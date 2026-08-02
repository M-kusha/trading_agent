# train/controllers/lr_controller.py
"""
Smart Learning Rate Controller with PID-based adaptive control.
Stage-aware bounds protect learned strategies in late stages.

Robustness upgrades:
- Learning-health is scale-aware (relative trends, CV-style stability metrics)
- Separate instability detector gates emergency LR reductions
- Slew limiting clamps around the *current* multiplier (safe under external overrides)
- Stage-aware “stuck” handling: early can increase LR; later prioritizes stability
- Strong numeric guards + partial-telemetry behavior
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .pid_controller import PIDController


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / float(len(xs))


def _safe_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.std(np.asarray(xs, dtype=np.float64)))


def _lin_slope(xs: list[float]) -> float:
    """
    Least-squares slope over equally spaced points.
    Returns slope per step index.
    """
    n = len(xs)
    if n < 3:
        return 0.0
    y = np.asarray(xs, dtype=np.float64)
    x = np.arange(n, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


class SmartLRController:
    """
    PID-based learning rate controller with stage awareness.

    Control objective:
      Maintain stage-specific "learning health" near target.
      - 0.0 => learning unhealthy / not improving
      - 1.0 => learning healthy / improving
      Targets decrease by stage to prefer stability/fine-tuning later.

    Additionally:
      Instability detector (loss volatility / divergence) can force emergency LR reduction,
      independent of health target.
    """

    MAX_STAGE = 9

    # Learning rate multiplier bounds by stage (relative to base_lr)
    STAGE_LR_MULTIPLIERS: Dict[int, Tuple[float, float]] = {
        0: (0.5, 2.0),
        1: (0.5, 1.8),
        2: (0.4, 1.5),
        3: (0.4, 1.3),
        4: (0.35, 1.2),
        5: (0.3, 1.0),
        6: (0.25, 0.8),
        7: (0.2, 0.6),
        8: (0.15, 0.5),
        9: (0.1, 0.4),
    }

    # Target "learning health" by stage (lower later)
    STAGE_HEALTH_TARGETS: Dict[int, float] = {
        0: 0.70,
        1: 0.65,
        2: 0.60,
        3: 0.55,
        4: 0.50,
        5: 0.45,
        6: 0.40,
        7: 0.35,
        8: 0.30,
        9: 0.25,
    }

    def __init__(
        self,
        base_lr: float = 1e-4,
        # Scheduling / stability knobs
        min_steps_between_updates: int = 40_000,
        cooldown_after_stage_change: int = 60_000,
        cooldown_after_emergency: int = 30_000,
        # Slew limiting per apply (relative to current multiplier)
        max_relative_step: float = 0.15,
        # PID tuning (conservative for LR)
        kp: float = 0.08,
        ki: float = 0.005,
        kd: float = 0.03,
        # History windows
        history_len_loss: int = 30,
        history_len_reward: int = 50,
        smooth_window: int = 12,
    ):
        self.base_lr = float(base_lr) if _is_finite(base_lr) and base_lr > 0 else 1e-4
        self.current_stage = 0

        self.min_steps_between_updates = int(max(1, min_steps_between_updates))
        self.cooldown_after_stage_change = int(max(0, cooldown_after_stage_change))
        self.cooldown_after_emergency = int(max(0, cooldown_after_emergency))
        self.max_relative_step = float(max(0.01, max_relative_step))

        self.history_len_loss = int(max(12, history_len_loss))
        self.history_len_reward = int(max(20, history_len_reward))
        self.smooth_window = int(max(6, smooth_window))

        self.cooldown_steps = 0
        self.steps_since_update = 0

        bounds = self.STAGE_LR_MULTIPLIERS[0]
        self.pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            setpoint=self.STAGE_HEALTH_TARGETS[0],
            output_min=bounds[0],  # multiplier
            output_max=bounds[1],
            deadband=0.08,
            d_filter_alpha=0.15,
            max_delta_per_update=0.05,  # absolute multiplier step per PID update
        )

        self._last_lr_mult = 1.0

        # History for computing health/instability
        self._p_loss_history: list[float] = []
        self._v_loss_history: list[float] = []
        self._reward_history: list[float] = []

    # ---------------------------- telemetry ----------------------------

    def update_history(
        self,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> None:
        """Update internal history buffers."""
        if policy_loss is not None and _is_finite(policy_loss):
            self._p_loss_history.append(float(policy_loss))
            self._p_loss_history = self._p_loss_history[-self.history_len_loss :]
        if value_loss is not None and _is_finite(value_loss):
            self._v_loss_history.append(float(value_loss))
            self._v_loss_history = self._v_loss_history[-self.history_len_loss :]
        if reward is not None and _is_finite(reward):
            self._reward_history.append(float(reward))
            self._reward_history = self._reward_history[-self.history_len_reward :]

    def _recent(self, xs: list[float], n: int) -> list[float]:
        if n <= 0:
            return []
        return xs[-n:] if len(xs) >= 1 else []

    # ---------------------------- health + instability ----------------------------

    def _policy_trend_score(self) -> float:
        """
        Policy loss trend score:
          decreasing loss => higher score
        Uses relative slope to reduce dependence on absolute scale.
        """
        xs = self._recent(self._p_loss_history, self.smooth_window)
        if len(xs) < 6:
            return 0.5
        slope = _lin_slope(xs)  # per index
        denom = abs(_mean(xs)) + 1e-8
        rel = float(slope) / float(denom)

        # rel < 0 (decreasing) is good; map to [0..1]
        # scale chosen so rel=-0.05 => high, rel=+0.05 => low
        scale = 0.06
        z = -rel / scale
        return _clamp(0.5 * (1.0 + math.tanh(z)), 0.0, 1.0)

    def _value_stability_score(self) -> float:
        """
        Value loss stability score:
          lower coefficient-of-variation => higher score
        """
        xs = self._recent(self._v_loss_history, self.smooth_window)
        if len(xs) < 6:
            return 0.5
        mu = abs(_mean(xs)) + 1e-8
        sd = _safe_std(xs)
        cv = float(sd) / float(mu)  # scale-free

        # CV <= 0.15 => very stable (near 1)
        # CV >= 1.00 => unstable (near 0)
        good = 0.15
        bad = 1.00
        if cv <= good:
            return 0.98
        if cv >= bad:
            return 0.05
        # linear interpolation
        return _clamp(1.0 - (cv - good) / (bad - good), 0.0, 1.0)

    def _reward_improvement_score(self) -> float:
        """
        Reward improvement score:
          increasing reward => higher score
        Uses relative change to reduce dependence on absolute scale.
        """
        rs = self._recent(self._reward_history, max(self.smooth_window, 20))
        if len(rs) < 20:
            return 0.5

        half = len(rs) // 2
        early = _mean(rs[:half])
        late = _mean(rs[half:])

        denom = abs(early) + 1e-6
        rel_improve = float(late - early) / float(denom)

        # rel_improve ~ +0.25 => strong improvement; -0.25 => bad
        scale = 0.30
        z = rel_improve / scale
        return _clamp(0.5 * (1.0 + math.tanh(z)), 0.0, 1.0)

    def _compute_learning_health(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute learning health in [0..1] + debug signals.
        Returns neutral if insufficient telemetry.
        """
        have_p = len(self._p_loss_history) >= 8
        have_v = len(self._v_loss_history) >= 8
        have_r = len(self._reward_history) >= 20

        if not (have_p or have_v or have_r):
            return 0.5, {"health": 0.5}

        p_score = self._policy_trend_score() if have_p else 0.5
        v_score = self._value_stability_score() if have_v else 0.5
        r_score = self._reward_improvement_score() if have_r else 0.5

        # Reward + policy trend dominate; value stability is a stabilizer term
        health = 0.40 * p_score + 0.25 * v_score + 0.35 * r_score
        health = _clamp(float(health), 0.0, 1.0)

        return health, {
            "health": health,
            "p_score": float(p_score),
            "v_score": float(v_score),
            "r_score": float(r_score),
        }

    def _compute_instability(self) -> Tuple[float, Dict[str, float]]:
        """
        Instability index in [0..1]:
          0 => stable
          1 => divergent/volatile

        Uses scale-free volatility + adverse trends.
        """
        xs_v = self._recent(self._v_loss_history, self.smooth_window)
        xs_p = self._recent(self._p_loss_history, self.smooth_window)

        v_mu = abs(_mean(xs_v)) + 1e-8
        v_cv = (_safe_std(xs_v) / v_mu) if len(xs_v) >= 6 else 0.0

        p_mu = abs(_mean(xs_p)) + 1e-8
        p_slope = _lin_slope(xs_p) if len(xs_p) >= 6 else 0.0
        p_rel = float(p_slope) / float(p_mu)

        # Convert signals to [0..1] “badness”
        # v_cv: 0.2 good, 1.2 bad
        v_bad = _clamp((v_cv - 0.20) / (1.20 - 0.20), 0.0, 1.0)

        # p_rel: positive slope is bad (loss increasing over time)
        # Threshold 0.06: In PPO, relative slope >3% is suspicious, >6% is alarming.
        # This scales linearly: 0% -> 0.0 badness, 6%+ -> 1.0 badness.
        p_bad = _clamp((p_rel - 0.00) / 0.06, 0.0, 1.0)

        # reward collapse indicator
        rs = self._recent(self._reward_history, max(self.smooth_window, 20))
        if len(rs) >= 20:
            half = len(rs) // 2
            early = _mean(rs[:half])
            late = _mean(rs[half:])
            denom = abs(early) + 1e-6
            rel_drop = float(early - late) / float(denom)  # positive => dropping
            r_bad = _clamp(rel_drop / 0.40, 0.0, 1.0)
        else:
            r_bad = 0.0

        # Weighted instability
        instability = 0.45 * v_bad + 0.35 * p_bad + 0.20 * r_bad
        instability = _clamp(float(instability), 0.0, 1.0)

        return instability, {
            "instability": instability,
            "v_cv": float(v_cv),
            "p_rel_slope": float(p_rel),
            "r_bad": float(r_bad),
        }

    def _instability_threshold(self, stage: int) -> float:
        """
        Later stages tolerate less instability.
        """
        stage = int(max(0, min(stage, self.MAX_STAGE)))
        t = stage / float(self.MAX_STAGE)  # 0..1
        return float((0.75 * (1.0 - t)) + (0.45 * t))  # 0.75 -> 0.45

    # ---------------------------- stage handling ----------------------------

    def on_stage_change(self, new_stage: int) -> None:
        """Handle curriculum stage transition."""
        try:
            ns = int(new_stage)
        except Exception:
            ns = self.current_stage

        ns = max(0, min(ns, self.MAX_STAGE))
        if ns == self.current_stage:
            return

        self.current_stage = ns
        bounds = self.STAGE_LR_MULTIPLIERS[self.current_stage]

        self.pid.set_setpoint(self.STAGE_HEALTH_TARGETS[self.current_stage])
        self.pid.output_min = bounds[0]
        self.pid.output_max = bounds[1]
        self.pid.reset()

        self.cooldown_steps = self.cooldown_after_stage_change
        self.steps_since_update = 0

        self._last_lr_mult = _clamp(self._last_lr_mult, bounds[0], bounds[1])

    # ---------------------------- main control ----------------------------

    def get_lr(
        self,
        current_lr: float,
        timesteps_elapsed: int,
    ) -> Tuple[float, str, bool]:
        """
        Compute recommended learning rate.

        Args:
            current_lr: current learning rate
            timesteps_elapsed: delta-steps since last call

        Returns:
            (lr, reason, should_apply)
        """
        # Defensive delta-steps
        try:
            delta_steps = int(timesteps_elapsed)
        except Exception:
            delta_steps = 0
        delta_steps = max(0, delta_steps)

        if not _is_finite(current_lr) or current_lr <= 0:
            current_lr = self.base_lr

        bounds = self.STAGE_LR_MULTIPLIERS[self.current_stage]
        current_mult = float(current_lr) / float(self.base_lr) if self.base_lr > 0 else 1.0
        if not _is_finite(current_mult) or current_mult <= 0:
            current_mult = 1.0

        current_mult = _clamp(current_mult, bounds[0], bounds[1])
        current_lr = self.base_lr * current_mult

        self.steps_since_update += delta_steps

        # If external systems changed LR abruptly (checkpoint load), resync last mult
        if self._last_lr_mult > 0 and abs(current_mult - self._last_lr_mult) / self._last_lr_mult > 0.35:
            self._last_lr_mult = current_mult

        # Cooldown
        if self.cooldown_steps > 0:
            self.cooldown_steps = max(0, self.cooldown_steps - delta_steps)
            return (
                current_lr,
                f"COOLDOWN: lr={current_lr:.2e} mult={current_mult:.3f} (stage {self.current_stage})",
                False,
            )

        # Minimum interval
        if self.steps_since_update < self.min_steps_between_updates:
            return (
                current_lr,
                f"WAITING: lr={current_lr:.2e} mult={current_mult:.3f} (stage {self.current_stage})",
                False,
            )

        # Compute health and instability
        health, hs = self._compute_learning_health()
        instability, ins = self._compute_instability()
        inst_th = self._instability_threshold(self.current_stage)

        # Emergency: instability too high -> reduce LR
        if instability > inst_th:
            new_mult = max(bounds[0], current_mult * (1.0 - max(self.max_relative_step, 0.20)))
            new_lr = self.base_lr * new_mult

            self.steps_since_update = 0
            self.cooldown_steps = self.cooldown_after_emergency
            self._last_lr_mult = new_mult

            if hasattr(self.pid, "_last_output"):
                try:
                    self.pid._last_output = new_mult
                except Exception:
                    pass

            return (
                new_lr,
                (
                    f"EMERGENCY_INSTABILITY: inst={instability:.3f}>{inst_th:.3f} "
                    f"v_cv={ins.get('v_cv', float('nan')):.3f} p_rel={ins.get('p_rel_slope', float('nan')):.3f} "
                    f"-> mult={new_mult:.3f} lr={new_lr:.2e}"
                ),
                True,
            )

        target = float(self.STAGE_HEALTH_TARGETS[self.current_stage])

        # Extremely low health handling (stage-aware):
        # - Early stages (0..2): increase LR to escape plateaus if stable
        # - Later stages (3..9): reduce LR slightly to regain stability
        if health < 0.12:
            if self.current_stage <= 2:
                new_mult = min(bounds[1], current_mult * (1.0 + max(0.10, self.max_relative_step)))
                action = "BOOST"
            else:
                new_mult = max(bounds[0], current_mult * (1.0 - max(0.10, self.max_relative_step)))
                action = "REDUCE"

            new_lr = self.base_lr * new_mult

            self.steps_since_update = 0
            self.cooldown_steps = self.cooldown_after_emergency
            self._last_lr_mult = new_mult

            if hasattr(self.pid, "_last_output"):
                try:
                    self.pid._last_output = new_mult
                except Exception:
                    pass

            return (
                new_lr,
                f"EMERGENCY_LOW_HEALTH_{action}: health={health:.3f} target={target:.2f} -> mult={new_mult:.3f} lr={new_lr:.2e}",
                True,
            )

        # PID control on health
        dt = max(1.0, float(self.steps_since_update) / float(self.min_steps_between_updates))
        proposed_mult, pid_reason = self.pid.update(float(health), dt=dt)

        # Slew limit around *current* multiplier
        upper = current_mult * (1.0 + self.max_relative_step)
        lower = current_mult * (1.0 - self.max_relative_step)
        proposed_mult = _clamp(float(proposed_mult), lower, upper)

        # Clamp to stage bounds
        proposed_mult = _clamp(float(proposed_mult), bounds[0], bounds[1])
        proposed_lr = self.base_lr * proposed_mult

        # Apply only if meaningful (>3%)
        if abs(proposed_mult - current_mult) / max(current_mult, 1e-6) < 0.03:
            self.steps_since_update = 0
            self._last_lr_mult = current_mult
            return (
                current_lr,
                f"PID_STABLE: health={health:.3f} inst={instability:.3f} mult={current_mult:.3f} | {pid_reason}",
                False,
            )

        self.steps_since_update = 0
        self._last_lr_mult = proposed_mult
        return (
            proposed_lr,
            f"PID_APPLY: health={health:.3f} target={target:.2f} inst={instability:.3f} mult={proposed_mult:.3f} lr={proposed_lr:.2e} | {pid_reason}",
            True,
        )

    def to_dict(self) -> dict:
        """Serialize controller state for checkpointing."""
        return {
            "current_stage": self.current_stage,
            "cooldown_steps": self.cooldown_steps,
            "steps_since_update": self.steps_since_update,
            "_last_lr_mult": self._last_lr_mult,
            "_p_loss_history": list(self._p_loss_history[-20:]),
            "_v_loss_history": list(self._v_loss_history[-20:]),
            "_reward_history": list(self._reward_history[-30:]),
            "pid_state": self.pid.to_dict(),
        }

    def load_from_dict(self, state: dict) -> None:
        """Restore controller state from checkpoint."""
        self.current_stage = int(state.get("current_stage", 0))
        self.cooldown_steps = int(state.get("cooldown_steps", 0))
        self.steps_since_update = int(state.get("steps_since_update", 0))
        self._last_lr_mult = float(state.get("_last_lr_mult", 1.0))
        self._p_loss_history = list(state.get("_p_loss_history", []))
        self._v_loss_history = list(state.get("_v_loss_history", []))
        self._reward_history = list(state.get("_reward_history", []))
        pid_state = state.get("pid_state", {})
        if pid_state:
            self.pid.load_from_dict(pid_state)
