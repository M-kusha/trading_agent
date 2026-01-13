# train/controllers/clip_controller.py
"""
Smart Clip Range Controller with PID-based adaptive control.
Stage-aware bounds protect learned strategies in late stages.

Robustness upgrades:
- Health mapping is truly 0..1 (stuck..too-fast) with smooth log-ratio transforms
- Emergency logic uses interpretable raw signals (KL ratio + clip fraction) and is reachable
- Slew limiting clamps around the *current* value (safe under external overrides/checkpoints)
- PID state alignment is done defensively (no hard dependency on private fields)
- Strong numeric guards + sane defaults under missing/partial telemetry
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import math

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


class SmartClipController:
    """
    PID-based PPO clip_range controller with stage awareness.

    Control objective:
      Maintain a stage-specific "update health" near target.
      - 0.0 => stuck / too little change
      - 1.0 => too aggressive / unstable change
      - ~0.5 => healthy update rate

    Telemetry inputs:
      - KL divergence (policy change rate)
      - clip_fraction (fraction of updates being clipped)

    Operational safety:
      - Stage bounds prevent catastrophic late-stage unlearning
      - Cooldowns + minimum update interval prevent oscillations
      - Emergency logic handles stuck or runaway policy updates
    """

    MAX_STAGE = 9

    # Clip range bounds by stage
    STAGE_CLIP_BOUNDS: Dict[int, Tuple[float, float]] = {
        0: (0.15, 0.40),  # Discovery
        1: (0.15, 0.35),  # Foundation
        2: (0.12, 0.30),  # Trend Student
        3: (0.12, 0.28),  # Session Student
        4: (0.10, 0.25),  # Timing Student
        5: (0.10, 0.22),  # Integrator
        6: (0.08, 0.20),  # Risk Manager
        7: (0.06, 0.18),  # Strategist
        8: (0.05, 0.15),  # Professional
        9: (0.04, 0.12),  # Live Ready
    }

    # Target update-health by stage (lower in later stages)
    STAGE_UPDATE_HEALTH_TARGETS: Dict[int, float] = {
        0: 0.55,
        1: 0.52,
        2: 0.50,
        3: 0.48,
        4: 0.46,
        5: 0.44,
        6: 0.42,
        7: 0.40,
        8: 0.38,
        9: 0.35,
    }

    # Stage-aware KL targets (lower in later stages)
    STAGE_KL_TARGETS: Dict[int, float] = {
        0: 0.018,
        1: 0.016,
        2: 0.014,
        3: 0.013,
        4: 0.012,
        5: 0.011,
        6: 0.010,
        7: 0.008,
        8: 0.006,
        9: 0.005,
    }

    # Clip fraction "ideal" region (used to shape health).
    # Interpretable anchor for PPO: too low => under-updating; too high => overly aggressive.
    IDEAL_CLIP_FRAC = 0.15

    def __init__(
        self,
        base_clip_range: float = 0.20,
        # Scheduling / stability knobs
        min_steps_between_updates: int = 25_000,
        cooldown_after_stage_change: int = 40_000,
        cooldown_after_emergency: int = 20_000,
        # Slew limiting per apply (relative to current value)
        max_relative_step: float = 0.20,  # 20% max change per apply
        # PID tuning
        kp: float = 0.12,
        ki: float = 0.008,
        kd: float = 0.04,
        # History windows
        history_len: int = 20,
        smooth_window: int = 6,
    ):
        self.current_stage = 0

        self.min_steps_between_updates = int(max(1, min_steps_between_updates))
        self.cooldown_after_stage_change = int(max(0, cooldown_after_stage_change))
        self.cooldown_after_emergency = int(max(0, cooldown_after_emergency))
        self.max_relative_step = float(max(0.01, max_relative_step))

        self.history_len = int(max(8, history_len))
        self.smooth_window = int(max(3, min(smooth_window, self.history_len)))

        self.cooldown_steps = 0
        self.steps_since_update = 0

        bounds = self.STAGE_CLIP_BOUNDS[0]
        base_clip = float(base_clip_range) if _is_finite(base_clip_range) else 0.20
        base_clip = _clamp(base_clip, bounds[0], bounds[1])

        self.pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            setpoint=self.STAGE_UPDATE_HEALTH_TARGETS[0],
            output_min=bounds[0],
            output_max=bounds[1],
            deadband=0.06,
            d_filter_alpha=0.18,
            max_delta_per_update=0.02,  # absolute cap per PID update
        )

        self._last_clip = base_clip

        # History for computing update health
        self._kl_history: list[float] = []
        self._clip_frac_history: list[float] = []

    # ---------------------------- telemetry ----------------------------

    def update_history(
        self,
        kl_divergence: Optional[float] = None,
        clip_fraction: Optional[float] = None,
    ) -> None:
        """Update internal history buffers."""
        if kl_divergence is not None and _is_finite(kl_divergence):
            self._kl_history.append(float(kl_divergence))
            self._kl_history = self._kl_history[-self.history_len :]
        if clip_fraction is not None and _is_finite(clip_fraction):
            # clip_fraction should be in [0, 1], but we clamp softly
            self._clip_frac_history.append(_clamp(float(clip_fraction), 0.0, 1.0))
            self._clip_frac_history = self._clip_frac_history[-self.history_len :]

    def _recent_avg(self, xs: list[float]) -> Optional[float]:
        if len(xs) < 3:
            return None
        return _mean(xs[-self.smooth_window :])

    # ---------------------------- health mapping ----------------------------

    def _kl_ratio(self) -> Optional[float]:
        """avg_kl / target_kl (smoothed)."""
        avg_kl = self._recent_avg(self._kl_history)
        if avg_kl is None:
            return None
        target_kl = float(self.STAGE_KL_TARGETS.get(self.current_stage, 0.012))
        if target_kl <= 0.0:
            return None
        return float(avg_kl) / float(target_kl)

    def _kl_score_01(self, kl_ratio: float) -> float:
        """
        Smoothly map KL ratio to [0..1], centered at 0.5 when ratio==1.

        Uses log-ratio + tanh so that:
          ratio << 1 -> near 0 (stuck)
          ratio == 1 -> 0.5 (on target)
          ratio >> 1 -> near 1 (too fast)
        """
        eps = 1e-12
        r = max(eps, float(kl_ratio))
        # scale chosen so ~0.3 -> ~0.15 and ~3 -> ~0.85 (approximately)
        scale = 1.35
        z = math.log(r) / scale
        return _clamp(0.5 * (1.0 + math.tanh(z)), 0.0, 1.0)

    def _clip_frac_score_01(self, clip_frac: float) -> float:
        """
        Map clip_fraction to [0..1], centered at 0.5 around IDEAL_CLIP_FRAC.
        Lower clip_frac => under-updating (low score).
        Higher clip_frac => over-updating (high score).
        """
        cf = _clamp(float(clip_frac), 0.0, 1.0)
        ideal = float(self.IDEAL_CLIP_FRAC)
        # scale sets how quickly score ramps away from ideal
        scale = 0.12
        z = (cf - ideal) / scale
        return _clamp(0.5 * (1.0 + math.tanh(z)), 0.0, 1.0)

    def _compute_update_health(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute update health in [0..1] plus debug signals.

        Returns:
          (health, signals)
        """
        kl_ratio = self._kl_ratio()
        avg_cf = self._recent_avg(self._clip_frac_history)

        # If telemetry is missing, return neutral.
        if kl_ratio is None and avg_cf is None:
            return 0.5, {"health": 0.5}

        # Use whichever signals are available.
        if kl_ratio is not None:
            kl_s = self._kl_score_01(kl_ratio)
        else:
            kl_s = 0.5

        if avg_cf is not None:
            cf_s = self._clip_frac_score_01(avg_cf)
        else:
            cf_s = 0.5

        # Favor KL slightly; clip_frac is a secondary proxy
        health = 0.65 * kl_s + 0.35 * cf_s
        health = _clamp(float(health), 0.0, 1.0)

        signals = {
            "health": health,
            "kl_ratio": float(kl_ratio) if kl_ratio is not None else float("nan"),
            "avg_clip_frac": float(avg_cf) if avg_cf is not None else float("nan"),
            "kl_score": float(kl_s),
            "cf_score": float(cf_s),
        }
        return health, signals

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
        bounds = self.STAGE_CLIP_BOUNDS[self.current_stage]

        self.pid.set_setpoint(self.STAGE_UPDATE_HEALTH_TARGETS[self.current_stage])
        self.pid.output_min = bounds[0]
        self.pid.output_max = bounds[1]
        self.pid.reset()

        self.cooldown_steps = self.cooldown_after_stage_change
        self.steps_since_update = 0

        # Keep last clip within the new stage bounds
        self._last_clip = _clamp(self._last_clip, bounds[0], bounds[1])

    # ---------------------------- emergency thresholds ----------------------------

    def _emergency_thresholds(self, stage: int) -> Tuple[float, float, float, float]:
        """
        Returns thresholds for emergency logic as:
          (stuck_kl_ratio, stuck_cf, fast_kl_ratio, fast_cf)

        Later stages are stricter (lower fast thresholds, slightly higher stuck thresholds),
        because we must protect learned behavior and avoid sudden unlearning.
        """
        stage = int(max(0, min(stage, self.MAX_STAGE)))

        # Interpolate in stage 0..9
        t = stage / float(self.MAX_STAGE)  # 0..1

        stuck_kl_ratio = (0.18 * (1.0 - t)) + (0.30 * t)  # 0.18 -> 0.30
        stuck_cf = (0.05 * (1.0 - t)) + (0.07 * t)        # 0.05 -> 0.07

        fast_kl_ratio = (3.20 * (1.0 - t)) + (2.40 * t)   # 3.2 -> 2.4
        fast_cf = (0.42 * (1.0 - t)) + (0.32 * t)         # 0.42 -> 0.32

        return float(stuck_kl_ratio), float(stuck_cf), float(fast_kl_ratio), float(fast_cf)

    # ---------------------------- main control ----------------------------

    def get_clip_range(
        self,
        current_clip: float,
        timesteps_elapsed: int,
    ) -> Tuple[float, str, bool]:
        """
        Compute recommended clip range.

        Args:
            current_clip: current clip_range
            timesteps_elapsed: delta-steps since last call

        Returns:
            (clip_range, reason, should_apply)
        """
        # Defensive inputs
        try:
            delta_steps = int(timesteps_elapsed)
        except Exception:
            delta_steps = 0
        delta_steps = max(0, delta_steps)

        if not _is_finite(current_clip):
            current_clip = self._last_clip

        bounds = self.STAGE_CLIP_BOUNDS[self.current_stage]
        current_clip = _clamp(float(current_clip), bounds[0], bounds[1])

        self.steps_since_update += delta_steps

        # If external systems changed clip abruptly (checkpoint load), resync last_clip
        if self._last_clip > 0 and abs(current_clip - self._last_clip) / self._last_clip > 0.35:
            self._last_clip = current_clip

        # Cooldown
        if self.cooldown_steps > 0:
            self.cooldown_steps = max(0, self.cooldown_steps - delta_steps)
            return (
                current_clip,
                f"COOLDOWN: clip={current_clip:.3f} (stage {self.current_stage})",
                False,
            )

        # Minimum interval
        if self.steps_since_update < self.min_steps_between_updates:
            return (
                current_clip,
                f"WAITING: clip={current_clip:.3f} (stage {self.current_stage})",
                False,
            )

        # Compute update health + signals
        health, sig = self._compute_update_health()

        # Emergency logic on raw interpretable signals (when available)
        stuck_kl_ratio, stuck_cf, fast_kl_ratio, fast_cf = self._emergency_thresholds(self.current_stage)

        kl_ratio = sig.get("kl_ratio", float("nan"))
        avg_cf = sig.get("avg_clip_frac", float("nan"))

        have_kl = _is_finite(kl_ratio)
        have_cf = _is_finite(avg_cf)

        is_stuck = False
        is_fast = False

        if have_kl and have_cf:
            is_stuck = (kl_ratio < stuck_kl_ratio) and (avg_cf < stuck_cf)
            is_fast = (kl_ratio > fast_kl_ratio) or (avg_cf > fast_cf)
        elif have_kl:
            # Single-signal fallback: use 0.85x threshold (15% tighter) because
            # with only one signal we need stronger evidence to declare "stuck".
            # This reduces false positives when clip_fraction data is unavailable.
            is_stuck = (kl_ratio < stuck_kl_ratio * 0.85)
            is_fast = (kl_ratio > fast_kl_ratio)
        elif have_cf:
            # Same logic: require 15% more extreme clip_fraction when KL is unavailable
            is_stuck = (avg_cf < stuck_cf * 0.85)
            is_fast = (avg_cf > fast_cf)

        if is_stuck:
            # Widen clip to allow larger policy steps
            widened = min(bounds[1], current_clip * (1.0 + self.max_relative_step))
            widened = _clamp(widened, bounds[0], bounds[1])

            self.steps_since_update = 0
            self.cooldown_steps = self.cooldown_after_emergency
            self._last_clip = widened

            # Align PID state (defensive)
            if hasattr(self.pid, "_last_output"):
                try:
                    self.pid._last_output = widened
                except Exception:
                    pass

            return (
                widened,
                f"EMERGENCY_STUCK: health={health:.3f} kl_ratio={kl_ratio:.3f} cf={avg_cf:.3f} -> clip={widened:.3f}",
                True,
            )

        if is_fast:
            # Tighten clip to prevent runaway updates
            tightened = max(bounds[0], current_clip * (1.0 - self.max_relative_step))
            tightened = _clamp(tightened, bounds[0], bounds[1])

            self.steps_since_update = 0
            self.cooldown_steps = self.cooldown_after_emergency
            self._last_clip = tightened

            if hasattr(self.pid, "_last_output"):
                try:
                    self.pid._last_output = tightened
                except Exception:
                    pass

            return (
                tightened,
                f"EMERGENCY_FAST: health={health:.3f} kl_ratio={kl_ratio:.3f} cf={avg_cf:.3f} -> clip={tightened:.3f}",
                True,
            )

        # PID control on update health
        dt = max(1.0, float(self.steps_since_update) / float(self.min_steps_between_updates))
        proposed_clip, pid_reason = self.pid.update(float(health), dt=dt)

        # Slew limit around *current* value (safe under external changes)
        upper = current_clip * (1.0 + self.max_relative_step)
        lower = current_clip * (1.0 - self.max_relative_step)
        proposed_clip = _clamp(float(proposed_clip), lower, upper)

        # Clamp to stage bounds
        proposed_clip = _clamp(float(proposed_clip), bounds[0], bounds[1])

        # Apply only if meaningful (>5%)
        if abs(proposed_clip - current_clip) / max(current_clip, 1e-6) < 0.05:
            self.steps_since_update = 0  # we still “consumed” an evaluation bucket
            self._last_clip = current_clip
            return (
                current_clip,
                f"PID_STABLE: health={health:.3f} clip={current_clip:.3f} | {pid_reason}",
                False,
            )

        self.steps_since_update = 0
        self._last_clip = proposed_clip
        return (
            proposed_clip,
            f"PID_APPLY: health={health:.3f} clip={proposed_clip:.3f} | {pid_reason}",
            True,
        )

    def to_dict(self) -> dict:
        """Serialize controller state for checkpointing."""
        return {
            "current_stage": self.current_stage,
            "cooldown_steps": self.cooldown_steps,
            "steps_since_update": self.steps_since_update,
            "_last_clip": self._last_clip,
            "_kl_history": list(self._kl_history[-15:]),
            "_clip_frac_history": list(self._clip_frac_history[-15:]),
            "pid_state": self.pid.to_dict(),
        }

    def load_from_dict(self, state: dict) -> None:
        """Restore controller state from checkpoint."""
        self.current_stage = int(state.get("current_stage", 0))
        self.cooldown_steps = int(state.get("cooldown_steps", 0))
        self.steps_since_update = int(state.get("steps_since_update", 0))
        self._last_clip = float(state.get("_last_clip", 0.20))
        self._kl_history = list(state.get("_kl_history", []))
        self._clip_frac_history = list(state.get("_clip_frac_history", []))
        pid_state = state.get("pid_state", {})
        if pid_state:
            self.pid.load_from_dict(pid_state)
