"""
train/controllers/entropy_controller.py

Smart Entropy Controller with PID-based adaptive control.
Upgraded: consistent stage tolerances, mask-aware effective action count,
output slew limiting, and better numeric guards.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import math
import numpy as np

from .pid_controller import PIDController


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SmartEntropyController:
    """
    High-level entropy management using PID control + stage awareness.

    Controls ent_coef using normalized entropy:
      H_norm = H_raw / log(K_eff)
    where K_eff is an estimate of valid actions (mask-aware).

    Goals:
    - Early stages: allow exploration, only intervene on entropy collapse.
    - Later stages: enforce predictable behavior with tight tolerances.
    - Avoid oscillations: PID + cooldown + update interval + slew limiting.
    """

    # Normalized entropy targets (0=deterministic, 1=uniform random over valid actions)
    STAGE_TARGETS_NORMALIZED: Dict[int, float] = {
        0: 0.85,
        1: 0.75,
        2: 0.60,
        3: 0.50,
        4: 0.42,
        5: 0.32,
        6: 0.22,
        7: 0.16,
        8: 0.10,
        9: 0.06,
    }

    # ent_coef output bounds by stage
    # Lower ent_coef = less entropy bonus = more deterministic policy
    # Higher stages need lower ent_coef to achieve lower entropy targets
    STAGE_ENT_COEF_BOUNDS: Dict[int, Tuple[float, float]] = {
        0: (0.05, 0.15),    # Discovery: high exploration
        1: (0.04, 0.12),    # Foundation: moderate exploration
        2: (0.03, 0.10),    # Trend Student: learning patterns
        3: (0.025, 0.08),   # Session Student: time awareness
        4: (0.02, 0.06),    # Timing Student: entry timing
        5: (0.015, 0.05),   # Integrator: combining skills
        6: (0.01, 0.04),    # Risk Manager: capital preservation
        7: (0.008, 0.03),   # Optimizer: fine-tuning
        8: (0.005, 0.02),   # Consistency: stable execution
        9: (0.003, 0.015),  # Live Ready: minimal exploration
    }

    # Default action space size
    DEFAULT_N_ACTIONS = 10

    def __init__(
        self,
        initial_ent_coef: float = 0.15,
        n_actions: int = DEFAULT_N_ACTIONS,
        min_valid_floor: int = 4,
        # Scheduling / stability knobs
        min_steps_between_updates: int = 20_000,
        cooldown_after_stage_change: int = 40_000,
        cooldown_after_emergency: int = 15_000,
        # Max ent_coef relative adjustment per apply (prevents huge jumps)
        max_relative_step: float = 0.30,  # 30% max change per apply
        # PID tuning (kept conservative)
        kp: float = 0.10,
        ki: float = 0.01,
        kd: float = 0.05,
    ):
        self.current_stage = 0
        self.cooldown_steps = 0
        self.min_steps_between_updates = int(max(1, min_steps_between_updates))
        self.steps_since_update = 0

        self.cooldown_after_stage_change = int(max(0, cooldown_after_stage_change))
        self.cooldown_after_emergency = int(max(0, cooldown_after_emergency))

        self.n_actions = int(max(1, n_actions))
        self.min_valid_floor = int(max(2, min_valid_floor))

        self._valid_actions_estimate = float(self.n_actions)

        self.max_relative_step = float(max(0.01, max_relative_step))

        bounds = self.STAGE_ENT_COEF_BOUNDS[0]
        self.pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            setpoint=self.STAGE_TARGETS_NORMALIZED[0],
            output_min=bounds[0],
            output_max=bounds[1],
            deadband=0.12,
            d_filter_alpha=0.2,
            max_delta_per_update=None,  # we rate-limit at controller level instead
        )

        self._last_ent_coef = float(initial_ent_coef)

    # ---------------------------- entropy normalization ----------------------------

    def _get_max_entropy(self, n_valid_actions: Optional[int] = None) -> float:
        k = self._effective_valid_actions(n_valid_actions)
        return float(np.log(k))

    def _effective_valid_actions(self, n_valid_actions: Optional[int]) -> int:
        if n_valid_actions is not None:
            try:
                n_valid = int(n_valid_actions)
            except Exception:
                n_valid = self.min_valid_floor
        else:
            n_valid = int(round(self._valid_actions_estimate))

        n_valid = max(self.min_valid_floor, n_valid)
        n_valid = min(n_valid, self.n_actions)
        return n_valid

    def _normalize_entropy(self, raw_entropy: float, n_valid_actions: Optional[int] = None) -> float:
        if not _is_finite(raw_entropy):
            return 0.5
        max_h = self._get_max_entropy(n_valid_actions)
        if max_h <= 0.0 or not _is_finite(max_h):
            return 0.5
        return _clamp(float(raw_entropy) / max_h, 0.0, 1.0)

    def update_valid_actions_estimate(self, n_valid: int) -> None:
        """EMA update of valid-action estimate; clamped to sane range."""
        try:
            n_valid = int(n_valid)
        except Exception:
            return
        n_valid = max(self.min_valid_floor, min(self.n_actions, n_valid))

        # EMA on linear scale (stable enough); alpha can be stage-dependent if desired
        alpha = 0.10
        self._valid_actions_estimate = alpha * float(n_valid) + (1.0 - alpha) * float(self._valid_actions_estimate)

    # ---------------------------- stage handling ----------------------------

    def on_stage_change(self, new_stage: int) -> None:
        if new_stage == self.current_stage:
            return

        try:
            new_stage = int(new_stage)
        except Exception:
            new_stage = self.current_stage

        self.current_stage = max(0, min(new_stage, 9))
        bounds = self.STAGE_ENT_COEF_BOUNDS[self.current_stage]

        self.pid.set_setpoint(self.STAGE_TARGETS_NORMALIZED[self.current_stage])
        self.pid.output_min = bounds[0]
        self.pid.output_max = bounds[1]
        self.pid.reset()

        self.cooldown_steps = self.cooldown_after_stage_change
        self.steps_since_update = 0

    # ---------------------------- tolerances ----------------------------

    def _tolerance_band(self, stage: int) -> Tuple[float, float]:
        """
        Returns (low_tol, high_tol) where:
        - emergency low triggers if norm_entropy < target - low_tol
        - emergency high triggers if norm_entropy > target + high_tol

        Early stages: tolerate high entropy; only protect against collapse.
        Late stages: tight both sides.
        """
        stage = int(max(0, min(stage, 9)))

        if stage <= 1:
            # Discovery: only guard against collapse
            return (0.35, 10.0)  # effectively disables high-side emergency

        # Interpolate tolerances: stage 2..9 tighten progressively
        t = (stage - 2) / 7.0  # 0..1
        low_tol = (0.30 * (1.0 - t)) + (0.06 * t)   # 0.30 -> 0.06
        high_tol = (0.35 * (1.0 - t)) + (0.08 * t)  # 0.35 -> 0.08
        return (low_tol, high_tol)

    # ---------------------------- main control ----------------------------

    def get_ent_coef(
        self,
        current_entropy: float,
        current_ent_coef: float,
        timesteps_elapsed: int,
        n_valid_actions: Optional[int] = None,
    ) -> Tuple[float, str, bool]:
        """
        Compute recommended ent_coef.

        Args:
            current_entropy: raw entropy from policy (not normalized)
            current_ent_coef: current ent_coef
            timesteps_elapsed: delta-steps since last call (expected)
            n_valid_actions: number of valid actions (mask-aware). Optional.

        Returns:
            (ent_coef, reason, should_apply)
        """
        # Defensive delta-steps
        try:
            delta_steps = int(timesteps_elapsed)
        except Exception:
            delta_steps = 0
        delta_steps = max(0, delta_steps)

        self.steps_since_update += delta_steps

        if n_valid_actions is not None:
            self.update_valid_actions_estimate(n_valid_actions)

        norm_entropy = self._normalize_entropy(current_entropy, n_valid_actions)
        k_eff = self._effective_valid_actions(n_valid_actions)
        max_h = self._get_max_entropy(n_valid_actions)

        # Cooldown
        if self.cooldown_steps > 0:
            self.cooldown_steps = max(0, self.cooldown_steps - delta_steps)
            return (
                current_ent_coef,
                f"COOLDOWN: H_raw={current_entropy:.4f} H_norm={norm_entropy:.3f} K_eff={k_eff} logK={max_h:.3f}",
                False,
            )

        # Minimum interval
        if self.steps_since_update < self.min_steps_between_updates:
            return (
                current_ent_coef,
                f"WAITING: H_norm={norm_entropy:.3f} (stage={self.current_stage}, target={self.STAGE_TARGETS_NORMALIZED[self.current_stage]:.2f})",
                False,
            )

        target = self.STAGE_TARGETS_NORMALIZED[self.current_stage]
        bounds = self.STAGE_ENT_COEF_BOUNDS[self.current_stage]
        low_tol, high_tol = self._tolerance_band(self.current_stage)

        low_threshold = _clamp(target - low_tol, 0.0, 1.0)
        high_threshold = _clamp(target + high_tol, 0.0, 1.0)

        # ---------------- Emergency logic ----------------

        # Discovery phases: intervene only on collapse
        if self.current_stage <= 1:
            if norm_entropy < low_threshold:
                new_coef = min(bounds[1], float(current_ent_coef) * (1.0 + self.max_relative_step))
                self.steps_since_update = 0
                self.cooldown_steps = max(10_000, self.cooldown_after_emergency)
                # keep PID output aligned to prevent a follow-up snap-back
                self.pid._last_output = new_coef
                return new_coef, f"DISCOVERY_BOOST: H_norm={norm_entropy:.3f} < {low_threshold:.3f}", True

            return (
                current_ent_coef,
                f"DISCOVERY_PASSIVE: H_norm={norm_entropy:.3f} target={target:.2f}",
                False,
            )

        # Foundation+: emergency low/high
        if norm_entropy < low_threshold:
            new_coef = min(bounds[1], float(current_ent_coef) * (1.0 + self.max_relative_step))
            self.steps_since_update = 0
            self.cooldown_steps = self.cooldown_after_emergency
            self.pid._last_output = new_coef
            return new_coef, f"EMERGENCY_LOW: H_norm={norm_entropy:.3f} < {low_threshold:.3f}", True

        if norm_entropy > high_threshold:
            new_coef = max(bounds[0], float(current_ent_coef) * (1.0 - self.max_relative_step))
            self.steps_since_update = 0
            self.cooldown_steps = self.cooldown_after_emergency
            self.pid._last_output = new_coef
            return new_coef, f"EMERGENCY_HIGH: H_norm={norm_entropy:.3f} > {high_threshold:.3f}", True

        # ---------------- PID control ----------------

        # Let PID operate on normalized entropy; use dt proportional to elapsed steps bucket
        # (keeps I/D scaling roughly consistent if caller varies frequency)
        dt = max(1.0, float(self.steps_since_update) / float(self.min_steps_between_updates))

        proposed, pid_reason = self.pid.update(norm_entropy, dt=dt)

        # Rate limit relative step (hard safety clamp)
        if current_ent_coef > 0:
            upper = float(current_ent_coef) * (1.0 + self.max_relative_step)
            lower = float(current_ent_coef) * (1.0 - self.max_relative_step)
            proposed = _clamp(float(proposed), lower, upper)

        # Clamp to stage bounds
        proposed = _clamp(float(proposed), bounds[0], bounds[1])

        # Apply only if meaningful (>2%)
        if current_ent_coef > 0 and abs(proposed - current_ent_coef) / float(current_ent_coef) < 0.02:
            return current_ent_coef, f"PID_STABLE: H_norm={norm_entropy:.3f} | {pid_reason}", False

        self.steps_since_update = 0
        self._last_ent_coef = proposed
        return (
            proposed,
            f"PID_APPLY: H_norm={norm_entropy:.3f} target={target:.2f} K_eff={k_eff} | {pid_reason}",
            True,
        )
