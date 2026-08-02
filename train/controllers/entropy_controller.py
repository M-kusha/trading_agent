
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


class SmartEntropyController:


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


    STAGE_ENT_COEF_BOUNDS: Dict[int, Tuple[float, float]] = {
        0: (0.08, 0.25),
        1: (0.06, 0.20),
        2: (0.04, 0.15),
        3: (0.03, 0.12),
        4: (0.025, 0.10),
        5: (0.02, 0.08),
        6: (0.01, 0.04),
        7: (0.008, 0.03),
        8: (0.005, 0.02),
        9: (0.003, 0.015),
    }


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


    STAGE_CLIP_RANGE_BOUNDS: Dict[int, Tuple[float, float]] = {
        0: (0.15, 0.40),
        1: (0.15, 0.35),
        2: (0.12, 0.30),
        3: (0.12, 0.28),
        4: (0.10, 0.25),
        5: (0.10, 0.22),
        6: (0.08, 0.20),
        7: (0.06, 0.18),
        8: (0.05, 0.15),
        9: (0.04, 0.12),
    }


    DEFAULT_N_ACTIONS = 10

    def __init__(
        self,
        initial_ent_coef: float = 0.15,
        n_actions: int = DEFAULT_N_ACTIONS,
        min_valid_floor: int = 4,

        min_steps_between_updates: int = 20_000,
        cooldown_after_stage_change: int = 40_000,
        cooldown_after_emergency: int = 15_000,

        max_relative_step: float = 0.30,

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
            max_delta_per_update=None,
        )

        self._last_ent_coef = float(initial_ent_coef)
        self._emergency_active = False


    def _get_max_entropy(self, n_valid_actions: Optional[int] = None) -> float:
        # Deliberately the unrounded estimate. The effective action count is
        # genuinely fractional when averaged over a rollout that mixes flat
        # states with in-position ones, and rounding it re-introduced the swing
        # this normaliser exists to remove: an EMA hovering near k.5 flips
        # between log(k) and log(k+1) on alternating updates.
        # _effective_valid_actions stays integral for display.
        k = self._effective_valid_actions_float(n_valid_actions)
        return float(np.log(k))

    def _effective_valid_actions_float(self, n_valid_actions: Optional[int]) -> float:
        if n_valid_actions is not None:
            try:
                n_valid = float(n_valid_actions)
            except Exception:
                n_valid = float(self.min_valid_floor)
        else:
            n_valid = float(self._valid_actions_estimate)

        n_valid = max(float(self.min_valid_floor), n_valid)
        return min(n_valid, float(self.n_actions))

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
        try:
            n_valid = int(n_valid)
        except Exception:
            return
        n_valid = max(self.min_valid_floor, min(self.n_actions, n_valid))


        alpha = 0.10
        self._valid_actions_estimate = alpha * float(n_valid) + (1.0 - alpha) * float(self._valid_actions_estimate)


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


    def _tolerance_band(self, stage: int) -> Tuple[float, float]:
        stage = int(max(0, min(stage, 9)))

        if stage <= 1:

            return (0.35, 10.0)


        t = (stage - 2) / 7.0
        low_tol = (0.30 * (1.0 - t)) + (0.06 * t)
        high_tol = (0.35 * (1.0 - t)) + (0.08 * t)
        return (low_tol, high_tol)


    def get_ent_coef(
        self,
        current_entropy: float,
        current_ent_coef: float,
        timesteps_elapsed: int,
        n_valid_actions: Optional[int] = None,
    ) -> Tuple[float, str, bool]:

        try:
            delta_steps = int(timesteps_elapsed)
        except Exception:
            delta_steps = 0
        delta_steps = max(0, delta_steps)

        self.steps_since_update += delta_steps

        if n_valid_actions is not None:
            self.update_valid_actions_estimate(n_valid_actions)

        # current_entropy is SB3's rollout MEAN policy entropy - averaged over
        # thousands of steps that mix flat states (hold + K longs + K shorts
        # valid) with in-position states (hold + close). n_valid_actions is a
        # single instantaneous mask read at callback time. Dividing a batch mean
        # by a point sample made the same raw entropy normalise to 0.42 or 0.67
        # depending only on whether a position happened to be open at that
        # instant, and the PID chased the swing: ent_coef climbed 0.100 -> 0.130
        # -> 0.169 across three consecutive updates on a stationary policy.
        #
        # The EMA is already maintained above and reflects the average valid
        # action count over the rollout, which is the denominator that matches
        # the numerator. Passing None selects it.
        norm_entropy = self._normalize_entropy(current_entropy, None)
        k_eff = self._effective_valid_actions(None)
        max_h = self._get_max_entropy(None)


        if self.cooldown_steps > 0:
            self.cooldown_steps = max(0, self.cooldown_steps - delta_steps)
            return (
                current_ent_coef,
                f"COOLDOWN: H_raw={current_entropy:.4f} H_norm={norm_entropy:.3f} K_eff={k_eff} logK={max_h:.3f}",
                False,
            )


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


        if self.current_stage <= 1:
            if norm_entropy < low_threshold:
                new_coef = min(bounds[1], float(current_ent_coef) * (1.0 + self.max_relative_step))
                self.steps_since_update = 0
                self.cooldown_steps = max(10_000, self.cooldown_after_emergency)

                self.pid._last_output = new_coef
                return new_coef, f"DISCOVERY_BOOST: H_norm={norm_entropy:.3f} < {low_threshold:.3f}", True

            return (
                current_ent_coef,
                f"DISCOVERY_PASSIVE: H_norm={norm_entropy:.3f} target={target:.2f}",
                False,
            )


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


        dt = max(1.0, float(self.steps_since_update) / float(self.min_steps_between_updates))

        proposed, pid_reason = self.pid.update(norm_entropy, dt=dt)


        if current_ent_coef > 0:
            upper = float(current_ent_coef) * (1.0 + self.max_relative_step)
            lower = float(current_ent_coef) * (1.0 - self.max_relative_step)
            proposed = _clamp(float(proposed), lower, upper)


        proposed = _clamp(float(proposed), bounds[0], bounds[1])


        if current_ent_coef > 0 and abs(proposed - current_ent_coef) / float(current_ent_coef) < 0.02:
            return current_ent_coef, f"PID_STABLE: H_norm={norm_entropy:.3f} | {pid_reason}", False

        self.steps_since_update = 0
        self._last_ent_coef = proposed
        return (
            proposed,
            f"PID_APPLY: H_norm={norm_entropy:.3f} target={target:.2f} K_eff={k_eff} | {pid_reason}",
            True,
        )

    def to_dict(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "cooldown_steps": self.cooldown_steps,
            "steps_since_update": self.steps_since_update,
            "_valid_actions_estimate": self._valid_actions_estimate,
            "_last_ent_coef": self._last_ent_coef,
            "_emergency_active": self._emergency_active,
            "pid_state": self.pid.to_dict(),
        }

    def load_from_dict(self, state: dict) -> None:
        self.current_stage = int(state.get("current_stage", 0))
        self.cooldown_steps = int(state.get("cooldown_steps", 0))
        self.steps_since_update = int(state.get("steps_since_update", 0))
        self._valid_actions_estimate = float(state.get("_valid_actions_estimate", float(self.n_actions)))
        self._last_ent_coef = float(state.get("_last_ent_coef", 0.0))
        self._emergency_active = bool(state.get("_emergency_active", False))
        pid_state = state.get("pid_state", {})
        if pid_state:
            self.pid.load_from_dict(pid_state)
