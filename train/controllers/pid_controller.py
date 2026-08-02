
from __future__ import annotations

import math
from typing import Optional, Tuple


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


class PIDController:

    def __init__(
        self,
        kp: float = 0.3,
        ki: float = 0.05,
        kd: float = 0.1,
        setpoint: float = 0.7,
        output_min: float = 0.03,
        output_max: float = 0.15,
        integral_limit: float = 0.5,
        deadband: float = 0.05,

        d_filter_alpha: float = 0.2,
        max_delta_per_update: Optional[float] = None,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)

        self.output_min = float(output_min)
        self.output_max = float(output_max)
        if self.output_max < self.output_min:
            self.output_min, self.output_max = self.output_max, self.output_min

        self.integral_limit = float(abs(integral_limit))
        self.deadband = float(abs(deadband))

        self.d_filter_alpha = float(max(0.0, min(1.0, d_filter_alpha)))
        self.max_delta_per_update = float(max_delta_per_update) if max_delta_per_update is not None else None
        if self.max_delta_per_update is not None:
            self.max_delta_per_update = abs(self.max_delta_per_update)


        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = (self.output_min + self.output_max) / 2.0
        self._d_filtered = 0.0

    def update(self, current_value: float, dt: float = 1.0) -> Tuple[float, str]:
        if not _is_finite(current_value):
            return self._last_output, "PID: non-finite current_value; holding last output"

        if not _is_finite(dt) or dt <= 0.0:
            dt = 1.0

        error = self.setpoint - float(current_value)


        if abs(error) < self.deadband:
            return self._last_output, (
                f"IN_TARGET: val={current_value:.3f} target={self.setpoint:.3f} ±{self.deadband:.3f}"
            )


        p_term = self.kp * error


        d_raw = (error - self._last_error) / dt
        self._d_filtered = (1.0 - self.d_filter_alpha) * self._d_filtered + self.d_filter_alpha * d_raw
        d_term = self.kd * self._d_filtered


        tentative = self._last_output + p_term + d_term


        saturated_high = tentative >= self.output_max and error > 0
        saturated_low = tentative <= self.output_min and error < 0
        if not (saturated_high or saturated_low):
            self._integral += error * dt
            self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        i_term = self.ki * self._integral


        raw_output = self._last_output + p_term + i_term + d_term


        output = max(self.output_min, min(self.output_max, raw_output))


        if self.max_delta_per_update is not None:
            delta = output - self._last_output
            if abs(delta) > self.max_delta_per_update:
                output = self._last_output + (self.max_delta_per_update if delta > 0 else -self.max_delta_per_update)

        direction = "up" if output > self._last_output else "down" if output < self._last_output else "flat"
        reason = (
            f"PID {direction}: err={error:+.3f} "
            f"[P={p_term:+.4f}, I={i_term:+.4f}, D={d_term:+.4f}] "
            f"out={output:.5f}"
        )


        self._last_error = error
        self._last_output = output

        return output, reason

    def set_setpoint(self, new_setpoint: float) -> None:
        new_setpoint = float(new_setpoint)
        if abs(new_setpoint - self.setpoint) > 0.05:
            self._integral *= 0.5
        self.setpoint = new_setpoint

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = 0.0
        self._d_filtered = 0.0
        self._last_output = (self.output_min + self.output_max) / 2.0

    def to_dict(self) -> dict:
        return {
            "setpoint": self.setpoint,
            "output_min": self.output_min,
            "output_max": self.output_max,
            "_integral": self._integral,
            "_last_error": self._last_error,
            "_last_output": self._last_output,
            "_d_filtered": self._d_filtered,
        }

    def load_from_dict(self, state: dict) -> None:
        self.setpoint = float(state.get("setpoint", self.setpoint))
        self.output_min = float(state.get("output_min", self.output_min))
        self.output_max = float(state.get("output_max", self.output_max))
        self._integral = float(state.get("_integral", 0.0))
        self._last_error = float(state.get("_last_error", 0.0))
        self._last_output = float(state.get("_last_output", (self.output_min + self.output_max) / 2.0))
        self._d_filtered = float(state.get("_d_filtered", 0.0))
