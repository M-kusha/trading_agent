# training/rules/safety_filter.py
"""
Safety Filter for Training (TRAINING-ONLY)
==========================================

Filters raw agent actions through safety constraints before execution.
Enforces session rules, prop limits, gates, cooldowns, trade/session limits,
and position limits.

Key design points:
- Uses SIMULATION datetime (not wall-clock).
- Treats naive datetimes as local (Europe/Berlin) where possible.
- Blocks NEW entries during:
  - off-session windows (unless allow_off_hours_trading is enabled upstream)
  - final exit window before hard close
  - weekend (if weekend holding is disallowed)
- Allows CLOSE/HOLD behavior during blocked windows.
  * If current_position_direction is provided, reversals can be blocked explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from training.contracts.obs_contract import OBS_IDX, OBS_SIZE, GATE_SLOTS
from training.rules.rule_adapter import RuleAdapter
from training.rules.prop_constraints import PropConstraints


@dataclass
class FilteredAction:
    """
    Result of safety filtering.

    Attributes:
        direction: Filtered direction (-1=short, 0=flat, 1=long)
        size_mult: Position size multiplier (0.0 to 1.0)
        blocked: True if action was blocked (for NEW entry / reversal)
        reasons: List of blocking/adjustment reasons
        original_direction: Original parsed direction
        original_size: Original parsed size (0..1)
        forced_flatten: True if the filter forced a flat action
    """
    direction: int = 0
    size_mult: float = 1.0
    blocked: bool = False
    reasons: List[str] = field(default_factory=list)
    original_direction: int = 0
    original_size: float = 1.0
    forced_flatten: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "size_mult": self.size_mult,
            "blocked": self.blocked,
            "reasons": list(self.reasons),
            "original_direction": self.original_direction,
            "original_size": self.original_size,
            "forced_flatten": self.forced_flatten,
        }


@dataclass
class ViolationCounters:
    """Tracks violation counts across episode for metrics."""

    session_blocked: int = 0
    final_exit_blocked: int = 0
    weekend_blocked: int = 0
    prop_dd_blocked: int = 0
    trade_limit_blocked: int = 0
    session_trade_limit_blocked: int = 0
    cooldown_blocked: int = 0
    loss_streak_blocked: int = 0
    max_positions_blocked: int = 0
    gate_blocked: int = 0
    quality_blocked: int = 0
    size_clamped: int = 0
    forced_flatten: int = 0
    total_blocked: int = 0
    total_filtered: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "session_blocked": self.session_blocked,
            "final_exit_blocked": self.final_exit_blocked,
            "weekend_blocked": self.weekend_blocked,
            "prop_dd_blocked": self.prop_dd_blocked,
            "trade_limit_blocked": self.trade_limit_blocked,
            "session_trade_limit_blocked": self.session_trade_limit_blocked,
            "cooldown_blocked": self.cooldown_blocked,
            "loss_streak_blocked": self.loss_streak_blocked,
            "max_positions_blocked": self.max_positions_blocked,
            "gate_blocked": self.gate_blocked,
            "quality_blocked": self.quality_blocked,
            "size_clamped": self.size_clamped,
            "forced_flatten": self.forced_flatten,
            "total_blocked": self.total_blocked,
            "total_filtered": self.total_filtered,
        }

    def reset(self) -> None:
        for k in self.to_dict().keys():
            setattr(self, k, 0)


class SafetyFilter:
    """
    Filters agent actions through safety constraints.

    Checks (in order) for NEW entries / reversals:
    1. Weekend holding rule (if disallowed)
    2. Final exit window (block new entries)
    3. Session allowed (trading hours)
    4. Prop constraints (DD limits)
    5. Trade limits (per-day, per-session)
    6. Cooldown (bars since last trade / loss)
    7. Loss streak cap
    8. Max positions (global)
    9. Gates from observation (risk_gate, memory_gate, mode_entry_allowed)
    10. Entry quality threshold
    11. Position size scaling (risk multiplier + prime-hours boost)
    """

    def __init__(
        self,
        rule_adapter: RuleAdapter,
        prop_constraints: PropConstraints,
        direction_long_threshold: float = 0.3,
        direction_short_threshold: float = -0.3,
        enable_gate_checks: bool = True,
        enable_quality_check: bool = True,
        strict_mode: bool = False,  # terminate on serious violations
        force_flatten_on_weekend: bool = True,
        force_flatten_in_final_exit_window: bool = False,
    ):
        self.rule_adapter = rule_adapter
        self.prop_constraints = prop_constraints
        self.direction_long_threshold = float(direction_long_threshold)
        self.direction_short_threshold = float(direction_short_threshold)
        self.enable_gate_checks = bool(enable_gate_checks)
        self.enable_quality_check = bool(enable_quality_check)
        self.strict_mode = bool(strict_mode)

        self.force_flatten_on_weekend = bool(force_flatten_on_weekend)
        self.force_flatten_in_final_exit_window = bool(force_flatten_in_final_exit_window)

        self.counters = ViolationCounters()

    def reset(self) -> None:
            """Reset counters for new episode."""
            self.counters.reset()

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _as_obs_1d(obs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if obs is None:
            return None
        arr = np.asarray(obs)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        return None

    def _session_key(self, dt: datetime) -> int:
        # Prefer adapter implementation if present (supports timezone semantics).
        if hasattr(self.rule_adapter, "session_key"):
            try:
                return int(self.rule_adapter.session_key(dt))  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback: YYYYMMDD of (possibly local) dt
        try:
            local_dt = self.rule_adapter._to_local_time(dt)  # type: ignore[attr-defined]
        except Exception:
            local_dt = dt
        return local_dt.year * 10000 + local_dt.month * 100 + local_dt.day

    def _in_final_exit_window(self, dt: datetime, instrument: str) -> bool:
        if hasattr(self.rule_adapter, "is_hard_close_window"):
            try:
                return bool(self.rule_adapter.is_hard_close_window(dt, instrument))
            except Exception:
                pass

        # Conservative fallback based on configured hard_close_hour + window_minutes.
        try:
            session = self.rule_adapter._get_session_config()  # type: ignore[attr-defined]
        except Exception:
            session = {}

        hard_close = int(session.get("hard_close_hour", 22))
        window_min = int(session.get("final_exit_window_minutes", 60))

        try:
            local_dt = self.rule_adapter._to_local_time(dt)  # type: ignore[attr-defined]
        except Exception:
            local_dt = dt

        close_dt = local_dt.replace(hour=hard_close, minute=0, second=0, microsecond=0)
        window_start = close_dt - timedelta(minutes=max(0, window_min))

        # IMPORTANT: window_start and everything after it is "hard-close window"
        return local_dt >= window_start


    def _weekend_policy_disallows_holding(self) -> bool:
        # Prefer adapter methods if present.
        if hasattr(self.rule_adapter, "_get_prop_firm_config"):
            try:
                prop = self.rule_adapter._get_prop_firm_config()  # type: ignore[attr-defined]
                return not bool(prop.get("allow_weekend_holding", True))
            except Exception:
                return False
        # Fallback: unknown -> do not enforce
        return False

    def _should_block_for_weekend(self, dt: datetime, instrument: str) -> bool:
        if hasattr(self.rule_adapter, "should_flatten_for_weekend"):
            try:
                return bool(self.rule_adapter.should_flatten_for_weekend(dt, instrument))  # type: ignore[attr-defined]
            except Exception:
                pass

        if not self._weekend_policy_disallows_holding():
            return False

        try:
            local_dt = self.rule_adapter._to_local_time(dt)  # type: ignore[attr-defined]
        except Exception:
            local_dt = dt

        # Block entries on Saturday/Sunday, and from Friday final-exit-window onwards.
        if local_dt.weekday() in (5, 6):
            return True
        if local_dt.weekday() == 4:  # Friday
            return self._in_final_exit_window(dt, instrument) or (local_dt.hour >= int(getattr(self.rule_adapter, "hard_close_hour", 22)))
        return False

    def _entry_allowed_by_session(self, dt: datetime, instrument: str) -> bool:
        # Prefer adapter.entry_allowed if available (may include final-exit-window logic).
        if hasattr(self.rule_adapter, "entry_allowed"):
            try:
                return bool(self.rule_adapter.entry_allowed(dt, instrument))  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback to classic session check.
        try:
            return bool(self.rule_adapter.is_trading_session_allowed(dt, instrument))
        except Exception:
            return True

    def _get_session_trade_limit(self, instrument: str) -> int:
        try:
            return int(self.rule_adapter.get_max_trades_per_session(instrument))
        except Exception:
            return 10

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def filter_action(
        self,
        direction_score: float,
        size_score: float,
        obs: Optional[np.ndarray],
        current_dt: datetime,
        current_step: int,
        instrument: str,
        current_positions: int,
        has_position_in_instrument: bool,
        current_position_direction: Optional[int] = None,
    ) -> FilteredAction:
        """
        Filter a raw action through safety constraints.
        """
        self.counters.total_filtered += 1

        # Parse direction
        if direction_score > self.direction_long_threshold:
            raw_direction = 1
        elif direction_score < self.direction_short_threshold:
            raw_direction = -1
        else:
            raw_direction = 0

        # Parse size robustly -> [0,1]
        s = float(size_score)
        if -1.0 <= s <= 1.0:
            raw_size = (s + 1.0) / 2.0
        else:
            raw_size = s
        raw_size = float(np.clip(raw_size, 0.0, 1.0))

        result = FilteredAction(
            direction=raw_direction,
            size_mult=raw_size,
            original_direction=raw_direction,
            original_size=raw_size,
        )

        obs1 = self._as_obs_1d(obs)

        # Flat action: allow (env interprets as close/hold)
        if raw_direction == 0:
            return result

        # Determine NEW entry / reversal
        is_reversal = False
        if has_position_in_instrument and current_position_direction in (-1, 1):
            is_reversal = (raw_direction != current_position_direction)

        is_new_entry = (not has_position_in_instrument) or is_reversal

        # Holding same direction -> allow
        if not is_new_entry:
            return result

        # ─────────────────────────────────────────
        # NEW ENTRY / REVERSAL CHECKS
        # ─────────────────────────────────────────

        # 0) Weekend holding rule
        if self._should_block_for_weekend(current_dt, instrument):
            if (has_position_in_instrument and self.force_flatten_on_weekend) or (not has_position_in_instrument):
                result.direction = 0
                result.size_mult = 0.0
                result.blocked = True
                result.forced_flatten = bool(has_position_in_instrument)
                result.reasons.append("weekend_blocked")
                if result.forced_flatten:
                    result.reasons.append("forced_flatten_weekend")
                    self.counters.forced_flatten += 1
                self.counters.weekend_blocked += 1
                self.counters.total_blocked += 1
                return result

        # 1) Final exit window (block new entries; optionally force flatten)
        if self._in_final_exit_window(current_dt, instrument):
            if has_position_in_instrument and self.force_flatten_in_final_exit_window:
                result.direction = 0
                result.size_mult = 0.0
                result.blocked = True
                result.forced_flatten = True
                result.reasons.append("final_exit_window")
                result.reasons.append("forced_flatten_final_exit")
                self.counters.forced_flatten += 1
            else:
                result.direction = 0
                result.size_mult = 0.0
                result.blocked = True
                result.reasons.append("final_exit_window")
            self.counters.final_exit_blocked += 1
            self.counters.total_blocked += 1
            return result

        # 2) Session check (trading hours / no-new-trades window)
        if not self._entry_allowed_by_session(current_dt, instrument):
            result.direction = 0
            result.size_mult = 0.0
            result.blocked = True
            result.reasons.append("session_blocked")
            self.counters.session_blocked += 1
            self.counters.total_blocked += 1
            return result

        # 3) Prop DD violations
        v = self.prop_constraints.violations()
        if v.get("daily_dd_exceeded", False):
            result.direction = 0
            result.size_mult = 0.0
            result.blocked = True
            result.reasons.append("daily_dd_exceeded")
            self.counters.prop_dd_blocked += 1
            self.counters.total_blocked += 1
            return result

        if v.get("total_dd_exceeded", False):
            result.direction = 0
            result.size_mult = 0.0
            result.blocked = True
            result.reasons.append("total_dd_exceeded")
            self.counters.prop_dd_blocked += 1
            self.counters.total_blocked += 1
            return result

        # 4) Trade limits (per-day)
        if v.get("max_trades_exceeded", False):
            result.direction = 0
            result.size_mult = 0.0
            result.blocked = True
            result.reasons.append("max_trades_per_day_exceeded")
            self.counters.trade_limit_blocked += 1
            self.counters.total_blocked += 1
            return result

        # 5) Trade limits (per-session) - authoritative via PropConstraints
        if v.get("max_trades_session_exceeded", False):
            result.direction = 0
            result.size_mult = 0.0
            result.blocked = True
            result.reasons.append("max_trades_per_session_exceeded")
            self.counters.session_trade_limit_blocked += 1
            self.counters.total_blocked += 1
            return result

        # 6) Cooldown (bars since last trade / loss)
        if self.prop_constraints.is_cooldown_active(current_step):
            result.direction = 0
            result.size_mult = 0.0
            result.blocked = True
            result.reasons.append("cooldown_active")
            self.counters.cooldown_blocked += 1
            self.counters.total_blocked += 1
            return result

        # 7) Loss streak cap
        if v.get("loss_streak_exceeded", False):
            result.direction = 0
            result.size_mult = 0.0
            result.blocked = True
            result.reasons.append("loss_streak_exceeded")
            self.counters.loss_streak_blocked += 1
            self.counters.total_blocked += 1
            return result

        # 8) Max positions (global)
        max_positions = int(self.rule_adapter.get_max_positions(instrument))
        if current_positions >= max_positions:
            result.direction = 0
            result.size_mult = 0.0
            result.blocked = True
            result.reasons.append("max_positions_reached")
            self.counters.max_positions_blocked += 1
            self.counters.total_blocked += 1
            return result

        # 9) Observation gates
        if self.enable_gate_checks and obs1 is not None and obs1.shape == (OBS_SIZE,):
            risk_gate = float(obs1[GATE_SLOTS["risk_gate"]])
            if risk_gate < 0.5:
                result.direction = 0
                result.size_mult = 0.0
                result.blocked = True
                result.reasons.append("risk_gate_closed")
                self.counters.gate_blocked += 1
                self.counters.total_blocked += 1
                return result

            memory_gate = float(obs1[GATE_SLOTS["memory_gate"]])
            if memory_gate < 0.5:
                result.direction = 0
                result.size_mult = 0.0
                result.blocked = True
                result.reasons.append("memory_gate_closed")
                self.counters.gate_blocked += 1
                self.counters.total_blocked += 1
                return result

            mode_entry = float(obs1[GATE_SLOTS["mode_entry_allowed"]])
            if mode_entry < 0.5:
                result.direction = 0
                result.size_mult = 0.0
                result.blocked = True
                result.reasons.append("mode_entry_blocked")
                self.counters.gate_blocked += 1
                self.counters.total_blocked += 1
                return result

        # 10) Entry quality gate (instrument-aware)
        if self.enable_quality_check and self.rule_adapter.is_entry_quality_gate_enabled(instrument):
            if obs1 is not None and obs1.shape == (OBS_SIZE,):
                quality_idx = OBS_IDX.get("mode_entry_quality", 58)
                entry_quality = float(obs1[quality_idx])
                threshold = float(self.rule_adapter.get_entry_quality_threshold(instrument))
                if entry_quality < threshold:
                    result.direction = 0
                    result.size_mult = 0.0
                    result.blocked = True
                    result.reasons.append(f"quality_too_low:{entry_quality:.2f}<{threshold:.2f}")
                    self.counters.quality_blocked += 1
                    self.counters.total_blocked += 1
                    return result

        # 11) Size scaling based on drawdown risk multiplier
        risk_mult = float(np.clip(self.prop_constraints.get_risk_multiplier(), 0.0, 1.0))
        if risk_mult < 1.0:
            result.size_mult = float(np.clip(result.size_mult * risk_mult, 0.0, 1.0))
            result.reasons.append(f"size_scaled:{risk_mult:.2f}")
            self.counters.size_clamped += 1

        # Prime-hours boost (size only, capped at 1.0)
        if self.rule_adapter.is_boost_session(current_dt, instrument):
            lot_mult = float(self.rule_adapter.get_lot_multiplier(current_dt, instrument))
            if lot_mult != 1.0:
                result.size_mult = float(min(1.0, result.size_mult * lot_mult))
                result.reasons.append(f"prime_boost:{lot_mult:.2f}")

        return result


    def filter_multi_instrument(
        self,
        action: np.ndarray,
        obs: Optional[np.ndarray],
        current_dt: datetime,
        current_step: int,
        instruments: List[str],
        positions: Dict[str, Any],  # instrument -> Position-like
    ) -> Dict[str, FilteredAction]:
        """
        Filter actions for multiple instruments.

        Args:
            action: Raw action array [dir1, size1, dir2, size2, ...]
            obs: Current observation (64 dims)
            current_dt: Current simulation datetime
            current_step: Current episode step
            instruments: List of instruments
            positions: Dict of current positions by instrument

        Returns:
            Dict mapping instrument -> FilteredAction
        """
        results: Dict[str, FilteredAction] = {}
        current_positions = len(positions)

        for i, inst in enumerate(instruments):
            j = i * 2
            if j + 1 >= len(action):
                break

            direction_score = float(action[j])
            size_score = float(action[j + 1])
            has_position = inst in positions

            # Attempt to infer current position direction if present
            cur_dir: Optional[int] = None
            if has_position:
                pos = positions.get(inst)
                # Common conventions: pos.direction or pos.side or pos.is_long
                try:
                    if hasattr(pos, "direction"):
                        cur_dir = int(getattr(pos, "direction"))
                    elif hasattr(pos, "side"):
                        side = str(getattr(pos, "side")).lower()
                        cur_dir = 1 if "long" in side or "buy" in side else (-1 if "short" in side or "sell" in side else None)
                    elif hasattr(pos, "is_long"):
                        cur_dir = 1 if bool(getattr(pos, "is_long")) else -1
                except Exception:
                    cur_dir = None

            result = self.filter_action(
                direction_score=direction_score,
                size_score=size_score,
                obs=obs,
                current_dt=current_dt,
                current_step=current_step,
                instrument=inst,
                current_positions=current_positions,
                has_position_in_instrument=has_position,
                current_position_direction=cur_dir,
            )

            results[inst] = result

            # Update position count if this action opens a new position
            if (not has_position) and (result.direction != 0) and (not result.blocked):
                current_positions += 1

        return results

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of violations for logging/metrics."""
        total = max(1, self.counters.total_filtered)
        return {
            "counters": self.counters.to_dict(),
            "block_rate": self.counters.total_blocked / total,
            "session_block_rate": self.counters.session_blocked / total,
            "final_exit_block_rate": self.counters.final_exit_blocked / total,
            "weekend_block_rate": self.counters.weekend_blocked / total,
            "prop_block_rate": self.counters.prop_dd_blocked / total,
            "gate_block_rate": self.counters.gate_blocked / total,
            "quality_block_rate": self.counters.quality_blocked / total,
        }

    def should_terminate(self) -> Tuple[bool, str]:
        """
        Check if episode should terminate due to serious violations.
        """
        if self.prop_constraints.should_terminate_episode():
            return True, "emergency_drawdown"

        if self.strict_mode:
            v = self.prop_constraints.violations()
            if v.get("total_dd_exceeded", False):
                return True, "max_drawdown_exceeded"

        return False, ""


if __name__ == "__main__":
    # Self-test (lightweight)
    from datetime import datetime as _dt

    adapter = RuleAdapter(mode="training")
    prop = PropConstraints.from_rule_adapter(adapter)
    prop.reset(100_000.0, _dt(2024, 1, 15, 10, 0))

    safety = SafetyFilter(rule_adapter=adapter, prop_constraints=prop)

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[OBS_IDX["risk_gate"]] = 1.0
    obs[OBS_IDX["memory_gate"]] = 1.0
    obs[OBS_IDX["mode_entry_allowed"]] = 1.0
    obs[OBS_IDX["mode_entry_quality"]] = 0.60

    r1 = safety.filter_action(
        direction_score=0.5,
        size_score=0.5,
        obs=obs,
        current_dt=_dt(2024, 1, 15, 15, 0),  # prime hours
        current_step=10,
        instrument="EURUSD",
        current_positions=0,
        has_position_in_instrument=False,
    )
    print("Prime hours:", r1.to_dict())

    r2 = safety.filter_action(
        direction_score=0.5,
        size_score=0.5,
        obs=obs,
        current_dt=_dt(2024, 1, 15, 20, 0),  # typically blocked hours
        current_step=11,
        instrument="EURUSD",
        current_positions=0,
        has_position_in_instrument=False,
    )
    print("After hours:", r2.to_dict())

    print("\nSummary:", safety.get_violation_summary())
