# training/rules/rule_adapter.py
"""
Rule Adapter for Training (TRAINING-ONLY)
==========================================

Provides a clean interface for training env to query trading rules,
reusing existing config values from the repo.

Loads from:
- config/risk_policy.yaml (prop limits, session management, trade limits)
- config/timing_policy.yaml (trade spacing, entry quality, instrument overrides)
- config/base.yaml + training.yaml (instruments, timeframes)

Key semantics:
- ALL trading time semantics are Europe/Berlin local.
- Naive datetimes are treated as *local Europe/Berlin*, NOT UTC.
- Training-only: does not modify live trading behavior.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, time, timedelta, date
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from config import load_app_config, load_risk_policy
except ImportError:
    load_app_config = None
    load_risk_policy = None

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


class RuleAdapter:
    """
    Training-only adapter for trading rules.

    Reads existing config files to provide:
    - Session allowed/blocked logic (entries vs close/defend)
    - Prime/boost session detection
    - Hard close + final-exit window detection (minute-precise)
    - Max positions limits
    - Cooldown rules (with instrument overrides)
    - Prop firm limits
    """

    def __init__(self, mode: str = "training"):
        """
        Initialize rule adapter.

        Args:
            mode: Config mode ("training" or "live")
        """
        self.mode = mode
        self._risk_policy: Dict[str, Any] = {}
        self._timing_policy: Dict[str, Any] = {}
        self._app_config: Optional[Any] = None
        self._timezone: Optional[Any] = None  # ZoneInfo or compatible tzinfo

        self._load_configs()

    def _load_configs(self) -> None:
        """Load configuration from repo config files."""
        # Load risk policy
        if load_risk_policy is not None:
            try:
                self._risk_policy = load_risk_policy()
            except Exception as e:
                warnings.warn(f"Failed to load risk_policy.yaml: {e}")
                self._risk_policy = {}

        # Load timing policy
        timing_path = REPO_ROOT / "config" / "timing_policy.yaml"
        if timing_path.exists():
            try:
                import yaml

                with open(timing_path, "r", encoding="utf-8") as f:
                    self._timing_policy = yaml.safe_load(f) or {}
            except Exception as e:
                warnings.warn(f"Failed to load timing_policy.yaml: {e}")
                self._timing_policy = {}

        # Load app config (optional)
        if load_app_config is not None:
            try:
                self._app_config = load_app_config(self.mode)
            except Exception as e:
                warnings.warn(f"Failed to load app config: {e}")
                self._app_config = None

        # Setup timezone (Europe/Berlin local by default)
        tz_name = self._get_session_config().get("timezone", "Europe/Berlin")
        if ZoneInfo is not None:
            try:
                self._timezone = ZoneInfo(tz_name)
            except Exception as e:
                warnings.warn(
                    f"Failed to load ZoneInfo('{tz_name}'). "
                    f"Timezone conversions will be skipped. Error: {e}"
                )
                self._timezone = None
        else:
            self._timezone = None

    # ═══════════════════════════════════════════════════════════════
    # CONFIG ACCESSORS
    # ═══════════════════════════════════════════════════════════════

    def _get_session_config(self) -> Dict[str, Any]:
        """Get session management config from risk policy."""
        return self._risk_policy.get("session_management", {}) or {}

    def _get_prop_firm_config(self) -> Dict[str, Any]:
        """Get prop firm config from risk policy."""
        return self._risk_policy.get("prop_firm", {}) or {}

    def _get_trade_limits(self) -> Dict[str, Any]:
        """Get trade limits from risk policy."""
        return self._risk_policy.get("trade_limits", {}) or {}

    def _get_lot_sizing(self) -> Dict[str, Any]:
        """Get lot sizing config from risk policy."""
        return self._risk_policy.get("lot_sizing", {}) or {}

    def _get_limits(self) -> Dict[str, Any]:
        """Get portfolio limits from risk policy."""
        return self._risk_policy.get("limits", {}) or {}

    def _get_position_manager(self) -> Dict[str, Any]:
        """Get position manager config from risk policy."""
        return self._risk_policy.get("position_manager", {}) or {}

    def _timing_for(self, instrument: str) -> Dict[str, Any]:
        """
        Get effective timing policy for an instrument (base + instrument_overrides).
        """
        base = dict(self._timing_policy or {})
        overrides_all = base.pop("instrument_overrides", {}) or {}
        overrides = dict(overrides_all.get(instrument, {}) or {})
        base.update(overrides)
        return base

    def _to_local_time(self, dt: datetime) -> datetime:
        """
        Convert datetime to Europe/Berlin local timezone if possible.

        IMPORTANT:
        - If dt is naive, treat it as *already local* Europe/Berlin.
        - If dt is timezone-aware, convert it to Europe/Berlin.
        """
        if self._timezone is None:
            return dt

        try:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=self._timezone)  # interpret as local
            return dt.astimezone(self._timezone)
        except Exception:
            return dt

    # ═══════════════════════════════════════════════════════════════
    # CONVENIENCE PROPERTIES
    # ═══════════════════════════════════════════════════════════════

    @property
    def session_hours(self) -> Tuple[int, int]:
        """Allowed new-entry trading session hours (start, end)."""
        session = self._get_session_config()
        start = int(session.get("no_new_trades_end", 9))     # 09:00
        end = int(session.get("no_new_trades_start", 18))    # 18:00
        return (start, end)

    @property
    def boost_hours(self) -> Tuple[int, int]:
        """Prime/boost session hours (start, end)."""
        session = self._get_session_config()
        start = int(session.get("prime_hours_start", 14))    # 14:00
        end = int(session.get("prime_hours_end", 17))        # 17:00
        return (start, end)

    @property
    def hard_close_hour(self) -> int:
        """Hard close hour (local)."""
        session = self._get_session_config()
        return int(session.get("hard_close_hour", 22))

    # ═══════════════════════════════════════════════════════════════
    # SESSION RULES (ENTRIES VS CLOSE/DEFEND)
    # ═══════════════════════════════════════════════════════════════

    def is_trading_session_allowed(self, dt: datetime, instrument: str = "EURUSD") -> bool:
        """
        Check if NEW ENTRIES are allowed at the given simulation datetime.

        Uses session_management config from risk_policy.yaml:
        - no_new_trades_start: hour when new trades blocked (default 18)
        - no_new_trades_end: hour when new trades resume (default 9)

        Also respects timing_policy allow_off_hours_trading (global or per-instrument).
        """
        timing = self._timing_for(instrument)
        if bool(timing.get("allow_off_hours_trading", False)):
            return True

        session = self._get_session_config()
        block_start = int(session.get("no_new_trades_start", 18))  # 18:00
        block_end = int(session.get("no_new_trades_end", 9))       # 09:00

        local_dt = self._to_local_time(dt)
        t = local_dt.time()
        start_t = time(block_start, 0)
        end_t = time(block_end, 0)

        # Overnight block (e.g., 18:00 -> 09:00)
        if block_start > block_end:
            return not (t >= start_t or t < end_t)

        # Same-day block (rare)
        return not (start_t <= t < end_t)

    def is_boost_session(self, dt: datetime, instrument: str = "EURUSD") -> bool:
        """True if dt is within prime/boost hours (local)."""
        session = self._get_session_config()
        prime_start = int(session.get("prime_hours_start", 14))
        prime_end = int(session.get("prime_hours_end", 17))

        local_dt = self._to_local_time(dt)
        return prime_start <= local_dt.hour < prime_end

    def get_confidence_boost(self, dt: datetime, instrument: str = "EURUSD") -> float:
        """Confidence boost during prime hours, else 0.0."""
        if not self.is_boost_session(dt, instrument):
            return 0.0
        session = self._get_session_config()
        return float(session.get("prime_hours_confidence_boost", 0.15))

    def get_lot_multiplier(self, dt: datetime, instrument: str = "EURUSD") -> float:
        """Lot multiplier during prime hours, else 1.0."""
        if not self.is_boost_session(dt, instrument):
            return 1.0
        session = self._get_session_config()
        return float(session.get("prime_hours_lot_multiplier", 1.25))

    def is_hard_close_window(self, dt: datetime, instrument: str = "EURUSD") -> bool:
        """
        True if dt is in the final-exit window OR at/after hard close.

        Minute-precise:
        - final_exit_window_minutes before hard_close_hour (e.g., 21:00-22:00)
        - and any time at/after hard close (e.g., >= 22:00)
        """
        session = self._get_session_config()
        hard_close = int(session.get("hard_close_hour", 22))
        window_min = int(session.get("final_exit_window_minutes", 60))

        local_dt = self._to_local_time(dt)

        close_dt = local_dt.replace(hour=hard_close, minute=0, second=0, microsecond=0)
        window_start = close_dt - timedelta(minutes=max(0, window_min))

        return local_dt >= window_start

    def entry_allowed(self, dt: datetime, instrument: str = "EURUSD") -> bool:
        """
        Composite entry gate:
        - must be within allowed session
        - must NOT be in hard close window (final-exit window or after hard close)
        """
        if not self.is_trading_session_allowed(dt, instrument):
            return False
        if self.is_hard_close_window(dt, instrument):
            return False
        return True

    def allow_close_actions(self, dt: datetime, instrument: str = "EURUSD") -> bool:
        """Close/tighten actions are allowed when blocked (per session_management flags)."""
        session = self._get_session_config()
        return bool(session.get("allow_close_during_block", True))

    def allow_defend_actions(self, dt: datetime, instrument: str = "EURUSD") -> bool:
        """Defensive actions are allowed when blocked (per session_management flags)."""
        session = self._get_session_config()
        return bool(session.get("allow_defend_during_block", True))

    def session_key(self, dt: datetime) -> int:
        """
        Session key for 'trades per session' accounting.

        Session is defined as the new-entry window starting at no_new_trades_end
        (e.g., 09:00 local). Times before that belong to the previous session day.
        """
        session = self._get_session_config()
        start_hour = int(session.get("no_new_trades_end", 9))

        local_dt = self._to_local_time(dt)
        d = local_dt.date()
        if local_dt.time() < time(start_hour, 0):
            d = d - timedelta(days=1)
        return d.year * 10000 + d.month * 100 + d.day

    # Weekend handling (training-friendly approximation)
    def weekend_holding_allowed(self) -> bool:
        """Whether weekend holding is allowed (prop firm rule)."""
        prop = self._get_prop_firm_config()
        return bool(prop.get("allow_weekend_holding", False))

    def should_flatten_for_weekend(self, dt: datetime, instrument: str = "EURUSD") -> bool:
        """
        True if weekend holding is disallowed and it's effectively weekend risk.

        Approximation:
        - Saturday/Sunday always flatten-required.
        - Friday in hard close window also flatten-required.
        """
        if self.weekend_holding_allowed():
            return False

        local_dt = self._to_local_time(dt)
        wd = local_dt.weekday()  # Mon=0 ... Sun=6

        if wd >= 5:  # Saturday/Sunday
            return True

        # Friday: flatten required from final-exit-window onward
        if wd == 4 and self.is_hard_close_window(local_dt, instrument):
            return True

        return False


    # ═══════════════════════════════════════════════════════════════
    # POSITION LIMITS
    # ═══════════════════════════════════════════════════════════════

    def get_max_positions(self, instrument: str = "EURUSD") -> int:
        """Max concurrent positions allowed (lot_sizing.max_positions)."""
        lot_sizing = self._get_lot_sizing()
        return int(lot_sizing.get("max_positions", 2))

    def get_max_positions_per_symbol(self, instrument: str = "EURUSD") -> int:
        """Max positions per symbol (smart_position.max_positions_per_symbol)."""
        smart_pos = self._risk_policy.get("smart_position", {}) or {}
        return int(smart_pos.get("max_positions_per_symbol", 1))

    # ═══════════════════════════════════════════════════════════════
    # TRADE LIMITS
    # ═══════════════════════════════════════════════════════════════

    def get_max_trades_per_day(self, instrument: str = "EURUSD") -> int:
        """
        Max trades per day.

        Training uses trade_limits.training_mode_limit (effectively unlimited)
        unless you run adapter in non-training mode.
        """
        trade_limits = self._get_trade_limits()
        if self.mode == "training":
            return int(trade_limits.get("training_mode_limit", 9999))
        return int(trade_limits.get("max_trades_per_day", 20))

    def get_max_trades_per_session(self, instrument: str = "EURUSD") -> int:
        """Max trades per session (timing_policy, supports instrument_overrides)."""
        timing = self._timing_for(instrument)
        return int(timing.get("max_trades_per_session", 10))

    # ═══════════════════════════════════════════════════════════════
    # COOLDOWNS (BAR-BASED, M15 DEFAULT)
    # ═══════════════════════════════════════════════════════════════

    def get_cooldown_bars(self, instrument: str = "EURUSD", bar_minutes: int = 15) -> int:
        """
        Minimum bars between entries.

        Converts min_minutes_between_entries to bars (ceil-ish, min 1).
        """
        timing = self._timing_for(instrument)
        min_minutes = float(timing.get("min_minutes_between_entries", 5.0))
        bars = max(1, int((min_minutes + (bar_minutes - 1)) // bar_minutes))
        return bars

    def get_cooldown_after_loss_bars(self, instrument: str = "EURUSD", bar_minutes: int = 15) -> int:
        """Cooldown bars after a losing trade."""
        timing = self._timing_for(instrument)
        min_minutes = float(timing.get("min_minutes_after_loss", 15.0))
        bars = max(1, int((min_minutes + (bar_minutes - 1)) // bar_minutes))
        return bars

    def get_max_consecutive_losses(self) -> int:
        """Max consecutive losses before emergency regime (risk_policy.position_manager)."""
        pm = self._get_position_manager()
        return int(pm.get("max_consecutive_losses", 3))

    # ═══════════════════════════════════════════════════════════════
    # PROP FIRM LIMITS
    # ═══════════════════════════════════════════════════════════════

    def get_prop_limits(self) -> Dict[str, Any]:
        """Get prop firm limits from config (core + effective/buffered)."""
        prop = self._get_prop_firm_config()
        limits = self._get_limits()
        lot = self._get_lot_sizing()

        return {
            "daily_dd_limit": float(prop.get("daily_drawdown_limit", 0.05)),
            "max_dd_limit": float(prop.get("max_drawdown_limit", 0.10)),
            "daily_dd_safety_buffer": float(prop.get("daily_dd_safety_buffer", 0.008)),
            "max_dd_safety_buffer": float(prop.get("max_dd_safety_buffer", 0.015)),
            "effective_daily_dd": float(limits.get("max_daily_loss", 0.042)),
            "effective_max_dd": float(limits.get("max_drawdown", 0.085)),
            "emergency_close_all": float(prop.get("emergency_close_all_threshold", 0.09)),
            "account_size": float(prop.get("account_size", 100000.0)),
            "profit_target": float(prop.get("profit_target", 0.10)),
            "allow_weekend_holding": bool(prop.get("allow_weekend_holding", False)),
            "risk_per_trade_pct": float(lot.get("risk_per_trade_pct", 0.003)),
            "max_risk_per_trade_pct": float(lot.get("max_risk_per_trade_pct", 0.007)),
            "max_consecutive_losses": self.get_max_consecutive_losses(),
        }

    # ═══════════════════════════════════════════════════════════════
    # ENTRY QUALITY
    # ═══════════════════════════════════════════════════════════════

    def is_entry_quality_gate_enabled(self, instrument: str = "EURUSD") -> bool:
        """Entry quality gating enabled? (supports instrument_overrides)."""
        timing = self._timing_for(instrument)
        return bool(timing.get("entry_quality_gate_enabled", True))

    def get_entry_quality_threshold(self, instrument: str = "EURUSD") -> float:
        """Minimum entry quality threshold (supports instrument_overrides)."""
        timing = self._timing_for(instrument)
        return float(timing.get("entry_quality_min_threshold", 0.55))

    # ═══════════════════════════════════════════════════════════════
    # INSTRUMENT CONFIG (RISK POLICY)
    # ═══════════════════════════════════════════════════════════════

    def get_instrument_config(self, instrument: str) -> Dict[str, Any]:
        """Per-instrument overrides from risk_policy.smart_position.per_instrument."""
        smart_pos = self._risk_policy.get("smart_position", {}) or {}
        per_inst = smart_pos.get("per_instrument", {}) or {}
        return dict(per_inst.get(instrument, {}) or {})

    def describe(self) -> str:
        """Human-readable config summary."""
        session = self._get_session_config()
        prop = self.get_prop_limits()

        lines = [
            f"RuleAdapter ({self.mode} mode)",
            f"  Timezone: {session.get('timezone', 'Europe/Berlin')}",
            f"  New-entry hours: {session.get('no_new_trades_end', 9)}:00 - {session.get('no_new_trades_start', 18)}:00",
            f"  Prime hours: {session.get('prime_hours_start', 14)}:00 - {session.get('prime_hours_end', 17)}:00",
            f"  Hard close: {session.get('hard_close_hour', 22)}:00 (final window: {session.get('final_exit_window_minutes', 60)} min)",
            f"  Max positions: {self.get_max_positions()}",
            f"  Max trades/day: {self.get_max_trades_per_day()}",
            f"  Max trades/session: {self.get_max_trades_per_session()}",
            f"  Cooldown bars: {self.get_cooldown_bars()}",
            f"  After-loss cooldown bars: {self.get_cooldown_after_loss_bars()}",
            f"  Max consecutive losses: {self.get_max_consecutive_losses()}",
            f"  Daily DD limit: {prop['effective_daily_dd']*100:.1f}%",
            f"  Max DD limit: {prop['effective_max_dd']*100:.1f}%",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    adapter = RuleAdapter(mode="training")
    print(adapter.describe())

    test_times = [
        datetime(2024, 1, 15, 8, 0),    # off-hours (pre 09:00)
        datetime(2024, 1, 15, 10, 0),   # trading hours
        datetime(2024, 1, 15, 15, 0),   # prime hours
        datetime(2024, 1, 15, 21, 15),  # final exit window
        datetime(2024, 1, 15, 22, 15),  # after hard close
        datetime(2024, 1, 19, 21, 15),  # Friday final window
        datetime(2024, 1, 20, 10, 0),   # Saturday
    ]

    print("\nSession tests (naive datetimes treated as Europe/Berlin local):")
    for dt in test_times:
        allowed = adapter.is_trading_session_allowed(dt)
        boost = adapter.is_boost_session(dt)
        hard = adapter.is_hard_close_window(dt)
        entry_ok = adapter.entry_allowed(dt)
        wk_flat = adapter.should_flatten_for_weekend(dt)
        print(
            f"  {dt} | entry_session={allowed} boost={boost} hard_window={hard} "
            f"entry_allowed={entry_ok} weekend_flat={wk_flat} session_key={adapter.session_key(dt)}"
        )
