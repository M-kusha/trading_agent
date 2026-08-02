# envs/prop_firm/session/timing.py
# pyright: reportAttributeAccessIssue=false
"""
Session timing mixin for PropFirmTradingEnv.

Contains all session and time-related methods.

Fixes / upgrades:
- Correct time-window logic for both same-day windows and cross-midnight windows
- Session precedence: no_new_trades > prime > regular
- Anchor session date for any session window that crosses midnight (no_new_trades and prime)
- Normalize timeframe key lookup to reduce silent "missing TF" issues
- Minor DST robustness when localizing timestamps
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from envs.core.env_types import PropFirmConfig


class SessionTimingMixin:
    """Mixin providing session and timing methods.

    Expected attributes from PropFirmTradingEnv:
    - config: PropFirmConfig
    - data: Dict[str, Dict[str, pd.DataFrame]]
    - current_step: int
    - tz: ZoneInfo
    - equity: float
    - _current_day: Optional[date]
    - _current_session_key: Optional[Tuple[date, str]]
    - _session_trades: int
    - day_start_balance: float
    - daily_trades: int
    - daily_pnl: float
    """

    config: "PropFirmConfig"
    data: Dict[str, Dict[str, pd.DataFrame]]
    current_step: int
    tz: "ZoneInfo"
    equity: float
    _current_day: Optional[date]
    _current_session_key: Optional[Tuple[date, str]]
    _session_trades: int
    day_start_balance: float
    daily_trades: int
    daily_pnl: float

    # ---------------------------
    # Timeframe helpers
    # ---------------------------

    def _tf_minutes(self) -> int:
        """Get minutes per bar for primary timeframe."""
        from envs.core.shared_utils import DEFAULT_PRIMARY_TIMEFRAME, timeframe_to_minutes

        tf = (self.config.primary_timeframe or DEFAULT_PRIMARY_TIMEFRAME)
        tf = str(tf).upper().strip()
        return timeframe_to_minutes(tf)

    # ---------------------------
    # Datetime extraction
    # ---------------------------

    def _get_bar_dt(self, instrument: str) -> Optional[datetime]:
        """Get bar datetime at current step."""
        return self._get_bar_dt_at(instrument, self.current_step)

    def _get_bar_dt_at(self, instrument: str, step_idx: int) -> Optional[datetime]:
        """Get bar datetime at specific step."""
        from envs.core.shared_utils import DEFAULT_PRIMARY_TIMEFRAME

        tf = str(self.config.primary_timeframe or DEFAULT_PRIMARY_TIMEFRAME).upper().strip()

        inst_data = self.data.get(instrument, {})
        df = inst_data.get(tf)

        # Defensive fallback: if TF key casing differs, try a direct scan once
        if (df is None or df.empty) and isinstance(inst_data, dict) and inst_data:
            for k, v in inst_data.items():
                if str(k).upper().strip() == tf:
                    df = v
                    break

        if df is None or df.empty:
            return None

        idx = int(np.clip(step_idx, 0, len(df) - 1))

        ts: Any = None
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[idx]
        else:
            for col in ("time", "timestamp", "datetime", "date"):
                if col in df.columns:
                    ts = df[col].iloc[idx]
                    break

        if ts is None:
            return None

        t = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t):
            return None

        try:
            if getattr(t, "tzinfo", None) is None:
                # DST-safe localization: prefer shifting nonexistent forward; ambiguous choose earliest
                t = t.tz_localize(self.tz, nonexistent="shift_forward", ambiguous="infer")
            else:
                t = t.tz_convert(self.tz)
        except Exception:
            # Fall back to simple behavior if pandas can't infer DST
            if getattr(t, "tzinfo", None) is None:
                t = t.tz_localize(self.tz)
            else:
                t = t.tz_convert(self.tz)

        return t.to_pydatetime()

    # ---------------------------
    # Window logic (CRITICAL)
    # ---------------------------

    @staticmethod
    def _time_in_window(t: time, start: time, end: time) -> bool:
        """
        Returns True if time `t` is in [start, end), correctly handling:
        - Disabled windows: start == end -> False
        - Same-day windows: start < end -> start <= t < end
        - Cross-midnight: start > end -> t >= start OR t < end
        """
        if start == end:
            return False
        if start < end:
            return start <= t < end
        return (t >= start) or (t < end)

    def _is_weekend(self, dt: datetime) -> bool:
        """Check if datetime falls on weekend."""
        return dt.weekday() >= 5

    def _in_no_new_trades_window(self, dt: datetime) -> bool:
        """Check if in no-new-trades window (may cross midnight)."""
        start = self.config.no_new_trades_start
        end = self.config.no_new_trades_end
        t = dt.timetz().replace(tzinfo=None)
        return self._time_in_window(t, start, end)

    def _in_prime_window(self, dt: datetime) -> bool:
        """Check if in prime trading window (may cross midnight if configured)."""
        start = self.config.prime_start
        end = self.config.prime_end
        t = dt.timetz().replace(tzinfo=None)
        return self._time_in_window(t, start, end)

    def _in_final_exit_window(self, dt: datetime) -> bool:
        """Check if in final exit window before hard close."""
        # Use dt.replace to preserve timezone behavior
        hc_t = self.config.hard_close_time
        hc = dt.replace(hour=hc_t.hour, minute=hc_t.minute, second=0, microsecond=0)
        start = hc - timedelta(minutes=int(self.config.final_exit_window_minutes))
        return start <= dt < hc

    def _at_or_after_hard_close(self, dt: datetime) -> bool:
        """Check if at or after hard close time."""
        hc_t = self.config.hard_close_time
        hc = dt.replace(hour=hc_t.hour, minute=hc_t.minute, second=0, microsecond=0)
        return dt >= hc

    # ---------------------------
    # Session naming / anchoring
    # ---------------------------

    def _session_name(self, dt: datetime) -> str:
        """
        Get session name for datetime.

        Precedence is important:
        - no_new_trades must win over prime if the windows overlap.
        """
        if self._in_no_new_trades_window(dt):
            return "no_new_trades"
        if self._in_prime_window(dt):
            return "prime"
        return "regular"

    @staticmethod
    def _anchor_for_window(dt: datetime, start: time, end: time) -> date:
        """
        For windows that cross midnight (start > end), anchor the after-midnight
        portion (t < end) to the previous date, preventing midnight reset exploits.
        """
        if start == end:
            return dt.date()
        crosses = start > end
        if not crosses:
            return dt.date()

        t = dt.timetz().replace(tzinfo=None)
        if t < end:
            return (dt - timedelta(days=1)).date()
        return dt.date()

    def _session_anchor_date(self, dt: datetime) -> date:
        """
        Anchor date based on the active session's window.
        - no_new_trades anchored using its window
        - prime anchored using its window (if you ever configure it to cross midnight)
        - regular anchored to calendar date
        """
        sname = self._session_name(dt)
        if sname == "no_new_trades":
            return self._anchor_for_window(dt, self.config.no_new_trades_start, self.config.no_new_trades_end)
        if sname == "prime":
            return self._anchor_for_window(dt, self.config.prime_start, self.config.prime_end)
        return dt.date()

    # ---------------------------
    # dt=None fallback
    # ---------------------------

    def _bars_per_day(self) -> int:
        """Get bars per trading day for primary timeframe."""
        from envs.core.shared_utils import DEFAULT_PRIMARY_TIMEFRAME, bars_per_day_for_timeframe

        tf = str(self.config.primary_timeframe or DEFAULT_PRIMARY_TIMEFRAME).upper().strip()
        return bars_per_day_for_timeframe(tf)

    def _maybe_roll_day_session(self, dt: Optional[datetime]) -> None:
        """Roll day/session counters if crossing day or session boundary."""
        if dt is None:
            bpd = max(1, self._bars_per_day())
            day_idx = int(self.current_step // bpd)
            cur_day = (datetime(2000, 1, 1, tzinfo=self.tz) + timedelta(days=day_idx)).date()

            session = "regular"
            key = (cur_day, f"bar_day_{day_idx}_{session}")

            if self._current_session_key != key:
                self._current_session_key = key
                self._session_trades = 0
                # Session budget reset (v5.5)
                self.session_start_balance = self.equity
                self.session_pnl = 0.0
                self.session_consecutive_losses = 0
                self.session_start_step = self.current_step

            if self._current_day != cur_day:
                self._current_day = cur_day
                # CRIT FIX: Use equity (includes unrealized PnL), not balance (cash only)
                self.day_start_balance = self.equity
                self.daily_trades = 0
                self.daily_pnl = 0.0
            return

        cur_day = dt.date()
        if self._current_day != cur_day:
            self._current_day = cur_day
            # CRIT FIX: Use equity (includes unrealized PnL), not balance (cash only)
            self.day_start_balance = self.equity
            self.daily_trades = 0
            self.daily_pnl = 0.0

        sname = self._session_name(dt)
        anchor = self._session_anchor_date(dt)
        key = (anchor, sname)
        if self._current_session_key != key:
            self._current_session_key = key
            self._session_trades = 0
            # Session budget reset (v5.5)
            self.session_start_balance = self.equity
            self.session_pnl = 0.0
            self.session_consecutive_losses = 0
            self.session_start_step = self.current_step
