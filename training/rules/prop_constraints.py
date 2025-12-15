# training/rules/prop_constraints.py
"""
Prop Firm Constraint State (TRAINING-ONLY)
==========================================

Fast, deterministic tracking of prop-firm-like constraints during training.

Design goals:
- Uses ONLY simulation datetime (no wall-clock)
- Treats naive datetimes as Europe/Berlin local time
- Tracks daily drawdown (reset at local midnight)
- Tracks total drawdown (from initial equity by default; optional trailing-from-peak)
- Enforces:
  - trades per day
  - trades per session (session-key tracking)
  - cooldowns (bars since last entry / since last loss)
  - max consecutive losses (wired from config via RuleAdapter)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


BERLIN_TZ = "Europe/Berlin"


def _to_berlin_wallclock(dt: datetime) -> datetime:
    """
    Normalize simulation datetime to Europe/Berlin wall-clock.

    Rules:
    - If dt is tz-aware: convert to Europe/Berlin and drop tzinfo (naive local wall-clock)
    - If dt is naive: assume it's already Europe/Berlin wall-clock (do NOT treat as UTC)
    """
    if dt.tzinfo is None:
        return dt

    if ZoneInfo is None:
        # If zoneinfo unavailable, keep as-is but drop tzinfo to avoid mixed arithmetic.
        return dt.replace(tzinfo=None)

    return dt.astimezone(ZoneInfo(BERLIN_TZ)).replace(tzinfo=None)


@dataclass
class PropConstraintState:
    """
    Mutable state for prop constraint tracking.
    All values are updated each step based on simulation time.
    """

    # Equity tracking
    initial_equity: float = 100_000.0
    day_start_equity: float = 100_000.0
    day_peak_equity: float = 100_000.0
    total_peak_equity: float = 100_000.0
    current_equity: float = 100_000.0

    # Day / session tracking (fast integer keys)
    current_day_key: int = 0          # YYYYMMDD (local)
    current_session_key: int = 0      # YYYYMMDD of the session start day (local)

    # Drawdown tracking
    daily_drawdown: float = 0.0                # fraction, peak->current within day
    daily_drawdown_peak: float = 0.0           # worst daily DD observed (across days)

    total_drawdown_initial: float = 0.0        # fraction, initial->current
    total_drawdown_trailing: float = 0.0       # fraction, total peak->current
    total_drawdown: float = 0.0                # effective total DD used for enforcement
    total_drawdown_peak: float = 0.0           # worst effective total DD observed

    # Trade tracking
    trades_today: int = 0
    trades_this_session: int = 0
    total_trades: int = 0
    last_trade_step: int = -999
    last_loss_step: int = -999

    # Consecutive tracking
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0

    def copy(self) -> "PropConstraintState":
        return PropConstraintState(
            initial_equity=self.initial_equity,
            day_start_equity=self.day_start_equity,
            day_peak_equity=self.day_peak_equity,
            total_peak_equity=self.total_peak_equity,
            current_equity=self.current_equity,
            current_day_key=self.current_day_key,
            current_session_key=self.current_session_key,
            daily_drawdown=self.daily_drawdown,
            daily_drawdown_peak=self.daily_drawdown_peak,
            total_drawdown_initial=self.total_drawdown_initial,
            total_drawdown_trailing=self.total_drawdown_trailing,
            total_drawdown=self.total_drawdown,
            total_drawdown_peak=self.total_drawdown_peak,
            trades_today=self.trades_today,
            trades_this_session=self.trades_this_session,
            total_trades=self.total_trades,
            last_trade_step=self.last_trade_step,
            last_loss_step=self.last_loss_step,
            consecutive_losses=self.consecutive_losses,
            max_consecutive_losses=self.max_consecutive_losses,
        )


@dataclass
class PropConstraints:
    """
    Prop firm constraint manager for training.

    Notes:
    - daily_dd_limit: effective daily drawdown stop (e.g., 0.042 = 4.2%)
    - max_dd_limit: effective max drawdown stop for NEW trades (e.g., 0.085 = 8.5%)
    - emergency_dd: emergency close-all / terminate threshold (e.g., 0.09 = 9%)
    - trailing_drawdown: if True, enforce max_dd_limit against peak->valley instead of initial->current
    """

    # Limits (loaded from config via RuleAdapter)
    daily_dd_limit: float = 0.042
    max_dd_limit: float = 0.085
    emergency_dd: float = 0.09

    trailing_drawdown: bool = False

    # Trade limits
    max_trades_per_day: int = 9999          # training default: unlimited
    max_trades_per_session: int = 9999      # timing-policy style session cap

    # Cooldowns (bars)
    cooldown_bars: int = 1
    cooldown_after_loss_bars: int = 1

    # Loss streak protection
    max_consecutive_losses: int = 5

    # Session definition (local wall-clock hours)
    # Defaults match canonical risk_policy.yaml: no_new_trades_end=9, no_new_trades_start=18
    session_start_hour: int = 9
    session_end_hour: int = 18

    # State
    state: PropConstraintState = field(default_factory=PropConstraintState)

    def reset(self, initial_equity: float, start_dt: datetime) -> None:
        dt = _to_berlin_wallclock(start_dt)
        day_key = self._dt_to_day_key(dt)
        session_key = self._dt_to_session_key(dt, self.session_start_hour)

        self.state = PropConstraintState(
            initial_equity=initial_equity,
            day_start_equity=initial_equity,
            day_peak_equity=initial_equity,
            total_peak_equity=initial_equity,
            current_equity=initial_equity,
            current_day_key=day_key,
            current_session_key=session_key,
        )

    def update(
        self,
        step_dt: datetime,
        equity: float,
        current_step: int,
        trade_opened: bool = False,
        trade_closed_pnl: Optional[float] = None,
    ) -> None:
        dt = _to_berlin_wallclock(step_dt)
        day_key = self._dt_to_day_key(dt)
        session_key = self._dt_to_session_key(dt, self.session_start_hour)

        # Day/session transitions
        if day_key != self.state.current_day_key:
            self._handle_day_change(day_key, equity)

        if session_key != self.state.current_session_key:
            self._handle_session_change(session_key)

        # Equity
        self.state.current_equity = equity

        # Peaks
        if equity > self.state.day_peak_equity:
            self.state.day_peak_equity = equity
        if equity > self.state.total_peak_equity:
            self.state.total_peak_equity = equity

        # Drawdowns
        self._calculate_drawdowns()

        # Trade opening
        if trade_opened:
            self.state.trades_today += 1
            self.state.trades_this_session += 1
            self.state.total_trades += 1
            self.state.last_trade_step = current_step

        # Trade closing
        if trade_closed_pnl is not None:
            if trade_closed_pnl < 0:
                self.state.consecutive_losses += 1
                self.state.last_loss_step = current_step
                if self.state.consecutive_losses > self.state.max_consecutive_losses:
                    self.state.max_consecutive_losses = self.state.consecutive_losses
            else:
                self.state.consecutive_losses = 0

    def _handle_day_change(self, new_day_key: int, equity: float) -> None:
        # Carry forward worst daily DD (also tracked continuously, but keep this safe)
        if self.state.daily_drawdown > self.state.daily_drawdown_peak:
            self.state.daily_drawdown_peak = self.state.daily_drawdown

        self.state.current_day_key = new_day_key
        self.state.day_start_equity = equity
        self.state.day_peak_equity = equity
        self.state.trades_today = 0
        # Do not reset consecutive_losses: streak continues across days

    def _handle_session_change(self, new_session_key: int) -> None:
        self.state.current_session_key = new_session_key
        self.state.trades_this_session = 0

    def _calculate_drawdowns(self) -> None:
        eq = self.state.current_equity

        # Daily drawdown: from max(day_start, day_peak) => conservative, prevents "resetting" mid-day.
        day_ref = max(self.state.day_start_equity, self.state.day_peak_equity)
        self.state.daily_drawdown = max(0.0, (day_ref - eq) / day_ref) if day_ref > 0 else 0.0
        if self.state.daily_drawdown > self.state.daily_drawdown_peak:
            self.state.daily_drawdown_peak = self.state.daily_drawdown

        # Total DD from initial (non-trailing)
        init_ref = self.state.initial_equity
        self.state.total_drawdown_initial = max(0.0, (init_ref - eq) / init_ref) if init_ref > 0 else 0.0

        # Trailing DD from peak
        peak_ref = max(self.state.initial_equity, self.state.total_peak_equity)
        self.state.total_drawdown_trailing = max(0.0, (peak_ref - eq) / peak_ref) if peak_ref > 0 else 0.0

        # Effective total DD used for enforcement
        self.state.total_drawdown = (
            self.state.total_drawdown_trailing if self.trailing_drawdown else self.state.total_drawdown_initial
        )

        if self.state.total_drawdown > self.state.total_drawdown_peak:
            self.state.total_drawdown_peak = self.state.total_drawdown

    @staticmethod
    def _dt_to_day_key(dt: datetime) -> int:
        return dt.year * 10000 + dt.month * 100 + dt.day

    @staticmethod
    def _dt_to_session_key(dt: datetime, session_start_hour: int) -> int:
        """
        Session-key = YYYYMMDD of the session *start day*.

        If time is before session_start_hour (e.g., 02:00), we attribute it to the previous day session.
        This makes counting stable even if simulation includes off-hours steps.
        """
        if dt.hour < session_start_hour:
            prev = dt.replace(hour=12, minute=0, second=0, microsecond=0)  # safe anchor
            # subtract one day without importing timedelta repeatedly
            from datetime import timedelta
            prev = prev - timedelta(days=1)
            return prev.year * 10000 + prev.month * 100 + prev.day
        return dt.year * 10000 + dt.month * 100 + dt.day

    # ═══════════════════════════════════════════════════════════════
    # VIOLATION CHECKS
    # ═══════════════════════════════════════════════════════════════

    def violations(self) -> Dict[str, Any]:
        total_dd = self.state.total_drawdown
        return {
            # Daily drawdown
            "daily_dd_exceeded": self.state.daily_drawdown >= self.daily_dd_limit,
            "daily_dd_magnitude": self.state.daily_drawdown,
            "daily_dd_limit": self.daily_dd_limit,
            "daily_dd_margin": self.daily_dd_limit - self.state.daily_drawdown,

            # Total drawdown (effective)
            "total_dd_exceeded": total_dd >= self.max_dd_limit,
            "total_dd_magnitude": total_dd,
            "total_dd_limit": self.max_dd_limit,
            "total_dd_margin": self.max_dd_limit - total_dd,

            # Extra transparency
            "total_dd_initial": self.state.total_drawdown_initial,
            "total_dd_trailing": self.state.total_drawdown_trailing,
            "total_dd_mode": "trailing" if self.trailing_drawdown else "initial",

            # Emergency
            "emergency_dd_exceeded": total_dd >= self.emergency_dd,

            # Trade limits (day)
            "max_trades_exceeded": self.state.trades_today >= self.max_trades_per_day,
            "trades_today": self.state.trades_today,
            "trades_remaining_today": max(0, self.max_trades_per_day - self.state.trades_today),

            # Trade limits (session)
            "max_trades_session_exceeded": self.state.trades_this_session >= self.max_trades_per_session,
            "trades_this_session": self.state.trades_this_session,
            "trades_remaining_session": max(0, self.max_trades_per_session - self.state.trades_this_session),
            "session_key": self.state.current_session_key,

            # Loss streak
            "loss_streak_exceeded": self.state.consecutive_losses >= self.max_consecutive_losses,
            "consecutive_losses": self.state.consecutive_losses,
            "loss_streak_limit": self.max_consecutive_losses,
        }

    def is_cooldown_active(self, current_step: int) -> bool:
        bars_since_trade = current_step - self.state.last_trade_step
        if bars_since_trade < self.cooldown_bars:
            return True

        bars_since_loss = current_step - self.state.last_loss_step
        if bars_since_loss < self.cooldown_after_loss_bars:
            return True

        return False

    def trading_allowed(self, current_step: int) -> bool:
        v = self.violations()

        if v["daily_dd_exceeded"]:
            return False
        if v["total_dd_exceeded"]:
            return False

        if v["max_trades_exceeded"]:
            return False
        if v["max_trades_session_exceeded"]:
            return False

        if self.is_cooldown_active(current_step):
            return False

        if v["loss_streak_exceeded"]:
            return False

        return True

    def should_terminate_episode(self) -> bool:
        v = self.violations()
        return bool(v["emergency_dd_exceeded"])

    def get_risk_multiplier(self) -> float:
        dd = self.state.total_drawdown

        if dd < 0.02:
            return 1.0
        if dd < 0.04:
            return 0.75
        if dd < 0.06:
            return 0.50
        if dd < 0.08:
            return 0.25
        return 0.10

    @classmethod
    def from_rule_adapter(cls, adapter: Any) -> "PropConstraints":
        """
        Create PropConstraints with limits from RuleAdapter.

        Adapter expectations (best-effort):
        - adapter.get_prop_limits() -> dict (effective_daily_dd, effective_max_dd, emergency_close_all, trailing_drawdown?)
        - adapter.get_max_trades_per_day()
        - adapter.get_max_trades_per_session()  (optional)
        - adapter.get_cooldown_bars()
        - adapter.get_cooldown_after_loss_bars()
        - adapter.get_max_consecutive_losses()  (optional; otherwise use config position_manager.max_consecutive_losses via adapter limits if present)
        """
        limits = adapter.get_prop_limits()

        # Trade caps
        max_trades_per_day = adapter.get_max_trades_per_day()
        max_trades_per_session = (
            adapter.get_max_trades_per_session()
            if hasattr(adapter, "get_max_trades_per_session")
            else max_trades_per_day
        )

        # Loss streak cap (wire from config if adapter provides it)
        max_consecutive_losses = (
            adapter.get_max_consecutive_losses()
            if hasattr(adapter, "get_max_consecutive_losses")
            else limits.get("max_consecutive_losses", 5)
        )

        trailing = bool(limits.get("trailing_drawdown", False))

        return cls(
            daily_dd_limit=float(limits.get("effective_daily_dd", 0.042)),
            max_dd_limit=float(limits.get("effective_max_dd", 0.085)),
            emergency_dd=float(limits.get("emergency_close_all", 0.09)),
            trailing_drawdown=trailing,
            max_trades_per_day=int(max_trades_per_day),
            max_trades_per_session=int(max_trades_per_session),
            cooldown_bars=int(adapter.get_cooldown_bars()),
            cooldown_after_loss_bars=int(adapter.get_cooldown_after_loss_bars()),
            max_consecutive_losses=int(max_consecutive_losses),
        )

    def describe(self, current_step: Optional[int] = None) -> str:
        v = self.violations()
        step_for_check = current_step if current_step is not None else 10**9  # ignore cooldown in summary by default
        lines = [
            "PropConstraints State:",
            f"  Equity: {self.state.current_equity:,.0f} (initial: {self.state.initial_equity:,.0f})",
            f"  Daily DD: {self.state.daily_drawdown*100:.2f}% (limit: {self.daily_dd_limit*100:.1f}%)",
            f"  Total DD ({v['total_dd_mode']}): {self.state.total_drawdown*100:.2f}% (limit: {self.max_dd_limit*100:.1f}%)",
            f"  Trades today: {self.state.trades_today} / {self.max_trades_per_day}",
            f"  Trades session({self.state.current_session_key}): {self.state.trades_this_session} / {self.max_trades_per_session}",
            f"  Loss streak: {self.state.consecutive_losses} / {self.max_consecutive_losses}",
            f"  Trading allowed (step={step_for_check}): {self.trading_allowed(step_for_check)}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    from datetime import datetime, timezone

    pc = PropConstraints(
        daily_dd_limit=0.042,
        max_dd_limit=0.085,
        max_trades_per_day=3,
        max_trades_per_session=2,
        cooldown_bars=1,
        cooldown_after_loss_bars=2,
        max_consecutive_losses=3,
    )

    # Reset with tz-aware time (will convert to Berlin wall-clock)
    pc.reset(100_000.0, datetime(2024, 1, 15, 8, 55, tzinfo=timezone.utc))
    print("After reset:")
    print(pc.describe())

    # Simulate some trading
    pc.update(datetime(2024, 1, 15, 10, 0), 100_500.0, 1, trade_opened=True)
    pc.update(datetime(2024, 1, 15, 11, 0), 100_200.0, 2, trade_closed_pnl=-300)
    pc.update(datetime(2024, 1, 15, 12, 0), 99_800.0, 3, trade_opened=True)

    print("\nAfter events:")
    print(pc.describe(current_step=3))
    print(f"\nViolations: {pc.violations()}")
