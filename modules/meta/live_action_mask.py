#!/usr/bin/env python3

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dtime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.utils import simulation_time as simclock

try:
    from config import get_trade_limits
    _TRADE_LIMITS = get_trade_limits()
except ImportError:
    _TRADE_LIMITS = {"max_trades_per_day": 20, "max_trades_per_session": 10}


@dataclass
class LiveMaskConfig:

    daily_drawdown_limit: float = 0.05
    max_drawdown_limit: float = 0.10
    daily_dd_safety_buffer: float = 0.008
    max_dd_safety_buffer: float = 0.015


    max_trades_per_day: int = field(default_factory=lambda: _TRADE_LIMITS.get("max_trades_per_day", 20))
    max_trades_per_session: int = field(default_factory=lambda: _TRADE_LIMITS.get("max_trades_per_session", 10))
    max_consecutive_losses: int = 3


    min_minutes_between_entries: int = 5
    min_minutes_after_loss: int = 15


    enforce_hard_rules: bool = False


    no_new_trades_start: dtime = dtime(18, 0)
    no_new_trades_end: dtime = dtime(9, 0)
    hard_close_time: dtime = dtime(22, 0)


    size_buckets: Tuple[float, ...] = (0.35, 0.60, 0.85, 1.10)

    @property
    def n_actions(self) -> int:
        return 2 * len(self.size_buckets) + 2


class LiveActionMaskBuilder:

    def __init__(self, config: Optional[LiveMaskConfig] = None) -> None:
        self.config = config or LiveMaskConfig()
        self._K = len(self.config.size_buckets)


        self._ACTION_HOLD = 0
        self._ACTION_LONG_START = 1
        self._ACTION_SHORT_START = 1 + self._K
        self._ACTION_CLOSE = 1 + 2 * self._K
        self._N_ACTIONS = 2 * self._K + 2

        self.logger = logging.getLogger("LiveActionMask")

    def get_action_mask(
        self,
        has_position: bool,
        trade_open_allowed: Optional[bool] = None,
        has_pending_entry: bool = False,
        has_pending_exit: bool = False,
        current_dd: float = 0.0,
        daily_dd: float = 0.0,
        daily_trades: int = 0,
        session_trades: int = 0,
        consecutive_losses: int = 0,
        last_entry_time: Optional[datetime] = None,
        last_loss_time: Optional[datetime] = None,
        current_time: Optional[datetime] = None,
    ) -> np.ndarray:
        mask = np.ones(self._N_ACTIONS, dtype=np.bool_)


        can_enter = (not has_position) and (not has_pending_entry)


        if trade_open_allowed is False:
            can_enter = False


        if consecutive_losses >= int(self.config.max_consecutive_losses):
            can_enter = False


        if self.config.enforce_hard_rules and can_enter:
            # Read the shared clock rather than the wall clock. In live these are
            # identical; under simulation the wall clock would make the
            # minutes-between-entries and minutes-after-loss rules meaningless,
            # because thousands of bars replay inside one real second.
            now = current_time or simclock.now()
            hard_allowed, _ = self._hard_entry_allowed(
                current_dd=current_dd,
                daily_dd=daily_dd,
                daily_trades=daily_trades,
                session_trades=session_trades,
                consecutive_losses=consecutive_losses,
                last_entry_time=last_entry_time,
                last_loss_time=last_loss_time,
                current_time=now,
            )
            if not hard_allowed:
                can_enter = False

        if not can_enter:

            mask[self._ACTION_LONG_START: self._ACTION_LONG_START + self._K] = False
            mask[self._ACTION_SHORT_START: self._ACTION_SHORT_START + self._K] = False


        if not has_position and not has_pending_exit:
            mask[self._ACTION_CLOSE] = False


        mask[self._ACTION_HOLD] = True

        return mask

    def _hard_entry_allowed(
        self,
        current_dd: float,
        daily_dd: float,
        daily_trades: int,
        session_trades: int,
        consecutive_losses: int,
        last_entry_time: Optional[datetime],
        last_loss_time: Optional[datetime],
        current_time: datetime,
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []


        max_dd_threshold = self.config.max_drawdown_limit - self.config.max_dd_safety_buffer
        if current_dd >= max_dd_threshold:
            reasons.append(f"max_dd_breach({current_dd:.2%} >= {max_dd_threshold:.2%})")

        daily_dd_threshold = self.config.daily_drawdown_limit - self.config.daily_dd_safety_buffer
        if daily_dd >= daily_dd_threshold:
            reasons.append(f"daily_dd_breach({daily_dd:.2%} >= {daily_dd_threshold:.2%})")


        if daily_trades >= self.config.max_trades_per_day:
            reasons.append(f"daily_trades_limit({daily_trades} >= {self.config.max_trades_per_day})")

        if session_trades >= self.config.max_trades_per_session:
            reasons.append(f"session_trades_limit({session_trades} >= {self.config.max_trades_per_session})")


        if consecutive_losses >= self.config.max_consecutive_losses:
            reasons.append(f"consecutive_losses({consecutive_losses} >= {self.config.max_consecutive_losses})")


        t = current_time.time()


        if self._in_no_new_trades_window(t):
            reasons.append("no_new_trades_window")


        if t >= self.config.hard_close_time:
            reasons.append("after_hard_close")


        if current_time.weekday() >= 5:
            reasons.append("weekend")


        if last_entry_time is not None:
            minutes_since_entry = (current_time - last_entry_time).total_seconds() / 60
            if minutes_since_entry < self.config.min_minutes_between_entries:
                reasons.append(f"entry_spacing({minutes_since_entry:.1f}m < {self.config.min_minutes_between_entries}m)")

        if last_loss_time is not None:
            minutes_since_loss = (current_time - last_loss_time).total_seconds() / 60
            if minutes_since_loss < self.config.min_minutes_after_loss:
                reasons.append(f"post_loss_cooldown({minutes_since_loss:.1f}m < {self.config.min_minutes_after_loss}m)")

        return len(reasons) == 0, reasons

    def _in_no_new_trades_window(self, t: dtime) -> bool:
        start = self.config.no_new_trades_start
        end = self.config.no_new_trades_end

        if start > end:
            return t >= start or t < end
        else:
            return start <= t < end

    def decode_action(self, action_id: int) -> Tuple[str, float]:
        a = int(action_id)

        if a == self._ACTION_HOLD:
            return "hold", 0.0
        if a == self._ACTION_CLOSE:
            return "close", 0.0
        if self._ACTION_LONG_START <= a < self._ACTION_SHORT_START:
            idx = a - self._ACTION_LONG_START
            return "long", float(self.config.size_buckets[idx])
        if self._ACTION_SHORT_START <= a < self._ACTION_CLOSE:
            idx = a - self._ACTION_SHORT_START
            return "short", float(self.config.size_buckets[idx])

        return "hold", 0.0

    def get_mask_summary(self, mask: np.ndarray) -> Dict[str, Any]:
        long_allowed = mask[self._ACTION_LONG_START:self._ACTION_SHORT_START].sum()
        short_allowed = mask[self._ACTION_SHORT_START:self._ACTION_CLOSE].sum()

        return {
            "hold_allowed": bool(mask[self._ACTION_HOLD]),
            "long_allowed": int(long_allowed),
            "short_allowed": int(short_allowed),
            "close_allowed": bool(mask[self._ACTION_CLOSE]),
            "total_allowed": int(mask.sum()),
            "total_actions": len(mask),
        }
