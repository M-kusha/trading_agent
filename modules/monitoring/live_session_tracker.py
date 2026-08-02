# modules/monitoring/live_session_tracker.py
"""
Live Session Tracker - Tracks session-level metrics for live trading.

This module tracks the same session-level metrics that PropFirmTradingEnv tracks
in training, ensuring train/live parity for the governor observation block (v5.5).

Publishes to InfoBus:
- 'governor_state': Dict with all 8 governor observation values
- 'trade_statistics': Dict with consecutive losses, session trades, etc.

Required for live PPO observation construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np


@dataclass
class LiveSessionConfig:
    """Configuration for live session tracking."""
    # Session windows (trading hours in session timezone)
    session_timezone: str = "Europe/Berlin"
    session_start_hour: int = 8  # 08:00 local time
    session_end_hour: int = 22  # 22:00 local time
    
    # Session budget limits (should match live config)
    session_loss_limit_pct: float = 0.99  # Default: effectively infinite
    session_consecutive_loss_limit: int = 99  # Default: effectively infinite
    max_trades_per_session: int = 50
    
    # Governor limits
    loss_layer_stop: int = 5
    max_consecutive_losses: int = 10


class LiveSessionTracker:
    """
    Tracks session-level metrics for live trading governor observation.
    
    Mirrors the session tracking done by PropFirmTradingEnv to ensure
    train/live parity in the observation vector.
    
    Usage:
        tracker = LiveSessionTracker(smart_bus)
        tracker.on_trade_close(pnl, is_win)  # Call when trade closes
        tracker.update()  # Call periodically to publish to InfoBus
    """
    
    def __init__(
        self,
        smart_bus: Any,
        config: Optional[LiveSessionConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.smart_bus = smart_bus
        self.config = config or LiveSessionConfig()
        self.logger = logger or logging.getLogger("LiveSessionTracker")
        
        self.tz = ZoneInfo(self.config.session_timezone)
        
        # Session state
        self.session_start_balance: float = 0.0
        self.session_pnl: float = 0.0
        self.session_trades: int = 0
        self.session_consecutive_losses: int = 0
        self.session_start_time: Optional[datetime] = None
        
        # Overall state
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.total_trades: int = 0
        
        # Pending order tracking
        self.pending_entry_start_time: Optional[datetime] = None
        self.max_pending_bars: int = 3  # Matches training env
        
        # Last update (date only for session rollover)
        self._last_session_date: Optional[date] = None
        
    def initialize_from_account(self, balance: float) -> None:
        """Initialize session with current account balance."""
        self.session_start_balance = balance
        self.session_start_time = datetime.now(self.tz)
        self._last_session_date = self.session_start_time.date()
        self.logger.info(f"Session initialized: balance={balance:.2f}")
        
    def on_trade_close(self, pnl: float, *, is_win: Optional[bool] = None) -> None:
        """
        Called when a trade closes. Updates all session metrics.
        
        Args:
            pnl: Realized PnL of the trade
            is_win: Whether the trade was profitable (defaults to pnl > 0)
        """
        if is_win is None:
            is_win = pnl > 0
            
        # Update session metrics
        self.session_pnl += pnl
        self.session_trades += 1
        self.total_trades += 1
        
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.session_consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.session_consecutive_losses += 1
            
        self.logger.debug(
            f"Trade closed: pnl={pnl:.2f}, session_pnl={self.session_pnl:.2f}, "
            f"consec_losses={self.consecutive_losses}, session_consec={self.session_consecutive_losses}"
        )
        
    def on_pending_entry_created(self) -> None:
        """Called when a pending entry order is created."""
        self.pending_entry_start_time = datetime.now(self.tz)
        
    def on_pending_entry_filled_or_cancelled(self) -> None:
        """Called when pending entry is filled or cancelled."""
        self.pending_entry_start_time = None
        
    def check_session_rollover(self) -> bool:
        """
        Check if we need to roll over to a new session.
        Returns True if rollover occurred.
        """
        now = datetime.now(self.tz)
        current_date = now.date()
        current_hour = now.hour
        
        # Check if we're in a new session (day changed or passed session start)
        should_rollover = False
        
        if self._last_session_date is None:
            should_rollover = True
        elif current_date > self._last_session_date:
            # New day - check if we're past session start
            if current_hour >= self.config.session_start_hour:
                should_rollover = True
        elif current_date == self._last_session_date:
            # Same day - rollover only if we crossed session start
            if self.session_start_time and self.session_start_time.hour < self.config.session_start_hour:
                if current_hour >= self.config.session_start_hour:
                    should_rollover = True
                    
        if should_rollover:
            self._rollover_session()
            
        return should_rollover
        
    def _rollover_session(self) -> None:
        """Reset session-level metrics for a new session."""
        # Get current balance from InfoBus
        try:
            account_state = self.smart_bus.get("account_state", "LiveSessionTracker") or {}
            current_balance = float(account_state.get("balance", 100000))
        except Exception:
            current_balance = self.session_start_balance or 100000
            
        old_session_pnl = self.session_pnl
        old_session_trades = self.session_trades
        
        self.session_start_balance = current_balance
        self.session_pnl = 0.0
        self.session_trades = 0
        self.session_consecutive_losses = 0
        self.session_start_time = datetime.now(self.tz)
        self._last_session_date = self.session_start_time.date()
        
        self.logger.info(
            f"Session rolled over: prev_pnl={old_session_pnl:.2f}, "
            f"prev_trades={old_session_trades}, new_balance={current_balance:.2f}"
        )
        
    def get_governor_state(self) -> Dict[str, float]:
        """
        Get the governor state dict for observation construction.
        
        Returns same fields as PropFirmTradingEnv._get_governor_state().
        """
        cfg = self.config
        
        # Loss layer ratio
        loss_layer_ratio = self.consecutive_losses / max(cfg.loss_layer_stop, 1)
        loss_layer_level = min(self.consecutive_losses, 5) / 5.0
        
        # Win streak
        win_streak_ratio = min(self.consecutive_wins, 5) / 5.0
        
        # Session PnL headroom
        if self.session_start_balance > 0:
            session_pnl_pct = self.session_pnl / self.session_start_balance
        else:
            session_pnl_pct = 0.0
        headroom = (cfg.session_loss_limit_pct + session_pnl_pct) / max(cfg.session_loss_limit_pct, 0.001)
        
        # Session trade budget
        session_trade_budget = 1.0 - (self.session_trades / max(cfg.max_trades_per_session, 1))
        
        # Session consecutive loss ratio
        session_consec_ratio = self.session_consecutive_losses / max(cfg.session_consecutive_loss_limit, 1)
        
        # Session progress (time-based)
        session_progress = 0.5
        if self.session_start_time:
            now = datetime.now(self.tz)
            session_duration_hours = self.config.session_end_hour - self.config.session_start_hour
            elapsed = (now - self.session_start_time).total_seconds() / 3600
            session_progress = min(1.0, max(0.0, elapsed / session_duration_hours))
            
        # Pending order progress
        pending_progress = 0.0
        if self.pending_entry_start_time:
            now = datetime.now(self.tz)
            # Assume 15-minute bars, so max_pending_bars * 15 minutes
            max_pending_minutes = self.max_pending_bars * 15
            elapsed_minutes = (now - self.pending_entry_start_time).total_seconds() / 60
            pending_progress = min(1.0, elapsed_minutes / max_pending_minutes)
            
        return {
            "loss_layer_ratio": float(np.clip(loss_layer_ratio, 0.0, 1.0)),
            "loss_layer_level": float(np.clip(loss_layer_level, 0.0, 1.0)),
            "win_streak_ratio": float(np.clip(win_streak_ratio, 0.0, 1.0)),
            "session_pnl_headroom": float(np.clip(headroom, 0.0, 2.0)),
            "session_trade_budget": float(np.clip(session_trade_budget, 0.0, 1.0)),
            "session_consec_loss_ratio": float(np.clip(session_consec_ratio, 0.0, 1.0)),
            "session_progress": float(session_progress),
            "pending_order_progress": float(pending_progress),
        }
        
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get trade statistics dict for InfoBus publishing."""
        return {
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "session_trades": self.session_trades,
            "session_pnl": self.session_pnl,
            "session_consecutive_losses": self.session_consecutive_losses,
            "session_start_balance": self.session_start_balance,
            "total_trades": self.total_trades,
        }
        
    def update_and_publish(self) -> None:
        """
        Check for session rollover and publish all metrics to InfoBus.
        Call this periodically (e.g., every bar or every minute).
        """
        # Check for session rollover
        self.check_session_rollover()
        
        # Publish governor state
        governor_state = self.get_governor_state()
        self.smart_bus.set(
            "governor_state",
            governor_state,
            module="LiveSessionTracker",
            thesis="Governor observation state for PPO",
        )
        
        # Publish trade statistics
        trade_stats = self.get_trade_statistics()
        self.smart_bus.set(
            "trade_statistics",
            trade_stats,
            module="LiveSessionTracker",
            thesis="Trade statistics for observation builder",
        )
        
        # Publish session metrics (for other modules that might need it)
        session_metrics = {
            "session_pnl": self.session_pnl,
            "session_trades": self.session_trades,
            "session_start_balance": self.session_start_balance,
            "session_start_time": self.session_start_time.isoformat() if self.session_start_time else None,
        }
        self.smart_bus.set(
            "session_metrics",
            session_metrics,
            module="LiveSessionTracker",
            thesis="Session metrics for budget tracking",
        )


__all__ = ["LiveSessionConfig", "LiveSessionTracker"]
