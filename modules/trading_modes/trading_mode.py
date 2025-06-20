# modules/trading_modes/trading_mode.py

from typing import List, Dict, Any, Optional
import numpy as np
import logging
import datetime

from modules.core.core import Module

class TradingModeManager(Module):
    MODES = ["safe", "normal", "aggressive", "extreme"]

    def __init__(
        self,
        initial_mode: str = "normal",  # FIXED: Changed from "extreme" to "normal"
        window: int = 20,              # FIXED: Reduced from 50 to 20 for faster adaptation
        log_file: Optional[str] = None,
        market_schedule: Optional[Dict[str, Any]] = None,
        audit_log_size: int = 100,
    ):
        if initial_mode not in self.MODES:
            raise ValueError(f"Invalid initial_mode '{initial_mode}'. Must be one of {self.MODES}")
        self.mode = initial_mode
        self.auto = True
        self.window = window
        self.stats_history: List[Dict[str, Any]] = []
        self.log_file = log_file or "logs/mode_manager.log"
        self._setup_logger()
        self.last_reason = ""
        self.last_switch_time = None

        # Decision/audit trace for every mode decision
        self._decision_trace: List[Dict[str, Any]] = []
        self._audit_log_size = audit_log_size

        # Market schedule for close/holiday awareness
        self.market_schedule = market_schedule
        
        # NEW: Track mode persistence to avoid rapid switching
        self._mode_persistence = 0
        self._min_persistence = 5  # Minimum steps before allowing mode change

    def _setup_logger(self):
        """Sets up the logger for the TradingModeManager to log mode changes and decisions to a file."""
        logger_name = f"TradingModeManager.{id(self)}"  # Unique logger per instance
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to prevent duplicates
        self.logger.handlers.clear()
        
        # Add file handler
        try:
            fh = logging.FileHandler(self.log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)
        except Exception as e:
            print(f"[TradingModeManager] Failed to create log file: {e}")

    def set_mode(self, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.MODES}")
        prev_mode = self.mode
        self.mode = mode
        self.auto = False
        self._mode_persistence = 0
        self._log_switch(prev_mode, mode, "Manual override")
        self._append_trace(prev_mode, mode, "Manual override", is_auto=False)

    def set_auto(self, auto: bool):
        self.auto = auto

    def update_stats(
        self,
        trade_result: str,     # "win", "loss", or "hold"
        pnl: float,
        consensus: float,      # 0-1, from voting/confidence
        volatility: float,
        drawdown: float,
        sharpe: Optional[float] = None
    ):
        self.stats_history.append({
            "result": trade_result,
            "pnl": pnl,
            "consensus": consensus,
            "volatility": volatility,
            "drawdown": drawdown,
            "sharpe": sharpe,
            "time": datetime.datetime.utcnow().isoformat()
        })
        if len(self.stats_history) > self.window * 2:
            self.stats_history = self.stats_history[-self.window*2:]

    def _rolling_stats(self):
        """Calculate rolling statistics with better handling of edge cases"""
        if not self.stats_history:
            # Return neutral stats that allow normal trading when no history
            return dict(
                win_rate=0.5,      # Assume neutral
                avg_pnl=0.0,       # No profit/loss yet
                consensus=0.5,     # Neutral consensus
                drawdown=0.0,      # No drawdown yet
                volatility=0.02,   # Assume normal volatility
                sharpe=0.0,        # Neutral Sharpe
                trade_count=0
            )
            
        last = self.stats_history[-self.window:]
        
        # Count actual trades (not holds)
        actual_trades = [h for h in last if h["result"] in ["win", "loss"]]
        trade_count = len(actual_trades)
        
        if trade_count == 0:
            # If only holds, return neutral stats
            consensus = sum(h["consensus"] for h in last) / len(last)
            volatility = sum(h["volatility"] for h in last) / len(last)
            drawdown = max((h["drawdown"] for h in last), default=0)
            
            return dict(
                win_rate=0.5,
                avg_pnl=0.0,
                consensus=consensus,
                drawdown=drawdown,
                volatility=volatility,
                sharpe=0.0,
                trade_count=0
            )
        
        # Calculate actual statistics
        win_rate = sum(1 for h in actual_trades if h["result"] == "win") / trade_count
        avg_pnl = sum(h["pnl"] for h in last) / len(last)
        consensus = sum(h["consensus"] for h in last) / len(last)
        drawdown = max((h["drawdown"] for h in last), default=0)
        volatility = sum(h["volatility"] for h in last) / len(last)
        
        sharpe_values = [h["sharpe"] for h in last if h["sharpe"] is not None]
        sharpe = sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0
        
        return dict(
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            consensus=consensus,
            drawdown=drawdown,
            volatility=volatility,
            sharpe=sharpe,
            trade_count=trade_count
        )

    def _market_is_open(self) -> bool:
        """Returns True if market is open based on the current time and self.market_schedule."""
        if not self.market_schedule:
            return True  # assume always open if not specified

        try:
            import pytz
            tzname = self.market_schedule.get("timezone", "UTC")
            tz = pytz.timezone(tzname)
            now = datetime.datetime.now(tz)
            day = now.weekday()  # Monday=0, Sunday=6
            hour = now.hour

            # Closed for weekends
            if day in self.market_schedule.get("close_days", [5, 6]):
                return False
                
            # Check holidays
            if "holidays" in self.market_schedule:
                today = now.strftime("%Y-%m-%d")
                if today in self.market_schedule["holidays"]:
                    return False
                    
            # Check hours
            open_hour = self.market_schedule.get("open_hour", 0)
            close_hour = self.market_schedule.get("close_hour", 23)
            if hour < open_hour or hour >= close_hour:
                return False
                
            return True
        except Exception:
            # If timezone handling fails, assume market is open
            return True

    def decide_mode(self) -> str:
        """Decide trading mode based on recent performance with realistic thresholds"""
        
        # Increment persistence counter
        self._mode_persistence += 1
        
        # Market schedule logic (always safe when closed)
        if not self._market_is_open():
            prev_mode = self.mode
            self.mode = "safe"
            reason = "Market is closed – set to SAFE"
            if prev_mode != self.mode:
                self._log_switch(prev_mode, self.mode, reason)
                self.last_switch_time = datetime.datetime.utcnow().isoformat()
                self.last_reason = reason
                self._append_trace(prev_mode, self.mode, reason, is_auto=True)
                self._mode_persistence = 0
            return self.mode

        # Manual mode - no auto switching
        if not self.auto:
            return self.mode

        stats = self._rolling_stats()
        prev_mode = self.mode
        reason = ""
        new_mode = self.mode  # Default to current mode
        
        # FIXED: More realistic thresholds and bootstrap-friendly logic
        
        # Check for safe mode conditions (loosened)
        if stats["drawdown"] > 0.30 or (stats["trade_count"] >= 5 and stats["win_rate"] < 0.35):
            new_mode = "safe"
            reason = (
                f"Switched to SAFE due to drawdown ({stats['drawdown']:.2f}) "
                f"or poor performance (win_rate={stats['win_rate']:.2f})"
            )
            
        # For modes requiring performance history, check trade count
        elif stats["trade_count"] < 3:
            # Not enough trades - stay in normal mode to gather data
            new_mode = "normal"
            reason = f"Staying in NORMAL - insufficient trades ({stats['trade_count']})"
            
        # Extreme mode - very relaxed requirements
        elif (stats["win_rate"] >= 0.65 and stats["avg_pnl"] > 0.5 and 
              stats["consensus"] >= 0.60 and stats["volatility"] < 0.10):
            new_mode = "extreme"
            reason = (
                f"Switched to EXTREME: good performance "
                f"(win={stats['win_rate']:.2f}, pnl={stats['avg_pnl']:.2f}, "
                f"consensus={stats['consensus']:.2f})"
            )
            
        # Aggressive mode - moderate requirements
        elif (stats["win_rate"] >= 0.55 and stats["avg_pnl"] > 0.0 and 
              stats["consensus"] >= 0.50):
            new_mode = "aggressive"
            reason = (
                f"Switched to AGGRESSIVE: decent performance "
                f"(win={stats['win_rate']:.2f}, pnl={stats['avg_pnl']:.2f}, "
                f"consensus={stats['consensus']:.2f})"
            )
            
        # Normal mode - default for average conditions
        elif stats["drawdown"] <= 0.15 and stats["volatility"] <= 0.05:
            new_mode = "normal"
            reason = (
                f"Set to NORMAL: stable conditions "
                f"(drawdown={stats['drawdown']:.2f}, vol={stats['volatility']:.3f})"
            )
            
        # Safe mode - when unsure
        else:
            new_mode = "safe"
            reason = (
                f"Defaulting to SAFE: "
                f"(win={stats['win_rate']:.2f}, drawdown={stats['drawdown']:.2f}, "
                f"vol={stats['volatility']:.3f})"
            )

        # Check if we should actually switch (persistence requirement)
        if new_mode != self.mode:
            if self._mode_persistence < self._min_persistence:
                # Not enough persistence, don't switch yet
                return self.mode
            else:
                # Switch mode
                self.mode = new_mode
                self._mode_persistence = 0
                self._log_switch(prev_mode, new_mode, reason)
                self.last_switch_time = datetime.datetime.utcnow().isoformat()
                self.last_reason = reason
                self._append_trace(prev_mode, new_mode, reason, is_auto=True)

        return self.mode

    def get_mode(self) -> str:
        return self.mode

    def get_stats(self) -> Dict[str, Any]:
        stats = self._rolling_stats()
        stats["mode"] = self.mode
        stats["auto"] = self.auto
        stats["last_switch_time"] = self.last_switch_time
        stats["last_reason"] = self.last_reason
        stats["mode_persistence"] = self._mode_persistence
        return stats

    def _log_switch(self, prev_mode, new_mode, reason):
        msg = f"Mode changed: {prev_mode} → {new_mode} | {reason}"
        self.logger.info(msg)
        print(f"[ModeManager] {msg}")

    def _append_trace(self, prev_mode, new_mode, reason, is_auto=True):
        entry = {
            "prev_mode": prev_mode,
            "new_mode": new_mode,
            "reason": reason,
            "auto": is_auto,
            "time": datetime.datetime.utcnow().isoformat(),
            "stats": self._rolling_stats(),
        }
        self._decision_trace.append(entry)
        if len(self._decision_trace) > self._audit_log_size:
            self._decision_trace = self._decision_trace[-self._audit_log_size:]

    def get_last_decisions(self, n=5) -> List[Dict[str, Any]]:
        return self._decision_trace[-n:]

    def reset(self):
        """Reset to initial state"""
        self.stats_history.clear()
        self.last_switch_time = None
        self.last_reason = ""
        self._decision_trace.clear()
        self._mode_persistence = 0
        # Reset to normal mode for fresh start
        self.mode = "normal"

    def step(self, trade_result=None, pnl=None, consensus=None, volatility=None, 
             drawdown=None, sharpe=None, **kwargs):
        """Update stats and decide mode"""
        if trade_result is not None:
            # Allow 'hold' as a valid result
            if trade_result not in ["win", "loss", "hold"]:
                trade_result = "hold"
                
            # Provide defaults for missing values
            pnl = pnl if pnl is not None else 0.0
            consensus = consensus if consensus is not None else 0.5
            volatility = volatility if volatility is not None else 0.02
            drawdown = drawdown if drawdown is not None else 0.0
            
            self.update_stats(trade_result, pnl, consensus, volatility, drawdown, sharpe)
            
        self.decide_mode()

    def get_observation_components(self):
        """Return one-hot encoding of current mode"""
        arr = np.zeros(len(self.MODES), np.float32)
        arr[self.MODES.index(self.mode)] = 1.0
        return arr

    def get_state(self):
        return {
            "mode": self.mode,
            "auto": self.auto,
            "stats_history": self.stats_history,
            "last_switch_time": self.last_switch_time,
            "last_reason": self.last_reason,
            "market_schedule": self.market_schedule,
            "decision_trace": self._decision_trace,
            "_mode_persistence": self._mode_persistence,
            "_min_persistence": self._min_persistence,
        }

    def set_state(self, state):
        self.mode = state.get("mode", "normal")
        self.auto = state.get("auto", True)
        self.stats_history = state.get("stats_history", [])
        self.last_switch_time = state.get("last_switch_time", None)
        self.last_reason = state.get("last_reason", "")
        self.market_schedule = state.get("market_schedule", None)
        self._decision_trace = state.get("decision_trace", [])
        self._mode_persistence = state.get("_mode_persistence", 0)
        self._min_persistence = state.get("_min_persistence", 5)