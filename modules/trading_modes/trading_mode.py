# modules/trading_mode.py

from typing import List, Dict, Any, Optional
import numpy as np
import logging
import datetime

from modules.core.core import Module

class TradingModeManager(Module):
    MODES = ["safe", "normal", "aggressive", "extreme"]

    def __init__(
        self,
        initial_mode: str = "extreme",
        window: int = 50,
        log_file: Optional[str] = None,
        market_schedule: Optional[Dict[str, Any]] = None,  # NEW
        audit_log_size: int = 100,  # NEW
    ):
        assert initial_mode in self.MODES
        self.mode = initial_mode
        self.auto = True
        self.window = window
        self.stats_history: List[Dict[str, Any]] = []
        self.log_file = log_file or "logs/mode_manager.log"
        self._setup_logger()
        self.last_reason = ""
        self.last_switch_time = None

        # NEW: Decision/audit trace for every mode decision
        self._decision_trace: List[Dict[str, Any]] = []
        self._audit_log_size = audit_log_size

        # NEW: Market schedule for close/holiday awareness
        self.market_schedule = market_schedule

    def _setup_logger(self):
        self.logger = logging.getLogger("TradingModeManager")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

    def set_mode(self, mode: str):
        assert mode in self.MODES
        prev_mode = self.mode
        self.mode = mode
        self.auto = False
        self._log_switch(prev_mode, mode, "Manual override")
        self._append_trace(prev_mode, mode, "Manual override", is_auto=False)

    def set_auto(self, auto: bool):
        self.auto = auto

    def update_stats(
        self,
        trade_result: str,     # "win" or "loss"
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
        last = self.stats_history[-self.window:]
        if not last:
            return dict(win_rate=0, avg_pnl=0, consensus=0, drawdown=0, volatility=0, sharpe=0)
        win_rate = sum(1 for h in last if h["result"] == "win") / len(last)
        avg_pnl = sum(h["pnl"] for h in last) / len(last)
        consensus = sum(h["consensus"] for h in last) / len(last)
        volatility = sum(h["volatility"] for h in last) / len(last)
        drawdown = max(h["drawdown"] for h in last)
        sharpe = (
            sum(h["sharpe"] for h in last if h["sharpe"] is not None) / 
            sum(1 for h in last if h["sharpe"] is not None)
            if any(h["sharpe"] is not None for h in last) else 0
        )
        return dict(
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            consensus=consensus,
            drawdown=drawdown,
            volatility=volatility,
            sharpe=sharpe
        )

    def _market_is_open(self) -> bool:
        """
        Returns True if market is open based on the current time and self.market_schedule.
        Schedule format:
        {
            "timezone": "Europe/Berlin",
            "open_hour": 0,
            "close_hour": 23,
            "close_days": [5, 6],  # Saturday=5, Sunday=6
            "holidays": ["2024-12-25", ...]
        }
        """
        if not self.market_schedule:
            return True  # assume always open if not specified

        import pytz
        tzname = self.market_schedule.get("timezone", "UTC")
        tz = pytz.timezone(tzname)
        now = datetime.datetime.now(tz)
        day = now.weekday()  # Monday=0, Sunday=6
        hour = now.hour

        # Closed for weekends/holidays
        if day in self.market_schedule.get("close_days", [5, 6]):
            return False
        if "holidays" in self.market_schedule:
            today = now.strftime("%Y-%m-%d")
            if today in self.market_schedule["holidays"]:
                return False
        open_hour = self.market_schedule.get("open_hour", 0)
        close_hour = self.market_schedule.get("close_hour", 23)
        if hour < open_hour or hour >= close_hour:
            return False
        return True

    def decide_mode(self) -> str:
        # --------- Market schedule logic (always safe when closed) ---------
        if not self._market_is_open():
            prev_mode = self.mode
            self.mode = "safe"
            reason = "Market is closed – set to SAFE"
            if prev_mode != self.mode:
                self._log_switch(prev_mode, self.mode, reason)
                self.last_switch_time = datetime.datetime.utcnow().isoformat()
                self.last_reason = reason
                self._append_trace(prev_mode, self.mode, reason, is_auto=True)
            return self.mode

        # --------- Normal auto logic ----------
        if not self.auto:
            return self.mode

        stats = self._rolling_stats()
        prev_mode = self.mode
        reason = ""

        if stats["drawdown"] > 0.20 or stats["win_rate"] < 0.45 or stats["volatility"] > 2.5:
            self.mode = "safe"
            reason = (
                f"Switched to SAFE due to drawdown ({stats['drawdown']:.2f}), "
                f"win_rate ({stats['win_rate']:.2f}), or high volatility ({stats['volatility']:.2f})"
            )
        elif stats["win_rate"] > 0.85 and stats["avg_pnl"] > 1.5 and stats["consensus"] > 0.80:
            self.mode = "extreme"
            reason = (
                f"Switched to EXTREME: win streak ({stats['win_rate']:.2f}), "
                f"avg pnl ({stats['avg_pnl']:.2f}), consensus ({stats['consensus']:.2f})"
            )
        elif stats["win_rate"] > 0.70 and stats["avg_pnl"] > 0.8 and stats["consensus"] > 0.65:
            self.mode = "aggressive"
            reason = (
                f"Switched to AGGRESSIVE: win_rate ({stats['win_rate']:.2f}), "
                f"avg pnl ({stats['avg_pnl']:.2f}), consensus ({stats['consensus']:.2f})"
            )
        else:
            self.mode = "normal"
            reason = (
                f"Set to NORMAL: win_rate ({stats['win_rate']:.2f}), "
                f"avg pnl ({stats['avg_pnl']:.2f}), consensus ({stats['consensus']:.2f})"
            )

        if prev_mode != self.mode:
            self._log_switch(prev_mode, self.mode, reason)
            self.last_switch_time = datetime.datetime.utcnow().isoformat()
            self.last_reason = reason
            self._append_trace(prev_mode, self.mode, reason, is_auto=True)

        return self.mode

    def get_mode(self) -> str:
        return self.mode

    def get_stats(self) -> Dict[str, Any]:
        stats = self._rolling_stats()
        stats["mode"] = self.mode
        stats["auto"] = self.auto
        stats["last_switch_time"] = self.last_switch_time
        stats["last_reason"] = self.last_reason
        return stats

    def _log_switch(self, prev_mode, new_mode, reason):
        msg = f"Mode changed: {prev_mode}  {new_mode} | {reason}"
        self.logger.info(msg)
        print(f"[ModeManager] {msg}")

    def _append_trace(self, prev_mode, new_mode, reason, is_auto=True):
        entry = {
            "prev_mode": prev_mode,
            "new_mode": new_mode,
            "reason": reason,
            "auto": is_auto,
            "time": datetime.datetime.utcnow().isoformat(),
        }
        self._decision_trace.append(entry)
        if len(self._decision_trace) > self._audit_log_size:
            self._decision_trace = self._decision_trace[-self._audit_log_size:]

    def get_last_decisions(self, n=5) -> List[Dict[str, Any]]:
        return self._decision_trace[-n:]

    def reset(self):
        self.stats_history.clear()
        self.last_switch_time = None
        self.last_reason = ""
        self._decision_trace.clear()

    def step(self, trade_result=None, pnl=None, consensus=None, volatility=None, drawdown=None, sharpe=None, **kwargs):
        if trade_result is not None:
            self.update_stats(trade_result, pnl, consensus, volatility, drawdown, sharpe)
        self.decide_mode()

    def get_observation_components(self):
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
        }

    def set_state(self, state):
        self.mode = state.get("mode", "safe")
        self.auto = state.get("auto", True)
        self.stats_history = state.get("stats_history", [])
        self.last_switch_time = state.get("last_switch_time", None)
        self.last_reason = state.get("last_reason", "")
        self.market_schedule = state.get("market_schedule", None)
        self._decision_trace = state.get("decision_trace", [])
