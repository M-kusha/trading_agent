#modules/trading_mode.py

from typing import List, Dict, Any, Optional
import numpy as np
import logging
import datetime

from modules.core.core import Module

class TradingModeManager(Module):
    MODES = ["safe", "normal", "aggressive", "extreme"]

    def __init__(self, initial_mode: str = "safe", window: int = 50, log_file: Optional[str] = None):
        assert initial_mode in self.MODES
        self.mode = initial_mode
        self.auto = True  # Default to auto mode
        self.window = window  # Rolling stats window size
        self.stats_history: List[Dict[str, Any]] = []
        self.log_file = log_file or "mode_manager.log"
        self._setup_logger()
        self.last_reason = ""
        self.last_switch_time = None

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

    def decide_mode(self) -> str:
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
        msg = f"Mode changed: {prev_mode} → {new_mode} | {reason}"
        self.logger.info(msg)
        print(f"[ModeManager] {msg}")

    def reset(self):
        self.stats_history.clear()
        self.last_switch_time = None
        self.last_reason = ""

    def step(self, trade_result=None, pnl=None, consensus=None, volatility=None, drawdown=None, sharpe=None, **kwargs):
        if trade_result is not None:
            self.update_stats(trade_result, pnl, consensus, volatility, drawdown, sharpe)
        self.decide_mode()

    def get_observation_components(self):
        arr = np.zeros(len(self.MODES), np.float32)
        arr[self.MODES.index(self.mode)] = 1.0
        return arr
