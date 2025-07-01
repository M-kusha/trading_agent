
#
# modules/strategy/curriculum_planner_plus.py
from __future__ import annotations
import logging
import numpy as np
from typing import Dict,List
from modules.core.core import Module

class CurriculumPlannerPlus(Module):
    def __init__(self, window: int=10, debug=True):
        self.window = window
        self.debug = debug
        self._history: List[Dict[str,float]] = []
        self._step_count = 0
        
        # FIX: Track cumulative metrics for better learning
        self._total_trades = 0
        self._total_wins = 0
        self._cumulative_pnl = 0.0

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"CurriculumPlannerPlus_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/curriculum/curriculum_planner.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"CurriculumPlannerPlus initialized - window={window}")

    def reset(self):
        self._history.clear()
        self._total_trades = 0
        self._total_wins = 0
        self._cumulative_pnl = 0.0
        self._step_count = 0
        self.logger.info("CurriculumPlannerPlus reset - all metrics cleared")

    def step(self, **kwargs):
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record_episode(self, summary: Dict[str,float]):
        """Record episode with comprehensive validation and logging"""
        try:
            # Validate summary
            if not isinstance(summary, dict):
                self.logger.error(f"Invalid summary type: {type(summary)}")
                return
                
            self.logger.debug(f"Recording episode summary: {summary}")
            
            self._history.append(summary)
            if len(self._history) > self.window:
                removed = self._history.pop(0)
                self.logger.debug(f"Removed old episode: {removed}")
            
            # FIX: Update cumulative metrics with validation
            if "total_trades" in summary:
                trades = summary["total_trades"]
                if isinstance(trades, (int, float)) and trades >= 0:
                    self._total_trades += trades
                else:
                    self.logger.warning(f"Invalid total_trades: {trades}")
                    
            if "wins" in summary:
                wins = summary["wins"]
                if isinstance(wins, (int, float)) and wins >= 0:
                    self._total_wins += wins
                else:
                    self.logger.warning(f"Invalid wins: {wins}")
                    
            if "pnl" in summary:
                pnl = summary["pnl"]
                if isinstance(pnl, (int, float)) and not np.isnan(pnl):
                    self._cumulative_pnl += pnl
                else:
                    self.logger.warning(f"Invalid pnl: {pnl}")
            
            # Log episode statistics
            self.logger.info(f"Episode recorded: total_trades={self._total_trades}, total_wins={self._total_wins}, cumulative_pnl=€{self._cumulative_pnl:.2f}")
            
            # Log detailed statistics periodically
            if len(self._history) % 5 == 0:
                self._log_curriculum_stats()
                
        except Exception as e:
            self.logger.error(f"Error recording episode: {e}")

    def _log_curriculum_stats(self):
        """Log detailed curriculum statistics"""
        try:
            if not self._history:
                return
                
            win_rates = [e.get("win_rate", 0) for e in self._history if "win_rate" in e]
            durations = [e.get("avg_duration", 0) for e in self._history if "avg_duration" in e]
            drawdowns = [e.get("avg_drawdown", 0) for e in self._history if "avg_drawdown" in e]
            
            self.logger.info(f"Curriculum Statistics - Episodes: {len(self._history)}")
            if win_rates:
                self.logger.info(f"  Win Rates: mean={np.mean(win_rates):.3f}, std={np.std(win_rates):.3f}")
            if durations:
                self.logger.info(f"  Durations: mean={np.mean(durations):.3f}, std={np.std(durations):.3f}")
            if drawdowns:
                self.logger.info(f"  Drawdowns: mean={np.mean(drawdowns):.3f}, std={np.std(drawdowns):.3f}")
                
            overall_win_rate = self._total_wins / max(1, self._total_trades)
            self.logger.info(f"  Overall: win_rate={overall_win_rate:.3f}, avg_pnl=€{self._cumulative_pnl/max(1, len(self._history)):.2f}")
            
        except Exception as e:
            self.logger.error(f"Error logging curriculum stats: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Get curriculum metrics with validation"""
        try:
            if not self._history:
                # FIX: Return meaningful defaults for bootstrap
                defaults = np.array([0.5, 0.0, 0.01], dtype=np.float32)  # [win_rate, avg_duration, avg_drawdown]
                self.logger.debug("Using default curriculum metrics")
                return defaults
            
            # Calculate rolling metrics with validation
            win_rates = [e.get("win_rate", 0) for e in self._history if isinstance(e.get("win_rate"), (int, float))]
            durations = [e.get("avg_duration", 0) for e in self._history if isinstance(e.get("avg_duration"), (int, float))]
            drawdowns = [e.get("avg_drawdown", 0) for e in self._history if isinstance(e.get("avg_drawdown"), (int, float))]
            
            # FIX: Use robust averaging with fallbacks
            avg_wr = np.mean(win_rates) if win_rates else 0.5
            avg_dur = np.mean(durations) if durations else 0.0
            avg_dd = np.mean(drawdowns) if drawdowns else 0.01
            
            # Validate for NaN
            metrics = np.array([avg_wr, avg_dur, avg_dd], dtype=np.float32)
            if np.any(np.isnan(metrics)):
                self.logger.error(f"NaN in curriculum metrics: {metrics}")
                metrics = np.nan_to_num(metrics, nan=0.0)
                
            self.logger.debug(f"Curriculum metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.5, 0.0, 0.01], dtype=np.float32)
