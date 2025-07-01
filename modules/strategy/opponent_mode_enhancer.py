#
# modules/strategy/opponent_mode_enhancer.py

from __future__ import annotations
import logging
import numpy as np
from typing import List
from modules.core.core import Module


class OpponentModeEnhancer(Module):

    def __init__(self, modes: List[str]=None, debug=True):
        self.modes = modes or ["trending", "ranging", "volatile"]  # FIX: Better market modes
        self.debug = debug
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"OpponentModeEnhancer_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/opponent_mode_enhancer.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"OpponentModeEnhancer initialized - modes={self.modes}")
        
        self.reset()

    def reset(self):
        self.pnl = {m: 0.0 for m in self.modes}
        self.counts = {m: 0 for m in self.modes}
        self._step_count = 0
        self.logger.info("OpponentModeEnhancer reset - all mode statistics cleared")

    def step(self, **kwargs): 
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record_result(self, mode: str, pnl: float):
        """Record mode result with validation and logging"""
        try:
            # Validate inputs
            if not isinstance(mode, str):
                self.logger.error(f"Invalid mode type: {type(mode)}")
                return
                
            if np.isnan(pnl):
                self.logger.error(f"NaN PnL for mode {mode}, setting to 0")
                pnl = 0.0
                
            if mode not in self.modes:
                self.logger.warning(f"Unknown mode '{mode}', adding to tracking")
                self.modes.append(mode)
                self.pnl[mode] = 0.0
                self.counts[mode] = 0
                
            self.pnl[mode] += pnl
            self.counts[mode] += 1
            
            avg_pnl = self.pnl[mode] / self.counts[mode]
            self.logger.info(f"Mode result: {mode}, PnL: €{pnl:.2f}, Count: {self.counts[mode]}, Avg: €{avg_pnl:.2f}")
            
            # Log mode statistics periodically
            if sum(self.counts.values()) % 10 == 0:
                self._log_mode_stats()
                
        except Exception as e:
            self.logger.error(f"Error recording result for mode {mode}: {e}")

    def _log_mode_stats(self):
        """Log detailed mode statistics"""
        try:
            total_trades = sum(self.counts.values())
            if total_trades == 0:
                return
                
            self.logger.info(f"Mode Statistics - Total trades: {total_trades}")
            
            for mode in self.modes:
                if self.counts[mode] > 0:
                    avg_pnl = self.pnl[mode] / self.counts[mode]
                    frequency = self.counts[mode] / total_trades
                    self.logger.info(f"  {mode}: trades={self.counts[mode]} ({frequency:.1%}), total=€{self.pnl[mode]:.2f}, avg=€{avg_pnl:.2f}")
                else:
                    self.logger.info(f"  {mode}: no trades yet")
                    
        except Exception as e:
            self.logger.error(f"Error logging mode stats: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Get mode weights with validation"""
        try:
            # FIX: Return profit-per-trade for each mode
            profits_per_trade = []
            for m in self.modes:
                if self.counts[m] > 0:
                    avg_pnl = self.pnl[m] / self.counts[m]
                else:
                    avg_pnl = 0.0
                profits_per_trade.append(avg_pnl)
                
            # Normalize to weights
            arr = np.array(profits_per_trade, dtype=np.float32)
            
            # Validate array
            if np.any(np.isnan(arr)):
                self.logger.error(f"NaN in profits per trade: {arr}")
                arr = np.nan_to_num(arr)
            
            if arr.sum() > 0:
                # Shift to positive values and normalize
                shifted = arr - arr.min() + 1e-6
                weights = shifted / shifted.sum()
            else:
                weights = np.ones(len(self.modes)) / len(self.modes)
                
            # Final validation
            if np.any(np.isnan(weights)):
                self.logger.error(f"NaN in final weights: {weights}")
                weights = np.ones(len(self.modes)) / len(self.modes)
                
            self.logger.debug(f"Mode weights: {dict(zip(self.modes, weights))}")
            return weights
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.ones(len(self.modes), dtype=np.float32) / len(self.modes)
