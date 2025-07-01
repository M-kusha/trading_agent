# modules/strategy/strategy_introspector.py

from __future__ import annotations
import logging
import numpy as np
from typing import Dict, List
from modules.core.core import Module

class StrategyIntrospector(Module):

    def __init__(self, history_len: int = 10, debug: bool = True):
        self.history_len = history_len
        self.debug = debug
        self._records: List[Dict[str, float]] = []
        self._step_count = 0
        
        # FIX: Initialize with some baseline data to avoid zero observations
        self._baseline_wr = 0.5  # 50% win rate baseline
        self._baseline_sl = 1.0  # 1% stop loss baseline
        self._baseline_tp = 1.5  # 1.5% take profit baseline

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"StrategyIntrospector_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/introspection/strategy_introspector.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"StrategyIntrospector initialized - history_len={history_len}, baselines: wr={self._baseline_wr}, sl={self._baseline_sl}, tp={self._baseline_tp}")

    def reset(self) -> None:
        self._records.clear()
        self._step_count = 0
        self.logger.info("StrategyIntrospector reset - all records cleared")

    def step(self, **kwargs) -> None:
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record(self, theme: np.ndarray, win_rate: float, sl: float, tp: float) -> None:
        """Record strategy performance with validation"""
        try:
            # Validate inputs
            if not (0 <= win_rate <= 1):
                self.logger.warning(f"Invalid win_rate {win_rate}, clamping to [0,1]")
                win_rate = np.clip(win_rate, 0, 1)
            
            if sl <= 0:
                self.logger.warning(f"Invalid sl {sl}, using baseline {self._baseline_sl}")
                sl = self._baseline_sl
                
            if tp <= 0:
                self.logger.warning(f"Invalid tp {tp}, using baseline {self._baseline_tp}")
                tp = self._baseline_tp
            
            record = {"wr": win_rate, "sl": sl, "tp": tp}
            self._records.append(record)
            
            if len(self._records) > self.history_len:
                removed = self._records.pop(0)
                self.logger.debug(f"Removed old record: {removed}")
                
            self.logger.info(f"Recorded strategy: wr={win_rate:.3f}, sl={sl:.3f}, tp={tp:.3f}, total_records={len(self._records)}")
            
            # Log statistics periodically
            if len(self._records) % 5 == 0:
                self._log_statistics()
                
        except Exception as e:
            self.logger.error(f"Error recording strategy: {e}")

    def _log_statistics(self):
        """Log current strategy statistics"""
        try:
            if not self._records:
                return
                
            arr = np.array([[r["wr"], r["sl"], r["tp"]] for r in self._records], dtype=np.float32)
            means = arr.mean(axis=0)
            stds = arr.std(axis=0)
            
            self.logger.info(f"Strategy Statistics - Records: {len(self._records)}")
            self.logger.info(f"  Win Rate: {means[0]:.3f} ± {stds[0]:.3f}")
            self.logger.info(f"  Stop Loss: {means[1]:.3f} ± {stds[1]:.3f}")
            self.logger.info(f"  Take Profit: {means[2]:.3f} ± {stds[2]:.3f}")
        except Exception as e:
            self.logger.error(f"Error logging statistics: {e}")

    def profile(self) -> np.ndarray:
        """Get strategy profile with comprehensive validation"""
        try:
            if not self._records:
                # FIX: Return baseline values instead of zeros
                baseline = np.array([
                    self._baseline_wr, self._baseline_sl, self._baseline_tp,
                    0.0, 0.0  # No variance yet
                ], dtype=np.float32)
                self.logger.debug(f"Using baseline profile: {baseline}")
                return baseline
            
            arr = np.array([[r["wr"], r["sl"], r["tp"]] for r in self._records], dtype=np.float32)
            
            # Validate array
            if np.any(np.isnan(arr)):
                self.logger.error(f"NaN values in records array: {arr}")
                arr = np.nan_to_num(arr, nan=0.0)
            
            # FIX: Calculate mean and variance for better signal
            mean_vals = arr.mean(axis=0)
            var_vals = arr.var(axis=0) if len(arr) > 1 else np.zeros(3)
            
            # Combine mean and variance info
            profile = np.concatenate([mean_vals, var_vals[:2]])  # Total 5 values
            
            # Final validation
            if np.any(np.isnan(profile)):
                self.logger.error(f"NaN in final profile: {profile}")
                profile = np.nan_to_num(profile, nan=0.0)
                
            self.logger.debug(f"Generated profile: {profile}")
            return profile.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error generating profile: {e}")
            return np.array([self._baseline_wr, self._baseline_sl, self._baseline_tp, 0.0, 0.0], dtype=np.float32)

    def get_observation_components(self) -> np.ndarray:
        return self.profile()
