# ─────────────────────────────────────────────────────────────
# modules/meta/meta_agent.py

from __future__ import annotations
import logging
import numpy as np
from typing import List
from modules.core.core import Module
# ──────────────────────────────────────────────
class MetaAgent(Module):
    def __init__(self, window: int=20, debug=True, profit_target=150.0):
        self.window = window
        self.debug = debug
        self.profit_target = profit_target
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"MetaAgent_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/meta_agent.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"MetaAgent initialized - window={window}, profit_target=€{profit_target}")
        
        self.reset()

    def reset(self):
        self.history: List[float] = []
        self.trade_count = 0
        self.consecutive_losses = 0
        self._step_count = 0
        self.logger.info("MetaAgent reset - all history and counters cleared")

    def step(self, pnl: float=0.0):
        """Step with comprehensive validation and logging"""
        self._step_count += 1
        
        try:
            # Validate PnL
            if np.isnan(pnl):
                self.logger.error(f"NaN PnL received, setting to 0")
                pnl = 0.0
            
            self.history.append(pnl)
            if len(self.history) > self.window:
                removed = self.history.pop(0)
                self.logger.debug(f"Removed old PnL: €{removed:.2f}")
            
            # Track consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
                self.logger.debug(f"Loss recorded: €{pnl:.2f}, consecutive losses: {self.consecutive_losses}")
            else:
                if self.consecutive_losses > 0:
                    self.logger.info(f"Loss streak broken after {self.consecutive_losses} losses with profit: €{pnl:.2f}")
                self.consecutive_losses = 0
                
            self.trade_count += 1
            
            # Log statistics periodically
            if self.trade_count % 10 == 0:
                self._log_meta_stats()
                
        except Exception as e:
            self.logger.error(f"Error in step: {e}")

    def record(self, pnl: float):
        """Record PnL with logging"""
        self.logger.debug(f"Recording PnL: €{pnl:.2f}")
        self.step(pnl)

    def _log_meta_stats(self):
        """Log detailed meta agent statistics"""
        try:
            if not self.history:
                return
                
            total_pnl = sum(self.history)
            avg_pnl = total_pnl / len(self.history)
            wins = sum(1 for p in self.history if p > 0)
            win_rate = wins / len(self.history)
            
            self.logger.info(f"Meta Statistics - Trades: {self.trade_count}")
            self.logger.info(f"  Total PnL: €{total_pnl:.2f}, Avg: €{avg_pnl:.2f}")
            self.logger.info(f"  Win Rate: {win_rate:.3f} ({wins}/{len(self.history)})")
            self.logger.info(f"  Consecutive Losses: {self.consecutive_losses}")
            self.logger.info(f"  Progress vs Target: {(total_pnl/self.profit_target)*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error logging meta stats: {e}")

    def get_observation_components(self)->np.ndarray:
        """Get observation with validation"""
        try:
            if not self.history:
                observation = np.array([0.0, 0.0], dtype=np.float32)
                self.logger.debug("Using default observation (empty history)")
                return observation
                
            arr = np.array(self.history, dtype=np.float32)
            
            # Validate array
            if np.any(np.isnan(arr)):
                self.logger.error(f"NaN values in history: {arr}")
                arr = np.nan_to_num(arr)
                
            mean_val = float(arr.mean())
            std_val = float(arr.std())
            
            # Validate results
            if np.isnan(mean_val):
                self.logger.error("NaN in mean calculation")
                mean_val = 0.0
            if np.isnan(std_val):
                self.logger.error("NaN in std calculation")
                std_val = 0.0
                
            observation = np.array([mean_val, std_val], dtype=np.float32)
            self.logger.debug(f"Observation: mean={mean_val:.3f}, std={std_val:.3f}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def get_intensity(self, instrument: str) -> float:
        """
        FIX: Smarter intensity calculation for profitable trading with comprehensive logging
        """
        try:
            self.logger.debug(f"Calculating intensity for {instrument}")
            
            # Bootstrap intensity for initial trades
            if self.trade_count < 5:
                # Start with moderate positive intensity to encourage trading
                intensity = 0.3 + np.random.uniform(-0.1, 0.1)
                self.logger.info(f"Bootstrap intensity for {instrument}: {intensity:.3f} (trade_count={self.trade_count})")
                return float(intensity)
            
            if not self.history:
                self.logger.debug(f"No history for {instrument}, returning 0")
                return 0.0
                
            # Calculate recent performance
            recent_window = min(10, len(self.history))
            recent_pnl = self.history[-recent_window:]
            avg_pnl = np.mean(recent_pnl)
            
            self.logger.debug(f"Recent performance for {instrument}: avg_pnl=€{avg_pnl:.3f} over {recent_window} trades")
            
            # FIX: Profit-aware intensity
            if avg_pnl > self.profit_target / self.window:
                # Above target: maintain momentum
                intensity = 0.7 + min(0.3, avg_pnl / (self.profit_target * 2))
                self.logger.info(f"Above target performance - high intensity: {intensity:.3f}")
            elif avg_pnl > 0:
                # Profitable but below target: increase aggression
                intensity = 0.3 + (avg_pnl / self.profit_target)
                self.logger.info(f"Profitable but below target - moderate intensity: {intensity:.3f}")
            else:
                # Losing: reduce but don't stop
                intensity = max(-0.5, -0.1 - self.consecutive_losses * 0.05)
                self.logger.warning(f"Losing streak - reduced intensity: {intensity:.3f}")
                
            # Clamp to safe range
            intensity = np.clip(intensity, -0.8, 0.9)
            
            # Final validation
            if np.isnan(intensity):
                self.logger.error("NaN intensity calculated, using 0")
                intensity = 0.0
            
            self.logger.info(f"Final intensity for {instrument}: {intensity:.3f} (avg_pnl=€{avg_pnl:.3f}, losses={self.consecutive_losses})")
            return float(intensity)
            
        except Exception as e:
            self.logger.error(f"Error calculating intensity for {instrument}: {e}")
            return 0.0
