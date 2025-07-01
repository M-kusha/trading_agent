
# ─────────────────────────────────────────────────────────────
# modules/meta/metacognitive_planner.py

from __future__ import annotations
import logging
import os
import numpy as np
from typing import Dict,List
from modules.core.core import Module

class MetaCognitivePlanner(Module):
    def __init__(self, window: int=20, debug=True):
        self.window = window
        self.debug = debug
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"MetaCognitivePlanner_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/metacognitive_planner.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"MetaCognitivePlanner initialized - window={window}")
        
        self.reset()

    def reset(self):
        self.history: List[Dict[str,float]] = []
        self.total_episodes = 0
        self.profitable_episodes = 0
        self._step_count = 0
        self.logger.info("MetaCognitivePlanner reset - all episode history cleared")

    def step(self, **kwargs): 
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record_episode(self, result: Dict[str,float]):
        """Record episode with comprehensive validation and logging"""
        try:
            # Validate input
            if not isinstance(result, dict):
                self.logger.error(f"Invalid result type: {type(result)}")
                return
                
            self.logger.debug(f"Recording episode result: {result}")
            
            # Validate PnL
            pnl = result.get("pnl", 0)
            if np.isnan(pnl):
                self.logger.error(f"NaN PnL in episode result, setting to 0")
                result = result.copy()
                result["pnl"] = 0
                pnl = 0
            
            self.history.append(result)
            if len(self.history) > self.window:
                removed = self.history.pop(0)
                self.logger.debug(f"Removed old episode: {removed}")
                
            # Track profitable episodes
            self.total_episodes += 1
            if pnl > 0:
                self.profitable_episodes += 1
                self.logger.info(f"Profitable episode recorded: €{pnl:.2f}")
            else:
                self.logger.debug(f"Loss episode recorded: €{pnl:.2f}")
                
            # Log episode statistics
            win_rate = self.profitable_episodes / self.total_episodes
            self.logger.info(f"Episode {self.total_episodes}: Win rate: {win_rate:.3f} ({self.profitable_episodes}/{self.total_episodes})")
            
            # Log detailed statistics periodically
            if self.total_episodes % 10 == 0:
                self._log_cognitive_stats()
                
        except Exception as e:
            self.logger.error(f"Error recording episode: {e}")

    def _log_cognitive_stats(self):
        """Log detailed cognitive planning statistics"""
        try:
            if not self.history:
                return
                
            pnls = [r.get("pnl", 0) for r in self.history]
            total_pnl = sum(pnls)
            avg_pnl = total_pnl / len(pnls)
            
            profitable_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p < 0]
            
            self.logger.info(f"Cognitive Statistics - Episodes: {len(self.history)}")
            self.logger.info(f"  Total PnL: €{total_pnl:.2f}, Avg: €{avg_pnl:.2f}")
            
            if profitable_pnls:
                avg_win = np.mean(profitable_pnls)
                self.logger.info(f"  Avg Win: €{avg_win:.2f} ({len(profitable_pnls)} wins)")
                
            if losing_pnls:
                avg_loss = np.mean(losing_pnls)
                self.logger.info(f"  Avg Loss: €{avg_loss:.2f} ({len(losing_pnls)} losses)")
                
                if profitable_pnls:
                    risk_reward = avg_win / abs(avg_loss)
                    self.logger.info(f"  Risk/Reward Ratio: {risk_reward:.2f}")
                    
        except Exception as e:
            self.logger.error(f"Error logging cognitive stats: {e}")

    def get_observation_components(self)->np.ndarray:
        """Get cognitive metrics with validation"""
        try:
            if not self.history:
                # FIX: Better bootstrap values
                defaults = np.array([0.5, 0.0, 1.0], dtype=np.float32)
                self.logger.debug("Using default cognitive metrics")
                return defaults
                
            pnls = np.array([r.get("pnl",0) for r in self.history], dtype=np.float32)
            
            # Validate PnLs
            if np.any(np.isnan(pnls)):
                self.logger.error(f"NaN values in PnL history: {pnls}")
                pnls = np.nan_to_num(pnls)
                
            win_rate = float((pnls>0).sum() / len(pnls)) if len(pnls)>0 else 0.5
            
            # FIX: Add risk-adjusted return metric
            profitable_pnls = pnls[pnls > 0]
            losing_pnls = pnls[pnls < 0]
            
            avg_win = profitable_pnls.mean() if len(profitable_pnls) > 0 else 0.0
            avg_loss = abs(losing_pnls.mean()) if len(losing_pnls) > 0 else 1.0
            risk_reward = avg_win / avg_loss if avg_loss > 0 else 1.0
            
            # Validate components
            if np.isnan(win_rate):
                self.logger.error("NaN in win_rate")
                win_rate = 0.5
            if np.isnan(risk_reward):
                self.logger.error("NaN in risk_reward")
                risk_reward = 1.0
                
            mean_pnl = float(pnls.mean())
            if np.isnan(mean_pnl):
                self.logger.error("NaN in mean_pnl")
                mean_pnl = 0.0
            
            observation = np.array([win_rate, mean_pnl, risk_reward], dtype=np.float32)
            self.logger.debug(f"Cognitive metrics: win_rate={win_rate:.3f}, mean_pnl={mean_pnl:.3f}, risk_reward={risk_reward:.3f}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.5, 0.0, 1.0], dtype=np.float32)
