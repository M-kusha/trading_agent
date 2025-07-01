#
# modules/strategy/thesis_evolution_engine.py

from __future__ import annotations
import logging
import numpy as np
from typing import Dict,List
from modules.core.core import Module

class ThesisEvolutionEngine(Module):
    def __init__(self, capacity: int=20, debug=True):
        self.capacity = capacity
        self.debug = debug
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"ThesisEvolutionEngine_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/thesis_evolution.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"ThesisEvolutionEngine initialized - capacity={capacity}")
        
        self.reset()

    def reset(self):
        self.theses: List[str] = []  # Changed from Any to str for clarity
        self.pnls: List[float] = []
        self.thesis_performance: Dict[str, List[float]] = {}
        self._step_count = 0
        self.logger.info("ThesisEvolutionEngine reset - all thesis data cleared")

    def step(self, **kwargs): 
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record_thesis(self, thesis: str):
        """FIX: Track thesis properly with validation and logging"""
        try:
            if not isinstance(thesis, str):
                self.logger.error(f"Invalid thesis type: {type(thesis)}")
                thesis = str(thesis)
                
            self.theses.append(thesis)
            if thesis not in self.thesis_performance:
                self.thesis_performance[thesis] = []
                self.logger.info(f"New thesis recorded: '{thesis}'")
            else:
                self.logger.debug(f"Existing thesis recorded: '{thesis}'")
                
            # Maintain capacity
            if len(self.theses) > self.capacity:
                removed = self.theses.pop(0)
                self.logger.debug(f"Removed old thesis: '{removed}'")
                
        except Exception as e:
            self.logger.error(f"Error recording thesis: {e}")

    def record_pnl(self, pnl: float):
        """Record PnL for current thesis with validation and logging"""
        try:
            # Validate PnL
            if np.isnan(pnl):
                self.logger.error("NaN PnL recorded, setting to 0")
                pnl = 0.0
                
            if self.theses:
                current_thesis = self.theses[-1]
                self.pnls.append(pnl)
                self.thesis_performance[current_thesis].append(pnl)
                
                # Calculate thesis performance
                thesis_pnls = self.thesis_performance[current_thesis]
                thesis_avg = np.mean(thesis_pnls)
                thesis_total = sum(thesis_pnls)
                
                self.logger.info(f"PnL €{pnl:.2f} recorded for thesis '{current_thesis}' - Total: €{thesis_total:.2f}, Avg: €{thesis_avg:.2f}, Count: {len(thesis_pnls)}")
                
                # Maintain capacity
                if len(self.pnls) > self.capacity:
                    removed = self.pnls.pop(0)
                    self.logger.debug(f"Removed old PnL: €{removed:.2f}")
                    
                # Log thesis statistics periodically
                if len(thesis_pnls) % 5 == 0:
                    self._log_thesis_stats()
            else:
                self.logger.warning(f"No thesis to record PnL €{pnl:.2f} against")
                
        except Exception as e:
            self.logger.error(f"Error recording PnL: {e}")

    def _log_thesis_stats(self):
        """Log detailed thesis statistics"""
        try:
            if not self.thesis_performance:
                return
                
            self.logger.info(f"Thesis Performance Summary - {len(self.thesis_performance)} theses:")
            
            for thesis, pnls in self.thesis_performance.items():
                if pnls:
                    total_pnl = sum(pnls)
                    avg_pnl = np.mean(pnls)
                    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                    
                    self.logger.info(f"  '{thesis[:30]}...': trades={len(pnls)}, total=€{total_pnl:.2f}, avg=€{avg_pnl:.2f}, win_rate={win_rate:.3f}")
                    
        except Exception as e:
            self.logger.error(f"Error logging thesis stats: {e}")

    def get_observation_components(self)->np.ndarray:
        """Get thesis metrics with validation"""
        try:
            if not self.pnls:
                defaults = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self.logger.debug("Using default thesis metrics")
                return defaults
                
            # FIX: Better metrics
            uniq = len(self.thesis_performance)
            mean_p = float(np.mean(self.pnls))
            
            # Validate mean
            if np.isnan(mean_p):
                self.logger.error("NaN in mean PnL")
                mean_p = 0.0
            
            # Find best performing thesis
            best_thesis_pnl = 0.0
            if self.thesis_performance:
                for thesis, pnls in self.thesis_performance.items():
                    if pnls:
                        thesis_avg = np.mean(pnls)
                        if not np.isnan(thesis_avg):
                            best_thesis_pnl = max(best_thesis_pnl, thesis_avg)
                            
            observation = np.array([float(uniq), mean_p, best_thesis_pnl], dtype=np.float32)
            
            # Final validation
            if np.any(np.isnan(observation)):
                self.logger.error(f"NaN in thesis observation: {observation}")
                observation = np.nan_to_num(observation)
                
            self.logger.debug(f"Thesis metrics: unique={uniq}, mean_pnl={mean_p:.3f}, best_thesis_pnl={best_thesis_pnl:.3f}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
