
# modules/strategy/bias_auditor.py

from __future__ import annotations
import logging
import numpy as np
from collections import deque
from modules.core.core import Module

class BiasAuditor(Module):

    def __init__(self, history_len: int=100, debug=True):
        self.history_len = history_len
        self.debug = debug
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"BiasAuditor_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/bias_auditor.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"BiasAuditor initialized - history_len={history_len}")
        
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.history_len)
        self.bias_corrections = {"revenge": 0, "fear": 0, "greed": 0}
        self._step_count = 0
        self.logger.info("BiasAuditor reset - all bias history and corrections cleared")

    def step(self, **kwargs): 
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record(self, bias: str, pnl: float = 0):
        """FIX: Record bias with outcome and comprehensive logging"""
        try:
            # Validate inputs
            if not isinstance(bias, str):
                self.logger.error(f"Invalid bias type: {type(bias)}")
                bias = "unknown"
                
            if np.isnan(pnl):
                self.logger.error("NaN PnL in bias record, setting to 0")
                pnl = 0.0
                
            self.logger.debug(f"Recording bias: {bias}, PnL: €{pnl:.2f}")
            
            self.hist.append((bias, pnl))
            
            # Learn bias corrections
            if pnl < 0:
                if bias in self.bias_corrections:
                    self.bias_corrections[bias] += 1
                    self.logger.warning(f"Negative outcome for bias '{bias}' - correction count: {self.bias_corrections[bias]}")
                else:
                    self.bias_corrections[bias] = 1
                    self.logger.warning(f"New negative bias detected: {bias}")
            else:
                self.logger.info(f"Positive outcome for bias '{bias}': €{pnl:.2f}")
                
            # Log bias statistics periodically
            if len(self.hist) % 20 == 0:
                self._log_bias_stats()
                
        except Exception as e:
            self.logger.error(f"Error recording bias: {e}")

    def _log_bias_stats(self):
        """Log detailed bias statistics"""
        try:
            if not self.hist:
                return
                
            bias_counts = {}
            bias_pnls = {}
            
            for bias, pnl in self.hist:
                if bias not in bias_counts:
                    bias_counts[bias] = 0
                    bias_pnls[bias] = []
                bias_counts[bias] += 1
                bias_pnls[bias].append(pnl)
            
            self.logger.info(f"Bias Statistics - Total records: {len(self.hist)}")
            
            for bias in bias_counts:
                count = bias_counts[bias]
                pnls = bias_pnls[bias]
                avg_pnl = np.mean(pnls)
                negative_count = sum(1 for p in pnls if p < 0)
                
                self.logger.info(f"  {bias}: count={count}, avg_pnl=€{avg_pnl:.2f}, negative={negative_count}, correction_level={self.bias_corrections.get(bias, 0)}")
                
        except Exception as e:
            self.logger.error(f"Error logging bias stats: {e}")

    def get_observation_components(self)->np.ndarray:
        """Get bias frequencies with validation"""
        try:
            total = len(self.hist)
            if total == 0:
                # FIX: Balanced initial state
                defaults = np.array([0.33, 0.33, 0.33], dtype=np.float32)
                self.logger.debug("Using default bias frequencies")
                return defaults
                
            cnt = {"revenge":0,"fear":0,"greed":0}
            for b, _ in self.hist:
                if b in cnt: 
                    cnt[b] += 1
                    
            # FIX: Apply corrections to discourage losing biases
            for bias, correction_count in self.bias_corrections.items():
                if correction_count > 5:  # Significant negative bias
                    reduction = correction_count // 2
                    cnt[bias] = max(0, cnt[bias] - reduction)
                    if reduction > 0:
                        self.logger.debug(f"Applied correction to {bias}: reduced by {reduction}")
                    
            total_corrected = sum(cnt.values()) or 1
            freqs = np.array([cnt["revenge"],cnt["fear"],cnt["greed"]], dtype=np.float32) / total_corrected
            
            # Validate frequencies
            if np.any(np.isnan(freqs)):
                self.logger.error(f"NaN in bias frequencies: {freqs}")
                freqs = np.nan_to_num(freqs)
                
            # Ensure they sum to 1
            if freqs.sum() > 0:
                freqs = freqs / freqs.sum()
            else:
                freqs = np.array([0.33, 0.33, 0.33], dtype=np.float32)
                
            self.logger.debug(f"Bias frequencies: revenge={freqs[0]:.3f}, fear={freqs[1]:.3f}, greed={freqs[2]:.3f}")
            return freqs
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.33, 0.33, 0.33], dtype=np.float32)
