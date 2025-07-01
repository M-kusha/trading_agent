# ──────────────────────────────────────────────────────────────
# File: modules/risk/anomaly_detector.py
# ──────────────────────────────────────────────────────────────

import numpy as np
import logging
import json
import os
from collections import deque
from typing import Dict, Any, List, Optional
from modules.core.core import Module
from utils.get_dir import utcnow

class AnomalyDetector(Module):

    AUDIT_PATH = "logs/risk/anomaly_detector_audit.jsonl"
    LOG_PATH   = "logs/risk/anomaly_detector.log"

    def __init__(
        self,
        pnl_limit: float = 1000.0,  # FIXED: Realistic threshold for training
        volume_zscore: float = 3.0,
        price_zscore: float = 3.0,
        enabled: bool = True,
        audit_log_size: int = 100,
        history_size: int = 100,
        training_mode: bool = True,  # NEW: Separate training vs live behavior
        debug: bool = True
    ):
        super().__init__()
        self.enabled = enabled
        self.pnl_limit = pnl_limit
        self.volume_zscore = volume_zscore
        self.price_zscore = price_zscore
        self.training_mode = training_mode
        self.debug = debug
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)
        
        # History tracking
        self.pnl_history = deque(maxlen=history_size)
        self.volume_history = deque(maxlen=history_size)
        self.price_history = deque(maxlen=history_size)
        self.observation_history = deque(maxlen=20)
        
        # State
        self.anomalies: Dict[str, List[Dict[str, Any]]] = {
            "pnl": [],
            "volume": [],
            "price": [],
            "observation": [],
            "pattern": []
        }
        self.anomaly_score = 0.0
        self.step_count = 0
        self.adaptive_thresholds = {
            "pnl": self.pnl_limit,
            "volume": 0.0,
            "price": 0.0
        }
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size

        # Logger setup - Fixed
        self.logger = logging.getLogger("AnomalyDetector")
        if not self.logger.handlers:
            self.logger.handlers.clear()
            self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
            self.logger.propagate = False
            
            fh = logging.FileHandler(self.LOG_PATH, mode='a')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
            if debug:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
        
        self.logger.info(f"AnomalyDetector initialized - pnl_limit={pnl_limit}, training_mode={training_mode}")

    def reset(self):
        """Reset detector state"""
        self.pnl_history.clear()
        self.volume_history.clear()
        self.price_history.clear()
        self.observation_history.clear()
        for key in self.anomalies:
            self.anomalies[key].clear()
        self.anomaly_score = 0.0
        self.step_count = 0
        self._audit.clear()
        self.logger.info("AnomalyDetector reset")

    def step(
        self,
        pnl: Optional[float] = None,
        obs: Optional[np.ndarray] = None,
        volume: Optional[float] = None,
        price: Optional[float] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        balance: Optional[float] = None,
        equity: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        FIXED: Detect anomalies with proper training vs live mode handling.
        """
        self.step_count += 1
        
        # Reduced logging frequency
        if self.debug and self.step_count % 50 == 0:
            self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}, pnl={pnl}, training={self.training_mode}")
        
        if not self.enabled:
            self.anomaly_score = 0.0
            return False
            
        # Extract data from different sources
        pnl = self._extract_pnl(pnl, balance, equity, kwargs)
        volume = self._extract_volume(volume, trades, kwargs)
        price = self._extract_price(price, kwargs)
        obs = self._extract_observation(obs, kwargs)
        
        # Clear previous anomalies
        for key in self.anomalies:
            self.anomalies[key].clear()
            
        critical_found = False
        
        try:
            # Check PnL anomaly with training-appropriate thresholds
            if pnl is not None:
                self.pnl_history.append(pnl)
                if self._check_pnl_anomaly(pnl):
                    critical_found = True
            elif self.training_mode:
                # FIXED: Only generate synthetic data in training mode when needed
                synthetic_pnl = np.random.normal(0, 100)  # More realistic for training
                self.pnl_history.append(synthetic_pnl)
                if self.debug and self.step_count % 100 == 0:
                    self.logger.debug(f"Generated synthetic PnL: {synthetic_pnl:.2f}")
                    
            # Check observation anomaly
            if obs is not None:
                if self._check_observation_anomaly(obs):
                    critical_found = True
                    
            # Check volume anomaly  
            if volume is not None:
                self.volume_history.append(volume)
                self._check_volume_anomaly(volume)
            elif self.training_mode and len(self.volume_history) < 10:
                # Generate synthetic volume only when bootstrapping in training
                synthetic_volume = abs(np.random.normal(10000, 3000))
                self.volume_history.append(synthetic_volume)
                if self.debug and self.step_count % 100 == 0:
                    self.logger.debug(f"Generated synthetic volume: {synthetic_volume:.0f}")
                
            # Check price anomaly
            if price is not None:
                self.price_history.append(price)
                self._check_price_anomaly(price)
            elif self.training_mode and len(self.price_history) < 10:
                # Generate synthetic price only when bootstrapping in training
                if self.price_history:
                    last_price = self.price_history[-1]
                    synthetic_price = last_price * (1 + np.random.normal(0, 0.001))
                else:
                    synthetic_price = 100.0 + np.random.normal(0, 1)
                self.price_history.append(synthetic_price)
                if self.debug and self.step_count % 100 == 0:
                    self.logger.debug(f"Generated synthetic price: {synthetic_price:.2f}")
                
            # Check pattern anomalies
            if trades:
                self._check_pattern_anomalies(trades)
                
            # Update adaptive thresholds
            self._update_adaptive_thresholds()
            
            # Calculate anomaly score
            self._calculate_anomaly_score()
            
            # Log summary less frequently
            total_anomalies = sum(len(anomalies) for anomalies in self.anomalies.values())
            if total_anomalies > 0 or self.step_count % 200 == 0:
                self.logger.info(f"Step {self.step_count} summary: anomaly_score={self.anomaly_score:.3f}, total_anomalies={total_anomalies}, critical={critical_found}")
            
            # Record audit if anomalies found
            if total_anomalies > 0:
                self._record_audit()
                
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            
        return critical_found

    def _extract_pnl(self, pnl: Optional[float], balance: Optional[float], equity: Optional[float], kwargs: Dict[str, Any]) -> Optional[float]:
        """Extract PnL from various sources"""
        if pnl is not None:
            return float(pnl)
            
        # Try balance change
        if balance is not None:
            if hasattr(self, '_last_balance') and self._last_balance is not None:
                pnl = balance - self._last_balance
                self._last_balance = balance
                return pnl
            else:
                self._last_balance = balance
                
        # Try equity change
        if equity is not None:
            if hasattr(self, '_last_equity') and self._last_equity is not None:
                pnl = equity - self._last_equity
                self._last_equity = equity
                return pnl
            else:
                self._last_equity = equity
                
        # Try from kwargs
        for key in ['profit', 'profit_loss', 'realized_pnl', 'unrealized_pnl']:
            if key in kwargs and kwargs[key] is not None:
                return float(kwargs[key])
                
        return None

    def _extract_volume(self, volume: Optional[float], trades: Optional[List[Dict[str, Any]]], kwargs: Dict[str, Any]) -> Optional[float]:
        """Extract volume from various sources"""
        if volume is not None:
            return float(volume)
            
        # Calculate from trades
        if trades:
            total_volume = sum(abs(trade.get("size", trade.get("volume", 0))) for trade in trades)
            if total_volume > 0:
                return total_volume
                
        # Try from kwargs
        for key in ['trade_volume', 'total_volume', 'size', 'tick_volume']:
            if key in kwargs and kwargs[key] is not None:
                return float(kwargs[key])
                
        return None

    def _extract_price(self, price: Optional[float], kwargs: Dict[str, Any]) -> Optional[float]:
        """Extract price from various sources"""
        if price is not None:
            return float(price)
            
        # Try from kwargs - handle CSV column names
        for key in ['current_price', 'last_price', 'close_price', 'close', 'bid', 'ask']:
            if key in kwargs and kwargs[key] is not None:
                return float(kwargs[key])
                
        return None

    def _extract_observation(self, obs: Optional[np.ndarray], kwargs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract observation array from various sources"""
        if obs is not None:
            return obs
            
        # Try from kwargs
        for key in ['observation', 'state', 'features']:
            if key in kwargs and kwargs[key] is not None:
                try:
                    return np.array(kwargs[key])
                except:
                    continue
                    
        return None

    def _check_pnl_anomaly(self, pnl: float) -> bool:
        """Check for PnL anomalies with training-appropriate thresholds"""
        # FIXED: Use adaptive threshold that makes sense for training
        current_threshold = self.adaptive_thresholds["pnl"]
        
        if abs(pnl) > current_threshold:
            self.anomalies["pnl"].append({
                "type": "absolute",
                "value": pnl,
                "threshold": current_threshold,
                "severity": "critical"
            })
            self.logger.error(f"Critical PnL anomaly: {pnl:.2f} > {current_threshold:.2f}")
            return True
            
        # Statistical check if enough history
        if len(self.pnl_history) >= 20:
            pnl_array = np.array(self.pnl_history)
            mean = np.mean(pnl_array)
            std = np.std(pnl_array)
            if std > 0:
                z_score = abs((pnl - mean) / std)
                if z_score > 4:  # More lenient for training
                    self.anomalies["pnl"].append({
                        "type": "statistical",
                        "value": pnl,
                        "z_score": z_score,
                        "severity": "warning"
                    })
                    if self.debug:
                        self.logger.warning(f"Statistical PnL anomaly: z-score={z_score:.2f}")
                    
        return False

    def _check_observation_anomaly(self, obs: np.ndarray) -> bool:
        """Check for anomalies in observation vector"""
        try:
            # Check for NaN/Inf
            if np.isnan(obs).any() or np.isinf(obs).any():
                self.anomalies["observation"].append({
                    "type": "invalid_values",
                    "nan_count": int(np.isnan(obs).sum()),
                    "inf_count": int(np.isinf(obs).sum()),
                    "severity": "critical"
                })
                self.logger.error("Critical: Observation contains NaN/Inf")
                return True
                
            # Check for extreme values
            self.observation_history.append(obs)
            if len(self.observation_history) >= 5:
                all_obs = np.vstack(self.observation_history)
                mean = np.mean(all_obs, axis=0)
                std = np.std(all_obs, axis=0)
                
                # Calculate z-scores for current observation
                z_scores = np.abs((obs - mean) / (std + 1e-8))
                extreme_indices = np.where(z_scores > 5)[0]  # More lenient for training
                
                if len(extreme_indices) > 0:
                    self.anomalies["observation"].append({
                        "type": "extreme_values",
                        "indices": extreme_indices.tolist(),
                        "z_scores": z_scores[extreme_indices].tolist(),
                        "severity": "warning"
                    })
                    if self.debug:
                        self.logger.warning(f"Extreme observation values at indices: {extreme_indices.tolist()}")
                    
        except Exception as e:
            self.logger.error(f"Error checking observation anomaly: {e}")
            
        return False

    def _check_volume_anomaly(self, volume: float):
        """Check for volume anomalies"""
        if len(self.volume_history) >= 10:
            volume_array = np.array(self.volume_history)
            mean = np.mean(volume_array)
            std = np.std(volume_array)
            if std > 0:
                z_score = abs((volume - mean) / std)
                if z_score > self.volume_zscore:
                    self.anomalies["volume"].append({
                        "type": "statistical",
                        "value": volume,
                        "z_score": z_score,
                        "severity": "warning"
                    })
                    if self.debug:
                        self.logger.warning(f"Volume anomaly: z-score={z_score:.2f}")

    def _check_price_anomaly(self, price: float):
        """Check for price anomalies"""
        if len(self.price_history) >= 2:
            # Check for price jumps
            prev_price = self.price_history[-2]
            if prev_price > 0:
                price_change = abs((price - prev_price) / prev_price)
                if price_change > 0.1:  # 10% jump - more realistic for training
                    self.anomalies["price"].append({
                        "type": "price_jump",
                        "change": price_change,
                        "prev_price": prev_price,
                        "current_price": price,
                        "severity": "warning"
                    })
                    if self.debug:
                        self.logger.warning(f"Price jump anomaly: {price_change:.3%} change from {prev_price:.2f} to {price:.2f}")
                        
        # Statistical check
        if len(self.price_history) >= 10:
            price_array = np.array(self.price_history)
            mean = np.mean(price_array)
            std = np.std(price_array)
            if std > 0:
                z_score = abs((price - mean) / std)
                if z_score > self.price_zscore:
                    self.anomalies["price"].append({
                        "type": "statistical",
                        "value": price,
                        "z_score": z_score,
                        "severity": "info"
                    })

    def _check_pattern_anomalies(self, trades: List[Dict[str, Any]]):
        """Check for anomalous trading patterns"""
        if not trades:
            return
            
        try:
            # Check for suspicious patterns
            directions = []
            sizes = []
            for trade in trades:
                size = trade.get("size", trade.get("volume", 0))
                if size != 0:
                    directions.append(np.sign(size))
                    sizes.append(abs(size))
                    
            if directions and len(set(directions)) == 1 and len(trades) > 5:  # More lenient
                self.anomalies["pattern"].append({
                    "type": "unidirectional",
                    "count": len(trades),
                    "direction": directions[0],
                    "severity": "info"
                })
                
            # Rapid fire trades - more lenient for training
            if len(trades) > 20:
                self.anomalies["pattern"].append({
                    "type": "high_frequency",
                    "count": len(trades),
                    "severity": "warning"
                })
                
        except Exception as e:
            self.logger.error(f"Error checking pattern anomalies: {e}")

    def _update_adaptive_thresholds(self):
        """Update thresholds based on recent history"""
        try:
            # FIXED: More intelligent adaptive PnL threshold for training
            if len(self.pnl_history) >= 50:
                recent_pnls = list(self.pnl_history)[-50:]
                pnl_array = np.array(recent_pnls)
                mean = abs(np.mean(pnl_array))
                std = np.std(pnl_array)
                
                # Set threshold to 3 standard deviations above mean, but with reasonable bounds
                new_threshold = max(self.pnl_limit, mean + 3 * std)
                new_threshold = min(new_threshold, self.pnl_limit * 5)  # Cap at 5x original
                
                if abs(new_threshold - self.adaptive_thresholds["pnl"]) > self.pnl_limit * 0.1:
                    if self.debug:
                        self.logger.debug(f"Updating PnL threshold: {self.adaptive_thresholds['pnl']:.0f} -> {new_threshold:.0f}")
                    self.adaptive_thresholds["pnl"] = new_threshold
                    
        except Exception as e:
            self.logger.error(f"Error updating adaptive thresholds: {e}")

    def _calculate_anomaly_score(self):
        """Calculate overall anomaly score"""
        score = 0.0
        weights = {
            "critical": 0.5,
            "warning": 0.3,
            "info": 0.1
        }
        
        for anomaly_type, anomalies in self.anomalies.items():
            for anomaly in anomalies:
                severity = anomaly.get("severity", "info")
                score += weights.get(severity, 0.1)
                
        self.anomaly_score = min(score, 1.0)

    def _record_audit(self):
        """Record audit entry for anomalies"""
        entry = {
            "timestamp": utcnow(),
            "step": self.step_count,
            "anomaly_score": self.anomaly_score,
            "anomalies": {
                k: len(v) for k, v in self.anomalies.items() if v
            },
            "details": {
                k: v for k, v in self.anomalies.items() if v
            },
            "thresholds": self.adaptive_thresholds.copy(),
            "history_sizes": {
                "pnl": len(self.pnl_history),
                "volume": len(self.volume_history),
                "price": len(self.price_history),
                "observation": len(self.observation_history)
            }
        }
        
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit.pop(0)
            
        try:
            with open(self.AUDIT_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            if self.debug:
                self.logger.error(f"Failed to write audit: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Return anomaly metrics as observation"""
        try:
            has_critical = float(any(
                a.get("severity") == "critical"
                for anomalies in self.anomalies.values()
                for a in anomalies
            ))
            
            total_anomalies = sum(len(v) for v in self.anomalies.values())
            
            # Data sufficiency metrics
            pnl_sufficiency = min(len(self.pnl_history) / 50.0, 1.0)
            volume_sufficiency = min(len(self.volume_history) / 20.0, 1.0)
            
            return np.array([
                float(self.anomaly_score),
                has_critical,
                min(float(total_anomalies) / 10.0, 1.0),
                pnl_sufficiency,
                volume_sufficiency
            ], dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "limits": {
                "pnl_limit": self.pnl_limit,
                "volume_zscore": self.volume_zscore,
                "price_zscore": self.price_zscore
            },
            "enabled": self.enabled,
            "step_count": self.step_count,
            "anomaly_score": self.anomaly_score,
            "adaptive_thresholds": self.adaptive_thresholds.copy(),
            "training_mode": self.training_mode,
        }

