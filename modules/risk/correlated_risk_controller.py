# ──────────────────────────────────────────────────────────────
# modules/risk/correlated_risk_controller.py
# ──────────────────────────────────────────────────────────────

import numpy as np
import logging
import json
import os
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Union
from modules.core.core import Module
from utils.get_dir import utcnow

class CorrelatedRiskController(Module):
    AUDIT_PATH = "logs/risk/correlated_risk_controller_audit.jsonl"
    LOG_PATH   = "logs/risk/correlated_risk_controller.log"

    def __init__(
        self,
        max_corr: float = 0.9,
        warning_corr: float = 0.7,
        info_corr: float = 0.5,
        enabled: bool = True,
        audit_log_size: int = 100,
        history_size: int = 20,
        debug: bool = True
    ):
        super().__init__()
        self.max_corr = max_corr
        self.warning_corr = warning_corr
        self.info_corr = info_corr
        self.enabled = enabled
        self.debug = debug
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)
        
        # State tracking
        self.correlation_history = deque(maxlen=history_size)
        self.current_correlations: Dict[Tuple[str, str], float] = {}
        self.risk_score = 0.0
        self.step_count = 0
        self.alerts: Dict[str, List[Tuple[str, str]]] = {
            "info": [],
            "warning": [],
            "critical": []
        }
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size

        # Logger setup - Fixed to prevent multiple handlers
        self.logger = logging.getLogger(f"CorrelatedRiskController")
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
        
        self.logger.info(f"CorrelatedRiskController initialized - thresholds: {info_corr}/{warning_corr}/{max_corr}")

    def reset(self):
        """Reset controller state"""
        self.correlation_history.clear()
        self.current_correlations.clear()
        self.risk_score = 0.0
        self.step_count = 0
        for key in self.alerts:
            self.alerts[key].clear()
        self._audit.clear()
        self.logger.info("CorrelatedRiskController reset")

    def step(
        self,
        correlations: Optional[Union[Dict, List, np.ndarray]] = None,
        positions: Optional[Dict[str, Any]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        instruments: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """
        FIXED: Monitor correlation risk with proper error handling for all input types.
        Returns True if critical correlation detected.
        """
        self.step_count += 1
        
        if self.debug and self.step_count % 20 == 0:  # Reduced logging frequency
            self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}")
        
        if not self.enabled:
            self.risk_score = 0.0
            return False
        
        try:
            # FIXED: Robust correlation input processing
            processed_correlations = self._safe_process_correlation_input(
                correlations, correlation_matrix, instruments, kwargs
            )
            
            if not processed_correlations:
                self.risk_score = 0.0
                if self.debug and self.step_count % 50 == 0:  # Less frequent logging
                    self.logger.debug("No correlation data provided")
                return False
                
            # Update current correlations
            self.current_correlations = processed_correlations.copy()
            
            # Clear alerts
            for key in self.alerts:
                self.alerts[key].clear()
                
            # Analyze correlations
            critical_found = self._analyze_correlations(processed_correlations, positions)
            
            # Log summary less frequently
            if critical_found or self.step_count % 100 == 0:
                total_alerts = sum(len(alerts) for alerts in self.alerts.values())
                self.logger.info(f"Step {self.step_count} summary: risk_score={self.risk_score:.3f}, alerts={total_alerts}, critical={critical_found}")
            
            return critical_found
            
        except Exception as e:
            self.logger.error(f"Error in step {self.step_count}: {e}")
            self.risk_score = 0.0
            return False

    def _safe_process_correlation_input(
        self,
        correlations: Optional[Union[Dict, List, np.ndarray]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        instruments: Optional[List[str]] = None,
        kwargs: Dict[str, Any] = None
    ) -> Dict[Tuple[str, str], float]:
        """
        FIXED: Safely process ALL possible correlation input formats.
        This is where the "too many values to unpack" error was happening.
        """
        result = {}
        
        try:
            # Method 1: Direct correlation dict
            if correlations is not None:
                if isinstance(correlations, dict):
                    # Handle different key formats safely
                    for key, value in correlations.items():
                        try:
                            # FIXED: Safe tuple creation without unpacking assumptions
                            if isinstance(key, (tuple, list)) and len(key) >= 2:
                                # Convert to tuple safely
                                clean_key = (str(key[0]), str(key[1]))
                                result[clean_key] = float(value)
                            elif isinstance(key, str):
                                # Handle string formats like "EUR/USD_GBP/USD"
                                if '_' in key:
                                    parts = key.split('_', 1)
                                    if len(parts) == 2:
                                        result[(parts[0], parts[1])] = float(value)
                                elif '-' in key:
                                    parts = key.split('-', 1)
                                    if len(parts) == 2:
                                        result[(parts[0], parts[1])] = float(value)
                        except (ValueError, IndexError, TypeError) as e:
                            if self.debug:
                                self.logger.debug(f"Skipping invalid correlation entry {key}: {e}")
                            continue
                            
                elif isinstance(correlations, (list, np.ndarray)):
                    # Handle list/array format - create default instrument names
                    corr_array = np.array(correlations)
                    if corr_array.ndim == 2:
                        n = min(corr_array.shape[0], corr_array.shape[1])
                        default_instruments = [f"INST_{i}" for i in range(n)]
                        for i in range(n):
                            for j in range(i + 1, n):
                                if i < corr_array.shape[0] and j < corr_array.shape[1]:
                                    corr_value = corr_array[i, j]
                                    if not np.isnan(corr_value) and not np.isinf(corr_value):
                                        result[(default_instruments[i], default_instruments[j])] = float(corr_value)
                        
            # Method 2: Correlation matrix with instruments
            if correlation_matrix is not None and instruments and len(instruments) > 1:
                try:
                    n = min(len(instruments), correlation_matrix.shape[0], correlation_matrix.shape[1])
                    for i in range(n):
                        for j in range(i + 1, n):
                            if i < correlation_matrix.shape[0] and j < correlation_matrix.shape[1]:
                                corr_value = correlation_matrix[i, j]
                                if not np.isnan(corr_value) and not np.isinf(corr_value):
                                    result[(instruments[i], instruments[j])] = float(corr_value)
                except (IndexError, ValueError) as e:
                    if self.debug:
                        self.logger.debug(f"Error processing correlation matrix: {e}")
                        
            # Method 3: Extract from kwargs - FIXED to handle various formats
            if kwargs and not result:
                for key in ['correlation_data', 'corr_data', 'correlations', 'corr_matrix']:
                    if key in kwargs and kwargs[key] is not None:
                        try:
                            # Recursive call with better error handling
                            recursive_result = self._safe_process_correlation_input(kwargs[key])
                            if recursive_result:
                                result.update(recursive_result)
                                break
                        except:
                            continue
                            
        except Exception as e:
            self.logger.error(f"Error processing correlation input: {e}")
            
        return result

    def _analyze_correlations(
        self,
        correlations: Dict[Tuple[str, str], float],
        positions: Optional[Dict[str, Any]]
    ) -> bool:
        """Analyze correlations and generate alerts"""
        all_correlations = []
        critical_found = False
        
        for (inst1, inst2), corr in correlations.items():
            try:
                abs_corr = abs(float(corr))
                all_correlations.append(abs_corr)
                
                # Determine severity
                severity = self._get_correlation_severity(abs_corr)
                
                if severity != "none":
                    self.alerts[severity].append((inst1, inst2))
                    
                    # Log significant correlations only
                    if severity in ["warning", "critical"]:
                        msg = f"{inst1}/{inst2} correlation {corr:.3f} - {severity.upper()}"
                        if severity == "critical":
                            self.logger.error(msg)
                            critical_found = True
                        else:
                            self.logger.warning(msg)
                            
                        # Record audit for significant events only
                        self._record_audit(inst1, inst2, corr, severity, positions)
                        
            except (ValueError, TypeError):
                continue
        
        # Calculate risk metrics
        if all_correlations:
            avg_correlation = float(np.mean(all_correlations))
            max_correlation = float(np.max(all_correlations))
        else:
            avg_correlation = 0.0
            max_correlation = 0.0
            
        # Update history
        self.correlation_history.append({
            "avg": avg_correlation,
            "max": max_correlation,
            "critical_count": len(self.alerts["critical"])
        })
        
        # Calculate risk score
        self._calculate_risk_score(avg_correlation, max_correlation)
        
        return critical_found

    def _get_correlation_severity(self, abs_corr: float) -> str:
        """Determine correlation severity"""
        if abs_corr >= self.max_corr:
            return "critical"
        elif abs_corr >= self.warning_corr:
            return "warning"
        elif abs_corr >= self.info_corr:
            return "info"
        else:
            return "none"

    def _calculate_risk_score(self, avg_correlation: float, max_correlation: float):
        """Calculate overall correlation risk score"""
        base_score = 0.0
        
        # Weight by alert counts
        base_score += len(self.alerts["critical"]) * 0.4
        base_score += len(self.alerts["warning"]) * 0.2
        base_score += len(self.alerts["info"]) * 0.1
        
        # Factor in correlation levels
        base_score += avg_correlation * 0.3
        base_score += max_correlation * 0.2
        
        self.risk_score = min(base_score, 1.0)

    def _record_audit(
        self,
        inst1: str,
        inst2: str,
        corr: float,
        severity: str,
        positions: Optional[Dict[str, Any]]
    ):
        """Record audit entry (reduced frequency)"""
        entry = {
            "timestamp": utcnow(),
            "step": self.step_count,
            "pair": [inst1, inst2],
            "correlation": float(corr),
            "severity": severity,
        }
        
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit.pop(0)
            
        # Write to file less frequently
        if severity == "critical" or len(self._audit) % 20 == 0:
            try:
                with open(self.AUDIT_PATH, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass

    def get_observation_components(self) -> np.ndarray:
        """Return correlation metrics as observation"""
        try:
            avg_corr = np.mean([h["avg"] for h in self.correlation_history]) if self.correlation_history else 0.0
            max_corr = max([h["max"] for h in self.correlation_history]) if self.correlation_history else 0.0
            
            return np.array([
                float(self.risk_score),
                float(avg_corr),
                float(max_corr),
                float(len(self.alerts["critical"]) / 10.0)
            ], dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "thresholds": {
                "max_corr": self.max_corr,
                "warning_corr": self.warning_corr,
                "info_corr": self.info_corr
            },
            "enabled": self.enabled,
            "step_count": self.step_count,
            "risk_score": float(self.risk_score),
        }

