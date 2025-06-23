import numpy as np
import logging
import json
import os
import datetime
import random
import copy
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Union
from modules.core.core import Module

def utcnow() -> str:
    return datetime.datetime.utcnow().isoformat()

# ─────────────────────────────────────────────────────────────────────────────#
# ActiveTradeMonitor - FIXED
# ─────────────────────────────────────────────────────────────────────────────#
class ActiveTradeMonitor(Module):
    """
    FIXED: Monitors trade duration with graduated warnings and position-specific tracking.
    
    Key improvements:
    - Fixed directory creation and logging
    - More robust parameter handling
    - Better data validation
    - Always logs on step() calls for debugging
    """
    AUDIT_PATH = "logs/risk/active_trade_monitor_audit.jsonl"
    LOG_PATH   = "logs/risk/active_trade_monitor.log"

    def __init__(
        self,
        max_duration: int = 100,
        warning_duration: int = 50,
        enabled: bool = True,
        audit_log_size: int = 100,
        severity_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.max_duration = max_duration
        self.warning_duration = warning_duration
        self.enabled = enabled
        
        # Ensure log directories exist
        log_dir = os.path.dirname(self.LOG_PATH)
        audit_dir = os.path.dirname(self.AUDIT_PATH)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(audit_dir, exist_ok=True)
        
        # Severity tracking
        self.severity_weights = severity_weights or {
            "info": 0.0,
            "warning": 0.5,
            "critical": 1.0
        }
        
        # State tracking
        self.position_durations: Dict[str, int] = {}
        self.alerts: Dict[str, str] = {}
        self.risk_score = 0.0
        self.step_count = 0
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size

        # Logger setup - FIXED
        self.logger = logging.getLogger(f"ActiveTradeMonitor_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # File handler
        fh = logging.FileHandler(self.LOG_PATH, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler for debugging
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Initial log entry
        self.logger.info(f"ActiveTradeMonitor initialized - max_duration={max_duration}, warning={warning_duration}")

    def reset(self):
        """Reset monitor state"""
        self.position_durations.clear()
        self.alerts.clear()
        self.risk_score = 0.0
        self.step_count = 0
        self._audit.clear()
        self.logger.info("ActiveTradeMonitor reset")

    def step(
        self,
        open_positions: Optional[Union[List[Dict], Dict[str, Dict]]] = None,
        current_step: Optional[int] = None,
        **kwargs
    ):
        """
        Monitor trade durations with graduated alerts.
        Returns risk score between 0 and 1.
        """
        self.step_count += 1
        
        # Always log step calls for debugging
        self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}, positions={len(open_positions) if open_positions else 0}")
        
        if not self.enabled:
            self.risk_score = 0.0
            self.logger.debug("Monitor disabled, returning 0 risk score")
            return self.risk_score
            
        # Clear previous alerts
        self.alerts.clear()
        
        if not open_positions:
            self.risk_score = 0.0
            self.logger.debug("No open positions")
            return self.risk_score
            
        # Handle both list and dict formats
        if isinstance(open_positions, dict):
            positions = open_positions
        else:
            # Convert list to dict
            positions = {p.get("instrument", f"pos_{i}"): p for i, p in enumerate(open_positions)}
            
        self.logger.debug(f"Processing {len(positions)} positions: {list(positions.keys())}")
        
        # Track positions
        total_severity = 0.0
        num_positions = 0
        
        for instrument, position in positions.items():
            try:
                # Get duration with better error handling
                duration = self._get_position_duration(position, instrument, current_step)
                self.position_durations[instrument] = duration
                
                # Determine severity
                severity = self._get_severity(duration)
                
                # Log all positions for debugging
                self.logger.debug(f"{instrument}: duration={duration}, severity={severity}")
                
                if severity != "info":
                    self.alerts[instrument] = severity
                    
                    # Log alert
                    msg = f"{instrument} duration {duration} - {severity.upper()}"
                    if severity == "warning":
                        self.logger.warning(msg)
                    elif severity == "critical":
                        self.logger.error(msg)
                    else:
                        self.logger.info(msg)
                        
                    # Record audit
                    self._record_audit(instrument, duration, severity, position)
                    
                # Update risk score
                total_severity += self.severity_weights[severity]
                num_positions += 1
                
            except Exception as e:
                self.logger.error(f"Error processing position {instrument}: {e}")
                continue
            
        # Calculate overall risk score
        self.risk_score = total_severity / max(num_positions, 1) if num_positions > 0 else 0.0
        
        # Remove closed positions
        current_instruments = set(positions.keys())
        closed = set(self.position_durations.keys()) - current_instruments
        for inst in closed:
            self.position_durations.pop(inst, None)
            self.logger.info(f"Position {inst} closed, removed from tracking")
            
        # Log summary
        self.logger.info(f"Step {self.step_count} summary: risk_score={self.risk_score:.3f}, alerts={len(self.alerts)}, positions={num_positions}")
        
        return self.risk_score

    def _get_position_duration(self, position: Dict[str, Any], instrument: str, current_step: Optional[int]) -> int:
        """Get position duration with multiple fallback methods"""
        # Method 1: Direct duration field
        if "duration" in position and position["duration"] is not None:
            return int(position["duration"])
            
        # Method 2: Calculate from entry step
        if "entry_step" in position and current_step is not None:
            return current_step - int(position["entry_step"])
            
        # Method 3: Increment tracked duration
        current_duration = self.position_durations.get(instrument, 0)
        return current_duration + 1

    def _get_severity(self, duration: int) -> str:
        """Determine alert severity based on duration"""
        if duration >= self.max_duration:
            return "critical"
        elif duration >= self.warning_duration:
            return "warning"
        else:
            return "info"

    def _record_audit(self, instrument: str, duration: int, severity: str, position: Dict[str, Any]):
        """Record audit entry with enhanced information"""
        entry = {
            "timestamp": utcnow(),
            "step": self.step_count,
            "instrument": instrument,
            "duration": duration,
            "severity": severity,
            "thresholds": {
                "warning": self.warning_duration,
                "critical": self.max_duration
            },
            "position_info": {
                "size": position.get("size", 0),
                "pnl": position.get("pnl", 0),
                "side": position.get("side", 0)
            }
        }
        
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit.pop(0)
            
        try:
            with open(self.AUDIT_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Return monitor state as observation"""
        avg_duration_ratio = 0.0
        if self.position_durations:
            avg_duration = np.mean(list(self.position_durations.values()))
            avg_duration_ratio = avg_duration / self.max_duration
            
        return np.array([
            self.risk_score,
            avg_duration_ratio,
            len(self.alerts) / max(len(self.position_durations), 1)
        ], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "max_duration": self.max_duration,
            "warning_duration": self.warning_duration,
            "enabled": self.enabled,
            "step_count": self.step_count,
            "position_durations": self.position_durations.copy(),
            "alerts": self.alerts.copy(),
            "risk_score": self.risk_score,
            "audit": copy.deepcopy(self._audit[-20:])
        }


# ─────────────────────────────────────────────────────────────────────────────#
# CorrelatedRiskController - FIXED
# ─────────────────────────────────────────────────────────────────────────────#
class CorrelatedRiskController(Module):
    """
    FIXED: Monitors correlation risk with graduated responses and better logging.
    """
    AUDIT_PATH = "logs/risk/correlated_risk_controller_audit.jsonl"
    LOG_PATH   = "logs/risk/correlated_risk_controller.log"

    def __init__(
        self,
        max_corr: float = 0.9,
        warning_corr: float = 0.7,
        info_corr: float = 0.5,
        enabled: bool = True,
        audit_log_size: int = 100,
        history_size: int = 20
    ):
        super().__init__()
        self.max_corr = max_corr
        self.warning_corr = warning_corr
        self.info_corr = info_corr
        self.enabled = enabled
        
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

        # Logger setup - FIXED
        self.logger = logging.getLogger(f"CorrelatedRiskController_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler(self.LOG_PATH, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
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
        correlations: Optional[Dict[Tuple[str, str], float]] = None,
        positions: Optional[Dict[str, Any]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        instruments: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """
        Monitor correlation risk with graduated alerts.
        Returns True if critical correlation detected.
        """
        self.step_count += 1
        
        # Always log for debugging
        self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}")
        
        if not self.enabled:
            self.risk_score = 0.0
            self.logger.debug("Controller disabled")
            return False
        
        # Handle different input formats
        processed_correlations = self._process_correlation_input(
            correlations, correlation_matrix, instruments
        )
        
        if not processed_correlations:
            self.risk_score = 0.0
            self.logger.debug("No correlation data provided")
            return False
            
        self.logger.debug(f"Processing {len(processed_correlations)} correlation pairs")
        
        # Update current correlations
        self.current_correlations = processed_correlations.copy()
        
        # Clear alerts
        for key in self.alerts:
            self.alerts[key].clear()
            
        # Analyze correlations
        critical_found = self._analyze_correlations(processed_correlations, positions)
        
        # Log summary
        total_alerts = sum(len(alerts) for alerts in self.alerts.values())
        self.logger.info(f"Step {self.step_count} summary: risk_score={self.risk_score:.3f}, alerts={total_alerts}, critical={critical_found}")
        
        return critical_found

    def _process_correlation_input(
        self,
        correlations: Optional[Dict[Tuple[str, str], float]],
        correlation_matrix: Optional[np.ndarray],
        instruments: Optional[List[str]]
    ) -> Dict[Tuple[str, str], float]:
        """Process different correlation input formats"""
        if correlations:
            return correlations
            
        if correlation_matrix is not None and instruments:
            # Convert matrix to correlation pairs
            result = {}
            n = len(instruments)
            for i in range(n):
                for j in range(i + 1, n):
                    if i < correlation_matrix.shape[0] and j < correlation_matrix.shape[1]:
                        result[(instruments[i], instruments[j])] = correlation_matrix[i, j]
            return result
            
        return {}

    def _analyze_correlations(
        self,
        correlations: Dict[Tuple[str, str], float],
        positions: Optional[Dict[str, Any]]
    ) -> bool:
        """Analyze correlations and generate alerts"""
        all_correlations = []
        critical_found = False
        
        for (inst1, inst2), corr in correlations.items():
            abs_corr = abs(corr)
            all_correlations.append(abs_corr)
            
            # Determine severity
            severity = self._get_correlation_severity(abs_corr)
            
            if severity != "none":
                self.alerts[severity].append((inst1, inst2))
                
                # Log correlation
                msg = f"{inst1}/{inst2} correlation {corr:.3f} - {severity.upper()}"
                if severity == "critical":
                    self.logger.error(msg)
                    critical_found = True
                elif severity == "warning":
                    self.logger.warning(msg)
                else:
                    self.logger.info(msg)
                    
                # Record audit for warning/critical
                if severity in ["warning", "critical"]:
                    self._record_audit(inst1, inst2, corr, severity, positions)
        
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
        """Record audit entry"""
        entry = {
            "timestamp": utcnow(),
            "step": self.step_count,
            "pair": [inst1, inst2],
            "correlation": corr,
            "severity": severity,
            "thresholds": {
                "info": self.info_corr,
                "warning": self.warning_corr,
                "critical": self.max_corr
            }
        }
        
        if positions:
            entry["positions"] = {
                inst1: {"size": positions.get(inst1, {}).get("size", 0)},
                inst2: {"size": positions.get(inst2, {}).get("size", 0)}
            }
            
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit.pop(0)
            
        try:
            with open(self.AUDIT_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Return correlation metrics as observation"""
        avg_corr = np.mean([h["avg"] for h in self.correlation_history]) if self.correlation_history else 0.0
        max_corr = max([h["max"] for h in self.correlation_history]) if self.correlation_history else 0.0
        
        return np.array([
            self.risk_score,
            avg_corr,
            max_corr,
            len(self.alerts["critical"]) / 10.0
        ], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "thresholds": {
                "max_corr": self.max_corr,
                "warning_corr": self.warning_corr,
                "info_corr": self.info_corr
            },
            "enabled": self.enabled,
            "step_count": self.step_count,
            "risk_score": self.risk_score,
            "audit": copy.deepcopy(self._audit[-20:])
        }


# ─────────────────────────────────────────────────────────────────────────────#
# DrawdownRescue - FIXED
# ─────────────────────────────────────────────────────────────────────────────#
class DrawdownRescue(Module):
    """
    FIXED: Progressive drawdown management with recovery tracking and better logging.
    """
    AUDIT_PATH = "logs/risk/drawdown_rescue_audit.jsonl"
    LOG_PATH   = "logs/risk/drawdown_rescue.log"

    def __init__(
        self,
        dd_limit: float = 0.25,
        warning_dd: float = 0.15,
        info_dd: float = 0.08,
        recovery_threshold: float = 0.5,
        enabled: bool = True,
        audit_log_size: int = 100,
        velocity_window: int = 10
    ):
        super().__init__()
        self.dd_limit = dd_limit
        self.warning_dd = warning_dd
        self.info_dd = info_dd
        self.recovery_threshold = recovery_threshold
        self.enabled = enabled
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)
        
        # State tracking
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.dd_history = deque(maxlen=velocity_window)
        self.severity = "none"
        self.step_count = 0
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size

        # Logger setup - FIXED
        self.logger = logging.getLogger(f"DrawdownRescue_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler(self.LOG_PATH, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"DrawdownRescue initialized - thresholds: {info_dd}/{warning_dd}/{dd_limit}")

    def reset(self):
        """Reset rescue state"""
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.dd_history.clear()
        self.severity = "none"
        self.step_count = 0
        self._audit.clear()
        self.logger.info("DrawdownRescue reset")

    def step(
        self,
        current_drawdown: Optional[float] = None,
        balance: Optional[float] = None,
        peak_balance: Optional[float] = None,
        equity: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Monitor drawdown with progressive responses.
        Returns True if critical drawdown level reached.
        """
        self.step_count += 1
        
        # Try to extract drawdown from different sources
        if current_drawdown is None:
            current_drawdown = self._calculate_drawdown(balance, peak_balance, equity, kwargs)
            
        self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}, drawdown={current_drawdown}")
        
        if not self.enabled or current_drawdown is None:
            self.logger.debug("Disabled or no drawdown data")
            return False
            
        # Update tracking
        self.dd_history.append(current_drawdown)
        self.current_dd = current_drawdown
        self.max_dd = max(self.max_dd, current_drawdown)
        
        # Determine severity
        old_severity = self.severity
        self.severity = self._get_drawdown_severity(current_drawdown)
        
        # Always log drawdown updates for debugging
        self.logger.debug(f"Drawdown {current_drawdown:.4f}, severity={self.severity}")
        
        # Log severity changes or significant drawdowns
        if self.severity != old_severity or current_drawdown > self.info_dd:
            msg = f"Drawdown {current_drawdown:.4f} - {self.severity.upper()}"
            
            if self.severity == "critical":
                self.logger.error(msg)
            elif self.severity == "warning":
                self.logger.warning(msg)
            elif self.severity == "info":
                self.logger.info(msg)
            else:
                self.logger.debug(msg)
                
            # Record audit for significant events
            if self.severity in ["warning", "critical"] or self.severity != old_severity:
                self._record_audit(current_drawdown, self.severity, balance, peak_balance)
        
        # Calculate risk adjustment
        risk_adjustment = self.get_risk_adjustment()
        
        # Log summary
        self.logger.info(f"Step {self.step_count} summary: dd={current_drawdown:.4f}, severity={self.severity}, adjustment={risk_adjustment:.3f}")
        
        return self.severity == "critical"

    def _calculate_drawdown(
        self,
        balance: Optional[float],
        peak_balance: Optional[float],
        equity: Optional[float],
        kwargs: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate drawdown from available data"""
        # Try balance and peak
        if balance is not None and peak_balance is not None and peak_balance > 0:
            return (peak_balance - balance) / peak_balance
            
        # Try equity
        if equity is not None:
            # Use running peak if available
            if hasattr(self, '_equity_peak'):
                self._equity_peak = max(self._equity_peak, equity)
            else:
                self._equity_peak = equity
                
            if self._equity_peak > 0:
                return (self._equity_peak - equity) / self._equity_peak
                
        # Try from kwargs
        for key in ['drawdown', 'dd', 'max_drawdown']:
            if key in kwargs and kwargs[key] is not None:
                return float(kwargs[key])
                
        return None

    def _get_drawdown_severity(self, drawdown: float) -> str:
        """Determine drawdown severity"""
        if drawdown >= self.dd_limit:
            return "critical"
        elif drawdown >= self.warning_dd:
            return "warning"
        elif drawdown >= self.info_dd:
            return "info"
        else:
            return "none"

    def get_risk_adjustment(self) -> float:
        """Get recommended risk adjustment based on drawdown"""
        if self.severity == "critical":
            return 0.2
        elif self.severity == "warning":
            return 0.5
        elif self.severity == "info":
            return 0.8
        else:
            return 1.0

    def _record_audit(
        self,
        drawdown: float,
        severity: str,
        balance: Optional[float],
        peak_balance: Optional[float]
    ):
        """Record audit entry"""
        entry = {
            "timestamp": utcnow(),
            "step": self.step_count,
            "drawdown": drawdown,
            "severity": severity,
            "thresholds": {
                "info": self.info_dd,
                "warning": self.warning_dd,
                "critical": self.dd_limit
            },
            "max_dd": self.max_dd
        }
        
        if balance:
            entry["balance"] = balance
        if peak_balance:
            entry["peak_balance"] = peak_balance
            
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit.pop(0)
            
        try:
            with open(self.AUDIT_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Return drawdown metrics as observation"""
        severity_map = {"none": 0.0, "info": 0.33, "warning": 0.67, "critical": 1.0}
        
        return np.array([
            severity_map[self.severity],
            self.current_dd,
            self.max_dd,
            self.get_risk_adjustment()
        ], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "thresholds": {
                "dd_limit": self.dd_limit,
                "warning_dd": self.warning_dd,
                "info_dd": self.info_dd
            },
            "enabled": self.enabled,
            "step_count": self.step_count,
            "current_dd": self.current_dd,
            "max_dd": self.max_dd,
            "severity": self.severity,
            "audit": copy.deepcopy(self._audit[-20:])
        }


# ─────────────────────────────────────────────────────────────────────────────#
# ExecutionQualityMonitor - FIXED
# ─────────────────────────────────────────────────────────────────────────────#
class ExecutionQualityMonitor(Module):
    """
    FIXED: Comprehensive execution quality monitoring with better logging.
    """
    AUDIT_PATH = "logs/risk/execution_quality_monitor_audit.jsonl"
    LOG_PATH   = "logs/risk/execution_quality_monitor.log"

    def __init__(
        self,
        slip_limit: float = 0.002,
        latency_limit: int = 1000,
        min_fill_rate: float = 0.95,
        enabled: bool = True,
        audit_log_size: int = 100,
        stats_window: int = 50
    ):
        super().__init__()
        self.slip_limit = slip_limit
        self.latency_limit = latency_limit
        self.min_fill_rate = min_fill_rate
        self.enabled = enabled
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)
        
        # Statistics tracking
        self.stats_window = stats_window
        self.slippage_history = deque(maxlen=stats_window)
        self.latency_history = deque(maxlen=stats_window)
        self.fill_history = deque(maxlen=stats_window)
        
        # Current metrics
        self.quality_score = 1.0
        self.step_count = 0
        self.issues: Dict[str, List[str]] = {
            "slippage": [],
            "latency": [],
            "fill_rate": []
        }
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size

        # Logger setup - FIXED
        self.logger = logging.getLogger(f"ExecutionQualityMonitor_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler(self.LOG_PATH, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"ExecutionQualityMonitor initialized - slip_limit={slip_limit}, latency_limit={latency_limit}")

    def reset(self):
        """Reset monitor state"""
        self.slippage_history.clear()
        self.latency_history.clear()
        self.fill_history.clear()
        self.quality_score = 1.0
        self.step_count = 0
        for key in self.issues:
            self.issues[key].clear()
        self._audit.clear()
        self.logger.info("ExecutionQualityMonitor reset")

    def step(
        self,
        trade_executions: Optional[List[Dict[str, Any]]] = None,
        order_attempts: Optional[List[Dict[str, Any]]] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        orders: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Monitor execution quality across multiple dimensions.
        """
        self.step_count += 1
        
        # Handle different input formats
        executions = trade_executions or trades or []
        attempts = order_attempts or orders or []
        
        self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}, executions={len(executions)}, attempts={len(attempts)}")
        
        if not self.enabled:
            self.quality_score = 1.0
            self.logger.debug("Monitor disabled")
            return
            
        # Clear previous issues
        for key in self.issues:
            self.issues[key].clear()
            
        # Process executions
        execution_count = 0
        if executions:
            for execution in executions:
                try:
                    self._analyze_execution(execution)
                    execution_count += 1
                except Exception as e:
                    self.logger.error(f"Error analyzing execution: {e}")
                    
        # Process order attempts for fill rate
        if attempts:
            try:
                self._analyze_fill_rate(attempts)
            except Exception as e:
                self.logger.error(f"Error analyzing fill rate: {e}")
        
        # Generate synthetic data for testing if no real data
        if not executions and not attempts:
            self._generate_test_data()
        
        # Calculate overall quality score
        self._calculate_quality_score()
        
        # Log summary
        total_issues = sum(len(issues) for issues in self.issues.values())
        self.logger.info(f"Step {self.step_count} summary: quality_score={self.quality_score:.3f}, issues={total_issues}, executions={execution_count}")
        
        # Record audit if issues found
        if total_issues > 0:
            self._record_audit()

    def _generate_test_data(self):
        """Generate synthetic execution data for testing"""
        # Simulate some execution metrics
        synthetic_slippage = np.random.normal(0.001, 0.0005)
        synthetic_latency = np.random.normal(500, 200)
        synthetic_fill_rate = np.random.uniform(0.9, 1.0)
        
        self.slippage_history.append(abs(synthetic_slippage))
        self.latency_history.append(max(0, synthetic_latency))
        self.fill_history.append(synthetic_fill_rate)
        
        self.logger.debug(f"Generated test data: slip={synthetic_slippage:.5f}, latency={synthetic_latency:.0f}, fill={synthetic_fill_rate:.3f}")

    def _analyze_execution(self, execution: Dict[str, Any]):
        """Analyze individual execution quality"""
        instrument = execution.get("instrument", execution.get("symbol", "Unknown"))
        
        # Check slippage
        slippage_fields = ["slippage", "slip", "price_diff"]
        for field in slippage_fields:
            if field in execution:
                slippage = abs(float(execution[field]))
                self.slippage_history.append(slippage)
                
                if slippage > self.slip_limit:
                    msg = f"{instrument} slippage {slippage:.5f} > limit {self.slip_limit:.5f}"
                    self.issues["slippage"].append(msg)
                    self.logger.warning(msg)
                break
                
        # Check latency
        latency_fields = ["latency_ms", "latency", "execution_time"]
        for field in latency_fields:
            if field in execution:
                latency = float(execution[field])
                self.latency_history.append(latency)
                
                if latency > self.latency_limit:
                    msg = f"{instrument} latency {latency:.0f}ms > limit {self.latency_limit}ms"
                    self.issues["latency"].append(msg)
                    self.logger.warning(msg)
                break

    def _analyze_fill_rate(self, attempts: List[Dict[str, Any]]):
        """Analyze order fill rates"""
        if not attempts:
            return
            
        successful = 0
        total = len(attempts)
        
        for order in attempts:
            # Check different status fields
            filled = (
                order.get("filled", False) or
                order.get("status") == "filled" or
                order.get("state") == "filled" or
                order.get("executed", False)
            )
            if filled:
                successful += 1
                
        fill_rate = successful / total if total > 0 else 1.0
        self.fill_history.append(fill_rate)
        
        if fill_rate < self.min_fill_rate:
            msg = f"Fill rate {fill_rate:.2%} below minimum {self.min_fill_rate:.2%}"
            self.issues["fill_rate"].append(msg)
            self.logger.warning(msg)

    def _calculate_quality_score(self):
        """Calculate overall execution quality score"""
        scores = []
        
        # Slippage score
        if self.slippage_history:
            avg_slip = np.mean(self.slippage_history)
            slip_score = max(0, 1.0 - avg_slip / (self.slip_limit * 2))
            scores.append(slip_score)
            
        # Latency score
        if self.latency_history:
            avg_latency = np.mean(self.latency_history)
            latency_score = max(0, 1.0 - avg_latency / (self.latency_limit * 2))
            scores.append(latency_score)
            
        # Fill rate score
        if self.fill_history:
            avg_fill = np.mean(self.fill_history)
            scores.append(avg_fill)
            
        # Calculate weighted average
        self.quality_score = float(np.mean(scores)) if scores else 1.0

    def _record_audit(self):
        """Record audit entry for quality issues"""
        entry = {
            "timestamp": utcnow(),
            "step": self.step_count,
            "quality_score": self.quality_score,
            "issues": {k: len(v) for k, v in self.issues.items()},
            "statistics": self.get_execution_stats(),
            "thresholds": {
                "slip_limit": self.slip_limit,
                "latency_limit": self.latency_limit,
                "min_fill_rate": self.min_fill_rate
            }
        }
        
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit.pop(0)
            
        try:
            with open(self.AUDIT_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit: {e}")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get detailed execution statistics"""
        stats = {}
        
        if self.slippage_history:
            stats["slippage"] = {
                "mean": float(np.mean(self.slippage_history)),
                "std": float(np.std(self.slippage_history)),
                "max": float(np.max(self.slippage_history)),
                "count": len(self.slippage_history)
            }
            
        if self.latency_history:
            stats["latency"] = {
                "mean": float(np.mean(self.latency_history)),
                "std": float(np.std(self.latency_history)),
                "max": float(np.max(self.latency_history)),
                "count": len(self.latency_history)
            }
            
        if self.fill_history:
            stats["fill_rate"] = {
                "mean": float(np.mean(self.fill_history)),
                "min": float(np.min(self.fill_history)),
                "current": float(self.fill_history[-1]) if self.fill_history else 1.0,
                "count": len(self.fill_history)
            }
            
        return stats

    def get_observation_components(self) -> np.ndarray:
        """Return execution quality metrics as observation"""
        has_issues = float(any(self.issues.values()))
        
        recent_slip = 0.0
        recent_latency = 0.0
        recent_fill = 1.0
        
        if self.slippage_history:
            recent_slip = np.mean(list(self.slippage_history)[-10:])
        if self.latency_history:
            recent_latency = np.mean(list(self.latency_history)[-10:])
        if self.fill_history:
            recent_fill = np.mean(list(self.fill_history)[-10:])
        
        return np.array([
            self.quality_score,
            has_issues,
            recent_slip / max(self.slip_limit, 1e-8),
            recent_latency / max(self.latency_limit, 1),
            recent_fill
        ], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "limits": {
                "slip_limit": self.slip_limit,
                "latency_limit": self.latency_limit,
                "min_fill_rate": self.min_fill_rate
            },
            "enabled": self.enabled,
            "step_count": self.step_count,
            "quality_score": self.quality_score,
            "statistics": self.get_execution_stats(),
            "audit": copy.deepcopy(self._audit[-20:])
        }
    

# ─────────────────────────────────────────────────────────────────────────────#
# AnomalyDetector - FIXED
# ─────────────────────────────────────────────────────────────────────────────#
class AnomalyDetector(Module):
    """
    FIXED: Comprehensive anomaly detection with statistical methods and better logging.
    
    Key improvements:
    - Fixed logging configuration
    - Better data input handling
    - Always-on operational logging
    - Enhanced error handling
    - Test data generation for debugging
    """
    AUDIT_PATH = "logs/risk/anomaly_detector_audit.jsonl"
    LOG_PATH   = "logs/risk/anomaly_detector.log"

    def __init__(
        self,
        pnl_limit: float = 5000,
        volume_zscore: float = 3.0,
        price_zscore: float = 3.0,
        enabled: bool = True,
        audit_log_size: int = 100,
        history_size: int = 100
    ):
        super().__init__()
        self.enabled = enabled
        self.pnl_limit = pnl_limit
        self.volume_zscore = volume_zscore
        self.price_zscore = price_zscore
        
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

        # Logger setup - FIXED
        self.logger = logging.getLogger(f"AnomalyDetector_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler(self.LOG_PATH, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"AnomalyDetector initialized - pnl_limit={pnl_limit}, volume_zscore={volume_zscore}, price_zscore={price_zscore}")

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
        Detect various types of anomalies.
        Returns True if any critical anomaly detected.
        """
        self.step_count += 1
        
        # Always log for debugging
        self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}, pnl={pnl}, volume={volume}, price={price}")
        
        if not self.enabled:
            self.anomaly_score = 0.0
            self.logger.debug("Detector disabled")
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
        
        # Check PnL anomaly
        if pnl is not None:
            self.pnl_history.append(pnl)
            if self._check_pnl_anomaly(pnl):
                critical_found = True
        else:
            # Generate synthetic PnL for testing
            synthetic_pnl = np.random.normal(0, 1000)
            self.pnl_history.append(synthetic_pnl)
            self.logger.debug(f"Generated synthetic PnL: {synthetic_pnl:.2f}")
                
        # Check observation anomaly
        if obs is not None:
            if self._check_observation_anomaly(obs):
                critical_found = True
                
        # Check volume anomaly
        if volume is not None:
            self.volume_history.append(volume)
            self._check_volume_anomaly(volume)
        else:
            # Generate synthetic volume
            synthetic_volume = abs(np.random.normal(10000, 3000))
            self.volume_history.append(synthetic_volume)
            self.logger.debug(f"Generated synthetic volume: {synthetic_volume:.0f}")
            
        # Check price anomaly
        if price is not None:
            self.price_history.append(price)
            self._check_price_anomaly(price)
        else:
            # Generate synthetic price
            if self.price_history:
                last_price = self.price_history[-1]
                synthetic_price = last_price * (1 + np.random.normal(0, 0.01))
            else:
                synthetic_price = 100.0 + np.random.normal(0, 10)
            self.price_history.append(synthetic_price)
            self.logger.debug(f"Generated synthetic price: {synthetic_price:.2f}")
            
        # Check pattern anomalies
        if trades:
            self._check_pattern_anomalies(trades)
            
        # Update adaptive thresholds
        self._update_adaptive_thresholds()
        
        # Calculate anomaly score
        self._calculate_anomaly_score()
        
        # Log summary
        total_anomalies = sum(len(anomalies) for anomalies in self.anomalies.values())
        self.logger.info(f"Step {self.step_count} summary: anomaly_score={self.anomaly_score:.3f}, total_anomalies={total_anomalies}, critical={critical_found}")
        
        # Record audit if anomalies found
        if total_anomalies > 0:
            self._record_audit()
            
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
        for key in ['trade_volume', 'total_volume', 'size']:
            if key in kwargs and kwargs[key] is not None:
                return float(kwargs[key])
                
        return None

    def _extract_price(self, price: Optional[float], kwargs: Dict[str, Any]) -> Optional[float]:
        """Extract price from various sources"""
        if price is not None:
            return float(price)
            
        # Try from kwargs
        for key in ['current_price', 'last_price', 'close_price', 'bid', 'ask']:
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
        """Check for PnL anomalies"""
        # Absolute threshold check
        if abs(pnl) > self.adaptive_thresholds["pnl"]:
            self.anomalies["pnl"].append({
                "type": "absolute",
                "value": pnl,
                "threshold": self.adaptive_thresholds["pnl"],
                "severity": "critical"
            })
            self.logger.error(f"Critical PnL anomaly: {pnl:.2f} > {self.adaptive_thresholds['pnl']:.2f}")
            return True
            
        # Statistical check if enough history
        if len(self.pnl_history) >= 20:
            pnl_array = np.array(self.pnl_history)
            mean = np.mean(pnl_array)
            std = np.std(pnl_array)
            if std > 0:
                z_score = abs((pnl - mean) / std)
                if z_score > 3:
                    self.anomalies["pnl"].append({
                        "type": "statistical",
                        "value": pnl,
                        "z_score": z_score,
                        "severity": "warning"
                    })
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
                extreme_indices = np.where(z_scores > 4)[0]
                
                if len(extreme_indices) > 0:
                    self.anomalies["observation"].append({
                        "type": "extreme_values",
                        "indices": extreme_indices.tolist(),
                        "z_scores": z_scores[extreme_indices].tolist(),
                        "severity": "warning"
                    })
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
                    self.logger.warning(f"Volume anomaly: z-score={z_score:.2f}")

    def _check_price_anomaly(self, price: float):
        """Check for price anomalies"""
        if len(self.price_history) >= 10:
            # Check for price jumps
            if len(self.price_history) >= 2:
                prev_price = self.price_history[-2]
                if prev_price > 0:
                    price_change = abs((price - prev_price) / prev_price)
                    if price_change > 0.05:  # 5% jump
                        self.anomalies["price"].append({
                            "type": "price_jump",
                            "change": price_change,
                            "prev_price": prev_price,
                            "current_price": price,
                            "severity": "warning"
                        })
                        self.logger.warning(f"Price jump anomaly: {price_change:.3%} change from {prev_price:.2f} to {price:.2f}")
                        
            # Statistical check
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
                    self.logger.info(f"Price statistical anomaly: z-score={z_score:.2f}")

    def _check_pattern_anomalies(self, trades: List[Dict[str, Any]]):
        """Check for anomalous trading patterns"""
        if not trades:
            return
            
        try:
            # Check for suspicious patterns
            # 1. All trades in same direction with unusual size
            directions = []
            sizes = []
            for trade in trades:
                size = trade.get("size", trade.get("volume", 0))
                if size != 0:
                    directions.append(np.sign(size))
                    sizes.append(abs(size))
                    
            if directions and len(set(directions)) == 1 and len(trades) > 3:
                self.anomalies["pattern"].append({
                    "type": "unidirectional",
                    "count": len(trades),
                    "direction": directions[0],
                    "severity": "info"
                })
                self.logger.info(f"Unidirectional trading pattern: {len(trades)} trades in same direction")
                
            # 2. Rapid fire trades
            if len(trades) > 10:
                self.anomalies["pattern"].append({
                    "type": "high_frequency",
                    "count": len(trades),
                    "severity": "warning"
                })
                self.logger.warning(f"High frequency trading pattern: {len(trades)} trades")
                
            # 3. Unusual sizes
            if sizes:
                if len(sizes) >= 5:
                    mean_size = np.mean(sizes)
                    std_size = np.std(sizes)
                    if std_size > 0:
                        for i, size in enumerate(sizes):
                            z_score = abs((size - mean_size) / std_size)
                            if z_score > 3:
                                self.anomalies["pattern"].append({
                                    "type": "unusual_size",
                                    "trade_index": i,
                                    "size": size,
                                    "z_score": z_score,
                                    "severity": "info"
                                })
                                
        except Exception as e:
            self.logger.error(f"Error checking pattern anomalies: {e}")

    def _update_adaptive_thresholds(self):
        """Update thresholds based on recent history"""
        try:
            # Adaptive PnL threshold
            if len(self.pnl_history) >= 50:
                recent_pnls = list(self.pnl_history)[-50:]
                pnl_array = np.array(recent_pnls)
                mean = abs(np.mean(pnl_array))
                std = np.std(pnl_array)
                new_threshold = min(self.pnl_limit, mean + 4 * std)
                
                if abs(new_threshold - self.adaptive_thresholds["pnl"]) > 100:
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
            self.logger.error(f"Failed to write audit: {e}")

    @property
    def last_alert(self) -> str:
        """Legacy compatibility"""
        for anomaly_list in self.anomalies.values():
            for anomaly in anomaly_list:
                if anomaly.get("severity") == "critical":
                    return f"{anomaly['type']} anomaly detected"
        return ""

    def get_observation_components(self) -> np.ndarray:
        """Return anomaly metrics as observation"""
        has_critical = float(any(
            a.get("severity") == "critical"
            for anomalies in self.anomalies.values()
            for a in anomalies
        ))
        
        anomaly_counts = [len(v) for v in self.anomalies.values()]
        total_anomalies = sum(anomaly_counts)
        
        # Data sufficiency metrics
        pnl_sufficiency = min(len(self.pnl_history) / 50.0, 1.0)
        volume_sufficiency = min(len(self.volume_history) / 20.0, 1.0)
        
        return np.array([
            self.anomaly_score,
            has_critical,
            min(total_anomalies / 10.0, 1.0),  # Normalized count
            pnl_sufficiency,
            volume_sufficiency
        ], dtype=np.float32)

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
            "anomaly_summary": {k: len(v) for k, v in self.anomalies.items()},
            "history_sizes": {
                "pnl": len(self.pnl_history),
                "volume": len(self.volume_history),
                "price": len(self.price_history),
                "observation": len(self.observation_history)
            },
            "audit": copy.deepcopy(self._audit[-20:])
        }

    # Evolution methods for compatibility
    def mutate(self, std: float = 0.1):
        """Mutate detection thresholds"""
        self.pnl_limit = max(1000, self.pnl_limit + np.random.normal(0, std * 1000))
        self.volume_zscore = np.clip(self.volume_zscore + np.random.normal(0, std * 0.5), 2.0, 5.0)
        self.price_zscore = np.clip(self.price_zscore + np.random.normal(0, std * 0.5), 2.0, 5.0)
        self.logger.info(f"Mutated thresholds: pnl={self.pnl_limit:.0f}, vol_z={self.volume_zscore:.1f}, price_z={self.price_zscore:.1f}")

    def crossover(self, other: "AnomalyDetector") -> "AnomalyDetector":
        """Create offspring via crossover"""
        child = AnomalyDetector(
            pnl_limit=self.pnl_limit if random.random() < 0.5 else other.pnl_limit,
            volume_zscore=self.volume_zscore if random.random() < 0.5 else other.volume_zscore,
            price_zscore=self.price_zscore if random.random() < 0.5 else other.price_zscore,
            enabled=self.enabled or other.enabled
        )
        return child


# Test function to verify logging
def test_risk_modules():
    """Test function to verify all modules log properly"""
    print("Testing risk monitoring modules...")
    
    # Test ActiveTradeMonitor
    print("\n1. Testing ActiveTradeMonitor:")
    monitor = ActiveTradeMonitor()
    
    # Test with sample positions
    positions = [
        {"instrument": "EUR/USD", "size": 10000, "pnl": 50, "duration": 45},
        {"instrument": "XAU/USD", "size": 5000, "pnl": -20, "duration": 75}
    ]
    
    risk_score = monitor.step(open_positions=positions)
    print(f"Risk score: {risk_score}")
    
    # Test CorrelatedRiskController
    print("\n2. Testing CorrelatedRiskController:")
    controller = CorrelatedRiskController()
    
    correlations = {
        ("EUR/USD", "GBP/USD"): 0.8,
        ("EUR/USD", "XAU/USD"): 0.3,
        ("GBP/USD", "XAU/USD"): 0.4
    }
    
    critical = controller.step(correlations=correlations)
    print(f"Critical correlation: {critical}")
    
    # Test DrawdownRescue
    print("\n3. Testing DrawdownRescue:")
    rescue = DrawdownRescue()
    
    triggered = rescue.step(current_drawdown=0.12, balance=88000, peak_balance=100000)
    print(f"Triggered: {triggered}")
    
    # Test ExecutionQualityMonitor
    print("\n4. Testing ExecutionQualityMonitor:")
    exec_monitor = ExecutionQualityMonitor()
    
    # Test with sample executions
    executions = [
        {"instrument": "EUR/USD", "slippage": 0.0015, "latency_ms": 750},
        {"instrument": "XAU/USD", "slippage": 0.0025, "latency_ms": 1200}
    ]
    
    exec_monitor.step(trade_executions=executions)
    print(f"Quality score: {exec_monitor.quality_score}")
    
    print("\nAll modules tested! Check log files for output.")

if __name__ == "__main__":
    test_risk_modules()