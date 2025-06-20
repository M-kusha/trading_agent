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
    - Graduated warnings (info, warning, critical)
    - Per-position tracking
    - Integration with position manager
    - More realistic defaults
    """
    AUDIT_PATH = "logs/risk/active_trade_monitor_audit.jsonl"
    LOG_PATH   = "logs/risk/active_trade_monitor.log"

    def __init__(
        self,
        max_duration: int = 100,  # Increased from 50
        warning_duration: int = 50,  # New: warning level
        enabled: bool = True,
        audit_log_size: int = 100,
        severity_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.max_duration = max_duration
        self.warning_duration = warning_duration
        self.enabled = enabled
        
        # Severity tracking
        self.severity_weights = severity_weights or {
            "info": 0.0,
            "warning": 0.5,
            "critical": 1.0
        }
        
        # State tracking
        self.position_durations: Dict[str, int] = {}
        self.alerts: Dict[str, str] = {}  # instrument -> severity
        self.risk_score = 0.0
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        # Logger
        self.logger = logging.getLogger("ActiveTradeMonitor")
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        """Reset monitor state"""
        self.position_durations.clear()
        self.alerts.clear()
        self.risk_score = 0.0
        self._audit.clear()

    def update_position_duration(self, instrument: str, duration: int):
        """Update duration for a specific position"""
        self.position_durations[instrument] = duration

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
        if not self.enabled:
            self.risk_score = 0.0
            return self.risk_score
            
        # Clear previous alerts
        self.alerts.clear()
        
        if not open_positions:
            self.risk_score = 0.0
            return self.risk_score
            
        # Handle both list and dict formats
        if isinstance(open_positions, dict):
            positions = open_positions
        else:
            # Convert list to dict
            positions = {p.get("instrument", f"pos_{i}"): p for i, p in enumerate(open_positions)}
            
        # Track positions
        total_severity = 0.0
        num_positions = 0
        
        for instrument, position in positions.items():
            # Get duration
            if "duration" in position:
                duration = position["duration"]
            elif "entry_step" in position and current_step is not None:
                duration = current_step - position["entry_step"]
            else:
                # Increment tracked duration
                duration = self.position_durations.get(instrument, 0) + 1
                
            self.position_durations[instrument] = duration
            
            # Determine severity
            severity = self._get_severity(duration)
            if severity != "info":
                self.alerts[instrument] = severity
                
                # Log alert
                msg = f"{instrument} duration {duration} - {severity.upper()}"
                if severity == "warning":
                    self.logger.warning(msg)
                elif severity == "critical":
                    self.logger.error(msg)
                    
                # Record audit
                self._record_audit(instrument, duration, severity, position)
                
            # Update risk score
            total_severity += self.severity_weights[severity]
            num_positions += 1
            
        # Calculate overall risk score
        self.risk_score = total_severity / max(num_positions, 1)
        
        # Remove closed positions
        current_instruments = set(positions.keys())
        closed = set(self.position_durations.keys()) - current_instruments
        for inst in closed:
            self.position_durations.pop(inst, None)
            
        return self.risk_score

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

    def get_position_risks(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed risk assessment for each position"""
        risks = {}
        for instrument, duration in self.position_durations.items():
            severity = self._get_severity(duration)
            risks[instrument] = {
                "duration": duration,
                "severity": severity,
                "risk_score": self.severity_weights[severity],
                "time_to_warning": max(0, self.warning_duration - duration),
                "time_to_critical": max(0, self.max_duration - duration)
            }
        return risks

    @property
    def alerted(self) -> bool:
        """Legacy compatibility - returns True if any critical alerts"""
        return any(sev == "critical" for sev in self.alerts.values())

    def get_observation_components(self) -> np.ndarray:
        """Return monitor state as observation"""
        # Average duration ratio, risk score, number of alerts
        avg_duration_ratio = 0.0
        if self.position_durations:
            avg_duration = np.mean(list(self.position_durations.values()))
            avg_duration_ratio = avg_duration / self.max_duration
            
        return np.array([
            self.risk_score,
            avg_duration_ratio,
            len(self.alerts) / max(len(self.position_durations), 1)
        ], dtype=np.float32)

    # Evolution methods
    def mutate(self, std: float = 0.1):
        """Mutate duration thresholds"""
        # Mutate thresholds
        self.warning_duration = int(np.clip(
            self.warning_duration + np.random.normal(0, std * 20),
            10, 200
        ))
        self.max_duration = int(np.clip(
            self.max_duration + np.random.normal(0, std * 30),
            self.warning_duration + 10, 500
        ))
        self.logger.info(f"Mutated: warning={self.warning_duration}, max={self.max_duration}")

    def crossover(self, other: "ActiveTradeMonitor") -> "ActiveTradeMonitor":
        """Create offspring via crossover"""
        child = ActiveTradeMonitor(
            max_duration=self.max_duration if random.random() < 0.5 else other.max_duration,
            warning_duration=self.warning_duration if random.random() < 0.5 else other.warning_duration,
            enabled=self.enabled or other.enabled,
            severity_weights=self.severity_weights.copy()
        )
        return child

    def get_state(self) -> Dict[str, Any]:
        return {
            "max_duration": self.max_duration,
            "warning_duration": self.warning_duration,
            "enabled": self.enabled,
            "position_durations": self.position_durations.copy(),
            "alerts": self.alerts.copy(),
            "risk_score": self.risk_score,
            "audit": copy.deepcopy(self._audit[-20:])  # Last 20 entries
        }

    def set_state(self, state: Dict[str, Any]):
        self.max_duration = state.get("max_duration", self.max_duration)
        self.warning_duration = state.get("warning_duration", self.warning_duration)
        self.enabled = state.get("enabled", self.enabled)
        self.position_durations = state.get("position_durations", {}).copy()
        self.alerts = state.get("alerts", {}).copy()
        self.risk_score = state.get("risk_score", 0.0)
        self._audit = copy.deepcopy(state.get("audit", []))

    def get_last_audit(self) -> Dict[str, Any]:
        return self._audit[-1] if self._audit else {}

    def get_audit_trail(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._audit[-n:]


# ─────────────────────────────────────────────────────────────────────────────#
# CorrelatedRiskController - FIXED
# ─────────────────────────────────────────────────────────────────────────────#
class CorrelatedRiskController(Module):
    """
    FIXED: Monitors correlation risk with graduated responses.
    
    Key improvements:
    - Multiple correlation thresholds
    - Portfolio-wide correlation metrics
    - Integration with position sizing
    - Historical correlation tracking
    """
    AUDIT_PATH = "logs/risk/correlated_risk_controller_audit.jsonl"
    LOG_PATH   = "logs/risk/correlated_risk_controller.log"

    def __init__(
        self,
        max_corr: float = 0.9,  # Increased from 0.8
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
        
        # State tracking
        self.correlation_history = deque(maxlen=history_size)
        self.current_correlations: Dict[Tuple[str, str], float] = {}
        self.risk_score = 0.0
        self.alerts: Dict[str, List[Tuple[str, str]]] = {
            "info": [],
            "warning": [],
            "critical": []
        }
        
        # Risk metrics
        self.avg_correlation = 0.0
        self.max_correlation = 0.0
        self.correlation_clusters: List[List[str]] = []
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        # Logger
        self.logger = logging.getLogger("CorrelatedRiskController")
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        """Reset controller state"""
        self.correlation_history.clear()
        self.current_correlations.clear()
        self.risk_score = 0.0
        for key in self.alerts:
            self.alerts[key].clear()
        self.avg_correlation = 0.0
        self.max_correlation = 0.0
        self.correlation_clusters.clear()
        self._audit.clear()

    def step(
        self,
        correlations: Optional[Dict[Tuple[str, str], float]] = None,
        positions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Monitor correlation risk with graduated alerts.
        
        Returns True if critical correlation detected.
        """
        if not self.enabled or not correlations:
            self.risk_score = 0.0
            return False
            
        # Update current correlations
        self.current_correlations = correlations.copy()
        
        # Clear alerts
        for key in self.alerts:
            self.alerts[key].clear()
            
        # Analyze correlations
        all_correlations = []
        for (inst1, inst2), corr in correlations.items():
            abs_corr = abs(corr)
            all_correlations.append(abs_corr)
            
            # Determine severity
            if abs_corr >= self.max_corr:
                severity = "critical"
                self.alerts["critical"].append((inst1, inst2))
            elif abs_corr >= self.warning_corr:
                severity = "warning"
                self.alerts["warning"].append((inst1, inst2))
            elif abs_corr >= self.info_corr:
                severity = "info"
                self.alerts["info"].append((inst1, inst2))
            else:
                continue
                
            # Log if warning or critical
            if severity in ["warning", "critical"]:
                msg = f"{inst1}/{inst2} correlation {corr:.3f} - {severity.upper()}"
                if severity == "warning":
                    self.logger.warning(msg)
                else:
                    self.logger.error(msg)
                    
                # Record audit
                self._record_audit(inst1, inst2, corr, severity, positions)
                
        # Calculate metrics
        if all_correlations:
            self.avg_correlation = float(np.mean(all_correlations))
            self.max_correlation = float(np.max(all_correlations))
        else:
            self.avg_correlation = 0.0
            self.max_correlation = 0.0
            
        # Update history
        self.correlation_history.append({
            "avg": self.avg_correlation,
            "max": self.max_correlation,
            "critical_count": len(self.alerts["critical"])
        })
        
        # Find correlation clusters
        self._find_correlation_clusters()
        
        # Calculate risk score
        self._calculate_risk_score()
        
        return len(self.alerts["critical"]) > 0

    def _find_correlation_clusters(self):
        """Identify groups of highly correlated instruments"""
        # Build adjacency for high correlations
        high_corr_pairs = []
        for (inst1, inst2), corr in self.current_correlations.items():
            if abs(corr) >= self.warning_corr:
                high_corr_pairs.append({inst1, inst2})
                
        # Merge overlapping pairs into clusters
        clusters = []
        for pair in high_corr_pairs:
            merged = False
            for cluster in clusters:
                if pair & cluster:  # Intersection exists
                    cluster.update(pair)
                    merged = True
                    break
            if not merged:
                clusters.append(pair)
                
        self.correlation_clusters = [list(c) for c in clusters]

    def _calculate_risk_score(self):
        """Calculate overall correlation risk score"""
        # Base score on correlation levels
        base_score = 0.0
        
        # Critical correlations have highest weight
        base_score += len(self.alerts["critical"]) * 0.4
        base_score += len(self.alerts["warning"]) * 0.2
        base_score += len(self.alerts["info"]) * 0.1
        
        # Factor in average correlation
        base_score += self.avg_correlation * 0.3
        
        # Factor in clustering
        cluster_penalty = sum(len(c) - 1 for c in self.correlation_clusters) * 0.1
        base_score += cluster_penalty
        
        # Normalize
        self.risk_score = min(base_score, 1.0)

    def _record_audit(
        self,
        inst1: str,
        inst2: str,
        corr: float,
        severity: str,
        positions: Optional[Dict[str, Any]]
    ):
        """Record audit entry with enhanced information"""
        entry = {
            "timestamp": utcnow(),
            "pair": [inst1, inst2],
            "correlation": corr,
            "severity": severity,
            "thresholds": {
                "info": self.info_corr,
                "warning": self.warning_corr,
                "critical": self.max_corr
            },
            "metrics": {
                "avg_correlation": self.avg_correlation,
                "max_correlation": self.max_correlation,
                "clusters": self.correlation_clusters
            }
        }
        
        # Add position info if available
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

    def get_correlation_impact(self) -> Dict[str, float]:
        """Get recommended position adjustments based on correlations"""
        impact = {}
        
        # Reduce exposure for instruments in correlation clusters
        for cluster in self.correlation_clusters:
            cluster_size = len(cluster)
            if cluster_size > 1:
                # Recommend reduced position size
                reduction = 1.0 / cluster_size
                for inst in cluster:
                    impact[inst] = reduction
                    
        return impact

    @property
    def high_corr(self) -> bool:
        """Legacy compatibility - returns True if any critical correlations"""
        return len(self.alerts["critical"]) > 0

    def get_observation_components(self) -> np.ndarray:
        """Return correlation metrics as observation"""
        return np.array([
            self.risk_score,
            self.avg_correlation,
            self.max_correlation,
            len(self.correlation_clusters) / 10.0  # Normalized cluster count
        ], dtype=np.float32)

    # Evolution and state methods...
    def mutate(self, std: float = 0.05):
        """Mutate correlation thresholds"""
        self.info_corr = float(np.clip(
            self.info_corr + np.random.normal(0, std),
            0.3, 0.7
        ))
        self.warning_corr = float(np.clip(
            self.warning_corr + np.random.normal(0, std),
            self.info_corr + 0.1, 0.85
        ))
        self.max_corr = float(np.clip(
            self.max_corr + np.random.normal(0, std),
            self.warning_corr + 0.1, 0.99
        ))
        self.logger.info(f"Mutated thresholds: {self.info_corr:.2f}/{self.warning_corr:.2f}/{self.max_corr:.2f}")

    def get_state(self) -> Dict[str, Any]:
        return {
            "thresholds": {
                "max_corr": self.max_corr,
                "warning_corr": self.warning_corr,
                "info_corr": self.info_corr
            },
            "enabled": self.enabled,
            "risk_score": self.risk_score,
            "metrics": {
                "avg_correlation": self.avg_correlation,
                "max_correlation": self.max_correlation
            },
            "correlation_clusters": self.correlation_clusters,
            "audit": copy.deepcopy(self._audit[-20:])
        }


# ─────────────────────────────────────────────────────────────────────────────#
# DrawdownRescue - FIXED
# ─────────────────────────────────────────────────────────────────────────────#
class DrawdownRescue(Module):
    """
    FIXED: Progressive drawdown management with recovery tracking.
    
    Key improvements:
    - Multiple drawdown levels with different responses
    - Recovery tracking and metrics
    - Integration with position sizing
    - Drawdown velocity monitoring
    """
    AUDIT_PATH = "logs/risk/drawdown_rescue_audit.jsonl"
    LOG_PATH   = "logs/risk/drawdown_rescue.log"

    def __init__(
        self,
        dd_limit: float = 0.25,  # Reduced from 0.3
        warning_dd: float = 0.15,
        info_dd: float = 0.08,
        recovery_threshold: float = 0.5,  # Recovery to this fraction of peak
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
        
        # State tracking
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.dd_history = deque(maxlen=velocity_window)
        self.severity = "none"
        self.in_recovery = False
        self.recovery_start_balance = 0.0
        
        # Metrics
        self.dd_velocity = 0.0  # Rate of drawdown change
        self.time_in_dd = 0
        self.recovery_ratio = 0.0
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        # Logger
        self.logger = logging.getLogger("DrawdownRescue")
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        """Reset rescue state"""
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.dd_history.clear()
        self.severity = "none"
        self.in_recovery = False
        self.recovery_start_balance = 0.0
        self.dd_velocity = 0.0
        self.time_in_dd = 0
        self.recovery_ratio = 0.0
        self._audit.clear()

    def step(
        self,
        current_drawdown: Optional[float] = None,
        balance: Optional[float] = None,
        peak_balance: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Monitor drawdown with progressive responses.
        
        Returns True if critical drawdown level reached.
        """
        if not self.enabled or current_drawdown is None:
            return False
            
        # Update drawdown tracking
        self.dd_history.append(current_drawdown)
        self.current_dd = current_drawdown
        self.max_dd = max(self.max_dd, current_drawdown)
        
        # Calculate drawdown velocity
        if len(self.dd_history) >= 2:
            recent_changes = [
                self.dd_history[i] - self.dd_history[i-1]
                for i in range(1, len(self.dd_history))
            ]
            self.dd_velocity = np.mean(recent_changes)
        else:
            self.dd_velocity = 0.0
            
        # Update time in drawdown
        if current_drawdown > self.info_dd:
            self.time_in_dd += 1
        else:
            self.time_in_dd = 0
            
        # Determine severity
        old_severity = self.severity
        if current_drawdown >= self.dd_limit:
            self.severity = "critical"
        elif current_drawdown >= self.warning_dd:
            self.severity = "warning"
        elif current_drawdown >= self.info_dd:
            self.severity = "info"
        else:
            self.severity = "none"
            
        # Check recovery status
        if balance and peak_balance:
            self._check_recovery(balance, peak_balance)
            
        # Log severity changes
        if self.severity != old_severity and self.severity != "none":
            msg = (f"Drawdown {current_drawdown:.3f} - {self.severity.upper()} "
                   f"(velocity: {self.dd_velocity:+.4f})")
            
            if self.severity == "critical":
                self.logger.error(msg)
            elif self.severity == "warning":
                self.logger.warning(msg)
            else:
                self.logger.info(msg)
                
            # Record audit
            self._record_audit(current_drawdown, self.severity, balance, peak_balance)
            
        return self.severity == "critical"

    def _check_recovery(self, balance: float, peak_balance: float):
        """Track recovery from drawdown"""
        if self.in_recovery:
            # Calculate recovery progress
            recovery_target = peak_balance * self.recovery_threshold
            if balance >= recovery_target:
                # Recovery complete
                self.in_recovery = False
                recovery_gain = balance - self.recovery_start_balance
                self.logger.info(
                    f"Recovery complete: gained {recovery_gain:.2f} "
                    f"from {self.recovery_start_balance:.2f} to {balance:.2f}"
                )
                self.time_in_dd = 0
            else:
                # Update recovery ratio
                self.recovery_ratio = (balance - self.recovery_start_balance) / (
                    recovery_target - self.recovery_start_balance + 1e-8
                )
        elif self.severity in ["warning", "critical"] and not self.in_recovery:
            # Start recovery tracking
            self.in_recovery = True
            self.recovery_start_balance = balance
            self.recovery_ratio = 0.0

    def _record_audit(
        self,
        drawdown: float,
        severity: str,
        balance: Optional[float],
        peak_balance: Optional[float]
    ):
        """Record audit entry with enhanced information"""
        entry = {
            "timestamp": utcnow(),
            "drawdown": drawdown,
            "severity": severity,
            "thresholds": {
                "info": self.info_dd,
                "warning": self.warning_dd,
                "critical": self.dd_limit
            },
            "metrics": {
                "max_dd": self.max_dd,
                "dd_velocity": self.dd_velocity,
                "time_in_dd": self.time_in_dd,
                "in_recovery": self.in_recovery,
                "recovery_ratio": self.recovery_ratio
            }
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

    def get_risk_adjustment(self) -> float:
        """Get recommended risk adjustment based on drawdown"""
        if self.severity == "critical":
            base_adj = 0.2
        elif self.severity == "warning":
            base_adj = 0.5
        elif self.severity == "info":
            base_adj = 0.8
        else:
            base_adj = 1.0
            
        # Adjust for velocity (faster drawdown = more conservative)
        if self.dd_velocity > 0.01:  # Worsening quickly
            base_adj *= 0.8
        elif self.dd_velocity < -0.01:  # Recovering
            base_adj *= 1.1
            
        # Adjust for recovery progress
        if self.in_recovery:
            base_adj *= (1.0 + 0.2 * self.recovery_ratio)
            
        return float(np.clip(base_adj, 0.1, 1.0))

    @property
    def triggered(self) -> bool:
        """Legacy compatibility"""
        return self.severity == "critical"

    def get_observation_components(self) -> np.ndarray:
        """Return drawdown metrics as observation"""
        severity_map = {"none": 0.0, "info": 0.33, "warning": 0.67, "critical": 1.0}
        return np.array([
            severity_map[self.severity],
            self.current_dd,
            self.dd_velocity + 0.5,  # Centered around 0.5
            self.recovery_ratio,
            min(self.time_in_dd / 100.0, 1.0)  # Normalized time
        ], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "thresholds": {
                "dd_limit": self.dd_limit,
                "warning_dd": self.warning_dd,
                "info_dd": self.info_dd
            },
            "enabled": self.enabled,
            "current_dd": self.current_dd,
            "max_dd": self.max_dd,
            "severity": self.severity,
            "metrics": {
                "dd_velocity": self.dd_velocity,
                "time_in_dd": self.time_in_dd,
                "in_recovery": self.in_recovery,
                "recovery_ratio": self.recovery_ratio
            },
            "audit": copy.deepcopy(self._audit[-20:])
        }


# ─────────────────────────────────────────────────────────────────────────────#
# ExecutionQualityMonitor - FIXED
# ─────────────────────────────────────────────────────────────────────────────#
class ExecutionQualityMonitor(Module):
    """
    FIXED: Comprehensive execution quality monitoring.
    
    Key improvements:
    - Multiple quality metrics (slippage, latency, fill rate)
    - Statistical tracking
    - Integration with order management
    - Cost analysis
    """
    AUDIT_PATH = "logs/risk/execution_quality_monitor_audit.jsonl"
    LOG_PATH   = "logs/risk/execution_quality_monitor.log"

    def __init__(
        self,
        slip_limit: float = 0.002,  # 2 pips, more realistic
        latency_limit: int = 1000,  # 1 second in milliseconds
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
        
        # Statistics tracking
        self.stats_window = stats_window
        self.slippage_history = deque(maxlen=stats_window)
        self.latency_history = deque(maxlen=stats_window)
        self.fill_history = deque(maxlen=stats_window)
        
        # Current metrics
        self.quality_score = 1.0
        self.issues: Dict[str, List[str]] = {
            "slippage": [],
            "latency": [],
            "fill_rate": []
        }
        self.cost_impact = 0.0
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        # Logger
        self.logger = logging.getLogger("ExecutionQualityMonitor")
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        """Reset monitor state"""
        self.slippage_history.clear()
        self.latency_history.clear()
        self.fill_history.clear()
        self.quality_score = 1.0
        for key in self.issues:
            self.issues[key].clear()
        self.cost_impact = 0.0
        self._audit.clear()

    def step(
        self,
        trade_executions: Optional[List[Dict[str, Any]]] = None,
        order_attempts: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Monitor execution quality across multiple dimensions.
        """
        if not self.enabled:
            self.quality_score = 1.0
            return
            
        # Clear previous issues
        for key in self.issues:
            self.issues[key].clear()
            
        # Reset cost impact
        daily_cost = 0.0
        
        # Process executions
        if trade_executions:
            for execution in trade_executions:
                self._analyze_execution(execution)
                
                # Calculate cost impact
                if "slippage_cost" in execution:
                    daily_cost += abs(execution["slippage_cost"])
                    
        # Process order attempts for fill rate
        if order_attempts:
            successful = sum(1 for o in order_attempts if o.get("filled", False))
            total = len(order_attempts)
            if total > 0:
                fill_rate = successful / total
                self.fill_history.append(fill_rate)
                
                if fill_rate < self.min_fill_rate:
                    msg = f"Fill rate {fill_rate:.2%} below minimum {self.min_fill_rate:.2%}"
                    self.issues["fill_rate"].append(msg)
                    self.logger.warning(msg)
                    
        # Update cost impact
        self.cost_impact = daily_cost
        
        # Calculate overall quality score
        self._calculate_quality_score()
        
        # Record significant issues
        if any(self.issues.values()):
            self._record_audit()

    def _analyze_execution(self, execution: Dict[str, Any]):
        """Analyze individual execution quality"""
        instrument = execution.get("instrument", "Unknown")
        
        # Check slippage
        if "slippage" in execution:
            slippage = abs(execution["slippage"])
            self.slippage_history.append(slippage)
            
            if slippage > self.slip_limit:
                msg = f"{instrument} slippage {slippage:.5f} > limit {self.slip_limit:.5f}"
                self.issues["slippage"].append(msg)
                self.logger.warning(msg)
                
        # Check latency
        if "latency_ms" in execution:
            latency = execution["latency_ms"]
            self.latency_history.append(latency)
            
            if latency > self.latency_limit:
                msg = f"{instrument} latency {latency}ms > limit {self.latency_limit}ms"
                self.issues["latency"].append(msg)
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
            fill_score = avg_fill
            scores.append(fill_score)
            
        # Calculate weighted average
        if scores:
            self.quality_score = float(np.mean(scores))
        else:
            self.quality_score = 1.0

    def _record_audit(self):
        """Record audit entry for quality issues"""
        stats = self.get_execution_stats()
        
        entry = {
            "timestamp": utcnow(),
            "quality_score": self.quality_score,
            "issues": {k: len(v) for k, v in self.issues.items()},
            "statistics": stats,
            "cost_impact": self.cost_impact,
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
                "percentile_95": float(np.percentile(self.slippage_history, 95))
            }
            
        if self.latency_history:
            stats["latency"] = {
                "mean": float(np.mean(self.latency_history)),
                "std": float(np.std(self.latency_history)),
                "max": float(np.max(self.latency_history)),
                "percentile_95": float(np.percentile(self.latency_history, 95))
            }
            
        if self.fill_history:
            stats["fill_rate"] = {
                "mean": float(np.mean(self.fill_history)),
                "min": float(np.min(self.fill_history)),
                "current": float(self.fill_history[-1]) if self.fill_history else 1.0
            }
            
        return stats

    def get_observation_components(self) -> np.ndarray:
        """Return execution quality metrics as observation"""
        has_issues = float(any(self.issues.values()))
        
        # Get recent averages
        recent_slip = np.mean(list(self.slippage_history)[-10:]) if self.slippage_history else 0.0
        recent_latency = np.mean(list(self.latency_history)[-10:]) if self.latency_history else 0.0
        recent_fill = np.mean(list(self.fill_history)[-10:]) if self.fill_history else 1.0
        
        return np.array([
            self.quality_score,
            has_issues,
            recent_slip / max(self.slip_limit, 1e-8),  # Normalized
            recent_latency / max(self.latency_limit, 1),  # Normalized
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
            "quality_score": self.quality_score,
            "cost_impact": self.cost_impact,
            "statistics": self.get_execution_stats(),
            "audit": copy.deepcopy(self._audit[-20:])
        }


# ─────────────────────────────────────────────────────────────────────────────#
# AnomalyDetector - FIXED
# ─────────────────────────────────────────────────────────────────────────────#
class AnomalyDetector(Module):
    """
    FIXED: Comprehensive anomaly detection with statistical methods.
    
    Key improvements:
    - Multiple anomaly types (PnL, volume, price, pattern)
    - Statistical anomaly detection
    - Adaptive thresholds
    - Pattern recognition
    """
    AUDIT_PATH = "logs/risk/anomaly_detector_audit.jsonl"
    LOG_PATH   = "logs/risk/anomaly_detector.log"

    def __init__(
        self,
        pnl_limit: float = 5000,  # More reasonable
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
        self.adaptive_thresholds = {
            "pnl": self.pnl_limit,
            "volume": 0.0,
            "price": 0.0
        }
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        # Logger
        self.logger = logging.getLogger("AnomalyDetector")
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        """Reset detector state"""
        self.pnl_history.clear()
        self.volume_history.clear()
        self.price_history.clear()
        self.observation_history.clear()
        for key in self.anomalies:
            self.anomalies[key].clear()
        self.anomaly_score = 0.0
        self._audit.clear()

    def step(
        self,
        pnl: Optional[float] = None,
        obs: Optional[np.ndarray] = None,
        volume: Optional[float] = None,
        price: Optional[float] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> bool:
        """
        Detect various types of anomalies.
        
        Returns True if any critical anomaly detected.
        """
        if not self.enabled:
            self.anomaly_score = 0.0
            return False
            
        # Clear previous anomalies
        for key in self.anomalies:
            self.anomalies[key].clear()
            
        critical_found = False
        
        # Check PnL anomaly
        if pnl is not None:
            self.pnl_history.append(pnl)
            if self._check_pnl_anomaly(pnl):
                critical_found = True
                
        # Check observation anomaly
        if obs is not None:
            if self._check_observation_anomaly(obs):
                critical_found = True
                
        # Check volume anomaly
        if volume is not None:
            self.volume_history.append(volume)
            self._check_volume_anomaly(volume)
            
        # Check price anomaly
        if price is not None:
            self.price_history.append(price)
            self._check_price_anomaly(price)
            
        # Check pattern anomalies
        if trades:
            self._check_pattern_anomalies(trades)
            
        # Update adaptive thresholds
        self._update_adaptive_thresholds()
        
        # Calculate anomaly score
        self._calculate_anomaly_score()
        
        # Record audit if anomalies found
        if any(self.anomalies.values()):
            self._record_audit()
            
        return critical_found

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
            self.logger.error(f"Critical PnL anomaly: {pnl:.2f}")
            return True
            
        # Statistical check if enough history
        if len(self.pnl_history) >= 20:
            mean = np.mean(self.pnl_history)
            std = np.std(self.pnl_history)
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
                
        return False

    def _check_volume_anomaly(self, volume: float):
        """Check for volume anomalies"""
        if len(self.volume_history) >= 10:
            mean = np.mean(self.volume_history)
            std = np.std(self.volume_history)
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
                            "severity": "warning"
                        })
                        
            # Statistical check
            mean = np.mean(self.price_history)
            std = np.std(self.price_history)
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
            
        # Check for suspicious patterns
        # 1. All trades in same direction with unusual size
        directions = [np.sign(t.get("size", 0)) for t in trades]
        if len(set(directions)) == 1 and len(trades) > 3:
            self.anomalies["pattern"].append({
                "type": "unidirectional",
                "count": len(trades),
                "severity": "info"
            })
            
        # 2. Rapid fire trades
        if len(trades) > 10:
            self.anomalies["pattern"].append({
                "type": "high_frequency",
                "count": len(trades),
                "severity": "warning"
            })

    def _update_adaptive_thresholds(self):
        """Update thresholds based on recent history"""
        # Adaptive PnL threshold
        if len(self.pnl_history) >= 50:
            recent_pnls = list(self.pnl_history)[-50:]
            mean = abs(np.mean(recent_pnls))
            std = np.std(recent_pnls)
            self.adaptive_thresholds["pnl"] = min(
                self.pnl_limit,
                mean + 4 * std
            )

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
            "anomaly_score": self.anomaly_score,
            "anomalies": {
                k: len(v) for k, v in self.anomalies.items() if v
            },
            "details": {
                k: v for k, v in self.anomalies.items() if v
            },
            "thresholds": self.adaptive_thresholds.copy()
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
        
        return np.array([
            self.anomaly_score,
            has_critical,
            min(total_anomalies / 10.0, 1.0),  # Normalized count
            len(self.pnl_history) / 100.0,  # Data sufficiency
            self.adaptive_thresholds["pnl"] / self.pnl_limit  # Threshold adaptation
        ], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "limits": {
                "pnl_limit": self.pnl_limit,
                "volume_zscore": self.volume_zscore,
                "price_zscore": self.price_zscore
            },
            "enabled": self.enabled,
            "anomaly_score": self.anomaly_score,
            "adaptive_thresholds": self.adaptive_thresholds.copy(),
            "anomaly_summary": {k: len(v) for k, v in self.anomalies.items()},
            "audit": copy.deepcopy(self._audit[-20:])
        }