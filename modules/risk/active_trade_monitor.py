#─────────────────────────────────────────────────────────────
# modules/risk/active_trade_monitor.py
#─────────────────────────────────────────────────────────────

import numpy as np
import logging
import json
import os
import datetime
from typing import Dict, Any, List, Optional, Union
from modules.core.core import Module
from utils.get_dir import utcnow


class ActiveTradeMonitor(Module):

    AUDIT_PATH = "logs/risk/active_trade_monitor_audit.jsonl"
    LOG_PATH   = "logs/risk/active_trade_monitor.log"

    def __init__(
        self,
        max_duration: int = 200,  # KEPT original sophisticated defaults
        warning_duration: int = 50,
        enabled: bool = True,
        audit_log_size: int = 100,
        severity_weights: Optional[Dict[str, float]] = None,
        debug: bool = True
    ):
        super().__init__()
        self.max_duration = max_duration
        self.warning_duration = warning_duration
        self.enabled = enabled
        self.debug = debug
        
        # Ensure log directories exist - FIXED without removing features
        log_dir = os.path.dirname(self.LOG_PATH)
        audit_dir = os.path.dirname(self.AUDIT_PATH)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(audit_dir, exist_ok=True)
        
        # Full severity tracking - KEPT all original functionality
        self.severity_weights = severity_weights or {
            "info": 0.0,
            "warning": 0.5,
            "critical": 1.0
        }
        
        # Full state tracking - KEPT all original features
        self.position_durations: Dict[str, int] = {}
        self.alerts: Dict[str, str] = {}
        self.risk_score = 0.0
        self.step_count = 0
        self.last_log_step = 0  # ADDED: For intelligent logging frequency
        
        # Full audit system - KEPT all original functionality
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size

        # FIXED Logger setup - maintains full functionality but optimizes instances
        self.logger = logging.getLogger("ActiveTradeMonitor")
        if not self.logger.handlers:
            self.logger.handlers.clear()
            self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
            self.logger.propagate = False
            
            # File handler with full debug capability
            fh = logging.FileHandler(self.LOG_PATH, mode='a', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
            # Console handler with intelligent filtering
            if debug:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
        
        # Initial log entry with full system info
        self.logger.info(f"ActiveTradeMonitor initialized - max_duration={max_duration}, warning={warning_duration}, debug={debug}")

    def reset(self):
        """Reset monitor state - FULL IMPLEMENTATION MAINTAINED"""
        self.position_durations.clear()
        self.alerts.clear()
        self.risk_score = 0.0
        self.step_count = 0
        self.last_log_step = 0
        self._audit.clear()
        self.logger.info("ActiveTradeMonitor reset - all state cleared")

    def step(
        self,
        open_positions: Optional[Union[List[Dict], Dict[str, Dict]]] = None,
        current_step: Optional[int] = None,
        **kwargs
    ) -> float:
        """
        Monitor trade durations with graduated alerts.
        FULL IMPLEMENTATION - Returns risk score between 0 and 1.
        
        FIXES APPLIED:
        - Better error handling without losing functionality
        - Intelligent logging frequency while maintaining full audit
        - Enhanced position processing with all original features
        """
        self.step_count += 1
        
        # Intelligent debug logging - not every step to reduce spam
        should_log_debug = (
            self.debug and 
            (self.step_count - self.last_log_step > 25 or 
             len(self.alerts) > 0 or 
             self.step_count % 100 == 0)
        )
        
        if should_log_debug:
            self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}, positions={len(open_positions) if open_positions else 0}")
            self.last_log_step = self.step_count
        
        if not self.enabled:
            self.risk_score = 0.0
            return self.risk_score
            
        # Clear previous alerts
        self.alerts.clear()
        
        if not open_positions:
            self.risk_score = 0.0
            if should_log_debug:
                self.logger.debug("No open positions")
            return self.risk_score
            
        try:
            # FIXED: Enhanced position format handling
            if isinstance(open_positions, dict):
                positions = open_positions
            elif isinstance(open_positions, list):
                # Convert list to dict with better error handling
                positions = {}
                for i, p in enumerate(open_positions):
                    if isinstance(p, dict):
                        inst_name = p.get("instrument", p.get("symbol", f"pos_{i}"))
                        positions[inst_name] = p
                    else:
                        positions[f"pos_{i}"] = {"size": p} if isinstance(p, (int, float)) else {}
            else:
                self.logger.warning(f"Unexpected position format: {type(open_positions)}")
                return self.risk_score
                
            if should_log_debug:
                self.logger.debug(f"Processing {len(positions)} positions: {list(positions.keys())}")
            
            # Track positions with full error recovery
            total_severity = 0.0
            num_positions = 0
            
            for instrument, position in positions.items():
                try:
                    # ENHANCED: Get duration with multiple fallback methods
                    duration = self._get_position_duration(position, instrument, current_step)
                    self.position_durations[instrument] = duration
                    
                    # Determine severity with full logic
                    severity = self._get_severity(duration)
                    
                    # Log positions when debugging or alerts
                    if should_log_debug or severity != "info":
                        self.logger.debug(f"{instrument}: duration={duration}, severity={severity}")
                    
                    if severity != "info":
                        self.alerts[instrument] = severity
                        
                        # Log alert with appropriate level
                        msg = f"{instrument} duration {duration} - {severity.upper()}"
                        if severity == "warning":
                            self.logger.warning(msg)
                        elif severity == "critical":
                            self.logger.error(msg)
                        else:
                            self.logger.info(msg)
                            
                        # Record audit for significant events
                        self._record_audit(instrument, duration, severity, position)
                        
                    # Update risk score
                    total_severity += self.severity_weights[severity]
                    num_positions += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing position {instrument}: {e}")
                    # Continue processing other positions
                    continue
                
            # Calculate overall risk score
            self.risk_score = total_severity / max(num_positions, 1) if num_positions > 0 else 0.0
            
            # Clean up closed positions
            current_instruments = set(positions.keys())
            closed = set(self.position_durations.keys()) - current_instruments
            for inst in closed:
                self.position_durations.pop(inst, None)
                if should_log_debug:
                    self.logger.info(f"Position {inst} closed, removed from tracking")
                
            # Log summary with intelligent frequency
            if len(self.alerts) > 0 or self.step_count % 100 == 0:
                self.logger.info(f"Step {self.step_count} summary: risk_score={self.risk_score:.3f}, alerts={len(self.alerts)}, positions={num_positions}")
            
        except Exception as e:
            self.logger.error(f"Critical error in ActiveTradeMonitor step: {e}")
            self.risk_score = 0.0
            
        return self.risk_score

    def _get_position_duration(self, position: Dict[str, Any], instrument: str, current_step: Optional[int]) -> int:
        """ENHANCED: Get position duration with multiple robust fallback methods"""
        try:
            # Method 1: Direct duration field
            for duration_field in ["duration", "time_open", "bars_held", "steps_open"]:
                if duration_field in position and position[duration_field] is not None:
                    return max(0, int(position[duration_field]))
                    
            # Method 2: Calculate from entry step
            for entry_field in ["entry_step", "open_step", "start_step"]:
                if entry_field in position and position[entry_field] is not None and current_step is not None:
                    entry_step = int(position[entry_field])
                    return max(0, current_step - entry_step)
                    
            # Method 3: Calculate from timestamps
            if "entry_time" in position and "current_time" in position:
                try:
                    entry_time = position["entry_time"]
                    current_time = position["current_time"]
                    if isinstance(entry_time, str):
                        entry_time = datetime.datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    if isinstance(current_time, str):
                        current_time = datetime.datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                    duration_seconds = (current_time - entry_time).total_seconds()
                    return max(0, int(duration_seconds / 60))  # Convert to minutes
                except:
                    pass
                    
            # Method 4: Increment tracked duration (fallback)
            current_duration = self.position_durations.get(instrument, 0)
            return current_duration + 1
            
        except Exception as e:
            self.logger.debug(f"Error calculating duration for {instrument}: {e}")
            # Ultimate fallback
            return self.position_durations.get(instrument, 1)

    def _get_severity(self, duration: int) -> str:
        """Determine alert severity based on duration - FULL IMPLEMENTATION"""
        if duration >= self.max_duration:
            return "critical"
        elif duration >= self.warning_duration:
            return "warning"
        else:
            return "info"

    def _record_audit(self, instrument: str, duration: int, severity: str, position: Dict[str, Any]):
        """Record audit entry with enhanced information - FULL IMPLEMENTATION"""
        try:
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
                    "size": position.get("size", position.get("volume", 0)),
                    "pnl": position.get("pnl", position.get("profit", 0)),
                    "side": position.get("side", position.get("direction", 0)),
                    "entry_price": position.get("entry_price", position.get("open_price", 0))
                }
            }
            
            self._audit.append(entry)
            if len(self._audit) > self._max_audit:
                self._audit.pop(0)
                
            # Write to file with reduced frequency
            if severity in ["warning", "critical"] or len(self._audit) % 10 == 0:
                try:
                    with open(self.AUDIT_PATH, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                except Exception as e:
                    if self.debug:
                        self.logger.error(f"Failed to write audit: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error recording audit: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Return monitor state as observation - FULL IMPLEMENTATION"""
        try:
            avg_duration_ratio = 0.0
            if self.position_durations:
                avg_duration = np.mean(list(self.position_durations.values()))
                avg_duration_ratio = avg_duration / max(self.max_duration, 1)
                
            alert_ratio = len(self.alerts) / max(len(self.position_durations), 1)
                
            return np.array([
                float(self.risk_score),
                float(np.clip(avg_duration_ratio, 0.0, 2.0)),  # Clipped for stability
                float(np.clip(alert_ratio, 0.0, 1.0))
            ], dtype=np.float32)
            
        except Exception:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        """Get full state - COMPLETE IMPLEMENTATION"""
        return {
            "max_duration": self.max_duration,
            "warning_duration": self.warning_duration,
            "enabled": self.enabled,
            "step_count": self.step_count,
            "position_durations": self.position_durations.copy(),
            "alerts": self.alerts.copy(),
            "risk_score": float(self.risk_score),
            "severity_weights": self.severity_weights.copy(),
            "audit_summary": {
                "total_entries": len(self._audit),
                "recent_entries": len([a for a in self._audit if a.get("severity") in ["warning", "critical"]])
            }
        }
