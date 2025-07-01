# ──────────────────────────────────────────────────────────────
# modules/risk/drawdown_rescue.py
# ──────────────────────────────────────────────────────────────

import numpy as np
import logging
import json
import os
from collections import deque
from typing import Dict, Any, List, Optional
from modules.core.core import Module
from utils.get_dir import utcnow


class DrawdownRescue(Module):
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
        velocity_window: int = 10,
        training_mode: bool = True,  # NEW: Training vs live mode
        debug: bool = True
    ):
        super().__init__()
        self.dd_limit = dd_limit
        self.warning_dd = warning_dd
        self.info_dd = info_dd
        self.recovery_threshold = recovery_threshold
        self.enabled = enabled
        self.training_mode = training_mode
        self.debug = debug
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)
        
        # Enhanced state tracking
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.dd_history = deque(maxlen=velocity_window)
        self.balance_history = deque(maxlen=50)  # Track balance for better DD calculation
        self.peak_balance = 0.0
        self.severity = "none"
        self.step_count = 0
        self.last_significant_log = 0
        
        # Recovery tracking
        self.dd_velocity = 0.0  # Rate of drawdown change
        self.recovery_progress = 0.0
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size

        # FIXED Logger setup
        self.logger = logging.getLogger("DrawdownRescue")
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
        
        self.logger.info(f"DrawdownRescue initialized - thresholds: {info_dd}/{warning_dd}/{dd_limit}, training={training_mode}")

    def reset(self):
        """Reset rescue state - FULL IMPLEMENTATION"""
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.peak_balance = 0.0
        self.dd_history.clear()
        self.balance_history.clear()
        self.severity = "none"
        self.step_count = 0
        self.last_significant_log = 0
        self.dd_velocity = 0.0
        self.recovery_progress = 0.0
        self._audit.clear()
        self.logger.info("DrawdownRescue reset - all state cleared")

    def step(
        self,
        current_drawdown: Optional[float] = None,
        balance: Optional[float] = None,
        peak_balance: Optional[float] = None,
        equity: Optional[float] = None,
        portfolio_value: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        ENHANCED: Monitor drawdown with progressive responses and multiple data sources.
        Returns True if critical drawdown level reached.
        """
        self.step_count += 1
        
        try:
            # ENHANCED: Extract drawdown from multiple sources
            calculated_dd = self._calculate_drawdown_enhanced(
                current_drawdown, balance, peak_balance, equity, portfolio_value, kwargs
            )
                
            # Intelligent logging
            should_log = (
                self.debug and 
                (self.step_count - self.last_significant_log > 50 or 
                 calculated_dd != self.current_dd or
                 self.step_count % 200 == 0)
            )
            
            if should_log:
                self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}, drawdown={calculated_dd}")
                self.last_significant_log = self.step_count
            
            if not self.enabled or calculated_dd is None:
                if should_log:
                    self.logger.debug("Disabled or no drawdown data")
                return False
                
            # Update tracking with velocity calculation
            if len(self.dd_history) > 0:
                self.dd_velocity = calculated_dd - self.dd_history[-1]
            else:
                self.dd_velocity = 0.0
                
            self.dd_history.append(calculated_dd)
            self.current_dd = calculated_dd
            self.max_dd = max(self.max_dd, calculated_dd)
            
            # Calculate recovery progress
            if self.max_dd > 0:
                self.recovery_progress = max(0, (self.max_dd - calculated_dd) / self.max_dd)
            
            # Determine severity with enhanced logic
            old_severity = self.severity
            self.severity = self._get_drawdown_severity_enhanced(calculated_dd)
            
            # Log severity changes or significant drawdowns
            should_log_severity = (
                self.severity != old_severity or 
                calculated_dd > self.info_dd or
                self.severity in ["warning", "critical"]
            )
            
            if should_log_severity:
                msg = f"Drawdown {calculated_dd:.4f} - {self.severity.upper()}"
                if self.dd_velocity != 0:
                    msg += f" (velocity: {self.dd_velocity:+.4f})"
                if self.recovery_progress > 0:
                    msg += f" (recovery: {self.recovery_progress:.1%})"
                
                if self.severity == "critical":
                    self.logger.error(msg)
                elif self.severity == "warning":
                    self.logger.warning(msg)
                elif self.severity == "info":
                    self.logger.info(msg)
                elif self.debug:
                    self.logger.debug(msg)
                    
                # Record audit for significant events
                if self.severity in ["warning", "critical"] or self.severity != old_severity:
                    self._record_audit(calculated_dd, self.severity, balance, peak_balance)
            
            # Calculate risk adjustment
            risk_adjustment = self.get_risk_adjustment()
            
            # Log summary with intelligent frequency
            if should_log_severity or self.step_count % 100 == 0:
                self.logger.info(f"Step {self.step_count} summary: dd={calculated_dd:.4f}, severity={self.severity}, adjustment={risk_adjustment:.3f}")
            
            return self.severity == "critical"
            
        except Exception as e:
            self.logger.error(f"Error in drawdown monitoring: {e}")
            return False

    def _calculate_drawdown_enhanced(
        self,
        current_drawdown: Optional[float],
        balance: Optional[float],
        peak_balance: Optional[float],
        equity: Optional[float],
        portfolio_value: Optional[float],
        kwargs: Dict[str, Any]
    ) -> Optional[float]:
        """ENHANCED: Calculate drawdown from multiple sources with better accuracy"""
        try:
            # Method 1: Direct drawdown provided
            if current_drawdown is not None and not np.isnan(current_drawdown):
                return max(0.0, float(current_drawdown))
                
            # Method 2: Balance and peak balance
            if balance is not None and peak_balance is not None and peak_balance > 0:
                dd = (peak_balance - balance) / peak_balance
                return max(0.0, float(dd))
                
            # Method 3: Track peak balance automatically
            current_value = balance or equity or portfolio_value
            if current_value is not None and current_value > 0:
                current_value = float(current_value)
                self.balance_history.append(current_value)
                
                # Update peak balance
                if current_value > self.peak_balance:
                    self.peak_balance = current_value
                    
                if self.peak_balance > 0:
                    dd = (self.peak_balance - current_value) / self.peak_balance
                    return max(0.0, dd)
                    
            # Method 4: Running peak from balance history
            if len(self.balance_history) > 5:
                balances = list(self.balance_history)
                current_bal = balances[-1]
                peak_bal = max(balances)
                if peak_bal > 0:
                    dd = (peak_bal - current_bal) / peak_bal
                    return max(0.0, dd)
                    
            # Method 5: Extract from kwargs
            for key in ['drawdown', 'dd', 'max_drawdown', 'current_dd']:
                if key in kwargs and kwargs[key] is not None:
                    try:
                        return max(0.0, float(kwargs[key]))
                    except:
                        continue
                        
            return None
            
        except Exception as e:
            if self.debug:
                self.logger.debug(f"Error calculating drawdown: {e}")
            return None

    def _get_drawdown_severity_enhanced(self, drawdown: float) -> str:
        """ENHANCED: Determine drawdown severity with velocity consideration"""
        base_severity = self._get_base_severity(drawdown)
        
        # Adjust severity based on velocity (rapid drawdown is more concerning)
        if self.dd_velocity > 0.02:  # Rapidly increasing drawdown
            if base_severity == "info":
                base_severity = "warning"
            elif base_severity == "warning":
                base_severity = "critical"
                
        return base_severity

    def _get_base_severity(self, drawdown: float) -> str:
        """Base severity determination"""
        if drawdown >= self.dd_limit:
            return "critical"
        elif drawdown >= self.warning_dd:
            return "warning"
        elif drawdown >= self.info_dd:
            return "info"
        else:
            return "none"

    def get_risk_adjustment(self) -> float:
        """ENHANCED: Get recommended risk adjustment based on drawdown and velocity"""
        base_adjustment = self._get_base_adjustment()
        
        # Adjust for velocity
        if self.dd_velocity > 0.01:  # Rapid drawdown
            base_adjustment *= 0.8  # More conservative
        elif self.dd_velocity < -0.005:  # Recovering
            base_adjustment = min(1.0, base_adjustment * 1.2)  # Less conservative
            
        return float(np.clip(base_adjustment, 0.1, 1.0))

    def _get_base_adjustment(self) -> float:
        """Base risk adjustment calculation"""
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
        """Record audit entry - ENHANCED"""
        try:
            entry = {
                "timestamp": utcnow(),
                "step": self.step_count,
                "drawdown": float(drawdown),
                "severity": severity,
                "velocity": float(self.dd_velocity),
                "recovery_progress": float(self.recovery_progress),
                "thresholds": {
                    "info": self.info_dd,
                    "warning": self.warning_dd,
                    "critical": self.dd_limit
                },
                "max_dd": float(self.max_dd),
                "risk_adjustment": self.get_risk_adjustment()
            }
            
            if balance is not None:
                entry["balance"] = float(balance)
            if peak_balance is not None:
                entry["peak_balance"] = float(peak_balance)
            if self.peak_balance > 0:
                entry["tracked_peak"] = float(self.peak_balance)
                
            self._audit.append(entry)
            if len(self._audit) > self._max_audit:
                self._audit.pop(0)
                
            # Write to file with intelligent frequency
            if severity in ["warning", "critical"] or len(self._audit) % 15 == 0:
                try:
                    with open(self.AUDIT_PATH, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                except Exception as e:
                    if self.debug:
                        self.logger.error(f"Failed to write audit: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error recording audit: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Return drawdown metrics as observation - ENHANCED"""
        try:
            severity_map = {"none": 0.0, "info": 0.25, "warning": 0.5, "critical": 1.0}
            
            return np.array([
                severity_map.get(self.severity, 0.0),
                float(np.clip(self.current_dd, 0.0, 1.0)),
                float(np.clip(self.max_dd, 0.0, 1.0)),
                float(self.get_risk_adjustment()),
                float(np.clip(self.dd_velocity + 0.5, 0.0, 1.0)),  # Normalized velocity
                float(self.recovery_progress)
            ], dtype=np.float32)
            
        except Exception:
            return np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.0], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        """Get complete state - FULL IMPLEMENTATION"""
        return {
            "thresholds": {
                "dd_limit": self.dd_limit,
                "warning_dd": self.warning_dd,
                "info_dd": self.info_dd
            },
            "enabled": self.enabled,
            "training_mode": self.training_mode,
            "step_count": self.step_count,
            "current_dd": float(self.current_dd),
            "max_dd": float(self.max_dd),
            "peak_balance": float(self.peak_balance),
            "severity": self.severity,
            "dd_velocity": float(self.dd_velocity),
            "recovery_progress": float(self.recovery_progress),
            "risk_adjustment": self.get_risk_adjustment(),
            "balance_history_size": len(self.balance_history),
            "audit_summary": {
                "total_entries": len(self._audit),
                "recent_critical": len([a for a in self._audit if a.get("severity") == "critical"])
            }
        }

