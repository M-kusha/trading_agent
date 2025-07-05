# ─────────────────────────────────────────────────────────────
# File: modules/risk/drawdown_rescue.py
# Enhanced with InfoBus integration & intelligent rescue mechanisms
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import RiskMixin, TradingMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import AuditTracker, format_operator_message


class DrawdownRescue(Module, RiskMixin, TradingMixin, StateManagementMixin):
    """
    Enhanced drawdown rescue system with InfoBus integration.
    Provides intelligent drawdown monitoring and progressive risk reduction.
    """

    def __init__(
        self,
        dd_limit: float = 0.25,
        warning_dd: float = 0.15,
        info_dd: float = 0.08,
        recovery_threshold: float = 0.5,
        enabled: bool = True,
        velocity_window: int = 10,
        training_mode: bool = True,
        adaptive_rescue: bool = True,
        debug: bool = True,
        **kwargs
    ):
        # Initialize with enhanced config
        config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 100),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(config)
        
        # Initialize mixins
        self._initialize_risk_state()
        self._initialize_trading_state()
        
        # Core configuration
        self.enabled = enabled
        self.training_mode = training_mode
        self.adaptive_rescue = adaptive_rescue
        
        # Drawdown thresholds
        self.base_thresholds = {
            'dd_limit': float(dd_limit),
            'warning_dd': float(warning_dd),
            'info_dd': float(info_dd),
            'recovery_threshold': float(recovery_threshold)
        }
        
        # Current thresholds (may be adjusted dynamically)
        self.current_thresholds = self.base_thresholds.copy()
        
        # Enhanced state tracking
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.peak_balance = 0.0
        self.severity = "none"
        self.step_count = 0
        
        # Velocity and trend analysis
        self.dd_history = deque(maxlen=velocity_window)
        self.balance_history = deque(maxlen=50)
        self.dd_velocity = 0.0
        self.dd_acceleration = 0.0
        self.recovery_progress = 0.0
        
        # Risk adjustment system
        self.risk_adjustment = 1.0
        self.rescue_mode = False
        self.rescue_start_time = None
        self.consecutive_rescue_days = 0
        
        # Performance tracking
        self.drawdown_analytics = defaultdict(list)
        self.regime_drawdowns = defaultdict(list)
        self.recovery_events = []
        
        # Rescue effectiveness tracking
        self._rescue_interventions = 0
        self._successful_recoveries = 0
        self._false_alarms = 0
        self._analysis_history = deque(maxlen=200)
        self._risk_metrics_history = deque(maxlen=100)
        
        # Audit system
        self.audit_manager = AuditTracker("DrawdownRescue")
        self._last_significant_event = None
        
        self.log_operator_info(
            "🛟 Enhanced Drawdown Rescue initialized",
            critical_limit=f"{dd_limit:.1%}",
            warning_threshold=f"{warning_dd:.1%}",
            info_threshold=f"{info_dd:.1%}",
            training_mode=training_mode,
            adaptive_rescue=adaptive_rescue,
            enabled=enabled
        )


    def _update_risk_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update risk metrics tracking"""
        try:
            if not hasattr(self, '_risk_metrics_history'):
                self._risk_metrics_history = deque(maxlen=100)
            
            # Add timestamp
            metrics['timestamp'] = datetime.datetime.now().isoformat()
            metrics['step'] = self.step_count
            
            self._risk_metrics_history.append(metrics)
            
            # Update current metrics for external access
            for key, value in metrics.items():
                if hasattr(self, f'_current_{key}'):
                    setattr(self, f'_current_{key}', value)
                    
        except Exception as e:
            self.log_operator_warning(f"Risk metrics update failed: {e}")


    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_risk_state()
        self._reset_trading_state()
        
        # Reset drawdown tracking
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.peak_balance = 0.0
        self.severity = "none"
        self.step_count = 0
        
        # Reset velocity tracking
        self.dd_history.clear()
        self.balance_history.clear()
        self.dd_velocity = 0.0
        self.dd_acceleration = 0.0
        self.recovery_progress = 0.0
        
        # Reset rescue system
        self.risk_adjustment = 1.0
        self.rescue_mode = False
        self.rescue_start_time = None
        self.consecutive_rescue_days = 0
        
        # Reset analytics
        self.drawdown_analytics.clear()
        self.regime_drawdowns.clear()
        self.recovery_events.clear()
        
        # Reset performance tracking
        self._rescue_interventions = 0
        self._successful_recoveries = 0
        self._false_alarms = 0
        
        # Reset thresholds to base values
        self.current_thresholds = self.base_thresholds.copy()
        
        self.log_operator_info("🔄 Drawdown Rescue reset - all tracking cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - drawdown rescue inactive")
            return
        
        if not self.enabled:
            self.risk_adjustment = 1.0
            self.rescue_mode = False
            return
        
        self.step_count += 1
        
        # Extract context for intelligent analysis
        context = extract_standard_context(info_bus)
        
        # Extract drawdown data from InfoBus
        calculated_dd = self._extract_drawdown_from_info_bus(info_bus)
        
        if calculated_dd is None:
            self._handle_no_drawdown_data(context)
            return
        
        # Process comprehensive drawdown analysis
        critical_found = self._analyze_drawdown_comprehensive(calculated_dd, info_bus, context)
        
        # Update rescue mechanisms
        self._update_rescue_system(context)
        
        # Calculate risk adjustment
        self._calculate_risk_adjustment(context)
        
        # Update InfoBus with results
        self._update_info_bus(info_bus, critical_found)
        
        # Record audit for significant events
        if critical_found or self.severity in ["warning", "critical"]:
            self._record_comprehensive_audit(info_bus, context)
        
        # Update performance metrics
        self._update_rescue_metrics()

    def _extract_drawdown_from_info_bus(self, info_bus: InfoBus) -> Optional[float]:
        """Extract drawdown from InfoBus with multiple sophisticated methods"""
        
        try:
            # Method 1: Direct drawdown from risk snapshot
            risk_data = info_bus.get('risk', {})
            current_drawdown = risk_data.get('current_drawdown')
            if current_drawdown is not None and not np.isnan(current_drawdown):
                return max(0.0, float(current_drawdown))
            
            # Method 2: Calculate from balance and peak balance
            balance = risk_data.get('balance', risk_data.get('equity'))
            peak_balance = risk_data.get('peak_balance')
            
            if balance is not None and peak_balance is not None and peak_balance > 0:
                dd = (peak_balance - balance) / peak_balance
                return max(0.0, float(dd))
            
            # Method 3: Track peak balance automatically from InfoBus
            if balance is not None and balance > 0:
                balance = float(balance)
                self.balance_history.append(balance)
                
                # Update peak balance
                if balance > self.peak_balance:
                    self.peak_balance = balance
                
                if self.peak_balance > 0:
                    dd = (self.peak_balance - balance) / self.peak_balance
                    return max(0.0, dd)
            
            # Method 4: Calculate from positions unrealized PnL
            positions = InfoBusExtractor.get_positions(info_bus)
            if positions and balance is not None:
                total_unrealized = sum(pos.get('unrealised_pnl', 0) for pos in positions)
                
                # If we have significant unrealized losses, include in drawdown
                if total_unrealized < -100:  # Significant unrealized loss
                    effective_balance = balance + total_unrealized
                    if self.peak_balance > 0:
                        dd = (self.peak_balance - effective_balance) / self.peak_balance
                        return max(0.0, dd)
            
            # Method 5: Extract from module data
            module_data = info_bus.get('module_data', {})
            for module_name in ['risk_manager', 'portfolio_manager', 'balance_tracker']:
                if module_name in module_data:
                    module_dd = module_data[module_name].get('drawdown')
                    if module_dd is not None:
                        return max(0.0, float(module_dd))
            
            # Method 6: Running peak from balance history
            if len(self.balance_history) >= 5:
                balances = list(self.balance_history)
                current_bal = balances[-1]
                peak_bal = max(balances)
                if peak_bal > 0:
                    dd = (peak_bal - current_bal) / peak_bal
                    return max(0.0, dd)
            
        except Exception as e:
            self.log_operator_warning(f"Drawdown extraction failed: {e}")
        
        return None

    def _handle_no_drawdown_data(self, context: Dict[str, Any]) -> None:
        """Handle case when no drawdown data is available"""
        
        self.current_dd = 0.0
        self.severity = "none"
        self.risk_adjustment = 1.0
        self.rescue_mode = False
        
        # Log occasionally
        if self.step_count % 100 == 0:
            self.log_operator_info(
                "📊 No drawdown data available",
                step=self.step_count,
                regime=context.get('regime', 'unknown'),
                training_mode=self.training_mode
            )

    def _analyze_drawdown_comprehensive(self, calculated_dd: float, 
                                       info_bus: InfoBus, context: Dict[str, Any]) -> bool:
        """Comprehensive drawdown analysis with advanced metrics"""
        
        # Update basic state
        old_dd = self.current_dd
        old_severity = self.severity
        self.current_dd = calculated_dd
        self.max_dd = max(self.max_dd, calculated_dd)
        
        # Calculate velocity and acceleration
        self._calculate_drawdown_dynamics()
        
        # Determine severity with enhanced logic
        self.severity = self._determine_drawdown_severity_enhanced(calculated_dd, context)
        
        # Calculate recovery progress
        self._calculate_recovery_progress()
        
        # Log significant changes
        critical_found = self._log_drawdown_changes(old_dd, old_severity, context)
        
        # Update analytics
        self._update_drawdown_analytics(context)
        
        # Store drawdown snapshot
        self._store_drawdown_snapshot(info_bus, context)
        
        return critical_found

    def _calculate_drawdown_dynamics(self) -> None:
        """Calculate drawdown velocity and acceleration"""
        
        # Add current drawdown to history
        self.dd_history.append(self.current_dd)
        
        try:
            if len(self.dd_history) >= 2:
                # Calculate velocity (rate of change)
                self.dd_velocity = self.dd_history[-1] - self.dd_history[-2]
                
                if len(self.dd_history) >= 3:
                    # Calculate acceleration (rate of velocity change)
                    prev_velocity = self.dd_history[-2] - self.dd_history[-3]
                    self.dd_acceleration = self.dd_velocity - prev_velocity
                else:
                    self.dd_acceleration = 0.0
            else:
                self.dd_velocity = 0.0
                self.dd_acceleration = 0.0
                
        except Exception as e:
            self.log_operator_warning(f"Drawdown dynamics calculation failed: {e}")
            self.dd_velocity = 0.0
            self.dd_acceleration = 0.0

    def _determine_drawdown_severity_enhanced(self, drawdown: float, 
                                            context: Dict[str, Any]) -> str:
        """Enhanced severity determination with context and velocity"""
        
        # Get current thresholds
        thresholds = self.current_thresholds
        
        # Base severity assessment
        if drawdown >= thresholds['dd_limit']:
            base_severity = "critical"
        elif drawdown >= thresholds['warning_dd']:
            base_severity = "warning"
        elif drawdown >= thresholds['info_dd']:
            base_severity = "info"
        else:
            base_severity = "none"
        
        # Velocity-based adjustments
        if self.dd_velocity > 0.02:  # Rapidly increasing drawdown (>2%)
            if base_severity == "info":
                base_severity = "warning"
            elif base_severity == "warning":
                base_severity = "critical"
        elif self.dd_velocity < -0.01:  # Recovering (improving by >1%)
            if base_severity == "warning" and drawdown < thresholds['warning_dd'] * 0.9:
                base_severity = "info"
        
        # Acceleration-based adjustments
        if self.dd_acceleration > 0.01:  # Accelerating downward
            if base_severity in ["info", "warning"]:
                # Escalate severity for accelerating drawdowns
                severity_escalation = {"info": "warning", "warning": "critical"}
                base_severity = severity_escalation.get(base_severity, base_severity)
        
        # Context-based adjustments
        regime = context.get('regime', 'unknown')
        volatility_level = context.get('volatility_level', 'medium')
        
        # More tolerant in volatile markets
        if volatility_level in ['high', 'extreme'] and base_severity == "warning":
            if drawdown < thresholds['dd_limit'] * 0.8:  # 20% tolerance
                base_severity = "info"
        
        return base_severity

    def _calculate_recovery_progress(self) -> None:
        """Calculate recovery progress from maximum drawdown"""
        
        try:
            if self.max_dd > 0:
                recovery_amount = max(0, self.max_dd - self.current_dd)
                self.recovery_progress = recovery_amount / self.max_dd
            else:
                self.recovery_progress = 0.0
                
            # Check for recovery milestone
            if (self.recovery_progress >= self.current_thresholds['recovery_threshold'] and
                self.max_dd > self.current_thresholds['info_dd']):
                
                # Record recovery event
                if not hasattr(self, '_last_recovery_recorded') or self._last_recovery_recorded != self.step_count:
                    self.recovery_events.append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'step': self.step_count,
                        'max_dd': self.max_dd,
                        'current_dd': self.current_dd,
                        'recovery_progress': self.recovery_progress
                    })
                    self._successful_recoveries += 1
                    self._last_recovery_recorded = self.step_count
                    
                    self.log_operator_info(
                        f"🎯 Recovery milestone achieved",
                        progress=f"{self.recovery_progress:.1%}",
                        max_dd=f"{self.max_dd:.1%}",
                        current_dd=f"{self.current_dd:.1%}"
                    )
                    
        except Exception as e:
            self.log_operator_warning(f"Recovery progress calculation failed: {e}")

    def _log_drawdown_changes(self, old_dd: float, old_severity: str, 
                             context: Dict[str, Any]) -> bool:
        """Log significant drawdown changes"""
        
        critical_found = False
        
        try:
            # Log severity changes or significant movements
            should_log = (
                self.severity != old_severity or
                abs(self.current_dd - old_dd) > 0.01 or  # >1% change
                self.severity in ["warning", "critical"] or
                self.dd_velocity > 0.015  # Rapid deterioration
            )
            
            if should_log:
                # Build comprehensive message
                msg_parts = [f"Drawdown {self.current_dd:.1%} - {self.severity.upper()}"]
                
                if self.dd_velocity != 0:
                    velocity_text = "deteriorating" if self.dd_velocity > 0 else "improving"
                    msg_parts.append(f"({velocity_text} {abs(self.dd_velocity):.1%})")
                
                if self.recovery_progress > 0:
                    msg_parts.append(f"(recovery: {self.recovery_progress:.1%})")
                
                message = " ".join(msg_parts)
                
                # Log with appropriate severity
                if self.severity == "critical":
                    self.log_operator_error(
                        f"🚨 CRITICAL: {message}",
                        limit=f"{self.current_thresholds['dd_limit']:.1%}",
                        velocity=f"{self.dd_velocity:+.1%}",
                        regime=context.get('regime', 'unknown')
                    )
                    critical_found = True
                elif self.severity == "warning":
                    self.log_operator_warning(
                        f"⚠️ WARNING: {message}",
                        threshold=f"{self.current_thresholds['warning_dd']:.1%}",
                        velocity=f"{self.dd_velocity:+.1%}"
                    )
                elif self.severity == "info":
                    self.log_operator_info(
                        f"ℹ️ INFO: {message}",
                        threshold=f"{self.current_thresholds['info_dd']:.1%}"
                    )
                elif self.config.debug:
                    self.log_operator_info(f"📊 {message}")
                    
        except Exception as e:
            self.log_operator_warning(f"Drawdown logging failed: {e}")
        
        return critical_found

    def _update_drawdown_analytics(self, context: Dict[str, Any]) -> None:
        """Update comprehensive drawdown analytics"""
        
        try:
            # Update general analytics
            self.drawdown_analytics['current_dd'].append(self.current_dd)
            self.drawdown_analytics['max_dd'].append(self.max_dd)
            self.drawdown_analytics['velocity'].append(self.dd_velocity)
            self.drawdown_analytics['recovery_progress'].append(self.recovery_progress)
            
            # Update regime-specific analytics
            regime = context.get('regime', 'unknown')
            self.regime_drawdowns[regime].append({
                'drawdown': self.current_dd,
                'severity': self.severity,
                'velocity': self.dd_velocity,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            # Update risk tracking
            self._update_risk_metrics({
                'current_drawdown': self.current_dd,
                'max_drawdown': self.max_dd,
                'drawdown_velocity': self.dd_velocity,
                'recovery_progress': self.recovery_progress
            })
            
        except Exception as e:
            self.log_operator_warning(f"Drawdown analytics update failed: {e}")

    def _store_drawdown_snapshot(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Store comprehensive drawdown snapshot"""
        
        try:
            snapshot = {
                'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
                'step_idx': info_bus.get('step_idx', self.step_count),
                'current_dd': self.current_dd,
                'max_dd': self.max_dd,
                'severity': self.severity,
                'velocity': self.dd_velocity,
                'acceleration': self.dd_acceleration,
                'recovery_progress': self.recovery_progress,
                'risk_adjustment': self.risk_adjustment,
                'rescue_mode': self.rescue_mode,
                'context': context.copy(),
                'balance': info_bus.get('risk', {}).get('balance'),
                'peak_balance': self.peak_balance
            }
            
            # Store in analysis history
            self._analysis_history.append(snapshot)
            
        except Exception as e:
            self.log_operator_warning(f"Drawdown snapshot storage failed: {e}")

    def _update_rescue_system(self, context: Dict[str, Any]) -> None:
        """Update intelligent rescue system"""
        
        try:
            old_rescue_mode = self.rescue_mode
            
            # Determine if rescue mode should be activated
            should_activate_rescue = (
                self.severity in ["warning", "critical"] or
                (self.dd_velocity > 0.02 and self.current_dd > self.current_thresholds['info_dd']) or
                (self.dd_acceleration > 0.01 and self.current_dd > 0.05)
            )
            
            # Activate rescue mode
            if should_activate_rescue and not self.rescue_mode:
                self.rescue_mode = True
                self.rescue_start_time = datetime.datetime.now()
                self._rescue_interventions += 1
                
                self.log_operator_warning(
                    "🛟 Rescue mode ACTIVATED",
                    drawdown=f"{self.current_dd:.1%}",
                    velocity=f"{self.dd_velocity:+.1%}",
                    reason=self.severity
                )
            
            # Deactivate rescue mode
            elif self.rescue_mode and (
                self.severity == "none" and 
                self.dd_velocity < 0 and 
                self.recovery_progress > 0.3
            ):
                self.rescue_mode = False
                rescue_duration = (datetime.datetime.now() - self.rescue_start_time).total_seconds() / 60
                
                self.log_operator_info(
                    "✅ Rescue mode DEACTIVATED",
                    duration=f"{rescue_duration:.1f} minutes",
                    recovery=f"{self.recovery_progress:.1%}",
                    final_dd=f"{self.current_dd:.1%}"
                )
            
            # Update consecutive rescue days tracking
            if self.rescue_mode and not old_rescue_mode:
                self.consecutive_rescue_days += 1
            elif not self.rescue_mode and old_rescue_mode:
                self.consecutive_rescue_days = 0
                
        except Exception as e:
            self.log_operator_warning(f"Rescue system update failed: {e}")

    def _calculate_risk_adjustment(self, context: Dict[str, Any]) -> None:
        """Calculate intelligent risk adjustment factor"""
        
        try:
            base_adjustment = self._get_base_risk_adjustment()
            
            # Velocity-based adjustments
            if self.dd_velocity > 0.02:  # Rapid deterioration
                base_adjustment *= 0.7
            elif self.dd_velocity > 0.01:  # Moderate deterioration
                base_adjustment *= 0.85
            elif self.dd_velocity < -0.01:  # Recovering
                base_adjustment = min(1.0, base_adjustment * 1.15)
            
            # Acceleration-based adjustments
            if self.dd_acceleration > 0.01:  # Accelerating downward
                base_adjustment *= 0.8
            
            # Context-based adjustments
            regime = context.get('regime', 'unknown')
            volatility_level = context.get('volatility_level', 'medium')
            
            # More conservative in volatile markets during drawdown
            if self.severity != "none" and volatility_level in ['high', 'extreme']:
                base_adjustment *= 0.9
            
            # More aggressive recovery in trending markets
            if regime == 'trending' and self.recovery_progress > 0.2:
                base_adjustment = min(1.0, base_adjustment * 1.1)
            
            # Apply adaptive rescue if enabled
            if self.adaptive_rescue and self.rescue_mode:
                # Progressive risk reduction during rescue
                rescue_duration = (datetime.datetime.now() - self.rescue_start_time).total_seconds() / 3600  # hours
                rescue_factor = max(0.5, 1.0 - rescue_duration * 0.1)  # Reduce 10% per hour
                base_adjustment *= rescue_factor
            
            # Final bounds and smoothing
            target_adjustment = np.clip(base_adjustment, 0.1, 1.0)
            
            # Smooth transitions to avoid abrupt changes
            if hasattr(self, 'risk_adjustment'):
                smoothing_factor = 0.3
                self.risk_adjustment = (
                    self.risk_adjustment * (1 - smoothing_factor) + 
                    target_adjustment * smoothing_factor
                )
            else:
                self.risk_adjustment = target_adjustment
            
            self.risk_adjustment = float(np.clip(self.risk_adjustment, 0.1, 1.0))
            
        except Exception as e:
            self.log_operator_warning(f"Risk adjustment calculation failed: {e}")
            self.risk_adjustment = 0.5  # Conservative fallback

    def _get_base_risk_adjustment(self) -> float:
        """Get base risk adjustment based on severity"""
        
        severity_adjustments = {
            "none": 1.0,
            "info": 0.8,
            "warning": 0.5,
            "critical": 0.2
        }
        
        return severity_adjustments.get(self.severity, 0.5)

    def _update_info_bus(self, info_bus: InfoBus, critical_found: bool) -> None:
        """Update InfoBus with drawdown rescue results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'drawdown_rescue', {
            'current_drawdown': self.current_dd,
            'max_drawdown': self.max_dd,
            'severity': self.severity,
            'velocity': self.dd_velocity,
            'acceleration': self.dd_acceleration,
            'recovery_progress': self.recovery_progress,
            'risk_adjustment': self.risk_adjustment,
            'rescue_mode': self.rescue_mode,
            'rescue_interventions': self._rescue_interventions,
            'successful_recoveries': self._successful_recoveries,
            'thresholds': self.current_thresholds.copy()
        })
        
        # Update risk snapshot
        InfoBusUpdater.update_risk_snapshot(info_bus, {
            'drawdown_risk_score': min(self.current_dd * 2, 1.0),  # Normalized risk score
            'rescue_active': self.rescue_mode,
            'risk_reduction_factor': self.risk_adjustment
        })
        
        # Add alerts for critical situations
        if critical_found:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Critical drawdown: {self.current_dd:.1%} (limit: {self.current_thresholds['dd_limit']:.1%})",
                severity="critical",
                module="DrawdownRescue"
            )
        elif self.rescue_mode:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Rescue mode active: {self.current_dd:.1%} drawdown",
                severity="warning",
                module="DrawdownRescue"
            )

    def _record_comprehensive_audit(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Record comprehensive audit trail"""
        
        audit_data = {
            'drawdown_metrics': {
                'current_dd': self.current_dd,
                'max_dd': self.max_dd,
                'severity': self.severity,
                'velocity': self.dd_velocity,
                'acceleration': self.dd_acceleration,
                'recovery_progress': self.recovery_progress
            },
            'rescue_system': {
                'risk_adjustment': self.risk_adjustment,
                'rescue_mode': self.rescue_mode,
                'rescue_interventions': self._rescue_interventions,
                'successful_recoveries': self._successful_recoveries
            },
            'context': context,
            'thresholds': self.current_thresholds.copy(),
            'step_count': self.step_count,
            'balance_info': {
                'current_balance': info_bus.get('risk', {}).get('balance'),
                'peak_balance': self.peak_balance,
                'balance_history_size': len(self.balance_history)
            }
        }
        
        self.audit_manager.record_event(
            event_type="drawdown_analysis",
            module="DrawdownRescue",
            details=audit_data,
            severity="critical" if self.severity == "critical" else "warning" if self.severity == "warning" else "info"
        )

    def _update_rescue_metrics(self) -> None:
        """Update performance and rescue metrics"""
        
        # Update performance metrics
        self._update_performance_metric('current_drawdown', self.current_dd)
        self._update_performance_metric('max_drawdown', self.max_dd)
        self._update_performance_metric('risk_adjustment', self.risk_adjustment)
        self._update_performance_metric('recovery_progress', self.recovery_progress)
        
        # Update rescue system metrics
        self._update_performance_metric('rescue_interventions', self._rescue_interventions)
        self._update_performance_metric('successful_recoveries', self._successful_recoveries)
        
        # Calculate rescue effectiveness
        if self._rescue_interventions > 0:
            effectiveness = self._successful_recoveries / self._rescue_interventions
            self._update_performance_metric('rescue_effectiveness', effectiveness)

    def _determine_current_session(self, info_bus: Optional[InfoBus] = None) -> str:
                """Determine current trading session with enhanced detection"""
                
                try:
                    # Method 1: Try InfoBus session data
                    if info_bus:
                        session = info_bus.get('session')
                        if session and session != 'unknown':
                            return session
                        
                        # Method 2: Try market context
                        market_context = info_bus.get('market_context', {})
                        session = market_context.get('session')
                        if session and session != 'unknown':
                            return session
                    
                    # Method 3: Determine from current time (UTC)
                    import datetime
                    import pytz
                    
                    now_utc = datetime.datetime.now(pytz.UTC)
                    hour_utc = now_utc.hour
                    
                    # Trading session detection based on UTC hours
                    if 22 <= hour_utc or hour_utc < 6:  # 22:00-06:00 UTC
                        return 'asian'
                    elif 6 <= hour_utc < 14:  # 06:00-14:00 UTC  
                        return 'european'
                    elif 14 <= hour_utc < 22:  # 14:00-22:00 UTC
                        return 'american'
                    else:
                        return 'rollover'
                        
                except Exception as e:
                    self.log_operator_warning(f"Session detection failed: {e}")
                    return 'unknown'

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components for model integration"""
        
        try:
            # Severity mapping
            severity_map = {"none": 0.0, "info": 0.25, "warning": 0.5, "critical": 1.0}
            severity_score = severity_map.get(self.severity, 0.0)
            
            # Normalized components
            drawdown_norm = float(np.clip(self.current_dd, 0.0, 1.0))
            max_drawdown_norm = float(np.clip(self.max_dd, 0.0, 1.0))
            risk_adjustment = float(self.risk_adjustment)
            
            # Velocity (centered at 0.5 for neutral)
            velocity_norm = float(np.clip(self.dd_velocity + 0.5, 0.0, 1.0))
            
            # Recovery progress
            recovery_norm = float(self.recovery_progress)
            
            # Rescue mode indicator
            rescue_mode_indicator = float(self.rescue_mode)
            
            # Trend indicator (based on recent velocity)
            if len(self.dd_history) >= 3:
                recent_trend = np.mean([
                    self.dd_history[-1] - self.dd_history[-2],
                    self.dd_history[-2] - self.dd_history[-3]
                ])
                trend_indicator = float(np.clip(recent_trend + 0.5, 0.0, 1.0))
            else:
                trend_indicator = 0.5
            
            return np.array([
                severity_score,            # Drawdown severity
                drawdown_norm,            # Current drawdown
                max_drawdown_norm,        # Maximum drawdown
                risk_adjustment,          # Risk adjustment factor
                velocity_norm,            # Drawdown velocity
                recovery_norm,            # Recovery progress
                rescue_mode_indicator,    # Rescue mode active
                trend_indicator           # Trend indicator
            ], dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Drawdown observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)

    def get_drawdown_report(self) -> str:
        """Generate operator-friendly drawdown report"""
        
        # Status indicators
        if self.severity == "critical":
            status = "🚨 Critical"
        elif self.severity == "warning":
            status = "⚠️ Warning"
        elif self.severity == "info":
            status = "ℹ️ Monitored"
        else:
            status = "✅ Normal"
        
        # Rescue status
        if self.rescue_mode:
            rescue_status = "🛟 Active"
            rescue_duration = (datetime.datetime.now() - self.rescue_start_time).total_seconds() / 60
            rescue_info = f"(Active for {rescue_duration:.1f} min)"
        else:
            rescue_status = "💤 Standby"
            rescue_info = ""
        
        # Velocity status
        if self.dd_velocity > 0.01:
            velocity_status = "📉 Deteriorating"
        elif self.dd_velocity < -0.01:
            velocity_status = "📈 Improving"
        else:
            velocity_status = "➡️ Stable"
        
        # Recovery status
        if self.recovery_progress > 0.5:
            recovery_status = "🎯 Strong Recovery"
        elif self.recovery_progress > 0.2:
            recovery_status = "⚡ Moderate Recovery"
        elif self.recovery_progress > 0:
            recovery_status = "🔄 Early Recovery"
        else:
            recovery_status = "📊 No Recovery"
        
        # Recent rescue history
        rescue_history_lines = []
        for event in self.recovery_events[-3:]:  # Show last 3 recoveries
            timestamp = event['timestamp'][:19]  # Remove microseconds
            rescue_history_lines.append(
                f"  ✅ {timestamp}: {event['max_dd']:.1%} → {event['current_dd']:.1%} "
                f"({event['recovery_progress']:.1%} recovery)"
            )
        
        return f"""
🛟 ENHANCED DRAWDOWN RESCUE
═══════════════════════════════════════
🎯 Status: {status} ({self.current_dd:.1%} drawdown)
🛟 Rescue Mode: {rescue_status} {rescue_info}
📊 Velocity: {velocity_status} ({self.dd_velocity:+.1%})
🔄 Recovery: {recovery_status} ({self.recovery_progress:.1%})

⚖️ DRAWDOWN THRESHOLDS
• Info Level: {self.current_thresholds['info_dd']:.1%}
• Warning Level: {self.current_thresholds['warning_dd']:.1%}
• Critical Level: {self.current_thresholds['dd_limit']:.1%}
• Recovery Threshold: {self.current_thresholds['recovery_threshold']:.1%}

📊 CURRENT METRICS
• Current Drawdown: {self.current_dd:.1%}
• Maximum Drawdown: {self.max_dd:.1%}
• Peak Balance: €{self.peak_balance:,.0f}
• Velocity: {self.dd_velocity:+.1%} per step
• Acceleration: {self.dd_acceleration:+.1%} per step²

🛡️ RISK MANAGEMENT
• Risk Adjustment: {self.risk_adjustment:.1%}
• Training Mode: {'✅ Enabled' if self.training_mode else '❌ Disabled'}
• Adaptive Rescue: {'✅ Enabled' if self.adaptive_rescue else '❌ Disabled'}
• Consecutive Rescue Days: {self.consecutive_rescue_days}

📈 RESCUE PERFORMANCE
• Total Interventions: {self._rescue_interventions}
• Successful Recoveries: {self._successful_recoveries}
• Success Rate: {(self._successful_recoveries / max(self._rescue_interventions, 1)):.1%}
• False Alarms: {self._false_alarms}

📜 RECENT RECOVERY HISTORY
{chr(10).join(rescue_history_lines) if rescue_history_lines else "  📭 No recent recoveries"}

💡 SYSTEM STATUS
• Step Count: {self.step_count:,}
• Balance History: {len(self.balance_history)} records
• DD History: {len(self.dd_history)} records
• Regime Tracking: {len(self.regime_drawdowns)} regimes
        """

    # ================== RESCUE SYSTEM INTERFACE ==================

    def get_risk_adjustment(self) -> float:
        """Get current risk adjustment factor for external use"""
        return self.risk_adjustment

    def is_rescue_mode_active(self) -> bool:
        """Check if rescue mode is currently active"""
        return self.rescue_mode

    def get_rescue_recommendations(self) -> Dict[str, Any]:
        """Get current rescue recommendations"""
        
        recommendations = {
            'reduce_position_size': self.risk_adjustment < 0.8,
            'close_losing_positions': self.severity in ["warning", "critical"],
            'stop_new_trades': self.rescue_mode and self.dd_velocity > 0.02,
            'increase_stops': self.severity != "none",
            'wait_for_recovery': self.recovery_progress > 0 and self.dd_velocity < 0
        }
        
        return recommendations

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, current_drawdown: Optional[float] = None,
            balance: Optional[float] = None, peak_balance: Optional[float] = None,
            equity: Optional[float] = None, **kwargs) -> bool:
        """Legacy compatibility method"""
        
        # Create mock InfoBus from legacy parameters
        mock_info_bus = {
            'step_idx': self.step_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'risk': {}
        }
        
        # Add drawdown data to mock InfoBus
        if current_drawdown is not None:
            mock_info_bus['risk']['current_drawdown'] = current_drawdown
        
        if balance is not None:
            mock_info_bus['risk']['balance'] = balance
        
        if peak_balance is not None:
            mock_info_bus['risk']['peak_balance'] = peak_balance
        
        if equity is not None:
            mock_info_bus['risk']['equity'] = equity
        
        # Use enhanced step method
        self._step_impl(mock_info_bus)
        
        # Return True if critical drawdown reached
        return self.severity == "critical"