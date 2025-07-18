"""
Enhanced Drawdown Rescue with SmartInfoBus Integration
Intelligent drawdown monitoring and rescue mechanisms
"""

import numpy as np
import datetime
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import deque, defaultdict

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="DrawdownRescue",
    version="3.0.0",
    category="risk",
    provides=["drawdown_risk", "rescue_status", "risk_adjustment"],
    requires=["balance", "equity", "positions", "market_context"],
    description="Enhanced drawdown monitoring with intelligent rescue mechanisms and risk adjustment",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class DrawdownRescue(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin):
    """
    Enhanced Drawdown Rescue with SmartInfoBus Integration
    
    Monitors portfolio drawdown with advanced analytics including velocity analysis,
    progressive rescue mechanisms, and intelligent risk adjustment factors.
    """
    
    def __init__(self, config=None, **kwargs):
        self.config = config or {}
        super().__init__()
        self._initialize_advanced_systems()
        
        # Configuration with intelligent defaults
        self.dd_limit = self.config.get('dd_limit', 0.25)
        self.warning_dd = self.config.get('warning_dd', 0.15)
        self.info_dd = self.config.get('info_dd', 0.08)
        self.recovery_threshold = self.config.get('recovery_threshold', 0.5)
        self.enabled = self.config.get('enabled', True)
        
        # Advanced rescue configuration
        self.velocity_window = self.config.get('velocity_window', 10)
        self.rescue_mode_enabled = self.config.get('rescue_mode', True)
        self.adaptive_thresholds = self.config.get('adaptive_thresholds', True)
        
        # State tracking
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.peak_balance = 0.0
        self.dd_velocity = 0.0
        self.dd_acceleration = 0.0
        self.severity_level = "normal"
        
        # Rescue system state
        self.rescue_mode = False
        self.rescue_start_time = None
        self.risk_adjustment_factor = 1.0
        self.rescue_intervention_count = 0
        
        # Advanced analytics
        self.dd_history = deque(maxlen=self.velocity_window)
        self.balance_history = deque(maxlen=100)
        self.recovery_events = []
        self.regime_drawdowns = defaultdict(list)
        
        # Performance tracking
        self.step_count = 0
        self.successful_recoveries = 0
        self.false_alarms = 0
        self.emergency_interventions = 0
        
        # Dynamic thresholds
        self.current_thresholds = {
            'dd_limit': self.dd_limit,
            'warning_dd': self.warning_dd,
            'info_dd': self.info_dd
        }
        
        # Context-aware adjustments
        self.regime_multipliers = {
            'volatile': {'warning': 1.3, 'critical': 1.2, 'velocity_tolerance': 0.8},
            'trending': {'warning': 0.9, 'critical': 0.95, 'velocity_tolerance': 1.1},
            'ranging': {'warning': 1.0, 'critical': 1.0, 'velocity_tolerance': 1.0},
            'crisis': {'warning': 0.8, 'critical': 0.85, 'velocity_tolerance': 0.7}
        }
        
        self.logger.info(format_operator_message(
            message="Enhanced Drawdown Rescue initialized",
            icon="🛟",
            critical_limit=f"{self.dd_limit:.1%}",
            warning_threshold=f"{self.warning_dd:.1%}",
            rescue_mode=self.rescue_mode_enabled,
            adaptive_thresholds=self.adaptive_thresholds,
            enabled=self.enabled
        ))
    
    def _initialize_advanced_systems(self):
        """Initialize advanced monitoring and error handling systems"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="DrawdownRescue",
            log_path="logs/risk/drawdown_rescue.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("DrawdownRescue", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
    
    def _initialize(self) -> None:
        """Initialize the drawdown rescue module (required by BaseModule)"""
        try:
            # Validate configuration
            if not self.enabled:
                self.logger.warning("Drawdown Rescue is disabled")
                return
            
            # Initialize state tracking
            self.current_dd = 0.0
            self.max_dd = 0.0
            self.peak_balance = 0.0
            self.dd_velocity = 0.0
            self.dd_acceleration = 0.0
            self.severity_level = "normal"
            
            # Initialize rescue system state
            self.rescue_mode = False
            self.rescue_start_time = None
            self.risk_adjustment_factor = 1.0
            self.rescue_intervention_count = 0
            
            # Initialize performance tracking
            self.step_count = 0
            self.successful_recoveries = 0
            self.false_alarms = 0
            self.emergency_interventions = 0
            
            self.logger.info("Drawdown Rescue module initialization completed successfully")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "rescue_initialization")
            self.logger.error(f"Rescue initialization failed: {error_context}")
    
    async def calculate_confidence(self, action: Dict[str, Any], **kwargs) -> float:
        """Calculate confidence score for drawdown rescue assessment"""
        try:
            # Base confidence starts high for rescue operations
            confidence = 0.9
            
            # Factors that affect confidence
            factors = {}
            
            # Severity level affects confidence
            severity_penalties = {
                'normal': 1.0,
                'info': 0.9,
                'warning': 0.8,
                'critical': 0.6,
                'error': 0.3,
                'disabled': 0.1
            }
            severity_factor = severity_penalties.get(self.severity_level, 0.5)
            factors['severity_factor'] = severity_factor
            confidence *= severity_factor
            
            # Rescue mode affects confidence
            if self.rescue_mode:
                # Lower confidence during rescue operations (more conservative)
                rescue_penalty = 0.7
                factors['rescue_penalty'] = rescue_penalty
                confidence *= rescue_penalty
            
            # Velocity affects confidence (rapid changes reduce confidence)
            if hasattr(self, 'dd_velocity') and abs(self.dd_velocity) > 0.02:
                velocity_penalty = max(0.5, 1.0 - abs(self.dd_velocity) * 10)
                factors['velocity_penalty'] = velocity_penalty
                confidence *= velocity_penalty
            
            # Data availability affects confidence
            if hasattr(self, 'dd_history') and len(self.dd_history) >= 5:
                data_factor = min(1.0, len(self.dd_history) / 10.0)
            else:
                data_factor = 0.5  # Lower confidence with limited data
            factors['data_availability'] = data_factor
            confidence *= data_factor
            
            # Recovery track record affects confidence
            if self.rescue_intervention_count > 0:
                success_rate = self.successful_recoveries / self.rescue_intervention_count
                factors['success_rate'] = success_rate
                confidence *= (0.5 + success_rate * 0.5)  # 50% base + 50% based on success
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            # Log confidence calculation for debugging
            self.logger.debug(f"Rescue confidence: {confidence:.3f}, factors: {factors}")
            
            return float(confidence)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "confidence_calculation")
            self.logger.error(f"Confidence calculation failed: {error_context}")
            return 0.5  # Default medium confidence on error
    
    async def propose_action(self, **kwargs) -> Dict[str, Any]:
        """Propose drawdown rescue actions based on current state"""
        try:
            action_proposal = {
                'action_type': 'drawdown_rescue',
                'timestamp': time.time(),
                'current_drawdown': self.current_dd,
                'severity_level': self.severity_level,
                'rescue_mode': self.rescue_mode,
                'risk_adjustment_factor': self.risk_adjustment_factor,
                'recommendations': [],
                'warnings': [],
                'adjustments': {}
            }
            
            # Generate recommendations based on drawdown state
            if self.severity_level == 'critical':
                action_proposal['recommendations'].append({
                    'type': 'emergency_intervention',
                    'reason': f'Critical drawdown level: {self.current_dd:.1%}',
                    'suggested_action': 'Immediately reduce position sizes and halt new entries',
                    'priority': 'critical'
                })
                
                action_proposal['adjustments']['position_reduction'] = 0.5  # Reduce by 50%
                action_proposal['adjustments']['new_entry_halt'] = True
            
            elif self.severity_level == 'warning':
                action_proposal['recommendations'].append({
                    'type': 'risk_reduction',
                    'reason': f'Warning drawdown level: {self.current_dd:.1%}',
                    'suggested_action': 'Reduce position sizes and increase stop-loss levels',
                    'priority': 'high'
                })
                
                action_proposal['adjustments']['position_reduction'] = 0.7  # Reduce by 30%
                action_proposal['adjustments']['tighter_stops'] = True
            
            elif self.severity_level == 'info':
                action_proposal['recommendations'].append({
                    'type': 'caution',
                    'reason': f'Elevated drawdown level: {self.current_dd:.1%}',
                    'suggested_action': 'Monitor closely and prepare for potential action',
                    'priority': 'medium'
                })
            
            # Velocity-based recommendations
            if hasattr(self, 'dd_velocity') and self.dd_velocity > 0.02:
                action_proposal['warnings'].append({
                    'type': 'rapid_deterioration',
                    'velocity': self.dd_velocity,
                    'threshold': 0.02,
                    'risk_level': 'high'
                })
                
                action_proposal['recommendations'].append({
                    'type': 'velocity_control',
                    'reason': f'Rapid drawdown increase detected: {self.dd_velocity:.2%}',
                    'suggested_action': 'Implement immediate position size reduction',
                    'priority': 'high'
                })
            
            # Rescue mode specific actions
            if self.rescue_mode:
                action_proposal['recommendations'].append({
                    'type': 'rescue_active',
                    'reason': 'Rescue mode is active',
                    'suggested_action': 'Continue conservative risk management until recovery',
                    'priority': 'ongoing'
                })
                
                # Calculate rescue duration
                if self.rescue_start_time:
                    rescue_duration = (datetime.datetime.now() - self.rescue_start_time).total_seconds() / 3600
                    if rescue_duration > 24:  # More than 24 hours
                        action_proposal['warnings'].append({
                            'type': 'prolonged_rescue',
                            'duration_hours': rescue_duration,
                            'risk_level': 'medium'
                        })
            
            # Risk adjustment factor recommendations
            if self.risk_adjustment_factor < 0.8:
                action_proposal['adjustments']['conservative_mode'] = True
                action_proposal['adjustments']['risk_factor'] = self.risk_adjustment_factor
            
            self.logger.debug(f"Rescue action proposed: {len(action_proposal['recommendations'])} recommendations, "
                            f"{len(action_proposal['warnings'])} warnings")
            
            return action_proposal
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "action_proposal")
            self.logger.error(f"Action proposal failed: {error_context}")
            return {
                'action_type': 'drawdown_rescue',
                'timestamp': time.time(),
                'error': str(e),
                'recommendations': [],
                'warnings': [],
                'adjustments': {}
            }
    
    async def process(self, **kwargs) -> Dict[str, Any]:
        """
        Enhanced drawdown monitoring with comprehensive rescue mechanisms
        
        Returns:
            Dict containing drawdown risk assessment, rescue status, and risk adjustments
        """
        try:
            if not self.enabled:
                return self._generate_disabled_response()
            
            self.step_count += 1
            
            # Extract comprehensive financial data
            balance = self.smart_bus.get('balance', 'DrawdownRescue') or 10000.0
            equity = self.smart_bus.get('equity', 'DrawdownRescue') or balance
            positions = self.smart_bus.get('positions', 'DrawdownRescue') or []
            market_context = self.smart_bus.get('market_context', 'DrawdownRescue') or {}
            
            # Update balance tracking
            self._update_balance_tracking(balance, equity)
            
            # Calculate comprehensive drawdown metrics
            drawdown_analysis = await self._analyze_drawdown_comprehensive(balance, equity, market_context)
            
            # Update rescue system
            rescue_status = self._update_rescue_system(drawdown_analysis, market_context)
            
            # Calculate risk adjustment
            risk_adjustment = self._calculate_risk_adjustment(drawdown_analysis, rescue_status)
            
            # Generate intelligent thesis
            thesis = await self._generate_drawdown_thesis(drawdown_analysis, rescue_status, market_context)
            
            # Calculate comprehensive metrics
            drawdown_metrics = self._calculate_drawdown_metrics(drawdown_analysis, rescue_status)
            
            # Update SmartInfoBus
            self._update_smart_info_bus(drawdown_analysis, rescue_status, risk_adjustment, thesis)
            
            # Record performance metrics
            self.performance_tracker.record_metric(
                'DrawdownRescue', 'analysis_cycle', 
                drawdown_analysis.get('processing_time_ms', 0), True
            )
            
            return {
                'current_drawdown': self.current_dd,
                'severity_level': self.severity_level,
                'rescue_mode': self.rescue_mode,
                'risk_adjustment_factor': self.risk_adjustment_factor,
                'drawdown_analysis': drawdown_analysis,
                'rescue_status': rescue_status,
                'risk_adjustment': risk_adjustment,
                'drawdown_metrics': drawdown_metrics,
                'thesis': thesis,
                'recommendations': self._generate_recommendations(drawdown_analysis, rescue_status)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "DrawdownRescue")
            self.logger.error(f"Drawdown analysis failed: {error_context}")
            return self._generate_error_response(str(error_context))
    
    def _update_balance_tracking(self, balance: float, equity: float):
        """Update balance and equity tracking"""
        try:
            # Use equity for drawdown calculation (includes unrealized P&L)
            effective_balance = equity if equity > 0 else balance
            
            # Track balance history
            self.balance_history.append({
                'balance': balance,
                'equity': equity,
                'effective_balance': effective_balance,
                'timestamp': datetime.datetime.now()
            })
            
            # Update peak balance
            if effective_balance > self.peak_balance:
                self.peak_balance = effective_balance
                
                # Log new peak achievement
                if self.step_count > 1:  # Avoid logging on first step
                    self.logger.info(format_operator_message(
                        message="New peak balance achieved",
                        icon="[CHART]",
                        peak_balance=f"€{self.peak_balance:,.2f}",
                        previous_max_dd=f"{self.max_dd:.1%}"
                    ))
                    
                    # Reset max drawdown on new peak
                    if self.max_dd > 0:
                        self.max_dd = 0.0
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "balance_tracking")
            self.logger.warning(f"Balance tracking failed: {error_context}")
    
    async def _analyze_drawdown_comprehensive(self, balance: float, equity: float,
                                            market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive drawdown analysis with advanced metrics"""
        start_time = datetime.datetime.now()
        
        try:
            effective_balance = equity if equity > 0 else balance
            
            # Calculate current drawdown
            if self.peak_balance > 0:
                self.current_dd = max(0.0, (self.peak_balance - effective_balance) / self.peak_balance)
            else:
                self.current_dd = 0.0
            
            # Update maximum drawdown
            self.max_dd = max(self.max_dd, self.current_dd)
            
            # Calculate velocity and acceleration
            velocity_metrics = self._calculate_velocity_metrics()
            
            # Adjust thresholds for market context
            context_thresholds = self._calculate_context_adjusted_thresholds(market_context)
            
            # Determine severity level
            severity_assessment = self._assess_drawdown_severity(velocity_metrics, context_thresholds)
            
            # Calculate recovery metrics
            recovery_metrics = self._calculate_recovery_metrics()
            
            # Regime-specific analysis
            regime_analysis = self._analyze_regime_patterns(market_context)
            
            # Calculate processing time
            processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'current_drawdown': self.current_dd,
                'max_drawdown': self.max_dd,
                'peak_balance': self.peak_balance,
                'effective_balance': effective_balance,
                'velocity_metrics': velocity_metrics,
                'context_thresholds': context_thresholds,
                'severity_assessment': severity_assessment,
                'recovery_metrics': recovery_metrics,
                'regime_analysis': regime_analysis,
                'processing_time_ms': processing_time,
                'market_context': market_context
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "drawdown_analysis")
            self.logger.error(f"Drawdown analysis failed: {error_context}")
            return self._generate_analysis_error_response(str(error_context))
    
    def _calculate_velocity_metrics(self) -> Dict[str, Any]:
        """Calculate drawdown velocity and acceleration metrics"""
        try:
            # Add current drawdown to history
            self.dd_history.append(self.current_dd)
            
            # Calculate velocity (rate of change)
            if len(self.dd_history) >= 2:
                self.dd_velocity = self.dd_history[-1] - self.dd_history[-2]
            else:
                self.dd_velocity = 0.0
            
            # Calculate acceleration (rate of velocity change)
            if len(self.dd_history) >= 3:
                prev_velocity = self.dd_history[-2] - self.dd_history[-3]
                self.dd_acceleration = self.dd_velocity - prev_velocity
            else:
                self.dd_acceleration = 0.0
            
            # Calculate trend metrics
            trend_direction = 'deteriorating' if self.dd_velocity > 0 else 'improving' if self.dd_velocity < 0 else 'stable'
            
            # Calculate momentum
            momentum = 0.0
            if len(self.dd_history) >= 5:
                recent_dds = list(self.dd_history)[-5:]
                trend_slope = np.polyfit(range(len(recent_dds)), recent_dds, 1)[0]
                momentum = trend_slope
            
            return {
                'velocity': self.dd_velocity,
                'acceleration': self.dd_acceleration,
                'trend_direction': trend_direction,
                'momentum': momentum,
                'volatility': np.std(list(self.dd_history)) if len(self.dd_history) >= 3 else 0.0
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "velocity_calculation")
            self.logger.warning(f"Velocity calculation failed: {error_context}")
            return {'velocity': 0.0, 'acceleration': 0.0, 'trend_direction': 'unknown', 'momentum': 0.0}
    
    def _calculate_context_adjusted_thresholds(self, market_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate context-adjusted drawdown thresholds"""
        try:
            if not self.adaptive_thresholds:
                return self.current_thresholds.copy()
            
            regime = market_context.get('regime', 'unknown')
            volatility_level = market_context.get('volatility_level', 'medium')
            
            # Get base thresholds
            adjusted_thresholds = {
                'info_dd': self.info_dd,
                'warning_dd': self.warning_dd,
                'dd_limit': self.dd_limit
            }
            
            # Apply regime-specific adjustments
            if regime in self.regime_multipliers:
                regime_config = self.regime_multipliers[regime]
                adjusted_thresholds['warning_dd'] *= regime_config['warning']
                adjusted_thresholds['dd_limit'] *= regime_config['critical']
            
            # Apply volatility adjustments
            volatility_multipliers = {
                'low': 0.8,      # Stricter in low volatility
                'medium': 1.0,   # Normal thresholds
                'high': 1.2,     # More lenient in high volatility
                'extreme': 1.4   # Very lenient in extreme volatility
            }
            
            vol_multiplier = volatility_multipliers.get(volatility_level, 1.0)
            for key in adjusted_thresholds:
                adjusted_thresholds[key] *= vol_multiplier
            
            # Apply bounds to prevent extreme adjustments
            adjusted_thresholds['info_dd'] = max(0.03, min(0.15, adjusted_thresholds['info_dd']))
            adjusted_thresholds['warning_dd'] = max(0.08, min(0.25, adjusted_thresholds['warning_dd']))
            adjusted_thresholds['dd_limit'] = max(0.15, min(0.50, adjusted_thresholds['dd_limit']))
            
            # Update current thresholds
            self.current_thresholds = adjusted_thresholds
            
            return adjusted_thresholds
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "threshold_adjustment")
            self.logger.warning(f"Threshold adjustment failed: {error_context}")
            return self.current_thresholds.copy()
    
    def _assess_drawdown_severity(self, velocity_metrics: Dict[str, float],
                                 context_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Assess drawdown severity with velocity consideration"""
        try:
            # Base severity assessment
            if self.current_dd >= context_thresholds['dd_limit']:
                base_severity = 'critical'
            elif self.current_dd >= context_thresholds['warning_dd']:
                base_severity = 'warning'
            elif self.current_dd >= context_thresholds['info_dd']:
                base_severity = 'info'
            else:
                base_severity = 'normal'
            
            # Velocity-based adjustments
            velocity = velocity_metrics['velocity']
            acceleration = velocity_metrics['acceleration']
            
            severity_factors = []
            
            # Rapid deterioration escalation
            if velocity > 0.02:  # 2% increase per step
                if base_severity == 'info':
                    base_severity = 'warning'
                elif base_severity == 'warning':
                    base_severity = 'critical'
                severity_factors.append('rapid_deterioration')
            
            # Acceleration-based escalation
            if acceleration > 0.01:  # Accelerating downward
                if base_severity in ['normal', 'info']:
                    base_severity = 'warning'
                severity_factors.append('accelerating_decline')
            
            # Recovery de-escalation
            if velocity < -0.01 and base_severity == 'warning':  # Improving by >1%
                if self.current_dd < context_thresholds['warning_dd'] * 0.9:
                    base_severity = 'info'
                    severity_factors.append('recovering')
            
            self.severity_level = base_severity
            
            return {
                'level': base_severity,
                'base_assessment': base_severity,
                'severity_factors': severity_factors,
                'velocity_impact': velocity > 0.015,
                'acceleration_impact': acceleration > 0.008,
                'threshold_used': context_thresholds.get(f'{base_severity}_dd', context_thresholds['info_dd'])
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "severity_assessment")
            self.logger.warning(f"Severity assessment failed: {error_context}")
            return {'level': 'unknown', 'base_assessment': 'unknown', 'severity_factors': []}
    
    def _calculate_recovery_metrics(self) -> Dict[str, Any]:
        """Calculate recovery progress and metrics"""
        try:
            # Basic recovery calculation
            if self.max_dd > 0:
                recovery_amount = max(0.0, self.max_dd - self.current_dd)
                recovery_progress = recovery_amount / self.max_dd
            else:
                recovery_progress = 0.0
            
            # Recovery velocity
            recovery_velocity = 0.0
            if len(self.dd_history) >= 3:
                recent_recovery = self.dd_history[-3] - self.current_dd  # Recovery over 3 steps
                recovery_velocity = recent_recovery / 3.0
            
            # Recovery stability
            recovery_stability = 0.0
            if len(self.dd_history) >= 5:
                recent_dds = list(self.dd_history)[-5:]
                if all(dd <= recent_dds[0] for dd in recent_dds):  # Consistent improvement
                    recovery_stability = 1.0
                else:
                    recovery_stability = 0.5
            
            # Check for recovery milestone
            milestone_achieved = False
            if (recovery_progress >= self.recovery_threshold and 
                self.max_dd > self.current_thresholds['info_dd']):
                
                milestone_achieved = True
                
                # Record recovery event
                self.recovery_events.append({
                    'timestamp': datetime.datetime.now(),
                    'max_dd': self.max_dd,
                    'recovery_progress': recovery_progress,
                    'step_count': self.step_count
                })
                
                self.successful_recoveries += 1
                
                self.logger.info(format_operator_message(
                    message="Recovery milestone achieved",
                    icon="[TARGET]",
                    progress=f"{recovery_progress:.1%}",
                    max_dd=f"{self.max_dd:.1%}",
                    current_dd=f"{self.current_dd:.1%}"
                ))
            
            return {
                'recovery_progress': recovery_progress,
                'recovery_velocity': recovery_velocity,
                'recovery_stability': recovery_stability,
                'milestone_achieved': milestone_achieved,
                'total_recoveries': self.successful_recoveries
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recovery_metrics")
            self.logger.warning(f"Recovery metrics calculation failed: {error_context}")
            return {'recovery_progress': 0.0, 'recovery_velocity': 0.0, 'recovery_stability': 0.0}
    
    def _analyze_regime_patterns(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime-specific drawdown patterns"""
        try:
            regime = market_context.get('regime', 'unknown')
            
            # Store current drawdown in regime history
            if regime != 'unknown':
                self.regime_drawdowns[regime].append({
                    'drawdown': self.current_dd,
                    'timestamp': datetime.datetime.now(),
                    'step_count': self.step_count
                })
            
            # Calculate regime statistics
            regime_stats = {}
            for regime_name, regime_data in self.regime_drawdowns.items():
                if regime_data:
                    drawdowns = [entry['drawdown'] for entry in regime_data]
                    regime_stats[regime_name] = {
                        'avg_drawdown': float(np.mean(drawdowns)),
                        'max_drawdown': float(np.max(drawdowns)),
                        'drawdown_volatility': float(np.std(drawdowns)),
                        'sample_count': len(drawdowns)
                    }
            
            # Assess current regime relative to historical patterns
            regime_assessment = 'normal'
            if regime in regime_stats:
                current_stats = regime_stats[regime]
                if self.current_dd > current_stats['avg_drawdown'] * 1.5:
                    regime_assessment = 'above_average'
                elif self.current_dd > current_stats['max_drawdown'] * 0.9:
                    regime_assessment = 'near_maximum'
                elif self.current_dd < current_stats['avg_drawdown'] * 0.5:
                    regime_assessment = 'below_average'
            
            return {
                'current_regime': regime,
                'regime_stats': regime_stats,
                'regime_assessment': regime_assessment,
                'regime_pattern': self._identify_regime_pattern(regime, regime_stats)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_analysis")
            self.logger.warning(f"Regime analysis failed: {error_context}")
            return {'current_regime': 'unknown', 'regime_stats': {}, 'regime_assessment': 'unknown'}
    
    def _identify_regime_pattern(self, current_regime: str, regime_stats: Dict[str, Dict]) -> str:
        """Identify patterns in regime-specific drawdowns"""
        try:
            if current_regime not in regime_stats or len(regime_stats) < 2:
                return 'insufficient_data'
            
            current_avg = regime_stats[current_regime]['avg_drawdown']
            other_regimes = {k: v for k, v in regime_stats.items() if k != current_regime}
            
            if not other_regimes:
                return 'no_comparison'
            
            other_avg = np.mean([stats['avg_drawdown'] for stats in other_regimes.values()])
            
            if current_avg > other_avg * 1.3:
                return 'high_risk_regime'
            elif current_avg < other_avg * 0.7:
                return 'low_risk_regime'
            else:
                return 'normal_risk_regime'
                
        except Exception:
            return 'pattern_analysis_error'
    
    def _update_rescue_system(self, drawdown_analysis: Dict[str, Any],
                             market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Update rescue system status and interventions"""
        try:
            severity_assessment = drawdown_analysis['severity_assessment']
            velocity_metrics = drawdown_analysis['velocity_metrics']
            
            old_rescue_mode = self.rescue_mode
            
            # Determine if rescue mode should be activated
            should_activate = (
                severity_assessment['level'] in ['warning', 'critical'] or
                (velocity_metrics['velocity'] > 0.02 and self.current_dd > self.current_thresholds['info_dd']) or
                velocity_metrics['acceleration'] > 0.01
            )
            
            # Activate rescue mode
            if should_activate and not self.rescue_mode and self.rescue_mode_enabled:
                self.rescue_mode = True
                self.rescue_start_time = datetime.datetime.now()
                self.rescue_intervention_count += 1
                
                self.logger.warning(format_operator_message(
                    message="Rescue mode ACTIVATED",
                    icon="🛟",
                    drawdown=f"{self.current_dd:.1%}",
                    velocity=f"{velocity_metrics['velocity']:+.2%}",
                    reason=severity_assessment['level'],
                    intervention_count=self.rescue_intervention_count
                ))
            
            # Deactivate rescue mode
            elif self.rescue_mode and (
                severity_assessment['level'] == 'normal' and
                velocity_metrics['velocity'] < 0 and
                drawdown_analysis['recovery_metrics']['recovery_progress'] > 0.3
            ):
                rescue_duration = (datetime.datetime.now() - (self.rescue_start_time or datetime.datetime.now())).total_seconds() / 60
                self.rescue_mode = False
                
                self.logger.info(format_operator_message(
                    message="Rescue mode DEACTIVATED",
                    icon="[OK]",
                    duration=f"{rescue_duration:.1f} minutes",
                    final_dd=f"{self.current_dd:.1%}",
                    recovery=f"{drawdown_analysis['recovery_metrics']['recovery_progress']:.1%}"
                ))
            
            # Emergency intervention check
            emergency_intervention = False
            if (self.current_dd > self.current_thresholds['dd_limit'] * 1.2 and
                velocity_metrics['velocity'] > 0.03):
                
                emergency_intervention = True
                self.emergency_interventions += 1
                
                self.logger.error(format_operator_message(
                    message="EMERGENCY INTERVENTION",
                    icon="[ALERT]",
                    drawdown=f"{self.current_dd:.1%}",
                    velocity=f"{velocity_metrics['velocity']:+.2%}",
                    intervention_number=self.emergency_interventions
                ))
            
            return {
                'rescue_mode': self.rescue_mode,
                'rescue_activated': not old_rescue_mode and self.rescue_mode,
                'rescue_deactivated': old_rescue_mode and not self.rescue_mode,
                'rescue_duration_minutes': (
                    (datetime.datetime.now() - self.rescue_start_time).total_seconds() / 60
                    if self.rescue_mode and self.rescue_start_time else 0.0
                ),
                'intervention_count': self.rescue_intervention_count,
                'emergency_intervention': emergency_intervention,
                'emergency_count': self.emergency_interventions
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "rescue_system")
            self.logger.error(f"Rescue system update failed: {error_context}")
            return {'rescue_mode': False, 'error': error_context}
    
    def _calculate_risk_adjustment(self, drawdown_analysis: Dict[str, Any],
                                  rescue_status: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate intelligent risk adjustment factor"""
        try:
            severity_level = drawdown_analysis['severity_assessment']['level']
            velocity_metrics = drawdown_analysis['velocity_metrics']
            recovery_metrics = drawdown_analysis['recovery_metrics']
            
            # Base adjustment based on severity
            base_adjustments = {
                'normal': 1.0,
                'info': 0.9,
                'warning': 0.7,
                'critical': 0.4
            }
            
            base_adjustment = base_adjustments.get(severity_level, 0.5)
            
            # Velocity-based adjustments
            velocity_adjustment = 1.0
            if velocity_metrics['velocity'] > 0.02:  # Rapid deterioration
                velocity_adjustment = 0.7
            elif velocity_metrics['velocity'] > 0.01:  # Moderate deterioration
                velocity_adjustment = 0.85
            elif velocity_metrics['velocity'] < -0.01:  # Recovering
                velocity_adjustment = min(1.2, 1.0 + abs(velocity_metrics['velocity']) * 5)
            
            # Recovery-based adjustments
            recovery_adjustment = 1.0
            if recovery_metrics['recovery_progress'] > 0.5:
                recovery_adjustment = 1.1  # Slightly more aggressive during recovery
            elif recovery_metrics['recovery_velocity'] > 0.01:
                recovery_adjustment = 1.05
            
            # Rescue mode adjustments
            rescue_adjustment = 1.0
            if rescue_status['rescue_mode']:
                # Progressive risk reduction during rescue
                rescue_duration = rescue_status['rescue_duration_minutes']
                rescue_adjustment = max(0.3, 1.0 - rescue_duration * 0.02)  # 2% reduction per minute
            
            # Emergency intervention
            if rescue_status.get('emergency_intervention', False):
                rescue_adjustment = min(rescue_adjustment, 0.2)  # Severe reduction
            
            # Calculate final adjustment
            raw_adjustment = base_adjustment * velocity_adjustment * recovery_adjustment * rescue_adjustment
            
            # Smooth transitions to avoid abrupt changes
            if hasattr(self, 'risk_adjustment_factor'):
                smoothing_factor = 0.3
                self.risk_adjustment_factor = (
                    self.risk_adjustment_factor * (1 - smoothing_factor) +
                    raw_adjustment * smoothing_factor
                )
            else:
                self.risk_adjustment_factor = raw_adjustment
            
            # Apply bounds
            self.risk_adjustment_factor = float(np.clip(self.risk_adjustment_factor, 0.1, 1.5))
            
            return {
                'risk_adjustment_factor': self.risk_adjustment_factor,
                'base_adjustment': base_adjustment,
                'velocity_adjustment': velocity_adjustment,
                'recovery_adjustment': recovery_adjustment,
                'rescue_adjustment': rescue_adjustment,
                'raw_adjustment': raw_adjustment,
                'adjustment_reason': self._determine_adjustment_reason(severity_level, rescue_status)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "risk_adjustment")
            self.logger.error(f"Risk adjustment calculation failed: {error_context}")
            return {'risk_adjustment_factor': 0.5, 'error': error_context}
    
    def _determine_adjustment_reason(self, severity_level: str, rescue_status: Dict[str, Any]) -> str:
        """Determine the primary reason for risk adjustment"""
        if rescue_status.get('emergency_intervention', False):
            return 'emergency_intervention'
        elif rescue_status['rescue_mode']:
            return 'rescue_mode_active'
        elif severity_level == 'critical':
            return 'critical_drawdown'
        elif severity_level == 'warning':
            return 'warning_drawdown'
        elif severity_level == 'info':
            return 'elevated_drawdown'
        else:
            return 'normal_operation'
    
    def _calculate_drawdown_metrics(self, drawdown_analysis: Dict[str, Any],
                                   rescue_status: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive drawdown metrics"""
        try:
            velocity_metrics = drawdown_analysis['velocity_metrics']
            recovery_metrics = drawdown_analysis['recovery_metrics']
            
            # Performance metrics
            total_interventions = self.rescue_intervention_count + self.emergency_interventions
            intervention_success_rate = (
                self.successful_recoveries / max(total_interventions, 1)
            )
            
            # Risk metrics
            risk_trend = 'improving' if velocity_metrics['velocity'] < 0 else 'deteriorating' if velocity_metrics['velocity'] > 0 else 'stable'
            
            # System effectiveness
            avg_recovery_time = 0.0
            if self.recovery_events:
                recovery_times = [
                    (event['step_count'] - (self.recovery_events[i-1]['step_count'] if i > 0 else 0))
                    for i, event in enumerate(self.recovery_events)
                ]
                avg_recovery_time = np.mean(recovery_times)
            
            return {
                'current_drawdown': self.current_dd,
                'max_drawdown': self.max_dd,
                'severity_level': self.severity_level,
                'risk_adjustment_factor': self.risk_adjustment_factor,
                'drawdown_velocity': velocity_metrics['velocity'],
                'recovery_progress': recovery_metrics['recovery_progress'],
                'rescue_interventions': self.rescue_intervention_count,
                'emergency_interventions': self.emergency_interventions,
                'successful_recoveries': self.successful_recoveries,
                'intervention_success_rate': intervention_success_rate,
                'risk_trend': risk_trend,
                'avg_recovery_time': avg_recovery_time,
                'rescue_mode_active': rescue_status['rescue_mode']
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "drawdown_metrics")
            self.logger.error(f"Drawdown metrics calculation failed: {error_context}")
            return {'current_drawdown': self.current_dd, 'error': error_context}
    
    async def _generate_drawdown_thesis(self, drawdown_analysis: Dict[str, Any],
                                       rescue_status: Dict[str, Any],
                                       market_context: Dict[str, Any]) -> str:
        """Generate intelligent thesis explaining drawdown analysis"""
        try:
            thesis_parts = []
            
            # Drawdown status
            if self.current_dd > 0:
                thesis_parts.append(
                    f"Portfolio drawdown: {self.current_dd:.1%} (max: {self.max_dd:.1%}) "
                    f"from peak balance of €{self.peak_balance:,.0f}"
                )
            else:
                thesis_parts.append("Portfolio at or near peak performance with no significant drawdown")
            
            # Severity assessment
            severity_level = drawdown_analysis['severity_assessment']['level']
            velocity = drawdown_analysis['velocity_metrics']['velocity']
            
            if severity_level == 'critical':
                thesis_parts.append(f"CRITICAL drawdown level requiring immediate intervention")
            elif severity_level == 'warning':
                thesis_parts.append(f"WARNING level drawdown with active monitoring")
            elif severity_level == 'info':
                thesis_parts.append(f"Elevated drawdown within acceptable parameters")
            else:
                thesis_parts.append("Drawdown levels normal and well-controlled")
            
            # Velocity analysis
            if velocity > 0.02:
                thesis_parts.append(f"RAPID deterioration detected (velocity: {velocity:+.2%} per step)")
            elif velocity > 0.01:
                thesis_parts.append(f"Moderate deterioration trend (velocity: {velocity:+.2%})")
            elif velocity < -0.01:
                thesis_parts.append(f"Recovery in progress (velocity: {velocity:+.2%})")
            
            # Rescue system status
            if rescue_status['rescue_mode']:
                duration = rescue_status['rescue_duration_minutes']
                thesis_parts.append(f"Rescue mode ACTIVE for {duration:.1f} minutes - risk reduction engaged")
            elif rescue_status.get('emergency_intervention', False):
                thesis_parts.append("EMERGENCY intervention triggered - maximum risk reduction applied")
            
            # Recovery analysis
            recovery_progress = drawdown_analysis['recovery_metrics']['recovery_progress']
            if recovery_progress > 0.5:
                thesis_parts.append(f"Strong recovery progress: {recovery_progress:.1%} from maximum drawdown")
            elif recovery_progress > 0.2:
                thesis_parts.append(f"Moderate recovery underway: {recovery_progress:.1%}")
            
            # Market context impact
            regime = market_context.get('regime', 'unknown')
            if regime != 'unknown' and self.adaptive_thresholds:
                adjusted_threshold = drawdown_analysis['context_thresholds']['warning_dd']
                thesis_parts.append(f"Thresholds adjusted for {regime} market regime (warning: {adjusted_threshold:.1%})")
            
            # Risk adjustment conclusion
            thesis_parts.append(
                f"Risk adjustment factor: {self.risk_adjustment_factor:.1%} "
                f"({drawdown_analysis.get('risk_adjustment', {}).get('adjustment_reason', 'normal_operation')})"
            )
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Thesis generation failed: {error_context}"
    
    def _generate_recommendations(self, drawdown_analysis: Dict[str, Any],
                                 rescue_status: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on drawdown analysis"""
        recommendations = []
        
        try:
            severity_level = drawdown_analysis['severity_assessment']['level']
            velocity_metrics = drawdown_analysis['velocity_metrics']
            recovery_metrics = drawdown_analysis['recovery_metrics']
            
            # Critical situation recommendations
            if severity_level == 'critical':
                recommendations.append("IMMEDIATE: Reduce position sizes significantly")
                recommendations.append("Consider closing losing positions to halt drawdown progression")
                recommendations.append("Implement emergency risk management protocols")
            
            # Warning level recommendations
            elif severity_level == 'warning':
                recommendations.append("Reduce risk exposure and position sizes")
                recommendations.append("Tighten stop losses and review position management")
                recommendations.append("Monitor drawdown velocity closely")
            
            # Velocity-based recommendations
            if velocity_metrics['velocity'] > 0.02:
                recommendations.append("URGENT: Rapid drawdown detected - immediate risk reduction required")
            elif velocity_metrics['velocity'] > 0.01:
                recommendations.append("Drawdown accelerating - consider defensive positioning")
            
            # Recovery recommendations
            if recovery_metrics['recovery_progress'] > 0.3:
                recommendations.append("Recovery in progress - maintain current risk management approach")
                if recovery_metrics['recovery_velocity'] > 0:
                    recommendations.append("Positive recovery momentum - consider gradual risk increase")
            
            # Rescue mode recommendations
            if rescue_status['rescue_mode']:
                recommendations.append("Rescue mode active - follow reduced risk guidelines")
                recommendations.append("Focus on capital preservation over profit maximization")
            
            # Emergency intervention recommendations
            if rescue_status.get('emergency_intervention', False):
                recommendations.append("EMERGENCY: Implement maximum risk reduction immediately")
                recommendations.append("Consider halting new position entries temporarily")
            
            # Performance-based recommendations
            if self.successful_recoveries < self.rescue_intervention_count:
                recommendations.append("Review rescue system effectiveness - consider parameter adjustments")
            
            # Proactive recommendations
            if not recommendations:
                recommendations.append("Drawdown management optimal - continue current approach")
                recommendations.append("Monitor for early warning signs of drawdown acceleration")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendations")
            recommendations.append(f"Recommendation generation failed: {error_context}")
        
        return recommendations
    
    def _update_smart_info_bus(self, drawdown_analysis: Dict[str, Any],
                              rescue_status: Dict[str, Any],
                              risk_adjustment: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with drawdown analysis results"""
        try:
            # Core drawdown risk data
            self.smart_bus.set('drawdown_risk', {
                'current_drawdown': self.current_dd,
                'severity_level': self.severity_level,
                'drawdown_analysis': drawdown_analysis,
                'rescue_status': rescue_status,
                'risk_adjustment': risk_adjustment,
                'thesis': thesis
            }, module='DrawdownRescue', thesis=thesis)
            
            # Rescue status for other modules
            self.smart_bus.set('rescue_status', {
                'rescue_mode': self.rescue_mode,
                'rescue_duration': rescue_status.get('rescue_duration_minutes', 0.0),
                'intervention_count': self.rescue_intervention_count,
                'emergency_active': rescue_status.get('emergency_intervention', False)
            }, module='DrawdownRescue', 
            thesis=f"Rescue mode: {'ACTIVE' if self.rescue_mode else 'STANDBY'}")
            
            # Risk adjustment factor for position sizing
            self.smart_bus.set('risk_adjustment', self.risk_adjustment_factor,
                             module='DrawdownRescue',
                             thesis=f"Risk adjustment: {self.risk_adjustment_factor:.1%} ({risk_adjustment.get('adjustment_reason', 'normal')})")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smart_info_bus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")
    
    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'current_drawdown': 0.0,
            'severity_level': 'disabled',
            'rescue_mode': False,
            'risk_adjustment_factor': 1.0,
            'drawdown_analysis': {},
            'rescue_status': {'rescue_mode': False},
            'risk_adjustment': {'risk_adjustment_factor': 1.0},
            'drawdown_metrics': {},
            'thesis': "Drawdown Rescue is disabled",
            'recommendations': ["Enable Drawdown Rescue for portfolio protection"]
        }
    
    def _generate_error_response(self, error_context: str) -> Dict[str, Any]:
        """Generate response when processing fails"""
        return {
            'current_drawdown': 0.0,
            'severity_level': 'error',
            'rescue_mode': False,
            'risk_adjustment_factor': 0.5,
            'drawdown_analysis': {'error': error_context},
            'rescue_status': {'error': error_context},
            'risk_adjustment': {'risk_adjustment_factor': 0.5, 'error': error_context},
            'drawdown_metrics': {'error': error_context},
            'thesis': f"Drawdown analysis failed: {error_context}",
            'recommendations': ["Investigate drawdown analysis system errors"]
        }
    
    def _generate_analysis_error_response(self, error_context: str) -> Dict[str, Any]:
        """Generate response when analysis fails"""
        return {
            'current_drawdown': self.current_dd,
            'max_drawdown': self.max_dd,
            'peak_balance': self.peak_balance,
            'velocity_metrics': {'velocity': 0.0, 'acceleration': 0.0},
            'severity_assessment': {'level': 'unknown'},
            'recovery_metrics': {'recovery_progress': 0.0},
            'processing_time_ms': 0,
            'error': error_context
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete module state for hot-reload"""
        return {
            'current_dd': self.current_dd,
            'max_dd': self.max_dd,
            'peak_balance': self.peak_balance,
            'dd_velocity': self.dd_velocity,
            'dd_acceleration': self.dd_acceleration,
            'severity_level': self.severity_level,
            'rescue_mode': self.rescue_mode,
            'rescue_start_time': self.rescue_start_time.isoformat() if self.rescue_start_time else None,
            'risk_adjustment_factor': self.risk_adjustment_factor,
            'rescue_intervention_count': self.rescue_intervention_count,
            'successful_recoveries': self.successful_recoveries,
            'emergency_interventions': self.emergency_interventions,
            'step_count': self.step_count,
            'config': self.config.copy()
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state for hot-reload"""
        self.current_dd = state.get('current_dd', 0.0)
        self.max_dd = state.get('max_dd', 0.0)
        self.peak_balance = state.get('peak_balance', 0.0)
        self.dd_velocity = state.get('dd_velocity', 0.0)
        self.dd_acceleration = state.get('dd_acceleration', 0.0)
        self.severity_level = state.get('severity_level', 'normal')
        self.rescue_mode = state.get('rescue_mode', False)
        
        rescue_time_str = state.get('rescue_start_time')
        self.rescue_start_time = datetime.datetime.fromisoformat(rescue_time_str) if rescue_time_str else None
        
        self.risk_adjustment_factor = state.get('risk_adjustment_factor', 1.0)
        self.rescue_intervention_count = state.get('rescue_intervention_count', 0)
        self.successful_recoveries = state.get('successful_recoveries', 0)
        self.emergency_interventions = state.get('emergency_interventions', 0)
        self.step_count = state.get('step_count', 0)
        self.config.update(state.get('config', {}))
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring"""
        return {
            'current_drawdown': self.current_dd,
            'max_drawdown': self.max_dd,
            'severity_level': self.severity_level,
            'rescue_mode': self.rescue_mode,
            'risk_adjustment_factor': self.risk_adjustment_factor,
            'intervention_success_rate': self.successful_recoveries / max(self.rescue_intervention_count, 1),
            'rescue_interventions': self.rescue_intervention_count,
            'emergency_interventions': self.emergency_interventions,
            'enabled': self.enabled
        }