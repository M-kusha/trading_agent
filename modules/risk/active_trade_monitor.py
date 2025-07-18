"""
Enhanced Active Trade Monitor with SmartInfoBus Integration
Monitors position duration and provides intelligent alerts with context awareness
"""

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import deque, defaultdict

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="ActiveTradeMonitor",
    version="3.0.0",
    category="risk",
    provides=["position_duration_risk", "duration_alerts", "position_tracking"],
    requires=["positions", "market_context"],
    description="Enhanced position duration monitoring with intelligent risk assessment",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class ActiveTradeMonitor(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Enhanced Active Trade Monitor with SmartInfoBus Integration
    
    Monitors position durations with intelligent context-aware thresholds,
    velocity analysis, and progressive risk assessment.
    """
    
    def __init__(self, config=None, **kwargs):
        self.config = config or {}
        self._fully_initialized = False
        
        # Initialize advanced systems first
        self._initialize_advanced_systems()
        
        # Call parent init
        super().__init__()
        
        # Configuration with intelligent defaults
        self.max_duration = self.config.get('max_duration', 200)
        self.warning_duration = self.config.get('warning_duration', 50)
        self.critical_duration = self.config.get('critical_duration', 150)
        self.enabled = self.config.get('enabled', True)
        
        # Mark as fully initialized
        self._fully_initialized = True
        
        # Enhanced tracking systems
        self.position_durations: Dict[str, int] = {}
        self.position_first_seen: Dict[str, str] = {}
        self.position_velocity: Dict[str, float] = {}
        self.duration_history = deque(maxlen=100)
        
        # Risk assessment
        self.risk_score = 0.0
        self.severity_level = "normal"
        self.alert_count = 0
        self.step_count = 0
        
        # Context-aware thresholds
        self.dynamic_thresholds = {
            'volatile_market': {'multiplier': 1.5, 'velocity_tolerance': 0.8},
            'trending_market': {'multiplier': 0.8, 'velocity_tolerance': 1.2},
            'ranging_market': {'multiplier': 1.0, 'velocity_tolerance': 1.0}
        }
        
        # Performance analytics
        self.closure_analytics = {'normal': 0, 'timeout': 0, 'emergency': 0}
        self.regime_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'durations': [], 'closures': 0})
        
        self.logger.info(format_operator_message(
            icon="[SEARCH]",
            message="Enhanced Active Trade Monitor initialized",
            max_duration=f"{self.max_duration} steps",
            warning_threshold=f"{self.warning_duration} steps",
            enabled=self.enabled
        ))
    
    def _initialize_advanced_systems(self):
        """Initialize advanced monitoring and error handling systems"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="ActiveTradeMonitor",
            log_path="logs/risk/active_trade_monitor.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("ActiveTradeMonitor", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
    
    async def process(self, **kwargs) -> Dict[str, Any]:
        """
        Enhanced position duration monitoring with comprehensive analysis
        
        Returns:
            Dict containing duration risk assessment, alerts, and recommendations
        """
        try:
            if not self.enabled:
                return self._generate_disabled_response()
            
            self.step_count += 1
            
            # Extract comprehensive market context
            market_context = self.smart_bus.get('market_context', 'ActiveTradeMonitor') or {}
            positions = self.smart_bus.get('positions', 'ActiveTradeMonitor') or []
            
            # Process position monitoring with context awareness
            monitoring_results = await self._monitor_positions_comprehensive(positions, market_context)
            
            # Generate intelligent thesis
            thesis = await self._generate_monitoring_thesis(monitoring_results, market_context)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_comprehensive_risk_metrics(monitoring_results)
            
            # Update SmartInfoBus
            self._update_smart_info_bus(monitoring_results, risk_metrics, thesis)
            
            # Record performance metrics
            self.performance_tracker.record_metric(
                'ActiveTradeMonitor', 'monitoring_cycle', 
                monitoring_results.get('processing_time_ms', 0), True
            )
            
            return {
                'risk_score': self.risk_score,
                'severity_level': self.severity_level,
                'monitoring_results': monitoring_results,
                'risk_metrics': risk_metrics,
                'thesis': thesis,
                'recommendations': self._generate_recommendations(monitoring_results)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "ActiveTradeMonitor")
            self.logger.error(f"Position monitoring failed: {error_context}")
            return self._generate_error_response(str(error_context))
    
    async def _monitor_positions_comprehensive(self, positions: List[Dict], 
                                             market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive position monitoring with intelligent analysis"""
        start_time = datetime.datetime.now()
        
        # Clear previous alerts
        alerts = {'critical': [], 'warning': [], 'info': []}
        current_symbols = set()
        
        # Process each position with enhanced analytics
        for position in positions:
            symbol = 'UNKNOWN'  # Initialize symbol to ensure it's always defined
            try:
                symbol = position.get('symbol', position.get('instrument', 'UNKNOWN'))
                current_symbols.add(symbol)
                
                # Calculate enhanced duration metrics
                duration_info = self._calculate_enhanced_duration(position, symbol)
                self.position_durations[symbol] = duration_info['duration']
                self.position_velocity[symbol] = duration_info['velocity']
                
                # Track first seen timestamp
                if symbol not in self.position_first_seen:
                    self.position_first_seen[symbol] = datetime.datetime.now().isoformat()
                
                # Assess severity with context awareness
                severity_info = self._assess_position_severity_enhanced(
                    symbol, duration_info, position, market_context
                )
                
                if severity_info['level'] != 'normal':
                    alerts[severity_info['level']].append({
                        'symbol': symbol,
                        'duration': duration_info['duration'],
                        'velocity': duration_info['velocity'],
                        'severity': severity_info,
                        'position_info': position,
                        'context_factors': severity_info.get('context_factors', [])
                    })
                
            except Exception as e:
                error_context = self.error_pinpointer.analyze_error(e, "position_processing")
                self.logger.warning(f"Position processing failed for {symbol}: {error_context}")
        
        # Handle position closures
        closure_info = self._process_position_closures(current_symbols, market_context)
        
        # Calculate processing time
        processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'alerts': alerts,
            'positions_tracked': len(current_symbols),
            'duration_statistics': self._calculate_duration_statistics(),
            'closure_info': closure_info,
            'processing_time_ms': processing_time,
            'market_context': market_context
        }
    
    def _calculate_enhanced_duration(self, position: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Calculate enhanced duration metrics with velocity analysis"""
        try:
            # Method 1: Direct duration from position
            duration = position.get('duration', position.get('bars_held', 0))
            if duration and duration > 0:
                base_duration = int(duration)
            else:
                # Method 2: Calculate from entry step
                entry_step = position.get('entry_step', 0)
                current_step = self.smart_bus.get('step_idx', 'ActiveTradeMonitor') or self.step_count
                base_duration = max(0, current_step - entry_step) if entry_step else 0
            
            # Calculate velocity (rate of change)
            previous_duration = self.position_durations.get(symbol, 0)
            velocity = base_duration - previous_duration if previous_duration else 0
            
            # Store in history for analytics
            self.duration_history.append({
                'symbol': symbol,
                'duration': base_duration,
                'velocity': velocity,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            return {
                'duration': base_duration,
                'velocity': velocity,
                'acceleration': velocity - self.position_velocity.get(symbol, 0),
                'trend': 'increasing' if velocity > 0 else 'stable' if velocity == 0 else 'decreasing'
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "duration_calculation")
            self.logger.warning(f"Duration calculation failed for {symbol}: {error_context}")
            return {'duration': 0, 'velocity': 0, 'acceleration': 0, 'trend': 'unknown'}
    
    def _assess_position_severity_enhanced(self, symbol: str, duration_info: Dict[str, Any],
                                         position: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced severity assessment with context awareness and velocity analysis"""
        try:
            duration = duration_info['duration']
            velocity = duration_info['velocity']
            
            # Get context-adjusted thresholds
            regime = market_context.get('regime', 'unknown')
            volatility = market_context.get('volatility_level', 'medium')
            
            thresholds = self._get_context_adjusted_thresholds(regime, volatility)
            
            # Base severity assessment
            if duration >= thresholds['critical']:
                base_level = 'critical'
            elif duration >= thresholds['warning']:
                base_level = 'warning'
            elif duration >= thresholds['info']:
                base_level = 'info'
            else:
                base_level = 'normal'
            
            # Velocity-based adjustments
            context_factors = []
            
            # Rapid duration increase
            if velocity > 5 and base_level == 'info':
                base_level = 'warning'
                context_factors.append('rapid_duration_increase')
            
            # Profitable position tolerance
            position_pnl = position.get('unrealised_pnl', position.get('pnl', 0))
            if position_pnl > 0 and base_level == 'warning' and duration < thresholds['critical']:
                base_level = 'info'
                context_factors.append('profitable_position_tolerance')
            
            # High volatility tolerance
            if volatility in ['high', 'extreme'] and base_level == 'warning':
                if duration < thresholds['critical'] * 0.9:
                    base_level = 'info'
                    context_factors.append('high_volatility_tolerance')
            
            return {
                'level': base_level,
                'threshold_used': thresholds[base_level] if base_level != 'normal' else thresholds['info'],
                'context_factors': context_factors,
                'position_pnl': position_pnl,
                'regime_factor': regime,
                'volatility_factor': volatility
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "severity_assessment")
            self.logger.warning(f"Severity assessment failed for {symbol}: {error_context}")
            return {'level': 'unknown', 'threshold_used': 0, 'context_factors': ['assessment_error']}
    
    def _get_context_adjusted_thresholds(self, regime: str, volatility: str) -> Dict[str, int]:
        """Get context-adjusted duration thresholds"""
        base_thresholds = {
            'info': self.warning_duration,
            'warning': self.critical_duration,
            'critical': self.max_duration
        }
        
        # Get regime multiplier
        regime_config = self.dynamic_thresholds.get(f"{regime}_market", self.dynamic_thresholds['ranging_market'])
        multiplier = regime_config['multiplier']
        
        # Volatility adjustments
        volatility_multipliers = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3,
            'extreme': 1.6
        }
        
        vol_multiplier = volatility_multipliers.get(volatility, 1.0)
        final_multiplier = multiplier * vol_multiplier
        
        return {
            level: int(threshold * final_multiplier)
            for level, threshold in base_thresholds.items()
        }
    
    def _process_position_closures(self, current_symbols: set, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process position closures and update analytics"""
        closed_symbols = set(self.position_durations.keys()) - current_symbols
        closure_info = {'closed_count': len(closed_symbols), 'closure_details': []}
        
        for symbol in closed_symbols:
            duration = self.position_durations.get(symbol, 0)
            
            # Classify closure type
            if duration >= self.max_duration:
                closure_type = 'timeout'
            elif duration >= self.critical_duration:
                closure_type = 'emergency'
            else:
                closure_type = 'normal'
            
            self.closure_analytics[closure_type] += 1
            
            # Update regime performance
            regime = market_context.get('regime', 'unknown')
            if regime not in self.regime_performance:
                self.regime_performance[regime] = {'durations': [], 'closures': 0}
            self.regime_performance[regime]['durations'].append(duration)
            self.regime_performance[regime]['closures'] += 1
            
            closure_info['closure_details'].append({
                'symbol': symbol,
                'duration': duration,
                'type': closure_type,
                'first_seen': self.position_first_seen.get(symbol)
            })
            
            # Clean up tracking
            self.position_durations.pop(symbol, None)
            self.position_first_seen.pop(symbol, None)
            self.position_velocity.pop(symbol, None)
            
            # Log significant closures
            if closure_type in ['timeout', 'emergency']:
                self.logger.warning(format_operator_message(
                    icon="⏰",
                    message=f"Position closed - {closure_type}",
                    symbol=symbol,
                    duration=f"{duration} steps",
                    regime=regime
                ))
        
        return closure_info
    
    def _calculate_duration_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive duration statistics"""
        try:
            if not self.position_durations:
                return {'active_positions': 0, 'avg_duration': 0, 'max_duration': 0}
            
            durations = list(self.position_durations.values())
            
            return {
                'active_positions': len(durations),
                'avg_duration': float(np.mean(durations)),
                'max_duration': int(np.max(durations)),
                'min_duration': int(np.min(durations)),
                'std_duration': float(np.std(durations)),
                'median_duration': float(np.median(durations)),
                'positions_over_warning': len([d for d in durations if d >= self.warning_duration]),
                'positions_over_critical': len([d for d in durations if d >= self.critical_duration])
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "statistics_calculation")
            self.logger.warning(f"Statistics calculation failed: {error_context}")
            return {'active_positions': 0, 'avg_duration': 0, 'max_duration': 0}
    
    def _calculate_comprehensive_risk_metrics(self, monitoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            alerts = monitoring_results['alerts']
            stats = monitoring_results['duration_statistics']
            
            # Alert-based risk score
            alert_risk = (
                len(alerts['critical']) * 1.0 +
                len(alerts['warning']) * 0.6 +
                len(alerts['info']) * 0.3
            ) / max(stats['active_positions'], 1)
            
            # Duration concentration risk
            concentration_risk = 0.0
            if stats['active_positions'] > 0:
                over_warning_ratio = stats['positions_over_warning'] / stats['active_positions']
                over_critical_ratio = stats['positions_over_critical'] / stats['active_positions']
                concentration_risk = over_warning_ratio * 0.5 + over_critical_ratio * 1.0
            
            # Velocity risk (positions increasing duration rapidly)
            velocity_risk = 0.0
            rapid_increase_count = sum(1 for v in self.position_velocity.values() if v > 3)
            if self.position_velocity:
                velocity_risk = rapid_increase_count / len(self.position_velocity)
            
            # Combined risk score
            self.risk_score = min(1.0, alert_risk + concentration_risk + velocity_risk)
            
            # Determine severity level
            if self.risk_score > 0.7 or len(alerts['critical']) > 0:
                self.severity_level = 'critical'
            elif self.risk_score > 0.4 or len(alerts['warning']) > 0:
                self.severity_level = 'warning'
            elif self.risk_score > 0.1 or len(alerts['info']) > 0:
                self.severity_level = 'elevated'
            else:
                self.severity_level = 'normal'
            
            return {
                'risk_score': self.risk_score,
                'severity_level': self.severity_level,
                'alert_risk': alert_risk,
                'concentration_risk': concentration_risk,
                'velocity_risk': velocity_risk,
                'total_alerts': len(alerts['critical']) + len(alerts['warning']) + len(alerts['info'])
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "risk_metrics")
            self.logger.error(f"Risk metrics calculation failed: {error_context}")
            return {'risk_score': 0.5, 'severity_level': 'unknown'}
    
    async def _generate_monitoring_thesis(self, monitoring_results: Dict[str, Any], 
                                        market_context: Dict[str, Any]) -> str:
        """Generate intelligent thesis explaining monitoring decisions"""
        try:
            stats = monitoring_results['duration_statistics']
            alerts = monitoring_results['alerts']
            regime = market_context.get('regime', 'unknown')
            volatility = market_context.get('volatility_level', 'medium')
            
            # Build thesis components
            thesis_parts = []
            
            # Position overview
            if stats['active_positions'] > 0:
                thesis_parts.append(
                    f"Monitoring {stats['active_positions']} active positions with "
                    f"average duration of {stats['avg_duration']:.1f} steps"
                )
                
                # Duration distribution analysis
                if stats['positions_over_critical'] > 0:
                    thesis_parts.append(
                        f"CRITICAL: {stats['positions_over_critical']} positions exceed "
                        f"{self.critical_duration} step threshold, indicating potential issues"
                    )
                elif stats['positions_over_warning'] > 0:
                    thesis_parts.append(
                        f"WARNING: {stats['positions_over_warning']} positions approaching "
                        f"duration limits, monitoring closely"
                    )
                else:
                    thesis_parts.append("All positions within acceptable duration ranges")
            else:
                thesis_parts.append("No active positions to monitor")
            
            # Market context analysis
            if regime != 'unknown':
                adjusted_thresholds = self._get_context_adjusted_thresholds(regime, volatility)
                thesis_parts.append(
                    f"Market regime ({regime}) and volatility ({volatility}) result in "
                    f"adjusted warning threshold of {adjusted_thresholds['warning']} steps"
                )
            
            # Alert analysis
            total_alerts = len(alerts['critical']) + len(alerts['warning']) + len(alerts['info'])
            if total_alerts > 0:
                thesis_parts.append(
                    f"Generated {total_alerts} alerts: {len(alerts['critical'])} critical, "
                    f"{len(alerts['warning'])} warning, {len(alerts['info'])} informational"
                )
            
            # Risk assessment conclusion
            thesis_parts.append(
                f"Overall position duration risk: {self.severity_level.upper()} "
                f"(score: {self.risk_score:.2f})"
            )
            
            # Performance insights
            if self.closure_analytics['timeout'] > 0:
                timeout_rate = self.closure_analytics['timeout'] / sum(self.closure_analytics.values())
                thesis_parts.append(
                    f"Historical analysis shows {timeout_rate:.1%} timeout closure rate, "
                    f"suggesting need for duration management improvements"
                )
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Thesis generation failed: {error_context}"
    
    def _generate_recommendations(self, monitoring_results: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on monitoring results"""
        recommendations = []
        
        try:
            alerts = monitoring_results['alerts']
            stats = monitoring_results['duration_statistics']
            
            # Critical situation recommendations
            if len(alerts['critical']) > 0:
                recommendations.append("IMMEDIATE: Close or reduce positions exceeding maximum duration")
                recommendations.append("Consider emergency risk reduction measures")
            
            # Warning level recommendations
            if len(alerts['warning']) > 0:
                recommendations.append("Review positions approaching duration limits")
                recommendations.append("Consider tightening stop losses or taking partial profits")
            
            # Concentration recommendations
            if stats['active_positions'] > 5:
                recommendations.append("High position concentration detected - consider reducing position count")
            
            # Velocity-based recommendations
            rapid_positions = [symbol for symbol, velocity in self.position_velocity.items() if velocity > 3]
            if rapid_positions:
                recommendations.append(f"Monitor rapidly aging positions: {', '.join(rapid_positions[:3])}")
            
            # Performance-based recommendations
            if self.closure_analytics['timeout'] > self.closure_analytics['normal']:
                recommendations.append("High timeout closure rate - review position management strategy")
            
            # Proactive recommendations
            if not recommendations:
                recommendations.append("Position duration monitoring optimal - continue current strategy")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendations")
            recommendations.append(f"Recommendation generation failed: {error_context}")
        
        return recommendations
    
    def _update_smart_info_bus(self, monitoring_results: Dict[str, Any], 
                              risk_metrics: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with monitoring results"""
        try:
            # Core monitoring data
            self.smart_bus.set('position_duration_risk', {
                'risk_score': self.risk_score,
                'severity_level': self.severity_level,
                'monitoring_results': monitoring_results,
                'risk_metrics': risk_metrics,
                'thesis': thesis
            }, module='ActiveTradeMonitor', thesis=thesis)
            
            # Duration alerts for other modules
            self.smart_bus.set('duration_alerts', monitoring_results['alerts'], 
                             module='ActiveTradeMonitor', 
                             thesis=f"Duration alerts: {len(monitoring_results['alerts']['critical'])} critical")
            
            # Position tracking data
            self.smart_bus.set('position_tracking', {
                'durations': self.position_durations.copy(),
                'velocities': self.position_velocity.copy(),
                'statistics': monitoring_results['duration_statistics']
            }, module='ActiveTradeMonitor', thesis="Position tracking metrics updated")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smart_info_bus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")
    
    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'risk_score': 0.0,
            'severity_level': 'disabled',
            'monitoring_results': {'alerts': {'critical': [], 'warning': [], 'info': []}},
            'risk_metrics': {'risk_score': 0.0, 'severity_level': 'disabled'},
            'thesis': "Active Trade Monitor is disabled",
            'recommendations': ["Enable Active Trade Monitor for position duration tracking"]
        }
    
    def _generate_error_response(self, error_context: str) -> Dict[str, Any]:
        """Generate response when processing fails"""
        return {
            'risk_score': 0.5,
            'severity_level': 'error',
            'monitoring_results': {'alerts': {'critical': [], 'warning': [], 'info': []}},
            'risk_metrics': {'risk_score': 0.5, 'severity_level': 'error'},
            'thesis': f"Position monitoring failed: {error_context}",
            'recommendations': ["Investigate position monitoring system errors"]
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete module state for hot-reload"""
        return {
            'position_durations': self.position_durations.copy(),
            'position_first_seen': self.position_first_seen.copy(),
            'position_velocity': self.position_velocity.copy(),
            'risk_score': self.risk_score,
            'severity_level': self.severity_level,
            'closure_analytics': self.closure_analytics.copy(),
            'step_count': self.step_count,
            'config': self.config.copy()
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state for hot-reload"""
        self.position_durations = state.get('position_durations', {})
        self.position_first_seen = state.get('position_first_seen', {})
        self.position_velocity = state.get('position_velocity', {})
        self.risk_score = state.get('risk_score', 0.0)
        self.severity_level = state.get('severity_level', 'normal')
        self.closure_analytics = state.get('closure_analytics', {'normal': 0, 'timeout': 0, 'emergency': 0})
        self.step_count = state.get('step_count', 0)
        self.config.update(state.get('config', {}))
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring"""
        total_closures = sum(self.closure_analytics.values())
        return {
            'positions_tracked': len(self.position_durations),
            'risk_score': self.risk_score,
            'severity_level': self.severity_level,
            'timeout_rate': self.closure_analytics['timeout'] / max(total_closures, 1),
            'avg_position_duration': np.mean(list(self.position_durations.values())) if self.position_durations else 0,
            'enabled': self.enabled
        }

    def _initialize(self):
        """Initialize module-specific state (called by BaseModule.__init__)"""
        # Check if we're fully initialized yet
        if not getattr(self, '_fully_initialized', False):
            return
            
        self.logger.info("[RELOAD] ActiveTradeMonitor async initialization")
        
        # Set initial data in SmartInfoBus
        self.smart_bus.set(
            'trade_monitor_status',
            {
                'initialized': True,
                'enabled': self.enabled,
                'max_duration': self.max_duration,
                'warning_duration': self.warning_duration,
                'positions_tracked': 0
            },
            module='ActiveTradeMonitor',
            thesis="Trade monitor initialization status for system awareness"
        )

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in position duration risk assessment"""
        try:
            base_confidence = 0.7  # High confidence in duration monitoring
            
            # Adjust confidence based on position tracking accuracy
            if len(self.position_durations) > 0:
                # Higher confidence with more tracking data
                tracking_confidence = min(0.3, len(self.position_durations) / 10 * 0.3)
                base_confidence += tracking_confidence
            
            # Confidence based on risk assessment quality
            if self.risk_score < 0.3:
                base_confidence += 0.2  # High confidence in low-risk scenarios
            elif self.risk_score > 0.7:
                base_confidence += 0.1  # Moderate confidence in high-risk scenarios
            
            # Adjust based on severity level
            severity_adjustments = {
                'normal': 0.1,
                'warning': 0.0,
                'critical': -0.1,
                'emergency': -0.2
            }
            base_confidence += severity_adjustments.get(self.severity_level, 0.0)
            
            return float(np.clip(base_confidence, 0.1, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Calculate confidence failed: {e}")
            return 0.5

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose risk management actions based on position duration analysis"""
        try:
            # Get current position data
            positions = inputs.get('positions', {})
            if not positions:
                positions = self.smart_bus.get('positions', 'ActiveTradeMonitor') or {}
            
            # Analyze current position durations
            duration_risks = {}
            recommendations = []
            
            for position_id, position in positions.items():
                if position_id in self.position_durations:
                    duration = self.position_durations[position_id]
                    
                    # Calculate risk level
                    if duration > self.critical_duration:
                        risk_level = 'critical'
                        recommendations.append({
                            'position_id': position_id,
                            'action': 'close_position',
                            'reason': f'Duration {duration} exceeds critical threshold {self.critical_duration}',
                            'urgency': 'high'
                        })
                    elif duration > self.warning_duration:
                        risk_level = 'warning'
                        recommendations.append({
                            'position_id': position_id,
                            'action': 'review_position',
                            'reason': f'Duration {duration} exceeds warning threshold {self.warning_duration}',
                            'urgency': 'medium'
                        })
                    else:
                        risk_level = 'normal'
                    
                    duration_risks[position_id] = {
                        'duration': duration,
                        'risk_level': risk_level,
                        'threshold_ratio': duration / self.max_duration
                    }
            
            # Overall risk assessment
            overall_action = 'monitor'
            if self.severity_level == 'emergency':
                overall_action = 'reduce_exposure'
            elif self.severity_level == 'critical':
                overall_action = 'close_risky_positions'
            elif self.severity_level == 'warning':
                overall_action = 'increase_monitoring'
            
            return {
                'action_type': 'duration_risk_management',
                'overall_action': overall_action,
                'severity_level': self.severity_level,
                'risk_score': self.risk_score,
                'position_risks': duration_risks,
                'recommendations': recommendations,
                'confidence': await self.calculate_confidence({}, **inputs),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Propose action failed: {e}")
            return {
                'action_type': 'duration_risk_management',
                'overall_action': 'monitor',
                'error': str(e),
                'confidence': 0.1
            }