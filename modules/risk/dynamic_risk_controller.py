# ─────────────────────────────────────────────────────────────
# File: modules/risk/dynamic_risk_controller.py
# [ROCKET] PRODUCTION-READY Enhanced Dynamic Risk Controller
# Advanced risk scaling with SmartInfoBus integration and intelligent automation
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
import datetime
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


class RiskControlMode(Enum):
    """Risk control operational modes"""
    INITIALIZATION = "initialization"
    CALIBRATION = "calibration"
    NORMAL = "normal"
    PROTECTIVE = "protective"
    AGGRESSIVE_REDUCTION = "aggressive_reduction"
    EMERGENCY = "emergency"
    RECOVERY = "recovery"


@dataclass
class DynamicRiskConfig:
    """Configuration for Dynamic Risk Controller"""
    base_risk_scale: float = 1.0
    min_risk_scale: float = 0.1
    max_risk_scale: float = 1.5
    vol_history_len: int = 30
    dd_threshold: float = 0.15
    vol_ratio_threshold: float = 2.0
    recovery_speed: float = 0.15
    risk_decay: float = 0.95
    adaptive_scaling: bool = True
    regime_sensitivity: float = 1.0
    correlation_sensitivity: float = 0.8
    
    # Performance thresholds
    max_processing_time_ms: float = 100
    circuit_breaker_threshold: int = 5
    min_risk_quality: float = 0.3
    
    # Adaptation parameters
    adaptive_learning_rate: float = 0.02
    risk_adaptation_speed: float = 1.0


@module(
    name="DynamicRiskController",
    version="4.0.0",
    category="risk",
    provides=["risk_scaling", "risk_factors", "risk_analytics", "risk_alerts"],
    requires=["risk_data", "performance_data", "market_data", "position_data"],
    description="Advanced dynamic risk scaling with intelligent adaptation and comprehensive market analysis",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    is_voting_member=True
)
class DynamicRiskController(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    [ROCKET] Advanced dynamic risk controller with SmartInfoBus integration.
    Provides intelligent risk scaling based on comprehensive market analysis.
    """

    def __init__(self, 
                 config: Optional[DynamicRiskConfig] = None,
                 action_dim: int = 1,
                 adaptive_scaling: bool = True,
                 regime_aware: bool = True,
                 **kwargs):
        
        # Handle config properly BEFORE calling super()
        if config is None:
            self.config = DynamicRiskConfig()
        elif isinstance(config, dict):
            self.config = DynamicRiskConfig(**config)
        else:
            self.config = config
            
        self.action_dim = int(action_dim)
        self.adaptive_scaling = adaptive_scaling
        self.regime_aware = regime_aware
        
        # Store original config before super() call
        original_config = self.config
        super().__init__()
        # Restore config after super() call in case it was overridden
        self.config = original_config
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize risk control state
        self._initialize_risk_control_state()
        
        self.logger.info(format_operator_message(
            message="Enhanced dynamic risk controller ready",
            icon="⚙️",
            adaptive=adaptive_scaling,
            regime_aware=regime_aware,
            config_loaded=True
        ))

    def _initialize_advanced_systems(self):
        """Initialize advanced systems for risk control"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="DynamicRiskController", 
            log_path="logs/risk/dynamic_risk_controller.log", 
            max_lines=5000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("DynamicRiskController", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for risk operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        # Note: Don't start monitoring here, wait until after risk control state init

    def _initialize_risk_control_state(self):
        """Initialize risk control state"""
        # Initialize mixin states
        self._initialize_risk_state()
        self._initialize_trading_state() 
        self._initialize_state_management()
        
        # Current operational mode
        self.current_mode = RiskControlMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        
        # Enhanced state tracking
        self.current_risk_scale = self.config.base_risk_scale
        self.risk_factors: Dict[str, float] = {
            "drawdown": 1.0,
            "volatility": 1.0,
            "correlation": 1.0,
            "losing_streak": 1.0,
            "market_stress": 1.0,
            "liquidity": 1.0,
            "news_sentiment": 1.0,
            "execution_quality": 1.0,
            "portfolio_concentration": 1.0
        }
        
        # Enhanced history tracking
        self.vol_history = deque(maxlen=self.config.vol_history_len)
        self.dd_history = deque(maxlen=50)
        self.risk_scale_history = deque(maxlen=100)
        self.consecutive_losses = 0
        self.last_pnl = 0.0
        
        # Market context tracking
        self.market_regime = "normal"
        self.market_regime_history = deque(maxlen=20)
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        
        # Enhanced risk event tracking
        self.risk_events: List[Dict[str, Any]] = []
        self.risk_adjustments_made = 0
        self.emergency_interventions = 0
        
        # Performance analytics
        self.risk_analytics = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self._last_significant_change = 0
        
        # External integrations
        self.external_risk_scale = 1.0
        self.external_signals = {}
        
        # Adaptive parameters with enhanced learning
        self._adaptive_params = {
            'dynamic_penalty_scaling': 1.0,
            'regime_sensitivity_multiplier': 1.0,
            'volatility_tolerance': 1.0,
            'risk_adaptation_confidence': 0.5,
            'learning_momentum': 0.0,
            'emergency_threshold_adaptation': 1.0
        }
        
        # Risk quality tracking
        self._risk_quality = 0.5
        self._risk_effectiveness_history = deque(maxlen=50)
        
        # Start monitoring after all state is initialized
        self._start_monitoring()

    def _start_monitoring(self):
        """Start background monitoring for risk control"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_risk_health()
                    self._analyze_risk_effectiveness()
                    self._adapt_risk_parameters()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Risk control monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    def _initialize(self) -> None:
        """Initialize module with SmartInfoBus integration"""
        try:
            # Set initial risk scaling status
            initial_status = {
                "current_mode": self.current_mode.value,
                "current_risk_scale": self.current_risk_scale,
                "base_risk_scale": self.config.base_risk_scale,
                "adaptive_scaling": self.adaptive_scaling,
                "regime_aware": self.regime_aware
            }
            
            self.smart_bus.set(
                'risk_scaling',
                initial_status,
                module='DynamicRiskController',
                thesis="Initial dynamic risk controller status"
            )
            
        except Exception as e:
            self.logger.error(f"Risk controller initialization failed: {e}")

    async def calculate_confidence(self, action: Dict[str, Any], **kwargs) -> float:
        """Calculate confidence score for risk scaling decisions"""
        try:
            # Base confidence starts high for risk management
            confidence = 0.9
            
            # Factors that affect confidence
            factors = {}
            
            # Risk quality affects confidence
            factors['risk_quality'] = self._risk_quality
            confidence *= self._risk_quality
            
            # Circuit breaker state affects confidence
            if self.circuit_breaker['state'] == 'OPEN':
                factors['circuit_breaker_penalty'] = 0.3
                confidence *= 0.3
            else:
                factors['circuit_breaker_penalty'] = 1.0
            
            # Market regime affects confidence
            regime_confidence_map = {
                'normal': 1.0,
                'trending': 0.95,
                'volatile': 0.8,
                'ranging': 0.9,
                'unknown': 0.7
            }
            regime_factor = regime_confidence_map.get(self.market_regime, 0.7)
            factors['regime_factor'] = regime_factor
            confidence *= regime_factor
            
            # Data availability affects confidence
            data_availability = min(len(self.vol_history) / 10.0, 1.0)
            factors['data_availability'] = data_availability
            confidence *= data_availability
            
            # Recent performance affects confidence
            if self._risk_effectiveness_history:
                recent_effectiveness = sum(list(self._risk_effectiveness_history)[-5:]) / min(5, len(self._risk_effectiveness_history))
                factors['recent_effectiveness'] = recent_effectiveness
                confidence *= recent_effectiveness
            
            # Risk scale stability affects confidence
            if len(self.risk_scale_history) >= 5:
                recent_scales = list(self.risk_scale_history)[-5:]
                scale_volatility = np.std(recent_scales) if len(recent_scales) > 1 else 0.0
                stability_factor = max(0.5, 1.0 - scale_volatility * 2)
                factors['stability_factor'] = stability_factor
                confidence *= stability_factor
            
            # Emergency mode reduces confidence
            if self.current_mode == RiskControlMode.EMERGENCY:
                factors['emergency_penalty'] = 0.6
                confidence *= 0.6
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            # Log confidence calculation for debugging
            self.logger.debug(f"Risk controller confidence: {confidence:.3f}, factors: {factors}")
            
            return float(confidence)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "confidence_calculation")
            self.logger.error(f"Confidence calculation failed: {error_context}")
            return 0.5  # Default medium confidence on error

    async def propose_action(self, **kwargs) -> Dict[str, Any]:
        """Propose risk scaling actions based on current state"""
        try:
            action_proposal = {
                'action_type': 'risk_scaling',
                'timestamp': time.time(),
                'current_risk_scale': self.current_risk_scale,
                'current_mode': self.current_mode.value,
                'market_regime': self.market_regime,
                'recommendations': [],
                'warnings': [],
                'adjustments': {}
            }
            
            # Generate recommendations based on current mode
            if self.current_mode == RiskControlMode.EMERGENCY:
                action_proposal['recommendations'].append({
                    'type': 'emergency_risk_reduction',
                    'reason': 'Emergency mode active',
                    'suggested_action': 'Maintain minimum risk exposure until conditions improve',
                    'priority': 'critical'
                })
                
                action_proposal['adjustments']['emergency_mode'] = True
                action_proposal['adjustments']['risk_scale'] = self.config.min_risk_scale
                
            elif self.current_mode == RiskControlMode.AGGRESSIVE_REDUCTION:
                action_proposal['recommendations'].append({
                    'type': 'aggressive_risk_reduction',
                    'reason': 'Multiple risk factors elevated',
                    'suggested_action': 'Significantly reduce position sizes and avoid new entries',
                    'priority': 'high'
                })
                
                action_proposal['adjustments']['position_reduction'] = 0.3  # Reduce by 70%
                
            elif self.current_mode == RiskControlMode.PROTECTIVE:
                action_proposal['recommendations'].append({
                    'type': 'protective_measures',
                    'reason': 'Risk factors showing warning signals',
                    'suggested_action': 'Reduce risk exposure and tighten risk management',
                    'priority': 'medium'
                })
                
                action_proposal['adjustments']['position_reduction'] = 0.7  # Reduce by 30%
                action_proposal['adjustments']['tighter_stops'] = True
            
            # Risk factor specific recommendations
            critical_factors = [name for name, value in self.risk_factors.items() if value < 0.5]
            if critical_factors:
                action_proposal['warnings'].append({
                    'type': 'critical_risk_factors',
                    'factors': critical_factors,
                    'risk_level': 'high'
                })
                
                action_proposal['recommendations'].append({
                    'type': 'factor_based_adjustment',
                    'reason': f'Critical risk factors detected: {", ".join(critical_factors)}',
                    'suggested_action': 'Address specific risk factor causes',
                    'priority': 'high'
                })
            
            # Volatility regime recommendations
            if self.volatility_regime == 'extreme':
                action_proposal['recommendations'].append({
                    'type': 'volatility_adjustment',
                    'reason': 'Extreme volatility detected',
                    'suggested_action': 'Reduce position sizes and increase monitoring frequency',
                    'priority': 'high'
                })
            
            # Losing streak recommendations
            if self.consecutive_losses > 5:
                action_proposal['warnings'].append({
                    'type': 'losing_streak',
                    'consecutive_losses': self.consecutive_losses,
                    'risk_level': 'medium'
                })
                
                action_proposal['recommendations'].append({
                    'type': 'streak_management',
                    'reason': f'{self.consecutive_losses} consecutive losses detected',
                    'suggested_action': 'Consider trading break or strategy review',
                    'priority': 'medium'
                })
            
            # Circuit breaker recommendations
            if self.circuit_breaker['state'] == 'OPEN':
                action_proposal['warnings'].append({
                    'type': 'circuit_breaker_open',
                    'failures': self.circuit_breaker['failures'],
                    'risk_level': 'critical'
                })
                
                action_proposal['recommendations'].append({
                    'type': 'system_recovery',
                    'reason': 'Circuit breaker triggered',
                    'suggested_action': 'System recovery mode - minimal risk until stabilized',
                    'priority': 'critical'
                })
            
            # External signal recommendations
            if self.external_signals:
                low_signals = [name for name, value in self.external_signals.items() if value < 0.5]
                if low_signals:
                    action_proposal['recommendations'].append({
                        'type': 'external_risk_signals',
                        'reason': f'External risk signals warning: {", ".join(low_signals)}',
                        'suggested_action': 'Consider external risk factor implications',
                        'priority': 'medium'
                    })
            
            # Risk scale adjustment suggestions
            if self.current_risk_scale < 0.3:
                action_proposal['adjustments']['recovery_readiness'] = True
            elif self.current_risk_scale > 1.2:
                action_proposal['adjustments']['risk_monitoring'] = 'increased'
            
            self.logger.debug(f"Risk action proposed: {len(action_proposal['recommendations'])} recommendations, "
                            f"{len(action_proposal['warnings'])} warnings")
            
            return action_proposal
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "action_proposal")
            self.logger.error(f"Action proposal failed: {error_context}")
            return {
                'action_type': 'risk_scaling',
                'timestamp': time.time(),
                'error': str(e),
                'recommendations': [],
                'warnings': [],
                'adjustments': {}
            }

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process dynamic risk scaling with enhanced analytics"""
        start_time = time.time()
        
        try:
            # Extract risk data from SmartInfoBus
            risk_data = await self._extract_risk_data(**inputs)
            
            if not risk_data:
                return await self._handle_no_data_fallback()
            
            # Update market context
            context_result = await self._update_market_context_async(risk_data)
            
            # Update external integrations
            external_result = await self._update_external_integrations_async(risk_data)
            
            # Perform comprehensive risk adjustment
            adjustment_result = await self._adjust_risk_comprehensive_async(risk_data)
            
            # Apply adaptive scaling if enabled
            adaptive_result = {}
            if self.adaptive_scaling:
                adaptive_result = await self._apply_adaptive_scaling_async(risk_data)
            
            # Apply emergency interventions if needed
            emergency_result = await self._apply_emergency_interventions_async(risk_data)
            
            # Calculate final risk scale
            final_result = await self._calculate_final_risk_scale_async()
            
            # Update operational mode
            mode_result = await self._update_operational_mode_async(risk_data)
            
            # Combine results
            result = {**context_result, **external_result, **adjustment_result,
                     **adaptive_result, **emergency_result, **final_result, **mode_result}
            
            # Generate thesis
            thesis = await self._generate_comprehensive_risk_thesis(risk_data, result)
            
            # Update SmartInfoBus
            await self._update_risk_smart_bus(result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return result
            
        except Exception as e:
            return await self._handle_risk_error(e, start_time)

    async def _extract_risk_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract comprehensive risk data from SmartInfoBus"""
        try:
            # Get risk data from SmartInfoBus
            risk_data_bus = self.smart_bus.get('risk_data', 'DynamicRiskController') or {}
            
            # Get performance data
            performance_data = self.smart_bus.get('performance_data', 'DynamicRiskController') or {}
            
            # Get market data
            market_data = self.smart_bus.get('market_data', 'DynamicRiskController') or {}
            
            # Get position data
            position_data = self.smart_bus.get('position_data', 'DynamicRiskController') or {}
            
            # Get direct inputs (legacy compatibility)
            drawdown = inputs.get('drawdown', inputs.get('current_drawdown', 0.0))
            volatility = inputs.get('volatility', 0.01)
            pnl = inputs.get('pnl', 0.0)
            balance = inputs.get('balance', inputs.get('current_balance', 0.0))
            
            # Extract from SmartInfoBus data
            risk_snapshot = risk_data_bus.get('risk_snapshot', {})
            if not drawdown and 'current_drawdown' in risk_snapshot:
                drawdown = risk_snapshot['current_drawdown']
            
            if not balance and 'balance' in risk_snapshot:
                balance = risk_snapshot['balance']
            
            # Get correlation data
            correlation = inputs.get('correlation', 0.0)
            if 'correlation_risk' in risk_snapshot:
                correlation = risk_snapshot['correlation_risk']
            
            return {
                'drawdown': float(drawdown),
                'volatility': float(volatility),
                'pnl': float(pnl),
                'balance': float(balance),
                'correlation': float(correlation),
                'risk_data': risk_data_bus,
                'performance_data': performance_data,
                'market_data': market_data,
                'position_data': position_data,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract risk data: {e}")
            return None

    async def _update_market_context_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update market context awareness asynchronously"""
        try:
            # Extract market context from SmartInfoBus
            market_context = self.smart_bus.get('market_context', 'DynamicRiskController') or {}
            
            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = market_context.get('regime', 'unknown')
            self.volatility_regime = market_context.get('volatility_level', 'medium')
            self.market_session = market_context.get('session', 'unknown')
            
            # Track regime changes
            if self.market_regime != old_regime:
                self.market_regime_history.append({
                    'regime': self.market_regime,
                    'timestamp': risk_data.get('timestamp', datetime.datetime.now().isoformat()),
                    'old_regime': old_regime
                })
                
                self.logger.info(format_operator_message(
                    message="Market regime changed",
                    icon="[STATS]",
                    old_regime=old_regime,
                    new_regime=self.market_regime,
                    volatility=self.volatility_regime,
                    session=self.market_session
                ))
                
                # Update regime-specific risk factors
                if self.regime_aware:
                    await self._update_regime_risk_factors_async()
            
            return {
                'market_context_updated': True,
                'regime_change': old_regime != self.market_regime,
                'current_regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'market_session': self.market_session
            }
            
        except Exception as e:
            self.logger.error(f"Market context update failed: {e}")
            return {'market_context_updated': False, 'error': str(e)}

    async def _update_regime_risk_factors_async(self) -> None:
        """Update risk factors based on market regime asynchronously"""
        try:
            # Base regime adjustments
            regime_adjustments = {
                "trending": {"market_stress": 0.9, "volatility": 1.1},
                "volatile": {"market_stress": 0.7, "volatility": 0.8},
                "ranging": {"market_stress": 1.1, "volatility": 1.2},
                "unknown": {"market_stress": 1.0, "volatility": 1.0}
            }
            
            # Volatility level adjustments
            vol_adjustments = {
                "low": {"volatility": 1.2, "market_stress": 1.1},
                "medium": {"volatility": 1.0, "market_stress": 1.0},
                "high": {"volatility": 0.8, "market_stress": 0.8},
                "extreme": {"volatility": 0.6, "market_stress": 0.6}
            }
            
            # Apply regime adjustments
            if self.market_regime in regime_adjustments:
                for factor, multiplier in regime_adjustments[self.market_regime].items():
                    if factor in self.risk_factors:
                        sensitivity = self._adaptive_params['regime_sensitivity_multiplier']
                        self.risk_factors[factor] *= multiplier * self.config.regime_sensitivity * sensitivity
            
            # Apply volatility adjustments
            if self.volatility_regime in vol_adjustments:
                for factor, multiplier in vol_adjustments[self.volatility_regime].items():
                    if factor in self.risk_factors:
                        self.risk_factors[factor] *= multiplier
                        
        except Exception as e:
            self.logger.warning(f"Regime risk factor update failed: {e}")

    async def _update_external_integrations_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update external integrations from other modules asynchronously"""
        try:
            # Get module data from SmartInfoBus
            portfolio_risk_data = self.smart_bus.get('portfolio_risk', 'DynamicRiskController') or {}
            execution_quality_data = self.smart_bus.get('execution_quality', 'DynamicRiskController') or {}
            anomaly_data = self.smart_bus.get('anomaly_detector', 'DynamicRiskController') or {}
            
            external_signals = {}
            
            # Update from portfolio risk
            if 'risk_adjustment' in portfolio_risk_data:
                external_signals['portfolio_risk'] = portfolio_risk_data['risk_adjustment']
            
            # Update from execution quality
            if 'quality_score' in execution_quality_data:
                quality_score = execution_quality_data['quality_score']
                external_signals['execution_quality'] = quality_score
                self.risk_factors['execution_quality'] = quality_score
            
            # Update from anomaly detector
            if 'anomaly_score' in anomaly_data:
                anomaly_score = anomaly_data['anomaly_score']
                external_signals['anomaly_risk'] = 1.0 - anomaly_score  # Invert for risk factor
            
            # Update from compliance module
            compliance_data = self.smart_bus.get('compliance', 'DynamicRiskController') or {}
            if 'risk_budget_used' in compliance_data:
                compliance_factor = 1.0 - compliance_data['risk_budget_used']
                external_signals['compliance'] = compliance_factor
            
            # Store external signals
            self.external_signals = external_signals
            
            # Aggregate external signals
            if external_signals:
                self.external_risk_scale = np.mean(list(external_signals.values()))
            else:
                self.external_risk_scale = 1.0
            
            return {
                'external_integrations_updated': True,
                'external_signals_count': len(external_signals),
                'external_risk_scale': self.external_risk_scale,
                'external_signals': external_signals.copy()
            }
                
        except Exception as e:
            self.logger.warning(f"External integration update failed: {e}")
            return {'external_integrations_updated': False, 'error': str(e)}

    async def _adjust_risk_comprehensive_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive risk adjustment based on all available factors"""
        try:
            old_scale = self.current_risk_scale
            
            # Update individual risk factors
            factor_results = {}
            factor_results['drawdown'] = await self._update_drawdown_factor_async(risk_data.get("drawdown", 0.0))
            factor_results['volatility'] = await self._update_volatility_factor_async(risk_data.get("volatility", 0.01))
            factor_results['correlation'] = await self._update_correlation_factor_async(risk_data.get("correlation", 0.0))
            factor_results['losing_streak'] = await self._update_losing_streak_factor_async(risk_data.get("pnl", 0.0))
            factor_results['liquidity'] = await self._update_liquidity_factor_async(risk_data)
            factor_results['news_sentiment'] = await self._update_news_sentiment_factor_async(risk_data)
            factor_results['portfolio_concentration'] = await self._update_portfolio_concentration_factor_async(risk_data)
            
            # Calculate preliminary risk scale
            preliminary_scale = await self._calculate_preliminary_risk_scale_async()
            
            # Track significant changes
            scale_change = abs(preliminary_scale - old_scale)
            if scale_change > 0.1:
                self.risk_adjustments_made += 1
                await self._record_risk_adjustment_event_async(old_scale, preliminary_scale, risk_data)
            
            return {
                'risk_adjustment_completed': True,
                'old_scale': old_scale,
                'preliminary_scale': preliminary_scale,
                'scale_change': scale_change,
                'factor_results': factor_results,
                'significant_change': scale_change > 0.1
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive risk adjustment failed: {e}")
            # Conservative fallback
            self.current_risk_scale = max(self.config.min_risk_scale, self.current_risk_scale * 0.9)
            return {'risk_adjustment_completed': False, 'error': str(e)}

    async def _update_drawdown_factor_async(self, drawdown: float) -> Dict[str, Any]:
        """Update drawdown risk factor with enhanced logic"""
        try:
            self.dd_history.append(drawdown)
            
            if drawdown <= 0.05:  # <5% drawdown
                factor = 1.0
                severity = "normal"
            elif drawdown <= self.config.dd_threshold:  # Normal range
                reduction = (drawdown - 0.05) / (self.config.dd_threshold - 0.05) * 0.4
                factor = 1.0 - reduction
                severity = "elevated"
            else:  # Excessive drawdown
                excess = drawdown - self.config.dd_threshold
                factor = 0.6 * np.exp(-excess * 8)  # Exponential reduction
                severity = "critical"
            
            # Apply regime adjustment
            if self.market_regime == 'volatile':
                factor *= 1.1  # More tolerant in volatile markets
            elif self.market_regime == 'trending':
                factor *= 0.9  # Less tolerant in trending markets
            
            self.risk_factors["drawdown"] = factor
            
            return {
                'drawdown_factor': factor,
                'drawdown_value': drawdown,
                'severity': severity,
                'regime_adjusted': self.market_regime != 'unknown'
            }
            
        except Exception as e:
            self.logger.warning(f"Drawdown factor update failed: {e}")
            return {'drawdown_factor': 0.8, 'error': str(e)}

    async def _update_volatility_factor_async(self, volatility: float) -> Dict[str, Any]:
        """Update volatility risk factor with enhanced logic"""
        try:
            # Update volatility history
            self.vol_history.append(volatility)
            
            if len(self.vol_history) >= 5:
                avg_vol = np.mean(list(self.vol_history)[-10:])  # Recent average
                vol_ratio = volatility / (avg_vol + 1e-8)
                
                if vol_ratio <= 1.2:  # Normal volatility
                    factor = 1.0
                    severity = "normal"
                elif vol_ratio <= self.config.vol_ratio_threshold:  # Elevated
                    reduction = (vol_ratio - 1.2) / (self.config.vol_ratio_threshold - 1.2) * 0.3
                    factor = 1.0 - reduction
                    severity = "elevated"
                else:  # Extreme volatility
                    excess = vol_ratio - self.config.vol_ratio_threshold
                    factor = 0.7 * np.exp(-excess * 3)
                    severity = "extreme"
            else:
                factor = 1.0
                severity = "insufficient_data"
                vol_ratio = 1.0
            
            # Apply adaptive volatility tolerance
            tolerance = self._adaptive_params['volatility_tolerance']
            factor = min(1.0, factor * tolerance)
            
            self.risk_factors["volatility"] = float(factor)
            
            return {
                'volatility_factor': factor,
                'volatility_value': volatility,
                'vol_ratio': vol_ratio,
                'severity': severity,
                'tolerance_applied': tolerance != 1.0
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility factor update failed: {e}")
            return {'volatility_factor': 0.8, 'error': str(e)}

    async def _update_correlation_factor_async(self, correlation_risk: float) -> Dict[str, Any]:
        """Update correlation risk factor"""
        try:
            if correlation_risk <= 0.3:
                factor = 1.0
                severity = "low"
            elif correlation_risk <= 0.6:
                reduction = (correlation_risk - 0.3) / 0.3 * 0.2
                factor = 1.0 - reduction
                severity = "moderate"
            else:
                excess = correlation_risk - 0.6
                factor = 0.8 * (1.0 - excess * self.config.correlation_sensitivity)
                severity = "high"
            
            self.risk_factors["correlation"] = factor
            
            return {
                'correlation_factor': factor,
                'correlation_risk': correlation_risk,
                'severity': severity
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation factor update failed: {e}")
            return {'correlation_factor': 1.0, 'error': str(e)}

    async def _update_losing_streak_factor_async(self, pnl: float) -> Dict[str, Any]:
        """Update losing streak risk factor"""
        try:
            # Track consecutive losses
            if pnl < 0 and self.last_pnl < 0:
                self.consecutive_losses += 1
            elif pnl > 0:
                self.consecutive_losses = max(0, self.consecutive_losses - 1)
            
            self.last_pnl = pnl
            
            # Calculate factor
            if self.consecutive_losses <= 2:
                factor = 1.0
                severity = "normal"
            elif self.consecutive_losses <= 5:
                reduction = (self.consecutive_losses - 2) * 0.15
                factor = 1.0 - reduction
                severity = "elevated"
            else:
                factor = 0.4
                severity = "critical"
            
            self.risk_factors["losing_streak"] = factor
            
            return {
                'losing_streak_factor': factor,
                'consecutive_losses': self.consecutive_losses,
                'pnl': pnl,
                'severity': severity
            }
            
        except Exception as e:
            self.logger.warning(f"Losing streak factor update failed: {e}")
            return {'losing_streak_factor': 1.0, 'error': str(e)}

    async def _update_liquidity_factor_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update liquidity risk factor"""
        try:
            # Extract liquidity data from market data
            market_data = risk_data.get('market_data', {})
            market_status = market_data.get('market_status', {})
            liquidity_score = market_status.get('liquidity_score', 1.0)
            
            if liquidity_score >= 0.8:
                factor = 1.0
                severity = "normal"
            elif liquidity_score >= 0.5:
                factor = 0.8 + liquidity_score * 0.2
                severity = "reduced"
            else:
                factor = 0.5 + liquidity_score * 0.3
                severity = "low"
                
            self.risk_factors["liquidity"] = factor
            
            return {
                'liquidity_factor': factor,
                'liquidity_score': liquidity_score,
                'severity': severity
            }
                
        except Exception:
            self.risk_factors["liquidity"] = 1.0  # Default to normal
            return {'liquidity_factor': 1.0, 'severity': 'unknown'}

    async def _update_news_sentiment_factor_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update news sentiment risk factor"""
        try:
            # Extract sentiment data from market data
            market_data = risk_data.get('market_data', {})
            market_context = market_data.get('market_context', {})
            news_sentiment = market_context.get('news_sentiment', 0.0)
            
            if news_sentiment >= -0.2:  # Positive or neutral sentiment
                factor = 1.0
                severity = "positive"
            elif news_sentiment >= -0.5:  # Moderately negative
                factor = 0.9
                severity = "negative"
            else:  # Very negative sentiment
                factor = 0.7
                severity = "very_negative"
                
            self.risk_factors["news_sentiment"] = factor
            
            return {
                'news_sentiment_factor': factor,
                'news_sentiment': news_sentiment,
                'severity': severity
            }
                
        except Exception:
            self.risk_factors["news_sentiment"] = 1.0  # Default to neutral
            return {'news_sentiment_factor': 1.0, 'severity': 'unknown'}

    async def _update_portfolio_concentration_factor_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update portfolio concentration risk factor"""
        try:
            # Extract position data
            position_data = risk_data.get('position_data', {})
            positions = position_data.get('positions', [])
            herfindahl = 0.0  # Initialize variable
            
            if not positions:
                factor = 1.0
                severity = "no_positions"
            else:
                # Calculate concentration (simplified Herfindahl index)
                total_exposure = sum(abs(pos.get('size', 0)) for pos in positions)
                if total_exposure > 0:
                    concentrations = [(abs(pos.get('size', 0)) / total_exposure) ** 2 for pos in positions]
                    herfindahl = sum(concentrations)
                    
                    if herfindahl <= 0.3:  # Well diversified
                        factor = 1.0
                        severity = "diversified"
                    elif herfindahl <= 0.6:  # Moderate concentration
                        factor = 0.9
                        severity = "moderate"
                    else:  # High concentration
                        factor = 0.7
                        severity = "concentrated"
                else:
                    factor = 1.0
                    severity = "no_exposure"
                    herfindahl = 0.0
                
            self.risk_factors["portfolio_concentration"] = factor
            
            return {
                'portfolio_concentration_factor': factor,
                'position_count': len(positions),
                'concentration_index': herfindahl,
                'severity': severity
            }
                
        except Exception as e:
            self.logger.warning(f"Portfolio concentration factor update failed: {e}")
            self.risk_factors["portfolio_concentration"] = 1.0
            return {'portfolio_concentration_factor': 1.0, 'error': str(e)}

    async def _calculate_preliminary_risk_scale_async(self) -> float:
        """Calculate preliminary risk scale from all factors"""
        try:
            scale = self.config.base_risk_scale
            
            # Apply all risk factors
            for factor_name, factor_value in self.risk_factors.items():
                scale *= factor_value
            
            # Apply external risk scale
            scale *= self.external_risk_scale
            
            # Apply bounds
            return float(np.clip(scale, self.config.min_risk_scale, self.config.max_risk_scale))
            
        except Exception as e:
            self.logger.error(f"Preliminary risk scale calculation failed: {e}")
            return self.config.min_risk_scale

    async def _apply_adaptive_scaling_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive scaling based on recent performance"""
        try:
            if len(self.risk_scale_history) < 10:
                return {'adaptive_scaling_applied': False, 'reason': 'insufficient_history'}
            
            # Analyze recent risk scale effectiveness
            recent_scales = list(self.risk_scale_history)[-10:]
            recent_performance = await self._calculate_recent_performance_async(risk_data)
            
            adaptation_factor = 1.0
            adaptation_reason = "no_change"
            
            # If recent performance is poor despite conservative scaling, be more aggressive
            if np.mean(recent_scales) < 0.7 and recent_performance < -0.1:
                adaptation_factor = 1.2  # Increase risk slightly
                adaptation_reason = "poor_performance_conservative"
            # If recent performance is good with conservative scaling, maintain conservatism
            elif np.mean(recent_scales) < 0.7 and recent_performance > 0.1:
                adaptation_factor = 0.9  # Be more conservative
                adaptation_reason = "good_performance_conservative"
            # If recent performance is poor with aggressive scaling, be more conservative
            elif np.mean(recent_scales) > 0.8 and recent_performance < -0.1:
                adaptation_factor = 0.8  # Reduce risk significantly
                adaptation_reason = "poor_performance_aggressive"
            else:
                adaptation_factor = 1.0  # No adjustment
                adaptation_reason = "stable_performance"
            
            # Apply learning rate
            learning_rate = self.config.adaptive_learning_rate
            current_confidence = self._adaptive_params['risk_adaptation_confidence']
            
            # Update adaptation confidence
            if abs(adaptation_factor - 1.0) > 0.1:
                self._adaptive_params['risk_adaptation_confidence'] = min(1.0, current_confidence + learning_rate)
            else:
                self._adaptive_params['risk_adaptation_confidence'] = max(0.1, current_confidence - learning_rate * 0.5)
            
            # Apply adaptation with confidence weighting
            final_adaptation = 1.0 + (adaptation_factor - 1.0) * current_confidence
            
            return {
                'adaptive_scaling_applied': True,
                'adaptation_factor': adaptation_factor,
                'final_adaptation': final_adaptation,
                'adaptation_reason': adaptation_reason,
                'recent_performance': recent_performance,
                'adaptation_confidence': self._adaptive_params['risk_adaptation_confidence']
            }
            
        except Exception as e:
            self.logger.warning(f"Adaptive scaling failed: {e}")
            return {'adaptive_scaling_applied': False, 'error': str(e)}

    async def _calculate_recent_performance_async(self, risk_data: Dict[str, Any]) -> float:
        """Calculate recent performance score"""
        try:
            # Simple performance based on drawdown and PnL trends
            drawdown = risk_data.get("drawdown", 0)
            pnl = risk_data.get("pnl", 0)
            
            drawdown_score = max(0, 1.0 - drawdown * 5)  # Penalize drawdown
            pnl_score = np.tanh(pnl / 100.0)  # Normalize PnL
            
            # Consider volatility in performance assessment
            if len(self.vol_history) >= 5:
                recent_vol = np.mean(list(self.vol_history)[-5:])
                vol_adjustment = 1.0 - min(0.3, recent_vol * 10)  # Penalize high volatility
            else:
                vol_adjustment = 1.0
            
            performance = (drawdown_score + pnl_score) / 2.0 * vol_adjustment
            
            return performance
            
        except Exception:
            return 0.0

    async def _apply_emergency_interventions_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply emergency interventions for extreme risk situations"""
        try:
            interventions_applied = []
            emergency_scale = self.current_risk_scale
            
            # Emergency intervention for extreme drawdown
            drawdown = risk_data.get("drawdown", 0)
            emergency_dd_threshold = 0.2 * self._adaptive_params['emergency_threshold_adaptation']
            if drawdown > emergency_dd_threshold:
                emergency_scale = min(emergency_scale, 0.3)
                interventions_applied.append(f"extreme_drawdown_{drawdown:.1%}")
                
            # Emergency intervention for extreme volatility
            if (len(self.vol_history) > 5 and 
                risk_data.get("volatility", 0) > np.mean(self.vol_history) * 3):
                emergency_scale = min(emergency_scale, 0.4)
                interventions_applied.append("extreme_volatility")
                
            # Emergency intervention for multiple risk factors
            active_risk_factors = sum(1 for factor in self.risk_factors.values() if factor < 0.8)
            if active_risk_factors >= 4:
                emergency_scale = min(emergency_scale, 0.5)
                interventions_applied.append(f"multiple_risk_factors_{active_risk_factors}")
                
            # Emergency intervention for extreme correlation
            correlation = risk_data.get("correlation", 0)
            if correlation > 0.8:
                emergency_scale = min(emergency_scale, 0.6)
                interventions_applied.append(f"extreme_correlation_{correlation:.2f}")
            
            # Update emergency count
            if interventions_applied:
                self.emergency_interventions += 1
                self.logger.warning(format_operator_message(
                    message="Emergency risk intervention triggered",
                    icon="[ALERT]",
                    interventions=len(interventions_applied),
                    old_scale=f"{self.current_risk_scale:.2f}",
                    new_scale=f"{emergency_scale:.2f}",
                    reasons=", ".join(interventions_applied[:2])
                ))
            
            # Update current scale
            self.current_risk_scale = emergency_scale
            
            return {
                'emergency_interventions_applied': len(interventions_applied) > 0,
                'interventions_count': len(interventions_applied),
                'interventions': interventions_applied,
                'emergency_scale': emergency_scale,
                'total_emergency_interventions': self.emergency_interventions
            }
            
        except Exception as e:
            self.logger.warning(f"Emergency intervention failed: {e}")
            return {'emergency_interventions_applied': False, 'error': str(e)}

    async def _calculate_final_risk_scale_async(self) -> Dict[str, Any]:
        """Calculate final risk scale with all adjustments"""
        try:
            # Apply decay factor to gradually return to base scale
            if self.current_risk_scale < self.config.base_risk_scale:
                recovery_adjustment = (self.config.base_risk_scale - self.current_risk_scale) * (1 - self.config.risk_decay)
                self.current_risk_scale = min(
                    self.config.base_risk_scale,
                    self.current_risk_scale + recovery_adjustment
                )
            
            # Final bounds check
            self.current_risk_scale = float(np.clip(
                self.current_risk_scale, 
                self.config.min_risk_scale, 
                self.config.max_risk_scale
            ))
            
            # Add to history
            self.risk_scale_history.append(self.current_risk_scale)
            
            # Calculate risk quality
            await self._calculate_risk_quality_async()
            
            return {
                'final_risk_scale_calculated': True,
                'final_risk_scale': self.current_risk_scale,
                'risk_quality': self._risk_quality,
                'scale_within_bounds': self.config.min_risk_scale <= self.current_risk_scale <= self.config.max_risk_scale
            }
            
        except Exception as e:
            self.logger.warning(f"Final risk scale calculation failed: {e}")
            return {'final_risk_scale_calculated': False, 'error': str(e)}

    async def _calculate_risk_quality_async(self) -> None:
        """Calculate comprehensive risk quality score"""
        try:
            quality_factors = []
            
            # Scale appropriateness (closer to base scale is better when conditions are normal)
            risk_factor_health = np.mean([max(0.3, factor) for factor in self.risk_factors.values()])
            if risk_factor_health > 0.8:
                scale_appropriateness = 1.0 - abs(self.current_risk_scale - self.config.base_risk_scale)
            else:
                scale_appropriateness = 1.0 - self.current_risk_scale  # Lower is better in poor conditions
            quality_factors.append(max(0, scale_appropriateness))
            
            # Risk factor stability
            if len(self.risk_scale_history) >= 10:
                recent_scales = list(self.risk_scale_history)[-10:]
                scale_stability = 1.0 - np.std(recent_scales)
                quality_factors.append(max(0, scale_stability))
            
            # Adaptation effectiveness
            adaptation_confidence = self._adaptive_params.get('risk_adaptation_confidence', 0.5)
            quality_factors.append(adaptation_confidence)
            
            # Emergency intervention frequency (fewer is better)
            emergency_frequency = max(0, 1.0 - (self.emergency_interventions / 10.0))
            quality_factors.append(emergency_frequency)
            
            self._risk_quality = float(np.mean(quality_factors)) if quality_factors else 0.5
            
        except Exception as e:
            self.logger.warning(f"Risk quality calculation failed: {e}")
            self._risk_quality = 0.5

    async def _update_operational_mode_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update operational mode based on risk level"""
        try:
            old_mode = self.current_mode
            
            # Determine new mode based on risk conditions
            if self.emergency_interventions > 0 and self.current_risk_scale < 0.4:
                new_mode = RiskControlMode.EMERGENCY
            elif self.current_risk_scale < 0.6 or sum(1 for f in self.risk_factors.values() if f < 0.7) >= 3:
                new_mode = RiskControlMode.AGGRESSIVE_REDUCTION
            elif self.current_risk_scale < 0.8 or risk_data.get('drawdown', 0) > 0.1:
                new_mode = RiskControlMode.PROTECTIVE
            elif self._risk_quality < 0.3:
                new_mode = RiskControlMode.CALIBRATION
            elif len(self.risk_scale_history) >= 20 and self.current_risk_scale > self.config.base_risk_scale * 0.8:
                new_mode = RiskControlMode.RECOVERY
            else:
                new_mode = RiskControlMode.NORMAL
            
            # Update mode if changed
            mode_changed = False
            if new_mode != old_mode:
                self.current_mode = new_mode
                self.mode_start_time = datetime.datetime.now()
                mode_changed = True
                
                self.logger.info(format_operator_message(
                    message="Risk mode changed",
                    icon="[RELOAD]",
                    old_mode=old_mode.value,
                    new_mode=new_mode.value,
                    risk_scale=f"{self.current_risk_scale:.2f}",
                    risk_quality=f"{self._risk_quality:.2f}"
                ))
            
            return {
                'mode_updated': True,
                'current_mode': self.current_mode.value,
                'mode_changed': mode_changed,
                'old_mode': old_mode.value if mode_changed else None,
                'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds()
            }
            
        except Exception as e:
            self.logger.warning(f"Mode update failed: {e}")
            return {'mode_updated': False, 'error': str(e)}

    async def _record_risk_adjustment_event_async(self, old_scale: float, new_scale: float,
                                                 risk_data: Dict[str, Any]) -> None:
        """Record significant risk adjustment events"""
        try:
            event = {
                "timestamp": datetime.datetime.now().isoformat(),
                "old_scale": old_scale,
                "new_scale": new_scale,
                "change": new_scale - old_scale,
                "risk_data": risk_data.copy(),
                "risk_factors": self.risk_factors.copy(),
                "reason": await self._determine_adjustment_reason_async(old_scale, new_scale, risk_data)
            }
            
            self.risk_events.append(event)
            
            # Keep only recent events
            if len(self.risk_events) > 50:
                self.risk_events = self.risk_events[-50:]
                
            # Log significant adjustments
            if abs(new_scale - old_scale) > 0.2:
                self.logger.warning(format_operator_message(
                    message="Significant risk adjustment made",
                    icon="⚙️",
                    old_scale=f"{old_scale:.2f}",
                    new_scale=f"{new_scale:.2f}",
                    reason=event["reason"],
                    regime=self.market_regime
                ))
                
        except Exception as e:
            self.logger.warning(f"Risk event recording failed: {e}")

    async def _determine_adjustment_reason_async(self, old_scale: float, new_scale: float, 
                                               risk_data: Dict[str, Any]) -> str:
        """Determine the primary reason for risk adjustment"""
        try:
            if new_scale < old_scale:  # Risk reduction
                if risk_data.get("drawdown", 0) > 0.1:
                    return "drawdown_protection"
                elif risk_data.get("correlation", 0) > 0.6:
                    return "correlation_risk"
                elif self.consecutive_losses > 3:
                    return "losing_streak"
                elif len(self.vol_history) > 5 and risk_data.get("volatility", 0) > np.mean(self.vol_history) * 2:
                    return "volatility_protection"
                else:
                    return "general_risk_reduction"
            else:  # Risk increase
                if self.config.recovery_speed > 0.1:
                    return "recovery_mode"
                else:
                    return "favorable_conditions"
                    
        except Exception:
            return "unknown"

    async def _generate_comprehensive_risk_thesis(self, risk_data: Dict[str, Any], 
                                  result: Dict[str, Any]) -> str:
        """Generate comprehensive risk thesis"""
        try:
            # Core metrics
            risk_scale = self.current_risk_scale
            mode = self.current_mode.value
            risk_quality = self._risk_quality
            
            thesis_parts = [
                f"Risk Control: {mode.upper()} mode with {risk_scale:.1%} scaling factor",
                f"Risk Quality: {risk_quality:.2f} assessment score"
            ]
            
            # Risk level assessment
            if risk_scale < 0.5:
                thesis_parts.append(f"DEFENSIVE: Significant risk reduction active")
            elif risk_scale > 0.9:
                thesis_parts.append(f"AGGRESSIVE: Near-normal risk exposure")
            
            # Active risk factors
            active_factors = [name for name, value in self.risk_factors.items() if value < 0.9]
            if active_factors:
                thesis_parts.append(f"Active factors: {', '.join(active_factors[:3])}")
            
            # Market context
            thesis_parts.append(f"Market: {self.market_regime.upper()} regime, {self.volatility_regime.upper()} volatility")
            
            # Performance metrics
            if result.get('significant_change', False):
                change = result.get('scale_change', 0)
                thesis_parts.append(f"Adjustment: {change:+.2f} scale change applied")
            
            # Emergency status
            if result.get('emergency_interventions_applied', False):
                interventions = result.get('interventions_count', 0)
                thesis_parts.append(f"EMERGENCY: {interventions} interventions triggered")
            
            # External signals
            external_count = len(self.external_signals)
            if external_count > 0:
                thesis_parts.append(f"External signals: {external_count} integrated")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Risk thesis generation failed: {str(e)} - Core risk scaling functional"

    async def _update_risk_smart_bus(self, result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with risk results"""
        try:
            # Risk scaling
            scaling_data = {
                'current_mode': self.current_mode.value,
                'current_risk_scale': self.current_risk_scale,
                'base_risk_scale': self.config.base_risk_scale,
                'min_risk_scale': self.config.min_risk_scale,
                'max_risk_scale': self.config.max_risk_scale,
                'adaptive_scaling': self.adaptive_scaling,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.smart_bus.set(
                'risk_scaling',
                scaling_data,
                module='DynamicRiskController',
                thesis=thesis
            )
            
            # Risk factors
            factors_data = {
                'risk_factors': self.risk_factors.copy(),
                'external_risk_scale': self.external_risk_scale,
                'external_signals': self.external_signals.copy(),
                'consecutive_losses': self.consecutive_losses,
                'risk_adjustments_made': self.risk_adjustments_made,
                'emergency_interventions': self.emergency_interventions
            }
            
            self.smart_bus.set(
                'risk_factors',
                factors_data,
                module='DynamicRiskController',
                thesis="Current risk factors and external signal integration"
            )
            
            # Risk analytics
            analytics_data = {
                'risk_quality': self._risk_quality,
                'adaptive_params': self._adaptive_params.copy(),
                'regime_performance': {
                    regime: {
                        'scale_history': len(data['risk_scales']),
                        'avg_scale': np.mean(data['risk_scales'][-10:]) if data['risk_scales'] else self.current_risk_scale
                    }
                    for regime, data in self.regime_performance.items()
                },
                'risk_events': len(self.risk_events),
                'scale_history_size': len(self.risk_scale_history),
                'volatility_history_size': len(self.vol_history)
            }
            
            self.smart_bus.set(
                'risk_analytics',
                analytics_data,
                module='DynamicRiskController',
                thesis="Risk control analytics and performance tracking"
            )
            
            # Risk alerts
            alerts_data = {
                'emergency_interventions': self.emergency_interventions,
                'risk_adjustments_made': self.risk_adjustments_made,
                'critical_mode': self.current_mode in [RiskControlMode.EMERGENCY, RiskControlMode.AGGRESSIVE_REDUCTION],
                'low_risk_quality': self._risk_quality < self.config.min_risk_quality,
                'recent_events': len([e for e in self.risk_events if 
                                    (datetime.datetime.now() - datetime.datetime.fromisoformat(e['timestamp'])).total_seconds() < 3600])
            }
            
            self.smart_bus.set(
                'risk_alerts',
                alerts_data,
                module='DynamicRiskController',
                thesis="Risk control alerts and emergency status tracking"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no risk data is available"""
        self.logger.warning("No risk data available - maintaining current scale")
        
        return {
            'current_mode': self.current_mode.value,
            'current_risk_scale': self.current_risk_scale,
            'risk_quality': self._risk_quality,
            'fallback_reason': 'no_risk_data'
        }

    async def _handle_risk_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle risk control errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self._health_status = 'warning'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "DynamicRiskController")
        explanation = self.english_explainer.explain_error(
            "DynamicRiskController", str(error), "risk scaling"
        )
        
        self.logger.error(format_operator_message(
            message="Risk controller error",
            icon="[CRASH]",
            error=str(error),
            details=explanation,
            processing_time_ms=processing_time,
            circuit_breaker_state=self.circuit_breaker['state']
        ))
        
        # Record failure
        self._record_failure(error)
        
        return self._create_error_fallback_response(f"error: {str(error)}")

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'current_mode': RiskControlMode.EMERGENCY.value,
            'current_risk_scale': self.config.min_risk_scale,  # Conservative fallback
            'risk_quality': 0.1,  # Poor quality due to error
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    def _update_risk_health(self):
        """Update risk control health metrics"""
        try:
            # Check if all required attributes are initialized
            if not hasattr(self, '_risk_quality'):
                return  # Skip if not fully initialized yet
                
            # Check risk quality
            if self._risk_quality < self.config.min_risk_quality:
                self._health_status = 'warning'
            else:
                self._health_status = 'healthy'
            
            # Check circuit breaker
            if self.circuit_breaker['state'] == 'OPEN':
                self._health_status = 'warning'
            
            # Check for excessive emergency interventions
            if self.emergency_interventions > 5:
                self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Risk health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_risk_effectiveness(self):
        """Analyze risk control effectiveness"""
        try:
            # Check if all required attributes are initialized
            if not hasattr(self, 'risk_scale_history'):
                return  # Skip if not fully initialized yet
                
            if len(self.risk_scale_history) >= 20:
                effectiveness = self._risk_quality
                
                if effectiveness > 0.8:
                    self.logger.info(format_operator_message(
                        message="High risk effectiveness achieved",
                        icon="[TARGET]",
                        quality_score=f"{effectiveness:.2f}",
                        current_scale=f"{self.current_risk_scale:.2f}"
                    ))
                elif effectiveness < 0.4:
                    self.logger.warning(format_operator_message(
                        message="Low risk effectiveness detected",
                        icon="[WARN]",
                        quality_score=f"{effectiveness:.2f}",
                        emergency_interventions=self.emergency_interventions
                    ))
            
        except Exception as e:
            self.logger.error(f"Risk effectiveness analysis failed: {e}")

    def _adapt_risk_parameters(self):
        """Continuous risk parameter adaptation"""
        try:
            # Check if all required attributes are initialized
            if not hasattr(self, 'market_regime') or not hasattr(self, 'market_regime_history'):
                return  # Skip if not fully initialized yet
                
            # Adapt emergency threshold based on market conditions
            if self.market_regime == 'volatile':
                self._adaptive_params['emergency_threshold_adaptation'] = min(
                    1.3, self._adaptive_params['emergency_threshold_adaptation'] * 1.005
                )
            else:
                self._adaptive_params['emergency_threshold_adaptation'] = max(
                    0.8, self._adaptive_params['emergency_threshold_adaptation'] * 0.999
                )
            
            # Adapt regime sensitivity based on regime changes
            if len(self.market_regime_history) >= 5:
                recent_changes = len([h for h in list(self.market_regime_history)[-5:]])
                if recent_changes > 2:  # Frequent regime changes
                    self._adaptive_params['regime_sensitivity_multiplier'] = min(
                        1.5, self._adaptive_params['regime_sensitivity_multiplier'] * 1.01
                    )
                else:
                    self._adaptive_params['regime_sensitivity_multiplier'] = max(
                        0.7, self._adaptive_params['regime_sensitivity_multiplier'] * 0.995
                    )
            
        except Exception as e:
            self.logger.warning(f"Risk parameter adaptation failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'DynamicRiskController', 'risk_scaling', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'DynamicRiskController', 'risk_scaling', 0, False
        )

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_current_risk_scale(self) -> float:
        """Get current risk scale for external use"""
        return self.current_risk_scale

    def set_external_risk_scale(self, scale: float) -> None:
        """Set external risk scale override"""
        self.external_risk_scale = float(np.clip(scale, 0.1, 2.0))

    def get_risk_factors(self) -> Dict[str, float]:
        """Get current risk factors"""
        return self.risk_factors.copy()

    def force_emergency_mode(self, reason: str = "manual_override") -> None:
        """Force emergency risk reduction"""
        old_scale = self.current_risk_scale
        self.current_risk_scale = self.config.min_risk_scale
        self.emergency_interventions += 1
        self.current_mode = RiskControlMode.EMERGENCY
        
        self.logger.error(format_operator_message(
            message="Emergency mode forced",
            icon="[ALERT]",
            reason=reason,
            old_scale=f"{old_scale:.2f}",
            new_scale=f"{self.current_risk_scale:.2f}"
        ))

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components for model integration"""
        try:
            return np.array([
                float(self.current_risk_scale),
                float(self.risk_factors["drawdown"]),
                float(self.risk_factors["volatility"]),
                float(self.risk_factors["correlation"]),
                float(self.risk_factors["losing_streak"]),
                float(min(self.consecutive_losses / 10.0, 1.0)),
                float(self.external_risk_scale),
                float(1.0 if self.market_regime in ["volatile", "extreme"] else 0.0),
                float(self._risk_quality),
                float(1.0 if self.current_mode in [RiskControlMode.EMERGENCY, RiskControlMode.AGGRESSIVE_REDUCTION] else 0.0)
            ], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Risk observation generation failed: {e}")
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0], dtype=np.float32)

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'current_mode': self.current_mode.value,
            'current_risk_scale': self.current_risk_scale,
            'risk_quality': self._risk_quality,
            'emergency_interventions': self.emergency_interventions
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    def get_risk_control_report(self) -> str:
        """Generate operator-friendly risk control report"""
        
        # Status indicators
        if self.current_risk_scale < 0.3:
            risk_status = "[ALERT] Emergency"
        elif self.current_risk_scale < 0.6:
            risk_status = "[WARN] High Reduction"
        elif self.current_risk_scale < 0.8:
            risk_status = "[FAST] Moderate Reduction"
        else:
            risk_status = "[OK] Normal"
        
        # Mode status
        mode_emoji = {
            RiskControlMode.INITIALIZATION: "[RELOAD]",
            RiskControlMode.CALIBRATION: "[TOOL]",
            RiskControlMode.NORMAL: "[OK]",
            RiskControlMode.PROTECTIVE: "[SAFE]",
            RiskControlMode.AGGRESSIVE_REDUCTION: "[WARN]",
            RiskControlMode.EMERGENCY: "🆘",
            RiskControlMode.RECOVERY: "[CHART]"
        }
        
        mode_status = f"{mode_emoji.get(self.current_mode, '❓')} {self.current_mode.value.upper()}"
        
        # Health status
        health_emoji = "[OK]" if self._health_status == 'healthy' else "[WARN]"
        cb_status = "[RED] OPEN" if self.circuit_breaker['state'] == 'OPEN' else "[GREEN] CLOSED"
        
        # Risk factor status
        risk_factor_lines = []
        for factor_name, factor_value in self.risk_factors.items():
            if factor_value < 0.9:
                emoji = "[ALERT]" if factor_value < 0.5 else "[WARN]" if factor_value < 0.7 else "[FAST]"
                risk_factor_lines.append(f"  {emoji} {factor_name.replace('_', ' ').title()}: {factor_value:.1%}")
        
        return f"""
⚙️ ENHANCED DYNAMIC RISK CONTROLLER v4.0
═══════════════════════════════════════════════════
[TARGET] Risk Status: {risk_status} ({self.current_risk_scale:.1%} scale)
[TOOL] Control Mode: {mode_status}
[STATS] Market Regime: {self.market_regime.title()}
[CRASH] Volatility Level: {self.volatility_regime.title()}
🕐 Market Session: {self.market_session.title()}

[HEALTH] SYSTEM HEALTH
• Status: {health_emoji} {self._health_status.upper()}
• Circuit Breaker: {cb_status}
• Risk Quality: {self._risk_quality:.2f}

[BALANCE] RISK SCALE CONFIGURATION
• Current Scale: {self.current_risk_scale:.1%}
• Base Scale: {self.config.base_risk_scale:.1%}
• Min Scale: {self.config.min_risk_scale:.1%}
• Max Scale: {self.config.max_risk_scale:.1%}
• External Scale: {self.external_risk_scale:.1%}

[STATS] ACTIVE RISK FACTORS
{chr(10).join(risk_factor_lines) if risk_factor_lines else "  [OK] All risk factors normal"}

[TOOL] CONTROLLER PERFORMANCE
• Risk Adjustments: {self.risk_adjustments_made}
• Emergency Interventions: {self.emergency_interventions}
• Consecutive Losses: {self.consecutive_losses}
• Adaptive Scaling: {'[OK] Enabled' if self.adaptive_scaling else '[FAIL] Disabled'}
• Regime Awareness: {'[OK] Enabled' if self.regime_aware else '[FAIL] Disabled'}

[CHART] ADAPTIVE PARAMETERS
• Dynamic Penalty Scaling: {self._adaptive_params['dynamic_penalty_scaling']:.2f}
• Regime Sensitivity: {self._adaptive_params['regime_sensitivity_multiplier']:.2f}
• Volatility Tolerance: {self._adaptive_params['volatility_tolerance']:.2f}
• Adaptation Confidence: {self._adaptive_params['risk_adaptation_confidence']:.2f}

🔗 EXTERNAL INTEGRATIONS
• External Signals: {len(self.external_signals)}
{chr(10).join([f"  • {name}: {value:.2f}" for name, value in self.external_signals.items()]) if self.external_signals else "  📭 No external signals"}

💡 SYSTEM STATUS
• History Tracking: {len(self.risk_scale_history)} records
• Volatility History: {len(self.vol_history)} records
• Risk Events: {len(self.risk_events)} events
• Recovery Speed: {self.config.recovery_speed:.1%}
        """

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> Dict[str, Any]:
        """Legacy step interface for backward compatibility"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(self.process(**kwargs))
            return result
        finally:
            loop.close()

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        # Reset mixin states
        # Note: Mixin reset methods will be implemented as needed
        
        # Reset core state
        self.current_risk_scale = self.config.base_risk_scale
        self.risk_factors = {
            "drawdown": 1.0,
            "volatility": 1.0,
            "correlation": 1.0,
            "losing_streak": 1.0,
            "market_stress": 1.0,
            "liquidity": 1.0,
            "news_sentiment": 1.0,
            "execution_quality": 1.0,
            "portfolio_concentration": 1.0
        }
        
        # Reset history
        self.vol_history.clear()
        self.dd_history.clear()
        self.risk_scale_history.clear()
        self.market_regime_history.clear()
        
        # Reset tracking
        self.consecutive_losses = 0
        self.last_pnl = 0.0
        self.risk_adjustments_made = 0
        self.emergency_interventions = 0
        
        # Reset market context
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        
        # Reset analytics
        self.risk_analytics.clear()
        self.regime_performance.clear()
        self.risk_events.clear()
        
        # Reset external integrations
        self.external_risk_scale = 1.0
        self.external_signals.clear()
        
        # Reset mode
        self.current_mode = RiskControlMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        
        # Reset circuit breaker
        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['state'] = 'CLOSED'
        self._health_status = 'healthy'
        
        # Reset adaptive parameters
        self._adaptive_params = {
            'dynamic_penalty_scaling': 1.0,
            'regime_sensitivity_multiplier': 1.0,
            'volatility_tolerance': 1.0,
            'risk_adaptation_confidence': 0.5,
            'learning_momentum': 0.0,
            'emergency_threshold_adaptation': 1.0
        }
        
        # Reset quality tracking
        self._risk_quality = 0.5
        self._risk_effectiveness_history.clear()
        
        self.logger.info("[RELOAD] Enhanced Dynamic Risk Controller reset - all state cleared")

    def adjust_risk(self, stats: Dict[str, float]) -> None:
        """Legacy risk adjustment interface"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(self.process(**stats))
        finally:
            loop.close()

    def calculate_risk_scale(self) -> float:
        """Legacy interface to get risk scale"""
        return self.current_risk_scale

# End of enhanced DynamicRiskController class