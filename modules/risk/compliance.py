"""
Enhanced Compliance Module with SmartInfoBus Integration
Comprehensive trade validation and regulatory compliance monitoring
"""

import numpy as np
import datetime
import time
import os
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from collections import deque, defaultdict

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="ComplianceModule",
    version="3.0.0",
    category="risk",
    provides=["compliance_status", "validation_results", "risk_limits"],
    requires=["positions", "pending_orders", "market_context"],
    description="Enhanced compliance validation with intelligent risk assessment and regulatory monitoring",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class ComplianceModule(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin):
    """
    Enhanced Compliance Module with SmartInfoBus Integration
    
    Provides comprehensive trade validation, regulatory compliance monitoring,
    and intelligent risk limit enforcement with context-aware adjustments.
    """
    
    # Default allowed instruments
    DEFAULT_ALLOWED_INSTRUMENTS = {
        "EUR/USD", "EURUSD", "GBP/USD", "GBPUSD", "USD/JPY", "USDJPY",
        "XAU/USD", "XAUUSD", "AUD/USD", "AUDUSD", "USD/CHF", "USDCHF",
        "NZD/USD", "NZDUSD", "EUR/GBP", "EURGBP", "EUR/JPY", "EURJPY",
        "GBP/JPY", "GBPJPY", "CHF/JPY", "CHFJPY", "AUD/JPY", "AUDJPY"
    }
    
    def __init__(self, config=None, **kwargs):
        self.config = config or {}
        super().__init__()
        self._initialize_advanced_systems()
        
        # Core compliance configuration
        self.max_leverage = self.config.get('max_leverage', 30.0)
        self.max_position_risk = self.config.get('max_position_risk', 0.20)
        self.max_total_risk = self.config.get('max_total_risk', 0.50)
        self.max_daily_trades = self.config.get('max_daily_trades', 100)
        self.min_trade_size = self.config.get('min_trade_size', 0.01)
        self.max_trade_size = self.config.get('max_trade_size', 10.0)
        self.enabled = self.config.get('enabled', True)
        
        # Dynamic risk management
        self.dynamic_limits = self.config.get('dynamic_limits', True)
        self.regime_aware = self.config.get('regime_aware', True)
        
        # Allowed instruments management
        self.allowed_instruments = self._initialize_allowed_instruments()
        self.restricted_hours = set(self.config.get('restricted_hours', []))
        
        # State tracking
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.total_exposure = 0.0
        self.current_leverage = 0.0
        
        # Risk budget tracking
        self.risk_budget_usage = 0.0
        self.position_limits = {}
        self.compliance_score = 1.0
        
        # Validation tracking
        self.validation_stats = {
            'total_validations': 0,
            'approved': 0,
            'rejected': 0,
            'violations': defaultdict(int)
        }
        
        # Performance analytics
        self.rejection_history = deque(maxlen=100)
        self.approval_rate_history = deque(maxlen=50)
        self.compliance_violations = deque(maxlen=200)
        
        # Context-aware adjustments
        self.regime_adjustments = {
            'volatile': {'leverage': 0.7, 'position_risk': 0.8, 'daily_trades': 1.2},
            'trending': {'leverage': 1.1, 'position_risk': 1.0, 'daily_trades': 0.9},
            'ranging': {'leverage': 1.0, 'position_risk': 1.1, 'daily_trades': 1.0},
            'crisis': {'leverage': 0.5, 'position_risk': 0.6, 'daily_trades': 0.7}
        }
        
        self.logger.info(format_operator_message(
            message="Enhanced Compliance Module initialized",
            icon="[SAFE]",
            max_leverage=f"{self.max_leverage:.1f}x",
            position_risk_limit=f"{self.max_position_risk:.1%}",
            allowed_instruments=len(self.allowed_instruments),
            dynamic_limits=self.dynamic_limits,
            enabled=self.enabled
        ))
    
    def _initialize(self) -> None:
        """Initialize the compliance module (required by BaseModule)"""
        try:
            # Validate configuration
            if not self.enabled:
                self.logger.warning("Compliance module is disabled")
                return
            
            # Initialize risk tracking
            self.daily_trade_count = 0
            self.last_trade_date = None
            self.total_exposure = 0.0
            self.current_leverage = 0.0
            self.risk_budget_usage = 0.0
            self.compliance_score = 1.0
            
            # Reset validation stats
            self.validation_stats = {
                'total_validations': 0,
                'approved': 0,
                'rejected': 0,
                'violations': defaultdict(int)
            }
            
            self.logger.info("Compliance module initialization completed successfully")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "compliance_initialization")
            self.logger.error(f"Compliance initialization failed: {error_context}")
    
    async def calculate_confidence(self, action: Dict[str, Any], **kwargs) -> float:
        """Calculate confidence score for compliance validation"""
        try:
            # Base confidence starts high for compliance
            confidence = 0.9
            
            # Factors that affect confidence
            factors = {}
            
            # Check compliance score
            factors['compliance_score'] = self.compliance_score
            confidence *= self.compliance_score
            
            # Check recent validation history
            if self.approval_rate_history:
                avg_approval_rate = sum(self.approval_rate_history) / len(self.approval_rate_history)
                factors['approval_rate'] = avg_approval_rate
                confidence *= avg_approval_rate
            
            # Check if we have recent violations
            recent_violations = len([v for v in self.compliance_violations if v.get('timestamp', 0) > time.time() - 3600])
            if recent_violations > 0:
                violation_penalty = max(0.5, 1.0 - (recent_violations * 0.1))
                factors['violation_penalty'] = violation_penalty
                confidence *= violation_penalty
            
            # Check system load and performance (simplified check)
            if hasattr(self, 'performance_tracker') and self.performance_tracker:
                # Use a simple performance metric instead of unknown method
                perf_score = 0.9  # Default good performance
                factors['performance'] = perf_score
                confidence *= perf_score
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            # Log confidence calculation for debugging
            self.logger.debug(f"Compliance confidence: {confidence:.3f}, factors: {factors}")
            
            return confidence
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "confidence_calculation")
            self.logger.error(f"Confidence calculation failed: {error_context}")
            return 0.5  # Default medium confidence on error
    
    async def propose_action(self, **kwargs) -> Dict[str, Any]:
        """Propose compliance actions based on current state"""
        try:
            action_proposal = {
                'action_type': 'compliance_check',
                'timestamp': time.time(),
                'compliance_score': self.compliance_score,
                'recommendations': [],
                'warnings': [],
                'adjustments': {}
            }
            
            # Check current risk levels with simple analysis
            risk_analysis = {
                'total_exposure': self.total_exposure,
                'current_leverage': self.current_leverage,
                'risk_budget_usage': self.risk_budget_usage,
                'compliance_score': self.compliance_score,
                'daily_trades': self.daily_trade_count
            }
            action_proposal['risk_analysis'] = risk_analysis
            
            # Generate recommendations based on compliance state
            if self.compliance_score < 0.8:
                action_proposal['recommendations'].append({
                    'type': 'reduce_risk',
                    'reason': 'Low compliance score detected',
                    'suggested_action': 'Reduce position sizes or close risky positions'
                })
            
            # Check for position limit violations
            current_positions = kwargs.get('positions', [])
            for pos in current_positions:
                instrument = pos.get('instrument', 'Unknown')
                size = abs(pos.get('size', 0))
                risk_ratio = size * pos.get('price', 1) / kwargs.get('balance', 10000)
                
                if risk_ratio > self.max_position_risk:
                    action_proposal['warnings'].append({
                        'type': 'position_risk_violation',
                        'instrument': instrument,
                        'current_risk': risk_ratio,
                        'max_allowed': self.max_position_risk
                    })
            
            # Suggest dynamic limit adjustments
            market_regime = kwargs.get('market_regime', 'normal')
            if market_regime in self.regime_adjustments:
                adjustments = self.regime_adjustments[market_regime]
                action_proposal['adjustments'] = {
                    'leverage_multiplier': adjustments.get('leverage', 1.0),
                    'position_risk_multiplier': adjustments.get('position_risk', 1.0),
                    'trade_frequency_multiplier': adjustments.get('daily_trades', 1.0)
                }
            
            self.logger.debug(f"Compliance action proposed: {len(action_proposal['recommendations'])} recommendations, "
                            f"{len(action_proposal['warnings'])} warnings")
            
            return action_proposal
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "action_proposal")
            self.logger.error(f"Action proposal failed: {error_context}")
            return {
                'action_type': 'compliance_check',
                'timestamp': time.time(),
                'error': str(e),
                'recommendations': [],
                'warnings': [],
                'adjustments': {}
            }
    
    def _initialize_advanced_systems(self):
        """Initialize advanced monitoring and error handling systems"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="ComplianceModule",
            log_path="logs/risk/compliance.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("ComplianceModule", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
    
    def _initialize_allowed_instruments(self) -> Set[str]:
        """Initialize allowed instruments from config or environment"""
        try:
            # Check environment variable first
            env_instruments = os.getenv("COMPLIANCE_INSTRUMENTS")
            if env_instruments:
                instruments = [inst.strip().upper() for inst in env_instruments.split(",")]
                return self._normalize_instrument_symbols(instruments)
            
            # Use config or defaults
            config_instruments = self.config.get('allowed_instruments', list(self.DEFAULT_ALLOWED_INSTRUMENTS))
            return self._normalize_instrument_symbols(config_instruments)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "instrument_initialization")
            self.logger.warning(f"Instrument initialization failed: {error_context}")
            return self._normalize_instrument_symbols(list(self.DEFAULT_ALLOWED_INSTRUMENTS))
    
    def _normalize_instrument_symbols(self, instruments: List[str]) -> Set[str]:
        """Normalize instrument symbols to handle different formats"""
        normalized = set()
        
        for instrument in instruments:
            instrument = instrument.strip().upper()
            normalized.add(instrument)
            
            # Add both slash and non-slash formats
            if "/" in instrument:
                normalized.add(instrument.replace("/", ""))
            elif len(instrument) == 6:  # EURUSD format
                normalized.add(f"{instrument[:3]}/{instrument[3:]}")
        
        return normalized
    
    async def process(self, **kwargs) -> Dict[str, Any]:
        """
        Enhanced compliance validation with comprehensive analysis
        
        Returns:
            Dict containing compliance status, validation results, and risk assessment
        """
        try:
            if not self.enabled:
                return self._generate_disabled_response()
            
            # Extract comprehensive market context
            market_context = self.smart_bus.get('market_context', 'ComplianceModule') or {}
            positions = self.smart_bus.get('positions', 'ComplianceModule') or []
            pending_orders = self.smart_bus.get('pending_orders', 'ComplianceModule') or []
            balance = self.smart_bus.get('balance', 'ComplianceModule') or 10000.0
            
            # Update daily tracking
            self._update_daily_tracking()
            
            # Adjust limits based on market context
            current_limits = self._calculate_dynamic_limits(market_context)
            
            # Validate pending orders
            validation_results = await self._validate_pending_orders_comprehensive(
                pending_orders, positions, balance, market_context, current_limits
            )
            
            # Assess current risk exposure
            risk_assessment = self._assess_current_risk_exposure(positions, balance, current_limits)
            
            # Generate intelligent thesis
            thesis = await self._generate_compliance_thesis(validation_results, risk_assessment, market_context)
            
            # Calculate compliance metrics
            compliance_metrics = self._calculate_compliance_metrics(validation_results, risk_assessment)
            
            # Update SmartInfoBus
            self._update_smart_info_bus(validation_results, risk_assessment, compliance_metrics, thesis)
            
            # Record performance metrics
            self.performance_tracker.record_metric(
                'ComplianceModule', 'validation_cycle', 
                validation_results.get('processing_time_ms', 0), True
            )
            
            return {
                'compliance_score': self.compliance_score,
                'validation_results': validation_results,
                'risk_assessment': risk_assessment,
                'compliance_metrics': compliance_metrics,
                'current_limits': current_limits,
                'thesis': thesis,
                'recommendations': self._generate_recommendations(validation_results, risk_assessment)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "ComplianceModule")
            self.logger.error(f"Compliance processing failed: {error_context}")
            return self._generate_error_response(str(error_context))
    
    def _update_daily_tracking(self):
        """Update daily trade count tracking"""
        try:
            current_date = datetime.datetime.now().date()
            
            # Reset counter for new day
            if self.last_trade_date != current_date:
                if self.last_trade_date and self.daily_trade_count > 0:
                    self.logger.info(format_operator_message(
                        message="Daily trading summary",
                        icon="[STATS]",
                        date=str(self.last_trade_date),
                        trades=self.daily_trade_count,
                        limit=self.max_daily_trades
                    ))
                
                self.daily_trade_count = 0
                self.last_trade_date = current_date
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "daily_tracking")
            self.logger.warning(f"Daily tracking update failed: {error_context}")
    
    def _calculate_dynamic_limits(self, market_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic limits based on market context"""
        try:
            # Start with base limits
            current_limits = {
                'max_leverage': self.max_leverage,
                'max_position_risk': self.max_position_risk,
                'max_total_risk': self.max_total_risk,
                'max_daily_trades': self.max_daily_trades,
                'min_trade_size': self.min_trade_size,
                'max_trade_size': self.max_trade_size
            }
            
            if not self.dynamic_limits:
                return current_limits
            
            # Apply regime-based adjustments
            regime = market_context.get('regime', 'unknown')
            volatility_level = market_context.get('volatility_level', 'medium')
            
            # Regime adjustments
            if regime in self.regime_adjustments:
                adjustments = self.regime_adjustments[regime]
                current_limits['max_leverage'] *= adjustments['leverage']
                current_limits['max_position_risk'] *= adjustments['position_risk']
                current_limits['max_daily_trades'] = int(current_limits['max_daily_trades'] * adjustments['daily_trades'])
            
            # Volatility adjustments
            volatility_multipliers = {
                'low': {'leverage': 1.1, 'position_risk': 1.1, 'trade_size': 1.0},
                'medium': {'leverage': 1.0, 'position_risk': 1.0, 'trade_size': 1.0},
                'high': {'leverage': 0.8, 'position_risk': 0.8, 'trade_size': 0.9},
                'extreme': {'leverage': 0.6, 'position_risk': 0.6, 'trade_size': 0.8}
            }
            
            if volatility_level in volatility_multipliers:
                vol_adj = volatility_multipliers[volatility_level]
                current_limits['max_leverage'] *= vol_adj['leverage']
                current_limits['max_position_risk'] *= vol_adj['position_risk']
                current_limits['max_trade_size'] *= vol_adj['trade_size']
            
            # Apply bounds to prevent extreme adjustments
            current_limits['max_leverage'] = max(5.0, min(100.0, current_limits['max_leverage']))
            current_limits['max_position_risk'] = max(0.05, min(0.5, current_limits['max_position_risk']))
            current_limits['max_total_risk'] = max(0.2, min(1.0, current_limits['max_total_risk']))
            
            return current_limits
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "dynamic_limits")
            self.logger.warning(f"Dynamic limits calculation failed: {error_context}")
            return {
                'max_leverage': self.max_leverage,
                'max_position_risk': self.max_position_risk,
                'max_total_risk': self.max_total_risk,
                'max_daily_trades': self.max_daily_trades,
                'min_trade_size': self.min_trade_size,
                'max_trade_size': self.max_trade_size
            }
    
    async def _validate_pending_orders_comprehensive(self, pending_orders: List[Dict], 
                                                   positions: List[Dict], balance: float,
                                                   market_context: Dict[str, Any], 
                                                   current_limits: Dict[str, float]) -> Dict[str, Any]:
        """Comprehensive validation of pending orders"""
        start_time = datetime.datetime.now()
        
        validation_results = {
            'total_orders': len(pending_orders),
            'approved': [],
            'rejected': [],
            'violations': [],
            'validation_details': []
        }
        
        try:
            for order in pending_orders:
                order_validation = await self._validate_single_order(
                    order, positions, balance, market_context, current_limits
                )
                
                validation_results['validation_details'].append(order_validation)
                
                if order_validation['approved']:
                    validation_results['approved'].append(order)
                    self.validation_stats['approved'] += 1
                else:
                    validation_results['rejected'].append(order)
                    validation_results['violations'].extend(order_validation['violations'])
                    
                    # Track rejection reasons
                    for violation in order_validation['violations']:
                        self.validation_stats['violations'][violation['type']] += 1
                
                self.validation_stats['total_validations'] += 1
            
            # Calculate processing time
            processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            validation_results['processing_time_ms'] = processing_time
            
            # Update approval rate
            if self.validation_stats['total_validations'] > 0:
                approval_rate = self.validation_stats['approved'] / self.validation_stats['total_validations']
                self.approval_rate_history.append(approval_rate)
            
            return validation_results
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "order_validation")
            self.logger.error(f"Order validation failed: {error_context}")
            return {'error': error_context, 'total_orders': len(pending_orders)}
    
    async def _validate_single_order(self, order: Dict[str, Any], positions: List[Dict],
                                    balance: float, market_context: Dict[str, Any],
                                    current_limits: Dict[str, float]) -> Dict[str, Any]:
        """Validate a single order with comprehensive checks"""
        # Initialize order_details with defaults to ensure it's always defined
        order_details = {
            'instrument': 'UNKNOWN',
            'size': 0.0,
            'side': 'UNKNOWN',
            'price': 1.0
        }
        
        try:
            violations = []
            order_details = {
                'instrument': order.get('instrument', order.get('symbol', 'UNKNOWN')),
                'size': abs(order.get('size', order.get('volume', 0))),
                'side': order.get('side', 'BUY' if order.get('size', 0) > 0 else 'SELL'),
                'price': order.get('price', 1.0)
            }
            
            # 1. Daily trade limit check
            if self.daily_trade_count >= current_limits['max_daily_trades']:
                violations.append({
                    'type': 'daily_trade_limit',
                    'message': f"Daily trade limit exceeded: {self.daily_trade_count}/{current_limits['max_daily_trades']}",
                    'severity': 'critical'
                })
            
            # 2. Instrument allowlist check
            if not self._is_instrument_allowed(order_details['instrument']):
                violations.append({
                    'type': 'instrument_not_allowed',
                    'message': f"Instrument {order_details['instrument']} not in allowlist",
                    'severity': 'critical'
                })
            
            # 3. Trading hours check
            if self._is_trading_restricted():
                violations.append({
                    'type': 'restricted_hours',
                    'message': f"Trading restricted during current hour",
                    'severity': 'warning'
                })
            
            # 4. Trade size validation
            size_violations = self._validate_trade_size(order_details['size'], current_limits)
            violations.extend(size_violations)
            
            # 5. Position risk validation
            position_risk_violations = self._validate_position_risk(
                order_details, positions, balance, current_limits
            )
            violations.extend(position_risk_violations)
            
            # 6. Total exposure validation
            exposure_violations = self._validate_total_exposure(
                order_details, positions, balance, current_limits
            )
            violations.extend(exposure_violations)
            
            # 7. Leverage validation
            leverage_violations = self._validate_leverage_limits(
                order_details, positions, balance, current_limits
            )
            violations.extend(leverage_violations)
            
            # 8. Market context validation
            context_violations = self._validate_market_context(order_details, market_context)
            violations.extend(context_violations)
            
            # Determine approval status
            critical_violations = [v for v in violations if v.get('severity') == 'critical']
            approved = len(critical_violations) == 0
            
            return {
                'order_details': order_details,
                'approved': approved,
                'violations': violations,
                'risk_score': len(violations) / 8.0,  # Normalize by number of checks
                'validation_timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "single_order_validation")
            return {
                'order_details': order_details,
                'approved': False,
                'violations': [{'type': 'validation_error', 'message': error_context, 'severity': 'critical'}],
                'risk_score': 1.0,
                'error': error_context
            }
    
    def _is_instrument_allowed(self, instrument: str) -> bool:
        """Check if instrument is in allowlist"""
        return instrument.upper() in self.allowed_instruments
    
    def _is_trading_restricted(self) -> bool:
        """Check if trading is restricted at current hour"""
        if not self.restricted_hours:
            return False
        
        current_hour = datetime.datetime.now().hour
        return current_hour in self.restricted_hours
    
    def _validate_trade_size(self, size: float, current_limits: Dict[str, float]) -> List[Dict[str, Any]]:
        """Validate trade size against limits"""
        violations = []
        
        if size < current_limits['min_trade_size']:
            violations.append({
                'type': 'size_too_small',
                'message': f"Trade size {size:.4f} below minimum {current_limits['min_trade_size']:.4f}",
                'severity': 'warning'
            })
        
        if size > current_limits['max_trade_size']:
            violations.append({
                'type': 'size_too_large',
                'message': f"Trade size {size:.4f} exceeds maximum {current_limits['max_trade_size']:.4f}",
                'severity': 'critical'
            })
        
        return violations
    
    def _validate_position_risk(self, order_details: Dict[str, Any], positions: List[Dict],
                               balance: float, current_limits: Dict[str, float]) -> List[Dict[str, Any]]:
        """Validate position risk limits"""
        violations = []
        
        try:
            # Calculate position value (simplified for forex)
            size = order_details['size']
            price = order_details['price']
            position_value = size * price * 100000  # Standard lot size
            
            # Calculate risk as percentage of balance
            position_risk = position_value / max(balance, 1.0)
            
            if position_risk > current_limits['max_position_risk']:
                violations.append({
                    'type': 'position_risk_exceeded',
                    'message': f"Position risk {position_risk:.1%} exceeds limit {current_limits['max_position_risk']:.1%}",
                    'severity': 'critical'
                })
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "position_risk_validation")
            violations.append({
                'type': 'position_risk_calculation_error',
                'message': f"Risk calculation failed: {error_context}",
                'severity': 'warning'
            })
        
        return violations
    
    def _validate_total_exposure(self, order_details: Dict[str, Any], positions: List[Dict],
                                balance: float, current_limits: Dict[str, float]) -> List[Dict[str, Any]]:
        """Validate total portfolio exposure"""
        violations = []
        
        try:
            # Calculate existing exposure
            existing_exposure = 0.0
            for position in positions:
                pos_size = abs(position.get('size', position.get('volume', 0)))
                pos_price = position.get('current_price', position.get('price', 1.0))
                existing_exposure += pos_size * pos_price * 100000
            
            # Add new order exposure
            new_position_value = order_details['size'] * order_details['price'] * 100000
            total_exposure = existing_exposure + new_position_value
            
            # Calculate total risk
            total_risk = total_exposure / max(balance, 1.0)
            
            if total_risk > current_limits['max_total_risk']:
                violations.append({
                    'type': 'total_risk_exceeded',
                    'message': f"Total risk {total_risk:.1%} exceeds limit {current_limits['max_total_risk']:.1%}",
                    'severity': 'critical'
                })
            
            # Update exposure tracking
            self.total_exposure = total_exposure
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "total_exposure_validation")
            violations.append({
                'type': 'exposure_calculation_error',
                'message': f"Exposure calculation failed: {error_context}",
                'severity': 'warning'
            })
        
        return violations
    
    def _validate_leverage_limits(self, order_details: Dict[str, Any], positions: List[Dict],
                                 balance: float, current_limits: Dict[str, float]) -> List[Dict[str, Any]]:
        """Validate leverage limits"""
        violations = []
        
        try:
            # Calculate current leverage including new order
            leverage = self.total_exposure / max(balance, 1.0)
            self.current_leverage = leverage
            
            if leverage > current_limits['max_leverage']:
                violations.append({
                    'type': 'leverage_exceeded',
                    'message': f"Leverage {leverage:.1f}x exceeds limit {current_limits['max_leverage']:.1f}x",
                    'severity': 'critical'
                })
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "leverage_validation")
            violations.append({
                'type': 'leverage_calculation_error',
                'message': f"Leverage calculation failed: {error_context}",
                'severity': 'warning'
            })
        
        return violations
    
    def _validate_market_context(self, order_details: Dict[str, Any], 
                                market_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate against market context restrictions"""
        violations = []
        
        try:
            volatility_level = market_context.get('volatility_level', 'medium')
            regime = market_context.get('regime', 'unknown')
            
            # Restrict large trades during extreme volatility
            if volatility_level == 'extreme' and order_details['size'] > self.max_trade_size * 0.5:
                violations.append({
                    'type': 'extreme_volatility_restriction',
                    'message': f"Large trade restricted during extreme volatility",
                    'severity': 'warning'
                })
            
            # Crisis regime restrictions
            if regime == 'crisis' and order_details['size'] > self.max_trade_size * 0.3:
                violations.append({
                    'type': 'crisis_regime_restriction',
                    'message': f"Trade size restricted during crisis regime",
                    'severity': 'warning'
                })
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_context_validation")
            violations.append({
                'type': 'context_validation_error',
                'message': f"Context validation failed: {error_context}",
                'severity': 'info'
            })
        
        return violations
    
    def _assess_current_risk_exposure(self, positions: List[Dict], balance: float,
                                     current_limits: Dict[str, float]) -> Dict[str, Any]:
        """Assess current risk exposure and compliance status"""
        try:
            # Calculate current metrics
            total_positions = len(positions)
            current_exposure = self.total_exposure
            current_leverage = self.current_leverage
            
            # Calculate risk budget usage
            leverage_usage = current_leverage / current_limits['max_leverage']
            exposure_usage = (current_exposure / balance) / current_limits['max_total_risk']
            daily_trades_usage = self.daily_trade_count / current_limits['max_daily_trades']
            
            # Overall risk budget usage
            self.risk_budget_usage = max(leverage_usage, exposure_usage, daily_trades_usage)
            
            # Calculate compliance score
            violations_count = sum(self.validation_stats['violations'].values())
            total_validations = max(self.validation_stats['total_validations'], 1)
            violation_rate = violations_count / total_validations
            
            self.compliance_score = max(0.0, 1.0 - violation_rate - (self.risk_budget_usage - 1.0) * 0.5)
            
            return {
                'total_positions': total_positions,
                'current_exposure': current_exposure,
                'current_leverage': current_leverage,
                'risk_budget_usage': self.risk_budget_usage,
                'compliance_score': self.compliance_score,
                'daily_trade_count': self.daily_trade_count,
                'leverage_usage': leverage_usage,
                'exposure_usage': exposure_usage,
                'daily_trades_usage': daily_trades_usage,
                'violation_rate': violation_rate
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "risk_assessment")
            self.logger.error(f"Risk assessment failed: {error_context}")
            return {'error': error_context, 'compliance_score': 0.5}
    
    def _calculate_compliance_metrics(self, validation_results: Dict[str, Any],
                                     risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive compliance metrics"""
        try:
            # Validation metrics
            total_orders = validation_results.get('total_orders', 0)
            approved_orders = len(validation_results.get('approved', []))
            rejected_orders = len(validation_results.get('rejected', []))
            
            approval_rate = approved_orders / max(total_orders, 1)
            rejection_rate = rejected_orders / max(total_orders, 1)
            
            # Risk metrics
            risk_budget_usage = risk_assessment.get('risk_budget_usage', 0.0)
            compliance_score = risk_assessment.get('compliance_score', 1.0)
            
            # Historical metrics
            avg_approval_rate = np.mean(self.approval_rate_history) if self.approval_rate_history else 1.0
            
            # Violation breakdown
            violation_breakdown = dict(self.validation_stats['violations'])
            
            return {
                'approval_rate': approval_rate,
                'rejection_rate': rejection_rate,
                'avg_approval_rate': avg_approval_rate,
                'compliance_score': compliance_score,
                'risk_budget_usage': risk_budget_usage,
                'total_validations': self.validation_stats['total_validations'],
                'total_violations': sum(violation_breakdown.values()),
                'violation_breakdown': violation_breakdown,
                'daily_trade_utilization': self.daily_trade_count / self.max_daily_trades
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "compliance_metrics")
            self.logger.error(f"Compliance metrics calculation failed: {error_context}")
            return {'compliance_score': 0.5, 'error': error_context}
    
    async def _generate_compliance_thesis(self, validation_results: Dict[str, Any],
                                         risk_assessment: Dict[str, Any],
                                         market_context: Dict[str, Any]) -> str:
        """Generate intelligent thesis explaining compliance decisions"""
        try:
            thesis_parts = []
            
            # Validation overview
            total_orders = validation_results.get('total_orders', 0)
            approved = len(validation_results.get('approved', []))
            rejected = len(validation_results.get('rejected', []))
            
            if total_orders > 0:
                thesis_parts.append(
                    f"Processed {total_orders} orders: {approved} approved, {rejected} rejected "
                    f"({approved/total_orders:.1%} approval rate)"
                )
            else:
                thesis_parts.append("No pending orders to validate")
            
            # Risk assessment
            compliance_score = risk_assessment.get('compliance_score', 1.0)
            risk_budget_usage = risk_assessment.get('risk_budget_usage', 0.0)
            
            if compliance_score >= 0.9:
                thesis_parts.append(f"EXCELLENT compliance maintained ({compliance_score:.1%})")
            elif compliance_score >= 0.7:
                thesis_parts.append(f"GOOD compliance status ({compliance_score:.1%})")
            elif compliance_score >= 0.5:
                thesis_parts.append(f"FAIR compliance with room for improvement ({compliance_score:.1%})")
            else:
                thesis_parts.append(f"POOR compliance requiring immediate attention ({compliance_score:.1%})")
            
            # Risk budget analysis
            if risk_budget_usage > 0.8:
                thesis_parts.append(f"HIGH risk budget utilization ({risk_budget_usage:.1%}) - approaching limits")
            elif risk_budget_usage > 0.5:
                thesis_parts.append(f"MODERATE risk budget usage ({risk_budget_usage:.1%})")
            else:
                thesis_parts.append(f"Conservative risk budget usage ({risk_budget_usage:.1%})")
            
            # Violation analysis
            violations = validation_results.get('violations', [])
            if violations:
                violation_types = set(v['type'] for v in violations)
                thesis_parts.append(f"Detected {len(violations)} violations: {', '.join(list(violation_types)[:3])}")
            
            # Market context impact
            regime = market_context.get('regime', 'unknown')
            volatility = market_context.get('volatility_level', 'medium')
            
            if self.dynamic_limits:
                thesis_parts.append(f"Dynamic limits adjusted for {regime} regime and {volatility} volatility")
            
            # Daily trading status
            daily_usage = self.daily_trade_count / self.max_daily_trades
            if daily_usage > 0.8:
                thesis_parts.append(f"Daily trade limit utilization HIGH ({daily_usage:.1%})")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Thesis generation failed: {error_context}"
    
    def _generate_recommendations(self, validation_results: Dict[str, Any],
                                 risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on compliance analysis"""
        recommendations = []
        
        try:
            # Risk budget recommendations
            risk_budget_usage = risk_assessment.get('risk_budget_usage', 0.0)
            
            if risk_budget_usage > 0.9:
                recommendations.append("IMMEDIATE: Reduce position sizes - risk budget critically high")
                recommendations.append("Consider closing some positions to free up risk capacity")
            elif risk_budget_usage > 0.7:
                recommendations.append("Monitor risk budget closely - approaching limits")
            
            # Violation-based recommendations
            violations = validation_results.get('violations', [])
            violation_types = [v['type'] for v in violations]
            
            if 'daily_trade_limit' in violation_types:
                recommendations.append("Daily trade limit reached - wait for next trading day")
            
            if 'leverage_exceeded' in violation_types:
                recommendations.append("Reduce leverage by closing positions or increasing margin")
            
            if 'position_risk_exceeded' in violation_types:
                recommendations.append("Consider smaller position sizes or better risk management")
            
            # Compliance score recommendations
            compliance_score = risk_assessment.get('compliance_score', 1.0)
            
            if compliance_score < 0.7:
                recommendations.append("Review and improve trade validation processes")
                recommendations.append("Consider more conservative position sizing")
            
            # Daily trading recommendations
            daily_usage = self.daily_trade_count / self.max_daily_trades
            if daily_usage > 0.8:
                recommendations.append("Approaching daily trade limit - prioritize high-conviction trades")
            
            # Proactive recommendations
            if not recommendations:
                recommendations.append("Compliance status optimal - maintain current risk management practices")
                recommendations.append("Continue monitoring for regime changes that may affect limits")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendations")
            recommendations.append(f"Recommendation generation failed: {error_context}")
        
        return recommendations
    
    def _update_smart_info_bus(self, validation_results: Dict[str, Any],
                              risk_assessment: Dict[str, Any], 
                              compliance_metrics: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with compliance results"""
        try:
            # Core compliance status
            self.smart_bus.set('compliance_status', {
                'compliance_score': self.compliance_score,
                'risk_budget_usage': self.risk_budget_usage,
                'validation_results': validation_results,
                'risk_assessment': risk_assessment,
                'compliance_metrics': compliance_metrics,
                'thesis': thesis
            }, module='ComplianceModule', thesis=thesis)
            
            # Validation results for other modules
            self.smart_bus.set('validation_results', {
                'approved_orders': validation_results.get('approved', []),
                'rejected_orders': validation_results.get('rejected', []),
                'approval_rate': compliance_metrics.get('approval_rate', 1.0)
            }, module='ComplianceModule', 
            thesis=f"Order validation: {len(validation_results.get('approved', []))} approved")
            
            # Risk limits for risk management modules
            self.smart_bus.set('risk_limits', {
                'max_leverage': self.max_leverage,
                'max_position_risk': self.max_position_risk,
                'max_total_risk': self.max_total_risk,
                'current_leverage': self.current_leverage,
                'risk_budget_usage': self.risk_budget_usage
            }, module='ComplianceModule', thesis="Risk limits and current utilization updated")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smart_info_bus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")
    
    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'compliance_score': 1.0,
            'validation_results': {'total_orders': 0, 'approved': [], 'rejected': []},
            'risk_assessment': {'compliance_score': 1.0, 'risk_budget_usage': 0.0},
            'compliance_metrics': {'compliance_score': 1.0, 'approval_rate': 1.0},
            'current_limits': {},
            'thesis': "Compliance Module is disabled",
            'recommendations': ["Enable Compliance Module for trade validation"]
        }
    
    def _generate_error_response(self, error_context: str) -> Dict[str, Any]:
        """Generate response when processing fails"""
        return {
            'compliance_score': 0.5,
            'validation_results': {'error': error_context},
            'risk_assessment': {'compliance_score': 0.5, 'error': error_context},
            'compliance_metrics': {'compliance_score': 0.5, 'error': error_context},
            'current_limits': {},
            'thesis': f"Compliance processing failed: {error_context}",
            'recommendations': ["Investigate compliance system errors"]
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete module state for hot-reload"""
        return {
            'daily_trade_count': self.daily_trade_count,
            'last_trade_date': self.last_trade_date.isoformat() if self.last_trade_date else None,
            'total_exposure': self.total_exposure,
            'current_leverage': self.current_leverage,
            'risk_budget_usage': self.risk_budget_usage,
            'compliance_score': self.compliance_score,
            'validation_stats': dict(self.validation_stats),
            'config': self.config.copy()
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state for hot-reload"""
        self.daily_trade_count = state.get('daily_trade_count', 0)
        last_date_str = state.get('last_trade_date')
        self.last_trade_date = datetime.datetime.fromisoformat(last_date_str).date() if last_date_str else None
        self.total_exposure = state.get('total_exposure', 0.0)
        self.current_leverage = state.get('current_leverage', 0.0)
        self.risk_budget_usage = state.get('risk_budget_usage', 0.0)
        self.compliance_score = state.get('compliance_score', 1.0)
        self.validation_stats.update(state.get('validation_stats', {}))
        self.config.update(state.get('config', {}))
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring"""
        return {
            'compliance_score': self.compliance_score,
            'risk_budget_usage': self.risk_budget_usage,
            'approval_rate': self.validation_stats['approved'] / max(self.validation_stats['total_validations'], 1),
            'daily_trade_count': self.daily_trade_count,
            'current_leverage': self.current_leverage,
            'total_violations': sum(self.validation_stats['violations'].values()),
            'enabled': self.enabled
        }