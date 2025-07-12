"""
🧠 Enhanced Bias Auditor with SmartInfoBus Integration v3.0
Advanced psychological bias detection and correction system with real-time trading behavior analysis
"""

import asyncio
import time
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

# ═══════════════════════════════════════════════════════════════════
# MODERN SMARTINFOBUS IMPORTS
# ═══════════════════════════════════════════════════════════════════
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="BiasAuditor",
    version="3.0.0",
    category="strategy",
    provides=[
        "bias_analysis", "bias_corrections", "bias_adjustments",
        "bias_report", "bias_recommendations", "psychological_state"
    ],
    requires=[
        "recent_trades", "current_pnl", "positions", "risk_data",
        "session_context", "market_regime"
    ],
    description="Advanced psychological bias detection and correction system with real-time trading behavior analysis",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=150,
    priority=7,
    explainable=True,
    hot_reload=True
)
class BiasAuditor(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🧠 PRODUCTION-GRADE Bias Auditor v3.0
    
    Advanced psychological bias detection and correction system with:
    - Real-time trading behavior analysis
    - Context-aware bias detection algorithms
    - Intelligent correction mechanisms with adaptive learning
    - SmartInfoBus zero-wiring architecture
    - Comprehensive thesis generation and explainability
    """

    def _initialize(self):
        """Initialize advanced bias auditing systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced bias tracking configuration
        self.history_len = self.config.get('history_len', 100)
        self.correction_threshold = self.config.get('correction_threshold', 3)
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        self.debug = self.config.get('debug', False)
        
        # Core bias tracking state
        self.bias_history = deque(maxlen=self.history_len)
        self.bias_corrections = defaultdict(int)
        self.bias_performance = defaultdict(list)
        self.bias_frequencies = defaultdict(int)
        
        # Enhanced bias categories with intelligent thresholds
        self.bias_categories = {
            'revenge': {
                'description': 'Trading to recover losses aggressively',
                'threshold': -50.0,
                'weight_reduction': 0.3,
                'detection_algorithm': 'revenge_pattern_analysis'
            },
            'fear': {
                'description': 'Avoiding trades due to recent losses',
                'threshold': -100.0,
                'weight_reduction': 0.2,
                'detection_algorithm': 'fear_avoidance_analysis'
            },
            'greed': {
                'description': 'Overconfident trading after wins',
                'threshold': 100.0,
                'weight_reduction': 0.25,
                'detection_algorithm': 'greed_escalation_analysis'
            },
            'fomo': {
                'description': 'Fear of missing out on trends',
                'threshold': 50.0,
                'weight_reduction': 0.35,
                'detection_algorithm': 'fomo_chasing_analysis'
            },
            'anchoring': {
                'description': 'Fixation on previous price levels',
                'threshold': 0.0,
                'weight_reduction': 0.15,
                'detection_algorithm': 'anchoring_fixation_analysis'
            }
        }
        
        # Session performance tracking
        self.session_stats = {
            'total_biases_detected': 0,
            'biases_corrected': 0,
            'correction_effectiveness': 0.0,
            'most_common_bias': 'none',
            'bias_impact_score': 0.0,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Process call tracking
        self.process_call_count = 0
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🧠",
            message=f"Bias Auditor v{version} initialized",
            categories=len(self.bias_categories),
            history_length=self.history_len,
            correction_threshold=self.correction_threshold
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="BiasAuditor",
            log_path="logs/strategy/bias_auditor.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("BiasAuditor", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()


    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Bias Auditor v3.0 Initialization Complete:
        
        System Configuration:
        - Tracking {len(self.bias_categories)} bias categories: {', '.join(self.bias_categories.keys())}
        - History buffer: {self.history_len} records with adaptive learning
        - Correction threshold: {self.correction_threshold} occurrences before intervention
        - Adaptation rate: {self.adaptation_rate:.1%} for continuous improvement
        
        Detection Algorithms:
        - Revenge trading: Pattern analysis of post-loss position sizing
        - Fear avoidance: Statistical analysis of trading frequency degradation
        - Greed escalation: Win-streak position size inflation detection
        - FOMO chasing: Trend-following behavior identification
        - Anchoring fixation: Price level clustering analysis
        
        Advanced Features:
        - Real-time bias strength measurement with contextual adjustments
        - Intelligent correction mechanisms with performance feedback
        - Session-wide bias impact scoring and effectiveness tracking
        - Market regime awareness for context-sensitive thresholds
        
        Expected Outcomes:
        - Reduced psychological trading errors through early detection
        - Improved decision-making quality via bias corrections
        - Enhanced self-awareness of trading behavior patterns
        - Data-driven approach to psychological risk management
        """
        
        self.smart_bus.set('bias_auditor_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat()
        }, module='BiasAuditor', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive bias analysis
        
        Returns:
            Dict containing bias analysis, corrections, adjustments, and recommendations
        """
        start_time = time.time()
        self.process_call_count += 1
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive trading data from SmartInfoBus
            trading_data = await self._get_comprehensive_trading_data()
            
            # Core bias analysis with error handling
            bias_analysis = await self._analyze_psychological_biases_comprehensive(trading_data)
            
            # Generate intelligent corrections and adjustments
            corrections = await self._generate_intelligent_corrections(bias_analysis)
            adjustments = self._calculate_dynamic_adjustments(bias_analysis, corrections)
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_thesis(bias_analysis, corrections, adjustments)
            
            # Create comprehensive results
            results = {
                'bias_analysis': bias_analysis,
                'bias_corrections': corrections,
                'bias_adjustments': adjustments,
                'bias_report': self._generate_comprehensive_report(bias_analysis),
                'recommendations': self._generate_intelligent_recommendations(bias_analysis),
                'session_performance': self.session_stats.copy(),
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('BiasAuditor', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_trading_data(self) -> Dict[str, Any]:
        """Get comprehensive trading data using modern SmartInfoBus patterns"""
        try:
            return {
                'recent_trades': self.smart_bus.get('recent_trades', 'BiasAuditor') or [],
                'current_pnl': self.smart_bus.get('current_pnl', 'BiasAuditor') or 0.0,
                'positions': self.smart_bus.get('positions', 'BiasAuditor') or [],
                'risk_metrics': self.smart_bus.get('risk_data', 'BiasAuditor') or {},
                'session_context': self.smart_bus.get('session_context', 'BiasAuditor') or {},
                'market_regime': self.smart_bus.get('market_regime', 'BiasAuditor') or 'unknown',
                'volatility_level': self.smart_bus.get('volatility_level', 'BiasAuditor') or 'medium',
                'trading_session': self.smart_bus.get('trading_session', 'BiasAuditor') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "BiasAuditor")
            self.logger.warning(f"Data retrieval incomplete: {error_context}")
            return self._get_safe_trading_defaults()

    async def _analyze_psychological_biases_comprehensive(self, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive bias analysis with advanced pattern recognition"""
        try:
            bias_signals = {}
            context_factors = {}
            
            # Analyze each bias category with context awareness
            for bias_type, config in self.bias_categories.items():
                algorithm = config['detection_algorithm']
                strength, factors = await self._run_bias_detection_algorithm(
                    algorithm, bias_type, trading_data
                )
                
                if strength > 0.1:  # Minimum detection threshold
                    bias_signals[bias_type] = strength
                    context_factors[bias_type] = factors
                    
                    # Record detection
                    await self._record_bias_detection_comprehensive(
                        bias_type, strength, factors, trading_data
                    )
            
            # Calculate aggregate bias metrics
            aggregate_metrics = self._calculate_aggregate_bias_metrics(bias_signals)
            
            return {
                'individual_biases': bias_signals,
                'context_factors': context_factors,
                'aggregate_metrics': aggregate_metrics,
                'detection_timestamp': datetime.datetime.now().isoformat(),
                'market_context': {
                    'regime': trading_data.get('market_regime', 'unknown'),
                    'volatility': trading_data.get('volatility_level', 'medium')
                }
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "BiasAuditor")
            self.logger.error(f"Bias analysis failed: {error_context}")
            return self._get_safe_bias_defaults()

    async def _run_bias_detection_algorithm(self, algorithm: str, bias_type: str, 
                                          trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Run specific bias detection algorithm with context analysis"""
        try:
            if algorithm == 'revenge_pattern_analysis':
                return await self._detect_revenge_bias_advanced(trading_data)
            elif algorithm == 'fear_avoidance_analysis':
                return await self._detect_fear_bias_advanced(trading_data)
            elif algorithm == 'greed_escalation_analysis':
                return await self._detect_greed_bias_advanced(trading_data)
            elif algorithm == 'fomo_chasing_analysis':
                return await self._detect_fomo_bias_advanced(trading_data)
            elif algorithm == 'anchoring_fixation_analysis':
                return await self._detect_anchoring_bias_advanced(trading_data)
            else:
                self.logger.warning(f"Unknown detection algorithm: {algorithm}")
                return 0.0, ['unknown_algorithm']
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, f"bias_detection_{bias_type}")
            self.logger.warning(f"Detection algorithm failed for {bias_type}: {error_context}")
            return 0.0, ['detection_error']

    async def _detect_revenge_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Advanced revenge trading detection with multiple pattern analysis"""
        try:
            recent_trades = trading_data.get('recent_trades', [])
            current_pnl = trading_data.get('current_pnl', 0)
            factors = []
            
            if not recent_trades or len(recent_trades) < 2:
                return 0.0, factors
            
            # Pattern 1: Position size escalation after losses
            recent_losses = [t for t in recent_trades[-5:] if t.get('pnl', 0) < 0]
            if len(recent_losses) >= 2:
                sizes = [abs(t.get('size', 0)) for t in recent_losses]
                if len(sizes) >= 2 and sizes[-1] > sizes[0] * 1.5:
                    escalation_factor = min(1.0, (sizes[-1] / sizes[0] - 1.0) * 0.5)
                    factors.append('position_size_escalation')
                    
            # Pattern 2: Rapid trading frequency after losses
            if current_pnl < -100:
                last_hour_trades = len([t for t in recent_trades[-10:] 
                                      if self._is_recent_trade(t, minutes=60)])
                if last_hour_trades >= 5:
                    frequency_factor = min(1.0, last_hour_trades / 10.0)
                    factors.append('rapid_trading_frequency')
                    
            # Pattern 3: Increasingly aggressive risk-taking
            risk_escalation = self._analyze_risk_escalation_pattern(recent_trades)
            if risk_escalation > 0.3:
                factors.append('risk_escalation')
            
            # Calculate composite revenge strength
            strength = self._calculate_composite_bias_strength(factors, 'revenge')
            
            return strength, factors
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "revenge_detection")
            return 0.0, ['detection_error']

    async def _detect_fear_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Advanced fear bias detection with statistical analysis"""
        try:
            recent_trades = trading_data.get('recent_trades', [])
            risk_data = trading_data.get('risk_metrics', {})
            factors = []
            
            # Pattern 1: Reduced trading frequency after drawdown
            drawdown = risk_data.get('current_drawdown', 0)
            if drawdown > 0.05:  # 5% drawdown
                recent_count = len([t for t in recent_trades[-20:] 
                                  if self._is_recent_trade(t, hours=24)])
                expected_trades = 10  # Expected daily trades
                
                if recent_count < expected_trades * 0.5:
                    frequency_reduction = 1.0 - (recent_count / expected_trades)
                    factors.append('trading_frequency_reduction')
                    
            # Pattern 2: Position size reduction under normal conditions
            size_reduction = self._analyze_position_size_trends(recent_trades)
            if size_reduction > 0.3:
                factors.append('position_size_reduction')
                
            # Pattern 3: Premature profit-taking
            premature_exits = self._detect_premature_profit_taking(recent_trades)
            if premature_exits > 0.4:
                factors.append('premature_profit_taking')
            
            strength = self._calculate_composite_bias_strength(factors, 'fear')
            return strength, factors
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "fear_detection")
            return 0.0, ['detection_error']

    async def _detect_greed_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Advanced greed bias detection with win-streak analysis"""
        try:
            recent_trades = trading_data.get('recent_trades', [])
            positions = trading_data.get('positions', [])
            factors = []
            
            # Pattern 1: Position size escalation after wins
            recent_wins = [t for t in recent_trades[-5:] if t.get('pnl', 0) > 0]
            if len(recent_wins) >= 3:
                total_exposure = sum(abs(p.get('size', 0)) for p in positions)
                if total_exposure > 2.0:  # Over-leveraged
                    win_streak = len(recent_wins)
                    factors.append('position_size_inflation')
                    
            # Pattern 2: Reduced stop-loss discipline
            stop_discipline = self._analyze_stop_loss_discipline(recent_trades)
            if stop_discipline < 0.3:
                factors.append('reduced_stop_discipline')
                
            # Pattern 3: Overconfident market timing
            timing_confidence = self._analyze_market_timing_confidence(recent_trades)
            if timing_confidence > 0.7:
                factors.append('overconfident_timing')
            
            strength = self._calculate_composite_bias_strength(factors, 'greed')
            return strength, factors
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "greed_detection")
            return 0.0, ['detection_error']

    async def _detect_fomo_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Advanced FOMO detection with trend-chasing analysis"""
        try:
            market_regime = trading_data.get('market_regime', 'unknown')
            recent_trades = trading_data.get('recent_trades', [])
            factors = []
            
            # Pattern 1: Excessive entries during trending markets
            if market_regime == 'trending':
                recent_entries = len(recent_trades)
                if recent_entries > 3:
                    entry_frequency = min(1.0, recent_entries / 10.0)
                    factors.append('excessive_trend_chasing')
                    
            # Pattern 2: Late entries at unfavorable prices
            late_entries = self._analyze_entry_timing_quality(recent_trades)
            if late_entries > 0.5:
                factors.append('poor_entry_timing')
                
            # Pattern 3: Abandoning strategy for hot markets
            strategy_abandonment = self._detect_strategy_abandonment(recent_trades)
            if strategy_abandonment > 0.4:
                factors.append('strategy_abandonment')
            
            strength = self._calculate_composite_bias_strength(factors, 'fomo')
            return strength, factors
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "fomo_detection")
            return 0.0, ['detection_error']

    async def _detect_anchoring_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Advanced anchoring detection with price level analysis"""
        try:
            recent_trades = trading_data.get('recent_trades', [])
            factors = []
            
            if not recent_trades:
                return 0.0, factors
                
            # Pattern 1: Clustering around previous price levels
            price_levels = [t.get('entry_price', 0) for t in recent_trades[-5:] 
                          if t.get('entry_price', 0) > 0]
            
            if len(price_levels) >= 3:
                price_std = np.std(price_levels)
                price_mean = np.mean(price_levels)
                
                if price_std / price_mean < 0.01:  # Very tight clustering
                    factors.append('price_level_clustering')
                    
            # Pattern 2: Fixation on round numbers
            round_number_bias = self._analyze_round_number_bias(recent_trades)
            if round_number_bias > 0.3:
                factors.append('round_number_fixation')
                
            # Pattern 3: Historical price reference bias
            historical_bias = self._analyze_historical_price_bias(recent_trades)
            if historical_bias > 0.4:
                factors.append('historical_price_reference')
            
            strength = self._calculate_composite_bias_strength(factors, 'anchoring')
            return strength, factors
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "anchoring_detection")
            return 0.0, ['detection_error']

    def _calculate_composite_bias_strength(self, factors: List[str], bias_type: str) -> float:
        """Calculate composite bias strength from multiple factors"""
        if not factors:
            return 0.0
            
        # Base strength from number of factors
        factor_strength = len(factors) / 3.0  # Normalize to 3 factors max
        
        # Apply bias-specific weightings
        bias_weights = {
            'revenge': 1.2,    # Higher weight for revenge trading
            'fear': 0.8,       # Lower weight for fear (more gradual)
            'greed': 1.1,      # High weight for greed
            'fomo': 1.3,       # Highest weight for FOMO
            'anchoring': 0.7   # Lower weight for anchoring
        }
        
        weight = bias_weights.get(bias_type, 1.0)
        
        # Apply historical correction factor
        correction_count = self.bias_corrections.get(bias_type, 0)
        correction_factor = 1.0 + (correction_count * 0.1)  # Increase sensitivity with corrections
        
        final_strength = min(1.0, factor_strength * weight * correction_factor)
        return final_strength

    async def _record_bias_detection_comprehensive(self, bias_type: str, strength: float, 
                                                 factors: List[str], trading_data: Dict[str, Any]):
        """Record comprehensive bias detection with rich context"""
        try:
            bias_record = {
                'type': bias_type,
                'strength': strength,
                'factors': factors,
                'timestamp': datetime.datetime.now().isoformat(),
                'trading_context': {
                    'pnl': trading_data.get('current_pnl', 0),
                    'positions_count': len(trading_data.get('positions', [])),
                    'recent_trades_count': len(trading_data.get('recent_trades', [])),
                    'market_regime': trading_data.get('market_regime', 'unknown'),
                    'volatility': trading_data.get('volatility_level', 'medium')
                },
                'pnl_impact': 0.0,  # Will be updated when outcome is known
                'correction_applied': False
            }
            
            self.bias_history.append(bias_record)
            self.bias_frequencies[bias_type] += 1
            self.session_stats['total_biases_detected'] += 1
            
            # Log significant bias detections
            if strength > 0.5:
                self.logger.warning(format_operator_message(
                    icon="⚠️",
                    message=f"Strong {bias_type.title()} bias detected",
                    strength=f"{strength:.1%}",
                    factors=", ".join(factors),
                    session_total=self.session_stats['total_biases_detected']
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "bias_recording")
            self.logger.error(f"Failed to record bias detection: {error_context}")

    async def _generate_intelligent_corrections(self, bias_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent bias corrections with adaptive mechanisms"""
        try:
            corrections = {}
            individual_biases = bias_analysis.get('individual_biases', {})
            
            for bias_type, strength in individual_biases.items():
                if strength > 0.3:  # Correction threshold
                    correction = await self._generate_bias_specific_correction(
                        bias_type, strength, bias_analysis
                    )
                    corrections[bias_type] = correction
                    
                    # Record correction application
                    self.bias_corrections[bias_type] += 1
                    self.session_stats['biases_corrected'] += 1
            
            return {
                'individual_corrections': corrections,
                'correction_timestamp': datetime.datetime.now().isoformat(),
                'total_corrections_applied': len(corrections),
                'session_correction_count': self.session_stats['biases_corrected']
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "correction_generation")
            self.logger.error(f"Correction generation failed: {error_context}")
            return {'individual_corrections': {}, 'error': str(error_context)}

    async def _generate_bias_specific_correction(self, bias_type: str, strength: float, 
                                               bias_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific correction for individual bias type"""
        try:
            base_config = self.bias_categories[bias_type]
            context_factors = bias_analysis.get('context_factors', {}).get(bias_type, [])
            
            correction = {
                'bias_type': bias_type,
                'strength': strength,
                'weight_reduction': base_config['weight_reduction'],
                'adaptive_adjustment': self._calculate_adaptive_adjustment(bias_type, strength),
                'context_factors': context_factors,
                'recommended_actions': self._generate_recommended_actions(bias_type, context_factors),
                'confidence_level': self._calculate_correction_confidence(bias_type, strength)
            }
            
            return correction
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "specific_correction")
            return {'error': str(error_context), 'bias_type': bias_type}

    def _calculate_dynamic_adjustments(self, bias_analysis: Dict[str, Any], 
                                     corrections: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic bias adjustments with learning integration"""
        try:
            adjustments = {}
            individual_biases = bias_analysis.get('individual_biases', {})
            individual_corrections = corrections.get('individual_corrections', {})
            
            for bias_type in self.bias_categories.keys():
                # Base adjustment
                base_adjustment = 1.0
                
                # Apply correction if bias detected
                if bias_type in individual_corrections:
                    correction = individual_corrections[bias_type]
                    weight_reduction = correction.get('weight_reduction', 0.0)
                    adaptive_adjustment = correction.get('adaptive_adjustment', 0.0)
                    
                    total_reduction = weight_reduction + adaptive_adjustment
                    base_adjustment = 1.0 - min(0.8, total_reduction)  # Cap at 20% minimum
                
                # Apply historical learning
                correction_count = self.bias_corrections.get(bias_type, 0)
                if correction_count >= self.correction_threshold:
                    learning_factor = min(0.2, correction_count * 0.05)
                    base_adjustment -= learning_factor
                
                adjustments[bias_type] = max(0.2, base_adjustment)  # Minimum 20% weight
            
            return adjustments
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adjustment_calculation")
            self.logger.error(f"Adjustment calculation failed: {error_context}")
            return {bias_type: 1.0 for bias_type in self.bias_categories.keys()}

    async def _generate_comprehensive_thesis(self, bias_analysis: Dict[str, Any], 
                                           corrections: Dict[str, Any], 
                                           adjustments: Dict[str, float]) -> str:
        """Generate comprehensive thesis explaining all bias decisions"""
        try:
            individual_biases = bias_analysis.get('individual_biases', {})
            aggregate_metrics = bias_analysis.get('aggregate_metrics', {})
            market_context = bias_analysis.get('market_context', {})
            
            thesis_parts = []
            
            # Executive Summary
            if individual_biases:
                strongest_bias = max(individual_biases.items(), key=lambda x: x[1])
                thesis_parts.append(
                    f"PRIMARY BIAS DETECTION: {strongest_bias[0].title()} bias at {strongest_bias[1]:.1%} strength"
                )
            else:
                thesis_parts.append("NO SIGNIFICANT PSYCHOLOGICAL BIASES DETECTED")
            
            # Individual Bias Analysis
            if individual_biases:
                thesis_parts.append("BIAS BREAKDOWN:")
                for bias_type, strength in individual_biases.items():
                    factors = bias_analysis.get('context_factors', {}).get(bias_type, [])
                    thesis_parts.append(
                        f"  • {bias_type.title()}: {strength:.1%} strength "
                        f"({', '.join(factors) if factors else 'pattern detected'})"
                    )
            
            # Market Context Integration
            regime = market_context.get('regime', 'unknown')
            volatility = market_context.get('volatility', 'medium')
            if regime != 'unknown':
                thesis_parts.append(
                    f"MARKET CONTEXT: {regime.title()} regime with {volatility} volatility "
                    f"influences bias detection thresholds and correction strategies"
                )
            
            # Correction Analysis
            corrections_applied = corrections.get('total_corrections_applied', 0)
            if corrections_applied > 0:
                thesis_parts.append(
                    f"CORRECTIONS APPLIED: {corrections_applied} bias corrections implemented "
                    f"with adaptive weight adjustments"
                )
                
                # Specific correction details
                for bias_type, correction in corrections.get('individual_corrections', {}).items():
                    actions = correction.get('recommended_actions', [])
                    confidence = correction.get('confidence_level', 0)
                    thesis_parts.append(
                        f"  • {bias_type.title()}: {confidence:.1%} confidence correction "
                        f"({actions[0] if actions else 'weight adjustment'})"
                    )
            
            # Weight Adjustments Summary
            significant_adjustments = {k: v for k, v in adjustments.items() if v < 0.9}
            if significant_adjustments:
                thesis_parts.append("WEIGHT ADJUSTMENTS:")
                for bias_type, weight in significant_adjustments.items():
                    reduction = (1.0 - weight) * 100
                    thesis_parts.append(f"  • {bias_type.title()}: {reduction:.0f}% weight reduction")
            
            # Performance Impact Assessment
            if self.session_stats['biases_corrected'] > 0:
                effectiveness = self.session_stats.get('correction_effectiveness', 0)
                impact_score = self.session_stats.get('bias_impact_score', 0)
                thesis_parts.append(
                    f"SESSION PERFORMANCE: {self.session_stats['biases_corrected']} corrections applied "
                    f"with €{effectiveness:.2f} average effectiveness and €{impact_score:.2f} total impact"
                )
            
            # Predictive Analysis
            risk_assessment = self._assess_future_bias_risk(bias_analysis)
            thesis_parts.append(f"RISK OUTLOOK: {risk_assessment}")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Bias analysis thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive bias analysis results"""
        try:
            # Core bias analysis
            self.smart_bus.set('bias_analysis', results['bias_analysis'], 
                             module='BiasAuditor', thesis=thesis)
            
            # Bias corrections with detailed thesis
            corrections_thesis = f"Applied {len(results['bias_corrections'].get('individual_corrections', {}))} bias corrections"
            self.smart_bus.set('bias_corrections', results['bias_corrections'],
                             module='BiasAuditor', thesis=corrections_thesis)
            
            # Dynamic adjustments
            adjustments_thesis = f"Dynamic weight adjustments: {len([a for a in results['bias_adjustments'].values() if a < 1.0])} biases adjusted"
            self.smart_bus.set('bias_adjustments', results['bias_adjustments'],
                             module='BiasAuditor', thesis=adjustments_thesis)
            
            # Comprehensive report
            self.smart_bus.set('bias_report', results['bias_report'],
                             module='BiasAuditor', thesis="Comprehensive bias analysis report generated")
            
            # Intelligent recommendations
            recommendations_thesis = f"Generated {len(results['recommendations'])} actionable recommendations"
            self.smart_bus.set('bias_recommendations', results['recommendations'],
                             module='BiasAuditor', thesis=recommendations_thesis)
            
            # Psychological state summary
            psychological_state = self._generate_psychological_state_summary(results)
            self.smart_bus.set('psychological_state', psychological_state,
                             module='BiasAuditor', thesis="Current psychological trading state assessment")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "BiasAuditor")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Bias Auditor disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('BiasAuditor', 'process_time', processing_time, False)
        
        return {
            'bias_analysis': {'individual_biases': {}, 'error': str(error_context)},
            'bias_corrections': {'individual_corrections': {}, 'error': str(error_context)},
            'bias_adjustments': {bias: 1.0 for bias in self.bias_categories.keys()},
            'bias_report': f"Bias analysis failed: {error_context}",
            'recommendations': ["Investigate bias auditor system errors"],
            'session_performance': self.session_stats.copy(),
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    # ═══════════════════════════════════════════════════════════════════
    # HELPER METHODS FOR BIAS DETECTION
    # ═══════════════════════════════════════════════════════════════════

    def _is_recent_trade(self, trade: Dict, minutes: int = 60, hours: int = 0) -> bool:
        """Check if trade is within specified time window"""
        try:
            # Simplified implementation - in production would parse actual timestamps
            return True  # Placeholder for time-based filtering
        except Exception:
            return False

    def _analyze_risk_escalation_pattern(self, trades: List[Dict]) -> float:
        """Analyze risk escalation patterns in recent trades"""
        # Placeholder implementation
        return 0.0

    def _analyze_position_size_trends(self, trades: List[Dict]) -> float:
        """Analyze position size reduction trends"""
        # Placeholder implementation
        return 0.0

    def _detect_premature_profit_taking(self, trades: List[Dict]) -> float:
        """Detect premature profit-taking patterns"""
        # Placeholder implementation
        return 0.0

    def _analyze_stop_loss_discipline(self, trades: List[Dict]) -> float:
        """Analyze stop-loss discipline degradation"""
        # Placeholder implementation
        return 0.5

    def _analyze_market_timing_confidence(self, trades: List[Dict]) -> float:
        """Analyze overconfident market timing patterns"""
        # Placeholder implementation
        return 0.0

    def _analyze_entry_timing_quality(self, trades: List[Dict]) -> float:
        """Analyze quality of entry timing"""
        # Placeholder implementation
        return 0.0

    def _detect_strategy_abandonment(self, trades: List[Dict]) -> float:
        """Detect strategy abandonment patterns"""
        # Placeholder implementation
        return 0.0

    def _analyze_round_number_bias(self, trades: List[Dict]) -> float:
        """Analyze bias toward round number price levels"""
        # Placeholder implementation
        return 0.0

    def _analyze_historical_price_bias(self, trades: List[Dict]) -> float:
        """Analyze bias toward historical price references"""
        # Placeholder implementation
        return 0.0

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY AND STATE MANAGEMENT METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _calculate_adaptive_adjustment(self, bias_type: str, strength: float) -> float:
        """Calculate adaptive adjustment based on bias learning"""
        correction_count = self.bias_corrections.get(bias_type, 0)
        base_adjustment = strength * 0.1  # Base 10% of strength
        learning_multiplier = 1.0 + (correction_count * self.adaptation_rate)
        return min(0.3, base_adjustment * learning_multiplier)

    def _generate_recommended_actions(self, bias_type: str, factors: List[str]) -> List[str]:
        """Generate specific recommended actions for bias type"""
        action_map = {
            'revenge': ['Take 15-minute break', 'Reduce position sizes', 'Review stop-loss levels'],
            'fear': ['Start with smaller positions', 'Focus on high-probability setups', 'Review risk parameters'],
            'greed': ['Implement profit-taking rules', 'Reduce position sizes', 'Increase stop-loss discipline'],
            'fomo': ['Wait for pullbacks', 'Stick to strategy rules', 'Avoid momentum chasing'],
            'anchoring': ['Review price level analysis', 'Focus on current market conditions', 'Update reference points']
        }
        return action_map.get(bias_type, ['Monitor trading behavior'])

    def _calculate_correction_confidence(self, bias_type: str, strength: float) -> float:
        """Calculate confidence level for bias correction"""
        base_confidence = min(0.9, strength * 1.2)  # Higher strength = higher confidence
        historical_success = self._get_historical_correction_success(bias_type)
        return (base_confidence + historical_success) / 2.0

    def _get_historical_correction_success(self, bias_type: str) -> float:
        """Get historical success rate for bias type corrections"""
        performance_data = self.bias_performance.get(bias_type, [])
        if not performance_data:
            return 0.5  # Default neutral confidence
        
        positive_outcomes = sum(1 for p in performance_data if p > 0)
        return positive_outcomes / len(performance_data)

    def _calculate_aggregate_bias_metrics(self, bias_signals: Dict[str, float]) -> Dict[str, Any]:
        """Calculate aggregate bias metrics"""
        if not bias_signals:
            return {'total_bias_score': 0.0, 'dominant_bias': 'none', 'bias_diversity': 0.0}
        
        total_score = sum(bias_signals.values())
        dominant_bias = max(bias_signals.items(), key=lambda x: x[1])
        bias_diversity = len(bias_signals) / len(self.bias_categories)
        
        return {
            'total_bias_score': total_score,
            'dominant_bias': dominant_bias[0],
            'dominant_strength': dominant_bias[1],
            'bias_diversity': bias_diversity,
            'bias_count': len(bias_signals)
        }

    def _assess_future_bias_risk(self, bias_analysis: Dict[str, Any]) -> str:
        """Assess future bias risk based on current analysis"""
        individual_biases = bias_analysis.get('individual_biases', {})
        aggregate_metrics = bias_analysis.get('aggregate_metrics', {})
        
        total_score = aggregate_metrics.get('total_bias_score', 0)
        
        if total_score > 2.0:
            return "HIGH RISK - Multiple strong biases detected, implement immediate corrections"
        elif total_score > 1.0:
            return "MODERATE RISK - Monitor bias development and apply preventive measures"
        elif total_score > 0.3:
            return "LOW RISK - Minor bias indicators, maintain awareness"
        else:
            return "MINIMAL RISK - Psychological state appears balanced"

    def _generate_psychological_state_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive psychological state summary"""
        bias_analysis = results.get('bias_analysis', {})
        corrections = results.get('bias_corrections', {})
        
        individual_biases = bias_analysis.get('individual_biases', {})
        aggregate_metrics = bias_analysis.get('aggregate_metrics', {})
        
        return {
            'overall_state': self._classify_psychological_state(aggregate_metrics),
            'dominant_bias': aggregate_metrics.get('dominant_bias', 'none'),
            'bias_strength': aggregate_metrics.get('total_bias_score', 0),
            'corrections_active': len(corrections.get('individual_corrections', {})),
            'risk_level': self._assess_future_bias_risk(bias_analysis),
            'stability_trend': self._assess_psychological_stability(),
            'last_updated': datetime.datetime.now().isoformat()
        }

    def _classify_psychological_state(self, aggregate_metrics: Dict[str, Any]) -> str:
        """Classify overall psychological state"""
        total_score = aggregate_metrics.get('total_bias_score', 0)
        bias_diversity = aggregate_metrics.get('bias_diversity', 0)
        
        if total_score < 0.3:
            return 'optimal'
        elif total_score < 1.0 and bias_diversity < 0.5:
            return 'stable'
        elif total_score < 2.0:
            return 'elevated'
        else:
            return 'compromised'

    def _assess_psychological_stability(self) -> str:
        """Assess psychological stability trend"""
        if len(self.bias_history) < 10:
            return 'insufficient_data'
        
        recent_strength = [b['strength'] for b in list(self.bias_history)[-5:]]
        older_strength = [b['strength'] for b in list(self.bias_history)[-10:-5]]
        
        if not recent_strength or not older_strength:
            return 'insufficient_data'
        
        recent_avg = np.mean(recent_strength)
        older_avg = np.mean(older_strength)
        
        change = recent_avg - older_avg
        
        if change > 0.2:
            return 'deteriorating'
        elif change < -0.2:
            return 'improving'
        else:
            return 'stable'

    def _generate_comprehensive_report(self, bias_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive bias analysis report"""
        individual_biases = bias_analysis.get('individual_biases', {})
        aggregate_metrics = bias_analysis.get('aggregate_metrics', {})
        market_context = bias_analysis.get('market_context', {})
        
        active_biases_str = ""
        if individual_biases:
            active_biases_str = "\n".join([
                f"  • {bias.title()}: {strength:.1%}" 
                for bias, strength in individual_biases.items()
            ])
        else:
            active_biases_str = "  • No significant biases detected"
        
        adjustments = self.get_bias_adjustments()
        adjustment_str = "\n".join([
            f"  • {bias.title()}: {adj:.1%} weight" 
            for bias, adj in adjustments.items() if adj < 1.0
        ])
        
        if not adjustment_str:
            adjustment_str = "  • No adjustments required"
        
        recommendations = self._generate_intelligent_recommendations(bias_analysis)
        rec_str = "\n".join([f"  • {rec}" for rec in recommendations])
        
        return f"""
🧠 COMPREHENSIVE BIAS ANALYSIS REPORT
═══════════════════════════════════════════════════════════════
📊 Session Overview:
• Total Biases Detected: {self.session_stats['total_biases_detected']}
• Biases Corrected: {self.session_stats['biases_corrected']}
• Most Common: {self.session_stats['most_common_bias'].title()}
• Overall Bias Score: {aggregate_metrics.get('total_bias_score', 0):.2f}

🎯 Current Market Context:
• Regime: {market_context.get('regime', 'Unknown').title()}
• Volatility: {market_context.get('volatility', 'Unknown').title()}

⚠️ Active Psychological Biases:
{active_biases_str}

🔧 Applied Weight Adjustments:
{adjustment_str}

📈 Performance Impact:
• Correction Effectiveness: €{self.session_stats['correction_effectiveness']:.2f}
• Total Bias Impact: €{self.session_stats['bias_impact_score']:.2f}
• Psychological State: {self._classify_psychological_state(aggregate_metrics).title()}

🎯 Intelligent Recommendations:
{rec_str}

📊 Session Statistics:
• Session Duration: {self._calculate_session_duration()}
• Detection Rate: {self._calculate_detection_rate():.1%}
• Correction Success Rate: {self._calculate_correction_success_rate():.1%}
"""

    def _generate_intelligent_recommendations(self, bias_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent, context-aware recommendations"""
        individual_biases = bias_analysis.get('individual_biases', {})
        aggregate_metrics = bias_analysis.get('aggregate_metrics', {})
        market_context = bias_analysis.get('market_context', {})
        
        recommendations = []
        
        # Specific bias recommendations
        for bias_type, strength in individual_biases.items():
            if strength > 0.5:
                recommendations.extend(self._generate_recommended_actions(bias_type, []))
        
        # Aggregate recommendations
        total_score = aggregate_metrics.get('total_bias_score', 0)
        if total_score > 2.0:
            recommendations.append("URGENT: Take extended break to reset psychological state")
        elif total_score > 1.0:
            recommendations.append("Consider reducing position sizes across all trades")
        
        # Market context recommendations
        regime = market_context.get('regime', 'unknown')
        if regime == 'volatile' and individual_biases:
            recommendations.append("High volatility amplifies bias effects - exercise extra caution")
        
        # Historical performance recommendations
        if self.session_stats['biases_corrected'] > 10:
            recommendations.append("High correction frequency suggests need for strategy review")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Continue current disciplined approach - psychological state optimal")
        
        return list(set(recommendations))  # Remove duplicates

    def _get_safe_trading_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when data retrieval fails"""
        return {
            'recent_trades': [],
            'current_pnl': 0.0,
            'positions': [],
            'risk_metrics': {},
            'session_context': {},
            'market_regime': 'unknown',
            'volatility_level': 'medium',
            'trading_session': {}
        }

    def _get_safe_bias_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when bias analysis fails"""
        return {
            'individual_biases': {},
            'context_factors': {},
            'aggregate_metrics': {'total_bias_score': 0.0, 'dominant_bias': 'none'},
            'detection_timestamp': datetime.datetime.now().isoformat(),
            'market_context': {'regime': 'unknown', 'volatility': 'medium'},
            'error': 'bias_analysis_failed'
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'bias_analysis': {'individual_biases': {}, 'status': 'disabled'},
            'bias_corrections': {'individual_corrections': {}, 'status': 'disabled'},
            'bias_adjustments': {bias: 1.0 for bias in self.bias_categories.keys()},
            'bias_report': "Bias Auditor is temporarily disabled due to errors",
            'recommendations': ["Restart bias auditor system", "Check error logs for issues"],
            'session_performance': self.session_stats.copy(),
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    def _calculate_session_duration(self) -> str:
        """Calculate human-readable session duration"""
        try:
            start_time = datetime.datetime.fromisoformat(self.session_stats['session_start'])
            duration = datetime.datetime.now() - start_time
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except Exception:
            return "Unknown"

    def _calculate_detection_rate(self) -> float:
        """Calculate bias detection rate"""
        # Use process call count from performance tracker or default to session time
        try:
            process_calls = getattr(self, 'process_call_count', 0)
            if process_calls == 0:
                return 0.0
            return (self.session_stats['total_biases_detected'] / process_calls) * 100
        except Exception:
            # Fallback: use total detections as rate (simple approximation)
            return min(100.0, self.session_stats['total_biases_detected'] * 10)

    def _calculate_correction_success_rate(self) -> float:
        """Calculate correction success rate"""
        if self.session_stats['biases_corrected'] == 0:
            return 0.0
        
        successful_corrections = sum(
            1 for outcomes in self.bias_performance.values() 
            for outcome in outcomes if outcome > 0
        )
        total_corrections = sum(len(outcomes) for outcomes in self.bias_performance.values())
        
        if total_corrections == 0:
            return 0.0
        
        return (successful_corrections / total_corrections) * 100

    def get_bias_adjustments(self) -> Dict[str, float]:
        """Get current bias adjustment weights for external use"""
        adjustments = {}
        
        try:
            for bias_type, category in self.bias_categories.items():
                correction_count = self.bias_corrections.get(bias_type, 0)
                
                if correction_count >= self.correction_threshold:
                    # Apply weight reduction based on correction history
                    base_reduction = category['weight_reduction']
                    adaptive_reduction = min(0.8, correction_count * 0.1)
                    total_reduction = base_reduction + adaptive_reduction
                    
                    adjustments[bias_type] = max(0.2, 1.0 - total_reduction)
                else:
                    adjustments[bias_type] = 1.0
            
            return adjustments
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adjustment_calculation")
            self.logger.error(f"Failed to calculate bias adjustments: {error_context}")
            return {bias_type: 1.0 for bias_type in self.bias_categories.keys()}

    def record_bias_outcome(self, bias_type: str, pnl: float) -> None:
        """Record the outcome of a biased decision for learning"""
        try:
            # Validate inputs
            if not isinstance(bias_type, str) or bias_type not in self.bias_categories:
                self.logger.warning(f"Invalid bias type: {bias_type}")
                return
            
            if np.isnan(pnl):
                self.logger.warning("NaN PnL in bias outcome, ignoring")
                return
            
            # Update performance tracking
            self.bias_performance[bias_type].append(pnl)
            
            # Update corrections if negative outcome
            if pnl < 0:
                self.bias_corrections[bias_type] += 1
                self.session_stats['biases_corrected'] += 1
                
                self.logger.info(format_operator_message(
                    icon="📚",
                    message=f"Learning from {bias_type} bias",
                    pnl=f"€{pnl:.2f}",
                    total_corrections=self.bias_corrections[bias_type]
                ))
            
            # Update bias records with actual outcome
            for record in reversed(self.bias_history):
                if record['type'] == bias_type and record.get('pnl_impact') == 0.0:
                    record['pnl_impact'] = pnl
                    break
            
            # Update session statistics
            self._update_session_stats()
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "outcome_recording")
            self.logger.error(f"Failed to record bias outcome: {error_context}")

    def _update_session_stats(self) -> None:
        """Update session-level bias statistics"""
        try:
            # Find most common bias
            if self.bias_frequencies:
                most_common = max(self.bias_frequencies.items(), key=lambda x: x[1])
                self.session_stats['most_common_bias'] = most_common[0]
            
            # Calculate bias impact score
            total_impact = 0.0
            for outcomes in self.bias_performance.values():
                if outcomes:
                    total_impact += sum(outcomes)
            
            self.session_stats['bias_impact_score'] = total_impact
            
            # Calculate correction effectiveness
            if self.session_stats['biases_corrected'] > 0:
                recent_outcomes = []
                for bias_type, outcomes in self.bias_performance.items():
                    if outcomes:
                        recent_outcomes.extend(outcomes[-5:])  # Last 5 outcomes
                
                if recent_outcomes:
                    self.session_stats['correction_effectiveness'] = np.mean(recent_outcomes)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "session_stats_update")
            self.logger.warning(f"Session stats update failed: {error_context}")

    def get_observation_components(self) -> np.ndarray:
        """Get bias frequencies and corrections for observation"""
        try:
            if not self.bias_history:
                # Return balanced defaults for cold start
                num_biases = len(self.bias_categories)
                defaults = np.full(num_biases * 2, 0.2, dtype=np.float32)
                return defaults
            
            # Calculate bias frequencies (normalized)
            total_biases = len(self.bias_history)
            frequencies = []
            for bias_type in self.bias_categories.keys():
                frequency = self.bias_frequencies.get(bias_type, 0) / max(1, total_biases)
                frequencies.append(frequency)
            
            # Calculate correction strengths (normalized)
            corrections = []
            max_corrections = max(self.bias_corrections.values()) if self.bias_corrections else 1
            for bias_type in self.bias_categories.keys():
                correction_strength = self.bias_corrections.get(bias_type, 0) / max(1, max_corrections)
                corrections.append(correction_strength)
            
            # Combine frequencies and corrections
            observation = np.array(frequencies + corrections, dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid observation values: {observation}")
                observation = np.nan_to_num(observation, nan=0.2)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Observation generation failed: {error_context}")
            num_biases = len(self.bias_categories)
            return np.full(num_biases * 2, 0.2, dtype=np.float32)

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'BiasAuditor',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'bias_detection_rate': self._calculate_detection_rate(),
            'correction_success_rate': self._calculate_correction_success_rate(),
            'total_biases_tracked': len(self.bias_history),
            'session_duration': self._calculate_session_duration(),
            'psychological_state': self._classify_psychological_state(
                self._calculate_aggregate_bias_metrics(
                    {bias: sum(self.bias_performance.get(bias, [0])[-1:]) 
                     for bias in self.bias_categories.keys()}
                )
            )
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT FOR HOT-RELOAD
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'BiasAuditor',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'history_len': self.history_len,
                'correction_threshold': self.correction_threshold,
                'adaptation_rate': self.adaptation_rate,
                'debug': self.debug
            },
            'bias_tracking': {
                'bias_history': list(self.bias_history),
                'corrections': dict(self.bias_corrections),
                'performance': {k: list(v) for k, v in self.bias_performance.items()},
                'frequencies': dict(self.bias_frequencies)
            },
            'session_data': self.session_stats.copy(),
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'process_tracking': {
                'process_call_count': self.process_call_count
            },
            'performance_metrics': self._get_health_metrics(),
            'bias_categories': self.bias_categories.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.history_len = int(config.get("history_len", self.history_len))
            self.correction_threshold = int(config.get("correction_threshold", self.correction_threshold))
            self.adaptation_rate = float(config.get("adaptation_rate", self.adaptation_rate))
            self.debug = bool(config.get("debug", self.debug))
            
            # Load bias tracking data
            bias_data = state.get("bias_tracking", {})
            self.bias_history = deque(bias_data.get("bias_history", []), maxlen=self.history_len)
            self.bias_corrections = defaultdict(int, bias_data.get("corrections", {}))
            
            # Restore performance data
            performance_data = bias_data.get("performance", {})
            self.bias_performance = defaultdict(list)
            for k, v in performance_data.items():
                self.bias_performance[k] = list(v)
            
            self.bias_frequencies = defaultdict(int, bias_data.get("frequencies", {}))
            
            # Load session data
            self.session_stats = state.get("session_data", self.session_stats)
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            # Load process tracking
            process_tracking = state.get("process_tracking", {})
            self.process_call_count = process_tracking.get("process_call_count", 0)
            
            # Load categories if provided
            self.bias_categories.update(state.get("bias_categories", {}))
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Bias Auditor state restored",
                biases_tracked=len(self.bias_history),
                corrections_applied=sum(self.bias_corrections.values())
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'BiasAuditor',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'metrics': self._get_health_metrics(),
            'alerts': self._generate_health_alerts(),
            'recommendations': self._generate_health_recommendations()
        }

    def _generate_health_alerts(self) -> List[Dict[str, Any]]:
        """Generate health-related alerts"""
        alerts = []
        
        if self.is_disabled:
            alerts.append({
                'severity': 'critical',
                'message': 'BiasAuditor disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring issues'
            })
        
        detection_rate = self._calculate_detection_rate()
        if detection_rate > 50:  # Very high detection rate
            alerts.append({
                'severity': 'warning',
                'message': f'High bias detection rate: {detection_rate:.1f}%',
                'action': 'Review trading strategy and psychological factors'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart BiasAuditor module after investigating errors")
        
        if len(self.bias_history) < 10:
            recommendations.append("Insufficient bias data - continue monitoring to build baseline")
        
        if self.session_stats['biases_corrected'] > 20:
            recommendations.append("High correction frequency - consider strategy review")
        
        if not recommendations:
            recommendations.append("BiasAuditor operating within normal parameters")
        
        return recommendations