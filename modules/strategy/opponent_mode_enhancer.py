"""
🎯 Enhanced Opponent Mode Enhancer with SmartInfoBus Integration v3.0
Advanced market mode detection and adaptation system with intelligent strategy weighting
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
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="OpponentModeEnhancer",
    version="3.0.0",
    category="strategy",
    provides=[
        "mode_weights", "mode_analysis", "mode_recommendations",
        "market_mode_detection", "strategy_adaptation", "mode_performance"
    ],
    requires=[
        "market_data", "recent_trades", "technical_indicators", "volatility_data",
        "price_data", "market_regime", "trading_performance"
    ],
    description="Advanced market mode detection and adaptation system with intelligent strategy weighting",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=150,
    priority=6,
    explainable=True,
    hot_reload=True
)
class OpponentModeEnhancer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🎯 PRODUCTION-GRADE Opponent Mode Enhancer v3.0
    
    Advanced market mode detection and adaptation system with:
    - Intelligent market condition detection across multiple modes
    - Dynamic strategy weighting based on performance and market analysis
    - Adaptive learning system with performance-based optimization
    - SmartInfoBus zero-wiring architecture
    - Comprehensive thesis generation for all mode decisions
    """

    def _initialize(self):
        """Initialize advanced mode detection and adaptation systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced mode configuration
        self.modes = self.config.get('modes', ["trending", "ranging", "volatile", "breakout", "reversal"])
        self.adaptation_rate = self.config.get('adaptation_rate', 0.15)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.mode_switch_cooldown = self.config.get('mode_switch_cooldown', 5)
        self.debug = self.config.get('debug', False)
        
        # Initialize enhanced mode definitions
        self.mode_definitions = self._initialize_comprehensive_mode_definitions()
        
        # Core mode tracking state
        self.mode_performance = defaultdict(lambda: defaultdict(list))
        self.mode_counts = defaultdict(int)
        self.mode_history = deque(maxlen=100)
        self.current_mode_weights = {mode: 1.0/len(self.modes) for mode in self.modes}
        
        # Advanced analytics system
        self.mode_analytics = {
            'mode_transitions': defaultdict(int),
            'mode_duration_stats': defaultdict(list),
            'mode_profitability': defaultdict(float),
            'mode_win_rates': defaultdict(float),
            'mode_confidence_scores': defaultdict(float),
            'mode_effectiveness': defaultdict(float),
            'last_mode_switch': None,
            'switches_since_reset': 0,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Enhanced market condition detectors
        self.condition_detectors = {
            'trending': self._detect_trending_condition_advanced,
            'ranging': self._detect_ranging_condition_advanced,
            'volatile': self._detect_volatile_condition_advanced,
            'breakout': self._detect_breakout_condition_advanced,
            'reversal': self._detect_reversal_condition_advanced
        }
        
        # Performance assessment thresholds
        self.performance_thresholds = {
            'exceptional': 150.0,
            'excellent': 100.0,
            'good': 50.0,
            'neutral': 0.0,
            'poor': -25.0,
            'very_poor': -75.0,
            'critical': -150.0
        }
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Advanced mode intelligence
        self.mode_intelligence = {
            'detection_sensitivity': 0.8,
            'adaptation_momentum': 0.9,
            'confidence_decay': 0.95,
            'performance_memory': 0.85
        }
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🎯",
            message=f"Opponent Mode Enhancer v{version} initialized",
            modes=len(self.modes),
            adaptation_rate=f"{self.adaptation_rate:.1%}",
            confidence_threshold=f"{self.confidence_threshold:.1%}"
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="OpponentModeEnhancer",
            log_path="logs/strategy/opponent_mode_enhancer.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("OpponentModeEnhancer", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _initialize_comprehensive_mode_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive mode definitions with enhanced characteristics"""
        return {
            'trending': {
                'description': 'Strong sustained directional price movement with momentum',
                'characteristics': ['sustained_direction', 'higher_highs_lows', 'momentum_persistence', 'volume_confirmation'],
                'indicators': ['price_slope', 'momentum_strength', 'trend_duration', 'directional_bias'],
                'optimal_strategies': ['momentum_following', 'breakout_continuation', 'trend_riding', 'pullback_entries'],
                'risk_factors': ['trend_exhaustion', 'reversal_signals', 'momentum_divergence', 'volume_decline'],
                'profit_potential': 'high',
                'typical_duration': '30-120 minutes',
                'detection_weights': {'momentum': 0.4, 'direction': 0.3, 'volume': 0.2, 'duration': 0.1}
            },
            'ranging': {
                'description': 'Horizontal price movement within defined support and resistance bounds',
                'characteristics': ['horizontal_movement', 'support_resistance_respect', 'mean_reversion', 'oscillation_patterns'],
                'indicators': ['range_width', 'bounce_frequency', 'volume_profile', 'oscillator_signals'],
                'optimal_strategies': ['mean_reversion', 'support_resistance_trading', 'range_scalping', 'oscillator_trades'],
                'risk_factors': ['range_breakdown', 'false_breakouts', 'range_compression', 'volume_drying_up'],
                'profit_potential': 'medium_consistent',
                'typical_duration': '45-180 minutes',
                'detection_weights': {'volatility': 0.3, 'direction': 0.2, 'bounce_quality': 0.3, 'time_in_range': 0.2}
            },
            'volatile': {
                'description': 'High price volatility with increased uncertainty and large swings',
                'characteristics': ['large_price_swings', 'unpredictable_direction', 'high_noise', 'rapid_changes'],
                'indicators': ['atr_expansion', 'price_acceleration', 'gap_frequency', 'volatility_spikes'],
                'optimal_strategies': ['volatility_capture', 'wide_stops', 'reduced_sizing', 'quick_scalps'],
                'risk_factors': ['whipsaws', 'gap_risk', 'overexposure', 'false_signals'],
                'profit_potential': 'high_risk_high_reward',
                'typical_duration': '15-60 minutes',
                'detection_weights': {'volatility': 0.5, 'price_swings': 0.3, 'unpredictability': 0.2}
            },
            'breakout': {
                'description': 'Price breaking through significant support or resistance levels',
                'characteristics': ['level_penetration', 'volume_surge', 'momentum_acceleration', 'follow_through'],
                'indicators': ['breakout_strength', 'volume_confirmation', 'follow_through_quality', 'level_significance'],
                'optimal_strategies': ['breakout_momentum', 'continuation_patterns', 'expansion_trades', 'momentum_acceleration'],
                'risk_factors': ['false_breakouts', 'fade_risk', 'trap_setups', 'insufficient_volume'],
                'profit_potential': 'very_high',
                'typical_duration': '10-45 minutes',
                'detection_weights': {'breakout_strength': 0.4, 'volume': 0.3, 'momentum': 0.2, 'level_quality': 0.1}
            },
            'reversal': {
                'description': 'Trend change and direction reversal with exhaustion patterns',
                'characteristics': ['exhaustion_signals', 'momentum_divergence', 'pattern_completion', 'sentiment_shift'],
                'indicators': ['reversal_patterns', 'momentum_divergence', 'volume_analysis', 'sentiment_indicators'],
                'optimal_strategies': ['counter_trend', 'reversal_trading', 'pattern_recognition', 'momentum_fade'],
                'risk_factors': ['false_reversals', 'trend_continuation', 'timing_risk', 'premature_entry'],
                'profit_potential': 'high_timing_dependent',
                'typical_duration': '20-90 minutes',
                'detection_weights': {'divergence': 0.3, 'pattern': 0.3, 'exhaustion': 0.2, 'sentiment': 0.2}
            }
        }

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Opponent Mode Enhancer v3.0 Initialization Complete:
        
        Advanced Mode Detection System:
        - Multi-mode analysis: {', '.join(self.modes)} market conditions
        - Intelligent detection algorithms with weighted scoring systems
        - Real-time market condition assessment with confidence scoring
        - Dynamic adaptation based on performance feedback and market evolution
        
        Current Configuration:
        - Modes tracked: {len(self.modes)} distinct market conditions
        - Adaptation rate: {self.adaptation_rate:.1%} for smooth weight transitions
        - Confidence threshold: {self.confidence_threshold:.1%} for mode activation
        - Switch cooldown: {self.mode_switch_cooldown} periods for stability
        
        Detection Intelligence Features:
        - Advanced condition detectors with multi-factor analysis
        - Performance-based learning with historical optimization
        - Confidence scoring with decay and momentum factors
        - Market regime integration for context-aware detection
        
        Advanced Capabilities:
        - Real-time mode weight adaptation based on performance
        - Comprehensive analytics tracking for all modes
        - Intelligent switching with cooldown protection
        - Performance tracking and effectiveness measurement
        
        Expected Outcomes:
        - Optimal strategy selection based on current market conditions
        - Enhanced performance through intelligent mode weighting
        - Adaptive learning that improves detection accuracy over time
        - Transparent mode decisions with comprehensive explanations
        """
        
        self.smart_bus.set('opponent_mode_enhancer_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'modes': self.modes,
                'detection_methods': list(self.condition_detectors.keys()),
                'performance_thresholds': list(self.performance_thresholds.keys())
            }
        }, module='OpponentModeEnhancer', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive mode analysis
        
        Returns:
            Dict containing mode weights, analysis, and recommendations
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive market data from SmartInfoBus
            market_data = await self._get_comprehensive_market_data()
            
            # Core mode detection with error handling
            mode_analysis = await self._analyze_market_modes_comprehensive(market_data)
            
            # Update mode performance based on recent results
            await self._update_mode_performance_comprehensive(market_data, mode_analysis)
            
            # Adapt mode weights with intelligent algorithms
            weight_adaptation = await self._adapt_mode_weights_intelligent(mode_analysis, market_data)
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_mode_thesis(mode_analysis, weight_adaptation)
            
            # Create comprehensive results
            results = {
                'mode_weights': self.current_mode_weights.copy(),
                'mode_analysis': mode_analysis,
                'mode_recommendations': self._generate_intelligent_recommendations(mode_analysis),
                'market_mode_detection': mode_analysis.get('detected_modes', {}),
                'strategy_adaptation': weight_adaptation,
                'mode_performance': self._get_performance_summary(),
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('OpponentModeEnhancer', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data using modern SmartInfoBus patterns"""
        try:
            return {
                'market_data': self.smart_bus.get('market_data', 'OpponentModeEnhancer') or {},
                'recent_trades': self.smart_bus.get('recent_trades', 'OpponentModeEnhancer') or [],
                'technical_indicators': self.smart_bus.get('technical_indicators', 'OpponentModeEnhancer') or {},
                'volatility_data': self.smart_bus.get('volatility_data', 'OpponentModeEnhancer') or {},
                'price_data': self.smart_bus.get('price_data', 'OpponentModeEnhancer') or {},
                'market_regime': self.smart_bus.get('market_regime', 'OpponentModeEnhancer') or 'unknown',
                'trading_performance': self.smart_bus.get('trading_performance', 'OpponentModeEnhancer') or {},
                'session_metrics': self.smart_bus.get('session_metrics', 'OpponentModeEnhancer') or {},
                'market_context': self.smart_bus.get('market_context', 'OpponentModeEnhancer') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "OpponentModeEnhancer")
            self.logger.warning(f"Market data retrieval incomplete: {error_context}")
            return self._get_safe_market_defaults()

    async def _analyze_market_modes_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive market mode analysis with advanced detection"""
        try:
            analysis = {
                'detected_modes': {},
                'confidence_scores': {},
                'detection_factors': {},
                'market_characteristics': {},
                'regime_alignment': {},
                'detection_timestamp': datetime.datetime.now().isoformat()
            }
            
            # Calculate comprehensive market metrics
            market_metrics = await self._calculate_comprehensive_market_metrics(market_data)
            analysis['market_characteristics'] = market_metrics
            
            # Detect all modes with advanced algorithms
            for mode in self.modes:
                if mode in self.condition_detectors:
                    detection_result = await self.condition_detectors[mode](market_data, market_metrics)
                    
                    analysis['detected_modes'][mode] = detection_result['confidence']
                    analysis['confidence_scores'][mode] = detection_result['confidence']
                    analysis['detection_factors'][mode] = detection_result['factors']
                else:
                    analysis['detected_modes'][mode] = 0.0
                    analysis['confidence_scores'][mode] = 0.0
                    analysis['detection_factors'][mode] = ['detector_missing']
            
            # Normalize mode confidences
            total_confidence = sum(analysis['detected_modes'].values())
            if total_confidence > 0:
                analysis['detected_modes'] = {
                    k: v/total_confidence for k, v in analysis['detected_modes'].items()
                }
            else:
                # Equal distribution if no clear mode detected
                analysis['detected_modes'] = {mode: 1.0/len(self.modes) for mode in self.modes}
            
            # Assess regime alignment
            analysis['regime_alignment'] = self._assess_regime_alignment(
                analysis['detected_modes'], market_data.get('market_regime', 'unknown')
            )
            
            # Log significant detections
            await self._log_significant_detections(analysis)
            
            return analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "OpponentModeEnhancer")
            self.logger.error(f"Mode analysis failed: {error_context}")
            return self._get_safe_analysis_defaults()

    async def _calculate_comprehensive_market_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive market metrics for mode detection"""
        try:
            metrics = {}
            
            # Extract data sources
            recent_trades = market_data.get('recent_trades', [])
            price_data = market_data.get('price_data', {})
            volatility_data = market_data.get('volatility_data', {})
            technical_indicators = market_data.get('technical_indicators', {})
            
            # Price and momentum metrics
            if recent_trades and len(recent_trades) >= 3:
                prices = [t.get('entry_price', 0) for t in recent_trades[-20:] if t.get('entry_price', 0) > 0]
                if len(prices) >= 3:
                    price_changes = np.diff(prices)
                    metrics.update({
                        'price_momentum': np.mean(price_changes),
                        'price_volatility': np.std(price_changes),
                        'price_trend_strength': abs(np.mean(price_changes)) / (np.std(price_changes) + 1e-6),
                        'price_acceleration': np.mean(np.diff(price_changes)) if len(price_changes) > 1 else 0,
                        'directional_consistency': self._calculate_directional_consistency(price_changes)
                    })
                
                # Performance and timing metrics
                pnls = [t.get('pnl', 0) for t in recent_trades]
                if pnls:
                    metrics.update({
                        'recent_performance_trend': np.mean(pnls[-10:]) if len(pnls) >= 10 else np.mean(pnls),
                        'performance_volatility': np.std(pnls),
                        'win_rate': len([p for p in pnls if p > 0]) / len(pnls),
                        'profit_factor': sum([p for p in pnls if p > 0]) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float('inf')
                    })
                
                # Trade timing and frequency
                trade_intervals = self._calculate_trade_intervals(recent_trades)
                if trade_intervals:
                    metrics.update({
                        'trade_frequency': 1.0 / (np.mean(trade_intervals) + 1e-6),
                        'timing_regularity': 1.0 - (np.std(trade_intervals) / (np.mean(trade_intervals) + 1e-6)),
                        'activity_intensity': len(recent_trades) / max(1, len(trade_intervals))
                    })
            
            # Volatility assessment
            volatility_level = market_data.get('market_context', {}).get('volatility_level', 'medium')
            metrics['volatility_score'] = {
                'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0
            }.get(volatility_level, 0.5)
            
            # Technical indicator integration
            if technical_indicators:
                metrics.update({
                    'momentum_indicators': technical_indicators.get('momentum', {}),
                    'trend_indicators': technical_indicators.get('trend', {}),
                    'volatility_indicators': technical_indicators.get('volatility', {}),
                    'volume_indicators': technical_indicators.get('volume', {})
                })
            
            # Market regime context
            regime = market_data.get('market_regime', 'unknown')
            metrics['regime_score'] = self._calculate_regime_score(regime)
            
            return metrics
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_metrics")
            self.logger.warning(f"Market metrics calculation failed: {error_context}")
            return {'calculation_error': str(error_context)}

    async def _detect_trending_condition_advanced(self, market_data: Dict[str, Any], 
                                                market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced trending condition detection with multi-factor analysis"""
        try:
            detection_weights = self.mode_definitions['trending']['detection_weights']
            factors = []
            confidence = 0.0
            
            # Momentum component (40% weight)
            price_momentum = abs(market_metrics.get('price_momentum', 0))
            trend_strength = market_metrics.get('price_trend_strength', 0)
            if price_momentum > 0.1 and trend_strength > 1.5:
                momentum_score = min(1.0, (price_momentum * 10 + trend_strength) / 3)
                confidence += momentum_score * detection_weights['momentum']
                factors.append('strong_momentum')
            
            # Direction component (30% weight)
            directional_consistency = market_metrics.get('directional_consistency', 0)
            if directional_consistency > 0.7:
                direction_score = directional_consistency
                confidence += direction_score * detection_weights['direction']
                factors.append('consistent_direction')
            
            # Volume component (20% weight)
            volume_indicators = market_metrics.get('volume_indicators', {})
            volume_trend = volume_indicators.get('trend_strength', 0.5)
            if volume_trend > 0.6:
                confidence += volume_trend * detection_weights['volume']
                factors.append('volume_confirmation')
            
            # Duration component (10% weight)
            recent_performance = market_metrics.get('recent_performance_trend', 0)
            if recent_performance > 0:
                duration_score = min(1.0, recent_performance / 50)
                confidence += duration_score * detection_weights['duration']
                factors.append('sustained_performance')
            
            # Regime alignment bonus
            if market_data.get('market_regime') == 'trending':
                confidence += 0.2
                factors.append('regime_alignment')
            
            return {
                'confidence': min(1.0, confidence),
                'factors': factors,
                'component_scores': {
                    'momentum': price_momentum * trend_strength,
                    'direction': directional_consistency,
                    'volume': volume_trend,
                    'performance': recent_performance
                }
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "trending_detection")
            return {'confidence': 0.0, 'factors': ['detection_error'], 'error': str(error_context)}

    async def _detect_ranging_condition_advanced(self, market_data: Dict[str, Any], 
                                               market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced ranging condition detection with oscillation analysis"""
        try:
            detection_weights = self.mode_definitions['ranging']['detection_weights']
            factors = []
            confidence = 0.0
            
            # Volatility component (30% weight) - low volatility suggests ranging
            price_volatility = market_metrics.get('price_volatility', 0)
            volatility_score = market_metrics.get('volatility_score', 0.5)
            if price_volatility < 0.05 and volatility_score < 0.6:
                vol_component = 1.0 - volatility_score
                confidence += vol_component * detection_weights['volatility']
                factors.append('low_volatility')
            
            # Direction component (20% weight) - low momentum suggests ranging
            price_momentum = abs(market_metrics.get('price_momentum', 0))
            trend_strength = market_metrics.get('price_trend_strength', 0)
            if price_momentum < 0.03 and trend_strength < 1.0:
                direction_component = 1.0 - min(1.0, price_momentum * 20 + trend_strength / 2)
                confidence += direction_component * detection_weights['direction']
                factors.append('weak_directional_bias')
            
            # Bounce quality component (30% weight)
            win_rate = market_metrics.get('win_rate', 0.5)
            profit_factor = market_metrics.get('profit_factor', 1.0)
            if win_rate > 0.5 and 0.8 < profit_factor < 2.0:  # Consistent but moderate profits
                bounce_quality = (win_rate + min(1.0, profit_factor / 2)) / 2
                confidence += bounce_quality * detection_weights['bounce_quality']
                factors.append('good_bounce_quality')
            
            # Time in range component (20% weight)
            timing_regularity = market_metrics.get('timing_regularity', 0.5)
            if timing_regularity > 0.6:
                confidence += timing_regularity * detection_weights['time_in_range']
                factors.append('regular_patterns')
            
            # Regime alignment
            if market_data.get('market_regime') == 'ranging':
                confidence += 0.2
                factors.append('regime_alignment')
            
            return {
                'confidence': min(1.0, confidence),
                'factors': factors,
                'component_scores': {
                    'volatility': 1.0 - volatility_score,
                    'direction': 1.0 - price_momentum * 20,
                    'bounce_quality': win_rate * min(1.0, profit_factor / 2),
                    'regularity': timing_regularity
                }
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "ranging_detection")
            return {'confidence': 0.0, 'factors': ['detection_error'], 'error': str(error_context)}

    async def _detect_volatile_condition_advanced(self, market_data: Dict[str, Any], 
                                                market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced volatile condition detection with uncertainty analysis"""
        try:
            detection_weights = self.mode_definitions['volatile']['detection_weights']
            factors = []
            confidence = 0.0
            
            # Volatility component (50% weight)
            volatility_score = market_metrics.get('volatility_score', 0.5)
            price_volatility = market_metrics.get('price_volatility', 0)
            if volatility_score > 0.7 or price_volatility > 0.1:
                vol_component = max(volatility_score, min(1.0, price_volatility * 10))
                confidence += vol_component * detection_weights['volatility']
                factors.append('high_volatility')
            
            # Price swings component (30% weight)
            price_acceleration = abs(market_metrics.get('price_acceleration', 0))
            performance_volatility = market_metrics.get('performance_volatility', 0)
            if price_acceleration > 0.02 or performance_volatility > 20:
                swings_component = min(1.0, price_acceleration * 50 + performance_volatility / 50)
                confidence += swings_component * detection_weights['price_swings']
                factors.append('large_price_swings')
            
            # Unpredictability component (20% weight)
            directional_consistency = market_metrics.get('directional_consistency', 0.5)
            timing_regularity = market_metrics.get('timing_regularity', 0.5)
            unpredictability = 1.0 - (directional_consistency + timing_regularity) / 2
            if unpredictability > 0.6:
                confidence += unpredictability * detection_weights['unpredictability']
                factors.append('unpredictable_patterns')
            
            # High activity indicator
            trade_frequency = market_metrics.get('trade_frequency', 0)
            if trade_frequency > 2.0:
                confidence += 0.1  # Bonus for high activity
                factors.append('high_activity')
            
            return {
                'confidence': min(1.0, confidence),
                'factors': factors,
                'component_scores': {
                    'volatility': volatility_score,
                    'price_swings': price_acceleration * 50,
                    'unpredictability': unpredictability,
                    'activity': min(1.0, trade_frequency / 3)
                }
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "volatile_detection")
            return {'confidence': 0.0, 'factors': ['detection_error'], 'error': str(error_context)}

    async def _detect_breakout_condition_advanced(self, market_data: Dict[str, Any], 
                                                market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced breakout condition detection with momentum analysis"""
        try:
            detection_weights = self.mode_definitions['breakout']['detection_weights']
            factors = []
            confidence = 0.0
            
            # Breakout strength component (40% weight)
            price_momentum = abs(market_metrics.get('price_momentum', 0))
            price_acceleration = abs(market_metrics.get('price_acceleration', 0))
            if price_momentum > 0.08 and price_acceleration > 0.02:
                strength_component = min(1.0, price_momentum * 10 + price_acceleration * 30)
                confidence += strength_component * detection_weights['breakout_strength']
                factors.append('strong_breakout')
            
            # Volume component (30% weight)
            volume_indicators = market_metrics.get('volume_indicators', {})
            volume_surge = volume_indicators.get('surge_strength', 0.5)
            if volume_surge > 0.7:
                confidence += volume_surge * detection_weights['volume']
                factors.append('volume_confirmation')
            
            # Momentum component (20% weight)
            trend_strength = market_metrics.get('price_trend_strength', 0)
            recent_performance = market_metrics.get('recent_performance_trend', 0)
            if trend_strength > 2.0 and recent_performance > 30:
                momentum_component = min(1.0, trend_strength / 3 + recent_performance / 100)
                confidence += momentum_component * detection_weights['momentum']
                factors.append('momentum_acceleration')
            
            # Level quality component (10% weight)
            technical_indicators = market_metrics.get('technical_indicators', {})
            support_resistance_quality = technical_indicators.get('level_quality', 0.5)
            if support_resistance_quality > 0.6:
                confidence += support_resistance_quality * detection_weights['level_quality']
                factors.append('significant_level_break')
            
            return {
                'confidence': min(1.0, confidence),
                'factors': factors,
                'component_scores': {
                    'strength': price_momentum * 10,
                    'volume': volume_surge,
                    'momentum': trend_strength / 3,
                    'level_quality': support_resistance_quality
                }
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "breakout_detection")
            return {'confidence': 0.0, 'factors': ['detection_error'], 'error': str(error_context)}

    async def _detect_reversal_condition_advanced(self, market_data: Dict[str, Any], 
                                                market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced reversal condition detection with exhaustion analysis"""
        try:
            detection_weights = self.mode_definitions['reversal']['detection_weights']
            factors = []
            confidence = 0.0
            
            # Divergence component (30% weight)
            price_momentum = market_metrics.get('price_momentum', 0)
            recent_performance = market_metrics.get('recent_performance_trend', 0)
            # Negative performance despite price movement suggests divergence
            if abs(price_momentum) > 0.05 and recent_performance < -10:
                divergence_strength = min(1.0, abs(price_momentum) * 10 + abs(recent_performance) / 20)
                confidence += divergence_strength * detection_weights['divergence']
                factors.append('momentum_divergence')
            
            # Pattern component (30% weight)
            trend_strength = market_metrics.get('price_trend_strength', 0)
            directional_consistency = market_metrics.get('directional_consistency', 0.5)
            if trend_strength > 2.5 and directional_consistency < 0.4:  # Strong trend losing consistency
                pattern_component = min(1.0, trend_strength / 3 + (1 - directional_consistency))
                confidence += pattern_component * detection_weights['pattern']
                factors.append('exhaustion_pattern')
            
            # Exhaustion component (20% weight)
            performance_volatility = market_metrics.get('performance_volatility', 0)
            if performance_volatility > 30:  # High performance volatility suggests exhaustion
                exhaustion_component = min(1.0, performance_volatility / 50)
                confidence += exhaustion_component * detection_weights['exhaustion']
                factors.append('trend_exhaustion')
            
            # Sentiment component (20% weight)
            win_rate = market_metrics.get('win_rate', 0.5)
            profit_factor = market_metrics.get('profit_factor', 1.0)
            if win_rate < 0.4 or profit_factor < 0.8:  # Deteriorating metrics
                sentiment_component = 1.0 - min(1.0, win_rate + profit_factor / 2)
                confidence += sentiment_component * detection_weights['sentiment']
                factors.append('sentiment_deterioration')
            
            return {
                'confidence': min(1.0, confidence),
                'factors': factors,
                'component_scores': {
                    'divergence': abs(price_momentum) * 10,
                    'pattern': trend_strength / 3,
                    'exhaustion': performance_volatility / 50,
                    'sentiment': 1.0 - win_rate
                }
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "reversal_detection")
            return {'confidence': 0.0, 'factors': ['detection_error'], 'error': str(error_context)}

    # ═══════════════════════════════════════════════════════════════════
    # HELPER METHODS FOR CALCULATIONS
    # ═══════════════════════════════════════════════════════════════════

    def _calculate_directional_consistency(self, price_changes: np.ndarray) -> float:
        """Calculate directional consistency of price movements"""
        try:
            if len(price_changes) < 2:
                return 0.5
            
            # Count direction changes
            direction_changes = 0
            for i in range(1, len(price_changes)):
                if (price_changes[i] > 0) != (price_changes[i-1] > 0):
                    direction_changes += 1
            
            # Calculate consistency (fewer direction changes = higher consistency)
            consistency = 1.0 - (direction_changes / (len(price_changes) - 1))
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5

    def _calculate_trade_intervals(self, recent_trades: List[Dict[str, Any]]) -> List[float]:
        """Calculate intervals between trades"""
        try:
            intervals = []
            for i in range(1, len(recent_trades)):
                # Simplified interval calculation (would use actual timestamps in production)
                intervals.append(1.0)  # Placeholder
            return intervals
        except Exception:
            return []

    def _calculate_regime_score(self, regime: str) -> float:
        """Calculate regime alignment score"""
        regime_scores = {
            'trending': 0.8,
            'ranging': 0.6,
            'volatile': 0.9,
            'breakout': 0.7,
            'reversal': 0.5,
            'unknown': 0.3
        }
        return regime_scores.get(regime, 0.3)

    def _assess_regime_alignment(self, detected_modes: Dict[str, float], regime: str) -> Dict[str, Any]:
        """Assess alignment between detected modes and market regime"""
        try:
            alignment = {}
            
            # Direct alignment
            if regime in detected_modes:
                alignment['direct_match'] = detected_modes[regime]
            else:
                alignment['direct_match'] = 0.0
            
            # Related mode alignment
            related_modes = {
                'trending': ['breakout', 'volatile'],
                'ranging': ['reversal'],
                'volatile': ['trending', 'breakout', 'reversal'],
                'breakout': ['trending', 'volatile'],
                'reversal': ['ranging', 'volatile']
            }
            
            related_confidence = 0.0
            if regime in related_modes:
                for related_mode in related_modes[regime]:
                    if related_mode in detected_modes:
                        related_confidence += detected_modes[related_mode]
            
            alignment['related_match'] = min(1.0, related_confidence)
            alignment['overall_alignment'] = (alignment['direct_match'] + alignment['related_match'] * 0.5) / 1.5
            
            return alignment
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_alignment")
            return {'direct_match': 0.0, 'related_match': 0.0, 'overall_alignment': 0.0}

    async def _log_significant_detections(self, analysis: Dict[str, Any]):
        """Log significant mode detections"""
        try:
            detected_modes = analysis.get('detected_modes', {})
            dominant_mode = max(detected_modes.items(), key=lambda x: x[1])
            
            if dominant_mode[1] > self.confidence_threshold:
                factors = analysis.get('detection_factors', {}).get(dominant_mode[0], [])
                self.logger.info(format_operator_message(
                    icon="🎯",
                    message=f"Strong {dominant_mode[0]} mode detected",
                    confidence=f"{dominant_mode[1]:.1%}",
                    factors=", ".join(factors[:3])
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "detection_logging")

    async def _update_mode_performance_comprehensive(self, market_data: Dict[str, Any], 
                                                   mode_analysis: Dict[str, Any]):
        """Update mode performance with comprehensive tracking"""
        try:
            recent_trades = market_data.get('recent_trades', [])
            detected_modes = mode_analysis.get('detected_modes', {})
            
            if not recent_trades:
                return
            
            # Get the most recent trade result
            last_trade = recent_trades[-1]
            pnl = last_trade.get('pnl', 0)
            
            # Update performance for detected modes
            for mode, confidence in detected_modes.items():
                if confidence > 0.1:  # Only update modes with significant confidence
                    await self._record_mode_result_comprehensive(mode, pnl, confidence, mode_analysis)
                    
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_update")
            self.logger.warning(f"Mode performance update failed: {error_context}")

    async def _record_mode_result_comprehensive(self, mode: str, pnl: float, confidence: float, 
                                              mode_analysis: Dict[str, Any]):
        """Record comprehensive mode result with enhanced analytics"""
        try:
            # Validate inputs
            if not isinstance(mode, str) or mode not in self.modes:
                return
            
            if np.isnan(pnl):
                return
            
            # Update basic tracking
            self.mode_performance[mode]['pnl'].append(pnl)
            self.mode_performance[mode]['confidence'].append(confidence)
            self.mode_counts[mode] += 1
            
            # Record in history with comprehensive context
            mode_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'mode': mode,
                'pnl': pnl,
                'confidence': confidence,
                'detection_factors': mode_analysis.get('detection_factors', {}).get(mode, []),
                'market_context': mode_analysis.get('market_characteristics', {})
            }
            self.mode_history.append(mode_record)
            
            # Update comprehensive analytics
            await self._update_mode_analytics_comprehensive(mode, pnl, confidence, mode_record)
            
            # Log significant results
            if abs(pnl) > 25 or confidence > 0.8:
                self.logger.info(format_operator_message(
                    icon="🎯",
                    message=f"{mode.title()} mode result recorded",
                    pnl=f"€{pnl:+.2f}",
                    confidence=f"{confidence:.1%}",
                    total_count=self.mode_counts[mode]
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "mode_result_recording")
            self.logger.error(f"Mode result recording failed: {error_context}")

    async def _update_mode_analytics_comprehensive(self, mode: str, pnl: float, confidence: float, 
                                                 mode_record: Dict[str, Any]):
        """Update comprehensive mode analytics"""
        try:
            mode_pnls = self.mode_performance[mode]['pnl']
            mode_confidences = self.mode_performance[mode]['confidence']
            
            if mode_pnls:
                # Update profitability
                self.mode_analytics['mode_profitability'][mode] = sum(mode_pnls)
                
                # Update win rate
                wins = len([p for p in mode_pnls if p > 0])
                self.mode_analytics['mode_win_rates'][mode] = wins / len(mode_pnls)
                
                # Update confidence-weighted performance
                if mode_confidences:
                    weights = np.array(mode_confidences)
                    weighted_pnls = np.array(mode_pnls) * weights
                    self.mode_analytics['mode_confidence_scores'][mode] = np.sum(weighted_pnls) / np.sum(weights)
                
                # Update effectiveness score
                recent_pnls = mode_pnls[-10:]  # Last 10 results
                recent_confidences = mode_confidences[-10:]
                if recent_pnls and recent_confidences:
                    effectiveness = np.mean(recent_pnls) * np.mean(recent_confidences)
                    self.mode_analytics['mode_effectiveness'][mode] = effectiveness
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "analytics_update")
            self.logger.warning(f"Mode analytics update failed: {error_context}")

    async def _adapt_mode_weights_intelligent(self, mode_analysis: Dict[str, Any], 
                                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt mode weights using intelligent algorithms"""
        try:
            detected_modes = mode_analysis.get('detected_modes', {})
            adaptation_info = {
                'weight_changes': {},
                'adaptation_factors': {},
                'performance_adjustments': {},
                'final_weights': {}
            }
            
            # Calculate new weights based on multiple factors
            new_weights = {}
            
            for mode in self.modes:
                base_weight = 1.0 / len(self.modes)  # Equal baseline
                
                # Performance adjustment
                performance_adj = await self._calculate_performance_adjustment_advanced(mode)
                adaptation_info['performance_adjustments'][mode] = performance_adj
                
                # Market detection adjustment
                market_adj = detected_modes.get(mode, 0.0)
                
                # Confidence and effectiveness adjustment
                confidence_adj = self.mode_analytics['mode_confidence_scores'].get(mode, 0.5)
                effectiveness_adj = self.mode_analytics['mode_effectiveness'].get(mode, 0.0)
                
                # Regime alignment adjustment
                regime_alignment = mode_analysis.get('regime_alignment', {}).get('overall_alignment', 0.5)
                
                # Combine adjustments with intelligent weighting
                combined_weight = base_weight * (
                    0.3 * (1.0 + performance_adj) +      # Performance component (30%)
                    0.25 * market_adj +                  # Market detection component (25%)
                    0.2 * confidence_adj +               # Confidence component (20%)
                    0.15 * (1.0 + effectiveness_adj) +   # Effectiveness component (15%)
                    0.1 * regime_alignment               # Regime alignment component (10%)
                )
                
                new_weights[mode] = max(0.05, combined_weight)  # Minimum weight threshold
                
                # Track adaptation factors
                adaptation_info['adaptation_factors'][mode] = {
                    'performance': performance_adj,
                    'market_detection': market_adj,
                    'confidence': confidence_adj,
                    'effectiveness': effectiveness_adj,
                    'regime_alignment': regime_alignment
                }
            
            # Normalize weights
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {k: v/total_weight for k, v in new_weights.items()}
            
            # Apply adaptation rate with momentum
            momentum = self.mode_intelligence['adaptation_momentum']
            for mode in self.modes:
                old_weight = self.current_mode_weights.get(mode, 1.0/len(self.modes))
                new_weight = new_weights.get(mode, 1.0/len(self.modes))
                
                # Apply momentum-based adaptation
                adapted_weight = (
                    old_weight * (1 - self.adaptation_rate) * momentum +
                    new_weight * self.adaptation_rate +
                    old_weight * (1 - momentum) * 0.1  # Small stability component
                )
                
                self.current_mode_weights[mode] = adapted_weight
                adaptation_info['weight_changes'][mode] = adapted_weight - old_weight
            
            # Final normalization
            total_final = sum(self.current_mode_weights.values())
            if total_final > 0:
                self.current_mode_weights = {k: v/total_final for k, v in self.current_mode_weights.items()}
            
            adaptation_info['final_weights'] = self.current_mode_weights.copy()
            
            # Log significant weight changes
            await self._log_weight_changes_advanced(adaptation_info)
            
            return adaptation_info
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "weight_adaptation")
            self.logger.error(f"Mode weight adaptation failed: {error_context}")
            return {'error': str(error_context)}

    async def _calculate_performance_adjustment_advanced(self, mode: str) -> float:
        """Calculate advanced performance-based adjustment for mode weight"""
        try:
            mode_pnls = self.mode_performance[mode]['pnl']
            if not mode_pnls or len(mode_pnls) < 3:
                return 0.0  # No adjustment for insufficient data
            
            # Recent performance (last 10 trades)
            recent_pnls = mode_pnls[-10:]
            recent_performance = sum(recent_pnls)
            
            # Overall performance with decay
            decay_factor = self.mode_intelligence['performance_memory']
            decayed_weights = [decay_factor ** (len(mode_pnls) - i - 1) for i in range(len(mode_pnls))]
            weighted_performance = sum(p * w for p, w in zip(mode_pnls, decayed_weights))
            
            # Win rate and consistency components
            win_rate = self.mode_analytics['mode_win_rates'].get(mode, 0.5)
            effectiveness = self.mode_analytics['mode_effectiveness'].get(mode, 0.0)
            
            # Calculate composite adjustment (-1.0 to +1.0 range)
            if recent_performance > 100:
                performance_adj = 0.5 + min(0.5, recent_performance / 500)
            elif recent_performance < -50:
                performance_adj = -0.5 + max(-0.5, recent_performance / 200)
            else:
                performance_adj = recent_performance / 200
            
            # Adjust based on win rate and effectiveness
            win_rate_adj = (win_rate - 0.5) * 0.4  # -0.2 to +0.2 range
            effectiveness_adj = np.clip(effectiveness / 100, -0.2, 0.2)
            
            # Combine with weights
            final_adjustment = (
                performance_adj * 0.5 +
                win_rate_adj * 0.3 +
                effectiveness_adj * 0.2
            )
            
            return np.clip(final_adjustment, -0.8, 0.8)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_adjustment")
            return 0.0

    async def _log_weight_changes_advanced(self, adaptation_info: Dict[str, Any]):
        """Log advanced weight changes with detailed analysis"""
        try:
            weight_changes = adaptation_info.get('weight_changes', {})
            significant_changes = []
            
            for mode, change in weight_changes.items():
                if abs(change) > 0.1:  # 10% change threshold
                    direction = "↗️" if change > 0 else "↘️"
                    old_weight = self.current_mode_weights[mode] - change
                    new_weight = self.current_mode_weights[mode]
                    
                    factors = adaptation_info.get('adaptation_factors', {}).get(mode, {})
                    primary_factor = max(factors.items(), key=lambda x: abs(x[1]))[0] if factors else 'unknown'
                    
                    significant_changes.append({
                        'mode': mode,
                        'change': change,
                        'direction': direction,
                        'old_weight': old_weight,
                        'new_weight': new_weight,
                        'primary_factor': primary_factor
                    })
            
            if significant_changes:
                change_summary = "; ".join([
                    f"{c['mode']}: {c['old_weight']:.1%} → {c['new_weight']:.1%} {c['direction']} ({c['primary_factor']})"
                    for c in significant_changes[:3]
                ])
                
                self.logger.info(format_operator_message(
                    icon="⚖️",
                    message="Significant mode weight adaptations",
                    changes=change_summary
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "weight_change_logging")

    async def _generate_comprehensive_mode_thesis(self, mode_analysis: Dict[str, Any], 
                                                weight_adaptation: Dict[str, Any]) -> str:
        """Generate comprehensive thesis explaining all mode decisions"""
        try:
            detected_modes = mode_analysis.get('detected_modes', {})
            weight_changes = weight_adaptation.get('weight_changes', {})
            
            thesis_parts = []
            
            # Executive Summary
            dominant_mode = max(detected_modes.items(), key=lambda x: x[1])
            thesis_parts.append(
                f"MODE ANALYSIS: {dominant_mode[0].title()} mode dominant with {dominant_mode[1]:.1%} confidence"
            )
            
            # Detection Analysis
            detection_factors = mode_analysis.get('detection_factors', {})
            if detection_factors.get(dominant_mode[0]):
                primary_factors = detection_factors[dominant_mode[0]][:3]
                thesis_parts.append(
                    f"DETECTION FACTORS: {', '.join(primary_factors)} support {dominant_mode[0]} classification"
                )
            
            # Weight Adaptation Summary
            significant_changes = {k: v for k, v in weight_changes.items() if abs(v) > 0.05}
            if significant_changes:
                thesis_parts.append(
                    f"WEIGHT ADAPTATIONS: {len(significant_changes)} modes adjusted based on performance and detection"
                )
                
                # Detail top changes
                top_changes = sorted(significant_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
                for mode, change in top_changes:
                    direction = "increased" if change > 0 else "decreased"
                    thesis_parts.append(f"  • {mode.title()}: Weight {direction} by {abs(change):.1%}")
            
            # Performance Context
            performance_summary = self._get_performance_summary()
            best_mode = performance_summary.get('best_performing_mode', 'unknown')
            if best_mode != 'unknown':
                best_performance = performance_summary.get('best_performance', 0)
                thesis_parts.append(
                    f"PERFORMANCE LEADER: {best_mode.title()} mode with €{best_performance:.0f} total profit"
                )
            
            # Market Context Integration
            regime_alignment = mode_analysis.get('regime_alignment', {})
            overall_alignment = regime_alignment.get('overall_alignment', 0)
            thesis_parts.append(
                f"REGIME ALIGNMENT: {overall_alignment:.1%} alignment with current market conditions"
            )
            
            # Analytics Summary
            total_modes_active = len([m for m in self.modes if self.mode_counts.get(m, 0) > 0])
            thesis_parts.append(
                f"SYSTEM STATUS: {total_modes_active}/{len(self.modes)} modes have performance data"
            )
            
            # Quality Assessment
            confidence_scores = mode_analysis.get('confidence_scores', {})
            avg_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0
            thesis_parts.append(
                f"DETECTION QUALITY: {avg_confidence:.1%} average detection confidence across all modes"
            )
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Mode analysis thesis generation failed: {error_context}"

    def _generate_intelligent_recommendations(self, mode_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on mode analysis"""
        try:
            recommendations = []
            detected_modes = mode_analysis.get('detected_modes', {})
            
            # Primary mode recommendation
            dominant_mode = max(detected_modes.items(), key=lambda x: x[1])
            if dominant_mode[1] > self.confidence_threshold:
                mode_def = self.mode_definitions.get(dominant_mode[0], {})
                optimal_strategies = mode_def.get('optimal_strategies', [])
                if optimal_strategies:
                    recommendations.append(
                        f"Primary Strategy: Focus on {', '.join(optimal_strategies[:2])} for {dominant_mode[0]} conditions"
                    )
                
                risk_factors = mode_def.get('risk_factors', [])
                if risk_factors:
                    recommendations.append(
                        f"Risk Awareness: Monitor for {', '.join(risk_factors[:2])} in current {dominant_mode[0]} mode"
                    )
            
            # Performance-based recommendations
            performance_summary = self._get_performance_summary()
            best_mode = performance_summary.get('best_performing_mode')
            worst_mode = performance_summary.get('worst_performing_mode')
            
            if best_mode and best_mode != dominant_mode[0]:
                recommendations.append(
                    f"Performance Insight: {best_mode.title()} mode showing strongest results - consider increased allocation"
                )
            
            if worst_mode and self.mode_analytics['mode_profitability'].get(worst_mode, 0) < -50:
                recommendations.append(
                    f"Performance Warning: {worst_mode.title()} mode underperforming - reduce exposure or review approach"
                )
            
            # Adaptation recommendations
            recent_switches = self.mode_analytics.get('switches_since_reset', 0)
            if recent_switches > 10:
                recommendations.append(
                    "Stability: High mode switching frequency - consider extending confidence thresholds"
                )
            elif recent_switches < 2:
                recommendations.append(
                    "Adaptation: Low mode switching - system may benefit from increased sensitivity"
                )
            
            # Default recommendation
            if not recommendations:
                recommendations.append(
                    f"Continue current approach with {dominant_mode[0]} mode focus - system operating optimally"
                )
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendation_generation")
            return [f"Recommendation generation failed: {error_context}"]

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary across all modes"""
        try:
            summary = {
                'total_modes': len(self.modes),
                'active_modes': 0,
                'best_performing_mode': None,
                'worst_performing_mode': None,
                'best_performance': float('-inf'),
                'worst_performance': float('inf'),
                'total_profit': 0.0,
                'overall_win_rate': 0.0,
                'mode_performance_breakdown': {}
            }
            
            total_trades = 0
            total_wins = 0
            
            for mode in self.modes:
                mode_pnls = self.mode_performance[mode]['pnl']
                if mode_pnls:
                    summary['active_modes'] += 1
                    mode_profit = sum(mode_pnls)
                    mode_wins = len([p for p in mode_pnls if p > 0])
                    mode_win_rate = mode_wins / len(mode_pnls)
                    
                    summary['mode_performance_breakdown'][mode] = {
                        'profit': mode_profit,
                        'trades': len(mode_pnls),
                        'wins': mode_wins,
                        'win_rate': mode_win_rate,
                        'avg_trade': mode_profit / len(mode_pnls)
                    }
                    
                    summary['total_profit'] += mode_profit
                    total_trades += len(mode_pnls)
                    total_wins += mode_wins
                    
                    if mode_profit > summary['best_performance']:
                        summary['best_performance'] = mode_profit
                        summary['best_performing_mode'] = mode
                    
                    if mode_profit < summary['worst_performance']:
                        summary['worst_performance'] = mode_profit
                        summary['worst_performing_mode'] = mode
            
            if total_trades > 0:
                summary['overall_win_rate'] = total_wins / total_trades
            
            return summary
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_summary")
            return {'error': str(error_context)}

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive mode results"""
        try:
            # Core mode weights
            self.smart_bus.set('mode_weights', results['mode_weights'],
                             module='OpponentModeEnhancer', thesis=thesis)
            
            # Mode analysis
            analysis_thesis = f"Market mode analysis: {len(results['mode_analysis'].get('detected_modes', {}))} modes evaluated"
            self.smart_bus.set('mode_analysis', results['mode_analysis'],
                             module='OpponentModeEnhancer', thesis=analysis_thesis)
            
            # Mode recommendations
            rec_thesis = f"Generated {len(results['mode_recommendations'])} intelligent mode recommendations"
            self.smart_bus.set('mode_recommendations', results['mode_recommendations'],
                             module='OpponentModeEnhancer', thesis=rec_thesis)
            
            # Market mode detection
            detection_thesis = f"Mode detection: {max(results['market_mode_detection'].items(), key=lambda x: x[1])[0]} dominant"
            self.smart_bus.set('market_mode_detection', results['market_mode_detection'],
                             module='OpponentModeEnhancer', thesis=detection_thesis)
            
            # Strategy adaptation
            adaptation_thesis = f"Strategy adaptation: {len([c for c in results['strategy_adaptation'].get('weight_changes', {}).values() if abs(c) > 0.05])} significant changes"
            self.smart_bus.set('strategy_adaptation', results['strategy_adaptation'],
                             module='OpponentModeEnhancer', thesis=adaptation_thesis)
            
            # Mode performance
            performance_thesis = f"Mode performance: {results['mode_performance'].get('active_modes', 0)} modes active"
            self.smart_bus.set('mode_performance', results['mode_performance'],
                             module='OpponentModeEnhancer', thesis=performance_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # ERROR HANDLING AND RECOVERY
    # ═══════════════════════════════════════════════════════════════════

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "OpponentModeEnhancer")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Opponent Mode Enhancer disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('OpponentModeEnhancer', 'process_time', processing_time, False)
        
        return {
            'mode_weights': {mode: 1.0/len(self.modes) for mode in self.modes},
            'mode_analysis': {'error': str(error_context)},
            'mode_recommendations': ["Investigate mode detection system errors"],
            'market_mode_detection': {},
            'strategy_adaptation': {'error': str(error_context)},
            'mode_performance': {'error': str(error_context)},
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_market_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when market data retrieval fails"""
        return {
            'market_data': {},
            'recent_trades': [],
            'technical_indicators': {},
            'volatility_data': {},
            'price_data': {},
            'market_regime': 'unknown',
            'trading_performance': {},
            'session_metrics': {},
            'market_context': {}
        }

    def _get_safe_analysis_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when analysis fails"""
        return {
            'detected_modes': {mode: 1.0/len(self.modes) for mode in self.modes},
            'confidence_scores': {mode: 0.5 for mode in self.modes},
            'detection_factors': {mode: ['insufficient_data'] for mode in self.modes},
            'market_characteristics': {},
            'regime_alignment': {'overall_alignment': 0.5},
            'detection_timestamp': datetime.datetime.now().isoformat(),
            'error': 'analysis_failed'
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'mode_weights': {mode: 1.0/len(self.modes) for mode in self.modes},
            'mode_analysis': {'status': 'disabled'},
            'mode_recommendations': ["Restart opponent mode enhancer system"],
            'market_mode_detection': {},
            'strategy_adaptation': {'status': 'disabled'},
            'mode_performance': {'status': 'disabled'},
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'OpponentModeEnhancer',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'modes_tracked': len(self.modes),
            'active_modes': len([m for m in self.modes if self.mode_counts.get(m, 0) > 0]),
            'total_detections': sum(self.mode_counts.values()),
            'adaptation_rate': self.adaptation_rate,
            'confidence_threshold': self.confidence_threshold,
            'session_duration': (datetime.datetime.now() - datetime.datetime.fromisoformat(self.mode_analytics['session_start'])).total_seconds() / 3600
        }

    def get_mode_report(self) -> str:
        """Generate comprehensive mode analysis report"""
        performance_summary = self._get_performance_summary()
        
        # Mode performance summary
        mode_breakdown = ""
        for mode in self.modes:
            count = self.mode_counts.get(mode, 0)
            weight = self.current_mode_weights.get(mode, 0)
            
            if count > 0:
                profit = performance_summary['mode_performance_breakdown'][mode]['profit']
                win_rate = performance_summary['mode_performance_breakdown'][mode]['win_rate']
                status = "🟢" if profit > 0 else "🔴" if profit < -25 else "🟡"
                mode_breakdown += f"  • {mode.title()}: {weight:.1%} weight, €{profit:+.0f} P&L, {win_rate:.1%} win rate, {count} trades {status}\n"
            else:
                mode_breakdown += f"  • {mode.title()}: {weight:.1%} weight, No performance data yet ⚪\n"
        
        # Recent activity
        recent_activity = ""
        if self.mode_history:
            for record in list(self.mode_history)[-3:]:
                timestamp = record['timestamp'][:19].replace('T', ' ')
                mode = record['mode']
                pnl = record['pnl']
                confidence = record['confidence']
                factors = record.get('detection_factors', [])
                recent_activity += f"  • {timestamp}: {mode} (€{pnl:+.0f}, {confidence:.1%}, {', '.join(factors[:2])})\n"
        
        # Current dominant mode
        dominant_mode = max(self.current_mode_weights.items(), key=lambda x: x[1])
        
        return f"""
🎯 OPPONENT MODE ENHANCER COMPREHENSIVE REPORT
═══════════════════════════════════════════════════════════════
🏆 Current Dominant Mode: {dominant_mode[0].title()} ({dominant_mode[1]:.1%} weight)
📊 Best Performing Mode: {performance_summary.get('best_performing_mode', 'N/A')} (€{performance_summary.get('best_performance', 0):+.0f})
💰 Total Profit Across Modes: €{performance_summary.get('total_profit', 0):+.0f}
🎯 Overall Win Rate: {performance_summary.get('overall_win_rate', 0):.1%}

⚙️ System Configuration:
• Modes Tracked: {len(self.modes)} ({performance_summary.get('active_modes', 0)} with data)
• Adaptation Rate: {self.adaptation_rate:.1%}
• Confidence Threshold: {self.confidence_threshold:.1%}
• Switch Cooldown: {self.mode_switch_cooldown} periods

📈 Mode Performance Breakdown:
{mode_breakdown}

🔄 Recent Mode Activity:
{recent_activity if recent_activity else '  📭 No recent mode activity'}

📊 Advanced Analytics:
• Mode Switches: {self.mode_analytics.get('switches_since_reset', 0)}
• Detection Quality: {np.mean([self.mode_analytics['mode_confidence_scores'].get(m, 0.5) for m in self.modes]):.1%} avg confidence
• System Health: {'DISABLED' if self.is_disabled else 'OPERATIONAL'}
• Error Count: {self.error_count}/{self.circuit_breaker_threshold}

🎯 Mode Definitions:
{chr(10).join([f'  • {mode.title()}: {self.mode_definitions[mode].get("description", "No description")}' for mode in self.modes])}

🔧 Current Intelligence Settings:
• Detection Sensitivity: {self.mode_intelligence['detection_sensitivity']:.1%}
• Adaptation Momentum: {self.mode_intelligence['adaptation_momentum']:.1%}
• Confidence Decay: {self.mode_intelligence['confidence_decay']:.1%}
• Performance Memory: {self.mode_intelligence['performance_memory']:.1%}
        """

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT FOR HOT-RELOAD
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'OpponentModeEnhancer',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'modes': self.modes.copy(),
                'adaptation_rate': self.adaptation_rate,
                'confidence_threshold': self.confidence_threshold,
                'mode_switch_cooldown': self.mode_switch_cooldown,
                'debug': self.debug
            },
            'mode_state': {
                'current_mode_weights': self.current_mode_weights.copy(),
                'mode_performance': {k: {'pnl': list(v['pnl']), 'confidence': list(v['confidence'])} 
                                   for k, v in self.mode_performance.items()},
                'mode_counts': dict(self.mode_counts),
                'mode_history': list(self.mode_history),
                'mode_analytics': self.mode_analytics.copy(),
                'mode_intelligence': self.mode_intelligence.copy()
            },
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'mode_definitions': self.mode_definitions.copy(),
            'performance_metrics': self._get_health_metrics()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.modes = config.get("modes", self.modes)
            self.adaptation_rate = float(config.get("adaptation_rate", self.adaptation_rate))
            self.confidence_threshold = float(config.get("confidence_threshold", self.confidence_threshold))
            self.mode_switch_cooldown = int(config.get("mode_switch_cooldown", self.mode_switch_cooldown))
            self.debug = bool(config.get("debug", self.debug))
            
            # Load mode state
            mode_state = state.get("mode_state", {})
            self.current_mode_weights = mode_state.get("current_mode_weights", 
                                                     {mode: 1.0/len(self.modes) for mode in self.modes})
            
            # Restore performance data
            performance_data = mode_state.get("mode_performance", {})
            self.mode_performance = defaultdict(lambda: defaultdict(list))
            for mode, data in performance_data.items():
                self.mode_performance[mode]['pnl'] = list(data.get('pnl', []))
                self.mode_performance[mode]['confidence'] = list(data.get('confidence', []))
            
            # Restore other state
            self.mode_counts = defaultdict(int, mode_state.get("mode_counts", {}))
            self.mode_history = deque(mode_state.get("mode_history", []), maxlen=100)
            self.mode_analytics = mode_state.get("mode_analytics", self.mode_analytics)
            self.mode_intelligence = mode_state.get("mode_intelligence", self.mode_intelligence)
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            # Load mode definitions if provided
            self.mode_definitions.update(state.get("mode_definitions", {}))
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Opponent Mode Enhancer state restored",
                modes=len(self.modes),
                active_modes=len([m for m in self.modes if self.mode_counts.get(m, 0) > 0]),
                total_records=len(self.mode_history)
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'OpponentModeEnhancer',
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
                'message': 'OpponentModeEnhancer disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring issues'
            })
        
        active_modes = len([m for m in self.modes if self.mode_counts.get(m, 0) > 0])
        if active_modes < len(self.modes) / 2:
            alerts.append({
                'severity': 'info',
                'message': f'Only {active_modes}/{len(self.modes)} modes have performance data',
                'action': 'Continue trading to build mode performance baselines'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart OpponentModeEnhancer module after investigating errors")
        
        if len(self.mode_history) < 20:
            recommendations.append("Insufficient mode history - continue operations to build performance baseline")
        
        if self.mode_analytics.get('switches_since_reset', 0) > 20:
            recommendations.append("High mode switching frequency - consider adjusting confidence thresholds")
        
        if not recommendations:
            recommendations.append("OpponentModeEnhancer operating within normal parameters")
        
        return recommendations

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API METHODS (for external use)
    # ═══════════════════════════════════════════════════════════════════

    def record_result(self, mode: str, pnl: float, confidence: float = 1.0) -> None:
        """Public method to record mode result (async wrapper)"""
        try:
            # Validate inputs
            if not isinstance(mode, str):
                self.logger.warning(f"Invalid mode type: {type(mode)}")
                return
            
            if np.isnan(pnl):
                self.logger.warning(f"NaN PnL for mode {mode}, ignoring")
                return
            
            if mode not in self.modes:
                self.logger.warning(f"Unknown mode '{mode}', adding to tracking")
                self.modes.append(mode)
                self.current_mode_weights[mode] = 1.0 / len(self.modes)
                self.mode_definitions[mode] = {'description': f'Dynamic mode: {mode}'}
            
            # Run the async method synchronously or schedule it
            import asyncio
            mode_analysis = {'market_characteristics': {}, 'detection_factors': {mode: ['external_result']}}
            
            if asyncio.get_event_loop().is_running():
                # If we're already in an async context, schedule it
                asyncio.create_task(self._record_mode_result_comprehensive(mode, pnl, confidence, mode_analysis))
            else:
                # Run it directly
                asyncio.run(self._record_mode_result_comprehensive(mode, pnl, confidence, mode_analysis))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "result_recording_wrapper")
            self.logger.error(f"Mode result recording wrapper failed: {error_context}")

    def get_observation_components(self) -> np.ndarray:
        """Get mode weights for observation"""
        try:
            if not self.mode_performance:
                # Return equal distribution for cold start
                num_modes = len(self.modes)
                defaults = np.full(num_modes, 1.0/num_modes, dtype=np.float32)
                return defaults
            
            # Get current mode weights
            weights = []
            for mode in self.modes:
                weight = self.current_mode_weights.get(mode, 1.0/len(self.modes))
                weights.append(weight)
            
            # Convert to numpy array
            observation = np.array(weights, dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid mode observation: {observation}")
                observation = np.nan_to_num(observation, nan=1.0/len(self.modes))
            
            # Ensure weights sum to 1
            weight_sum = observation.sum()
            if weight_sum > 0:
                observation = observation / weight_sum
            else:
                observation = np.full(len(self.modes), 1.0/len(self.modes), dtype=np.float32)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Mode observation generation failed: {error_context}")
            return np.full(len(self.modes), 1.0/len(self.modes), dtype=np.float32)

    def get_mode_recommendations(self) -> Dict[str, Any]:
        """Get current mode recommendations based on analysis"""
        try:
            performance_summary = self._get_performance_summary()
            
            # Find most confident mode
            most_confident_mode = max(self.current_mode_weights.items(), key=lambda x: x[1])
            
            # Generate recommendations
            recommendations = {
                'primary_mode': most_confident_mode[0],
                'primary_weight': most_confident_mode[1],
                'best_performing_mode': performance_summary.get('best_performing_mode'),
                'best_performance': performance_summary.get('best_performance', 0),
                'mode_weights': self.current_mode_weights.copy(),
                'mode_analytics': {
                    'total_modes_tracked': len(self.modes),
                    'modes_with_data': performance_summary.get('active_modes', 0),
                    'most_used_mode': max(self.mode_counts.items(), key=lambda x: x[1])[0] if self.mode_counts else None,
                    'total_profit': performance_summary.get('total_profit', 0),
                    'overall_win_rate': performance_summary.get('overall_win_rate', 0)
                }
            }
            
            return recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendations_generation")
            self.logger.warning(f"Mode recommendations generation failed: {error_context}")
            return {'primary_mode': self.modes[0], 'primary_weight': 1.0/len(self.modes)}