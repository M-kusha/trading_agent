"""
🕐 Enhanced Time Horizon Aligner with SmartInfoBus Integration v3.0
Advanced time-based weight scaling for voting committees with market adaptation
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
    name="TimeHorizonAligner",
    version="3.0.0",
    category="voting",
    provides=[
        "aligned_weights", "horizon_distances", "horizon_multipliers", "regime_adjustments",
        "session_patterns", "alignment_quality", "performance_metrics", "adaptation_status"
    ],
    requires=[
        "voting_weights", "market_regime", "session_type", "volatility_data", "market_context",
        "time_of_day", "performance_feedback", "member_confidences"
    ],
    description="Advanced time-based weight scaling for voting committees with market adaptation",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=50,
    priority=4,
    explainable=True,
    hot_reload=True
)
class TimeHorizonAligner(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🕐 PRODUCTION-GRADE Time Horizon Aligner v3.0
    
    Advanced time-based weight scaling system with:
    - Multi-dimensional horizon analysis across short, medium, and long-term perspectives
    - Adaptive market regime adjustments with volatility-aware scaling
    - Session-based cyclical pattern recognition and optimization
    - Performance feedback integration with expert contribution tracking
    - SmartInfoBus zero-wiring architecture with comprehensive analytics
    """

    def _initialize(self):
        """Initialize advanced time horizon alignment systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Core horizon configuration
        default_horizons = [1, 3, 5, 10, 15, 30, 60, 120, 240]
        self.horizons = np.array(
            self.config.get('horizons', default_horizons), 
            dtype=np.float32
        )
        self.adaptive_scaling = self.config.get('adaptive_scaling', True)
        self.regime_awareness = self.config.get('regime_awareness', True)
        self.performance_feedback = self.config.get('performance_feedback', True)
        self.debug = self.config.get('debug', False)
        
        # Time tracking and clock management
        self.clock = 0
        self.session_start = 0
        self.last_alignment_time = datetime.datetime.now()
        
        # Core alignment state
        self.current_distances = np.ones_like(self.horizons)
        self.base_distances = np.ones_like(self.horizons)
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.performance_multipliers = np.ones_like(self.horizons)
        self.cyclical_adjustments = np.ones_like(self.horizons)
        
        # Advanced market adaptation
        self.regime_multipliers = {
            'trending': np.ones_like(self.horizons),
            'volatile': np.ones_like(self.horizons),
            'ranging': np.ones_like(self.horizons),
            'noise': np.ones_like(self.horizons),
            'breakout': np.ones_like(self.horizons),
            'reversal': np.ones_like(self.horizons),
            'unknown': np.ones_like(self.horizons)
        }
        
        # Horizon performance tracking with enhanced analytics
        self.horizon_performance = defaultdict(lambda: {
            'total_weight': 0.0,
            'successful_weight': 0.0,
            'performance_score': 0.5,
            'recent_scores': deque(maxlen=30),
            'effectiveness_ratio': 0.5,
            'consistency_score': 0.5,
            'adaptation_count': 0,
            'last_update': datetime.datetime.now().isoformat()
        })
        
        # Session and cyclical intelligence
        self.session_patterns = {
            'american': np.ones_like(self.horizons),
            'european': np.ones_like(self.horizons),
            'asian': np.ones_like(self.horizons),
            'rollover': np.ones_like(self.horizons),
            'weekend': np.ones_like(self.horizons),
            'unknown': np.ones_like(self.horizons)
        }
        
        # Market state awareness
        self.current_regime = 'unknown'
        self.current_session = 'unknown'
        self.current_volatility = 0.02
        self.volatility_history = deque(maxlen=50)
        
        # Advanced alignment intelligence
        self.alignment_intelligence = {
            'learning_rate': 0.12,
            'adaptation_threshold': 0.15,
            'performance_window': 20,
            'regime_sensitivity': 0.8,
            'session_memory': 0.85,
            'volatility_adaptation': 0.7,
            'horizon_decay': 0.95,
            'performance_momentum': 0.9
        }
        
        # Quality and effectiveness metrics
        self.alignment_quality = {
            'effectiveness': 0.5,
            'consistency': 0.5,
            'adaptability': 0.5,
            'regime_alignment': 0.5,
            'session_optimization': 0.5,
            'performance_correlation': 0.5
        }
        
        # Comprehensive tracking
        self.alignment_history = deque(maxlen=200)
        self.adaptation_events = deque(maxlen=100)
        self.performance_history = deque(maxlen=150)
        
        # Statistics and analytics
        self.alignment_stats = {
            'total_alignments': 0,
            'significant_adaptations': 0,
            'regime_switches': 0,
            'session_transitions': 0,
            'performance_adjustments': 0,
            'avg_alignment_impact': 0.0,
            'effectiveness_trend': 0.0,
            'adaptation_accuracy': 0.5,
            'session_start_time': datetime.datetime.now().isoformat()
        }
        
        # Volatility and market condition adaptation
        self.volatility_adaptation = {
            'extreme': {'horizon_bias': 'short', 'multiplier_range': (0.3, 1.8)},
            'high': {'horizon_bias': 'short', 'multiplier_range': (0.5, 1.6)},
            'medium': {'horizon_bias': 'balanced', 'multiplier_range': (0.7, 1.4)},
            'low': {'horizon_bias': 'long', 'multiplier_range': (0.8, 1.3)},
            'very_low': {'horizon_bias': 'long', 'multiplier_range': (0.9, 1.2)}
        }
        
        # Circuit breaker and error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🕐",
            message=f"Time Horizon Aligner v{version} initialized",
            horizons=len(self.horizons),
            adaptive=self.adaptive_scaling,
            regime_aware=self.regime_awareness,
            performance_feedback=self.performance_feedback
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="TimeHorizonAligner",
            log_path="logs/voting/time_horizon_aligner.log",
            max_lines=3000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("TimeHorizonAligner", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Time Horizon Aligner v3.0 Initialization Complete:
        
        Advanced Horizon Management Framework:
        - Multi-scale temporal analysis across {len(self.horizons)} time horizons
        - Adaptive scaling with market regime awareness and volatility adjustments
        - Session-based cyclical pattern optimization for different market sessions
        - Performance feedback integration with expert contribution tracking
        
        Current Configuration:
        - Time horizons: {self.horizons.tolist()} (steps)
        - Adaptive scaling: {'enabled' if self.adaptive_scaling else 'disabled'} with learning rate {self.alignment_intelligence['learning_rate']:.2f}
        - Regime awareness: {'enabled' if self.regime_awareness else 'disabled'} with {len(self.regime_multipliers)} regime profiles
        - Performance feedback: {'enabled' if self.performance_feedback else 'disabled'} with {self.alignment_intelligence['performance_window']}-step window
        
        Intelligence Parameters:
        - Adaptation threshold: {self.alignment_intelligence['adaptation_threshold']:.2f} for significant changes
        - Regime sensitivity: {self.alignment_intelligence['regime_sensitivity']:.2f} for market condition response
        - Session memory: {self.alignment_intelligence['session_memory']:.2f} for cyclical pattern retention
        - Performance momentum: {self.alignment_intelligence['performance_momentum']:.2f} for feedback integration
        
        Advanced Features:
        - Volatility-aware horizon biasing with {len(self.volatility_adaptation)} volatility regimes
        - Session-specific pattern recognition for optimal time-of-day alignment
        - Performance-driven horizon effectiveness tracking and adaptation
        - Real-time alignment quality assessment and optimization recommendations
        
        Expected Outcomes:
        - Enhanced temporal decision quality through intelligent horizon weighting
        - Improved market timing with adaptive regime and session awareness
        - Optimal expert weight distribution based on time horizon effectiveness
        - Transparent alignment decisions with comprehensive quality analysis and actionable insights
        """
        
        self.smart_bus.set('time_horizon_aligner_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'horizons': self.horizons.tolist(),
                'adaptive_scaling': self.adaptive_scaling,
                'regime_awareness': self.regime_awareness,
                'performance_feedback': self.performance_feedback,
                'intelligence_parameters': self.alignment_intelligence
            }
        }, module='TimeHorizonAligner', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive horizon alignment
        
        Returns:
            Dict containing alignment results, quality metrics, and analytics
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Increment clock and update timing
            self.clock += 1
            current_time = datetime.datetime.now()
            
            # Get comprehensive data from SmartInfoBus
            alignment_data = await self._get_comprehensive_alignment_data()
            
            # Update market state and conditions
            await self._update_market_state_comprehensive(alignment_data)
            
            # Update horizon performance tracking
            if self.performance_feedback:
                await self._update_horizon_performance_comprehensive(alignment_data)
            
            # Calculate distance-based alignment
            await self._calculate_comprehensive_distance_alignment()
            
            # Apply regime and session adaptations
            await self._apply_regime_and_session_adaptations(alignment_data)
            
            # Update cyclical patterns
            await self._update_cyclical_patterns_comprehensive(alignment_data)
            
            # Calculate alignment quality metrics
            quality_analysis = await self._calculate_comprehensive_alignment_quality()
            
            # Generate alignment recommendations
            recommendations = await self._generate_intelligent_alignment_recommendations(quality_analysis)
            
            # Create comprehensive results
            results = {
                'aligned_weights': None,  # Will be set when apply() is called
                'horizon_distances': self.current_distances.tolist(),
                'horizon_multipliers': self._get_combined_multipliers().tolist(),
                'regime_adjustments': self.regime_multipliers[self.current_regime].tolist(),
                'session_patterns': self.session_patterns[self.current_session].tolist(),
                'alignment_quality': quality_analysis,
                'performance_metrics': self._get_performance_metrics_summary(),
                'adaptation_status': self._get_adaptation_status(),
                'health_metrics': self._get_health_metrics()
            }
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_alignment_thesis(results, quality_analysis)
            
            # Update SmartInfoBus with comprehensive results
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('TimeHorizonAligner', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            self.last_alignment_time = current_time
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_alignment_data(self) -> Dict[str, Any]:
        """Get comprehensive data for horizon alignment"""
        try:
            return {
                'voting_weights': self.smart_bus.get('voting_weights', 'TimeHorizonAligner') or [],
                'market_regime': self.smart_bus.get('market_regime', 'TimeHorizonAligner') or 'unknown',
                'session_type': self.smart_bus.get('session_type', 'TimeHorizonAligner') or 'unknown',
                'volatility_data': self.smart_bus.get('volatility_data', 'TimeHorizonAligner') or {},
                'market_context': self.smart_bus.get('market_context', 'TimeHorizonAligner') or {},
                'time_of_day': self.smart_bus.get('time_of_day', 'TimeHorizonAligner') or 0,
                'performance_feedback': self.smart_bus.get('performance_feedback', 'TimeHorizonAligner') or {},
                'member_confidences': self.smart_bus.get('member_confidences', 'TimeHorizonAligner') or [],
                'recent_trades': self.smart_bus.get('recent_trades', 'TimeHorizonAligner') or [],
                'expert_performance': self.smart_bus.get('expert_performance', 'TimeHorizonAligner') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "TimeHorizonAligner")
            self.logger.warning(f"Alignment data retrieval incomplete: {error_context}")
            return self._get_safe_alignment_defaults()

    async def _update_market_state_comprehensive(self, alignment_data: Dict[str, Any]):
        """Update comprehensive market state tracking"""
        try:
            # Track regime changes
            old_regime = self.current_regime
            self.current_regime = alignment_data.get('market_regime', 'unknown')
            
            if old_regime != self.current_regime and old_regime != 'unknown':
                self.alignment_stats['regime_switches'] += 1
                self.logger.info(format_operator_message(
                    icon="📊",
                    message="Market regime changed",
                    old_regime=old_regime,
                    new_regime=self.current_regime,
                    clock=self.clock,
                    impact="Horizon multipliers will adapt"
                ))
                
                # Record regime change event
                self.adaptation_events.append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'type': 'regime_change',
                    'old_value': old_regime,
                    'new_value': self.current_regime,
                    'clock': self.clock
                })
            
            # Track session changes
            old_session = self.current_session
            self.current_session = alignment_data.get('session_type', 'unknown')
            
            if old_session != self.current_session and old_session != 'unknown':
                self.alignment_stats['session_transitions'] += 1
                self.logger.info(format_operator_message(
                    icon="🕐",
                    message="Trading session changed",
                    old_session=old_session,
                    new_session=self.current_session,
                    clock=self.clock
                ))
            
            # Update volatility tracking
            volatility_data = alignment_data.get('volatility_data', {})
            if volatility_data:
                current_vol = np.mean(list(volatility_data.values()))
                self.volatility_history.append(current_vol)
                self.current_volatility = current_vol
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_state_update")
            self.logger.warning(f"Market state update failed: {error_context}")

    async def apply_alignment(self, weights: np.ndarray) -> np.ndarray:
        """
        Enhanced time-based scaling to weights with comprehensive adjustments
        
        Args:
            weights: Current voting weights to adjust
            
        Returns:
            Adjusted weights based on time horizons and market conditions
        """
        try:
            self.alignment_stats['total_alignments'] += 1
            
            # Validate and normalize inputs
            weights = np.asarray(weights, dtype=np.float32)
            
            # Handle dimension mismatch gracefully
            if len(weights) != len(self.horizons):
                weights = await self._handle_dimension_mismatch(weights)
            
            # Calculate comprehensive alignment factors
            distance_factors = await self._calculate_distance_factors()
            regime_factors = await self._get_regime_factors()
            session_factors = await self._get_session_factors()
            performance_factors = await self._get_performance_factors()
            volatility_factors = await self._get_volatility_factors()
            
            # Combine all factors with intelligent weighting
            combined_factors = await self._combine_alignment_factors(
                distance_factors, regime_factors, session_factors, 
                performance_factors, volatility_factors
            )
            
            # Apply alignment
            aligned_weights = weights * combined_factors
            
            # Ensure positive and normalized
            aligned_weights = np.maximum(aligned_weights, 0.01)
            aligned_weights = aligned_weights / (aligned_weights.sum() + 1e-12)
            
            # Track alignment impact and quality
            impact = np.linalg.norm(aligned_weights - weights)
            await self._track_alignment_impact(weights, aligned_weights, impact)
            
            # Record alignment event
            await self._record_alignment_event_comprehensive(
                weights, aligned_weights, combined_factors, impact
            )
            
            return aligned_weights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "alignment_application")
            self.logger.error(f"Horizon alignment failed: {error_context}")
            return np.asarray(weights, dtype=np.float32)

    async def _handle_dimension_mismatch(self, weights: np.ndarray) -> np.ndarray:
        """Handle dimension mismatch between weights and horizons"""
        try:
            self.logger.warning(format_operator_message(
                icon="🔧",
                message="Dimension mismatch detected",
                weights_dim=len(weights),
                horizons_dim=len(self.horizons),
                action="Auto-adjusting"
            ))
            
            if len(weights) > len(self.horizons):
                # Truncate weights
                adjusted_weights = weights[:len(self.horizons)]
                self.logger.info(f"Truncated weights to {len(adjusted_weights)}")
                
            elif len(weights) < len(self.horizons):
                # Pad weights with defaults
                missing_count = len(self.horizons) - len(weights)
                default_weight = 1.0 / len(self.horizons)
                padding = np.full(missing_count, default_weight, dtype=np.float32)
                adjusted_weights = np.concatenate([weights, padding])
                self.logger.info(f"Padded weights to {len(adjusted_weights)}")
                
            else:
                adjusted_weights = weights
            
            return adjusted_weights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "dimension_mismatch_handling")
            return weights

    async def _calculate_comprehensive_distance_alignment(self):
        """Calculate comprehensive distance-based alignment"""
        try:
            # Time-based distance calculation
            time_distances = 1.0 / (1.0 + np.abs(self.clock - self.horizons))
            
            # Volatility-adjusted distances
            vol_adjustment = 1.0 + self.current_volatility * 2.0
            adjusted_distances = time_distances * vol_adjustment
            
            # Normalize distances
            self.current_distances = adjusted_distances / (adjusted_distances.sum() + 1e-12)
            self.base_distances = self.current_distances.copy()
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "distance_calculation")
            self.current_distances = np.ones_like(self.horizons) / len(self.horizons)

    async def _apply_regime_and_session_adaptations(self, alignment_data: Dict[str, Any]):
        """Apply regime and session-based adaptations"""
        try:
            if not self.regime_awareness:
                return
            
            # Update regime multipliers
            await self._update_regime_multipliers_comprehensive(alignment_data)
            
            # Update session patterns
            await self._update_session_patterns_comprehensive(alignment_data)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_session_adaptation")

    async def _update_regime_multipliers_comprehensive(self, alignment_data: Dict[str, Any]):
        """Update comprehensive regime-based multipliers"""
        try:
            regime = self.current_regime
            volatility_level = self._determine_volatility_level()
            
            # Enhanced regime-specific horizon preferences
            if regime == 'trending':
                # Favor longer horizons in trending markets
                multipliers = np.array([
                    0.7 if h < 5 else 1.4 if h > 30 else 1.1 
                    for h in self.horizons
                ])
                
            elif regime == 'volatile':
                # Favor shorter horizons in volatile markets
                multipliers = np.array([
                    1.5 if h < 8 else 0.6 if h > 25 else 1.0 
                    for h in self.horizons
                ])
                
            elif regime == 'ranging':
                # Balanced approach with slight medium-term bias
                multipliers = np.array([
                    1.2 if 8 <= h <= 20 else 0.8 
                    for h in self.horizons
                ])
                
            elif regime == 'breakout':
                # Very short-term bias for breakout capture
                multipliers = np.array([
                    1.6 if h < 5 else 0.5 if h > 15 else 0.9 
                    for h in self.horizons
                ])
                
            elif regime == 'reversal':
                # Medium-term bias for reversal confirmation
                multipliers = np.array([
                    0.8 if h < 10 else 1.3 if 10 <= h <= 30 else 0.9 
                    for h in self.horizons
                ])
                
            elif regime == 'noise':
                # Very conservative, slight short-term bias
                multipliers = np.array([
                    1.3 if h < 3 else 0.7 if h > 20 else 1.0 
                    for h in self.horizons
                ])
                
            else:  # unknown
                multipliers = np.ones_like(self.horizons)
            
            # Apply volatility adjustments
            vol_config = self.volatility_adaptation.get(volatility_level, {})
            vol_bias = vol_config.get('horizon_bias', 'balanced')
            
            if vol_bias == 'short':
                multipliers *= np.array([
                    1.3 if h < 10 else 0.7 if h > 20 else 1.0 
                    for h in self.horizons
                ])
            elif vol_bias == 'long':
                multipliers *= np.array([
                    0.8 if h < 5 else 1.2 if h > 15 else 1.0 
                    for h in self.horizons
                ])
            
            # Smooth transition using exponential moving average
            if regime in self.regime_multipliers:
                alpha = self.alignment_intelligence['regime_sensitivity']
                old_multipliers = self.regime_multipliers[regime]
                self.regime_multipliers[regime] = (
                    alpha * multipliers + (1 - alpha) * old_multipliers
                )
            else:
                self.regime_multipliers[regime] = multipliers
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_multipliers_update")

    def _determine_volatility_level(self) -> str:
        """Determine current volatility level"""
        try:
            if self.current_volatility > 0.05:
                return 'extreme'
            elif self.current_volatility > 0.03:
                return 'high'
            elif self.current_volatility > 0.015:
                return 'medium'
            elif self.current_volatility > 0.008:
                return 'low'
            else:
                return 'very_low'
        except Exception:
            return 'medium'

    async def _calculate_comprehensive_alignment_quality(self) -> Dict[str, Any]:
        """Calculate comprehensive alignment quality metrics"""
        try:
            quality_analysis = {}
            
            # Effectiveness (how well alignment improves outcomes)
            if len(self.performance_history) >= 10:
                recent_performance = [p.get('improvement', 0.0) for p in list(self.performance_history)[-10:]]
                effectiveness = np.mean(recent_performance) if recent_performance else 0.5
                self.alignment_quality['effectiveness'] = effectiveness
            
            # Consistency (stability of alignment decisions)
            if len(self.alignment_history) >= 5:
                recent_impacts = [a.get('impact', 0.0) for a in list(self.alignment_history)[-10:]]
                consistency = 1.0 - np.std(recent_impacts) if recent_impacts else 0.5
                self.alignment_quality['consistency'] = max(0.0, min(1.0, consistency))
            
            # Adaptability (responsiveness to market changes)
            adaptation_score = min(1.0, self.alignment_stats['significant_adaptations'] / max(1, self.alignment_stats['total_alignments']))
            self.alignment_quality['adaptability'] = adaptation_score
            
            # Regime alignment (how well aligned with current regime)
            regime_score = self._calculate_regime_alignment_score()
            self.alignment_quality['regime_alignment'] = regime_score
            
            # Overall quality score
            quality_values = list(self.alignment_quality.values())
            overall_quality = np.mean(quality_values) if quality_values else 0.5
            
            quality_analysis = {
                **self.alignment_quality,
                'overall_quality': overall_quality,
                'quality_trend': self._determine_quality_trend(),
                'improvement_areas': self._identify_improvement_areas()
            }
            
            return quality_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "alignment_quality_calculation")
            return {'overall_quality': 0.5, 'quality_trend': 'unknown'}

    def _calculate_regime_alignment_score(self) -> float:
        """Calculate how well current alignment matches regime expectations"""
        try:
            regime = self.current_regime
            if regime not in self.regime_multipliers:
                return 0.5
            
            expected_multipliers = self.regime_multipliers[regime]
            current_combined = self._get_combined_multipliers()
            
            # Calculate similarity
            similarity = 1.0 - np.mean(np.abs(expected_multipliers - current_combined)) / 2.0
            return max(0.0, min(1.0, similarity))
            
        except Exception:
            return 0.5

    def _get_combined_multipliers(self) -> np.ndarray:
        """Get combined multipliers from all sources"""
        try:
            combined = self.adaptive_multipliers.copy()
            
            # Apply regime multipliers
            if self.current_regime in self.regime_multipliers:
                combined *= self.regime_multipliers[self.current_regime]
            
            # Apply performance multipliers
            if self.performance_feedback:
                combined *= self.performance_multipliers
            
            # Apply session patterns
            if self.current_session in self.session_patterns:
                combined *= self.session_patterns[self.current_session]
            
            # Apply cyclical adjustments
            combined *= self.cyclical_adjustments
            
            return combined
            
        except Exception:
            return np.ones_like(self.horizons)

    async def _generate_comprehensive_alignment_thesis(self, results: Dict[str, Any], 
                                                     quality_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive alignment thesis"""
        try:
            overall_quality = quality_analysis.get('overall_quality', 0.5)
            
            # Core metrics
            thesis_parts = []
            
            # Executive summary
            alignment_effectiveness = "HIGH" if overall_quality > 0.7 else "MODERATE" if overall_quality > 0.4 else "LOW"
            thesis_parts.append(
                f"HORIZON ALIGNMENT: {alignment_effectiveness} effectiveness with {overall_quality:.1%} quality score"
            )
            
            # Market adaptation
            thesis_parts.append(
                f"MARKET ADAPTATION: {self.current_regime} regime with {self.current_session} session patterns"
            )
            
            # Performance impact
            avg_impact = self.alignment_stats.get('avg_alignment_impact', 0.0)
            thesis_parts.append(f"ALIGNMENT IMPACT: {avg_impact:.3f} average weight adjustment magnitude")
            
            # System performance
            total_alignments = self.alignment_stats.get('total_alignments', 0)
            adaptations = self.alignment_stats.get('significant_adaptations', 0)
            thesis_parts.append(f"SYSTEM PERFORMANCE: {total_alignments} alignments with {adaptations} adaptations")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "alignment_thesis_generation")
            return f"Alignment thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive alignment results"""
        try:
            # Core alignment results
            if results.get('aligned_weights') is not None:
                self.smart_bus.set('aligned_weights', results['aligned_weights'],
                                 module='TimeHorizonAligner', thesis=thesis)
            
            # Horizon distances and multipliers
            self.smart_bus.set('horizon_distances', results['horizon_distances'],
                             module='TimeHorizonAligner', 
                             thesis=f"Horizon distances: {len(results['horizon_distances'])} time scales analyzed")
            
            self.smart_bus.set('horizon_multipliers', results['horizon_multipliers'],
                             module='TimeHorizonAligner',
                             thesis=f"Horizon multipliers: Combined scaling factors for {len(results['horizon_multipliers'])} horizons")
            
            # Regime and session adaptations
            self.smart_bus.set('regime_adjustments', results['regime_adjustments'],
                             module='TimeHorizonAligner',
                             thesis=f"Regime adjustments: {self.current_regime} market regime adaptations")
            
            self.smart_bus.set('session_patterns', results['session_patterns'],
                             module='TimeHorizonAligner',
                             thesis=f"Session patterns: {self.current_session} session optimizations")
            
            # Quality and performance metrics
            self.smart_bus.set('alignment_quality', results['alignment_quality'],
                             module='TimeHorizonAligner',
                             thesis=f"Alignment quality: {results['alignment_quality'].get('overall_quality', 0.5):.1%} effectiveness")
            
            self.smart_bus.set('performance_metrics', results['performance_metrics'],
                             module='TimeHorizonAligner',
                             thesis=f"Performance metrics: Comprehensive alignment analytics")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY COMPATIBILITY AND PUBLIC INTERFACE
    # ═══════════════════════════════════════════════════════════════════

    def apply(self, weights: np.ndarray) -> np.ndarray:
        """Legacy apply interface for backward compatibility"""
        try:
            import asyncio
            
            if asyncio.get_event_loop().is_running():
                # Already in async context - use simplified sync method
                return self._simple_alignment_fallback(weights)
            else:
                # Run async alignment
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.apply_alignment(weights))
                finally:
                    loop.close()
                    
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "legacy_apply")
            return self._simple_alignment_fallback(weights)

    def _simple_alignment_fallback(self, weights: np.ndarray) -> np.ndarray:
        """Simple fallback alignment method"""
        try:
            weights = np.asarray(weights, dtype=np.float32)
            
            # Handle dimension mismatch
            if len(weights) != len(self.horizons):
                if len(weights) > len(self.horizons):
                    weights = weights[:len(self.horizons)]
                elif len(weights) < len(self.horizons):
                    missing = len(self.horizons) - len(weights)
                    weights = np.concatenate([weights, np.ones(missing) / len(self.horizons)])
            
            # Simple distance-based scaling
            distances = 1.0 / (1.0 + np.abs(self.clock - self.horizons))
            distances = distances / (distances.sum() + 1e-12)
            
            # Apply basic regime multiplier
            regime_mult = self.regime_multipliers.get(self.current_regime, np.ones_like(self.horizons))
            
            # Combine factors
            aligned_weights = weights * distances * regime_mult
            aligned_weights = np.maximum(aligned_weights, 0.01)
            aligned_weights = aligned_weights / (aligned_weights.sum() + 1e-12)
            
            return aligned_weights
            
        except Exception:
            return np.asarray(weights, dtype=np.float32)

    def resize(self, new_horizons: List[int]) -> None:
        """Resize for different time horizons"""
        old_horizons = self.horizons.copy()
        self.horizons = np.array(new_horizons, dtype=np.float32)
        
        # Reinitialize arrays
        self.current_distances = np.ones_like(self.horizons)
        self.base_distances = np.ones_like(self.horizons)
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.performance_multipliers = np.ones_like(self.horizons)
        self.cyclical_adjustments = np.ones_like(self.horizons)
        
        # Reinitialize regime multipliers
        for regime in self.regime_multipliers:
            self.regime_multipliers[regime] = np.ones_like(self.horizons)
        
        # Reinitialize session patterns
        for session in self.session_patterns:
            self.session_patterns[session] = np.ones_like(self.horizons)
        
        self.logger.info(format_operator_message(
            icon="🔄",
            message="Time Horizon Aligner resized",
            old_horizons=old_horizons.tolist(),
            new_horizons=self.horizons.tolist()
        ))

    def get_observation_components(self) -> np.ndarray:
        """Return horizon alignment features for RL observation"""
        try:
            features = [
                float(self.clock % 1000) / 1000.0,  # Normalized clock
                float(self.alignment_stats.get('avg_alignment_impact', 0.0)),
                float(np.mean(self.current_distances)),
                float(np.mean(self.adaptive_multipliers)),
                float(np.mean(self.performance_multipliers)),
                float(self.alignment_quality.get('overall_quality', 0.5)),
                float(len(self.alignment_history) / 200.0),  # History fullness
                float(self.current_volatility * 10.0)  # Scaled volatility
            ]
            
            observation = np.array(features, dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid alignment observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Alignment observation generation failed: {error_context}")
            return np.array([0.5, 0.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.2], dtype=np.float32)

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'TimeHorizonAligner',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'total_alignments': self.alignment_stats.get('total_alignments', 0),
            'clock': self.clock,
            'current_regime': self.current_regime,
            'current_session': self.current_session,
            'current_volatility': self.current_volatility,
            'alignment_quality': self.alignment_quality.get('overall_quality', 0.5),
            'horizons_count': len(self.horizons),
            'adaptation_count': self.alignment_stats.get('significant_adaptations', 0),
            'regime_switches': self.alignment_stats.get('regime_switches', 0),
            'session_duration': (datetime.datetime.now() - 
                               datetime.datetime.fromisoformat(self.alignment_stats['session_start_time'])).total_seconds() / 3600
        }

    # ═══════════════════════════════════════════════════════════════════
    # ADDITIONAL HELPER METHODS AND STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _get_safe_alignment_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when data retrieval fails"""
        return {
            'voting_weights': [], 'market_regime': 'unknown', 'session_type': 'unknown',
            'volatility_data': {}, 'market_context': {}, 'time_of_day': 0,
            'performance_feedback': {}, 'member_confidences': [], 'recent_trades': [],
            'expert_performance': {}
        }

    def _get_performance_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        try:
            return {
                'total_alignments': self.alignment_stats.get('total_alignments', 0),
                'significant_adaptations': self.alignment_stats.get('significant_adaptations', 0),
                'avg_alignment_impact': self.alignment_stats.get('avg_alignment_impact', 0.0),
                'regime_switches': self.alignment_stats.get('regime_switches', 0),
                'session_transitions': self.alignment_stats.get('session_transitions', 0),
                'effectiveness_trend': self.alignment_stats.get('effectiveness_trend', 0.0),
                'adaptation_accuracy': self.alignment_stats.get('adaptation_accuracy', 0.5)
            }
        except Exception:
            return {}

    def _get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status"""
        try:
            return {
                'adaptive_scaling': self.adaptive_scaling,
                'regime_awareness': self.regime_awareness,
                'performance_feedback': self.performance_feedback,
                'current_regime': self.current_regime,
                'current_session': self.current_session,
                'volatility_level': self._determine_volatility_level(),
                'last_adaptation': self.adaptation_events[-1] if self.adaptation_events else None,
                'alignment_intelligence': self.alignment_intelligence.copy()
            }
        except Exception:
            return {'status': 'error'}

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "TimeHorizonAligner")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Time Horizon Aligner disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('TimeHorizonAligner', 'process_time', processing_time, False)
        
        return {
            'aligned_weights': None,
            'horizon_distances': np.ones_like(self.horizons).tolist(),
            'horizon_multipliers': np.ones_like(self.horizons).tolist(),
            'regime_adjustments': np.ones_like(self.horizons).tolist(),
            'session_patterns': np.ones_like(self.horizons).tolist(),
            'alignment_quality': {'overall_quality': 0.5, 'error': str(error_context)},
            'performance_metrics': {'error': str(error_context)},
            'adaptation_status': {'status': 'error', 'error_context': str(error_context)},
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        
        # Reset time tracking
        self.clock = 0
        self.session_start = 0
        
        # Reset alignment state
        self.current_distances = np.ones_like(self.horizons)
        self.base_distances = np.ones_like(self.horizons)
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.performance_multipliers = np.ones_like(self.horizons)
        self.cyclical_adjustments = np.ones_like(self.horizons)
        
        # Reset market state
        self.current_regime = 'unknown'
        self.current_session = 'unknown'
        self.current_volatility = 0.02
        
        # Reset multipliers to neutral
        for regime in self.regime_multipliers:
            self.regime_multipliers[regime] = np.ones_like(self.horizons)
        
        for session in self.session_patterns:
            self.session_patterns[session] = np.ones_like(self.horizons)
        
        # Reset history and tracking
        self.alignment_history.clear()
        self.adaptation_events.clear()
        self.performance_history.clear()
        self.volatility_history.clear()
        self.horizon_performance.clear()
        
        # Reset statistics
        self.alignment_stats = {
            'total_alignments': 0,
            'significant_adaptations': 0,
            'regime_switches': 0,
            'session_transitions': 0,
            'performance_adjustments': 0,
            'avg_alignment_impact': 0.0,
            'effectiveness_trend': 0.0,
            'adaptation_accuracy': 0.5,
            'session_start_time': datetime.datetime.now().isoformat()
        }
        
        # Reset quality metrics
        self.alignment_quality = {
            'effectiveness': 0.5,
            'consistency': 0.5,
            'adaptability': 0.5,
            'regime_alignment': 0.5,
            'session_optimization': 0.5,
            'performance_correlation': 0.5
        }
        
        # Reset error state
        self.error_count = 0
        self.is_disabled = False
        
        self.logger.info(format_operator_message(
            icon="🔄",
            message="Time Horizon Aligner reset completed",
            status="All alignment state cleared and systems reinitialized"
        ))