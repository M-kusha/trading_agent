"""
🎲 Enhanced Alternative Reality Sampler with SmartInfoBus Integration v3.0
Advanced alternative voting outcome sampling for robustness testing and uncertainty quantification
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
    name="AlternativeRealitySampler",
    version="3.0.0",
    category="voting",
    provides=[
        "alternative_samples", "sampling_uncertainty", "diversity_score", "sampling_stats",
        "effective_samples", "confidence_bounds", "sampling_recommendations"
    ],
    requires=[
        "votes", "voting_summary", "strategy_arbiter_weights", "market_context", "volatility_data",
        "consensus_direction", "agreement_score", "market_regime"
    ],
    description="Advanced alternative voting outcome sampling for robustness testing and uncertainty quantification",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=80,
    priority=4,
    explainable=True,
    hot_reload=True
)
class AlternativeRealitySampler(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🎲 PRODUCTION-GRADE Alternative Reality Sampler v3.0
    
    Advanced alternative reality sampling system with:
    - Intelligent adaptive sampling based on market conditions and uncertainty
    - Multi-dimensional exploration with structured and random sampling techniques
    - Comprehensive uncertainty quantification and robustness testing
    - SmartInfoBus zero-wiring architecture
    - Real-time effectiveness monitoring and optimization
    """

    def _initialize(self):
        """Initialize advanced alternative reality sampling systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced sampling configuration
        self.dim = self.config.get('dim', 5)
        self.n_samples = self.config.get('n_samples', 8)
        self.base_sigma = self.config.get('sigma', 0.05)
        self.current_sigma = self.base_sigma
        self.adaptive_sigma = self.config.get('adaptive_sigma', True)
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.3)
        self.debug = self.config.get('debug', False)
        
        # Initialize comprehensive sampling methods
        self.sampling_methods = self._initialize_sampling_methods()
        
        # Core sampling state with enhanced tracking
        self.last_samples = None
        self.last_weights = None
        self.last_base_weights = None
        self.sampling_history = deque(maxlen=100)
        self.uncertainty_history = deque(maxlen=200)
        self.effectiveness_history = deque(maxlen=150)
        
        # Enhanced performance tracking
        self.sampling_stats = {
            'samples_generated': 0,
            'total_samples_created': 0,
            'avg_uncertainty': 0.5,
            'sigma_adaptations': 0,
            'effective_samples': 0,
            'diversity_score': 0.0,
            'convergence_rate': 0.0,
            'exploration_efficiency': 0.0,
            'best_uncertainty_estimate': 0.5,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Adaptive intelligence parameters
        self.sampling_intelligence = {
            'sigma_bounds': (0.005, 0.25),
            'adaptation_rate': 0.12,
            'diversity_target': 0.15,
            'convergence_threshold': 0.02,
            'exploration_momentum': 0.85,
            'uncertainty_sensitivity': 0.7,
            'effectiveness_memory': 0.9
        }
        
        # Advanced sampling strategies
        self.sampling_strategies = {
            'random_gaussian': {'weight': 0.4, 'active': True},
            'structured_perturbation': {'weight': 0.3, 'active': True},
            'systematic_exploration': {'weight': 0.2, 'active': True},
            'uncertainty_guided': {'weight': 0.1, 'active': True}
        }
        
        # Market condition adaptation
        self.market_adaptation = {
            'regime_multipliers': {
                'trending': 0.8,
                'ranging': 1.0,
                'volatile': 1.6,
                'breakout': 1.2,
                'reversal': 1.4,
                'unknown': 1.1
            },
            'volatility_multipliers': {
                'very_low': 0.6,
                'low': 0.8,
                'medium': 1.0,
                'high': 1.4,
                'extreme': 2.0
            }
        }
        
        # Quality metrics and analytics
        self.quality_metrics = {
            'sample_diversity': 0.0,
            'coverage_efficiency': 0.0,
            'uncertainty_accuracy': 0.0,
            'adaptation_success_rate': 0.0,
            'exploration_completeness': 0.0
        }
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🎲",
            message=f"Alternative Reality Sampler v{version} initialized",
            dimensions=self.dim,
            samples=self.n_samples,
            base_sigma=f"{self.base_sigma:.3f}",
            adaptive=self.adaptive_sigma
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="AlternativeRealitySampler",
            log_path="logs/voting/alternative_reality_sampler.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("AlternativeRealitySampler", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _initialize_sampling_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive sampling method definitions"""
        return {
            'random_gaussian': {
                'description': 'Standard Gaussian noise sampling around base weights',
                'parameters': {'sigma_multiplier': 1.0, 'clipping': True},
                'use_cases': ['general_exploration', 'baseline_sampling'],
                'effectiveness_threshold': 0.4,
                'computational_cost': 'low'
            },
            'structured_perturbation': {
                'description': 'Systematic single-dimension perturbations for targeted exploration',
                'parameters': {'perturbation_strength': 2.0, 'rotation_enabled': True},
                'use_cases': ['dimension_analysis', 'sensitivity_testing'],
                'effectiveness_threshold': 0.6,
                'computational_cost': 'medium'
            },
            'systematic_exploration': {
                'description': 'Grid-based exploration with adaptive spacing',
                'parameters': {'grid_density': 0.1, 'adaptive_spacing': True},
                'use_cases': ['comprehensive_coverage', 'boundary_testing'],
                'effectiveness_threshold': 0.5,
                'computational_cost': 'high'
            },
            'uncertainty_guided': {
                'description': 'Sampling guided by uncertainty estimates and confidence bounds',
                'parameters': {'uncertainty_scaling': 1.5, 'confidence_targeting': True},
                'use_cases': ['robustness_testing', 'confidence_bounds'],
                'effectiveness_threshold': 0.7,
                'computational_cost': 'medium'
            },
            'adaptive_momentum': {
                'description': 'Momentum-based sampling with historical gradient information',
                'parameters': {'momentum_factor': 0.8, 'gradient_memory': 10},
                'use_cases': ['convergence_acceleration', 'trend_following'],
                'effectiveness_threshold': 0.65,
                'computational_cost': 'medium'
            }
        }

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Alternative Reality Sampler v3.0 Initialization Complete:
        
        Advanced Sampling Framework:
        - Multi-dimensional sampling space: {self.dim} dimensions with {self.n_samples} samples per iteration
        - Adaptive sampling algorithms with intelligent sigma adjustment ({self.base_sigma:.3f} base)
        - Comprehensive uncertainty quantification and robustness testing capabilities
        - Market-aware sampling adaptation based on regime and volatility conditions
        
        Current Configuration:
        - Sampling methods: {len(self.sampling_methods)} distinct approaches available
        - Adaptive sigma: {'enabled' if self.adaptive_sigma else 'disabled'} with bounds [{self.sampling_intelligence['sigma_bounds'][0]:.3f}, {self.sampling_intelligence['sigma_bounds'][1]:.3f}]
        - Uncertainty threshold: {self.uncertainty_threshold:.1%} for exploration triggering
        - Diversity target: {self.sampling_intelligence['diversity_target']:.1%} for sample spread optimization
        
        Sampling Intelligence Features:
        - Market regime adaptation with volatility-aware scaling
        - Multi-strategy sampling with effectiveness-based weighting
        - Real-time convergence monitoring and exploration efficiency tracking
        - Comprehensive quality metrics and performance analytics
        
        Advanced Capabilities:
        - Uncertainty-guided exploration for targeted robustness testing
        - Structured perturbation analysis for sensitivity assessment
        - Adaptive momentum sampling for convergence acceleration
        - Real-time effectiveness monitoring and strategy optimization
        
        Expected Outcomes:
        - Enhanced decision robustness through comprehensive alternative scenario testing
        - Improved uncertainty quantification with confidence bounds estimation
        - Optimal exploration efficiency adapted to current market conditions
        - Transparent sampling decisions with detailed quality metrics and analytics
        """
        
        self.smart_bus.set('alternative_reality_sampler_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'dimensions': self.dim,
                'samples_per_iteration': self.n_samples,
                'sampling_methods': list(self.sampling_methods.keys()),
                'intelligence_parameters': self.sampling_intelligence
            }
        }, module='AlternativeRealitySampler', thesis=thesis)

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive sampling analysis
        
        Returns:
            Dict containing sampling results, uncertainty metrics, and analytics
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive voting data from SmartInfoBus
            voting_data = await self._get_comprehensive_voting_data()
            
            # Update sampling parameters based on market conditions
            await self._update_sampling_parameters_comprehensive(voting_data)
            
            # Analyze recent sampling effectiveness
            effectiveness_analysis = await self._analyze_sampling_effectiveness_comprehensive(voting_data)
            
            # Update sampling strategy weights based on performance
            strategy_updates = await self._update_sampling_strategy_weights(effectiveness_analysis)
            
            # Calculate comprehensive quality metrics
            quality_analysis = await self._calculate_comprehensive_quality_metrics()
            
            # Generate sampling recommendations
            recommendations = await self._generate_intelligent_sampling_recommendations(
                effectiveness_analysis, quality_analysis
            )
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_sampling_thesis(
                effectiveness_analysis, quality_analysis, strategy_updates
            )
            
            # Create comprehensive results
            results = {
                'alternative_samples': self.last_samples.tolist() if self.last_samples is not None else [],
                'sampling_uncertainty': self.sampling_stats.get('avg_uncertainty', 0.5),
                'diversity_score': self.sampling_stats.get('diversity_score', 0.0),
                'sampling_stats': self._get_comprehensive_sampling_stats(),
                'effective_samples': self.sampling_stats.get('effective_samples', 0),
                'confidence_bounds': self._calculate_confidence_bounds(),
                'sampling_recommendations': recommendations,
                'quality_metrics': quality_analysis,
                'strategy_performance': strategy_updates,
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('AlternativeRealitySampler', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_voting_data(self) -> Dict[str, Any]:
        """Get comprehensive voting data using modern SmartInfoBus patterns"""
        try:
            return {
                'votes': self.smart_bus.get('votes', 'AlternativeRealitySampler') or [],
                'voting_summary': self.smart_bus.get('voting_summary', 'AlternativeRealitySampler') or {},
                'strategy_arbiter_weights': self.smart_bus.get('strategy_arbiter_weights', 'AlternativeRealitySampler') or [],
                'market_context': self.smart_bus.get('market_context', 'AlternativeRealitySampler') or {},
                'volatility_data': self.smart_bus.get('volatility_data', 'AlternativeRealitySampler') or {},
                'consensus_direction': self.smart_bus.get('consensus_direction', 'AlternativeRealitySampler') or 'neutral',
                'agreement_score': self.smart_bus.get('agreement_score', 'AlternativeRealitySampler') or 0.5,
                'market_regime': self.smart_bus.get('market_regime', 'AlternativeRealitySampler') or 'unknown',
                'recent_trades': self.smart_bus.get('recent_trades', 'AlternativeRealitySampler') or [],
                'session_metrics': self.smart_bus.get('session_metrics', 'AlternativeRealitySampler') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "AlternativeRealitySampler")
            self.logger.warning(f"Voting data retrieval incomplete: {error_context}")
            return self._get_safe_voting_defaults()

    async def _update_sampling_parameters_comprehensive(self, voting_data: Dict[str, Any]):
        """Update comprehensive sampling parameters with intelligent adaptation"""
        try:
            if not self.adaptive_sigma:
                return
            
            # Extract market intelligence
            market_context = voting_data.get('market_context', {})
            regime = voting_data.get('market_regime', 'unknown')
            volatility_level = market_context.get('volatility_level', 'medium')
            agreement_score = voting_data.get('agreement_score', 0.5)
            
            # Calculate market uncertainty factor
            market_uncertainty = self._calculate_market_uncertainty_factor(voting_data)
            
            # Calculate base adaptation multiplier
            base_multiplier = 1.0
            
            # Apply regime-based adaptation
            regime_multiplier = self.market_adaptation['regime_multipliers'].get(regime, 1.0)
            base_multiplier *= regime_multiplier
            
            # Apply volatility-based adaptation
            volatility_multiplier = self.market_adaptation['volatility_multipliers'].get(volatility_level, 1.0)
            base_multiplier *= volatility_multiplier
            
            # Apply agreement-based adaptation (inverse relationship)
            if agreement_score < 0.3:
                base_multiplier *= 1.5  # High disagreement = more exploration
            elif agreement_score > 0.8:
                base_multiplier *= 0.7  # High agreement = less exploration
            
            # Apply uncertainty-based scaling
            uncertainty_factor = 1.0 + (market_uncertainty - 0.5) * self.sampling_intelligence['uncertainty_sensitivity']
            base_multiplier *= uncertainty_factor
            
            # Calculate target sigma with intelligent bounds
            target_sigma = self.base_sigma * base_multiplier
            target_sigma = np.clip(
                target_sigma, 
                self.sampling_intelligence['sigma_bounds'][0], 
                self.sampling_intelligence['sigma_bounds'][1]
            )
            
            # Apply momentum-based smooth adaptation
            momentum = self.sampling_intelligence['exploration_momentum']
            adaptation_rate = self.sampling_intelligence['adaptation_rate']
            
            old_sigma = self.current_sigma
            self.current_sigma = (
                old_sigma * momentum * (1 - adaptation_rate) +
                target_sigma * adaptation_rate +
                old_sigma * (1 - momentum) * 0.1  # Stability component
            )
            
            # Track significant adaptations
            sigma_change = abs(self.current_sigma - old_sigma)
            if sigma_change > 0.005:  # Threshold for significant change
                self.sampling_stats['sigma_adaptations'] += 1
                
                adaptation_record = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'old_sigma': old_sigma,
                    'new_sigma': self.current_sigma,
                    'regime': regime,
                    'volatility_level': volatility_level,
                    'agreement_score': agreement_score,
                    'market_uncertainty': market_uncertainty,
                    'base_multiplier': base_multiplier
                }
                
                # Log significant adaptation
                self.logger.info(format_operator_message(
                    icon="[STATS]",
                    message="Sampling sigma adapted",
                    old_sigma=f"{old_sigma:.4f}",
                    new_sigma=f"{self.current_sigma:.4f}",
                    regime=regime,
                    volatility=volatility_level,
                    agreement=f"{agreement_score:.1%}",
                    uncertainty=f"{market_uncertainty:.3f}"
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "sampling_parameters_update")
            self.logger.warning(f"Sampling parameter update failed: {error_context}")

    def _calculate_market_uncertainty_factor(self, voting_data: Dict[str, Any]) -> float:
        """Calculate comprehensive market uncertainty factor"""
        try:
            uncertainty_components = []
            
            # Agreement uncertainty
            agreement_score = voting_data.get('agreement_score', 0.5)
            agreement_uncertainty = 1.0 - agreement_score
            uncertainty_components.append(agreement_uncertainty)
            
            # Volatility uncertainty
            market_context = voting_data.get('market_context', {})
            volatility_level = market_context.get('volatility_level', 'medium')
            volatility_uncertainty = {
                'very_low': 0.1, 'low': 0.3, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0
            }.get(volatility_level, 0.5)
            uncertainty_components.append(volatility_uncertainty)
            
            # Regime uncertainty
            regime = voting_data.get('market_regime', 'unknown')
            regime_uncertainty = 0.9 if regime == 'unknown' else 0.2
            uncertainty_components.append(regime_uncertainty)
            
            # Recent performance uncertainty
            recent_trades = voting_data.get('recent_trades', [])
            if len(recent_trades) >= 3:
                recent_pnls = [t.get('pnl', 0) for t in recent_trades[-5:]]
                if recent_pnls:
                    pnl_volatility = np.std(recent_pnls) / (abs(np.mean(recent_pnls)) + 10)
                    performance_uncertainty = min(1.0, pnl_volatility)
                    uncertainty_components.append(performance_uncertainty)
            
            # Weighted combination
            if uncertainty_components:
                weights = [0.3, 0.3, 0.2, 0.2][:len(uncertainty_components)]
                weights = np.array(weights) / np.sum(weights)  # Normalize
                total_uncertainty = np.average(uncertainty_components, weights=weights)
            else:
                total_uncertainty = 0.5
            
            return np.clip(total_uncertainty, 0.0, 1.0)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_uncertainty_calculation")
            return 0.5

    async def _analyze_sampling_effectiveness_comprehensive(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comprehensive sampling effectiveness with advanced metrics"""
        try:
            effectiveness_analysis = {
                'overall_effectiveness': 0.5,
                'diversity_analysis': {},
                'convergence_analysis': {},
                'exploration_efficiency': 0.0,
                'uncertainty_accuracy': 0.0,
                'strategy_performance': {}
            }
            
            if len(self.sampling_history) < 5:
                effectiveness_analysis['overall_effectiveness'] = 0.5
                effectiveness_analysis['insufficient_data'] = True
                return effectiveness_analysis
            
            # Analyze recent sampling diversity
            diversity_analysis = await self._analyze_sampling_diversity()
            effectiveness_analysis['diversity_analysis'] = diversity_analysis
            
            # Analyze convergence patterns
            convergence_analysis = await self._analyze_convergence_patterns()
            effectiveness_analysis['convergence_analysis'] = convergence_analysis
            
            # Calculate exploration efficiency
            exploration_efficiency = await self._calculate_exploration_efficiency()
            effectiveness_analysis['exploration_efficiency'] = exploration_efficiency
            
            # Analyze uncertainty estimation accuracy
            uncertainty_accuracy = await self._analyze_uncertainty_accuracy(voting_data)
            effectiveness_analysis['uncertainty_accuracy'] = uncertainty_accuracy
            
            # Evaluate strategy-specific performance
            strategy_performance = await self._evaluate_strategy_performance()
            effectiveness_analysis['strategy_performance'] = strategy_performance
            
            # Calculate overall effectiveness score
            overall_effectiveness = (
                0.25 * diversity_analysis.get('diversity_score', 0.5) +
                0.25 * convergence_analysis.get('convergence_score', 0.5) +
                0.20 * exploration_efficiency +
                0.15 * uncertainty_accuracy +
                0.15 * np.mean(list(strategy_performance.values())) if strategy_performance else 0.5
            )
            
            effectiveness_analysis['overall_effectiveness'] = np.clip(overall_effectiveness, 0.0, 1.0)
            
            # Update tracking
            self.effectiveness_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'overall_effectiveness': effectiveness_analysis['overall_effectiveness'],
                'diversity_score': diversity_analysis.get('diversity_score', 0.5),
                'exploration_efficiency': exploration_efficiency,
                'uncertainty_accuracy': uncertainty_accuracy
            })
            
            return effectiveness_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "effectiveness_analysis")
            self.logger.warning(f"Sampling effectiveness analysis failed: {error_context}")
            return {'overall_effectiveness': 0.5, 'analysis_error': str(error_context)}

    async def _analyze_sampling_diversity(self) -> Dict[str, Any]:
        """Analyze sampling diversity with comprehensive metrics"""
        try:
            if len(self.sampling_history) < 3:
                return {'diversity_score': 0.5, 'insufficient_data': True}
            
            recent_samples = list(self.sampling_history)[-10:]
            diversities = []
            spreads = []
            coverage_scores = []
            
            for sample_data in recent_samples:
                samples = sample_data.get('samples', [])
                if len(samples) > 1:
                    samples_array = np.array(samples)
                    
                    # Calculate pairwise diversity
                    pairwise_distances = []
                    for i in range(len(samples_array)):
                        for j in range(i + 1, len(samples_array)):
                            distance = np.linalg.norm(samples_array[i] - samples_array[j])
                            pairwise_distances.append(distance)
                    
                    if pairwise_distances:
                        diversity = np.mean(pairwise_distances)
                        diversities.append(diversity)
                        
                        # Calculate spread (max distance from center)
                        center = np.mean(samples_array, axis=0)
                        distances_from_center = [np.linalg.norm(s - center) for s in samples_array]
                        spread = np.max(distances_from_center)
                        spreads.append(spread)
                        
                        # Calculate coverage score (how well samples cover the space)
                        coverage = 1.0 - (np.std(pairwise_distances) / (np.mean(pairwise_distances) + 1e-8))
                        coverage_scores.append(coverage)
            
            # Aggregate metrics
            diversity_analysis = {
                'diversity_score': np.mean(diversities) if diversities else 0.5,
                'spread_score': np.mean(spreads) if spreads else 0.0,
                'coverage_score': np.mean(coverage_scores) if coverage_scores else 0.0,
                'diversity_trend': self._calculate_diversity_trend(diversities),
                'sample_count': len(recent_samples)
            }
            
            # Normalize diversity score
            target_diversity = self.sampling_intelligence['diversity_target']
            normalized_diversity = min(1.0, diversity_analysis['diversity_score'] / target_diversity)
            diversity_analysis['normalized_diversity'] = normalized_diversity
            
            return diversity_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "diversity_analysis")
            return {'diversity_score': 0.5, 'analysis_error': str(error_context)}

    def _calculate_coverage_score(self, samples: np.ndarray) -> float:
        """Calculate how well samples cover the sampling space"""
        try:
            if len(samples) < 2:
                return 0.0
            
            # Calculate coverage using nearest neighbor distances
            min_distances = []
            for i, sample in enumerate(samples):
                distances = [np.linalg.norm(sample - other) for j, other in enumerate(samples) if i != j]
                if distances:
                    min_distances.append(min(distances))
            
            if not min_distances:
                return 0.0
            
            # Coverage score based on uniformity of minimum distances
            coverage = 1.0 - (np.std(min_distances) / (np.mean(min_distances) + 1e-8))
            return float(max(0.0, min(1.0, coverage)))
            
        except Exception:
            return 0.0

    def _calculate_diversity_trend(self, diversities: List[float]) -> str:
        """Calculate trend in diversity over time"""
        try:
            if len(diversities) < 3:
                return 'insufficient_data'
            
            # Linear regression for trend
            x = np.arange(len(diversities))
            slope = np.polyfit(x, diversities, 1)[0]
            
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'

    async def _analyze_convergence_patterns(self) -> Dict[str, Any]:
        """Analyze convergence patterns in sampling"""
        try:
            if len(self.uncertainty_history) < 10:
                return {'convergence_score': 0.5, 'insufficient_data': True}
            
            recent_uncertainties = list(self.uncertainty_history)[-20:]
            
            # Calculate convergence rate
            convergence_rate = self._calculate_convergence_rate(recent_uncertainties)
            
            # Analyze stability
            stability_score = self._calculate_stability_score(recent_uncertainties)
            
            # Detect convergence patterns
            convergence_pattern = self._detect_convergence_pattern(recent_uncertainties)
            
            convergence_analysis = {
                'convergence_rate': convergence_rate,
                'stability_score': stability_score,
                'convergence_pattern': convergence_pattern,
                'convergence_score': (convergence_rate + stability_score) / 2,
                'recent_uncertainty_trend': self._calculate_uncertainty_trend(recent_uncertainties)
            }
            
            return convergence_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "convergence_analysis")
            return {'convergence_score': 0.5, 'analysis_error': str(error_context)}

    def _calculate_convergence_rate(self, uncertainties: List[float]) -> float:
        """Calculate convergence rate from uncertainty history"""
        try:
            if len(uncertainties) < 5:
                return 0.5
            
            # Calculate how quickly uncertainty is stabilizing
            recent = uncertainties[-5:]
            earlier = uncertainties[-10:-5] if len(uncertainties) >= 10 else uncertainties[:-5]
            
            if not earlier:
                return 0.5
            
            recent_std = np.std(recent)
            earlier_std = np.std(earlier)
            
            if earlier_std == 0:
                return 1.0 if recent_std < 0.01 else 0.5
            
            improvement = (earlier_std - recent_std) / earlier_std
            convergence_rate = max(0.0, min(1.0, 0.5 + improvement))
            
            return float(convergence_rate)
            
        except Exception:
            return 0.5

    def _calculate_stability_score(self, uncertainties: List[float]) -> float:
        """Calculate stability score from uncertainty history"""
        try:
            if len(uncertainties) < 3:
                return 0.5
            
            # Stability based on variance in recent uncertainties
            variance = np.var(uncertainties)
            stability = max(0.0, 1.0 - variance * 10)  # Scale variance appropriately
            
            return float(min(1.0, stability))
            
        except Exception:
            return 0.5

    def _detect_convergence_pattern(self, uncertainties: List[float]) -> str:
        """Detect convergence pattern from uncertainty history"""
        try:
            if len(uncertainties) < 5:
                return 'insufficient_data'
            
            recent_slope = self._calculate_slope(uncertainties[-5:])
            overall_slope = self._calculate_slope(uncertainties)
            
            if abs(recent_slope) < 0.01 and abs(overall_slope) < 0.01:
                return 'converged'
            elif recent_slope < -0.05:
                return 'converging'
            elif recent_slope > 0.05:
                return 'diverging'
            elif abs(recent_slope) < 0.03:
                return 'oscillating'
            else:
                return 'trending'
                
        except Exception:
            return 'unknown'

    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate slope of values using linear regression"""
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return slope
            
        except Exception:
            return 0.0

    def _calculate_uncertainty_trend(self, uncertainties: List[float]) -> str:
        """Calculate trend in uncertainty values"""
        try:
            if len(uncertainties) < 3:
                return 'insufficient_data'
            
            slope = self._calculate_slope(uncertainties)
            
            if slope > 0.02:
                return 'increasing'
            elif slope < -0.02:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'

    async def _calculate_exploration_efficiency(self) -> float:
        """Calculate exploration efficiency score"""
        try:
            if len(self.sampling_history) < 3:
                return 0.5
            
            # Calculate ratio of effective samples to total samples
            total_samples = self.sampling_stats.get('total_samples_created', 1)
            effective_samples = self.sampling_stats.get('effective_samples', 0)
            
            efficiency = effective_samples / max(total_samples, 1)
            
            # Adjust for recent trend
            if len(self.effectiveness_history) >= 3:
                recent_efficiencies = [e.get('exploration_efficiency', 0.5) for e in list(self.effectiveness_history)[-3:]]
                trend = self._calculate_slope(recent_efficiencies)
                efficiency += trend * 0.1  # Small trend bonus/penalty
            
            return np.clip(efficiency, 0.0, 1.0)
            
        except Exception:
            return 0.5

    async def _analyze_uncertainty_accuracy(self, voting_data: Dict[str, Any]) -> float:
        """Analyze accuracy of uncertainty estimation"""
        try:
            if len(self.uncertainty_history) < 5:
                return 0.5
            
            # Simplified accuracy assessment based on consistency
            recent_uncertainties = list(self.uncertainty_history)[-10:]
            
            # Check if uncertainty estimates are consistent with market conditions
            market_uncertainty = self._calculate_market_uncertainty_factor(voting_data)
            
            # Calculate alignment between estimated and expected uncertainty
            avg_estimated_uncertainty = np.mean(recent_uncertainties)
            alignment = 1.0 - abs(avg_estimated_uncertainty - market_uncertainty)
            
            return float(max(0.0, min(1.0, alignment)))
            
        except Exception:
            return 0.5

    async def _evaluate_strategy_performance(self) -> Dict[str, float]:
        """Evaluate performance of different sampling strategies"""
        try:
            strategy_performance = {}
            
            for strategy_name in self.sampling_strategies:
                # Simplified performance evaluation based on current weights
                current_weight = self.sampling_strategies[strategy_name]['weight']
                is_active = self.sampling_strategies[strategy_name]['active']
                
                # Performance score based on weight and activity
                performance = current_weight if is_active else 0.0
                
                # Add effectiveness bonus/penalty based on recent results
                if len(self.effectiveness_history) >= 3:
                    recent_effectiveness = [e.get('overall_effectiveness', 0.5) for e in list(self.effectiveness_history)[-3:]]
                    avg_effectiveness = np.mean(recent_effectiveness)
                    performance *= avg_effectiveness
                
                strategy_performance[strategy_name] = performance
            
            return strategy_performance
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "strategy_performance_evaluation")
            return {}

    async def _update_sampling_strategy_weights(self, effectiveness_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Update sampling strategy weights based on effectiveness analysis"""
        try:
            strategy_updates = {
                'weight_changes': {},
                'activation_changes': {},
                'overall_improvement': 0.0
            }
            
            overall_effectiveness = effectiveness_analysis.get('overall_effectiveness', 0.5)
            strategy_performance = effectiveness_analysis.get('strategy_performance', {})
            
            # Update strategy weights based on performance
            total_weight = 0.0
            for strategy_name in self.sampling_strategies:
                current_weight = self.sampling_strategies[strategy_name]['weight']
                performance = strategy_performance.get(strategy_name, 0.5)
                
                # Adaptive weight adjustment
                if performance > 0.7:
                    new_weight = min(0.5, current_weight * 1.1)  # Increase high performers
                elif performance < 0.3:
                    new_weight = max(0.05, current_weight * 0.9)  # Decrease poor performers
                else:
                    new_weight = current_weight  # Maintain moderate performers
                
                weight_change = new_weight - current_weight
                if abs(weight_change) > 0.01:
                    strategy_updates['weight_changes'][strategy_name] = {
                        'old_weight': current_weight,
                        'new_weight': new_weight,
                        'change': weight_change,
                        'performance': performance
                    }
                
                self.sampling_strategies[strategy_name]['weight'] = new_weight
                total_weight += new_weight
            
            # Normalize weights
            if total_weight > 0:
                for strategy_name in self.sampling_strategies:
                    self.sampling_strategies[strategy_name]['weight'] /= total_weight
            
            return strategy_updates
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "strategy_weight_update")
            return {'weight_changes': {}, 'activation_changes': {}, 'overall_improvement': 0.0}

    async def _calculate_comprehensive_quality_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for sampling"""
        try:
            quality_metrics = {
                'sample_diversity': self.quality_metrics.get('sample_diversity', 0.0),
                'coverage_efficiency': self.quality_metrics.get('coverage_efficiency', 0.0),
                'uncertainty_accuracy': self.quality_metrics.get('uncertainty_accuracy', 0.0),
                'adaptation_success_rate': self.quality_metrics.get('adaptation_success_rate', 0.0),
                'exploration_completeness': self.quality_metrics.get('exploration_completeness', 0.0),
                'overall_quality_score': 0.0
            }
            
            # Update diversity score
            if len(self.sampling_history) > 0:
                recent_sample_data = list(self.sampling_history)[-1]
                samples = recent_sample_data.get('samples', [])
                if samples:
                    diversity = self._calculate_sample_diversity(np.array(samples))
                    quality_metrics['sample_diversity'] = diversity
            
            # Update coverage efficiency
            if len(self.effectiveness_history) > 0:
                recent_effectiveness = list(self.effectiveness_history)[-1]
                exploration_efficiency = recent_effectiveness.get('exploration_efficiency', 0.0)
                quality_metrics['coverage_efficiency'] = exploration_efficiency
            
            # Update uncertainty accuracy
            quality_metrics['uncertainty_accuracy'] = self.sampling_stats.get('avg_uncertainty', 0.5)
            
            # Calculate adaptation success rate
            if self.sampling_stats.get('sigma_adaptations', 0) > 0:
                # Simplified success rate based on effectiveness improvement
                quality_metrics['adaptation_success_rate'] = min(1.0, 
                    self.sampling_stats.get('effective_samples', 0) / 
                    max(self.sampling_stats.get('sigma_adaptations', 1), 1)
                )
            
            # Calculate exploration completeness
            quality_metrics['exploration_completeness'] = min(1.0, 
                len(self.sampling_history) / 50.0  # Completeness based on history length
            )
            
            # Calculate overall quality score
            weights = [0.25, 0.20, 0.20, 0.20, 0.15]
            values = [
                quality_metrics['sample_diversity'],
                quality_metrics['coverage_efficiency'],
                quality_metrics['uncertainty_accuracy'],
                quality_metrics['adaptation_success_rate'],
                quality_metrics['exploration_completeness']
            ]
            
            quality_metrics['overall_quality_score'] = float(np.average(values, weights=weights))
            
            # Update internal quality metrics
            self.quality_metrics.update(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "quality_metrics_calculation")
            return {'overall_quality_score': 0.5, 'calculation_error': str(error_context)}

    def _calculate_sample_diversity(self, samples: np.ndarray) -> float:
        """Calculate diversity score for a set of samples"""
        try:
            if len(samples) < 2:
                return 0.0
            
            # Calculate average pairwise distance
            total_distance = 0.0
            pair_count = 0
            
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    distance = np.linalg.norm(samples[i] - samples[j])
                    total_distance += distance
                    pair_count += 1
            
            if pair_count == 0:
                return 0.0
            
            avg_distance = total_distance / pair_count
            
            # Normalize by expected distance for this dimensionality
            expected_distance = np.sqrt(self.dim) * self.current_sigma
            normalized_diversity = avg_distance / max(expected_distance, 1e-8)
            
            return float(min(1.0, normalized_diversity))
            
        except Exception:
            return 0.0

    async def _generate_intelligent_sampling_recommendations(self, effectiveness_analysis: Dict[str, Any], 
                                                           quality_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent sampling recommendations"""
        try:
            recommendations = []
            
            # Effectiveness-based recommendations
            overall_effectiveness = effectiveness_analysis.get('overall_effectiveness', 0.5)
            if overall_effectiveness < 0.3:
                recommendations.append("Low sampling effectiveness detected - consider increasing exploration sigma")
            elif overall_effectiveness > 0.8:
                recommendations.append("High sampling effectiveness - current parameters are optimal")
            
            # Diversity-based recommendations
            diversity_analysis = effectiveness_analysis.get('diversity_analysis', {})
            diversity_score = diversity_analysis.get('diversity_score', 0.5)
            if diversity_score < self.sampling_intelligence['diversity_target'] * 0.7:
                recommendations.append("Insufficient sample diversity - increase sigma or enable more exploration strategies")
            
            # Convergence-based recommendations
            convergence_analysis = effectiveness_analysis.get('convergence_analysis', {})
            convergence_pattern = convergence_analysis.get('convergence_pattern', 'unknown')
            if convergence_pattern == 'diverging':
                recommendations.append("Sampling is diverging - reduce sigma or improve convergence criteria")
            elif convergence_pattern == 'oscillating':
                recommendations.append("Oscillating convergence detected - stabilize sampling parameters")
            
            # Quality-based recommendations
            overall_quality = quality_analysis.get('overall_quality_score', 0.5)
            if overall_quality < 0.4:
                recommendations.append("Low sampling quality - review strategy weights and parameters")
            
            # Strategy-specific recommendations
            strategy_performance = effectiveness_analysis.get('strategy_performance', {})
            for strategy, performance in strategy_performance.items():
                if performance < 0.3:
                    recommendations.append(f"Consider disabling or adjusting {strategy} strategy due to poor performance")
            
            # Adaptation recommendations
            if self.sampling_stats.get('sigma_adaptations', 0) > 20:
                recommendations.append("High adaptation frequency - consider more stable market assessment")
            elif self.sampling_stats.get('sigma_adaptations', 0) == 0:
                recommendations.append("No sigma adaptations - enable adaptive sampling for better performance")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Sampling system operating optimally - continue current approach")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendation_generation")
            return [f"Recommendation generation failed: {error_context}"]

    async def _generate_comprehensive_sampling_thesis(self, effectiveness_analysis: Dict[str, Any], 
                                                    quality_analysis: Dict[str, Any], 
                                                    strategy_updates: Dict[str, Any]) -> str:
        """Generate comprehensive sampling thesis"""
        try:
            # Core metrics
            overall_effectiveness = effectiveness_analysis.get('overall_effectiveness', 0.5)
            diversity_score = effectiveness_analysis.get('diversity_analysis', {}).get('diversity_score', 0.5)
            overall_quality = quality_analysis.get('overall_quality_score', 0.5)
            
            # Strategy updates
            weight_changes = strategy_updates.get('weight_changes', {})
            
            thesis_parts = []
            
            # Executive summary
            thesis_parts.append(
                f"SAMPLING ANALYSIS: Overall effectiveness {overall_effectiveness:.1%} with {diversity_score:.3f} diversity score"
            )
            
            # Quality assessment
            thesis_parts.append(
                f"QUALITY METRICS: Overall quality {overall_quality:.1%} across {len(self.sampling_methods)} sampling methods"
            )
            
            # Effectiveness breakdown
            if effectiveness_analysis.get('diversity_analysis'):
                diversity_trend = effectiveness_analysis['diversity_analysis'].get('diversity_trend', 'unknown')
                thesis_parts.append(f"DIVERSITY TREND: {diversity_trend} with target alignment assessment")
            
            # Adaptation status
            sigma_adaptations = self.sampling_stats.get('sigma_adaptations', 0)
            if sigma_adaptations > 0:
                thesis_parts.append(
                    f"ADAPTATION STATUS: {sigma_adaptations} sigma adaptations applied for market alignment"
                )
            
            # Strategy performance
            if weight_changes:
                strategy_count = len(weight_changes)
                thesis_parts.append(f"STRATEGY OPTIMIZATION: {strategy_count} strategies adjusted based on performance")
            
            # Current configuration
            thesis_parts.append(
                f"CURRENT CONFIG: {self.n_samples} samples per iteration with σ={self.current_sigma:.4f}"
            )
            
            # System status
            effective_samples = self.sampling_stats.get('effective_samples', 0)
            total_samples = self.sampling_stats.get('total_samples_created', 1)
            efficiency = effective_samples / max(total_samples, 1)
            thesis_parts.append(f"SYSTEM EFFICIENCY: {efficiency:.1%} effective sample ratio")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Sampling thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive sampling results"""
        try:
            # Core sampling results
            self.smart_bus.set('alternative_samples', results['alternative_samples'],
                             module='AlternativeRealitySampler', thesis=thesis)
            
            # Uncertainty metrics
            uncertainty_thesis = f"Sampling uncertainty: {results['sampling_uncertainty']:.3f} with {results['effective_samples']} effective samples"
            self.smart_bus.set('sampling_uncertainty', results['sampling_uncertainty'],
                             module='AlternativeRealitySampler', thesis=uncertainty_thesis)
            
            # Diversity score
            diversity_thesis = f"Sample diversity: {results['diversity_score']:.3f} across {self.dim} dimensions"
            self.smart_bus.set('diversity_score', results['diversity_score'],
                             module='AlternativeRealitySampler', thesis=diversity_thesis)
            
            # Comprehensive stats
            stats_thesis = f"Sampling statistics: {results['sampling_stats']['samples_generated']} total samples generated"
            self.smart_bus.set('sampling_stats', results['sampling_stats'],
                             module='AlternativeRealitySampler', thesis=stats_thesis)
            
            # Effective samples count
            effective_thesis = f"Effective samples: {results['effective_samples']} meaningful variations generated"
            self.smart_bus.set('effective_samples', results['effective_samples'],
                             module='AlternativeRealitySampler', thesis=effective_thesis)
            
            # Confidence bounds
            bounds_thesis = f"Confidence bounds calculated from {len(results['confidence_bounds'])} sample points"
            self.smart_bus.set('confidence_bounds', results['confidence_bounds'],
                             module='AlternativeRealitySampler', thesis=bounds_thesis)
            
            # Recommendations
            rec_thesis = f"Generated {len(results['sampling_recommendations'])} sampling recommendations"
            self.smart_bus.set('sampling_recommendations', results['sampling_recommendations'],
                             module='AlternativeRealitySampler', thesis=rec_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # CORE SAMPLING METHODS
    # ═══════════════════════════════════════════════════════════════════

    async def sample_comprehensive(self, weights: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate comprehensive alternative weight configurations with enhanced features
        
        Args:
            weights: Base weights to sample around
            context: Optional context for adaptive sampling
            
        Returns:
            Array of alternative weight samples with comprehensive analysis
        """
        try:
            # Validate and prepare input
            weights = np.asarray(weights, dtype=np.float32).flatten()
            if weights.size != self.dim:
                self.logger.warning(format_operator_message(
                    icon="[WARN]",
                    message="Weight dimension mismatch detected",
                    expected=self.dim,
                    received=weights.size,
                    action="Adjusting dimensions"
                ))
                weights = np.pad(weights, (0, max(0, self.dim - weights.size)))[:self.dim]
            
            # Store base weights for analysis
            self.last_weights = weights.copy()
            self.last_base_weights = weights.copy()
            
            # Adapt sampling parameters based on context
            sampling_sigma = await self._calculate_adaptive_sigma(context)
            
            # Generate samples using multiple strategies
            samples = await self._generate_multi_strategy_samples(weights, sampling_sigma, context)
            
            # Post-process and validate samples
            samples = await self._post_process_samples(samples, weights)
            
            # Store and analyze results
            await self._record_sampling_results(samples, weights, sampling_sigma, context)
            
            # Update statistics
            await self._update_sampling_statistics(samples, weights)
            
            return samples
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "comprehensive_sampling")
            self.logger.error(f"Comprehensive sampling failed: {error_context}")
            # Return safe fallback
            return np.tile(weights, (self.n_samples, 1))

    async def _calculate_adaptive_sigma(self, context: Optional[Dict[str, Any]]) -> float:
        """Calculate adaptive sigma based on context and current conditions"""
        try:
            base_sigma = self.current_sigma
            
            if context:
                # Context-based adjustments
                uncertainty = context.get('uncertainty', 0.0)
                if uncertainty > self.uncertainty_threshold:
                    base_sigma *= (1.0 + uncertainty * 0.5)  # Increase exploration under uncertainty
                
                # Market volatility adjustment
                volatility = context.get('volatility', 0.5)
                base_sigma *= (0.8 + volatility * 0.4)  # Scale with volatility
                
                # Agreement-based adjustment
                agreement = context.get('agreement_score', 0.5)
                base_sigma *= (1.5 - agreement)  # More exploration when disagreement is high
            
            # Ensure sigma stays within bounds
            return np.clip(base_sigma, 
                          self.sampling_intelligence['sigma_bounds'][0], 
                          self.sampling_intelligence['sigma_bounds'][1])
            
        except Exception:
            return self.current_sigma

    async def _generate_multi_strategy_samples(self, weights: np.ndarray, sigma: float, 
                                             context: Optional[Dict[str, Any]]) -> np.ndarray:
        """Generate samples using multiple sampling strategies"""
        try:
            all_samples = []
            
            # Determine number of samples per strategy
            active_strategies = [name for name, config in self.sampling_strategies.items() if config['active']]
            if not active_strategies:
                active_strategies = ['random_gaussian']  # Fallback
            
            samples_per_strategy = max(1, self.n_samples // len(active_strategies))
            remaining_samples = self.n_samples
            
            for i, strategy_name in enumerate(active_strategies):
                if remaining_samples <= 0:
                    break
                
                # Calculate samples for this strategy
                if i == len(active_strategies) - 1:  # Last strategy gets remaining samples
                    strategy_samples = remaining_samples
                else:
                    strategy_samples = min(samples_per_strategy, remaining_samples)
                
                # Generate samples using specific strategy
                strategy_samples_array = await self._generate_strategy_samples(
                    strategy_name, weights, sigma, strategy_samples, context
                )
                
                if strategy_samples_array is not None and len(strategy_samples_array) > 0:
                    all_samples.extend(strategy_samples_array)
                    remaining_samples -= len(strategy_samples_array)
            
            # Ensure we have the right number of samples
            if len(all_samples) < self.n_samples:
                # Fill remaining with random Gaussian
                needed = self.n_samples - len(all_samples)
                additional = await self._generate_random_gaussian_samples(weights, sigma, needed)
                all_samples.extend(additional)
            
            return np.array(all_samples[:self.n_samples])
        
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "multi_strategy_sampling")
            # Fallback to simple Gaussian sampling
            fallback_samples = await self._generate_random_gaussian_samples(weights, sigma, self.n_samples)
            return np.array(fallback_samples)

    async def _generate_strategy_samples(self, strategy_name: str, weights: np.ndarray, sigma: float, 
                                       n_samples: int, context: Optional[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate samples using a specific strategy"""
        try:
            if strategy_name == 'random_gaussian':
                return await self._generate_random_gaussian_samples(weights, sigma, n_samples)
            elif strategy_name == 'structured_perturbation':
                return await self._generate_structured_perturbation_samples(weights, sigma, n_samples)
            elif strategy_name == 'systematic_exploration':
                return await self._generate_systematic_exploration_samples(weights, sigma, n_samples)
            elif strategy_name == 'uncertainty_guided':
                return await self._generate_uncertainty_guided_samples(weights, sigma, n_samples, context)
            elif strategy_name == 'adaptive_momentum':
                return await self._generate_adaptive_momentum_samples(weights, sigma, n_samples)
            else:
                # Fallback to Gaussian
                return await self._generate_random_gaussian_samples(weights, sigma, n_samples)
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, f"strategy_{strategy_name}")
            return await self._generate_random_gaussian_samples(weights, sigma, n_samples)

    async def _generate_random_gaussian_samples(self, weights: np.ndarray, sigma: float, n_samples: int) -> List[np.ndarray]:
        """Generate random Gaussian samples around base weights"""
        try:
            noise = np.random.randn(n_samples, self.dim) * sigma
            samples = weights[None, :] + noise
            return [sample for sample in samples]
        except Exception:
            return [weights.copy() for _ in range(n_samples)]

    async def _generate_structured_perturbation_samples(self, weights: np.ndarray, sigma: float, 
                                                      n_samples: int) -> List[np.ndarray]:
        """Generate structured perturbation samples"""
        try:
            samples = []
            perturbation_strength = 2.0 * sigma
            
            for i in range(min(n_samples, self.dim)):
                perturbed = weights.copy()
                # Single-dimension perturbation
                perturbed[i % self.dim] += perturbation_strength * (2 * np.random.random() - 1)
                samples.append(perturbed)
            
            # Fill remaining with random perturbations
            while len(samples) < n_samples:
                perturbed = weights.copy()
                dim_idx = np.random.randint(0, self.dim)
                perturbed[dim_idx] += perturbation_strength * (2 * np.random.random() - 1)
                samples.append(perturbed)
            
            return samples
            
        except Exception:
            return await self._generate_random_gaussian_samples(weights, sigma, n_samples)

    async def _generate_systematic_exploration_samples(self, weights: np.ndarray, sigma: float, 
                                                     n_samples: int) -> List[np.ndarray]:
        """Generate systematic exploration samples"""
        try:
            samples = []
            grid_spacing = sigma * 1.5
            
            # Create systematic grid around base weights
            directions = np.eye(self.dim)  # Unit vectors for each dimension
            
            for i in range(min(n_samples, self.dim * 2)):
                direction = directions[i % self.dim]
                sign = 1 if i < self.dim else -1
                
                perturbed = weights + sign * grid_spacing * direction
                samples.append(perturbed)
            
            # Fill remaining with diagonal movements
            while len(samples) < n_samples:
                # Random diagonal direction
                direction = np.random.randn(self.dim)
                direction = direction / np.linalg.norm(direction)
                
                perturbed = weights + grid_spacing * direction
                samples.append(perturbed)
            
            return samples
            
        except Exception:
            return await self._generate_random_gaussian_samples(weights, sigma, n_samples)

    async def _generate_uncertainty_guided_samples(self, weights: np.ndarray, sigma: float, 
                                                 n_samples: int, context: Optional[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate uncertainty-guided samples"""
        try:
            samples = []
            uncertainty_scaling = 1.5
            
            # Use uncertainty information from context if available
            if context and 'uncertainty_regions' in context:
                uncertainty_regions = context['uncertainty_regions']
                for region in uncertainty_regions[:n_samples]:
                    direction = np.array(region.get('direction', np.random.randn(self.dim)))
                    direction = direction / max(np.linalg.norm(direction), 1e-8)
                    
                    magnitude = sigma * uncertainty_scaling * region.get('uncertainty', 1.0)
                    perturbed = weights + magnitude * direction
                    samples.append(perturbed)
            
            # Fill remaining with uncertainty-weighted random sampling
            while len(samples) < n_samples:
                # Higher noise in dimensions with higher uncertainty
                uncertainty_weights = context.get('dimension_uncertainties', np.ones(self.dim)) if context else np.ones(self.dim)
                noise = np.random.randn(self.dim) * sigma * uncertainty_scaling * uncertainty_weights
                perturbed = weights + noise
                samples.append(perturbed)
            
            return samples
            
        except Exception:
            return await self._generate_random_gaussian_samples(weights, sigma, n_samples)

    async def _generate_adaptive_momentum_samples(self, weights: np.ndarray, sigma: float, 
                                                n_samples: int) -> List[np.ndarray]:
        """Generate adaptive momentum samples"""
        try:
            samples = []
            momentum_factor = 0.8
            
            # Use historical sampling direction if available
            if len(self.sampling_history) >= 2:
                recent_samples = list(self.sampling_history)[-2:]
                
                # Calculate momentum direction
                prev_weights = np.array(recent_samples[-2].get('base_weights', weights))
                curr_weights = np.array(recent_samples[-1].get('base_weights', weights))
                momentum_direction = curr_weights - prev_weights
                
                # Generate momentum-based samples
                for i in range(n_samples // 2):
                    # Forward momentum
                    momentum_sample = weights + momentum_factor * momentum_direction + np.random.randn(self.dim) * sigma
                    samples.append(momentum_sample)
                    
                    # Reverse momentum (for exploration)
                    reverse_sample = weights - momentum_factor * momentum_direction + np.random.randn(self.dim) * sigma
                    samples.append(reverse_sample)
            
            # Fill remaining with regular Gaussian
            while len(samples) < n_samples:
                noise_sample = weights + np.random.randn(self.dim) * sigma
                samples.append(noise_sample)
            
            return samples[:n_samples]
            
        except Exception:
            return await self._generate_random_gaussian_samples(weights, sigma, n_samples)

    async def _post_process_samples(self, samples: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
        """Post-process samples to ensure validity"""
        try:
            # Ensure positive values
            samples = np.abs(samples)
            
            # Normalize to ensure weights sum to 1
            row_sums = samples.sum(axis=1, keepdims=True)
            samples = samples / (row_sums + 1e-12)
            
            # Store processed samples
            self.last_samples = samples.copy()
            
            return samples
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "sample_post_processing")
            self.logger.warning(f"Sample post-processing failed: {error_context}")
            return np.tile(base_weights, (self.n_samples, 1))

    async def _record_sampling_results(self, samples: np.ndarray, base_weights: np.ndarray, 
                                     sigma: float, context: Optional[Dict[str, Any]]):
        """Record comprehensive sampling results"""
        try:
            sample_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'base_weights': base_weights.tolist(),
                'samples': samples.tolist(),
                'sigma_used': sigma,
                'n_samples': len(samples),
                'context': context.copy() if context else {},
                'strategies_used': [name for name, config in self.sampling_strategies.items() if config['active']],
                'diversity_score': self._calculate_sample_diversity(samples),
                'effective_samples': self._count_effective_samples_comprehensive(samples, base_weights)
            }
            
            self.sampling_history.append(sample_data)
            
            # Calculate and record uncertainty
            uncertainty = await self._calculate_uncertainty_estimate_comprehensive(samples, base_weights)
            self.uncertainty_history.append(uncertainty)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "sampling_results_recording")

    def _count_effective_samples_comprehensive(self, samples: np.ndarray, base_weights: np.ndarray) -> int:
        """Count effective samples with comprehensive analysis"""
        try:
            threshold = self.current_sigma * 0.5
            effective = 0
            
            for sample in samples:
                distance = np.linalg.norm(sample - base_weights)
                if distance > threshold:
                    effective += 1
            
            return effective
            
        except Exception:
            return len(samples)

    async def _calculate_uncertainty_estimate_comprehensive(self, samples: np.ndarray, 
                                                          base_weights: np.ndarray) -> float:
        """Calculate comprehensive uncertainty estimate"""
        try:
            if len(samples) < 2:
                return 0.5
            
            # Calculate spread-based uncertainty
            sample_norms = [np.linalg.norm(s) for s in samples]
            spread_uncertainty = np.std(sample_norms) / (np.mean(sample_norms) + 1e-12)
            
            # Calculate distance-based uncertainty
            distances = [np.linalg.norm(s - base_weights) for s in samples]
            distance_uncertainty = np.mean(distances) / (self.current_sigma * np.sqrt(self.dim))
            
            # Combine uncertainties
            combined_uncertainty = (0.6 * spread_uncertainty + 0.4 * distance_uncertainty)
            
            return float(np.clip(combined_uncertainty, 0.0, 1.0))
            
        except Exception:
            return 0.5

    async def _update_sampling_statistics(self, samples: np.ndarray, base_weights: np.ndarray):
        """Update comprehensive sampling statistics"""
        try:
            # Update basic counts
            self.sampling_stats['samples_generated'] += 1
            self.sampling_stats['total_samples_created'] += len(samples)
            
            # Update effective samples
            effective = self._count_effective_samples_comprehensive(samples, base_weights)
            self.sampling_stats['effective_samples'] += effective
            
            # Update diversity score
            diversity = self._calculate_sample_diversity(samples)
            self.sampling_stats['diversity_score'] = diversity
            
            # Update uncertainty average
            if self.uncertainty_history:
                self.sampling_stats['avg_uncertainty'] = np.mean(list(self.uncertainty_history)[-10:])
            
            # Update exploration efficiency
            total_samples = self.sampling_stats['total_samples_created']
            total_effective = self.sampling_stats['effective_samples']
            self.sampling_stats['exploration_efficiency'] = total_effective / max(total_samples, 1)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "sampling_statistics_update")

    def _calculate_confidence_bounds(self) -> Dict[str, Any]:
        """Calculate confidence bounds from recent sampling"""
        try:
            if self.last_samples is None or len(self.last_samples) < 3:
                return {'lower_bound': [], 'upper_bound': [], 'confidence_level': 0.0}
            
            # Calculate percentile-based confidence bounds
            lower_percentile = 5  # 5th percentile
            upper_percentile = 95  # 95th percentile
            
            lower_bounds = np.percentile(self.last_samples, lower_percentile, axis=0)
            upper_bounds = np.percentile(self.last_samples, upper_percentile, axis=0)
            
            # Calculate average confidence level based on bound width
            bound_widths = upper_bounds - lower_bounds
            avg_width = np.mean(bound_widths)
            confidence_level = max(0.0, min(1.0, 1.0 - avg_width * 2))  # Inverse relationship
            
            return {
                'lower_bound': lower_bounds.tolist(),
                'upper_bound': upper_bounds.tolist(),
                'confidence_level': confidence_level,
                'bound_width': avg_width,
                'percentiles': {'lower': lower_percentile, 'upper': upper_percentile}
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "confidence_bounds_calculation")
            return {'lower_bound': [], 'upper_bound': [], 'confidence_level': 0.0}

    def _get_comprehensive_sampling_stats(self) -> Dict[str, Any]:
        """Get comprehensive sampling statistics"""
        return {
            **self.sampling_stats,
            'current_sigma': self.current_sigma,
            'base_sigma': self.base_sigma,
            'adaptive_enabled': self.adaptive_sigma,
            'dimensions': self.dim,
            'samples_per_iteration': self.n_samples,
            'strategy_weights': {name: config['weight'] for name, config in self.sampling_strategies.items()},
            'quality_metrics': self.quality_metrics.copy(),
            'recent_uncertainty_trend': self._calculate_uncertainty_trend(list(self.uncertainty_history)[-10:]) if len(self.uncertainty_history) >= 3 else 'insufficient_data'
        }

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'AlternativeRealitySampler',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'samples_generated': self.sampling_stats.get('samples_generated', 0),
            'effective_sample_ratio': self.sampling_stats.get('exploration_efficiency', 0.0),
            'diversity_score': self.sampling_stats.get('diversity_score', 0.0),
            'uncertainty_level': self.sampling_stats.get('avg_uncertainty', 0.5),
            'sigma_adaptations': self.sampling_stats.get('sigma_adaptations', 0),
            'strategy_count': len([s for s in self.sampling_strategies.values() if s['active']]),
            'session_duration': (datetime.datetime.now() - datetime.datetime.fromisoformat(self.sampling_stats['session_start'])).total_seconds() / 3600
        }

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "AlternativeRealitySampler")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="[ALERT]",
                message="Alternative Reality Sampler disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('AlternativeRealitySampler', 'process_time', processing_time, False)
        
        return {
            'alternative_samples': [],
            'sampling_uncertainty': 0.5,
            'diversity_score': 0.0,
            'sampling_stats': {'error': str(error_context)},
            'effective_samples': 0,
            'confidence_bounds': {'error': str(error_context)},
            'sampling_recommendations': ["Investigate alternative reality sampler errors"],
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_voting_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when voting data retrieval fails"""
        return {
            'votes': [], 'voting_summary': {}, 'strategy_arbiter_weights': [],
            'market_context': {}, 'volatility_data': {}, 'consensus_direction': 'neutral',
            'agreement_score': 0.5, 'market_regime': 'unknown', 'recent_trades': [],
            'session_metrics': {}
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'alternative_samples': [],
            'sampling_uncertainty': 0.5,
            'diversity_score': 0.0,
            'sampling_stats': {'status': 'disabled'},
            'effective_samples': 0,
            'confidence_bounds': {'status': 'disabled'},
            'sampling_recommendations': ["Restart alternative reality sampler system"],
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT AND LEGACY COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'AlternativeRealitySampler',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'dim': self.dim,
                'n_samples': self.n_samples,
                'base_sigma': self.base_sigma,
                'adaptive_sigma': self.adaptive_sigma,
                'uncertainty_threshold': self.uncertainty_threshold,
                'debug': self.debug
            },
            'sampling_state': {
                'current_sigma': self.current_sigma,
                'last_weights': self.last_weights.tolist() if self.last_weights is not None else None,
                'last_samples': self.last_samples.tolist() if self.last_samples is not None else None,
                'last_base_weights': self.last_base_weights.tolist() if self.last_base_weights is not None else None,
                'sampling_stats': self.sampling_stats.copy(),
                'quality_metrics': self.quality_metrics.copy()
            },
            'intelligence_state': {
                'sampling_intelligence': self.sampling_intelligence.copy(),
                'sampling_strategies': {k: v.copy() for k, v in self.sampling_strategies.items()},
                'market_adaptation': self.market_adaptation.copy()
            },
            'history_state': {
                'sampling_history': list(self.sampling_history)[-20:],
                'uncertainty_history': list(self.uncertainty_history)[-50:],
                'effectiveness_history': list(self.effectiveness_history)[-30:]
            },
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'performance_metrics': self._get_health_metrics()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.dim = int(config.get("dim", self.dim))
            self.n_samples = int(config.get("n_samples", self.n_samples))
            self.base_sigma = float(config.get("base_sigma", self.base_sigma))
            self.adaptive_sigma = bool(config.get("adaptive_sigma", self.adaptive_sigma))
            self.uncertainty_threshold = float(config.get("uncertainty_threshold", self.uncertainty_threshold))
            self.debug = bool(config.get("debug", self.debug))
            
            # Load sampling state
            sampling_state = state.get("sampling_state", {})
            self.current_sigma = float(sampling_state.get("current_sigma", self.base_sigma))
            
            if sampling_state.get("last_weights"):
                self.last_weights = np.array(sampling_state["last_weights"], dtype=np.float32)
            if sampling_state.get("last_samples"):
                self.last_samples = np.array(sampling_state["last_samples"], dtype=np.float32)
            if sampling_state.get("last_base_weights"):
                self.last_base_weights = np.array(sampling_state["last_base_weights"], dtype=np.float32)
            
            self.sampling_stats.update(sampling_state.get("sampling_stats", {}))
            self.quality_metrics.update(sampling_state.get("quality_metrics", {}))
            
            # Load intelligence state
            intelligence_state = state.get("intelligence_state", {})
            self.sampling_intelligence.update(intelligence_state.get("sampling_intelligence", {}))
            
            strategies_data = intelligence_state.get("sampling_strategies", {})
            for name, strategy_data in strategies_data.items():
                if name in self.sampling_strategies:
                    self.sampling_strategies[name].update(strategy_data)
            
            self.market_adaptation.update(intelligence_state.get("market_adaptation", {}))
            
            # Load history state
            history_state = state.get("history_state", {})
            
            self.sampling_history.clear()
            for entry in history_state.get("sampling_history", []):
                self.sampling_history.append(entry)
            
            self.uncertainty_history.clear()
            for entry in history_state.get("uncertainty_history", []):
                self.uncertainty_history.append(entry)
            
            self.effectiveness_history.clear()
            for entry in history_state.get("effectiveness_history", []):
                self.effectiveness_history.append(entry)
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            self.logger.info(format_operator_message(
                icon="[RELOAD]",
                message="Alternative Reality Sampler state restored",
                dimensions=self.dim,
                samples=self.n_samples,
                current_sigma=f"{self.current_sigma:.4f}",
                total_samples=self.sampling_stats.get('samples_generated', 0)
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    # Legacy compatibility methods
    def sample(self, weights: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Legacy sampling interface for backward compatibility"""
        import asyncio
        try:
            # Run async method synchronously
            if asyncio.get_event_loop().is_running():
                # If already in async context, create a coroutine and run it
                coro = self.sample_comprehensive(weights, context)
                return asyncio.run_coroutine_threadsafe(coro, asyncio.get_event_loop()).result()
            else:
                # Run directly
                return asyncio.run(self.sample_comprehensive(weights, context))
        except Exception:
            # Fallback to simple implementation
            return self._simple_sample_fallback(weights, context)

    def _simple_sample_fallback(self, weights: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Simple fallback sampling method"""
        try:
            weights = np.asarray(weights, dtype=np.float32).flatten()
            if weights.size != self.dim:
                weights = np.pad(weights, (0, max(0, self.dim - weights.size)))[:self.dim]
            
            # Simple Gaussian noise
            noise = np.random.randn(self.n_samples, self.dim) * self.current_sigma
            samples = weights[None, :] + noise
            
            # Ensure positive and normalized
            samples = np.abs(samples)
            row_sums = samples.sum(axis=1, keepdims=True)
            samples = samples / (row_sums + 1e-12)
            
            self.last_samples = samples
            return samples
            
        except Exception:
            return np.tile(weights, (self.n_samples, 1))

    def get_uncertainty_estimate(self, weights: np.ndarray) -> float:
        """Legacy uncertainty estimation interface"""
        try:
            if self.last_samples is None:
                samples = self.sample(weights)
            else:
                samples = self.last_samples
            
            # Calculate spread of sample outcomes
            sample_norms = [np.linalg.norm(s) for s in samples]
            uncertainty = np.std(sample_norms) / (np.mean(sample_norms) + 1e-12)
            
            return float(np.clip(uncertainty, 0.0, 1.0))
            
        except Exception:
            return 0.5

    def get_observation_components(self) -> np.ndarray:
        """Return sampler features for RL observation"""
        try:
            features = [
                float(self.current_sigma / self.base_sigma),  # Sigma ratio
                float(self.sampling_stats.get('avg_uncertainty', 0.5)),  # Average uncertainty
                float(len(self.sampling_history) / 100),  # History fullness
                float(self.sampling_stats.get('exploration_efficiency', 0.5)),  # Exploration efficiency
                float(self.adaptive_sigma),  # Adaptive mode enabled
                float(self.sampling_stats.get('diversity_score', 0.0)),  # Diversity score
                float(len([s for s in self.sampling_strategies.values() if s['active']]) / len(self.sampling_strategies)),  # Active strategies ratio
                float(self.quality_metrics.get('overall_quality_score', 0.5))  # Overall quality
            ]
            
            observation = np.array(features, dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid sampler observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Sampler observation generation failed: {error_context}")
            return np.array([1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5], dtype=np.float32)

    def get_sampling_report(self) -> str:
        """Generate comprehensive sampling report"""
        # Recent sampling effectiveness
        if len(self.uncertainty_history) > 0:
            recent_uncertainty = np.mean(list(self.uncertainty_history)[-10:])
            uncertainty_trend = self._calculate_uncertainty_trend(list(self.uncertainty_history)[-10:])
        else:
            recent_uncertainty = 0.5
            uncertainty_trend = "📭 No data"
        
        # Sigma adaptation status
        sigma_change = (self.current_sigma - self.base_sigma) / self.base_sigma * 100
        if abs(sigma_change) < 5:
            sigma_status = "[BALANCE] Stable"
        elif sigma_change > 0:
            sigma_status = f"[CHART] Increased ({sigma_change:+.1f}%)"
        else:
            sigma_status = f"📉 Decreased ({sigma_change:+.1f}%)"
        
        # Strategy status
        active_strategies = [name for name, config in self.sampling_strategies.items() if config['active']]
        strategy_status = f"{len(active_strategies)}/{len(self.sampling_strategies)} active"
        
        # Quality assessment
        overall_quality = self.quality_metrics.get('overall_quality_score', 0.5)
        if overall_quality > 0.8:
            quality_status = "[OK] Excellent"
        elif overall_quality > 0.6:
            quality_status = "[FAST] Good"
        elif overall_quality > 0.4:
            quality_status = "[WARN] Fair"
        else:
            quality_status = "[ALERT] Poor"
        
        return f"""
🎲 ALTERNATIVE REALITY SAMPLER v3.0
═══════════════════════════════════════════════════════════════
[STATS] Current Configuration:
• Dimensions: {self.dim}
• Samples per Iteration: {self.n_samples}
• Base Sigma: {self.base_sigma:.4f}
• Current Sigma: {self.current_sigma:.4f}
• Adaptive Sampling: {'[OK] Enabled' if self.adaptive_sigma else '[FAIL] Disabled'}
• Uncertainty Threshold: {self.uncertainty_threshold:.1%}

[TARGET] Sampling Performance:
• Total Samples Generated: {self.sampling_stats.get('samples_generated', 0)}
• Total Sample Points: {self.sampling_stats.get('total_samples_created', 0)}
• Effective Samples: {self.sampling_stats.get('effective_samples', 0)}
• Exploration Efficiency: {self.sampling_stats.get('exploration_efficiency', 0.0):.1%}
• Diversity Score: {self.sampling_stats.get('diversity_score', 0.0):.3f}

[CHART] Current Status:
• Sigma Status: {sigma_status}
• Recent Uncertainty: {recent_uncertainty:.3f}
• Uncertainty Trend: {uncertainty_trend.title()}
• Quality Assessment: {quality_status} ({overall_quality:.1%})
• Strategy Status: {strategy_status}

🧠 Intelligence Parameters:
• Adaptation Rate: {self.sampling_intelligence.get('adaptation_rate', 0.12):.1%}
• Diversity Target: {self.sampling_intelligence.get('diversity_target', 0.15):.3f}
• Uncertainty Sensitivity: {self.sampling_intelligence.get('uncertainty_sensitivity', 0.7):.1%}
• Exploration Momentum: {self.sampling_intelligence.get('exploration_momentum', 0.85):.1%}

[STATS] Sampling Strategies:
{chr(10).join([f'  • {name.replace("_", " ").title()}: {"[OK]" if config["active"] else "[FAIL]"} (Weight: {config["weight"]:.1%})' for name, config in self.sampling_strategies.items()])}

[TARGET] Quality Metrics:
• Sample Diversity: {self.quality_metrics.get('sample_diversity', 0.0):.3f}
• Coverage Efficiency: {self.quality_metrics.get('coverage_efficiency', 0.0):.1%}
• Uncertainty Accuracy: {self.quality_metrics.get('uncertainty_accuracy', 0.0):.1%}
• Adaptation Success Rate: {self.quality_metrics.get('adaptation_success_rate', 0.0):.1%}
• Exploration Completeness: {self.quality_metrics.get('exploration_completeness', 0.0):.1%}

[STATS] Recent Activity:
• Sampling History: {len(self.sampling_history)} entries
• Uncertainty History: {len(self.uncertainty_history)} entries
• Effectiveness History: {len(self.effectiveness_history)} entries
• Sigma Adaptations: {self.sampling_stats.get('sigma_adaptations', 0)}

[TOOL] System Health:
• Error Count: {self.error_count}/{self.circuit_breaker_threshold}
• Status: {'[ALERT] DISABLED' if self.is_disabled else '[OK] OPERATIONAL'}
• Session Duration: {(datetime.datetime.now() - datetime.datetime.fromisoformat(self.sampling_stats['session_start'])).total_seconds() / 3600:.1f} hours
        """

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'AlternativeRealitySampler',
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
                'message': 'AlternativeRealitySampler disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring issues'
            })
        
        exploration_efficiency = self.sampling_stats.get('exploration_efficiency', 0.0)
        if exploration_efficiency < 0.3:
            alerts.append({
                'severity': 'warning',
                'message': f'Low exploration efficiency: {exploration_efficiency:.1%}',
                'action': 'Consider adjusting sigma or sampling strategies'
            })
        
        if len(self.sampling_history) < 5:
            alerts.append({
                'severity': 'info',
                'message': 'Insufficient sampling history',
                'action': 'Continue operations to build sampling baseline'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart AlternativeRealitySampler module after investigating errors")
        
        if len(self.sampling_history) < 10:
            recommendations.append("Insufficient sampling history - continue operations to establish patterns")
        
        diversity_score = self.sampling_stats.get('diversity_score', 0.0)
        if diversity_score < self.sampling_intelligence['diversity_target'] * 0.7:
            recommendations.append("Low sampling diversity - consider increasing sigma or enabling more strategies")
        
        if self.sampling_stats.get('sigma_adaptations', 0) > 50:
            recommendations.append("High adaptation frequency - review market assessment sensitivity")
        
        if not recommendations:
            recommendations.append("AlternativeRealitySampler operating within normal parameters")
        
        return recommendations

    # ═══════════════════════════════════════════════════════════════════
    # BASEMODULE ABSTRACT METHOD IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in sampling-based action recommendations"""
        try:
            # For sampling modules, confidence is based on sampling quality and uncertainty
            diversity_score = self.sampling_stats.get('diversity_score', 0.5)
            uncertainty = self.sampling_stats.get('current_uncertainty', 0.5)
            exploration_efficiency = self.sampling_stats.get('exploration_efficiency', 0.5)
            
            # Base confidence from sampling quality
            base_confidence = (diversity_score + exploration_efficiency) / 2.0
            
            # Adjust for uncertainty (lower uncertainty = higher confidence)
            uncertainty_adjustment = 1.0 - min(uncertainty, 0.8)
            
            # Final confidence calculation
            confidence = base_confidence * uncertainty_adjustment
            
            # Ensure valid range
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.4  # Conservative default

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose sampling-based action for robustness testing"""
        try:
            # Get current market data
            market_data = await self._get_comprehensive_voting_data()
            
            # Extract current weights if available
            weights = market_data.get('strategy_arbiter_weights')
            if weights is None or len(weights) == 0:
                weights = np.ones(4) / 4  # Default equal weights
            
            # Generate alternative samples
            samples = await self.sample_comprehensive(weights, market_data)
            
            # Calculate sampling statistics
            uncertainty = np.std(samples, axis=0).mean() if len(samples) > 1 else 0.3
            diversity = self._calculate_sample_diversity(samples) if len(samples) > 1 else 0.5
            
            # Propose action based on sampling results
            if uncertainty > self.uncertainty_threshold:
                action_type = 'conservative'
                signal_strength = 0.3
                reasoning = f"High uncertainty ({uncertainty:.3f}) detected in sampling"
            elif diversity < 0.3:
                action_type = 'diversify'
                signal_strength = 0.6
                reasoning = f"Low diversity ({diversity:.3f}) suggests need for exploration"
            else:
                action_type = 'normal'
                signal_strength = 0.7
                reasoning = f"Balanced sampling metrics (uncertainty: {uncertainty:.3f}, diversity: {diversity:.3f})"
            
            return {
                'action': action_type,
                'signal_strength': signal_strength,
                'reasoning': reasoning,
                'sampling_metrics': {
                    'uncertainty': uncertainty,
                    'diversity': diversity,
                    'n_samples': len(samples),
                    'effective_samples': self.sampling_stats.get('effective_samples', len(samples))
                },
                'confidence': await self.calculate_confidence({}, **inputs)
            }
            
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                'action': 'abstain',
                'signal_strength': 0.0,
                'reasoning': f'Sampling error: {str(e)}',
                'confidence': 0.1
            }