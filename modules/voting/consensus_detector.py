"""
🤝 Enhanced Consensus Detector with SmartInfoBus Integration v3.0
Advanced consensus analysis and agreement measurement for voting committees
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
    name="ConsensusDetector",
    version="3.0.0",
    category="voting",
    provides=[
        "consensus_score", "consensus_quality", "consensus_components", "directional_consensus",
        "magnitude_consensus", "confidence_consensus", "member_contributions", "consensus_trends",
        "quality_metrics", "consensus_recommendations"
    ],
    requires=[
        "votes", "raw_proposals", "member_confidences", "voting_summary", "alpha_weights",
        "blended_action", "market_context", "agreement_score", "consensus_direction"
    ],
    description="Advanced consensus analysis and agreement measurement for voting committees",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=80,
    priority=3,
    explainable=True,
    hot_reload=True
)
class ConsensusDetector(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🤝 PRODUCTION-GRADE Consensus Detector v3.0
    
    Advanced consensus detection system with:
    - Multi-dimensional agreement analysis across directional, magnitude, and confidence metrics
    - Adaptive quality weighting and temporal smoothing for robust consensus measurement
    - Comprehensive member contribution tracking and influence analysis
    - SmartInfoBus zero-wiring architecture
    - Real-time consensus trend analysis and quality assessment
    """

    def _initialize(self):
        """Initialize advanced consensus detection systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced consensus configuration
        self.n_members = self.config.get('n_members', 5)
        self.threshold = self.config.get('threshold', 0.6)
        self.quality_weighting = self.config.get('quality_weighting', True)
        self.temporal_smoothing = self.config.get('temporal_smoothing', True)
        self.consensus_methods = self.config.get('consensus_methods', ['cosine_agreement', 'direction_alignment', 'confidence_weighted'])
        self.debug = self.config.get('debug', False)
        
        # Initialize comprehensive consensus methods
        self.consensus_algorithms = self._initialize_consensus_algorithms()
        
        # Core consensus state with enhanced tracking
        self.last_consensus = 0.0
        self.consensus_quality = 0.0
        self.consensus_components = {}
        self.consensus_history = deque(maxlen=100)
        self.consensus_trends = deque(maxlen=50)
        
        # Advanced consensus dimensions
        self.directional_consensus = 0.0
        self.magnitude_consensus = 0.0
        self.confidence_consensus = 0.0
        self.temporal_stability = 0.0
        self.network_consensus = 0.0
        
        # Member contribution and influence tracking
        self.member_contributions = defaultdict(lambda: {
            'avg_alignment': 0.5,
            'consistency': 0.5,
            'influence_weight': 1.0 / max(self.n_members, 1),
            'consensus_contribution': 0.5,
            'reliability_score': 0.5,
            'coordination_factor': 0.0
        })
        
        # Advanced quality and performance metrics
        self.quality_metrics = {
            'coherence': 0.5,
            'stability': 0.5,
            'diversity': 0.5,
            'reliability': 0.5,
            'predictive_accuracy': 0.5,
            'temporal_consistency': 0.5
        }
        
        # Consensus intelligence parameters
        self.consensus_intelligence = {
            'smoothing_alpha': 0.3,
            'stability_window': 10,
            'quality_threshold': 0.7,
            'trend_sensitivity': 0.15,
            'adaptation_rate': 0.12,
            'confidence_weighting': 0.8,
            'temporal_memory': 0.85
        }
        
        # Market condition adaptation
        self.market_adaptation = {
            'regime_adjustments': {
                'trending': {'weight_multiplier': 1.1, 'stability_factor': 1.2},
                'ranging': {'weight_multiplier': 0.95, 'stability_factor': 0.9},
                'volatile': {'weight_multiplier': 0.85, 'stability_factor': 0.7},
                'breakout': {'weight_multiplier': 1.2, 'stability_factor': 1.3},
                'reversal': {'weight_multiplier': 1.05, 'stability_factor': 1.1},
                'unknown': {'weight_multiplier': 1.0, 'stability_factor': 1.0}
            },
            'volatility_adjustments': {
                'very_low': 0.9,
                'low': 0.95,
                'medium': 1.0,
                'high': 1.1,
                'extreme': 1.2
            }
        }
        
        # Enhanced statistics and analytics
        self.consensus_stats = {
            'total_computations': 0,
            'high_consensus_count': 0,
            'low_consensus_count': 0,
            'avg_consensus': 0.5,
            'consensus_volatility': 0.0,
            'quality_score': 0.5,
            'trend_accuracy': 0.0,
            'prediction_accuracy': 0.0,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Temporal and regime analysis
        self.regime_consensus_history = defaultdict(lambda: deque(maxlen=30))
        self.consensus_patterns = defaultdict(list)
        self.prediction_history = deque(maxlen=20)
        
        # Advanced analytics and insights
        self.consensus_insights = {
            'dominant_patterns': [],
            'member_dynamics': {},
            'consensus_predictors': {},
            'quality_drivers': {}
        }
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🤝",
            message=f"Consensus Detector v{version} initialized",
            members=self.n_members,
            threshold=f"{self.threshold:.3f}",
            methods=len(self.consensus_methods),
            quality_weighting=self.quality_weighting,
            temporal_smoothing=self.temporal_smoothing
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="ConsensusDetector",
            log_path="logs/voting/consensus_detector.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("ConsensusDetector", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _initialize_consensus_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive consensus algorithm definitions"""
        return {
            'cosine_agreement': {
                'description': 'Cosine similarity analysis for directional agreement measurement',
                'parameters': {'normalization': True, 'weight_threshold': 0.1},
                'use_cases': ['directional_consensus', 'vector_alignment'],
                'effectiveness_threshold': 0.7,
                'computational_cost': 'low'
            },
            'direction_alignment': {
                'description': 'Binary direction alignment analysis for voting coherence',
                'parameters': {'confidence_weighting': True, 'threshold_adaptive': True},
                'use_cases': ['binary_voting', 'directional_coherence'],
                'effectiveness_threshold': 0.8,
                'computational_cost': 'low'
            },
            'confidence_weighted': {
                'description': 'Confidence-weighted consensus with reliability factors',
                'parameters': {'variance_penalty': 0.3, 'reliability_boost': 1.2},
                'use_cases': ['quality_consensus', 'reliability_analysis'],
                'effectiveness_threshold': 0.75,
                'computational_cost': 'medium'
            },
            'magnitude_consensus': {
                'description': 'Action magnitude similarity for strength agreement',
                'parameters': {'cv_normalization': True, 'outlier_handling': True},
                'use_cases': ['strength_consensus', 'magnitude_alignment'],
                'effectiveness_threshold': 0.6,
                'computational_cost': 'low'
            },
            'network_consensus': {
                'description': 'Network-based consensus considering member interconnections',
                'parameters': {'network_threshold': 0.5, 'influence_weighting': True},
                'use_cases': ['network_analysis', 'influence_consensus'],
                'effectiveness_threshold': 0.7,
                'computational_cost': 'high'
            },
            'temporal_consensus': {
                'description': 'Time-series consensus analysis with trend consideration',
                'parameters': {'window_size': 10, 'trend_weighting': 0.3},
                'use_cases': ['temporal_analysis', 'trend_consensus'],
                'effectiveness_threshold': 0.65,
                'computational_cost': 'medium'
            }
        }

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Consensus Detector v3.0 Initialization Complete:
        
        Advanced Consensus Framework:
        - Multi-member committee analysis: {self.n_members} members with {self.threshold:.1%} consensus threshold
        - Comprehensive agreement algorithms with quality weighting and temporal smoothing
        - Advanced member contribution tracking and influence analysis capabilities
        - Market-aware consensus adaptation based on regime and volatility conditions
        
        Current Configuration:
        - Consensus methods: {len(self.consensus_algorithms)} distinct approaches available
        - Quality weighting: {'enabled' if self.quality_weighting else 'disabled'} with adaptive component weights
        - Temporal smoothing: {'enabled' if self.temporal_smoothing else 'disabled'} with α={self.consensus_intelligence['smoothing_alpha']:.2f}
        - Stability analysis: {self.consensus_intelligence['stability_window']}-step window for trend detection
        
        Consensus Intelligence Features:
        - Market regime adaptation with volatility-aware scaling
        - Multi-dimensional analysis across direction, magnitude, and confidence metrics
        - Real-time member contribution and influence tracking
        - Comprehensive quality metrics and performance analytics
        
        Advanced Capabilities:
        - Network consensus analysis for complex member interactions
        - Temporal consensus patterns with predictive accuracy measurement
        - Adaptive quality thresholds based on market conditions
        - Real-time consensus trend analysis and recommendation generation
        
        Expected Outcomes:
        - Enhanced decision quality through comprehensive agreement measurement
        - Improved voting coherence with member contribution optimization
        - Optimal consensus sensitivity adapted to current market conditions
        - Transparent consensus decisions with detailed quality analysis and actionable insights
        """
        
        self.smart_bus.set('consensus_detector_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'members': self.n_members,
                'threshold': self.threshold,
                'consensus_methods': list(self.consensus_algorithms.keys()),
                'intelligence_parameters': self.consensus_intelligence
            }
        }, module='ConsensusDetector', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive consensus analysis
        
        Returns:
            Dict containing consensus results, quality metrics, and recommendations
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive voting data from SmartInfoBus
            voting_data = await self._get_comprehensive_voting_data()
            
            # Update consensus parameters based on market conditions
            await self._update_consensus_parameters_comprehensive(voting_data)
            
            # Perform comprehensive consensus analysis
            consensus_analysis = await self._perform_comprehensive_consensus_analysis(voting_data)
            
            # Update member contribution profiles
            contribution_updates = await self._update_member_contributions_comprehensive(voting_data)
            
            # Analyze consensus trends and patterns
            trend_analysis = await self._analyze_consensus_trends_comprehensive(voting_data)
            
            # Calculate comprehensive quality metrics
            quality_analysis = await self._calculate_comprehensive_quality_metrics()
            
            # Generate intelligent consensus recommendations
            recommendations = await self._generate_intelligent_consensus_recommendations(
                consensus_analysis, quality_analysis, trend_analysis
            )
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_consensus_thesis(
                consensus_analysis, quality_analysis, recommendations
            )
            
            # Create comprehensive results
            results = {
                'consensus_score': self.last_consensus,
                'consensus_quality': self.consensus_quality,
                'consensus_components': self.consensus_components.copy(),
                'directional_consensus': self.directional_consensus,
                'magnitude_consensus': self.magnitude_consensus,
                'confidence_consensus': self.confidence_consensus,
                'member_contributions': self._get_member_contributions_summary(),
                'consensus_trends': self._get_consensus_trends_summary(),
                'quality_metrics': quality_analysis,
                'consensus_recommendations': recommendations,
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('ConsensusDetector', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_voting_data(self) -> Dict[str, Any]:
        """Get comprehensive voting data using modern SmartInfoBus patterns"""
        try:
            return {
                'votes': self.smart_bus.get('votes', 'ConsensusDetector') or [],
                'raw_proposals': self.smart_bus.get('raw_proposals', 'ConsensusDetector') or [],
                'member_confidences': self.smart_bus.get('member_confidences', 'ConsensusDetector') or [],
                'voting_summary': self.smart_bus.get('voting_summary', 'ConsensusDetector') or {},
                'alpha_weights': self.smart_bus.get('alpha_weights', 'ConsensusDetector') or [],
                'blended_action': self.smart_bus.get('blended_action', 'ConsensusDetector') or [],
                'market_context': self.smart_bus.get('market_context', 'ConsensusDetector') or {},
                'agreement_score': self.smart_bus.get('agreement_score', 'ConsensusDetector') or 0.5,
                'consensus_direction': self.smart_bus.get('consensus_direction', 'ConsensusDetector') or 'neutral',
                'market_regime': self.smart_bus.get('market_regime', 'ConsensusDetector') or 'unknown',
                'volatility_data': self.smart_bus.get('volatility_data', 'ConsensusDetector') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "ConsensusDetector")
            self.logger.warning(f"Voting data retrieval incomplete: {error_context}")
            return self._get_safe_voting_defaults()

    async def _update_consensus_parameters_comprehensive(self, voting_data: Dict[str, Any]):
        """Update comprehensive consensus parameters with intelligent adaptation"""
        try:
            # Extract market intelligence
            market_context = voting_data.get('market_context', {})
            regime = voting_data.get('market_regime', 'unknown')
            volatility_data = voting_data.get('volatility_data', {})
            agreement_score = voting_data.get('agreement_score', 0.5)
            
            # Calculate market consensus factors
            market_consensus_factor = self._calculate_market_consensus_factor(voting_data)
            
            # Apply regime-based adaptations
            regime_config = self.market_adaptation['regime_adjustments'].get(regime, {})
            weight_multiplier = regime_config.get('weight_multiplier', 1.0)
            stability_factor = regime_config.get('stability_factor', 1.0)
            
            # Apply volatility-based adaptations
            volatility_level = volatility_data.get('level', 'medium')
            volatility_multiplier = self.market_adaptation['volatility_adjustments'].get(volatility_level, 1.0)
            
            # Update smoothing parameters
            old_alpha = self.consensus_intelligence['smoothing_alpha']
            base_alpha = 0.3
            
            # Adjust smoothing based on market conditions
            if regime == 'volatile' or volatility_level in ['high', 'extreme']:
                # More smoothing in volatile conditions
                new_alpha = base_alpha * 0.7
            elif regime == 'trending':
                # Less smoothing in trending conditions (more responsive)
                new_alpha = base_alpha * 1.3
            else:
                new_alpha = base_alpha
            
            # Apply market consensus factor
            new_alpha *= market_consensus_factor
            new_alpha = np.clip(new_alpha, 0.1, 0.8)
            
            # Update with momentum
            adaptation_rate = self.consensus_intelligence['adaptation_rate']
            self.consensus_intelligence['smoothing_alpha'] = (
                old_alpha * (1 - adaptation_rate) + new_alpha * adaptation_rate
            )
            
            # Update stability window based on volatility
            if volatility_level in ['high', 'extreme']:
                self.consensus_intelligence['stability_window'] = min(15, self.consensus_intelligence['stability_window'] + 1)
            elif volatility_level in ['very_low', 'low']:
                self.consensus_intelligence['stability_window'] = max(5, self.consensus_intelligence['stability_window'] - 1)
            
            # Log significant parameter changes
            alpha_change = abs(self.consensus_intelligence['smoothing_alpha'] - old_alpha)
            if alpha_change > 0.05:
                self.logger.info(format_operator_message(
                    icon="⚙️",
                    message="Consensus parameters adapted",
                    old_alpha=f"{old_alpha:.3f}",
                    new_alpha=f"{self.consensus_intelligence['smoothing_alpha']:.3f}",
                    regime=regime,
                    volatility=volatility_level,
                    market_factor=f"{market_consensus_factor:.3f}"
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_parameters_update")
            self.logger.warning(f"Consensus parameter update failed: {error_context}")

    def _calculate_market_consensus_factor(self, voting_data: Dict[str, Any]) -> float:
        """Calculate comprehensive market consensus factor"""
        try:
            consensus_factors = []
            
            # Existing agreement factor
            agreement_score = voting_data.get('agreement_score', 0.5)
            consensus_factors.append(agreement_score)
            
            # Market regime factor
            regime = voting_data.get('market_regime', 'unknown')
            regime_factor = {
                'trending': 0.8,    # Trending markets often have clearer consensus
                'ranging': 0.6,     # Ranging markets have mixed consensus
                'volatile': 0.3,    # Volatile markets have low consensus
                'breakout': 0.9,    # Breakouts have high consensus
                'reversal': 0.4,    # Reversals have conflicted consensus
                'unknown': 0.5
            }.get(regime, 0.5)
            consensus_factors.append(regime_factor)
            
            # Volatility factor
            volatility_data = voting_data.get('volatility_data', {})
            volatility_level = volatility_data.get('level', 'medium')
            volatility_factor = {
                'very_low': 0.9, 'low': 0.8, 'medium': 0.6, 'high': 0.4, 'extreme': 0.2
            }.get(volatility_level, 0.6)
            consensus_factors.append(volatility_factor)
            
            # Member confidence factor
            member_confidences = voting_data.get('member_confidences', [])
            if member_confidences:
                avg_confidence = np.mean(member_confidences)
                confidence_consistency = 1.0 - np.std(member_confidences)
                confidence_factor = (avg_confidence + confidence_consistency) / 2.0
                consensus_factors.append(confidence_factor)
            
            # Weighted combination
            if consensus_factors:
                weights = [0.3, 0.25, 0.25, 0.2][:len(consensus_factors)]
                weights = np.array(weights) / np.sum(weights)  # Normalize
                total_factor = np.average(consensus_factors, weights=weights)
            else:
                total_factor = 0.5
            
            return np.clip(total_factor, 0.0, 1.0)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_consensus_factor_calculation")
            return 0.5

    async def _perform_comprehensive_consensus_analysis(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive consensus analysis with multiple algorithms"""
        try:
            self.consensus_stats['total_computations'] += 1
            
            # Extract voting actions and confidences
            actions = self._extract_voting_actions(voting_data)
            confidences = self._extract_member_confidences(voting_data)
            
            if len(actions) < 2:
                return {'consensus_score': 0.5, 'analysis_status': 'insufficient_data'}
            
            # Validate and normalize inputs
            actions, confidences = await self._validate_and_normalize_inputs(actions, confidences)
            
            # Calculate comprehensive consensus components
            consensus_components = await self._calculate_comprehensive_consensus_components(actions, confidences)
            
            # Apply advanced consensus weighting
            weighted_consensus = await self._apply_advanced_consensus_weighting(
                consensus_components, actions, confidences, voting_data
            )
            
            # Apply temporal smoothing if enabled
            final_consensus = await self._apply_temporal_smoothing(weighted_consensus)
            
            # Update consensus state
            await self._update_consensus_state_comprehensive(
                final_consensus, consensus_components, actions, confidences
            )
            
            # Calculate advanced consensus quality
            consensus_quality = await self._calculate_advanced_consensus_quality(
                actions, confidences, consensus_components
            )
            
            # Record comprehensive consensus event
            await self._record_consensus_event_comprehensive(
                final_consensus, consensus_components, consensus_quality, voting_data
            )
            
            return {
                'consensus_score': final_consensus,
                'consensus_components': consensus_components,
                'consensus_quality': consensus_quality,
                'analysis_status': 'complete',
                'member_count': len(actions),
                'avg_confidence': np.mean(confidences) if confidences else 0.0
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_analysis")
            self.logger.error(f"Consensus analysis failed: {error_context}")
            return {'consensus_score': 0.5, 'analysis_status': 'error', 'error': str(error_context)}

    def _extract_voting_actions(self, voting_data: Dict[str, Any]) -> List[np.ndarray]:
        """Extract voting actions from comprehensive voting data"""
        try:
            # Try raw proposals first (most detailed)
            raw_proposals = voting_data.get('raw_proposals', [])
            if raw_proposals and len(raw_proposals) >= 2:
                actions = []
                for proposal in raw_proposals[:self.n_members]:
                    if isinstance(proposal, (list, np.ndarray)) and len(proposal) > 0:
                        actions.append(np.array(proposal, dtype=np.float32))
                if len(actions) >= 2:
                    return actions
            
            # Fallback to blended action or votes
            blended_action = voting_data.get('blended_action', [])
            if blended_action:
                # Create variations around blended action
                base_action = np.array(blended_action, dtype=np.float32)
                actions = [base_action]
                for i in range(min(self.n_members - 1, 4)):
                    noise = np.random.randn(*base_action.shape) * 0.1
                    actions.append(base_action + noise)
                return actions
            
            # Final fallback to votes
            votes = voting_data.get('votes', [])
            if votes and len(votes) >= 2:
                return [np.array([float(vote)], dtype=np.float32) for vote in votes[:self.n_members]]
            
            return []
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "voting_actions_extraction")
            return []

    def _extract_member_confidences(self, voting_data: Dict[str, Any]) -> List[float]:
        """Extract member confidences from voting data"""
        try:
            # Try member confidences first
            confidences = voting_data.get('member_confidences', [])
            if confidences:
                return [max(0.0, min(1.0, float(c))) for c in confidences[:self.n_members]]
            
            # Fallback to alpha weights as confidence proxy
            alpha_weights = voting_data.get('alpha_weights', [])
            if alpha_weights:
                # Normalize alpha weights as confidence measures
                weights = np.array(alpha_weights[:self.n_members])
                if np.sum(weights) > 0:
                    normalized_weights = weights / np.sum(weights)
                    return [max(0.1, min(1.0, float(w) * self.n_members)) for w in normalized_weights]
            
            # Default to moderate confidence
            return [0.5] * min(self.n_members, 5)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "confidence_extraction")
            return [0.5] * min(self.n_members, 5)

    async def _validate_and_normalize_inputs(self, actions: List[np.ndarray], 
                                           confidences: List[float]) -> Tuple[List[np.ndarray], List[float]]:
        """Validate and normalize voting inputs"""
        try:
            # Ensure equal lengths
            min_len = min(len(actions), len(confidences), self.n_members)
            actions = actions[:min_len]
            confidences = confidences[:min_len]
            
            # Validate actions
            validated_actions = []
            for action in actions:
                if isinstance(action, (list, np.ndarray)):
                    action_array = np.array(action, dtype=np.float32).flatten()
                    if len(action_array) > 0:
                        validated_actions.append(action_array)
            
            # Validate confidences
            validated_confidences = [max(0.0, min(1.0, float(c))) for c in confidences]
            
            # Ensure we have sufficient data
            min_validated = min(len(validated_actions), len(validated_confidences))
            if min_validated < 2:
                # Create minimal valid data
                default_action = np.array([0.0], dtype=np.float32)
                validated_actions = [default_action, default_action.copy()]
                validated_confidences = [0.5, 0.5]
            
            return validated_actions[:min_validated], validated_confidences[:min_validated]
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "input_validation")
            # Return safe defaults
            default_action = np.array([0.0], dtype=np.float32)
            return [default_action, default_action.copy()], [0.5, 0.5]

    async def _calculate_comprehensive_consensus_components(self, actions: List[np.ndarray], 
                                                          confidences: List[float]) -> Dict[str, float]:
        """Calculate comprehensive consensus components using multiple algorithms"""
        try:
            components = {}
            
            # 1. Cosine agreement (directional consensus)
            if 'cosine_agreement' in self.consensus_methods:
                components['cosine_agreement'] = await self._calculate_cosine_agreement_advanced(actions, confidences)
            
            # 2. Direction alignment (binary consensus)
            if 'direction_alignment' in self.consensus_methods:
                components['direction_alignment'] = await self._calculate_direction_alignment_advanced(actions, confidences)
            
            # 3. Confidence-weighted consensus
            if 'confidence_weighted' in self.consensus_methods:
                components['confidence_weighted'] = await self._calculate_confidence_weighted_consensus_advanced(actions, confidences)
            
            # 4. Magnitude consensus
            components['magnitude_consensus'] = await self._calculate_magnitude_consensus_advanced(actions, confidences)
            
            # 5. Network consensus (if enabled)
            if len(actions) >= 3:
                components['network_consensus'] = await self._calculate_network_consensus_advanced(actions, confidences)
            
            # 6. Temporal consensus (if sufficient history)
            if len(self.consensus_history) >= 3:
                components['temporal_consensus'] = await self._calculate_temporal_consensus_advanced(actions, confidences)
            
            return components
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_components_calculation")
            return {'fallback_consensus': 0.5}

    async def _calculate_cosine_agreement_advanced(self, actions: List[np.ndarray], 
                                                 confidences: List[float]) -> float:
        """Calculate advanced cosine agreement with confidence weighting"""
        try:
            agreements = []
            weights = []
            
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    a1, a2 = actions[i], actions[j]
                    
                    # Ensure same dimensionality
                    max_len = max(len(a1), len(a2))
                    a1_padded = np.pad(a1, (0, max_len - len(a1)))
                    a2_padded = np.pad(a2, (0, max_len - len(a2)))
                    
                    # Calculate cosine similarity
                    norm1, norm2 = np.linalg.norm(a1_padded), np.linalg.norm(a2_padded)
                    if norm1 > 1e-6 and norm2 > 1e-6:
                        similarity = np.dot(a1_padded, a2_padded) / (norm1 * norm2)
                        # Convert to 0-1 scale
                        agreement = (similarity + 1.0) / 2.0
                        
                        # Advanced confidence weighting
                        conf_weight = (confidences[i] * confidences[j]) ** 0.5  # Geometric mean
                        quality_weight = min(confidences[i], confidences[j])    # Quality factor
                        total_weight = conf_weight * quality_weight
                        
                        agreements.append(agreement)
                        weights.append(total_weight)
            
            if agreements and sum(weights) > 0:
                weighted_agreement = np.average(agreements, weights=weights)
                return float(np.clip(weighted_agreement, 0.0, 1.0))
            else:
                return 0.5
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "cosine_agreement_advanced")
            return 0.5

    async def _calculate_direction_alignment_advanced(self, actions: List[np.ndarray], 
                                                    confidences: List[float]) -> float:
        """Calculate advanced direction alignment with adaptive thresholds"""
        try:
            # Multi-dimensional direction analysis
            alignment_scores = []
            
            # Analyze each dimension if multi-dimensional
            max_dims = max(len(action) for action in actions)
            
            for dim in range(max_dims):
                directions = []
                valid_confidences = []
                
                for action, confidence in zip(actions, confidences):
                    if dim < len(action) and abs(action[dim]) > 1e-6:
                        directions.append(np.sign(action[dim]))
                        valid_confidences.append(confidence)
                
                if len(directions) >= 2:
                    # Calculate weighted directional agreement
                    positive_weight = sum(conf for dir, conf in zip(directions, valid_confidences) if dir > 0)
                    negative_weight = sum(conf for dir, conf in zip(directions, valid_confidences) if dir < 0)
                    total_weight = positive_weight + negative_weight
                    
                    if total_weight > 0:
                        alignment = abs(positive_weight - negative_weight) / total_weight
                        alignment_scores.append(alignment)
            
            if alignment_scores:
                # Average across dimensions
                return float(np.mean(alignment_scores))
            else:
                return 0.5
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "direction_alignment_advanced")
            return 0.5

    async def _calculate_confidence_weighted_consensus_advanced(self, actions: List[np.ndarray], 
                                                              confidences: List[float]) -> float:
        """Calculate advanced confidence-weighted consensus"""
        try:
            if not confidences:
                return 0.5
            
            # Multiple confidence-based metrics
            confidence_metrics = []
            
            # 1. Confidence variance (lower variance = higher consensus)
            conf_variance = np.var(confidences)
            conf_mean = np.mean(confidences)
            if conf_mean > 0:
                relative_variance = conf_variance / (conf_mean ** 2)
                variance_consensus = max(0.0, 1.0 - relative_variance * 1.5)
                confidence_metrics.append(variance_consensus)
            
            # 2. High-confidence agreement (focus on confident members)
            high_conf_threshold = np.percentile(confidences, 70)
            high_conf_members = [i for i, c in enumerate(confidences) if c >= high_conf_threshold]
            
            if len(high_conf_members) >= 2:
                high_conf_actions = [actions[i] for i in high_conf_members]
                high_conf_agreement = await self._calculate_action_agreement(high_conf_actions)
                confidence_metrics.append(high_conf_agreement)
            
            # 3. Confidence-action correlation
            if len(actions) >= 3:
                action_magnitudes = [np.linalg.norm(action) for action in actions]
                if np.std(action_magnitudes) > 0 and np.std(confidences) > 0:
                    correlation = np.corrcoef(confidences, action_magnitudes)[0, 1]
                    if not np.isnan(correlation):
                        # High correlation indicates confident members take stronger actions
                        correlation_consensus = (abs(correlation) + 1.0) / 2.0
                        confidence_metrics.append(correlation_consensus)
            
            if confidence_metrics:
                return float(np.mean(confidence_metrics))
            else:
                return 0.5
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "confidence_weighted_consensus_advanced")
            return 0.5

    async def _calculate_action_agreement(self, actions: List[np.ndarray]) -> float:
        """Calculate agreement between a set of actions"""
        try:
            if len(actions) < 2:
                return 0.5
            
            agreements = []
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    a1, a2 = actions[i], actions[j]
                    
                    # Ensure same dimensionality
                    max_len = max(len(a1), len(a2))
                    a1_padded = np.pad(a1, (0, max_len - len(a1)))
                    a2_padded = np.pad(a2, (0, max_len - len(a2)))
                    
                    # Calculate similarity
                    norm1, norm2 = np.linalg.norm(a1_padded), np.linalg.norm(a2_padded)
                    if norm1 > 1e-6 and norm2 > 1e-6:
                        similarity = np.dot(a1_padded, a2_padded) / (norm1 * norm2)
                        agreement = (similarity + 1.0) / 2.0
                        agreements.append(agreement)
            
            return float(np.mean(agreements)) if agreements else 0.5
            
        except Exception:
            return 0.5

    async def _calculate_magnitude_consensus_advanced(self, actions: List[np.ndarray], 
                                                    confidences: List[float]) -> float:
        """Calculate advanced magnitude consensus with outlier handling"""
        try:
            magnitudes = [np.linalg.norm(action) for action in actions]
            
            if len(magnitudes) < 2:
                return 0.5
            
            # Confidence-weighted magnitude analysis
            weighted_magnitudes = []
            for mag, conf in zip(magnitudes, confidences):
                weighted_magnitudes.append(mag * conf)
            
            # Calculate multiple magnitude consensus metrics
            magnitude_metrics = []
            
            # 1. Coefficient of variation (lower CV = higher consensus)
            mean_magnitude = np.mean(weighted_magnitudes)
            if mean_magnitude > 1e-6:
                std_magnitude = np.std(weighted_magnitudes)
                cv = std_magnitude / mean_magnitude
                cv_consensus = max(0.0, 1.0 - cv)
                magnitude_metrics.append(cv_consensus)
            
            # 2. Outlier-robust consensus (using median absolute deviation)
            median_magnitude = np.median(magnitudes)
            mad = np.median(np.abs(np.array(magnitudes) - median_magnitude))
            if mad > 1e-6:
                outlier_scores = np.abs(np.array(magnitudes) - median_magnitude) / mad
                # Percentage of non-outliers (outlier threshold = 2.5 MADs)
                non_outlier_ratio = np.mean(outlier_scores <= 2.5)
                magnitude_metrics.append(non_outlier_ratio)
            
            # 3. Range-based consensus
            if len(magnitudes) > 2:
                mag_range = np.max(magnitudes) - np.min(magnitudes)
                avg_magnitude = np.mean(magnitudes)
                if avg_magnitude > 1e-6:
                    relative_range = mag_range / avg_magnitude
                    range_consensus = max(0.0, 1.0 - relative_range / 2.0)
                    magnitude_metrics.append(range_consensus)
            
            if magnitude_metrics:
                return float(np.mean(magnitude_metrics))
            else:
                return 0.5
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "magnitude_consensus_advanced")
            return 0.5

    async def _calculate_network_consensus_advanced(self, actions: List[np.ndarray], 
                                                  confidences: List[float]) -> float:
        """Calculate network-based consensus considering member interconnections"""
        try:
            n = len(actions)
            if n < 3:
                return 0.5
            
            # Build similarity matrix
            similarity_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        a1, a2 = actions[i], actions[j]
                        max_len = max(len(a1), len(a2))
                        a1_padded = np.pad(a1, (0, max_len - len(a1)))
                        a2_padded = np.pad(a2, (0, max_len - len(a2)))
                        
                        norm1, norm2 = np.linalg.norm(a1_padded), np.linalg.norm(a2_padded)
                        if norm1 > 1e-6 and norm2 > 1e-6:
                            similarity = np.dot(a1_padded, a2_padded) / (norm1 * norm2)
                            similarity_matrix[i, j] = (similarity + 1.0) / 2.0
            
            # Weight by confidence
            conf_weights = np.array(confidences)
            weighted_similarity = similarity_matrix * np.outer(conf_weights, conf_weights)
            
            # Calculate network metrics
            network_metrics = []
            
            # 1. Average network density
            network_density = np.mean(weighted_similarity[weighted_similarity > 0])
            network_metrics.append(network_density)
            
            # 2. Network clustering coefficient
            clustering_scores = []
            for i in range(n):
                neighbors = [j for j in range(n) if i != j and weighted_similarity[i, j] > 0.5]
                if len(neighbors) >= 2:
                    # Calculate clustering among neighbors
                    neighbor_similarities = []
                    for ni in neighbors:
                        for nj in neighbors:
                            if ni != nj:
                                neighbor_similarities.append(weighted_similarity[ni, nj])
                    if neighbor_similarities:
                        clustering_scores.append(np.mean(neighbor_similarities))
            
            if clustering_scores:
                network_clustering = np.mean(clustering_scores)
                network_metrics.append(network_clustering)
            
            # 3. Network consensus strength
            # Higher values when most members are similar to each other
            consensus_strength = np.mean(weighted_similarity)
            network_metrics.append(consensus_strength)
            
            if network_metrics:
                return float(np.mean(network_metrics))
            else:
                return 0.5
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "network_consensus_advanced")
            return 0.5

    async def _calculate_temporal_consensus_advanced(self, actions: List[np.ndarray], 
                                                   confidences: List[float]) -> float:
        """Calculate temporal consensus based on historical patterns"""
        try:
            if len(self.consensus_history) < 3:
                return 0.5
            
            # Get recent consensus history
            recent_consensus = [event.get('consensus', 0.5) for event in list(self.consensus_history)[-10:]]
            
            # Temporal consensus metrics
            temporal_metrics = []
            
            # 1. Consensus stability (low variance = high temporal consensus)
            consensus_variance = np.var(recent_consensus)
            consensus_mean = np.mean(recent_consensus)
            if consensus_mean > 0:
                relative_variance = consensus_variance / (consensus_mean ** 2)
                stability_consensus = max(0.0, 1.0 - relative_variance)
                temporal_metrics.append(stability_consensus)
            
            # 2. Trend consistency
            if len(recent_consensus) >= 5:
                # Calculate trend slope
                x = np.arange(len(recent_consensus))
                slope = np.polyfit(x, recent_consensus, 1)[0]
                
                # Trend consistency (smaller slope = more stable)
                trend_consistency = max(0.0, 1.0 - abs(slope) * 10)
                temporal_metrics.append(trend_consistency)
            
            # 3. Predictive accuracy
            if len(recent_consensus) >= 4:
                # Simple prediction: next consensus = average of last 3
                predicted_consensus = np.mean(recent_consensus[-3:])
                actual_consensus = recent_consensus[-1]
                prediction_error = abs(predicted_consensus - actual_consensus)
                prediction_accuracy = max(0.0, 1.0 - prediction_error * 2)
                temporal_metrics.append(prediction_accuracy)
            
            if temporal_metrics:
                return float(np.mean(temporal_metrics))
            else:
                return 0.5
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "temporal_consensus_advanced")
            return 0.5

    async def _apply_advanced_consensus_weighting(self, consensus_components: Dict[str, float], 
                                                actions: List[np.ndarray], confidences: List[float],
                                                voting_data: Dict[str, Any]) -> float:
        """Apply advanced weighting to consensus components"""
        try:
            if not consensus_components:
                return 0.5
            
            # Calculate dynamic weights based on data quality and context
            weights = await self._calculate_dynamic_component_weights(
                consensus_components, actions, confidences, voting_data
            )
            
            # Apply weights
            weighted_sum = 0.0
            total_weight = 0.0
            
            for component, score in consensus_components.items():
                weight = weights.get(component, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_consensus = weighted_sum / total_weight
            else:
                weighted_consensus = np.mean(list(consensus_components.values()))
            
            return float(np.clip(weighted_consensus, 0.0, 1.0))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_weighting")
            return np.mean(list(consensus_components.values())) if consensus_components else 0.5

    async def _calculate_dynamic_component_weights(self, consensus_components: Dict[str, float],
                                                 actions: List[np.ndarray], confidences: List[float],
                                                 voting_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic weights for consensus components"""
        try:
            weights = {}
            
            # Base weights from algorithm effectiveness
            for component in consensus_components:
                algo_info = self.consensus_algorithms.get(component, {})
                effectiveness = algo_info.get('effectiveness_threshold', 0.5)
                weights[component] = effectiveness
            
            # Adjust based on data characteristics
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            # High confidence: trust directional measures more
            if avg_confidence > 0.8:
                weights['cosine_agreement'] = weights.get('cosine_agreement', 1.0) * 1.3
                weights['direction_alignment'] = weights.get('direction_alignment', 1.0) * 1.2
            
            # Low confidence: rely more on robust measures
            elif avg_confidence < 0.4:
                weights['magnitude_consensus'] = weights.get('magnitude_consensus', 1.0) * 1.4
                weights['network_consensus'] = weights.get('network_consensus', 1.0) * 1.2
            
            # Market regime adjustments
            regime = voting_data.get('market_regime', 'unknown')
            if regime == 'volatile':
                # In volatile markets, trust temporal consensus less
                weights['temporal_consensus'] = weights.get('temporal_consensus', 1.0) * 0.7
            elif regime == 'trending':
                # In trending markets, directional consensus is more important
                weights['direction_alignment'] = weights.get('direction_alignment', 1.0) * 1.2
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight * len(weights) for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "dynamic_weights_calculation")
            return {component: 1.0 for component in consensus_components}

    async def _apply_temporal_smoothing(self, consensus: float) -> float:
        """Apply temporal smoothing to consensus score"""
        try:
            if not self.temporal_smoothing or len(self.consensus_history) == 0:
                return consensus
            
            # Get previous consensus
            previous_consensus = self.consensus_history[-1].get('consensus', 0.5)
            
            # Apply exponential smoothing
            alpha = self.consensus_intelligence['smoothing_alpha']
            smoothed_consensus = alpha * consensus + (1 - alpha) * previous_consensus
            
            return float(np.clip(smoothed_consensus, 0.0, 1.0))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "temporal_smoothing")
            return consensus

    async def _update_consensus_state_comprehensive(self, consensus: float, 
                                                  consensus_components: Dict[str, float],
                                                  actions: List[np.ndarray], 
                                                  confidences: List[float]):
        """Update comprehensive consensus state"""
        try:
            # Update core consensus state
            self.last_consensus = consensus
            self.consensus_components = consensus_components.copy()
            
            # Update individual consensus dimensions
            self.directional_consensus = consensus_components.get('direction_alignment', 0.5)
            self.magnitude_consensus = consensus_components.get('magnitude_consensus', 0.5)
            self.confidence_consensus = consensus_components.get('confidence_weighted', 0.5)
            self.network_consensus = consensus_components.get('network_consensus', 0.5)
            
            # Calculate temporal stability
            if len(self.consensus_history) >= self.consensus_intelligence['stability_window']:
                recent_consensus = [
                    event.get('consensus', 0.5) 
                    for event in list(self.consensus_history)[-self.consensus_intelligence['stability_window']:]
                ]
                self.temporal_stability = 1.0 - np.std(recent_consensus)
                self.temporal_stability = max(0.0, min(1.0, self.temporal_stability))
            
            # Update statistics
            await self._update_consensus_statistics_comprehensive(consensus, consensus_components)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_state_update")

    async def _calculate_advanced_consensus_quality(self, actions: List[np.ndarray], 
                                                  confidences: List[float],
                                                  consensus_components: Dict[str, float]) -> float:
        """Calculate advanced consensus quality metrics"""
        try:
            quality_components = []
            
            # 1. Component coherence (how well components agree)
            if len(consensus_components) > 1:
                component_values = list(consensus_components.values())
                coherence = 1.0 - np.std(component_values) / max(np.mean(component_values), 1e-6)
                self.quality_metrics['coherence'] = max(0.0, min(1.0, coherence))
                quality_components.append(self.quality_metrics['coherence'])
            
            # 2. Temporal stability
            if len(self.consensus_history) >= self.consensus_intelligence['stability_window']:
                recent_consensus = [
                    event.get('consensus', 0.5) 
                    for event in list(self.consensus_history)[-self.consensus_intelligence['stability_window']:]
                ]
                stability = 1.0 - np.std(recent_consensus)
                self.quality_metrics['stability'] = max(0.0, min(1.0, stability))
                quality_components.append(self.quality_metrics['stability'])
            
            # 3. Input diversity (balanced inputs indicate higher quality)
            if len(actions) > 1:
                diversity = await self._calculate_input_diversity_advanced(actions)
                self.quality_metrics['diversity'] = diversity
                quality_components.append(diversity)
            
            # 4. Reliability (based on confidence levels and consistency)
            if confidences:
                avg_confidence = np.mean(confidences)
                conf_consistency = 1.0 - np.std(confidences) / max(np.mean(confidences), 1e-6)
                reliability = (avg_confidence + conf_consistency) / 2.0
                self.quality_metrics['reliability'] = max(0.0, min(1.0, reliability))
                quality_components.append(self.quality_metrics['reliability'])
            
            # 5. Predictive accuracy (if sufficient history)
            if len(self.prediction_history) >= 3:
                prediction_accuracy = await self._calculate_prediction_accuracy()
                self.quality_metrics['predictive_accuracy'] = prediction_accuracy
                quality_components.append(prediction_accuracy)
            
            # 6. Temporal consistency
            if len(self.consensus_history) >= 5:
                temporal_consistency = await self._calculate_temporal_consistency()
                self.quality_metrics['temporal_consistency'] = temporal_consistency
                quality_components.append(temporal_consistency)
            
            # Calculate overall quality
            if quality_components:
                overall_quality = np.mean(quality_components)
            else:
                overall_quality = 0.5
            
            self.consensus_quality = float(np.clip(overall_quality, 0.0, 1.0))
            return self.consensus_quality
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_quality_calculation")
            return 0.5

    async def _calculate_input_diversity_advanced(self, actions: List[np.ndarray]) -> float:
        """Calculate advanced input diversity metrics"""
        try:
            if len(actions) < 2:
                return 0.0
            
            diversity_metrics = []
            
            # 1. Pairwise distance diversity
            distances = []
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    a1, a2 = actions[i], actions[j]
                    max_len = max(len(a1), len(a2))
                    a1_padded = np.pad(a1, (0, max_len - len(a1)))
                    a2_padded = np.pad(a2, (0, max_len - len(a2)))
                    distance = np.linalg.norm(a1_padded - a2_padded)
                    distances.append(distance)
            
            if distances:
                # Normalize by maximum distance
                max_distance = max(distances)
                if max_distance > 0:
                    avg_distance = np.mean(distances) / max_distance
                    diversity_metrics.append(avg_distance)
            
            # 2. Direction diversity (entropy of directions)
            if all(len(action) > 0 for action in actions):
                directions = [np.sign(action[0]) for action in actions if abs(action[0]) > 1e-6]
                if len(directions) > 1:
                    unique_directions = len(set(directions))
                    max_directions = min(2, len(directions))  # Binary directions
                    direction_diversity = unique_directions / max_directions
                    diversity_metrics.append(direction_diversity)
            
            # 3. Magnitude diversity
            magnitudes = [np.linalg.norm(action) for action in actions]
            if len(magnitudes) > 1 and np.mean(magnitudes) > 0:
                magnitude_cv = np.std(magnitudes) / np.mean(magnitudes)
                magnitude_diversity = min(1.0, magnitude_cv)
                diversity_metrics.append(magnitude_diversity)
            
            if diversity_metrics:
                return float(np.mean(diversity_metrics))
            else:
                return 0.5
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "input_diversity_calculation")
            return 0.5

    async def _calculate_prediction_accuracy(self) -> float:
        """Calculate predictive accuracy based on historical predictions"""
        try:
            if len(self.prediction_history) < 3:
                return 0.5
            
            errors = []
            for prediction_event in self.prediction_history:
                predicted = prediction_event.get('predicted_consensus', 0.5)
                actual = prediction_event.get('actual_consensus', 0.5)
                error = abs(predicted - actual)
                errors.append(error)
            
            if errors:
                avg_error = np.mean(errors)
                accuracy = max(0.0, 1.0 - avg_error * 2)  # Scale error to accuracy
                return float(accuracy)
            else:
                return 0.5
                
        except Exception:
            return 0.5

    async def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of consensus measurements"""
        try:
            if len(self.consensus_history) < 5:
                return 0.5
            
            # Get recent consensus values
            recent_consensus = [
                event.get('consensus', 0.5) 
                for event in list(self.consensus_history)[-10:]
            ]
            
            # Calculate autocorrelation at lag 1
            if len(recent_consensus) >= 3:
                lag1_correlation = np.corrcoef(recent_consensus[:-1], recent_consensus[1:])[0, 1]
                if not np.isnan(lag1_correlation):
                    # Convert correlation to consistency score
                    consistency = (abs(lag1_correlation) + 1.0) / 2.0
                    return float(consistency)
            
            return 0.5
            
        except Exception:
            return 0.5

    async def _record_consensus_event_comprehensive(self, consensus: float, 
                                                  consensus_components: Dict[str, float],
                                                  consensus_quality: float, 
                                                  voting_data: Dict[str, Any]):
        """Record comprehensive consensus event"""
        try:
            timestamp = datetime.datetime.now().isoformat()
            
            consensus_event = {
                'timestamp': timestamp,
                'consensus': consensus,
                'consensus_quality': consensus_quality,
                'components': consensus_components.copy(),
                'directional_consensus': self.directional_consensus,
                'magnitude_consensus': self.magnitude_consensus,
                'confidence_consensus': self.confidence_consensus,
                'network_consensus': self.network_consensus,
                'temporal_stability': self.temporal_stability,
                'member_count': len(voting_data.get('raw_proposals', [])),
                'avg_confidence': np.mean(voting_data.get('member_confidences', [0.5])),
                'market_regime': voting_data.get('market_regime', 'unknown'),
                'agreement_score': voting_data.get('agreement_score', 0.5),
                'quality_metrics': self.quality_metrics.copy()
            }
            
            self.consensus_history.append(consensus_event)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_event_recording")

    async def _update_consensus_statistics_comprehensive(self, consensus: float, 
                                                       consensus_components: Dict[str, float]):
        """Update comprehensive consensus statistics"""
        try:
            # Update counts based on consensus levels
            if consensus > 0.7:
                self.consensus_stats['high_consensus_count'] += 1
            elif consensus < 0.3:
                self.consensus_stats['low_consensus_count'] += 1
            
            # Update running averages
            total = self.consensus_stats['total_computations']
            old_avg = self.consensus_stats['avg_consensus']
            self.consensus_stats['avg_consensus'] = (old_avg * (total - 1) + consensus) / total
            
            # Update volatility
            if len(self.consensus_history) >= 10:
                recent_consensus = [
                    event.get('consensus', 0.5) 
                    for event in list(self.consensus_history)[-10:]
                ]
                self.consensus_stats['consensus_volatility'] = float(np.std(recent_consensus))
            
            # Update quality score
            self.consensus_stats['quality_score'] = self.consensus_quality
            
            # Update performance metrics
            self._update_performance_metric('consensus_score', consensus)
            self._update_performance_metric('consensus_quality', self.consensus_quality)
            self._update_performance_metric('directional_consensus', self.directional_consensus)
            self._update_performance_metric('temporal_stability', self.temporal_stability)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_statistics_update")

    async def _update_member_contributions_comprehensive(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update comprehensive member contribution analysis"""
        try:
            contribution_updates = {
                'updated_members': [],
                'influence_changes': {},
                'reliability_updates': {}
            }
            
            raw_proposals = voting_data.get('raw_proposals', [])
            confidences = voting_data.get('member_confidences', [])
            
            if len(raw_proposals) < 2:
                return contribution_updates
            
            # Analyze each member's contribution
            for i, proposal in enumerate(raw_proposals[:self.n_members]):
                if not isinstance(proposal, (list, np.ndarray)) or len(proposal) == 0:
                    continue
                
                proposal_array = np.array(proposal, dtype=np.float32)
                contribution = self.member_contributions[i]
                
                # Calculate alignment with other members
                alignments = []
                for j, other_proposal in enumerate(raw_proposals[:self.n_members]):
                    if i != j and isinstance(other_proposal, (list, np.ndarray)) and len(other_proposal) > 0:
                        other_array = np.array(other_proposal, dtype=np.float32)
                        
                        # Ensure same dimensionality
                        max_len = max(len(proposal_array), len(other_array))
                        prop_padded = np.pad(proposal_array, (0, max_len - len(proposal_array)))
                        other_padded = np.pad(other_array, (0, max_len - len(other_array)))
                        
                        if np.linalg.norm(prop_padded) > 0 and np.linalg.norm(other_padded) > 0:
                            alignment = np.dot(prop_padded, other_padded) / (
                                np.linalg.norm(prop_padded) * np.linalg.norm(other_padded)
                            )
                            alignments.append((alignment + 1.0) / 2.0)  # Convert to 0-1
                
                if alignments:
                    # Update member contribution metrics with exponential smoothing
                    memory_factor = self.consensus_intelligence['temporal_memory']
                    
                    old_alignment = contribution.get('avg_alignment', 0.5)
                    new_alignment = np.mean(alignments)
                    contribution['avg_alignment'] = (
                        old_alignment * memory_factor + new_alignment * (1 - memory_factor)
                    )
                    
                    contribution['consistency'] = 1.0 - np.std(alignments)
                    
                    # Calculate consensus contribution (how much member helps overall consensus)
                    consensus_contribution = await self._calculate_member_consensus_contribution(
                        i, proposal_array, raw_proposals, confidences
                    )
                    contribution['consensus_contribution'] = consensus_contribution
                    
                    # Update reliability score
                    if i < len(confidences):
                        confidence = confidences[i]
                        reliability = (confidence + contribution['consistency'] + contribution['avg_alignment']) / 3.0
                        contribution['reliability_score'] = reliability
                    
                    # Calculate influence weight
                    base_weight = 1.0 / max(self.n_members, 1)
                    quality_multiplier = (contribution['reliability_score'] + contribution['consensus_contribution']) / 2.0
                    contribution['influence_weight'] = base_weight * quality_multiplier
                    
                    # Track updates
                    if abs(new_alignment - old_alignment) > 0.1:
                        contribution_updates['updated_members'].append(i)
                        contribution_updates['influence_changes'][i] = {
                            'old_influence': base_weight,
                            'new_influence': contribution['influence_weight'],
                            'change_reason': 'alignment_update'
                        }
            
            return contribution_updates
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "member_contributions_update")
            return {'updated_members': [], 'influence_changes': {}, 'reliability_updates': {}}

    async def _calculate_member_consensus_contribution(self, member_idx: int, member_proposal: np.ndarray,
                                                     all_proposals: List, confidences: List[float]) -> float:
        """Calculate how much a member contributes to overall consensus"""
        try:
            # Calculate consensus with this member
            proposals_with = [np.array(prop) for prop in all_proposals if isinstance(prop, (list, np.ndarray))]
            if len(proposals_with) < 2:
                return 0.5
            
            consensus_with = await self._calculate_action_agreement(proposals_with)
            
            # Calculate consensus without this member
            proposals_without = [
                np.array(prop) for i, prop in enumerate(all_proposals) 
                if i != member_idx and isinstance(prop, (list, np.ndarray))
            ]
            
            if len(proposals_without) >= 2:
                consensus_without = await self._calculate_action_agreement(proposals_without)
                
                # Contribution is the difference
                contribution = consensus_with - consensus_without
                # Normalize to 0-1 scale
                contribution = (contribution + 1.0) / 2.0
                return float(np.clip(contribution, 0.0, 1.0))
            else:
                return 0.5
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "member_consensus_contribution")
            return 0.5

    async def _analyze_consensus_trends_comprehensive(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comprehensive consensus trends and patterns"""
        try:
            trend_analysis = {
                'current_trend': 'stable',
                'trend_strength': 0.0,
                'prediction': {},
                'pattern_analysis': {},
                'regime_analysis': {}
            }
            
            if len(self.consensus_history) < 5:
                trend_analysis['status'] = 'insufficient_history'
                return trend_analysis
            
            # Calculate trend direction and strength
            recent_consensus = [
                event.get('consensus', 0.5) 
                for event in list(self.consensus_history)[-10:]
            ]
            
            if len(recent_consensus) >= 3:
                x = np.arange(len(recent_consensus))
                slope, intercept = np.polyfit(x, recent_consensus, 1)
                
                trend_analysis['trend_strength'] = abs(slope)
                
                if slope > 0.02:
                    trend_analysis['current_trend'] = 'increasing'
                elif slope < -0.02:
                    trend_analysis['current_trend'] = 'decreasing'
                else:
                    trend_analysis['current_trend'] = 'stable'
                
                # Generate prediction for next consensus
                next_consensus = slope * len(recent_consensus) + intercept
                next_consensus = np.clip(next_consensus, 0.0, 1.0)
                
                trend_analysis['prediction'] = {
                    'next_consensus': float(next_consensus),
                    'confidence': min(1.0, 1.0 - trend_analysis['trend_strength'] * 5),
                    'trend_slope': float(slope)
                }
                
                # Record prediction for future accuracy assessment
                self.prediction_history.append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'predicted_consensus': next_consensus,
                    'actual_consensus': self.last_consensus,
                    'prediction_method': 'linear_trend'
                })
            
            # Analyze patterns by market regime
            regime = voting_data.get('market_regime', 'unknown')
            self.regime_consensus_history[regime].append(self.last_consensus)
            
            if len(self.regime_consensus_history[regime]) >= 3:
                regime_consensus = list(self.regime_consensus_history[regime])
                trend_analysis['regime_analysis'][regime] = {
                    'avg_consensus': np.mean(regime_consensus),
                    'consensus_volatility': np.std(regime_consensus),
                    'sample_count': len(regime_consensus)
                }
            
            # Store trend entry
            trend_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'trend_direction': trend_analysis['current_trend'],
                'trend_strength': trend_analysis['trend_strength'],
                'current_consensus': self.last_consensus,
                'regime': regime,
                'quality': self.consensus_quality
            }
            self.consensus_trends.append(trend_entry)
            
            return trend_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_trends_analysis")
            return {'current_trend': 'unknown', 'status': 'analysis_error'}

    async def _calculate_comprehensive_quality_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for consensus system"""
        try:
            # Quality metrics are updated in _calculate_advanced_consensus_quality
            # Here we compile additional insights
            
            quality_analysis = {
                **self.quality_metrics,
                'overall_quality_score': self.consensus_quality,
                'quality_trend': 'unknown',
                'quality_drivers': {},
                'improvement_areas': []
            }
            
            # Analyze quality trend
            if len(self.consensus_history) >= 5:
                recent_quality = [
                    event.get('consensus_quality', 0.5) 
                    for event in list(self.consensus_history)[-5:]
                ]
                
                if len(recent_quality) >= 3:
                    x = np.arange(len(recent_quality))
                    slope = np.polyfit(x, recent_quality, 1)[0]
                    
                    if slope > 0.02:
                        quality_analysis['quality_trend'] = 'improving'
                    elif slope < -0.02:
                        quality_analysis['quality_trend'] = 'declining'
                    else:
                        quality_analysis['quality_trend'] = 'stable'
            
            # Identify quality drivers
            quality_components = [(k, v) for k, v in self.quality_metrics.items() if v > 0.7]
            quality_components.sort(key=lambda x: x[1], reverse=True)
            quality_analysis['quality_drivers'] = dict(quality_components[:3])
            
            # Identify improvement areas
            improvement_areas = [(k, v) for k, v in self.quality_metrics.items() if v < 0.5]
            improvement_areas.sort(key=lambda x: x[1])
            quality_analysis['improvement_areas'] = [k for k, v in improvement_areas[:3]]
            
            return quality_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "quality_metrics_calculation")
            return {'overall_quality_score': 0.5, 'quality_trend': 'unknown'}

    async def _generate_intelligent_consensus_recommendations(self, consensus_analysis: Dict[str, Any],
                                                            quality_analysis: Dict[str, Any],
                                                            trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent consensus recommendations"""
        try:
            recommendations = []
            
            # Consensus level recommendations
            consensus_score = consensus_analysis.get('consensus_score', 0.5)
            if consensus_score > 0.9:
                recommendations.append("VERY HIGH CONSENSUS: Consider if decision-making is too homogeneous")
            elif consensus_score > 0.8:
                recommendations.append("HIGH CONSENSUS: Strong agreement detected - good for decisive action")
            elif consensus_score < 0.3:
                recommendations.append("LOW CONSENSUS: Significant disagreement - consider more discussion or analysis")
            elif consensus_score < 0.2:
                recommendations.append("CRITICAL: Very low consensus - decision may be premature")
            
            # Quality-based recommendations
            overall_quality = quality_analysis.get('overall_quality_score', 0.5)
            if overall_quality < 0.4:
                recommendations.append("QUALITY: Low consensus quality detected - review member input methods")
            
            improvement_areas = quality_analysis.get('improvement_areas', [])
            if 'reliability' in improvement_areas:
                recommendations.append("RELIABILITY: Consider member confidence calibration or training")
            if 'diversity' in improvement_areas:
                recommendations.append("DIVERSITY: Encourage more diverse perspectives in committee")
            if 'stability' in improvement_areas:
                recommendations.append("STABILITY: High consensus volatility - review decision-making process")
            
            # Trend-based recommendations
            current_trend = trend_analysis.get('current_trend', 'stable')
            if current_trend == 'decreasing':
                recommendations.append("TREND: Consensus declining - investigate source of disagreement")
            elif current_trend == 'increasing':
                recommendations.append("TREND: Consensus improving - current approach is working well")
            
            # Member contribution recommendations
            low_contributors = [
                member_id for member_id, contrib in self.member_contributions.items()
                if contrib.get('consensus_contribution', 0.5) < 0.3
            ]
            if len(low_contributors) > self.n_members // 3:
                recommendations.append(f"MEMBERS: {len(low_contributors)} members have low consensus contribution")
            
            # Market adaptation recommendations
            if hasattr(self, 'consensus_intelligence'):
                alpha = self.consensus_intelligence.get('smoothing_alpha', 0.3)
                if alpha > 0.6:
                    recommendations.append("PARAMETERS: High smoothing may be masking important consensus changes")
                elif alpha < 0.1:
                    recommendations.append("PARAMETERS: Low smoothing may cause excessive consensus volatility")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("SYSTEM: Consensus detection operating within normal parameters")
            
            return recommendations[:6]  # Limit to top 6 recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_recommendations")
            return [f"Recommendation generation failed: {error_context}"]

    async def _generate_comprehensive_consensus_thesis(self, consensus_analysis: Dict[str, Any],
                                                     quality_analysis: Dict[str, Any],
                                                     recommendations: List[str]) -> str:
        """Generate comprehensive consensus thesis"""
        try:
            # Core metrics
            consensus_score = consensus_analysis.get('consensus_score', 0.5)
            consensus_quality = quality_analysis.get('overall_quality_score', 0.5)
            member_count = consensus_analysis.get('member_count', self.n_members)
            
            thesis_parts = []
            
            # Executive summary
            consensus_level = "HIGH" if consensus_score > 0.7 else "MODERATE" if consensus_score > 0.4 else "LOW"
            thesis_parts.append(
                f"CONSENSUS ANALYSIS: {consensus_level} agreement with {consensus_score:.1%} consensus score"
            )
            
            # Quality assessment
            thesis_parts.append(
                f"QUALITY METRICS: {consensus_quality:.1%} overall quality across {len(self.consensus_components)} components"
            )
            
            # Component breakdown
            if self.consensus_components:
                best_component = max(self.consensus_components.items(), key=lambda x: x[1])
                thesis_parts.append(
                    f"COMPONENT ANALYSIS: {best_component[0]} leads with {best_component[1]:.1%} agreement"
                )
            
            # Member dynamics
            thesis_parts.append(
                f"MEMBER DYNAMICS: {member_count} active members with {len(self.member_contributions)} tracked"
            )
            
            # Temporal insights
            if self.temporal_stability > 0:
                thesis_parts.append(f"STABILITY: {self.temporal_stability:.1%} temporal stability measured")
            
            # System performance
            total_computations = self.consensus_stats.get('total_computations', 0)
            thesis_parts.append(f"SYSTEM PERFORMANCE: {total_computations} consensus computations completed")
            
            # Recommendations summary
            priority_recommendations = [rec for rec in recommendations if any(keyword in rec 
                                      for keyword in ['CRITICAL', 'HIGH', 'LOW CONSENSUS'])]
            if priority_recommendations:
                thesis_parts.append(f"ACTION ITEMS: {len(priority_recommendations)} priority recommendations")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "consensus_thesis_generation")
            return f"Consensus thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive consensus results"""
        try:
            # Core consensus results
            self.smart_bus.set('consensus_score', results['consensus_score'],
                             module='ConsensusDetector', thesis=thesis)
            
            # Consensus quality
            quality_thesis = f"Consensus quality: {results['consensus_quality']:.3f} across multiple quality dimensions"
            self.smart_bus.set('consensus_quality', results['consensus_quality'],
                             module='ConsensusDetector', thesis=quality_thesis)
            
            # Consensus components
            components_thesis = f"Consensus components: {len(results['consensus_components'])} algorithms analyzed"
            self.smart_bus.set('consensus_components', results['consensus_components'],
                             module='ConsensusDetector', thesis=components_thesis)
            
            # Directional consensus
            directional_thesis = f"Directional consensus: {results['directional_consensus']:.3f} alignment score"
            self.smart_bus.set('directional_consensus', results['directional_consensus'],
                             module='ConsensusDetector', thesis=directional_thesis)
            
            # Magnitude consensus
            magnitude_thesis = f"Magnitude consensus: {results['magnitude_consensus']:.3f} strength agreement"
            self.smart_bus.set('magnitude_consensus', results['magnitude_consensus'],
                             module='ConsensusDetector', thesis=magnitude_thesis)
            
            # Confidence consensus
            confidence_thesis = f"Confidence consensus: {results['confidence_consensus']:.3f} reliability score"
            self.smart_bus.set('confidence_consensus', results['confidence_consensus'],
                             module='ConsensusDetector', thesis=confidence_thesis)
            
            # Member contributions
            contrib_thesis = f"Member contributions: {len(results['member_contributions'])} profiles analyzed"
            self.smart_bus.set('member_contributions', results['member_contributions'],
                             module='ConsensusDetector', thesis=contrib_thesis)
            
            # Consensus trends
            trends_thesis = f"Consensus trends: {len(self.consensus_trends)} trend points tracked"
            self.smart_bus.set('consensus_trends', results['consensus_trends'],
                             module='ConsensusDetector', thesis=trends_thesis)
            
            # Quality metrics
            quality_metrics_thesis = f"Quality metrics: {len(results['quality_metrics'])} dimensions evaluated"
            self.smart_bus.set('quality_metrics', results['quality_metrics'],
                             module='ConsensusDetector', thesis=quality_metrics_thesis)
            
            # Consensus recommendations
            rec_thesis = f"Consensus recommendations: {len(results['consensus_recommendations'])} actionable insights"
            self.smart_bus.set('consensus_recommendations', results['consensus_recommendations'],
                             module='ConsensusDetector', thesis=rec_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY COMPATIBILITY AND PUBLIC INTERFACE
    # ═══════════════════════════════════════════════════════════════════

    def compute_consensus(self, actions: List[np.ndarray], confidences: List[float]) -> float:
        """Legacy consensus computation interface for backward compatibility"""
        try:
            # Run async method synchronously
            import asyncio
            
            if asyncio.get_event_loop().is_running():
                # If already in async context, use simplified sync method
                return self._simple_consensus_computation_fallback(actions, confidences)
            else:
                # Run async analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Create minimal voting data
                    voting_data = {
                        'raw_proposals': actions,
                        'member_confidences': confidences,
                        'agreement_score': 0.5,
                        'market_regime': 'unknown'
                    }
                    
                    consensus_analysis = loop.run_until_complete(
                        self._perform_comprehensive_consensus_analysis(voting_data)
                    )
                    return consensus_analysis.get('consensus_score', 0.5)
                finally:
                    loop.close()
                    
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "legacy_consensus_computation")
            return self._simple_consensus_computation_fallback(actions, confidences)

    def _simple_consensus_computation_fallback(self, actions: List[np.ndarray], 
                                             confidences: List[float]) -> float:
        """Simple fallback consensus computation method"""
        try:
            if not actions or len(actions) < 2:
                return 0.5
            
            # Simple cosine similarity calculation
            agreements = []
            weights = []
            
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    a1, a2 = actions[i], actions[j]
                    
                    # Ensure same dimensionality
                    max_len = max(len(a1), len(a2))
                    a1_padded = np.pad(a1, (0, max_len - len(a1)))
                    a2_padded = np.pad(a2, (0, max_len - len(a2)))
                    
                    norm1, norm2 = np.linalg.norm(a1_padded), np.linalg.norm(a2_padded)
                    if norm1 > 1e-6 and norm2 > 1e-6:
                        similarity = np.dot(a1_padded, a2_padded) / (norm1 * norm2)
                        agreement = (similarity + 1.0) / 2.0
                        
                        # Weight by confidence if available
                        if i < len(confidences) and j < len(confidences):
                            weight = confidences[i] * confidences[j]
                        else:
                            weight = 1.0
                        
                        agreements.append(agreement)
                        weights.append(weight)
            
            if agreements and sum(weights) > 0:
                consensus = np.average(agreements, weights=weights)
                
                # Apply temporal smoothing if enabled
                if self.temporal_smoothing and len(self.consensus_history) > 0:
                    previous_consensus = self.consensus_history[-1].get('consensus', 0.5)
                    alpha = self.consensus_intelligence.get('smoothing_alpha', 0.3)
                    consensus = alpha * consensus + (1 - alpha) * previous_consensus
                
                self.last_consensus = float(np.clip(consensus, 0.0, 1.0))
                return self.last_consensus
            else:
                return 0.5
                
        except Exception:
            return 0.5

    def resize(self, n_members: int) -> None:
        """Resize for different number of members"""
        old_members = self.n_members
        self.n_members = int(n_members)
        
        # Clear member-specific data if size changed significantly
        if abs(self.n_members - old_members) > 2:
            self.member_contributions.clear()
            
        self.logger.info(format_operator_message(
            icon="🔄",
            message="Consensus Detector resized",
            old_members=old_members,
            new_members=self.n_members
        ))

    def _get_member_contributions_summary(self) -> Dict[str, Any]:
        """Get summary of member contributions"""
        try:
            summary = {}
            
            for member_id, contribution in self.member_contributions.items():
                summary[f'member_{member_id}'] = {
                    'avg_alignment': contribution.get('avg_alignment', 0.5),
                    'consistency': contribution.get('consistency', 0.5),
                    'influence_weight': contribution.get('influence_weight', 1.0 / max(self.n_members, 1)),
                    'consensus_contribution': contribution.get('consensus_contribution', 0.5),
                    'reliability_score': contribution.get('reliability_score', 0.5)
                }
            
            return summary
            
        except Exception:
            return {}

    def _get_consensus_trends_summary(self) -> Dict[str, Any]:
        """Get summary of consensus trends"""
        try:
            if not self.consensus_trends:
                return {'status': 'no_trends'}
            
            recent_trends = list(self.consensus_trends)[-5:]
            
            summary = {
                'recent_trend_count': len(recent_trends),
                'current_trend': recent_trends[-1].get('trend_direction', 'unknown') if recent_trends else 'unknown',
                'trend_strength': recent_trends[-1].get('trend_strength', 0.0) if recent_trends else 0.0,
                'avg_consensus_trend': np.mean([t.get('current_consensus', 0.5) for t in recent_trends]),
                'consensus_volatility': np.std([t.get('current_consensus', 0.5) for t in recent_trends]) if len(recent_trends) > 1 else 0.0
            }
            
            return summary
            
        except Exception:
            return {'status': 'analysis_error'}

    def get_observation_components(self) -> np.ndarray:
        """Return consensus features for RL observation"""
        try:
            features = [
                float(self.last_consensus),
                float(self.consensus_quality),
                float(self.directional_consensus),
                float(self.magnitude_consensus),
                float(self.confidence_consensus),
                float(self.temporal_stability),
                float(len(self.consensus_history) / 100),  # History fullness
                float(self.consensus_stats.get('consensus_volatility', 0.0))
            ]
            
            observation = np.array(features, dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid consensus observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Consensus observation generation failed: {error_context}")
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'ConsensusDetector',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'total_computations': self.consensus_stats.get('total_computations', 0),
            'consensus_score': self.last_consensus,
            'consensus_quality': self.consensus_quality,
            'consensus_volatility': self.consensus_stats.get('consensus_volatility', 0.0),
            'high_consensus_count': self.consensus_stats.get('high_consensus_count', 0),
            'low_consensus_count': self.consensus_stats.get('low_consensus_count', 0),
            'member_contributions_tracked': len(self.member_contributions),
            'consensus_history_length': len(self.consensus_history),
            'temporal_stability': self.temporal_stability,
            'session_duration': (datetime.datetime.now() - 
                               datetime.datetime.fromisoformat(self.consensus_stats['session_start'])).total_seconds() / 3600
        }

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Internal method for health metrics (for compatibility)"""
        return self.get_health_metrics()

    def get_consensus_report(self) -> str:
        """Generate comprehensive operator-friendly consensus report"""
        # Consensus level assessment
        if self.last_consensus > 0.8:
            consensus_level = "🟢 HIGH"
        elif self.last_consensus > 0.6:
            consensus_level = "🟡 MODERATE"
        elif self.last_consensus > 0.4:
            consensus_level = "🟠 LOW-MODERATE"