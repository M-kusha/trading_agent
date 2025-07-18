"""
🏛️ Enhanced Strategy Arbiter with SmartInfoBus Integration v3.0
Advanced multi-expert coordination and sophisticated voting mechanisms
"""

import asyncio
import time
import numpy as np
import datetime
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
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
from utils.get_dir import _BASE_GATE, _smart_gate


@module(
    name="StrategyArbiter",
    version="3.0.0",
    category="voting",
    provides=[
        "blended_action", "alpha_weights", "member_weights", "gate_decision", "voting_quality",
        "member_performance", "decision_statistics", "proposal_analysis", "arbiter_recommendations"
    ],
    requires=[
        "market_context", "recent_trades", "current_positions", "member_proposals", "member_confidences",
        "consensus_score", "collusion_score", "horizon_alignment", "volatility_data", "market_regime"
    ],
    description="Advanced multi-expert coordination and sophisticated voting mechanisms",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=120,
    priority=1,
    explainable=True,
    hot_reload=True
)
class StrategyArbiter(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🏛️ PRODUCTION-GRADE Strategy Arbiter v3.0
    
    Advanced multi-expert coordination system with:
    - Sophisticated voting mechanisms with REINFORCE learning
    - Multi-criteria smart gating with adaptive thresholds
    - Comprehensive member performance tracking and weight adaptation
    - SmartInfoBus zero-wiring architecture
    - Real-time decision quality assessment and audit trails
    """

    # Enhanced REINFORCE parameters
    REINFORCE_LR: float = 0.001
    REINFORCE_LAMBDA: float = 0.95
    PRIOR_BLEND: float = 0.30

    def _initialize(self):
        """Initialize advanced strategy arbitration systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced arbitration configuration
        self.members = self.config.get('members', [])
        init_weights = self.config.get('init_weights', [1.0] * len(self.members))
        self.weights = np.asarray(init_weights, dtype=np.float32)
        self.action_dim = self.config.get('action_dim', 4)
        self.adapt_rate = self.config.get('adapt_rate', 0.01)
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.bootstrap_steps = self.config.get('bootstrap_steps', 50)
        self.debug = self.config.get('debug', True)
        
        # Validate configuration
        if len(init_weights) != len(self.members):
            raise ValueError(f"Weight count ({len(init_weights)}) must match member count ({len(self.members)})")
        
        # Sub-modules (will be injected via InfoBus)
        self.consensus = None
        self.collusion = None
        self.horizon_aligner = None
        
        # Enhanced market state tracking
        self.curr_vol = 0.01
        self.market_regime = 'unknown'
        self.market_session = 'unknown'
        self.market_context = {}
        
        # Advanced REINFORCE learning state
        self.last_alpha = None
        self._baseline = 0.0
        self._baseline_beta = 0.98
        self.learning_history = deque(maxlen=100)
        
        # Comprehensive decision tracking
        self._trace = []
        self._log_size = self.config.get('audit_log_size', 100)
        self.decision_history = deque(maxlen=200)
        self.proposal_history = deque(maxlen=100)
        self.gate_decisions = deque(maxlen=150)
        
        # Enhanced gate statistics with intelligence
        self._gate_passes = 0
        self._gate_attempts = 0
        self._step_count = 0
        self.gate_intelligence = {
            'adaptive_threshold': True,
            'criteria_weights': [0.25, 0.20, 0.20, 0.20, 0.15],
            'bootstrap_factor': 0.5,
            'regime_adjustments': {
                'volatile': 0.8,
                'trending': 1.2,
                'ranging': 1.0,
                'noise': 0.7
            }
        }
        
        # Advanced member performance tracking
        self.member_performance = defaultdict(lambda: {
            'proposals_made': 0,
            'successful_proposals': 0,
            'avg_confidence': 0.5,
            'recent_performance': deque(maxlen=20),
            'weight_evolution': deque(maxlen=50),
            'quality_scores': deque(maxlen=30),
            'contribution_score': 0.5,
            'reliability_index': 0.5,
            'specialization_score': 0.5
        })
        
        # Additional tracking attributes
        self._last_proposals: Optional[List[np.ndarray]] = None
        self.voting_quality_metrics = {'overall_quality_score': 0.5}
        self.member_analytics = {'performance_consistency': 0.5}
        self.regime_analytics = {'current_regime_fit': 0.5}
        self.coordination_analytics = {'coordination_effectiveness': 0.5}
        
        # Enhanced voting quality metrics
        self.voting_quality = {
            'avg_consensus': 0.5,
            'collusion_risk': 0.0,
            'gate_effectiveness': 0.5,
            'member_diversity': 0.5,
            'decision_confidence': 0.5,
            'proposal_quality': 0.5,
            'learning_efficiency': 0.5,
            'adaptation_rate': 0.0
        }
        
        # Comprehensive performance statistics
        self.arbiter_stats = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'weight_adaptations': 0,
            'consensus_failures': 0,
            'collusion_detected': 0,
            'gate_pass_rate': 0.0,
            'avg_proposal_quality': 0.5,
            'learning_convergence': 0.0,
            'member_coordination': 0.5,
            'decision_latency': 0.0,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Advanced decision intelligence
        self.decision_intelligence = {
            'quality_threshold': 0.7,
            'adaptation_sensitivity': 0.15,
            'member_learning_rate': 0.05,
            'consensus_weight': 0.3,
            'performance_memory': 0.9,
            'regime_adaptation': True,
            'dynamic_weighting': True
        }
        
        # Market condition adaptation
        self.market_adaptation = {
            'regime_multipliers': {
                'trending': {'confidence_boost': 1.1, 'gate_adjustment': 1.2},
                'volatile': {'confidence_boost': 0.9, 'gate_adjustment': 0.8},
                'ranging': {'confidence_boost': 1.0, 'gate_adjustment': 1.0},
                'noise': {'confidence_boost': 0.8, 'gate_adjustment': 0.7},
                'unknown': {'confidence_boost': 1.0, 'gate_adjustment': 1.0}
            },
            'session_adjustments': {
                'american': 1.0,
                'european': 0.95,
                'asian': 0.9,
                'rollover': 0.6
            }
        }
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🏛️",
            message=f"Strategy Arbiter v{version} initialized",
            members=len(self.members),
            action_dim=self.action_dim,
            bootstrap_steps=self.bootstrap_steps,
            adaptive_learning=True
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="StrategyArbiter",
            log_path="logs/voting/strategy_arbiter.log",
            max_lines=10000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("StrategyArbiter", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Strategy Arbiter v3.0 Initialization Complete:
        
        Advanced Coordination Framework:
        - Multi-expert committee: {len(self.members)} members with {self.action_dim}-dimensional actions
        - REINFORCE learning with adaptive baseline and {self.adapt_rate:.4f} learning rate
        - Smart gating with {len(self.gate_intelligence['criteria_weights'])} evaluation criteria
        - Bootstrap period: {self.bootstrap_steps} steps for learning stabilization
        
        Current Configuration:
        - Weight adaptation: Dynamic REINFORCE with {self._baseline_beta:.3f} baseline momentum
        - Gate intelligence: Adaptive thresholds with regime-aware adjustments
        - Performance tracking: {len(self.member_performance)} member profiles with quality scoring
        - Decision auditing: Comprehensive trace logging with {self._log_size} entry capacity
        
        Arbitration Intelligence Features:
        - Market regime adaptation with session-aware scaling
        - Multi-criteria gate evaluation with weighted scoring
        - Real-time member performance analysis and weight optimization
        - Comprehensive decision quality metrics and learning efficiency tracking
        
        Advanced Capabilities:
        - Collusion detection integration with automatic weight penalties
        - Consensus analysis with quality-weighted voting
        - Horizon alignment with temporal strategy coordination
        - Real-time decision intelligence and recommendation generation
        
        Expected Outcomes:
        - Enhanced decision quality through expert coordination and learning
        - Improved risk management with intelligent gating and member selection
        - Optimal strategy allocation adapted to current market conditions
        - Transparent arbitration decisions with detailed audit trails and performance analytics
        """
        
        self.smart_bus.set('strategy_arbiter_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'members': len(self.members),
                'action_dim': self.action_dim,
                'intelligence_parameters': self.decision_intelligence,
                'gate_parameters': self.gate_intelligence
            }
        }, module='StrategyArbiter', thesis=thesis)

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive strategy arbitration
        
        Returns:
            Dict containing arbitration results, quality metrics, and recommendations
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive market data from SmartInfoBus
            market_data = await self._get_comprehensive_market_data()
            
            # Update market state and regime tracking
            await self._update_market_state_comprehensive(market_data)
            
            # Perform comprehensive member performance analysis
            performance_analysis = await self._analyze_member_performance_comprehensive(market_data)
            
            # Update voting quality metrics
            quality_updates = await self._update_voting_quality_metrics_comprehensive()
            
            # Generate intelligent arbitration recommendations
            recommendations = await self._generate_intelligent_arbitration_recommendations(
                performance_analysis, quality_updates
            )
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_arbitration_thesis(
                performance_analysis, recommendations
            )
            
            # Create comprehensive results
            results = {
                'blended_action': self.last_alpha.tolist() if self.last_alpha is not None else [],
                'alpha_weights': self.last_alpha.tolist() if self.last_alpha is not None else [],
                'member_weights': self.weights.tolist(),
                'gate_decision': self._get_recent_gate_decision(),
                'voting_quality': self.voting_quality.copy(),
                'member_performance': self._get_member_performance_summary(),
                'decision_statistics': self._get_comprehensive_arbiter_stats(),
                'proposal_analysis': self._get_recent_proposal_analysis(),
                'arbiter_recommendations': recommendations,
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('StrategyArbiter', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data using modern SmartInfoBus patterns"""
        try:
            return {
                'market_context': self.smart_bus.get('market_context', 'StrategyArbiter') or {},
                'recent_trades': self.smart_bus.get('recent_trades', 'StrategyArbiter') or [],
                'current_positions': self.smart_bus.get('current_positions', 'StrategyArbiter') or [],
                'member_proposals': self.smart_bus.get('member_proposals', 'StrategyArbiter') or [],
                'member_confidences': self.smart_bus.get('member_confidences', 'StrategyArbiter') or [],
                'consensus_score': self.smart_bus.get('consensus_score', 'StrategyArbiter') or 0.5,
                'collusion_score': self.smart_bus.get('collusion_score', 'StrategyArbiter') or 0.0,
                'horizon_alignment': self.smart_bus.get('horizon_alignment', 'StrategyArbiter') or {},
                'volatility_data': self.smart_bus.get('volatility_data', 'StrategyArbiter') or {},
                'market_regime': self.smart_bus.get('market_regime', 'StrategyArbiter') or 'unknown',
                'session_data': self.smart_bus.get('session_data', 'StrategyArbiter') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "StrategyArbiter")
            self.logger.warning(f"Market data retrieval incomplete: {error_context}")
            return self._get_safe_market_defaults()

    async def _update_market_state_comprehensive(self, market_data: Dict[str, Any]):
        """Update comprehensive market state for decision making"""
        try:
            # Update regime and session
            old_regime = self.market_regime
            self.market_regime = market_data.get('market_regime', 'unknown')
            session_data = market_data.get('session_data', {})
            self.market_session = session_data.get('current_session', 'unknown')
            
            # Update volatility with multiple sources
            volatility_data = market_data.get('volatility_data', {})
            if volatility_data:
                if isinstance(volatility_data, dict):
                    volatilities = list(volatility_data.values())
                    self.curr_vol = max(0.001, np.mean(volatilities))
                else:
                    self.curr_vol = max(0.001, float(volatility_data))
            
            # Update market context
            self.market_context = market_data.get('market_context', {})
            
            # Log significant regime changes
            if old_regime != self.market_regime and old_regime != 'unknown':
                self.logger.info(format_operator_message(
                    icon="[STATS]",
                    message="Market regime transition detected",
                    old_regime=old_regime,
                    new_regime=self.market_regime,
                    volatility=f"{self.curr_vol:.3f}",
                    session=self.market_session,
                    impact="Strategy weights will adapt"
                ))
                
                # Update regime-based adaptations
                await self._apply_regime_adaptations()
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_state_update")
            self.logger.warning(f"Market state update failed: {error_context}")

    async def _apply_regime_adaptations(self):
        """Apply regime-based adaptations to arbitration parameters"""
        try:
            regime_config = self.market_adaptation['regime_multipliers'].get(self.market_regime, {})
            
            # Adjust gate intelligence based on regime
            if self.market_regime == 'volatile':
                self.gate_intelligence['criteria_weights'] = [0.3, 0.25, 0.15, 0.15, 0.15]  # Emphasize signal strength
            elif self.market_regime == 'trending':
                self.gate_intelligence['criteria_weights'] = [0.2, 0.15, 0.25, 0.25, 0.15]  # Emphasize direction and consensus
            elif self.market_regime == 'ranging':
                self.gate_intelligence['criteria_weights'] = [0.25, 0.20, 0.20, 0.20, 0.15]  # Balanced approach
            else:  # noise or unknown
                self.gate_intelligence['criteria_weights'] = [0.35, 0.20, 0.15, 0.15, 0.15]  # Very conservative
            
            # Adjust learning parameters
            confidence_boost = regime_config.get('confidence_boost', 1.0)
            self.decision_intelligence['adaptation_sensitivity'] *= confidence_boost
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_adaptations")

    async def _analyze_member_performance_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comprehensive member performance with advanced metrics"""
        try:
            performance_analysis = {
                'member_updates': {},
                'weight_changes': {},
                'quality_assessments': {},
                'specialization_analysis': {},
                'coordination_effectiveness': 0.0
            }
            
            recent_trades = market_data.get('recent_trades', [])
            member_proposals = market_data.get('member_proposals', [])
            member_confidences = market_data.get('member_confidences', [])
            
            # Calculate recent performance indicators
            recent_success_rate = 0.5
            recent_pnl = []
            if recent_trades:
                recent_pnl = [trade.get('pnl', 0) for trade in recent_trades[-10:]]
                recent_success_rate = sum(1 for pnl in recent_pnl if pnl > 0) / len(recent_pnl)
            
            # Analyze each member's performance
            for i, member in enumerate(self.members):
                if i >= len(self.weights):
                    continue
                
                member_analysis = await self._analyze_individual_member_performance(
                    i, member, member_proposals, member_confidences, recent_success_rate
                )
                performance_analysis['member_updates'][i] = member_analysis
                
                # Update member weights based on performance
                weight_update = await self._calculate_adaptive_weight_update(i, member_analysis)
                if abs(weight_update) > 0.05:
                    performance_analysis['weight_changes'][i] = {
                        'old_weight': self.weights[i],
                        'weight_change': weight_update,
                        'reason': member_analysis.get('primary_factor', 'performance')
                    }
                    self.weights[i] = max(0.01, self.weights[i] + weight_update)
            
            # Renormalize weights
            self.weights = self.weights / (self.weights.sum() + 1e-12)
            
            # Calculate coordination effectiveness
            coordination_score = await self._calculate_coordination_effectiveness(performance_analysis)
            performance_analysis['coordination_effectiveness'] = coordination_score
            
            return performance_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "member_performance_analysis")
            return {'member_updates': {}, 'coordination_effectiveness': 0.5}

    async def _analyze_individual_member_performance(self, member_idx: int, member: Any,
                                                   proposals: List, confidences: List,
                                                   recent_success_rate: float) -> Dict[str, Any]:
        """Analyze individual member performance with detailed metrics"""
        try:
            perf_data = self.member_performance[member_idx]
            
            # Update basic counters
            current_proposals = int(perf_data['proposals_made']) if isinstance(perf_data['proposals_made'], (int, float)) else 0
            perf_data['proposals_made'] = current_proposals + 1
            
            # Calculate current confidence
            current_confidence = 0.5
            if member_idx < len(confidences):
                current_confidence = max(0.1, min(1.0, confidences[member_idx]))
            
            # Update average confidence with exponential smoothing
            old_conf = float(perf_data['avg_confidence']) if isinstance(perf_data['avg_confidence'], (int, float)) else 0.5
            perf_data['avg_confidence'] = old_conf * 0.9 + current_confidence * 0.1
            
            # Analyze proposal quality
            proposal_quality = 0.5
            if member_idx < len(proposals) and len(proposals[member_idx]) > 0:
                proposal = np.array(proposals[member_idx])
                proposal_quality = await self._assess_proposal_quality(proposal, current_confidence)
            
            # Safely append to quality_scores with defensive type checking
            quality_scores = perf_data.get('quality_scores')
            try:
                if isinstance(quality_scores, deque):
                    quality_scores.append(proposal_quality)
                elif isinstance(quality_scores, list):
                    quality_scores.append(proposal_quality)
                else:
                    # Initialize as deque if it's not already a collection
                    perf_data['quality_scores'] = deque([proposal_quality], maxlen=30)
            except (AttributeError, TypeError):
                # Fallback: initialize as new deque
                perf_data['quality_scores'] = deque([proposal_quality], maxlen=30)
            
            # Calculate contribution score
            contribution_score = (proposal_quality + current_confidence + recent_success_rate) / 3.0
            perf_data['contribution_score'] = contribution_score
            
            # Update reliability index
            quality_scores = perf_data['quality_scores']
            if isinstance(quality_scores, deque) and len(quality_scores) >= 5:
                recent_scores = list(quality_scores)[-5:]
                quality_consistency = 1.0 - float(np.std(recent_scores))
                perf_data['reliability_index'] = (perf_data['avg_confidence'] + quality_consistency) / 2.0
            
            # Calculate specialization score (how unique member's contributions are)
            specialization = await self._calculate_member_specialization(member_idx, proposals)
            perf_data['specialization_score'] = specialization
            
            return {
                'contribution_score': contribution_score,
                'proposal_quality': proposal_quality,
                'confidence': current_confidence,
                'reliability': perf_data['reliability_index'],
                'specialization': specialization,
                'primary_factor': 'contribution' if contribution_score > 0.7 else 'reliability'
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "individual_member_analysis")
            return {'contribution_score': 0.5, 'proposal_quality': 0.5, 'confidence': 0.5}

    async def _assess_proposal_quality(self, proposal: np.ndarray, confidence: float) -> float:
        """Assess the quality of a member's proposal"""
        try:
            quality_factors = []
            
            # Factor 1: Signal strength
            signal_strength = np.linalg.norm(proposal)
            normalized_strength = min(1.0, signal_strength / 2.0)  # Normalize to reasonable range
            quality_factors.append(normalized_strength)
            
            # Factor 2: Confidence alignment
            confidence_factor = confidence
            quality_factors.append(confidence_factor)
            
            # Factor 3: Consistency (if we have history)
            if self._last_proposals is not None and len(self._last_proposals) > 0:
                consistency = 1.0 - float(np.linalg.norm(proposal - self._last_proposals[-1])) / 2.0
                consistency = max(0.0, consistency)
                quality_factors.append(consistency)
            
            # Factor 4: Market appropriateness
            regime_appropriateness = 0.5
            if self.market_regime == 'volatile' and signal_strength < 0.5:
                regime_appropriateness = 0.8  # Conservative in volatile markets
            elif self.market_regime == 'trending' and signal_strength > 0.3:
                regime_appropriateness = 0.8  # Decisive in trending markets
            quality_factors.append(regime_appropriateness)
            
            # Weighted average
            weights = [0.3, 0.3, 0.2, 0.2][:len(quality_factors)]
            quality_score = np.average(quality_factors, weights=weights)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "proposal_quality_assessment")
            return 0.5

    async def _calculate_member_specialization(self, member_idx: int, proposals: List) -> float:
        """Calculate how specialized/unique a member's contributions are"""
        try:
            if member_idx >= len(proposals) or len(proposals) < 2:
                return 0.5
            
            member_proposal = np.array(proposals[member_idx])
            other_proposals = [np.array(proposals[i]) for i in range(len(proposals)) if i != member_idx]
            
            if not other_proposals:
                return 0.5
            
            # Calculate uniqueness as average distance to other proposals
            distances = []
            for other_proposal in other_proposals:
                if len(member_proposal) == len(other_proposal):
                    distance = np.linalg.norm(member_proposal - other_proposal)
                    distances.append(distance)
            
            if distances:
                avg_distance = np.mean(distances)
                # Normalize to 0-1 range (higher distance = more specialized)
                specialization = min(1.0, avg_distance / 2.0)
                return float(specialization)
            
            return 0.5
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "member_specialization_calculation")
            return 0.5

    async def _calculate_adaptive_weight_update(self, member_idx: int, analysis: Dict[str, Any]) -> float:
        """Calculate adaptive weight update for a member"""
        try:
            contribution_score = analysis.get('contribution_score', 0.5)
            reliability = analysis.get('reliability', 0.5)
            specialization = analysis.get('specialization', 0.5)
            
            # Base weight change calculation
            performance_factor = (contribution_score + reliability) / 2.0
            
            # Calculate desired weight change
            current_weight = self.weights[member_idx]
            target_weight = performance_factor / len(self.members)  # Ideal equal distribution baseline
            
            # Apply specialization bonus
            if specialization > 0.7:
                target_weight *= 1.2  # Reward unique contributors
            
            # Calculate change with adaptive learning rate
            learning_rate = self.decision_intelligence['member_learning_rate']
            weight_change = (target_weight - current_weight) * learning_rate
            
            # Apply regime-based adjustments
            regime_factor = 1.0
            if self.market_regime == 'volatile' and contribution_score > 0.8:
                regime_factor = 1.3  # Reward good performance in volatile markets
            elif self.market_regime == 'trending' and specialization > 0.6:
                regime_factor = 1.2  # Reward specialists in trending markets
            
            weight_change *= regime_factor
            
            # Limit change magnitude
            max_change = 0.1
            weight_change = np.clip(weight_change, -max_change, max_change)
            
            return float(weight_change)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adaptive_weight_update")
            return 0.0

    async def _calculate_coordination_effectiveness(self, performance_analysis: Dict[str, Any]) -> float:
        """Calculate overall coordination effectiveness"""
        try:
            member_updates = performance_analysis.get('member_updates', {})
            
            if not member_updates:
                return 0.5
            
            # Calculate average contribution scores
            contribution_scores = [update.get('contribution_score', 0.5) for update in member_updates.values()]
            avg_contribution = np.mean(contribution_scores)
            
            # Calculate diversity (standard deviation of contributions)
            contribution_diversity = np.std(contribution_scores) if len(contribution_scores) > 1 else 0.0
            normalized_diversity = min(1.0, contribution_diversity * 2.0)
            
            # Calculate specialization spread
            specialization_scores = [update.get('specialization', 0.5) for update in member_updates.values()]
            avg_specialization = np.mean(specialization_scores)
            
            # Combine factors
            coordination_effectiveness = (
                0.4 * avg_contribution +
                0.3 * normalized_diversity +
                0.3 * avg_specialization
            )
            
            return float(np.clip(coordination_effectiveness, 0.0, 1.0))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "coordination_effectiveness_calculation")
            return 0.5

    async def _update_voting_quality_metrics_comprehensive(self) -> Dict[str, Any]:
        """Update comprehensive voting quality metrics"""
        try:
            quality_updates = {
                'metric_changes': {},
                'trend_analysis': {},
                'quality_drivers': {}
            }
            
            # Update gate effectiveness
            if self._gate_attempts > 0:
                new_gate_effectiveness = self._gate_passes / self._gate_attempts
                old_effectiveness = self.voting_quality['gate_effectiveness']
                self.voting_quality['gate_effectiveness'] = new_gate_effectiveness
                
                if abs(new_gate_effectiveness - old_effectiveness) > 0.1:
                    quality_updates['metric_changes']['gate_effectiveness'] = {
                        'old_value': old_effectiveness,
                        'new_value': new_gate_effectiveness,
                        'trend': 'improving' if new_gate_effectiveness > old_effectiveness else 'declining'
                    }
            
            # Update decision confidence
            if len(self.decision_history) >= 5:
                recent_decisions = list(self.decision_history)[-5:]
                confidence_scores = [d.get('signal_strength', 0.5) for d in recent_decisions]
                avg_confidence = np.mean(confidence_scores)
                self.voting_quality['decision_confidence'] = float(avg_confidence)
            
            # Update member diversity
            if self.member_performance:
                contribution_scores = [p.get('contribution_score', 0.5) for p in self.member_performance.values()]
                if len(contribution_scores) > 1:
                    diversity = np.std(contribution_scores)
                    self.voting_quality['member_diversity'] = float(min(1.0, diversity * 2.0))
            
            # Update learning efficiency
            if len(self.learning_history) >= 10:
                recent_rewards = [entry.get('reward', 0.0) for entry in list(self.learning_history)[-10:]]
                if len(recent_rewards) > 1:
                    learning_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                    learning_efficiency = 0.5 + np.tanh(learning_trend * 10) * 0.5
                    self.voting_quality['learning_efficiency'] = learning_efficiency
            
            # Calculate adaptation rate
            if len(self.decision_history) >= 20:
                weight_changes = []
                for i in range(1, min(20, len(self.decision_history))):
                    if i < len(self.decision_history):
                        # Could calculate weight change magnitude here if stored
                        pass
                
                # Simplified adaptation rate
                adaptation_rate = min(1.0, self.arbiter_stats.get('weight_adaptations', 0) / max(self._step_count, 1))
                self.voting_quality['adaptation_rate'] = adaptation_rate
            
            return quality_updates
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "voting_quality_update")
            return {'metric_changes': {}}

    async def _generate_intelligent_arbitration_recommendations(self, performance_analysis: Dict[str, Any],
                                                              quality_updates: Dict[str, Any]) -> List[str]:
        """Generate intelligent arbitration recommendations"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            coordination_effectiveness = performance_analysis.get('coordination_effectiveness', 0.5)
            if coordination_effectiveness < 0.4:
                recommendations.append("LOW COORDINATION: Consider member rebalancing or additional training")
            elif coordination_effectiveness > 0.8:
                recommendations.append("HIGH COORDINATION: Excellent member performance - maintain current approach")
            
            # Gate effectiveness recommendations
            gate_effectiveness = self.voting_quality.get('gate_effectiveness', 0.5)
            if gate_effectiveness < 0.3:
                recommendations.append("GATE RESTRICTIVE: Consider loosening gate criteria or reviewing thresholds")
            elif gate_effectiveness > 0.8:
                recommendations.append("GATE PERMISSIVE: Consider tightening criteria for better risk management")
            
            # Weight adaptation recommendations
            weight_changes = performance_analysis.get('weight_changes', {})
            if len(weight_changes) > len(self.members) * 0.6:
                recommendations.append("HIGH ADAPTATION: Many members changing weights - ensure stability")
            elif len(weight_changes) == 0 and self._step_count > self.bootstrap_steps:
                recommendations.append("NO ADAPTATION: Weights static - consider increasing learning sensitivity")
            
            # Member diversity recommendations
            member_diversity = self.voting_quality.get('member_diversity', 0.5)
            if member_diversity < 0.3:
                recommendations.append("LOW DIVERSITY: Members too similar - encourage specialization")
            elif member_diversity > 0.8:
                recommendations.append("HIGH DIVERSITY: Good member specialization - balance coordination")
            
            # Learning efficiency recommendations
            learning_efficiency = self.voting_quality.get('learning_efficiency', 0.5)
            if learning_efficiency < 0.3:
                recommendations.append("LEARNING ISSUES: Poor learning convergence - review reward signals")
            elif learning_efficiency > 0.8:
                recommendations.append("LEARNING EFFECTIVE: Strong learning progress - consider advanced strategies")
            
            # Market regime recommendations
            if self.market_regime == 'volatile':
                recommendations.append("VOLATILE REGIME: Emphasize risk management and shorter horizons")
            elif self.market_regime == 'trending':
                recommendations.append("TRENDING REGIME: Favor momentum strategies and position sizing")
            elif self.market_regime == 'noise':
                recommendations.append("NOISE REGIME: Reduce position sizes and increase quality thresholds")
            
            # Bootstrap recommendations
            if self._step_count < self.bootstrap_steps:
                remaining = self.bootstrap_steps - self._step_count
                recommendations.append(f"BOOTSTRAP MODE: {remaining} steps remaining for learning stabilization")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("SYSTEM OPTIMAL: Strategy arbitration operating within normal parameters")
            
            return recommendations[:6]  # Limit to top 6 recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "arbitration_recommendations")
            return [f"Recommendation generation failed: {error_context}"]

    async def _generate_comprehensive_arbitration_thesis(self, performance_analysis: Dict[str, Any],
                                                        recommendations: List[str]) -> str:
        """Generate comprehensive arbitration thesis"""
        try:
            # Core metrics
            coordination_effectiveness = performance_analysis.get('coordination_effectiveness', 0.5)
            gate_effectiveness = self.voting_quality.get('gate_effectiveness', 0.5)
            learning_efficiency = self.voting_quality.get('learning_efficiency', 0.5)
            
            thesis_parts = []
            
            # Executive summary
            coordination_level = "HIGH" if coordination_effectiveness > 0.7 else "MODERATE" if coordination_effectiveness > 0.4 else "LOW"
            thesis_parts.append(
                f"ARBITRATION STATUS: {coordination_level} coordination with {coordination_effectiveness:.1%} effectiveness"
            )
            
            # Member dynamics
            thesis_parts.append(
                f"MEMBER DYNAMICS: {len(self.members)} experts with {self.voting_quality.get('member_diversity', 0.5):.1%} diversity"
            )
            
            # Gate performance
            gate_status = "EFFECTIVE" if gate_effectiveness > 0.6 else "RESTRICTIVE" if gate_effectiveness < 0.4 else "MODERATE"
            thesis_parts.append(f"GATE PERFORMANCE: {gate_status} with {gate_effectiveness:.1%} pass rate")
            
            # Learning progress
            learning_status = "STRONG" if learning_efficiency > 0.7 else "STABLE" if learning_efficiency > 0.4 else "WEAK"
            thesis_parts.append(f"LEARNING STATUS: {learning_status} with {learning_efficiency:.1%} efficiency")
            
            # Market alignment
            thesis_parts.append(f"MARKET ALIGNMENT: {self.market_regime.upper()} regime with {self.curr_vol:.2%} volatility")
            
            # Weight dynamics
            weight_adaptations = self.arbiter_stats.get('weight_adaptations', 0)
            thesis_parts.append(f"WEIGHT DYNAMICS: {weight_adaptations} adaptations across {self._step_count} decisions")
            
            # System performance
            total_decisions = self.arbiter_stats.get('total_decisions', 0)
            successful_decisions = self.arbiter_stats.get('successful_decisions', 0)
            success_rate = (successful_decisions / max(total_decisions, 1)) if total_decisions > 0 else 0.0
            thesis_parts.append(f"SYSTEM PERFORMANCE: {success_rate:.1%} success rate over {total_decisions} decisions")
            
            # Recommendations summary
            priority_recommendations = [rec for rec in recommendations if any(keyword in rec 
                                      for keyword in ['LOW', 'HIGH', 'CRITICAL', 'URGENT'])]
            if priority_recommendations:
                thesis_parts.append(f"ACTION ITEMS: {len(priority_recommendations)} priority recommendations")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "arbitration_thesis_generation")
            return f"Arbitration thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive arbitration results"""
        try:
            # Core arbitration results
            self.smart_bus.set('blended_action', results['blended_action'],
                             module='StrategyArbiter', thesis=thesis)
            
            # Alpha weights
            alpha_thesis = f"Alpha weights: {len(results['alpha_weights'])} member allocations computed"
            self.smart_bus.set('alpha_weights', results['alpha_weights'],
                             module='StrategyArbiter', thesis=alpha_thesis)
            
            # Member weights
            weights_thesis = f"Member weights: {len(results['member_weights'])} experts balanced"
            self.smart_bus.set('member_weights', results['member_weights'],
                             module='StrategyArbiter', thesis=weights_thesis)
            
            # Gate decision
            gate_thesis = f"Gate decision: {results['gate_decision'].get('decision', 'unknown')} with quality criteria"
            self.smart_bus.set('gate_decision', results['gate_decision'],
                             module='StrategyArbiter', thesis=gate_thesis)
            
            # Voting quality
            quality_thesis = f"Voting quality: {len(results['voting_quality'])} metrics tracked"
            self.smart_bus.set('voting_quality', results['voting_quality'],
                             module='StrategyArbiter', thesis=quality_thesis)
            
            # Member performance
            performance_thesis = f"Member performance: {len(results['member_performance'])} profiles analyzed"
            self.smart_bus.set('member_performance', results['member_performance'],
                             module='StrategyArbiter', thesis=performance_thesis)
            
            # Decision statistics
            stats_thesis = f"Decision statistics: {results['decision_statistics']['total_decisions']} decisions processed"
            self.smart_bus.set('decision_statistics', results['decision_statistics'],
                             module='StrategyArbiter', thesis=stats_thesis)
            
            # Proposal analysis
            proposal_thesis = f"Proposal analysis: Recent member proposals evaluated"
            self.smart_bus.set('proposal_analysis', results['proposal_analysis'],
                             module='StrategyArbiter', thesis=proposal_thesis)
            
            # Arbitration recommendations
            rec_thesis = f"Arbitration recommendations: {len(results['arbiter_recommendations'])} insights generated"
            self.smart_bus.set('arbiter_recommendations', results['arbiter_recommendations'],
                             module='StrategyArbiter', thesis=rec_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY COMPATIBILITY AND PUBLIC INTERFACE
    # ═══════════════════════════════════════════════════════════════════

    def propose(self, obs: Any) -> np.ndarray:
        """Legacy proposal interface for backward compatibility"""
        try:
            # Run simplified proposal generation
            return self._simple_proposal_fallback(obs)
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "legacy_proposal")
            self.logger.error(f"Legacy proposal failed: {error_context}")
            return np.zeros(self.action_dim, dtype=np.float32)

    def _simple_proposal_fallback(self, obs: Any) -> np.ndarray:
        """Simple fallback proposal method"""
        try:
            self._step_count += 1
            
            # Collect simplified proposals from members
            proposals = []
            confidences = []
            
            for i, member in enumerate(self.members):
                try:
                    if hasattr(member, 'propose_action'):
                        prop = member.propose_action(obs)
                    elif hasattr(member, 'propose'):
                        prop = member.propose(obs)
                    else:
                        prop = np.zeros(self.action_dim, dtype=np.float32)
                    
                    prop = np.asarray(prop, dtype=np.float32).flatten()
                    if prop.size < self.action_dim:
                        prop = np.pad(prop, (0, self.action_dim - prop.size))
                    elif prop.size > self.action_dim:
                        prop = prop[:self.action_dim]
                    
                    proposals.append(prop)
                    
                    if hasattr(member, 'confidence'):
                        conf = float(member.confidence(obs))
                    else:
                        conf = 0.5
                    
                    confidences.append(max(conf, self.min_confidence))
                    
                except Exception as e:
                    proposals.append(np.zeros(self.action_dim, dtype=np.float32))
                    confidences.append(self.min_confidence)
            
            # Simple weighted blend
            if proposals:
                w_norm = self.weights / (self.weights.sum() + 1e-12)
                c_norm = np.array(confidences) / (np.sum(confidences) + 1e-12)
                alpha = w_norm * c_norm
                alpha = alpha / (alpha.sum() + 1e-12)
                
                self.last_alpha = alpha.copy()
                
                action = np.zeros(self.action_dim, dtype=np.float32)
                for i, (prop, a) in enumerate(zip(proposals, alpha)):
                    action += a * prop
                
                # Simple gate check
                signal_strength = np.abs(action).mean()
                gate_threshold = _smart_gate(float(self.curr_vol), 0) if self._step_count >= self.bootstrap_steps else _BASE_GATE * 0.5
                
                if signal_strength >= gate_threshold:
                    self._gate_passes += 1
                    final_action = action
                else:
                    final_action = action * 0.2
                
                self._gate_attempts += 1
                
                return final_action
            
            return np.zeros(self.action_dim, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Simple proposal fallback failed: {e}")
            return np.zeros(self.action_dim, dtype=np.float32)

    def update_weights(self, reward: float) -> None:
        """Enhanced REINFORCE weight update with comprehensive tracking"""
        if self.last_alpha is None:
            return
        
        try:
            # Update baseline
            self._baseline = self._baseline_beta * self._baseline + (1 - self._baseline_beta) * reward
            
            # Calculate advantage
            advantage = reward - self._baseline
            
            # REINFORCE update with adaptive learning rate
            regime_lr_multiplier = self.market_adaptation['regime_multipliers'].get(
                self.market_regime, {}
            ).get('confidence_boost', 1.0)
            
            effective_lr = self.adapt_rate * regime_lr_multiplier
            grad = advantage * (self.last_alpha - self.weights)
            old_weights = self.weights.copy()
            self.weights += effective_lr * grad
            
            # Ensure positive weights
            self.weights = np.maximum(self.weights, 0.01)
            
            # Normalize
            self.weights = self.weights / self.weights.sum()
            
            # Track learning
            learning_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'reward': reward,
                'advantage': advantage,
                'baseline': self._baseline,
                'weight_change': np.linalg.norm(self.weights - old_weights),
                'regime': self.market_regime
            }
            self.learning_history.append(learning_entry)
            
            # Track significant adaptations
            weight_change = np.linalg.norm(self.weights - old_weights)
            if weight_change > 0.05:
                self.arbiter_stats['weight_adaptations'] += 1
                self.logger.info(format_operator_message(
                    icon="[BALANCE]",
                    message="Significant weight adaptation completed",
                    reward=f"{reward:+.3f}",
                    advantage=f"{advantage:+.3f}",
                    change=f"{weight_change:.3f}",
                    regime=self.market_regime
                ))
            
            # Update success tracking
            if reward > 0:
                self.arbiter_stats['successful_decisions'] += 1
            
            # Update performance metrics
            self._update_performance_metric('weight_adaptation_magnitude', float(weight_change))
            self._update_performance_metric('learning_advantage', advantage)
            self._update_performance_metric('baseline_estimate', self._baseline)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "weight_update")
            self.logger.error(f"Weight update failed: {error_context}")

    def _update_performance_metric(self, metric_name: str, value: float) -> None:
        """Update performance metric for tracking and analysis"""
        try:
            if hasattr(self, 'performance_tracker') and self.performance_tracker:
                self.performance_tracker.record_metric('StrategyArbiter', metric_name, value, True)
            
            # Also store in internal tracking for historical analysis
            if not hasattr(self, '_performance_history'):
                self._performance_history = defaultdict(lambda: deque(maxlen=100))
            
            self._performance_history[metric_name].append({
                'timestamp': datetime.datetime.now().isoformat(),
                'value': value
            })
            
        except Exception as e:
            # Don't let performance tracking failures affect main functionality
            if hasattr(self, 'logger'):
                self.logger.warning(f"Performance metric update failed for {metric_name}: {e}")

    def _get_recent_gate_decision(self) -> Dict[str, Any]:
        """Get most recent gate decision information"""
        try:
            if self.gate_decisions:
                return dict(self.gate_decisions[-1])
            else:
                return {
                    'decision': 'unknown',
                    'criteria_met': 0,
                    'total_criteria': 5,
                    'timestamp': datetime.datetime.now().isoformat()
                }
        except Exception:
            return {'decision': 'unknown'}

    def _get_member_performance_summary(self) -> Dict[str, Any]:
        """Get summary of member performance"""
        try:
            summary = {}
            
            for member_idx, perf_data in self.member_performance.items():
                quality_scores = perf_data.get('quality_scores', [0.5])
                if isinstance(quality_scores, deque) and len(quality_scores) > 0:
                    recent_quality = float(np.mean(list(quality_scores)[-5:]))
                else:
                    recent_quality = 0.5
                    
                summary[f'member_{member_idx}'] = {
                    'contribution_score': perf_data.get('contribution_score', 0.5),
                    'reliability_index': perf_data.get('reliability_index', 0.5),
                    'specialization_score': perf_data.get('specialization_score', 0.5),
                    'avg_confidence': perf_data.get('avg_confidence', 0.5),
                    'proposals_made': perf_data.get('proposals_made', 0),
                    'recent_quality': recent_quality
                }
            
            return summary
            
        except Exception:
            return {}

    def _get_comprehensive_arbiter_stats(self) -> Dict[str, Any]:
        """Get comprehensive arbitration statistics"""
        return {
            **self.arbiter_stats,
            'current_weights': self.weights.tolist(),
            'last_alpha': self.last_alpha.tolist() if self.last_alpha is not None else None,
            'step_count': self._step_count,
            'gate_passes': self._gate_passes,
            'gate_attempts': self._gate_attempts,
            'baseline_estimate': self._baseline,
            'market_regime': self.market_regime,
            'market_session': self.market_session,
            'volatility': self.curr_vol,
            'bootstrap_complete': self._step_count >= self.bootstrap_steps,
            'learning_trend': self._calculate_learning_trend()
        }

    def _calculate_learning_trend(self) -> str:
        """Calculate recent learning trend"""
        try:
            if len(self.learning_history) < 5:
                return 'insufficient_data'
            
            recent_rewards = [entry.get('reward', 0.0) for entry in list(self.learning_history)[-5:]]
            slope = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
            
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'declining'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'

    def _get_recent_proposal_analysis(self) -> Dict[str, Any]:
        """Get analysis of recent proposals"""
        try:
            if not self.proposal_history:
                return {'status': 'no_proposals'}
            
            recent_proposals = list(self.proposal_history)[-5:]
            
            analysis = {
                'proposal_count': len(recent_proposals),
                'avg_quality': np.mean([p.get('quality', 0.5) for p in recent_proposals]),
                'quality_trend': 'stable',
                'diversity_score': 0.5
            }
            
            # Calculate quality trend
            if len(recent_proposals) >= 3:
                qualities = [p.get('quality', 0.5) for p in recent_proposals]
                slope = np.polyfit(range(len(qualities)), qualities, 1)[0]
                
                if slope > 0.05:
                    analysis['quality_trend'] = 'improving'
                elif slope < -0.05:
                    analysis['quality_trend'] = 'declining'
            
            return analysis
            
        except Exception:
            return {'status': 'analysis_error'}

    def get_observation_components(self) -> np.ndarray:
        """Return arbitration features for RL observation"""
        try:
            features = [
                float(self._gate_passes / max(self._gate_attempts, 1)),  # Gate pass rate
                float(self.curr_vol),  # Current volatility
                float(self._baseline),  # Learning baseline
                float(self.voting_quality['avg_consensus']),  # Average consensus
                float(self.voting_quality['decision_confidence']),  # Decision confidence
                float(self.voting_quality['member_diversity']),  # Member diversity
                float(len(self.decision_history) / 200),  # History fullness
                float(self.arbiter_stats['weight_adaptations'] / max(self._step_count, 1))  # Adaptation rate
            ]
            
            # Add normalized weights
            features.extend(self.weights.tolist())
            
            observation = np.array(features, dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid arbitration observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Arbitration observation generation failed: {error_context}")
            # Return safe defaults
            default_features = [0.5, 0.02, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0]
            default_features.extend([1.0 / len(self.members)] * len(self.members))
            return np.array(default_features, dtype=np.float32)

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'StrategyArbiter',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'total_decisions': self.arbiter_stats.get('total_decisions', 0),
            'successful_decisions': self.arbiter_stats.get('successful_decisions', 0),
            'gate_pass_rate': self._gate_passes / max(self._gate_attempts, 1),
            'weight_adaptations': self.arbiter_stats.get('weight_adaptations', 0),
            'learning_baseline': self._baseline,
            'coordination_effectiveness': self.voting_quality.get('member_diversity', 0.5),
            'members_count': len(self.members),
            'step_count': self._step_count,
            'bootstrap_complete': self._step_count >= self.bootstrap_steps,
            'market_regime': self.market_regime,
            'session_duration': (datetime.datetime.now() - 
                               datetime.datetime.fromisoformat(self.arbiter_stats['session_start'])).total_seconds() / 3600
        }

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Internal method for health metrics (for compatibility)"""
        return self.get_health_metrics()

    def get_arbiter_report(self) -> str:
        """Generate comprehensive operator-friendly arbitration report"""
        # Decision quality assessment
        decision_conf = self.voting_quality['decision_confidence']
        if decision_conf > 0.8:
            quality_status = "[OK] EXCELLENT"
        elif decision_conf > 0.6:
            quality_status = "[FAST] GOOD"
        elif decision_conf > 0.4:
            quality_status = "[WARN] FAIR"
        else:
            quality_status = "[ALERT] POOR"
        
        # Gate effectiveness
        gate_rate = self.voting_quality['gate_effectiveness']
        if gate_rate > 0.7:
            gate_status = "[GREEN] EFFECTIVE"
        elif gate_rate > 0.4:
            gate_status = "[YELLOW] MODERATE"
        else:
            gate_status = "[RED] RESTRICTIVE"
        
        # Top performing members
        member_lines = []
        for member_idx, perf_data in list(self.member_performance.items())[:5]:
            if member_idx < len(self.weights):
                weight = self.weights[member_idx]
                contribution = perf_data.get('contribution_score', 0.5)
                reliability = perf_data.get('reliability_index', 0.5)
                
                contribution_float = float(contribution) if isinstance(contribution, (int, float)) else 0.5
                if contribution_float > 0.7:
                    emoji = "🌟"
                elif contribution_float > 0.5:
                    emoji = "[FAST]"
                else:
                    emoji = "[WARN]"
                
                member_lines.append(f"  {emoji} Member {member_idx}: Weight {weight:.3f}, Contrib {contribution:.1%}, Rel {reliability:.1%}")
        
        # Learning status
        learning_efficiency = self.voting_quality.get('learning_efficiency', 0.5)
        if learning_efficiency > 0.7:
            learning_status = "[CHART] Strong"
        elif learning_efficiency > 0.4:
            learning_status = "→ Stable"
        else:
            learning_status = "📉 Weak"
        
        return f"""
🏛️ STRATEGY ARBITER v3.0
═══════════════════════════════════════════════════════════════
[TARGET] Decision Quality: {quality_status} ({decision_conf:.1%})
🚪 Gate Status: {gate_status} ({gate_rate:.1%})
[STATS] Consensus Level: {self.voting_quality['avg_consensus']:.1%}
[CHART] Learning Status: {learning_status} ({learning_efficiency:.1%})

[BALANCE] Committee Overview:
• Total Members: {len(self.members)}
• Action Dimensions: {self.action_dim}
• Current Step: {self._step_count}
• Bootstrap Mode: {'[OK] Active' if self._step_count < self.bootstrap_steps else '[FAIL] Complete'}
• Learning Baseline: {self._baseline:.3f}

[STATS] Market Context:
• Regime: {self.market_regime.title()}
• Session: {self.market_session.title()}
• Volatility: {self.curr_vol:.2%}

[TARGET] Voting Quality Metrics:
• Decision Confidence: {self.voting_quality['decision_confidence']:.1%}
• Average Consensus: {self.voting_quality['avg_consensus']:.1%}
• Member Diversity: {self.voting_quality['member_diversity']:.1%}
• Collusion Risk: {self.voting_quality['collusion_risk']:.1%}
• Gate Effectiveness: {self.voting_quality['gate_effectiveness']:.1%}
• Proposal Quality: {self.voting_quality['proposal_quality']:.1%}
• Learning Efficiency: {self.voting_quality['learning_efficiency']:.1%}

[CHART] Performance Statistics:
• Total Decisions: {self.arbiter_stats['total_decisions']}
• Successful Decisions: {self.arbiter_stats['successful_decisions']}
• Success Rate: {(self.arbiter_stats['successful_decisions'] / max(self.arbiter_stats['total_decisions'], 1)):.1%}
• Weight Adaptations: {self.arbiter_stats['weight_adaptations']}
• Collusion Events: {self.arbiter_stats['collusion_detected']}
• Learning Trend: {self._calculate_learning_trend().title()}

🚪 Gate Intelligence:
• Gate Attempts: {self._gate_attempts}
• Gate Passes: {self._gate_passes}
• Pass Rate: {(self._gate_passes / max(self._gate_attempts, 1)):.1%}
• Criteria Weights: {', '.join([f'{w:.2f}' for w in self.gate_intelligence['criteria_weights']])}
• Adaptive Threshold: {'[OK] Enabled' if self.gate_intelligence['adaptive_threshold'] else '[FAIL] Disabled'}

👥 Top Performing Members:
{chr(10).join(member_lines) if member_lines else "  📭 No member performance data available"}

[TOOL] Configuration:
• Adapt Rate: {self.adapt_rate:.4f}
• Min Confidence: {self.min_confidence:.2f}
• Bootstrap Steps: {self.bootstrap_steps}
• REINFORCE Beta: {self._baseline_beta:.3f}
• Learning Rate: {self.REINFORCE_LR:.4f}

[STATS] Recent Activity:
• Decision History: {len(self.decision_history)} entries
• Learning History: {len(self.learning_history)} entries
• Gate Decisions: {len(self.gate_decisions)} recorded
• Proposal History: {len(self.proposal_history)} entries

[TOOL] System Health:
• Error Count: {self.error_count}/{self.circuit_breaker_threshold}
• Status: {'[ALERT] DISABLED' if self.is_disabled else '[OK] OPERATIONAL'}
• Session Duration: {(datetime.datetime.now() - datetime.datetime.fromisoformat(self.arbiter_stats['session_start'])).total_seconds() / 3600:.1f} hours
        """

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'StrategyArbiter',
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
                'message': 'StrategyArbiter disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring issues'
            })
        
        # Gate effectiveness alerts
        gate_rate = self._gate_passes / max(self._gate_attempts, 1)
        if gate_rate < 0.2:
            alerts.append({
                'severity': 'warning',
                'message': f'Very low gate pass rate: {gate_rate:.1%}',
                'action': 'Review gate criteria and thresholds'
            })
        elif gate_rate > 0.9:
            alerts.append({
                'severity': 'info',
                'message': f'Very high gate pass rate: {gate_rate:.1%}',
                'action': 'Consider tightening gate criteria for better risk management'
            })
        
        # Learning alerts
        learning_efficiency = self.voting_quality.get('learning_efficiency', 0.5)
        if learning_efficiency < 0.3:
            alerts.append({
                'severity': 'warning',
                'message': f'Poor learning efficiency: {learning_efficiency:.1%}',
                'action': 'Review reward signals and learning parameters'
            })
        
        # Member coordination alerts
        member_diversity = self.voting_quality.get('member_diversity', 0.5)
        if member_diversity < 0.2:
            alerts.append({
                'severity': 'info',
                'message': f'Low member diversity: {member_diversity:.1%}',
                'action': 'Encourage member specialization'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart StrategyArbiter module after investigating errors")
        
        if self._step_count < self.bootstrap_steps:
            remaining = self.bootstrap_steps - self._step_count
            recommendations.append(f"Bootstrap mode: {remaining} steps remaining for learning stabilization")
        
        # Gate recommendations
        gate_rate = self._gate_passes / max(self._gate_attempts, 1)
        if gate_rate < 0.3:
            recommendations.append("Gate too restrictive - consider loosening criteria")
        elif gate_rate > 0.8:
            recommendations.append("Gate too permissive - consider tightening criteria")
        
        # Learning recommendations
        if len(self.learning_history) < 10:
            recommendations.append("Insufficient learning history - continue operations to establish patterns")
        
        learning_trend = self._calculate_learning_trend()
        if learning_trend == 'declining':
            recommendations.append("Learning performance declining - review reward signals")
        
        # Weight adaptation recommendations
        adaptation_rate = self.arbiter_stats.get('weight_adaptations', 0) / max(self._step_count, 1)
        if adaptation_rate > 0.2:
            recommendations.append("High weight adaptation frequency - ensure stability")
        elif adaptation_rate < 0.05 and self._step_count > self.bootstrap_steps:
            recommendations.append("Low weight adaptation - consider increasing learning sensitivity")
        
        if not recommendations:
            recommendations.append("StrategyArbiter operating within normal parameters")
        
        return recommendations

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "StrategyArbiter")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="[ALERT]",
                message="Strategy Arbiter disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('StrategyArbiter', 'process_time', processing_time, False)
        
        return {
            'blended_action': [],
            'alpha_weights': [],
            'member_weights': self.weights.tolist(),
            'gate_decision': {'decision': 'error', 'error_context': str(error_context)},
            'voting_quality': {'error': str(error_context)},
            'member_performance': {},
            'decision_statistics': {'error': str(error_context)},
            'proposal_analysis': {'error': str(error_context)},
            'arbiter_recommendations': ["Investigate strategy arbiter errors"],
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_market_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when market data retrieval fails"""
        return {
            'market_context': {}, 'recent_trades': [], 'current_positions': [],
            'member_proposals': [], 'member_confidences': [], 'consensus_score': 0.5,
            'collusion_score': 0.0, 'horizon_alignment': {}, 'volatility_data': {},
            'market_regime': 'unknown', 'session_data': {}
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'blended_action': [],
            'alpha_weights': [],
            'member_weights': self.weights.tolist(),
            'gate_decision': {'decision': 'disabled'},
            'voting_quality': {'status': 'disabled'},
            'member_performance': {},
            'decision_statistics': {'status': 'disabled'},
            'proposal_analysis': {'status': 'disabled'},
            'arbiter_recommendations': ["Restart strategy arbiter system"],
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT AND HOT-RELOAD SUPPORT
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'StrategyArbiter',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'action_dim': self.action_dim,
                'adapt_rate': self.adapt_rate,
                'min_confidence': self.min_confidence,
                'bootstrap_steps': self.bootstrap_steps,
                'debug': self.debug
            },
            'arbitration_state': {
                'weights': self.weights.tolist(),
                'last_alpha': self.last_alpha.tolist() if self.last_alpha is not None else None,
                'baseline': self._baseline,
                'step_count': self._step_count,
                'gate_passes': self._gate_passes,
                'gate_attempts': self._gate_attempts
            },
            'market_state': {
                'curr_vol': self.curr_vol,
                'market_regime': self.market_regime,
                'market_session': self.market_session,
                'market_context': self.market_context.copy()
            },
            'intelligence_state': {
                'decision_intelligence': self.decision_intelligence.copy(),
                'gate_intelligence': self.gate_intelligence.copy(),
                'market_adaptation': self.market_adaptation.copy(),
                'voting_quality': self.voting_quality.copy()
            },
            'performance_state': {
                'member_performance': {k: {
                    'contribution_score': v.get('contribution_score', 0.5),
                    'reliability_index': v.get('reliability_index', 0.5),
                    'specialization_score': v.get('specialization_score', 0.5),
                    'avg_confidence': v.get('avg_confidence', 0.5),
                    'proposals_made': v.get('proposals_made', 0),
                    'quality_scores': []  # Simplified to avoid type issues
                } for k, v in self.member_performance.items()},
                'arbiter_stats': self.arbiter_stats.copy()
            },
            'history_state': {
                'decision_history': list(self.decision_history)[-50:],
                'learning_history': list(self.learning_history)[-30:],
                'gate_decisions': list(self.gate_decisions)[-20:],
                'proposal_history': list(self.proposal_history)[-20:],
                'trace': self._trace[-20:] if self._trace else []
            },
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'performance_metrics': self.get_health_metrics()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.action_dim = int(config.get("action_dim", self.action_dim))
            self.adapt_rate = float(config.get("adapt_rate", self.adapt_rate))
            self.min_confidence = float(config.get("min_confidence", self.min_confidence))
            self.bootstrap_steps = int(config.get("bootstrap_steps", self.bootstrap_steps))
            self.debug = bool(config.get("debug", self.debug))
            
            # Load arbitration state
            arbitration_state = state.get("arbitration_state", {})
            weights = arbitration_state.get("weights", self.weights.tolist())
            self.weights = np.array(weights, dtype=np.float32)
            
            last_alpha = arbitration_state.get("last_alpha")
            if last_alpha:
                self.last_alpha = np.array(last_alpha, dtype=np.float32)
            
            self._baseline = float(arbitration_state.get("baseline", 0.0))
            self._step_count = int(arbitration_state.get("step_count", 0))
            self._gate_passes = int(arbitration_state.get("gate_passes", 0))
            self._gate_attempts = int(arbitration_state.get("gate_attempts", 0))
            
            # Load market state
            market_state = state.get("market_state", {})
            self.curr_vol = float(market_state.get("curr_vol", 0.01))
            self.market_regime = market_state.get("market_regime", "unknown")
            self.market_session = market_state.get("market_session", "unknown")
            self.market_context = market_state.get("market_context", {})
            
            # Load intelligence state
            intelligence_state = state.get("intelligence_state", {})
            self.decision_intelligence.update(intelligence_state.get("decision_intelligence", {}))
            self.gate_intelligence.update(intelligence_state.get("gate_intelligence", {}))
            self.market_adaptation.update(intelligence_state.get("market_adaptation", {}))
            self.voting_quality.update(intelligence_state.get("voting_quality", {}))
            
            # Load performance state
            performance_state = state.get("performance_state", {})
            member_performance_data = performance_state.get("member_performance", {})
            self.member_performance.clear()
            for member_id, perf_data in member_performance_data.items():
                member_idx = int(member_id)
                # Use regular dict instead of defaultdict to avoid type conflicts
                restored_perf = {
                    'proposals_made': perf_data.get('proposals_made', 0),
                    'successful_proposals': perf_data.get('successful_proposals', 0),
                    'avg_confidence': perf_data.get('avg_confidence', 0.5),
                    'contribution_score': perf_data.get('contribution_score', 0.5),
                    'reliability_index': perf_data.get('reliability_index', 0.5),
                    'specialization_score': perf_data.get('specialization_score', 0.5),
                    'recent_performance': deque(maxlen=20),
                    'weight_evolution': deque(maxlen=50),
                    'quality_scores': deque(perf_data.get('quality_scores', []), maxlen=30)
                }
                self.member_performance[member_idx] = restored_perf
            
            self.arbiter_stats.update(performance_state.get("arbiter_stats", {}))
            
            # Load history state
            history_state = state.get("history_state", {})
            
            # Load decision history
            self.decision_history.clear()
            for entry in history_state.get("decision_history", []):
                self.decision_history.append(entry)
            
            # Load learning history
            self.learning_history.clear()
            for entry in history_state.get("learning_history", []):
                self.learning_history.append(entry)
            
            # Load gate decisions
            self.gate_decisions.clear()
            for entry in history_state.get("gate_decisions", []):
                self.gate_decisions.append(entry)
            
            # Load proposal history
            self.proposal_history.clear()
            for entry in history_state.get("proposal_history", []):
                self.proposal_history.append(entry)
            
            # Load trace
            self._trace = history_state.get("trace", [])
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            self.logger.info(format_operator_message(
                icon="[RELOAD]",
                message="Strategy Arbiter state restored",
                members=len(self.members),
                action_dim=self.action_dim,
                step_count=self._step_count,
                total_decisions=self.arbiter_stats.get('total_decisions', 0)
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # RESET AND CLEANUP METHODS
    # ═══════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        
        # Reset learning state
        self._baseline = 0.0
        self.last_alpha = None
        
        # Reset tracking
        self._gate_passes = 0
        self._gate_attempts = 0
        self._step_count = 0
        
        # Reset history
        self._trace.clear()
        self.decision_history.clear()
        self.proposal_history.clear()
        self.gate_decisions.clear()
        self.learning_history.clear()
        
        # Reset member tracking
        self.member_performance.clear()
        
        # Reset quality metrics
        self.voting_quality = {
            'avg_consensus': 0.5,
            'collusion_risk': 0.0,
            'gate_effectiveness': 0.5,
            'member_diversity': 0.5,
            'decision_confidence': 0.5,
            'proposal_quality': 0.5,
            'learning_efficiency': 0.5,
            'adaptation_rate': 0.0
        }
        
        # Reset statistics
        self.arbiter_stats = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'weight_adaptations': 0,
            'consensus_failures': 0,
            'collusion_detected': 0,
            'gate_pass_rate': 0.0,
            'avg_proposal_quality': 0.5,
            'learning_convergence': 0.0,
            'member_coordination': 0.5,
            'decision_latency': 0.0,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Reset error state
        self.error_count = 0
        self.is_disabled = False
        
        self.logger.info(format_operator_message(
            icon="[RELOAD]",
            message="Strategy Arbiter reset completed",
            status="All arbitration state cleared and systems reinitialized"
        ))

    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(format_operator_message(
                    icon="👋",
                    message="Strategy Arbiter shutting down",
                    total_decisions=self.arbiter_stats.get('total_decisions', 0),
                    weight_adaptations=self.arbiter_stats.get('weight_adaptations', 0),
                    gate_pass_rate=f"{(self._gate_passes / max(self._gate_attempts, 1)):.1%}"
                ))
        except Exception:
            pass  # Ignore cleanup errors

    # ═══════════════════════════════════════════════════════════════════
    # BASEMODULE ABSTRACT METHOD IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in strategy arbitration decisions"""
        try:
            # Base confidence from voting quality
            voting_quality = self.voting_quality_metrics.get('overall_quality_score', 0.5)
            
            # Member performance consistency
            member_consistency = self.member_analytics.get('performance_consistency', 0.5)
            
            # Market regime alignment
            regime_alignment = self.regime_analytics.get('current_regime_fit', 0.5)
            
            # Recent gate decision success
            recent_gate_success = self._gate_passes / max(self._gate_attempts, 1)
            
            # Consensus strength
            consensus_strength = 1.0 - (np.std(self.weights) if len(self.weights) > 1 else 0.0)
            
            # Combine factors
            confidence = (
                voting_quality * 0.3 +
                member_consistency * 0.25 +
                regime_alignment * 0.2 +
                recent_gate_success * 0.15 +
                consensus_strength * 0.1
            )
            
            # Ensure valid range
            return float(max(0.1, min(0.95, confidence)))
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.4  # Conservative default

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose arbitration action based on member coordination and market state"""
        try:
            # Get current market data
            market_data = await self._get_comprehensive_market_data()
            
            # Analyze current arbitration state
            voting_quality = self.voting_quality_metrics.get('overall_quality_score', 0.5)
            member_coordination = self.coordination_analytics.get('coordination_effectiveness', 0.5)
            
            # Calculate blended proposal if members available
            if len(self.members) > 0:
                try:
                    # Simple observation for demonstration
                    obs = np.array([0.0, 0.0, 0.0, 0.0])  # Basic observation
                    blended_proposal = self.propose(obs)
                    proposal_strength = np.linalg.norm(blended_proposal)
                except Exception:
                    blended_proposal = np.array([0.0, 0.0, 0.0, 0.0])
                    proposal_strength = 0.0
            else:
                blended_proposal = np.array([0.0, 0.0, 0.0, 0.0])
                proposal_strength = 0.0
            
            # Determine action based on arbitration state
            if voting_quality < 0.3:
                action_type = 'rebalance_members'
                signal_strength = 0.8
                reasoning = f"Poor voting quality ({voting_quality:.3f}) requires member rebalancing"
            elif member_coordination < 0.4:
                action_type = 'improve_coordination'
                signal_strength = 0.6
                reasoning = f"Low member coordination ({member_coordination:.3f}) needs attention"
            elif proposal_strength > 0.7:
                action_type = 'execute_proposal'
                signal_strength = min(proposal_strength, 0.9)
                reasoning = f"Strong blended proposal (strength: {proposal_strength:.3f})"
            else:
                action_type = 'monitor'
                signal_strength = 0.3
                reasoning = f"Normal arbitration state - continue monitoring"
            
            return {
                'action': action_type,
                'signal_strength': signal_strength,
                'reasoning': reasoning,
                'arbitration_metrics': {
                    'voting_quality': voting_quality,
                    'member_coordination': member_coordination,
                    'proposal_strength': proposal_strength,
                    'member_count': len(self.members),
                    'weight_distribution': self.weights.tolist() if hasattr(self.weights, 'tolist') else []
                },
                'blended_proposal': blended_proposal.tolist() if hasattr(blended_proposal, 'tolist') else [],
                'confidence': await self.calculate_confidence({}, **inputs)
            }
            
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                'action': 'abstain',
                'signal_strength': 0.0,
                'reasoning': f'Arbitration error: {str(e)}',
                'confidence': 0.1
            }