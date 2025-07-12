"""
🕵️ Enhanced Collusion Auditor with SmartInfoBus Integration v3.0
Advanced collusion detection and anti-manipulation safeguards for voting committees
"""

import asyncio
import time
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
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
    name="CollusionAuditor",
    version="3.0.0",
    category="voting",
    provides=[
        "collusion_score", "suspicious_pairs", "member_independence_scores", "collusion_alerts",
        "behavioral_profiles", "coordination_events", "detection_statistics", "audit_recommendations"
    ],
    requires=[
        "votes", "voting_summary", "strategy_arbiter_weights", "raw_proposals", "member_confidences",
        "consensus_direction", "agreement_score", "market_context", "recent_trades"
    ],
    description="Advanced collusion detection and anti-manipulation safeguards for voting committees",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=100,
    priority=2,
    explainable=True,
    hot_reload=True
)
class CollusionAuditor(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🕵️ PRODUCTION-GRADE Collusion Auditor v3.0
    
    Advanced collusion detection system with:
    - Multi-dimensional similarity analysis for detecting coordination patterns
    - Adaptive threshold management based on market conditions
    - Comprehensive behavioral profiling and temporal pattern analysis
    - SmartInfoBus zero-wiring architecture
    - Real-time threat assessment and alert management
    """

    def _initialize(self):
        """Initialize advanced collusion detection systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced detection configuration
        self.n_members = self.config.get('n_members', 5)
        self.window = self.config.get('window', 10)
        self.base_threshold = self.config.get('threshold', 0.9)
        self.current_threshold = self.base_threshold
        self.adaptive_threshold = self.config.get('adaptive_threshold', True)
        self.similarity_methods = self.config.get('similarity_methods', ['cosine', 'correlation', 'euclidean'])
        self.debug = self.config.get('debug', False)
        
        # Initialize comprehensive detection methods
        self.detection_methods = self._initialize_detection_methods()
        
        # Core collusion detection state
        self.vote_history = deque(maxlen=self.window * 2)
        self.collusion_score = 0.0
        self.suspicious_pairs = set()
        self.collusion_history = deque(maxlen=100)
        
        # Advanced behavioral analysis
        self.pair_agreement_history = defaultdict(lambda: deque(maxlen=self.window))
        self.member_behavior_profiles = defaultdict(lambda: {
            'avg_similarity': 0.0,
            'volatility': 0.0,
            'consistency_score': 0.0,
            'independence_score': 1.0,
            'coordination_frequency': 0.0,
            'anomaly_score': 0.0
        })
        
        # Temporal and coordination analysis
        self.temporal_patterns = defaultdict(list)
        self.coordination_events = deque(maxlen=50)
        self.alert_patterns = defaultdict(list)
        
        # Enhanced statistics and analytics
        self.detection_stats = {
            'total_checks': 0,
            'alerts_raised': 0,
            'false_positive_rate': 0.0,
            'confirmed_collusion_events': 0,
            'avg_pair_similarity': 0.0,
            'member_independence_scores': {},
            'detection_accuracy': 0.95,
            'alert_frequency': 0.0,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Adaptive intelligence parameters
        self.detection_intelligence = {
            'threshold_bounds': (0.7, 0.98),
            'adaptation_rate': 0.15,
            'sensitivity_target': 0.85,
            'false_positive_threshold': 0.1,
            'confirmation_threshold': 0.8,
            'temporal_sensitivity': 0.6,
            'behavioral_memory': 0.9
        }
        
        # Market condition adaptation
        self.market_adaptation = {
            'regime_multipliers': {
                'trending': 1.1,    # Stricter during trending (coordination easier)
                'ranging': 0.9,     # More lenient during ranging
                'volatile': 0.85,   # More lenient during high volatility
                'breakout': 1.2,    # Very strict during breakouts
                'reversal': 1.15,   # Strict during reversals
                'unknown': 1.0
            },
            'agreement_adjustments': {
                'high_agreement': 1.2,    # Stricter when natural agreement is high
                'medium_agreement': 1.0,
                'low_agreement': 0.8      # More lenient when natural disagreement
            }
        }
        
        # Alert management system
        self.alert_system = {
            'cooldown_period': 10,  # Steps between alerts for same pair
            'escalation_threshold': 3,  # Alerts before escalation
            'severity_levels': ['info', 'warning', 'critical'],
            'auto_investigation': True,
            'last_alerts': defaultdict(int),
            'alert_history': deque(maxlen=100)
        }
        
        # Quality and performance metrics
        self.quality_metrics = {
            'detection_precision': 0.0,
            'detection_recall': 0.0,
            'behavioral_accuracy': 0.0,
            'temporal_consistency': 0.0,
            'overall_effectiveness': 0.0
        }
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🕵️",
            message=f"Collusion Auditor v{version} initialized",
            members=self.n_members,
            window=self.window,
            base_threshold=f"{self.base_threshold:.3f}",
            methods=len(self.similarity_methods),
            adaptive=self.adaptive_threshold
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="CollusionAuditor",
            log_path="logs/voting/collusion_auditor.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("CollusionAuditor", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _initialize_detection_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive detection method definitions"""
        return {
            'cosine_similarity': {
                'description': 'Cosine similarity analysis for detecting aligned voting patterns',
                'parameters': {'normalization': True, 'weight_threshold': 0.1},
                'use_cases': ['general_coordination', 'direction_alignment'],
                'effectiveness_threshold': 0.7,
                'computational_cost': 'low'
            },
            'correlation_analysis': {
                'description': 'Statistical correlation analysis for temporal coordination patterns',
                'parameters': {'min_samples': 3, 'confidence_level': 0.95},
                'use_cases': ['temporal_coordination', 'sequential_patterns'],
                'effectiveness_threshold': 0.8,
                'computational_cost': 'medium'
            },
            'euclidean_distance': {
                'description': 'Distance-based analysis for detecting similar magnitude responses',
                'parameters': {'distance_normalization': True, 'outlier_detection': True},
                'use_cases': ['magnitude_coordination', 'precision_collusion'],
                'effectiveness_threshold': 0.6,
                'computational_cost': 'low'
            },
            'behavioral_profiling': {
                'description': 'Long-term behavioral pattern analysis for identifying systematic coordination',
                'parameters': {'profile_memory': 50, 'anomaly_sensitivity': 0.3},
                'use_cases': ['systematic_collusion', 'long_term_coordination'],
                'effectiveness_threshold': 0.75,
                'computational_cost': 'high'
            },
            'temporal_clustering': {
                'description': 'Time-based clustering analysis for detecting coordinated timing patterns',
                'parameters': {'time_window': 5, 'clustering_threshold': 0.8},
                'use_cases': ['timing_coordination', 'synchronized_responses'],
                'effectiveness_threshold': 0.7,
                'computational_cost': 'medium'
            }
        }

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Collusion Auditor v3.0 Initialization Complete:
        
        Advanced Detection Framework:
        - Multi-member committee surveillance: {self.n_members} members with {self.window}-step analysis window
        - Adaptive detection algorithms with intelligent threshold adjustment ({self.base_threshold:.3f} base)
        - Comprehensive behavioral profiling and temporal pattern analysis capabilities
        - Market-aware detection adaptation based on regime and agreement conditions
        
        Current Configuration:
        - Detection methods: {len(self.detection_methods)} distinct approaches available
        - Adaptive threshold: {'enabled' if self.adaptive_threshold else 'disabled'} with bounds [{self.detection_intelligence['threshold_bounds'][0]:.3f}, {self.detection_intelligence['threshold_bounds'][1]:.3f}]
        - Similarity methods: {', '.join(self.similarity_methods)} for multi-dimensional analysis
        - Alert management: {self.alert_system['cooldown_period']}-step cooldown with {len(self.alert_system['severity_levels'])} severity levels
        
        Detection Intelligence Features:
        - Market regime adaptation with agreement-aware scaling
        - Multi-method analysis with effectiveness-based weighting
        - Real-time behavioral profiling and anomaly detection
        - Comprehensive quality metrics and performance analytics
        
        Advanced Capabilities:
        - Temporal clustering analysis for coordinated timing detection
        - Behavioral profiling for systematic collusion identification
        - Adaptive alert management with escalation procedures
        - Real-time effectiveness monitoring and threshold optimization
        
        Expected Outcomes:
        - Enhanced voting integrity through comprehensive coordination detection
        - Improved threat identification with behavioral pattern analysis
        - Optimal detection sensitivity adapted to current market conditions
        - Transparent audit decisions with detailed forensic analysis and recommendations
        """
        
        self.smart_bus.set('collusion_auditor_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'members': self.n_members,
                'window': self.window,
                'detection_methods': list(self.detection_methods.keys()),
                'intelligence_parameters': self.detection_intelligence
            }
        }, module='CollusionAuditor', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive collusion analysis
        
        Returns:
            Dict containing detection results, behavioral analysis, and recommendations
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive voting data from SmartInfoBus
            voting_data = await self._get_comprehensive_voting_data()
            
            # Update detection parameters based on market conditions
            await self._update_detection_parameters_comprehensive(voting_data)
            
            # Perform comprehensive collusion analysis
            collusion_analysis = await self._perform_comprehensive_collusion_analysis(voting_data)
            
            # Update behavioral profiles
            behavioral_updates = await self._update_behavioral_profiles_comprehensive(voting_data)
            
            # Analyze temporal coordination patterns
            temporal_analysis = await self._analyze_temporal_coordination_patterns(voting_data)
            
            # Calculate comprehensive quality metrics
            quality_analysis = await self._calculate_comprehensive_quality_metrics()
            
            # Generate detection recommendations
            recommendations = await self._generate_intelligent_detection_recommendations(
                collusion_analysis, behavioral_updates, temporal_analysis
            )
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_detection_thesis(
                collusion_analysis, quality_analysis, recommendations
            )
            
            # Create comprehensive results
            results = {
                'collusion_score': self.collusion_score,
                'suspicious_pairs': list(self.suspicious_pairs),
                'member_independence_scores': self.get_member_independence_scores(),
                'collusion_alerts': self._get_recent_collusion_alerts(),
                'behavioral_profiles': self._get_behavioral_profiles_summary(),
                'coordination_events': list(self.coordination_events)[-10:],
                'detection_statistics': self._get_comprehensive_detection_stats(),
                'audit_recommendations': recommendations,
                'quality_metrics': quality_analysis,
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('CollusionAuditor', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_voting_data(self) -> Dict[str, Any]:
        """Get comprehensive voting data using modern SmartInfoBus patterns"""
        try:
            return {
                'votes': self.smart_bus.get('votes', 'CollusionAuditor') or [],
                'voting_summary': self.smart_bus.get('voting_summary', 'CollusionAuditor') or {},
                'strategy_arbiter_weights': self.smart_bus.get('strategy_arbiter_weights', 'CollusionAuditor') or [],
                'raw_proposals': self.smart_bus.get('raw_proposals', 'CollusionAuditor') or [],
                'member_confidences': self.smart_bus.get('member_confidences', 'CollusionAuditor') or [],
                'consensus_direction': self.smart_bus.get('consensus_direction', 'CollusionAuditor') or 'neutral',
                'agreement_score': self.smart_bus.get('agreement_score', 'CollusionAuditor') or 0.5,
                'market_context': self.smart_bus.get('market_context', 'CollusionAuditor') or {},
                'recent_trades': self.smart_bus.get('recent_trades', 'CollusionAuditor') or [],
                'market_regime': self.smart_bus.get('market_regime', 'CollusionAuditor') or 'unknown',
                'volatility_data': self.smart_bus.get('volatility_data', 'CollusionAuditor') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "CollusionAuditor")
            self.logger.warning(f"Voting data retrieval incomplete: {error_context}")
            return self._get_safe_voting_defaults()

    async def _update_detection_parameters_comprehensive(self, voting_data: Dict[str, Any]):
        """Update comprehensive detection parameters with intelligent adaptation"""
        try:
            if not self.adaptive_threshold:
                return
            
            # Extract market intelligence
            market_context = voting_data.get('market_context', {})
            regime = voting_data.get('market_regime', 'unknown')
            agreement_score = voting_data.get('agreement_score', 0.5)
            recent_trades = voting_data.get('recent_trades', [])
            
            # Calculate market uncertainty factor
            market_uncertainty = self._calculate_market_uncertainty_factor(voting_data)
            
            # Calculate base adaptation multiplier
            base_multiplier = 1.0
            
            # Apply regime-based adaptation
            regime_multiplier = self.market_adaptation['regime_multipliers'].get(regime, 1.0)
            base_multiplier *= regime_multiplier
            
            # Apply agreement-based adaptation
            if agreement_score > 0.8:
                agreement_category = 'high_agreement'
            elif agreement_score > 0.4:
                agreement_category = 'medium_agreement'
            else:
                agreement_category = 'low_agreement'
            
            agreement_multiplier = self.market_adaptation['agreement_adjustments'].get(agreement_category, 1.0)
            base_multiplier *= agreement_multiplier
            
            # Apply performance-based adaptation
            if recent_trades:
                recent_performance = self._calculate_recent_performance(recent_trades)
                if abs(recent_performance) > 0.05:  # High performance volatility
                    base_multiplier *= 1.1  # Stricter detection during high performance volatility
            
            # Calculate target threshold with intelligent bounds
            target_threshold = self.base_threshold * base_multiplier
            target_threshold = np.clip(
                target_threshold,
                self.detection_intelligence['threshold_bounds'][0],
                self.detection_intelligence['threshold_bounds'][1]
            )
            
            # Apply momentum-based smooth adaptation
            adaptation_rate = self.detection_intelligence['adaptation_rate']
            old_threshold = self.current_threshold
            self.current_threshold = (
                old_threshold * (1 - adaptation_rate) +
                target_threshold * adaptation_rate
            )
            
            # Track significant adaptations
            threshold_change = abs(self.current_threshold - old_threshold)
            if threshold_change > 0.01:  # Threshold for significant change
                adaptation_record = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'old_threshold': old_threshold,
                    'new_threshold': self.current_threshold,
                    'regime': regime,
                    'agreement_score': agreement_score,
                    'market_uncertainty': market_uncertainty,
                    'base_multiplier': base_multiplier
                }
                
                # Log significant adaptation
                self.logger.info(format_operator_message(
                    icon="🎯",
                    message="Detection threshold adapted",
                    old_threshold=f"{old_threshold:.4f}",
                    new_threshold=f"{self.current_threshold:.4f}",
                    regime=regime,
                    agreement=f"{agreement_score:.1%}",
                    uncertainty=f"{market_uncertainty:.3f}"
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "detection_parameters_update")
            self.logger.warning(f"Detection parameter update failed: {error_context}")

    def _calculate_market_uncertainty_factor(self, voting_data: Dict[str, Any]) -> float:
        """Calculate comprehensive market uncertainty factor"""
        try:
            uncertainty_components = []
            
            # Agreement uncertainty (inverse relationship)
            agreement_score = voting_data.get('agreement_score', 0.5)
            agreement_uncertainty = 1.0 - agreement_score
            uncertainty_components.append(agreement_uncertainty)
            
            # Regime uncertainty
            regime = voting_data.get('market_regime', 'unknown')
            regime_uncertainty = 0.8 if regime == 'unknown' else 0.2
            uncertainty_components.append(regime_uncertainty)
            
            # Volatility uncertainty
            volatility_data = voting_data.get('volatility_data', {})
            volatility_level = volatility_data.get('level', 'medium')
            volatility_uncertainty = {
                'very_low': 0.1, 'low': 0.3, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0
            }.get(volatility_level, 0.5)
            uncertainty_components.append(volatility_uncertainty)
            
            # Performance uncertainty
            recent_trades = voting_data.get('recent_trades', [])
            if len(recent_trades) >= 3:
                recent_pnls = [t.get('pnl', 0) for t in recent_trades[-5:]]
                if recent_pnls:
                    pnl_volatility = np.std(recent_pnls) / (abs(np.mean(recent_pnls)) + 0.01)
                    performance_uncertainty = min(1.0, pnl_volatility)
                    uncertainty_components.append(performance_uncertainty)
            
            # Weighted combination
            if uncertainty_components:
                weights = [0.4, 0.2, 0.3, 0.1][:len(uncertainty_components)]
                weights = np.array(weights) / np.sum(weights)  # Normalize
                total_uncertainty = np.average(uncertainty_components, weights=weights)
            else:
                total_uncertainty = 0.5
            
            return np.clip(total_uncertainty, 0.0, 1.0)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_uncertainty_calculation")
            return 0.5

    def _calculate_recent_performance(self, recent_trades: List[Dict]) -> float:
        """Calculate recent trading performance"""
        try:
            if not recent_trades:
                return 0.0
            
            recent_pnl = [trade.get('pnl', 0) for trade in recent_trades[-10:]]
            return float(np.mean(recent_pnl))
        except Exception:
            return 0.0

    async def _perform_comprehensive_collusion_analysis(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive collusion analysis with multiple detection methods"""
        try:
            self.detection_stats['total_checks'] += 1
            
            # Extract voting actions
            actions = self._extract_voting_actions(voting_data)
            if len(actions) < 2:
                return {'collusion_score': 0.0, 'suspicious_pairs': [], 'analysis_status': 'insufficient_data'}
            
            # Add to history
            timestamp = datetime.datetime.now().isoformat()
            vote_entry = {
                'timestamp': timestamp,
                'actions': [action.copy() for action in actions],
                'n_members': len(actions),
                'agreement_score': voting_data.get('agreement_score', 0.5),
                'market_regime': voting_data.get('market_regime', 'unknown')
            }
            self.vote_history.append(vote_entry)
            
            # Need sufficient history for meaningful analysis
            if len(self.vote_history) < 3:
                return {'collusion_score': 0.0, 'suspicious_pairs': [], 'analysis_status': 'building_history'}
            
            # Perform multi-method similarity analysis
            similarity_analysis = await self._calculate_comprehensive_similarities(actions)
            
            # Update pair agreement history
            await self._update_pair_agreements_comprehensive(similarity_analysis)
            
            # Detect suspicious coordination patterns
            coordination_analysis = await self._detect_coordination_patterns(similarity_analysis)
            
            # Update suspicious pairs and generate alerts
            alert_updates = await self._update_suspicious_pairs_and_alerts(coordination_analysis)
            
            # Calculate overall collusion score
            self.collusion_score = await self._calculate_overall_collusion_score(coordination_analysis)
            
            # Record comprehensive collusion event
            collusion_event = {
                'timestamp': timestamp,
                'collusion_score': self.collusion_score,
                'suspicious_pairs': list(self.suspicious_pairs),
                'similarity_analysis': {str(k): v for k, v in similarity_analysis.items()},
                'threshold_used': self.current_threshold,
                'coordination_analysis': coordination_analysis,
                'alert_updates': alert_updates
            }
            self.collusion_history.append(collusion_event)
            
            # Update comprehensive statistics
            await self._update_detection_statistics_comprehensive(similarity_analysis, coordination_analysis)
            
            return {
                'collusion_score': self.collusion_score,
                'suspicious_pairs': list(self.suspicious_pairs),
                'similarity_analysis': similarity_analysis,
                'coordination_analysis': coordination_analysis,
                'alert_updates': alert_updates,
                'analysis_status': 'complete'
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "collusion_analysis")
            self.logger.error(f"Collusion analysis failed: {error_context}")
            return {'collusion_score': 0.0, 'suspicious_pairs': [], 'analysis_status': 'error', 'error': str(error_context)}

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
            
            # Fallback to votes
            votes = voting_data.get('votes', [])
            if votes and len(votes) >= 2:
                actions = []
                for vote in votes[:self.n_members]:
                    if isinstance(vote, (int, float)):
                        actions.append(np.array([float(vote)], dtype=np.float32))
                    elif isinstance(vote, (list, np.ndarray)):
                        actions.append(np.array(vote, dtype=np.float32))
                if len(actions) >= 2:
                    return actions
            
            # Fallback to strategy weights
            weights = voting_data.get('strategy_arbiter_weights', [])
            if weights and len(weights) >= 2:
                return [np.array(w, dtype=np.float32) for w in weights[:self.n_members]]
            
            return []
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "voting_actions_extraction")
            return []

    async def _calculate_comprehensive_similarities(self, actions: List[np.ndarray]) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Calculate comprehensive similarities using multiple advanced methods"""
        try:
            similarities = {}
            
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    pair = (i, j)
                    v1, v2 = actions[i], actions[j]
                    
                    pair_similarities = {}
                    
                    # Only calculate if both vectors have magnitude
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        
                        # Cosine similarity
                        if 'cosine' in self.similarity_methods:
                            cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            pair_similarities['cosine'] = float(np.clip(cosine_sim, -1, 1))
                        
                        # Correlation (for multi-dimensional vectors)
                        if 'correlation' in self.similarity_methods and len(v1) > 1:
                            if np.std(v1) > 1e-6 and np.std(v2) > 1e-6:
                                correlation = np.corrcoef(v1, v2)[0, 1]
                                if not np.isnan(correlation):
                                    pair_similarities['correlation'] = float(correlation)
                        
                        # Inverse Euclidean distance (normalized)
                        if 'euclidean' in self.similarity_methods:
                            distance = np.linalg.norm(v1 - v2)
                            max_distance = np.linalg.norm(v1) + np.linalg.norm(v2)
                            if max_distance > 1e-6:
                                euclidean_sim = 1.0 - (distance / max_distance)
                                pair_similarities['euclidean'] = float(euclidean_sim)
                        
                        # Angular similarity (direction-focused)
                        if len(v1) > 1 and len(v2) > 1:
                            # Angle between vectors
                            dot_product = np.dot(v1, v2)
                            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                            if norms > 1e-6:
                                cos_angle = np.clip(dot_product / norms, -1, 1)
                                angle_similarity = (cos_angle + 1) / 2  # Normalize to [0, 1]
                                pair_similarities['angular'] = float(angle_similarity)
                    
                    similarities[pair] = pair_similarities
            
            return similarities
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "similarity_calculation")
            return {}

    async def _update_pair_agreements_comprehensive(self, similarity_analysis: Dict[Tuple[int, int], Dict[str, float]]):
        """Update comprehensive pair agreement tracking with advanced analytics"""
        try:
            for pair, similarities in similarity_analysis.items():
                if similarities:
                    # Calculate weighted average of similarity measures
                    valid_similarities = list(similarities.values())
                    if valid_similarities:
                        # Weight different similarity measures
                        weights = {
                            'cosine': 0.4,
                            'correlation': 0.3,
                            'euclidean': 0.2,
                            'angular': 0.1
                        }
                        
                        weighted_similarity = 0.0
                        total_weight = 0.0
                        
                        for method, similarity in similarities.items():
                            weight = weights.get(method, 0.25)
                            weighted_similarity += similarity * weight
                            total_weight += weight
                        
                        if total_weight > 0:
                            final_similarity = weighted_similarity / total_weight
                            self.pair_agreement_history[pair].append(final_similarity)
                        
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "pair_agreements_update")

    async def _detect_coordination_patterns(self, similarity_analysis: Dict[Tuple[int, int], Dict[str, float]]) -> Dict[str, Any]:
        """Detect comprehensive coordination patterns using advanced analysis"""
        try:
            coordination_analysis = {
                'coordinated_pairs': [],
                'coordination_strength': {},
                'temporal_patterns': {},
                'behavioral_anomalies': {},
                'network_effects': {}
            }
            
            for pair, similarities in similarity_analysis.items():
                if not similarities:
                    continue
                
                # Get historical agreement for this pair
                pair_history = self.pair_agreement_history.get(pair, deque())
                if len(pair_history) < 3:
                    continue
                
                # Calculate various coordination metrics
                historical_avg = np.mean(list(pair_history))
                recent_avg = np.mean(list(pair_history)[-3:]) if len(pair_history) >= 3 else historical_avg
                trend = recent_avg - historical_avg
                consistency = 1.0 - (np.std(list(pair_history)) / max(historical_avg, 0.1))
                
                # Detect coordination based on multiple criteria
                is_coordinated = (
                    historical_avg > self.current_threshold and
                    consistency > 0.7 and
                    len(pair_history) >= 5
                )
                
                if is_coordinated:
                    coordination_analysis['coordinated_pairs'].append(pair)
                    coordination_analysis['coordination_strength'][pair] = {
                        'historical_avg': historical_avg,
                        'recent_avg': recent_avg,
                        'trend': trend,
                        'consistency': consistency,
                        'coordination_score': historical_avg * consistency
                    }
                
                # Analyze temporal patterns
                if len(pair_history) >= 5:
                    temporal_pattern = self._analyze_temporal_pattern(list(pair_history))
                    coordination_analysis['temporal_patterns'][pair] = temporal_pattern
                
                # Detect behavioral anomalies
                if len(pair_history) >= 10:
                    anomaly_score = self._calculate_anomaly_score(list(pair_history))
                    if anomaly_score > 0.7:
                        coordination_analysis['behavioral_anomalies'][pair] = anomaly_score
            
            # Analyze network effects (multi-member coordination)
            coordination_analysis['network_effects'] = await self._analyze_network_coordination_effects(
                coordination_analysis['coordinated_pairs']
            )
            
            return coordination_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "coordination_pattern_detection")
            return {'coordinated_pairs': [], 'coordination_strength': {}, 'analysis_error': str(error_context)}

    def _analyze_temporal_pattern(self, history: List[float]) -> Dict[str, Any]:
        """Analyze temporal patterns in coordination history"""
        try:
            if len(history) < 3:
                return {'pattern': 'insufficient_data'}
            
            # Calculate trend
            x = np.arange(len(history))
            slope = np.polyfit(x, history, 1)[0]
            
            # Calculate volatility
            volatility = np.std(history)
            
            # Detect pattern type
            if abs(slope) < 0.01 and volatility < 0.1:
                pattern = 'stable_high' if np.mean(history) > 0.8 else 'stable_low'
            elif slope > 0.05:
                pattern = 'increasing'
            elif slope < -0.05:
                pattern = 'decreasing'
            elif volatility > 0.3:
                pattern = 'volatile'
            else:
                pattern = 'moderate'
            
            return {
                'pattern': pattern,
                'slope': slope,
                'volatility': volatility,
                'mean_value': np.mean(history),
                'recent_trend': 'up' if len(history) >= 3 and history[-1] > history[-3] else 'down'
            }
            
        except Exception:
            return {'pattern': 'unknown'}

    def _calculate_anomaly_score(self, history: List[float]) -> float:
        """Calculate behavioral anomaly score"""
        try:
            if len(history) < 5:
                return 0.0
            
            # Calculate z-scores for recent values
            mean_val = np.mean(history[:-3])  # Exclude recent values from baseline
            std_val = np.std(history[:-3])
            
            if std_val < 1e-6:
                return 0.0
            
            recent_values = history[-3:]
            z_scores = [(val - mean_val) / std_val for val in recent_values]
            
            # Anomaly score based on how many standard deviations from normal
            max_z_score = max(abs(z) for z in z_scores)
            anomaly_score = min(1.0, max_z_score / 3.0)  # Normalize to [0, 1]
            
            return float(anomaly_score)
            
        except Exception:
            return 0.0

    async def _analyze_network_coordination_effects(self, coordinated_pairs: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze network-level coordination effects"""
        try:
            if not coordinated_pairs:
                return {'network_score': 0.0, 'clusters': [], 'coordination_density': 0.0}
            
            # Build coordination network
            coordination_network = defaultdict(set)
            for pair in coordinated_pairs:
                i, j = pair
                coordination_network[i].add(j)
                coordination_network[j].add(i)
            
            # Find coordination clusters
            clusters = []
            visited = set()
            
            for member in coordination_network:
                if member not in visited:
                    cluster = self._find_coordination_cluster(member, coordination_network, visited)
                    if len(cluster) > 2:  # Only clusters with 3+ members are significant
                        clusters.append(cluster)
            
            # Calculate network metrics
            total_possible_pairs = self.n_members * (self.n_members - 1) / 2
            coordination_density = len(coordinated_pairs) / max(total_possible_pairs, 1)
            
            # Network coordination score
            network_score = coordination_density
            if clusters:
                # Bonus for large clusters
                largest_cluster_size = max(len(cluster) for cluster in clusters)
                cluster_bonus = (largest_cluster_size - 2) / max(self.n_members - 2, 1)
                network_score += cluster_bonus * 0.5
            
            network_score = min(1.0, network_score)
            
            return {
                'network_score': network_score,
                'clusters': clusters,
                'coordination_density': coordination_density,
                'largest_cluster_size': max(len(cluster) for cluster in clusters) if clusters else 0,
                'total_coordinated_pairs': len(coordinated_pairs)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "network_coordination_analysis")
            return {'network_score': 0.0, 'clusters': [], 'coordination_density': 0.0}

    def _find_coordination_cluster(self, start_member: int, network: Dict[int, Set[int]], visited: Set[int]) -> List[int]:
        """Find coordination cluster using DFS"""
        cluster = []
        stack = [start_member]
        
        while stack:
            member = stack.pop()
            if member not in visited:
                visited.add(member)
                cluster.append(member)
                
                # Add connected members to stack
                for connected_member in network.get(member, set()):
                    if connected_member not in visited:
                        stack.append(connected_member)
        
        return sorted(cluster)

    async def _update_suspicious_pairs_and_alerts(self, coordination_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Update suspicious pairs and manage comprehensive alert system"""
        try:
            alert_updates = {
                'new_alerts': [],
                'escalated_alerts': [],
                'resolved_alerts': [],
                'alert_summary': {}
            }
            
            old_suspicious = self.suspicious_pairs.copy()
            self.suspicious_pairs.clear()
            
            # Update suspicious pairs from coordination analysis
            coordinated_pairs = coordination_analysis.get('coordinated_pairs', [])
            self.suspicious_pairs.update(coordinated_pairs)
            
            # Generate alerts for new or escalated suspicious activity
            for pair in self.suspicious_pairs:
                coordination_strength = coordination_analysis.get('coordination_strength', {}).get(pair, {})
                coordination_score = coordination_strength.get('coordination_score', 0.0)
                
                # Check if this is a new alert or needs escalation
                last_alert_step = self.alert_system['last_alerts'].get(pair, 0)
                steps_since_last_alert = self.detection_stats['total_checks'] - last_alert_step
                
                should_alert = (
                    pair not in old_suspicious or  # New suspicious pair
                    steps_since_last_alert > self.alert_system['cooldown_period']  # Cooldown period passed
                )
                
                if should_alert:
                    alert_severity = self._determine_alert_severity(coordination_score)
                    alert_info = await self._generate_coordination_alert(pair, coordination_strength, alert_severity)
                    
                    alert_updates['new_alerts'].append(alert_info)
                    self.alert_system['last_alerts'][pair] = self.detection_stats['total_checks']
                    
                    # Record coordination event
                    coordination_event = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'pair': pair,
                        'coordination_score': coordination_score,
                        'alert_severity': alert_severity,
                        'coordination_strength': coordination_strength,
                        'alert_type': 'coordination_detection'
                    }
                    self.coordination_events.append(coordination_event)
            
            # Check for resolved alerts (pairs no longer suspicious)
            resolved_pairs = old_suspicious - self.suspicious_pairs
            for pair in resolved_pairs:
                alert_updates['resolved_alerts'].append({
                    'pair': pair,
                    'resolution_timestamp': datetime.datetime.now().isoformat(),
                    'resolution_reason': 'coordination_below_threshold'
                })
            
            # Update alert statistics
            self.detection_stats['alerts_raised'] += len(alert_updates['new_alerts'])
            
            alert_updates['alert_summary'] = {
                'total_suspicious_pairs': len(self.suspicious_pairs),
                'new_alerts_count': len(alert_updates['new_alerts']),
                'resolved_alerts_count': len(alert_updates['resolved_alerts']),
                'active_cooldowns': len(self.alert_system['last_alerts'])
            }
            
            return alert_updates
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "alert_system_update")
            return {'new_alerts': [], 'escalated_alerts': [], 'resolved_alerts': [], 'alert_summary': {}}

    def _determine_alert_severity(self, coordination_score: float) -> str:
        """Determine alert severity based on coordination score"""
        try:
            if coordination_score > 0.9:
                return 'critical'
            elif coordination_score > 0.8:
                return 'warning'
            else:
                return 'info'
        except Exception:
            return 'info'

    async def _generate_coordination_alert(self, pair: Tuple[int, int], coordination_strength: Dict[str, Any], 
                                         severity: str) -> Dict[str, Any]:
        """Generate comprehensive coordination alert"""
        try:
            member_i, member_j = pair
            historical_avg = coordination_strength.get('historical_avg', 0.0)
            consistency = coordination_strength.get('consistency', 0.0)
            trend = coordination_strength.get('trend', 0.0)
            
            # Generate human-readable alert message
            if severity == 'critical':
                icon = "🚨"
                message = f"CRITICAL: High coordination detected between members {member_i} and {member_j}"
            elif severity == 'warning':
                icon = "⚠️"
                message = f"WARNING: Suspicious coordination between members {member_i} and {member_j}"
            else:
                icon = "ℹ️"
                message = f"INFO: Monitoring coordination between members {member_i} and {member_j}"
            
            # Log the alert
            self.logger.warning(format_operator_message(
                icon=icon,
                message=message,
                coordination=f"{historical_avg:.3f}",
                threshold=f"{self.current_threshold:.3f}",
                consistency=f"{consistency:.3f}",
                trend=f"{trend:+.3f}",
                action_required="Monitor these members closely"
            ))
            
            return {
                'timestamp': datetime.datetime.now().isoformat(),
                'pair': pair,
                'severity': severity,
                'message': message,
                'coordination_score': historical_avg,
                'threshold_used': self.current_threshold,
                'consistency_score': consistency,
                'trend': trend,
                'recommended_action': self._get_recommended_action(severity, coordination_strength)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "alert_generation")
            return {
                'timestamp': datetime.datetime.now().isoformat(),
                'pair': pair,
                'severity': 'info',
                'message': f"Alert generation failed: {error_context}",
                'coordination_score': 0.0
            }

    def _get_recommended_action(self, severity: str, coordination_strength: Dict[str, Any]) -> str:
        """Get recommended action based on alert severity and coordination details"""
        try:
            if severity == 'critical':
                return "Immediate investigation required - consider member rotation or voting weight adjustment"
            elif severity == 'warning':
                return "Enhanced monitoring recommended - review member behavior patterns"
            else:
                return "Continue standard monitoring - document coordination patterns"
        except Exception:
            return "Monitor situation and investigate if patterns persist"

    async def _calculate_overall_collusion_score(self, coordination_analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive overall collusion score"""
        try:
            coordinated_pairs = coordination_analysis.get('coordinated_pairs', [])
            network_effects = coordination_analysis.get('network_effects', {})
            
            # Base score from pair coordination
            max_possible_pairs = self.n_members * (self.n_members - 1) / 2
            pair_score = len(coordinated_pairs) / max(max_possible_pairs, 1)
            
            # Network effects bonus
            network_score = network_effects.get('network_score', 0.0)
            
            # Weighted combination
            overall_score = (0.7 * pair_score + 0.3 * network_score)
            
            # Apply temporal consistency factor
            if len(self.collusion_history) >= 3:
                recent_scores = [event.get('collusion_score', 0.0) for event in list(self.collusion_history)[-3:]]
                consistency_factor = 1.0 - (np.std(recent_scores) / max(np.mean(recent_scores), 0.1))
                overall_score *= (0.8 + 0.2 * consistency_factor)
            
            return float(np.clip(overall_score, 0.0, 1.0))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "overall_collusion_score_calculation")
            return 0.0

    async def _update_detection_statistics_comprehensive(self, similarity_analysis: Dict[Tuple[int, int], Dict[str, float]], 
                                                       coordination_analysis: Dict[str, Any]):
        """Update comprehensive detection statistics and analytics"""
        try:
            # Calculate average pair similarity
            all_similarities = []
            for similarities in similarity_analysis.values():
                all_similarities.extend(similarities.values())
            
            if all_similarities:
                self.detection_stats['avg_pair_similarity'] = float(np.mean(all_similarities))
            
            # Update member independence scores
            for member_id in range(self.n_members):
                member_similarities = []
                for pair, similarities in similarity_analysis.items():
                    if member_id in pair and similarities:
                        member_similarities.extend(similarities.values())
                
                if member_similarities:
                    avg_similarity = np.mean(member_similarities)
                    independence_score = max(0.0, 1.0 - avg_similarity)
                    self.detection_stats['member_independence_scores'][f'member_{member_id}'] = independence_score
            
            # Update alert frequency
            if self.detection_stats['total_checks'] > 0:
                self.detection_stats['alert_frequency'] = (
                    self.detection_stats['alerts_raised'] / self.detection_stats['total_checks']
                )
            
            # Update performance metrics
            self._update_performance_metric('collusion_score', self.collusion_score)
            self._update_performance_metric('suspicious_pairs_count', len(self.suspicious_pairs))
            self._update_performance_metric('avg_pair_similarity', self.detection_stats['avg_pair_similarity'])
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "detection_statistics_update")

    async def _update_behavioral_profiles_comprehensive(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update comprehensive behavioral profiles for all members"""
        try:
            behavioral_updates = {
                'profile_updates': {},
                'anomaly_detections': {},
                'behavioral_trends': {}
            }
            
            raw_proposals = voting_data.get('raw_proposals', [])
            if len(raw_proposals) < 2:
                return behavioral_updates
            
            # Analyze each member's behavior
            for i, proposal in enumerate(raw_proposals[:self.n_members]):
                if not isinstance(proposal, (list, np.ndarray)) or len(proposal) == 0:
                    continue
                
                proposal_array = np.array(proposal)
                profile = self.member_behavior_profiles[i]
                
                # Calculate similarity with all other members
                similarities = []
                for j, other_proposal in enumerate(raw_proposals[:self.n_members]):
                    if i != j and isinstance(other_proposal, (list, np.ndarray)) and len(other_proposal) > 0:
                        other_array = np.array(other_proposal)
                        if np.linalg.norm(proposal_array) > 0 and np.linalg.norm(other_array) > 0:
                            sim = np.dot(proposal_array, other_array) / (
                                np.linalg.norm(proposal_array) * np.linalg.norm(other_array)
                            )
                            similarities.append(sim)
                
                if similarities:
                    # Update member profile with comprehensive metrics
                    old_avg_similarity = profile.get('avg_similarity', 0.0)
                    new_avg_similarity = np.mean(similarities)
                    
                    # Exponential moving average for smoothing
                    memory_factor = self.detection_intelligence['behavioral_memory']
                    profile['avg_similarity'] = (
                        old_avg_similarity * memory_factor + 
                        new_avg_similarity * (1 - memory_factor)
                    )
                    
                    profile['volatility'] = float(np.std(similarities))
                    profile['independence_score'] = max(0.0, 1.0 - profile['avg_similarity'])
                    profile['consistency_score'] = float(max(0.0, 1.0 - profile['volatility']))
                    
                    # Calculate coordination frequency
                    high_similarity_count = sum(1 for sim in similarities if sim > self.current_threshold)
                    profile['coordination_frequency'] = high_similarity_count / len(similarities)
                    
                    # Calculate anomaly score
                    if len(similarities) >= 3:
                        profile['anomaly_score'] = self._calculate_member_anomaly_score(similarities, profile)
                    
                    # Track behavioral changes
                    similarity_change = abs(new_avg_similarity - old_avg_similarity)
                    if similarity_change > 0.2:  # Significant behavioral change
                        behavioral_updates['profile_updates'][i] = {
                            'old_similarity': old_avg_similarity,
                            'new_similarity': new_avg_similarity,
                            'change_magnitude': similarity_change,
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                    
                    # Detect anomalies
                    if profile['anomaly_score'] > 0.7:
                        behavioral_updates['anomaly_detections'][i] = {
                            'anomaly_score': profile['anomaly_score'],
                            'anomaly_type': self._classify_behavioral_anomaly(profile),
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                
                # Update global statistics
                self.detection_stats['member_independence_scores'][f'member_{i}'] = profile['independence_score']
            
            # Analyze behavioral trends across all members
            behavioral_updates['behavioral_trends'] = await self._analyze_behavioral_trends()
            
            return behavioral_updates
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "behavioral_profile_update")
            return {'profile_updates': {}, 'anomaly_detections': {}, 'behavioral_trends': {}}

    def _calculate_member_anomaly_score(self, similarities: List[float], profile: Dict[str, Any]) -> float:
        """Calculate anomaly score for individual member behavior"""
        try:
            historical_avg = profile.get('avg_similarity', 0.0)
            historical_volatility = profile.get('volatility', 0.0)
            
            current_avg = np.mean(similarities)
            current_volatility = np.std(similarities)
            
            # Anomaly based on deviation from historical behavior
            avg_deviation = abs(current_avg - historical_avg) / max(historical_avg, 0.1)
            volatility_deviation = abs(current_volatility - historical_volatility) / max(historical_volatility, 0.1)
            
            # Combined anomaly score
            anomaly_score = (avg_deviation + volatility_deviation) / 2
            return min(1.0, anomaly_score)
            
        except Exception:
            return 0.0

    def _classify_behavioral_anomaly(self, profile: Dict[str, Any]) -> str:
        """Classify the type of behavioral anomaly"""
        try:
            avg_similarity = profile.get('avg_similarity', 0.0)
            volatility = profile.get('volatility', 0.0)
            coordination_frequency = profile.get('coordination_frequency', 0.0)
            
            if avg_similarity > 0.8 and coordination_frequency > 0.7:
                return 'high_coordination'
            elif volatility > 0.5:
                return 'erratic_behavior'
            elif avg_similarity < 0.2:
                return 'isolation_behavior'
            else:
                return 'moderate_anomaly'
                
        except Exception:
            return 'unknown_anomaly'

    async def _analyze_behavioral_trends(self) -> Dict[str, Any]:
        """Analyze behavioral trends across all committee members"""
        try:
            trends = {
                'overall_coordination_trend': 'stable',
                'independence_distribution': {},
                'coordination_network_density': 0.0,
                'behavioral_diversity': 0.0
            }
            
            # Calculate overall coordination trend
            if len(self.collusion_history) >= 5:
                recent_scores = [event.get('collusion_score', 0.0) for event in list(self.collusion_history)[-5:]]
                trend_slope = self._calculate_slope(recent_scores)
                
                if trend_slope > 0.1:
                    trends['overall_coordination_trend'] = 'increasing'
                elif trend_slope < -0.1:
                    trends['overall_coordination_trend'] = 'decreasing'
                else:
                    trends['overall_coordination_trend'] = 'stable'
            
            # Analyze independence distribution
            independence_scores = [
                profile.get('independence_score', 1.0) 
                for profile in self.member_behavior_profiles.values()
            ]
            
            if independence_scores:
                trends['independence_distribution'] = {
                    'mean': np.mean(independence_scores),
                    'std': np.std(independence_scores),
                    'min': np.min(independence_scores),
                    'max': np.max(independence_scores)
                }
                
                # Behavioral diversity (higher std = more diverse behavior)
                trends['behavioral_diversity'] = np.std(independence_scores)
            
            # Coordination network density
            if len(self.suspicious_pairs) > 0:
                max_possible_pairs = self.n_members * (self.n_members - 1) / 2
                trends['coordination_network_density'] = len(self.suspicious_pairs) / max(max_possible_pairs, 1)
            
            return trends
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "behavioral_trends_analysis")
            return {'overall_coordination_trend': 'unknown'}

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

    def _update_performance_metric(self, metric_name: str, value: float) -> None:
        """Update performance metric for tracking"""
        try:
            # Track performance metrics for monitoring
            if hasattr(self, 'performance_tracker'):
                self.performance_tracker.record_metric('CollusionAuditor', metric_name, value, True)
        except Exception:
            # Silently fail if performance tracking is not available
            pass

    async def _analyze_temporal_coordination_patterns(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal coordination patterns and timing-based collusion"""
        try:
            temporal_analysis = {
                'timing_patterns': {},
                'coordination_clusters': [],
                'temporal_consistency': 0.0,
                'synchronized_responses': {}
            }
            
            if len(self.vote_history) < 5:
                temporal_analysis['status'] = 'insufficient_history'
                return temporal_analysis
            
            # Analyze timing patterns in recent voting history
            recent_votes = list(self.vote_history)[-10:]
            
            # Look for synchronized response patterns
            for i, vote_entry in enumerate(recent_votes):
                timestamp = vote_entry.get('timestamp', '')
                if timestamp:
                    # Analyze time-based clustering
                    sync_analysis = await self._analyze_vote_synchronization(vote_entry, recent_votes[max(0, i-2):i])
                    if sync_analysis['synchronization_score'] > 0.7:
                        temporal_analysis['synchronized_responses'][timestamp] = sync_analysis
            
            # Calculate temporal consistency across all pairs
            consistency_scores = []
            for pair in self.pair_agreement_history:
                history = list(self.pair_agreement_history[pair])
                if len(history) >= 5:
                    consistency = self._calculate_temporal_consistency(history)
                    consistency_scores.append(consistency)
                    temporal_analysis['timing_patterns'][str(pair)] = {
                        'consistency': consistency,
                        'pattern_type': self._classify_temporal_pattern(history)
                    }
            
            if consistency_scores:
                temporal_analysis['temporal_consistency'] = np.mean(consistency_scores)
            
            # Identify coordination clusters based on timing
            temporal_analysis['coordination_clusters'] = await self._identify_temporal_coordination_clusters()
            
            return temporal_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "temporal_coordination_analysis")
            return {'timing_patterns': {}, 'analysis_error': str(error_context)}

    async def _analyze_vote_synchronization(self, current_vote: Dict[str, Any], 
                                          recent_votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze synchronization patterns in voting timing"""
        try:
            sync_analysis = {
                'synchronization_score': 0.0,
                'synchronized_members': [],
                'timing_deviation': 0.0
            }
            
            current_actions = current_vote.get('actions', [])
            if len(current_actions) < 2:
                return sync_analysis
            
            # Simple synchronization analysis based on action similarity timing
            synchronization_count = 0
            total_pairs = 0
            
            for i in range(len(current_actions)):
                for j in range(i + 1, len(current_actions)):
                    action_i = np.array(current_actions[i])
                    action_j = np.array(current_actions[j])
                    
                    if np.linalg.norm(action_i) > 0 and np.linalg.norm(action_j) > 0:
                        similarity = np.dot(action_i, action_j) / (
                            np.linalg.norm(action_i) * np.linalg.norm(action_j)
                        )
                        
                        if similarity > 0.9:  # High similarity suggests synchronization
                            synchronization_count += 1
                            sync_analysis['synchronized_members'].append((i, j))
                        
                        total_pairs += 1
            
            if total_pairs > 0:
                sync_analysis['synchronization_score'] = synchronization_count / total_pairs
            
            return sync_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "vote_synchronization_analysis")
            return {'synchronization_score': 0.0, 'synchronized_members': [], 'timing_deviation': 0.0}

    def _calculate_temporal_consistency(self, history: List[float]) -> float:
        """Calculate temporal consistency in coordination patterns"""
        try:
            if len(history) < 3:
                return 0.0
            
            # Consistency based on how stable the coordination pattern is over time
            variance = np.var(history)
            mean_val = np.mean(history)
            
            if mean_val < 1e-6:
                return 0.0
            
            # Consistency is inverse of coefficient of variation
            coefficient_of_variation = np.sqrt(variance) / mean_val
            consistency = max(0.0, 1.0 - coefficient_of_variation)
            
            return consistency
            
        except Exception:
            return 0.0

    def _classify_temporal_pattern(self, history: List[float]) -> str:
        """Classify temporal coordination pattern"""
        try:
            if len(history) < 3:
                return 'insufficient_data'
            
            # Calculate trend and volatility
            slope = self._calculate_slope(history)
            volatility = np.std(history)
            
            if volatility < 0.1:
                if np.mean(history) > 0.8:
                    return 'consistently_high'
                elif np.mean(history) < 0.3:
                    return 'consistently_low'
                else:
                    return 'stable_moderate'
            elif abs(slope) > 0.1:
                return 'trending_up' if slope > 0 else 'trending_down'
            else:
                return 'volatile'
                
        except Exception:
            return 'unknown'

    async def _identify_temporal_coordination_clusters(self) -> List[Dict[str, Any]]:
        """Identify temporal coordination clusters"""
        try:
            clusters = []
            
            # Simple clustering based on coordination timing patterns
            if len(self.coordination_events) < 3:
                return clusters
            
            # Group coordination events by time windows
            time_windows = defaultdict(list)
            
            for event in list(self.coordination_events)[-20:]:  # Recent events
                timestamp_str = event.get('timestamp', '')
                if timestamp_str:
                    try:
                        timestamp = datetime.datetime.fromisoformat(timestamp_str)
                        # Group by 5-minute windows
                        window_key = timestamp.replace(second=0, microsecond=0)
                        window_key = window_key.replace(minute=(window_key.minute // 5) * 5)
                        time_windows[window_key].append(event)
                    except Exception:
                        continue
            
            # Identify significant clusters (multiple coordination events in same window)
            for window_time, events in time_windows.items():
                if len(events) >= 2:  # Multiple coordination events in same window
                    cluster = {
                        'window_start': window_time.isoformat(),
                        'event_count': len(events),
                        'involved_pairs': [event.get('pair') for event in events],
                        'avg_coordination_score': np.mean([
                            event.get('coordination_score', 0.0) for event in events
                        ]),
                        'cluster_significance': len(events) / len(list(self.coordination_events)[-20:])
                    }
                    clusters.append(cluster)
            
            # Sort by significance
            clusters.sort(key=lambda x: x['cluster_significance'], reverse=True)
            
            return clusters[:5]  # Return top 5 clusters
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "temporal_clustering")
            return []

    async def _calculate_comprehensive_quality_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for detection system"""
        try:
            quality_metrics = {
                'detection_precision': self.quality_metrics.get('detection_precision', 0.0),
                'detection_recall': self.quality_metrics.get('detection_recall', 0.0),
                'behavioral_accuracy': self.quality_metrics.get('behavioral_accuracy', 0.0),
                'temporal_consistency': self.quality_metrics.get('temporal_consistency', 0.0),
                'overall_effectiveness': 0.0
            }
            
            # Update detection precision (how many detected events are actually problematic)
            if self.detection_stats['alerts_raised'] > 0:
                # Simplified precision based on confirmed vs total alerts
                confirmed_events = self.detection_stats.get('confirmed_collusion_events', 0)
                quality_metrics['detection_precision'] = confirmed_events / self.detection_stats['alerts_raised']
            
            # Update detection recall (estimated based on behavioral consistency)
            if len(self.member_behavior_profiles) > 0:
                independence_scores = [
                    profile.get('independence_score', 1.0) 
                    for profile in self.member_behavior_profiles.values()
                ]
                avg_independence = np.mean(independence_scores)
                # Higher average independence suggests good recall (catching coordination)
                quality_metrics['detection_recall'] = float(1.0 - avg_independence)
            
            # Update behavioral accuracy (consistency of behavioral profiling)
            if len(self.collusion_history) >= 5:
                recent_scores = [event.get('collusion_score', 0.0) for event in list(self.collusion_history)[-5:]]
                behavioral_consistency = 1.0 - (np.std(recent_scores) / max(np.mean(recent_scores), 0.1))
                quality_metrics['behavioral_accuracy'] = float(max(0.0, behavioral_consistency))
            
            # Update temporal consistency
            if self.pair_agreement_history:
                consistency_scores = []
                for pair_history in self.pair_agreement_history.values():
                    if len(pair_history) >= 3:
                        consistency = self._calculate_temporal_consistency(list(pair_history))
                        consistency_scores.append(consistency)
                
                if consistency_scores:
                    quality_metrics['temporal_consistency'] = float(np.mean(consistency_scores))
            
            # Calculate overall effectiveness
            weights = [0.3, 0.3, 0.2, 0.2]
            values = [
                quality_metrics['detection_precision'],
                quality_metrics['detection_recall'],
                quality_metrics['behavioral_accuracy'],
                quality_metrics['temporal_consistency']
            ]
            
            quality_metrics['overall_effectiveness'] = float(np.average(values, weights=weights))
            
            # Update internal quality metrics
            self.quality_metrics.update(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "quality_metrics_calculation")
            return {'overall_effectiveness': 0.5, 'calculation_error': str(error_context)}

    async def _generate_intelligent_detection_recommendations(self, collusion_analysis: Dict[str, Any], 
                                                           behavioral_updates: Dict[str, Any], 
                                                           temporal_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent detection recommendations"""
        try:
            recommendations = []
            
            # Collusion-based recommendations
            collusion_score = collusion_analysis.get('collusion_score', 0.0)
            if collusion_score > 0.8:
                recommendations.append("HIGH PRIORITY: Immediate investigation of detected coordination patterns required")
            elif collusion_score > 0.5:
                recommendations.append("MODERATE: Enhanced monitoring and analysis of suspicious member pairs")
            
            # Behavioral anomaly recommendations
            anomaly_detections = behavioral_updates.get('anomaly_detections', {})
            if len(anomaly_detections) > 0:
                member_count = len(anomaly_detections)
                recommendations.append(f"BEHAVIORAL: {member_count} members showing anomalous behavior patterns - investigate")
            
            # Temporal pattern recommendations
            temporal_consistency = temporal_analysis.get('temporal_consistency', 0.0)
            if temporal_consistency > 0.8:
                recommendations.append("TEMPORAL: High temporal coordination detected - review timing-based collusion")
            
            # Network effect recommendations
            network_effects = collusion_analysis.get('coordination_analysis', {}).get('network_effects', {})
            clusters = network_effects.get('clusters', [])
            if len(clusters) > 0:
                largest_cluster = max(len(cluster) for cluster in clusters)
                if largest_cluster >= 3:
                    recommendations.append(f"NETWORK: Large coordination cluster detected ({largest_cluster} members) - consider member rotation")
            
            # Alert frequency recommendations
            alert_frequency = self.detection_stats.get('alert_frequency', 0.0)
            if alert_frequency > 0.3:
                recommendations.append("SYSTEM: High alert frequency - review detection sensitivity")
            elif alert_frequency < 0.05:
                recommendations.append("SYSTEM: Low alert frequency - consider increasing detection sensitivity")
            
            # Threshold adjustment recommendations
            if self.adaptive_threshold:
                threshold_change = abs(self.current_threshold - self.base_threshold) / self.base_threshold
                if threshold_change > 0.2:
                    recommendations.append("THRESHOLD: Significant threshold adaptation - review market condition sensitivity")
            
            # Quality-based recommendations
            overall_effectiveness = self.quality_metrics.get('overall_effectiveness', 0.0)
            if overall_effectiveness < 0.4:
                recommendations.append("QUALITY: Low detection effectiveness - review detection parameters and methods")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("SYSTEM: Collusion detection operating within normal parameters")
            
            return recommendations[:6]  # Limit to top 6 recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendation_generation")
            return [f"Recommendation generation failed: {error_context}"]

    async def _generate_comprehensive_detection_thesis(self, collusion_analysis: Dict[str, Any], 
                                                     quality_analysis: Dict[str, Any], 
                                                     recommendations: List[str]) -> str:
        """Generate comprehensive detection thesis"""
        try:
            # Core metrics
            collusion_score = collusion_analysis.get('collusion_score', 0.0)
            suspicious_pairs_count = len(collusion_analysis.get('suspicious_pairs', []))
            overall_effectiveness = quality_analysis.get('overall_effectiveness', 0.0)
            
            thesis_parts = []
            
            # Executive summary
            risk_level = "HIGH" if collusion_score > 0.7 else "MODERATE" if collusion_score > 0.4 else "LOW"
            thesis_parts.append(
                f"COLLUSION ANALYSIS: {risk_level} risk with {collusion_score:.1%} coordination score"
            )
            
            # Detection summary
            thesis_parts.append(
                f"DETECTION STATUS: {suspicious_pairs_count} suspicious pairs identified from {self.n_members} members"
            )
            
            # Quality assessment
            thesis_parts.append(
                f"SYSTEM EFFECTIVENESS: {overall_effectiveness:.1%} detection quality across multiple analysis methods"
            )
            
            # Alert status
            alerts_raised = self.detection_stats.get('alerts_raised', 0)
            if alerts_raised > 0:
                thesis_parts.append(f"ALERT STATUS: {alerts_raised} alerts raised with managed escalation")
            
            # Threshold status
            if self.adaptive_threshold:
                threshold_change = (self.current_threshold - self.base_threshold) / self.base_threshold
                thesis_parts.append(
                    f"THRESHOLD ADAPTATION: {threshold_change:+.1%} adjustment for market conditions"
                )
            
            # Behavioral insights
            behavioral_anomalies = len([p for p in self.member_behavior_profiles.values() 
                                     if p.get('anomaly_score', 0) > 0.5])
            if behavioral_anomalies > 0:
                thesis_parts.append(f"BEHAVIORAL ANALYSIS: {behavioral_anomalies} members with anomalous patterns")
            
            # System performance
            total_checks = self.detection_stats.get('total_checks', 0)
            thesis_parts.append(f"SYSTEM PERFORMANCE: {total_checks} checks completed with comprehensive analysis")
            
            # Recommendations summary
            priority_recommendations = [rec for rec in recommendations if any(keyword in rec 
                                      for keyword in ['HIGH PRIORITY', 'CRITICAL', 'IMMEDIATE'])]
            if priority_recommendations:
                thesis_parts.append(f"ACTION REQUIRED: {len(priority_recommendations)} high-priority recommendations")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Detection thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive detection results"""
        try:
            # Core detection results
            self.smart_bus.set('collusion_score', results['collusion_score'],
                             module='CollusionAuditor', thesis=thesis)
            
            # Suspicious pairs
            pairs_thesis = f"Suspicious coordination: {len(results['suspicious_pairs'])} pairs under monitoring"
            self.smart_bus.set('suspicious_pairs', results['suspicious_pairs'],
                             module='CollusionAuditor', thesis=pairs_thesis)
            
            # Member independence scores
            independence_thesis = f"Member independence: {len(results['member_independence_scores'])} profiles analyzed"
            self.smart_bus.set('member_independence_scores', results['member_independence_scores'],
                             module='CollusionAuditor', thesis=independence_thesis)
            
            # Collusion alerts
            alerts_thesis = f"Alert system: {len(results['collusion_alerts'])} active alerts managed"
            self.smart_bus.set('collusion_alerts', results['collusion_alerts'],
                             module='CollusionAuditor', thesis=alerts_thesis)
            
            # Behavioral profiles
            behavioral_thesis = f"Behavioral analysis: {len(results['behavioral_profiles'])} member profiles updated"
            self.smart_bus.set('behavioral_profiles', results['behavioral_profiles'],
                             module='CollusionAuditor', thesis=behavioral_thesis)
            
            # Coordination events
            events_thesis = f"Coordination tracking: {len(results['coordination_events'])} recent events recorded"
            self.smart_bus.set('coordination_events', results['coordination_events'],
                             module='CollusionAuditor', thesis=events_thesis)
            
            # Detection statistics
            stats_thesis = f"Detection statistics: {results['detection_statistics']['total_checks']} total checks performed"
            self.smart_bus.set('detection_statistics', results['detection_statistics'],
                             module='CollusionAuditor', thesis=stats_thesis)
            
            # Audit recommendations
            rec_thesis = f"Audit recommendations: {len(results['audit_recommendations'])} actionable insights"
            self.smart_bus.set('audit_recommendations', results['audit_recommendations'],
                             module='CollusionAuditor', thesis=rec_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY COMPATIBILITY AND PUBLIC INTERFACE
    # ═══════════════════════════════════════════════════════════════════

    def check_collusion(self, actions: List[np.ndarray]) -> float:
        """Legacy collusion checking interface for backward compatibility"""
        try:
            # Run async method synchronously
            import asyncio
            
            # Create minimal voting data from actions
            voting_data = {
                'raw_proposals': actions,
                'votes': [np.mean(action) if len(action) > 0 else 0.0 for action in actions],
                'agreement_score': 0.5,
                'market_regime': 'unknown',
                'market_context': {},
                'recent_trades': []
            }
            
            if asyncio.get_event_loop().is_running():
                # If already in async context, use simplified sync method
                return self._simple_collusion_check_fallback(actions)
            else:
                # Run async analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    collusion_analysis = loop.run_until_complete(
                        self._perform_comprehensive_collusion_analysis(voting_data)
                    )
                    return collusion_analysis.get('collusion_score', 0.0)
                finally:
                    loop.close()
                    
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "legacy_collusion_check")
            return self._simple_collusion_check_fallback(actions)

    def _simple_collusion_check_fallback(self, actions: List[np.ndarray]) -> float:
        """Simple fallback collusion checking method"""
        try:
            if len(actions) < 2:
                return 0.0
            
            # Simple pairwise similarity analysis
            similarities = []
            suspicious_count = 0
            
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    v1, v2 = actions[i], actions[j]
                    
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        similarities.append(cosine_sim)
                        
                        if cosine_sim > self.current_threshold:
                            suspicious_count += 1
            
            # Calculate collusion score
            max_pairs = len(actions) * (len(actions) - 1) / 2
            collusion_score = suspicious_count / max(max_pairs, 1)
            
            # Update basic state
            self.collusion_score = collusion_score
            if similarities:
                self.detection_stats['avg_pair_similarity'] = float(np.mean(similarities))
            
            return collusion_score
            
        except Exception:
            return 0.0

    def get_member_independence_scores(self) -> Dict[int, float]:
        """Get independence scores for all members"""
        scores = {}
        for member_id, profile in self.member_behavior_profiles.items():
            scores[member_id] = profile.get('independence_score', 1.0)
        return scores

    def _get_recent_collusion_alerts(self) -> List[Dict[str, Any]]:
        """Get recent collusion alerts"""
        try:
            recent_events = list(self.coordination_events)[-5:]
            alerts = []
            
            for event in recent_events:
                if event.get('alert_type') == 'coordination_detection':
                    alert = {
                        'timestamp': event.get('timestamp'),
                        'pair': event.get('pair'),
                        'severity': event.get('alert_severity', 'info'),
                        'coordination_score': event.get('coordination_score', 0.0),
                        'alert_type': 'coordination'
                    }
                    alerts.append(alert)
            
            return alerts
            
        except Exception:
            return []

    def _get_behavioral_profiles_summary(self) -> Dict[str, Any]:
        """Get summary of behavioral profiles"""
        try:
            summary = {}
            
            for member_id, profile in self.member_behavior_profiles.items():
                summary[f'member_{member_id}'] = {
                    'independence_score': profile.get('independence_score', 1.0),
                    'coordination_frequency': profile.get('coordination_frequency', 0.0),
                    'anomaly_score': profile.get('anomaly_score', 0.0),
                    'consistency_score': profile.get('consistency_score', 0.0)
                }
            
            return summary
            
        except Exception:
            return {}

    def _get_comprehensive_detection_stats(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics"""
        return {
            **self.detection_stats,
            'current_threshold': self.current_threshold,
            'base_threshold': self.base_threshold,
            'adaptive_enabled': self.adaptive_threshold,
            'members_monitored': self.n_members,
            'analysis_window': self.window,
            'similarity_methods': self.similarity_methods,
            'quality_metrics': self.quality_metrics.copy(),
            'recent_collusion_trend': self._calculate_recent_collusion_trend()
        }

    def _calculate_recent_collusion_trend(self) -> str:
        """Calculate recent collusion trend"""
        try:
            if len(self.collusion_history) < 3:
                return 'insufficient_data'
            
            recent_scores = [event.get('collusion_score', 0.0) for event in list(self.collusion_history)[-5:]]
            slope = self._calculate_slope(recent_scores)
            
            if slope > 0.1:
                return 'increasing'
            elif slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'

    def get_observation_components(self) -> np.ndarray:
        """Return collusion features for RL observation"""
        try:
            features = [
                float(self.collusion_score),
                float(len(self.suspicious_pairs) / max(self.n_members, 1)),
                float(self.current_threshold),
                float(self.detection_stats.get('avg_pair_similarity', 0)),
                float(len(self.vote_history) / self.window),
                float(self.quality_metrics.get('overall_effectiveness', 0.5)),
                float(self.detection_stats.get('alert_frequency', 0.0)),
                float(len(self.coordination_events) / 50)  # Coordination event density
            ]
            
            observation = np.array(features, dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid collusion observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Collusion observation generation failed: {error_context}")
            return np.array([0.0, 0.0, 0.9, 0.5, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'CollusionAuditor',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'total_checks': self.detection_stats.get('total_checks', 0),
            'alerts_raised': self.detection_stats.get('alerts_raised', 0),
            'suspicious_pairs_count': len(self.suspicious_pairs),
            'avg_pair_similarity': self.detection_stats.get('avg_pair_similarity', 0.0),
            'detection_effectiveness': self.quality_metrics.get('overall_effectiveness', 0.0),
            'alert_frequency': self.detection_stats.get('alert_frequency', 0.0),
            'behavioral_profiles_count': len(self.member_behavior_profiles),
            'coordination_events_count': len(self.coordination_events),
            'session_duration': (datetime.datetime.now() - 
                               datetime.datetime.fromisoformat(self.detection_stats['session_start'])).total_seconds() / 3600
        }

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Internal method for health metrics (for compatibility)"""
        return self.get_health_metrics()

    def get_collusion_report(self) -> str:
        """Generate comprehensive operator-friendly collusion report"""
        # Risk assessment
        if self.collusion_score > 0.8:
            risk_level = "🚨 CRITICAL RISK"
        elif self.collusion_score > 0.5:
            risk_level = "⚠️ HIGH RISK"
        elif self.collusion_score > 0.2:
            risk_level = "🟡 MODERATE RISK"
        else:
            risk_level = "✅ LOW RISK"
        
        # Recent activity
        recent_alerts = len([e for e in self.coordination_events if 
                           (datetime.datetime.now() - 
                            datetime.datetime.fromisoformat(e['timestamp'])).seconds < 600])
        
        # Suspicious pairs details
        suspicious_details = []
        for pair in list(self.suspicious_pairs)[:5]:  # Show top 5
            i, j = pair
            history = self.pair_agreement_history.get(pair, [])
            if history:
                avg_sim = np.mean(list(history))
                suspicious_details.append(f"  🔍 Members {i}-{j}: {avg_sim:.1%} similarity")
        
        # Member independence summary
        independence_summary = []
        for member_id, profile in list(self.member_behavior_profiles.items())[:5]:
            independence = profile.get('independence_score', 1.0)
            anomaly_score = profile.get('anomaly_score', 0.0)
            if independence < 0.7 or anomaly_score > 0.5:
                status = "🚨" if anomaly_score > 0.7 else "⚠️"
                independence_summary.append(f"  {status} Member {member_id}: {independence:.1%} independence, {anomaly_score:.1%} anomaly")
        
        # System effectiveness
        effectiveness = self.quality_metrics.get('overall_effectiveness', 0.0)
        if effectiveness > 0.8:
            effectiveness_status = "✅ Excellent"
        elif effectiveness > 0.6:
            effectiveness_status = "⚡ Good"
        elif effectiveness > 0.4:
            effectiveness_status = "⚠️ Fair"
        else:
            effectiveness_status = "🚨 Poor"
        
        return f"""
🕵️ COLLUSION AUDITOR v3.0
═══════════════════════════════════════════════════════════════
🎯 Current Status: {risk_level}
📊 Collusion Score: {self.collusion_score:.1%}
🎚️ Detection Threshold: {self.current_threshold:.1%} (Base: {self.base_threshold:.1%})

📈 Detection Performance:
• Total Checks: {self.detection_stats['total_checks']}
• Alerts Raised: {self.detection_stats['alerts_raised']}
• Recent Alerts (10min): {recent_alerts}
• Alert Frequency: {self.detection_stats.get('alert_frequency', 0.0):.1%}
• Average Pair Similarity: {self.detection_stats.get('avg_pair_similarity', 0):.1%}

🔍 Current Surveillance:
• Committee Size: {self.n_members} members
• Suspicious Pairs: {len(self.suspicious_pairs)}
• Coordination Events: {len(self.coordination_events)}
• Members Under Watch: {len(self.alert_system['last_alerts'])}
• Behavioral Profiles: {len(self.member_behavior_profiles)}

📊 System Configuration:
• Analysis Window: {self.window} votes
• Similarity Methods: {', '.join(self.similarity_methods)}
• Adaptive Threshold: {'✅ Enabled' if self.adaptive_threshold else '❌ Disabled'}
• Alert Cooldown: {self.alert_system['cooldown_period']} steps
• Detection Methods: {len(self.detection_methods)} active

🔍 Suspicious Pairs:
{chr(10).join(suspicious_details) if suspicious_details else "  ✅ No suspicious pairs detected"}

⚠️ Member Alerts:
{chr(10).join(independence_summary) if independence_summary else "  ✅ All members showing normal behavior"}

📊 Quality Metrics:
• Detection Precision: {self.quality_metrics.get('detection_precision', 0.0):.1%}
• Detection Recall: {self.quality_metrics.get('detection_recall', 0.0):.1%}
• Behavioral Accuracy: {self.quality_metrics.get('behavioral_accuracy', 0.0):.1%}
• Temporal Consistency: {self.quality_metrics.get('temporal_consistency', 0.0):.1%}
• Overall Effectiveness: {effectiveness_status} ({effectiveness:.1%})

📊 Recent Activity:
• Vote History: {len(self.vote_history)} entries
• Collusion History: {len(self.collusion_history)} events
• Coordination Events: {len(self.coordination_events)} recorded
• Alert History: {len(self.alert_system.get('alert_history', []))} alerts

🔧 System Health:
• Error Count: {self.error_count}/{self.circuit_breaker_threshold}
• Status: {'🚨 DISABLED' if self.is_disabled else '✅ OPERATIONAL'}
• Session Duration: {(datetime.datetime.now() - datetime.datetime.fromisoformat(self.detection_stats['session_start'])).total_seconds() / 3600:.1f} hours
• Detection Trend: {self._calculate_recent_collusion_trend().title()}

🎯 Intelligence Metrics:
• Adaptation Rate: {self.detection_intelligence.get('adaptation_rate', 0.15):.1%}
• Sensitivity Target: {self.detection_intelligence.get('sensitivity_target', 0.85):.1%}
• False Positive Threshold: {self.detection_intelligence.get('false_positive_threshold', 0.1):.1%}
• Behavioral Memory: {self.detection_intelligence.get('behavioral_memory', 0.9):.1%}
        """

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'CollusionAuditor',
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
                'message': 'CollusionAuditor disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring issues'
            })
        
        # High collusion score alert
        if self.collusion_score > 0.7:
            alerts.append({
                'severity': 'critical',
                'message': f'High collusion score detected: {self.collusion_score:.1%}',
                'action': 'Immediate investigation of member coordination required'
            })
        
        # High alert frequency
        alert_frequency = self.detection_stats.get('alert_frequency', 0.0)
        if alert_frequency > 0.4:
            alerts.append({
                'severity': 'warning',
                'message': f'High alert frequency: {alert_frequency:.1%}',
                'action': 'Review detection sensitivity and threshold settings'
            })
        
        # Low detection effectiveness
        effectiveness = self.quality_metrics.get('overall_effectiveness', 0.0)
        if effectiveness < 0.3:
            alerts.append({
                'severity': 'warning',
                'message': f'Low detection effectiveness: {effectiveness:.1%}',
                'action': 'Review detection methods and parameters'
            })
        
        # Insufficient data
        if len(self.vote_history) < 5:
            alerts.append({
                'severity': 'info',
                'message': 'Insufficient voting history for reliable detection',
                'action': 'Continue operations to build detection baseline'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart CollusionAuditor module after investigating errors")
        
        if len(self.vote_history) < 10:
            recommendations.append("Insufficient voting history - continue operations to establish detection patterns")
        
        # Threshold recommendations
        threshold_deviation = abs(self.current_threshold - self.base_threshold) / self.base_threshold
        if threshold_deviation > 0.3:
            recommendations.append("Large threshold adaptation detected - review market sensitivity settings")
        
        # Coordination recommendations
        if len(self.suspicious_pairs) > self.n_members // 2:
            recommendations.append("High number of suspicious pairs - consider member rotation or voting methodology review")
        
        # Alert management recommendations
        if len(self.alert_system['last_alerts']) > self.n_members:
            recommendations.append("Many members under alert - review overall committee composition")
        
        # Performance recommendations
        effectiveness = self.quality_metrics.get('overall_effectiveness', 0.0)
        if effectiveness < 0.5:
            recommendations.append("Low detection effectiveness - consider adjusting similarity methods or thresholds")
        
        if not recommendations:
            recommendations.append("CollusionAuditor operating within normal parameters")
        
        return recommendations

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "CollusionAuditor")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Collusion Auditor disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('CollusionAuditor', 'process_time', processing_time, False)
        
        return {
            'collusion_score': 0.0,
            'suspicious_pairs': [],
            'member_independence_scores': {},
            'collusion_alerts': [],
            'behavioral_profiles': {},
            'coordination_events': [],
            'detection_statistics': {'error': str(error_context)},
            'audit_recommendations': ["Investigate collusion auditor errors"],
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_voting_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when voting data retrieval fails"""
        return {
            'votes': [], 'voting_summary': {}, 'strategy_arbiter_weights': [],
            'raw_proposals': [], 'member_confidences': [], 'consensus_direction': 'neutral',
            'agreement_score': 0.5, 'market_context': {}, 'recent_trades': [],
            'market_regime': 'unknown', 'volatility_data': {}
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'collusion_score': 0.0,
            'suspicious_pairs': [],
            'member_independence_scores': {},
            'collusion_alerts': [],
            'behavioral_profiles': {},
            'coordination_events': [],
            'detection_statistics': {'status': 'disabled'},
            'audit_recommendations': ["Restart collusion auditor system"],
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT AND HOT-RELOAD SUPPORT
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'CollusionAuditor',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'n_members': self.n_members,
                'window': self.window,
                'base_threshold': self.base_threshold,
                'adaptive_threshold': self.adaptive_threshold,
                'similarity_methods': self.similarity_methods,
                'debug': self.debug
            },
            'detection_state': {
                'current_threshold': self.current_threshold,
                'collusion_score': self.collusion_score,
                'suspicious_pairs': list(self.suspicious_pairs),
                'detection_stats': self.detection_stats.copy(),
                'quality_metrics': self.quality_metrics.copy()
            },
            'intelligence_state': {
                'detection_intelligence': self.detection_intelligence.copy(),
                'market_adaptation': self.market_adaptation.copy(),
                'alert_system': {k: (v.copy() if isinstance(v, dict) else v) for k, v in self.alert_system.items()}
            },
            'behavioral_state': {
                'member_behavior_profiles': {k: v.copy() for k, v in self.member_behavior_profiles.items()},
                'pair_agreement_history': {str(k): list(v) for k, v in self.pair_agreement_history.items()},
                'temporal_patterns': {k: v.copy() for k, v in self.temporal_patterns.items()}
            },
            'history_state': {
                'vote_history': list(self.vote_history)[-20:],
                'collusion_history': list(self.collusion_history)[-30:],
                'coordination_events': list(self.coordination_events)[-20:],
                'alert_patterns': {k: v.copy() for k, v in self.alert_patterns.items()}
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
            self.n_members = int(config.get("n_members", self.n_members))
            self.window = int(config.get("window", self.window))
            self.base_threshold = float(config.get("base_threshold", self.base_threshold))
            self.adaptive_threshold = bool(config.get("adaptive_threshold", self.adaptive_threshold))
            self.similarity_methods = config.get("similarity_methods", self.similarity_methods)
            self.debug = bool(config.get("debug", self.debug))
            
            # Load detection state
            detection_state = state.get("detection_state", {})
            self.current_threshold = float(detection_state.get("current_threshold", self.base_threshold))
            self.collusion_score = float(detection_state.get("collusion_score", 0.0))
            
            suspicious_pairs = detection_state.get("suspicious_pairs", [])
            self.suspicious_pairs = set(tuple(pair) for pair in suspicious_pairs)
            
            self.detection_stats.update(detection_state.get("detection_stats", {}))
            self.quality_metrics.update(detection_state.get("quality_metrics", {}))
            
            # Load intelligence state
            intelligence_state = state.get("intelligence_state", {})
            self.detection_intelligence.update(intelligence_state.get("detection_intelligence", {}))
            self.market_adaptation.update(intelligence_state.get("market_adaptation", {}))
            
            alert_system_data = intelligence_state.get("alert_system", {})
            for key, value in alert_system_data.items():
                if key in self.alert_system:
                    if isinstance(value, dict) and isinstance(self.alert_system[key], dict):
                        self.alert_system[key].update(value)
                    else:
                        self.alert_system[key] = value
            
            # Load behavioral state
            behavioral_state = state.get("behavioral_state", {})
            
            # Load member behavior profiles
            profiles_data = behavioral_state.get("member_behavior_profiles", {})
            self.member_behavior_profiles.clear()
            for member_id, profile_data in profiles_data.items():
                self.member_behavior_profiles[int(member_id)] = profile_data
            
            # Load pair agreement history
            pair_history_data = behavioral_state.get("pair_agreement_history", {})
            self.pair_agreement_history.clear()
            for pair_str, history_list in pair_history_data.items():
                try:
                    # Parse pair string like "(0, 1)" back to tuple
                    pair_str_clean = pair_str.strip('()')
                    pair_parts = [int(x.strip()) for x in pair_str_clean.split(',')]
                    if len(pair_parts) == 2:
                        pair = tuple(pair_parts)
                        self.pair_agreement_history[pair] = deque(history_list, maxlen=self.window)
                except Exception:
                    continue
            
            # Load temporal patterns
            self.temporal_patterns.clear()
            temporal_data = behavioral_state.get("temporal_patterns", {})
            for key, pattern_data in temporal_data.items():
                self.temporal_patterns[key] = pattern_data
            
            # Load history state
            history_state = state.get("history_state", {})
            
            # Load vote history
            self.vote_history.clear()
            for entry in history_state.get("vote_history", []):
                self.vote_history.append(entry)
            
            # Load collusion history
            self.collusion_history.clear()
            for entry in history_state.get("collusion_history", []):
                self.collusion_history.append(entry)
            
            # Load coordination events
            self.coordination_events.clear()
            for entry in history_state.get("coordination_events", []):
                self.coordination_events.append(entry)
            
            # Load alert patterns
            self.alert_patterns.clear()
            alert_patterns_data = history_state.get("alert_patterns", {})
            for key, pattern_data in alert_patterns_data.items():
                self.alert_patterns[key] = pattern_data
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Collusion Auditor state restored",
                members=self.n_members,
                threshold=f"{self.current_threshold:.3f}",
                suspicious_pairs=len(self.suspicious_pairs),
                total_checks=self.detection_stats.get('total_checks', 0)
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
        
        # Reset detection state
        self.collusion_score = 0.0
        self.suspicious_pairs.clear()
        self.current_threshold = self.base_threshold
        
        # Reset history
        self.vote_history.clear()
        self.collusion_history.clear()
        self.coordination_events.clear()
        
        # Reset behavioral tracking
        self.pair_agreement_history.clear()
        self.member_behavior_profiles.clear()
        self.temporal_patterns.clear()
        self.alert_patterns.clear()
        
        # Reset statistics
        self.detection_stats = {
            'total_checks': 0,
            'alerts_raised': 0,
            'false_positive_rate': 0.0,
            'confirmed_collusion_events': 0,
            'avg_pair_similarity': 0.0,
            'member_independence_scores': {},
            'detection_accuracy': 0.95,
            'alert_frequency': 0.0,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Reset quality metrics
        self.quality_metrics = {
            'detection_precision': 0.0,
            'detection_recall': 0.0,
            'behavioral_accuracy': 0.0,
            'temporal_consistency': 0.0,
            'overall_effectiveness': 0.0
        }
        
        # Reset alert system
        self.alert_system['last_alerts'].clear()
        if 'alert_history' in self.alert_system:
            self.alert_system['alert_history'].clear()
        
        # Reset error state
        self.error_count = 0
        self.is_disabled = False
        
        self.logger.info(format_operator_message(
            icon="🔄",
            message="Collusion Auditor reset completed",
            status="All detection state cleared and systems reinitialized"
        ))

    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(format_operator_message(
                    icon="👋",
                    message="Collusion Auditor shutting down",
                    total_checks=self.detection_stats.get('total_checks', 0),
                    alerts_raised=self.detection_stats.get('alerts_raised', 0)
                ))
        except Exception:
            pass  # Ignore cleanup errors