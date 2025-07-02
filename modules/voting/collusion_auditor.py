# ─────────────────────────────────────────────────────────────
# File: modules/voting/collusion_auditor.py
# Enhanced Collusion Auditor with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class CollusionAuditor(Module, AnalysisMixin, StateManagementMixin):
    """
    Enhanced collusion auditor with InfoBus integration.
    Detects suspicious voting patterns and coordination between committee members.
    Provides robust anti-manipulation safeguards for the voting system.
    """

    def __init__(
        self,
        n_members: int,
        window: int = 10,
        threshold: float = 0.9,
        adaptive_threshold: bool = True,
        similarity_methods: List[str] = None,
        debug: bool = False,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 200),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        
        # Core parameters
        self.n_members = int(n_members)
        self.window = int(window)
        self.base_threshold = float(threshold)
        self.current_threshold = self.base_threshold
        self.adaptive_threshold = bool(adaptive_threshold)
        
        # Similarity analysis methods
        if similarity_methods is None:
            self.similarity_methods = ['cosine', 'correlation', 'euclidean']
        else:
            self.similarity_methods = similarity_methods
        
        # Collusion detection state
        self.vote_history = deque(maxlen=window * 2)
        self.collusion_score = 0.0
        self.suspicious_pairs = set()
        self.collusion_history = deque(maxlen=100)
        
        # Advanced detection features
        self.pair_agreement_history = defaultdict(lambda: deque(maxlen=window))
        self.member_behavior_profiles = defaultdict(lambda: {
            'avg_similarity': 0.0,
            'volatility': 0.0,
            'consistency_score': 0.0,
            'independence_score': 1.0
        })
        
        # Temporal analysis
        self.temporal_patterns = defaultdict(list)
        self.coordination_events = deque(maxlen=50)
        
        # Statistical tracking
        self.collusion_stats = {
            'total_checks': 0,
            'alerts_raised': 0,
            'false_positive_rate': 0.0,
            'confirmed_collusion_events': 0,
            'avg_pair_similarity': 0.0,
            'member_independence_scores': {}
        }
        
        # Alert system
        self.alert_cooldown = 10  # Steps between alerts for same pair
        self.last_alerts = defaultdict(int)
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "CollusionAuditor",
            "logs/voting/collusion_auditor.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("CollusionAuditor")
        
        self.log_operator_info(
            "🕵️ Collusion Auditor initialized",
            members=self.n_members,
            window=self.window,
            threshold=self.base_threshold,
            methods=len(self.similarity_methods)
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset detection state
        self.collusion_score = 0.0
        self.suspicious_pairs.clear()
        self.current_threshold = self.base_threshold
        
        # Reset history
        self.vote_history.clear()
        self.collusion_history.clear()
        self.coordination_events.clear()
        
        # Reset tracking
        self.pair_agreement_history.clear()
        self.member_behavior_profiles.clear()
        self.temporal_patterns.clear()
        
        # Reset statistics
        self.collusion_stats = {
            'total_checks': 0,
            'alerts_raised': 0,
            'false_positive_rate': 0.0,
            'confirmed_collusion_events': 0,
            'avg_pair_similarity': 0.0,
            'member_independence_scores': {}
        }
        
        # Reset alerts
        self.last_alerts.clear()
        
        self.log_operator_info("🔄 Collusion Auditor reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using standalone mode")
            return
        
        # Extract context and voting data
        context = extract_standard_context(info_bus)
        voting_data = self._extract_voting_data_from_info_bus(info_bus)
        
        # Update threshold based on market conditions
        self._update_adaptive_threshold(context, voting_data)
        
        # Analyze member behavior patterns
        self._analyze_member_behavior_patterns(voting_data)
        
        # Update temporal analysis
        self._update_temporal_analysis(voting_data, context)
        
        # Update InfoBus with collusion analysis
        self._update_info_bus(info_bus)

    def _extract_voting_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract voting data for collusion analysis"""
        
        data = {}
        
        try:
            # Get votes
            votes = info_bus.get('votes', [])
            data['votes'] = votes
            data['vote_count'] = len(votes)
            
            # Extract member proposals if available
            module_data = info_bus.get('module_data', {})
            arbiter_data = module_data.get('strategy_arbiter', {})
            data['raw_proposals'] = arbiter_data.get('raw_proposals', [])
            data['member_confidences'] = arbiter_data.get('confidences', [])
            data['alpha_weights'] = arbiter_data.get('last_alpha', [])
            
            # Get consensus information
            voting_summary = InfoBusExtractor.get_voting_summary(info_bus)
            data['consensus_direction'] = voting_summary.get('consensus_direction', 'neutral')
            data['agreement_score'] = voting_summary.get('agreement_score', 0.5)
            
            # Extract performance context
            recent_trades = info_bus.get('recent_trades', [])
            data['recent_performance'] = self._calculate_recent_performance(recent_trades)
            
        except Exception as e:
            self.log_operator_warning(f"Voting data extraction failed: {e}")
            data = {
                'votes': [],
                'vote_count': 0,
                'raw_proposals': [],
                'member_confidences': [],
                'alpha_weights': [],
                'consensus_direction': 'neutral',
                'agreement_score': 0.5,
                'recent_performance': 0.0
            }
        
        return data

    def _calculate_recent_performance(self, recent_trades: List[Dict]) -> float:
        """Calculate recent trading performance"""
        
        if not recent_trades:
            return 0.0
        
        try:
            recent_pnl = [trade.get('pnl', 0) for trade in recent_trades[-10:]]
            return float(np.mean(recent_pnl))
        except Exception:
            return 0.0

    def _update_adaptive_threshold(self, context: Dict[str, Any], 
                                  voting_data: Dict[str, Any]) -> None:
        """Update detection threshold based on market conditions"""
        
        if not self.adaptive_threshold:
            return
        
        try:
            # Base threshold adjustments
            regime = context.get('regime', 'unknown')
            volatility_level = context.get('volatility_level', 'medium')
            agreement_score = voting_data.get('agreement_score', 0.5)
            
            # Calculate adjustment factor
            adjustment = 1.0
            
            # In volatile markets, slightly lower threshold (easier to detect collusion)
            if regime == 'volatile':
                adjustment *= 0.95
            elif regime == 'trending':
                adjustment *= 1.05  # Stricter in trending markets
            
            # Adjust based on natural agreement
            if agreement_score > 0.8:
                adjustment *= 1.1  # Stricter when high natural agreement
            elif agreement_score < 0.3:
                adjustment *= 0.9  # More lenient when low natural agreement
            
            # Apply adjustment
            old_threshold = self.current_threshold
            self.current_threshold = self.base_threshold * adjustment
            self.current_threshold = np.clip(self.current_threshold, 0.7, 0.98)
            
            # Log significant changes
            if abs(self.current_threshold - old_threshold) > 0.05:
                self.log_operator_info(
                    f"🎯 Collusion threshold adjusted: {old_threshold:.3f} → {self.current_threshold:.3f}",
                    regime=regime,
                    agreement=f"{agreement_score:.1%}"
                )
            
        except Exception as e:
            self.log_operator_warning(f"Threshold adaptation failed: {e}")

    def check_collusion(self, actions: List[np.ndarray]) -> float:
        """
        Enhanced collusion detection with multiple similarity metrics.
        
        Args:
            actions: List of action vectors from committee members
            
        Returns:
            Collusion score between 0 (no collusion) and 1 (definite collusion)
        """
        
        self.collusion_stats['total_checks'] += 1
        
        if len(actions) < 2:
            return 0.0
        
        try:
            # Add to history
            timestamp = datetime.datetime.now().isoformat()
            vote_entry = {
                'timestamp': timestamp,
                'actions': [action.copy() for action in actions],
                'n_members': len(actions)
            }
            self.vote_history.append(vote_entry)
            
            # Need sufficient history
            if len(self.vote_history) < 3:
                return 0.0
            
            # Analyze with multiple methods
            similarity_scores = self._calculate_comprehensive_similarities(actions)
            
            # Update pair agreement history
            self._update_pair_agreements(similarity_scores)
            
            # Find suspicious pairs
            old_suspicious = self.suspicious_pairs.copy()
            self.suspicious_pairs.clear()
            suspicious_count = 0
            
            for pair, similarities in similarity_scores.items():
                # Use multiple similarity measures
                avg_similarity = np.mean(list(similarities.values()))
                
                # Get historical agreement for this pair
                pair_history = self.pair_agreement_history[pair]
                if len(pair_history) >= 3:
                    historical_avg = np.mean(list(pair_history))
                    
                    # Check if consistently above threshold
                    if historical_avg > self.current_threshold:
                        self.suspicious_pairs.add(pair)
                        suspicious_count += 1
                        
                        # Alert if new or after cooldown
                        if (pair not in old_suspicious or 
                            self.collusion_stats['total_checks'] - self.last_alerts.get(pair, 0) > self.alert_cooldown):
                            
                            self._raise_collusion_alert(pair, historical_avg, similarities)
                            self.last_alerts[pair] = self.collusion_stats['total_checks']
            
            # Calculate overall collusion score
            max_pairs = self.n_members * (self.n_members - 1) / 2
            self.collusion_score = suspicious_count / max_pairs if max_pairs > 0 else 0.0
            
            # Record collusion event
            collusion_event = {
                'timestamp': timestamp,
                'score': self.collusion_score,
                'suspicious_pairs': list(self.suspicious_pairs),
                'similarity_scores': {str(k): v for k, v in similarity_scores.items()},
                'threshold_used': self.current_threshold
            }
            self.collusion_history.append(collusion_event)
            
            # Update statistics
            self._update_collusion_statistics(similarity_scores)
            
            return self.collusion_score
            
        except Exception as e:
            self.log_operator_error(f"Collusion check failed: {e}")
            return 0.0

    def _calculate_comprehensive_similarities(self, actions: List[np.ndarray]) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Calculate similarities using multiple methods"""
        
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
                        pair_similarities['cosine'] = float(cosine_sim)
                    
                    # Correlation
                    if 'correlation' in self.similarity_methods:
                        if len(v1) > 1 and np.std(v1) > 0 and np.std(v2) > 0:
                            correlation = np.corrcoef(v1, v2)[0, 1]
                            if not np.isnan(correlation):
                                pair_similarities['correlation'] = float(correlation)
                    
                    # Inverse Euclidean distance (normalized)
                    if 'euclidean' in self.similarity_methods:
                        distance = np.linalg.norm(v1 - v2)
                        max_distance = np.linalg.norm(v1) + np.linalg.norm(v2)
                        if max_distance > 0:
                            euclidean_sim = 1.0 - (distance / max_distance)
                            pair_similarities['euclidean'] = float(euclidean_sim)
                
                similarities[pair] = pair_similarities
        
        return similarities

    def _update_pair_agreements(self, similarity_scores: Dict[Tuple[int, int], Dict[str, float]]) -> None:
        """Update historical pair agreement tracking"""
        
        for pair, similarities in similarity_scores.items():
            if similarities:
                # Use average of available similarity measures
                avg_similarity = np.mean(list(similarities.values()))
                self.pair_agreement_history[pair].append(avg_similarity)

    def _raise_collusion_alert(self, pair: Tuple[int, int], similarity_score: float, 
                              similarities: Dict[str, float]) -> None:
        """Raise collusion alert with detailed information"""
        
        member_i, member_j = pair
        
        # Format similarity details
        sim_details = ", ".join([f"{method}: {score:.3f}" for method, score in similarities.items()])
        
        self.log_operator_warning(
            f"🚨 Potential collusion detected between members {member_i} and {member_j}",
            similarity=f"{similarity_score:.3f}",
            threshold=f"{self.current_threshold:.3f}",
            details=sim_details,
            action_required="Monitor these members closely"
        )
        
        # Update statistics
        self.collusion_stats['alerts_raised'] += 1
        
        # Record coordination event
        self.coordination_events.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'pair': pair,
            'similarity': similarity_score,
            'similarities': similarities.copy(),
            'alert_type': 'collusion_detection'
        })

    def _analyze_member_behavior_patterns(self, voting_data: Dict[str, Any]) -> None:
        """Analyze individual member behavior patterns"""
        
        try:
            raw_proposals = voting_data.get('raw_proposals', [])
            if len(raw_proposals) < 2:
                return
            
            # Analyze each member's behavior
            for i, proposal in enumerate(raw_proposals):
                if i >= self.n_members:
                    break
                
                profile = self.member_behavior_profiles[i]
                
                # Calculate similarity with all other members
                similarities = []
                for j, other_proposal in enumerate(raw_proposals):
                    if i != j and np.linalg.norm(proposal) > 0 and np.linalg.norm(other_proposal) > 0:
                        sim = np.dot(proposal, other_proposal) / (
                            np.linalg.norm(proposal) * np.linalg.norm(other_proposal)
                        )
                        similarities.append(sim)
                
                if similarities:
                    # Update member profile
                    profile['avg_similarity'] = np.mean(similarities)
                    profile['volatility'] = np.std(similarities)
                    
                    # Independence score (lower similarity = higher independence)
                    profile['independence_score'] = max(0.0, 1.0 - profile['avg_similarity'])
                    
                    # Consistency score (lower volatility = higher consistency)
                    profile['consistency_score'] = max(0.0, 1.0 - profile['volatility'])
                
                # Update global statistics
                self.collusion_stats['member_independence_scores'][f'member_{i}'] = profile['independence_score']
            
        except Exception as e:
            self.log_operator_warning(f"Member behavior analysis failed: {e}")

    def _update_temporal_analysis(self, voting_data: Dict[str, Any], 
                                 context: Dict[str, Any]) -> None:
        """Update temporal pattern analysis"""
        
        try:
            # Track patterns over time
            timestamp = datetime.datetime.now().isoformat()
            regime = context.get('regime', 'unknown')
            
            pattern_entry = {
                'timestamp': timestamp,
                'regime': regime,
                'collusion_score': self.collusion_score,
                'suspicious_pairs_count': len(self.suspicious_pairs),
                'avg_agreement': voting_data.get('agreement_score', 0.5)
            }
            
            self.temporal_patterns[regime].append(pattern_entry)
            
            # Keep only recent patterns (last 50 per regime)
            if len(self.temporal_patterns[regime]) > 50:
                self.temporal_patterns[regime] = self.temporal_patterns[regime][-50:]
            
        except Exception as e:
            self.log_operator_warning(f"Temporal analysis update failed: {e}")

    def _update_collusion_statistics(self, similarity_scores: Dict[Tuple[int, int], Dict[str, float]]) -> None:
        """Update comprehensive collusion statistics"""
        
        try:
            # Calculate average pair similarity
            all_similarities = []
            for similarities in similarity_scores.values():
                all_similarities.extend(similarities.values())
            
            if all_similarities:
                self.collusion_stats['avg_pair_similarity'] = float(np.mean(all_similarities))
            
            # Update performance metrics
            self._update_performance_metric('collusion_score', self.collusion_score)
            self._update_performance_metric('suspicious_pairs_count', len(self.suspicious_pairs))
            self._update_performance_metric('avg_pair_similarity', self.collusion_stats['avg_pair_similarity'])
            
        except Exception as e:
            self.log_operator_warning(f"Statistics update failed: {e}")

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with collusion analysis results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'collusion_auditor', {
            'collusion_score': self.collusion_score,
            'suspicious_pairs': list(self.suspicious_pairs),
            'threshold': self.current_threshold,
            'base_threshold': self.base_threshold,
            'adaptive_threshold': self.adaptive_threshold,
            'n_members': self.n_members,
            'window': self.window,
            'statistics': self.collusion_stats.copy(),
            'member_profiles': {
                str(k): v for k, v in self.member_behavior_profiles.items()
            },
            'alert_status': {
                'recent_alerts': len([e for e in self.coordination_events if 
                                    (datetime.datetime.now() - 
                                     datetime.datetime.fromisoformat(e['timestamp'])).seconds < 300]),
                'cooldown_pairs': len(self.last_alerts)
            }
        })
        
        # Add alerts for significant collusion
        if self.collusion_score > 0.7:
            InfoBusUpdater.add_alert(
                info_bus,
                f"High collusion risk detected: {self.collusion_score:.1%}",
                severity="warning",
                module="CollusionAuditor"
            )
        elif len(self.suspicious_pairs) > 0:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Monitoring {len(self.suspicious_pairs)} suspicious member pairs",
                severity="info",
                module="CollusionAuditor"
            )

    def get_collusion_report(self) -> str:
        """Generate operator-friendly collusion report"""
        
        # Risk assessment
        if self.collusion_score > 0.8:
            risk_level = "🚨 HIGH RISK"
        elif self.collusion_score > 0.5:
            risk_level = "⚠️ MODERATE RISK"
        elif self.collusion_score > 0.2:
            risk_level = "🟡 LOW RISK"
        else:
            risk_level = "✅ CLEAN"
        
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
            if independence < 0.7:
                independence_summary.append(f"  ⚠️ Member {member_id}: {independence:.1%} independence")
        
        return f"""
🕵️ COLLUSION AUDITOR
═══════════════════════════════════════
🎯 Current Status: {risk_level}
📊 Collusion Score: {self.collusion_score:.1%}
🎚️ Detection Threshold: {self.current_threshold:.1%}

📈 Detection Statistics:
• Total Checks: {self.collusion_stats['total_checks']}
• Alerts Raised: {self.collusion_stats['alerts_raised']}
• Recent Alerts (10min): {recent_alerts}
• Average Pair Similarity: {self.collusion_stats.get('avg_pair_similarity', 0):.1%}

🔍 Suspicious Activity:
• Suspicious Pairs: {len(self.suspicious_pairs)}
• Coordination Events: {len(self.coordination_events)}
• Members Under Watch: {len(self.last_alerts)}

📊 Configuration:
• Committee Size: {self.n_members} members
• Analysis Window: {self.window} votes
• Similarity Methods: {', '.join(self.similarity_methods)}
• Adaptive Threshold: {'✅ Enabled' if self.adaptive_threshold else '❌ Disabled'}

🔍 Suspicious Pairs:
{chr(10).join(suspicious_details) if suspicious_details else "  ✅ No suspicious pairs detected"}

⚠️ Low Independence Members:
{chr(10).join(independence_summary) if independence_summary else "  ✅ All members showing good independence"}

📊 Recent Activity:
• Vote History: {len(self.vote_history)} entries
• Behavior Profiles: {len(self.member_behavior_profiles)} members
• Temporal Patterns: {len(self.temporal_patterns)} regimes tracked

🎯 Alert System:
• Alert Cooldown: {self.alert_cooldown} steps
• Active Cooldowns: {len(self.last_alerts)}
        """

    def get_member_independence_scores(self) -> Dict[int, float]:
        """Get independence scores for all members"""
        
        scores = {}
        for member_id, profile in self.member_behavior_profiles.items():
            scores[member_id] = profile.get('independence_score', 1.0)
        
        return scores

    def get_observation_components(self) -> np.ndarray:
        """Return collusion features for observation"""
        
        try:
            features = [
                float(self.collusion_score),
                float(len(self.suspicious_pairs) / max(self.n_members, 1)),
                float(self.current_threshold),
                float(self.collusion_stats.get('avg_pair_similarity', 0)),
                float(len(self.vote_history) / self.window)
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.array([0.0, 0.0, 0.9, 0.5, 0.0], dtype=np.float32)

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "n_members": self.n_members,
                "window": self.window,
                "base_threshold": self.base_threshold,
                "adaptive_threshold": self.adaptive_threshold,
                "similarity_methods": self.similarity_methods
            },
            "detection_state": {
                "collusion_score": self.collusion_score,
                "current_threshold": self.current_threshold,
                "suspicious_pairs": list(self.suspicious_pairs)
            },
            "statistics": self.collusion_stats.copy(),
            "member_profiles": {
                str(k): v for k, v in self.member_behavior_profiles.items()
            },
            "history": {
                "vote_history": list(self.vote_history)[-20:],
                "collusion_history": list(self.collusion_history)[-20:],
                "coordination_events": list(self.coordination_events)[-10:]
            },
            "alerts": {
                "last_alerts": dict(self.last_alerts),
                "alert_cooldown": self.alert_cooldown
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.n_members = int(config.get("n_members", self.n_members))
        self.window = int(config.get("window", self.window))
        self.base_threshold = float(config.get("base_threshold", self.base_threshold))
        self.adaptive_threshold = bool(config.get("adaptive_threshold", self.adaptive_threshold))
        self.similarity_methods = config.get("similarity_methods", self.similarity_methods)
        
        # Load detection state
        detection_state = state.get("detection_state", {})
        self.collusion_score = float(detection_state.get("collusion_score", 0.0))
        self.current_threshold = float(detection_state.get("current_threshold", self.base_threshold))
        
        suspicious_pairs = detection_state.get("suspicious_pairs", [])
        self.suspicious_pairs = set(tuple(pair) for pair in suspicious_pairs)
        
        # Load statistics
        self.collusion_stats.update(state.get("statistics", {}))
        
        # Load member profiles
        member_profiles = state.get("member_profiles", {})
        self.member_behavior_profiles.clear()
        for member_id, profile in member_profiles.items():
            self.member_behavior_profiles[int(member_id)] = profile
        
        # Load history
        history = state.get("history", {})
        
        vote_history = history.get("vote_history", [])
        self.vote_history.clear()
        for entry in vote_history:
            self.vote_history.append(entry)
            
        collusion_history = history.get("collusion_history", [])
        self.collusion_history.clear()
        for entry in collusion_history:
            self.collusion_history.append(entry)
            
        coordination_events = history.get("coordination_events", [])
        self.coordination_events.clear()
        for entry in coordination_events:
            self.coordination_events.append(entry)
        
        # Load alerts
        alerts = state.get("alerts", {})
        last_alerts = alerts.get("last_alerts", {})
        self.last_alerts = defaultdict(int)
        for pair_str, step in last_alerts.items():
            pair = tuple(map(int, pair_str.strip('()').split(', ')))
            self.last_alerts[pair] = step
        
        self.alert_cooldown = int(alerts.get("alert_cooldown", self.alert_cooldown))