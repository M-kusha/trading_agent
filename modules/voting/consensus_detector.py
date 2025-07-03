# ─────────────────────────────────────────────────────────────
# File: modules/voting/consensus_detector.py
# Enhanced Consensus Detector with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class ConsensusDetector(Module, AnalysisMixin, StateManagementMixin):
    """
    Enhanced consensus detector with InfoBus integration.
    Computes consensus levels from member actions and confidences with
    sophisticated agreement analysis and consensus quality metrics.
    """

    def __init__(
        self,
        n_members: int,
        threshold: float = 0.6,
        consensus_methods: List[str] = None,
        quality_weighting: bool = True,
        temporal_smoothing: bool = True,
        debug: bool = False,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 100),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        
        # Core parameters
        self.n_members = int(n_members)
        self.threshold = float(threshold)
        self.quality_weighting = bool(quality_weighting)
        self.temporal_smoothing = bool(temporal_smoothing)
        
        # Consensus calculation methods
        if consensus_methods is None:
            self.consensus_methods = ['cosine_agreement', 'direction_alignment', 'confidence_weighted']
        else:
            self.consensus_methods = consensus_methods
        
        # Consensus state
        self.last_consensus = 0.0
        self.consensus_history = deque(maxlen=100)
        self.consensus_quality = 0.0
        self.consensus_components = {}
        
        # Advanced consensus analysis
        self.directional_consensus = 0.0
        self.magnitude_consensus = 0.0
        self.confidence_consensus = 0.0
        self.temporal_stability = 0.0
        
        # Member contribution tracking
        self.member_contributions = defaultdict(lambda: {
            'avg_alignment': 0.5,
            'consistency': 0.5,
            'influence_weight': 1.0 / max(n_members, 1)
        })
        
        # Consensus quality metrics
        self.quality_metrics = {
            'coherence': 0.5,
            'stability': 0.5,
            'diversity': 0.5,
            'reliability': 0.5
        }
        
        # Temporal analysis
        self.consensus_trends = deque(maxlen=50)
        self.regime_consensus_history = defaultdict(lambda: deque(maxlen=30))
        
        # Statistics
        self.consensus_stats = {
            'total_computations': 0,
            'high_consensus_count': 0,
            'low_consensus_count': 0,
            'avg_consensus': 0.5,
            'consensus_volatility': 0.0,
            'quality_score': 0.5
        }
        
        # Smoothing parameters
        self.smoothing_alpha = 0.3
        self.stability_window = 10
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "ConsensusDetector",
            "logs/voting/consensus_detector.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("ConsensusDetector")
        
        self.log_operator_info(
            "🤝 Consensus Detector initialized",
            members=self.n_members,
            threshold=self.threshold,
            methods=len(self.consensus_methods),
            quality_weighting=self.quality_weighting
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset consensus state
        self.last_consensus = 0.0
        self.consensus_quality = 0.0
        self.directional_consensus = 0.0
        self.magnitude_consensus = 0.0
        self.confidence_consensus = 0.0
        self.temporal_stability = 0.0
        
        # Reset history
        self.consensus_history.clear()
        self.consensus_trends.clear()
        self.regime_consensus_history.clear()
        
        # Reset tracking
        self.member_contributions.clear()
        self.consensus_components.clear()
        
        # Reset metrics
        self.quality_metrics = {
            'coherence': 0.5,
            'stability': 0.5,
            'diversity': 0.5,
            'reliability': 0.5
        }
        
        # Reset statistics
        self.consensus_stats = {
            'total_computations': 0,
            'high_consensus_count': 0,
            'low_consensus_count': 0,
            'avg_consensus': 0.5,
            'consensus_volatility': 0.0,
            'quality_score': 0.5
        }
        
        self.log_operator_info("🔄 Consensus Detector reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using standalone mode")
            return
        
        # Extract context and voting data
        context = extract_standard_context(info_bus)
        voting_data = self._extract_voting_data_from_info_bus(info_bus)
        
        # Analyze consensus trends
        self._analyze_consensus_trends(context)
        
        # Update member contribution analysis
        self._update_member_contributions(voting_data)
        
        # Update quality metrics
        self._update_quality_metrics(voting_data, context)
        
        # Update InfoBus with consensus analysis
        self._update_info_bus(info_bus)

    def _extract_voting_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract voting data for consensus analysis"""
        
        data = {}
        
        try:
            # Get votes and proposals
            votes = info_bus.get('votes', [])
            data['votes'] = votes
            data['vote_count'] = len(votes)
            
            # Extract detailed voting information
            module_data = info_bus.get('module_data', {})
            arbiter_data = module_data.get('strategy_arbiter', {})
            data['raw_proposals'] = arbiter_data.get('raw_proposals', [])
            data['member_confidences'] = arbiter_data.get('confidences', [])
            data['alpha_weights'] = arbiter_data.get('last_alpha', [])
            data['blended_action'] = arbiter_data.get('blended_action', [])
            
            # Get voting summary
            voting_summary = InfoBusExtractor.get_voting_summary(info_bus)
            data['existing_consensus'] = voting_summary.get('agreement_score', 0.5)
            data['consensus_direction'] = voting_summary.get('consensus_direction', 'neutral')
            
        except Exception as e:
            self.log_operator_warning(f"Voting data extraction failed: {e}")
            data = {
                'votes': [],
                'vote_count': 0,
                'raw_proposals': [],
                'member_confidences': [],
                'alpha_weights': [],
                'blended_action': [],
                'existing_consensus': 0.5,
                'consensus_direction': 'neutral'
            }
        
        return data

    def compute_consensus(self, actions: List[np.ndarray], confidences: List[float]) -> float:
        """
        Enhanced consensus computation using multiple methods.
        
        Args:
            actions: List of action vectors from committee members
            confidences: List of confidence scores for each member
            
        Returns:
            Consensus level between 0 (no consensus) and 1 (perfect consensus)
        """
        
        self.consensus_stats['total_computations'] += 1
        
        if not actions or len(actions) < 2:
            self.last_consensus = 0.5
            return self.last_consensus
        
        try:
            # Validate inputs
            actions = [np.asarray(action, dtype=np.float32).flatten() for action in actions]
            confidences = [max(0.0, min(1.0, float(c))) for c in confidences]
            
            # Ensure equal lengths
            min_len = min(len(actions), len(confidences))
            actions = actions[:min_len]
            confidences = confidences[:min_len]
            
            if min_len < 2:
                self.last_consensus = 0.5
                return self.last_consensus
            
            # Calculate consensus components
            consensus_components = {}
            
            # 1. Cosine agreement (directional consensus)
            if 'cosine_agreement' in self.consensus_methods:
                consensus_components['cosine'] = self._calculate_cosine_agreement(actions, confidences)
            
            # 2. Direction alignment (same direction votes)
            if 'direction_alignment' in self.consensus_methods:
                consensus_components['direction'] = self._calculate_direction_alignment(actions, confidences)
            
            # 3. Confidence-weighted agreement
            if 'confidence_weighted' in self.consensus_methods:
                consensus_components['confidence'] = self._calculate_confidence_weighted_consensus(actions, confidences)
            
            # 4. Magnitude consensus (similar action strengths)
            consensus_components['magnitude'] = self._calculate_magnitude_consensus(actions, confidences)
            
            # Combine consensus measures
            if consensus_components:
                if self.quality_weighting:
                    # Weight by quality indicators
                    weights = self._calculate_component_weights(consensus_components, actions, confidences)
                    consensus = sum(score * weights.get(method, 1.0) 
                                   for method, score in consensus_components.items())
                    consensus = consensus / sum(weights.values())
                else:
                    # Simple average
                    consensus = np.mean(list(consensus_components.values()))
            else:
                consensus = 0.5
            
            # Apply temporal smoothing if enabled
            if self.temporal_smoothing and len(self.consensus_history) > 0:
                previous_consensus = self.consensus_history[-1]['consensus']
                consensus = (self.smoothing_alpha * consensus + 
                           (1 - self.smoothing_alpha) * previous_consensus)
            
            # Clip to valid range
            consensus = float(np.clip(consensus, 0.0, 1.0))
            
            # Store components for analysis
            self.consensus_components = consensus_components
            self.directional_consensus = consensus_components.get('direction', 0.5)
            self.magnitude_consensus = consensus_components.get('magnitude', 0.5)
            self.confidence_consensus = consensus_components.get('confidence', 0.5)
            
            # Update consensus state
            self.last_consensus = consensus
            
            # Record consensus event
            consensus_event = {
                'timestamp': datetime.datetime.now().isoformat(),
                'consensus': consensus,
                'components': consensus_components.copy(),
                'n_members': len(actions),
                'avg_confidence': np.mean(confidences),
                'action_count': len(actions)
            }
            self.consensus_history.append(consensus_event)
            
            # Update statistics
            self._update_consensus_statistics(consensus, consensus_components)
            
            # Calculate consensus quality
            self._calculate_consensus_quality(actions, confidences, consensus_components)
            
            return consensus
            
        except Exception as e:
            self.log_operator_error(f"Consensus computation failed: {e}")
            self.last_consensus = 0.5
            return self.last_consensus

    def _calculate_cosine_agreement(self, actions: List[np.ndarray], confidences: List[float]) -> float:
        """Calculate consensus based on cosine similarity between actions"""
        
        try:
            agreements = []
            weights = []
            
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    a1, a2 = actions[i], actions[j]
                    
                    # Only calculate if both have magnitude
                    norm1, norm2 = np.linalg.norm(a1), np.linalg.norm(a2)
                    if norm1 > 1e-6 and norm2 > 1e-6:
                        similarity = np.dot(a1, a2) / (norm1 * norm2)
                        # Convert to 0-1 scale (cosine ranges from -1 to 1)
                        agreement = (similarity + 1.0) / 2.0
                        
                        # Weight by confidence
                        weight = confidences[i] * confidences[j]
                        
                        agreements.append(agreement)
                        weights.append(weight)
            
            if agreements and sum(weights) > 0:
                weighted_agreement = np.average(agreements, weights=weights)
                return float(weighted_agreement)
            else:
                return 0.5
                
        except Exception as e:
            self.log_operator_warning(f"Cosine agreement calculation failed: {e}")
            return 0.5

    def _calculate_direction_alignment(self, actions: List[np.ndarray], confidences: List[float]) -> float:
        """Calculate consensus based on directional alignment"""
        
        try:
            # Get primary direction for each action (first component sign)
            directions = []
            valid_confidences = []
            
            for action, confidence in zip(actions, confidences):
                if len(action) > 0 and abs(action[0]) > 1e-6:
                    directions.append(np.sign(action[0]))
                    valid_confidences.append(confidence)
            
            if len(directions) < 2:
                return 0.5
            
            # Calculate agreement in direction
            positive_weight = sum(conf for dir, conf in zip(directions, valid_confidences) if dir > 0)
            negative_weight = sum(conf for dir, conf in zip(directions, valid_confidences) if dir < 0)
            total_weight = positive_weight + negative_weight
            
            if total_weight > 0:
                # Stronger agreement = higher consensus
                alignment = abs(positive_weight - negative_weight) / total_weight
                return float(alignment)
            else:
                return 0.5
                
        except Exception as e:
            self.log_operator_warning(f"Direction alignment calculation failed: {e}")
            return 0.5

    def _calculate_confidence_weighted_consensus(self, actions: List[np.ndarray], confidences: List[float]) -> float:
        """Calculate consensus weighted by member confidence"""
        
        try:
            if not confidences:
                return 0.5
            
            # Higher variance in confidence = lower consensus
            conf_variance = np.var(confidences)
            conf_mean = np.mean(confidences)
            
            # Normalize variance by mean to get relative spread
            if conf_mean > 0:
                relative_variance = conf_variance / (conf_mean ** 2)
                # Convert to consensus (lower variance = higher consensus)
                confidence_consensus = max(0.0, 1.0 - relative_variance * 2)
            else:
                confidence_consensus = 0.5
            
            return float(confidence_consensus)
            
        except Exception as e:
            self.log_operator_warning(f"Confidence weighted consensus calculation failed: {e}")
            return 0.5

    def _calculate_magnitude_consensus(self, actions: List[np.ndarray], confidences: List[float]) -> float:
        """Calculate consensus based on action magnitude similarity"""
        
        try:
            magnitudes = [np.linalg.norm(action) for action in actions]
            
            if len(magnitudes) < 2:
                return 0.5
            
            # Calculate coefficient of variation
            mean_magnitude = np.mean(magnitudes)
            if mean_magnitude > 0:
                std_magnitude = np.std(magnitudes)
                cv = std_magnitude / mean_magnitude
                # Convert to consensus (lower CV = higher consensus)
                magnitude_consensus = max(0.0, 1.0 - cv)
            else:
                magnitude_consensus = 0.5
            
            return float(magnitude_consensus)
            
        except Exception as e:
            self.log_operator_warning(f"Magnitude consensus calculation failed: {e}")
            return 0.5

    def _calculate_component_weights(self, consensus_components: Dict[str, float], 
                                   actions: List[np.ndarray], confidences: List[float]) -> Dict[str, float]:
        """Calculate weights for different consensus components"""
        
        weights = {}
        
        try:
            # Base weights
            for method in consensus_components:
                weights[method] = 1.0
            
            # Adjust based on data quality
            avg_confidence = np.mean(confidences)
            
            # If high confidence, trust directional measures more
            if avg_confidence > 0.8:
                weights['direction'] = weights.get('direction', 1.0) * 1.2
                weights['cosine'] = weights.get('cosine', 1.0) * 1.2
            
            # If low confidence, rely more on magnitude consensus
            if avg_confidence < 0.4:
                weights['magnitude'] = weights.get('magnitude', 1.0) * 1.3
                weights['confidence'] = weights.get('confidence', 1.0) * 1.3
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
        except Exception as e:
            self.log_operator_warning(f"Component weight calculation failed: {e}")
            # Fallback to equal weights
            weights = {method: 1.0 / len(consensus_components) for method in consensus_components}
        
        return weights

    def _calculate_consensus_quality(self, actions: List[np.ndarray], confidences: List[float], 
                                   components: Dict[str, float]) -> None:
        """Calculate overall consensus quality metrics"""
        
        try:
            # Coherence: How well do components agree
            if len(components) > 1:
                component_values = list(components.values())
                coherence = 1.0 - np.std(component_values) / max(np.mean(component_values), 1e-6)
                self.quality_metrics['coherence'] = max(0.0, min(1.0, coherence))
            
            # Stability: How stable is consensus over time
            if len(self.consensus_history) >= self.stability_window:
                recent_consensus = [h['consensus'] for h in list(self.consensus_history)[-self.stability_window:]]
                stability = 1.0 - np.std(recent_consensus)
                self.quality_metrics['stability'] = max(0.0, min(1.0, stability))
            
            # Diversity: Are we getting diverse inputs
            if len(actions) > 1:
                action_diversity = self._calculate_action_diversity(actions)
                self.quality_metrics['diversity'] = action_diversity
            
            # Reliability: Based on confidence levels
            if confidences:
                avg_confidence = np.mean(confidences)
                conf_consistency = 1.0 - np.std(confidences)
                reliability = (avg_confidence + conf_consistency) / 2.0
                self.quality_metrics['reliability'] = max(0.0, min(1.0, reliability))
            
            # Overall quality score
            self.consensus_quality = np.mean(list(self.quality_metrics.values()))
            
        except Exception as e:
            self.log_operator_warning(f"Quality calculation failed: {e}")

    def _calculate_action_diversity(self, actions: List[np.ndarray]) -> float:
        """Calculate diversity of actions"""
        
        try:
            if len(actions) < 2:
                return 0.0
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    dist = np.linalg.norm(actions[i] - actions[j])
                    distances.append(dist)
            
            if distances:
                # Normalize by maximum possible distance
                max_dist = max(distances)
                if max_dist > 0:
                    avg_distance = np.mean(distances) / max_dist
                    return float(min(1.0, avg_distance))
            
            return 0.5
            
        except Exception as e:
            self.log_operator_warning(f"Action diversity calculation failed: {e}")
            return 0.5

    def _analyze_consensus_trends(self, context: Dict[str, Any]) -> None:
        """Analyze consensus trends over time"""
        
        try:
            if len(self.consensus_history) < 5:
                return
            
            # Calculate trend
            recent_consensus = [h['consensus'] for h in list(self.consensus_history)[-10:]]
            if len(recent_consensus) >= 3:
                # Simple linear trend
                x = np.arange(len(recent_consensus))
                trend_slope = np.polyfit(x, recent_consensus, 1)[0]
                
                trend_entry = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'trend_slope': float(trend_slope),
                    'current_consensus': self.last_consensus,
                    'regime': context.get('regime', 'unknown')
                }
                self.consensus_trends.append(trend_entry)
                
                # Track by regime
                regime = context.get('regime', 'unknown')
                self.regime_consensus_history[regime].append(self.last_consensus)
            
        except Exception as e:
            self.log_operator_warning(f"Trend analysis failed: {e}")

    def _update_member_contributions(self, voting_data: Dict[str, Any]) -> None:
        """Update member contribution analysis"""
        
        try:
            raw_proposals = voting_data.get('raw_proposals', [])
            confidences = voting_data.get('member_confidences', [])
            
            if len(raw_proposals) < 2:
                return
            
            # Analyze each member's contribution to consensus
            for i, proposal in enumerate(raw_proposals):
                if i >= self.n_members:
                    break
                
                contribution = self.member_contributions[i]
                
                # Calculate alignment with others
                alignments = []
                for j, other_proposal in enumerate(raw_proposals):
                    if i != j and np.linalg.norm(proposal) > 0 and np.linalg.norm(other_proposal) > 0:
                        alignment = np.dot(proposal, other_proposal) / (
                            np.linalg.norm(proposal) * np.linalg.norm(other_proposal)
                        )
                        alignments.append((alignment + 1.0) / 2.0)  # Convert to 0-1
                
                if alignments:
                    contribution['avg_alignment'] = np.mean(alignments)
                    contribution['consistency'] = 1.0 - np.std(alignments)
                
                # Update influence weight based on confidence and alignment
                if i < len(confidences):
                    confidence = confidences[i]
                    alignment = contribution['avg_alignment']
                    contribution['influence_weight'] = (confidence + alignment) / 2.0
            
        except Exception as e:
            self.log_operator_warning(f"Member contribution update failed: {e}")

    def _update_consensus_statistics(self, consensus: float, components: Dict[str, float]) -> None:
        """Update consensus statistics"""
        
        try:
            # Update counts
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
                recent_consensus = [h['consensus'] for h in list(self.consensus_history)[-10:]]
                self.consensus_stats['consensus_volatility'] = float(np.std(recent_consensus))
            
            # Update quality score
            self.consensus_stats['quality_score'] = self.consensus_quality
            
            # Update performance metrics
            self._update_performance_metric('consensus_score', consensus)
            self._update_performance_metric('consensus_quality', self.consensus_quality)
            self._update_performance_metric('directional_consensus', self.directional_consensus)
            
        except Exception as e:
            self.log_operator_warning(f"Statistics update failed: {e}")

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with consensus analysis results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'consensus_detector', {
            'consensus_score': self.last_consensus,
            'consensus_quality': self.consensus_quality,
            'threshold': self.threshold,
            'n_members': self.n_members,
            'consensus_components': self.consensus_components.copy(),
            'directional_consensus': self.directional_consensus,
            'magnitude_consensus': self.magnitude_consensus,
            'confidence_consensus': self.confidence_consensus,
            'temporal_stability': self.temporal_stability,
            'quality_metrics': self.quality_metrics.copy(),
            'statistics': self.consensus_stats.copy(),
            'member_contributions': {
                str(k): v for k, v in self.member_contributions.items()
            }
        })
        
        # Update main consensus score in InfoBus
        if 'consensus' not in info_bus:
            info_bus['consensus'] = {}
        
        info_bus['consensus'].update({
            'score': self.last_consensus,
            'quality': self.consensus_quality,
            'components': self.consensus_components.copy(),
            'direction': 'high' if self.last_consensus > 0.7 else 'low' if self.last_consensus < 0.3 else 'moderate'
        })
        
        # Add alerts for extreme consensus conditions
        if self.last_consensus > 0.9:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Very high consensus detected: {self.last_consensus:.1%}",
                severity="info",
                module="ConsensusDetector"
            )
        elif self.last_consensus < 0.2:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Very low consensus detected: {self.last_consensus:.1%}",
                severity="warning",
                module="ConsensusDetector"
            )

    def resize(self, n_members: int) -> None:
        """Resize for different number of members"""
        old_members = self.n_members
        self.n_members = int(n_members)
        
        # Clear member-specific data if size changed significantly
        if abs(self.n_members - old_members) > 2:
            self.member_contributions.clear()
            
        self.log_operator_info(
            f"🔄 Consensus Detector resized: {old_members} → {self.n_members} members"
        )

    def get_consensus_report(self) -> str:
        """Generate operator-friendly consensus report"""
        
        # Consensus level assessment
        if self.last_consensus > 0.8:
            consensus_level = "🟢 HIGH"
        elif self.last_consensus > 0.6:
            consensus_level = "🟡 MODERATE"
        elif self.last_consensus > 0.4:
            consensus_level = "🟠 LOW-MODERATE"
        elif self.last_consensus > 0.2:
            consensus_level = "🟠 LOW"
        else:
            consensus_level = "🔴 VERY LOW"
        
        # Quality assessment
        if self.consensus_quality > 0.8:
            quality_level = "✅ Excellent"
        elif self.consensus_quality > 0.6:
            quality_level = "⚡ Good"
        elif self.consensus_quality > 0.4:
            quality_level = "⚠️ Fair"
        else:
            quality_level = "🚨 Poor"
        
        # Trend analysis
        if len(self.consensus_trends) > 0:
            recent_trend = self.consensus_trends[-1]
            trend_slope = recent_trend.get('trend_slope', 0)
            if trend_slope > 0.01:
                trend_direction = "📈 Improving"
            elif trend_slope < -0.01:
                trend_direction = "📉 Declining"
            else:
                trend_direction = "→ Stable"
        else:
            trend_direction = "📭 No data"
        
        # Component breakdown
        component_lines = []
        for method, score in self.consensus_components.items():
            if score > 0.7:
                emoji = "✅"
            elif score > 0.5:
                emoji = "⚡"
            elif score > 0.3:
                emoji = "⚠️"
            else:
                emoji = "🚨"
            component_lines.append(f"  {emoji} {method.replace('_', ' ').title()}: {score:.1%}")
        
        return f"""
🤝 CONSENSUS DETECTOR
═══════════════════════════════════════
📊 Current Consensus: {consensus_level} ({self.last_consensus:.1%})
🎯 Quality Level: {quality_level} ({self.consensus_quality:.1%})
📈 Trend: {trend_direction}

📊 Consensus Components:
{chr(10).join(component_lines) if component_lines else "  📭 No components available"}

🎯 Quality Metrics:
• Coherence: {self.quality_metrics.get('coherence', 0):.1%}
• Stability: {self.quality_metrics.get('stability', 0):.1%}
• Diversity: {self.quality_metrics.get('diversity', 0):.1%}
• Reliability: {self.quality_metrics.get('reliability', 0):.1%}

📈 Performance Statistics:
• Total Computations: {self.consensus_stats['total_computations']}
• High Consensus Events: {self.consensus_stats['high_consensus_count']}
• Low Consensus Events: {self.consensus_stats['low_consensus_count']}
• Average Consensus: {self.consensus_stats['avg_consensus']:.1%}
• Consensus Volatility: {self.consensus_stats['consensus_volatility']:.3f}

⚙️ Configuration:
• Members: {self.n_members}
• Threshold: {self.threshold:.1%}
• Methods: {', '.join(self.consensus_methods)}
• Quality Weighting: {'✅ Enabled' if self.quality_weighting else '❌ Disabled'}
• Temporal Smoothing: {'✅ Enabled' if self.temporal_smoothing else '❌ Disabled'}

📊 Recent Activity:
• History Length: {len(self.consensus_history)}
• Trend Data Points: {len(self.consensus_trends)}
• Member Contributions: {len(self.member_contributions)} tracked
        """

    def _get_observation_impl(self) -> np.ndarray:
            """
            Provides the consensus-based components for the RL agent's observation.
            This is the required implementation for the Module abstract base class.
            """
            try:
                # This logic is copied directly from your get_observation_components method
                features = [
                    float(self.last_consensus),
                    float(self.consensus_quality),
                    float(self.directional_consensus),
                    float(self.magnitude_consensus),
                    float(self.confidence_consensus)
                ]
                
                return np.array(features, dtype=np.float32)
                
            except Exception as e:
                self.log_operator_error(f"Observation generation failed: {e}")
                # Return a zero vector of the correct shape on failure
                return np.zeros(5, dtype=np.float32)

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "n_members": self.n_members,
                "threshold": self.threshold,
                "consensus_methods": self.consensus_methods,
                "quality_weighting": self.quality_weighting,
                "temporal_smoothing": self.temporal_smoothing
            },
            "consensus_state": {
                "last_consensus": self.last_consensus,
                "consensus_quality": self.consensus_quality,
                "directional_consensus": self.directional_consensus,
                "magnitude_consensus": self.magnitude_consensus,
                "confidence_consensus": self.confidence_consensus,
                "temporal_stability": self.temporal_stability
            },
            "consensus_components": self.consensus_components.copy(),
            "quality_metrics": self.quality_metrics.copy(),
            "statistics": self.consensus_stats.copy(),
            "member_contributions": {
                str(k): v for k, v in self.member_contributions.items()
            },
            "history": {
                "consensus_history": list(self.consensus_history)[-20:],
                "consensus_trends": list(self.consensus_trends)[-10:],
                "regime_consensus": {
                    regime: list(history)[-10:] 
                    for regime, history in self.regime_consensus_history.items()
                }
            },
            "parameters": {
                "smoothing_alpha": self.smoothing_alpha,
                "stability_window": self.stability_window
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.n_members = int(config.get("n_members", self.n_members))
        self.threshold = float(config.get("threshold", self.threshold))
        self.consensus_methods = config.get("consensus_methods", self.consensus_methods)
        self.quality_weighting = bool(config.get("quality_weighting", self.quality_weighting))
        self.temporal_smoothing = bool(config.get("temporal_smoothing", self.temporal_smoothing))
        
        # Load consensus state
        consensus_state = state.get("consensus_state", {})
        self.last_consensus = float(consensus_state.get("last_consensus", 0.0))
        self.consensus_quality = float(consensus_state.get("consensus_quality", 0.0))
        self.directional_consensus = float(consensus_state.get("directional_consensus", 0.0))
        self.magnitude_consensus = float(consensus_state.get("magnitude_consensus", 0.0))
        self.confidence_consensus = float(consensus_state.get("confidence_consensus", 0.0))
        self.temporal_stability = float(consensus_state.get("temporal_stability", 0.0))
        
        # Load components and metrics
        self.consensus_components = state.get("consensus_components", {})
        self.quality_metrics.update(state.get("quality_metrics", {}))
        self.consensus_stats.update(state.get("statistics", {}))
        
        # Load member contributions
        member_contributions = state.get("member_contributions", {})
        self.member_contributions.clear()
        for member_id, contribution in member_contributions.items():
            self.member_contributions[int(member_id)] = contribution
        
        # Load history
        history = state.get("history", {})
        
        consensus_history = history.get("consensus_history", [])
        self.consensus_history.clear()
        for entry in consensus_history:
            self.consensus_history.append(entry)
            
        consensus_trends = history.get("consensus_trends", [])
        self.consensus_trends.clear()
        for entry in consensus_trends:
            self.consensus_trends.append(entry)
            
        regime_consensus = history.get("regime_consensus", {})
        self.regime_consensus_history.clear()
        for regime, history_list in regime_consensus.items():
            regime_deque = deque(maxlen=30)
            for entry in history_list:
                regime_deque.append(entry)
            self.regime_consensus_history[regime] = regime_deque
        
        # Load parameters
        parameters = state.get("parameters", {})
        self.smoothing_alpha = float(parameters.get("smoothing_alpha", self.smoothing_alpha))
        self.stability_window = int(parameters.get("stability_window", self.stability_window))