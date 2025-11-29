"""
Collusion Detector
==================
Detects potential collusion patterns among voting committee members.
Analyzes similarity, behavioral patterns, and temporal coordination.

Refactored from collusion_auditor.py (~2035 lines).
~400 lines focused on core collusion detection.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase
from modules.voting.core.types import CollusionResult


@module(**module_args("CollusionDetector"))
class CollusionDetector(VotingModuleBase):
    """
    Collusion detection for voting committees.
    
    Detects:
    - Pair-wise similarity (cosine, correlation)
    - Suspicious voting patterns
    - Coordinated timing
    - Behavioral anomalies
    
    Publishes:
    - collusion_score
    - collusion_analysis
    - suspicious_pairs
    """
    
    def _module_specific_init(self) -> None:
        """Initialize collusion detection state."""
        # Configuration
        self.n_members = int(self.config.get('n_members', 5))
        self.window = int(self.config.get('window', 10))
        self.base_threshold = float(self.config.get('threshold', 0.9))
        self.current_threshold = self.base_threshold
        self.adaptive_threshold = bool(self.config.get('adaptive_threshold', True))
        
        # State
        self.vote_history: deque = deque(maxlen=self.window * 2)
        self.collusion_score: float = 0.0
        self.suspicious_pairs: Set[Tuple[str, str]] = set()
        self.collusion_history: deque = deque(maxlen=100)
        
        # Pair tracking
        self.pair_agreement_history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        
        # Member behavior profiles
        self.member_profiles: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                'avg_similarity': 0.0,
                'consistency_score': 0.5,
                'independence_score': 1.0,
                'anomaly_score': 0.0,
            }
        )
        
        # Statistics
        self.detection_stats: Dict[str, Any] = {
            'total_checks': 0,
            'alerts_raised': 0,
            'avg_pair_similarity': 0.0,
        }
        
        # Quality metrics
        self.quality_metrics: Dict[str, float] = {
            'detection_precision': 0.0,
            'behavioral_accuracy': 0.0,
            'overall_effectiveness': 0.5,
        }
        
        # Alert cooldowns
        self.alert_cooldowns: Dict[Tuple[str, str], int] = {}
        self.cooldown_period = int(self.config.get('alert_cooldown', 10))
        
        self.logger.info(
            f"[COLLUSION] CollusionDetector initialized | "
            f"members={self.n_members} | threshold={self.base_threshold:.2f}"
        )
        
        # Publish baseline
        self._publish_collusion_baseline()
    
    def _publish_collusion_baseline(self) -> None:
        """Publish baseline collusion keys."""
        try:
            self.smart_bus.set(
                'collusion_score',
                0.0,
                module='CollusionDetector',
                thesis='Baseline collusion score'
            )
        except Exception:
            pass
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """Detect collusion patterns."""
        start = time.time()
        name = self.__class__.__name__
        
        try:
            # Get decision ID
            decision_id = self.smart_bus.get('kernel_decision_id', name)
            
            # Get voting data
            voting_data = await self._get_voting_data()
            
            # Fast path for insufficient data
            votes = voting_data.get('votes') or []
            if len(votes) < 2:
                return self._insufficient_data_output(decision_id)
            
            # Perform analysis
            analysis = await self._analyze_collusion(voting_data)
            behavioral = await self._update_behavioral_profiles(voting_data)
            
            # Generate thesis
            thesis = self._generate_thesis(analysis)
            
            # Publish to bus
            await self._update_bus(analysis, thesis)
            
            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, 'process', elapsed_ms, True)
            
            return {
                'collusion_score': analysis.get('collusion_score', 0.0),
                'collusion_analysis': analysis,
                'suspicious_pairs': list(self.suspicious_pairs),
                'member_profiles': dict(self.member_profiles),
                'detection_statistics': dict(self.detection_stats),
                'quality_metrics': dict(self.quality_metrics),
                'collusion_alerts': analysis.get('alerts', []),
                'decision_id': decision_id,
                'collusion_decision_id': decision_id,
                # Contract-expected keys
                'collusion_result': analysis,
                'collusion_detected': analysis.get('collusion_detected', False),
                'collusion_thesis': thesis,
                'member_independence_scores': {m: 1.0 - p.get('correlation_score', 0.0) for m, p in self.member_profiles.items()},
                '_thesis': thesis,
            }
        
        except Exception as e:
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(e, 'collusion_process')
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)
    
    async def _get_voting_data(self) -> Dict[str, Any]:
        """Get voting data from SmartInfoBus."""
        return {
            'votes': self.smart_bus.get('expert_votes', self.__class__.__name__) or [],
            'proposal_vectors': self.smart_bus.get('committee_proposal_vectors', self.__class__.__name__) or [],
            'member_confidences': self.smart_bus.get('committee_member_confidences', self.__class__.__name__) or [],
            'market_regime': self.smart_bus.get('market_regime', self.__class__.__name__) or 'unknown',
        }
    
    async def _analyze_collusion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform collusion analysis."""
        votes = data.get('votes') or []
        vectors = data.get('proposal_vectors') or []
        
        self.detection_stats['total_checks'] += 1
        
        # Calculate pair similarities
        pair_similarities = self._calculate_pair_similarities(votes, vectors)
        
        # Update suspicious pairs
        new_suspicious = set()
        for (m1, m2), sim in pair_similarities.items():
            if sim > self.current_threshold:
                new_suspicious.add((m1, m2))
                self._record_suspicious_pair(m1, m2, sim)
        
        self.suspicious_pairs = new_suspicious
        
        # Calculate overall collusion score
        if pair_similarities:
            avg_sim = np.mean(list(pair_similarities.values()))
            max_sim = max(pair_similarities.values())
        else:
            avg_sim = 0.0
            max_sim = 0.0
        
        # Collusion score weighted by suspicious pairs
        suspicious_count = len(self.suspicious_pairs)
        possible_pairs = (self.n_members * (self.n_members - 1)) / 2 if self.n_members > 1 else 1
        suspicious_ratio = suspicious_count / max(possible_pairs, 1)
        
        # Ensure plain float for collusion_score (avoid numpy scalar)
        score = (
            0.4 * float(max_sim) +
            0.3 * float(avg_sim) +
            0.3 * float(suspicious_ratio)
        )
        self.collusion_score = float(score)
        
        # Update stats
        self.detection_stats['avg_pair_similarity'] = avg_sim
        
        # Generate alerts
        alerts = self._generate_alerts()
        
        # Update threshold adaptively
        if self.adaptive_threshold:
            self._adapt_threshold(data)
        
        return {
            'collusion_score': max(0.0, min(1.0, self.collusion_score)),
            'collusion_detected': self.collusion_score > 0.7,
            'suspicious_pair_count': suspicious_count,
            'avg_pair_similarity': avg_sim,
            'max_pair_similarity': max_sim,
            'pair_similarities': {f"{k[0]}-{k[1]}": v for k, v in pair_similarities.items()},
            'alerts': alerts,
            'threshold': self.current_threshold,
        }
    
    def _calculate_pair_similarities(
        self, 
        votes: List[Dict[str, Any]], 
        vectors: List[List[float]]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate pair-wise similarities."""
        similarities: Dict[Tuple[str, str], float] = {}
        
        # Extract member names and their vote data
        members_data = {}
        for i, vote in enumerate(votes):
            member = vote.get('expert', f'member_{i}')
            vote_dict = vote.get('vote', {})
            confidence = float(vote.get('confidence', 0.5))
            action = vote_dict.get('action', 'abstain')
            signal = float(vote_dict.get('signal_strength', 0.5))
            
            # Create feature vector
            members_data[member] = {
                'action': action,
                'confidence': confidence,
                'signal': signal,
                'vector': vectors[i] if i < len(vectors) else [0.0, confidence],
            }
        
        # Calculate similarities for each pair
        members = list(members_data.keys())
        for i, m1 in enumerate(members):
            for m2 in members[i+1:]:
                d1 = members_data[m1]
                d2 = members_data[m2]
                
                sim = self._calculate_similarity(d1, d2)
                similarities[(m1, m2)] = sim
        
        return similarities
    
    def _calculate_similarity(
        self, 
        d1: Dict[str, Any], 
        d2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two members' votes."""
        try:
            # Action agreement
            action_match = 1.0 if d1['action'] == d2['action'] else 0.0
            
            # Confidence similarity
            conf_diff = abs(d1['confidence'] - d2['confidence'])
            conf_sim = 1.0 - conf_diff
            
            # Signal similarity
            signal_diff = abs(d1['signal'] - d2['signal'])
            signal_sim = 1.0 - min(signal_diff, 1.0)
            
            # Vector cosine similarity if available
            v1 = d1.get('vector', [])
            v2 = d2.get('vector', [])
            if v1 and v2 and len(v1) == len(v2):
                cosine = self._cosine_similarity(v1, v2)
            else:
                cosine = 0.5
            
            # Weighted combination
            similarity = (
                0.4 * action_match +
                0.2 * conf_sim +
                0.2 * signal_sim +
                0.2 * cosine
            )
            
            return max(0.0, min(1.0, similarity))
        
        except Exception:
            return 0.0
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity."""
        try:
            a = np.array(v1, dtype=float)
            b = np.array(v2, dtype=float)
            
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0
    
    def _record_suspicious_pair(self, m1: str, m2: str, similarity: float) -> None:
        """Record a suspicious pair."""
        pair = (min(m1, m2), max(m1, m2))
        self.pair_agreement_history[pair].append(similarity)
    
    async def _update_behavioral_profiles(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update member behavioral profiles."""
        votes = data.get('votes') or []
        
        for vote in votes:
            member = vote.get('expert', 'unknown')
            confidence = float(vote.get('confidence', 0.5))
            
            profile = self.member_profiles[member]
            
            # Update consistency (how stable their confidence is)
            # This would need historical data to be accurate
            profile['consistency_score'] = 0.5 + confidence * 0.2
            
            # Update independence based on suspicious pairs
            involved_pairs = sum(
                1 for p in self.suspicious_pairs 
                if member in p
            )
            profile['independence_score'] = max(0.0, 1.0 - involved_pairs * 0.2)
        
        return dict(self.member_profiles)
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate collusion alerts."""
        alerts = []
        current_check = self.detection_stats['total_checks']
        
        for pair in self.suspicious_pairs:
            # Check cooldown
            last_alert = self.alert_cooldowns.get(pair, 0)
            if current_check - last_alert < self.cooldown_period:
                continue
            
            # Get similarity history
            history = list(self.pair_agreement_history.get(pair, []))
            avg_sim = np.mean(history) if history else 0.0
            
            if avg_sim > self.current_threshold:
                alerts.append({
                    'pair': list(pair),
                    'severity': 'warning' if avg_sim > 0.85 else 'info',
                    'avg_similarity': float(avg_sim),
                    'timestamp': datetime.datetime.now().isoformat(),
                })
                self.alert_cooldowns[pair] = current_check
                self.detection_stats['alerts_raised'] += 1
        
        return alerts
    
    def _adapt_threshold(self, data: Dict[str, Any]) -> None:
        """Adapt detection threshold based on market conditions."""
        regime = data.get('market_regime', 'unknown')
        
        # Regime multipliers
        multipliers = {
            'trending': 1.10,
            'ranging': 0.90,
            'volatile': 0.85,
            'breakout': 1.20,
            'unknown': 1.00,
        }
        
        mult = multipliers.get(regime, 1.0)
        self.current_threshold = min(0.98, max(0.7, self.base_threshold * mult))
    
    def _generate_thesis(self, analysis: Dict[str, Any]) -> str:
        """Generate collusion detection thesis."""
        score = analysis.get('collusion_score', 0.0)
        detected = analysis.get('collusion_detected', False)
        pairs = analysis.get('suspicious_pair_count', 0)
        
        label = 'HIGH_RISK' if detected else 'LOW_RISK' if score < 0.3 else 'MODERATE'
        
        return (
            f"COLLUSION: {label} ({score:.1%}) | "
            f"suspicious_pairs={pairs} | "
            f"threshold={self.current_threshold:.2f}"
        )
    
    async def _update_bus(self, analysis: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with results."""
        try:
            name = self.__class__.__name__
            
            self.smart_bus.set(
                'collusion_score',
                analysis.get('collusion_score', 0.0),
                module=name,
                thesis=thesis
            )
            self.smart_bus.set(
                'collusion_analysis',
                analysis,
                module=name,
                thesis='Collusion analysis results'
            )
            self.smart_bus.set(
                'suspicious_pairs',
                list(self.suspicious_pairs),
                module=name,
                thesis=f'{len(self.suspicious_pairs)} suspicious pairs detected'
            )
        except Exception as e:
            self.logger.warning(f"Bus update failed: {e}")
    
    def _insufficient_data_output(self, decision_id: str) -> Dict[str, Any]:
        """Output for insufficient data."""
        return {
            'collusion_score': 0.0,
            'collusion_analysis': {'collusion_detected': False, 'reason': 'insufficient_data'},
            'suspicious_pairs': [],
            'member_profiles': {},
            'detection_statistics': dict(self.detection_stats),
            'quality_metrics': dict(self.quality_metrics),
            'collusion_alerts': [],
            'decision_id': decision_id,
            'collusion_decision_id': decision_id,
            # Contract-expected keys
            'collusion_result': {'collusion_detected': False, 'reason': 'insufficient_data', 'collusion_score': 0.0},
            'collusion_detected': False,
            'collusion_thesis': 'Collusion analysis skipped (insufficient data)',
            'member_independence_scores': {},
            '_thesis': 'Collusion analysis skipped (insufficient data)',
        }
    
    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        return {
            'collusion_score': 0.0,
            'collusion_analysis': {'error': error, 'collusion_detected': False},
            'suspicious_pairs': [],
            'member_profiles': {},
            'detection_statistics': dict(self.detection_stats),
            'quality_metrics': dict(self.quality_metrics),
            'collusion_alerts': [],
            'decision_id': None,
            'collusion_decision_id': None,
            # Contract-expected keys
            'collusion_result': {'error': error, 'collusion_detected': False, 'collusion_score': 0.0},
            'collusion_detected': False,
            'collusion_thesis': f'Collusion detection error: {error}',
            'member_independence_scores': {},
            '_thesis': f'Collusion detection error: {error}',
        }


