"""
Consensus Analyzer
==================
Analyzes consensus among voting proposals using multiple dimensions:
direction, magnitude, confidence, and temporal stability.

Refactored from consensus_detector.py (~2150 lines).
~400 lines focused on core consensus analysis.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase


@module(**module_args("ConsensusAnalyzer"))
class ConsensusAnalyzer(VotingModuleBase):
    """
    Consensus analysis for voting committees.
    
    Analyzes multiple dimensions:
    - Directional consensus (bullish/bearish alignment)
    - Magnitude consensus (signal strength agreement)
    - Confidence consensus (member confidence agreement)
    - Temporal stability (consistency over time)
    
    Publishes:
    - consensus_score
    - consensus_analysis
    - consensus_quality_metrics
    """
    
    def _module_specific_init(self) -> None:
        """Initialize consensus-specific state."""
        # Configuration
        self.n_members = int(self.config.get('n_members', 5))
        self.threshold = float(self.config.get('threshold', 0.6))
        self.quality_weighting = bool(self.config.get('quality_weighting', True))
        self.temporal_smoothing = bool(self.config.get('temporal_smoothing', True))
        self.smoothing_alpha = float(self.config.get('smoothing_alpha', 0.3))
        
        # State
        self.last_consensus: float = 0.0
        self.consensus_history: deque = deque(maxlen=150)
        
        # Dimension scores
        self.directional_consensus: float = 0.0     # 0–1 magnitude agreement
        self.directional_consensus_sign: float = 0.0  # -1 short, 0 neutral, +1 long
        self.magnitude_consensus: float = 0.0
        self.confidence_consensus: float = 0.0
        self.temporal_stability: float = 0.0
        
        # Quality metrics
        self.consensus_quality_metrics: Dict[str, float] = {
            'coherence': 0.5,
            'stability': 0.5,
            'diversity': 0.5,
            'reliability': 0.5,
            'overall_effectiveness': 0.5,
        }
        
        # Statistics
        self.consensus_stats: Dict[str, Any] = {
            'total_computations': 0,
            'high_consensus_count': 0,
            'low_consensus_count': 0,
            'avg_consensus': 0.5,
        }
        
        # Member contributions (placeholder for future enhancement)
        self.member_contributions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                'avg_alignment': 0.5,
                'consistency': 0.5,
                'reliability_score': 0.5,
            }
        )
        
        self.logger.info(
            f"[CONSENSUS] ConsensusAnalyzer initialized | "
            f"members={self.n_members} | threshold={self.threshold:.2f}"
        )
        
        # Publish baseline
        self._publish_consensus_baseline()
    
    def _publish_consensus_baseline(self) -> None:
        """Publish baseline consensus keys."""
        try:
            self.smart_bus.set(
                'consensus_score',
                0.0,
                module='ConsensusAnalyzer',
                thesis='Baseline consensus score'
            )
        except Exception:
            pass
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """Analyze consensus from voting data."""
        start = time.time()
        name = self.__class__.__name__
        
        try:
            # Get decision ID for coordination
            decision_id = self.smart_bus.get('kernel_decision_id', name)
            
            # Get voting data
            voting_data = await self._get_voting_data()
            
            # Perform analysis
            analysis = await self._analyze_consensus(voting_data)
            quality = await self._calculate_quality_metrics(voting_data, analysis)
            
            # Update state
            self._update_history(analysis)
            
            # Generate thesis
            thesis = self._generate_thesis(analysis, quality)
            
            # Publish to bus
            await self._update_bus(analysis, quality, thesis)
            
            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, 'process', elapsed_ms, True)

            # Derive a symbolic consensus direction from the stored sign
            direction_label = 'neutral'
            if analysis.get('consensus_exists', False):
                if self.directional_consensus_sign > 0:
                    direction_label = 'long'
                elif self.directional_consensus_sign < 0:
                    direction_label = 'short'
            
            return {
                'consensus_score': analysis['consensus_score'],
                'consensus_analysis': analysis,
                'consensus_quality_metrics': quality,
                'consensus_statistics': dict(self.consensus_stats),
                'directional_consensus': self.directional_consensus,
                'magnitude_consensus': self.magnitude_consensus,
                'confidence_consensus': self.confidence_consensus,
                'temporal_stability': self.temporal_stability,
                'member_contributions': dict(self.member_contributions),
                'decision_id': decision_id,
                'consensus_decision_id': decision_id,
                # Contract-expected keys
                'consensus_result': analysis,
                'agreement_score': analysis.get('consensus_score', 0.0),
                'consensus_direction': direction_label,
                'consensus_confidence': analysis.get('consensus_score', 0.0),
                'consensus_components': {
                    'directional': self.directional_consensus,
                    'magnitude': self.magnitude_consensus,
                    'confidence': self.confidence_consensus,
                    'temporal': self.temporal_stability
                },
                'consensus_quality': quality,
                'consensus_thesis': thesis,
                '_thesis': thesis,
            }
        
        except Exception as e:
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(e, 'consensus_process')
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)
    
    async def _get_voting_data(self) -> Dict[str, Any]:
        """Get voting data from SmartInfoBus."""
        return {
            'proposal_vectors': self.smart_bus.get('committee_proposal_vectors', self.__class__.__name__) or [],
            'member_confidences': self.smart_bus.get('committee_member_confidences', self.__class__.__name__) or [],
            'expert_votes': self.smart_bus.get('expert_votes', self.__class__.__name__) or [],
            'committee_members': self.smart_bus.get('committee_members', self.__class__.__name__) or [],
            'market_regime': self.smart_bus.get('market_regime', self.__class__.__name__) or 'unknown',
        }
    
    async def _analyze_consensus(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-dimensional consensus analysis."""
        vectors = data.get('proposal_vectors') or []
        confidences = data.get('member_confidences') or []
        
        if len(vectors) < 2:
            # Reset sign in case of sparse data
            self.directional_consensus_sign = 0.0
            self.directional_consensus = 0.0
            self.magnitude_consensus = 0.0
            self.confidence_consensus = 0.0
            return {
                'consensus_score': 0.0,
                'consensus_exists': False,
                'reason': 'insufficient_data',
            }
        
        # Directional consensus (sign agreement)
        self.directional_consensus = self._calculate_directional_consensus(vectors)
        
        # Magnitude consensus (strength agreement)
        self.magnitude_consensus = self._calculate_magnitude_consensus(vectors)
        
        # Confidence consensus (confidence agreement)
        self.confidence_consensus = self._calculate_confidence_consensus(confidences)
        
        # Combine with weights
        raw_score = (
            0.4 * self.directional_consensus +
            0.25 * self.magnitude_consensus +
            0.2 * self.confidence_consensus +
            0.15 * self.temporal_stability
        )
        
        # Apply temporal smoothing
        if self.temporal_smoothing and self.last_consensus > 0:
            alpha = self.smoothing_alpha
            consensus_score = alpha * raw_score + (1 - alpha) * self.last_consensus
        else:
            consensus_score = raw_score
        
        consensus_score = max(0.0, min(1.0, consensus_score))
        self.last_consensus = consensus_score
        
        return {
            'consensus_score': consensus_score,
            'consensus_exists': consensus_score >= self.threshold,
            'directional_consensus': self.directional_consensus,
            'direction_sign': self.directional_consensus_sign,
            'magnitude_consensus': self.magnitude_consensus,
            'confidence_consensus': self.confidence_consensus,
            'temporal_stability': self.temporal_stability,
            'raw_score': raw_score,
            'vote_count': len(vectors),
        }
    
    def _calculate_directional_consensus(self, vectors: List[List[float]]) -> float:
        """Calculate directional consensus using sign agreement (0–1 magnitude) 
        and store the majority direction sign (-1/0/+1)."""
        try:
            if len(vectors) < 2:
                self.directional_consensus_sign = 0.0
                return 0.0
            
            # Extract direction signs from vectors
            directions: List[int] = []
            for v in vectors:
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    val = float(v[0])
                    if val > 0.1:
                        directions.append(1)
                    elif val < -0.1:
                        directions.append(-1)
                    else:
                        directions.append(0)
            
            if not directions:
                self.directional_consensus_sign = 0.0
                return 0.0
            
            from collections import Counter
            counts = Counter(directions)
            most_common = counts.most_common(1)
            if not most_common:
                self.directional_consensus_sign = 0.0
                return 0.0
            
            majority_value, majority_count = most_common[0]
            
            # If the majority is "neutral" (0), treat as no directional consensus
            if majority_value == 0:
                self.directional_consensus_sign = 0.0
                return 0.0
            
            self.directional_consensus_sign = float(majority_value)
            agreement = majority_count / len(directions)
            return float(max(0.0, min(1.0, agreement)))
        
        except Exception:
            self.directional_consensus_sign = 0.0
            return 0.0
    
    def _calculate_magnitude_consensus(self, vectors: List[List[float]]) -> float:
        """Calculate magnitude consensus using coefficient of variation."""
        try:
            if len(vectors) < 2:
                return 0.0
            
            # Extract magnitudes (absolute directional strength)
            magnitudes: List[float] = []
            for v in vectors:
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    magnitudes.append(abs(float(v[0])))
            
            if len(magnitudes) < 2:
                return 0.0
            
            mean = float(np.mean(magnitudes))
            std = float(np.std(magnitudes))
            
            if mean <= 0:
                return 0.5
            
            cv: float = std / mean
            # Map CV to consensus (CV of 0 = 1.0, CV of 1 = 0.0, clamp)
            return float(max(0.0, min(1.0, 1.0 - cv)))
        
        except Exception:
            return 0.0
    
    def _calculate_confidence_consensus(self, confidences: List[float]) -> float:
        """Calculate confidence consensus via coefficient of variation."""
        try:
            if len(confidences) < 2:
                return 0.0
            
            conf_values = [float(c) for c in confidences if isinstance(c, (int, float))]
            if len(conf_values) < 2:
                return 0.0
            
            mean = float(np.mean(conf_values))
            std = float(np.std(conf_values))
            
            if mean <= 0:
                return 0.5
            
            cv: float = std / mean
            return float(max(0.0, min(1.0, 1.0 - cv)))
        
        except Exception:
            return 0.0
    
    async def _calculate_quality_metrics(
        self, 
        data: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate consensus quality metrics."""
        try:
            score = float(analysis.get('consensus_score', 0.0))
            
            # Coherence: how well do dimensions agree
            dimensions: List[float] = [
                self.directional_consensus,
                self.magnitude_consensus,
                self.confidence_consensus,
            ]
            coherence = float(1.0 - float(np.std(dimensions))) if dimensions else 0.5
            
            # Stability: consistency over time
            if len(self.consensus_history) >= 3:
                recent = list(self.consensus_history)[-5:]
                stability = float(1.0 - float(np.std(recent))) if len(recent) > 1 else 0.5
                self.temporal_stability = float(stability)
            else:
                stability = 0.5
            
            # Diversity: inverse of “too extreme” consensus
            if score > 0.5:
                diversity = float(1.0 - abs(score - 0.5) * 2)
            else:
                diversity = 0.5
            
            # Overall effectiveness
            effectiveness = float((coherence + stability + diversity) / 3.0)
            
            self.consensus_quality_metrics = {
                'coherence': float(max(0.0, min(1.0, coherence))),
                'stability': float(max(0.0, min(1.0, stability))),
                'diversity': float(max(0.0, min(1.0, diversity))),
                'reliability': float(score),
                'overall_effectiveness': float(max(0.0, min(1.0, effectiveness))),
            }
            
            return self.consensus_quality_metrics
        
        except Exception:
            return self.consensus_quality_metrics
    
    def _update_history(self, analysis: Dict[str, Any]) -> None:
        """Update consensus history and stats."""
        score = float(analysis.get('consensus_score', 0.0))
        self.consensus_history.append(score)
        
        # Update stats
        self.consensus_stats['total_computations'] += 1
        if score >= self.threshold:
            self.consensus_stats['high_consensus_count'] += 1
        else:
            self.consensus_stats['low_consensus_count'] += 1
        
        n = self.consensus_stats['total_computations']
        old_avg = float(self.consensus_stats['avg_consensus'])
        self.consensus_stats['avg_consensus'] = (old_avg * (n - 1) + score) / n
    
    def _generate_thesis(self, analysis: Dict[str, Any], quality: Dict[str, float]) -> str:
        """Generate consensus thesis string."""
        score = float(analysis.get('consensus_score', 0.0))
        exists = bool(analysis.get('consensus_exists', False))
        
        label = 'STRONG' if score > 0.7 else 'MODERATE' if score > 0.4 else 'WEAK'
        
        return (
            f"CONSENSUS: {label} ({score:.1%}) | "
            f"exists={exists} | "
            f"directional={self.directional_consensus:.2f} | "
            f"magnitude={self.magnitude_consensus:.2f} | "
            f"quality={quality.get('overall_effectiveness', 0.5):.2f}"
        )
    
    async def _update_bus(
        self, 
        analysis: Dict[str, Any], 
        quality: Dict[str, float], 
        thesis: str
    ) -> None:
        """Update SmartInfoBus with results."""
        try:
            name = self.__class__.__name__
            
            self.smart_bus.set(
                'consensus_score',
                analysis.get('consensus_score', 0.0),
                module=name,
                thesis=thesis
            )
            self.smart_bus.set(
                'consensus_analysis',
                analysis,
                module=name,
                thesis='Consensus analysis results'
            )
            self.smart_bus.set(
                'consensus_quality_metrics',
                quality,
                module=name,
                thesis='Quality metrics'
            )
        except Exception as e:
            self.logger.warning(f"Bus update failed: {e}")
    
    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        return {
            'consensus_score': 0.0,
            'consensus_analysis': {'error': error, 'consensus_exists': False},
            'consensus_quality_metrics': self.consensus_quality_metrics,
            'consensus_statistics': dict(self.consensus_stats),
            'directional_consensus': 0.0,
            'magnitude_consensus': 0.0,
            'confidence_consensus': 0.0,
            'temporal_stability': 0.0,
            'member_contributions': {},
            'decision_id': None,
            'consensus_decision_id': None,
            # Contract-expected keys
            'consensus_result': {'error': error, 'consensus_exists': False, 'consensus_score': 0.0},
            'agreement_score': 0.0,
            'consensus_direction': 'neutral',
            'consensus_confidence': 0.0,
            'consensus_thesis': f'Consensus error: {error}',
            'consensus_components': {},
            'consensus_quality': self.consensus_quality_metrics,
            '_thesis': f'Consensus error: {error}',
        }
