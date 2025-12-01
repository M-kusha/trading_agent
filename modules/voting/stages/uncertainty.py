"""
Uncertainty Sampler
===================
Alternative reality sampling for robustness and uncertainty quantification.
Generates perturbed voting outcomes to assess decision fragility.

Refactored from alternative_reality_sampler.py (~1313 lines).
~350 lines focused on core sampling logic.
"""

from __future__ import annotations

import datetime
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase


@module(**module_args("UncertaintySampler"))
class UncertaintySampler(VotingModuleBase):
    """
    Alternative reality sampling for uncertainty quantification.
    
    Generates perturbed voting scenarios to assess:
    - Decision robustness/fragility
    - Outcome uncertainty
    - Confidence calibration
    
    Publishes:
    - uncertainty_score
    - fragility_score
    - alternative_outcomes
    - uncertainty_analysis
    """
    
    def _module_specific_init(self) -> None:
        """Initialize sampling state."""
        # Configuration
        self.dim = int(self.config.get('dim', 5))
        self.n_samples = int(self.config.get('n_samples', 8))
        self.base_sigma = float(self.config.get('sigma', 0.05))
        self.current_sigma = self.base_sigma
        self.adaptive_sigma = bool(self.config.get('adaptive_sigma', True))
        self.uncertainty_threshold = float(self.config.get('uncertainty_threshold', 0.30))
        self.auto_dim = bool(self.config.get('auto_dim', True))
        
        # Sigma bounds
        self.sigma_bounds = (0.005, 0.25)
        
        # RNG
        seed = self.config.get('seed')
        self._rng = np.random.default_rng(seed)
        
        # History
        self.sampling_history: deque = deque(maxlen=120)
        self.uncertainty_history: deque = deque(maxlen=240)
        
        # Quality metrics
        self.quality_metrics: Dict[str, float] = {
            'sample_diversity': 0.0,
            'coverage_efficiency': 0.0,
            'uncertainty_accuracy': 0.0,
            'overall_quality_score': 0.5,
        }
        
        # Statistics
        self.sampling_stats: Dict[str, Any] = {
            'samples_generated': 0,
            'avg_uncertainty': 0.5,
            'sigma_adaptations': 0,
            'diversity_score': 0.0,
        }
        
        # Market adaptation multipliers
        self.regime_multipliers = {
            'trending': 0.8,
            'ranging': 1.0,
            'volatile': 1.6,
            'breakout': 1.2,
            'reversal': 1.4,
            'unknown': 1.1,
        }
        
        self.logger.info(
            f"[SAMPLER] UncertaintySampler initialized | "
            f"samples={self.n_samples} | sigma={self.base_sigma:.3f}"
        )
        
        # Publish baseline
        self._publish_uncertainty_baseline()
    
    def _publish_uncertainty_baseline(self) -> None:
        """Publish baseline uncertainty keys."""
        try:
            self.smart_bus.set(
                'uncertainty_score',
                0.5,
                module='UncertaintySampler',
                thesis='Baseline uncertainty score'
            )
            self.smart_bus.set(
                'fragility_score',
                0.5,
                module='UncertaintySampler',
                thesis='Baseline fragility score'
            )
        except Exception:
            pass
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """Sample alternative voting outcomes."""
        start = time.time()
        name = self.__class__.__name__
        
        try:
            # Get decision ID
            decision_id = self.smart_bus.get('kernel_decision_id', name)
            
            # Get voting data
            data = await self._get_voting_data()
            
            # Adapt sigma if needed
            if self.adaptive_sigma:
                await self._adapt_sigma(data)
            
            # Generate samples
            samples = await self._generate_samples(data)
            
            # Calculate uncertainty metrics
            analysis = await self._analyze_uncertainty(samples, data)
            
            # Generate thesis
            thesis = self._generate_thesis(analysis)
            
            # Publish to bus
            await self._update_bus(analysis, thesis)
            
            # Update stats
            self.sampling_stats['samples_generated'] += len(samples)
            
            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, 'process', elapsed_ms, True)
            
            return {
                'uncertainty_score': analysis.get('uncertainty_score', 0.5),
                'fragility_score': analysis.get('fragility_score', 0.5),
                'fragility': analysis.get('fragility_score', 0.5),  # alias
                'alternative_outcomes': samples,
                'uncertainty_analysis': analysis,
                'sampling_statistics': dict(self.sampling_stats),
                'quality_metrics': dict(self.quality_metrics),
                'current_sigma': self.current_sigma,
                'decision_id': decision_id,
                'uncertainty_decision_id': decision_id,
                # Contract-expected keys
                'uncertainty_result': analysis,
                'sampling_uncertainty': analysis.get('uncertainty_score', 0.5),
                'uncertainty_thesis': thesis,
                'effective_samples': len(samples),
                'alternative_samples': samples,
                'confidence_bounds': analysis.get('confidence_bounds', {'lower': 0.0, 'upper': 1.0}),
                'diversity_score': analysis.get('diversity_score', 0.5),
                'sampling_decision_id': decision_id,
                'sampling_fragility': analysis.get('fragility_score', 0.5),
                '_thesis': thesis,
            }
        
        except Exception as e:
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(e, 'uncertainty_process')
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)
    
    async def _get_voting_data(self) -> Dict[str, Any]:
        """Get voting data from SmartInfoBus."""
        return {
            'proposal_vectors': self.smart_bus.get('committee_proposal_vectors', self.__class__.__name__) or [],
            'member_confidences': self.smart_bus.get('committee_member_confidences', self.__class__.__name__) or [],
            'committee_decision': self.smart_bus.get('committee_decision', self.__class__.__name__) or {},
            'market_regime': self.smart_bus.get('market_regime', self.__class__.__name__) or 'unknown',
        }
    
    async def _adapt_sigma(self, data: Dict[str, Any]) -> None:
        """Adapt sampling sigma based on market conditions."""
        regime = str(data.get('market_regime', 'unknown')).lower()
        mult = self.regime_multipliers.get(regime, 1.0)
        
        # Adapt sigma
        target_sigma = self.base_sigma * mult
        target_sigma = max(self.sigma_bounds[0], min(self.sigma_bounds[1], target_sigma))
        
        # Smooth adaptation
        alpha = 0.2
        self.current_sigma = alpha * target_sigma + (1 - alpha) * self.current_sigma
        
        if abs(self.current_sigma - target_sigma) > 0.01:
            self.sampling_stats['sigma_adaptations'] += 1
    
    async def _generate_samples(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative voting scenarios."""
        base_vectors = data.get('proposal_vectors') or []
        base_confidences = data.get('member_confidences') or []
        
        if not base_vectors:
            # No real data - return empty samples (fragility will use neutral defaults)
            return []
        
        # Determine dimension
        if self.auto_dim and base_vectors:
            dim = len(base_vectors[0]) if base_vectors[0] else self.dim
        else:
            dim = self.dim
        
        samples = []
        base_array = np.array(base_vectors, dtype=np.float64)
        
        for i in range(self.n_samples):
            # Generate perturbation
            noise = self._rng.normal(0, self.current_sigma, base_array.shape)
            perturbed = base_array + noise
            
            # Clip to reasonable range
            perturbed = np.clip(perturbed, -2.0, 2.0)
            
            # Calculate outcome from perturbed vectors
            outcome = self._calculate_outcome(perturbed, base_confidences)
            
            samples.append({
                'sample_id': i,
                'perturbed_vectors': perturbed.tolist(),
                'outcome': outcome,
                'perturbation_magnitude': float(np.linalg.norm(noise)),
            })
        
        return samples
    
    def _calculate_outcome(
        self, 
        vectors: np.ndarray, 
        confidences: List[float]
    ) -> Dict[str, Any]:
        """Calculate voting outcome from perturbed vectors."""
        try:
            # Simple aggregation: mean direction weighted by confidence
            if len(vectors) == 0:
                return {'action': 'abstain', 'confidence': 0.5}
            
            # Get directional signal from first dimension
            directions = vectors[:, 0] if len(vectors.shape) > 1 else vectors
            
            # Weight by confidences
            if confidences:
                weights = np.array(confidences[:len(directions)])
                weights = weights / (weights.sum() + 1e-8)
                weighted_dir = np.dot(directions, weights)
            else:
                weighted_dir = np.mean(directions)
            
            # Determine action
            if weighted_dir > 0.1:
                action = 'long'
            elif weighted_dir < -0.1:
                action = 'short'
            else:
                action = 'hold'
            
            # Confidence from agreement
            conf = max(0.0, min(1.0, 0.5 + abs(weighted_dir) * 0.5))
            
            return {
                'action': action,
                'confidence': float(conf),
                'direction_score': float(weighted_dir),
            }
        
        except Exception:
            return {'action': 'abstain', 'confidence': 0.5}
    
    async def _analyze_uncertainty(
        self, 
        samples: List[Dict[str, Any]], 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze uncertainty from samples."""
        if not samples:
            return {
                'uncertainty_score': 0.5,
                'fragility_score': 0.5,
                'outcome_variance': 0.0,
            }
        
        # If no meaningful proposal vectors, return neutral fragility
        # This prevents fragility=1.0 when experts have no data yet
        proposal_vectors = data.get('proposal_vectors', [])
        if not proposal_vectors or len(proposal_vectors) == 0:
            return {
                'uncertainty_score': 0.5,
                'fragility_score': 0.5,
                'flip_rate': 0.0,
                'outcome_variance': 0.0,
                'n_samples': len(samples),
                'original_action': data.get('committee_decision', {}).get('action', 'abstain'),
                'sample_actions': [],
                'reason': 'No proposal vectors available - using neutral defaults',
            }
        
        # Get original decision
        original = data.get('committee_decision', {})
        original_action = original.get('action', 'abstain')
        
        # Count outcome changes
        actions = [s['outcome']['action'] for s in samples]
        confidences = [s['outcome']['confidence'] for s in samples]
        
        # How many samples flipped the decision?
        flips = sum(1 for a in actions if a != original_action)
        flip_rate = flips / len(samples)
        
        # Fragility: how easily does decision flip?
        fragility = float(min(1.0, flip_rate * 2.0))
        
        # Confidence variance
        conf_variance = float(np.var(confidences)) if len(confidences) > 1 else 0.0
        
        # Uncertainty combines fragility and variance
        uncertainty = float(0.6 * fragility + 0.4 * min(1.0, conf_variance * 4.0))
        
        # Update histories
        self.uncertainty_history.append(uncertainty)
        self.sampling_stats['avg_uncertainty'] = float(np.mean(list(self.uncertainty_history)))
        
        # Update quality
        sample_div = 1.0 - flip_rate if flip_rate < 0.5 else flip_rate
        self.quality_metrics['sample_diversity'] = float(sample_div)
        self.quality_metrics['overall_quality_score'] = float(
            0.5 * self.quality_metrics['sample_diversity'] +
            0.5 * (1.0 - uncertainty)
        )
        
        return {
            'uncertainty_score': float(max(0.0, min(1.0, uncertainty))),
            'fragility_score': float(max(0.0, min(1.0, fragility))),
            'flip_rate': float(flip_rate),
            'outcome_variance': float(conf_variance),
            'n_samples': len(samples),
            'original_action': original_action,
            'sample_actions': actions,
        }
    
    def _generate_thesis(self, analysis: Dict[str, Any]) -> str:
        """Generate uncertainty thesis."""
        unc = analysis.get('uncertainty_score', 0.5)
        frag = analysis.get('fragility_score', 0.5)
        n = analysis.get('n_samples', 0)
        
        label = 'HIGH' if unc > 0.6 else 'MODERATE' if unc > 0.3 else 'LOW'
        
        return (
            f"UNCERTAINTY: {label} ({unc:.1%}) | fragility={frag:.2f} | "
            f"samples={n} | sigma={self.current_sigma:.3f}"
        )
    
    async def _update_bus(self, analysis: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with results."""
        try:
            name = self.__class__.__name__
            
            self.smart_bus.set(
                'uncertainty_score',
                analysis.get('uncertainty_score', 0.5),
                module=name,
                thesis=thesis
            )
            self.smart_bus.set(
                'fragility_score',
                analysis.get('fragility_score', 0.5),
                module=name,
                thesis=f'Fragility: {analysis.get("fragility_score", 0.5):.2f}'
            )
            self.smart_bus.set(
                'fragility',
                analysis.get('fragility_score', 0.5),
                module=name,
                thesis='Fragility alias'
            )
            self.smart_bus.set(
                'uncertainty_analysis',
                analysis,
                module=name,
                thesis='Uncertainty analysis results'
            )
        except Exception as e:
            self.logger.warning(f"Bus update failed: {e}")
    
    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        return {
            'uncertainty_score': 0.5,
            'fragility_score': 0.5,
            'fragility': 0.5,
            'alternative_outcomes': [],
            'uncertainty_analysis': {'error': error},
            'sampling_statistics': dict(self.sampling_stats),
            'quality_metrics': dict(self.quality_metrics),
            'current_sigma': self.current_sigma,
            'decision_id': None,
            'uncertainty_decision_id': None,
            # Contract-expected keys
            'uncertainty_result': {'error': error, 'uncertainty_score': 0.5},
            'sampling_uncertainty': 0.5,
            'uncertainty_thesis': f'Uncertainty sampling error: {error}',
            'effective_samples': 0,
            'alternative_samples': [],
            'confidence_bounds': {'lower': 0.0, 'upper': 1.0},
            'diversity_score': 0.5,
            'sampling_decision_id': None,
            'sampling_fragility': 0.5,
            '_thesis': f'Uncertainty sampling error: {error}',
        }


