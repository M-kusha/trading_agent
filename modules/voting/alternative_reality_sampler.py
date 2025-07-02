# ─────────────────────────────────────────────────────────────
# File: modules/voting/alternative_reality_sampler.py
# Enhanced Alternative Reality Sampler with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class AlternativeRealitySampler(Module, AnalysisMixin, StateManagementMixin):
    """
    Enhanced alternative reality sampler with InfoBus integration.
    Samples alternative voting outcomes for robustness testing and
    uncertainty quantification in committee decisions.
    """

    def __init__(
        self,
        dim: int,
        n_samples: int = 5,
        sigma: float = 0.05,
        adaptive_sigma: bool = True,
        uncertainty_threshold: float = 0.3,
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
        self.dim = int(dim)
        self.n_samples = int(n_samples)
        self.base_sigma = float(sigma)
        self.current_sigma = self.base_sigma
        self.adaptive_sigma = bool(adaptive_sigma)
        self.uncertainty_threshold = float(uncertainty_threshold)
        
        # Sampling state
        self.last_samples = None
        self.last_weights = None
        self.sampling_history = deque(maxlen=50)
        self.uncertainty_history = deque(maxlen=100)
        
        # Performance tracking
        self.sampling_stats = {
            'samples_generated': 0,
            'avg_uncertainty': 0.0,
            'sigma_adaptations': 0,
            'effective_samples': 0
        }
        
        # Adaptive parameters
        self.sigma_bounds = (0.01, 0.2)
        self.adaptation_rate = 0.1
        self.diversity_target = 0.15
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "AlternativeRealitySampler",
            "logs/voting/alternative_reality_sampler.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("AlternativeRealitySampler")
        
        self.log_operator_info(
            "🎲 Alternative Reality Sampler initialized",
            dimensions=self.dim,
            samples=self.n_samples,
            sigma=self.base_sigma,
            adaptive=self.adaptive_sigma
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset sampling state
        self.last_samples = None
        self.last_weights = None
        self.current_sigma = self.base_sigma
        
        # Reset history
        self.sampling_history.clear()
        self.uncertainty_history.clear()
        
        # Reset statistics
        self.sampling_stats = {
            'samples_generated': 0,
            'avg_uncertainty': 0.0,
            'sigma_adaptations': 0,
            'effective_samples': 0
        }
        
        self.log_operator_info("🔄 Alternative Reality Sampler reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using standalone mode")
            return
        
        # Extract context and voting data
        context = extract_standard_context(info_bus)
        voting_data = self._extract_voting_data_from_info_bus(info_bus)
        
        # Update sampling parameters based on market conditions
        self._update_sampling_parameters(context, voting_data)
        
        # Analyze recent sampling effectiveness
        self._analyze_sampling_effectiveness(voting_data)
        
        # Update InfoBus with sampler state
        self._update_info_bus(info_bus)

    def _extract_voting_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract voting-related data from InfoBus"""
        
        data = {}
        
        try:
            # Get votes and consensus information
            votes = info_bus.get('votes', [])
            data['votes'] = votes
            data['vote_count'] = len(votes)
            
            # Extract voting summary if available
            voting_summary = InfoBusExtractor.get_voting_summary(info_bus)
            data['consensus_direction'] = voting_summary.get('consensus_direction', 'neutral')
            data['avg_confidence'] = voting_summary.get('avg_confidence', 0.5)
            data['agreement_score'] = voting_summary.get('agreement_score', 0.5)
            
            # Get arbiter weights if available
            module_data = info_bus.get('module_data', {})
            arbiter_data = module_data.get('strategy_arbiter', {})
            data['arbiter_weights'] = arbiter_data.get('weights', [])
            data['last_alpha'] = arbiter_data.get('last_alpha', [])
            
            # Get market uncertainty indicators
            market_context = info_bus.get('market_context', {})
            data['volatility'] = market_context.get('volatility', {})
            data['regime'] = market_context.get('regime', 'unknown')
            
        except Exception as e:
            self.log_operator_warning(f"Voting data extraction failed: {e}")
            data = {
                'votes': [],
                'vote_count': 0,
                'consensus_direction': 'neutral',
                'avg_confidence': 0.5,
                'agreement_score': 0.5,
                'arbiter_weights': [],
                'last_alpha': [],
                'volatility': {},
                'regime': 'unknown'
            }
        
        return data

    def _update_sampling_parameters(self, context: Dict[str, Any], 
                                   voting_data: Dict[str, Any]) -> None:
        """Update sampling parameters based on market conditions"""
        
        if not self.adaptive_sigma:
            return
        
        try:
            # Get market uncertainty indicators
            regime = context.get('regime', 'unknown')
            volatility_level = context.get('volatility_level', 'medium')
            agreement_score = voting_data.get('agreement_score', 0.5)
            
            # Calculate target sigma based on conditions
            base_multiplier = 1.0
            
            # Regime-based adjustments
            if regime == 'volatile':
                base_multiplier *= 1.5  # More sampling in volatile conditions
            elif regime == 'trending':
                base_multiplier *= 0.8  # Less sampling in clear trends
            elif regime == 'noise':
                base_multiplier *= 1.3  # More exploration in noisy conditions
            
            # Volatility-based adjustments
            if volatility_level == 'extreme':
                base_multiplier *= 1.8
            elif volatility_level == 'high':
                base_multiplier *= 1.3
            elif volatility_level == 'low':
                base_multiplier *= 0.7
            
            # Agreement-based adjustments
            if agreement_score < 0.3:
                base_multiplier *= 1.4  # More sampling when disagreement high
            elif agreement_score > 0.8:
                base_multiplier *= 0.6  # Less sampling when high agreement
            
            # Calculate new sigma
            target_sigma = self.base_sigma * base_multiplier
            target_sigma = np.clip(target_sigma, self.sigma_bounds[0], self.sigma_bounds[1])
            
            # Smooth adaptation
            old_sigma = self.current_sigma
            self.current_sigma += self.adaptation_rate * (target_sigma - self.current_sigma)
            
            # Track adaptations
            if abs(self.current_sigma - old_sigma) > 0.01:
                self.sampling_stats['sigma_adaptations'] += 1
                self.log_operator_info(
                    f"📊 Sampling sigma adapted: {old_sigma:.3f} → {self.current_sigma:.3f}",
                    regime=regime,
                    volatility=volatility_level,
                    agreement=f"{agreement_score:.1%}"
                )
            
        except Exception as e:
            self.log_operator_warning(f"Sampling parameter update failed: {e}")

    def _analyze_sampling_effectiveness(self, voting_data: Dict[str, Any]) -> None:
        """Analyze effectiveness of recent sampling"""
        
        try:
            if len(self.sampling_history) < 5:
                return
            
            # Analyze diversity of recent samples
            recent_samples = list(self.sampling_history)[-10:]
            
            diversities = []
            uncertainties = []
            
            for sample_data in recent_samples:
                samples = sample_data.get('samples', [])
                if len(samples) > 1:
                    # Calculate diversity (average pairwise distance)
                    pairwise_dists = []
                    for i in range(len(samples)):
                        for j in range(i + 1, len(samples)):
                            dist = np.linalg.norm(np.array(samples[i]) - np.array(samples[j]))
                            pairwise_dists.append(dist)
                    
                    if pairwise_dists:
                        diversity = np.mean(pairwise_dists)
                        diversities.append(diversity)
                        
                        # Calculate uncertainty from sample spread
                        uncertainty = np.std([np.linalg.norm(s) for s in samples])
                        uncertainties.append(uncertainty)
            
            if diversities:
                avg_diversity = np.mean(diversities)
                avg_uncertainty = np.mean(uncertainties)
                
                # Update statistics
                self.sampling_stats['avg_uncertainty'] = avg_uncertainty
                self.uncertainty_history.append(avg_uncertainty)
                
                # Check if we need to adjust sampling
                if avg_diversity < self.diversity_target * 0.7:
                    self.log_operator_info(
                        f"⚠️ Low sampling diversity detected: {avg_diversity:.3f}",
                        target=self.diversity_target,
                        recommendation="Consider increasing sigma"
                    )
                elif avg_diversity > self.diversity_target * 1.5:
                    self.log_operator_info(
                        f"📈 High sampling diversity: {avg_diversity:.3f}",
                        target=self.diversity_target,
                        recommendation="Consider decreasing sigma"
                    )
            
        except Exception as e:
            self.log_operator_warning(f"Sampling effectiveness analysis failed: {e}")

    def sample(self, weights: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate alternative weight configurations with enhanced features.
        
        Args:
            weights: Base weights to sample around
            context: Optional context for adaptive sampling
            
        Returns:
            Array of alternative weight samples
        """
        
        try:
            # Validate input
            weights = np.asarray(weights, dtype=np.float32).flatten()
            if weights.size != self.dim:
                self.log_operator_warning(
                    f"Weight dimension mismatch: {weights.size} vs {self.dim}"
                )
                weights = np.pad(weights, (0, max(0, self.dim - weights.size)))[:self.dim]
            
            # Store for analysis
            self.last_weights = weights.copy()
            
            # Adapt sampling based on context
            current_sigma = self.current_sigma
            if context:
                uncertainty = context.get('uncertainty', 0.0)
                if uncertainty > self.uncertainty_threshold:
                    current_sigma *= 1.3  # Increase exploration under uncertainty
            
            # Generate base samples
            noise = np.random.randn(self.n_samples, self.dim) * current_sigma
            samples = weights[None, :] + noise
            
            # Enhanced sampling methods
            if self.n_samples >= 5:
                # Add some structured sampling
                structured_samples = self._generate_structured_samples(weights, current_sigma)
                # Replace some random samples with structured ones
                n_structured = min(2, self.n_samples // 2)
                samples[-n_structured:] = structured_samples[:n_structured]
            
            # Ensure positive and normalized
            samples = np.abs(samples)
            row_sums = samples.sum(axis=1, keepdims=True)
            samples = samples / (row_sums + 1e-12)
            
            # Store samples for analysis
            self.last_samples = samples.copy()
            
            # Record sampling event
            sample_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'base_weights': weights.tolist(),
                'samples': samples.tolist(),
                'sigma_used': current_sigma,
                'context': context.copy() if context else {}
            }
            self.sampling_history.append(sample_data)
            
            # Update statistics
            self.sampling_stats['samples_generated'] += 1
            effective_samples = self._count_effective_samples(samples, weights)
            self.sampling_stats['effective_samples'] += effective_samples
            
            return samples
            
        except Exception as e:
            self.log_operator_error(f"Sampling failed: {e}")
            # Return safe fallback
            return np.tile(weights, (self.n_samples, 1))

    def _generate_structured_samples(self, weights: np.ndarray, sigma: float) -> np.ndarray:
        """Generate structured samples for better exploration"""
        
        try:
            structured = []
            
            # Systematic perturbations
            for i in range(min(self.dim, 3)):  # Max 3 structured samples
                perturbed = weights.copy()
                
                # Single-dimension perturbation
                if i < self.dim:
                    perturbed[i] += sigma * 2 * (np.random.random() - 0.5)
                
                # Normalize
                perturbed = np.abs(perturbed)
                perturbed = perturbed / (perturbed.sum() + 1e-12)
                structured.append(perturbed)
            
            return np.array(structured)
            
        except Exception as e:
            self.log_operator_warning(f"Structured sampling failed: {e}")
            return np.tile(weights, (1, 1))

    def _count_effective_samples(self, samples: np.ndarray, base_weights: np.ndarray) -> int:
        """Count how many samples are meaningfully different from base"""
        
        try:
            threshold = self.current_sigma * 0.5
            effective = 0
            
            for sample in samples:
                distance = np.linalg.norm(sample - base_weights)
                if distance > threshold:
                    effective += 1
            
            return effective
            
        except Exception as e:
            self.log_operator_warning(f"Effective sample counting failed: {e}")
            return len(samples)

    def get_uncertainty_estimate(self, weights: np.ndarray) -> float:
        """Get uncertainty estimate for given weights"""
        
        try:
            if self.last_samples is None or len(self.uncertainty_history) == 0:
                return 0.5  # Default uncertainty
            
            # Generate samples for uncertainty estimation
            samples = self.sample(weights)
            
            # Calculate spread of sample outcomes
            sample_norms = [np.linalg.norm(s) for s in samples]
            uncertainty = np.std(sample_norms) / (np.mean(sample_norms) + 1e-12)
            
            return float(np.clip(uncertainty, 0.0, 1.0))
            
        except Exception as e:
            self.log_operator_warning(f"Uncertainty estimation failed: {e}")
            return 0.5

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with sampler results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'alternative_reality_sampler', {
            'current_sigma': self.current_sigma,
            'base_sigma': self.base_sigma,
            'n_samples': self.n_samples,
            'dimensions': self.dim,
            'adaptive_sigma': self.adaptive_sigma,
            'sampling_stats': self.sampling_stats.copy(),
            'recent_uncertainty': list(self.uncertainty_history)[-10:],
            'last_sample_count': len(self.last_samples) if self.last_samples is not None else 0,
            'effectiveness': {
                'avg_uncertainty': self.sampling_stats.get('avg_uncertainty', 0.0),
                'sigma_adaptations': self.sampling_stats.get('sigma_adaptations', 0),
                'effective_sample_ratio': (
                    self.sampling_stats.get('effective_samples', 0) / 
                    max(self.sampling_stats.get('samples_generated', 1), 1)
                )
            }
        })

    def get_sampling_report(self) -> str:
        """Generate operator-friendly sampling report"""
        
        # Recent sampling effectiveness
        if len(self.uncertainty_history) > 0:
            recent_uncertainty = np.mean(list(self.uncertainty_history)[-10:])
            uncertainty_trend = "📈 Rising" if recent_uncertainty > self.uncertainty_threshold else "📉 Stable"
        else:
            recent_uncertainty = 0.0
            uncertainty_trend = "📭 No data"
        
        # Sigma adaptation status
        sigma_change = (self.current_sigma - self.base_sigma) / self.base_sigma * 100
        if abs(sigma_change) < 5:
            sigma_status = "⚖️ Stable"
        elif sigma_change > 0:
            sigma_status = f"📈 Increased ({sigma_change:+.1f}%)"
        else:
            sigma_status = f"📉 Decreased ({sigma_change:+.1f}%)"
        
        return f"""
🎲 ALTERNATIVE REALITY SAMPLER
═══════════════════════════════════════
📊 Configuration:
• Dimensions: {self.dim}
• Sample Count: {self.n_samples}
• Base Sigma: {self.base_sigma:.3f}
• Current Sigma: {self.current_sigma:.3f}
• Adaptive: {'✅ Enabled' if self.adaptive_sigma else '❌ Disabled'}

📈 Sampling Statistics:
• Total Samples Generated: {self.sampling_stats['samples_generated']}
• Effective Samples: {self.sampling_stats['effective_samples']}
• Sigma Adaptations: {self.sampling_stats['sigma_adaptations']}
• Effectiveness Ratio: {(self.sampling_stats.get('effective_samples', 0) / max(self.sampling_stats.get('samples_generated', 1), 1)):.1%}

🎯 Current Status:
• Sigma Status: {sigma_status}
• Recent Uncertainty: {recent_uncertainty:.3f}
• Uncertainty Trend: {uncertainty_trend}
• Target Diversity: {self.diversity_target:.3f}

📊 Recent Performance:
• Sampling History: {len(self.sampling_history)} entries
• Uncertainty History: {len(self.uncertainty_history)} entries
• Last Sample Size: {len(self.last_samples) if self.last_samples is not None else 0}

🎲 Sampling Quality:
• Sigma Bounds: [{self.sigma_bounds[0]:.3f}, {self.sigma_bounds[1]:.3f}]
• Adaptation Rate: {self.adaptation_rate:.3f}
• Uncertainty Threshold: {self.uncertainty_threshold:.3f}
        """

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "dim": self.dim,
                "n_samples": self.n_samples,
                "base_sigma": self.base_sigma,
                "adaptive_sigma": self.adaptive_sigma,
                "uncertainty_threshold": self.uncertainty_threshold
            },
            "current_state": {
                "current_sigma": self.current_sigma,
                "last_weights": self.last_weights.tolist() if self.last_weights is not None else None,
                "last_samples": self.last_samples.tolist() if self.last_samples is not None else None
            },
            "statistics": self.sampling_stats.copy(),
            "history": {
                "sampling_history": list(self.sampling_history)[-10:],
                "uncertainty_history": list(self.uncertainty_history)[-20:]
            },
            "parameters": {
                "sigma_bounds": self.sigma_bounds,
                "adaptation_rate": self.adaptation_rate,
                "diversity_target": self.diversity_target
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.dim = int(config.get("dim", self.dim))
        self.n_samples = int(config.get("n_samples", self.n_samples))
        self.base_sigma = float(config.get("base_sigma", self.base_sigma))
        self.adaptive_sigma = bool(config.get("adaptive_sigma", self.adaptive_sigma))
        self.uncertainty_threshold = float(config.get("uncertainty_threshold", self.uncertainty_threshold))
        
        # Load current state
        current_state = state.get("current_state", {})
        self.current_sigma = float(current_state.get("current_sigma", self.base_sigma))
        
        last_weights = current_state.get("last_weights")
        if last_weights:
            self.last_weights = np.array(last_weights, dtype=np.float32)
            
        last_samples = current_state.get("last_samples")
        if last_samples:
            self.last_samples = np.array(last_samples, dtype=np.float32)
        
        # Load statistics
        self.sampling_stats.update(state.get("statistics", {}))
        
        # Load history
        history = state.get("history", {})
        
        sampling_history = history.get("sampling_history", [])
        self.sampling_history.clear()
        for entry in sampling_history:
            self.sampling_history.append(entry)
            
        uncertainty_history = history.get("uncertainty_history", [])
        self.uncertainty_history.clear()
        for entry in uncertainty_history:
            self.uncertainty_history.append(entry)
        
        # Load parameters
        parameters = state.get("parameters", {})
        self.sigma_bounds = tuple(parameters.get("sigma_bounds", self.sigma_bounds))
        self.adaptation_rate = float(parameters.get("adaptation_rate", self.adaptation_rate))
        self.diversity_target = float(parameters.get("diversity_target", self.diversity_target))

    def get_observation_components(self) -> np.ndarray:
        """Return sampler features for observation"""
        
        try:
            features = [
                float(self.current_sigma / self.base_sigma),  # Sigma ratio
                float(self.sampling_stats.get('avg_uncertainty', 0.0)),
                float(len(self.sampling_history) / 50),  # History fullness
                float(self.sampling_stats.get('effective_samples', 0) / max(self.sampling_stats.get('samples_generated', 1), 1)),
                float(self.adaptive_sigma)
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.array([1.0, 0.5, 0.0, 0.5, 1.0], dtype=np.float32)