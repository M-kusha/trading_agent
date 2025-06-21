# modules/strategy/voting.py
"""
FIXED: Committee-of-experts voting and arbitration logic for strategy modules.
Less restrictive gating, better consensus logic, and improved debugging.
"""
from __future__ import annotations

# ── standard lib ──────────────────────────────────────────────────────────
import math
from statistics import mean
from typing import List, Any, Dict, Optional
import logging

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np

# ── internal ──────────────────────────────────────────────────────────────
try:
    from modules.core.core import Module
except ImportError:
    class Module:  # Stub for standalone testing
        def reset(self) -> None:
            pass
        def step(self, *a, **kw):
            pass
        def get_observation_components(self):
            pass

# ═══════════════════════════════════════════════════════════════════════════
# Helper functions for the new "smart gate" - MADE LESS RESTRICTIVE
# ═══════════════════════════════════════════════════════════════════════════
_LAYER_W = dict(
    liquidityheatmaplayer=2.0,      # LiquidityHeatmapLayer
    lhl=2.0,                        # Short form
    fractalregimeconfirmation=1.5,  # FractalRegimeConfirmation  
    frc=1.5,                        # Short form
    marketthemedetector=1.0,        # MarketThemeDetector
    mtd=1.0,                        # Short form
    markerregimeswitcher=1.0,       # MarketRegimeSwitcher
    switcher=1.0,                   # Short form
    # New additions for better voting
    positionmanager=1.5,            # Position manager has good judgment
    themeexpert=1.2,                # Theme expert
    regimebiasexpert=1.3,           # Regime expert
    seasonalityriskexpert=1.1,      # Seasonality expert
    metarlexpert=1.4,               # Meta-RL expert
    trademonitorvetoexpert=0.8,     # Veto expert (lower weight)
    dynamicriskcontroller=1.0,      # Risk controller
)

# ADJUSTED: Made less restrictive
_SIG_K     = 4.0        # REDUCED from 8.0 - gentler slope
_SIG_KNEE  = 0.15       # REDUCED from 0.20 - lower threshold
_BASE_GATE = 0.15       # REDUCED from 0.25 - easier base gate
_VOL_REF   = 0.02       # INCREASED from 0.01 - less sensitive to volatility

def _squash(c: float) -> float:
    """Gentler squashing function for confidence values."""
    return 1.0 / (1.0 + np.exp(-_SIG_K * (c - _SIG_KNEE)))

def _smart_gate(volatility: float, maj: int) -> float:
    """
    FIXED: Less restrictive gate that allows more trades through.
    
    Args:
        volatility: Current market volatility
        maj: Majority direction (+1 or -1)
        
    Returns:
        Gate threshold (lower = easier to pass)
    """
    # Base gate is already low
    gate = _BASE_GATE
    
    # Only increase gate slightly for high volatility
    if volatility > _VOL_REF * 2:
        gate *= 1.2  # Only 20% increase instead of doubling
    
    # Reduce gate if there's clear directional agreement
    if abs(maj) > 0:
        gate *= 0.8
        
    return gate

# ═══════════════════════════════════════════════════════════════════════════
# Supporting classes
# ═══════════════════════════════════════════════════════════════════════════

class ConsensusDetector(Module):
    """Detects consensus among voting members"""
    
    def __init__(self, n_members: int, threshold: float = 0.6):
        self.n_members = n_members
        self.threshold = threshold
        self.last_consensus = 0.0
        
    def reset(self):
        self.last_consensus = 0.0
        
    def step(self, **kwargs):
        pass
    
    def resize(self, n_members: int):
        self.n_members = n_members
        
    def get_observation_components(self) -> np.ndarray:
        return np.array([self.last_consensus], dtype=np.float32)
        
    def compute_consensus(self, actions: List[np.ndarray], confidences: List[float]) -> float:
        """
        Compute consensus level from member actions and confidences.
        
        Returns value between 0 (no consensus) and 1 (perfect consensus).
        """
        if actions is None or len(actions) < 2:
            return 0.5
            
        # Compute pairwise agreement
        agreements = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                # Cosine similarity between action vectors
                a1, a2 = actions[i], actions[j]
                norm1, norm2 = np.linalg.norm(a1), np.linalg.norm(a2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(a1, a2) / (norm1 * norm2)
                    # Weight by confidence
                    weight = confidences[i] * confidences[j]
                    agreements.append(similarity * weight)
                    
        if agreements:
            self.last_consensus = float(np.mean(agreements))
        else:
            self.last_consensus = 0.0
            
        return self.last_consensus


class CollusionAuditor(Module):
    """
    Detects potential collusion patterns in voting.
    FIXED: Now properly integrated but with relaxed thresholds.
    """
    
    def __init__(self, n_members: int, window: int = 10, threshold: float = 0.9, debug: bool = False):
        self.n_members = n_members
        self.window = window
        self.threshold = threshold  # High threshold to avoid false positives
        self.debug = debug
        
        self.vote_history = []
        self.collusion_score = 0.0
        self.suspicious_pairs = set()
        
    def reset(self):
        self.vote_history.clear()
        self.collusion_score = 0.0
        self.suspicious_pairs.clear()
        
    def step(self, **kwargs):
        pass
        
    def get_observation_components(self) -> np.ndarray:
        return np.array([self.collusion_score], dtype=np.float32)
        
    def check_collusion(self, actions: List[np.ndarray]) -> float:
        """
        Check for suspicious voting patterns.
        Returns score between 0 (no collusion) and 1 (definite collusion).
        """
        if len(actions) < 2:
            return 0.0
            
        # Add to history
        self.vote_history.append(actions)
        if len(self.vote_history) > self.window:
            self.vote_history.pop(0)
            
        # Need enough history
        if len(self.vote_history) < 5:
            return 0.0
            
        # Check for pairs that always vote together
        pair_agreements = {}
        
        for votes in self.vote_history:
            for i in range(len(votes)):
                for j in range(i + 1, len(votes)):
                    pair = (i, j)
                    
                    # Compute agreement
                    v1, v2 = votes[i], votes[j]
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        
                        if pair not in pair_agreements:
                            pair_agreements[pair] = []
                        pair_agreements[pair].append(similarity)
                        
        # Find suspicious pairs
        self.suspicious_pairs.clear()
        suspicious_count = 0
        
        for pair, agreements in pair_agreements.items():
            avg_agreement = np.mean(agreements)
            if avg_agreement > self.threshold:
                self.suspicious_pairs.add(pair)
                suspicious_count += 1
                
        # Calculate overall score
        max_pairs = self.n_members * (self.n_members - 1) / 2
        self.collusion_score = suspicious_count / max_pairs if max_pairs > 0 else 0.0
        
        return self.collusion_score


class TimeHorizonAligner(Module):
    """Aligns voting weights based on time horizons"""
    
    def __init__(self, horizons: List[int]):
        self.horizons = np.asarray(horizons, np.float32)
        self.clock = 0
        
    def reset(self):
        self.clock = 0
        
    def step(self, **kwargs):
        self.clock += 1
        
    def get_observation_components(self) -> np.ndarray:
        return np.asarray([self.clock], np.float32)
        
    def apply(self, weights: np.ndarray) -> np.ndarray:
        """Apply time-based scaling to weights"""
        # Distance from each horizon
        distances = 1.0 / (1.0 + np.abs(self.clock - self.horizons))
        
        # Normalize
        distances = distances / distances.sum()
        
        # Apply to weights
        if len(weights) == len(distances):
            return weights * distances
        else:
            # Fallback if size mismatch
            return weights


class AlternativeRealitySampler:
    """Samples alternative voting outcomes for robustness"""
    
    def __init__(self, dim: int, n_samples: int = 5, sigma: float = 0.05):
        self.dim = dim
        self.n_samples = n_samples
        self.sigma = sigma
        
    def sample(self, weights: np.ndarray) -> np.ndarray:
        """Generate alternative weight configurations"""
        # Base weights plus noise
        samples = weights[None, :] + np.random.randn(self.n_samples, self.dim) * self.sigma
        
        # Ensure positive and normalized
        samples = np.abs(samples)
        samples = samples / samples.sum(axis=1, keepdims=True)
        
        return samples


# ═══════════════════════════════════════════════════════════════════════════
# StrategyArbiter with improved smart-gate
# ═══════════════════════════════════════════════════════════════════════════
class StrategyArbiter(Module):
    """
    FIXED: Committee-of-experts blender with less restrictive gating.
    
    Key improvements:
    - More lenient gate thresholds
    - Better handling of bootstrapping
    - Clearer debug output
    - Support for dynamic member addition
    """
    
    # Reward settings
    REINFORCE_LR: float = 0.001
    REINFORCE_LAMBDA: float = 0.95
    PRIOR_BLEND: float = 0.30

    def __init__(
        self,
        members: List[Module],
        init_weights: List[float] | np.ndarray,
        action_dim: int,
        adapt_rate: float = 0.01,
        consensus: Optional[Module] = None,
        collusion: Optional[Module] = None,
        horizon_aligner: Optional[Module] = None,
        debug: bool = True,
        audit_log_size: int = 100,
        min_confidence: float = 0.3,  # REDUCED from 0.5
    ):
        self.members = members
        self.weights = np.asarray(init_weights, np.float32)
        assert self.weights.shape == (len(members),), f"Weight shape mismatch: {self.weights.shape} vs {len(members)}"
        
        self.action_dim = action_dim
        self.adapt_rate = adapt_rate
        self.consensus = consensus
        self.collusion = collusion
        self.haligner = horizon_aligner
        self.debug = debug
        self.min_confidence = min_confidence
        
        self.curr_vol: float = 0.01  # Default volatility
        self.last_alpha: Optional[np.ndarray] = None
        self._baseline = 0.0
        self._baseline_beta = 0.98
        self._trace: List[Dict[str, Any]] = []
        self._log_size = audit_log_size
        
        # Track gate statistics for debugging
        self._gate_passes = 0
        self._gate_attempts = 0
        self._bootstrap_steps = 50
        self._step_count = 0
        
        # Setup logging
        self.logger = logging.getLogger("StrategyArbiter")
        if self.debug and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
            
    def reset(self):
        """Reset arbiter state"""
        self._baseline = 0.0
        self._gate_passes = 0
        self._gate_attempts = 0
        self._step_count = 0
        self._trace.clear()
        self.last_alpha = None
        
        if self.consensus:
            self.consensus.reset()
        if self.collusion:
            self.collusion.reset()
        if self.haligner:
            self.haligner.reset()

    def update_market_state(self, volatility: float):
        """Update current market volatility"""
        self.curr_vol = float(max(volatility, 0.001))

    def _save_trace(self, trace: Dict[str, Any]):
        """Save trace for debugging"""
        self._trace.append(trace)
        if len(self._trace) > self._log_size:
            self._trace = self._trace[-self._log_size:]

    def get_last_traces(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get recent traces for debugging"""
        return self._trace[-n:]

    def propose(self, obs: Any) -> np.ndarray:
        """
        FIXED: Propose an action by blending member proposals with less restrictive gating.
        """
        self._step_count += 1
        trace = {"step": self._step_count}
        
        # Collect proposals from all members
        proposals = []
        confidences = []
        
        for i, member in enumerate(self.members):
            try:
                # Get action proposal
                if hasattr(member, 'propose_action'):
                    prop = member.propose_action(obs)
                elif hasattr(member, 'propose'):
                    prop = member.propose(obs)
                else:
                    # Fallback for modules without propose
                    prop = np.zeros(self.action_dim, dtype=np.float32)
                    
                # Ensure proper shape
                prop = np.asarray(prop, dtype=np.float32).reshape(-1)
                if prop.size < self.action_dim:
                    prop = np.pad(prop, (0, self.action_dim - prop.size))
                elif prop.size > self.action_dim:
                    prop = prop[:self.action_dim]
                    
                proposals.append(prop)
                
                # Get confidence
                if hasattr(member, 'confidence'):
                    conf = float(member.confidence(obs))
                else:
                    conf = 0.5  # Default confidence
                    
                confidences.append(max(conf, self.min_confidence))
                
            except Exception as e:
                if self.debug:
                    self.logger.warning(f"Member {i} ({member.__class__.__name__}) failed: {e}")
                proposals.append(np.zeros(self.action_dim, dtype=np.float32))
                confidences.append(self.min_confidence)
                
        # Convert to arrays
        proposals = np.array(proposals, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        
        trace["raw_proposals"] = proposals.tolist()
        trace["confidences"] = confidences.tolist()
        
        # Check consensus if available
        if self.consensus:
            consensus_score = self.consensus.compute_consensus(proposals, confidences)
            trace["consensus"] = float(consensus_score)
        else:
            consensus_score = 0.5
            
        # Check collusion if available
        if self.collusion:
            collusion_score = self.collusion.check_collusion(proposals)
            if collusion_score > 0.5:
                # Reduce confidence of colluding members
                for pair in self.collusion.suspicious_pairs:
                    confidences[pair[0]] *= 0.7
                    confidences[pair[1]] *= 0.7
                    
        # Apply time horizon alignment if available
        if self.haligner:
            self.weights = self.haligner.apply(self.weights)
            
        # Compute weighted blend
        # Normalize weights and confidences
        w_norm = self.weights / (self.weights.sum() + 1e-12)
        c_norm = confidences / (confidences.sum() + 1e-12)
        
        # Combined weights
        alpha = w_norm * c_norm
        alpha = alpha / (alpha.sum() + 1e-12)
        
        self.last_alpha = alpha.copy()
        trace["alpha"] = alpha.tolist()
        
        # Blend proposals
        action = np.zeros(self.action_dim, dtype=np.float32)
        for i, (prop, a) in enumerate(zip(proposals, alpha)):
            action += a * prop
            
        trace["blended_action"] = action.tolist()
        
        # Apply smart gate
        action = self._apply_smart_gate(action, proposals, confidences, trace)
        
        # Save trace
        self._save_trace(trace)
        
        return action

    def _apply_smart_gate(
        self,
        action: np.ndarray,
        proposals: np.ndarray,
        confidences: np.ndarray,
        trace: Dict[str, Any]
    ) -> np.ndarray:
        """
        FIXED: Apply less restrictive gating to allow more trades.
        """
        # Bootstrap mode - very lenient
        if self._step_count < self._bootstrap_steps:
            gate = _BASE_GATE * 0.5  # Half the normal gate
            trace["bootstrap"] = True
        else:
            gate = _smart_gate(self.curr_vol, 0)
            trace["bootstrap"] = False
            
        # Calculate signal strength
        signal_strength = np.abs(action).mean()
        
        # Check directional agreement
        directions = []
        for i in range(0, self.action_dim, 2):  # Check intensity components
            avg_dir = np.mean([p[i] for p in proposals])
            directions.append(np.sign(avg_dir))
            
        # Count agreement
        direction_agreement = sum(1 for p in proposals if all(
            np.sign(p[i]) == directions[i//2] for i in range(0, self.action_dim, 2)
        )) / len(proposals)
        
        # Gate decision logic - MUCH more lenient
        passed = False
        
        # Method 1: Signal strength
        if signal_strength >= gate:
            passed = True
            reason = "signal_strength"
            
        # Method 2: High confidence from multiple experts
        high_conf_count = sum(1 for c in confidences if c >= 0.6)
        if high_conf_count >= len(self.members) * 0.3:  # 30% high confidence
            passed = True
            reason = "high_confidence"
            
        # Method 3: Directional agreement
        if direction_agreement >= 0.5:  # 50% agreement
            passed = True
            reason = "direction_agreement"
            
        # Method 4: Bootstrap override
        if self._step_count < self._bootstrap_steps and signal_strength > gate * 0.3:
            passed = True
            reason = "bootstrap_override"
            
        # Track statistics
        self._gate_attempts += 1
        if passed:
            self._gate_passes += 1
        else:
            # Apply dampening instead of zeroing
            action *= 0.2  # Still allow small positions through
            reason = "dampened"
            
        # Update trace
        trace["gate"] = float(gate)
        trace["signal_strength"] = float(signal_strength)
        trace["passed"] = passed
        trace["reason"] = reason
        trace["pass_rate"] = self._gate_passes / max(self._gate_attempts, 1)
        
        if self.debug and self._gate_attempts % 10 == 0:
            self.logger.info(
                f"Gate stats: {self._gate_passes}/{self._gate_attempts} "
                f"({trace['pass_rate']:.1%} pass rate)"
            )
            
        return action

    def update_weights(self, reward: float):
        """
        Update member weights using REINFORCE.
        """
        if self.last_alpha is None:
            return
            
        # Update baseline
        self._baseline = self._baseline_beta * self._baseline + (1 - self._baseline_beta) * reward
        
        # Advantage
        advantage = reward - self._baseline
        
        # REINFORCE update
        grad = advantage * (self.last_alpha - self.weights)
        self.weights += self.adapt_rate * grad
        
        # Ensure positive weights
        self.weights = np.maximum(self.weights, 0.01)
        
        # Normalize
        self.weights = self.weights / self.weights.sum()

    def step(self, reward: float = 0.0, **kwargs):
        """Process step with optional weight update"""
        if reward != 0:
            self.update_weights(reward)
            
        # Step sub-modules
        if self.consensus:
            self.consensus.step(**kwargs)
        if self.collusion:
            self.collusion.step(**kwargs)
        if self.haligner:
            self.haligner.step(**kwargs)

    def get_observation_components(self) -> np.ndarray:
        """Get arbiter state for observation"""
        features = [
            float(self._gate_passes / max(self._gate_attempts, 1)),  # Pass rate
            float(self.curr_vol),
            float(self._baseline),
        ]
        
        # Add weights
        features.extend(self.weights.tolist())
        
        # Add sub-module features
        if self.consensus:
            features.extend(self.consensus.get_observation_components().tolist())
        if self.collusion:
            features.extend(self.collusion.get_observation_components().tolist())
            
        return np.array(features, dtype=np.float32)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information"""
        member_names = [m.__class__.__name__ for m in self.members]
        
        return {
            "members": member_names,
            "weights": self.weights.tolist(),
            "last_alpha": self.last_alpha.tolist() if self.last_alpha is not None else None,
            "gate_stats": {
                "attempts": self._gate_attempts,
                "passes": self._gate_passes,
                "pass_rate": self._gate_passes / max(self._gate_attempts, 1),
            },
            "baseline": float(self._baseline),
            "step_count": self._step_count,
            "bootstrap": self._step_count < self._bootstrap_steps,
        }