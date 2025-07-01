
# modules/voting/strategy_arbiter.py

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from modules.core.core import Module
from utils.get_dir import _BASE_GATE, _smart_gate


class StrategyArbiter(Module):
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