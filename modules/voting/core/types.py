# modules/voting/core/types.py
"""
Voting system data types.
Immutable dataclasses for vote proposals, bundles, and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

from .constants import VotingAction, VotingQuality


# ═══════════════════════════════════════════════════════════════════
# Vote Proposal (from individual experts)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VotingProposal:
    """
    A single vote from an expert.
    Immutable to ensure vote integrity throughout pipeline.
    """
    action: str                  # 'long', 'short', 'hold', 'abstain'
    confidence: float            # 0.0 - 1.0
    signal_strength: float       # 0.0 - 1.0 (intensity)
    reason: str                  # Human-readable explanation
    expert: str                  # Name of the voting expert
    timestamp: str               # ISO format timestamp
    
    # Optional enrichments
    instrument: Optional[str] = None
    timeframe: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Validate and clamp values
        object.__setattr__(self, 'confidence', float(np.clip(self.confidence, 0.0, 1.0)))
        object.__setattr__(self, 'signal_strength', float(np.clip(self.signal_strength, 0.0, 1.0)))
    
    @property
    def voting_action(self) -> VotingAction:
        """Parse action string to VotingAction enum."""
        return VotingAction.from_string(self.action)
    
    @property
    def canonical_action(self) -> str:
        """
        Canonical action string ('long', 'short', 'hold', 'abstain').
        
        Legacy aliases such as 'flat' or 'neutral' are normalized via
        VotingAction.from_string() and treated as neutral actions.
        """
        return self.voting_action.value
    
    @property
    def is_directional(self) -> bool:
        """True if this is a long or short signal."""
        return self.voting_action.is_directional
    
    @property
    def is_valid(self) -> bool:
        """
        Check if proposal has minimum required data.
        
        Uses VotingAction normalization so that legacy aliases like 'flat' or
        'neutral' are treated as valid neutral actions instead of being
        rejected outright.
        """
        action_enum = self.voting_action
        return (
            action_enum in (
                VotingAction.LONG,
                VotingAction.SHORT,
                VotingAction.HOLD,
                VotingAction.ABSTAIN,
            )
            and 0.0 <= self.confidence <= 1.0
            and 0.0 <= self.signal_strength <= 1.0
            and bool(self.expert)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for bus publication."""
        return {
            "action": self.action,
            "confidence": self.confidence,
            "signal_strength": self.signal_strength,
            "reason": self.reason,
            "expert": self.expert,
            "timestamp": self.timestamp,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "metadata": self.metadata or {},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VotingProposal":
        """Create from dictionary (bus consumption)."""
        return cls(
            action=str(data.get("action", "abstain")),
            confidence=float(data.get("confidence", 0.0)),
            signal_strength=float(data.get("signal_strength", data.get("intensity", 0.0))),
            reason=str(data.get("reason", data.get("thesis", ""))),
            expert=str(data.get("expert", data.get("module", "unknown"))),
            timestamp=str(data.get("timestamp", datetime.now().isoformat())),
            instrument=data.get("instrument"),
            timeframe=data.get("timeframe"),
            metadata=data.get("metadata"),
        )
    
    @classmethod
    def abstain(cls, expert: str, reason: str = "No signal") -> "VotingProposal":
        """Create an abstain proposal."""
        return cls(
            action="abstain",
            confidence=0.0,
            signal_strength=0.0,
            reason=reason,
            expert=expert,
            timestamp=datetime.now().isoformat(),
        )


# ═══════════════════════════════════════════════════════════════════
# Consensus Result (from consensus stage)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ConsensusResult:
    """Result from consensus analysis stage."""
    consensus_score: float           # 0.0 - 1.0 (agreement level)
    consensus_action: str            # Majority action
    vote_distribution: Dict[str, int]  # Count per action
    confidence_weighted_score: float # Weighted by expert confidence
    quality: str                     # 'high', 'medium', 'low', 'invalid'
    participating_experts: List[str]
    abstain_count: int
    timestamp: str
    
    def __post_init__(self):
        self.consensus_score = float(np.clip(self.consensus_score, 0.0, 1.0))
        self.confidence_weighted_score = float(np.clip(self.confidence_weighted_score, 0.0, 1.0))
    
    @property
    def has_quorum(self) -> bool:
        """Check if enough experts participated."""
        return len(self.participating_experts) >= 2
    
    @property
    def agreement_ratio(self) -> float:
        """Alias for confidence_weighted_score for backward compatibility."""
        return self.confidence_weighted_score
    
    @property
    def is_strong(self) -> bool:
        """Check if consensus is strong (>0.7)."""
        return self.consensus_score >= 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def empty(cls) -> "ConsensusResult":
        """Create empty/failed result."""
        return cls(
            consensus_score=0.0,
            consensus_action="abstain",
            vote_distribution={},
            confidence_weighted_score=0.0,
            quality="invalid",
            participating_experts=[],
            abstain_count=0,
            timestamp=datetime.now().isoformat(),
        )


# ═══════════════════════════════════════════════════════════════════
# Collusion Result (from collusion detection stage)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CollusionResult:
    """Result from collusion detection stage."""
    collusion_score: float           # 0.0 - 1.0 (1.0 = high collusion risk)
    is_suspicious: bool              # True if collusion detected
    correlation_matrix: Dict[str, Dict[str, float]]  # Expert-to-expert correlation
    flagged_pairs: List[tuple]       # Pairs with high correlation
    diversity_score: float           # 0.0 - 1.0 (1.0 = high diversity)
    adjustment_factor: float         # Multiplier for consensus (0.5-1.0)
    reason: str
    timestamp: str
    
    def __post_init__(self):
        self.collusion_score = float(np.clip(self.collusion_score, 0.0, 1.0))
        self.diversity_score = float(np.clip(self.diversity_score, 0.0, 1.0))
        self.adjustment_factor = float(np.clip(self.adjustment_factor, 0.5, 1.0))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "collusion_score": self.collusion_score,
            "is_suspicious": self.is_suspicious,
            "flagged_pairs": self.flagged_pairs,
            "diversity_score": self.diversity_score,
            "adjustment_factor": self.adjustment_factor,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }
    
    @property
    def collusion_detected(self) -> bool:
        """Alias for is_suspicious for backward compatibility."""
        return self.is_suspicious
    
    @property
    def suspicious_pairs(self) -> List[tuple]:
        """Alias for flagged_pairs for backward compatibility."""
        return self.flagged_pairs
    
    @property
    def thesis(self) -> str:
        """Alias for reason for backward compatibility."""
        return self.reason
    
    @classmethod
    def clean(cls) -> "CollusionResult":
        """Create result indicating no collusion."""
        return cls(
            collusion_score=0.0,
            is_suspicious=False,
            correlation_matrix={},
            flagged_pairs=[],
            diversity_score=1.0,
            adjustment_factor=1.0,
            reason="No collusion detected",
            timestamp=datetime.now().isoformat(),
        )


# ═══════════════════════════════════════════════════════════════════
# Vote Bundle (complete pipeline result)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VoteBundle:
    """
    Complete voting bundle after all pipeline stages.
    This is the final output of the voting kernel.
    """
    # Coordination
    decision_id: str
    tick_ts: str
    timestamp: str
    
    # Collected votes
    proposals: List[VotingProposal] = field(default_factory=list)
    
    # Stage results
    consensus: Optional[ConsensusResult] = None
    collusion: Optional[CollusionResult] = None
    
    # Horizon alignment
    aligned_weights: Dict[str, float] = field(default_factory=dict)
    horizon_score: float = 0.5
    
    # Uncertainty/fragility
    fragility: float = 0.0
    uncertainty_samples: int = 0
    
    # Final decision
    final_action: str = "abstain"
    final_confidence: float = 0.0
    final_intensity: float = 0.0
    
    # Meta
    processing_time_ms: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """Check if all stages completed successfully."""
        expected = ["committee", "consensus", "collusion", "horizon", "uncertainty", "arbiter"]
        return all(s in self.stages_completed for s in expected)
    
    @property
    def is_actionable(self) -> bool:
        """Check if this bundle has an actionable decision."""
        return (
            self.final_action in ("long", "short") and
            self.final_confidence >= 0.25 and
            self.final_intensity >= 0.15
        )
    
    @property
    def quality(self) -> VotingQuality:
        """Assess overall quality of the decision."""
        if not self.is_complete:
            return VotingQuality.INVALID
        if self.consensus and self.consensus.is_strong and self.final_confidence >= 0.6:
            return VotingQuality.HIGH
        if self.final_confidence >= 0.4:
            return VotingQuality.MEDIUM
        return VotingQuality.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for bus publication."""
        return {
            "decision_id": self.decision_id,
            "tick_ts": self.tick_ts,
            "timestamp": self.timestamp,
            "proposals": [p.to_dict() for p in self.proposals],
            "consensus": self.consensus.to_dict() if self.consensus else None,
            "collusion": self.collusion.to_dict() if self.collusion else None,
            "aligned_weights": self.aligned_weights,
            "horizon_score": self.horizon_score,
            "fragility": self.fragility,
            "uncertainty_samples": self.uncertainty_samples,
            "final_action": self.final_action,
            "final_confidence": self.final_confidence,
            "final_intensity": self.final_intensity,
            "processing_time_ms": self.processing_time_ms,
            "stages_completed": self.stages_completed,
            "warnings": self.warnings,
            "is_actionable": self.is_actionable,
            "quality": self.quality.value,
        }
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def mark_stage_complete(self, stage: str) -> None:
        """Mark a pipeline stage as completed."""
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)


# ═══════════════════════════════════════════════════════════════════
# Helper Factory Functions
# ═══════════════════════════════════════════════════════════════════

def create_empty_bundle(decision_id: str, tick_ts: str) -> VoteBundle:
    """Create an empty vote bundle for a new decision cycle."""
    return VoteBundle(
        decision_id=decision_id,
        tick_ts=tick_ts,
        timestamp=datetime.now().isoformat(),
    )


def create_abstain_bundle(decision_id: str, tick_ts: str, reason: str) -> VoteBundle:
    """Create a bundle that abstains from decision."""
    bundle = create_empty_bundle(decision_id, tick_ts)
    bundle.final_action = "abstain"
    bundle.final_confidence = 0.0
    bundle.final_intensity = 0.0
    bundle.add_warning(f"Abstain: {reason}")
    return bundle


def make_vote_bundle(
    decision_id: str = "",
    tick_ts: str = "",
    proposals: Optional[List[VotingProposal]] = None,
    **kwargs
) -> VoteBundle:
    """Factory function to create a VoteBundle with optional overrides."""
    return VoteBundle(
        decision_id=decision_id or f"vote_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tick_ts=tick_ts or datetime.now().isoformat(),
        timestamp=datetime.now().isoformat(),
        proposals=proposals or [],
        **kwargs
    )


def make_consensus_result(
    consensus_action: VotingAction = VotingAction.ABSTAIN,
    consensus_score: float = 0.0,
    agreement_ratio: float = 0.0,
    thesis: str = "",
    **kwargs
) -> ConsensusResult:
    """Factory function to create a ConsensusResult."""
    return ConsensusResult(
        consensus_score=consensus_score,
        consensus_action=consensus_action.value if isinstance(consensus_action, VotingAction) else str(consensus_action),
        vote_distribution=kwargs.get("vote_distribution", {}),
        confidence_weighted_score=kwargs.get("confidence_weighted_score", agreement_ratio),
        quality=kwargs.get("quality", "medium" if consensus_score >= 0.5 else "low"),
        participating_experts=kwargs.get("participating_experts", []),
        abstain_count=kwargs.get("abstain_count", 0),
        timestamp=datetime.now().isoformat(),
    )
