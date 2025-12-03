# modules/voting/core/types.py
"""
Voting system data types.
Immutable dataclasses for vote proposals, bundles, and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .constants import (
    VotingAction,
    VotingQuality,
    get_thresholds,
    FRAGILITY_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    """Return current time in ISO 8601 format."""
    return datetime.now().isoformat()


def _clamp_01(value: float, default: float = 0.0) -> float:
    """
    Clamp a value into [0.0, 1.0].

    - Non-numeric → default
    - NaN → default
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default

    # NaN check: NaN != NaN
    if v != v:
        return default

    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


# ═══════════════════════════════════════════════════════════════════
# Vote Proposal (from individual experts)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VotingProposal:
    """
    A single vote from an expert.
    Immutable to ensure vote integrity throughout pipeline.
    """
    action: str                  # 'long', 'short', 'hold', 'abstain' (canonical)
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
        # Normalize numeric fields
        object.__setattr__(self, "confidence", _clamp_01(self.confidence))
        object.__setattr__(
            self,
            "signal_strength",
            _clamp_01(self.signal_strength, default=0.0),
        )

        # Normalize action to canonical form using VotingAction
        normalized_action = self.voting_action.value
        object.__setattr__(self, "action", normalized_action)

        # Ensure metadata is always a dict internally
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        # Ensure timestamp is present
        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now_iso())

    @property
    def voting_action(self) -> VotingAction:
        """Parse action string/enum to VotingAction enum."""
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
            action_enum
            in (
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
            "action": self.canonical_action,
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
            action=str(data.get("action", data.get("canonical_action", "abstain"))),
            confidence=float(data.get("confidence", 0.0)),
            signal_strength=float(
                data.get("signal_strength", data.get("intensity", 0.0))
            ),
            reason=str(data.get("reason", data.get("thesis", ""))),
            expert=str(data.get("expert", data.get("module", "unknown"))),
            timestamp=str(data.get("timestamp", _now_iso())),
            instrument=data.get("instrument"),
            timeframe=data.get("timeframe"),
            metadata=data.get("metadata") or {},
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
            timestamp=_now_iso(),
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
        self.consensus_score = _clamp_01(self.consensus_score)
        self.confidence_weighted_score = _clamp_01(
            self.confidence_weighted_score
        )
        if not isinstance(self.vote_distribution, dict):
            self.vote_distribution = dict(self.vote_distribution or {})
        if not self.timestamp:
            self.timestamp = _now_iso()

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

    @property
    def quality_enum(self) -> VotingQuality:
        """Return quality as VotingQuality enum."""
        try:
            return VotingQuality(self.quality)
        except ValueError:
            return VotingQuality.INVALID

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
            timestamp=_now_iso(),
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
        self.collusion_score = _clamp_01(self.collusion_score)
        self.diversity_score = _clamp_01(self.diversity_score)
        # Keep adjustment in [0.5, 1.0] so we never amplify consensus
        try:
            adj = float(self.adjustment_factor)
        except (TypeError, ValueError):
            adj = 1.0
        if adj < 0.5:
            adj = 0.5
        if adj > 1.0:
            adj = 1.0
        self.adjustment_factor = adj

        if self.correlation_matrix is None:
            self.correlation_matrix = {}
        if self.flagged_pairs is None:
            self.flagged_pairs = []
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        # correlation_matrix can be large, but is extremely useful for debugging
        return {
            "collusion_score": self.collusion_score,
            "is_suspicious": self.is_suspicious,
            "correlation_matrix": self.correlation_matrix,
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
            timestamp=_now_iso(),
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

    def __post_init__(self):
        self.horizon_score = _clamp_01(self.horizon_score, default=0.5)
        self.fragility = _clamp_01(self.fragility, default=0.0)
        self.final_confidence = _clamp_01(self.final_confidence, default=0.0)
        self.final_intensity = _clamp_01(self.final_intensity, default=0.0)

        if not self.timestamp:
            self.timestamp = _now_iso()

        # Normalize final_action to canonical form
        if self.final_action:
            self.final_action = VotingAction.from_string(self.final_action).value
        else:
            self.final_action = VotingAction.ABSTAIN.value

        if self.aligned_weights is None:
            self.aligned_weights = {}
        else:
            # Ensure keys are strings and values are floats
            self.aligned_weights = {
                str(k): float(v) for k, v in self.aligned_weights.items()
            }

    @property
    def is_complete(self) -> bool:
        """Check if all stages completed successfully."""
        expected = [
            "committee",
            "consensus",
            "collusion",
            "horizon",
            "uncertainty",
            "arbiter",
        ]
        return all(stage in self.stages_completed for stage in expected)

    @property
    def is_actionable(self) -> bool:
        """
        Check if this bundle has an actionable decision.

        Uses mode-aware thresholds from constants.get_thresholds():
        - ARBITER_CONFIDENCE_FLOOR
        - ARBITER_INTENSITY_FLOOR
        """
        thresholds = get_thresholds()
        conf_floor = float(thresholds["ARBITER_CONFIDENCE_FLOOR"])
        intensity_floor = float(thresholds["ARBITER_INTENSITY_FLOOR"])

        return (
            self.final_action in (VotingAction.LONG.value, VotingAction.SHORT.value)
            and self.final_confidence >= conf_floor
            and self.final_intensity >= intensity_floor
        )

    @property
    def quality(self) -> VotingQuality:
        """Assess overall quality of the decision."""
        if not self.is_complete:
            return VotingQuality.INVALID

        thresholds = get_thresholds()
        high_conf = float(thresholds["HIGH_CONFIDENCE_THRESHOLD"])
        base_conf = float(thresholds["CONFIDENCE_THRESHOLD"])

        # High quality: strong consensus, high confidence, low fragility
        if (
            self.consensus
            and self.consensus.is_strong
            and self.final_confidence >= high_conf
            and self.fragility <= FRAGILITY_THRESHOLD
        ):
            return VotingQuality.HIGH

        # Medium quality: above base confidence threshold
        if self.final_confidence >= base_conf:
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
            "stages_completed": list(self.stages_completed),
            "warnings": list(self.warnings),
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
        timestamp=_now_iso(),
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
    **kwargs: Any,
) -> VoteBundle:
    """Factory function to create a VoteBundle with optional overrides."""
    return VoteBundle(
        decision_id=decision_id or f"vote_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tick_ts=tick_ts or _now_iso(),
        timestamp=_now_iso(),
        proposals=proposals or [],
        **kwargs,
    )


def make_consensus_result(
    consensus_action: VotingAction = VotingAction.ABSTAIN,
    consensus_score: float = 0.0,
    agreement_ratio: float = 0.0,
    thesis: str = "",  # kept for backward compatibility, not stored directly
    **kwargs: Any,
) -> ConsensusResult:
    """Factory function to create a ConsensusResult."""
    return ConsensusResult(
        consensus_score=consensus_score,
        consensus_action=(
            consensus_action.value
            if isinstance(consensus_action, VotingAction)
            else str(consensus_action)
        ),
        vote_distribution=kwargs.get("vote_distribution", {}),
        confidence_weighted_score=kwargs.get(
            "confidence_weighted_score", agreement_ratio
        ),
        quality=kwargs.get(
            "quality",
            "medium" if consensus_score >= 0.5 else "low",
        ),
        participating_experts=kwargs.get("participating_experts", []),
        abstain_count=kwargs.get("abstain_count", 0),
        timestamp=_now_iso(),
    )
