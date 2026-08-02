"""
Voting system data types.
Immutable dataclasses for vote proposals, bundles, and results.

Upgrades:
- Preserves position-management semantics at the type level (exit/tighten)
- Stronger normalization + safer dict conversions
- Instrument normalization compatibility is handled by callers (base provides canon())
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .constants import (
    FRAGILITY_THRESHOLD,
    VotingAction,
    VotingQuality,
    get_thresholds,
)

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now().isoformat()

def _clamp_01(value: float, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default

    if v != v:  # NaN
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
    action: str
    confidence: float
    signal_strength: float
    reason: str
    expert: str
    timestamp: str

    instrument: Optional[str] = None
    timeframe: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        object.__setattr__(self, "confidence", _clamp_01(self.confidence))
        object.__setattr__(self, "signal_strength", _clamp_01(self.signal_strength, default=0.0))

        normalized_action = self.voting_action.value
        object.__setattr__(self, "action", normalized_action)

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now_iso())

    @property
    def voting_action(self) -> VotingAction:
        return VotingAction.from_string(self.action)

    @property
    def canonical_action(self) -> str:
        return self.voting_action.value

    @property
    def is_directional(self) -> bool:
        return self.voting_action.is_directional

    @property
    def is_valid(self) -> bool:
        action_enum = self.voting_action
        return (
            action_enum in (
                VotingAction.LONG,
                VotingAction.SHORT,
                VotingAction.HOLD,
                VotingAction.ABSTAIN,
                VotingAction.EXIT,
                VotingAction.TIGHTEN,
            )
            and 0.0 <= self.confidence <= 1.0
            and 0.0 <= self.signal_strength <= 1.0
            and bool(self.expert)
        )

    def to_dict(self) -> Dict[str, Any]:
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
        return cls(
            action=str(data.get("action", data.get("canonical_action", "abstain"))),
            confidence=float(data.get("confidence", 0.0)),
            signal_strength=float(data.get("signal_strength", data.get("intensity", 0.0))),
            reason=str(data.get("reason", data.get("thesis", ""))),
            expert=str(data.get("expert", data.get("module", "unknown"))),
            timestamp=str(data.get("timestamp", _now_iso())),
            instrument=data.get("instrument"),
            timeframe=data.get("timeframe"),
            metadata=data.get("metadata") or {},
        )

    @classmethod
    def abstain(cls, expert: str, reason: str = "No signal") -> "VotingProposal":
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
    consensus_score: float
    consensus_action: str
    vote_distribution: Dict[str, int]
    confidence_weighted_score: float
    quality: str
    participating_experts: List[str]
    abstain_count: int
    timestamp: str

    def __post_init__(self):
        self.consensus_score = _clamp_01(self.consensus_score)
        self.confidence_weighted_score = _clamp_01(self.confidence_weighted_score)
        if not isinstance(self.vote_distribution, dict):
            self.vote_distribution = dict(self.vote_distribution or {})
        if not self.timestamp:
            self.timestamp = _now_iso()

        # Normalize to canonical action strings where possible
        self.consensus_action = VotingAction.from_string(self.consensus_action).value

    @property
    def has_quorum(self) -> bool:
        return len(self.participating_experts) >= 2

    @property
    def agreement_ratio(self) -> float:
        return self.confidence_weighted_score

    @property
    def is_strong(self) -> bool:
        return self.consensus_score >= 0.7

    @property
    def quality_enum(self) -> VotingQuality:
        try:
            return VotingQuality(self.quality)
        except ValueError:
            return VotingQuality.INVALID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def empty(cls) -> "ConsensusResult":
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
    collusion_score: float
    is_suspicious: bool
    correlation_matrix: Dict[str, Dict[str, float]]
    flagged_pairs: List[tuple]
    diversity_score: float
    adjustment_factor: float
    reason: str
    timestamp: str

    def __post_init__(self):
        self.collusion_score = _clamp_01(self.collusion_score)
        self.diversity_score = _clamp_01(self.diversity_score)

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
        return self.is_suspicious

    @property
    def suspicious_pairs(self) -> List[tuple]:
        return self.flagged_pairs

    @property
    def thesis(self) -> str:
        return self.reason

    @classmethod
    def clean(cls) -> "CollusionResult":
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
    decision_id: str
    tick_ts: str
    timestamp: str

    proposals: List[VotingProposal] = field(default_factory=list)

    consensus: Optional[ConsensusResult] = None
    collusion: Optional[CollusionResult] = None

    aligned_weights: Dict[str, float] = field(default_factory=dict)
    horizon_score: float = 0.5

    fragility: float = 0.0
    uncertainty_samples: int = 0

    final_action: str = "abstain"
    final_confidence: float = 0.0
    final_intensity: float = 0.0

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

        self.final_action = VotingAction.from_string(self.final_action).value if self.final_action else VotingAction.ABSTAIN.value

        if self.aligned_weights is None:
            self.aligned_weights = {}
        else:
            self.aligned_weights = {str(k): float(v) for k, v in self.aligned_weights.items()}

    @property
    def is_complete(self) -> bool:
        expected = ["committee", "consensus", "collusion", "horizon", "uncertainty", "arbiter"]
        return all(stage in self.stages_completed for stage in expected)

    @property
    def is_actionable(self) -> bool:
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
        if not self.is_complete:
            return VotingQuality.INVALID

        thresholds = get_thresholds()
        high_conf = float(thresholds["HIGH_CONFIDENCE_THRESHOLD"])
        base_conf = float(thresholds["CONFIDENCE_THRESHOLD"])

        if (
            self.consensus
            and self.consensus.is_strong
            and self.final_confidence >= high_conf
            and self.fragility <= FRAGILITY_THRESHOLD
        ):
            return VotingQuality.HIGH

        if self.final_confidence >= base_conf:
            return VotingQuality.MEDIUM

        return VotingQuality.LOW

    def to_dict(self) -> Dict[str, Any]:
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
        if warning not in self.warnings:
            self.warnings.append(warning)

    def mark_stage_complete(self, stage: str) -> None:
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)


# ═══════════════════════════════════════════════════════════════════
# Helper Factory Functions
# ═══════════════════════════════════════════════════════════════════

def create_empty_bundle(decision_id: str, tick_ts: str) -> VoteBundle:
    return VoteBundle(decision_id=decision_id, tick_ts=tick_ts, timestamp=_now_iso())

def create_abstain_bundle(decision_id: str, tick_ts: str, reason: str) -> VoteBundle:
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
    thesis: str = "",
    **kwargs: Any,
) -> ConsensusResult:
    return ConsensusResult(
        consensus_score=consensus_score,
        consensus_action=(
            consensus_action.value if isinstance(consensus_action, VotingAction) else str(consensus_action)
        ),
        vote_distribution=kwargs.get("vote_distribution", {}),
        confidence_weighted_score=kwargs.get("confidence_weighted_score", agreement_ratio),
        quality=kwargs.get("quality", "medium" if consensus_score >= 0.5 else "low"),
        participating_experts=kwargs.get("participating_experts", []),
        abstain_count=kwargs.get("abstain_count", 0),
        timestamp=_now_iso(),
    )
