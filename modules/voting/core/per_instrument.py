"""
Voting system data types.
Immutable / structured dataclasses for proposals, bundles, and stage results.
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

    if v != v:  # NaN check
        return default

    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def normalize_instrument(symbol: str) -> str:
    """
    Normalize instrument symbol to a canonical format (no separators).
    
    Examples:
        'EUR/USD' -> 'EURUSD'
        'xau_usd' -> 'XAUUSD'
        'GOLD-USD' -> 'XAUUSD'
    """
    if not symbol:
        return ""
    s = symbol.upper().replace("/", "").replace("_", "").replace("-", "").strip()
    mapping = {"EURUSD": "EURUSD", "XAUUSD": "XAUUSD", "GOLDUSD": "XAUUSD"}
    return mapping.get(s, s)


# ═══════════════════════════════════════════════════════════════════
# Default Instruments
# ═══════════════════════════════════════════════════════════════════

DEFAULT_INSTRUMENTS: List[str] = ["XAUUSD", "EURUSD"]


# ═══════════════════════════════════════════════════════════════════
# Aggregated Instrument Decision
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AggregatedInstrumentDecision:
    """
    Result of aggregating votes for a single instrument.
    Used as return type of aggregate_all_instruments.
    """
    action: str = "flat"
    confidence: float = 0.5
    consensus_score: float = 0.0
    vote_count: int = 0
    long_votes: int = 0
    short_votes: int = 0
    flat_votes: int = 0
    weighted_score: float = 0.0
    members: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "consensus_score": self.consensus_score,
            "vote_count": self.vote_count,
            "long_votes": self.long_votes,
            "short_votes": self.short_votes,
            "flat_votes": self.flat_votes,
            "weighted_score": self.weighted_score,
            "members": self.members,
        }


def aggregate_all_instruments(
    votes: List["PerInstrumentVote"],
    instruments: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, AggregatedInstrumentDecision]:
    """
    Aggregate per-instrument votes across all voting members.
    
    Returns a dict keyed by instrument with AggregatedInstrumentDecision objects
    containing action, confidence, consensus score, vote counts, and weighted score.
    
    Args:
        votes: List of PerInstrumentVote from all experts
        instruments: List of instruments to aggregate (defaults to DEFAULT_INSTRUMENTS)
        weights: Optional dict mapping member names to their voting weights (0.0-1.0)
    
    Returns:
        Dict[instrument, AggregatedInstrumentDecision]
    """
    if instruments is None:
        instruments = DEFAULT_INSTRUMENTS
    if weights is None:
        weights = {}
    
    result: Dict[str, AggregatedInstrumentDecision] = {}
    
    for inst in instruments:
        inst_norm = normalize_instrument(inst)
        long_count = 0
        short_count = 0
        flat_count = 0
        confidences: List[float] = []
        members: List[str] = []
        weighted_sum: float = 0.0  # For weighted direction scoring
        total_weight: float = 0.0
        
        for vote in votes:
            prop = vote.get_proposal(inst_norm)
            if prop:
                action = prop.action.lower()
                member_weight = weights.get(vote.member, 1.0)
                
                # Count by action
                if action in ("long", "buy", "bullish"):
                    long_count += 1
                    weighted_sum += member_weight * prop.confidence
                elif action in ("short", "sell", "bearish"):
                    short_count += 1
                    weighted_sum -= member_weight * prop.confidence
                else:
                    flat_count += 1
                    # Flat/neutral doesn't contribute to direction
                
                total_weight += member_weight
                confidences.append(prop.confidence)
                members.append(vote.member)
        
        vote_count = long_count + short_count + flat_count
        
        # Determine dominant action based on weighted score
        if weighted_sum > 0.1:
            dominant_action = "long"
        elif weighted_sum < -0.1:
            dominant_action = "short"
        else:
            dominant_action = "flat"
        
        # Compute consensus: how much agreement is there among voters?
        # Consensus is high if votes skew heavily one direction
        if vote_count > 0:
            max_votes = max(long_count, short_count, flat_count)
            consensus_score = max_votes / vote_count
        else:
            consensus_score = 0.0
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        result[inst_norm] = AggregatedInstrumentDecision(
            action=dominant_action,
            confidence=avg_confidence,
            consensus_score=consensus_score,
            vote_count=vote_count,
            long_votes=long_count,
            short_votes=short_count,
            flat_votes=flat_count,
            weighted_score=weighted_sum / max(total_weight, 1.0),
            members=members,
        )
    
    return result


def extract_instrument_data(data: Dict[str, Any], instrument: str) -> Dict[str, Any]:
    """
    Extract data for a specific instrument from nested market/feature data.
    
    Looks up the instrument using various key variations:
    - Exact match
    - Normalized form (no separators)
    - Upper/lower case variations
    - Common separator variations (EUR_USD, EUR/USD, EURUSD)
    
    Also handles SmartInfoBus wrapper format: {'value': {...}, 'timestamp': ..., 'version': ...}
    
    Args:
        data: Dict containing instrument-keyed data
        instrument: Instrument symbol to look up
        
    Returns:
        Dict with instrument-specific data, or original dict if not found
    """
    if not isinstance(data, dict):
        return {}
    
    # Handle SmartInfoBus wrapper format: {'value': {...}, 'timestamp': ..., 'version': ...}
    if 'value' in data and isinstance(data['value'], dict) and 'timestamp' in data:
        data = data['value']
    
    inst_norm = normalize_instrument(instrument)
    
    # Try various key formats
    for key in [instrument, inst_norm, instrument.upper(), instrument.lower()]:
        if key in data:
            return data[key] if isinstance(data[key], dict) else data
    
    # Try with common separators
    for sep in ["_", "/", "-", ""]:
        for pair in [f"EUR{sep}USD", f"XAU{sep}USD"]:
            norm_pair = normalize_instrument(pair)
            if norm_pair == inst_norm and pair in data:
                return data[pair] if isinstance(data[pair], dict) else data
    
    # Fallback: return original data (legacy flat format)
    return data


# ═══════════════════════════════════════════════════════════════════
# Vote Proposal (from individual experts)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VotingProposal:
    """
    A single vote from an expert.
    Immutable to ensure vote integrity throughout the pipeline.
    """
    action: str
    confidence: float
    signal_strength: float
    reason: str
    expert: str
    timestamp: str

    # Optional enrichments
    instrument: Optional[str] = None
    timeframe: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        object.__setattr__(self, "confidence", _clamp_01(self.confidence))
        object.__setattr__(
            self,
            "signal_strength",
            _clamp_01(self.signal_strength, default=0.0),
        )

        normalized_action = self.voting_action.value
        object.__setattr__(self, "action", normalized_action)

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now_iso())

    @property
    def voting_action(self) -> VotingAction:
        """Parse action string/enum to VotingAction enum."""
        return VotingAction.from_string(self.action)

    @property
    def canonical_action(self) -> str:
        """Canonical action string ('long', 'short', 'hold', 'abstain')."""
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
        'neutral' are treated as valid neutral actions.
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
# Per-Instrument Voting Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InstrumentProposal:
    """
    A vote proposal for a specific instrument.
    Used within PerInstrumentVote to track per-instrument decisions.
    """
    instrument: str
    action: str = "flat"
    confidence: float = 0.5
    magnitude: float = 0.5
    rationale: str = ""
    
    def __post_init__(self):
        self.confidence = _clamp_01(self.confidence)
        self.magnitude = _clamp_01(self.magnitude)
        self.action = self.action.lower() if self.action else "flat"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instrument": self.instrument,
            "action": self.action,
            "confidence": self.confidence,
            "magnitude": self.magnitude,
            "rationale": self.rationale,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstrumentProposal":
        """Create from dictionary."""
        return cls(
            instrument=str(data.get("instrument", "UNKNOWN")),
            action=str(data.get("action", "flat")),
            confidence=float(data.get("confidence", 0.5)),
            magnitude=float(data.get("magnitude", data.get("signal_strength", 0.5))),
            rationale=str(data.get("rationale", data.get("reason", ""))),
        )


@dataclass
class PerInstrumentVote:
    """
    Container for per-instrument voting proposals from a single expert/member.
    
    Tracks multiple instrument-specific votes from one voting member,
    allowing the committee to aggregate votes on a per-instrument basis.
    """
    member: str
    proposals: Dict[str, InstrumentProposal] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)
    
    def set_proposal(self, proposal: InstrumentProposal) -> None:
        """Add or update a proposal for an instrument."""
        self.proposals[proposal.instrument] = proposal
    
    def get_proposal(self, instrument: str) -> Optional[InstrumentProposal]:
        """Get proposal for a specific instrument."""
        return self.proposals.get(instrument)
    
    def get_action(self, instrument: str, default: str = "flat") -> str:
        """Get action for instrument, with default."""
        prop = self.proposals.get(instrument)
        return prop.action if prop else default
    
    def get_confidence(self, instrument: str, default: float = 0.5) -> float:
        """Get confidence for instrument, with default."""
        prop = self.proposals.get(instrument)
        return prop.confidence if prop else default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "member": self.member,
            "proposals": {
                inst: prop.to_dict() for inst, prop in self.proposals.items()
            },
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerInstrumentVote":
        """
        Create from dictionary.
        
        Supports both:
        - proposals as dict of InstrumentProposal dicts
        - Legacy format with global action/confidence applied to all instruments
        """
        member = str(data.get("member", "unknown"))
        proposals_data = data.get("proposals", {})
        
        piv = cls(member=member)
        
        if isinstance(proposals_data, dict):
            for inst, prop_data in proposals_data.items():
                if isinstance(prop_data, dict):
                    prop_data.setdefault("instrument", inst)
                    piv.set_proposal(InstrumentProposal.from_dict(prop_data))
                elif isinstance(prop_data, InstrumentProposal):
                    piv.set_proposal(prop_data)
        
        return piv


# ═══════════════════════════════════════════════════════════════════
# Consensus Result (from consensus stage)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ConsensusResult:
    """Result from consensus analysis stage."""
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
        self.confidence_weighted_score = _clamp_01(
            self.confidence_weighted_score
        )
        if not isinstance(self.vote_distribution, dict):
            self.vote_distribution = dict(self.vote_distribution or {})
        if not self.timestamp:
            self.timestamp = _now_iso()

    @property
    def has_quorum(self) -> bool:
        """True if enough experts participated."""
        return len(self.participating_experts) >= 2

    @property
    def agreement_ratio(self) -> float:
        """Alias for confidence_weighted_score (backward compatibility)."""
        return self.confidence_weighted_score

    @property
    def is_strong(self) -> bool:
        """True if consensus is strong (> 0.7)."""
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
        """Alias for is_suspicious."""
        return self.is_suspicious

    @property
    def suspicious_pairs(self) -> List[tuple]:
        """Alias for flagged_pairs."""
        return self.flagged_pairs

    @property
    def thesis(self) -> str:
        """Alias for reason."""
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
    Typically created by the monolithic voting kernel (legacy path) or
    by higher-level orchestrators when a structured result is preferred.
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

    # Uncertainty / fragility
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

        if self.final_action:
            self.final_action = VotingAction.from_string(self.final_action).value
        else:
            self.final_action = VotingAction.ABSTAIN.value

        if self.aligned_weights is None:
            self.aligned_weights = {}
        else:
            self.aligned_weights = {
                str(k): float(v) for k, v in self.aligned_weights.items()
            }

    @property
    def is_complete(self) -> bool:
        """True if all pipeline stages completed successfully."""
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
        True if this bundle has an actionable directional decision.

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
        """Add a warning message (deduplicated)."""
        if warning and warning not in self.warnings:
            self.warnings.append(warning)

    def mark_stage_complete(self, stage: str) -> None:
        """Mark a pipeline stage as completed."""
        if stage and stage not in self.stages_completed:
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


def create_abstain_bundle(
    decision_id: str, tick_ts: str, reason: str
) -> VoteBundle:
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
        decision_id=decision_id
        or f"vote_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
