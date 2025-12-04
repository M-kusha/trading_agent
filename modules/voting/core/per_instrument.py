# modules/voting/core/per_instrument.py
"""
Per-Instrument Voting Infrastructure
====================================
Provides utilities and data structures for per-instrument voting.
All voting members should use these to produce per-instrument proposals.

Design principles:
- Each instrument (EURUSD, XAUUSD) gets its own vote
- Votes are aggregated per-instrument by CommitteeCoordinator
- FinalArbiter receives per-instrument decisions
- Downstream modules (PositionManager, Executor) already handle per-instrument

Usage in voting members:
    from modules.voting.core.per_instrument import (
        InstrumentProposal,
        PerInstrumentVote,
        create_per_instrument_vote,
        DEFAULT_INSTRUMENTS,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from .constants import VotingAction, get_thresholds


# ═══════════════════════════════════════════════════════════════════
# Helpers / Constants
# ═══════════════════════════════════════════════════════════════════

# Default instruments - should match system_config.yaml
DEFAULT_INSTRUMENTS: List[str] = ["EURUSD", "XAUUSD"]


def _now_iso() -> str:
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


def normalize_instrument(inst: str) -> str:
    """
    Normalize instrument name to canonical format.

    Handles:
    - Case insensitivity
    - Separators ('EUR_USD', 'eur/usd', 'XAU-USD', etc.)
    """
    if inst is None:
        return "UNKNOWN"

    s = str(inst).strip()
    if not s:
        return "UNKNOWN"

    # Remove separators and upper-case
    key = s.replace("/", "").replace("_", "").replace("-", "").upper()

    if key == "EURUSD":
        return "EURUSD"
    if key == "XAUUSD":
        return "XAUUSD"

    # Fallback: normalized but not recognized
    return key


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InstrumentProposal:
    """
    A voting proposal for a single instrument.

    Attributes:
        instrument: The instrument symbol (e.g., 'EURUSD')
        action: The proposed action ('long', 'short', 'hold', 'abstain', legacy 'flat')
        confidence: Confidence in this proposal (0.0 to 1.0)
        magnitude: Signal strength/intensity (0.0 to 1.0)
        horizon: Time horizon ('scalp', 'intraday', 'swing')
        rationale: Human-readable explanation
        meta: Additional metadata
    """
    instrument: str
    action: str = "flat"  # will be normalized to canonical via VotingAction
    confidence: float = 0.5
    magnitude: float = 0.0
    horizon: str = "intraday"
    rationale: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize instrument and action
        self.instrument = normalize_instrument(self.instrument)
        self.action = VotingAction.from_string(self.action).value

        # Clamp numeric fields
        self.confidence = _clamp_01(self.confidence, default=0.0)
        self.magnitude = _clamp_01(self.magnitude, default=0.0)

        if self.meta is None:
            self.meta = {}

    @property
    def voting_action(self) -> VotingAction:
        return VotingAction.from_string(self.action)

    @property
    def is_directional(self) -> bool:
        """Returns True if this is a directional signal."""
        return self.voting_action.is_directional

    @property
    def is_neutral(self) -> bool:
        """True for non-directional but valid actions (hold/abstain)."""
        act = self.voting_action
        return act in (VotingAction.HOLD, VotingAction.ABSTAIN)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for bus publishing."""
        return {
            "instrument": self.instrument,
            "action": self.action,  # canonical
            "confidence": round(self.confidence, 4),
            "magnitude": round(self.magnitude, 4),
            "horizon": self.horizon,
            "rationale": self.rationale,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstrumentProposal":
        """Create from dictionary."""
        return cls(
            instrument=str(data.get("instrument", "UNKNOWN")),
            action=str(data.get("action", "flat")),
            confidence=float(data.get("confidence", 0.5)),
            magnitude=float(data.get("magnitude", 0.0)),
            horizon=str(data.get("horizon", "intraday")),
            rationale=str(data.get("rationale", "")),
            meta=dict(data.get("meta", {})),
        )


@dataclass
class PerInstrumentVote:
    """
    A complete per-instrument vote from a voting member.

    Contains proposals for ALL instruments the member analyzed.
    """
    member: str
    proposals: Dict[str, InstrumentProposal] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    # Legacy compatibility: also track a "global" fallback for old consumers
    global_action: str = "flat"
    global_confidence: float = 0.5

    def __post_init__(self) -> None:
        # Normalize timestamp
        if not self.timestamp:
            self.timestamp = _now_iso()

        # Normalize global_action
        self.global_action = VotingAction.from_string(self.global_action).value
        self.global_confidence = _clamp_01(self.global_confidence, default=0.0)

        # Ensure proposals are properly initialized
        normalized: Dict[str, InstrumentProposal] = {}
        for inst, prop in self.proposals.items():
            if not isinstance(prop, InstrumentProposal):
                prop = InstrumentProposal.from_dict(prop)
            prop.instrument = normalize_instrument(prop.instrument or inst)
            normalized[prop.instrument] = prop
        if normalized:
            self.proposals = normalized

        # Update global fallback from proposals if available
        if self.proposals:
            self._update_global()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for bus publishing."""
        return {
            "member": self.member,
            "proposals": {k: v.to_dict() for k, v in self.proposals.items()},
            "timestamp": self.timestamp,
            # Legacy compatibility fields
            "action": self.global_action,
            "confidence": self.global_confidence,
            "proposal": {
                "direction": self.global_action,
                "magnitude": max((p.magnitude for p in self.proposals.values()), default=0.0),
                "horizon": "intraday",
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerInstrumentVote":
        """Create from dictionary."""
        proposals: Dict[str, InstrumentProposal] = {}
        raw_props = data.get("proposals", {})
        if isinstance(raw_props, dict):
            for inst, prop_data in raw_props.items():
                proposals[inst] = InstrumentProposal.from_dict(prop_data)

        return cls(
            member=str(data.get("member", "Unknown")),
            proposals=proposals,
            timestamp=str(data.get("timestamp", _now_iso())),
            global_action=str(data.get("action", "flat")),
            global_confidence=float(data.get("confidence", 0.5)),
        )

    def get_proposal(self, instrument: str) -> Optional[InstrumentProposal]:
        """Get proposal for a specific instrument."""
        inst = normalize_instrument(instrument)
        return self.proposals.get(inst)

    def set_proposal(self, proposal: InstrumentProposal) -> None:
        """Set proposal for an instrument."""
        inst = normalize_instrument(proposal.instrument)
        proposal.instrument = inst
        self.proposals[inst] = proposal
        self._update_global()

    def _update_global(self) -> None:
        """
        Update global fallback from per-instrument proposals.

        Global action is chosen as:
        - Prefer directional proposals with highest confidence * magnitude
        - Fallback to strongest neutral proposal
        """
        if not self.proposals:
            return

        best_prop: Optional[InstrumentProposal] = None
        best_score = -1.0

        for prop in self.proposals.values():
            conf = _clamp_01(prop.confidence, default=0.0)
            mag = _clamp_01(prop.magnitude, default=0.0)
            act = prop.voting_action

            # Directional proposals get full weight, neutral ones get half
            base = conf * mag if act.is_directional else conf * mag * 0.5
            if base > best_score:
                best_score = base
                best_prop = prop

        if best_prop is not None:
            self.global_action = best_prop.voting_action.value
            self.global_confidence = _clamp_01(best_prop.confidence, default=0.0)


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def create_per_instrument_vote(
    member: str,
    proposals: Dict[str, Dict[str, Any]],
) -> PerInstrumentVote:
    """
    Create a PerInstrumentVote from a dictionary of proposals.

    Args:
        member: Name of the voting member
        proposals: Dict mapping instrument -> proposal dict
                   Each proposal dict should have: action, confidence, magnitude, etc.

    Returns:
        PerInstrumentVote instance
    """
    vote = PerInstrumentVote(member=member)

    for inst, prop_data in proposals.items():
        prop = InstrumentProposal(
            instrument=normalize_instrument(inst),
            action=str(prop_data.get("action", "flat")),
            confidence=float(prop_data.get("confidence", 0.5)),
            magnitude=float(prop_data.get("magnitude", 0.0)),
            horizon=str(prop_data.get("horizon", "intraday")),
            rationale=str(prop_data.get("rationale", "")),
            meta=dict(prop_data.get("meta", {})),
        )
        vote.set_proposal(prop)

    return vote


def create_flat_vote(member: str, instruments: Optional[List[str]] = None) -> PerInstrumentVote:
    """
    Create a neutral (no directional) vote for all instruments.

    This represents "I have evaluated, but I have no directional signal",
    which maps to a canonical HOLD action.
    """
    instruments = instruments or DEFAULT_INSTRUMENTS
    vote = PerInstrumentVote(member=member)

    for inst in instruments:
        vote.set_proposal(
            InstrumentProposal(
                instrument=inst,
                action="hold",      # canonical neutral
                confidence=0.3,     # low but non-zero confidence
                magnitude=0.0,
                rationale="No signal",
            )
        )

    return vote


def extract_instrument_data(
    market_data: Dict[str, Any],
    instrument: str,
) -> Dict[str, Any]:
    """
    Extract market data for a specific instrument.

    Handles mixed formats:
    - market_data[instrument] = {...}  (direct)
    - market_data['EURUSD'], market_data['eur_usd'], market_data['EUR/USD'], etc.
    """
    inst = normalize_instrument(instrument)

    # Try direct canonical key
    if inst in market_data:
        return market_data[inst]

    # Try matching by normalized key of any entry
    for key, value in market_data.items():
        if normalize_instrument(key) == inst:
            return value

    return {}


def analyze_instrument_trend(
    price_data: Dict[str, Any],
    indicators: Dict[str, Any],
) -> Tuple[str, float]:
    """
    Analyze trend direction for a single instrument.

    Returns:
        Tuple of (direction: 'long'|'short'|'hold', strength: 0.0-1.0)
    """
    try:
        # Get price movement
        open_price = float(price_data.get("open", 0) or 0)
        close_price = float(price_data.get("close", price_data.get("last", 0)) or 0)

        if open_price <= 0 or close_price <= 0:
            return VotingAction.HOLD.value, 0.0

        # Candle direction
        candle_change = (close_price - open_price) / open_price

        # RSI
        rsi = float(indicators.get("rsi", 50) or 50)

        # Momentum
        momentum = float(indicators.get("momentum", indicators.get("roc", 0)) or 0)

        signals: List[float] = []

        # Candle signal
        if abs(candle_change) > 0.001:  # 0.1% minimum
            signals.append(1.0 if candle_change > 0 else -1.0)

        # RSI signal
        if rsi > 60:
            signals.append(1.0 * min(1.0, (rsi - 50) / 30))
        elif rsi < 40:
            signals.append(-1.0 * min(1.0, (50 - rsi) / 30))

        # Momentum signal
        if abs(momentum) > 0.1:
            m = max(-1.0, min(1.0, momentum / 2.0))
            signals.append(m)

        if not signals:
            return VotingAction.HOLD.value, 0.0

        avg_signal = float(np.mean(signals))
        strength = abs(avg_signal)

        if strength < 0.2:
            return VotingAction.HOLD.value, float(strength)
        elif avg_signal > 0:
            return VotingAction.LONG.value, float(strength)
        else:
            return VotingAction.SHORT.value, float(strength)

    except Exception:
        return VotingAction.HOLD.value, 0.0


# ═══════════════════════════════════════════════════════════════════
# Aggregation Helpers (for CommitteeCoordinator)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AggregatedInstrumentDecision:
    """Aggregated committee decision for a single instrument."""
    instrument: str
    action: str = VotingAction.ABSTAIN.value
    confidence: float = 0.0
    consensus_score: float = 0.0
    vote_count: int = 0
    long_votes: int = 0
    short_votes: int = 0
    flat_votes: int = 0  # neutral/non-directional votes
    weighted_score: float = 0.0

    def __post_init__(self) -> None:
        self.instrument = normalize_instrument(self.instrument)
        self.action = VotingAction.from_string(self.action).value
        self.confidence = _clamp_01(self.confidence, default=0.0)
        self.consensus_score = _clamp_01(self.consensus_score, default=0.0)
        self.weighted_score = _clamp_01(self.weighted_score, default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_instrument_votes(
    votes: List[PerInstrumentVote],
    instrument: str,
    weights: Optional[Dict[str, float]] = None,
) -> AggregatedInstrumentDecision:
    """
    Aggregate votes from multiple members for a single instrument.

    Mode-aware:
    - Uses CONFIDENCE_THRESHOLD and MIN_SIGNAL_STRENGTH from get_thresholds()
      to classify strong directional vs weak/neutral votes.
    - Tracks actual confidence separately from magnitude-weighted scores.
    - For LIVE: requires both high confidence AND reasonable magnitude
    - Confidence output reflects actual expert confidence, not magnitude

    Args:
        votes: List of PerInstrumentVote from all voting members
        instrument: The instrument to aggregate for
        weights: Optional member weights (default: equal weights)

    Returns:
        AggregatedInstrumentDecision
    """
    thresholds = get_thresholds()
    conf_threshold = float(thresholds["CONFIDENCE_THRESHOLD"])
    min_strength = float(thresholds["MIN_SIGNAL_STRENGTH"])

    weights = weights or {}
    inst = normalize_instrument(instrument)

    decision = AggregatedInstrumentDecision(instrument=inst)

    long_score = 0.0
    short_score = 0.0
    flat_score = 0.0
    total_weight = 0.0
    
    # Track actual confidence values separately for proper output
    long_conf_sum = 0.0
    long_conf_weight = 0.0
    short_conf_sum = 0.0
    short_conf_weight = 0.0

    for vote in votes:
        prop = vote.get_proposal(inst)
        if prop is None:
            continue

        act = prop.voting_action
        conf = _clamp_01(prop.confidence, default=0.0)
        mag = _clamp_01(prop.magnitude, default=0.0)
        
        # IMPORTANT: If magnitude is not set (0), use confidence as proxy
        # Many experts don't set signal_strength/magnitude explicitly
        if mag == 0.0 and conf > 0.0:
            mag = conf  # Use confidence as magnitude fallback
            
        base_weight = float(weights.get(vote.member, 1.0))

        # Every non-missing proposal counts as one "vote" for consensus ratio
        decision.vote_count += 1

        # No usable magnitude or confidence → treat as neutral vote
        if conf == 0.0 and mag == 0.0:
            decision.flat_votes += 1
            continue

        # Strong directional: requires confidence above threshold
        # (magnitude already falls back to confidence, so we just check conf)
        strong_directional = act.is_directional and conf >= conf_threshold
        
        # Moderate directional: has reasonable confidence (70% of threshold)
        moderate_directional = act.is_directional and conf >= conf_threshold * 0.7
        
        if strong_directional:
            # Full weight for strong signals
            w = base_weight * conf
            total_weight += w
            
            if act is VotingAction.LONG:
                decision.long_votes += 1
                long_score += w * mag
                long_conf_sum += conf * base_weight
                long_conf_weight += base_weight
            elif act is VotingAction.SHORT:
                decision.short_votes += 1
                short_score += w * mag
                short_conf_sum += conf * base_weight
                short_conf_weight += base_weight
                
        elif moderate_directional:
            # Reduced weight for moderate signals (70% weight)
            w = base_weight * conf * 0.7
            total_weight += w
            
            if act is VotingAction.LONG:
                decision.long_votes += 1
                long_score += w * max(mag, conf * 0.5)  # Use conf as mag proxy if needed
                long_conf_sum += conf * base_weight * 0.7
                long_conf_weight += base_weight * 0.7
            elif act is VotingAction.SHORT:
                decision.short_votes += 1
                short_score += w * max(mag, conf * 0.5)
                short_conf_sum += conf * base_weight * 0.7
                short_conf_weight += base_weight * 0.7
        else:
            # Weak/neutral: count as neutral
            decision.flat_votes += 1
            w = base_weight * conf * 0.25
            flat_score += w * 0.5
            total_weight += w

    if total_weight <= 0.0 or decision.vote_count == 0:
        # No usable votes, leave defaults (ABSTAIN / 0)
        return decision

    # Normalize scores
    long_score /= total_weight
    short_score /= total_weight
    flat_score /= total_weight

    # Determine winning action
    scores = {
        VotingAction.LONG: long_score,
        VotingAction.SHORT: short_score,
        VotingAction.HOLD: flat_score,
    }
    winning_action = max(scores, key=lambda k: scores[k])
    winning_score = scores[winning_action]

    # Calculate consensus (how much agreement, including neutral votes)
    max_votes = max(decision.long_votes, decision.short_votes, decision.flat_votes)
    if decision.vote_count > 0:
        decision.consensus_score = max_votes / decision.vote_count

    decision.action = winning_action.value
    decision.weighted_score = winning_score
    
    # Use actual average confidence for the winning direction
    # This ensures output confidence reflects expert confidence, not magnitude
    if winning_action is VotingAction.LONG and long_conf_weight > 0:
        decision.confidence = long_conf_sum / long_conf_weight
    elif winning_action is VotingAction.SHORT and short_conf_weight > 0:
        decision.confidence = short_conf_sum / short_conf_weight
    else:
        # Fall back to weighted score for HOLD or when no directional votes
        decision.confidence = winning_score

    return decision


def aggregate_all_instruments(
    votes: List[PerInstrumentVote],
    instruments: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, AggregatedInstrumentDecision]:
    """
    Aggregate votes for all instruments.

    Returns:
        Dict mapping instrument -> AggregatedInstrumentDecision
    """
    instruments = instruments or DEFAULT_INSTRUMENTS

    return {
        inst: aggregate_instrument_votes(votes, inst, weights)
        for inst in instruments
    }
