
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .constants import normalize_instrument


def _now_iso() -> str:
    return datetime.now().isoformat()


def _clamp_01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)

    if v != v:
        return float(default)

    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


DEFAULT_INSTRUMENTS: List[str] = ["XAUUSD"]


@dataclass
class AggregatedInstrumentDecision:

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
            "members": list(self.members),
        }


@dataclass
class InstrumentProposal:
    instrument: str
    action: str = "flat"
    confidence: float = 0.5
    magnitude: float = 0.5
    rationale: str = ""

    def __post_init__(self) -> None:
        self.instrument = normalize_instrument(self.instrument)
        self.action = str(self.action or "flat").lower().strip()
        self.confidence = _clamp_01(self.confidence, default=0.5)
        self.magnitude = _clamp_01(self.magnitude, default=self.confidence)
        self.rationale = str(self.rationale or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "action": self.action,
            "confidence": self.confidence,
            "magnitude": self.magnitude,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstrumentProposal":
        return cls(
            instrument=str(data.get("instrument", "UNKNOWN")),
            action=str(data.get("action", "flat")),
            confidence=float(data.get("confidence", 0.5) or 0.5),
            magnitude=float(data.get("magnitude", data.get("signal_strength", 0.5)) or 0.5),
            rationale=str(data.get("rationale", data.get("reason", "")) or ""),
        )


@dataclass
class PerInstrumentVote:

    member: str
    proposals: Dict[str, InstrumentProposal] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def set_proposal(self, proposal: InstrumentProposal) -> None:
        inst = normalize_instrument(proposal.instrument)
        proposal.instrument = inst
        self.proposals[inst] = proposal

    def get_proposal(self, instrument: str) -> Optional[InstrumentProposal]:
        return self.proposals.get(normalize_instrument(instrument))

    def get_action(self, instrument: str, default: str = "flat") -> str:
        prop = self.get_proposal(instrument)
        return prop.action if prop else default

    def get_confidence(self, instrument: str, default: float = 0.5) -> float:
        prop = self.get_proposal(instrument)
        return prop.confidence if prop else default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "member": self.member,
            "proposals": {inst: prop.to_dict() for inst, prop in self.proposals.items()},
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerInstrumentVote":
        member = str(data.get("member", "unknown"))
        piv = cls(member=member, timestamp=str(data.get("timestamp") or _now_iso()))

        proposals_data = data.get("proposals", {})
        if isinstance(proposals_data, dict):
            for inst, prop_data in proposals_data.items():
                if isinstance(prop_data, InstrumentProposal):
                    piv.set_proposal(prop_data)
                elif isinstance(prop_data, dict):
                    prop_data = dict(prop_data)
                    prop_data.setdefault("instrument", inst)
                    piv.set_proposal(InstrumentProposal.from_dict(prop_data))

        return piv


def aggregate_all_instruments(
    votes: List[PerInstrumentVote],
    instruments: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, AggregatedInstrumentDecision]:
    instruments = instruments or DEFAULT_INSTRUMENTS
    weights = weights or {}

    result: Dict[str, AggregatedInstrumentDecision] = {}

    for inst in instruments:
        inst_norm = normalize_instrument(inst)
        long_count = 0
        short_count = 0
        flat_count = 0
        confidences: List[float] = []
        members: List[str] = []

        weighted_sum = 0.0
        total_weight = 0.0

        for vote in votes:
            prop = vote.get_proposal(inst_norm)
            if not prop:
                continue

            action = str(prop.action or "flat").lower().strip()
            member_weight = float(weights.get(vote.member, 1.0) or 1.0)

            if action in ("long", "buy", "bullish"):
                long_count += 1
                weighted_sum += member_weight * float(prop.confidence)
            elif action in ("short", "sell", "bearish"):
                short_count += 1
                weighted_sum -= member_weight * float(prop.confidence)
            else:
                flat_count += 1

            total_weight += member_weight
            confidences.append(float(prop.confidence))
            members.append(vote.member)

        vote_count = long_count + short_count + flat_count

        if weighted_sum > 0.1:
            dominant_action = "long"
        elif weighted_sum < -0.1:
            dominant_action = "short"
        else:
            dominant_action = "flat"

        if vote_count > 0:
            max_votes = max(long_count, short_count, flat_count)
            consensus_score = max_votes / vote_count
        else:
            consensus_score = 0.0

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        weighted_score = weighted_sum / max(total_weight, 1.0)

        result[inst_norm] = AggregatedInstrumentDecision(
            action=dominant_action,
            confidence=float(avg_confidence),
            consensus_score=float(consensus_score),
            vote_count=int(vote_count),
            long_votes=int(long_count),
            short_votes=int(short_count),
            flat_votes=int(flat_count),
            weighted_score=float(weighted_score),
            members=members,
        )

    return result


def extract_instrument_data(data: Dict[str, Any], instrument: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}


    if "value" in data and isinstance(data.get("value"), dict) and "timestamp" in data:
        data = data["value"]

    inst_norm = normalize_instrument(instrument)
    if not inst_norm:
        return data


    for key in (instrument, inst_norm, str(instrument).upper(), str(instrument).lower()):
        if key in data:
            return data[key] if isinstance(data[key], dict) else data


    for k, v in data.items():
        if isinstance(k, str) and normalize_instrument(k) == inst_norm:
            return v if isinstance(v, dict) else data

    return data
