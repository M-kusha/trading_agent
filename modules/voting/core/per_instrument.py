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
from enum import Enum

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# Default instruments - should match system_config.yaml
DEFAULT_INSTRUMENTS: List[str] = ["EURUSD", "XAUUSD"]

# Alternate key formats (for bus key normalization)
INSTRUMENT_ALIASES: Dict[str, str] = {
    "EUR_USD": "EURUSD",
    "XAU_USD": "XAUUSD",
    "EURUSD": "EURUSD",
    "XAUUSD": "XAUUSD",
}


def normalize_instrument(inst: str) -> str:
    """Normalize instrument name to canonical format."""
    return INSTRUMENT_ALIASES.get(inst, inst)


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InstrumentProposal:
    """
    A voting proposal for a single instrument.
    
    Attributes:
        instrument: The instrument symbol (e.g., 'EURUSD')
        action: The proposed action ('long', 'short', 'flat', 'abstain')
        confidence: Confidence in this proposal (0.0 to 1.0)
        magnitude: Signal strength/intensity (0.0 to 1.0)
        horizon: Time horizon ('scalp', 'intraday', 'swing')
        rationale: Human-readable explanation
        meta: Additional metadata
    """
    instrument: str
    action: str = "flat"
    confidence: float = 0.5
    magnitude: float = 0.0
    horizon: str = "intraday"
    rationale: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for bus publishing."""
        return {
            'instrument': self.instrument,
            'action': self.action,
            'confidence': round(self.confidence, 4),
            'magnitude': round(self.magnitude, 4),
            'horizon': self.horizon,
            'rationale': self.rationale,
            'meta': self.meta,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstrumentProposal":
        """Create from dictionary."""
        return cls(
            instrument=str(data.get('instrument', 'UNKNOWN')),
            action=str(data.get('action', 'flat')),
            confidence=float(data.get('confidence', 0.5)),
            magnitude=float(data.get('magnitude', 0.0)),
            horizon=str(data.get('horizon', 'intraday')),
            rationale=str(data.get('rationale', '')),
            meta=dict(data.get('meta', {})),
        )
    
    @property
    def is_directional(self) -> bool:
        """Returns True if this is a directional signal."""
        return self.action in ('long', 'short')


@dataclass
class PerInstrumentVote:
    """
    A complete per-instrument vote from a voting member.
    
    Contains proposals for ALL instruments the member analyzed.
    """
    member: str
    proposals: Dict[str, InstrumentProposal] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Legacy compatibility: also track a "global" fallback for old consumers
    global_action: str = "flat"
    global_confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for bus publishing."""
        return {
            'member': self.member,
            'proposals': {k: v.to_dict() for k, v in self.proposals.items()},
            'timestamp': self.timestamp,
            # Legacy compatibility fields
            'action': self.global_action,
            'confidence': self.global_confidence,
            'proposal': {
                'direction': self.global_action,
                'magnitude': max((p.magnitude for p in self.proposals.values()), default=0.0),
                'horizon': 'intraday',
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerInstrumentVote":
        """Create from dictionary."""
        proposals = {}
        if 'proposals' in data and isinstance(data['proposals'], dict):
            for inst, prop_data in data['proposals'].items():
                proposals[inst] = InstrumentProposal.from_dict(prop_data)
        
        return cls(
            member=str(data.get('member', 'Unknown')),
            proposals=proposals,
            timestamp=str(data.get('timestamp', datetime.now().isoformat())),
            global_action=str(data.get('action', 'flat')),
            global_confidence=float(data.get('confidence', 0.5)),
        )
    
    def get_proposal(self, instrument: str) -> Optional[InstrumentProposal]:
        """Get proposal for a specific instrument."""
        inst = normalize_instrument(instrument)
        return self.proposals.get(inst)
    
    def set_proposal(self, proposal: InstrumentProposal) -> None:
        """Set proposal for an instrument."""
        inst = normalize_instrument(proposal.instrument)
        self.proposals[inst] = proposal
        self._update_global()
    
    def _update_global(self) -> None:
        """Update global fallback from per-instrument proposals."""
        if not self.proposals:
            return
        
        # Use the proposal with highest confidence as global
        best = max(self.proposals.values(), key=lambda p: p.confidence)
        self.global_action = best.action
        self.global_confidence = best.confidence


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
            action=str(prop_data.get('action', 'flat')),
            confidence=float(prop_data.get('confidence', 0.5)),
            magnitude=float(prop_data.get('magnitude', 0.0)),
            horizon=str(prop_data.get('horizon', 'intraday')),
            rationale=str(prop_data.get('rationale', '')),
            meta=dict(prop_data.get('meta', {})),
        )
        vote.set_proposal(prop)
    
    return vote


def create_flat_vote(member: str, instruments: Optional[List[str]] = None) -> PerInstrumentVote:
    """Create a flat (no-action) vote for all instruments."""
    instruments = instruments or DEFAULT_INSTRUMENTS
    vote = PerInstrumentVote(member=member)
    
    for inst in instruments:
        vote.set_proposal(InstrumentProposal(
            instrument=inst,
            action="flat",
            confidence=0.3,
            magnitude=0.0,
            rationale="No signal",
        ))
    
    return vote


def extract_instrument_data(
    market_data: Dict[str, Any],
    instrument: str,
) -> Dict[str, Any]:
    """
    Extract market data for a specific instrument.
    
    Handles both formats:
    - market_data[instrument] = {...}  (direct)
    - market_data['EURUSD'] or market_data['EUR_USD']  (alias)
    """
    inst = normalize_instrument(instrument)
    
    # Try direct key
    if inst in market_data:
        return market_data[inst]
    
    # Try alternate formats
    for alias, canonical in INSTRUMENT_ALIASES.items():
        if canonical == inst and alias in market_data:
            return market_data[alias]
    
    return {}


def analyze_instrument_trend(
    price_data: Dict[str, Any],
    indicators: Dict[str, Any],
) -> Tuple[str, float]:
    """
    Analyze trend direction for a single instrument.
    
    Returns:
        Tuple of (direction: 'long'|'short'|'flat', strength: 0.0-1.0)
    """
    try:
        # Get price movement
        open_price = float(price_data.get('open', 0) or 0)
        close_price = float(price_data.get('close', price_data.get('last', 0)) or 0)
        high = float(price_data.get('high', 0) or 0)
        low = float(price_data.get('low', 0) or 0)
        
        if open_price <= 0 or close_price <= 0:
            return 'flat', 0.0
        
        # Candle direction
        candle_change = (close_price - open_price) / open_price
        
        # RSI
        rsi = float(indicators.get('rsi', 50) or 50)
        
        # Momentum
        momentum = float(indicators.get('momentum', indicators.get('roc', 0)) or 0)
        
        # Combine signals
        signals = []
        
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
            signals.append(np.clip(momentum / 2.0, -1.0, 1.0))
        
        if not signals:
            return 'flat', 0.0
        
        avg_signal = np.mean(signals)
        strength = abs(avg_signal)
        
        if strength < 0.2:
            return 'flat', float(strength)
        elif avg_signal > 0:
            return 'long', float(strength)
        else:
            return 'short', float(strength)
    
    except Exception:
        return 'flat', 0.0


# ═══════════════════════════════════════════════════════════════════
# Aggregation Helpers (for CommitteeCoordinator)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AggregatedInstrumentDecision:
    """Aggregated committee decision for a single instrument."""
    instrument: str
    action: str = "flat"
    confidence: float = 0.0
    consensus_score: float = 0.0
    vote_count: int = 0
    long_votes: int = 0
    short_votes: int = 0
    flat_votes: int = 0
    weighted_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_instrument_votes(
    votes: List[PerInstrumentVote],
    instrument: str,
    weights: Optional[Dict[str, float]] = None,
) -> AggregatedInstrumentDecision:
    """
    Aggregate votes from multiple members for a single instrument.
    
    Args:
        votes: List of PerInstrumentVote from all voting members
        instrument: The instrument to aggregate for
        weights: Optional member weights (default: equal weights)
    
    Returns:
        AggregatedInstrumentDecision
    """
    weights = weights or {}
    inst = normalize_instrument(instrument)
    
    decision = AggregatedInstrumentDecision(instrument=inst)
    
    long_score = 0.0
    short_score = 0.0
    flat_score = 0.0
    total_weight = 0.0
    
    for vote in votes:
        prop = vote.get_proposal(inst)
        if prop is None:
            continue
        
        weight = weights.get(vote.member, 1.0) * prop.confidence
        decision.vote_count += 1
        
        if prop.action == 'long':
            decision.long_votes += 1
            long_score += weight * prop.magnitude
        elif prop.action == 'short':
            decision.short_votes += 1
            short_score += weight * prop.magnitude
        else:
            decision.flat_votes += 1
            flat_score += weight * 0.5
        
        total_weight += weight
    
    if total_weight <= 0:
        return decision
    
    # Normalize scores
    long_score /= total_weight
    short_score /= total_weight
    flat_score /= total_weight
    
    # Determine winning action
    scores = {'long': long_score, 'short': short_score, 'flat': flat_score}
    winning_action = max(scores, key=lambda k: scores[k])
    winning_score = scores[winning_action]
    
    # Calculate consensus (how much agreement)
    if decision.vote_count > 0:
        max_votes = max(decision.long_votes, decision.short_votes, decision.flat_votes)
        decision.consensus_score = max_votes / decision.vote_count
    
    decision.action = winning_action
    decision.confidence = winning_score
    decision.weighted_score = winning_score
    
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
