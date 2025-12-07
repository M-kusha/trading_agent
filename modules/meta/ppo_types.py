#!/usr/bin/env python3
"""
PPO Types - Structured Decision Types for Multi-Instrument Trading
===================================================================

This module defines the core data types for the PPO arbiter system:
- InstrumentDecision: Per-instrument trading decision
- ArbiterMultiDecision: Container for all instrument decisions
- MemoryGateInfo: Memory/risk gate information
- RiskInfo: Risk assessment information
- InstrumentStats / InstrumentStatsTracker: Per-instrument stats

Version: 3.0.0 (Multi-instrument architecture)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# MEMORY / RISK GATE TYPES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MemoryGateInfo:
    """
    Normalized memory gate information from UnifiedMemory.

    This replaces the raw multiplier approach with structured data
    that can be used both as observation features and for gating logic.
    """
    risk_multiplier: float = 1.0      # 0..1, how much to scale position
    veto: bool = False                # Hard veto - block all trading
    risk_score: float = 0.0           # 0..1, overall memory risk
    danger_similarity: float = 0.0    # 0..1, similarity to past losses
    loss_prob: float = 0.0            # 0..1, probability of loss
    reasons: List[str] = field(default_factory=list)

    @classmethod
    def from_bus_data(cls, memory_gate: Any, danger_zones: Any) -> "MemoryGateInfo":
        """Construct from SmartInfoBus data."""
        # Helper to convert reason items to strings (reasons can be dicts or strings)
        def _reason_to_str(r: Any) -> str:
            if isinstance(r, str):
                return r
            if isinstance(r, dict):
                # Extract meaningful text from dict reason
                return (
                    r.get("message")
                    or r.get("reason")
                    or r.get("label")
                    or f"{r.get('type', 'unknown')}: {r.get('similarity', r.get('consecutive_losses', ''))}"
                )
            return str(r)
        
        # Parse memory_gate
        if isinstance(memory_gate, dict):
            risk_mult = float(memory_gate.get("risk_multiplier", 1.0))
            veto = bool(memory_gate.get("veto", False))
            risk_score = float(memory_gate.get("risk_score", 0.0))
            danger_sim = float(memory_gate.get("danger_similarity", 0.0))
            loss_prob = float(memory_gate.get("loss_prob", 0.0))
            raw_reasons = memory_gate.get("reasons", [])
            # Convert any dict reasons to strings
            reasons = [_reason_to_str(r) for r in raw_reasons] if raw_reasons else []
        elif memory_gate is not None:
            try:
                risk_mult = float(memory_gate)
            except (TypeError, ValueError):
                risk_mult = 1.0
            veto = False
            risk_score = max(0.0, min(1.0, 1.0 - risk_mult))
            danger_sim = 0.0
            loss_prob = 0.0
            reasons = []
        else:
            risk_mult = 1.0
            veto = False
            risk_score = 0.0
            danger_sim = 0.0
            loss_prob = 0.0
            reasons = []

        # Parse danger_zones for additional info
        if isinstance(danger_zones, dict):
            zone_count = int(danger_zones.get("zone_count", 0))
            if zone_count > 0 and "DANGER_ZONE" not in " ".join(reasons):
                reasons.append(f"DANGER_ZONE_COUNT={zone_count}")
            if "max_similarity" in danger_zones:
                try:
                    dz_sim = float(danger_zones["max_similarity"])
                except (TypeError, ValueError):
                    dz_sim = 0.0
                danger_sim = max(danger_sim, dz_sim)
        elif isinstance(danger_zones, list) and len(danger_zones) > 0:
            reasons.append(f"DANGER_ZONE_COUNT={len(danger_zones)}")

        return cls(
            risk_multiplier=max(0.0, min(1.0, risk_mult)),
            veto=veto,
            risk_score=max(0.0, min(1.0, risk_score)),
            danger_similarity=max(0.0, min(1.0, danger_sim)),
            loss_prob=max(0.0, min(1.0, loss_prob)),
            reasons=reasons,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InstrumentMemoryInfo(MemoryGateInfo):
    """Per-instrument memory gate information."""
    instrument: str = ""


@dataclass
class RiskInfo:
    """
    Normalized risk information from DynamicRiskController / PortfolioRiskSystem.

    Provides both features (for observation) and gates (for decision logic).
    
    v4.2.0: Enhanced to consume full DynamicRiskController intelligence:
    - risk_scale (0.1-1.5 dynamic multiplier)
    - risk_level (NORMAL/ELEVATED/HIGH/CRITICAL)
    - risk_factors (drawdown/volatility/correlation factors)
    - emergency_mode flag
    """
    portfolio_risk: float = 0.0       # 0..1, normalized portfolio risk
    instrument_risk: float = 0.0      # 0..1, instrument-specific risk
    max_dd: float = 0.0               # Current max drawdown (0..1)
    margin_usage: float = 0.0         # 0..1, margin utilization
    hard_block: bool = False          # If True, cannot trade at all
    hard_cap: float = 1.0             # Max position size (0..1)
    reasons: List[str] = field(default_factory=list)
    
    # v4.2.0: DynamicRiskController integration
    risk_scale: float = 1.0           # 0.1-1.5 from DynamicRiskController
    risk_level: str = "NORMAL"        # NORMAL/ELEVATED/HIGH/CRITICAL
    emergency_mode: bool = False      # True if DRC in emergency mode
    risk_factors: Dict[str, float] = field(default_factory=dict)  # drawdown/volatility/correlation factors

    @classmethod
    def from_bus_data(
        cls,
        risk_data: Optional[Dict[str, Any]] = None,
        portfolio_risk: Optional[Dict[str, Any]] = None,
        risk_scaling: Optional[Dict[str, Any]] = None,
        risk_assessment: Optional[Dict[str, Any]] = None,
        instrument: str = "",
    ) -> "RiskInfo":
        """
        Construct from SmartInfoBus data.
        
        v4.2.0: Enhanced to consume:
        - risk_data: Basic risk metrics
        - portfolio_risk: PortfolioRiskSystem output
        - risk_scaling: DynamicRiskController scaling output (current_risk_scale, etc)
        - risk_assessment: DynamicRiskController assessment (risk_level, emergency_active)
        """
        risk_data = risk_data or {}
        portfolio_risk = portfolio_risk or {}
        risk_scaling = risk_scaling or {}
        risk_assessment = risk_assessment or {}

        # Parse portfolio risk
        if isinstance(portfolio_risk, dict):
            port_risk = float(
                portfolio_risk.get("total_risk", portfolio_risk.get("risk_score", 0.0))
            )
            margin = float(portfolio_risk.get("margin_usage", 0.0))
        else:
            port_risk = 0.0
            margin = 0.0

        # Parse risk_data
        max_dd = float(risk_data.get("max_drawdown", risk_data.get("drawdown", 0.0)))
        hard_block = bool(risk_data.get("hard_block", risk_data.get("trading_blocked", False)))

        # Cap: prefer explicit numeric caps, fall back to 1.0 if not provided
        raw_cap = risk_data.get("position_cap", risk_data.get("max_position_size", 1.0))
        try:
            hard_cap = float(raw_cap)
        except (TypeError, ValueError):
            hard_cap = 1.0

        # Instrument-specific risk (if available)
        inst_risk = 0.0
        if instrument and isinstance(risk_data.get("instruments"), dict):
            inst_data = risk_data["instruments"].get(instrument, {})
            if isinstance(inst_data, dict):
                inst_risk = float(inst_data.get("risk", 0.0))

        # ═══════════════════════════════════════════════════════════════════
        # v4.2.0: DynamicRiskController integration
        # ═══════════════════════════════════════════════════════════════════
        
        # Extract risk_scale from risk_scaling
        risk_scale = 1.0
        if isinstance(risk_scaling, dict):
            rs = risk_scaling.get("current_risk_scale")
            if isinstance(rs, (int, float)):
                risk_scale = float(max(0.1, min(1.5, rs)))
        
        # Extract risk_level from risk_assessment
        risk_level = "NORMAL"
        emergency_mode = False
        if isinstance(risk_assessment, dict):
            rl = risk_assessment.get("risk_level")
            if isinstance(rl, str) and rl:
                risk_level = rl.upper()
            emergency_mode = bool(risk_assessment.get("emergency_active", False))
            
            # Also extract risk_scale from assessment if not in scaling
            if risk_scale == 1.0:
                rs = risk_assessment.get("risk_scale")
                if isinstance(rs, (int, float)):
                    risk_scale = float(max(0.1, min(1.5, rs)))
        
        # Extract risk_factors if available
        risk_factors: Dict[str, float] = {}
        if isinstance(risk_data, dict):
            rf = risk_data.get("risk_factors")
            if isinstance(rf, dict):
                for k, v in rf.items():
                    try:
                        risk_factors[k] = float(v)
                    except (TypeError, ValueError):
                        pass
        
        # ═══════════════════════════════════════════════════════════════════
        # Apply risk_level to hard_block and hard_cap
        # ═══════════════════════════════════════════════════════════════════
        
        # CRITICAL risk level = hard block
        if risk_level == "CRITICAL" or emergency_mode:
            hard_block = True
            hard_cap = 0.0
        # HIGH risk level = reduced cap
        elif risk_level == "HIGH":
            hard_cap = min(hard_cap, 0.3)
        # ELEVATED risk level = moderately reduced cap
        elif risk_level == "ELEVATED":
            hard_cap = min(hard_cap, 0.6)

        reasons: List[str] = []
        if hard_block:
            reasons.append("HARD_BLOCK_ACTIVE")
        if emergency_mode:
            reasons.append("EMERGENCY_MODE")
        if risk_level != "NORMAL":
            reasons.append(f"RISK_LEVEL={risk_level}")
        if max_dd > 0.05:
            reasons.append(f"DRAWDOWN={max_dd:.1%}")
        if margin > 0.8:
            reasons.append(f"HIGH_MARGIN={margin:.1%}")
        if risk_scale < 0.5:
            reasons.append(f"LOW_RISK_SCALE={risk_scale:.2f}")

        return cls(
            portfolio_risk=max(0.0, min(1.0, port_risk)),
            instrument_risk=max(0.0, min(1.0, inst_risk)),
            max_dd=max_dd,
            margin_usage=max(0.0, min(1.0, margin)),
            hard_block=hard_block,
            hard_cap=max(0.0, min(1.0, hard_cap)),
            reasons=reasons,
            risk_scale=risk_scale,
            risk_level=risk_level,
            emergency_mode=emergency_mode,
            risk_factors=risk_factors,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InstrumentRiskInfo(RiskInfo):
    """Per-instrument risk information."""
    instrument: str = ""


# ═══════════════════════════════════════════════════════════════════
# DECISION TYPES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InstrumentDecision:
    """
    A structured trading decision for a single instrument.

    Core output of ArbiterLogic for each instrument:
    - What to do (direction, size)
    - Why (reasoning, meta contributors)
    - Whether it's allowed (gate_passed)
    """
    instrument: str
    direction: str = "flat"           # "long" | "short" | "flat"
    confidence: float = 0.0           # 0..1
    position_size: float = 0.0        # 0..1 (risk-normalized)
    trust_score: float = 0.0          # PPO trust vs committee

    # Committee/expert context
    committee_action: str = "hold"
    committee_confidence: float = 0.0
    expert_consensus: str = "flat"
    expert_confidence: float = 0.0

    # Market context
    regime: str = "unknown"
    regime_strength: float = 0.0

    # PPO internals
    value_estimate: float = 0.0
    raw_action: List[float] = field(default_factory=list)

    # Gating
    gate_passed: bool = False
    gate_reasons: List[str] = field(default_factory=list)

    # Explanation
    reasoning: str = ""

    # Rich metadata for debugging/dashboard
    meta: Dict[str, Any] = field(default_factory=dict)

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy ppo_final_decision format for backward compatibility."""
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "trust_score": self.trust_score,
            "committee_action": self.committee_action,
            "committee_confidence": self.committee_confidence,
            "expert_consensus": self.expert_consensus,
            "expert_confidence": self.expert_confidence,
            "regime": self.regime,
            "value_estimate": self.value_estimate,
            "gate_passed": self.gate_passed,
            "position_size": self.position_size,
            "instrument": self.instrument,
        }


@dataclass
class ArbiterMultiDecision:
    """
    Container for multi-instrument arbiter decisions.

    This is the primary output of ArbiterLogic.make_multi_instrument_decision().
    """
    instruments: Dict[str, InstrumentDecision] = field(default_factory=dict)
    global_meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "PPOAgent"

    def get_primary_decision(self, primary_instrument: str = "XAUUSD") -> InstrumentDecision:
        """Get the primary instrument's decision (for legacy compatibility)."""
        if primary_instrument in self.instruments:
            return self.instruments[primary_instrument]
        if self.instruments:
            # Fallback to first instrument
            return next(iter(self.instruments.values()))
        # Return empty decision if nothing exists
        return InstrumentDecision(instrument=primary_instrument)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruments": {k: v.to_dict() for k, v in self.instruments.items()},
            "global_meta": self.global_meta,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    def to_legacy_format(self, primary_instrument: str = "XAUUSD") -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility."""
        primary = self.get_primary_decision(primary_instrument)
        return {
            "ppo_final_decision": primary.to_legacy_format(),
            "ppo_gate_passed": primary.gate_passed,
            "ppo_position_size": primary.position_size,
            "arbiter_decision": self.to_dict(),
        }


# ═══════════════════════════════════════════════════════════════════
# STATISTICS TRACKING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InstrumentStats:
    """
    Per-instrument statistics tracker for learning and diagnostics.

    Tracks decision patterns, win/loss rates, and recent history.
    """
    instrument: str = ""
    trades: int = 0
    longs: int = 0
    shorts: int = 0
    flats: int = 0
    wins: int = 0
    losses: int = 0
    avg_confidence: float = 0.0
    avg_position_size: float = 0.0
    total_pnl: float = 0.0

    # Running aggregates
    _confidence_sum: float = field(default=0.0, repr=False)
    _size_sum: float = field(default=0.0, repr=False)
    _decision_count: int = field(default=0, repr=False)

    # Recent decisions (for pattern analysis)
    last_decisions: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=50)
    )

    def record_decision(self, decision: InstrumentDecision) -> None:
        """Record a decision for statistics."""
        self._decision_count += 1
        self._confidence_sum += decision.confidence
        self._size_sum += decision.position_size

        self.avg_confidence = self._confidence_sum / max(self._decision_count, 1)
        self.avg_position_size = self._size_sum / max(self._decision_count, 1)

        if decision.direction == "long":
            self.longs += 1
        elif decision.direction == "short":
            self.shorts += 1
        else:
            self.flats += 1

        if decision.gate_passed:
            self.trades += 1

        # Store summary for pattern analysis
        self.last_decisions.append(
            {
                "direction": decision.direction,
                "confidence": decision.confidence,
                "position_size": decision.position_size,
                "gate_passed": decision.gate_passed,
                "timestamp": decision.timestamp,
            }
        )

    def record_trade_result(self, pnl: float) -> None:
        """Record trade outcome."""
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def long_bias(self) -> float:
        """Calculate long bias (positive = more longs than shorts)."""
        total = self.longs + self.shorts
        if total == 0:
            return 0.0
        return (self.longs - self.shorts) / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "trades": self.trades,
            "longs": self.longs,
            "shorts": self.shorts,
            "flats": self.flats,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_position_size": round(self.avg_position_size, 4),
            "total_pnl": round(self.total_pnl, 2),
            "long_bias": round(self.long_bias, 4),
        }


@dataclass
class InstrumentStatsTracker:
    """
    Manages statistics for all instruments.
    """
    stats: Dict[str, InstrumentStats] = field(default_factory=dict)

    def get_or_create(self, instrument: str) -> InstrumentStats:
        """Get or create stats for an instrument."""
        if instrument not in self.stats:
            self.stats[instrument] = InstrumentStats(instrument=instrument)
        return self.stats[instrument]

    def record_decision(self, decision: InstrumentDecision) -> None:
        """Record a single decision."""
        stats = self.get_or_create(decision.instrument)
        stats.record_decision(decision)

    def record_multi_decision(self, multi_decision: ArbiterMultiDecision) -> None:
        """Record all decisions from a multi-decision."""
        for decision in multi_decision.instruments.values():
            self.record_decision(decision)

    def record_trade_result(self, instrument: str, pnl: float) -> None:
        """Record trade outcome."""
        stats = self.get_or_create(instrument)
        stats.record_trade_result(pnl)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self.stats.items()}


# ═══════════════════════════════════════════════════════════════════
# GATING PIPELINE RESULT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GatingResult:
    """
    Result of the 3-stage gating pipeline.

    Stages:
    1. Hard veto (risk.hard_block or memory.veto)
    2. Soft scaling (risk_multiplier, caps)
    3. PPO decision (trust_score determines final action)
    """
    gate_passed: bool = True
    position_size_cap: float = 1.0
    confidence_multiplier: float = 1.0
    reasons: List[str] = field(default_factory=list)

    # Which stage blocked/modified
    hard_veto_triggered: bool = False
    soft_scaling_applied: bool = False
    ppo_override: bool = False

    @classmethod
    def apply_gates(
        cls,
        memory: MemoryGateInfo,
        risk: RiskInfo,
        trust_score: float,
    ) -> "GatingResult":
        """
        Apply the 3-stage gating pipeline.

        Stage 1: Hard veto (risk.hard_block, emergency_mode, or memory.veto)
        Stage 2: Soft scaling (risk_multiplier, risk_scale, caps)
        Stage 3: PPO override check
        
        v4.2.0: Enhanced with DynamicRiskController's risk_scale and risk_level.
        """
        result = cls()

        # Stage 1: Hard veto
        if risk.hard_block:
            result.gate_passed = False
            result.hard_veto_triggered = True
            result.reasons.append("RISK_HARD_BLOCK")
        
        # v4.2.0: Emergency mode from DynamicRiskController
        if risk.emergency_mode:
            result.gate_passed = False
            result.hard_veto_triggered = True
            result.reasons.append("EMERGENCY_MODE")

        if memory.veto:
            result.gate_passed = False
            result.hard_veto_triggered = True
            result.reasons.append("MEMORY_VETO")

        if result.hard_veto_triggered:
            result.position_size_cap = 0.0
            result.confidence_multiplier = 0.0
            return result

        # Stage 2: Soft scaling

        # Apply memory risk multiplier
        if memory.risk_multiplier < 1.0:
            result.confidence_multiplier *= max(0.0, min(1.0, memory.risk_multiplier))
            result.soft_scaling_applied = True
            result.reasons.append(f"MEMORY_SCALE={memory.risk_multiplier:.2f}")

        # Apply risk hard cap
        if risk.hard_cap < 1.0:
            cap = max(0.0, min(1.0, risk.hard_cap))
            result.position_size_cap = min(result.position_size_cap, cap)
            result.soft_scaling_applied = True
            result.reasons.append(f"RISK_CAP={cap:.2f}")

        # ═══════════════════════════════════════════════════════════════════
        # v4.2.0: DynamicRiskController risk_scale integration
        # Apply the dynamic risk scale (0.1-1.5) from DynamicRiskController
        # This is the key intelligence from all risk modules aggregated
        # ═══════════════════════════════════════════════════════════════════
        if risk.risk_scale < 1.0:
            # Low risk_scale = reduce position size and confidence
            result.position_size_cap *= risk.risk_scale
            result.confidence_multiplier *= (0.5 + 0.5 * risk.risk_scale)  # 0.55-1.0
            result.soft_scaling_applied = True
            result.reasons.append(f"DRC_SCALE={risk.risk_scale:.2f}")
        elif risk.risk_scale > 1.0:
            # High risk_scale (recovery mode) = can boost slightly
            boost = min(1.2, risk.risk_scale)  # Cap at 1.2x
            result.position_size_cap *= boost
            result.soft_scaling_applied = True
            result.reasons.append(f"DRC_BOOST={boost:.2f}")
        
        # v4.2.0: Risk level based scaling
        if risk.risk_level == "HIGH":
            result.position_size_cap = min(result.position_size_cap, 0.3)
            result.confidence_multiplier *= 0.6
            result.soft_scaling_applied = True
            result.reasons.append("RISK_LEVEL_HIGH")
        elif risk.risk_level == "ELEVATED":
            result.position_size_cap = min(result.position_size_cap, 0.6)
            result.confidence_multiplier *= 0.8
            result.soft_scaling_applied = True
            result.reasons.append("RISK_LEVEL_ELEVATED")

        # Danger zone penalty
        if memory.danger_similarity > 0.5:
            penalty = max(0.0, 1.0 - (memory.danger_similarity - 0.5))
            result.confidence_multiplier *= penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"DANGER_SIM={memory.danger_similarity:.2f}")

        # High portfolio risk penalty
        if risk.portfolio_risk > 0.7:
            penalty = 1.0 - (risk.portfolio_risk - 0.7) * 2.0
            penalty = max(0.3, min(1.0, penalty))
            result.confidence_multiplier *= penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"HIGH_RISK={risk.portfolio_risk:.2f}")

        # Stage 3: PPO override check
        # If trust_score is very negative, PPO is overriding committee
        if trust_score < -0.5:
            result.ppo_override = True
            result.gate_passed = False
            result.reasons.append(f"PPO_OVERRIDE={trust_score:.2f}")

        return result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

DEFAULT_INSTRUMENTS: List[str] = ["XAUUSD", "EURUSD"]
PRIMARY_INSTRUMENT: str = "XAUUSD"
