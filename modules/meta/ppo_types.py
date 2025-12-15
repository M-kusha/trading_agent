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
- GatingResult: Result of combined memory/risk/PPO gating

Version: 3.2.0 (Centralized thresholds, PPO-safe gating, richer diagnostics)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from modules.meta.numeric_utils import _safe_float, _clip


# ═══════════════════════════════════════════════════════════════════
# GATING / PPO THRESHOLDS (CENTRALIZED)
# ═══════════════════════════════════════════════════════════════════

# PPO signal: how strong the direction_score must be to consider a trade.
# MUST be aligned with ArbiterLogic._score_to_direction() (long/short at ~0.3).
PPO_SIGNAL_UNCERTAIN_THRESHOLD: float = 0.30  # |score| < this ⇒ "uncertain"

# Memory hard veto thresholds
MEMORY_HARD_VETO_RISK_SCORE: float = 0.98
MEMORY_HARD_VETO_LOSS_PROB: float = 0.98

# Memory soft scaling thresholds
MEMORY_SOFT_RISK_SCORE_START: float = 0.70  # above → soft penalty
MEMORY_SOFT_LOSS_PROB_START: float = 0.50   # above → soft penalty

# Portfolio risk soft scaling
PORTFOLIO_RISK_PENALTY_START: float = 0.70  # above → confidence penalty

# Instrument risk scaling (per-instrument)
INSTRUMENT_RISK_ELEVATED: float = 0.60
INSTRUMENT_RISK_HIGH: float = 0.80


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

    def __post_init__(self) -> None:
        # Clamp to valid ranges
        self.risk_multiplier = _clip(self.risk_multiplier, 0.0, 1.0)
        self.risk_score = _clip(self.risk_score, 0.0, 1.0)
        self.danger_similarity = _clip(self.danger_similarity, 0.0, 1.0)
        self.loss_prob = _clip(self.loss_prob, 0.0, 1.0)

    @classmethod
    def from_bus_data(cls, memory_gate: Any, danger_zones: Any) -> "MemoryGateInfo":
        """Construct from SmartInfoBus data."""
        # Helper to convert reason items to strings (reasons can be dicts or strings)
        def _reason_to_str(r: Any) -> str:
            if isinstance(r, str):
                return r
            if isinstance(r, dict):
                return (
                    r.get("message")
                    or r.get("reason")
                    or r.get("label")
                    or f"{r.get('type', 'unknown')}: {r.get('similarity', r.get('consecutive_losses', ''))}"
                )
            return str(r)

        # Parse memory_gate
        if isinstance(memory_gate, dict):
            risk_mult = _safe_float(memory_gate.get("risk_multiplier", 1.0), 1.0)
            veto = bool(memory_gate.get("veto", False))
            risk_score = _safe_float(memory_gate.get("risk_score", 0.0), 0.0)
            danger_sim = _safe_float(memory_gate.get("danger_similarity", 0.0), 0.0)
            loss_prob = _safe_float(memory_gate.get("loss_prob", 0.0), 0.0)
            raw_reasons = memory_gate.get("reasons", []) or []
            reasons = [_reason_to_str(r) for r in raw_reasons]
        elif memory_gate is not None:
            # Scalar multiplier style
            risk_mult = _safe_float(memory_gate, 1.0)
            veto = False
            # Convert multiplier into a crude "risk_score" if no struct provided
            risk_score = _clip(1.0 - risk_mult, 0.0, 1.0)
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
            zone_count = int(danger_zones.get("zone_count", 0) or 0)
            if zone_count > 0:
                reasons.append(f"DANGER_ZONE_COUNT={zone_count}")
            # Explicit max_similarity if provided
            if "max_similarity" in danger_zones:
                dz_sim = _safe_float(danger_zones.get("max_similarity", 0.0), 0.0)
                danger_sim = max(danger_sim, dz_sim)
            # Zones list with similarities
            zones = danger_zones.get("zones")
            if isinstance(zones, list):
                max_zone_sim = 0.0
                for z in zones:
                    if isinstance(z, dict) and "similarity" in z:
                        max_zone_sim = max(
                            max_zone_sim,
                            _safe_float(z.get("similarity"), 0.0),
                        )
                danger_sim = max(danger_sim, max_zone_sim)
        elif isinstance(danger_zones, list) and danger_zones:
            reasons.append(f"DANGER_ZONE_COUNT={len(danger_zones)}")
            max_zone_sim = 0.0
            for z in danger_zones:
                if isinstance(z, dict) and "similarity" in z:
                    max_zone_sim = max(
                        max_zone_sim,
                        _safe_float(z.get("similarity"), 0.0),
                    )
            danger_sim = max(danger_sim, max_zone_sim)

        # Optional explicit logging of loss probability
        if loss_prob > 0.0:
            reasons.append(f"LOSS_PROB={loss_prob:.2f}")

        return cls(
            risk_multiplier=risk_mult,
            veto=veto,
            risk_score=risk_score,
            danger_similarity=danger_sim,
            loss_prob=loss_prob,
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

    def __post_init__(self) -> None:
        # Clamp numeric fields
        self.portfolio_risk = _clip(self.portfolio_risk, 0.0, 1.0)
        self.instrument_risk = _clip(self.instrument_risk, 0.0, 1.0)
        self.max_dd = _clip(self.max_dd, 0.0, 1.0)
        self.margin_usage = _clip(self.margin_usage, 0.0, 1.0)
        self.hard_cap = _clip(self.hard_cap, 0.0, 1.0)
        # Risk scale strictly bounded to reasonable dynamic range
        self.risk_scale = max(0.1, min(1.5, _safe_float(self.risk_scale, 1.0)))
        # Normalize risk_level to uppercase for consistency
        if self.risk_level:
            self.risk_level = str(self.risk_level).upper()

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
        - risk_scaling: DynamicRiskController scaling output
        - risk_assessment: DynamicRiskController assessment output
        """
        risk_data = risk_data or {}
        portfolio_risk_raw = portfolio_risk
        risk_scaling = risk_scaling or {}
        risk_assessment = risk_assessment or {}

        # Portfolio risk & margin usage
        port_risk = 0.0
        margin = 0.0
        if isinstance(portfolio_risk_raw, dict):
            port_risk = _safe_float(
                portfolio_risk_raw.get("total_risk", portfolio_risk_raw.get("risk_score", 0.0)),
                0.0,
            )
            margin = _safe_float(portfolio_risk_raw.get("margin_usage", 0.0), 0.0)
        elif isinstance(portfolio_risk_raw, (int, float, str)):
            port_risk = _safe_float(portfolio_risk_raw, 0.0)

        # Basic risk_data metrics
        if isinstance(risk_data, dict):
            max_dd = _safe_float(
                risk_data.get("max_drawdown", risk_data.get("drawdown", 0.0)),
                0.0,
            )
            hard_block = bool(
                risk_data.get("hard_block", risk_data.get("trading_blocked", False))
            )
            raw_cap = risk_data.get("position_cap", risk_data.get("max_position_size", 1.0))
        else:
            max_dd = 0.0
            hard_block = False
            raw_cap = 1.0

        hard_cap = _safe_float(raw_cap, 1.0)

        # Instrument-specific risk (if instrument map provided)
        inst_risk = 0.0
        if instrument and isinstance(risk_data, dict):
            inst_map = risk_data.get("instruments")
            if isinstance(inst_map, dict):
                inst_data = inst_map.get(instrument, {})
                if isinstance(inst_data, dict):
                    inst_risk = _safe_float(inst_data.get("risk", 0.0), 0.0)

        # DynamicRiskController integration
        risk_scale = 1.0
        if isinstance(risk_scaling, dict):
            rs = risk_scaling.get("current_risk_scale")
            risk_scale = _safe_float(rs, 1.0) if rs is not None else 1.0

        risk_level = "NORMAL"
        emergency_mode = False
        if isinstance(risk_assessment, dict):
            rl = risk_assessment.get("risk_level")
            if isinstance(rl, str) and rl:
                risk_level = rl.upper()
            emergency_mode = bool(risk_assessment.get("emergency_active", False))

            if risk_scale == 1.0:
                rs = risk_assessment.get("risk_scale")
                if rs is not None:
                    risk_scale = _safe_float(rs, 1.0)

        # Extract risk_factors if available
        risk_factors: Dict[str, float] = {}
        if isinstance(risk_data, dict):
            rf = risk_data.get("risk_factors")
            if isinstance(rf, dict):
                for k, v in rf.items():
                    risk_factors[k] = _safe_float(v, 0.0)

        # Adjust block/cap based on risk_level and emergency
        if risk_level == "CRITICAL" or emergency_mode:
            hard_block = True
            hard_cap = 0.0
        elif risk_level == "HIGH":
            hard_cap = min(hard_cap, 0.3)
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
        if inst_risk > 0.0:
            reasons.append(f"INSTRUMENT_RISK={inst_risk:.2f}")

        return cls(
            portfolio_risk=port_risk,
            instrument_risk=inst_risk,
            max_dd=max_dd,
            margin_usage=margin,
            hard_block=hard_block,
            hard_cap=hard_cap,
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
    
    ACTION SEMANTICS (v4.1 - Autonomous PPO):
    =========================================
    direction_score: Raw PPO output from action[0] in [-1, 1]
        - > +0.3 ⇒ PPO intends LONG
        - < -0.3 ⇒ PPO intends SHORT
        - |score| ≤ 0.3 ⇒ PPO is uncertain (FLAT)
    
    trust_score: Legacy alias for direction_score (backwards compat)
    
    The final 'direction' field is derived from direction_score,
    potentially blended with experts based on autonomy phase.
    """
    instrument: str
    direction: str = "flat"           # "long" | "short" | "flat"
    confidence: float = 0.0           # 0..1
    position_size: float = 0.0        # 0..1 (risk-normalized)
    
    # PPO autonomous direction (v4.1)
    direction_score: float = 0.0      # Raw PPO output: action[0] in [-1, 1]
    trust_score: float = 0.0          # Legacy alias for direction_score

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

    @property
    def numeric_direction(self) -> int:
        """
        Map direction to {-1, 0, 1} for analysis:
        - long/buy   ->  1
        - short/sell -> -1
        - flat/hold/other -> 0
        """
        d = (self.direction or "").lower()
        if d in ("long", "buy"):
            return 1
        if d in ("short", "sell"):
            return -1
        return 0

    @property
    def is_trade(self) -> bool:
        """True if this decision corresponds to an executed trade."""
        d = (self.direction or "").lower()
        return self.gate_passed and d in ("long", "short", "buy", "sell")

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

        denom = max(self._decision_count, 1)
        self.avg_confidence = self._confidence_sum / denom
        self.avg_position_size = self._size_sum / denom

        d = (decision.direction or "").lower()
        if d in ("long", "buy"):
            self.longs += 1
        elif d in ("short", "sell"):
            self.shorts += 1
        else:
            self.flats += 1

        if decision.gate_passed and decision.is_trade:
            self.trades += 1

        # Store summary for pattern analysis
        self.last_decisions.append(
            {
                "direction": decision.direction,
                "numeric_direction": decision.numeric_direction,
                "confidence": decision.confidence,
                "position_size": decision.position_size,
                "gate_passed": decision.gate_passed,
                "trust_score": decision.trust_score,
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
    2. Soft scaling (risk_multiplier, caps, dynamic risk_scale)
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
        has_existing_position: bool = True,  # v5.2: Per-instrument independence
    ) -> "GatingResult":
        """
        Apply the 3-stage gating pipeline.

        Stage 1: Hard veto (risk.hard_block, emergency_mode, memory.veto,
                 or extreme memory risk/loss probability)
        Stage 2: Soft scaling (risk_multiplier, risk_scale, caps, risk levels)
        Stage 3: PPO uncertainty check (analytics + safety)

        v4.2.0+: Uses DynamicRiskController's risk_scale and risk_level,
        plus UnifiedMemory's loss_prob and risk_score.
        
        v5.2: When has_existing_position=False, skip portfolio-level penalties
        to allow independent trading on instruments without positions.
        """
        result = cls()

        # ─────────────────────────────────────────────────────────
        # Stage 1: Hard veto
        # ─────────────────────────────────────────────────────────
        if risk.hard_block:
            result.gate_passed = False
            result.hard_veto_triggered = True
            result.reasons.append("RISK_HARD_BLOCK")

        if risk.emergency_mode:
            result.gate_passed = False
            result.hard_veto_triggered = True
            result.reasons.append("EMERGENCY_MODE")

        if memory.veto:
            result.gate_passed = False
            result.hard_veto_triggered = True
            result.reasons.append("MEMORY_VETO")

        # Extreme memory-based risk: treat as hard veto
        if not result.hard_veto_triggered:
            if (
                memory.loss_prob >= MEMORY_HARD_VETO_LOSS_PROB
                or memory.risk_score >= MEMORY_HARD_VETO_RISK_SCORE
            ):
                result.gate_passed = False
                result.hard_veto_triggered = True
                result.reasons.append("MEMORY_MAX_RISK")

        if result.hard_veto_triggered:
            result.position_size_cap = 0.0
            result.confidence_multiplier = 0.0
            # No need to proceed with soft scaling or PPO
            return result

        # ─────────────────────────────────────────────────────────
        # Stage 2: Soft scaling
        # ─────────────────────────────────────────────────────────

        # Memory risk multiplier (0..1)
        if memory.risk_multiplier < 1.0:
            mem_scale = _clip(memory.risk_multiplier, 0.0, 1.0)
            result.confidence_multiplier *= mem_scale
            result.soft_scaling_applied = True
            result.reasons.append(f"MEMORY_SCALE={mem_scale:.2f}")

        # Hard cap from risk policy
        if risk.hard_cap < 1.0:
            cap = _clip(risk.hard_cap, 0.0, 1.0)
            result.position_size_cap = min(result.position_size_cap, cap)
            result.soft_scaling_applied = True
            result.reasons.append(f"RISK_CAP={cap:.2f}")

        # Dynamic risk_scale from DRC (0.1-1.5)
        if risk.risk_scale < 1.0:
            # Low risk_scale = reduce position size and confidence
            rs = max(0.1, min(1.0, risk.risk_scale))
            result.position_size_cap *= rs
            # Confidence less aggressive than size scaling
            result.confidence_multiplier *= (0.5 + 0.5 * rs)  # 0.55-1.0
            result.soft_scaling_applied = True
            result.reasons.append(f"DRC_SCALE={rs:.2f}")
        elif risk.risk_scale > 1.0:
            # Recovery mode: limited boost (size only; confidence stays as-is)
            boost = min(1.2, risk.risk_scale)
            result.position_size_cap *= boost
            result.soft_scaling_applied = True
            result.reasons.append(f"DRC_BOOST={boost:.2f}")

        # Risk level based scaling
        # v5.2: Only apply confidence penalty to instruments WITH positions
        # This prevents positions in one instrument from blocking new entries in another
        # M15 SCALPING v5.9: Raised caps - 0.30→0.50 (HIGH), 0.60→0.75 (ELEVATED)
        # Rationale: M15 scalps exit quickly via ExitManager, so tighter risk caps
        # are overly conservative for short-duration trades
        if risk.risk_level == "HIGH":
            result.position_size_cap = min(result.position_size_cap, 0.50)  # was 0.30
            if has_existing_position:
                result.confidence_multiplier *= 0.7  # was 0.6
            result.soft_scaling_applied = True
            result.reasons.append("RISK_LEVEL_HIGH")
        elif risk.risk_level == "ELEVATED":
            result.position_size_cap = min(result.position_size_cap, 0.75)  # was 0.60
            if has_existing_position:
                result.confidence_multiplier *= 0.85  # was 0.8
            result.soft_scaling_applied = True
            result.reasons.append("RISK_LEVEL_ELEVATED")

        # Memory danger similarity penalty
        if memory.danger_similarity > 0.5:
            # Linearly reduce confidence as similarity increases from 0.5 → 1.0
            penalty = max(0.0, 1.0 - (memory.danger_similarity - 0.5))
            result.confidence_multiplier *= penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"DANGER_SIM={memory.danger_similarity:.2f}")

        # Memory risk_score soft penalty (when not already hard veto)
        if MEMORY_SOFT_RISK_SCORE_START < memory.risk_score < MEMORY_HARD_VETO_RISK_SCORE:
            # Map [0.7,0.98] → [1.0,0.3]
            mem_penalty = 1.0 - 0.7 * (memory.risk_score - MEMORY_SOFT_RISK_SCORE_START) / (
                MEMORY_HARD_VETO_RISK_SCORE - MEMORY_SOFT_RISK_SCORE_START
            )
            mem_penalty = max(0.3, min(1.0, mem_penalty))
            result.confidence_multiplier *= mem_penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"MEM_RISK={memory.risk_score:.2f}")

        # Loss probability soft penalty
        if MEMORY_SOFT_LOSS_PROB_START < memory.loss_prob < MEMORY_HARD_VETO_LOSS_PROB:
            # Map [0.5,0.98] → [1.0,0.4]
            lp_penalty = 1.0 - 0.6 * (memory.loss_prob - MEMORY_SOFT_LOSS_PROB_START) / (
                MEMORY_HARD_VETO_LOSS_PROB - MEMORY_SOFT_LOSS_PROB_START
            )
            lp_penalty = max(0.4, min(1.0, lp_penalty))
            result.confidence_multiplier *= lp_penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"LOSS_PROB={memory.loss_prob:.2f}")

        # High portfolio risk penalty
        # v5.2: ONLY apply to instruments WITH existing positions
        # Instruments without positions are independent - don't penalize them
        # for risk from OTHER instruments' positions
        if risk.portfolio_risk > PORTFOLIO_RISK_PENALTY_START and has_existing_position:
            # Map [0.7,1.0] → [1.0,0.4]
            penalty = 1.0 - (risk.portfolio_risk - PORTFOLIO_RISK_PENALTY_START) * 2.0
            penalty = max(0.3, min(1.0, penalty))
            result.confidence_multiplier *= penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"HIGH_RISK={risk.portfolio_risk:.2f}")

        # Instrument-specific risk scaling
        if risk.instrument_risk > INSTRUMENT_RISK_HIGH:
            result.position_size_cap *= 0.4
            result.soft_scaling_applied = True
            result.reasons.append(f"INSTR_RISK_HIGH={risk.instrument_risk:.2f}")
        elif risk.instrument_risk > INSTRUMENT_RISK_ELEVATED:
            result.position_size_cap *= 0.7
            result.soft_scaling_applied = True
            result.reasons.append(f"INSTR_RISK_ELEVATED={risk.instrument_risk:.2f}")

        # ─────────────────────────────────────────────────────────
        # Stage 3: PPO uncertainty check (aligned with direction logic)
        # ─────────────────────────────────────────────────────────
        # If PPO's direction signal is very weak (near zero), it's uncertain.
        # We block trades when PPO is indecisive, not when it wants to SHORT.
        # Note: direction_score is signed (-1 to 1) where negative = SHORT.
        # A strong SHORT signal (e.g., -0.8) should PASS, not be blocked.
        direction_magnitude = abs(trust_score)
        if direction_magnitude < PPO_SIGNAL_UNCERTAIN_THRESHOLD:
            result.ppo_override = True
            result.gate_passed = False
            result.reasons.append(f"PPO_UNCERTAIN={trust_score:.2f}")

        # Final sanity clamps
        result.position_size_cap = max(0.0, min(2.0, result.position_size_cap))
        result.confidence_multiplier = max(0.0, min(2.0, result.confidence_multiplier))

        return result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

DEFAULT_INSTRUMENTS: List[str] = ["XAUUSD", "EURUSD"]
PRIMARY_INSTRUMENT: str = "XAUUSD"
