#!/usr/bin/env python3

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from modules.meta.numeric_utils import _clip, _safe_float

PPO_SIGNAL_UNCERTAIN_THRESHOLD: float = 0.30


MEMORY_HARD_VETO_RISK_SCORE: float = 0.98
MEMORY_HARD_VETO_LOSS_PROB: float = 0.98


MEMORY_SOFT_RISK_SCORE_START: float = 0.70
MEMORY_SOFT_LOSS_PROB_START: float = 0.50


PORTFOLIO_RISK_PENALTY_START: float = 0.70


INSTRUMENT_RISK_ELEVATED: float = 0.60
INSTRUMENT_RISK_HIGH: float = 0.80


@dataclass
class MemoryGateInfo:
    risk_multiplier: float = 1.0
    veto: bool = False
    risk_score: float = 0.0
    danger_similarity: float = 0.0
    loss_prob: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:

        self.risk_multiplier = _clip(self.risk_multiplier, 0.0, 1.0)
        self.risk_score = _clip(self.risk_score, 0.0, 1.0)
        self.danger_similarity = _clip(self.danger_similarity, 0.0, 1.0)
        self.loss_prob = _clip(self.loss_prob, 0.0, 1.0)

    @classmethod
    def from_bus_data(cls, memory_gate: Any, danger_zones: Any) -> "MemoryGateInfo":

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


        if isinstance(memory_gate, dict):
            risk_mult = _safe_float(memory_gate.get("risk_multiplier", 1.0), 1.0)
            veto = bool(memory_gate.get("veto", False))
            risk_score = _safe_float(memory_gate.get("risk_score", 0.0), 0.0)
            danger_sim = _safe_float(memory_gate.get("danger_similarity", 0.0), 0.0)
            loss_prob = _safe_float(memory_gate.get("loss_prob", 0.0), 0.0)
            raw_reasons = memory_gate.get("reasons", []) or []
            reasons = [_reason_to_str(r) for r in raw_reasons]
        elif memory_gate is not None:

            risk_mult = _safe_float(memory_gate, 1.0)
            veto = False

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


        if isinstance(danger_zones, dict):
            zone_count = int(danger_zones.get("zone_count", 0) or 0)
            if zone_count > 0:
                reasons.append(f"DANGER_ZONE_COUNT={zone_count}")

            if "max_similarity" in danger_zones:
                dz_sim = _safe_float(danger_zones.get("max_similarity", 0.0), 0.0)
                danger_sim = max(danger_sim, dz_sim)

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
    instrument: str = ""


@dataclass
class RiskInfo:
    portfolio_risk: float = 0.0
    instrument_risk: float = 0.0
    max_dd: float = 0.0
    margin_usage: float = 0.0
    hard_block: bool = False
    hard_cap: float = 1.0
    reasons: List[str] = field(default_factory=list)


    risk_scale: float = 1.0
    risk_level: str = "NORMAL"
    emergency_mode: bool = False
    risk_factors: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:

        self.portfolio_risk = _clip(self.portfolio_risk, 0.0, 1.0)
        self.instrument_risk = _clip(self.instrument_risk, 0.0, 1.0)
        self.max_dd = _clip(self.max_dd, 0.0, 1.0)
        self.margin_usage = _clip(self.margin_usage, 0.0, 1.0)
        self.hard_cap = _clip(self.hard_cap, 0.0, 1.0)

        self.risk_scale = max(0.1, min(1.5, _safe_float(self.risk_scale, 1.0)))

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
        risk_data = risk_data or {}
        portfolio_risk_raw = portfolio_risk
        risk_scaling = risk_scaling or {}
        risk_assessment = risk_assessment or {}


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


        inst_risk = 0.0
        if instrument and isinstance(risk_data, dict):
            inst_map = risk_data.get("instruments")
            if isinstance(inst_map, dict):
                inst_data = inst_map.get(instrument, {})
                if isinstance(inst_data, dict):
                    inst_risk = _safe_float(inst_data.get("risk", 0.0), 0.0)


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


        risk_factors: Dict[str, float] = {}
        if isinstance(risk_data, dict):
            rf = risk_data.get("risk_factors")
            if isinstance(rf, dict):
                for k, v in rf.items():
                    risk_factors[k] = _safe_float(v, 0.0)


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
    instrument: str = ""


@dataclass
class InstrumentDecision:
    instrument: str
    direction: str = "flat"
    confidence: float = 0.0
    position_size: float = 0.0


    direction_score: float = 0.0
    trust_score: float = 0.0


    committee_action: str = "hold"
    committee_confidence: float = 0.0
    expert_consensus: str = "flat"
    expert_confidence: float = 0.0


    regime: str = "unknown"
    regime_strength: float = 0.0


    value_estimate: float = 0.0
    raw_action: List[float] = field(default_factory=list)


    gate_passed: bool = False
    gate_reasons: List[str] = field(default_factory=list)


    reasoning: str = ""


    meta: Dict[str, Any] = field(default_factory=dict)


    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def numeric_direction(self) -> int:
        d = (self.direction or "").lower()
        if d in ("long", "buy"):
            return 1
        if d in ("short", "sell"):
            return -1
        return 0

    @property
    def is_trade(self) -> bool:
        d = (self.direction or "").lower()
        return self.gate_passed and d in ("long", "short", "buy", "sell")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_legacy_format(self) -> Dict[str, Any]:
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
    instruments: Dict[str, InstrumentDecision] = field(default_factory=dict)
    global_meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "PPOAgent"

    def get_primary_decision(self, primary_instrument: str = "XAUUSD") -> InstrumentDecision:
        if primary_instrument in self.instruments:
            return self.instruments[primary_instrument]
        if self.instruments:

            return next(iter(self.instruments.values()))

        return InstrumentDecision(instrument=primary_instrument)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruments": {k: v.to_dict() for k, v in self.instruments.items()},
            "global_meta": self.global_meta,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    def to_legacy_format(self, primary_instrument: str = "XAUUSD") -> Dict[str, Any]:
        primary = self.get_primary_decision(primary_instrument)
        return {
            "ppo_final_decision": primary.to_legacy_format(),
            "ppo_gate_passed": primary.gate_passed,
            "ppo_position_size": primary.position_size,
            "arbiter_decision": self.to_dict(),
        }


@dataclass
class InstrumentStats:
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


    _confidence_sum: float = field(default=0.0, repr=False)
    _size_sum: float = field(default=0.0, repr=False)
    _decision_count: int = field(default=0, repr=False)


    last_decisions: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=50)
    )

    def record_decision(self, decision: InstrumentDecision) -> None:
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
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def long_bias(self) -> float:
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
    stats: Dict[str, InstrumentStats] = field(default_factory=dict)

    def get_or_create(self, instrument: str) -> InstrumentStats:
        if instrument not in self.stats:
            self.stats[instrument] = InstrumentStats(instrument=instrument)
        return self.stats[instrument]

    def record_decision(self, decision: InstrumentDecision) -> None:
        stats = self.get_or_create(decision.instrument)
        stats.record_decision(decision)

    def record_multi_decision(self, multi_decision: ArbiterMultiDecision) -> None:
        for decision in multi_decision.instruments.values():
            self.record_decision(decision)

    def record_trade_result(self, instrument: str, pnl: float) -> None:
        stats = self.get_or_create(instrument)
        stats.record_trade_result(pnl)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self.stats.items()}


@dataclass
class GatingResult:
    gate_passed: bool = True
    position_size_cap: float = 1.0
    confidence_multiplier: float = 1.0
    reasons: List[str] = field(default_factory=list)


    hard_veto_triggered: bool = False
    soft_scaling_applied: bool = False
    ppo_override: bool = False

    @classmethod
    def apply_gates(
        cls,
        memory: MemoryGateInfo,
        risk: RiskInfo,
        trust_score: float,
        has_existing_position: bool = True,
    ) -> "GatingResult":
        result = cls()


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

            return result


        if memory.risk_multiplier < 1.0:
            mem_scale = _clip(memory.risk_multiplier, 0.0, 1.0)
            result.confidence_multiplier *= mem_scale
            result.soft_scaling_applied = True
            result.reasons.append(f"MEMORY_SCALE={mem_scale:.2f}")


        if risk.hard_cap < 1.0:
            cap = _clip(risk.hard_cap, 0.0, 1.0)
            result.position_size_cap = min(result.position_size_cap, cap)
            result.soft_scaling_applied = True
            result.reasons.append(f"RISK_CAP={cap:.2f}")


        if risk.risk_scale < 1.0:

            rs = max(0.1, min(1.0, risk.risk_scale))
            result.position_size_cap *= rs

            result.confidence_multiplier *= (0.5 + 0.5 * rs)
            result.soft_scaling_applied = True
            result.reasons.append(f"DRC_SCALE={rs:.2f}")
        elif risk.risk_scale > 1.0:

            boost = min(1.2, risk.risk_scale)
            result.position_size_cap *= boost
            result.soft_scaling_applied = True
            result.reasons.append(f"DRC_BOOST={boost:.2f}")


        if risk.risk_level == "HIGH":
            result.position_size_cap = min(result.position_size_cap, 0.3)
            if has_existing_position:
                result.confidence_multiplier *= 0.6
            result.soft_scaling_applied = True
            result.reasons.append("RISK_LEVEL_HIGH")
        elif risk.risk_level == "ELEVATED":
            result.position_size_cap = min(result.position_size_cap, 0.6)
            if has_existing_position:
                result.confidence_multiplier *= 0.8
            result.soft_scaling_applied = True
            result.reasons.append("RISK_LEVEL_ELEVATED")


        if memory.danger_similarity > 0.5:

            penalty = max(0.0, 1.0 - (memory.danger_similarity - 0.5))
            result.confidence_multiplier *= penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"DANGER_SIM={memory.danger_similarity:.2f}")


        if MEMORY_SOFT_RISK_SCORE_START < memory.risk_score < MEMORY_HARD_VETO_RISK_SCORE:

            mem_penalty = 1.0 - 0.7 * (memory.risk_score - MEMORY_SOFT_RISK_SCORE_START) / (
                MEMORY_HARD_VETO_RISK_SCORE - MEMORY_SOFT_RISK_SCORE_START
            )
            mem_penalty = max(0.3, min(1.0, mem_penalty))
            result.confidence_multiplier *= mem_penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"MEM_RISK={memory.risk_score:.2f}")


        if MEMORY_SOFT_LOSS_PROB_START < memory.loss_prob < MEMORY_HARD_VETO_LOSS_PROB:

            lp_penalty = 1.0 - 0.6 * (memory.loss_prob - MEMORY_SOFT_LOSS_PROB_START) / (
                MEMORY_HARD_VETO_LOSS_PROB - MEMORY_SOFT_LOSS_PROB_START
            )
            lp_penalty = max(0.4, min(1.0, lp_penalty))
            result.confidence_multiplier *= lp_penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"LOSS_PROB={memory.loss_prob:.2f}")


        if risk.portfolio_risk > PORTFOLIO_RISK_PENALTY_START and has_existing_position:

            penalty = 1.0 - (risk.portfolio_risk - PORTFOLIO_RISK_PENALTY_START) * 2.0
            penalty = max(0.3, min(1.0, penalty))
            result.confidence_multiplier *= penalty
            result.soft_scaling_applied = True
            result.reasons.append(f"HIGH_RISK={risk.portfolio_risk:.2f}")


        if risk.instrument_risk > INSTRUMENT_RISK_HIGH:
            result.position_size_cap *= 0.4
            result.soft_scaling_applied = True
            result.reasons.append(f"INSTR_RISK_HIGH={risk.instrument_risk:.2f}")
        elif risk.instrument_risk > INSTRUMENT_RISK_ELEVATED:
            result.position_size_cap *= 0.7
            result.soft_scaling_applied = True
            result.reasons.append(f"INSTR_RISK_ELEVATED={risk.instrument_risk:.2f}")


        direction_magnitude = abs(trust_score)
        if direction_magnitude < PPO_SIGNAL_UNCERTAIN_THRESHOLD:
            result.ppo_override = True
            result.gate_passed = False
            result.reasons.append(f"PPO_UNCERTAIN={trust_score:.2f}")


        result.position_size_cap = max(0.0, min(2.0, result.position_size_cap))
        result.confidence_multiplier = max(0.0, min(2.0, result.confidence_multiplier))

        return result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_INSTRUMENTS: List[str] = ["XAUUSD"]
PRIMARY_INSTRUMENT: str = "XAUUSD"
