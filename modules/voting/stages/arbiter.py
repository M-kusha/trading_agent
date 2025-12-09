"""
Final Arbiter
=============
Final gate decision for trading signals.

Combines all voting inputs (consensus, collusion, uncertainty, horizon, risk)
to produce the final trading decision.

Key properties (v2.0):
- GLOBAL gate is minimal: only blocks true global problems (collusion, memory veto)
- REAL per-instrument decisions:
    • Reads committee_decisions_by_instrument from SmartInfoBus
    • Each instrument has its own direction, confidence, consensus and fragility
    • Adaptive thresholds per instrument via DynamicThresholdManager
- Optional technical override:
    • MomentumExpert + TrendExpert can flip direction if both strongly disagree
- Fallback path:
    • When per-instrument decisions are missing, use global decision
      filtered by instrument-specific alignment (price, trend, volatility)
"""

from __future__ import annotations

import datetime
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase
from modules.voting.core.constants import (
    CONFIDENCE_THRESHOLD_F,
    CONSENSUS_THRESHOLD_F,
    VotingAction,
    is_training_mode,
    is_live_mode,
    get_voting_mode,
    get_instrument_threshold,  # Adaptive per-instrument thresholds
    _threshold_manager,        # For feedback recording
)
from modules.voting.core.per_instrument import (
    DEFAULT_INSTRUMENTS,
    normalize_instrument,
    extract_instrument_data,
)


@module(**module_args("FinalArbiter"))
class FinalArbiter(VotingModuleBase):
    """
    Final arbitration for voting pipeline.

    Makes final gate decision based on:
    - Committee decision
    - Consensus score
    - Collusion score
    - Fragility / uncertainty
    - Horizon alignment
    - Risk / memory constraints

    Publishes (SmartInfoBus):
    - final_decision
    - gate_passed
    - trading_signal
    - instrument_signals
    - arbiter_analysis
    """

    # ──────────────────────────────────────────────────────────────────────
    # INITIALIZATION & UTILITIES
    # ──────────────────────────────────────────────────────────────────────

    def _module_specific_init(self) -> None:
        """Initialize arbiter state and configuration."""
        # Configurable thresholds (fallback to dynamic functions if None)
        self._config_min_confidence = self.config.get("min_confidence")
        self._config_consensus_threshold = self.config.get("consensus_threshold")

        # Global safety limits
        self.max_fragility = float(self.config.get("max_fragility", 0.9))      # Warn-only
        self.max_collusion = float(self.config.get("max_collusion", 0.995))    # True global block
        self.bootstrap_steps = int(self.config.get("bootstrap_steps", 50))

        # Optional: allow technical experts to override RL/committee direction
        self.technical_override_enabled = bool(
            self.config.get("technical_override_enabled", False)
        )

        # Gate criteria weights for global weighted_score
        # (used to scale confidence, not to hard-block per instrument)
        self.criteria_weights = self.config.get(
            "criteria_weights",
            [0.25, 0.20, 0.20, 0.20, 0.15],  # strength, consensus, reliability, risk, novelty
        )

        # Counters
        self._step_count = 0
        self._gate_passes = 0
        self._gate_attempts = 0

        # Member blending weights (if needed)
        self.weights = np.ones(5, dtype=np.float32) / 5.0

        # History / stats
        self.decision_history: deque = deque(maxlen=200)
        self.gate_decisions: deque = deque(maxlen=150)

        self.arbiter_stats: Dict[str, Any] = {
            "total_decisions": 0,
            "gate_passes": 0,
            "gate_failures": 0,
            "gate_pass_rate": 0.0,
            "avg_confidence": 0.5,
        }

        self.voting_quality: Dict[str, float] = {
            "avg_consensus": 0.5,
            "gate_effectiveness": 0.5,
            "decision_confidence": 0.5,
        }

        # Instrument universe (default from per_instrument helpers)
        self.instruments: List[str] = self.config.get("instruments", DEFAULT_INSTRUMENTS)

        # Log current mode-aware thresholds
        mode = get_voting_mode()
        self.logger.info(
            f"[ARBITER] FinalArbiter initialized | MODE={mode} | "
            f"min_conf={self.min_confidence:.2f} | "
            f"consensus_thresh={self.consensus_threshold:.2f} | "
            f"technical_override_enabled={self.technical_override_enabled}"
        )

        # Baseline bus keys for downstream modules
        self._publish_arbiter_baseline()

    @property
    def min_confidence(self) -> float:
        """
        Minimum confidence threshold (mode-aware).

        If explicitly configured, use config value.
        Otherwise delegate to CONFIDENCE_THRESHOLD_F(), which already
        understands training/live modes.
        """
        if self._config_min_confidence is not None:
            return float(self._config_min_confidence)
        return CONFIDENCE_THRESHOLD_F()

    @property
    def consensus_threshold(self) -> float:
        """
        Minimum consensus threshold (mode-aware).

        If explicitly configured, use config value.
        Otherwise delegate to CONSENSUS_THRESHOLD_F().
        """
        if self._config_consensus_threshold is not None:
            return float(self._config_consensus_threshold)
        return CONSENSUS_THRESHOLD_F()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        Robust float conversion from SmartInfoBus values.

        Handles:
        - None → default
        - int/float → float(value)
        - dict → tries common numeric keys ('value', 'score', 'confidence', 'prob', 'p')
        - string → float(string) when possible

        On any error, returns default.
        """
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, dict):
                for key in ("value", "score", "confidence", "prob", "p"):
                    v = value.get(key)
                    if isinstance(v, (int, float)):
                        return float(v)
                return default
            # Last-resort: string/other → float()
            return float(value)
        except Exception:
            return default


    def _publish_arbiter_baseline(self) -> None:
        """Publish baseline arbiter keys so downstream modules always see a valid shape."""
        try:
            self.smart_bus.set(
                "final_decision",
                {"action": "abstain", "reason": "initializing"},
                module="FinalArbiter",
                thesis="Baseline final decision",
            )
            self.smart_bus.set(
                "gate_passed",
                False,
                module="FinalArbiter",
                thesis="Baseline gate status",
            )
            self.smart_bus.set(
                "instruments",
                self.instruments,
                module="FinalArbiter",
                thesis="Instrument universe",
            )
        except Exception:
            # Never crash on baseline publication
            pass

    def _extract_memory_gate(self, value: Any) -> float:
        """
        Normalize UnifiedMemory 'memory_gate' into a numeric multiplier.

        UnifiedMemory typically publishes a dict. We search in order for:
        - risk_multiplier
        - risk_score
        - score
        - value
        - gate
        and fall back gracefully.

        If a 'veto' flag is set and True, we treat it as a hard veto (0.0).
        """
        try:
            if isinstance(value, dict):
                for key in ("risk_multiplier", "risk_score", "score", "value", "gate"):
                    v = value.get(key)
                    if isinstance(v, (int, float)):
                        return float(v)
                # Explicit veto flag
                if value.get("veto") is True:
                    return 0.0
            if isinstance(value, (int, float)):
                return float(value)
        except Exception:
            pass
        return 1.0

    # ──────────────────────────────────────────────────────────────────────
    # MAIN PROCESSING
    # ──────────────────────────────────────────────────────────────────────

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """
        Main arbitration entrypoint.

        High-level steps:
        1. Collect voting data from SmartInfoBus
        2. Run GLOBAL gate (collusion + memory + fragility warning)
        3. Build global final_decision (mostly for logging / compatibility)
        4. Build per-instrument signals (with adaptive thresholds + overrides)
        5. Publish everything back to SmartInfoBus
        """
        start = time.time()
        name = self.__class__.__name__

        mode = get_voting_mode()
        if self._step_count == 0 or self._step_count % 100 == 0:
            self.logger.info(
                f"[ARBITER] Processing step {self._step_count} | MODE={mode} | "
                f"thresholds: min_conf={self.min_confidence:.2f}, "
                f"consensus={self.consensus_threshold:.2f}"
            )

        try:
            # Decision correlation ID (if kernel sets it)
            decision_id = self.smart_bus.get("kernel_decision_id", name)

            # 1) Collect voting data
            data = await self._collect_voting_data()

            # 2) Global gate: only true global issues
            gate_result = await self._evaluate_gate(data)

            # 3) Global final decision (used for logging / legacy consumers)
            final_decision = await self._generate_final_decision(data, gate_result)

            # 4) Per-instrument signals (true trade routing)
            instrument_signals = await self._generate_instrument_signals(
                final_decision, data
            )

            # 5) Human-readable thesis
            thesis = self._generate_thesis(final_decision, gate_result)

            # 6) Publish back to SmartInfoBus
            await self._update_bus(
                final_decision, gate_result, instrument_signals, thesis
            )

            # 7) Update in-memory stats
            self._update_stats(gate_result)

            elapsed_ms = (time.time() - start) * 1000.0
            # VotingModuleBase provides performance_tracker
            if getattr(self, "performance_tracker", None) is not None:
                self.performance_tracker.record_metric(
                    name, "process", elapsed_ms, True
                )

            # Contract-compliant output
            return {
                "final_decision": final_decision,
                "gate_passed": gate_result["passed"],
                "gate_result": gate_result,
                "trading_signal": final_decision,
                "instrument_signals": instrument_signals,
                "arbiter_analysis": {
                    "input_summary": self._summarize_inputs(data),
                    "gate_criteria": gate_result.get("criteria_scores", {}),
                    "final_confidence": final_decision.get("confidence", 0.0),
                },
                "arbiter_statistics": dict(self.arbiter_stats),
                "voting_quality": dict(self.voting_quality),
                "decision_id": decision_id,
                "arbiter_decision_id": decision_id,
                # Contract-expected aliases
                "trade_vote": final_decision,
                "gate_decision": gate_result,
                "arbiter_thesis": thesis,
                "decision_confidence": final_decision.get("confidence", 0.0),
                "decision_rationale": final_decision.get("rationale", thesis),
                "arbiter_recommendations": {
                    "action": final_decision.get("action", "abstain"),
                    "gate": gate_result["passed"],
                    "signals": instrument_signals,
                },
                "member_weights": data.get("aligned_weights", {}),
                "_thesis": thesis,
            }

        except Exception as e:
            # Use ErrorPinpointer if available for richer diagnostics
            err_pin = getattr(self, "error_pinpointer", None)
            if err_pin is not None:
                error_context = err_pin.analyze_error(e, "arbiter_process")
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)


    async def _collect_voting_data(self) -> Dict[str, Any]:
        """
        Collect all upstream voting inputs from SmartInfoBus.

        All numeric fields are passed through _safe_float() to avoid
        'float(dict)' type crashes when upstream modules change shape.
        """
        name = self.__class__.__name__

        committee_conf_raw = self.smart_bus.get("committee_confidence", name)
        consensus_raw = self.smart_bus.get("consensus_score", name)
        collusion_raw = self.smart_bus.get("collusion_score", name)
        fragility_raw = self.smart_bus.get("fragility", name)
        uncertainty_raw = self.smart_bus.get("uncertainty_score", name)

        return {
            # Committee / PPO
            "committee_decision": self.smart_bus.get("committee_decision", name) or {},
            "committee_confidence": self._safe_float(committee_conf_raw, 0.5),
            "trade_vote_v2": self.smart_bus.get("trade_vote_v2", name) or {},
            # Consensus / collusion
            "consensus_score": self._safe_float(consensus_raw, 0.5),
            "consensus_analysis": self.smart_bus.get("consensus_analysis", name) or {},
            "collusion_score": self._safe_float(collusion_raw, 0.0),
            "suspicious_pairs": self.smart_bus.get("suspicious_pairs", name) or [],
            # Uncertainty / fragility
            "fragility": self._safe_float(fragility_raw, 0.5),
            "uncertainty_score": self._safe_float(uncertainty_raw, 0.5),
            "instrument_fragility": self.smart_bus.get(
                "instrument_fragility", name
            ) or {},
            # Horizon alignment
            "horizon_alignment": self.smart_bus.get("horizon_alignment", name) or {},
            "aligned_weights": self.smart_bus.get("aligned_weights", name) or {},
            # Market context
            "market_regime": self.smart_bus.get("market_regime", name) or "unknown",
            "session": self.smart_bus.get("session_canonical", name) or "unknown",
            # Risk / memory
            "risk_data": self.smart_bus.get("risk_data", name) or {},
            "memory_gate": self._extract_memory_gate(
                self.smart_bus.get("memory_gate", name)
            ),
        }

    # ──────────────────────────────────────────────────────────────────────
    # GLOBAL GATE
    # ──────────────────────────────────────────────────────────────────────

    async def _evaluate_gate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate GLOBAL gate criteria only.

        This gate is intentionally minimal:

        Blocks only:
        - EXTREME collusion (likely data issue)
        - Explicit memory veto (UnifiedMemory, via memory_gate multiplier)

        Fragility is warn-only at global level; the real work happens
        in per-instrument gates.

        Confidence / consensus checks are **not** done globally anymore.
        They are delegated to _check_instrument_gate().
        """
        self._step_count += 1
        self._gate_attempts += 1

        confidence = data.get("committee_confidence", 0.5)  # Only for scoring
        consensus = data.get("consensus_score", 0.5)
        collusion = data.get("collusion_score", 0.0)
        fragility = data.get("fragility", 0.5)
        memory_gate = data.get("memory_gate", 1.0)

        criteria_scores = {
            "confidence": confidence,
            "consensus": consensus,
            "anti_collusion": 1.0 - collusion,
            "stability": 1.0 - fragility,
            "memory_clear": memory_gate,
        }

        weights = self.criteria_weights
        if len(weights) < 5:
            weights = [0.2] * 5

        weighted_score = (
            weights[0] * criteria_scores["confidence"]
            + weights[1] * criteria_scores["consensus"]
            + weights[2] * criteria_scores["anti_collusion"]
            + weights[3] * criteria_scores["stability"]
            + weights[4] * criteria_scores["memory_clear"]
        )

        # Global conditions
        collusion_ok = collusion < self.max_collusion
        memory_ok = memory_gate > 0.30
        fragility_warning = fragility >= self.max_fragility  # warn-only

        gate_passed = collusion_ok and memory_ok

        if gate_passed:
            self._gate_passes += 1
            fragility_msg = (
                f" ⚠️ HIGH_FRAGILITY={fragility:.2f}" if fragility_warning else ""
            )
            self.logger.debug(
                f"[ARBITER] Global gate PASSED [MODE={get_voting_mode()}]: "
                f"score={weighted_score:.2f}, collusion={collusion:.2f}, "
                f"memory={memory_gate:.2f}{fragility_msg}"
            )
        else:
            failed_criteria: List[str] = []
            if not collusion_ok:
                failed_criteria.append(
                    f"collusion({collusion:.2f}>{self.max_collusion:.2f})"
                )
            if not memory_ok:
                failed_criteria.append(f"memory_gate({memory_gate:.2f}<0.30)")
            if fragility_warning:
                failed_criteria.append(
                    f"⚠️fragility({fragility:.2f}>{self.max_fragility:.2f})[warn-only]"
                )

            self.logger.warning(
                f"[ARBITER] Global gate BLOCKED [MODE={get_voting_mode()}]: "
                f"failed=[{', '.join(failed_criteria)}]"
            )

        return {
            "passed": gate_passed,
            "weighted_score": weighted_score,
            "criteria_scores": criteria_scores,
            "criteria_passed": {
                "confidence": True,  # Per-instrument
                "consensus": True,   # Per-instrument
                "collusion": collusion_ok,
                "fragility": True,   # Warn only
                "memory": memory_ok,
            },
            "bootstrap_active": self._step_count < self.bootstrap_steps,
        }

    # ──────────────────────────────────────────────────────────────────────
    # GLOBAL DECISION
    # ──────────────────────────────────────────────────────────────────────

    async def _generate_final_decision(
        self, data: Dict[str, Any], gate_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate global trading decision.

        This is kept mostly for compatibility and logging. The true routing
        for live trading is done by _generate_instrument_signals().
        """
        committee = data.get("committee_decision", {})
        trade_vote = data.get("trade_vote_v2", {})

        if not gate_result["passed"]:
            fallback_action = committee.get("action", "abstain")
            fallback_conf = float(data.get("committee_confidence", 0.1)) * 0.5
            return {
                "action": fallback_action,
                "confidence": max(0.05, fallback_conf),
                "reason": "gate_failed",
                "gate_score": gate_result["weighted_score"],
                "original_action": committee.get("action", "unknown"),
                "gate_passed": False,
            }

        # Prefer trade_vote_v2 when available
        action = trade_vote.get("action") or committee.get("action", "abstain")
        confidence = float(
            trade_vote.get("confidence") or data.get("committee_confidence", 0.5)
        )

        adjusted_confidence = confidence * gate_result["weighted_score"]

        # Penalize high global fragility
        fragility = float(data.get("fragility", 0.5))
        if fragility > 0.5:
            adjusted_confidence *= (1.0 - (fragility - 0.5))

        adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))

        return {
            "action": action,
            "confidence": adjusted_confidence,
            "raw_confidence": confidence,
            "gate_score": gate_result["weighted_score"],
            "consensus_score": data.get("consensus_score", 0.5),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # ──────────────────────────────────────────────────────────────────────
    # PER-INSTRUMENT SIGNALS
    # ──────────────────────────────────────────────────────────────────────

    async def _generate_instrument_signals(
        self, decision: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate per-instrument trading signals.

        v2.0 architecture:

        1) Preferred path:
           - Use 'committee_decisions_by_instrument' from CommitteeCoordinator.
           - Each instrument gets:
             • its own action (LONG/SHORT/HOLD)
             • its own confidence and consensus
             • its own fragility
           - Per-instrument gate decides if the trade passes.

        2) Fallback path:
           - If per-instrument committee data is missing, fall back to
             global decision + simple alignment filter per instrument.
        """
        signals: Dict[str, Dict[str, Any]] = {}
        name = self.__class__.__name__

        # ── 1) Preferred path: per-instrument committee decisions ──
        per_inst_decisions: Dict[str, Dict[str, Any]] = (
            self.smart_bus.get("committee_decisions_by_instrument", name) or {}
        )

        if per_inst_decisions:
            self.logger.debug(
                f"[ARBITER] Using per-instrument decisions: "
                f"{list(per_inst_decisions.keys())}"
            )

            # Log raw per-instrument decisions for transparency
            for debug_inst, debug_dec in per_inst_decisions.items():
                raw_conf = self._safe_float(debug_dec.get("confidence"), 0.0)
                raw_cons = self._safe_float(debug_dec.get("consensus_score"), 0.0)
                self.logger.info(
                    f"[ARBITER][RAW] {debug_inst}: "
                    f"action={debug_dec.get('action')}, "
                    f"conf={raw_conf:.4f}, consensus={raw_cons:.4f}"
                )

            collusion_score = float(data.get("collusion_score", 0.0))
            global_fragility = float(data.get("fragility", 0.5))
            instrument_fragility_map: Dict[str, Any] = (
                data.get("instrument_fragility", {}) or {}
            )
            market_regime = str(data.get("market_regime", "UNKNOWN"))

            for inst in self.instruments:
                inst_normalized = normalize_instrument(inst)
                inst_decision = (
                    per_inst_decisions.get(inst_normalized)
                    or per_inst_decisions.get(inst)
                    or {}
                )

                if inst_decision:
                    inst_action = str(inst_decision.get("action", "flat")).upper()
                    inst_confidence = self._safe_float(
                        inst_decision.get("confidence"), 0.0
                    )
                    inst_consensus = self._safe_float(
                        inst_decision.get("consensus_score"), 0.0
                    )

                    # Prefer instrument-specific fragility if available
                    inst_fragility = self._safe_float(
                        instrument_fragility_map.get(inst)
                        or instrument_fragility_map.get(inst_normalized),
                        global_fragility,
                    )

                    gate_passed, final_action, final_confidence = (
                        self._check_instrument_gate(
                            instrument=inst,
                            action=inst_action,
                            confidence=inst_confidence,
                            consensus_score=inst_consensus,
                            collusion_score=collusion_score,
                            fragility=inst_fragility,
                            market_regime=market_regime,
                        )
                    )

                    if gate_passed and final_action.upper() in (
                        "LONG",
                        "SHORT",
                        "BUY",
                        "SELL",
                    ):
                        if final_action.upper() in ("LONG", "BUY"):
                            output_action = "BUY"
                            intensity = round(final_confidence, 4)
                        else:
                            output_action = "SELL"
                            intensity = round(-final_confidence, 4)

                        signals[inst] = {
                            "action": output_action,
                            "confidence": round(final_confidence, 4),
                            "size_multiplier": round(final_confidence, 4),
                            "intensity": intensity,  # For PositionManager
                            "instrument": inst,
                            "consensus_score": inst_consensus,
                            "gate_passed": True,
                            "reason": (
                                f"Per-instrument decision: {output_action} "
                                f"(conf={final_confidence:.2f})"
                            ),
                            "source": "per_instrument_committee",
                            "original_action": inst_action,
                        }
                    else:
                        signals[inst] = {
                            "action": "HOLD",
                            "confidence": 0.0,
                            "size_multiplier": 0.0,
                            "intensity": 0.0,
                            "instrument": inst,
                            "gate_passed": False,
                            "reason": (
                                f"Gate blocked or flat signal "
                                f"(action={inst_action}, gate={gate_passed})"
                            ),
                            "source": "per_instrument_committee",
                        }
                else:
                    # No decision for this instrument
                    signals[inst] = {
                        "action": "HOLD",
                        "confidence": 0.0,
                        "size_multiplier": 0.0,
                        "intensity": 0.0,
                        "instrument": inst,
                        "reason": "No committee decision for instrument",
                        "source": "fallback",
                    }

            self.logger.info(
                "[ARBITER] Per-instrument signals: "
                + ", ".join(f"{k}={v['action']}" for k, v in signals.items())
            )
            return signals

        # ── 2) Fallback path: global decision + alignment ──
        self.logger.debug("[ARBITER] Falling back to global decision + alignment")

        action = str(decision.get("action", "abstain")).upper()
        base_confidence = float(decision.get("confidence", 0.0))

        if action.lower() in ("hold", "abstain", "unknown", "flat"):
            for inst in self.instruments:
                signals[inst] = {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "size_multiplier": 0.0,
                    "intensity": 0.0,
                    "instrument": inst,
                    "reason": f"Global decision is {action}",
                    "source": "global_fallback",
                }
            return signals

        market_data = self.smart_bus.get("market_data", name) or {}
        price_data = self.smart_bus.get("price_data", name) or {}
        indicators = self.smart_bus.get("technical_indicators", name) or {}

        aligned_instruments: List[str] = []
        instrument_scores: Dict[str, float] = {}

        for inst in self.instruments:
            inst_market = extract_instrument_data(market_data, inst)
            inst_price = extract_instrument_data(price_data, inst)
            inst_indicators = extract_instrument_data(indicators, inst)

            alignment_score = self._calculate_instrument_alignment(
                inst, action, inst_market, inst_price, inst_indicators
            )
            instrument_scores[inst] = alignment_score

            if alignment_score > 0.3:
                aligned_instruments.append(inst)

        # If nothing aligns, still trade best instrument but with reduced confidence
        if not aligned_instruments and instrument_scores:
            best_inst = max(instrument_scores, key=lambda k: instrument_scores.get(k, 0.0))
            aligned_instruments = [best_inst]
            base_confidence *= 0.5

        for inst in self.instruments:
            if inst in aligned_instruments:
                inst_confidence = base_confidence * instrument_scores.get(inst, 0.5)

                if action in ("BUY", "LONG"):
                    intensity = round(inst_confidence, 4)
                elif action in ("SELL", "SHORT"):
                    intensity = round(-inst_confidence, 4)
                else:
                    intensity = 0.0

                signals[inst] = {
                    "action": action,
                    "confidence": round(inst_confidence, 4),
                    "size_multiplier": round(inst_confidence, 4),
                    "intensity": intensity,
                    "instrument": inst,
                    "alignment_score": instrument_scores.get(inst, 0.5),
                    "reason": f"Aligned with global {action}",
                    "source": "global_with_alignment",
                }
            else:
                signals[inst] = {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "size_multiplier": 0.0,
                    "intensity": 0.0,
                    "instrument": inst,
                    "alignment_score": instrument_scores.get(inst, 0.0),
                    "reason": (
                        f"Not aligned with global {action} "
                        f"(score={instrument_scores.get(inst, 0.0):.2f})"
                    ),
                    "source": "global_with_alignment",
                }

        self.logger.debug(
            f"[ARBITER] Instrument signals: aligned={aligned_instruments}, "
            f"scores={instrument_scores}"
        )
        return signals

    def _calculate_instrument_alignment(
        self,
        instrument: str,
        action: str,
        market_data: Dict[str, Any],
        price_data: Dict[str, Any],
        indicators: Dict[str, Any],
    ) -> float:
        """
        Calculate how well an instrument aligns with the proposed global action.

        Heuristics (all bounded to avoid extreme influence):
        - Candle direction (close vs. open) [high weight]
        - Momentum / ROC and RSI [medium weight]
        - Price position in local range [medium weight]
        - Volatility / ATR (penalizes very high volatility) [low weight]
        """
        try:
            is_buy = action.upper() == "BUY"
            scores: List[float] = []
            weights: List[float] = []

            # 1) Candle direction
            open_price = float(market_data.get("open", price_data.get("open", 0.0)) or 0.0)
            close_price = float(market_data.get("close", price_data.get("close", 0.0)) or 0.0)

            if open_price > 0.0 and close_price > 0.0:
                candle_direction = 1 if close_price > open_price else -1
                if is_buy:
                    candle_score = 0.8 if candle_direction > 0 else 0.3
                else:
                    candle_score = 0.8 if candle_direction < 0 else 0.3
                scores.append(candle_score)
                weights.append(2.0)

            # 2) Momentum + RSI
            momentum = float(indicators.get("momentum", indicators.get("roc", 0.0)) or 0.0)
            rsi = float(indicators.get("rsi", 50.0) or 50.0)

            # Normalize momentum into [-0.5, 0.5]
            norm_mom = max(-0.5, min(0.5, momentum / 10.0))

            if is_buy:
                momentum_score = 0.5 + norm_mom
                rsi_score = 1.0 if rsi < 70.0 else max(0.0, 1.0 - (rsi - 70.0) / 30.0)
            else:
                momentum_score = 0.5 - norm_mom
                rsi_score = 1.0 if rsi > 30.0 else max(0.0, rsi / 30.0)

            scores.extend([momentum_score, rsi_score])
            weights.extend([1.0, 1.0])

            # 3) Price position in local range
            high = float(price_data.get("high", 0.0) or 0.0)
            low = float(price_data.get("low", 0.0) or 0.0)
            close = float(price_data.get("close", price_data.get("last", 0.0)) or 0.0)

            if high > low and close > 0.0:
                price_position = (close - low) / (high - low)  # 0 = low, 1 = high
                if is_buy:
                    position_score = 1.0 - price_position * 0.5  # 1.0 → 0.5
                else:
                    position_score = 0.5 + price_position * 0.5  # 0.5 → 1.0
                scores.append(position_score)
                weights.append(0.8)

            # 4) Volatility / ATR penalty
            atr = float(indicators.get("atr", indicators.get("volatility", 0.0)) or 0.0)
            if atr > 0.0 and close > 0.0:
                atr_pct = atr / close
                if atr_pct > 0.02:
                    vol_score = max(0.3, 1.0 - (atr_pct - 0.02) / 0.03)
                else:
                    vol_score = 1.0
                scores.append(vol_score)
                weights.append(0.5)

            if scores and weights:
                total_weight = sum(weights[: len(scores)])
                if total_weight > 0:
                    weighted_sum = sum(
                        s * w for s, w in zip(scores, weights[: len(scores)])
                    )
                    return weighted_sum / total_weight
                return 0.5
            elif scores:
                return sum(scores) / len(scores)
            else:
                return 0.5

        except Exception as e:
            self.logger.warning(
                f"[ARBITER] Error calculating alignment for {instrument}: {e}"
            )
            return 0.5

    # ──────────────────────────────────────────────────────────────────────
    # TECHNICAL OVERRIDE
    # ──────────────────────────────────────────────────────────────────────

    def _check_technical_override(
        self, instrument: str, action: str
    ) -> Tuple[bool, str, float]:
        """
        Optional technical override by MomentumExpert + TrendExpert.

        If BOTH experts:
        - vote in the SAME direction (LONG/SHORT)
        - that direction is OPPOSITE to the proposed action
        - AND both confidence values >= 0.50

        then this method returns (True, new_action, avg_confidence).

        This is primarily useful early in RL training or in live mode
        when you want a strong technical safety belt.
        """
        try:
            name = self.__class__.__name__
            inst_normalized = normalize_instrument(instrument)

            momentum_votes = (
                self.smart_bus.get("MomentumExpert_per_instrument_votes", name) or {}
            )
            trend_votes = (
                self.smart_bus.get("TrendExpert_per_instrument_votes", name) or {}
            )

            momentum_vote = momentum_votes.get(inst_normalized, {})
            trend_vote = trend_votes.get(inst_normalized, {})

            momentum_action = str(momentum_vote.get("action", "flat")).upper()
            trend_action = str(trend_vote.get("action", "flat")).upper()
            momentum_conf = self._safe_float(momentum_vote.get("confidence"), 0.0)
            trend_conf = self._safe_float(trend_vote.get("confidence"), 0.0)

            proposed = action.upper()
            if proposed in ("BUY", "LONG"):
                proposed_direction = "LONG"
            elif proposed in ("SELL", "SHORT"):
                proposed_direction = "SHORT"
            else:
                # HOLD/FLAT → nothing to override
                return (False, action, 0.0)

            momentum_dir = (
                "LONG"
                if momentum_action == "LONG"
                else "SHORT"
                if momentum_action == "SHORT"
                else None
            )
            trend_dir = (
                "LONG"
                if trend_action == "LONG"
                else "SHORT"
                if trend_action == "SHORT"
                else None
            )

            if momentum_dir and trend_dir and momentum_dir == trend_dir:
                # Technical experts agree with each other
                if momentum_dir != proposed_direction:
                    min_override_conf = 0.50
                    if (
                        momentum_conf >= min_override_conf
                        and trend_conf >= min_override_conf
                    ):
                        avg_conf = (momentum_conf + trend_conf) / 2.0
                        new_action = "BUY" if momentum_dir == "LONG" else "SELL"

                        self.logger.warning(
                            f"[ARBITER] TECHNICAL OVERRIDE for {instrument}: "
                            f"{proposed_direction} → {momentum_dir} | "
                            f"Momentum={momentum_action}({momentum_conf:.2f}), "
                            f"Trend={trend_action}({trend_conf:.2f})"
                        )
                        return (True, new_action, avg_conf)
                    else:
                        self.logger.info(
                            f"[ARBITER] Technical disagreement noted but "
                            f"NOT overriding {instrument} {proposed_direction}: "
                            f"Momentum={momentum_action}({momentum_conf:.2f}), "
                            f"Trend={trend_action}({trend_conf:.2f}) "
                            f"(confidence too low)"
                        )

            return (False, action, 0.0)

        except Exception as e:
            self.logger.warning(f"[ARBITER] Error checking technical override: {e}")
            return (False, action, 0.0)

    # ──────────────────────────────────────────────────────────────────────
    # PER-INSTRUMENT GATE
    # ──────────────────────────────────────────────────────────────────────

    def _check_instrument_gate(
        self,
        instrument: str,
        action: str,
        confidence: float,
        consensus_score: float,
        collusion_score: float = 0.0,
        fragility: float = 0.5,
        market_regime: str = "UNKNOWN",
    ) -> Tuple[bool, str, float]:
        """
        Check if a per-instrument trade passes the gate.

        LIVE MODE:
        - Very strict, uses adaptive thresholds tuned for live behaviour
        - Requires both strong confidence and strong consensus

        TRAINING MODE:
        - More permissive, but still uses adaptive thresholds per instrument

        Returns:
            (gate_passed, final_action, final_confidence)
        """
        try:
            # HOLD / FLAT signals always "pass" (they are non-trades)
            if action.upper() in ("HOLD", "FLAT", "ABSTAIN"):
                return (True, "HOLD", 0.0)

            # Optional technical override
            if self.technical_override_enabled:
                should_override, override_action, override_conf = (
                    self._check_technical_override(instrument, action)
                )
                if should_override:
                    action = override_action
                    confidence = max(confidence * 0.8, override_conf)
                    self.logger.info(
                        f"[ARBITER] Using technical override: {instrument} → {action} "
                        f"(conf={confidence:.2f})"
                    )

            # Adaptive thresholds per instrument
            threshold_context = {
                "volatility": fragility,  # proxy
                "regime": market_regime,
                "recent_signals": len(self.gate_decisions),
            }

            min_confidence = float(
                get_instrument_threshold(instrument, "confidence", threshold_context)
            )
            min_consensus = float(
                get_instrument_threshold(instrument, "consensus", threshold_context)
            )

            self.logger.debug(
                f"[ARBITER] Adaptive thresholds for {instrument}: "
                f"conf={min_confidence:.3f}, consensus={min_consensus:.3f} "
                f"(regime={market_regime}, vol={threshold_context['volatility']:.4f})"
            )

            # LIVE mode: strict
            if is_live_mode():
                if confidence < min_confidence:
                    self.logger.warning(
                        f"[ARBITER] 🚫 Gate BLOCKED {instrument} {action} [LIVE]: "
                        f"confidence {confidence:.2f} < {min_confidence:.2f} (adaptive)"
                    )
                    if _threshold_manager:
                        _threshold_manager.record_signal(
                            instrument, confidence, consensus_score, passed=False
                        )
                    return (False, action, confidence)

                if consensus_score < min_consensus:
                    self.logger.warning(
                        f"[ARBITER] 🚫 Gate BLOCKED {instrument} {action} [LIVE]: "
                        f"consensus {consensus_score:.2f} < {min_consensus:.2f} (adaptive)"
                    )
                    if _threshold_manager:
                        _threshold_manager.record_signal(
                            instrument, confidence, consensus_score, passed=False
                        )
                    return (False, action, confidence)
            else:
                # TRAINING mode: adaptive but less strict
                if confidence < min_confidence:
                    self.logger.warning(
                        f"[ARBITER] Gate BLOCKED {instrument} {action} [TRAINING]: "
                        f"confidence {confidence:.2f} < {min_confidence:.2f} (adaptive)"
                    )
                    return (False, action, confidence)

                if consensus_score < min_consensus:
                    self.logger.warning(
                        f"[ARBITER] Gate BLOCKED {instrument} {action} [TRAINING]: "
                        f"consensus {consensus_score:.2f} < {min_consensus:.2f} (adaptive)"
                    )
                    return (False, action, confidence)

            # Extreme collusion check (true data-sanity guardrail)
            if collusion_score > 0.98:
                self.logger.warning(
                    f"[ARBITER] Gate BLOCKED {instrument} {action} "
                    f"[MODE={get_voting_mode()}]: extreme collusion {collusion_score:.2f} > 0.98"
                )
                if _threshold_manager and is_live_mode():
                    _threshold_manager.record_signal(
                        instrument, confidence, consensus_score, passed=False
                    )
                return (False, action, confidence)

            # Feed back successful signals to DynamicThresholdManager
            if _threshold_manager and is_live_mode():
                _threshold_manager.record_signal(
                    instrument, confidence, consensus_score, passed=True
                )

            self.logger.info(
                f"[ARBITER] ✅ Gate PASSED {instrument} {action} "
                f"[MODE={get_voting_mode()}]: "
                f"conf={confidence:.2f}, consensus={consensus_score:.2f}, "
                f"regime={market_regime}"
            )
            return (True, action, confidence)

        except Exception as e:
            self.logger.warning(
                f"[ARBITER] Error in instrument gate check for {instrument}: {e}"
            )
            # On error, fail-safe: block trade
            return (False, action, confidence)

    # ──────────────────────────────────────────────────────────────────────
    # STATS / THESIS / BUS UPDATE
    # ──────────────────────────────────────────────────────────────────────

    def _summarize_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Human-readable summary of main voting inputs."""
        return {
            "committee_action": data.get("committee_decision", {}).get(
                "action", "unknown"
            ),
            "committee_confidence": data.get("committee_confidence", 0.0),
            "consensus_score": data.get("consensus_score", 0.0),
            "collusion_score": data.get("collusion_score", 0.0),
            "fragility": data.get("fragility", 0.0),
            "regime": data.get("market_regime", "unknown"),
        }

    def _update_stats(self, gate_result: Dict[str, Any]) -> None:
        """Update aggregate statistics and gate history."""
        self.arbiter_stats["total_decisions"] += 1

        if gate_result["passed"]:
            self.arbiter_stats["gate_passes"] += 1
        else:
            self.arbiter_stats["gate_failures"] += 1

        self.arbiter_stats["gate_pass_rate"] = self._gate_passes / max(
            1, self._gate_attempts
        )

        self.gate_decisions.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "passed": gate_result["passed"],
                "score": gate_result["weighted_score"],
            }
        )

    def _generate_thesis(
        self, decision: Dict[str, Any], gate_result: Dict[str, Any]
    ) -> str:
        """Generate short textual summary for logs / dashboard."""
        action = decision.get("action", "unknown")
        conf = float(decision.get("confidence", 0.0))
        passed = gate_result["passed"]
        score = float(gate_result.get("weighted_score", 0.0))

        status = "PASSED" if passed else "BLOCKED"

        return (
            f"ARBITER: {status} | action={action} | "
            f"confidence={conf:.1%} | gate_score={score:.2f} | "
            f"pass_rate={self.arbiter_stats['gate_pass_rate']:.1%}"
        )

    async def _update_bus(
        self,
        decision: Dict[str, Any],
        gate_result: Dict[str, Any],
        instrument_signals: Dict[str, Dict[str, Any]],
        thesis: str,
    ) -> None:
        """
        Push all relevant outputs back to SmartInfoBus.

        Downstream modules (PositionManager, RL agent shell, dashboards)
        rely on these keys.
        """
        try:
            name = self.__class__.__name__

            self.smart_bus.set(
                "final_decision",
                decision,
                module=name,
                thesis=thesis,
                confidence=decision.get("confidence", 0.0),
            )
            self.smart_bus.set(
                "gate_passed",
                gate_result["passed"],
                module=name,
                thesis=f"Gate {'passed' if gate_result['passed'] else 'blocked'}",
            )
            self.smart_bus.set(
                "trading_signal",
                decision,
                module=name,
                thesis="Trading signal",
            )
            self.smart_bus.set(
                "instrument_signals",
                instrument_signals,
                module=name,
                thesis=f"Signals for {len(instrument_signals)} instruments",
            )
            self.smart_bus.set(
                "arbiter_analysis",
                {"gate_result": gate_result, "stats": dict(self.arbiter_stats)},
                module=name,
                thesis="Arbiter analysis",
            )
        except Exception as e:
            self.logger.warning(f"[ARBITER] Bus update failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # ERROR OUTPUT
    # ──────────────────────────────────────────────────────────────────────

    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error payload, fail-safe to ABSTAIN."""
        thesis = f"Arbiter error: {error}"
        return {
            "final_decision": {
                "action": "abstain",
                "reason": f"error: {error}",
                "confidence": 0.0,
            },
            "gate_passed": False,
            "gate_result": {"passed": False, "error": error},
            "trading_signal": {"action": "abstain"},
            "instrument_signals": {},
            "arbiter_analysis": {"error": error},
            "arbiter_statistics": dict(self.arbiter_stats),
            "voting_quality": dict(self.voting_quality),
            "decision_id": None,
            "arbiter_decision_id": None,
            # Contract-expected keys / aliases
            "trade_vote": {"action": "abstain", "confidence": 0.0},
            "gate_decision": {"passed": False, "error": error},
            "arbiter_thesis": thesis,
            "decision_confidence": 0.0,
            "decision_rationale": f"error: {error}",
            "arbiter_recommendations": {
                "action": "abstain",
                "gate": False,
                "signals": {},
            },
            "member_weights": {},
            "_thesis": thesis,
        }
