"""
Slim Voting Kernel
==================
Lightweight pipeline orchestrator that coordinates the voting stages.
This replaces the monolithic voting_kernel.py with a clean, modular design.

The kernel:
1. Collects votes from all registered experts (via CommitteeCoordinator)
2. Analyzes consensus (via ConsensusAnalyzer)
3. Detects collusion patterns (via CollusionDetector)
4. Applies horizon alignment (via HorizonAligner)
5. Samples uncertainty (via UncertaintySampler)
6. Makes final decision (via FinalArbiter)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase
from modules.voting.core.constants import (
    PipelineStage,
    VotingAction,
    VotingBusKeys,
)


# Action translation map: semantic actions -> trading actions
# Long/bullish actions map to BUY, Short/bearish to SELL, neutral/gate to HOLD
ACTION_TRANSLATION: Dict[str, str] = {
    # Long/bullish actions
    "long": "BUY",
    "buy": "BUY",
    "long_risk_assets": "BUY",
    "trend_following": "BUY",
    "breakout_long": "BUY",
    "momentum_long": "BUY",
    "risk_on": "BUY",
    "bullish": "BUY",
    "long_bias": "BUY",
    "seasonal_long_bias": "BUY",
    "breakout": "BUY",
    "increase_risk": "BUY",
    "aggressive": "BUY",  # Risk module bullish signals
    "trend_bullish": "BUY",  # TrendExpert bullish signal
    # Short/bearish actions
    "short": "SELL",
    "sell": "SELL",
    "short_risk_assets": "SELL",
    "safe_haven_rotation": "SELL",
    "volatility_hedging": "SELL",
    "breakout_short": "SELL",
    "momentum_short": "SELL",
    "risk_off": "SELL",
    "bearish": "SELL",
    "short_bias": "SELL",
    "seasonal_short_bias": "SELL",
    "mean_reversion": "SELL",
    "reduce_risk": "SELL",
    "emergency_stop": "SELL",  # Risk module bearish signals
    "trend_bearish": "SELL",  # TrendExpert bearish signal
    # Neutral/gate actions
    "abstain": "HOLD",
    "hold": "HOLD",
    "neutral": "HOLD",
    "flat": "HOLD",
    "proceed": "HOLD",
    "wait": "HOLD",
    "no_action": "HOLD",
    "caution": "HOLD",
    "halt": "HOLD",
    "continue": "HOLD",
    "confirm": "HOLD",
    "seasonal_neutral": "HOLD",
    "theme_neutral": "HOLD",
    "momentum_neutral": "HOLD",
    "trend_neutral": "HOLD",
    "session_avoid": "HOLD",
    "session_optimal": "HOLD",
    "high_impact_caution": "HOLD",
}


def translate_action(action: str) -> str:
    """
    Translate semantic voting action to trading action (BUY/SELL/HOLD).

    If the action is unknown, we infer from name patterns and fall back to HOLD.
    """
    if not action:
        return "HOLD"

    action_lower = str(action).lower().strip()

    # Direct mapping
    if action_lower in ACTION_TRANSLATION:
        return ACTION_TRANSLATION[action_lower]

    # Pattern matching for complex actions
    if "long" in action_lower or "buy" in action_lower or "bullish" in action_lower:
        return "BUY"
    if "short" in action_lower or "sell" in action_lower or "bearish" in action_lower:
        return "SELL"

    # Default to HOLD (safety)
    return "HOLD"


@module(**module_args("SlimVotingKernel"))
class SlimVotingKernel(VotingModuleBase):
    """
    Slim Voting Kernel - Lightweight Pipeline Orchestrator

    Coordinates all voting stages in sequence:
    COMMITTEE → CONSENSUS → COLLUSION → HORIZON → UNCERTAINTY → ARBITER

    Each stage is a separate module that handles one responsibility.
    The kernel aggregates results from the SmartInfoBus (stages are run
    by the main ModuleOrchestrator) and produces a unified decision.
    """

    MODULE_NAME = "SlimVotingKernel"

    # =======================================================================
    # Initialization
    # =======================================================================

    def _module_specific_init(self) -> None:
        """Initialize kernel-specific state and metrics."""
        # Pipeline state
        self._current_stage: PipelineStage = PipelineStage.IDLE
        self._pipeline_run_count: int = 0
        self._last_decision_id: Optional[str] = None

        # Hysteresis state to prevent BUY/SELL flip-flopping
        self._last_final_action: str = "HOLD"
        self._action_hold_count: int = 0
        self._min_hold_ticks: int = int(
            getattr(self.config, "get", lambda *_: 5)("min_hold_ticks", 5)
            if getattr(self, "config", None)
            else 5
        )

        # Stage skip configuration (soft skip: treat missing outputs as defaults)
        self._skip_stages: Set[PipelineStage] = set()
        config = getattr(self, "config", None) or {}
        if config:
            skip_list = config.get("skip_stages", [])
            for stage_name in skip_list:
                try:
                    self._skip_stages.add(PipelineStage[stage_name.upper()])
                except KeyError:
                    self.log_warning(f"[KERNEL] Unknown stage to skip: {stage_name}")

        # Timing stats (ms) for each stage
        self._stage_timings: Dict[str, List[float]] = {
            stage.name: [] for stage in PipelineStage
        }

        self.log_info("SlimVotingKernel initialized")

    # =======================================================================
    # Main Pipeline
    # =======================================================================

    async def run_pipeline(
        self,
        instruments: Optional[List[str]] = None,
        force_refresh: bool = False,  # reserved: freshness handled by stages themselves
    ) -> Dict[str, Any]:
        """
        Run the voting pipeline by aggregating results from the bus.

        The individual stage modules (CommitteeCoordinator, ConsensusAnalyzer, etc.)
        are executed by the main ModuleOrchestrator. This kernel reads their outputs
        from the SmartInfoBus and produces a unified decision.

        Args:
            instruments: List of instruments to process (default: all active)
            force_refresh: Reserved for future use

        Returns:
            Pipeline result with final decisions and thesis.
        """
        pipeline_start = datetime.now()
        self._pipeline_run_count += 1

        # Generate decision ID for coordination across modules
        decision_id = self._generate_decision_id()
        self._last_decision_id = decision_id

        self.log_info(
            f"[KERNEL] Pipeline run #{self._pipeline_run_count} started: {decision_id}"
        )

        # Determine instruments to process
        if instruments is None:
            instruments = self._get_active_instruments()

        if not instruments:
            reason = "No instruments available"
            self.log_warning(f"[KERNEL] {reason}")
            result = self._make_empty_result(decision_id, reason)
            # Even empty runs should publish pipeline_result so dashboards don't see stale state.
            self._publish_result(result)
            return result

        try:
            # ===============================================================
            # 1) COMMITTEE STAGE – global committee decision & votes
            # ===============================================================
            stage_start = datetime.now()
            self._current_stage = PipelineStage.COMMITTEE

            if PipelineStage.COMMITTEE in self._skip_stages:
                self.log_info("[KERNEL] Skipping COMMITTEE stage by config")
                committee_votes: List[Dict[str, Any]] = []
                committee_decision: Dict[str, Any] = {}
                committee_confidence: float = 0.0
                raw_proposals: Dict[str, Any] = {}
            else:
                committee_votes = (
                    self.bus_get(VotingBusKeys.COMMITTEE_VOTES, default=None)
                    or self.bus_get("committee_votes", default=[])
                    or []
                )
                committee_decision = (
                    self.bus_get(VotingBusKeys.COMMITTEE_DECISION, default=None)
                    or self.bus_get("committee_decision", default={})
                    or {}
                )
                committee_confidence = float(
                    self.bus_get(
                        VotingBusKeys.COMMITTEE_CONFIDENCE, default=None
                    )
                    or self.bus_get("committee_confidence", default=0.5)
                    or 0.5
                )
                raw_proposals = self.bus_get("raw_proposals", default={}) or {}

            self._record_stage_timing(PipelineStage.COMMITTEE.name, stage_start)

            # Warmup status: avoid noisy warnings during warmup
            warmup_status = self.bus_get("warmup_status", default={}) or {}
            is_warmup = (
                not warmup_status.get("complete", False) if warmup_status else True
            )

            if not committee_votes and not raw_proposals:
                if not is_warmup:
                    self.log_warning(
                        "[KERNEL] No votes available from committee - defaulting to ABSTAIN"
                    )
                    try:
                        raw_keys = (
                            list(raw_proposals.keys())
                            if isinstance(raw_proposals, dict)
                            else raw_proposals
                        )
                        self.log_debug(
                            f"[KERNEL][DATA] committee_votes_len={len(committee_votes)}, "
                            f"raw_proposals_keys={raw_keys}"
                        )
                    except Exception:
                        pass
                else:
                    warmup_pct = warmup_status.get("progress_pct", 0)
                    self.log_debug(
                        f"[KERNEL][WARMUP] Committee warming up ({warmup_pct:.0f}% complete)"
                    )

                result = self._make_abstain_result(
                    decision_id, "No votes from committee"
                )
                self._publish_result(result)
                self._current_stage = PipelineStage.IDLE
                return result

            # ===============================================================
            # 2) CONSENSUS STAGE – how aligned are the experts?
            # ===============================================================
            stage_start = datetime.now()
            self._current_stage = PipelineStage.CONSENSUS

            if PipelineStage.CONSENSUS in self._skip_stages:
                self.log_info("[KERNEL] Skipping CONSENSUS stage by config")
                consensus_score = 0.5
                consensus_result: Dict[str, Any] = {
                    "consensus_score": consensus_score,
                    "consensus_exists": False,
                    "reason": "consensus_stage_skipped",
                }
                agreement_score = consensus_score
            else:
                # Try dedicated consensus score; fall back to committee consensus_strength.
                consensus_score_raw = self.bus_get("consensus_score", default=None)
                if consensus_score_raw is None:
                    consensus_score_raw = self.bus_get(
                        "agreement_score", default=None
                    )

                committee_consensus = (
                    self.bus_get(
                        VotingBusKeys.COMMITTEE_CONSENSUS, default=None
                    )
                    or self.bus_get("committee_consensus", default=None)
                    or {}
                )
                if consensus_score_raw is None and isinstance(
                    committee_consensus, dict
                ):
                    consensus_score_raw = committee_consensus.get(
                        "consensus_strength", 0.5
                    )

                consensus_score = float(
                    consensus_score_raw if consensus_score_raw is not None else 0.5
                )

                consensus_result = (
                    self.bus_get("consensus_result", default=None) or {}
                )
                if not isinstance(consensus_result, dict) or not consensus_result:
                    consensus_result = {
                        "consensus_score": consensus_score,
                        "consensus_exists": consensus_score >= 0.60,
                    }

                agreement_score = float(
                    self.bus_get("agreement_score", default=consensus_score)
                    or consensus_score
                )

            self._record_stage_timing(PipelineStage.CONSENSUS.name, stage_start)

            # ===============================================================
            # 3) COLLUSION STAGE – detect unhealthy expert clustering
            # ===============================================================
            stage_start = datetime.now()
            self._current_stage = PipelineStage.COLLUSION

            if PipelineStage.COLLUSION in self._skip_stages:
                self.log_info("[KERNEL] Skipping COLLUSION stage by config")
                collusion_score = 0.0
                collusion_detected = False
            else:
                collusion_score = float(
                    self.bus_get("collusion_score", default=0.0) or 0.0
                )
                collusion_detected = bool(
                    self.bus_get("collusion_detected", default=False) or False
                )

            self._record_stage_timing(PipelineStage.COLLUSION.name, stage_start)

            if collusion_detected:
                self.log_warning(
                    f"[KERNEL] Collusion detected! Score: {collusion_score:.2%}"
                )

            # ===============================================================
            # 4) HORIZON STAGE – horizon alignment / time-consistency
            # ===============================================================
            stage_start = datetime.now()
            self._current_stage = PipelineStage.HORIZON

            if PipelineStage.HORIZON in self._skip_stages:
                self.log_info("[KERNEL] Skipping HORIZON stage by config")
                horizon_alignment: Dict[str, Any] = {}
                aligned_weights: List[float] = []
            else:
                horizon_alignment = (
                    self.bus_get("horizon_alignment", default={}) or {}
                )
                aligned_weights = (
                    self.bus_get("aligned_weights", default=[]) or []
                )

            self._record_stage_timing(PipelineStage.HORIZON.name, stage_start)

            # ===============================================================
            # 5) UNCERTAINTY STAGE – robustness & fragility
            # ===============================================================
            stage_start = datetime.now()
            self._current_stage = PipelineStage.UNCERTAINTY

            if PipelineStage.UNCERTAINTY in self._skip_stages:
                self.log_info("[KERNEL] Skipping UNCERTAINTY stage by config")
                fragility = 0.5
                uncertainty_result: Dict[str, Any] = {
                    "uncertainty_score": 0.5,
                    "fragility_score": fragility,
                    "robustness_score": 1.0 - fragility,
                    "reason": "uncertainty_stage_skipped",
                }
            else:
                fragility_raw = self.bus_get("fragility", default=None)
                if fragility_raw is None:
                    fragility_raw = self.bus_get("fragility_score", default=0.5)
                fragility = float(fragility_raw or 0.5)

                uncertainty_result = (
                    self.bus_get("uncertainty_result", default=None) or {}
                )
                if not isinstance(uncertainty_result, dict) or not uncertainty_result:
                    uncertainty_result = {
                        "uncertainty_score": 0.5,
                        "fragility_score": fragility,
                        "robustness_score": 1.0 - fragility,
                    }

            self._record_stage_timing(PipelineStage.UNCERTAINTY.name, stage_start)

            # ===============================================================
            # 6) ARBITER STAGE – final gated decision & instrument signals
            # ===============================================================
            stage_start = datetime.now()
            self._current_stage = PipelineStage.ARBITER

            if PipelineStage.ARBITER in self._skip_stages:
                self.log_info("[KERNEL] Skipping ARBITER stage by config")
                final_decision: Dict[str, Any] = committee_decision or {
                    "action": "abstain",
                    "confidence": committee_confidence,
                }
                gate_decision: Dict[str, Any] = {
                    "passed": True,
                    "reason": "arbiter_stage_skipped",
                }
                instrument_signals: Dict[str, Any] = {}
            else:
                final_decision = (
                    self.bus_get("final_decision", default=None)
                    or self.bus_get("trade_decision", default=None)
                    or committee_decision
                    or {"action": "abstain", "confidence": committee_confidence}
                )

                gate_decision = (
                    self.bus_get("gate_decision", default={"passed": True})
                    or {"passed": True}
                )

                instrument_signals = (
                    self.bus_get("instrument_signals", default=None)
                    or self.bus_get("arbiter_signals", default=None)
                    or {}
                )

            self._record_stage_timing(PipelineStage.ARBITER.name, stage_start)

            # Per-instrument committee decisions (for traceability)
            committee_decisions_by_instrument = (
                self.bus_get("committee_decisions_by_instrument", default={}) or {}
            )

            # ===============================================================
            # 7) Build unified result
            # ===============================================================
            self._current_stage = PipelineStage.IDLE

            pipeline_time_ms = (
                datetime.now() - pipeline_start
            ).total_seconds() * 1000.0

            # Raw semantic action from final arbiter or committee
            raw_action = final_decision.get(
                "action", committee_decision.get("action", "abstain")
            )
            confidence = float(
                final_decision.get("confidence", committee_confidence)
            )

            # Translate semantic action to trading action (BUY/SELL/HOLD)
            action = translate_action(raw_action)

            # Apply hysteresis to reduce flip-flopping between BUY/SELL
            action = self._apply_action_hysteresis(
                new_action=action,
                confidence=confidence,
                consensus_score=consensus_score,
            )

            # If gate/guardrail failed, keep action info but mark the gate result
            gate_passed = bool(gate_decision.get("passed", True))

            # Debug: log raw vs translated action
            if raw_action and action and str(raw_action).lower() != str(action).lower():
                self.log_info(
                    f"[KERNEL] Action translated: '{raw_action}' → '{action}'"
                )

            # Build human thesis
            thesis = self._build_pipeline_thesis_from_bus(
                committee_votes=committee_votes,
                consensus_score=consensus_score,
                collusion_detected=collusion_detected,
                collusion_score=collusion_score,
                fragility=fragility,
                action=action,
                confidence=confidence,
            )

            # Aggregate result
            result: Dict[str, Any] = {
                "decision_id": decision_id,
                "pipeline_run": self._pipeline_run_count,
                "instruments": instruments,
                "final_result": {
                    "action": action,
                    "confidence": confidence,
                    "gate_passed": gate_passed,
                    "gate_decision": gate_decision,
                    "thesis": thesis,
                    "instrument_signals": instrument_signals,
                },
                "committee_summary": {
                    "votes": len(committee_votes),
                    "decision": committee_decision,
                    "confidence": committee_confidence,
                },
                "consensus_result": consensus_result,
                "collusion_result": {
                    "collusion_detected": collusion_detected,
                    "collusion_score": collusion_score,
                },
                "uncertainty_result": uncertainty_result,
                "horizon_alignment": horizon_alignment,
                "instrument_signals": instrument_signals,
                "arbiter_signals": instrument_signals,  # alias for compatibility
                "kernel_consensus_score": consensus_score,
                "agreement_score": agreement_score,
                "pipeline_time_ms": pipeline_time_ms,
                "thesis": thesis,
                "fragility": fragility,
                "committee_decisions_by_instrument": committee_decisions_by_instrument,
            }

            # Publish to bus
            self._publish_result(result)

            self.log_info(
                f"[KERNEL] Pipeline #{self._pipeline_run_count} complete: "
                f"{action} (conf={confidence:.1%}, gate_passed={gate_passed}) "
                f"in {pipeline_time_ms:.0f}ms"
            )

            return result

        except Exception as e:
            self._current_stage = PipelineStage.IDLE
            error_msg = str(e)
            self.log_error(f"[KERNEL] Pipeline error: {error_msg}")
            result = self._make_error_result(decision_id, error_msg)
            self._publish_result(result)
            return result

    # =======================================================================
    # Internal helpers
    # =======================================================================

    def _build_pipeline_thesis_from_bus(
        self,
        committee_votes: List[Dict[str, Any]],
        consensus_score: float,
        collusion_detected: bool,
        collusion_score: float,
        fragility: float,
        action: str,
        confidence: float,
    ) -> str:
        """
        Build a compact, human-readable thesis string summarizing the pipeline.
        """
        parts: List[str] = [
            f"Collected {len(committee_votes)} votes.",
            f"Consensus: {consensus_score:.1%}.",
        ]

        if collusion_detected:
            parts.append(
                f"WARNING: Collusion detected (score={collusion_score:.1%})."
            )

        if fragility > 0.7:
            parts.append(f"High fragility ({fragility:.1%}).")
        elif fragility > 0.4:
            parts.append(f"Moderate fragility ({fragility:.1%}).")
        else:
            parts.append(f"Low fragility ({fragility:.1%}).")

        parts.append(
            f"Decision: {action.upper()} with {confidence:.1%} confidence."
        )

        return " ".join(parts)

    def _apply_action_hysteresis(
        self,
        new_action: str,
        confidence: float,
        consensus_score: float,
    ) -> str:
        """
        Apply hysteresis to prevent BUY/SELL flip-flopping.

        Rules:
        1. If confidence < 20% and consensus < 50%, stay with last action (if not HOLD).
        2. Must hold current direction for min_hold_ticks before reversing.
        3. Reversals (BUY↔SELL) require either enough hold time OR >= 40% confidence.
        4. Going to HOLD is always allowed (safety).
        """
        last_action = getattr(self, "_last_final_action", "HOLD")
        hold_count = getattr(self, "_action_hold_count", 0)
        min_hold = getattr(self, "_min_hold_ticks", 5)

        # Low confidence/consensus: stay with last action to avoid noise
        if (
            confidence < 0.20
            and consensus_score < 0.50
            and last_action != "HOLD"
        ):
            self._action_hold_count = hold_count + 1
            return last_action

        # Direction reversal (BUY ↔ SELL)
        is_reversal = (last_action == "BUY" and new_action == "SELL") or (
            last_action == "SELL" and new_action == "BUY"
        )

        if is_reversal:
            # Too soon and too weak → block reversal
            if hold_count < min_hold and confidence < 0.40:
                self._action_hold_count = hold_count + 1
                self.log_info(
                    f"[KERNEL][HYSTERESIS] Blocked reversal {last_action}→{new_action} "
                    f"(hold={hold_count}/{min_hold}, conf={confidence:.1%})"
                )
                return last_action

        # Action is allowed – update hysteresis state
        if new_action != last_action:
            self._action_hold_count = 0
            self._last_final_action = new_action
        else:
            self._action_hold_count = hold_count + 1

        return new_action

    # =======================================================================
    # Helper Methods
    # =======================================================================

    def _generate_decision_id(self) -> str:
        """Generate unique decision coordination ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"SLIM_{timestamp}_{self._pipeline_run_count}"

    def _get_active_instruments(self) -> List[str]:
        """
        Get list of active instruments from bus or config.

        Priority:
        1) VotingBusKeys.ACTIVE_INSTRUMENTS on SmartInfoBus
        2) Kernel config: config['instruments']
        3) Fallback: ['EURUSD', 'XAUUSD']
        """
        instruments = self.bus_get(
            VotingBusKeys.ACTIVE_INSTRUMENTS, default=None
        )
        if instruments:
            return instruments

        if self.config:
            cfg_instruments = self.config.get("instruments")
            if cfg_instruments:
                return cfg_instruments

        return ["EURUSD", "XAUUSD"]

    def _make_empty_result(self, decision_id: str, reason: str) -> Dict[str, Any]:
        """Create empty result when no instruments are available."""
        return {
            "decision_id": decision_id,
            "pipeline_run": self._pipeline_run_count,
            "instruments": [],
            "final_result": {
                "action": VotingAction.ABSTAIN.value,
                "confidence": 0.0,
                "thesis": reason,
                "instrument_signals": {},
            },
            "instrument_signals": {},
            "thesis": reason,
            "fragility": 0.5,
        }

    def _make_abstain_result(self, decision_id: str, reason: str) -> Dict[str, Any]:
        """Create abstain result when votes are unavailable or unusable."""
        return {
            "decision_id": decision_id,
            "pipeline_run": self._pipeline_run_count,
            "final_result": {
                "action": VotingAction.ABSTAIN.value,
                "confidence": 0.0,
                "thesis": reason,
                "instrument_signals": {},
            },
            "instrument_signals": {},
            "thesis": reason,
            "fragility": 0.5,
        }

    def _make_error_result(self, decision_id: str, error: str) -> Dict[str, Any]:
        """Create error result when pipeline fails."""
        thesis = f"Pipeline error: {error}"
        return {
            "decision_id": decision_id,
            "pipeline_run": self._pipeline_run_count,
            "error": error,
            "final_result": {
                "action": VotingAction.ABSTAIN.value,
                "confidence": 0.0,
                "thesis": thesis,
                "instrument_signals": {},
            },
            "instrument_signals": {},
            "thesis": f"Pipeline failed: {error}",
            "fragility": 0.5,
        }

    def _record_stage_timing(self, stage_name: str, start_time: datetime) -> None:
        """Record timing for a pipeline stage in milliseconds."""
        elapsed_ms = (
            datetime.now() - start_time
        ).total_seconds() * 1000.0

        timings = self._stage_timings.get(stage_name)
        if timings is None:
            self._stage_timings[stage_name] = [elapsed_ms]
            return

        timings.append(elapsed_ms)
        # Keep only last 100 timings per stage to bound memory
        if len(timings) > 100:
            self._stage_timings[stage_name] = timings[-100:]

    def _publish_result(self, result: Dict[str, Any]) -> None:
        """
        Publish pipeline result and key aliases to SmartInfoBus.

        This is the main bridge used by downstream modules and dashboards.
        """
        decision_id = result.get("decision_id")

        # Core coordination IDs
        self.bus_set(
            VotingBusKeys.KERNEL_DECISION_ID,
            decision_id,
            thesis="Pipeline decision ID",
        )

        # Full pipeline result bundle
        self.bus_set(
            VotingBusKeys.PIPELINE_RESULT,
            result,
            thesis=result.get("thesis", "Pipeline result"),
        )

        # Kernel-level consensus score
        consensus_score = result.get("kernel_consensus_score")
        if consensus_score is None:
            cr = result.get("consensus_result")
            if isinstance(cr, dict):
                consensus_score = cr.get("consensus_score", 0.0)
            else:
                consensus_score = 0.0

        self.bus_set(
            "kernel_consensus_score",
            consensus_score,
            thesis="Kernel consensus score",
        )

        # Instrument signals – both kernel and arbiter aliases
        instrument_signals = result.get("instrument_signals", {}) or {}
        self.bus_set(
            "kernel_instrument_signals",
            instrument_signals,
            thesis="Kernel instrument signals",
        )
        self.bus_set(
            "arbiter_instrument_signals",
            instrument_signals,
            thesis="Arbiter instrument signals (alias)",
        )

        # Publish per-instrument signals as individual keys for convenience
        final_result = result.get("final_result", {}) or {}
        per_instrument = final_result.get("instrument_signals", {}) or {}

        for instrument, signal in per_instrument.items():
            self.bus_set(
                f"voting_signal_{instrument}",
                signal,
                thesis=f"Voting signal for {instrument}",
            )

    # =======================================================================
    # Status and Metrics
    # =======================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current kernel status for monitoring/debugging."""
        return {
            "module": self.MODULE_NAME,
            "current_stage": self._current_stage.value,
            "pipeline_runs": self._pipeline_run_count,
            "last_decision_id": self._last_decision_id,
            "skip_stages": [s.name for s in self._skip_stages],
            "stage_timings": {
                stage: {
                    "count": len(timings),
                    "avg_ms": (sum(timings) / len(timings)) if timings else 0.0,
                    "max_ms": max(timings) if timings else 0.0,
                }
                for stage, timings in self._stage_timings.items()
                if timings
            },
        }

    def reset_stats(self) -> None:
        """Reset pipeline statistics and timing history."""
        self._pipeline_run_count = 0
        self._stage_timings = {stage.name: [] for stage in PipelineStage}
        self.log_info("[KERNEL] Pipeline statistics reset")

    # =======================================================================
    # Orchestrator Entry Point
    # =======================================================================

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """
        Process voting pipeline (ModuleOrchestrator interface).

        Args:
            **inputs: Input data from orchestrator

        Returns:
            Dict with voting results and contract-required keys.
        """
        # Extract parameters from orchestrator inputs
        instruments = inputs.get("instruments", None)
        force_refresh = inputs.get("force_refresh", False)

        # Run the full pipeline
        result = await self.run_pipeline(
            instruments=instruments,
            force_refresh=force_refresh,
        )

        final = result.get("final_result", {}) or {}
        consensus_result = result.get("consensus_result", None)

        # Normalize consensus_score
        consensus_score = 0.0
        if isinstance(consensus_result, dict):
            consensus_score = float(
                consensus_result.get("consensus_score", 0.0)
            )

        tick_ts = datetime.now().isoformat()
        decision_id = result.get("decision_id", "")
        status = self.get_status()

        # Contract fields
        voting_action = final.get("action", VotingAction.ABSTAIN.value)
        voting_confidence = float(final.get("confidence", 0.0))

        return {
            "voting_action": voting_action,
            "voting_confidence": voting_confidence,
            "kernel_decision_id": decision_id,
            "consensus_score": consensus_score,
            "pipeline_run_count": self._pipeline_run_count,
            "stage_timings": status.get("stage_timings", {}),
            # Contract-expected keys
            "kernel_decision": final,
            "trade_vote_v2": {
                "action": voting_action,
                "size": voting_confidence,
                "confidence": voting_confidence,
                "consensus_score": consensus_score,
                "decision_id": decision_id,
                "timestamp": tick_ts,
            },
            "decision_bundle": result,
            "voting_consensus": (
                consensus_result
                if isinstance(consensus_result, dict)
                else {"consensus_score": consensus_score}
            ),
            "consensus_summary": {
                "consensus_score": consensus_score,
                "action": voting_action,
            },
            "voting_metrics": status.get("stage_timings", {}),
            "decision_id": decision_id,
            "tick_ts": tick_ts,
            "kernel_tick_ts": tick_ts,
            "pipeline_status": self._current_stage.value,
            "pipeline_thesis": result.get("thesis", ""),
            "decision_coordination": {
                "decision_id": decision_id,
                "tick_ts": tick_ts,
            },
            "fragility": result.get("fragility", 0.5),
            # Contract-required aliases
            "pipeline_result": result,
            "kernel_consensus_score": consensus_score,
            "kernel_instrument_signals": result.get("instrument_signals", {}),
            "arbiter_instrument_signals": result.get("instrument_signals", {}),
            "_thesis": f"Voting pipeline completed with action {voting_action}",
        }
