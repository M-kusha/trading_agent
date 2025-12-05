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
    VOTING_DEFAULTS,
)


# Action translation map: semantic actions -> trading actions
# Long/bullish actions map to BUY, Short/bearish to SELL, neutral/gate to HOLD
ACTION_TRANSLATION = {
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
    """Translate semantic voting action to trading action (BUY/SELL/HOLD)."""
    if not action:
        return "HOLD"
    action_lower = str(action).lower().strip()
    # Direct match
    if action_lower in ACTION_TRANSLATION:
        return ACTION_TRANSLATION[action_lower]
    # Pattern matching for complex actions
    if "long" in action_lower or "buy" in action_lower or "bullish" in action_lower:
        return "BUY"
    if "short" in action_lower or "sell" in action_lower or "bearish" in action_lower:
        return "SELL"
    # Default to HOLD
    return "HOLD"


@module(**module_args("SlimVotingKernel"))
class SlimVotingKernel(VotingModuleBase):
    """
    Slim Voting Kernel - Lightweight Pipeline Orchestrator

    Coordinates all voting stages in sequence:
    COMMITTEE → CONSENSUS → COLLUSION → HORIZON → UNCERTAINTY → ARBITER

    Each stage is a separate module that handles one responsibility.
    The kernel aggregates results from the bus (stages run via ModuleOrchestrator).
    """

    MODULE_NAME = "SlimVotingKernel"

    def _module_specific_init(self) -> None:
        """Initialize kernel-specific state."""
        # Pipeline state
        self._current_stage: PipelineStage = PipelineStage.IDLE
        self._pipeline_run_count: int = 0
        self._last_decision_id: Optional[str] = None

        # Hysteresis for final action to prevent BUY/SELL flip-flopping
        self._last_final_action: str = "HOLD"
        self._action_hold_count: int = 0
        self._min_hold_ticks: int = 5  # Minimum ticks to hold direction before reversal

        # Stage skip configuration (currently advisory; wiring is in stages themselves)
        self._skip_stages: Set[PipelineStage] = set()
        config = getattr(self, "config", None) or {}
        if config:
            skip_list = config.get("skip_stages", [])
            for stage_name in skip_list:
                try:
                    self._skip_stages.add(PipelineStage[stage_name.upper()])
                except KeyError:
                    self.log_warning(f"Unknown stage to skip: {stage_name}")

        # Timing stats
        self._stage_timings: Dict[str, List[float]] = {
            stage.name: [] for stage in PipelineStage
        }

        self.log_info("SlimVotingKernel initialized")

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    async def run_pipeline(
        self,
        instruments: Optional[List[str]] = None,
        force_refresh: bool = False,  # kept for interface; stages handle freshness
    ) -> Dict[str, Any]:
        """
        Run the voting pipeline by aggregating results from the bus.

        The individual stage modules (CommitteeCoordinator, ConsensusAnalyzer, etc.)
        are executed by the main ModuleOrchestrator. This kernel reads their outputs
        from the SmartInfoBus (via the voting bus) and produces a unified decision.

        Args:
            instruments: List of instruments to process (default: all active)
            force_refresh: Reserved for future use

        Returns:
            Pipeline result with final decisions and thesis
        """
        pipeline_start = datetime.now()
        self._pipeline_run_count += 1

        # Generate decision ID
        decision_id = self._generate_decision_id()
        self._last_decision_id = decision_id

        self.log_info(
            f"Pipeline run #{self._pipeline_run_count} started: {decision_id}"
        )

        # Get instruments to process
        if instruments is None:
            instruments = self._get_active_instruments()

        if not instruments:
            self.log_warning("No instruments to process")
            return self._make_empty_result(decision_id, "No instruments")

        try:
            # ===============================================================
            # Read stage outputs from voting bus
            # (Stages are executed by the main orchestrator)
            # ===============================================================

            # Committee results
            stage_start = datetime.now()
            self._current_stage = PipelineStage.COMMITTEE
            committee_votes = self.bus_get("committee_votes", default=[]) or []
            committee_decision = (
                self.bus_get("committee_decision", default={}) or {}
            )
            committee_confidence = float(
                self.bus_get("committee_confidence", default=0.5) or 0.5
            )
            raw_proposals = self.bus_get("raw_proposals", default={}) or {}
            self._record_stage_timing(PipelineStage.COMMITTEE.name, stage_start)

            if not committee_votes and not raw_proposals:
                self.log_warning(
                    "No votes available from committee - defaulting to ABSTAIN"
                )
                try:
                    raw_keys = list(raw_proposals.keys()) if isinstance(raw_proposals, dict) else raw_proposals
                    self.log_debug(
                        f"[KERNEL][DATA] committee_votes_len={len(committee_votes)}, "
                        f"raw_proposals_keys={raw_keys}"
                    )
                except Exception:
                    pass
                return self._make_abstain_result(
                    decision_id, "No votes from committee"
                )

            # Consensus results
            stage_start = datetime.now()
            self._current_stage = PipelineStage.CONSENSUS
            consensus_score = float(
                self.bus_get("consensus_score", default=0.5) or 0.5
            )
            consensus_result = self.bus_get("consensus_result", default=None) or {
                "consensus_score": consensus_score,
                "consensus_exists": consensus_score >= 0.6,
            }
            agreement_score = float(
                self.bus_get("agreement_score", default=consensus_score)
                or consensus_score
            )
            self._record_stage_timing(PipelineStage.CONSENSUS.name, stage_start)

            # Collusion results
            stage_start = datetime.now()
            self._current_stage = PipelineStage.COLLUSION
            collusion_score = float(
                self.bus_get("collusion_score", default=0.0) or 0.0
            )
            collusion_detected = bool(
                self.bus_get("collusion_detected", default=False) or False
            )
            self._record_stage_timing(PipelineStage.COLLUSION.name, stage_start)

            if collusion_detected:
                self.log_warning(
                    f"Collusion detected! Score: {collusion_score:.2%}"
                )

            # Horizon results
            stage_start = datetime.now()
            self._current_stage = PipelineStage.HORIZON
            horizon_alignment = (
                self.bus_get("horizon_alignment", default={}) or {}
            )
            aligned_weights = self.bus_get("aligned_weights", default=[]) or []
            # (aligned_weights kept for future horizon-aware weighting)
            self._record_stage_timing(PipelineStage.HORIZON.name, stage_start)

            # Uncertainty results
            stage_start = datetime.now()
            self._current_stage = PipelineStage.UNCERTAINTY
            fragility = float(self.bus_get("fragility", default=0.5) or 0.5)
            uncertainty_result = self.bus_get(
                "uncertainty_result", default=None
            ) or {
                "fragility_score": fragility,
                "robustness_score": 1.0 - fragility,
            }
            self._record_stage_timing(
                PipelineStage.UNCERTAINTY.name, stage_start
            )

            # Final arbiter results (if already computed)
            stage_start = datetime.now()
            self._current_stage = PipelineStage.ARBITER
            final_decision = self.bus_get("final_decision", default=None) or committee_decision
            gate_decision = (
                self.bus_get("gate_decision", default={"passed": True})
                or {"passed": True}
            )
            instrument_signals = (
                self.bus_get("instrument_signals", default={}) or {}
            )
            self._record_stage_timing(PipelineStage.ARBITER.name, stage_start)

            # ===============================================================
            # Build unified result
            # ===============================================================
            self._current_stage = PipelineStage.IDLE

            # Calculate total pipeline time
            pipeline_time_ms = (
                datetime.now() - pipeline_start
            ).total_seconds() * 1000.0

            # Determine final action (raw semantic action from voting)
            raw_action = final_decision.get(
                "action", committee_decision.get("action", "abstain")
            )
            confidence = final_decision.get(
                "confidence", committee_confidence
            )

            # Translate semantic action to trading action (BUY/SELL/HOLD)
            action = translate_action(raw_action)

            # Apply hysteresis to prevent BUY/SELL flip-flopping
            action = self._apply_action_hysteresis(
                action, confidence, consensus_score
            )

            # Debug: log raw vs translated action
            if raw_action and action and raw_action.lower() != action.lower():
                self.log_info(f"Action translated: '{raw_action}' → '{action}'")

            # Build thesis
            thesis = self._build_pipeline_thesis_from_bus(
                committee_votes=committee_votes,
                consensus_score=consensus_score,
                collusion_detected=collusion_detected,
                fragility=fragility,
                action=action,
                confidence=confidence,
            )

            # Build complete result
            result = {
                "decision_id": decision_id,
                "pipeline_run": self._pipeline_run_count,
                "instruments": instruments,
                "final_result": {
                    "action": action,
                    "confidence": confidence,
                    "gate_passed": gate_decision.get("passed", True),
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
                "kernel_consensus_score": consensus_score,
                "pipeline_time_ms": pipeline_time_ms,
                "thesis": thesis,
                "fragility": fragility,
            }

            # Publish to bus
            self._publish_result(result)

            self.log_info(
                f"Pipeline #{self._pipeline_run_count} complete: "
                f"{action} (conf={confidence:.1%}) in {pipeline_time_ms:.0f}ms"
            )

            return result

        except Exception as e:
            self._current_stage = PipelineStage.IDLE
            error_msg = str(e)
            self.log_error(f"Pipeline error: {error_msg}")
            return self._make_error_result(decision_id, error_msg)

    def _build_pipeline_thesis_from_bus(
        self,
        committee_votes: List[Dict],
        consensus_score: float,
        collusion_detected: bool,
        fragility: float,
        action: str,
        confidence: float,
    ) -> str:
        """Build thesis from bus data."""
        parts = [
            f"Collected {len(committee_votes)} votes.",
            f"Consensus: {consensus_score:.1%}.",
        ]

        if collusion_detected:
            parts.append("WARNING: Collusion detected.")

        if fragility > 0.7:
            parts.append(f"High fragility ({fragility:.1%}).")

        parts.append(
            f"Decision: {action.upper()} with {confidence:.1%} confidence."
        )

        return " ".join(parts)

    def _apply_action_hysteresis(
        self, new_action: str, confidence: float, consensus_score: float
    ) -> str:
        """
        Apply hysteresis to prevent BUY/SELL flip-flopping.

        Rules:
        1. If confidence < 20% and consensus < 50%, stay with last action
        2. Must hold current direction for min_hold_ticks before reversing
        3. Reversals (BUY→SELL or SELL→BUY) require higher confidence than staying
        4. Going to HOLD is always allowed (safety)
        """
        last_action = getattr(self, "_last_final_action", "HOLD")
        hold_count = getattr(self, "_action_hold_count", 0)
        min_hold = getattr(self, "_min_hold_ticks", 5)

        # Low confidence/consensus: stay with last action (avoid noise)
        if (
            confidence < 0.20
            and consensus_score < 0.50
            and last_action != "HOLD"
        ):
            self._action_hold_count = hold_count + 1
            return last_action

        # Check for direction reversal (BUY↔SELL)
        is_reversal = (last_action == "BUY" and new_action == "SELL") or (
            last_action == "SELL" and new_action == "BUY"
        )

        # Reversals need higher confidence OR enough hold time
        if is_reversal:
            if hold_count < min_hold and confidence < 0.40:
                # Too soon and too weak - stay with current direction
                self._action_hold_count = hold_count + 1
                self.log_info(
                    f"Hysteresis: Blocked reversal {last_action}→{new_action} "
                    f"(hold={hold_count}/{min_hold}, conf={confidence:.1%})"
                )
                return last_action

        # Action is allowed - update state
        if new_action != last_action:
            self._action_hold_count = 0
            self._last_final_action = new_action
        else:
            self._action_hold_count = hold_count + 1

        return new_action

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_decision_id(self) -> str:
        """Generate unique decision coordination ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"SLIM_{timestamp}_{self._pipeline_run_count}"

    def _get_active_instruments(self) -> List[str]:
        """Get list of active instruments from bus or config."""
        # Try bus first
        instruments = self.bus_get(
            VotingBusKeys.ACTIVE_INSTRUMENTS, default=None
        )
        if instruments:
            return instruments

        # Fall back to config
        if self.config:
            return self.config.get("instruments", ["EURUSD", "XAUUSD"])

        return ["EURUSD", "XAUUSD"]

    def _make_empty_result(self, decision_id: str, reason: str) -> Dict[str, Any]:
        """Create empty result when no instruments available."""
        return {
            "decision_id": decision_id,
            "pipeline_run": self._pipeline_run_count,
            "instruments": [],
            "final_result": {
                "action": VotingAction.ABSTAIN.value,
                "confidence": 0.0,
                "thesis": reason,
            },
            "thesis": reason,
        }

    def _make_abstain_result(self, decision_id: str, reason: str) -> Dict[str, Any]:
        """Create abstain result when votes unavailable."""
        return {
            "decision_id": decision_id,
            "pipeline_run": self._pipeline_run_count,
            "final_result": {
                "action": VotingAction.ABSTAIN.value,
                "confidence": 0.0,
                "thesis": reason,
            },
            "thesis": reason,
        }

    def _make_error_result(self, decision_id: str, error: str) -> Dict[str, Any]:
        """Create error result when pipeline fails."""
        return {
            "decision_id": decision_id,
            "pipeline_run": self._pipeline_run_count,
            "error": error,
            "final_result": {
                "action": VotingAction.ABSTAIN.value,
                "confidence": 0.0,
                "thesis": f"Pipeline error: {error}",
            },
            "thesis": f"Pipeline failed: {error}",
        }

    def _record_stage_timing(self, stage_name: str, start_time: datetime) -> None:
        """Record timing for a pipeline stage."""
        elapsed_ms = (
            datetime.now() - start_time
        ).total_seconds() * 1000.0

        if stage_name in self._stage_timings:
            timings = self._stage_timings[stage_name]
            timings.append(elapsed_ms)

            # Keep only last 100 timings
            if len(timings) > 100:
                self._stage_timings[stage_name] = timings[-100:]

    def _publish_result(self, result: Dict[str, Any]) -> None:
        """Publish pipeline result to bus."""
        self.bus_set(
            VotingBusKeys.KERNEL_DECISION_ID,
            result["decision_id"],
            thesis="Pipeline decision ID",
        )

        self.bus_set(
            VotingBusKeys.PIPELINE_RESULT,
            result,
            thesis=result.get("thesis", "Pipeline result"),
        )

        # Convenience aliases for downstream consumers
        self.bus_set(
            "kernel_consensus_score",
            result.get("consensus_result", {}).get("consensus_score"),
            thesis="Kernel consensus score",
        )
        self.bus_set(
            "kernel_instrument_signals",
            result.get("instrument_signals", {}),
            thesis="Kernel instrument signals",
        )
        self.bus_set(
            "arbiter_instrument_signals",
            result.get("instrument_signals", {}),
            thesis="Arbiter instrument signals (alias)",
        )

        # Publish per-instrument signals if available
        final_result = result.get("final_result", {})
        instrument_signals = final_result.get("instrument_signals", {})

        for instrument, signal in instrument_signals.items():
            self.bus_set(
                f"voting_signal_{instrument}",
                signal,
                thesis=f"Voting signal for {instrument}",
            )

    # =========================================================================
    # Status and Metrics
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current kernel status."""
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
        """Reset pipeline statistics."""
        self._pipeline_run_count = 0
        self._stage_timings = {stage.name: [] for stage in PipelineStage}
        self.log_info("Pipeline statistics reset")

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Process voting pipeline (ModuleOrchestrator interface).

        Args:
            **inputs: Input data from orchestrator

        Returns:
            Dict with voting results
        """
        # Extract parameters from inputs
        instruments = inputs.get("instruments", None)
        force_refresh = inputs.get("force_refresh", False)

        # Run pipeline
        result = await self.run_pipeline(
            instruments=instruments,
            force_refresh=force_refresh,
        )

        # Format for orchestrator
        final = result.get("final_result", {})
        consensus_result = result.get("consensus_result", None)
        consensus_score = 0.0
        if consensus_result and hasattr(consensus_result, "consensus_score"):
            consensus_score = consensus_result.consensus_score
        elif isinstance(consensus_result, dict):
            consensus_score = consensus_result.get("consensus_score", 0.0)

        tick_ts = datetime.now().isoformat()
        decision_id = result.get("decision_id", "")
        status = self.get_status()

        return {
            "voting_action": final.get(
                "action", VotingAction.ABSTAIN.value
            ),
            "voting_confidence": final.get("confidence", 0.0),
            "kernel_decision_id": decision_id,
            "consensus_score": consensus_score,
            "pipeline_run_count": self._pipeline_run_count,
            "stage_timings": status.get("stage_timings", {}),
            # Contract-expected keys
            "kernel_decision": final,
            "trade_vote_v2": {
                "action": final.get("action", VotingAction.ABSTAIN.value),
                "size": final.get("confidence", 0.0),
                "confidence": final.get("confidence", 0.0),
                "consensus_score": consensus_score,
                "decision_id": decision_id,
                "timestamp": tick_ts,
            },
            "decision_bundle": result,
            "voting_consensus": (
                consensus_result
                if isinstance(consensus_result, dict)
                else {"score": consensus_score}
            ),
            "consensus_summary": {
                "consensus_score": consensus_score,
                "action": final.get("action", "abstain"),
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
            "_thesis": f"Voting pipeline completed with action {final.get('action', 'ABSTAIN')}",
        }
