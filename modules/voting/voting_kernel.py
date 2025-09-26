"""
Unified Voting Kernel v1.0
Deterministic orchestration of all voting modules with single debug surface
"""

from __future__ import annotations

import asyncio
import time
import datetime as dt
from typing import Dict, Any, List, Optional
from modules.contracts import module_args

# Core framework
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker

# Voting modules
from modules.voting.voting_wrappers import EnhancedVotingExpertBase
from modules.voting.consensus_detector import ConsensusDetector
from modules.voting.collusion_auditor import CollusionAuditor
from modules.voting.time_horizon_aligner import TimeHorizonAligner
from modules.voting.alternative_reality_sampler import AlternativeRealitySampler
from modules.voting.strategy_arbiter import StrategyArbiter


@module(**module_args(
    "VotingKernel",
    description="Unified voting orchestration with deterministic pipeline and schema validation",
    error_handling=True,
    hot_reload=True,
    timeout_ms=5000,
))
class VotingKernel(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    UNIFIED VOTING KERNEL v1.0

    Orchestrates all 6 voting modules in deterministic order:
    1. Committee (VotingWrappers) -> proposal_vectors, member_confidences, decision_id
    2. ConsensusDetector -> consensus_score, components
    3. CollusionAuditor -> collusion_score, suspicious_pairs
    4. TimeHorizonAligner -> aligned_weights, horizon_alignment
    5. AlternativeRealitySampler -> uncertainty level, fragility
    6. StrategyArbiter -> signals, final gate

    Emits single voting/decision_bundle per tick with full schema validation.
    """

    def _initialize(self) -> None:
        # Core services
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="VotingKernel",
            log_path="logs/voting/voting_kernel.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("VotingKernel", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor(auto_start=False)

        # Configuration
        self.enable_committee = bool(self.config.get("enable_committee", True))
        self.enable_consensus = bool(self.config.get("enable_consensus", True))
        self.enable_collusion = bool(self.config.get("enable_collusion", True))
        self.enable_horizon = bool(self.config.get("enable_horizon", True))
        self.enable_sampling = bool(self.config.get("enable_sampling", True))
        self.enable_arbiter = bool(self.config.get("enable_arbiter", True))
        self.schema_validation = bool(self.config.get("schema_validation", True))
        self.debug_timeline = bool(self.config.get("debug_timeline", True))

        # Pipeline state
        self.pipeline_stats = {
            "total_ticks": 0,
            "successful_ticks": 0,
            "failed_ticks": 0,
            "avg_processing_time_ms": 0.0,
            "module_success_rates": {},
            "last_error": None,
            "session_start": dt.datetime.now().isoformat(),
        }

        # Initialize submodules (would normally be injected or auto-discovered)
        self.submodules = {}
        if self.enable_committee:
            # Note: VotingWrappers/Committee would be initialized here
            pass  # For now, assume external committee module

        # Error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 3
        self.is_disabled = False

        self.logger.info(
            format_operator_message(
                icon="[VOTING]",
                message="VotingKernel v1.0 initialized",
                committee=self.enable_committee,
                consensus=self.enable_consensus,
                collusion=self.enable_collusion,
                horizon=self.enable_horizon,
                sampling=self.enable_sampling,
                arbiter=self.enable_arbiter,
                schema_validation=self.schema_validation,
            )
        )

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Deterministic voting pipeline with full orchestration
        """
        start_time = time.time()
        pipeline_results = {}
        timeline = []

        try:
            if self.is_disabled:
                return self._generate_disabled_response()

            # Initialize decision tracking
            decision_id = f"{dt.datetime.now().isoformat()}#{self.pipeline_stats['total_ticks']}"
            tick_ts = dt.datetime.now().isoformat()

            # Seed coordination keys on the bus
            self.smart_bus.set("decision_id", decision_id, module="VotingKernel",
                             thesis="Decision coordination ID")
            self.smart_bus.set("tick_ts", tick_ts, module="VotingKernel",
                             thesis="Tick timestamp for synchronization")

            self.pipeline_stats["total_ticks"] += 1

            # STAGE 1: Committee (proposal generation)
            if self.enable_committee:
                stage_start = time.time()
                try:
                    # Committee would normally be called here
                    # For now, check if committee data exists on bus
                    committee_data = await self._get_committee_data()
                    pipeline_results["committee"] = committee_data
                    timeline.append({
                        "stage": "committee",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "success" if committee_data.get("members") else "no_data",
                    })
                except Exception as e:
                    timeline.append({
                        "stage": "committee",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "error",
                        "error": str(e)
                    })

            # STAGE 2: Consensus Detection
            if self.enable_consensus:
                stage_start = time.time()
                try:
                    consensus_data = await self._get_consensus_data()
                    pipeline_results["consensus"] = consensus_data
                    timeline.append({
                        "stage": "consensus",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "success",
                    })
                except Exception as e:
                    timeline.append({
                        "stage": "consensus",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "error",
                        "error": str(e)
                    })

            # STAGE 3: Collusion Auditing
            if self.enable_collusion:
                stage_start = time.time()
                try:
                    collusion_data = await self._get_collusion_data()
                    pipeline_results["collusion"] = collusion_data
                    timeline.append({
                        "stage": "collusion",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "success",
                    })
                except Exception as e:
                    timeline.append({
                        "stage": "collusion",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "error",
                        "error": str(e)
                    })

            # STAGE 4: Time Horizon Alignment
            if self.enable_horizon:
                stage_start = time.time()
                try:
                    horizon_data = await self._get_horizon_data()
                    pipeline_results["weights"] = horizon_data
                    timeline.append({
                        "stage": "horizon",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "success",
                    })
                except Exception as e:
                    timeline.append({
                        "stage": "horizon",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "error",
                        "error": str(e)
                    })

            # STAGE 5: Alternative Reality Sampling
            if self.enable_sampling:
                stage_start = time.time()
                try:
                    uncertainty_data = await self._get_uncertainty_data()
                    pipeline_results["uncertainty"] = uncertainty_data
                    timeline.append({
                        "stage": "sampling",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "success",
                    })
                except Exception as e:
                    timeline.append({
                        "stage": "sampling",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "error",
                        "error": str(e)
                    })

            # STAGE 6: Strategy Arbiter (final gating)
            if self.enable_arbiter:
                stage_start = time.time()
                try:
                    arbiter_data = await self._get_arbiter_data()
                    pipeline_results["trade_vote_v2"] = arbiter_data.get("trade_vote_v2", {})
                    pipeline_results["signals"] = arbiter_data.get("signals", {})
                    timeline.append({
                        "stage": "arbiter",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "success",
                    })
                except Exception as e:
                    timeline.append({
                        "stage": "arbiter",
                        "duration_ms": (time.time() - stage_start) * 1000,
                        "status": "error",
                        "error": str(e)
                    })

            # Assemble final bundle
            bundle = await self._assemble_decision_bundle(
                decision_id, tick_ts, pipeline_results
            )

            # Schema validation
            if self.schema_validation:
                validation_errors = self._validate_schema_v1(bundle)
                if validation_errors:
                    self.logger.warning(f"Schema validation errors: {validation_errors}")

            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_pipeline_stats(processing_time, timeline, True)

            # Build contract surfaces
            decision_coordination = {
                "decision_id": decision_id,
                "tick_ts": tick_ts,
                "stages": len(timeline),
                "status": "ok",
            }
            voting_consensus = dict(pipeline_results.get("consensus", {}))
            consensus_summary = {
                "score": voting_consensus.get("score"),
                "components": voting_consensus.get("components", {}),
            }
            voting_metrics = {
                "processing_time_ms": processing_time,
                "avg_processing_time_ms": self.pipeline_stats.get("avg_processing_time_ms", 0.0),
                "total_ticks": self.pipeline_stats.get("total_ticks", 0),
                "successful_ticks": self.pipeline_stats.get("successful_ticks", 0),
                "failed_ticks": self.pipeline_stats.get("failed_ticks", 0),
            }
            trade_vote_v2 = dict(pipeline_results.get("trade_vote_v2", {}))
            # Derive fragility (uncertainty.fragility) for contract + bus exposure
            fragility_value = 0.0
            try:
                if isinstance(pipeline_results.get("uncertainty"), dict):
                    fragility_value = float(pipeline_results["uncertainty"].get("fragility", 0.0) or 0.0)
            except Exception:
                fragility_value = 0.0
            # Derive committee_members (contract surface) from committee stage or bus
            committee_section = pipeline_results.get("committee", {}) or {}
            committee_members = []
            try:
                if isinstance(committee_section, dict) and isinstance(committee_section.get("members"), list):
                    committee_members = list(committee_section.get("members", []))
                if not committee_members:
                    # Fallback direct bus read if committee stage absent
                    bus_members = self.smart_bus.get("committee_members", "VotingKernel")
                    if isinstance(bus_members, list):
                        committee_members = list(bus_members)
            except Exception:
                committee_members = []

            # Publish bundle and key surfaces to SmartInfoBus for downstreams
            self.smart_bus.set(
                "voting/decision_bundle",
                bundle,
                module="VotingKernel",
                thesis=f"Unified voting decision {decision_id} ({processing_time:.1f}ms)"
            )

            try:
                self.smart_bus.set("decision_coordination", decision_coordination, module="VotingKernel", thesis="Decision coordination")
                # Avoid writing 'voting_consensus' (owner: ConsensusDetector). Publish a committee-local consensus instead.
                self.smart_bus.set("committee_consensus", voting_consensus, module="VotingKernel", thesis="Committee consensus snapshot")
                self.smart_bus.set("consensus_summary", consensus_summary, module="VotingKernel", thesis="Consensus summary")
                self.smart_bus.set("voting_metrics", voting_metrics, module="VotingKernel", thesis="Voting metrics")
                self.smart_bus.set("trade_vote_v2", trade_vote_v2, module="VotingKernel", thesis="Final vote bundle (v2)")
                # If available, publish signals under VotingKernel namespace (contract owner)
                try:
                    signals_payload = dict(pipeline_results.get("signals", {}) or {})
                    self.smart_bus.set(
                        "signals",
                        signals_payload,
                        module="VotingKernel",
                        thesis=f"Voting signals ({len(signals_payload) if isinstance(signals_payload, dict) else 0})",
                    )
                except Exception:
                    pass
                # Publish standalone fragility surface (mirrors uncertainty.fragility)
                try:
                    self.smart_bus.set("fragility", fragility_value, module="VotingKernel", thesis=f"Voting fragility {fragility_value:.3f}")
                    # Record performance metric for fragility exposure (success assumed if bus write succeeds)
                    try:
                        self.performance_tracker.record_metric(
                            self.__class__.__name__,
                            'fragility_publish',
                            0.0,
                            True,
                            error=f"fragility={fragility_value:.4f}"  # reuse error/context slot for value annotation
                        )
                    except Exception:
                        pass
                    # Publish committee_members (even if empty) for contract consumers
                    try:
                        self.smart_bus.set("committee_members", committee_members, module="VotingKernel", thesis=f"Committee members ({len(committee_members)})")
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass

            if self.debug_timeline:
                self.smart_bus.set(
                    "voting/kernel_timeline",
                    {"decision_id": decision_id, "timeline": timeline},
                    module="VotingKernel",
                    thesis=f"Pipeline timeline for {decision_id}"
                )

            # Derive proposal_vectors (contract surface) from committee stage or bus
            try:
                committee_proposals = []
                if isinstance(committee_section.get("proposal_vectors"), list):
                    committee_proposals = list(committee_section.get("proposal_vectors", []))
                if not committee_proposals:
                    bus_proposals = self.smart_bus.get("proposal_vectors", "VotingKernel")
                    if isinstance(bus_proposals, list):
                        committee_proposals = list(bus_proposals)
            except Exception:
                committee_proposals = []

            return {
                "decision_bundle": bundle,
                "decision_coordination": decision_coordination,
                "voting_consensus": voting_consensus,
                "consensus_summary": consensus_summary,
                "voting_metrics": voting_metrics,
                "trade_vote_v2": trade_vote_v2,
                # Contract key: expose signals (may be empty dict)
                "signals": pipeline_results.get("signals", {}) if isinstance(pipeline_results.get("signals"), dict) else {},
                # Contract key: expose top-level fragility (mirror uncertainty.fragility when available)
                "fragility": fragility_value,
                # Contract key: expose committee_members (even if empty)
                "committee_members": committee_members,
                # Contract key: expose proposal_vectors (even if empty)
                "proposal_vectors": committee_proposals,
                "pipeline_timeline": timeline,
                "processing_time_ms": processing_time,
                "pipeline_stats": dict(self.pipeline_stats),
                "_thesis": f"VotingKernel: {len(timeline)} stages, {processing_time:.1f}ms",
            }

        except Exception as e:
            return await self._handle_pipeline_error(e, start_time, timeline)

    async def _get_committee_data(self) -> Dict[str, Any]:
        """Get committee data from bus"""
        g = self.smart_bus.get
        return {
            "members": g("committee_members", "VotingKernel") or [],
            "proposal_vectors": g("proposal_vectors", "VotingKernel") or [],
            "member_confidences": g("member_confidences_ordered", "VotingKernel") or [],
            "committee_consensus": g("committee_consensus", "VotingKernel") or {},
            "raw": {
                "votes": g("committee_votes", "VotingKernel") or [],
                "meta": {"n_members": 0}
            }
        }

    async def _get_consensus_data(self) -> Dict[str, Any]:
        """Get consensus data from bus"""
        g = self.smart_bus.get
        return {
            "score": g("consensus_score", "VotingKernel"),
            "components": g("consensus_components", "VotingKernel") or {}
        }

    async def _get_collusion_data(self) -> Dict[str, Any]:
        """Get collusion data from bus"""
        g = self.smart_bus.get
        return {
            "score": g("collusion_score", "VotingKernel"),
            "suspicious_pairs": g("suspicious_pairs", "VotingKernel") or [],
            "pair_penalties": []  # TODO: implement in CollusionAuditor
        }

    async def _get_horizon_data(self) -> Dict[str, Any]:
        """Get time horizon alignment data from bus"""
        g = self.smart_bus.get
        return {
            "raw": g("voting_weights", "VotingKernel") or [],
            "aligned": g("aligned_weights", "VotingKernel") or []
        }

    async def _get_uncertainty_data(self) -> Dict[str, Any]:
        """Get uncertainty/sampling data from bus"""
        g = self.smart_bus.get
        return {
            "level": g("sampling_uncertainty", "VotingKernel") or 0.0,
            "fragility": g("fragility", "VotingKernel") or 0.0,
            "n_alts": g("effective_samples", "VotingKernel") or 0
        }

    async def _get_arbiter_data(self) -> Dict[str, Any]:
        """Get strategy arbiter data from bus"""
        g = self.smart_bus.get
        return {
            "trade_vote_v2": g("trade_vote_v2", "VotingKernel") or {},
            "signals": g("signals", "VotingKernel") or {}
        }

    async def _assemble_decision_bundle(
        self, decision_id: str, tick_ts: str, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assemble the final voting decision bundle"""
        return {
            "decision_id": decision_id,
            "tick_ts": tick_ts,
            "committee": results.get("committee", {}),
            "consensus": results.get("consensus", {"score": None}),
            "collusion": results.get("collusion", {"score": None}),
            "weights": results.get("weights", {"raw": [], "aligned": []}),
            "uncertainty": results.get("uncertainty", {"level": 0.0, "fragility": 0.0, "n_alts": 0}),
            "trade_vote_v2": results.get("trade_vote_v2", {}),
            "signals": results.get("signals", {}),
            "_schema_version": "v1",
        }

    def _validate_schema_v1(self, bundle: Dict[str, Any]) -> List[str]:
        """Validate voting schema v1 compliance"""
        errors = []

        required_fields = ["decision_id", "tick_ts", "committee", "consensus",
                          "collusion", "weights", "uncertainty", "trade_vote_v2",
                          "signals", "_schema_version"]

        for field in required_fields:
            if field not in bundle:
                errors.append(f"Missing required field: {field}")

        if bundle.get("_schema_version") != "v1":
            errors.append(f"Invalid schema version: {bundle.get('_schema_version')}")

        # Validate committee structure
        committee = bundle.get("committee", {})
        if not isinstance(committee.get("members"), list):
            errors.append("committee.members must be a list")
        if not isinstance(committee.get("proposal_vectors"), list):
            errors.append("committee.proposal_vectors must be a list")
        if not isinstance(committee.get("member_confidences"), list):
            errors.append("committee.member_confidences must be a list")

        return errors

    def _update_pipeline_stats(self, processing_time: float, timeline: List[Dict], success: bool):
        """Update pipeline statistics"""
        if success:
            self.pipeline_stats["successful_ticks"] += 1
        else:
            self.pipeline_stats["failed_ticks"] += 1

        # Update average processing time
        total = self.pipeline_stats["successful_ticks"] + self.pipeline_stats["failed_ticks"]
        old_avg = self.pipeline_stats["avg_processing_time_ms"]
        self.pipeline_stats["avg_processing_time_ms"] = (old_avg * (total - 1) + processing_time) / total

        # Update module success rates
        for stage in timeline:
            stage_name = stage["stage"]
            if stage_name not in self.pipeline_stats["module_success_rates"]:
                self.pipeline_stats["module_success_rates"][stage_name] = {"success": 0, "total": 0}

            self.pipeline_stats["module_success_rates"][stage_name]["total"] += 1
            if stage["status"] == "success":
                self.pipeline_stats["module_success_rates"][stage_name]["success"] += 1

    async def _handle_pipeline_error(self, error: Exception, start_time: float, timeline: List[Dict]) -> Dict[str, Any]:
        """Handle pipeline errors"""
        self.error_count += 1
        processing_time = (time.time() - start_time) * 1000

        ctx = self.error_pinpointer.analyze_error(error, "VotingKernel")
        self.pipeline_stats["last_error"] = str(ctx)

        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(
                format_operator_message(
                    icon="[ALERT]",
                    message="VotingKernel disabled due to repeated errors",
                    error_count=self.error_count,
                    threshold=self.circuit_breaker_threshold,
                )
            )

        self._update_pipeline_stats(processing_time, timeline, False)

        return {
            "decision_bundle": {"error": str(ctx), "_schema_version": "v1"},
            "pipeline_timeline": timeline,
            "processing_time_ms": processing_time,
            "pipeline_stats": dict(self.pipeline_stats),
            "fragility": 0.0,
            "committee_members": [],
            "_thesis": f"VotingKernel error: {ctx}",
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when kernel is disabled"""
        return {
            "decision_bundle": {"status": "disabled", "_schema_version": "v1"},
            "pipeline_timeline": [],
            "processing_time_ms": 0.0,
            "pipeline_stats": dict(self.pipeline_stats),
            "fragility": 0.0,
            "committee_members": [],
            "_thesis": "VotingKernel disabled via circuit breaker",
        }

    def get_state(self) -> Dict[str, Any]:
        """Get kernel state for persistence"""
        base = super().get_state()
        base["custom_state"] = {
            "pipeline_stats": dict(self.pipeline_stats),
            "error_count": self.error_count,
            "is_disabled": self.is_disabled,
        }
        return base

    def set_state(self, state: Dict[str, Any]):
        """Set kernel state from persistence"""
        super().set_state(state)
        cs = state.get("custom_state", {})
        if cs.get("pipeline_stats"):
            self.pipeline_stats.update(cs["pipeline_stats"])
        self.error_count = int(cs.get("error_count", 0))
        self.is_disabled = bool(cs.get("is_disabled", False))

