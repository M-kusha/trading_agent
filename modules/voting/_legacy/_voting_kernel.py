"""
================================================================================
DEPRECATED - LEGACY MODULE
================================================================================
This module has been replaced by the unified voting system (v5.0).
See modules/voting/ for the new architecture.

This file is preserved for reference only. All code below is commented out.
DO NOT IMPORT THIS MODULE.

Deprecated: November 2025
================================================================================
"""

# DEPRECATED: The following code is commented out.
# If you need this functionality, use the new modules in:
#   - modules/voting/core/
#   - modules/voting/experts/
#   - modules/voting/stages/
#   - modules/voting/pipeline/

# """
# Unified Voting Kernel v2.0
# Deterministic orchestration with schema validation and decision_id coordination
# """

# from __future__ import annotations

# import asyncio
# import time
# import datetime as dt
# from typing import Dict, Any, List, Optional

# from modules.contracts import module_args

# # Core framework
# from modules.core.module_base import BaseModule, module
# from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
# from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
# from modules.utils.info_bus import InfoBusManager
# from modules.utils.audit_utils import RotatingLogger, format_operator_message
# from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
# from modules.monitoring.health_monitor import HealthMonitor
# from modules.monitoring.performance_tracker import PerformanceTracker

# # Voting modules
# from modules.voting.voting_wrappers import EnhancedVotingExpertBase
# from modules.voting.consensus_detector import ConsensusDetector
# from modules.voting.collusion_auditor import CollusionAuditor
# from modules.voting.time_horizon_aligner import TimeHorizonAligner
# from modules.voting.alternative_reality_sampler import AlternativeRealitySampler
# from modules.voting.strategy_arbiter import StrategyArbiter

# @module(**module_args(
#     "VotingKernel",
#     description="Unified voting orchestration v2.0 with schema validation and coordination fixes",
#     error_handling=True,
#     hot_reload=True,
#     timeout_ms=5000,
# ))
# class VotingKernel(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
#     """
#     UNIFIED VOTING KERNEL v2.0
    
#     FIXES APPLIED:
#     - Schema conflict resolution (deprecated voting_consensus, split into consensus_score + committee_consensus)
#     - decision_id coordination across all stages
#     - Namespaced bus keys (committee_*, consensus_*, collusion_*, etc.)
#     - Schema validation layer
#     - Enhanced error tracking
    
#     Pipeline stages:
#     1. Committee -> proposal_vectors, member_confidences, committee_consensus, decision_id
#     2. ConsensusDetector -> consensus_score, consensus_components
#     3. CollusionAuditor -> collusion_score, suspicious_pairs
#     4. TimeHorizonAligner -> aligned_weights, horizon_alignment
#     5. AlternativeRealitySampler -> sampling_uncertainty, fragility
#     6. StrategyArbiter -> instrument_signals, gate_decision
#     """

#     def _initialize(self) -> None:
#         # Core services
#         self.smart_bus = InfoBusManager.get_instance()
#         self.logger = RotatingLogger(
#             name="VotingKernel",
#             log_path="logs/voting/voting_kernel.log",
#             max_lines=5000,
#             operator_mode=True,
#             plain_english=True,
#         )
        
#         self.error_pinpointer = ErrorPinpointer()
#         self.error_handler = create_error_handler("VotingKernel", self.error_pinpointer)
#         self.english_explainer = EnglishExplainer()
#         self.system_utilities = SystemUtilities()
#         self.performance_tracker = PerformanceTracker()
#         self.health_monitor = HealthMonitor(auto_start=False)

#         # Configuration
#         self.enable_committee = bool(self.config.get("enable_committee", True))
#         self.enable_consensus = bool(self.config.get("enable_consensus", True))
#         self.enable_collusion = bool(self.config.get("enable_collusion", True))
#         self.enable_horizon = bool(self.config.get("enable_horizon", True))
#         self.enable_sampling = bool(self.config.get("enable_sampling", True))
#         self.enable_arbiter = bool(self.config.get("enable_arbiter", True))
#         self.schema_validation = bool(self.config.get("schema_validation", True))
#         self.debug_timeline = bool(self.config.get("debug_timeline", True))
#         # Increased staleness threshold to accommodate orchestration cycle (~5-6 seconds)
#         # Data from previous tick (N-1) is valid since pipeline modules run after VotingKernel
#         self.max_staleness_seconds = float(self.config.get("max_staleness_seconds", 15.0))
#         # Last known good results to avoid empty/stale-driven abstains
#         self._last_good_results: Dict[str, Dict[str, Any]] = {
#             "committee": {},
#             "consensus": {},
#             "collusion": {},
#             "horizon": {},
#             "uncertainty": {},
#             "arbiter": {},
#         }

#         # Pipeline state
#         self.pipeline_stats = {
#             "total_ticks": 0,
#             "successful_ticks": 0,
#             "failed_ticks": 0,
#             "avg_processing_time_ms": 0.0,
#             "module_success_rates": {},
#             "last_error": None,
#             "session_start": dt.datetime.now().isoformat(),
#             "schema_violations": 0,
#         }

#         # Error handling
#         self.error_count = 0
#         self.circuit_breaker_threshold = 5
#         self.is_disabled = False

#         self.logger.info(
#             format_operator_message(
#                 icon="[VOTING]",
#                 message="VotingKernel v2.0 initialized with coordination fixes",
#                 committee=self.enable_committee,
#                 consensus=self.enable_consensus,
#                 collusion=self.enable_collusion,
#                 horizon=self.enable_horizon,
#                 sampling=self.enable_sampling,
#                 arbiter=self.enable_arbiter,
#                 schema_validation=self.schema_validation,
#             )
#         )

#     def _check_freshness(self, current_decision_id: str, source_decision_id: Optional[str],
#                          source_tick_ts: Optional[str], stage: str) -> tuple[bool, str]:
#         """
#         Validate that upstream data is not stale.
#         Returns (is_fresh, reason_if_stale).
        
#         NOTE: We do NOT enforce decision_id matching because pipeline modules may run
#         in different orchestration stages. Committee/Collusion/Horizon modules run AFTER
#         VotingKernel in the stage order, so their decision_id is from the previous tick.
#         Timestamp-based staleness is sufficient to ensure data freshness.
#         """
#         # Skip decision_id matching - orchestration order means upstream data is from N-1 tick
#         # This is intentional: VotingKernel consumes the most recent data from each stage
        
#         if source_tick_ts:
#             try:
#                 ts = dt.datetime.fromisoformat(str(source_tick_ts))
#                 age = abs((dt.datetime.now(ts.tzinfo) - ts).total_seconds())
#                 if age > self.max_staleness_seconds:
#                     return False, f"{stage}:stale_ts:{age:.2f}s"
#             except Exception:
#                 # If timestamp cannot be parsed, don't block - allow data through
#                 pass
#         return True, ""

#     def _bus_get_first(self, key: str, modules: List[str], default: Any = None) -> Any:
#         """Return the first non-None value for key across candidate modules."""
#         g = self.smart_bus.get
#         for mod in modules:
#             try:
#                 val = g(key, mod, default=None)
#             except Exception:
#                 val = None
#             if val is not None:
#                 return val
#         return default

#     async def process(self, **inputs) -> Dict[str, Any]:
#         """
#         Deterministic voting pipeline with full coordination
#         """
#         start_time = time.time()
#         pipeline_results = {}
#         timeline = []

#         try:
#             if self.is_disabled:
#                 return self._generate_disabled_response()

#             # ============================================
#             # FIX 1: Generate decision_id at pipeline start
#             # ============================================
#             decision_id = f"{dt.datetime.now().isoformat()}#{self.pipeline_stats['total_ticks']}"
#             tick_ts = dt.datetime.now().isoformat()

#             # Seed coordination keys with namespacing
#             self.smart_bus.set("kernel_decision_id", decision_id, module="VotingKernel",
#                              thesis="Decision coordination ID for this tick")
#             self.smart_bus.set("kernel_tick_ts", tick_ts, module="VotingKernel",
#                              thesis="Tick timestamp for synchronization")
            
#             # NEW FIX: Publish canonical keys for zero-wiring discoverability (190 BUS MISS fix)
#             self.smart_bus.set("decision_id", decision_id, module="VotingKernel",
#                              thesis="Canonical decision ID for current voting cycle - allows modules to discover coordination ID")
#             self.smart_bus.set("tick_ts", tick_ts, module="VotingKernel",
#                              thesis="Canonical tick timestamp - allows modules to discover current tick time")
            
#             self.pipeline_stats["total_ticks"] += 1

#             # ============================================
#             # STAGE 1: Committee (proposal generation)
#             # ============================================
#             if self.enable_committee:
#                 stage_start = time.time()
#                 try:
#                     committee_data = await self._get_committee_data(decision_id)
#                     stale_reason = committee_data.get("_stale_reason")
#                     stage_status = "success" if committee_data.get("members") else "no_data"
#                     if stale_reason:
#                         stage_status = "stale"
#                         committee_data = {}
#                     pipeline_results["committee"] = committee_data
#                     timeline.append({
#                         "stage": "committee",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": stage_status,
#                         "stale_reason": stale_reason,
#                         "decision_id": decision_id,
#                     })
#                 except Exception as e:
#                     timeline.append({
#                         "stage": "committee",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": "error",
#                         "error": str(e),
#                         "decision_id": decision_id,
#                     })

#             # ============================================
#             # STAGE 2: Consensus Detection
#             # ============================================
#             if self.enable_consensus:
#                 stage_start = time.time()
#                 try:
#                     consensus_data = await self._get_consensus_data(decision_id)
#                     stale_reason = consensus_data.get("_stale_reason")
#                     stage_status = "success" if consensus_data.get("score") is not None else "no_data"
#                     if stale_reason:
#                         stage_status = "stale"
#                         consensus_data = {}
#                     pipeline_results["consensus"] = consensus_data
#                     timeline.append({
#                         "stage": "consensus",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": stage_status,
#                         "stale_reason": stale_reason,
#                         "decision_id": decision_id,
#                     })
#                 except Exception as e:
#                     timeline.append({
#                         "stage": "consensus",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": "error",
#                         "error": str(e),
#                         "decision_id": decision_id,
#                     })

#             # ============================================
#             # STAGE 3: Collusion Auditing
#             # ============================================
#             if self.enable_collusion:
#                 stage_start = time.time()
#                 try:
#                     collusion_data = await self._get_collusion_data(decision_id)
#                     stale_reason = collusion_data.get("_stale_reason")
#                     stage_status = "success" if collusion_data.get("score") is not None else "no_data"
#                     if stale_reason:
#                         stage_status = "stale"
#                         collusion_data = {}
#                     pipeline_results["collusion"] = collusion_data
#                     timeline.append({
#                         "stage": "collusion",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": stage_status,
#                         "stale_reason": stale_reason,
#                         "decision_id": decision_id,
#                     })
#                 except Exception as e:
#                     timeline.append({
#                         "stage": "collusion",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": "error",
#                         "error": str(e),
#                         "decision_id": decision_id,
#                     })

#             # ============================================
#             # STAGE 4: Time Horizon Alignment
#             # ============================================
#             if self.enable_horizon:
#                 stage_start = time.time()
#                 try:
#                     horizon_data = await self._get_horizon_data(decision_id)
#                     stale_reason = horizon_data.get("_stale_reason")
#                     stage_status = "success" if horizon_data.get("aligned_weights") else "no_data"
#                     if stale_reason:
#                         stage_status = "stale"
#                         horizon_data = {}
#                     pipeline_results["horizon"] = horizon_data
#                     timeline.append({
#                         "stage": "horizon",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": stage_status,
#                         "stale_reason": stale_reason,
#                         "decision_id": decision_id,
#                     })
#                 except Exception as e:
#                     timeline.append({
#                         "stage": "horizon",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": "error",
#                         "error": str(e),
#                         "decision_id": decision_id,
#                     })

#             # ============================================
#             # STAGE 5: Alternative Reality Sampling
#             # ============================================
#             if self.enable_sampling:
#                 stage_start = time.time()
#                 try:
#                     uncertainty_data = await self._get_uncertainty_data(decision_id)
#                     stale_reason = uncertainty_data.get("_stale_reason")
#                     stage_status = "success" if uncertainty_data else "no_data"
#                     if stale_reason:
#                         stage_status = "stale"
#                         uncertainty_data = {}
#                     pipeline_results["uncertainty"] = uncertainty_data
#                     timeline.append({
#                         "stage": "sampling",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": stage_status,
#                         "stale_reason": stale_reason,
#                         "decision_id": decision_id,
#                     })
#                 except Exception as e:
#                     timeline.append({
#                         "stage": "sampling",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": "error",
#                         "error": str(e),
#                         "decision_id": decision_id,
#                     })

#             # ============================================
#             # STAGE 6: Strategy Arbiter (final gating)
#             # ============================================
#             if self.enable_arbiter:
#                 stage_start = time.time()
#                 try:
#                     arbiter_data = await self._get_arbiter_data(decision_id)
#                     stale_reason = arbiter_data.get("_stale_reason")
#                     stage_status = "success" if arbiter_data.get("gate_decision") else "no_data"
#                     if stale_reason:
#                         stage_status = "stale"
#                         arbiter_data = {}
#                     pipeline_results["arbiter"] = arbiter_data
#                     timeline.append({
#                         "stage": "arbiter",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": stage_status,
#                         "stale_reason": stale_reason,
#                         "decision_id": decision_id,
#                     })
#                 except Exception as e:
#                     timeline.append({
#                         "stage": "arbiter",
#                         "duration_ms": (time.time() - stage_start) * 1000,
#                         "status": "error",
#                         "error": str(e),
#                         "decision_id": decision_id,
#                     })

#             # ============================================
#             # Assemble final bundle
#             # ============================================
#             # Fill missing stage data with last known good to avoid hard abstain when upstream is empty
#             for stage in ["committee", "consensus", "collusion", "horizon", "uncertainty", "arbiter"]:
#                 data = pipeline_results.get(stage) or {}
#                 has_data = bool(data)
#                 if stage == "committee":
#                     has_data = has_data and bool(data.get("members"))
#                 if has_data:
#                     self._last_good_results[stage] = dict(data)
#                 elif self._last_good_results.get(stage):
#                     cached = dict(self._last_good_results[stage])
#                     cached["decision_id"] = decision_id
#                     cached["tick_ts"] = tick_ts
#                     pipeline_results[stage] = cached

#             bundle = await self._assemble_decision_bundle(
#                 decision_id, tick_ts, pipeline_results
#             )

#             # ============================================
#             # FIX 2: Schema validation with detailed errors
#             # ============================================
#             if self.schema_validation:
#                 validation_errors = self._validate_schema_v2(bundle)
#                 if validation_errors:
#                     self.pipeline_stats["schema_violations"] += 1
#                     self.logger.warning(
#                         format_operator_message(
#                             icon="[SCHEMA]",
#                             message=f"Schema validation errors: {len(validation_errors)}",
#                             errors=validation_errors[:3],  # Log first 3
#                             decision_id=decision_id,
#                         )
#                     )

#             # Update statistics
#             processing_time = (time.time() - start_time) * 1000
#             self._update_pipeline_stats(processing_time, timeline, True)

#             # ============================================
#             # FIX 3: Publish with namespaced keys
#             # ============================================
#             await self._publish_bundle(bundle, decision_id, tick_ts, processing_time, timeline, pipeline_results)

#             # Log successful decision with key metrics
#             consensus_score = bundle.get("consensus", {}).get("score")  # From ConsensusDetector
#             committee_consensus_strength = bundle.get("committee", {}).get("committee_consensus", {}).get("consensus_strength")  # From Committee
#             gate_decision = bundle.get("arbiter", {}).get("gate_decision", {})
#             action = gate_decision.get("action", "abstain")
#             confidence = gate_decision.get("confidence", 0.0)
            
#             self.logger.info(
#                 f"[VOTE] ✅ Decision complete | ID: {decision_id[:30]}... | "
#                 f"Action: {action.upper()} | Confidence: {confidence:.1%} | "
#                 f"ConsensusDetector.score: {consensus_score if consensus_score is not None else 0.0:.1%} | "
#                 f"Committee.consensus_strength: {committee_consensus_strength if committee_consensus_strength is not None else 0.0:.1%} | "
#                 f"Time: {processing_time:.1f}ms"
#             )

#             try:
#                 if self.debug_timeline:
#                     self._trace_vote(bundle, pipeline_results, timeline)
#             except Exception:
#                 pass
#             return self._build_contract_output(bundle, timeline, processing_time)

#         except Exception as e:
#             return await self._handle_pipeline_error(e, start_time, timeline)

#     async def _get_committee_data(self, decision_id: str) -> Dict[str, Any]:
#         """Get committee data from bus - namespaced"""
#         sources = ["EnhancedVotingCommitteeCoordinator", "EnhancedVotingCommittee", "VotingKernel"]

#         # Read committee's decision_id (may be from previous cycle - this is normal)
#         bus_decision_id = (
#             self._bus_get_first("committee_decision_id", sources)
#             or self._bus_get_first("decision_id", sources)
#         )
#         bus_tick_ts = self._bus_get_first("tick_ts", sources)
#         fresh, reason = self._check_freshness(decision_id, bus_decision_id, bus_tick_ts, "committee")
#         if not fresh:
#             return {"_stale_reason": reason, "decision_id": bus_decision_id}

#         members = self._bus_get_first("committee_members", sources, default=[]) or []
#         proposal_vectors = (
#             self._bus_get_first("committee_proposal_vectors", sources)
#             or self._bus_get_first("proposal_vectors", sources, default=[])
#             or []
#         )
#         member_confidences = (
#             self._bus_get_first("committee_member_confidences", sources)
#             or self._bus_get_first("member_confidences_ordered", sources, default=[])
#             or []
#         )
#         committee_consensus = self._bus_get_first("committee_consensus", sources, default={}) or {}

#         return {
#             "members": members if isinstance(members, list) else [],
#             "proposal_vectors": proposal_vectors if isinstance(proposal_vectors, list) else [],
#             "member_confidences": member_confidences if isinstance(member_confidences, list) else [],
#             "committee_consensus": committee_consensus if isinstance(committee_consensus, dict) else {},
#             "decision_id": decision_id,  # Use kernel's current decision_id for coordination
#         }

#     async def _get_consensus_data(self, decision_id: str) -> Dict[str, Any]:
#         """Get consensus data from bus - namespaced"""
#         sources = ["ConsensusDetector", "VotingKernel"]

#         bus_decision_id = (
#             self._bus_get_first("consensus_decision_id", sources)
#             or self._bus_get_first("decision_id", sources)
#         )
#         bus_tick_ts = self._bus_get_first("tick_ts", sources)
#         fresh, reason = self._check_freshness(decision_id, bus_decision_id, bus_tick_ts, "consensus")
#         if not fresh:
#             return {"_stale_reason": reason, "decision_id": bus_decision_id}

#         return {
#             "score": self._bus_get_first("consensus_score", sources),
#             "components": self._bus_get_first("consensus_components", sources, default={}) or {},
#             "decision_id": decision_id,  # Use kernel's current decision_id
#         }

#     async def _get_collusion_data(self, decision_id: str) -> Dict[str, Any]:
#         """Get collusion data from bus - namespaced"""
#         sources = ["CollusionAuditor", "VotingKernel"]

#         bus_decision_id = (
#             self._bus_get_first("collusion_decision_id", sources)
#             or self._bus_get_first("decision_id", sources)
#         )
#         bus_tick_ts = self._bus_get_first("tick_ts", sources)
#         fresh, reason = self._check_freshness(decision_id, bus_decision_id, bus_tick_ts, "collusion")
#         if not fresh:
#             return {"_stale_reason": reason, "decision_id": bus_decision_id}

#         return {
#             "score": self._bus_get_first("collusion_score", sources),
#             "suspicious_pairs": self._bus_get_first("suspicious_pairs", sources, default=[]) or [],
#             "pair_penalties": self._bus_get_first("pair_penalties", sources, default={}) or {},
#             "decision_id": decision_id,  # Use kernel's current decision_id
#         }

#     async def _get_horizon_data(self, decision_id: str) -> Dict[str, Any]:
#         """Get time horizon alignment data from bus - namespaced"""
#         sources = ["TimeHorizonAligner", "VotingKernel"]

#         bus_decision_id = (
#             self._bus_get_first("horizon_decision_id", sources)
#             or self._bus_get_first("decision_id", sources)
#         )
#         bus_tick_ts = self._bus_get_first("tick_ts", sources)
#         fresh, reason = self._check_freshness(decision_id, bus_decision_id, bus_tick_ts, "horizon")
#         if not fresh:
#             return {"_stale_reason": reason, "decision_id": bus_decision_id}

#         return {
#             "raw_weights": [],
#             "aligned_weights": self._bus_get_first("aligned_weights", sources, default=[]) or [],
#             "alignment_meta": self._bus_get_first("horizon_alignment_meta", sources, default={}) or {},
#             "decision_id": decision_id,  # Use kernel's current decision_id
#         }

#     async def _get_uncertainty_data(self, decision_id: str) -> Dict[str, Any]:
#         """Get uncertainty/sampling data from bus - namespaced"""
#         sources = ["AlternativeRealitySampler", "VotingKernel"]

#         bus_decision_id = (
#             self._bus_get_first("sampling_decision_id", sources)
#             or self._bus_get_first("decision_id", sources)
#         )
#         bus_tick_ts = self._bus_get_first("tick_ts", sources)
#         fresh, reason = self._check_freshness(decision_id, bus_decision_id, bus_tick_ts, "sampling")
#         if not fresh:
#             return {"_stale_reason": reason, "decision_id": bus_decision_id}

#         return {
#             "sampling_uncertainty": self._bus_get_first("sampling_uncertainty", sources, default=0.0) or 0.0,
#             "fragility": self._bus_get_first("sampling_fragility", sources, default=0.0) or 0.0,
#             "effective_samples": (
#                 self._bus_get_first("sampling_effective_samples", sources)
#                 or self._bus_get_first("effective_samples", sources, default=0)
#                 or 0
#             ),
#             "decision_id": decision_id,  # Use kernel's current decision_id
#         }

#     async def _get_arbiter_data(self, decision_id: str) -> Dict[str, Any]:
#         """Get strategy arbiter data from bus - namespaced"""
#         sources = ["StrategyArbiter", "VotingKernel"]

#         bus_decision_id = (
#             self._bus_get_first("arbiter_decision_id", sources)
#             or self._bus_get_first("decision_id", sources)
#         )
#         bus_tick_ts = self._bus_get_first("tick_ts", sources)
#         fresh, reason = self._check_freshness(decision_id, bus_decision_id, bus_tick_ts, "arbiter")
#         if not fresh:
#             return {"_stale_reason": reason, "decision_id": bus_decision_id}

#         return {
#             "instrument_signals": (
#                 self._bus_get_first("arbiter_instrument_signals", sources)
#                 or self._bus_get_first("instrument_signals", sources, default={})
#                 or {}
#             ),
#             "gate_decision": (
#                 self._bus_get_first("arbiter_gate_decision", sources)
#                 or self._bus_get_first("gate_decision", sources, default={})
#                 or {}
#             ),
#             "gate_breakdown": self._bus_get_first("arbiter_gate_breakdown", sources, default={}) or {},
#             "decision_id": decision_id,  # Use kernel's current decision_id
#         }

#     async def _assemble_decision_bundle(
#         self, decision_id: str, tick_ts: str, results: Dict[str, Any]
#     ) -> Dict[str, Any]:
#         """Assemble the final voting decision bundle with v2 schema"""
#         committee_raw = results.get("committee", {}) or {}
#         consensus_raw = results.get("consensus", {"score": None}) or {"score": None}
#         collusion_raw = results.get("collusion", {"score": None}) or {"score": None}
#         horizon_raw = results.get("horizon", {"raw_weights": [], "aligned_weights": []}) or {"raw_weights": [], "aligned_weights": []}
#         uncertainty_raw = results.get("uncertainty", {"sampling_uncertainty": 0.0, "fragility": 0.0}) or {"sampling_uncertainty": 0.0, "fragility": 0.0}
#         arbiter_raw = results.get("arbiter", {}) or {}

#         committee: Dict[str, Any] = dict(committee_raw) if isinstance(committee_raw, dict) else {}
#         consensus: Dict[str, Any] = dict(consensus_raw) if isinstance(consensus_raw, dict) else {"score": None}
#         collusion: Dict[str, Any] = dict(collusion_raw) if isinstance(collusion_raw, dict) else {"score": None}
#         horizon: Dict[str, Any] = dict(horizon_raw) if isinstance(horizon_raw, dict) else {"raw_weights": [], "aligned_weights": []}
#         uncertainty: Dict[str, Any] = dict(uncertainty_raw) if isinstance(uncertainty_raw, dict) else {"sampling_uncertainty": 0.0, "fragility": 0.0}
#         arbiter: Dict[str, Any] = dict(arbiter_raw) if isinstance(arbiter_raw, dict) else {}

#         # Normalize to prevent schema violations when upstream is missing
#         if not isinstance(committee.get("members"), list):
#             committee["members"] = []
#         if not isinstance(committee.get("proposal_vectors"), list):
#             committee["proposal_vectors"] = []
#         if not isinstance(committee.get("member_confidences"), list):
#             committee["member_confidences"] = []
#         if not isinstance(committee.get("committee_consensus"), dict):
#             committee["committee_consensus"] = {}
#         if "decision_id" not in committee:
#             committee["decision_id"] = decision_id

#         if not isinstance(consensus, dict):
#             consensus = {"score": None, "components": {}}
#         consensus.setdefault("score", None)
#         if not isinstance(consensus.get("components"), dict):
#             consensus["components"] = {}
#         consensus["decision_id"] = decision_id

#         if not isinstance(collusion, dict):
#             collusion = {"score": None, "suspicious_pairs": []}
#         collusion.setdefault("score", None)
#         if not isinstance(collusion.get("suspicious_pairs"), list):
#             collusion["suspicious_pairs"] = []
#         collusion["decision_id"] = decision_id

#         if not isinstance(horizon, dict):
#             horizon = {"raw_weights": [], "aligned_weights": []}
#         horizon.setdefault("raw_weights", [])
#         horizon.setdefault("aligned_weights", [])
#         horizon.setdefault("alignment_meta", {})
#         horizon["decision_id"] = decision_id

#         if not isinstance(uncertainty, dict):
#             uncertainty = {"sampling_uncertainty": 0.0, "fragility": 0.0}
#         uncertainty.setdefault("sampling_uncertainty", 0.0)
#         uncertainty.setdefault("fragility", 0.0)
#         uncertainty.setdefault("effective_samples", 0)
#         uncertainty["decision_id"] = decision_id

#         if not isinstance(arbiter, dict):
#             arbiter = {}
#         arbiter.setdefault("instrument_signals", {})
#         arbiter.setdefault("gate_decision", {})
#         arbiter["decision_id"] = decision_id

#         return {
#             "decision_id": decision_id,
#             "tick_ts": tick_ts,
#             "committee": committee,
#             "consensus": consensus,
#             "collusion": collusion,
#             "horizon": horizon,
#             "uncertainty": uncertainty,
#             "arbiter": arbiter,
#             "_schema_version": "v2",
#         }

#     def _validate_schema_v2(self, bundle: Dict[str, Any]) -> List[str]:
#         """Validate voting schema v2 with stricter checks"""
#         errors = []
        
#         # Check schema version
#         if bundle.get("_schema_version") != "v2":
#             errors.append(f"Invalid schema version: {bundle.get('_schema_version')}")
        
#         # Check required top-level fields
#         required_fields = ["decision_id", "tick_ts", "committee", "consensus",
#                           "collusion", "horizon", "uncertainty", "arbiter"]
#         for field in required_fields:
#             if field not in bundle:
#                 errors.append(f"Missing required field: {field}")
        
#         # Validate committee structure
#         committee = bundle.get("committee", {})
#         if not isinstance(committee.get("members"), list):
#             errors.append("committee.members must be a list")
#         if not isinstance(committee.get("proposal_vectors"), list):
#             errors.append("committee.proposal_vectors must be a list")
#         if not isinstance(committee.get("member_confidences"), list):
#             errors.append("committee.member_confidences must be a list")
        
#         # Validate consensus structure
#         consensus = bundle.get("consensus", {})
#         if "score" not in consensus:
#             errors.append("consensus.score is required")
#         elif consensus["score"] is not None:
#             if not isinstance(consensus["score"], (int, float)):
#                 errors.append("consensus.score must be numeric or None")
        
#         # Validate collusion structure
#         collusion = bundle.get("collusion", {})
#         if "score" not in collusion:
#             errors.append("collusion.score is required")
        
#         # Validate horizon structure
#         horizon = bundle.get("horizon", {})
#         if not isinstance(horizon.get("aligned_weights"), list):
#             errors.append("horizon.aligned_weights must be a list")
        
#         # Validate uncertainty structure
#         uncertainty = bundle.get("uncertainty", {})
#         if "fragility" not in uncertainty:
#             errors.append("uncertainty.fragility is required")
#         if "sampling_uncertainty" not in uncertainty:
#             errors.append("uncertainty.sampling_uncertainty is required")
        
#         return errors

#     async def _publish_bundle(self, bundle: Dict[str, Any], decision_id: str,
#                             tick_ts: str, processing_time: float,
#                             timeline: List[Dict], pipeline_results: Dict[str, Any]):
#         """Publish bundle and surfaces with namespaced keys"""

#         # Publish main bundle
#         self.smart_bus.set(
#             "kernel_decision_bundle",
#             bundle,
#             module="VotingKernel",
#             thesis=f"Unified voting decision {decision_id} ({processing_time:.1f}ms)"
#         )

#         # Publish coordination data
#         decision_coordination = {
#             "decision_id": decision_id,
#             "tick_ts": tick_ts,
#             "stages": len(timeline),
#             "status": "ok",
#         }
#         # Publish canonical key expected by contracts
#         self.smart_bus.set("decision_coordination", decision_coordination,
#                            module="VotingKernel", thesis="Decision coordination")
#         # Keep kernel mirror for internal tooling/backward compatibility
#         try:
#             self.smart_bus.set("kernel_decision_coordination", decision_coordination,
#                                module="VotingKernel", thesis="Decision coordination (mirror)")
#         except Exception:
#             pass

#         # Publish consensus_score (float) - owned by ConsensusDetector, kernel just mirrors
#         consensus_data = pipeline_results.get("consensus", {})
#         if "score" in consensus_data:
#             self.smart_bus.set("kernel_consensus_score", consensus_data["score"],
#                              module="VotingKernel", thesis="Consensus score (kernel mirror)")

#         # Publish committee_consensus (dict) - owned by Committee, kernel mirrors
#         committee_data = pipeline_results.get("committee", {})
#         if "committee_consensus" in committee_data:
#             self.smart_bus.set("kernel_committee_consensus", committee_data["committee_consensus"],
#                              module="VotingKernel", thesis="Committee consensus (kernel mirror)")

#         # FIX: Republish committee data under VotingKernel namespace for backend access
#         # NOTE: EnhancedVotingCommitteeCoordinator is the canonical owner of these keys per contracts.py
#         # VotingKernel only mirrors under kernel_* namespace
#         if committee_data.get("members"):
#             self.smart_bus.set("kernel_committee_members", committee_data["members"],
#                              module="VotingKernel", thesis=f"Committee members ({len(committee_data['members'])} voters)")
#         if "proposal_vectors" in committee_data:
#             self.smart_bus.set("kernel_proposal_vectors", committee_data.get("proposal_vectors", []),
#                              module="VotingKernel", thesis="Committee proposal vectors")
#         if "member_confidences" in committee_data:
#             self.smart_bus.set("kernel_member_confidences_ordered", committee_data.get("member_confidences", []),
#                              module="VotingKernel", thesis="Committee member confidences")

#         # Republish member analytics and committee analytics from Committee module
#         member_analytics = self.smart_bus.get('member_analytics', 'EnhancedVotingCommittee', default=[]) or []
#         if member_analytics:
#             self.smart_bus.set("kernel_member_analytics", member_analytics,
#                              module="VotingKernel", thesis=f"Committee member analytics ({len(member_analytics)} members)")

#         committee_analytics = self.smart_bus.get('committee_analytics', 'EnhancedVotingCommittee', default={}) or {}
#         if committee_analytics:
#             self.smart_bus.set("kernel_committee_analytics", committee_analytics,
#                              module="VotingKernel", thesis="Committee performance analytics")

#         committee_votes = self.smart_bus.get('committee_votes', 'EnhancedVotingCommittee', default=[]) or []
#         if committee_votes:
#             self.smart_bus.set("kernel_committee_votes", committee_votes,
#                              module="VotingKernel", thesis=f"Committee votes ({len(committee_votes)} votes)")

#         # Publish metrics
#         voting_metrics = {
#             "processing_time_ms": processing_time,
#             "avg_processing_time_ms": self.pipeline_stats.get("avg_processing_time_ms", 0.0),
#             "total_ticks": self.pipeline_stats.get("total_ticks", 0),
#             "successful_ticks": self.pipeline_stats.get("successful_ticks", 0),
#             "failed_ticks": self.pipeline_stats.get("failed_ticks", 0),
#             "schema_violations": self.pipeline_stats.get("schema_violations", 0),
#         }
#         self.smart_bus.set("kernel_voting_metrics", voting_metrics,
#                           module="VotingKernel", thesis="Voting metrics")

#         # Publish pipeline stats for backend timeline endpoint
#         self.smart_bus.set("kernel_pipeline_stats", dict(self.pipeline_stats),
#                           module="VotingKernel", thesis="Pipeline statistics")

#         # Publish arbiter outputs - use namespaced keys only to avoid ownership conflicts
#         # StrategyArbiter owns 'instrument_signals' and 'gate_decision' per contracts.py
#         arbiter_data = pipeline_results.get("arbiter", {})
#         if "instrument_signals" in arbiter_data:
#             # Only publish namespaced version - let StrategyArbiter own the canonical 'instrument_signals'
#             self.smart_bus.set("kernel_instrument_signals", arbiter_data["instrument_signals"],
#                              module="VotingKernel", thesis="Instrument signals from arbiter (namespaced)")
#             # NOTE: Removed duplicate publish to 'instrument_signals' - StrategyArbiter is canonical owner
#         if "gate_decision" in arbiter_data:
#             # Only publish namespaced version - let StrategyArbiter own the canonical 'gate_decision'
#             self.smart_bus.set("kernel_gate_decision", arbiter_data["gate_decision"],
#                              module="VotingKernel", thesis="Gate decision from arbiter (namespaced)")
#             # NOTE: Removed duplicate publish to 'gate_decision' - StrategyArbiter is canonical owner

#         # Publish consensus components for backend
#         if "components" in consensus_data:
#             # NOTE: ConsensusDetector owns 'consensus_components' - VotingKernel publishes namespaced copy
#             self.smart_bus.set("kernel_consensus_components", consensus_data["components"],
#                              module="VotingKernel", thesis="Consensus components (namespaced)")

#         # Publish collusion data - CollusionAuditor owns these keys
#         collusion_data = pipeline_results.get("collusion", {})
#         if "score" in collusion_data:
#             # NOTE: CollusionAuditor owns 'collusion_score' - VotingKernel publishes namespaced copy
#             self.smart_bus.set("kernel_collusion_score", collusion_data["score"],
#                              module="VotingKernel", thesis="Collusion score (namespaced)")
#         if "suspicious_pairs" in collusion_data:
#             # NOTE: CollusionAuditor owns 'suspicious_pairs' - VotingKernel publishes namespaced copy
#             self.smart_bus.set("kernel_suspicious_pairs", collusion_data.get("suspicious_pairs", []),
#                              module="VotingKernel", thesis="Suspicious pairs (namespaced)")

#         # Publish horizon data - TimeHorizonAligner owns these keys
#         horizon_data = pipeline_results.get("horizon", {})
#         if "aligned_weights" in horizon_data:
#             # NOTE: TimeHorizonAligner owns 'aligned_weights' - VotingKernel publishes namespaced copy
#             self.smart_bus.set("kernel_aligned_weights", horizon_data["aligned_weights"],
#                              module="VotingKernel", thesis="Aligned weights (namespaced)")
#         if "alignment_meta" in horizon_data:
#             # NOTE: TimeHorizonAligner owns 'horizon_alignment_meta' - VotingKernel publishes namespaced copy
#             self.smart_bus.set("kernel_horizon_alignment_meta", horizon_data.get("alignment_meta", {}),
#                              module="VotingKernel", thesis="Horizon alignment meta (namespaced)")

#         # Publish sampling data - AlternativeRealitySampler owns these keys
#         uncertainty_data = pipeline_results.get("uncertainty", {})
#         if "sampling_uncertainty" in uncertainty_data:
#             # NOTE: AlternativeRealitySampler owns 'sampling_uncertainty' - VotingKernel publishes namespaced copy
#             self.smart_bus.set("kernel_sampling_uncertainty", uncertainty_data["sampling_uncertainty"],
#                              module="VotingKernel", thesis="Sampling uncertainty (namespaced)")
#         if "effective_samples" in uncertainty_data:
#             # NOTE: AlternativeRealitySampler owns 'effective_samples' - VotingKernel publishes namespaced copy
#             self.smart_bus.set("kernel_effective_samples", uncertainty_data.get("effective_samples", 0),
#                              module="VotingKernel", thesis="Effective samples (namespaced)")

#         # Publish fragility (VotingKernel owns 'fragility' per contract)
#         raw_fragility = uncertainty_data.get("fragility")
#         if isinstance(raw_fragility, (int, float)):
#             fragility_value = float(raw_fragility)
#         elif isinstance(raw_fragility, str):
#             try:
#                 fragility_value = float(raw_fragility)
#             except ValueError:
#                 fragility_value = 0.0
#         else:
#             fragility_value = 0.0
#         # VotingKernel owns 'fragility' per contract - publish both namespaced and canonical
#         self.smart_bus.set("kernel_fragility", fragility_value,
#                           module="VotingKernel", thesis=f"Voting fragility {fragility_value:.3f}")
#         self.smart_bus.set("fragility", fragility_value,
#                           module="VotingKernel", thesis=f"Fragility (canonical) {fragility_value:.3f}")
#         # NOTE: AlternativeRealitySampler owns 'sampling_fragility' - VotingKernel publishes namespaced copy
#         self.smart_bus.set("kernel_sampling_fragility", fragility_value,
#                           module="VotingKernel", thesis=f"Sampling fragility (namespaced)")

#         # Publish timeline if debug enabled
#         if self.debug_timeline:
#             self.smart_bus.set(
#                 "kernel_pipeline_timeline",
#                 {"decision_id": decision_id, "timeline": timeline},
#                 module="VotingKernel",
#                 thesis=f"Pipeline timeline for {decision_id}"
#             )
#             # Also publish timeline for backend
#             self.smart_bus.set("pipeline_timeline", timeline,
#                              module="VotingKernel", thesis="Pipeline stage timeline")

#     def _build_contract_output(self, bundle: Dict[str, Any], 
#                                timeline: List[Dict], 
#                                processing_time: float) -> Dict[str, Any]:
#         """Build the contract output for downstream consumers"""
#         committee = bundle.get("committee", {})
#         consensus = bundle.get("consensus", {})
#         collusion = bundle.get("collusion", {})
#         horizon = bundle.get("horizon", {})
#         uncertainty = bundle.get("uncertainty", {})
#         arbiter = bundle.get("arbiter", {})
        
#         # Build decision_coordination for contract compliance
#         decision_coordination = {
#             "decision_id": bundle.get("decision_id"),
#             "tick_ts": bundle.get("tick_ts"),
#             "stages": len(timeline),
#             "status": "ok",
#         }
        
#         decision_id = bundle.get("decision_id")
#         tick_ts = bundle.get("tick_ts")
        
#         return {
#             # Core bundle
#             "decision_bundle": bundle,
#             "decision_coordination": decision_coordination,  # FIX: Required by contract
#             "decision_id": decision_id,
#             "tick_ts": tick_ts,
            
#             # FIX: Contract-required namespaced coordination keys
#             "kernel_decision_id": decision_id,
#             "kernel_tick_ts": tick_ts,
            
#             # Surfaces (namespaced)
#             "consensus_score": consensus.get("score"),
#             "voting_consensus": consensus.get("score"),  # FIX: Contract alias for backward compatibility
#             "consensus_summary": consensus.get("components", {}),  # FIX: Required by contract
#             "committee_consensus": committee.get("committee_consensus", {}),
#             "collusion_score": collusion.get("score"),
#             "fragility": uncertainty.get("fragility", 0.0),
#             "sampling_uncertainty": uncertainty.get("sampling_uncertainty", 0.0),
            
#             # Voting metrics (required by contract)
#             "voting_metrics": {
#                 "processing_time_ms": processing_time,
#                 "total_ticks": self.pipeline_stats.get("total_ticks", 0),
#                 "successful_ticks": self.pipeline_stats.get("successful_ticks", 0),
#                 "failed_ticks": self.pipeline_stats.get("failed_ticks", 0),
#             },
            
#             # Trade vote v2 (required by contract)
#             "trade_vote_v2": arbiter.get("gate_decision", {
#                 "action": "abstain",
#                 "size": 0.0,
#                 "confidence": 0.0,
#                 "consensus_score": consensus.get("score", 0.0) or 0.0,  # Pass through consensus score
#                 "reason": "no_arbiter_decision",
#             }),
            
#             # Committee outputs
#             "committee_members": committee.get("members", []),
#             "proposal_vectors": committee.get("proposal_vectors", []),
#             "member_confidences": committee.get("member_confidences", []),
            
#             # Arbiter outputs
#             "instrument_signals": arbiter.get("instrument_signals", {}),
#             "gate_decision": arbiter.get("gate_decision", {}),
            
#             # Metrics
#             "pipeline_timeline": timeline,
#             "processing_time_ms": processing_time,
#             "pipeline_stats": dict(self.pipeline_stats),
            
#             "_thesis": f"VotingKernel v2.0: {len(timeline)} stages, {processing_time:.1f}ms",
#         }

#     def _update_pipeline_stats(self, processing_time: float, timeline: List[Dict], success: bool):
#         """Update pipeline statistics"""
#         if success:
#             self.pipeline_stats["successful_ticks"] += 1
#             self.error_count = max(0, self.error_count - 1)  # Decay error count on success
#         else:
#             self.pipeline_stats["failed_ticks"] += 1

#         # Update average processing time
#         total = self.pipeline_stats["successful_ticks"] + self.pipeline_stats["failed_ticks"]
#         old_avg = self.pipeline_stats["avg_processing_time_ms"]
#         self.pipeline_stats["avg_processing_time_ms"] = (old_avg * (total - 1) + processing_time) / total

#         # Update module success rates
#         for stage in timeline:
#             stage_name = stage["stage"]
#             if stage_name not in self.pipeline_stats["module_success_rates"]:
#                 self.pipeline_stats["module_success_rates"][stage_name] = {"success": 0, "total": 0}
            
#             self.pipeline_stats["module_success_rates"][stage_name]["total"] += 1
#             if stage["status"] == "success":
#                 self.pipeline_stats["module_success_rates"][stage_name]["success"] += 1

#     async def _handle_pipeline_error(self, error: Exception, start_time: float, 
#                                      timeline: List[Dict]) -> Dict[str, Any]:
#         """Handle pipeline errors with circuit breaker"""
#         self.error_count += 1
#         processing_time = (time.time() - start_time) * 1000
#         ctx = self.error_pinpointer.analyze_error(error, "VotingKernel")
#         self.pipeline_stats["last_error"] = str(ctx)

#         if self.error_count >= self.circuit_breaker_threshold:
#             self.is_disabled = True
#             self.logger.error(
#                 format_operator_message(
#                     icon="[ALERT]",
#                     message="VotingKernel disabled due to repeated errors",
#                     error_count=self.error_count,
#                     threshold=self.circuit_breaker_threshold,
#                 )
#             )

#         self._update_pipeline_stats(processing_time, timeline, False)

#         return {
#             "decision_bundle": {"error": str(ctx), "_schema_version": "v2"},
#             "decision_coordination": {"status": "error", "error": str(ctx)},  # FIX: Required by contract
#             "voting_consensus": 0.0,  # FIX: Required by contract
#             "consensus_summary": {},  # FIX: Required by contract
#             "voting_metrics": {"processing_time_ms": processing_time},  # FIX: Required by contract
#             "trade_vote_v2": {"action": "abstain", "size": 0.0, "confidence": 0.0, "consensus_score": 0.0, "reason": "kernel_error"},  # FIX: Required
#             "decision_id": None,
#             "pipeline_timeline": timeline,
#             "processing_time_ms": processing_time,
#             "pipeline_stats": dict(self.pipeline_stats),
#             "fragility": 1.0,  # Max uncertainty on error
#             "committee_members": [],
#             "_thesis": f"VotingKernel error: {ctx}",
#         }

#     def _generate_disabled_response(self) -> Dict[str, Any]:
#         """Generate response when kernel is disabled"""
#         return {
#             "decision_bundle": {"status": "disabled", "_schema_version": "v2"},
#             "decision_coordination": {"status": "disabled", "reason": "circuit_breaker"},  # FIX: Required by contract
#             "voting_consensus": 0.0,  # FIX: Required by contract
#             "consensus_summary": {},  # FIX: Required by contract
#             "voting_metrics": {"processing_time_ms": 0.0},  # FIX: Required by contract
#             "trade_vote_v2": {"action": "abstain", "size": 0.0, "confidence": 0.0, "consensus_score": 0.0, "reason": "disabled"},  # FIX: Required
#             "decision_id": None,
#             "pipeline_timeline": [],
#             "processing_time_ms": 0.0,
#             "pipeline_stats": dict(self.pipeline_stats),
#             "fragility": 1.0,  # Max uncertainty when disabled
#             "committee_members": [],
#             "_thesis": "VotingKernel disabled via circuit breaker",
#         }

#     def get_state(self) -> Dict[str, Any]:
#         """Get kernel state for persistence"""
#         base = super().get_state()
#         base["custom_state"] = {
#             "pipeline_stats": dict(self.pipeline_stats),
#             "error_count": self.error_count,
#             "is_disabled": self.is_disabled,
#         }
#         return base

#     def set_state(self, state: Dict[str, Any]):
#         """Set kernel state from persistence"""
#         super().set_state(state)
#         cs = state.get("custom_state", {})
#         if cs.get("pipeline_stats"):
#             self.pipeline_stats.update(cs["pipeline_stats"])
#         self.error_count = int(cs.get("error_count", 0))
#         self.is_disabled = bool(cs.get("is_disabled", False))

#     def _trace_vote(self, bundle: Dict[str, Any], results: Dict[str, Any], timeline: List[Dict[str, Any]]) -> None:
#         """Emit a compact, end-to-end trace of the voting decision for debugging."""
#         try:
#             committee = bundle.get("committee", {}) or {}
#             consensus = bundle.get("consensus", {}) or {}
#             collusion = bundle.get("collusion", {}) or {}
#             horizon = bundle.get("horizon", {}) or {}
#             uncertainty = bundle.get("uncertainty", {}) or {}
#             arbiter = bundle.get("arbiter", {}) or {}

#             cons_score_raw = consensus.get("score")
#             cons_score = float(cons_score_raw) if isinstance(cons_score_raw, (int, float)) else 0.0
#             cc = committee.get("committee_consensus", {}) or {}
#             comm_strength = float(cc.get("consensus_strength", 0.0)) if isinstance(cc.get("consensus_strength", 0.0), (int, float)) else 0.0

#             gate = arbiter.get("gate_decision", {}) or {}
#             instr = arbiter.get("instrument_signals", {}) or {}
#             tops = []
#             try:
#                 pairs = []
#                 for k, v in instr.items():
#                     val = v.get("intensity") if isinstance(v, dict) else None
#                     if isinstance(val, (int, float)):
#                         pairs.append((k, float(val)))
#                 tops = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:3]
#             except Exception:
#                 tops = []

#             self.logger.debug(
#                 format_operator_message(
#                     icon="[TRACE]",
#                     message="Voting pipeline",
#                     action=str(gate.get("action", "abstain")),
#                     confidence=float(gate.get("confidence", 0.0)) if isinstance(gate.get("confidence"), (int, float)) else 0.0,
#                     size=float(gate.get("size", 0.0)) if isinstance(gate.get("size"), (int, float)) else 0.0,
#                     consensus_score=f"{cons_score:.3f}",
#                     committee_strength=f"{comm_strength:.3f}",
#                     collusion_score=float(collusion.get("score", 0.0)) if isinstance(collusion.get("score", 0.0), (int, float)) else 0.0,
#                     aligned_weights=len(horizon.get("aligned_weights", [])) if isinstance(horizon.get("aligned_weights", []), list) else 0,
#                     sampling_uncertainty=float(uncertainty.get("sampling_uncertainty", 0.0)) if isinstance(uncertainty.get("sampling_uncertainty", 0.0), (int, float)) else 0.0,
#                     fragility=float(uncertainty.get("fragility", 0.0)) if isinstance(uncertainty.get("fragility", 0.0), (int, float)) else 0.0,
#                     top_signals=[{"instrument": k, "intensity": v} for k, v in tops],
#                     stages=len(timeline),
#                 )
#             )
#         except Exception:
#             pass
