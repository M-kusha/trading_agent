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
# 🤝 Enhanced Consensus Detector with SmartInfoBus Integration v3.2 - AUDIT FIXED
# Production-grade consensus analysis and agreement measurement for voting committees.
# """

# from __future__ import annotations

# import asyncio
# import time
# import math
# from modules.contracts import module_args
# import numpy as np
# import datetime as dt
# from typing import Dict, Any, List, Optional, Tuple, Deque
# from collections import deque, defaultdict

# # ═══════════════════════════════════════════════════════════════════
# # MODERN SMARTINFOBUS IMPORTS
# # ═══════════════════════════════════════════════════════════════════
# from modules.core.module_base import BaseModule, module
# from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
# from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
# from modules.utils.info_bus import InfoBusManager
# from modules.utils.audit_utils import RotatingLogger, format_operator_message
# from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
# from modules.monitoring.health_monitor import HealthMonitor
# from modules.monitoring.performance_tracker import PerformanceTracker


# @module(**module_args(
#     "ConsensusDetector",
#     description="Production-grade consensus analysis and agreement measurement for voting committees.",
#     error_handling=True,
#     hot_reload=True,
#     timeout_ms=3000,
# ))
# class ConsensusDetector(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
#     """
#     🤝 PRODUCTION-GRADE Consensus Detector v3.1

#     Key capabilities:
#     - Multi-dimensional agreement (direction, magnitude, confidence, network, temporal)
#     - Adaptive smoothing/weighting based on market regime & volatility
#     - Robust member contribution & reliability tracking
#     - Circuit breaker, health metrics, performance telemetry
#     - Zero-wiring SmartInfoBus integration with BUS-safe defaults
#     """

#     # ────────────────────────────
#     # INIT
#     # ────────────────────────────
#     def _initialize(self) -> None:
#         # Initialize core systems & mixins
#         self._initialize_trading_state()
#         self._initialize_state_management()
#         self._init_systems()

#         # Config (with safe defaults)
#         self.n_members: int = int(self.config.get("n_members", 5))
#         self.threshold: float = float(self.config.get("threshold", 0.6))
#         self.quality_weighting: bool = bool(self.config.get("quality_weighting", True))
#         self.temporal_smoothing: bool = bool(self.config.get("temporal_smoothing", True))
#         self.consensus_methods: List[str] = list(
#             self.config.get(
#                 "consensus_methods",
#                 ["cosine_agreement", "direction_alignment", "confidence_weighted"],
#             )
#         )
#         self.debug: bool = bool(self.config.get("debug", False))

#         # Algorithms registry
#         self.consensus_algorithms = self._initialize_consensus_algorithms()

#         # Core state
#         self.last_consensus: float = 0.0
#         self.consensus_quality: float = 0.0
#         self.consensus_components: Dict[str, float] = {}
#         self.consensus_history: deque = deque(maxlen=150)
#         self.consensus_trends: deque = deque(maxlen=80)

#         # Dimensions
#         self.directional_consensus: float = 0.0
#         self.magnitude_consensus: float = 0.0
#         self.confidence_consensus: float = 0.0
#         self.temporal_stability: float = 0.0
#         self.network_consensus: float = 0.0

#         # Member analytics
#         self.member_contributions: Dict[int, Dict[str, float]] = defaultdict(
#             lambda: {
#                 "avg_alignment": 0.5,
#                 "consistency": 0.5,
#                 "influence_weight": 1.0 / max(self.n_members, 1),
#                 "consensus_contribution": 0.5,
#                 "reliability_score": 0.5,
#                 "coordination_factor": 0.0,
#             }
#         )

#         # Quality metrics - FIX #10: consensus_quality_metrics is now primary
#         self.consensus_quality_metrics: Dict[str, float] = {
#             "coherence": 0.5, "stability": 0.5, "diversity": 0.5,
#             "reliability": 0.5, "predictive_accuracy": 0.5,
#             "temporal_consistency": 0.5, "overall_effectiveness": 0.5,
#         }
#         self.quality_metrics: Dict[str, float] = self.consensus_quality_metrics  # alias for backward compat

#         # Intelligence knobs
#         self.consensus_intelligence: Dict[str, Any] = {
#             "smoothing_alpha": 0.3, "stability_window": 10, "quality_threshold": 0.7,
#             "trend_sensitivity": 0.15, "adaptation_rate": 0.12,
#             "confidence_weighting": 0.8, "temporal_memory": 0.85,
#             "max_dim": 256,
#         }

#         # Market adaptation
#         self.market_adaptation: Dict[str, Any] = {
#             "regime_adjustments": {
#                 "trending": {"weight_multiplier": 1.10, "stability_factor": 1.20},
#                 "ranging": {"weight_multiplier": 0.95, "stability_factor": 0.90},
#                 "volatile": {"weight_multiplier": 0.85, "stability_factor": 0.70},
#                 "breakout": {"weight_multiplier": 1.20, "stability_factor": 1.30},
#                 "reversal": {"weight_multiplier": 1.05, "stability_factor": 1.10},
#                 "unknown": {"weight_multiplier": 1.00, "stability_factor": 1.00},
#             },
#             "volatility_adjustments": {
#                 "very_low": 0.90, "low": 0.95, "medium": 1.00, "high": 1.10, "extreme": 1.20,
#             },
#         }

#         # Stats & analysis
#         self.consensus_stats: Dict[str, Any] = {
#             "total_computations": 0, "high_consensus_count": 0, "low_consensus_count": 0,
#             "avg_consensus": 0.5, "consensus_volatility": 0.0, "quality_score": 0.5,
#             "trend_accuracy": 0.0, "prediction_accuracy": 0.0,
#             "session_start": dt.datetime.now().isoformat(),
#         }
#         self.regime_consensus_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
#         self.consensus_patterns: Dict[str, Any] = defaultdict(list)
#         self.prediction_history: deque = deque(maxlen=40)
#         self.consensus_insights: Dict[str, Any] = {
#             "dominant_patterns": [], "member_dynamics": {}, "consensus_predictors": {}, "quality_drivers": {},
#         }

#         # Circuit breaker
#         self.error_count: int = 0
#         self.circuit_breaker_threshold: int = 5
#         self.is_disabled: bool = False
#         self.last_failure_time: float = 0.0  # FIX #15: for circuit breaker persistence

#         # Concurrency guard
#         self._process_lock: asyncio.Lock = asyncio.Lock()

#         # Init payload
#         self._init_payload: Dict[str, Any] = {
#             "status": "initializing",
#             "timestamp": dt.datetime.now().isoformat(),
#             "configuration": {
#                 "members": self.n_members,
#                 "threshold": self.threshold,
#                 "consensus_methods": list(self.consensus_methods),
#             },
#         }

#         # Initialization thesis + early BUS publish
#         self._generate_initialization_thesis()
#         version = getattr(self.metadata, "version", "3.2.0") if self.metadata else "3.1.0"
#         self.logger.info(
#             format_operator_message(
#                 icon="🤝",
#                 message=f"Consensus Detector v{version} initialized",
#                 members=self.n_members,
#                 threshold=f"{self.threshold:.3f}",
#                 methods=len(self.consensus_methods),
#                 quality_weighting=self.quality_weighting,
#                 temporal_smoothing=self.temporal_smoothing,
#             )
#         )
#         try:
#             # Seed ONLY the canonical numeric output. Do NOT seed/write 'voting_consensus' on the bus.
#             self.smart_bus.set(
#                 "consensus_score",
#                 float(self.last_consensus),
#                 module="ConsensusDetector",
#                 thesis="Initial consensus score placeholder",
#             )
#         except Exception:
#             pass


#     def _init_systems(self) -> None:
#         """Initialize logging, error handling, telemetry, health."""
#         self.smart_bus = InfoBusManager.get_instance()
#         self.logger = RotatingLogger(
#             name="ConsensusDetector",
#             log_path="logs/voting/consensus_detector.log",
#             max_lines=5000,
#             operator_mode=True,
#             plain_english=True,
#         )
#         self.error_pinpointer = ErrorPinpointer()
#         self.error_handler = create_error_handler("ConsensusDetector", self.error_pinpointer)
#         self.english_explainer = EnglishExplainer()
#         self.system_utilities = SystemUtilities()
#         self.performance_tracker = PerformanceTracker()
#         self.health_monitor = HealthMonitor()

#     def _initialize_consensus_algorithms(self) -> Dict[str, Dict[str, Any]]:
#         """Definitions for algorithm metadata (used for dynamic weighting)."""
#         return {
#             "cosine_agreement": {
#                 "description": "Cosine similarity for directional agreement",
#                 "parameters": {"normalization": True, "weight_threshold": 0.1},
#                 "use_cases": ["directional_consensus", "vector_alignment"],
#                 "effectiveness_threshold": 0.7,
#                 "computational_cost": "low",
#             },
#             "direction_alignment": {
#                 "description": "Binary direction alignment w/ confidence weighting",
#                 "parameters": {"confidence_weighting": True, "threshold_adaptive": True},
#                 "use_cases": ["binary_voting", "directional_coherence"],
#                 "effectiveness_threshold": 0.8,
#                 "computational_cost": "low",
#             },
#             "confidence_weighted": {
#                 "description": "Consensus weighted by member confidence & reliability",
#                 "parameters": {"variance_penalty": 0.3, "reliability_boost": 1.2},
#                 "use_cases": ["quality_consensus", "reliability_analysis"],
#                 "effectiveness_threshold": 0.75,
#                 "computational_cost": "medium",
#             },
#             "magnitude_consensus": {
#                 "description": "Agreement on action magnitude/strength",
#                 "parameters": {"cv_normalization": True, "outlier_handling": True},
#                 "use_cases": ["strength_consensus", "magnitude_alignment"],
#                 "effectiveness_threshold": 0.6,
#                 "computational_cost": "low",
#             },
#             "network_consensus": {
#                 "description": "Network-based consensus considering pairwise similarities",
#                 "parameters": {"network_threshold": 0.5, "influence_weighting": True},
#                 "use_cases": ["network_analysis", "influence_consensus"],
#                 "effectiveness_threshold": 0.7,
#                 "computational_cost": "high",
#             },
#             "temporal_consensus": {
#                 "description": "Time-series stability & predictive agreement",
#                 "parameters": {"window_size": 10, "trend_weighting": 0.3},
#                 "use_cases": ["temporal_analysis", "trend_consensus"],
#                 "effectiveness_threshold": 0.65,
#                 "computational_cost": "medium",
#             },
#         }

#     def _generate_initialization_thesis(self) -> None:
#         thesis = f"""
# Consensus Detector v3.1 Initialization:
# - Members={self.n_members}, Threshold={self.threshold:.3f}
# - Methods={len(self.consensus_algorithms)} (active: {', '.join(self.consensus_methods)})
# - Quality weighting={'on' if self.quality_weighting else 'off'}, Temporal smoothing={'on' if self.temporal_smoothing else 'off'} (α={self.consensus_intelligence['smoothing_alpha']:.2f})
# - Regime/volatility adaptive with stability window={self.consensus_intelligence['stability_window']} steps
# """
#         self.smart_bus.set(
#             "consensus_detector_initialization",
#             {
#                 "status": "initialized",
#                 "thesis": thesis,
#                 "timestamp": dt.datetime.now().isoformat(),
#                 "configuration": {
#                     "members": self.n_members,
#                     "threshold": self.threshold,
#                     "consensus_methods": list(self.consensus_algorithms.keys()),
#                     "intelligence_parameters": self.consensus_intelligence,
#                 },
#             },
#             module="ConsensusDetector",
#             thesis=thesis,
#         )

#     # ────────────────────────────
#     # MAIN PROCESS
#     # ────────────────────────────
#     async def process(self, **inputs) -> Dict[str, Any]:
#         """Run a full consensus pass with adaptation, analysis, and BUS updates."""
#         async with self._process_lock:
#             start = time.time()
#             try:
#                 if self.is_disabled:
#                     return self._generate_disabled_response()

#                 # FIX #2: Read kernel's decision_id for coordination
#                 decision_id = self.smart_bus.get('kernel_decision_id', 'ConsensusDetector')
                
#                 voting_data = await self._get_comprehensive_voting_data()
#                 await self._update_consensus_parameters_comprehensive(voting_data)
#                 consensus_analysis = await self._perform_comprehensive_consensus_analysis(voting_data)
#                 contributions = await self._update_member_contributions_comprehensive(voting_data)
#                 trends = await self._analyze_consensus_trends_comprehensive(voting_data)
#                 quality = await self._calculate_comprehensive_quality_metrics()

#                 recommendations = await self._generate_intelligent_consensus_recommendations(
#                     consensus_analysis, quality, trends
#                 )
#                 thesis = await self._generate_comprehensive_consensus_thesis(
#                     consensus_analysis, quality, recommendations
#                 )

#                 results: Dict[str, Any] = {
#                     "consensus_score": float(self.last_consensus),
#                     "voting_consensus": float(self.last_consensus),  # contract alias
#                     "consensus_quality": float(self.consensus_quality),
#                     "consensus_components": dict(self.consensus_components),
#                     "directional_consensus": float(self.directional_consensus),
#                     "magnitude_consensus": float(self.magnitude_consensus),
#                     "confidence_consensus": float(self.confidence_consensus),
#                     "member_contributions": self._get_member_contributions_summary(),
#                     "consensus_trends": self._get_consensus_trends_summary(),
#                     "quality_metrics": dict(self.quality_metrics),
#                     "consensus_quality_metrics": dict(self.quality_metrics),
#                     "consensus_recommendations": list(recommendations),
#                     "health_metrics": self._get_health_metrics(),
#                     "consensus_detector_initialization": dict(
#                         getattr(self, "_init_payload", {
#                             "status": "initialized",
#                             "timestamp": dt.datetime.now().isoformat(),
#                             "configuration": {
#                                 "members": self.n_members,
#                                 "threshold": self.threshold,
#                                 "consensus_methods": list(self.consensus_methods),
#                             },
#                         })
#                     ),
#                     "decision_id": decision_id,  # FIX #2: Include decision_id for coordination
#                     "consensus_decision_id": decision_id,  # FIX: Contract-required namespaced decision_id
#                     "_thesis": thesis,
#                 }

#                 await self._update_smartinfobus_comprehensive(results, thesis)
#                 self.performance_tracker.record_metric(
#                     "ConsensusDetector", "process_time_ms", (time.time() - start) * 1000.0, True
#                 )
#                 self.error_count = 0
#                 return results

#             except Exception as e:
#                 return await self._handle_processing_error(e, start)

#     # ────────────────────────────
#     # BUS IO
#     # ────────────────────────────
#     async def _get_comprehensive_voting_data(self) -> Dict[str, Any]:
#         """Pull everything we may need from the bus with safe fallbacks (schema v1-aware)."""
#         try:
#             g = self.smart_bus.get
            
#             # FIX: Read from modules that PUBLISH voting data, not from self!
#             # Committee publishes the actual voting data
#             source_modules = ["EnhancedVotingCommitteeCoordinator", "VotingKernel"]
            
#             # Helper to try multiple modules
#             def get_from_sources(key: str, default=None):
#                 for module in source_modules:
#                     val = g(key, module)
#                     if val is not None:
#                         return val
#                 return default
            
#             # Prefer the canonical committee surfaces; keep legacy fallbacks.
#             member_confs = (
#                 get_from_sources("member_confidences_ordered")
#                 or get_from_sources("member_confidences")
#                 or []
#             )
            
#             return {
#                 # Canonical numeric vectors for analytics (preferred)
#                 "proposal_vectors": get_from_sources("proposal_vectors") or [],
#                 # Legacy / UI-friendly fallbacks
#                 "raw_proposals": get_from_sources("raw_proposals") or [],
#                 "votes": get_from_sources("votes") or get_from_sources("committee_votes") or [],
#                 # Confidences (aligned to proposal_vectors order when present)
#                 "member_confidences": member_confs,
#                 # Context
#                 "voting_summary": get_from_sources("voting_summary") or {},
#                 "alpha_weights": get_from_sources("alpha_weights") or [],
#                 "blended_action": get_from_sources("blended_action") or [],
#                 "market_context": get_from_sources("market_context") or g("market_conditions", "UnifiedMarketModule") or {},
#                 "agreement_score": get_from_sources("agreement_score") or 0.5,
#                 "consensus_direction": get_from_sources("consensus_direction") or "neutral",
#                 "market_regime": get_from_sources("market_regime") or g("market_regime", "UnifiedMarketModule") or "unknown",
#                 "volatility_data": get_from_sources("volatility_data") or g("volatility_level", "UnifiedMarketModule") or {},
#                 # Pass-through orchestration tags (set by the kernel)
#                 "decision_id": g("kernel_decision_id", "VotingKernel") or g("decision_id", "VotingKernel"),
#                 "tick_ts": g("kernel_tick_ts", "VotingKernel") or g("tick_ts", "VotingKernel"),
#             }
#         except Exception as e:
#             ctx = self.error_pinpointer.analyze_error(e, "ConsensusDetector")
#             self.logger.warning(f"Voting data retrieval incomplete: {ctx}")
#             return self._get_safe_voting_defaults()


#     # ────────────────────────────
#     # ADAPTIVE PARAMS
#     # ────────────────────────────
#     async def _update_consensus_parameters_comprehensive(self, voting_data: Dict[str, Any]) -> None:
#         """Adapt smoothing & internal weights to market conditions + agreement."""
#         try:
#             regime = str(voting_data.get("market_regime", "unknown"))
#             vol_level = self._extract_volatility_level(voting_data.get("volatility_data", {}))
#             agreement = float(voting_data.get("agreement_score", 0.5))

#             # Market consensus factor: combines agreement, regime & vol
#             mcf = self._calculate_market_consensus_factor(voting_data)

#             # Regime/volatility influence
#             base_alpha = 0.3
#             if regime == "volatile" or vol_level in {"high", "extreme"}:
#                 new_alpha = base_alpha * 0.7
#             elif regime == "trending":
#                 new_alpha = base_alpha * 1.3
#             else:
#                 new_alpha = base_alpha

#             # Adjust based on market consensus factor (more consensus → less smoothing)
#             new_alpha *= 0.9 + 0.2 * mcf  # clamp effect
#             new_alpha = float(np.clip(new_alpha, 0.1, 0.8))

#             # Smoothly adapt
#             old_alpha = float(self.consensus_intelligence["smoothing_alpha"])
#             rate = float(self.consensus_intelligence["adaptation_rate"])
#             self.consensus_intelligence["smoothing_alpha"] = float(old_alpha * (1 - rate) + new_alpha * rate)

#             # Adjust confidence weighting with agreement (more agreement → slightly higher weight)
#             old_cw = float(self.consensus_intelligence["confidence_weighting"])
#             target_cw = float(np.clip(0.6 + 0.6 * agreement, 0.6, 1.2))
#             self.consensus_intelligence["confidence_weighting"] = float(old_cw * (1 - rate) + target_cw * rate)

#             # Adjust stability window with volatility
#             if vol_level in {"high", "extreme"}:
#                 self.consensus_intelligence["stability_window"] = int(min(18, self.consensus_intelligence["stability_window"] + 1))
#             elif vol_level in {"very_low", "low"}:
#                 self.consensus_intelligence["stability_window"] = int(max(6, self.consensus_intelligence["stability_window"] - 1))

#             if abs(self.consensus_intelligence["smoothing_alpha"] - old_alpha) > 0.05:
#                 self.logger.info(
#                     format_operator_message(
#                         icon="⚙️",
#                         message="Consensus parameters adapted",
#                         old_alpha=f"{old_alpha:.3f}",
#                         new_alpha=f"{self.consensus_intelligence['smoothing_alpha']:.3f}",
#                         confidence_weighting=f"{self.consensus_intelligence['confidence_weighting']:.2f}",
#                         regime=regime,
#                         volatility=vol_level,
#                         market_factor=f"{mcf:.3f}",
#                     )
#                 )
#         except Exception as e:
#             ctx = self.error_pinpointer.analyze_error(e, "consensus_parameters_update")
#             self.logger.warning(f"Consensus parameter update failed: {ctx}")

#     def _calculate_market_consensus_factor(self, voting_data: Dict[str, Any]) -> float:
#         """Blend agreement + regime + volatility + confidence into [0,1]."""
#         try:
#             parts: List[float] = []

#             # 1) Agreement
#             parts.append(float(voting_data.get("agreement_score", 0.5)))

#             # 2) Regime
#             regime = str(voting_data.get("market_regime", "unknown"))
#             parts.append(
#                 {
#                     "trending": 0.8,
#                     "ranging": 0.6,
#                     "volatile": 0.3,
#                     "breakout": 0.9,
#                     "reversal": 0.4,
#                     "unknown": 0.5,
#                 }.get(regime, 0.5)
#             )

#             # 3) Volatility
#             vol_level = self._extract_volatility_level(voting_data.get("volatility_data", {}))
#             parts.append({"very_low": 0.9, "low": 0.8, "medium": 0.6, "high": 0.4, "extreme": 0.2}.get(vol_level, 0.6))

#             # 4) Confidence (avg & dispersion)
#             confidences = self._extract_member_confidences(voting_data)
#             if confidences:
#                 avg_c = float(np.mean(confidences))
#                 disp = float(np.std(confidences))
#                 conf_factor = float(np.clip((avg_c + (1.0 - disp)) / 2.0, 0.0, 1.0))
#                 parts.append(conf_factor)

#             w = np.array([0.30, 0.25, 0.25, 0.20][: len(parts)], dtype=np.float32)
#             w = w / float(w.sum()) if float(w.sum()) > 0 else w
#             return float(np.clip(float(np.dot(np.array(parts, dtype=np.float32), w)), 0.0, 1.0))
#         except Exception:
#             return 0.5

#     def _extract_volatility_level(self, volatility_data: Any) -> str:
#         """Map various volatility representations -> one of {very_low,low,medium,high,extreme}."""
#         try:
#             if isinstance(volatility_data, dict):
#                 level = volatility_data.get("level")
#                 if isinstance(level, str) and level in {"very_low", "low", "medium", "high", "extreme"}:
#                     return level
#                 for key in ("value", "atr", "sigma", "vol"):
#                     if key in volatility_data:
#                         return self._map_numeric_volatility_to_level(float(volatility_data[key]))
#                 return "medium"
#             if isinstance(volatility_data, (int, float, np.floating)):
#                 return self._map_numeric_volatility_to_level(float(volatility_data))
#             return "medium"
#         except Exception:
#             return "medium"

#     def _map_numeric_volatility_to_level(self, v: float) -> str:
#         try:
#             v = abs(float(v))
#             if v < 0.01:
#                 return "very_low"
#             if v < 0.02:
#                 return "low"
#             if v < 0.05:
#                 return "medium"
#             if v < 0.10:
#                 return "high"
#             return "extreme"
#         except Exception:
#             return "medium"

#     # ────────────────────────────
#     # CORE CONSENSUS
#     # ────────────────────────────
#     async def _perform_comprehensive_consensus_analysis(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             self.consensus_stats["total_computations"] += 1

#             actions = self._extract_voting_actions(voting_data)
#             confidences = self._extract_member_confidences(voting_data)

#             if len(actions) < 2:
#                 return {"consensus_score": 0.5, "analysis_status": "insufficient_data", "member_count": len(actions)}

#             actions, confidences = await self._validate_and_normalize_inputs(actions, confidences)

#             components = await self._calculate_comprehensive_consensus_components(actions, confidences)
#             weighted = await self._apply_advanced_consensus_weighting(components, actions, confidences, voting_data)
#             final = await self._apply_temporal_smoothing(weighted)

#             await self._update_consensus_state_comprehensive(final, components, actions, confidences)
#             q = await self._calculate_advanced_consensus_quality(actions, confidences, components)
#             await self._record_consensus_event_comprehensive(final, components, q, voting_data)

#             return {
#                 "consensus_score": float(final),
#                 "consensus_components": dict(components),
#                 "consensus_quality": float(q),
#                 "analysis_status": "complete",
#                 "member_count": len(actions),
#                 "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
#             }
#         except Exception as e:
#             ctx = self.error_pinpointer.analyze_error(e, "consensus_analysis")
#             self.logger.error(f"Consensus analysis failed: {ctx}")
#             return {"consensus_score": 0.5, "analysis_status": "error", "error": str(ctx)}

#     # Input extraction & validation
#     def _extract_voting_actions(self, voting_data: Dict[str, Any]) -> List[np.ndarray]:
#         try:
#             # Preferred: committee-published numeric vectors (schema v1)
#             pv = voting_data.get("proposal_vectors", [])
#             if isinstance(pv, list) and len(pv) >= 2:
#                 out: List[np.ndarray] = []
#                 for p in pv[: self.n_members]:
#                     if isinstance(p, (list, np.ndarray)) and len(p) > 0:
#                         out.append(np.asarray(p, dtype=np.float32).flatten())
#                 if len(out) >= 2:
#                     return out

#             # Legacy: raw_proposals may be vectors already
#             raw = voting_data.get("raw_proposals", [])
#             if raw and len(raw) >= 2:
#                 out = []
#                 for p in raw[: self.n_members]:
#                     if isinstance(p, (list, np.ndarray)) and len(p) > 0:
#                         out.append(np.asarray(p, dtype=np.float32).flatten())
#                 if len(out) >= 2:
#                     return out

#             # Fallback: synthesize from blended_action if present
#             blended = voting_data.get("blended_action", [])
#             if isinstance(blended, (list, np.ndarray)) and len(blended) > 0:
#                 base = np.asarray(blended, dtype=np.float32).flatten()
#                 out = [base]
#                 for _ in range(min(self.n_members - 1, 4)):
#                     noise = np.random.normal(0.0, 0.1, size=min(len(base), self.consensus_intelligence["max_dim"]))
#                     out.append((base[: len(noise)] + noise).astype(np.float32))
#                 return out

#             # Very old path: scalar votes -> 1D vectors
#             votes = voting_data.get("votes", [])
#             if votes and len(votes) >= 2:
#                 return [np.array([float(v)], dtype=np.float32) for v in votes[: self.n_members]]

#             return []
#         except Exception:
#             return []


#     def _extract_member_confidences(self, voting_data: Dict[str, Any]) -> List[float]:
#         try:
#             confs = voting_data.get("member_confidences", [])
#             if confs:
#                 return [float(np.clip(c, 0.0, 1.0)) for c in confs[: self.n_members]]
#             weights = voting_data.get("alpha_weights", [])
#             if weights:
#                 w = np.asarray(weights[: self.n_members], dtype=np.float32)
#                 s = float(w.sum())
#                 if s > 0:
#                     w = w / s
#                     return [float(np.clip(v * self.n_members, 0.1, 1.0)) for v in w]
#             return [0.5] * min(self.n_members, 5)
#         except Exception:
#             return [0.5] * min(self.n_members, 5)

#     async def _validate_and_normalize_inputs(
#         self, actions: List[np.ndarray], confidences: List[float]
#     ) -> Tuple[List[np.ndarray], List[float]]:
#         try:
#             m = min(len(actions), len(confidences), self.n_members)
#             actions = [np.asarray(a, dtype=np.float32).flatten() for a in actions[:m] if len(a) > 0]
#             confidences = [float(np.clip(c, 0.0, 1.0)) for c in confidences[:m]]

#             if len(actions) < 2 or len(confidences) < 2:
#                 a = np.array([0.0], dtype=np.float32)
#                 return [a, a.copy()], [0.5, 0.5]

#             # Cap vector dims for safety
#             capped_actions: List[np.ndarray] = []
#             cap = int(self.consensus_intelligence["max_dim"])
#             for a in actions:
#                 if a.size > cap:
#                     capped_actions.append(a[:cap].copy())
#                 else:
#                     capped_actions.append(a.copy())

#             return capped_actions, confidences
#         except Exception:
#             a = np.array([0.0], dtype=np.float32)
#             return [a, a.copy()], [0.5, 0.5]

#     # Component calculations
#     async def _calculate_comprehensive_consensus_components(
#         self, actions: List[np.ndarray], confidences: List[float]
#     ) -> Dict[str, float]:
#         try:
#             comp: Dict[str, float] = {}

#             if "cosine_agreement" in self.consensus_methods:
#                 comp["cosine_agreement"] = await self._cosine_agreement(actions, confidences)

#             if "direction_alignment" in self.consensus_methods:
#                 comp["direction_alignment"] = await self._direction_alignment(actions, confidences)

#             if "confidence_weighted" in self.consensus_methods:
#                 comp["confidence_weighted"] = await self._confidence_weighted(actions, confidences)

#             comp["magnitude_consensus"] = await self._magnitude_consensus(actions, confidences)

#             if len(actions) >= 3:
#                 comp["network_consensus"] = await self._network_consensus(actions, confidences)

#             if len(self.consensus_history) >= 3:
#                 comp["temporal_consensus"] = await self._temporal_consensus()

#             return comp
#         except Exception:
#             return {"fallback_consensus": 0.5}

#     def _pad_pair(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         L = max(a.size, b.size)
#         if L == 0:
#             return a, b
#         if a.size < L:
#             a = np.pad(a, (0, L - a.size))
#         if b.size < L:
#             b = np.pad(b, (0, L - b.size))
#         return a, b

#     async def _cosine_agreement(self, actions: List[np.ndarray], confidences: List[float]) -> float:
#         try:
#             vals: List[float] = []
#             ws: List[float] = []
#             for i in range(len(actions)):
#                 for j in range(i + 1, len(actions)):
#                     a1, a2 = self._pad_pair(actions[i], actions[j])
#                     n1 = float(np.linalg.norm(a1))
#                     n2 = float(np.linalg.norm(a2))
#                     if n1 > 1e-9 and n2 > 1e-9:
#                         sim = float(np.dot(a1, a2) / (n1 * n2))
#                         agree = (sim + 1.0) / 2.0
#                         cw = math.sqrt(max(0.0, confidences[i]) * max(0.0, confidences[j]))
#                         qw = min(confidences[i], confidences[j])
#                         vals.append(agree)
#                         ws.append(max(1e-9, cw * qw))
#             if vals and sum(ws) > 0:
#                 return float(np.clip(float(np.average(vals, weights=ws)), 0.0, 1.0))
#             return 0.5
#         except Exception:
#             return 0.5

#     async def _direction_alignment(self, actions: List[np.ndarray], confidences: List[float]) -> float:
#         try:
#             max_dims = max((a.size for a in actions), default=1)
#             scores: List[float] = []
#             for k in range(max_dims):
#                 dirs: List[int] = []
#                 confs: List[float] = []
#                 for a, c in zip(actions, confidences):
#                     if k < a.size and abs(float(a[k])) > 1e-9:
#                         dirs.append(1 if a[k] > 0 else -1)
#                         confs.append(c)
#                 if len(dirs) >= 2:
#                     pos = sum(c for d, c in zip(dirs, confs) if d > 0)
#                     neg = sum(c for d, c in zip(dirs, confs) if d < 0)
#                     tot = pos + neg
#                     if tot > 0:
#                         scores.append(abs(pos - neg) / tot)
#             return float(np.mean(scores)) if scores else 0.5
#         except Exception:
#             return 0.5

#     async def _confidence_weighted(self, actions: List[np.ndarray], confidences: List[float]) -> float:
#         try:
#             if not confidences:
#                 return 0.5
#             metrics: List[float] = []

#             # variance penalty
#             m = float(np.mean(confidences))
#             if m > 0:
#                 rv = float(np.var(confidences) / (m**2))
#                 metrics.append(max(0.0, 1.0 - 1.5 * rv))

#             # high-confidence sub-consensus
#             if len(confidences) >= 2:
#                 t = float(np.percentile(confidences, 70))
#                 idx = [i for i, c in enumerate(confidences) if c >= t]
#                 if len(idx) >= 2:
#                     sub_actions = [actions[i] for i in idx]
#                     metrics.append(await self._pairwise_agreement(sub_actions))

#             # correlation between confidence & magnitude
#             if len(actions) >= 3:
#                 mags = [float(np.linalg.norm(a)) for a in actions]
#                 if np.std(mags) > 1e-9 and np.std(confidences) > 1e-9:
#                     corr = float(np.corrcoef(confidences, mags)[0, 1])
#                     if not math.isnan(corr):
#                         metrics.append((abs(corr) + 1.0) / 2.0)

#             return float(np.mean(metrics)) if metrics else 0.5
#         except Exception:
#             return 0.5

#     async def _pairwise_agreement(self, actions: List[np.ndarray]) -> float:
#         try:
#             vals: List[float] = []
#             for i in range(len(actions)):
#                 for j in range(i + 1, len(actions)):
#                     a1, a2 = self._pad_pair(actions[i], actions[j])
#                     n1 = float(np.linalg.norm(a1))
#                     n2 = float(np.linalg.norm(a2))
#                     if n1 > 1e-9 and n2 > 1e-9:
#                         sim = float(np.dot(a1, a2) / (n1 * n2))
#                         vals.append((sim + 1.0) / 2.0)
#             return float(np.mean(vals)) if vals else 0.5
#         except Exception:
#             return 0.5

#     async def _magnitude_consensus(self, actions: List[np.ndarray], confidences: List[float]) -> float:
#         try:
#             mags = [float(np.linalg.norm(a)) for a in actions]
#             if len(mags) < 2:
#                 return 0.5

#             metrics: List[float] = []

#             # CV of confidence-weighted magnitudes
#             wmags = [m * c for m, c in zip(mags, confidences)]
#             mu = float(np.mean(wmags))
#             if mu > 1e-9:
#                 cv = float(np.std(wmags) / mu)
#                 metrics.append(max(0.0, 1.0 - cv))

#             # MAD outlier ratio
#             med = float(np.median(mags))
#             mad = float(np.median(np.abs(np.asarray(mags) - med)))
#             if mad > 1e-9:
#                 z = np.abs(np.asarray(mags) - med) / mad
#                 metrics.append(float(np.mean(z <= 2.5)))

#             # Relative range
#             if len(mags) >= 3:
#                 r = max(mags) - min(mags)
#                 am = float(np.mean(mags))
#                 if am > 1e-9:
#                     metrics.append(max(0.0, 1.0 - 0.5 * (r / am)))

#             return float(np.mean(metrics)) if metrics else 0.5
#         except Exception:
#             return 0.5

#     async def _network_consensus(self, actions: List[np.ndarray], confidences: List[float]) -> float:
#         try:
#             n = len(actions)
#             if n < 3:
#                 return 0.5

#             S = np.zeros((n, n), dtype=np.float32)
#             for i in range(n):
#                 for j in range(i + 1, n):
#                     a1, a2 = self._pad_pair(actions[i], actions[j])
#                     n1 = float(np.linalg.norm(a1))
#                     n2 = float(np.linalg.norm(a2))
#                     if n1 > 1e-9 and n2 > 1e-9:
#                         sim = float(np.dot(a1, a2) / (n1 * n2))
#                         s = (sim + 1.0) / 2.0
#                     else:
#                         s = 0.0
#                     S[i, j] = S[j, i] = s

#             cw = np.outer(confidences, confidences).astype(np.float32)
#             W = S * cw

#             metrics: List[float] = []

#             # density over > 0 entries
#             positives = W[W > 0]
#             if positives.size > 0:
#                 metrics.append(float(np.mean(positives)))

#             # clustering proxy: neighbor similarity avg
#             cluster_scores: List[float] = []
#             for i in range(n):
#                 neighbors = [j for j in range(n) if j != i and W[i, j] > 0.5]
#                 if len(neighbors) >= 2:
#                     vals = []
#                     for u in neighbors:
#                         for v in neighbors:
#                             if u < v:
#                                 vals.append(W[u, v])
#                     if vals:
#                         cluster_scores.append(float(np.mean(vals)))
#             if cluster_scores:
#                 metrics.append(float(np.mean(cluster_scores)))

#             # overall strength
#             metrics.append(float(np.mean(W)))

#             return float(np.clip(float(np.mean(metrics)) if metrics else 0.5, 0.0, 1.0))
#         except Exception:
#             return 0.5

#     async def _temporal_consensus(self) -> float:
#         try:
#             if len(self.consensus_history) < 3:
#                 return 0.5
#             recent = [float(e.get("consensus", 0.5)) for e in list(self.consensus_history)[-10:]]
#             metrics: List[float] = []

#             # stability (variance scaled)
#             mu = float(np.mean(recent))
#             var = float(np.var(recent))
#             if mu > 1e-9:
#                 metrics.append(max(0.0, 1.0 - float(var / (mu**2))))

#             # slope magnitude (smaller better)
#             if len(recent) >= 5:
#                 x = np.arange(len(recent))
#                 try:
#                     slope = float(np.polyfit(x, recent, 1)[0])
#                 except Exception:
#                     slope = 0.0
#                 metrics.append(max(0.0, 1.0 - 10.0 * abs(slope)))

#             # simple predictive accuracy
#             if len(recent) >= 4:
#                 pred = float(np.mean(recent[-3:]))
#                 err = abs(pred - recent[-1])
#                 metrics.append(max(0.0, 1.0 - 2.0 * err))

#             return float(np.mean(metrics)) if metrics else 0.5
#         except Exception:
#             return 0.5

#     # Weighting, smoothing, state update
#     async def _apply_advanced_consensus_weighting(
#         self,
#         consensus_components: Dict[str, float],
#         actions: List[np.ndarray],
#         confidences: List[float],
#         voting_data: Dict[str, Any],
#     ) -> float:
#         try:
#             if not consensus_components:
#                 return 0.5
#             weights = await self._dynamic_component_weights(consensus_components, actions, confidences, voting_data)

#             s = 0.0
#             wsum = 0.0
#             for k, v in consensus_components.items():
#                 w = float(weights.get(k, 1.0))
#                 s += float(v) * w
#                 wsum += w
#             return float(np.clip(s / wsum if wsum > 0 else float(np.mean(list(consensus_components.values()))), 0.0, 1.0))
#         except Exception:
#             return float(np.mean(list(consensus_components.values()))) if consensus_components else 0.5

#     async def _dynamic_component_weights(
#         self,
#         consensus_components: Dict[str, float],
#         actions: List[np.ndarray],
#         confidences: List[float],
#         voting_data: Dict[str, Any],
#     ) -> Dict[str, float]:
#         try:
#             w: Dict[str, float] = {}
#             for c in consensus_components:
#                 eff = float(self.consensus_algorithms.get(c, {}).get("effectiveness_threshold", 0.5))
#                 w[c] = eff

#             avg_c = float(np.mean(confidences)) if confidences else 0.5
#             regime = str(voting_data.get("market_regime", "unknown"))
#             vol = self._extract_volatility_level(voting_data.get("volatility_data", {}))

#             if avg_c > 0.8:
#                 w["cosine_agreement"] = w.get("cosine_agreement", 1.0) * 1.25
#                 w["direction_alignment"] = w.get("direction_alignment", 1.0) * 1.15
#             elif avg_c < 0.4:
#                 w["magnitude_consensus"] = w.get("magnitude_consensus", 1.0) * 1.35
#                 w["network_consensus"] = w.get("network_consensus", 1.0) * 1.15

#             if regime == "volatile" or vol in {"high", "extreme"}:
#                 w["temporal_consensus"] = w.get("temporal_consensus", 1.0) * 0.75
#             elif regime == "trending":
#                 w["direction_alignment"] = w.get("direction_alignment", 1.0) * 1.15

#             total = float(sum(w.values()))
#             if total > 0:
#                 # Normalize weights to keep average ~1.0
#                 w = {k: (v / total) * len(w) for k, v in w.items()}
#             return w
#         except Exception:
#             return {k: 1.0 for k in consensus_components.keys()}

#     async def _apply_temporal_smoothing(self, consensus: float) -> float:
#         try:
#             if not self.temporal_smoothing or len(self.consensus_history) == 0:
#                 return float(np.clip(consensus, 0.0, 1.0))
#             prev = float(self.consensus_history[-1].get("consensus", 0.5))
#             alpha = float(self.consensus_intelligence["smoothing_alpha"])
#             sm = float(alpha * consensus + (1 - alpha) * prev)
#             return float(np.clip(sm, 0.0, 1.0))
#         except Exception:
#             return float(np.clip(consensus, 0.0, 1.0))

#     async def _update_consensus_state_comprehensive(
#         self,
#         consensus: float,
#         components: Dict[str, float],
#         actions: List[np.ndarray],
#         confidences: List[float],
#     ) -> None:
#         try:
#             self.last_consensus = float(consensus)
#             self.consensus_components = dict(components)

#             self.directional_consensus = float(components.get("direction_alignment", 0.5))
#             self.magnitude_consensus = float(components.get("magnitude_consensus", 0.5))
#             self.confidence_consensus = float(components.get("confidence_weighted", 0.5))
#             self.network_consensus = float(components.get("network_consensus", 0.5))

#             if len(self.consensus_history) >= int(self.consensus_intelligence["stability_window"]):
#                 recent = [float(e.get("consensus", 0.5)) for e in list(self.consensus_history)[-int(self.consensus_intelligence["stability_window"]):]]
#                 self.temporal_stability = float(np.clip(1.0 - float(np.std(recent)), 0.0, 1.0))

#             await self._update_consensus_statistics_comprehensive(consensus, components)
#         except Exception:
#             pass

#     async def _calculate_advanced_consensus_quality(
#         self,
#         actions: List[np.ndarray],
#         confidences: List[float],
#         components: Dict[str, float],
#     ) -> float:
#         try:
#             q_parts: List[float] = []

#             # coherence
#             if len(components) > 1:
#                 vals = list(components.values())
#                 mu = float(np.mean(vals))
#                 std = float(np.std(vals))
#                 coherence = float(np.clip(1.0 - (std / max(mu, 1e-6)), 0.0, 1.0))
#                 self.consensus_quality_metrics["coherence"] = coherence
#                 q_parts.append(coherence)

#             # stability
#             if len(self.consensus_history) >= int(self.consensus_intelligence["stability_window"]):
#                 recent = [float(e.get("consensus", 0.5)) for e in list(self.consensus_history)[-int(self.consensus_intelligence["stability_window"]):]]
#                 stability = float(np.clip(1.0 - float(np.std(recent)), 0.0, 1.0))
#                 self.consensus_quality_metrics["stability"] = stability
#                 q_parts.append(stability)

#             # diversity
#             diversity = await self._input_diversity(actions)
#             self.consensus_quality_metrics["diversity"] = diversity
#             q_parts.append(diversity)

#             # reliability (confidence level + dispersion)
#             if confidences:
#                 avg_c = float(np.mean(confidences))
#                 disp = float(np.std(confidences))
#                 reliability = float(np.clip((avg_c + (1.0 - disp)) / 2.0, 0.0, 1.0))
#                 self.consensus_quality_metrics["reliability"] = reliability
#                 q_parts.append(reliability)

#             # predictive accuracy (history)
#             if len(self.prediction_history) >= 3:
#                 pa = await self._prediction_accuracy()
#                 self.consensus_quality_metrics["predictive_accuracy"] = pa
#                 q_parts.append(pa)

#             # temporal consistency (autocorr proxy)
#             if len(self.consensus_history) >= 5:
#                 tc = await self._temporal_consistency_score()
#                 self.consensus_quality_metrics["temporal_consistency"] = tc
#                 q_parts.append(tc)

#             overall = float(np.mean(q_parts)) if q_parts else 0.5
#             self.consensus_quality = float(np.clip(overall, 0.0, 1.0))

#             # derive overall_effectiveness for reporting
#             weights = np.array([0.25, 0.20, 0.15, 0.20, 0.10, 0.10], dtype=np.float32)
#             values = np.array(
#                 [
#                     self.quality_metrics.get("coherence", 0.5),
#                     self.quality_metrics.get("stability", 0.5),
#                     self.quality_metrics.get("diversity", 0.5),
#                     self.quality_metrics.get("reliability", 0.5),
#                     self.quality_metrics.get("predictive_accuracy", 0.5),
#                     self.quality_metrics.get("temporal_consistency", 0.5),
#                 ],
#                 dtype=np.float32,
#             )
#             self.consensus_quality_metrics["overall_effectiveness"] = float(np.dot(weights, values))
#             return self.consensus_quality
#         except Exception:
#             return 0.5

#     async def _input_diversity(self, actions: List[np.ndarray]) -> float:
#         try:
#             if len(actions) < 2:
#                 return 0.0
#             metrics: List[float] = []

#             # pairwise distances (normalized)
#             dists: List[float] = []
#             for i in range(len(actions)):
#                 for j in range(i + 1, len(actions)):
#                     a1, a2 = self._pad_pair(actions[i], actions[j])
#                     dists.append(float(np.linalg.norm(a1 - a2)))
#             if dists:
#                 dmax = max(dists)
#                 if dmax > 1e-9:
#                     metrics.append(float(np.mean(dists) / dmax))

#             # direction entropy (1D sign diversity)
#             signs = [int(1 if a[0] > 0 else -1) for a in actions if a.size > 0 and abs(float(a[0])) > 1e-9]
#             if len(signs) > 1:
#                 unique = len(set(signs))
#                 metrics.append(float(unique / min(2, len(signs))))

#             # magnitude CV
#             mags = [float(np.linalg.norm(a)) for a in actions]
#             if len(mags) > 1 and float(np.mean(mags)) > 1e-9:
#                 metrics.append(float(np.clip(float(np.std(mags) / np.mean(mags)), 0.0, 1.0)))

#             return float(np.mean(metrics)) if metrics else 0.5
#         except Exception:
#             return 0.5

#     async def _prediction_accuracy(self) -> float:
#         try:
#             if len(self.prediction_history) < 3:
#                 return 0.5
#             errors: List[float] = []
#             for ev in self.prediction_history:
#                 p = float(ev.get("predicted_consensus", 0.5))
#                 a = float(ev.get("actual_consensus", 0.5))
#                 errors.append(abs(p - a))
#             if errors:
#                 return float(np.clip(1.0 - 2.0 * float(np.mean(errors)), 0.0, 1.0))
#             return 0.5
#         except Exception:
#             return 0.5

#     async def _temporal_consistency_score(self) -> float:
#         try:
#             if len(self.consensus_history) < 5:
#                 return 0.5
#             recent = [float(e.get("consensus", 0.5)) for e in list(self.consensus_history)[-10:]]
#             if len(recent) >= 3:
#                 c = float(np.corrcoef(recent[:-1], recent[1:])[0, 1])
#                 if not math.isnan(c):
#                     return float((abs(c) + 1.0) / 2.0)
#             return 0.5
#         except Exception:
#             return 0.5

#     async def _record_consensus_event_comprehensive(
#         self,
#         consensus: float,
#         components: Dict[str, float],
#         consensus_quality: float,
#         voting_data: Dict[str, Any],
#     ) -> None:
#         try:
#             self.consensus_history.append(
#                 {
#                     "timestamp": dt.datetime.now().isoformat(),
#                     "consensus": float(consensus),
#                     "consensus_quality": float(consensus_quality),
#                     "components": dict(components),
#                     "directional_consensus": float(self.directional_consensus),
#                     "magnitude_consensus": float(self.magnitude_consensus),
#                     "confidence_consensus": float(self.confidence_consensus),
#                     "network_consensus": float(self.network_consensus),
#                     "temporal_stability": float(self.temporal_stability),
#                     "member_count": int(len(voting_data.get("raw_proposals", []))),
#                     "avg_confidence": float(np.mean(voting_data.get("member_confidences", [0.5]))),
#                     "market_regime": str(voting_data.get("market_regime", "unknown")),
#                     "agreement_score": float(voting_data.get("agreement_score", 0.5)),
#                     "quality_metrics": dict(self.quality_metrics),
#                 }
#             )
#         except Exception:
#             pass

#     async def _update_consensus_statistics_comprehensive(
#         self, consensus: float, components: Dict[str, float]
#     ) -> None:
#         try:
#             if consensus > 0.7:
#                 self.consensus_stats["high_consensus_count"] += 1
#             elif consensus < 0.3:
#                 self.consensus_stats["low_consensus_count"] += 1

#             total = int(self.consensus_stats["total_computations"])
#             if total > 0:
#                 self.consensus_stats["avg_consensus"] = float(
#                     (self.consensus_stats["avg_consensus"] * (total - 1) + consensus) / total
#                 )

#             if len(self.consensus_history) >= 10:
#                 recent = [float(e.get("consensus", 0.5)) for e in list(self.consensus_history)[-10:]]
#                 self.consensus_stats["consensus_volatility"] = float(np.std(recent))

#             self.consensus_stats["quality_score"] = float(self.consensus_quality)

#             # perf metrics
#             self._update_performance_metric("consensus_score", float(consensus))
#             self._update_performance_metric("consensus_quality", float(self.consensus_quality))
#             self._update_performance_metric("directional_consensus", float(self.directional_consensus))
#             self._update_performance_metric("temporal_stability", float(self.temporal_stability))
#         except Exception:
#             pass

#     # ────────────────────────────
#     # MEMBER CONTRIBUTIONS
#     # ────────────────────────────
#     async def _update_member_contributions_comprehensive(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             updates = {"updated_members": [], "influence_changes": {}, "reliability_updates": {}}
#             raw = voting_data.get("raw_proposals", [])
#             confs = voting_data.get("member_confidences", [])

#             if len(raw) < 2:
#                 return updates

#             mem = float(self.consensus_intelligence["temporal_memory"])

#             for i, proposal in enumerate(raw[: self.n_members]):
#                 if not isinstance(proposal, (list, np.ndarray)) or len(proposal) == 0:
#                     continue
#                 p = np.asarray(proposal, dtype=np.float32).flatten()
#                 aligns: List[float] = []
#                 for j, other in enumerate(raw[: self.n_members]):
#                     if i == j or not isinstance(other, (list, np.ndarray)) or len(other) == 0:
#                         continue
#                     q = np.asarray(other, dtype=np.float32).flatten()
#                     a1, a2 = self._pad_pair(p, q)
#                     n1 = float(np.linalg.norm(a1))
#                     n2 = float(np.linalg.norm(a2))
#                     if n1 > 1e-9 and n2 > 1e-9:
#                         sim = float(np.dot(a1, a2) / (n1 * n2))
#                         aligns.append((sim + 1.0) / 2.0)

#                 if not aligns:
#                     continue

#                 contrib = self.member_contributions[i]
#                 old_align = float(contrib.get("avg_alignment", 0.5))
#                 new_align = float(np.mean(aligns))
#                 contrib["avg_alignment"] = float(old_align * mem + new_align * (1 - mem))
#                 contrib["consistency"] = float(np.clip(1.0 - float(np.std(aligns)), 0.0, 1.0))

#                 # contribution delta (with vs without)
#                 contrib["consensus_contribution"] = await self._member_contribution_delta(i, raw)

#                 if i < len(confs):
#                     conf = float(np.clip(confs[i], 0.0, 1.0))
#                     contrib["reliability_score"] = float(
#                         np.clip((conf + contrib["consistency"] + contrib["avg_alignment"]) / 3.0, 0.0, 1.0)
#                     )

#                 base_w = 1.0 / max(self.n_members, 1)
#                 qual_mult = float((contrib["reliability_score"] + contrib["consensus_contribution"]) / 2.0)
#                 old_w = float(contrib.get("influence_weight", base_w))
#                 contrib["influence_weight"] = float(base_w * qual_mult)

#                 if abs(new_align - old_align) > 0.1 or abs(contrib["influence_weight"] - old_w) > 0.05 * base_w:
#                     updates["updated_members"].append(i)
#                     updates["influence_changes"][i] = {
#                         "old_influence": old_w,
#                         "new_influence": contrib["influence_weight"],
#                         "change_reason": "alignment_update",
#                     }

#             return updates
#         except Exception:
#             return {"updated_members": [], "influence_changes": {}, "reliability_updates": {}}

#     async def _member_contribution_delta(self, idx: int, all_props: List[Any]) -> float:
#         try:
#             with_idx = [np.asarray(p, dtype=np.float32).flatten() for p in all_props if isinstance(p, (list, np.ndarray))]
#             if len(with_idx) < 2:
#                 return 0.5
#             c_with = await self._pairwise_agreement(with_idx)

#             wo = [
#                 np.asarray(p, dtype=np.float32).flatten()
#                 for i, p in enumerate(all_props)
#                 if i != idx and isinstance(p, (list, np.ndarray))
#             ]
#             if len(wo) >= 2:
#                 c_wo = await self._pairwise_agreement(wo)
#                 delta = float(np.clip(((c_with - c_wo) + 1.0) / 2.0, 0.0, 1.0))
#                 return delta
#             return 0.5
#         except Exception:
#             return 0.5

#     # ────────────────────────────
#     # TRENDS & QUALITY
#     # ────────────────────────────
#     async def _analyze_consensus_trends_comprehensive(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             out: Dict[str, Any] = {
#                 "current_trend": "stable",
#                 "trend_strength": 0.0,
#                 "prediction": {},
#                 "pattern_analysis": {},
#                 "regime_analysis": {},
#             }
#             if len(self.consensus_history) < 5:
#                 out["status"] = "insufficient_history"
#                 return out

#             recent = [float(e.get("consensus", 0.5)) for e in list(self.consensus_history)[-10:]]
#             if len(recent) >= 3:
#                 x = np.arange(len(recent))
#                 try:
#                     slope, intercept = np.polyfit(x, recent, 1)
#                 except Exception:
#                     slope, intercept = 0.0, float(recent[-1])
#                 out["trend_strength"] = float(abs(slope))
#                 if slope > 0.02:
#                     out["current_trend"] = "increasing"
#                 elif slope < -0.02:
#                     out["current_trend"] = "decreasing"
#                 else:
#                     out["current_trend"] = "stable"

#                 nxt = float(np.clip(slope * len(recent) + intercept, 0.0, 1.0))
#                 out["prediction"] = {
#                     "next_consensus": nxt,
#                     "confidence": float(np.clip(1.0 - 5.0 * abs(slope), 0.0, 1.0)),
#                     "trend_slope": float(slope),
#                 }
#                 self.prediction_history.append(
#                     {
#                         "timestamp": dt.datetime.now().isoformat(),
#                         "predicted_consensus": nxt,
#                         "actual_consensus": float(self.last_consensus),
#                         "prediction_method": "linear_trend",
#                     }
#                 )

#             regime = str(voting_data.get("market_regime", "unknown"))
#             self.regime_consensus_history[regime].append(float(self.last_consensus))
#             if len(self.regime_consensus_history[regime]) >= 3:
#                 seq = list(self.regime_consensus_history[regime])
#                 out["regime_analysis"][regime] = {
#                     "avg_consensus": float(np.mean(seq)),
#                     "consensus_volatility": float(np.std(seq)),
#                     "sample_count": int(len(seq)),
#                 }

#             self.consensus_trends.append(
#                 {
#                     "timestamp": dt.datetime.now().isoformat(),
#                     "trend_direction": out["current_trend"],
#                     "trend_strength": out["trend_strength"],
#                     "current_consensus": float(self.last_consensus),
#                     "regime": regime,
#                     "quality": float(self.consensus_quality),
#                 }
#             )
#             return out
#         except Exception:
#             return {"current_trend": "unknown", "status": "analysis_error"}

#     async def _calculate_comprehensive_quality_metrics(self) -> Dict[str, Any]:
#         try:
#             out = {
#                 **self.quality_metrics,
#                 "overall_quality_score": float(self.consensus_quality),
#                 "quality_trend": "unknown",
#                 "quality_drivers": {},
#                 "improvement_areas": [],
#             }

#             if len(self.consensus_history) >= 5:
#                 q_recent = [float(e.get("consensus_quality", 0.5)) for e in list(self.consensus_history)[-5:]]
#                 if len(q_recent) >= 3:
#                     x = np.arange(len(q_recent))
#                     try:
#                         slope = float(np.polyfit(x, q_recent, 1)[0])
#                     except Exception:
#                         slope = 0.0
#                     out["quality_trend"] = "improving" if slope > 0.02 else "declining" if slope < -0.02 else "stable"

#             drivers = [(k, v) for k, v in self.quality_metrics.items() if k != "overall_effectiveness" and v > 0.7]
#             drivers.sort(key=lambda x: x[1], reverse=True)
#             out["quality_drivers"] = dict(drivers[:3])

#             improve = [(k, v) for k, v in self.quality_metrics.items() if k != "overall_effectiveness" and v < 0.5]
#             improve.sort(key=lambda x: x[1])
#             out["improvement_areas"] = [k for k, _ in improve[:3]]

#             return out
#         except Exception:
#             return {"overall_quality_score": 0.5, "quality_trend": "unknown"}

#     # ────────────────────────────
#     # RECOMMENDATIONS & THESIS
#     # ────────────────────────────
#     async def _generate_intelligent_consensus_recommendations(
#         self, consensus_analysis: Dict[str, Any], quality_analysis: Dict[str, Any], trend_analysis: Dict[str, Any]
#     ) -> List[str]:
#         try:
#             recs: List[str] = []
#             score = float(consensus_analysis.get("consensus_score", 0.5))
#             qual = float(quality_analysis.get("overall_quality_score", 0.5))
#             trend = str(trend_analysis.get("current_trend", "stable"))

#             if score > 0.9:
#                 recs.append("VERY HIGH CONSENSUS: Ensure decisions aren't overly homogeneous")
#             elif score > 0.8:
#                 recs.append("HIGH CONSENSUS: Suitable for decisive action")
#             elif score < 0.2:
#                 recs.append("CRITICAL: Very low consensus — postpone decision or add analysis")
#             elif score < 0.3:
#                 recs.append("LOW CONSENSUS: Facilitate discussion; explore dissenting views")

#             if qual < 0.4:
#                 recs.append("QUALITY: Low consensus quality — review inputs & calibration")
#             for area in quality_analysis.get("improvement_areas", []):
#                 if area == "reliability":
#                     recs.append("RELIABILITY: Calibrate or train members; review confidence usage")
#                 if area == "diversity":
#                     recs.append("DIVERSITY: Encourage broader viewpoints or rotate members")
#                 if area == "stability":
#                     recs.append("STABILITY: Consider slightly higher smoothing α or process tweaks")

#             if trend == "decreasing":
#                 recs.append("TREND: Consensus declining — investigate sources of conflict")
#             elif trend == "increasing":
#                 recs.append("TREND: Consensus improving — current approach appears effective")

#             # Member contribution check
#             low_contrib = [m for m, c in self.member_contributions.items() if c.get("consensus_contribution", 0.5) < 0.3]
#             if len(low_contrib) > max(1, self.n_members // 3):
#                 recs.append(f"MEMBERS: {len(low_contrib)} members with low contribution — consider coaching/rotation")

#             # Param hints
#             alpha = float(self.consensus_intelligence.get("smoothing_alpha", 0.3))
#             if alpha > 0.6:
#                 recs.append("PARAMETERS: High smoothing may mask important changes")
#             elif alpha < 0.1:
#                 recs.append("PARAMETERS: Low smoothing may cause excess volatility")

#             return recs[:6] if recs else ["SYSTEM: Consensus detection within normal parameters"]
#         except Exception as e:
#             ctx = self.error_pinpointer.analyze_error(e, "consensus_recommendations")
#             return [f"Recommendation generation failed: {ctx}"]

#     async def _generate_comprehensive_consensus_thesis(
#         self, consensus_analysis: Dict[str, Any], quality_analysis: Dict[str, Any], recommendations: List[str]
#     ) -> str:
#         try:
#             score = float(consensus_analysis.get("consensus_score", self.last_consensus))
#             qual = float(quality_analysis.get("overall_quality_score", self.consensus_quality))
#             members = int(consensus_analysis.get("member_count", self.n_members))
#             lvl = "HIGH" if score > 0.7 else "MODERATE" if score > 0.4 else "LOW"
#             parts = [
#                 f"CONSENSUS: {lvl} ({score:.1%})",
#                 f"QUALITY: {qual:.1%} across {len(self.consensus_components)} components",
#             ]
#             if self.consensus_components:
#                 k, v = max(self.consensus_components.items(), key=lambda kv: kv[1])
#                 parts.append(f"LEADING COMPONENT: {k} at {v:.1%}")
#             parts.append(f"MEMBERS: {members} evaluated")
#             if self.temporal_stability > 0:
#                 parts.append(f"STABILITY: {self.temporal_stability:.1%}")
#             parts.append(f"RUNS: {self.consensus_stats.get('total_computations', 0)} computations")
#             pri = [r for r in recommendations if any(s in r for s in ["CRITICAL", "HIGH", "LOW CONSENSUS"])]
#             if pri:
#                 parts.append(f"ACTION ITEMS: {len(pri)} priority recs")
#             return " | ".join(parts)
#         except Exception as e:
#             ctx = self.error_pinpointer.analyze_error(e, "consensus_thesis_generation")
#             return f"Consensus thesis generation failed: {ctx}"

#     async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str) -> None:
#         try:
#             s = self.smart_bus.set

#             # Canonical numeric consensus (single-writer)
#             s("consensus_score", results["consensus_score"], module="ConsensusDetector", thesis=thesis)

#             # NOTE: Do NOT write 'voting_consensus' to the bus. (Kept only as a return alias for compat.)

#             # Contract-critical diagnostics (all safe single-writer keys under our module)
#             s("consensus_detector_initialization", results["consensus_detector_initialization"],
#             module="ConsensusDetector", thesis="Initialization payload")
#             s("consensus_quality", results["consensus_quality"], module="ConsensusDetector",
#             thesis=f"Consensus quality: {results['consensus_quality']:.3f}")
#             s("consensus_components", results["consensus_components"], module="ConsensusDetector",
#             thesis=f"Components analyzed: {len(results['consensus_components'])}")
#             s("directional_consensus", results["directional_consensus"], module="ConsensusDetector",
#             thesis=f"Directional alignment: {results['directional_consensus']:.3f}")
#             s("magnitude_consensus", results["magnitude_consensus"], module="ConsensusDetector",
#             thesis=f"Magnitude agreement: {results['magnitude_consensus']:.3f}")
#             s("confidence_consensus", results["confidence_consensus"], module="ConsensusDetector",
#             thesis=f"Confidence-weighted: {results['confidence_consensus']:.3f}")
#             s("member_contributions", results["member_contributions"], module="ConsensusDetector",
#             thesis=f"Member contributions: {len(results['member_contributions'])} profiles")
#             s("consensus_trends", results["consensus_trends"], module="ConsensusDetector",
#             thesis=f"Trends tracked: {len(self.consensus_trends)} points")
#             s("consensus_quality_metrics", results["quality_metrics"], module="ConsensusDetector",
#             thesis=f"Quality metrics: {len(results['quality_metrics'])} dimensions")
#             s("consensus_recommendations", results["consensus_recommendations"], module="ConsensusDetector",
#             thesis=f"Recommendations: {len(results['consensus_recommendations'])}")
            
#             # FIX #2: Publish decision_id for kernel coordination
#             if results.get("decision_id"):
#                 s("consensus_decision_id", results["decision_id"], module="ConsensusDetector",
#                   thesis=f"Consensus decision ID: {results['decision_id']}")
#         except Exception as e:
#             ctx = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
#             self.logger.error(f"SmartInfoBus update failed: {ctx}")


#     # ────────────────────────────
#     # PUBLIC / LEGACY API
#     # ────────────────────────────
#     def compute_consensus(self, actions: List[np.ndarray], confidences: List[float]) -> float:
#         """Synchronous wrapper for legacy callers."""
#         try:
#             import asyncio

#             if asyncio.get_event_loop().is_running():
#                 return self._simple_consensus_computation_fallback(actions, confidences)
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             try:
#                 voting_data = {
#                     "raw_proposals": actions,
#                     "member_confidences": confidences,
#                     "agreement_score": 0.5,
#                     "market_regime": "unknown",
#                 }
#                 res = loop.run_until_complete(self._perform_comprehensive_consensus_analysis(voting_data))
#                 return float(res.get("consensus_score", 0.5))
#             finally:
#                 loop.close()
#         except Exception:
#             return self._simple_consensus_computation_fallback(actions, confidences)

#     def _simple_consensus_computation_fallback(self, actions: List[np.ndarray], confidences: List[float]) -> float:
#         """Minimal, safe cosine-average fallback with optional confidence weighting."""
#         try:
#             if not actions or len(actions) < 2:
#                 return 0.5
#             vals: List[float] = []
#             ws: List[float] = []
#             for i in range(len(actions)):
#                 for j in range(i + 1, len(actions)):
#                     a1 = np.asarray(actions[i], dtype=np.float32).flatten()
#                     a2 = np.asarray(actions[j], dtype=np.float32).flatten()
#                     a1, a2 = self._pad_pair(a1, a2)
#                     n1 = float(np.linalg.norm(a1))
#                     n2 = float(np.linalg.norm(a2))
#                     if n1 > 1e-9 and n2 > 1e-9:
#                         sim = float(np.dot(a1, a2) / (n1 * n2))
#                         agree = (sim + 1.0) / 2.0
#                         vals.append(agree)
#                         if i < len(confidences) and j < len(confidences):
#                             ws.append(float(confidences[i] * confidences[j]))
#                         else:
#                             ws.append(1.0)
#             if vals and sum(ws) > 0:
#                 c = float(np.average(vals, weights=ws))
#                 if self.temporal_smoothing and len(self.consensus_history) > 0:
#                     prev = float(self.consensus_history[-1].get("consensus", 0.5))
#                     alpha = float(self.consensus_intelligence.get("smoothing_alpha", 0.3))
#                     c = float(alpha * c + (1 - alpha) * prev)
#                 self.last_consensus = float(np.clip(c, 0.0, 1.0))
#                 return self.last_consensus
#             return 0.5
#         except Exception:
#             return 0.5

#     def resize(self, n_members: int) -> None:
#         old = self.n_members
#         self.n_members = int(n_members)
#         if abs(self.n_members - old) > 2:
#             self.member_contributions.clear()
#         self.logger.info(
#             format_operator_message(icon="[RELOAD]", message="Consensus Detector resized", old_members=old, new_members=self.n_members)
#         )

#     # ────────────────────────────
#     # SUMMARIES & HEALTH
#     # ────────────────────────────
#     def _get_member_contributions_summary(self) -> Dict[str, Any]:
#         try:
#             out: Dict[str, Any] = {}
#             for mid, c in self.member_contributions.items():
#                 out[f"member_{int(mid)}"] = {
#                     "avg_alignment": float(c.get("avg_alignment", 0.5)),
#                     "consistency": float(c.get("consistency", 0.5)),
#                     "influence_weight": float(c.get("influence_weight", 1.0 / max(self.n_members, 1))),
#                     "consensus_contribution": float(c.get("consensus_contribution", 0.5)),
#                     "reliability_score": float(c.get("reliability_score", 0.5)),
#                 }
#             return out
#         except Exception:
#             return {}

#     def _get_consensus_trends_summary(self) -> Dict[str, Any]:
#         try:
#             if not self.consensus_trends:
#                 return {"status": "no_trends"}
#             recent = list(self.consensus_trends)[-5:]
#             return {
#                 "recent_trend_count": len(recent),
#                 "current_trend": str(recent[-1].get("trend_direction", "unknown")),
#                 "trend_strength": float(recent[-1].get("trend_strength", 0.0)),
#                 "avg_consensus_trend": float(np.mean([t.get("current_consensus", 0.5) for t in recent])),
#                 "consensus_volatility": float(np.std([t.get("current_consensus", 0.5) for t in recent])) if len(recent) > 1 else 0.0,
#             }
#         except Exception:
#             return {"status": "analysis_error"}

#     def get_observation_components(self) -> np.ndarray:
#         try:
#             feats = np.asarray(
#                 [
#                     float(self.last_consensus),
#                     float(self.consensus_quality),
#                     float(self.directional_consensus),
#                     float(self.magnitude_consensus),
#                     float(self.confidence_consensus),
#                     float(self.temporal_stability),
#                     float(len(self.consensus_history) / 150.0),
#                     float(self.consensus_stats.get("consensus_volatility", 0.0)),
#                 ],
#                 dtype=np.float32,
#             )
#             if np.any(~np.isfinite(feats)):
#                 self.logger.error(f"Invalid consensus observation: {feats}")
#                 feats = np.nan_to_num(feats, nan=0.5)
#             return feats
#         except Exception:
#             return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)

#     def get_health_metrics(self) -> Dict[str, Any]:
#         return {
#             "module_name": "ConsensusDetector",
#             "status": "disabled" if self.is_disabled else "healthy",
#             "error_count": int(self.error_count),
#             "circuit_breaker_threshold": int(self.circuit_breaker_threshold),
#             "total_computations": int(self.consensus_stats.get("total_computations", 0)),
#             "consensus_score": float(self.last_consensus),
#             "consensus_quality": float(self.consensus_quality),
#             "consensus_volatility": float(self.consensus_stats.get("consensus_volatility", 0.0)),
#             "high_consensus_count": int(self.consensus_stats.get("high_consensus_count", 0)),
#             "low_consensus_count": int(self.consensus_stats.get("low_consensus_count", 0)),
#             "member_contributions_tracked": int(len(self.member_contributions)),
#             "consensus_history_length": int(len(self.consensus_history)),
#             "temporal_stability": float(self.temporal_stability),
#             "session_duration": (dt.datetime.now() - dt.datetime.fromisoformat(self.consensus_stats["session_start"])).total_seconds() / 3600.0,
#         }

#     def _get_health_metrics(self) -> Dict[str, Any]:
#         return self.get_health_metrics()

#     def get_consensus_report(self) -> str:
#         # Levels
#         if self.last_consensus > 0.8:
#             lvl = "[GREEN] HIGH"
#         elif self.last_consensus > 0.6:
#             lvl = "[YELLOW] MODERATE"
#         elif self.last_consensus > 0.4:
#             lvl = "🟠 LOW-MODERATE"
#         elif self.last_consensus > 0.2:
#             lvl = "🟠 LOW"
#         else:
#             lvl = "[RED] VERY LOW"

#         if self.consensus_quality > 0.8:
#             qlvl = "[OK] Excellent"
#         elif self.consensus_quality > 0.6:
#             qlvl = "[FAST] Good"
#         elif self.consensus_quality > 0.4:
#             qlvl = "[WARN] Fair"
#         else:
#             qlvl = "[ALERT] Poor"

#         trend_status = "📭 No data"
#         if len(self.consensus_trends) > 0:
#             td = self.consensus_trends[-1].get("trend_direction", "unknown")
#             trend_status = "[CHART] Improving" if td == "increasing" else "📉 Declining" if td == "decreasing" else "→ Stable"

#         comp_lines: List[str] = []
#         for m, s in self.consensus_components.items():
#             tag = "[OK]" if s > 0.7 else "[FAST]" if s > 0.5 else "[WARN]" if s > 0.3 else "[ALERT]"
#             comp_lines.append(f"  {tag} {m.replace('_', ' ').title()}: {s:.1%}")

#         eff = float(self.quality_metrics.get("overall_effectiveness", 0.0))
#         eff_status = "[OK] Excellent" if eff > 0.8 else "[FAST] Good" if eff > 0.6 else "[WARN] Fair" if eff > 0.4 else "[ALERT] Poor"

#         high_c = sum(1 for c in self.member_contributions.values() if c.get("consensus_contribution", 0.5) > 0.7)
#         low_c = sum(1 for c in self.member_contributions.values() if c.get("consensus_contribution", 0.5) < 0.3)

#         return f"""
# 🤝 CONSENSUS DETECTOR v3.1
# ═══════════════════════════════════════════════════════════════
# [STATS] Current Consensus: {lvl} ({self.last_consensus:.1%})
# [TARGET] Quality Level: {qlvl} ({self.consensus_quality:.1%})
# [CHART] Trend: {trend_status}

# [STATS] Consensus Components:
# {chr(10).join(comp_lines) if comp_lines else "  📭 No components available"}

# [TARGET] Advanced Metrics:
# • Directional: {self.directional_consensus:.1%}
# • Magnitude: {self.magnitude_consensus:.1%}
# • Confidence: {self.confidence_consensus:.1%}
# • Network: {self.network_consensus:.1%}
# • Temporal Stability: {self.temporal_stability:.1%}

# [TARGET] Quality Dimensions:
# • Coherence: {self.quality_metrics.get('coherence', 0):.1%}
# • Stability: {self.quality_metrics.get('stability', 0):.1%}
# • Diversity: {self.quality_metrics.get('diversity', 0):.1%}
# • Reliability: {self.quality_metrics.get('reliability', 0):.1%}
# • Predictive Accuracy: {self.quality_metrics.get('predictive_accuracy', 0):.1%}
# • Temporal Consistency: {self.quality_metrics.get('temporal_consistency', 0):.1%}
# • Overall Effectiveness: {eff_status} ({eff:.1%})

# [CHART] Performance Statistics:
# • Total Computations: {self.consensus_stats['total_computations']}
# • High Consensus Events: {self.consensus_stats['high_consensus_count']}
# • Low Consensus Events: {self.consensus_stats['low_consensus_count']}
# • Average Consensus: {self.consensus_stats['avg_consensus']:.1%}
# • Consensus Volatility: {self.consensus_stats['consensus_volatility']:.3f}

# 👥 Member Analysis:
# • Committee Size: {self.n_members} members
# • Contributions Tracked: {len(self.member_contributions)}
# • Contributor Summary: {high_c} high, {low_c} low contributors
# • Average Influence: {np.mean([c.get('influence_weight', 0) for c in self.member_contributions.values()]) if self.member_contributions else 0.0:.3f}

# ⚙️ System Configuration:
# • Consensus Threshold: {self.threshold:.1%}
# • Methods Active: {', '.join(self.consensus_methods)}
# • Quality Weighting: {'[OK] Enabled' if self.quality_weighting else '[FAIL] Disabled'}
# • Temporal Smoothing: {'[OK] Enabled' if self.temporal_smoothing else '[FAIL] Disabled'}
# • Smoothing Alpha: {self.consensus_intelligence.get('smoothing_alpha', 0.3):.3f}
# • Stability Window: {self.consensus_intelligence.get('stability_window', 10)} steps
# """

#     def get_health_status(self) -> Dict[str, Any]:
#         return {
#             "module_name": "ConsensusDetector",
#             "status": "disabled" if self.is_disabled else "healthy",
#             "metrics": self._get_health_metrics(),
#             "alerts": self._generate_health_alerts(),
#             "recommendations": self._generate_health_recommendations(),
#         }

#     def _generate_health_alerts(self) -> List[Dict[str, Any]]:
#         alerts: List[Dict[str, Any]] = []
#         if self.is_disabled:
#             alerts.append(
#                 {
#                     "severity": "critical",
#                     "message": "ConsensusDetector disabled due to errors",
#                     "action": "Inspect logs and restart module",
#                 }
#             )
#         if self.error_count > 2:
#             alerts.append({"severity": "warning", "message": f"Elevated error count: {self.error_count}", "action": "Monitor stability"})
#         if self.last_consensus < 0.2:
#             alerts.append({"severity": "warning", "message": f"Very low consensus: {self.last_consensus:.1%}", "action": "Facilitate mediation"})
#         if self.consensus_quality < 0.3:
#             alerts.append({"severity": "warning", "message": f"Low consensus quality: {self.consensus_quality:.1%}", "action": "Review inputs & methods"})
#         if float(self.consensus_stats.get("consensus_volatility", 0.0)) > 0.3:
#             alerts.append({"severity": "warning", "message": "High consensus volatility", "action": "Increase smoothing or stabilize process"})
#         if len(self.consensus_history) < 5:
#             alerts.append({"severity": "info", "message": "Limited history — reliability improving with time", "action": "Continue operations"})
#         return alerts

#     def _generate_health_recommendations(self) -> List[str]:
#         recs: List[str] = []
#         if self.is_disabled:
#             recs.append("Restart ConsensusDetector after resolving errors")
#         if len(self.consensus_history) < 10:
#             recs.append("Build more history for robust analytics")
#         if self.consensus_quality < 0.5:
#             recs.append("Tune algorithm weights & input quality")
#         alpha = float(self.consensus_intelligence.get("smoothing_alpha", 0.3))
#         if alpha > 0.6:
#             recs.append("High smoothing may hide signal changes")
#         elif alpha < 0.1:
#             recs.append("Low smoothing may cause excessive volatility")
#         low_contrib = sum(1 for c in self.member_contributions.values() if c.get("consensus_contribution", 0.5) < 0.3)
#         if low_contrib > max(1, self.n_members // 2):
#             recs.append(f"Many members ({low_contrib}) show low contribution — coaching/rotation suggested")
#         if float(self.consensus_stats.get("consensus_volatility", 0.0)) > 0.4:
#             recs.append("High volatility — consider process stabilization")
#         if not recs:
#             recs.append("ConsensusDetector operating within normal parameters")
#         return recs

#     # ────────────────────────────
#     # ERRORS & DISABLED RESPONSE
#     # ────────────────────────────
#     async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
#         self.error_count += 1
#         ctx = self.error_pinpointer.analyze_error(error, "ConsensusDetector")
#         if self.error_count >= self.circuit_breaker_threshold:
#             self.is_disabled = True
#             self.logger.error(
#                 format_operator_message(
#                     icon="[ALERT]",
#                     message="Consensus Detector disabled due to repeated errors",
#                     error_count=self.error_count,
#                     threshold=self.circuit_breaker_threshold,
#                 )
#             )
#         self.performance_tracker.record_metric("ConsensusDetector", "process_time_ms", (time.time() - start_time) * 1000.0, False)
#         return self._generate_error_response(ctx)

#     def _generate_error_response(self, ctx: Any) -> Dict[str, Any]:
#         return {
#             "consensus_score": 0.5,
#             "voting_consensus": 0.5,  # keep contract surface even in error
#             "consensus_quality": 0.5,
#             "consensus_components": {},
#             "directional_consensus": 0.5,
#             "magnitude_consensus": 0.5,
#             "confidence_consensus": 0.5,
#             "member_contributions": {},
#             "consensus_trends": {},
#             "quality_metrics": {"error": str(ctx)},
#             "consensus_recommendations": ["Investigate consensus detector errors"],
#             "health_metrics": {"status": "error", "error_context": str(ctx)},
#             "consensus_detector_initialization": dict(
#                 getattr(self, "_init_payload", {
#                     "status": "initialized",
#                     "timestamp": dt.datetime.now().isoformat(),
#                     "configuration": {
#                         "members": self.n_members,
#                         "threshold": self.threshold,
#                         "consensus_methods": list(self.consensus_methods),
#                     },
#                 })
#             ),
#             "_thesis": f"ConsensusDetector error: {ctx}",
#         }


#     def _get_safe_voting_defaults(self) -> Dict[str, Any]:
#         return {
#             "proposal_vectors": [],
#             "votes": [],
#             "raw_proposals": [],
#             "member_confidences": [],
#             "voting_summary": {},
#             "alpha_weights": [],
#             "blended_action": [],
#             "market_context": {},
#             "agreement_score": 0.5,
#             "consensus_direction": "neutral",
#             "market_regime": "unknown",
#             "volatility_data": {},
#             "decision_id": None,
#             "tick_ts": None,
#         }


#     def _generate_disabled_response(self) -> Dict[str, Any]:
#         return {
#             "consensus_score": 0.5,
#             "voting_consensus": 0.5,  # keep contract surface even in disabled state
#             "consensus_quality": 0.5,
#             "consensus_components": {},
#             "directional_consensus": 0.5,
#             "magnitude_consensus": 0.5,
#             "confidence_consensus": 0.5,
#             "member_contributions": {},
#             "consensus_trends": {},
#             "quality_metrics": {"status": "disabled"},
#             "consensus_recommendations": ["Restart consensus detector system"],
#             "health_metrics": {"status": "disabled", "reason": "circuit_breaker_triggered"},
#             "consensus_detector_initialization": dict(
#                 getattr(self, "_init_payload", {
#                     "status": "initialized",
#                     "timestamp": dt.datetime.now().isoformat(),
#                     "configuration": {
#                         "members": self.n_members,
#                         "threshold": self.threshold,
#                         "consensus_methods": list(self.consensus_methods),
#                     },
#                 })
#             ),
#             "_thesis": "ConsensusDetector disabled due to circuit breaker",
#         }


#     # ────────────────────────────
#     # STATE / HOT-RELOAD
#     # ────────────────────────────
#     def get_state(self) -> Dict[str, Any]:
#         return {
#             "module_info": {"name": "ConsensusDetector", "version": "3.1.0", "last_updated": dt.datetime.now().isoformat()},
#             "configuration": {
#                 "n_members": int(self.n_members),
#                 "threshold": float(self.threshold),
#                 "quality_weighting": bool(self.quality_weighting),
#                 "temporal_smoothing": bool(self.temporal_smoothing),
#                 "consensus_methods": list(self.consensus_methods),
#                 "debug": bool(self.debug),
#             },
#             "consensus_state": {
#                 "last_consensus": float(self.last_consensus),
#                 "consensus_quality": float(self.consensus_quality),
#                 "directional_consensus": float(self.directional_consensus),
#                 "magnitude_consensus": float(self.magnitude_consensus),
#                 "confidence_consensus": float(self.confidence_consensus),
#                 "network_consensus": float(self.network_consensus),
#                 "temporal_stability": float(self.temporal_stability),
#                 "consensus_components": dict(self.consensus_components),
#             },
#             "intelligence_state": {
#                 "consensus_intelligence": dict(self.consensus_intelligence),
#                 "market_adaptation": dict(self.market_adaptation),
#                 "quality_metrics": dict(self.quality_metrics),
#             },
#             "member_state": {
#                 "member_contributions": {int(k): dict(v) for k, v in self.member_contributions.items()},
#                 "consensus_insights": dict(self.consensus_insights),
#             },
#             "history_state": {
#                 "consensus_history": list(self.consensus_history)[-80:],
#                 "consensus_trends": list(self.consensus_trends)[-40:],
#                 "prediction_history": list(self.prediction_history)[-40:],
#                 "regime_consensus_history": {k: list(v)[-30:] for k, v in self.regime_consensus_history.items()},
#                 "consensus_patterns": {k: list(v) for k, v in self.consensus_patterns.items()},
#             },
#             "statistics_state": {"consensus_stats": dict(self.consensus_stats)},
#             "error_state": {"error_count": int(self.error_count), "is_disabled": bool(self.is_disabled)},
#             "performance_metrics": self.get_health_metrics(),
#         }

#     def set_state(self, state: Dict[str, Any]) -> None:
#         try:
#             cfg = state.get("configuration", {})
#             self.n_members = int(cfg.get("n_members", self.n_members))
#             self.threshold = float(cfg.get("threshold", self.threshold))
#             self.quality_weighting = bool(cfg.get("quality_weighting", self.quality_weighting))
#             self.temporal_smoothing = bool(cfg.get("temporal_smoothing", self.temporal_smoothing))
#             self.consensus_methods = list(cfg.get("consensus_methods", self.consensus_methods))
#             self.debug = bool(cfg.get("debug", self.debug))

#             cstate = state.get("consensus_state", {})
#             self.last_consensus = float(cstate.get("last_consensus", self.last_consensus))
#             self.consensus_quality = float(cstate.get("consensus_quality", self.consensus_quality))
#             self.directional_consensus = float(cstate.get("directional_consensus", self.directional_consensus))
#             self.magnitude_consensus = float(cstate.get("magnitude_consensus", self.magnitude_consensus))
#             self.confidence_consensus = float(cstate.get("confidence_consensus", self.confidence_consensus))
#             self.network_consensus = float(cstate.get("network_consensus", self.network_consensus))
#             self.temporal_stability = float(cstate.get("temporal_stability", self.temporal_stability))
#             self.consensus_components = dict(cstate.get("consensus_components", self.consensus_components))

#             intel = state.get("intelligence_state", {})
#             self.consensus_intelligence.update(intel.get("consensus_intelligence", {}))
#             self.market_adaptation.update(intel.get("market_adaptation", {}))
#             self.quality_metrics.update(intel.get("quality_metrics", {}))

#             memb = state.get("member_state", {})
#             self.member_contributions.clear()
#             for k, v in memb.get("member_contributions", {}).items():
#                 self.member_contributions[int(k)] = dict(v)
#             self.consensus_insights.update(memb.get("consensus_insights", {}))

#             hist = state.get("history_state", {})
#             self.consensus_history = deque(hist.get("consensus_history", []), maxlen=150)
#             self.consensus_trends = deque(hist.get("consensus_trends", []), maxlen=80)
#             self.prediction_history = deque(hist.get("prediction_history", []), maxlen=40)
#             self.regime_consensus_history.clear()
#             for k, v in hist.get("regime_consensus_history", {}).items():
#                 self.regime_consensus_history[str(k)] = deque(v, maxlen=40)
#             self.consensus_patterns = defaultdict(list, hist.get("consensus_patterns", {}))

#             stats = state.get("statistics_state", {})
#             self.consensus_stats.update(stats.get("consensus_stats", {}))

#             err = state.get("error_state", {})
#             self.error_count = int(err.get("error_count", self.error_count))
#             self.is_disabled = bool(err.get("is_disabled", self.is_disabled))

#             self.logger.info(
#                 format_operator_message(
#                     icon="[RELOAD]",
#                     message="Consensus Detector state restored",
#                     members=self.n_members,
#                     threshold=f"{self.threshold:.3f}",
#                     consensus_score=f"{self.last_consensus:.3f}",
#                     total_computations=self.consensus_stats.get("total_computations", 0),
#                 )
#             )
#         except Exception as e:
#             ctx = self.error_pinpointer.analyze_error(e, "state_restoration")
#             self.logger.error(f"State restoration failed: {ctx}")

#     # ────────────────────────────
#     # RESET & TEARDOWN
#     # ────────────────────────────
#     def reset(self) -> None:
#         super().reset()
#         self.last_consensus = 0.0
#         self.consensus_quality = 0.0
#         self.directional_consensus = 0.0
#         self.magnitude_consensus = 0.0
#         self.confidence_consensus = 0.0
#         self.network_consensus = 0.0
#         self.temporal_stability = 0.0

#         self.consensus_history.clear()
#         self.consensus_trends.clear()
#         self.prediction_history.clear()
#         self.member_contributions.clear()
#         self.consensus_components.clear()
#         self.consensus_patterns.clear()
#         self.regime_consensus_history.clear()

#         self.quality_metrics.update(
#             {
#                 "coherence": 0.5,
#                 "stability": 0.5,
#                 "diversity": 0.5,
#                 "reliability": 0.5,
#                 "predictive_accuracy": 0.5,
#                 "temporal_consistency": 0.5,
#                 "overall_effectiveness": 0.5,
#             }
#         )
#         self.consensus_stats = {
#             "total_computations": 0,
#             "high_consensus_count": 0,
#             "low_consensus_count": 0,
#             "avg_consensus": 0.5,
#             "consensus_volatility": 0.0,
#             "quality_score": 0.5,
#             "trend_accuracy": 0.0,
#             "prediction_accuracy": 0.0,
#             "session_start": dt.datetime.now().isoformat(),
#         }
#         self.consensus_insights = {
#             "dominant_patterns": [],
#             "member_dynamics": {},
#             "consensus_predictors": {},
#             "quality_drivers": {},
#         }
#         self.error_count = 0
#         self.is_disabled = False

#         self.logger.info(
#             format_operator_message(
#                 icon="[RELOAD]", message="Consensus Detector reset completed", status="All state cleared"
#             )
#         )

#     def _update_performance_metric(self, metric_name: str, value: float) -> None:
#         try:
#             if hasattr(self, "performance_tracker") and self.performance_tracker:
#                 self.performance_tracker.record_metric("ConsensusDetector", metric_name, float(value), True)
#             # Use a separate per-metric history map to avoid conflicting with BaseModule's deque _performance_history
#             if not hasattr(self, "_metric_history"):
#                 self._metric_history = defaultdict(lambda: deque(maxlen=50))  # type: ignore[attr-defined]
#             self._metric_history[metric_name].append({  # type: ignore[attr-defined]
#                 "timestamp": dt.datetime.now().isoformat(),
#                 "value": float(value),
#             })
#         except Exception as e:
#             if hasattr(self, "logger"):
#                 self.logger.warning(f"Performance metric update failed for {metric_name}: {e}")

#     def get_performance_history(self, metric_name: str) -> List[Dict[str, Any]]:
#         try:
#             # Read from the dedicated per-metric history map if present
#             if hasattr(self, "_metric_history"):
#                 return list(self._metric_history.get(metric_name, []))  # type: ignore[attr-defined]
#             return []
#         except Exception:
#             return []

#     def get_all_performance_metrics(self) -> Dict[str, Any]:
#         try:
#             m = {
#                 "consensus_score": float(self.last_consensus),
#                 "consensus_quality": float(self.consensus_quality),
#                 "directional_consensus": float(self.directional_consensus),
#                 "magnitude_consensus": float(self.magnitude_consensus),
#                 "confidence_consensus": float(self.confidence_consensus),
#                 "temporal_stability": float(self.temporal_stability),
#                 "consensus_volatility": float(self.consensus_stats.get("consensus_volatility", 0.0)),
#                 "avg_consensus": float(self.consensus_stats.get("avg_consensus", 0.5)),
#                 "total_computations": int(self.consensus_stats.get("total_computations", 0)),
#                 "high_consensus_ratio": float(
#                     self.consensus_stats.get("high_consensus_count", 0) / max(self.consensus_stats.get("total_computations", 1), 1)
#                 ),
#                 "low_consensus_ratio": float(
#                     self.consensus_stats.get("low_consensus_count", 0) / max(self.consensus_stats.get("total_computations", 1), 1)
#                 ),
#             }
#             m.update({f"quality_{k}": float(v) for k, v in self.quality_metrics.items()})
#             return m
#         except Exception:
#             return {}

#     def __del__(self) -> None:
#         try:
#             if hasattr(self, "logger") and self.logger:
#                 self.logger.info(
#                     format_operator_message(
#                         icon="👋",
#                         message="Consensus Detector shutting down",
#                         total_computations=self.consensus_stats.get("total_computations", 0),
#                         avg_consensus=f"{self.consensus_stats.get('avg_consensus', 0.5):.1%}",
#                     )
#                 )
#         except Exception:
#             pass

#     # ────────────────────────────
#     # BASEMODULE ABSTRACTS
#     # ────────────────────────────
#     async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
#         try:
#             cq = float(self.consensus_quality)
#             stab = float(self.temporal_stability)
#             part = float(len(self.member_contributions) / max(self.n_members, 1))
#             if len(self.consensus_history) > 5:
#                 recent = [float(e.get("consensus", 0.5)) for e in list(self.consensus_history)[-5:]]
#                 mu = float(np.mean(recent)) if recent else 0.5
#                 var = float(np.std(recent)) if recent else 0.0
#                 trend_stab = float(np.clip(1.0 - var / max(mu, 0.1), 0.0, 1.0)) if mu > 0 else 0.5
#             else:
#                 trend_stab = 0.5
#             conf = 0.4 * cq + 0.3 * stab + 0.2 * part + 0.1 * trend_stab
#             return float(np.clip(conf, 0.1, 0.95))
#         except Exception:
#             return 0.4

#     async def propose_action(self, **inputs) -> Dict[str, Any]:
#         try:
#             cs = float(self.last_consensus)
#             cq = float(self.consensus_quality)
#             dc = float(self.directional_consensus)

#             if cs > 0.8 and cq > 0.7:
#                 a, s, r = "strong_consensus", 0.9, f"Strong consensus (score={cs:.3f}, quality={cq:.3f})"
#             elif cs > 0.6:
#                 a, s, r = "moderate_consensus", cs * 0.7, f"Moderate consensus (score={cs:.3f})"
#             elif cs < 0.3:
#                 a, s, r = "low_consensus", 0.8, f"Low consensus ({cs:.3f}) — coordinate members"
#             elif dc < 0.4:
#                 a, s, r = "directional_conflict", 0.6, f"Directional disagreement ({dc:.3f})"
#             else:
#                 a, s, r = "monitor", 0.3, "Moderate consensus — continue monitoring"

#             return {
#                 "action": a,
#                 "signal_strength": float(s),
#                 "reasoning": r,
#                 "consensus_metrics": {
#                     "consensus_score": cs,
#                     "consensus_quality": cq,
#                     "directional_consensus": dc,
#                     "magnitude_consensus": float(self.magnitude_consensus),
#                     "confidence_consensus": float(self.confidence_consensus),
#                     "temporal_stability": float(self.temporal_stability),
#                 },
#                 "member_stats": {"active_members": int(len(self.member_contributions)), "target_members": int(self.n_members)},
#                 "confidence": await self.calculate_confidence({}, **inputs),
#             }
#         except Exception as e:
#             if hasattr(self, "logger"):
#                 self.logger.error(f"Action proposal failed: {e}")
#             return {"action": "abstain", "signal_strength": 0.0, "reasoning": f"Consensus detection error: {str(e)}", "confidence": 0.1}
