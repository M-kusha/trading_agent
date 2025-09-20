"""
🕵️ Enhanced Collusion Auditor with SmartInfoBus Integration v3.1
Advanced collusion detection and anti-manipulation safeguards for voting committees
"""

from __future__ import annotations

import asyncio
import time
import math
from modules.contracts import module_args
import numpy as np
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple, Set, Deque
from collections import deque, defaultdict

# ═══════════════════════════════════════════════════════════════════
# MODERN SMARTINFOBUS IMPORTS
# ═══════════════════════════════════════════════════════════════════
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


@module(**module_args(
    "CollusionAuditor",
    description="Advanced collusion detection and anti-manipulation safeguards for voting committees",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class CollusionAuditor(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🕵️ PRODUCTION-GRADE Collusion Auditor v3.1

    - Multi-dimensional similarity analysis for detecting coordination patterns
    - Adaptive threshold management based on market conditions
    - Behavioral profiling and temporal/network pattern analysis
    - SmartInfoBus zero-wiring architecture
    - Real-time threat assessment and alert management
    """

    # ────────────────────────────
    # INIT & SYSTEM WIRING
    # ────────────────────────────
    def _initialize(self) -> None:
        # Core services (no calls to unknown mixin init hooks)
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="CollusionAuditor",
            log_path="logs/voting/collusion_auditor.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("CollusionAuditor", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor(auto_start=False)

        # Configuration
        self.n_members: int = int(self.config.get("n_members", 5))
        self.window: int = int(self.config.get("window", 10))
        self.base_threshold: float = float(self.config.get("threshold", 0.9))
        self.current_threshold: float = float(self.base_threshold)
        self.adaptive_threshold: bool = bool(self.config.get("adaptive_threshold", True))
        self.similarity_methods: List[str] = list(
            self.config.get("similarity_methods", ["cosine", "correlation", "euclidean"])
        )
        self.debug: bool = bool(self.config.get("debug", False))

        # Method registry (metadata)
        self.detection_methods = self._initialize_detection_methods()

        # Collusion state
        self.vote_history: deque = deque(maxlen=self.window * 2)
        self.collusion_score: float = 0.0
        self.suspicious_pairs: Set[Tuple[int, int]] = set()
        self.collusion_history: deque = deque(maxlen=100)

        # Pair & member analytics
        self.pair_agreement_history: Dict[Tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.member_behavior_profiles: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {
                "avg_similarity": 0.0,
                "volatility": 0.0,
                "consistency_score": 0.0,
                "independence_score": 1.0,
                "coordination_frequency": 0.0,
                "anomaly_score": 0.0,
            }
        )

        # Temporal/network tracking
        self.temporal_patterns: Dict[str, Any] = defaultdict(list)
        self.coordination_events: deque = deque(maxlen=50)
        self.alert_patterns: Dict[str, Any] = defaultdict(list)

        # Statistics
        self.detection_stats: Dict[str, Any] = {
            "total_checks": 0,
            "alerts_raised": 0,
            "false_positive_rate": 0.0,
            "confirmed_collusion_events": 0,
            "avg_pair_similarity": 0.0,
            "member_independence_scores": {},
            "detection_accuracy": 0.95,
            "alert_frequency": 0.0,
            "session_start": dt.datetime.now().isoformat(),
        }

        # Intelligence knobs
        self.detection_intelligence: Dict[str, Any] = {
            "threshold_bounds": (0.7, 0.98),
            "adaptation_rate": 0.15,
            "sensitivity_target": 0.85,
            "false_positive_threshold": 0.10,
            "confirmation_threshold": 0.80,
            "temporal_sensitivity": 0.60,
            "behavioral_memory": 0.90,
        }

        # Market adaptation multipliers
        self.market_adaptation: Dict[str, Any] = {
            "regime_multipliers": {
                "trending": 1.10,
                "ranging": 0.90,
                "volatile": 0.85,
                "breakout": 1.20,
                "reversal": 1.15,
                "unknown": 1.00,
            },
            "agreement_adjustments": {
                "high_agreement": 1.20,
                "medium_agreement": 1.00,
                "low_agreement": 0.80,
            },
        }

        # Alerting system
        self.alert_system: Dict[str, Any] = {
            "cooldown_period": 10,  # checks between alerts for same pair
            "escalation_threshold": 3,
            "severity_levels": ["info", "warning", "critical"],
            "auto_investigation": True,
            "last_alerts": defaultdict(int),  # pair -> last check index
            "alert_history": deque(maxlen=200),  # store recent alerts
        }

        # Quality metrics
        self.quality_metrics: Dict[str, float] = {
            "detection_precision": 0.0,
            "detection_recall": 0.0,
            "behavioral_accuracy": 0.0,
            "temporal_consistency": 0.0,
            "overall_effectiveness": 0.0,
        }

        # Circuit breaker
        self.error_count: int = 0
        self.circuit_breaker_threshold: int = 5
        self.is_disabled: bool = False

        # Announce
        self._generate_initialization_thesis()
        version = getattr(self.metadata, "version", "3.1.0") if self.metadata else "3.1.0"
        self.logger.info(
            format_operator_message(
                icon="🕵️",
                message=f"Collusion Auditor v{version} initialized",
                members=self.n_members,
                window=self.window,
                base_threshold=f"{self.base_threshold:.3f}",
                methods=len(self.similarity_methods),
                adaptive=self.adaptive_threshold,
            )
        )

    def _initialize_detection_methods(self) -> Dict[str, Dict[str, Any]]:
        return {
            "cosine_similarity": {
                "description": "Cosine similarity analysis for detecting aligned voting patterns",
                "parameters": {"normalization": True, "weight_threshold": 0.1},
                "use_cases": ["general_coordination", "direction_alignment"],
                "effectiveness_threshold": 0.7,
                "computational_cost": "low",
            },
            "correlation_analysis": {
                "description": "Statistical correlation analysis for temporal coordination patterns",
                "parameters": {"min_samples": 3, "confidence_level": 0.95},
                "use_cases": ["temporal_coordination", "sequential_patterns"],
                "effectiveness_threshold": 0.8,
                "computational_cost": "medium",
            },
            "euclidean_distance": {
                "description": "Distance-based analysis for detecting similar magnitude responses",
                "parameters": {"distance_normalization": True, "outlier_detection": True},
                "use_cases": ["magnitude_coordination", "precision_collusion"],
                "effectiveness_threshold": 0.6,
                "computational_cost": "low",
            },
            "behavioral_profiling": {
                "description": "Long-term behavioral pattern analysis for identifying systematic coordination",
                "parameters": {"profile_memory": 50, "anomaly_sensitivity": 0.3},
                "use_cases": ["systematic_collusion", "long_term_coordination"],
                "effectiveness_threshold": 0.75,
                "computational_cost": "high",
            },
            "temporal_clustering": {
                "description": "Time-based clustering analysis for detecting coordinated timing patterns",
                "parameters": {"time_window": 5, "clustering_threshold": 0.8},
                "use_cases": ["timing_coordination", "synchronized_responses"],
                "effectiveness_threshold": 0.7,
                "computational_cost": "medium",
            },
        }

    def _generate_initialization_thesis(self) -> None:
        thesis = f"""
Collusion Auditor v3.1 Initialization:
- Members={self.n_members}, Window={self.window}, BaseThreshold={self.base_threshold:.3f}
- Adaptive={'on' if self.adaptive_threshold else 'off'} Bounds={self.detection_intelligence['threshold_bounds']}
- Methods={len(self.detection_methods)} Similarities={', '.join(self.similarity_methods)}
- Alerts cooldown={self.alert_system['cooldown_period']} levels={len(self.alert_system['severity_levels'])}
"""
        self.smart_bus.set(
            "collusion_auditor_initialization",
            {
                "status": "initialized",
                "thesis": thesis,
                "timestamp": dt.datetime.now().isoformat(),
                "configuration": {
                    "members": self.n_members,
                    "window": self.window,
                    "detection_methods": list(self.detection_methods.keys()),
                    "intelligence_parameters": self.detection_intelligence,
                },
            },
            module="CollusionAuditor",
            thesis=thesis,
        )

    # ────────────────────────────
    # MAIN PROCESS
    # ────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if self.is_disabled:
                return self._generate_disabled_response()

            voting_data = await self._get_comprehensive_voting_data()
            await self._update_detection_parameters_comprehensive(voting_data)

            collusion_analysis = await self._perform_comprehensive_collusion_analysis(voting_data)
            behavioral_updates = await self._update_behavioral_profiles_comprehensive(voting_data)
            temporal_analysis = await self._analyze_temporal_coordination_patterns(voting_data)
            quality_analysis = await self._calculate_comprehensive_quality_metrics()

            recommendations = await self._generate_intelligent_detection_recommendations(
                collusion_analysis, behavioral_updates, temporal_analysis
            )
            thesis = await self._generate_comprehensive_detection_thesis(
                collusion_analysis, quality_analysis, recommendations
            )

            results: Dict[str, Any] = {
                "collusion_score": float(self.collusion_score),
                "suspicious_pairs": [tuple(p) for p in self.suspicious_pairs],
                "member_independence_scores": self.get_member_independence_scores(),
                "collusion_alerts": self._get_recent_collusion_alerts(),
                "behavioral_profiles": self._get_behavioral_profiles_summary(),
                "coordination_events": list(self.coordination_events)[-10:],
                "detection_statistics": self._get_comprehensive_detection_stats(),
                "audit_recommendations": recommendations,
                "quality_metrics": quality_analysis,
                "health_metrics": self._get_health_metrics(),
                # contract heartbeat: include initialization view in results
                "collusion_auditor_initialization": self._get_collusion_init_view(),
                "decision_id": voting_data.get("decision_id"),
                "tick_ts": voting_data.get("tick_ts") or dt.datetime.now().isoformat(),
                "_thesis": thesis,
            }

            await self._update_smartinfobus_comprehensive(results, thesis)

            self.performance_tracker.record_metric(
                "CollusionAuditor", "process_time_ms", (time.time() - start_time) * 1000.0, True
            )
            self.error_count = 0
            return results

        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    # ────────────────────────────
    # BUS IO
    # ────────────────────────────
    async def _get_comprehensive_voting_data(self) -> Dict[str, Any]:
        try:
            g = self.smart_bus.get
            return {
                "votes": g("votes", "CollusionAuditor") or [],
                "voting_summary": g("voting_summary", "CollusionAuditor") or {},
                "strategy_arbiter_weights": g("strategy_arbiter_weights", "CollusionAuditor") or [],
                "raw_proposals": g("raw_proposals", "CollusionAuditor") or [],
                "member_confidences": g("member_confidences", "CollusionAuditor") or [],
                "consensus_direction": g("consensus_direction", "CollusionAuditor") or "neutral",
                "agreement_score": g("agreement_score", "CollusionAuditor") or 0.5,
                "market_context": g("market_context", "CollusionAuditor") or {},
                "recent_trades": g("recent_trades", "CollusionAuditor") or [],
                "market_regime": g("market_regime", "CollusionAuditor") or "unknown",
                "volatility_data": g("volatility_data", "CollusionAuditor") or {},
                "decision_id": g("decision_id", "CollusionAuditor"),
                "tick_ts": g("tick_ts", "CollusionAuditor"),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "CollusionAuditor")
            self.logger.warning(f"Voting data retrieval incomplete: {error_context}")
            return self._get_safe_voting_defaults()

    # ────────────────────────────
    # ADAPTIVE THRESHOLD
    # ────────────────────────────
    async def _update_detection_parameters_comprehensive(self, voting_data: Dict[str, Any]) -> None:
        try:
            if not self.adaptive_threshold:
                return

            regime = str(voting_data.get("market_regime", "unknown"))
            agreement_score = float(voting_data.get("agreement_score", 0.5))
            recent_trades = voting_data.get("recent_trades", [])

            market_uncertainty = float(self._calculate_market_uncertainty_factor(voting_data))

            base_multiplier = 1.0
            base_multiplier *= float(self.market_adaptation["regime_multipliers"].get(regime, 1.0))

            if agreement_score > 0.8:
                agreement_category = "high_agreement"
            elif agreement_score > 0.4:
                agreement_category = "medium_agreement"
            else:
                agreement_category = "low_agreement"
            base_multiplier *= float(self.market_adaptation["agreement_adjustments"].get(agreement_category, 1.0))

            if recent_trades:
                recent_performance = float(self._calculate_recent_performance(recent_trades))
                if abs(recent_performance) > 0.05:
                    base_multiplier *= 1.1

            target_threshold = float(
                np.clip(
                    self.base_threshold * base_multiplier,
                    self.detection_intelligence["threshold_bounds"][0],
                    self.detection_intelligence["threshold_bounds"][1],
                )
            )

            rate = float(self.detection_intelligence["adaptation_rate"])
            old = float(self.current_threshold)
            self.current_threshold = float(old * (1.0 - rate) + target_threshold * rate)

            if abs(self.current_threshold - old) > 0.01:
                self.logger.info(
                    format_operator_message(
                        icon="[TARGET]",
                        message="Detection threshold adapted",
                        old_threshold=f"{old:.4f}",
                        new_threshold=f"{self.current_threshold:.4f}",
                        regime=regime,
                        agreement=f"{agreement_score:.2f}",
                        uncertainty=f"{market_uncertainty:.3f}",
                    )
                )
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "detection_parameters_update")
            self.logger.warning(f"Detection parameter update failed: {error_context}")

    def _calculate_market_uncertainty_factor(self, voting_data: Dict[str, Any]) -> float:
        try:
            comps: List[float] = []
            # inverse of agreement
            comps.append(1.0 - float(voting_data.get("agreement_score", 0.5)))
            # regime
            comps.append(0.8 if str(voting_data.get("market_regime", "unknown")) == "unknown" else 0.2)
            # volatility
            vol_level = (
                voting_data.get("volatility_data", {}) or {}
            ).get("level", "medium")
            comps.append({"very_low": 0.1, "low": 0.3, "medium": 0.5, "high": 0.8, "extreme": 1.0}.get(vol_level, 0.5))
            # performance
            rtr = voting_data.get("recent_trades", [])
            if len(rtr) >= 3:
                pnls = [float(t.get("pnl", 0.0)) for t in rtr[-5:]]
                if pnls:
                    vol = float(np.std(pnls) / (abs(np.mean(pnls)) + 0.01))
                    comps.append(min(1.0, vol))

            w = np.array([0.4, 0.2, 0.3, 0.1][: len(comps)], dtype=np.float32)
            w = w / float(w.sum()) if float(w.sum()) > 0 else w
            total = float(np.dot(np.array(comps, dtype=np.float32), w))
            return float(np.clip(total, 0.0, 1.0))
        except Exception:
            return 0.5

    def _calculate_recent_performance(self, recent_trades: List[Dict[str, Any]]) -> float:
        try:
            if not recent_trades:
                return 0.0
            recent_pnl = [float(trade.get("pnl", 0.0)) for trade in recent_trades[-10:]]
            return float(np.mean(recent_pnl)) if recent_pnl else 0.0
        except Exception:
            return 0.0

    # ────────────────────────────
    # CORE ANALYSIS
    # ────────────────────────────
    async def _perform_comprehensive_collusion_analysis(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.detection_stats["total_checks"] += 1

            actions = self._extract_voting_actions(voting_data)
            if len(actions) < 2:
                return {"collusion_score": 0.0, "suspicious_pairs": [], "analysis_status": "insufficient_data"}

            timestamp = dt.datetime.now().isoformat()
            self.vote_history.append(
                {
                    "timestamp": timestamp,
                    "actions": [a.copy() for a in actions],
                    "n_members": len(actions),
                    "agreement_score": float(voting_data.get("agreement_score", 0.5)),
                    "market_regime": str(voting_data.get("market_regime", "unknown")),
                }
            )

            if len(self.vote_history) < 3:
                return {"collusion_score": 0.0, "suspicious_pairs": [], "analysis_status": "building_history"}

            similarity_analysis = await self._calculate_comprehensive_similarities(actions)
            await self._update_pair_agreements_comprehensive(similarity_analysis)
            coordination_analysis = await self._detect_coordination_patterns(similarity_analysis)
            alert_updates = await self._update_suspicious_pairs_and_alerts(coordination_analysis)

            self.collusion_score = await self._calculate_overall_collusion_score(coordination_analysis)

            self.collusion_history.append(
                {
                    "timestamp": timestamp,
                    "collusion_score": float(self.collusion_score),
                    "suspicious_pairs": list(self.suspicious_pairs),
                    "similarity_analysis": {str(k): v for k, v in similarity_analysis.items()},
                    "threshold_used": float(self.current_threshold),
                    "coordination_analysis": coordination_analysis,
                    "alert_updates": alert_updates,
                }
            )

            await self._update_detection_statistics_comprehensive(similarity_analysis, coordination_analysis)

            return {
                "collusion_score": float(self.collusion_score),
                "suspicious_pairs": list(self.suspicious_pairs),
                "similarity_analysis": similarity_analysis,
                "coordination_analysis": coordination_analysis,
                "alert_updates": alert_updates,
                "analysis_status": "complete",
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "collusion_analysis")
            self.logger.error(f"Collusion analysis failed: {error_context}")
            return {"collusion_score": 0.0, "suspicious_pairs": [], "analysis_status": "error", "error": str(error_context)}

    def _extract_voting_actions(self, voting_data: Dict[str, Any]) -> List[np.ndarray]:
        try:
            raw_proposals = voting_data.get("raw_proposals", [])
            if raw_proposals and len(raw_proposals) >= 2:
                actions: List[np.ndarray] = []
                for proposal in raw_proposals[: self.n_members]:
                    if isinstance(proposal, (list, np.ndarray)) and len(proposal) > 0:
                        actions.append(np.asarray(proposal, dtype=np.float32))
                if len(actions) >= 2:
                    return actions

            votes = voting_data.get("votes", [])
            if votes and len(votes) >= 2:
                actions = []
                for vote in votes[: self.n_members]:
                    if isinstance(vote, (int, float, np.floating)):
                        actions.append(np.array([float(vote)], dtype=np.float32))
                    elif isinstance(vote, (list, np.ndarray)):
                        arr = np.asarray(vote, dtype=np.float32)
                        if arr.size > 0:
                            actions.append(arr)
                if len(actions) >= 2:
                    return actions

            weights = voting_data.get("strategy_arbiter_weights", [])
            if weights and len(weights) >= 2:
                return [np.asarray(w, dtype=np.float32) for w in weights[: self.n_members]]

            return []
        except Exception:
            return []

    async def _calculate_comprehensive_similarities(
        self, actions: List[np.ndarray]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        try:
            sims: Dict[Tuple[int, int], Dict[str, float]] = {}
            n = len(actions)
            for i in range(n):
                for j in range(i + 1, n):
                    v1, v2 = actions[i], actions[j]
                    pair: Tuple[int, int] = (i, j)
                    pair_sims: Dict[str, float] = {}

                    n1 = float(np.linalg.norm(v1))
                    n2 = float(np.linalg.norm(v2))
                    if n1 <= 1e-9 or n2 <= 1e-9:
                        sims[pair] = pair_sims
                        continue

                    # Cosine
                    if "cosine" in self.similarity_methods:
                        c = float(np.dot(v1, v2) / (n1 * n2))
                        pair_sims["cosine"] = float(np.clip(c, -1.0, 1.0))

                    # Correlation (only if var>0 and dim>1)
                    if "correlation" in self.similarity_methods and v1.size > 1 and v2.size > 1:
                        s1 = float(np.std(v1))
                        s2 = float(np.std(v2))
                        if s1 > 1e-9 and s2 > 1e-9:
                            corr = float(np.corrcoef(v1, v2)[0, 1])
                            if not math.isnan(corr):
                                pair_sims["correlation"] = float(np.clip(corr, -1.0, 1.0))

                    # Euclidean (normalized to [0,1] as similarity)
                    if "euclidean" in self.similarity_methods:
                        dist = float(np.linalg.norm(v1 - v2))
                        maxd = n1 + n2
                        eu = 1.0 - (dist / maxd if maxd > 1e-9 else 0.0)
                        pair_sims["euclidean"] = float(np.clip(eu, -1.0, 1.0))

                    # Angular similarity in [0,1]
                    dotp = float(np.dot(v1, v2))
                    cos_angle = float(np.clip(dotp / (n1 * n2), -1.0, 1.0))
                    ang_sim = (cos_angle + 1.0) / 2.0
                    pair_sims["angular"] = float(ang_sim)

                    sims[pair] = pair_sims
            return sims
        except Exception:
            return {}

    async def _update_pair_agreements_comprehensive(
        self, similarity_analysis: Dict[Tuple[int, int], Dict[str, float]]
    ) -> None:
        try:
            weights = {"cosine": 0.4, "correlation": 0.3, "euclidean": 0.2, "angular": 0.1}
            for pair, smap in similarity_analysis.items():
                if not smap:
                    continue
                wsum = 0.0
                acc = 0.0
                for method, val in smap.items():
                    w = float(weights.get(method, 0.25))
                    acc += float(val) * w
                    wsum += w
                if wsum > 0.0:
                    final = float(acc / wsum)
                    self.pair_agreement_history[pair].append(final)
        except Exception:
            pass

    async def _detect_coordination_patterns(
        self, similarity_analysis: Dict[Tuple[int, int], Dict[str, float]]
    ) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "coordinated_pairs": [],
                "coordination_strength": {},
                "temporal_patterns": {},
                "behavioral_anomalies": {},
                "network_effects": {},
            }
            for pair in similarity_analysis.keys():
                hist = self.pair_agreement_history.get(pair, deque())
                if len(hist) < 3:
                    continue

                hist_list = list(hist)
                hist_avg = float(np.mean(hist_list))
                recent_avg = float(np.mean(hist_list[-3:])) if len(hist_list) >= 3 else hist_avg
                trend = float(recent_avg - hist_avg)
                stdv = float(np.std(hist_list))
                consistency = float(max(0.0, 1.0 - (stdv / max(hist_avg, 0.1))))

                is_coord = hist_avg > self.current_threshold and consistency > 0.7 and len(hist_list) >= 5
                if is_coord:
                    out["coordinated_pairs"].append(pair)
                    out["coordination_strength"][pair] = {
                        "historical_avg": hist_avg,
                        "recent_avg": recent_avg,
                        "trend": trend,
                        "consistency": consistency,
                        "coordination_score": float(hist_avg * consistency),
                    }

                if len(hist_list) >= 5:
                    out["temporal_patterns"][pair] = self._analyze_temporal_pattern(hist_list)
                if len(hist_list) >= 10:
                    anom = self._calculate_anomaly_score(hist_list)
                    if anom > 0.7:
                        out["behavioral_anomalies"][pair] = float(anom)

            out["network_effects"] = await self._analyze_network_coordination_effects(out["coordinated_pairs"])
            return out
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "coordination_pattern_detection")
            return {"coordinated_pairs": [], "coordination_strength": {}, "analysis_error": str(error_context)}

    def _analyze_temporal_pattern(self, history: List[float]) -> Dict[str, Any]:
        try:
            if len(history) < 3:
                return {"pattern": "insufficient_data"}
            x = np.arange(len(history))
            try:
                slope = float(np.polyfit(x, history, 1)[0])
            except Exception:
                slope = 0.0
            volatility = float(np.std(history))
            meanv = float(np.mean(history))
            if abs(slope) < 0.01 and volatility < 0.1:
                pattern = "stable_high" if meanv > 0.8 else "stable_low"
            elif slope > 0.05:
                pattern = "increasing"
            elif slope < -0.05:
                pattern = "decreasing"
            elif volatility > 0.3:
                pattern = "volatile"
            else:
                pattern = "moderate"
            recent_trend = "up" if (len(history) >= 3 and history[-1] > history[-3]) else "down"
            return {"pattern": pattern, "slope": slope, "volatility": volatility, "mean_value": meanv, "recent_trend": recent_trend}
        except Exception:
            return {"pattern": "unknown"}

    def _calculate_anomaly_score(self, history: List[float]) -> float:
        try:
            if len(history) < 5:
                return 0.0
            base = history[:-3]
            if len(base) < 2:
                return 0.0
            m = float(np.mean(base))
            s = float(np.std(base))
            if s < 1e-9:
                return 0.0
            z = [abs((float(v) - m) / s) for v in history[-3:]]
            return float(min(1.0, max(z) / 3.0))
        except Exception:
            return 0.0

    async def _analyze_network_coordination_effects(self, pairs: List[Tuple[int, int]]) -> Dict[str, Any]:
        try:
            if not pairs:
                return {"network_score": 0.0, "clusters": [], "coordination_density": 0.0, "largest_cluster_size": 0, "total_coordinated_pairs": 0}
            net: Dict[int, Set[int]] = defaultdict(set)
            for i, j in pairs:
                net[i].add(j)
                net[j].add(i)

            clusters: List[List[int]] = []
            visited: Set[int] = set()

            def dfs(start: int) -> List[int]:
                stack = [start]
                cluster: List[int] = []
                while stack:
                    u = stack.pop()
                    if u in visited:
                        continue
                    visited.add(u)
                    cluster.append(u)
                    for v in net.get(u, set()):
                        if v not in visited:
                            stack.append(v)
                return sorted(cluster)

            for node in list(net.keys()):
                if node not in visited:
                    c = dfs(node)
                    if len(c) > 2:
                        clusters.append(c)

            total_pairs = self.n_members * (self.n_members - 1) / 2.0
            density = float(len(pairs) / max(total_pairs, 1.0))
            largest = max((len(c) for c in clusters), default=0)
            score = density + 0.5 * ((largest - 2) / max(self.n_members - 2, 1.0)) if largest >= 3 else density
            return {
                "network_score": float(min(1.0, max(0.0, score))),
                "clusters": clusters,
                "coordination_density": float(density),
                "largest_cluster_size": int(largest),
                "total_coordinated_pairs": int(len(pairs)),
            }
        except Exception:
            return {"network_score": 0.0, "clusters": [], "coordination_density": 0.0, "largest_cluster_size": 0, "total_coordinated_pairs": 0}

    async def _update_suspicious_pairs_and_alerts(self, coord: Dict[str, Any]) -> Dict[str, Any]:
        try:
            updates = {"new_alerts": [], "escalated_alerts": [], "resolved_alerts": [], "alert_summary": {}}
            old = set(self.suspicious_pairs)
            self.suspicious_pairs = set(coord.get("coordinated_pairs", []))

            # New / periodic alerts
            for pair in self.suspicious_pairs:
                strength = coord.get("coordination_strength", {}).get(pair, {})
                score = float(strength.get("coordination_score", 0.0))
                last_step = int(self.alert_system["last_alerts"].get(pair, 0))
                steps_since = int(self.detection_stats["total_checks"] - last_step)
                if pair not in old or steps_since > int(self.alert_system["cooldown_period"]):
                    sev = self._determine_alert_severity(score)
                    alert = await self._generate_coordination_alert(pair, strength, sev)
                    updates["new_alerts"].append(alert)
                    self.alert_system["last_alerts"][pair] = int(self.detection_stats["total_checks"])
                    self.alert_system["alert_history"].append(alert)
                    # Escalation logic (simple heuristic based on repeats)
                    repeat_count = sum(1 for a in self.alert_system["alert_history"] if tuple(a.get("pair", (-1, -1))) == pair)
                    if repeat_count >= int(self.alert_system["escalation_threshold"]) and sev != "critical":
                        alert_escalated = dict(alert)
                        alert_escalated["severity"] = "critical"
                        updates["escalated_alerts"].append(alert_escalated)
                        self.alert_system["alert_history"].append(alert_escalated)

                    # Record coordination event
                    self.coordination_events.append(
                        {
                            "timestamp": dt.datetime.now().isoformat(),
                            "pair": pair,
                            "coordination_score": score,
                            "alert_severity": sev,
                            "coordination_strength": strength,
                            "alert_type": "coordination_detection",
                        }
                    )

            # Resolutions
            resolved = old - self.suspicious_pairs
            for pair in resolved:
                updates["resolved_alerts"].append(
                    {
                        "pair": pair,
                        "resolution_timestamp": dt.datetime.now().isoformat(),
                        "resolution_reason": "coordination_below_threshold",
                    }
                )

            self.detection_stats["alerts_raised"] += int(len(updates["new_alerts"]))
            updates["alert_summary"] = {
                "total_suspicious_pairs": int(len(self.suspicious_pairs)),
                "new_alerts_count": int(len(updates["new_alerts"])),
                "resolved_alerts_count": int(len(updates["resolved_alerts"])),
                "active_cooldowns": int(len(self.alert_system["last_alerts"])),
            }
            return updates
        except Exception:
            return {"new_alerts": [], "escalated_alerts": [], "resolved_alerts": [], "alert_summary": {}}

    def _determine_alert_severity(self, coordination_score: float) -> str:
        try:
            if coordination_score > 0.90:
                return "critical"
            if coordination_score > 0.80:
                return "warning"
            return "info"
        except Exception:
            return "info"

    async def _generate_coordination_alert(
        self, pair: Tuple[int, int], strength: Dict[str, Any], severity: str
    ) -> Dict[str, Any]:
        try:
            i, j = pair
            hist = float(strength.get("historical_avg", 0.0))
            cons = float(strength.get("consistency", 0.0))
            trend = float(strength.get("trend", 0.0))
            if severity == "critical":
                icon, msg = "[ALERT]", f"CRITICAL: High coordination detected between members {i} and {j}"
            elif severity == "warning":
                icon, msg = "[WARN]", f"WARNING: Suspicious coordination between members {i} and {j}"
            else:
                icon, msg = "ℹ️", f"INFO: Monitoring coordination between members {i} and {j}"

            self.logger.warning(
                format_operator_message(
                    icon=icon,
                    message=msg,
                    coordination=f"{hist:.3f}",
                    threshold=f"{self.current_threshold:.3f}",
                    consistency=f"{cons:.3f}",
                    trend=f"{trend:+.3f}",
                    action_required="Monitor these members closely",
                )
            )
            return {
                "timestamp": dt.datetime.now().isoformat(),
                "pair": (int(i), int(j)),
                "severity": severity,
                "message": msg,
                "coordination_score": hist,
                "threshold_used": float(self.current_threshold),
                "consistency_score": cons,
                "trend": trend,
                "recommended_action": self._get_recommended_action(severity, strength),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "alert_generation")
            return {
                "timestamp": dt.datetime.now().isoformat(),
                "pair": pair,
                "severity": "info",
                "message": f"Alert generation failed: {error_context}",
                "coordination_score": 0.0,
            }

    def _get_recommended_action(self, severity: str, strength: Dict[str, Any]) -> str:
        try:
            if severity == "critical":
                return "Immediate investigation required – consider member rotation or voting weight adjustment"
            if severity == "warning":
                return "Enhanced monitoring recommended – review member behavior patterns"
            return "Continue standard monitoring – document coordination patterns"
        except Exception:
            return "Monitor and investigate if patterns persist"

    async def _calculate_overall_collusion_score(self, coord: Dict[str, Any]) -> float:
        try:
            pairs = coord.get("coordinated_pairs", [])
            net = coord.get("network_effects", {})
            max_pairs = self.n_members * (self.n_members - 1) / 2.0
            pair_score = float(len(pairs) / max(max_pairs, 1.0))
            net_score = float(net.get("network_score", 0.0))
            overall = 0.7 * pair_score + 0.3 * net_score
            if len(self.collusion_history) >= 3:
                recent = [float(e.get("collusion_score", 0.0)) for e in list(self.collusion_history)[-3:]]
                if recent:
                    std = float(np.std(recent))
                    mean = float(np.mean(recent))
                    cons = 1.0 - (std / max(mean, 0.1))
                    overall *= float(np.clip(0.8 + 0.2 * cons, 0.5, 1.2))
            return float(np.clip(overall, 0.0, 1.0))
        except Exception:
            return 0.0

    async def _update_detection_statistics_comprehensive(
        self, similarity_analysis: Dict[Tuple[int, int], Dict[str, float]], coordination_analysis: Dict[str, Any]
    ) -> None:
        try:
            all_vals: List[float] = []
            for mp in similarity_analysis.values():
                for v in mp.values():
                    all_vals.append(float(v))
            if all_vals:
                self.detection_stats["avg_pair_similarity"] = float(np.mean(all_vals))

            for member_id in range(self.n_members):
                mvals: List[float] = []
                for (i, j), mp in similarity_analysis.items():
                    if (i == member_id or j == member_id) and mp:
                        mvals.extend([float(v) for v in mp.values()])
                if mvals:
                    avg = float(np.mean(mvals))
                    indep = max(0.0, 1.0 - avg)
                    self.detection_stats["member_independence_scores"][f"member_{member_id}"] = float(indep)

            tc = int(self.detection_stats["total_checks"])
            ar = int(self.detection_stats["alerts_raised"])
            self.detection_stats["alert_frequency"] = float(ar / tc) if tc > 0 else 0.0

            self._update_performance_metric("collusion_score", float(self.collusion_score))
            self._update_performance_metric("suspicious_pairs_count", int(len(self.suspicious_pairs)))
            self._update_performance_metric("avg_pair_similarity", float(self.detection_stats["avg_pair_similarity"]))
        except Exception:
            pass

    # ────────────────────────────
    # BEHAVIORAL ANALYTICS
    # ────────────────────────────
    async def _update_behavioral_profiles_comprehensive(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            out = {"profile_updates": {}, "anomaly_detections": {}, "behavioral_trends": {}}
            raw = voting_data.get("raw_proposals", [])
            if len(raw) < 2:
                return out

            for i, proposal in enumerate(raw[: self.n_members]):
                if not isinstance(proposal, (list, np.ndarray)) or len(proposal) == 0:
                    continue
                p = np.asarray(proposal, dtype=np.float32)
                prof = self.member_behavior_profiles[i]
                sims: List[float] = []
                for j, other in enumerate(raw[: self.n_members]):
                    if i == j or not isinstance(other, (list, np.ndarray)) or len(other) == 0:
                        continue
                    q = np.asarray(other, dtype=np.float32)
                    n1, n2 = float(np.linalg.norm(p)), float(np.linalg.norm(q))
                    if n1 > 1e-9 and n2 > 1e-9:
                        sims.append(float(np.dot(p, q) / (n1 * n2)))
                if not sims:
                    continue

                old_avg = float(prof.get("avg_similarity", 0.0))
                new_avg = float(np.mean(sims))
                mem = float(self.detection_intelligence["behavioral_memory"])
                prof["avg_similarity"] = float(old_avg * mem + new_avg * (1.0 - mem))
                prof["volatility"] = float(np.std(sims))
                prof["independence_score"] = float(max(0.0, 1.0 - prof["avg_similarity"]))
                prof["consistency_score"] = float(max(0.0, 1.0 - prof["volatility"]))
                high_sim = sum(1 for s in sims if s > self.current_threshold)
                prof["coordination_frequency"] = float(high_sim / max(len(sims), 1))
                if len(sims) >= 3:
                    prof["anomaly_score"] = float(self._calculate_member_anomaly_score(sims, prof))

                if abs(new_avg - old_avg) > 0.2:
                    out["profile_updates"][i] = {
                        "old_similarity": old_avg,
                        "new_similarity": new_avg,
                        "change_magnitude": float(abs(new_avg - old_avg)),
                        "timestamp": dt.datetime.now().isoformat(),
                    }
                if float(prof["anomaly_score"]) > 0.7:
                    out["anomaly_detections"][i] = {
                        "anomaly_score": float(prof["anomaly_score"]),
                        "anomaly_type": self._classify_behavioral_anomaly(prof),
                        "timestamp": dt.datetime.now().isoformat(),
                    }

                self.detection_stats["member_independence_scores"][f"member_{i}"] = float(prof["independence_score"])

            out["behavioral_trends"] = await self._analyze_behavioral_trends()
            return out
        except Exception:
            return {"profile_updates": {}, "anomaly_detections": {}, "behavioral_trends": {}}

    def _calculate_member_anomaly_score(self, sims: List[float], prof: Dict[str, Any]) -> float:
        try:
            h_avg = float(prof.get("avg_similarity", 0.0))
            h_vol = float(prof.get("volatility", 0.0))
            c_avg = float(np.mean(sims))
            c_vol = float(np.std(sims))
            avg_dev = abs(c_avg - h_avg) / max(h_avg, 0.1)
            vol_dev = abs(c_vol - h_vol) / max(h_vol, 0.1) if h_vol > 1e-9 else 0.0
            return float(min(1.0, (avg_dev + vol_dev) / 2.0))
        except Exception:
            return 0.0

    def _classify_behavioral_anomaly(self, profile: Dict[str, Any]) -> str:
        try:
            a = float(profile.get("avg_similarity", 0.0))
            v = float(profile.get("volatility", 0.0))
            f = float(profile.get("coordination_frequency", 0.0))
            if a > 0.8 and f > 0.7:
                return "high_coordination"
            if v > 0.5:
                return "erratic_behavior"
            if a < 0.2:
                return "isolation_behavior"
            return "moderate_anomaly"
        except Exception:
            return "unknown_anomaly"

    async def _analyze_behavioral_trends(self) -> Dict[str, Any]:
        try:
            trends: Dict[str, Any] = {
                "overall_coordination_trend": "stable",
                "independence_distribution": {},
                "coordination_network_density": 0.0,
                "behavioral_diversity": 0.0,
            }
            if len(self.collusion_history) >= 5:
                recent = [float(e.get("collusion_score", 0.0)) for e in list(self.collusion_history)[-5:]]
                slope = float(self._calculate_slope(recent))
                if slope > 0.1:
                    trends["overall_coordination_trend"] = "increasing"
                elif slope < -0.1:
                    trends["overall_coordination_trend"] = "decreasing"
                else:
                    trends["overall_coordination_trend"] = "stable"

            indeps = [float(p.get("independence_score", 1.0)) for p in self.member_behavior_profiles.values()]
            if indeps:
                trends["independence_distribution"] = {
                    "mean": float(np.mean(indeps)),
                    "std": float(np.std(indeps)),
                    "min": float(np.min(indeps)),
                    "max": float(np.max(indeps)),
                }
                trends["behavioral_diversity"] = float(np.std(indeps))

            if len(self.suspicious_pairs) > 0:
                max_pairs = self.n_members * (self.n_members - 1) / 2.0
                trends["coordination_network_density"] = float(len(self.suspicious_pairs) / max(max_pairs, 1.0))
            return trends
        except Exception:
            return {"overall_coordination_trend": "unknown"}

    def _calculate_slope(self, values: List[float]) -> float:
        try:
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            try:
                return float(np.polyfit(x, values, 1)[0])
            except Exception:
                return 0.0
        except Exception:
            return 0.0

    # ────────────────────────────
    # TEMPORAL ANALYTICS
    # ────────────────────────────
    async def _analyze_temporal_coordination_patterns(self, voting_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            out = {"timing_patterns": {}, "coordination_clusters": [], "temporal_consistency": 0.0, "synchronized_responses": {}}
            if len(self.vote_history) < 5:
                out["status"] = "insufficient_history"
                return out

            recent_votes = list(self.vote_history)[-10:]
            for idx, vote_entry in enumerate(recent_votes):
                ts = vote_entry.get("timestamp", "")
                if ts:
                    sync = await self._analyze_vote_synchronization(vote_entry, recent_votes[max(0, idx - 2) : idx])
                    if float(sync.get("synchronization_score", 0.0)) > 0.7:
                        out["synchronized_responses"][ts] = sync

            scores: List[float] = []
            for pair, hist in self.pair_agreement_history.items():
                lst = list(hist)
                if len(lst) >= 5:
                    scores.append(float(self._calculate_temporal_consistency(lst)))
                    out["timing_patterns"][str(pair)] = {"consistency": float(scores[-1]), "pattern_type": self._classify_temporal_pattern(lst)}
            out["temporal_consistency"] = float(np.mean(scores)) if scores else 0.0
            out["coordination_clusters"] = await self._identify_temporal_coordination_clusters()
            return out
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "temporal_coordination_analysis")
            return {"timing_patterns": {}, "analysis_error": str(error_context)}

    async def _analyze_vote_synchronization(self, current_vote: Dict[str, Any], recent_votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            res = {"synchronization_score": 0.0, "synchronized_members": [], "timing_deviation": 0.0}
            actions = current_vote.get("actions", [])
            if len(actions) < 2:
                return res
            sync_count = 0
            total = 0
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    a = np.asarray(actions[i], dtype=np.float32)
                    b = np.asarray(actions[j], dtype=np.float32)
                    n1, n2 = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                    if n1 <= 1e-9 or n2 <= 1e-9:
                        continue
                    sim = float(np.dot(a, b) / (n1 * n2))
                    if sim > 0.90:
                        sync_count += 1
                        res["synchronized_members"].append((i, j))
                    total += 1
            res["synchronization_score"] = float(sync_count / total) if total > 0 else 0.0
            return res
        except Exception:
            return {"synchronization_score": 0.0, "synchronized_members": [], "timing_deviation": 0.0}

    def _calculate_temporal_consistency(self, history: List[float]) -> float:
        try:
            if len(history) < 3:
                return 0.0
            var = float(np.var(history))
            mean = float(np.mean(history))
            if mean < 1e-9:
                return 0.0
            cv = math.sqrt(max(var, 0.0)) / mean
            return float(max(0.0, 1.0 - cv))
        except Exception:
            return 0.0

    def _classify_temporal_pattern(self, history: List[float]) -> str:
        try:
            if len(history) < 3:
                return "insufficient_data"
            slope = float(self._calculate_slope(history))
            vol = float(np.std(history))
            mean = float(np.mean(history))
            if vol < 0.1:
                if mean > 0.8:
                    return "consistently_high"
                if mean < 0.3:
                    return "consistently_low"
                return "stable_moderate"
            if abs(slope) > 0.1:
                return "trending_up" if slope > 0 else "trending_down"
            return "volatile"
        except Exception:
            return "unknown"

    async def _identify_temporal_coordination_clusters(self) -> List[Dict[str, Any]]:
        try:
            clusters: List[Dict[str, Any]] = []
            if len(self.coordination_events) < 3:
                return clusters
            time_windows: Dict[dt.datetime, List[Dict[str, Any]]] = defaultdict(list)
            recent = list(self.coordination_events)[-20:]
            for ev in recent:
                ts = ev.get("timestamp", "")
                try:
                    t = dt.datetime.fromisoformat(ts)
                    base = t.replace(second=0, microsecond=0, minute=(t.minute // 5) * 5)
                    time_windows[base].append(ev)
                except Exception:
                    continue
            for win, events in time_windows.items():
                if len(events) >= 2:
                    clusters.append(
                        {
                            "window_start": win.isoformat(),
                            "event_count": int(len(events)),
                            "involved_pairs": [tuple(e.get("pair", (-1, -1))) for e in events],
                            "avg_coordination_score": float(np.mean([float(e.get("coordination_score", 0.0)) for e in events])),
                            "cluster_significance": float(len(events) / max(len(recent), 1)),
                        }
                    )
            clusters.sort(key=lambda x: float(x["cluster_significance"]), reverse=True)
            return clusters[:5]
        except Exception:
            return []

    # ────────────────────────────
    # QUALITY METRICS & RECS
    # ────────────────────────────
    async def _calculate_comprehensive_quality_metrics(self) -> Dict[str, Any]:
        try:
            qm: Dict[str, float] = {
                "detection_precision": float(self.quality_metrics.get("detection_precision", 0.0)),
                "detection_recall": float(self.quality_metrics.get("detection_recall", 0.0)),
                "behavioral_accuracy": float(self.quality_metrics.get("behavioral_accuracy", 0.0)),
                "temporal_consistency": float(self.quality_metrics.get("temporal_consistency", 0.0)),
                "overall_effectiveness": 0.0,
            }
            ar = int(self.detection_stats.get("alerts_raised", 0))
            confirmed = int(self.detection_stats.get("confirmed_collusion_events", 0))
            if ar > 0:
                qm["detection_precision"] = float(confirmed / ar)

            if len(self.member_behavior_profiles) > 0:
                indeps = [float(p.get("independence_score", 1.0)) for p in self.member_behavior_profiles.values()]
                qm["detection_recall"] = float(1.0 - (np.mean(indeps) if indeps else 1.0))

            if len(self.collusion_history) >= 5:
                recent = [float(e.get("collusion_score", 0.0)) for e in list(self.collusion_history)[-5:]]
                if recent:
                    std = float(np.std(recent))
                    mean = float(np.mean(recent))
                    qm["behavioral_accuracy"] = float(max(0.0, 1.0 - (std / max(mean, 0.1))))

            scores: List[float] = []
            for hist in self.pair_agreement_history.values():
                lst = list(hist)
                if len(lst) >= 3:
                    scores.append(float(self._calculate_temporal_consistency(lst)))
            if scores:
                qm["temporal_consistency"] = float(np.mean(scores))

            weights = np.array([0.3, 0.3, 0.2, 0.2], dtype=np.float32)
            vals = np.array(
                [qm["detection_precision"], qm["detection_recall"], qm["behavioral_accuracy"], qm["temporal_consistency"]],
                dtype=np.float32,
            )
            qm["overall_effectiveness"] = float(np.dot(weights, vals))
            self.quality_metrics.update(qm)
            return qm
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "quality_metrics_calculation")
            return {"overall_effectiveness": 0.5, "calculation_error": str(error_context)}

    async def _generate_intelligent_detection_recommendations(
        self, collusion_analysis: Dict[str, Any], behavioral_updates: Dict[str, Any], temporal_analysis: Dict[str, Any]
    ) -> List[str]:
        try:
            recs: List[str] = []
            score = float(collusion_analysis.get("collusion_score", self.collusion_score))
            if score > 0.8:
                recs.append("HIGH PRIORITY: Immediate investigation of detected coordination patterns required")
            elif score > 0.5:
                recs.append("MODERATE: Enhanced monitoring and analysis of suspicious member pairs")

            anomalies = behavioral_updates.get("anomaly_detections", {})
            if len(anomalies) > 0:
                recs.append(f"BEHAVIORAL: {len(anomalies)} members showing anomalous behavior patterns – investigate")

            tcons = float(temporal_analysis.get("temporal_consistency", 0.0))
            if tcons > 0.8:
                recs.append("TEMPORAL: High temporal coordination detected – review timing-based collusion")

            net_effects = collusion_analysis.get("coordination_analysis", {}).get("network_effects", {}) if "coordination_analysis" in collusion_analysis else {}
            clusters = net_effects.get("clusters", [])
            if clusters:
                largest = max((len(c) for c in clusters), default=0)
                if largest >= 3:
                    recs.append(f"NETWORK: Large coordination cluster detected ({largest} members) – consider member rotation")

            af = float(self.detection_stats.get("alert_frequency", 0.0))
            if af > 0.3:
                recs.append("SYSTEM: High alert frequency – review detection sensitivity")
            elif af < 0.05:
                recs.append("SYSTEM: Low alert frequency – consider increasing detection sensitivity")

            if self.adaptive_threshold:
                delta = abs(self.current_threshold - self.base_threshold) / max(self.base_threshold, 1e-9)
                if delta > 0.2:
                    recs.append("THRESHOLD: Significant threshold adaptation – review market condition sensitivity")

            eff = float(self.quality_metrics.get("overall_effectiveness", 0.0))
            if eff < 0.4:
                recs.append("QUALITY: Low detection effectiveness – review detection parameters and methods")

            return recs[:6] if recs else ["SYSTEM: Collusion detection operating within normal parameters"]
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendation_generation")
            return [f"Recommendation generation failed: {error_context}"]

    async def _generate_comprehensive_detection_thesis(
        self, collusion_analysis: Dict[str, Any], quality_analysis: Dict[str, Any], recommendations: List[str]
    ) -> str:
        try:
            score = float(collusion_analysis.get("collusion_score", self.collusion_score))
            suspicious_count = len(collusion_analysis.get("suspicious_pairs", []))
            eff = float(quality_analysis.get("overall_effectiveness", 0.0))
            risk = "HIGH" if score > 0.7 else "MODERATE" if score > 0.4 else "LOW"
            parts = [
                f"COLLUSION ANALYSIS: {risk} risk with {score:.1%} coordination score",
                f"DETECTION STATUS: {suspicious_count} suspicious pairs identified from {self.n_members} members",
                f"SYSTEM EFFECTIVENESS: {eff:.1%} detection quality across multiple analysis methods",
            ]
            ar = int(self.detection_stats.get("alerts_raised", 0))
            if ar > 0:
                parts.append(f"ALERT STATUS: {ar} alerts raised with managed escalation")
            if self.adaptive_threshold:
                delta = (self.current_threshold - self.base_threshold) / max(self.base_threshold, 1e-9)
                parts.append(f"THRESHOLD ADAPTATION: {delta:+.1%} adjustment for market conditions")
            anomalies = len([p for p in self.member_behavior_profiles.values() if float(p.get("anomaly_score", 0.0)) > 0.5])
            if anomalies > 0:
                parts.append(f"BEHAVIORAL ANALYSIS: {anomalies} members with anomalous patterns")
            total = int(self.detection_stats.get("total_checks", 0))
            parts.append(f"SYSTEM PERFORMANCE: {total} checks completed with comprehensive analysis")
            priors = [r for r in recommendations if any(k in r for k in ["HIGH PRIORITY", "CRITICAL", "IMMEDIATE"])]
            if priors:
                parts.append(f"ACTION REQUIRED: {len(priors)} high-priority recommendations")
            return " | ".join(parts)
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Detection thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str) -> None:
        try:
            self.smart_bus.set("collusion_score", results["collusion_score"], module="CollusionAuditor", thesis=thesis)
            self.smart_bus.set(
                "suspicious_pairs",
                results["suspicious_pairs"],
                module="CollusionAuditor",
                thesis=f"Suspicious coordination: {len(results['suspicious_pairs'])} pairs under monitoring",
            )
            self.smart_bus.set(
                "member_independence_scores",
                results["member_independence_scores"],
                module="CollusionAuditor",
                thesis=f"Member independence: {len(results['member_independence_scores'])} profiles analyzed",
            )
            self.smart_bus.set(
                "collusion_alerts",
                results["collusion_alerts"],
                module="CollusionAuditor",
                thesis=f"Alert system: {len(results['collusion_alerts'])} active alerts managed",
            )
            self.smart_bus.set(
                "behavioral_profiles",
                results["behavioral_profiles"],
                module="CollusionAuditor",
                thesis=f"Behavioral analysis: {len(results['behavioral_profiles'])} member profiles updated",
            )
            self.smart_bus.set(
                "coordination_events",
                results["coordination_events"],
                module="CollusionAuditor",
                thesis=f"Coordination tracking: {len(results['coordination_events'])} recent events recorded",
            )
            self.smart_bus.set(
                "detection_statistics",
                results["detection_statistics"],
                module="CollusionAuditor",
                thesis=f"Detection statistics: {results['detection_statistics'].get('total_checks', 0)} total checks performed",
            )
            self.smart_bus.set(
                "audit_recommendations",
                results["audit_recommendations"],
                module="CollusionAuditor",
                thesis=f"Audit recommendations: {len(results['audit_recommendations'])} actionable insights",
            )
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    def _get_collusion_init_view(self) -> Dict[str, Any]:
        try:
            # try read back the init payload to mirror bus
            init_payload = self.smart_bus.get("collusion_auditor_initialization", "CollusionAuditor") or {}
            if isinstance(init_payload, dict) and init_payload.get("status"):
                return init_payload
        except Exception:
            pass
        return {
            "status": "initialized",
            "thesis": "Collusion Auditor initialization heartbeat",
            "timestamp": dt.datetime.now().isoformat(),
            "configuration": {
                "members": getattr(self, "n_members", 0),
                "window": getattr(self, "window", 0),
                "detection_methods": list(getattr(self, "detection_methods", {}).keys()) if hasattr(self, "detection_methods") else [],
                "intelligence_parameters": getattr(self, "detection_intelligence", {}),
            },
        }

    # ────────────────────────────
    # LEGACY / PUBLIC API
    # ────────────────────────────
    def check_collusion(self, actions: List[np.ndarray]) -> float:
        try:
            voting_data = {
                "raw_proposals": actions,
                "votes": [float(np.mean(a)) if len(a) > 0 else 0.0 for a in actions],
                "agreement_score": 0.5,
                "market_regime": "unknown",
                "market_context": {},
                "recent_trades": [],
            }
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return self._simple_collusion_check_fallback(actions)
            except RuntimeError:
                pass
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(self._perform_comprehensive_collusion_analysis(voting_data))
                return float(res.get("collusion_score", 0.0))
            finally:
                loop.close()
        except Exception:
            return self._simple_collusion_check_fallback(actions)

    def _simple_collusion_check_fallback(self, actions: List[np.ndarray]) -> float:
        try:
            if len(actions) < 2:
                return 0.0
            sims: List[float] = []
            suspicious = 0
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    v1, v2 = np.asarray(actions[i], dtype=np.float32), np.asarray(actions[j], dtype=np.float32)
                    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                    if n1 <= 1e-9 or n2 <= 1e-9:
                        continue
                    c = float(np.dot(v1, v2) / (n1 * n2))
                    sims.append(c)
                    if c > self.current_threshold:
                        suspicious += 1
            max_pairs = len(actions) * (len(actions) - 1) / 2.0
            score = float(suspicious / max(max_pairs, 1.0))
            self.collusion_score = score
            if sims:
                self.detection_stats["avg_pair_similarity"] = float(np.mean(sims))
            return score
        except Exception:
            return 0.0

    def get_member_independence_scores(self) -> Dict[int, float]:
        return {int(mid): float(p.get("independence_score", 1.0)) for mid, p in self.member_behavior_profiles.items()}

    def _get_recent_collusion_alerts(self) -> List[Dict[str, Any]]:
        try:
            recent = list(self.alert_system.get("alert_history", []))[-5:]
            cleaned: List[Dict[str, Any]] = []
            for a in recent:
                cleaned.append(
                    {
                        "timestamp": a.get("timestamp"),
                        "pair": tuple(a.get("pair", (-1, -1))),
                        "severity": a.get("severity", "info"),
                        "coordination_score": float(a.get("coordination_score", 0.0)),
                        "alert_type": "coordination",
                    }
                )
            return cleaned
        except Exception:
            return []

    def _get_behavioral_profiles_summary(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {}
            for mid, prof in self.member_behavior_profiles.items():
                out[f"member_{int(mid)}"] = {
                    "independence_score": float(prof.get("independence_score", 1.0)),
                    "coordination_frequency": float(prof.get("coordination_frequency", 0.0)),
                    "anomaly_score": float(prof.get("anomaly_score", 0.0)),
                    "consistency_score": float(prof.get("consistency_score", 0.0)),
                }
            return out
        except Exception:
            return {}

    def _get_comprehensive_detection_stats(self) -> Dict[str, Any]:
        return {
            **self.detection_stats,
            "current_threshold": float(self.current_threshold),
            "base_threshold": float(self.base_threshold),
            "adaptive_enabled": bool(self.adaptive_threshold),
            "members_monitored": int(self.n_members),
            "analysis_window": int(self.window),
            "similarity_methods": list(self.similarity_methods),
            "quality_metrics": dict(self.quality_metrics),
            "recent_collusion_trend": self._calculate_recent_collusion_trend(),
        }

    def _calculate_recent_collusion_trend(self) -> str:
        try:
            if len(self.collusion_history) < 3:
                return "insufficient_data"
            recent = [float(e.get("collusion_score", 0.0)) for e in list(self.collusion_history)[-5:]]
            slope = float(self._calculate_slope(recent))
            if slope > 0.1:
                return "increasing"
            if slope < -0.1:
                return "decreasing"
            return "stable"
        except Exception:
            return "unknown"

    def get_observation_components(self) -> np.ndarray:
        try:
            features = [
                float(self.collusion_score),
                float(len(self.suspicious_pairs) / max(self.n_members, 1)),
                float(self.current_threshold),
                float(self.detection_stats.get("avg_pair_similarity", 0.0)),
                float(len(self.vote_history) / max(self.window, 1)),
                float(self.quality_metrics.get("overall_effectiveness", 0.5)),
                float(self.detection_stats.get("alert_frequency", 0.0)),
                float(len(self.coordination_events) / 50.0),
            ]
            arr = np.asarray(features, dtype=np.float32)
            if np.any(~np.isfinite(arr)):
                self.logger.error(f"Invalid collusion observation: {arr}")
                arr = np.nan_to_num(arr, nan=0.5)
            return arr
        except Exception:
            return np.array([0.0, 0.0, 0.9, 0.5, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)

    # Health (public + internal alias)
    def get_health_metrics(self) -> Dict[str, Any]:
        return {
            "module_name": "CollusionAuditor",
            "status": "disabled" if self.is_disabled else "healthy",
            "error_count": int(self.error_count),
            "circuit_breaker_threshold": int(self.circuit_breaker_threshold),
            "total_checks": int(self.detection_stats.get("total_checks", 0)),
            "alerts_raised": int(self.detection_stats.get("alerts_raised", 0)),
            "suspicious_pairs_count": int(len(self.suspicious_pairs)),
            "avg_pair_similarity": float(self.detection_stats.get("avg_pair_similarity", 0.0)),
            "detection_effectiveness": float(self.quality_metrics.get("overall_effectiveness", 0.0)),
            "alert_frequency": float(self.detection_stats.get("alert_frequency", 0.0)),
            "behavioral_profiles_count": int(len(self.member_behavior_profiles)),
            "coordination_events_count": int(len(self.coordination_events)),
            "session_duration": (dt.datetime.now() - dt.datetime.fromisoformat(self.detection_stats["session_start"])).total_seconds() / 3600.0,
        }

    def _get_health_metrics(self) -> Dict[str, Any]:
        return self.get_health_metrics()

    # Operator report
    def get_collusion_report(self) -> str:
        risk = "[ALERT] CRITICAL RISK" if self.collusion_score > 0.8 else "[WARN] HIGH RISK" if self.collusion_score > 0.5 else "[YELLOW] MODERATE RISK" if self.collusion_score > 0.2 else "[OK] LOW RISK"
        recent_10m = len(
            [
                e
                for e in self.coordination_events
                if (dt.datetime.now() - dt.datetime.fromisoformat(e["timestamp"])).total_seconds() < 600.0
            ]
        )
        suspicious_details: List[str] = []
        for pair in list(self.suspicious_pairs)[:5]:
            hist = self.pair_agreement_history.get(pair, [])
            if hist:
                avg_sim = float(np.mean(list(hist)))
                suspicious_details.append(f"  [SEARCH] Members {pair[0]}-{pair[1]}: {avg_sim:.1%} similarity")

        indep_summary: List[str] = []
        for member_id, profile in list(self.member_behavior_profiles.items())[:5]:
            independence = float(profile.get("independence_score", 1.0))
            anomaly = float(profile.get("anomaly_score", 0.0))
            if independence < 0.7 or anomaly > 0.5:
                status = "[ALERT]" if anomaly > 0.7 else "[WARN]"
                indep_summary.append(f"  {status} Member {member_id}: {independence:.1%} independence, {anomaly:.1%} anomaly")

        eff = float(self.quality_metrics.get("overall_effectiveness", 0.0))
        eff_status = "[OK] Excellent" if eff > 0.8 else "[FAST] Good" if eff > 0.6 else "[WARN] Fair" if eff > 0.4 else "[ALERT] Poor"

        return f"""
🕵️ COLLUSION AUDITOR v3.1
═══════════════════════════════════════════════════════════════
[TARGET] Current Status: {risk}
[STATS] Collusion Score: {self.collusion_score:.1%}
🎚️ Detection Threshold: {self.current_threshold:.1%} (Base: {self.base_threshold:.1%})

[CHART] Detection Performance:
• Total Checks: {self.detection_stats['total_checks']}
• Alerts Raised: {self.detection_stats['alerts_raised']}
• Recent Alerts (10min): {recent_10m}
• Alert Frequency: {self.detection_stats.get('alert_frequency', 0.0):.1%}
• Average Pair Similarity: {self.detection_stats.get('avg_pair_similarity', 0.0):.1%}

[SEARCH] Current Surveillance:
• Committee Size: {self.n_members} members
• Suspicious Pairs: {len(self.suspicious_pairs)}
• Coordination Events: {len(self.coordination_events)}
• Members Under Watch: {len(self.alert_system['last_alerts'])}
• Behavioral Profiles: {len(self.member_behavior_profiles)}

[STATS] System Configuration:
• Analysis Window: {self.window} votes
• Similarity Methods: {', '.join(self.similarity_methods)}
• Adaptive Threshold: {'[OK] Enabled' if self.adaptive_threshold else '[FAIL] Disabled'}
• Alert Cooldown: {self.alert_system['cooldown_period']} checks
• Detection Methods: {len(self.detection_methods)} active

[SEARCH] Suspicious Pairs:
{chr(10).join(suspicious_details) if suspicious_details else "  [OK] No suspicious pairs detected"}

[WARN] Member Alerts:
{chr(10).join(indep_summary) if indep_summary else "  [OK] All members showing normal behavior"}

[STATS] Quality Metrics:
• Detection Precision: {self.quality_metrics.get('detection_precision', 0.0):.1%}
• Detection Recall: {self.quality_metrics.get('detection_recall', 0.0):.1%}
• Behavioral Accuracy: {self.quality_metrics.get('behavioral_accuracy', 0.0):.1%}
• Temporal Consistency: {self.quality_metrics.get('temporal_consistency', 0.0):.1%}
• Overall Effectiveness: {eff_status} ({eff:.1%})

[STATS] Recent Activity:
• Vote History: {len(self.vote_history)} entries
• Collusion History: {len(self.collusion_history)} events
• Coordination Events: {len(self.coordination_events)} recorded
• Alert History: {len(self.alert_system.get('alert_history', []))} alerts

[TOOL] System Health:
• Error Count: {self.error_count}/{self.circuit_breaker_threshold}
• Status: {'[ALERT] DISABLED' if self.is_disabled else '[OK] OPERATIONAL'}
• Session Duration: {(dt.datetime.now() - dt.datetime.fromisoformat(self.detection_stats['session_start'])).total_seconds() / 3600:.1f} hours
• Detection Trend: {self._calculate_recent_collusion_trend().title()}

[TARGET] Intelligence:
• Adaptation Rate: {self.detection_intelligence.get('adaptation_rate', 0.15):.1%}
• Sensitivity Target: {self.detection_intelligence.get('sensitivity_target', 0.85):.1%}
• False Positive Threshold: {self.detection_intelligence.get('false_positive_threshold', 0.1):.1%}
• Behavioral Memory: {self.detection_intelligence.get('behavioral_memory', 0.9):.1%}
"""

    # ────────────────────────────
    # HEALTH, ERRORS, STATE
    # ────────────────────────────
    def _update_performance_metric(self, metric_name: str, value: float) -> None:
        try:
            if hasattr(self, "performance_tracker") and self.performance_tracker:
                self.performance_tracker.record_metric("CollusionAuditor", metric_name, float(value), True)
            # Use a dedicated per-metric history map to avoid conflicting with BaseModule's deque _performance_history
            if not hasattr(self, "_metric_history"):
                self._metric_history = defaultdict(lambda: deque(maxlen=50))  # type: ignore[attr-defined]
            self._metric_history[metric_name].append({  # type: ignore[attr-defined]
                "timestamp": dt.datetime.now().isoformat(),
                "value": float(value),
            })
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"Performance metric update failed for {metric_name}: {e}")

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        self.error_count += 1
        ctx = self.error_pinpointer.analyze_error(error, "CollusionAuditor")
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(
                format_operator_message(
                    icon="[ALERT]",
                    message="Collusion Auditor disabled due to repeated errors",
                    error_count=self.error_count,
                    threshold=self.circuit_breaker_threshold,
                )
            )
        self.performance_tracker.record_metric("CollusionAuditor", "process_time_ms", (time.time() - start_time) * 1000.0, False)
        return {
            "collusion_score": 0.0,
            "suspicious_pairs": [],
            "member_independence_scores": {},
            "collusion_alerts": [],
            "behavioral_profiles": {},
            "coordination_events": [],
            "detection_statistics": {"error": str(ctx)},
            "audit_recommendations": ["Investigate collusion auditor errors"],
            "health_metrics": {"status": "error", "error_context": str(ctx)},
            "collusion_auditor_initialization": self._get_collusion_init_view(),
            "_thesis": f"CollusionAuditor error: {ctx}",
        }

    def _get_safe_voting_defaults(self) -> Dict[str, Any]:
        return {
            "votes": [],
            "voting_summary": {},
            "strategy_arbiter_weights": [],
            "raw_proposals": [],
            "member_confidences": [],
            "consensus_direction": "neutral",
            "agreement_score": 0.5,
            "market_context": {},
            "recent_trades": [],
            "market_regime": "unknown",
            "volatility_data": {},
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        return {
            "collusion_score": 0.0,
            "suspicious_pairs": [],
            "member_independence_scores": {},
            "collusion_alerts": [],
            "behavioral_profiles": {},
            "coordination_events": [],
            "detection_statistics": {"status": "disabled"},
            "audit_recommendations": ["Restart collusion auditor system"],
            "health_metrics": {"status": "disabled", "reason": "circuit_breaker_triggered"},
            "collusion_auditor_initialization": self._get_collusion_init_view(),
            "_thesis": "CollusionAuditor disabled via circuit breaker",
        }

    # ────────────────────────────
    # STATE & HOT RELOAD
    # ────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        return {
            "module_info": {"name": "CollusionAuditor", "version": "3.1.0", "last_updated": dt.datetime.now().isoformat()},
            "configuration": {
                "n_members": int(self.n_members),
                "window": int(self.window),
                "base_threshold": float(self.base_threshold),
                "adaptive_threshold": bool(self.adaptive_threshold),
                "similarity_methods": list(self.similarity_methods),
                "debug": bool(self.debug),
            },
            "detection_state": {
                "current_threshold": float(self.current_threshold),
                "collusion_score": float(self.collusion_score),
                "suspicious_pairs": [list(p) for p in self.suspicious_pairs],
                "detection_stats": dict(self.detection_stats),
                "quality_metrics": dict(self.quality_metrics),
            },
            "intelligence_state": {
                "detection_intelligence": dict(self.detection_intelligence),
                "market_adaptation": dict(self.market_adaptation),
                "alert_system": {
                    "cooldown_period": int(self.alert_system["cooldown_period"]),
                    "escalation_threshold": int(self.alert_system["escalation_threshold"]),
                    "severity_levels": list(self.alert_system["severity_levels"]),
                    "auto_investigation": bool(self.alert_system["auto_investigation"]),
                    "last_alerts": {f"{k}": int(v) for k, v in self.alert_system["last_alerts"].items()},
                },
            },
            "behavioral_state": {
                "member_behavior_profiles": {int(k): dict(v) for k, v in self.member_behavior_profiles.items()},
                "pair_agreement_history": {str(k): list(v) for k, v in self.pair_agreement_history.items()},
                "temporal_patterns": {str(k): list(v) for k, v in self.temporal_patterns.items()},
            },
            "history_state": {
                "vote_history": list(self.vote_history)[-20:],
                "collusion_history": list(self.collusion_history)[-30:],
                "coordination_events": list(self.coordination_events)[-20:],
                "alert_patterns": {k: list(v) for k, v in self.alert_patterns.items()},
                "alert_history": list(self.alert_system.get("alert_history", []))[-50:],
            },
            "error_state": {"error_count": int(self.error_count), "is_disabled": bool(self.is_disabled)},
            "performance_metrics": self.get_health_metrics(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        try:
            cfg = state.get("configuration", {})
            self.n_members = int(cfg.get("n_members", self.n_members))
            self.window = int(cfg.get("window", self.window))
            self.base_threshold = float(cfg.get("base_threshold", self.base_threshold))
            self.adaptive_threshold = bool(cfg.get("adaptive_threshold", self.adaptive_threshold))
            self.similarity_methods = list(cfg.get("similarity_methods", self.similarity_methods))
            self.debug = bool(cfg.get("debug", self.debug))

            det = state.get("detection_state", {})
            self.current_threshold = float(det.get("current_threshold", self.base_threshold))
            self.collusion_score = float(det.get("collusion_score", 0.0))
            self.suspicious_pairs = set(tuple(p) for p in det.get("suspicious_pairs", []))
            self.detection_stats.update(det.get("detection_stats", {}))
            self.quality_metrics.update(det.get("quality_metrics", {}))

            intel = state.get("intelligence_state", {})
            self.detection_intelligence.update(intel.get("detection_intelligence", {}))
            self.market_adaptation.update(intel.get("market_adaptation", {}))
            alert_sys = intel.get("alert_system", {})
            if alert_sys:
                self.alert_system["cooldown_period"] = int(alert_sys.get("cooldown_period", self.alert_system["cooldown_period"]))
                self.alert_system["escalation_threshold"] = int(alert_sys.get("escalation_threshold", self.alert_system["escalation_threshold"]))
                self.alert_system["severity_levels"] = list(alert_sys.get("severity_levels", self.alert_system["severity_levels"]))
                self.alert_system["auto_investigation"] = bool(alert_sys.get("auto_investigation", self.alert_system["auto_investigation"]))
                # last_alerts is a mapping from str(pair) -> step int. Keep as-is if present.

            beh = state.get("behavioral_state", {})
            self.member_behavior_profiles = defaultdict(
                lambda: {
                    "avg_similarity": 0.0,
                    "volatility": 0.0,
                    "consistency_score": 0.0,
                    "independence_score": 1.0,
                    "coordination_frequency": 0.0,
                    "anomaly_score": 0.0,
                }
            )
            for mid, prof in beh.get("member_behavior_profiles", {}).items():
                self.member_behavior_profiles[int(mid)] = dict(prof)

            self.pair_agreement_history.clear()
            for pair_str, hist in beh.get("pair_agreement_history", {}).items():
                try:
                    pair_str_clean = pair_str.strip("()")
                    parts = [int(x.strip()) for x in pair_str_clean.split(",")]
                    if len(parts) == 2:
                        self.pair_agreement_history[(parts[0], parts[1])] = deque(hist, maxlen=self.window)
                except Exception:
                    continue

            self.temporal_patterns = defaultdict(list)
            for k, v in beh.get("temporal_patterns", {}).items():
                self.temporal_patterns[str(k)] = list(v)

            hist_state = state.get("history_state", {})
            self.vote_history = deque(hist_state.get("vote_history", []), maxlen=self.window * 2)
            self.collusion_history = deque(hist_state.get("collusion_history", []), maxlen=100)
            self.coordination_events = deque(hist_state.get("coordination_events", []), maxlen=50)
            self.alert_patterns = defaultdict(list, hist_state.get("alert_patterns", {}))
            if "alert_history" in hist_state:
                self.alert_system["alert_history"] = deque(hist_state["alert_history"], maxlen=200)

            err = state.get("error_state", {})
            self.error_count = int(err.get("error_count", 0))
            self.is_disabled = bool(err.get("is_disabled", False))

            self.logger.info(
                format_operator_message(
                    icon="[RELOAD]",
                    message="Collusion Auditor state restored",
                    members=self.n_members,
                    threshold=f"{self.current_threshold:.3f}",
                    suspicious_pairs=len(self.suspicious_pairs),
                    total_checks=self.detection_stats.get("total_checks", 0),
                )
            )
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    # ────────────────────────────
    # RESET & TEARDOWN
    # ────────────────────────────
    def reset(self) -> None:
        super().reset()
        self.collusion_score = 0.0
        self.suspicious_pairs.clear()
        self.current_threshold = float(self.base_threshold)

        self.vote_history.clear()
        self.collusion_history.clear()
        self.coordination_events.clear()

        self.pair_agreement_history.clear()
        self.member_behavior_profiles.clear()
        self.temporal_patterns.clear()
        self.alert_patterns.clear()

        self.detection_stats = {
            "total_checks": 0,
            "alerts_raised": 0,
            "false_positive_rate": 0.0,
            "confirmed_collusion_events": 0,
            "avg_pair_similarity": 0.0,
            "member_independence_scores": {},
            "detection_accuracy": 0.95,
            "alert_frequency": 0.0,
            "session_start": dt.datetime.now().isoformat(),
        }

        # Reset quality metrics
        self.quality_metrics = {
            "detection_precision": 0.0,
            "detection_recall": 0.0,
            "behavioral_accuracy": 0.0,
            "temporal_consistency": 0.0,
            "overall_effectiveness": 0.0,
        }

        # Reset alert system
        self.alert_system["last_alerts"].clear()
        if "alert_history" in self.alert_system:
            self.alert_system["alert_history"].clear()

        # Reset errors/circuit breaker
        self.error_count = 0
        self.is_disabled = False

        self.logger.info(
            format_operator_message(
                icon="[RELOAD]",
                message="Collusion Auditor reset completed",
                status="All detection state cleared and systems reinitialized",
            )
        )

    def __del__(self) -> None:
        """Best-effort cleanup; never raise."""
        try:
            if hasattr(self, "logger") and self.logger:
                self.logger.info(
                    format_operator_message(
                        icon="👋",
                        message="Collusion Auditor shutting down",
                        total_checks=self.detection_stats.get("total_checks", 0),
                        alerts_raised=self.detection_stats.get("alerts_raised", 0),
                    )
                )
        except Exception:
            # Swallow all exceptions during interpreter teardown
            pass

    # ────────────────────────────
    # CONFIDENCE & ACTION
    # ────────────────────────────
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Confidence in the current collusion assessment."""
        try:
            # Lower collusion => higher confidence (bounded)
            detection_reliability = float(np.clip(1.0 - float(self.collusion_score), 0.0, 1.0))

            # Data quality: how filled our windowed history is
            data_quality = float(
                np.clip(len(self.vote_history) / max(float(self.window), 1.0), 0.0, 1.0)
            )

            # Participation: how many unique pairs we track vs. expected
            expected_pairs = float(max(self.n_members * (self.n_members - 1) // 2, 1))
            actual_pairs = float(len(self.pair_agreement_history))
            participation = float(np.clip(actual_pairs / expected_pairs, 0.0, 1.0))

            # Recent consistency of collusion scores
            if len(self.collusion_history) > 3:
                recent_scores = [float(e.get("collusion_score", 0.0)) for e in list(self.collusion_history)[-5:]]
                mean_val = float(np.mean(recent_scores)) if recent_scores else 0.0
                std_val = float(np.std(recent_scores)) if recent_scores else 0.0
                consistency = float(max(0.0, 1.0 - (std_val / max(mean_val, 0.1)))) if mean_val > 0 else 0.5
            else:
                consistency = 0.5

            confidence = (
                0.4 * detection_reliability
                + 0.3 * data_quality
                + 0.2 * participation
                + 0.1 * consistency
            )
            return float(np.clip(confidence, 0.1, 0.95))
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.4  # conservative fallback

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Recommend an integrity action based on current risk posture."""
        try:
            collusion_score = float(self.collusion_score)
            suspicious_pairs_count = int(len(self.suspicious_pairs))

            # Count recent high-severity alerts (last 10 entries)
            recent_alerts_list = list(self.alert_system.get("alert_history", []))[-10:]
            recent_high_severity = sum(
                1
                for a in recent_alerts_list
                if str(a.get("severity", "info")).lower() in {"warning", "critical"}
            )

            # Decision policy
            if collusion_score > 0.80:
                action_type = "emergency_intervention"
                signal_strength = 0.95
                reasoning = f"Critical collusion detected (score: {collusion_score:.3f}) – immediate intervention required"
            elif collusion_score > 0.60:
                action_type = "increase_monitoring"
                signal_strength = 0.80
                reasoning = f"High collusion risk (score: {collusion_score:.3f})"
            elif suspicious_pairs_count > max(self.n_members // 2, 1):
                action_type = "investigate_pairs"
                signal_strength = 0.70
                reasoning = f"Multiple suspicious pairs detected ({suspicious_pairs_count})"
            elif recent_high_severity > 0:
                action_type = "review_alerts"
                signal_strength = 0.60
                reasoning = f"{recent_high_severity} recent high-severity alerts require review"
            elif collusion_score < 0.20:
                action_type = "normal_monitoring"
                signal_strength = 0.20
                reasoning = f"Low collusion risk (score: {collusion_score:.3f}) – normal operations"
            else:
                action_type = "monitor"
                signal_strength = 0.40
                reasoning = "Moderate collusion metrics – continue monitoring"

            # Confidence
            confidence = await self.calculate_confidence({}, **inputs)

            return {
                "action": action_type,
                "signal_strength": float(signal_strength),
                "reasoning": reasoning,
                "collusion_metrics": {
                    "collusion_score": float(collusion_score),
                    "suspicious_pairs_count": int(suspicious_pairs_count),
                    "recent_high_severity_alerts": int(recent_high_severity),
                    "total_members": int(self.n_members),
                    "detection_quality": float(self.quality_metrics.get("overall_effectiveness", 0.5)),
                },
                "alert_summary": {
                    "total_alerts": int(self.detection_stats.get("alerts_raised", 0)),
                    "recent_high_severity": int(recent_high_severity),
                },
                "confidence": float(confidence),
            }
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.error(f"Action proposal failed: {e}")
            return {
                "action": "abstain",
                "signal_strength": 0.0,
                "reasoning": f"Collusion detection error: {str(e)}",
                "confidence": 0.1,
            }
