"""
Collusion Detector
==================
Detects potential collusion patterns among voting committee members.
Analyzes similarity, behavioral patterns, and temporal coordination.

Refactored from collusion_auditor.py (~2035 lines).
~400 lines focused on core collusion detection.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase
from modules.voting.core.types import CollusionResult  # kept for type hints / future use
from modules.voting.core.constants import VotingBusKeys


@module(**module_args("CollusionDetector"))
class CollusionDetector(VotingModuleBase):
    """
    Collusion detection for voting committees.
    
    Detects:
    - Pair-wise similarity (cosine, correlation)
    - Suspicious voting patterns
    - Coordinated timing
    - Behavioral anomalies
    
    Publishes (SmartInfoBus):
    - collusion_score          (float)
    - collusion_detected       (bool)
    - collusion_analysis       (dict)
    - suspicious_pairs         (list[(expert1, expert2)])
    """

    def _module_specific_init(self) -> None:
        """Initialize collusion detection state."""
        # Configuration
        self.n_members = int(self.config.get("n_members", 5))
        self.window = int(self.config.get("window", 10))
        # FIXED: Increased threshold from 0.9 to 0.95 - experts agreeing in trending
        # markets is NORMAL behavior, not collusion. Only flag truly suspicious patterns.
        self.base_threshold = float(self.config.get("threshold", 0.95))
        self.current_threshold = self.base_threshold
        self.adaptive_threshold = bool(self.config.get("adaptive_threshold", True))

        # State
        self.vote_history: deque = deque(maxlen=self.window * 2)
        self.collusion_score: float = 0.0
        self.suspicious_pairs: Set[Tuple[str, str]] = set()
        self.collusion_history: deque = deque(maxlen=100)

        # Pair tracking
        self.pair_agreement_history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

        # Member behavior profiles
        self.member_profiles: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "avg_similarity": 0.0,
                "consistency_score": 0.5,
                "independence_score": 1.0,
                "anomaly_score": 0.0,
            }
        )

        # Statistics
        self.detection_stats: Dict[str, Any] = {
            "total_checks": 0,
            "alerts_raised": 0,
            "avg_pair_similarity": 0.0,
        }

        # Quality metrics
        self.quality_metrics: Dict[str, float] = {
            "detection_precision": 0.0,
            "behavioral_accuracy": 0.0,
            "overall_effectiveness": 0.5,
        }

        # Alert cooldowns
        self.alert_cooldowns: Dict[Tuple[str, str], int] = {}
        self.cooldown_period = int(self.config.get("alert_cooldown", 10))

        self.logger.info(
            f"[COLLUSION] CollusionDetector initialized | "
            f"members={self.n_members} | threshold={self.base_threshold:.2f}"
        )

        # Publish baseline
        self._publish_collusion_baseline()

    # ====================================================================== #
    # Baseline / Bus
    # ====================================================================== #

    def _publish_collusion_baseline(self) -> None:
        """Publish baseline collusion keys."""
        try:
            name = self.__class__.__name__
            self.smart_bus.set(
                "collusion_score",
                0.0,
                module=name,
                thesis="Baseline collusion score",
            )
            self.smart_bus.set(
                "collusion_detected",
                False,
                module=name,
                thesis="Baseline collusion flag",
            )
        except Exception:
            pass

    # ====================================================================== #
    # Main process
    # ====================================================================== #

    async def process(self, **inputs) -> Dict[str, Any]:
        """Detect collusion patterns."""
        start = time.time()
        name = self.__class__.__name__

        try:
            # Decision ID for coordination
            decision_id = self.smart_bus.get("kernel_decision_id", name)

            # Get voting data (expert-level, global – not per instrument)
            voting_data = await self._get_voting_data()
            votes = voting_data.get("votes") or []

            # Fast path for insufficient data
            if len(votes) < 2:
                output = self._insufficient_data_output(decision_id)
                # Keep bus in sync so kernel never sees stale collusion flags
                await self._update_bus(output["collusion_analysis"], output["collusion_thesis"])
                return output

            # Core analysis
            analysis = await self._analyze_collusion(voting_data)
            await self._update_behavioral_profiles(voting_data)

            # Thesis and bus update
            thesis = self._generate_thesis(analysis)
            await self._update_bus(analysis, thesis)

            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, "process", elapsed_ms, True)

            return {
                "collusion_score": float(analysis.get("collusion_score", 0.0)),
                "collusion_analysis": analysis,
                "suspicious_pairs": list(self.suspicious_pairs),
                "member_profiles": dict(self.member_profiles),
                "detection_statistics": dict(self.detection_stats),
                "quality_metrics": dict(self.quality_metrics),
                "collusion_alerts": analysis.get("alerts", []),
                "decision_id": decision_id,
                "collusion_decision_id": decision_id,
                # Contract-expected keys
                "collusion_result": analysis,
                "collusion_detected": bool(analysis.get("collusion_detected", False)),
                "collusion_thesis": thesis,
                "member_independence_scores": {
                    m: float(p.get("independence_score", 1.0))
                    for m, p in self.member_profiles.items()
                },
                "_thesis": thesis,
            }

        except Exception as e:
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(e, "collusion_process")
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)

    async def _get_voting_data(self) -> Dict[str, Any]:
        """Get voting data from SmartInfoBus."""
        name = self.__class__.__name__

        # Note: expert_votes and committee_proposal_vectors are populated by CommitteeCoordinator
        votes = self.smart_bus.get("expert_votes", name) or []
        proposal_vectors = self.smart_bus.get("committee_proposal_vectors", name) or []

        # Member confidences are stored under VotingBusKeys.MEMBER_CONFIDENCES
        member_confidences = (
            self.smart_bus.get(VotingBusKeys.MEMBER_CONFIDENCES, name, default=None)
            or self.smart_bus.get("committee_member_confidences", name)
            or []
        )

        market_regime = self.smart_bus.get("market_regime", name) or "unknown"

        return {
            "votes": votes,
            "proposal_vectors": proposal_vectors,
            "member_confidences": member_confidences,
            "market_regime": market_regime,
        }

    # ====================================================================== #
    # Core analysis
    # ====================================================================== #

    async def _analyze_collusion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform collusion analysis."""
        votes = data.get("votes") or []
        vectors = data.get("proposal_vectors") or []

        self.detection_stats["total_checks"] += 1

        # Pairwise similarities
        pair_similarities = self._calculate_pair_similarities(votes, vectors)

        # Update suspicious pairs and per-member similarity stats
        new_suspicious: Set[Tuple[str, str]] = set()
        member_sim_sum: Dict[str, float] = defaultdict(float)
        member_sim_count: Dict[str, int] = defaultdict(int)

        for (m1, m2), sim in pair_similarities.items():
            sim_f = float(sim)
            if sim_f > self.current_threshold:
                new_suspicious.add((m1, m2))
                self._record_suspicious_pair(m1, m2, sim_f)

            member_sim_sum[m1] += sim_f
            member_sim_sum[m2] += sim_f
            member_sim_count[m1] += 1
            member_sim_count[m2] += 1

        self.suspicious_pairs = new_suspicious

        # Effective member count (actual, not config)
        members_in_pairs: Set[str] = set()
        for m1, m2 in pair_similarities.keys():
            members_in_pairs.add(m1)
            members_in_pairs.add(m2)

        if members_in_pairs:
            n_members_eff = len(members_in_pairs)
        else:
            # Fallback to unique experts in votes, then config
            n_members_eff = len({v.get("expert", "unknown") for v in votes}) or self.n_members

        possible_pairs = max(1, int(n_members_eff * (n_members_eff - 1) / 2))

        # Aggregate similarity stats
        if pair_similarities:
            sim_values = [float(v) for v in pair_similarities.values()]
            avg_sim = float(np.mean(sim_values))
            max_sim = float(max(sim_values))
        else:
            avg_sim = 0.0
            max_sim = 0.0

        suspicious_count = len(self.suspicious_pairs)
        suspicious_ratio = float(suspicious_count) / float(possible_pairs)

        # Collusion score = blend of max, avg, and suspicious ratio
        # FIXED: Reduced weights on max_sim and avg_sim since expert agreement
        # in trending markets is normal, not suspicious
        score = (
            0.25 * max_sim +     # Reduced from 0.4 - max similarity less important
            0.25 * avg_sim +     # Reduced from 0.3 - average similarity less important
            0.50 * suspicious_ratio  # Increased from 0.3 - focus on actual suspicious pairs
        )
        self.collusion_score = float(max(0.0, min(1.0, score)))
        self.collusion_history.append(self.collusion_score)

        # Update detection stats
        self.detection_stats["avg_pair_similarity"] = avg_sim

        # Update per-member avg_similarity
        for member, total in member_sim_sum.items():
            count = member_sim_count.get(member, 1)
            avg = float(total / max(count, 1))
            profile = self.member_profiles[member]
            profile["avg_similarity"] = avg

        # Basic quality metrics
        self._update_quality_metrics(
            avg_similarity=avg_sim,
            suspicious_ratio=suspicious_ratio,
            collusion_score=self.collusion_score,
        )

        # Alerts
        alerts = self._generate_alerts()

        # Adaptive threshold based on regime
        if self.adaptive_threshold:
            self._adapt_threshold(data)

        return {
            "collusion_score": self.collusion_score,
            # FIXED: Increased detection threshold from 0.7 to 0.85
            # Expert agreement in trending markets is normal, not collusion
            "collusion_detected": self.collusion_score > 0.85,
            "suspicious_pair_count": suspicious_count,
            "avg_pair_similarity": avg_sim,
            "max_pair_similarity": max_sim,
            "pair_similarities": {
                f"{k[0]}-{k[1]}": float(v) for k, v in pair_similarities.items()
            },
            "alerts": alerts,
            "threshold": float(self.current_threshold),
        }

    def _calculate_pair_similarities(
        self,
        votes: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> Dict[Tuple[str, str], float]:
        """Calculate pair-wise similarities."""
        similarities: Dict[Tuple[str, str], float] = {}

        # Map member → vote data
        members_data: Dict[str, Dict[str, Any]] = {}
        for i, vote in enumerate(votes):
            member = vote.get("expert", f"member_{i}")
            vote_dict = vote.get("vote", {}) or {}
            confidence = float(vote.get("confidence", 0.5))
            action = vote_dict.get("action", "abstain")
            signal = float(vote_dict.get("signal_strength", vote_dict.get("magnitude", 0.5)))

            members_data[member] = {
                "action": action,
                "confidence": confidence,
                "signal": signal,
                "vector": vectors[i] if i < len(vectors) else [0.0, confidence],
            }

        # All pairs
        members = list(members_data.keys())
        for i, m1 in enumerate(members):
            for m2 in members[i + 1 :]:
                d1 = members_data[m1]
                d2 = members_data[m2]
                sim = self._calculate_similarity(d1, d2)
                similarities[(m1, m2)] = float(sim)

        return similarities

    def _calculate_similarity(
        self,
        d1: Dict[str, Any],
        d2: Dict[str, Any],
    ) -> float:
        """Calculate similarity between two members' votes."""
        try:
            # Action agreement
            action_match = 1.0 if d1["action"] == d2["action"] else 0.0

            # Confidence similarity
            conf_diff = abs(d1["confidence"] - d2["confidence"])
            conf_sim = 1.0 - min(conf_diff, 1.0)

            # Signal similarity
            signal_diff = abs(d1["signal"] - d2["signal"])
            signal_sim = 1.0 - min(signal_diff, 1.0)

            # Vector cosine similarity
            v1 = d1.get("vector", [])
            v2 = d2.get("vector", [])
            if v1 and v2 and len(v1) == len(v2):
                cosine = self._cosine_similarity(v1, v2)
            else:
                cosine = 0.5

            # FIXED: Reduced weight of action_match from 0.4 to 0.15
            # Experts agreeing on direction (all LONG or all SHORT) is NORMAL
            # in trending markets. Focus more on suspicious patterns like
            # identical confidence values or vector similarity.
            similarity = (
                0.15 * action_match +    # Reduced from 0.4 - direction agreement is normal
                0.30 * conf_sim +        # Increased from 0.2 - identical confidence IS suspicious
                0.25 * signal_sim +      # Increased from 0.2 - identical signals more suspicious
                0.30 * cosine            # Increased from 0.2 - vector similarity matters more
            )
            return float(max(0.0, min(1.0, similarity)))
        except Exception:
            return 0.0

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity."""
        try:
            a = np.array(v1, dtype=float)
            b = np.array(v2, dtype=float)

            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))

            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0

            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0

    def _record_suspicious_pair(self, m1: str, m2: str, similarity: float) -> None:
        """Record a suspicious pair into history."""
        pair = (min(m1, m2), max(m1, m2))
        self.pair_agreement_history[pair].append(float(similarity))

    async def _update_behavioral_profiles(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update member behavioral profiles based on latest tick."""
        votes = data.get("votes") or []

        for vote in votes:
            member = vote.get("expert", "unknown")
            confidence = float(vote.get("confidence", 0.5))

            profile = self.member_profiles[member]

            # Consistency: simple proxy from confidence (placeholder for real history)
            profile["consistency_score"] = max(0.0, min(1.0, 0.5 + confidence * 0.2))

            # Independence: penalize members that appear in many suspicious pairs
            involved_pairs = sum(1 for p in self.suspicious_pairs if member in p)
            profile["independence_score"] = max(0.0, 1.0 - involved_pairs * 0.2)

            # Anomaly: high similarity + low independence
            profile["anomaly_score"] = max(
                0.0,
                min(1.0, profile["avg_similarity"] * (1.0 - profile["independence_score"])),
            )

        return dict(self.member_profiles)

    # ====================================================================== #
    # Thresholds, alerts, quality
    # ====================================================================== #

    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate collusion alerts."""
        alerts: List[Dict[str, Any]] = []
        current_check = int(self.detection_stats["total_checks"])

        for pair in self.suspicious_pairs:
            # Cooldown per pair
            last_alert = int(self.alert_cooldowns.get(pair, 0))
            if current_check - last_alert < self.cooldown_period:
                continue

            history = list(self.pair_agreement_history.get(pair, []))
            avg_sim = float(np.mean(history)) if history else 0.0

            if avg_sim > self.current_threshold:
                alerts.append(
                    {
                        "pair": list(pair),
                        "severity": "warning" if avg_sim > 0.85 else "info",
                        "avg_similarity": avg_sim,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                )
                self.alert_cooldowns[pair] = current_check
                self.detection_stats["alerts_raised"] += 1

        return alerts

    def _adapt_threshold(self, data: Dict[str, Any]) -> None:
        """Adapt detection threshold based on market conditions."""
        regime = str(data.get("market_regime", "unknown")).lower()

        multipliers = {
            "trending": 1.10,
            "ranging": 0.90,
            "volatile": 0.85,
            "breakout": 1.20,
            "unknown": 1.00,
        }

        mult = float(multipliers.get(regime, 1.0))
        new_threshold = self.base_threshold * mult
        self.current_threshold = float(min(0.98, max(0.7, new_threshold)))

    def _update_quality_metrics(
        self,
        avg_similarity: float,
        suspicious_ratio: float,
        collusion_score: float,
    ) -> None:
        """Update basic quality metrics for diagnostics."""
        # Heuristic: if suspicious_ratio is small but score is high, precision is lower.
        detection_precision = 1.0 - float(max(0.0, suspicious_ratio - 0.1) * 2.0)
        detection_precision = max(0.0, min(1.0, detection_precision))

        # Behavioral accuracy: we want high score only when similarities are genuinely high
        behavioral_accuracy = 1.0 - abs(collusion_score - avg_similarity)
        behavioral_accuracy = max(0.0, min(1.0, behavioral_accuracy))

        overall = 0.5 * detection_precision + 0.5 * behavioral_accuracy

        self.quality_metrics["detection_precision"] = detection_precision
        self.quality_metrics["behavioral_accuracy"] = behavioral_accuracy
        self.quality_metrics["overall_effectiveness"] = overall

    def _generate_thesis(self, analysis: Dict[str, Any]) -> str:
        """Generate collusion detection thesis."""
        score = float(analysis.get("collusion_score", 0.0))
        detected = bool(analysis.get("collusion_detected", False))
        pairs = int(analysis.get("suspicious_pair_count", 0))

        label = (
            "HIGH_RISK"
            if detected
            else "LOW_RISK"
            if score < 0.3
            else "MODERATE"
        )

        return (
            f"COLLUSION: {label} ({score:.1%}) | "
            f"suspicious_pairs={pairs} | "
            f"threshold={self.current_threshold:.2f}"
        )

    async def _update_bus(self, analysis: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with results."""
        try:
            name = self.__class__.__name__

            score = float(analysis.get("collusion_score", 0.0))
            detected = bool(analysis.get("collusion_detected", False))

            self.smart_bus.set(
                "collusion_score",
                score,
                module=name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "collusion_detected",
                detected,
                module=name,
                thesis="Collusion detection flag",
            )
            self.smart_bus.set(
                "collusion_analysis",
                analysis,
                module=name,
                thesis="Collusion analysis results",
            )
            self.smart_bus.set(
                "suspicious_pairs",
                list(self.suspicious_pairs),
                module=name,
                thesis=f"{len(self.suspicious_pairs)} suspicious pairs detected",
            )
        except Exception as e:
            self.logger.warning(f"Bus update failed: {e}")

    # ====================================================================== #
    # Outputs for degenerate/error cases
    # ====================================================================== #

    def _insufficient_data_output(self, decision_id: str) -> Dict[str, Any]:
        """Output for insufficient data."""
        thesis = "Collusion analysis skipped (insufficient data)"
        analysis = {
            "collusion_detected": False,
            "reason": "insufficient_data",
            "collusion_score": 0.0,
        }
        return {
            "collusion_score": 0.0,
            "collusion_analysis": analysis,
            "suspicious_pairs": [],
            "member_profiles": {},
            "detection_statistics": dict(self.detection_stats),
            "quality_metrics": dict(self.quality_metrics),
            "collusion_alerts": [],
            "decision_id": decision_id,
            "collusion_decision_id": decision_id,
            # Contract-expected keys
            "collusion_result": analysis,
            "collusion_detected": False,
            "collusion_thesis": thesis,
            "member_independence_scores": {},
            "_thesis": thesis,
        }

    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        thesis = f"Collusion detection error: {error}"
        analysis = {
            "error": error,
            "collusion_detected": False,
            "collusion_score": 0.0,
        }
        return {
            "collusion_score": 0.0,
            "collusion_analysis": analysis,
            "suspicious_pairs": [],
            "member_profiles": {},
            "detection_statistics": dict(self.detection_stats),
            "quality_metrics": dict(self.quality_metrics),
            "collusion_alerts": [],
            "decision_id": None,
            "collusion_decision_id": None,
            # Contract-expected keys
            "collusion_result": analysis,
            "collusion_detected": False,
            "collusion_thesis": thesis,
            "member_independence_scores": {},
            "_thesis": thesis,
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE PERSISTENCE - Save/Load module state
    # ═══════════════════════════════════════════════════════════════════

    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.
        
        Saves:
        - Vote history (recent votes for pattern detection)
        - Collusion history (detected collusion events)
        - Member profiles (behavioral profiles)
        - Detection statistics
        - Quality metrics
        """
        # Convert pair tuples to strings for JSON serialization
        pair_history = {}
        for pair, history in self.pair_agreement_history.items():
            key = f"{pair[0]}|{pair[1]}"
            pair_history[key] = list(history)
        
        return {
            "collusion_score": self.collusion_score,
            "vote_history": [
                v if isinstance(v, dict) else {"vote": v}
                for v in list(self.vote_history)[-50:]  # Last 50
            ],
            "collusion_history": list(self.collusion_history),
            "pair_agreement_history": pair_history,
            "member_profiles": {k: dict(v) for k, v in self.member_profiles.items()},
            "detection_stats": dict(self.detection_stats),
            "quality_metrics": dict(self.quality_metrics),
            "suspicious_pairs": [list(p) for p in self.suspicious_pairs],
            "current_threshold": self.current_threshold,
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        """
        Restore custom state from persistence.
        """
        if not state:
            return
        
        # Restore collusion score
        self.collusion_score = float(state.get("collusion_score", 0.0))
        
        # Restore vote history
        vote_hist = state.get("vote_history", [])
        self.vote_history = deque(vote_hist, maxlen=self.window * 2)
        
        # Restore collusion history
        coll_hist = state.get("collusion_history", [])
        self.collusion_history = deque(coll_hist, maxlen=100)
        
        # Restore pair agreement history (convert string keys back to tuples)
        pair_hist = state.get("pair_agreement_history", {})
        for key, history in pair_hist.items():
            parts = key.split("|")
            if len(parts) == 2:
                pair = (parts[0], parts[1])
                self.pair_agreement_history[pair] = deque(history, maxlen=self.window)
        
        # Restore member profiles
        profiles = state.get("member_profiles", {})
        for k, v in profiles.items():
            self.member_profiles[k].update(v)
        
        # Restore statistics
        stats = state.get("detection_stats", {})
        self.detection_stats.update(stats)
        
        # Restore quality metrics
        quality = state.get("quality_metrics", {})
        self.quality_metrics.update(quality)
        
        # Restore suspicious pairs
        susp_pairs = state.get("suspicious_pairs", [])
        self.suspicious_pairs = {tuple(p) for p in susp_pairs if len(p) == 2}
        
        # Restore threshold
        self.current_threshold = float(state.get("current_threshold", self.base_threshold))
        
        self.logger.info(
            f"📂 CollusionDetector state restored | "
            f"score={self.collusion_score:.2f} | "
            f"checks={self.detection_stats.get('total_checks', 0)}"
        )
