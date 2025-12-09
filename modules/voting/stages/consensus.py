"""
Consensus Analyzer
==================
Analyzes consensus among voting proposals using multiple dimensions:
direction, magnitude, confidence, and temporal stability.

Refactored from consensus_detector.py (~2150 lines).
~400+ lines focused on core consensus analysis.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase
from modules.voting.core.constants import VotingBusKeys


@module(**module_args("ConsensusAnalyzer"))
class ConsensusAnalyzer(VotingModuleBase):
    """
    Consensus analysis for voting committees.
    
    Analyzes multiple dimensions:
    - Directional consensus (bullish/bearish alignment)
    - Magnitude consensus (signal strength agreement)
    - Confidence consensus (member confidence agreement)
    - Temporal stability (consistency over time)
    
    Publishes on SmartInfoBus:
    - consensus_score               (float, 0–1)
    - consensus_analysis            (dict)
    - consensus_quality_metrics     (dict)
    - consensus_statistics          (dict)
    """

    # ====================================================================== #
    # Initialization
    # ====================================================================== #

    def _module_specific_init(self) -> None:
        """Initialize consensus-specific state."""
        # Configuration
        self.n_members = int(self.config.get("n_members", 5))
        self.threshold = float(self.config.get("threshold", 0.6))
        # If enabled, we publish an additional "effective_consensus_score"
        # that factors in quality metrics (but keep consensus_score stable).
        self.quality_weighting = bool(self.config.get("quality_weighting", True))
        self.temporal_smoothing = bool(self.config.get("temporal_smoothing", True))
        self.smoothing_alpha = float(self.config.get("smoothing_alpha", 0.3))
        self.min_votes = int(self.config.get("min_votes", 2))
        self.direction_tolerance = float(self.config.get("direction_tolerance", 0.1))

        # State
        self.last_consensus: float = 0.0
        self.consensus_history: deque = deque(maxlen=150)

        # Dimension scores (0–1 except sign)
        self.directional_consensus: float = 0.0       # magnitude of directional agreement
        self.directional_consensus_sign: float = 0.0  # -1 short, 0 neutral, +1 long
        self.magnitude_consensus: float = 0.0
        self.confidence_consensus: float = 0.0
        self.temporal_stability: float = 0.0

        # Quality metrics
        self.consensus_quality_metrics: Dict[str, float] = {
            "coherence": 0.5,
            "stability": 0.5,
            "diversity": 0.5,
            "reliability": 0.5,
            "overall_effectiveness": 0.5,
        }

        # Statistics
        self.consensus_stats: Dict[str, Any] = {
            "total_computations": 0,
            "high_consensus_count": 0,
            "low_consensus_count": 0,
            "avg_consensus": 0.5,
        }

        # Member contributions (per-expert diagnostics)
        self.member_contributions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "avg_alignment": 0.5,
                "consistency": 0.5,
                "reliability_score": 0.5,
            }
        )

        self.logger.info(
            f"[CONSENSUS] ConsensusAnalyzer initialized | "
            f"members={self.n_members} | threshold={self.threshold:.2f}"
        )

        # Publish baseline to avoid stale values
        self._publish_consensus_baseline()

    def _publish_consensus_baseline(self) -> None:
        """Publish baseline consensus keys."""
        try:
            name = self.__class__.__name__
            self.smart_bus.set(
                "consensus_score",
                0.0,
                module=name,
                thesis="Baseline consensus score",
            )
        except Exception:
            # Baseline is best-effort; failure here should not crash the module.
            pass

    # ====================================================================== #
    # Main process
    # ====================================================================== #

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """Top-level entry: analyze consensus from voting data."""
        start = time.time()
        name = self.__class__.__name__

        try:
            # Get decision ID for coordination across stages
            decision_id = self.smart_bus.get("kernel_decision_id", name)

            # Get voting data from SmartInfoBus
            voting_data = await self._get_voting_data()

            # Perform core analysis (multi-dimensional consensus)
            analysis = await self._analyze_consensus(voting_data)

            # Update history with the current (raw) consensus score
            self._update_history(analysis)

            # Compute quality metrics (coherence, stability, diversity, etc.)
            quality = await self._calculate_quality_metrics(voting_data, analysis)

            # Optionally derive a quality-weighted "effective" consensus score
            if self.quality_weighting:
                eff = self._apply_quality_weighting(
                    analysis.get("consensus_score", 0.0),
                    quality.get("overall_effectiveness", 0.5),
                )
                analysis["effective_consensus_score"] = eff
            else:
                analysis["effective_consensus_score"] = analysis.get("consensus_score", 0.0)

            # Derive a symbolic consensus direction from the stored sign
            direction_label = "neutral"
            if analysis.get("consensus_exists", False):
                if self.directional_consensus_sign > 0:
                    direction_label = "long"
                elif self.directional_consensus_sign < 0:
                    direction_label = "short"

            # Generate human-readable thesis
            thesis = self._generate_thesis(analysis, quality)

            # Publish to SmartInfoBus
            await self._update_bus(analysis, quality, thesis)

            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, "process", elapsed_ms, True)

            return {
                "consensus_score": analysis["consensus_score"],
                "consensus_analysis": analysis,
                "consensus_quality_metrics": quality,
                "consensus_statistics": dict(self.consensus_stats),
                "directional_consensus": self.directional_consensus,
                "magnitude_consensus": self.magnitude_consensus,
                "confidence_consensus": self.confidence_consensus,
                "temporal_stability": self.temporal_stability,
                "member_contributions": dict(self.member_contributions),
                "decision_id": decision_id,
                "consensus_decision_id": decision_id,
                # Contract-expected keys
                "consensus_result": analysis,
                "agreement_score": analysis.get("consensus_score", 0.0),
                "consensus_direction": direction_label,
                "consensus_confidence": analysis.get("consensus_score", 0.0),
                "consensus_components": {
                    "directional": self.directional_consensus,
                    "magnitude": self.magnitude_consensus,
                    "confidence": self.confidence_consensus,
                    "temporal": self.temporal_stability,
                },
                "consensus_quality": quality,
                "consensus_thesis": thesis,
                "_thesis": thesis,
            }

        except Exception as e:
            # Optional error pinpointer integration
            err_pin = getattr(self, "error_pinpointer", None)
            if err_pin is not None:
                error_context = err_pin.analyze_error(e, "consensus_process")
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)

    async def _get_voting_data(self) -> Dict[str, Any]:
        """Get voting data from SmartInfoBus."""
        name = self.__class__.__name__

        proposal_vectors = self.smart_bus.get("committee_proposal_vectors", name) or []

        member_confidences = (
            self.smart_bus.get(VotingBusKeys.MEMBER_CONFIDENCES, name, default=None)
            or self.smart_bus.get("committee_member_confidences", name)
            or []
        )

        expert_votes = self.smart_bus.get("expert_votes", name) or []
        committee_members = self.smart_bus.get("committee_members", name) or []
        market_regime = self.smart_bus.get("market_regime", name) or "unknown"

        return {
            "proposal_vectors": proposal_vectors,
            "member_confidences": member_confidences,
            "expert_votes": expert_votes,
            "committee_members": committee_members,
            "market_regime": market_regime,
        }

    # ====================================================================== #
    # Core analysis
    # ====================================================================== #

    async def _analyze_consensus(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform multi-dimensional consensus analysis.

        Dimensions:
        - directional_consensus: agreement on sign (long/short) of proposals
        - magnitude_consensus: agreement on absolute strength of proposals
        - confidence_consensus: agreement on member confidence levels
        - temporal_stability: consistency of consensus over past decisions
        """
        vectors = data.get("proposal_vectors") or []
        confidences = data.get("member_confidences") or []

        # Reset metrics for degenerate cases
        if len(vectors) < self.min_votes:
            self.directional_consensus_sign = 0.0
            self.directional_consensus = 0.0
            self.magnitude_consensus = 0.0
            self.confidence_consensus = 0.0
            self.temporal_stability = self._calculate_temporal_stability()
            return {
                "consensus_score": 0.0,
                "consensus_exists": False,
                "reason": "insufficient_data",
                "directional_consensus": 0.0,
                "direction_sign": 0.0,
                "magnitude_consensus": 0.0,
                "confidence_consensus": 0.0,
                "temporal_stability": self.temporal_stability,
                "raw_score": 0.0,
                "vote_count": len(vectors),
            }

        # Directional consensus (sign agreement)
        self.directional_consensus = self._calculate_directional_consensus(vectors)

        # Magnitude consensus (agreement on absolute strength)
        self.magnitude_consensus = self._calculate_magnitude_consensus(vectors)

        # Confidence consensus (agreement on confidence levels)
        self.confidence_consensus = self._calculate_confidence_consensus(confidences)

        # Temporal stability based on past consensus history
        self.temporal_stability = self._calculate_temporal_stability()

        # Combine dimensions into a raw consensus score
        raw_score = (
            0.4 * self.directional_consensus +
            0.25 * self.magnitude_consensus +
            0.20 * self.confidence_consensus +
            0.15 * self.temporal_stability
        )

        # Apply temporal smoothing on top of raw score
        if self.temporal_smoothing and self.last_consensus > 0.0:
            alpha = self.smoothing_alpha
            consensus_score = alpha * raw_score + (1.0 - alpha) * self.last_consensus
        else:
            consensus_score = raw_score

        # Clamp into [0, 1]
        consensus_score = float(max(0.0, min(1.0, consensus_score)))
        self.last_consensus = consensus_score

        return {
            "consensus_score": consensus_score,
            "consensus_exists": consensus_score >= self.threshold,
            "directional_consensus": self.directional_consensus,
            "direction_sign": self.directional_consensus_sign,
            "magnitude_consensus": self.magnitude_consensus,
            "confidence_consensus": self.confidence_consensus,
            "temporal_stability": self.temporal_stability,
            "raw_score": raw_score,
            "vote_count": len(vectors),
        }

    def _calculate_directional_consensus(self, vectors: List[List[float]]) -> float:
        """
        Calculate directional consensus using sign agreement (0–1 magnitude)
        and store the majority direction sign (-1/0/+1).

        We treat the first element of each proposal vector as the signed
        directional component (e.g., net long/short signal).
        """
        try:
            if len(vectors) < self.min_votes:
                self.directional_consensus_sign = 0.0
                return 0.0

            directions: List[int] = []
            for v in vectors:
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    try:
                        val = float(v[0])
                    except Exception:
                        val = 0.0

                    if val > self.direction_tolerance:
                        directions.append(1)
                    elif val < -self.direction_tolerance:
                        directions.append(-1)
                    else:
                        directions.append(0)

            if not directions:
                self.directional_consensus_sign = 0.0
                return 0.0

            from collections import Counter

            counts = Counter(directions)
            most_common = counts.most_common(1)
            if not most_common:
                self.directional_consensus_sign = 0.0
                return 0.0

            majority_value, majority_count = most_common[0]

            # If the majority is "neutral" (0), treat as no directional consensus.
            if majority_value == 0:
                self.directional_consensus_sign = 0.0
                return 0.0

            self.directional_consensus_sign = float(majority_value)
            agreement = majority_count / len(directions)
            return float(max(0.0, min(1.0, agreement)))

        except Exception as e:
            self.logger.warning(f"[CONSENSUS] Directional consensus failed: {e}")
            self.directional_consensus_sign = 0.0
            return 0.0

    def _calculate_magnitude_consensus(self, vectors: List[List[float]]) -> float:
        """
        Calculate magnitude consensus using the coefficient of variation (CV)
        of absolute directional strengths.
        
        CV = std / mean; we map:
        - CV = 0   → consensus = 1
        - CV >= 1  → consensus ~ 0 (clamped)
        """
        try:
            if len(vectors) < self.min_votes:
                return 0.0

            magnitudes: List[float] = []
            for v in vectors:
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    try:
                        magnitudes.append(abs(float(v[0])))
                    except Exception:
                        continue

            if len(magnitudes) < self.min_votes:
                return 0.0

            mean = float(np.mean(magnitudes))
            std = float(np.std(magnitudes))

            if mean <= 0.0:
                # If mean is zero, treat as "no magnitude consensus" but not an error.
                return 0.5

            cv: float = std / mean
            # Map CV into [0, 1] with simple linear rule and clamp.
            consensus = 1.0 - cv
            return float(max(0.0, min(1.0, consensus)))

        except Exception as e:
            self.logger.warning(f"[CONSENSUS] Magnitude consensus failed: {e}")
            return 0.0

    def _calculate_confidence_consensus(self, confidences: List[float]) -> float:
        """
        Calculate confidence consensus via coefficient of variation on
        member confidence values.
        
        Identical or very similar confidences → high consensus.
        Highly scattered confidences → low consensus.
        """
        try:
            if len(confidences) < self.min_votes:
                return 0.0

            conf_values = [
                float(c) for c in confidences if isinstance(c, (int, float))
            ]
            if len(conf_values) < self.min_votes:
                return 0.0

            mean = float(np.mean(conf_values))
            std = float(np.std(conf_values))

            if mean <= 0.0:
                return 0.5

            cv: float = std / mean
            consensus = 1.0 - cv
            return float(max(0.0, min(1.0, consensus)))

        except Exception as e:
            self.logger.warning(f"[CONSENSUS] Confidence consensus failed: {e}")
            return 0.0

    def _calculate_temporal_stability(self) -> float:
        """
        Calculate temporal stability of consensus based on recent history.

        High stability means consensus_score has been relatively stable
        (low standard deviation) over the last N decisions.
        """
        try:
            if len(self.consensus_history) < 3:
                # Not enough history; treat as neutral stability.
                return 0.5

            recent = list(self.consensus_history)[-10:]
            if len(recent) < 2:
                return 0.5

            std = float(np.std(recent))
            # Map std into [0, 1]: low std → high stability, high std → low stability
            # std=0 -> 1.0; std >= 0.5 → near 0.0 (clamped)
            stability = 1.0 - min(1.0, std * 2.0)
            return float(max(0.0, min(1.0, stability)))
        except Exception as e:
            self.logger.warning(f"[CONSENSUS] Temporal stability failed: {e}")
            return 0.5

    # ====================================================================== #
    # Quality metrics & member contributions
    # ====================================================================== #

    async def _calculate_quality_metrics(
        self,
        data: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Calculate consensus quality metrics.
        
        - coherence: how well dimensions align with each other
        - stability: temporal stability of consensus over recent history
        - diversity: how far from "dangerous unanimity" the score is
        - reliability: directly uses consensus_score as reliability proxy
        - overall_effectiveness: aggregate metric for diagnostics/weighting
        """
        try:
            score = float(analysis.get("consensus_score", 0.0))

            # 1) Coherence: if directional/magnitude/confidence strongly disagree,
            # std will be large → low coherence.
            dimensions: List[float] = [
                self.directional_consensus,
                self.magnitude_consensus,
                self.confidence_consensus,
            ]
            if len(dimensions) >= 2:
                dim_std = float(np.std(dimensions))
                coherence = 1.0 - min(1.0, dim_std * 2.0)
            else:
                coherence = 0.5

            # 2) Stability: directly use self.temporal_stability
            stability = float(self.temporal_stability)

            # 3) Diversity: we prefer some disagreement;
            # best diversity around consensus_score ~ 0.5.
            diversity = 1.0 - min(1.0, abs(score - 0.5) * 2.0)

            # 4) Overall effectiveness: simple average
            effectiveness = (coherence + stability + diversity) / 3.0

            self.consensus_quality_metrics = {
                "coherence": float(max(0.0, min(1.0, coherence))),
                "stability": float(max(0.0, min(1.0, stability))),
                "diversity": float(max(0.0, min(1.0, diversity))),
                "reliability": float(max(0.0, min(1.0, score))),
                "overall_effectiveness": float(max(0.0, min(1.0, effectiveness))),
            }

            # Update per-member contributions based on current consensus
            self._update_member_contributions(data, analysis)

            return self.consensus_quality_metrics

        except Exception as e:
            self.logger.warning(f"[CONSENSUS] Quality metrics failed: {e}")
            return self.consensus_quality_metrics

    def _update_member_contributions(
        self,
        data: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> None:
        """
        Update member contributions heuristically:

        - avg_alignment: EMA of how often an expert aligns with consensus direction.
        - consistency: proxy from expert confidence.
        - reliability_score: combines alignment with overall consensus reliability.
        """
        expert_votes = data.get("expert_votes") or []
        consensus_sign = float(analysis.get("direction_sign", 0.0))
        consensus_score = float(analysis.get("consensus_score", 0.0))

        if not expert_votes:
            return

        beta = 0.3  # EMA update rate

        for vote in expert_votes:
            member = vote.get("expert", "unknown")
            if not member:
                continue

            contrib = self.member_contributions[member]

            # Determine expert directional sign from vote
            direction_sign = 0.0
            vote_dict = vote.get("vote", {}) or {}
            action = str(vote_dict.get("action", "")).upper()
            try:
                signal_strength = float(
                    vote_dict.get("signal_strength", vote_dict.get("magnitude", 0.0)) or 0.0
                )
            except Exception:
                signal_strength = 0.0

            if abs(signal_strength) > self.direction_tolerance:
                direction_sign = 1.0 if signal_strength > 0 else -1.0
            else:
                # Fallback to action label if signal_strength is weak/absent
                if action in ("LONG", "BUY"):
                    direction_sign = 1.0
                elif action in ("SHORT", "SELL"):
                    direction_sign = -1.0
                else:
                    direction_sign = 0.0

            # Alignment: 1 if aligned with consensus direction, 0 if opposite,
            # 0.5 if either side is neutral.
            if consensus_sign == 0.0 or direction_sign == 0.0:
                alignment = 0.5
            else:
                alignment = 1.0 if consensus_sign == direction_sign else 0.0

            old_align = contrib["avg_alignment"]
            contrib["avg_alignment"] = (1.0 - beta) * old_align + beta * alignment

            # Consistency: proxy from expert confidence
            try:
                conf = float(vote.get("confidence", 0.5) or 0.5)
            except Exception:
                conf = 0.5
            contrib["consistency"] = max(0.0, min(1.0, 0.5 + conf * 0.5))

            # Reliability combines how often this expert aligns with consensus
            # and how reliable consensus itself is.
            contrib["reliability_score"] = float(
                max(0.0, min(1.0, contrib["avg_alignment"] * (0.5 + consensus_score * 0.5)))
            )

    def _apply_quality_weighting(self, score: float, effectiveness: float) -> float:
        """
        Apply a mild quality-based scaling to consensus_score.

        We keep this conservative to avoid destabilizing downstream consumers:
        effective_score = score * (0.7 + 0.3 * effectiveness)

        - If effectiveness = 1.0 → scale = 1.0
        - If effectiveness = 0.0 → scale = 0.7
        """
        base = float(score)
        eff = float(effectiveness)
        scale = 0.7 + 0.3 * max(0.0, min(1.0, eff))
        effective = base * scale
        return float(max(0.0, min(1.0, effective)))

    # ====================================================================== #
    # History / statistics / thesis
    # ====================================================================== #

    def _update_history(self, analysis: Dict[str, Any]) -> None:
        """Update consensus history and stats based on current score."""
        score = float(analysis.get("consensus_score", 0.0))
        self.consensus_history.append(score)

        # Update stats
        self.consensus_stats["total_computations"] += 1
        if score >= self.threshold:
            self.consensus_stats["high_consensus_count"] += 1
        else:
            self.consensus_stats["low_consensus_count"] += 1

        n = self.consensus_stats["total_computations"]
        old_avg = float(self.consensus_stats["avg_consensus"])
        self.consensus_stats["avg_consensus"] = (old_avg * (n - 1) + score) / max(1, n)

    def _generate_thesis(
        self,
        analysis: Dict[str, Any],
        quality: Dict[str, float],
    ) -> str:
        """Generate a human-readable consensus thesis string."""
        score = float(analysis.get("consensus_score", 0.0))
        exists = bool(analysis.get("consensus_exists", False))
        eff = float(analysis.get("effective_consensus_score", score))

        label = (
            "STRONG"
            if score > 0.7
            else "MODERATE"
            if score > 0.4
            else "WEAK"
        )

        return (
            f"CONSENSUS: {label} ({score:.1%}, effective={eff:.1%}) | "
            f"exists={exists} | "
            f"directional={self.directional_consensus:.2f} | "
            f"magnitude={self.magnitude_consensus:.2f} | "
            f"quality={quality.get('overall_effectiveness', 0.5):.2f}"
        )

    async def _update_bus(
        self,
        analysis: Dict[str, Any],
        quality: Dict[str, float],
        thesis: str,
    ) -> None:
        """Update SmartInfoBus with consensus results."""
        try:
            name = self.__class__.__name__

            self.smart_bus.set(
                "consensus_score",
                analysis.get("consensus_score", 0.0),
                module=name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "consensus_analysis",
                analysis,
                module=name,
                thesis="Consensus analysis results",
            )
            self.smart_bus.set(
                "consensus_quality_metrics",
                quality,
                module=name,
                thesis="Consensus quality metrics",
            )
        except Exception as e:
            self.logger.warning(f"[CONSENSUS] Bus update failed: {e}")

    # ====================================================================== #
    # Error handling
    # ====================================================================== #

    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        thesis = f"Consensus error: {error}"
        return {
            "consensus_score": 0.0,
            "consensus_analysis": {
                "error": error,
                "consensus_exists": False,
                "consensus_score": 0.0,
            },
            "consensus_quality_metrics": self.consensus_quality_metrics,
            "consensus_statistics": dict(self.consensus_stats),
            "directional_consensus": 0.0,
            "magnitude_consensus": 0.0,
            "confidence_consensus": 0.0,
            "temporal_stability": 0.0,
            "member_contributions": {},
            "decision_id": None,
            "consensus_decision_id": None,
            # Contract-expected keys
            "consensus_result": {
                "error": error,
                "consensus_exists": False,
                "consensus_score": 0.0,
            },
            "agreement_score": 0.0,
            "consensus_direction": "neutral",
            "consensus_confidence": 0.0,
            "consensus_thesis": thesis,
            "consensus_components": {},
            "consensus_quality": self.consensus_quality_metrics,
            "_thesis": thesis,
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE PERSISTENCE - Save/Load module state
    # ═══════════════════════════════════════════════════════════════════

    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.

        Saves:
        - Consensus history
        - Last consensus score
        - Member contributions
        - Quality metrics
        - Statistics
        """
        return {
            "last_consensus": self.last_consensus,
            "consensus_history": list(self.consensus_history),
            "member_contributions": {k: dict(v) for k, v in self.member_contributions.items()},
            "consensus_quality_metrics": dict(self.consensus_quality_metrics),
            "consensus_stats": dict(self.consensus_stats),
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        """
        Restore custom state from persistence.
        """
        if not state:
            return

        try:
            self.last_consensus = float(state.get("last_consensus", 0.0))
        except Exception:
            self.last_consensus = 0.0

        hist = state.get("consensus_history", [])
        self.consensus_history = deque(hist, maxlen=150)

        contribs = state.get("member_contributions", {})
        for k, v in contribs.items():
            self.member_contributions[k].update(v)

        qm = state.get("consensus_quality_metrics", {})
        self.consensus_quality_metrics.update(qm)

        stats = state.get("consensus_stats", {})
        self.consensus_stats.update(stats)

        self.logger.info(
            f"📂 ConsensusAnalyzer state restored | "
            f"last_consensus={self.last_consensus:.2f} | "
            f"computations={self.consensus_stats.get('total_computations', 0)}"
        )
