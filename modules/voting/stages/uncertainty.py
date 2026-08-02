"""
Uncertainty Sampler
===================
Alternative reality sampling for robustness and uncertainty quantification.
Generates perturbed voting outcomes to assess decision fragility.

Refactored from alternative_reality_sampler.py (~1313 lines).
~400 lines focused on core sampling + diagnostics logic.

v2 Highlights
-------------
- Global fragility/uncertainty with better mapping and normalization.
- Per-instrument fragility/uncertainty for EURUSD/XAUUSD etc.
- Regime-aware, history-aware adaptive sigma.
- Quality metrics: diversity, coverage, uncertainty "self-check".
- State persistence for sigma + histories.
"""

from __future__ import annotations

import datetime
import time
from collections import deque
from typing import Any, Dict, List

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase
from modules.voting.core.constants import VotingBusKeys


@module(**module_args("UncertaintySampler"))
class UncertaintySampler(VotingModuleBase):
    """
    Alternative reality sampling for uncertainty quantification.

    Generates perturbed voting scenarios to assess:
    - Decision robustness/fragility (how often the direction flips)
    - Outcome uncertainty (spread/variance of sampled outcomes)
    - Confidence calibration (how reliable committee confidence is)

    Publishes (global):
    - uncertainty_score       (0-1)
    - fragility_score         (0-1, 0 = robust, 1 = highly fragile)
    - fragility               (alias of fragility_score)
    - uncertainty_analysis    (rich dict)
    - alternative_outcomes    (list of sample dicts, for deep debugging)

    Publishes (per-instrument, if data available):
    - instrument_fragility:   {symbol -> fragility}
    - instrument_uncertainty: {symbol -> uncertainty}
    - fragility_<SYMBOL>      convenience scalar keys per instrument
    """

    # ====================================================================== #
    # Initialization
    # ====================================================================== #

    def _module_specific_init(self) -> None:
        """Initialize sampling state."""
        # Configuration
        self.dim = int(self.config.get("dim", 5))
        self.n_samples = int(self.config.get("n_samples", 8))
        self.base_sigma = float(self.config.get("sigma", 0.05))
        self.current_sigma = float(self.base_sigma)
        self.adaptive_sigma = bool(self.config.get("adaptive_sigma", True))
        self.uncertainty_threshold = float(
            self.config.get("uncertainty_threshold", 0.30)
        )
        self.auto_dim = bool(self.config.get("auto_dim", True))

        # Sigma bounds (hard safety rails)
        self.sigma_bounds = (0.005, 0.25)

        # RNG
        seed = self.config.get("seed")
        self._rng = np.random.default_rng(seed)

        # History
        self.sampling_history: deque = deque(maxlen=120)
        self.uncertainty_history: deque = deque(maxlen=240)
        self._last_uncertainty: float = 0.5

        # Quality metrics
        self.quality_metrics: Dict[str, float] = {
            "sample_diversity": 0.0,        # how many different outcomes we see
            "coverage_efficiency": 0.0,     # how well samples explore plausible space
            "uncertainty_accuracy": 0.0,    # heuristic self-check
            "overall_quality_score": 0.5,
        }

        # Statistics
        self.sampling_stats: Dict[str, Any] = {
            "samples_generated": 0,
            "avg_uncertainty": 0.5,
            "sigma_adaptations": 0,
            "diversity_score": 0.0,
        }

        # Market adaptation multipliers (regime-aware sigma scaling)
        self.regime_multipliers = {
            "trending": 0.8,
            "ranging": 1.0,
            "volatile": 1.6,
            "breakout": 1.2,
            "reversal": 1.4,
            "unknown": 1.1,
        }

        self.logger.info(
            f"[SAMPLER] UncertaintySampler initialized | "
            f"samples={self.n_samples} | sigma={self.base_sigma:.3f}"
        )

        # Publish baseline values so nothing downstream sees stale data
        self._publish_uncertainty_baseline()

    def _publish_uncertainty_baseline(self) -> None:
        """Publish baseline uncertainty keys."""
        try:
            name = self.__class__.__name__
            self.smart_bus.set(
                "uncertainty_score",
                0.5,
                module=name,
                thesis="Baseline uncertainty score",
            )
            self.smart_bus.set(
                "fragility_score",
                0.5,
                module=name,
                thesis="Baseline fragility score",
            )
            self.smart_bus.set(
                "fragility",
                0.5,
                module=name,
                thesis="Baseline fragility alias",
            )
            self.smart_bus.set(
                "instrument_fragility",
                {},
                module=name,
                thesis="Baseline per-instrument fragility",
            )
            self.smart_bus.set(
                "instrument_uncertainty",
                {},
                module=name,
                thesis="Baseline per-instrument uncertainty",
            )
        except Exception:
            # Baseline is best-effort only
            pass

    # ====================================================================== #
    # Main process
    # ====================================================================== #

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """Sample alternative voting outcomes and estimate uncertainty."""
        start = time.time()
        name = self.__class__.__name__

        try:
            # Correlate with kernel decision id
            decision_id = self.smart_bus.get("kernel_decision_id", name)

            # Get voting data (global + per instrument)
            data = await self._get_voting_data()

            # Adapt sigma before sampling
            if self.adaptive_sigma:
                self._adapt_sigma(data)

            # Generate alternative outcomes (global space)
            samples = self._generate_samples(data)

            # Analyze uncertainty (global + per-instrument)
            analysis = self._analyze_uncertainty(samples, data)

            # Generate thesis (global)
            thesis = self._generate_thesis(analysis)

            # Publish to SmartInfoBus
            await self._update_bus(analysis, thesis)

            # Update stats
            self.sampling_stats["samples_generated"] += len(samples)

            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(
                name, "process", elapsed_ms, True
            )

            return {
                "uncertainty_score": analysis.get("uncertainty_score", 0.5),
                "fragility_score": analysis.get("fragility_score", 0.5),
                "fragility": analysis.get("fragility_score", 0.5),  # alias
                "alternative_outcomes": samples,
                "uncertainty_analysis": analysis,
                "sampling_statistics": dict(self.sampling_stats),
                "quality_metrics": dict(self.quality_metrics),
                "current_sigma": float(self.current_sigma),
                "decision_id": decision_id,
                "uncertainty_decision_id": decision_id,
                # Per-instrument outputs
                "instrument_fragility": analysis.get("instrument_fragility", {}),
                "instrument_uncertainty": analysis.get("instrument_uncertainty", {}),
                # Contract-expected keys
                "uncertainty_result": analysis,
                "sampling_uncertainty": analysis.get("uncertainty_score", 0.5),
                "uncertainty_thesis": thesis,
                "effective_samples": len(samples),
                "alternative_samples": samples,
                "confidence_bounds": analysis.get(
                    "confidence_bounds", {"lower": 0.0, "upper": 1.0}
                ),
                "diversity_score": analysis.get("diversity_score", 0.5),
                "sampling_decision_id": decision_id,
                "sampling_fragility": analysis.get("fragility_score", 0.5),
                "_thesis": thesis,
            }

        except Exception as e:
            # ErrorPinpointer is optional
            err_pin = getattr(self, "error_pinpointer", None)
            if err_pin is not None:
                error_context = err_pin.analyze_error(e, "uncertainty_process")
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)

    # ====================================================================== #
    # Data collection & helpers
    # ====================================================================== #

    async def _get_voting_data(self) -> Dict[str, Any]:
        """Get voting data from SmartInfoBus."""
        name = self.__class__.__name__

        # Proposal vectors are the expert-space representation from Committee
        proposal_vectors = (
            self.smart_bus.get("committee_proposal_vectors", name) or []
        )

        # Member confidences: use canonical key with fallback
        member_confidences = (
            self.smart_bus.get(VotingBusKeys.MEMBER_CONFIDENCES, name, default=None)
            or self.smart_bus.get("committee_member_confidences", name)
            or []
        )

        # Global committee decision
        committee_decision = (
            self.smart_bus.get(VotingBusKeys.COMMITTEE_DECISION, name, default=None)
            or self.smart_bus.get("committee_decision", name)
            or {}
        )

        # Per-instrument decisions (from CommitteeCoordinator)
        per_instrument_decisions = (
            self.smart_bus.get("committee_decisions_by_instrument", name) or {}
        )

        market_regime = self.smart_bus.get("market_regime", name) or "unknown"

        return {
            "proposal_vectors": proposal_vectors,
            "member_confidences": member_confidences,
            "committee_decision": committee_decision,
            "committee_decisions_by_instrument": per_instrument_decisions,
            "market_regime": market_regime,
        }

    @staticmethod
    def _normalize_action(action: Any) -> str:
        """
        Normalize action labels to a small canonical set:
        - 'long' for BUY/LONG
        - 'short' for SELL/SHORT
        - 'hold' for FLAT/HOLD/ABSTAIN
        """
        a = str(action).lower()
        if a in ("buy", "long"):
            return "long"
        if a in ("sell", "short"):
            return "short"
        if a in ("hold", "flat", "abstain", "none", ""):
            return "hold"
        # Unknown → neutral/hold for fragility comparison purposes
        return "hold"

    # ====================================================================== #
    # Sigma adaptation
    # ====================================================================== #

    def _adapt_sigma(self, data: Dict[str, Any]) -> None:
        """
        Adapt sampling sigma based on:
        - market regime (coarse multiplier)
        - recent uncertainty (self-calibration)
        """
        regime = str(data.get("market_regime", "unknown")).lower()
        regime_mult = float(self.regime_multipliers.get(regime, 1.0))

        # Regime-based target sigma
        target_sigma = self.base_sigma * regime_mult

        # Self-calibration: if we are consistently too certain (low uncertainty),
        # expand sigma slightly; if everything is always fragile, shrink it.
        if self.uncertainty_history:
            avg_unc = float(np.mean(list(self.uncertainty_history)))
        else:
            avg_unc = self._last_uncertainty

        # Adjust target sigma by uncertainty feedback
        if avg_unc < 0.2:
            target_sigma *= 1.15
        elif avg_unc > 0.7:
            target_sigma *= 0.9

        # Clamp to safe bounds
        target_sigma = max(self.sigma_bounds[0], min(self.sigma_bounds[1], target_sigma))

        # Smooth adaptation
        alpha = 0.25
        new_sigma = float(alpha * target_sigma + (1.0 - alpha) * self.current_sigma)
        new_sigma = max(self.sigma_bounds[0], min(self.sigma_bounds[1], new_sigma))

        if abs(new_sigma - self.current_sigma) > 1e-4:
            self.sampling_stats["sigma_adaptations"] += 1
            self.logger.debug(
                f"[SAMPLER] Sigma adapted: {self.current_sigma:.4f} → {new_sigma:.4f} "
                f"(regime={regime}, avg_unc={avg_unc:.3f})"
            )
            self.current_sigma = new_sigma

    # ====================================================================== #
    # Sampling
    # ====================================================================== #

    def _generate_samples(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate alternative voting scenarios (global expert space).

        Each sample:
        - Adds Gaussian noise to committee_proposal_vectors
        - Clips to a reasonable range
        - Recomputes a simple aggregated outcome (direction + confidence)
        """
        base_vectors = data.get("proposal_vectors") or []
        raw_conf = data.get("member_confidences") or []

        if not base_vectors:
            # No real data - return empty samples (fragility will use neutral defaults)
            return []

        # Normalize confidences: may be dict or list
        if isinstance(raw_conf, dict):
            base_confidences = list(raw_conf.values())
        elif isinstance(raw_conf, list):
            base_confidences = raw_conf
        else:
            base_confidences = []

        base_array = np.array(base_vectors, dtype=np.float64)

        samples: List[Dict[str, Any]] = []

        for i in range(self.n_samples):
            # Slight per-sample jitter of sigma for better coverage
            local_sigma = float(
                self.current_sigma * (0.8 + 0.4 * self._rng.random())
            )

            # Generate perturbation and apply
            noise = self._rng.normal(0.0, local_sigma, base_array.shape)
            perturbed = base_array + noise

            # Clip to sane range (depends on how proposal_vectors are scaled;
            # here we assume roughly -2..2 is a good safety box).
            perturbed = np.clip(perturbed, -2.0, 2.0)

            outcome = self._calculate_outcome(perturbed, base_confidences)

            samples.append(
                {
                    "sample_id": i,
                    "perturbed_vectors": perturbed.tolist(),
                    "outcome": outcome,
                    "perturbation_magnitude": float(np.linalg.norm(noise)),
                    "sigma_used": local_sigma,
                }
            )

        # Track sampling meta in history
        self.sampling_history.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "n_samples": len(samples),
                "sigma": float(self.current_sigma),
            }
        )

        return samples

    def _calculate_outcome(
        self,
        vectors: np.ndarray,
        confidences: List[float],
    ) -> Dict[str, Any]:
        """
        Calculate voting outcome from perturbed vectors (global).

        Very simple aggregator:
        - Uses the first dimension as directional "score"
        - Weights by member confidence if available
        """
        try:
            if vectors.size == 0:
                return {"action": "hold", "confidence": 0.5}

            # First dimension is the signed directional component
            if vectors.ndim == 1:
                directions = vectors
            else:
                directions = vectors[:, 0]

            # Weight by confidences if we have them
            if confidences:
                weights = np.array(
                    confidences[: len(directions)], dtype=np.float64
                )
                # Avoid division by zero
                s = weights.sum()
                if s <= 1e-8:
                    weights = np.full_like(directions, 1.0 / len(directions))
                else:
                    weights = weights / s
                weighted_dir = float(np.dot(directions, weights))
            else:
                weighted_dir = float(np.mean(directions))

            # Map directional score to action
            if weighted_dir > 0.1:
                action = "long"
            elif weighted_dir < -0.1:
                action = "short"
            else:
                action = "hold"

            # Confidence from magnitude (soft)
            conf = max(0.0, min(1.0, 0.5 + abs(weighted_dir) * 0.5))

            return {
                "action": action,
                "confidence": float(conf),
                "direction_score": float(weighted_dir),
            }

        except Exception as e:
            self.logger.warning(f"[SAMPLER] Outcome calculation failed: {e}")
            return {"action": "hold", "confidence": 0.5}

    # ====================================================================== #
    # Uncertainty analysis (global + per instrument)
    # ====================================================================== #

    def _analyze_uncertainty(
        self,
        samples: List[Dict[str, Any]],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze uncertainty from samples.

        Global:
            - How often sampled actions disagree with committee_decision.action
            - How spread sample confidences are
        Per-instrument:
            - Monte Carlo around each instrument's signed score (direction+confidence)
        """
        # ---------------------------- Early exits ---------------------------- #
        per_inst_decisions = data.get("committee_decisions_by_instrument") or {}

        if not samples:
            # Neutral output with optional per-instrument neutral defaults
            instrument_fragility = {inst: 0.5 for inst in per_inst_decisions.keys()}
            instrument_uncertainty = {
                inst: 0.5 for inst in per_inst_decisions.keys()
            }
            return {
                "uncertainty_score": 0.5,
                "fragility_score": 0.5,
                "flip_rate": 0.0,
                "outcome_variance": 0.0,
                "n_samples": 0,
                "original_action": self._normalize_action(
                    data.get("committee_decision", {}).get("action", "hold")
                ),
                "sample_actions": [],
                "instrument_fragility": instrument_fragility,
                "instrument_uncertainty": instrument_uncertainty,
                "confidence_bounds": {"lower": 0.25, "upper": 0.75},
                "diversity_score": 0.5,
                "reason": "No samples generated - using neutral defaults",
            }

        proposal_vectors = data.get("proposal_vectors", [])
        if not proposal_vectors:
            instrument_fragility = {inst: 0.5 for inst in per_inst_decisions.keys()}
            instrument_uncertainty = {
                inst: 0.5 for inst in per_inst_decisions.keys()
            }
            return {
                "uncertainty_score": 0.5,
                "fragility_score": 0.5,
                "flip_rate": 0.0,
                "outcome_variance": 0.0,
                "n_samples": len(samples),
                "original_action": self._normalize_action(
                    data.get("committee_decision", {}).get("action", "hold")
                ),
                "sample_actions": [],
                "instrument_fragility": instrument_fragility,
                "instrument_uncertainty": instrument_uncertainty,
                "confidence_bounds": {"lower": 0.25, "upper": 0.75},
                "diversity_score": 0.5,
                "reason": "No proposal vectors available - using neutral defaults",
            }

        # ---------------------------- Global branch -------------------------- #
        original = data.get("committee_decision", {})
        original_action = self._normalize_action(original.get("action", "hold"))

        sample_actions_raw = [s["outcome"]["action"] for s in samples]
        sample_actions = [self._normalize_action(a) for a in sample_actions_raw]
        sample_confidences = [s["outcome"]["confidence"] for s in samples]

        # Flip rate: how often the sample disagrees with original direction
        flips = sum(1 for a in sample_actions if a != original_action)
        flip_rate = flips / float(len(samples))

        # Fragility mapping (softened; keeps robust decisions near 0)
        fragility = float(min(1.0, max(0.0, (flip_rate - 0.3) * 1.5)))

        # Confidence variance (dispersion of sample confidence)
        if len(sample_confidences) > 1:
            conf_variance = float(np.var(sample_confidences))
        else:
            conf_variance = 0.0

        # Overall uncertainty: blend fragility with confidence variance
        uncertainty = float(
            0.6 * fragility + 0.4 * min(1.0, conf_variance * 4.0)
        )
        uncertainty = float(max(0.0, min(1.0, uncertainty)))

        # Diversity: how balanced the action distribution is
        counts: Dict[str, int] = {}
        for a in sample_actions:
            counts[a] = counts.get(a, 0) + 1
        max_frac = max(counts.values()) / float(len(sample_actions)) if counts else 1.0
        diversity = float(1.0 - max_frac)  # 0 = single mode, 1 = uniform spread

        # Update histories
        self.uncertainty_history.append(uncertainty)
        self._last_uncertainty = uncertainty
        self.sampling_stats["avg_uncertainty"] = float(
            np.mean(list(self.uncertainty_history))
        )
        self.sampling_stats["diversity_score"] = diversity

        # Coverage efficiency: how much directional space we explored vs sigma
        direction_scores = [
            s["outcome"].get("direction_score", 0.0) for s in samples
        ]
        if len(direction_scores) > 1:
            dir_std = float(np.std(direction_scores))
        else:
            dir_std = 0.0
        # Compare standard deviation to sigma scale; clamp to [0,1]
        coverage_eff = float(
            max(0.0, min(1.0, dir_std / (self.current_sigma * 4.0 + 1e-8)))
        )

        # Uncertainty "accuracy": heuristic self-check
        # High fragility should correlate with high diversity and high coverage.
        ideal_unc = 0.5 * diversity + 0.5 * coverage_eff
        uncertainty_accuracy = float(1.0 - min(1.0, abs(uncertainty - ideal_unc) * 2.0))

        self.quality_metrics["sample_diversity"] = diversity
        self.quality_metrics["coverage_efficiency"] = coverage_eff
        self.quality_metrics["uncertainty_accuracy"] = uncertainty_accuracy
        self.quality_metrics["overall_quality_score"] = float(
            0.4 * diversity + 0.3 * coverage_eff + 0.3 * uncertainty_accuracy
        )

        # Confidence bounds: rough interval around 0.5 based on uncertainty
        lower = max(0.0, 0.5 - uncertainty * 0.5)
        upper = min(1.0, 0.5 + uncertainty * 0.5)

        # ---------------------- Per-instrument branch ----------------------- #
        instrument_fragility: Dict[str, float] = {}
        instrument_uncertainty: Dict[str, float] = {}

        # Use instrument decisions as seeds and run lightweight Monte Carlo
        for inst, d in per_inst_decisions.items():
            action_raw = d.get("action", "flat")
            action = self._normalize_action(action_raw)
            conf = float(d.get("confidence", 0.0))
            weighted_score = float(d.get("weighted_score", 0.0))

            if action == "hold":
                # Flat / neutral instruments are inherently low directional fragility
                instrument_fragility[inst] = 0.3
                instrument_uncertainty[inst] = 0.3
                continue

            # Derive base signed score per instrument:
            # direction = sign, magnitude = confidence (plus optional weighted_score)
            base_score = max(0.0, conf)
            if action == "short":
                base_score = -base_score

            if weighted_score != 0.0:
                # Blend the two: 70% confidence, 30% weighted score
                base_score = 0.7 * base_score + 0.3 * weighted_score

            # Monte-Carlo around base_score
            num_inst_samples = max(4, self.n_samples)
            flips_inst = 0
            directions_inst: List[float] = []

            for _ in range(num_inst_samples):
                # Slightly smaller sigma for high-confidence instruments
                # (we already probed global space above)
                local_sigma = self.current_sigma * (0.6 + 0.4 * (1.0 - conf))
                noise = float(self._rng.normal(0.0, local_sigma))
                s = base_score + noise
                directions_inst.append(s)

                if base_score > 0 and s <= 0:
                    flips_inst += 1
                elif base_score < 0 and s >= 0:
                    flips_inst += 1

            flip_rate_inst = flips_inst / float(num_inst_samples)
            frag_i = float(min(1.0, max(0.0, (flip_rate_inst - 0.3) * 1.5)))

            if len(directions_inst) > 1:
                var_i = float(np.var(directions_inst))
            else:
                var_i = 0.0

            unc_i = float(0.6 * frag_i + 0.4 * min(1.0, var_i * 4.0))

            instrument_fragility[inst] = frag_i
            instrument_uncertainty[inst] = unc_i

        return {
            "uncertainty_score": uncertainty,
            "fragility_score": fragility,
            "flip_rate": float(flip_rate),
            "outcome_variance": float(conf_variance),
            "n_samples": len(samples),
            "original_action": original_action,
            "sample_actions": sample_actions,
            "instrument_fragility": instrument_fragility,
            "instrument_uncertainty": instrument_uncertainty,
            "confidence_bounds": {"lower": lower, "upper": upper},
            "diversity_score": diversity,
        }

    # ====================================================================== #
    # Thesis, bus, error
    # ====================================================================== #

    def _generate_thesis(self, analysis: Dict[str, Any]) -> str:
        """Generate uncertainty thesis (global, human-readable)."""
        unc = float(analysis.get("uncertainty_score", 0.5))
        frag = float(analysis.get("fragility_score", 0.5))
        n = int(analysis.get("n_samples", 0))
        flip_rate = float(analysis.get("flip_rate", 0.0))

        label = "HIGH" if unc > 0.6 else "MODERATE" if unc > 0.3 else "LOW"

        return (
            f"UNCERTAINTY: {label} ({unc:.1%}) | fragility={frag:.2f} | "
            f"flip_rate={flip_rate:.2f} | samples={n} | sigma={self.current_sigma:.3f}"
        )

    async def _update_bus(self, analysis: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with results."""
        try:
            name = self.__class__.__name__

            unc = float(analysis.get("uncertainty_score", 0.5))
            frag = float(analysis.get("fragility_score", 0.5))

            self.smart_bus.set(
                "uncertainty_score",
                unc,
                module=name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "fragility_score",
                frag,
                module=name,
                thesis=f"Fragility: {frag:.2f}",
            )
            self.smart_bus.set(
                "fragility",
                frag,
                module=name,
                thesis="Fragility alias",
            )
            self.smart_bus.set(
                "uncertainty_analysis",
                analysis,
                module=name,
                thesis="Uncertainty analysis results",
            )

            # Per-instrument publication
            inst_frag = analysis.get("instrument_fragility", {}) or {}
            inst_unc = analysis.get("instrument_uncertainty", {}) or {}

            self.smart_bus.set(
                "instrument_fragility",
                inst_frag,
                module=name,
                thesis="Per-instrument fragility map",
            )
            self.smart_bus.set(
                "instrument_uncertainty",
                inst_unc,
                module=name,
                thesis="Per-instrument uncertainty map",
            )

            # Convenience per-instrument scalar keys: fragility_<SYMBOL>
            for inst, f in inst_frag.items():
                key = f"fragility_{inst}"
                self.smart_bus.set(
                    key,
                    float(f),
                    module=name,
                    thesis=f"Fragility for {inst}",
                )
        except Exception as e:
            self.logger.warning(f"[SAMPLER] Bus update failed: {e}")

    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        thesis = f"Uncertainty sampling error: {error}"
        return {
            "uncertainty_score": 0.5,
            "fragility_score": 0.5,
            "fragility": 0.5,
            "alternative_outcomes": [],
            "uncertainty_analysis": {"error": error},
            "sampling_statistics": dict(self.sampling_stats),
            "quality_metrics": dict(self.quality_metrics),
            "current_sigma": float(self.current_sigma),
            "decision_id": None,
            "uncertainty_decision_id": None,
            # Per-instrument error defaults
            "instrument_fragility": {},
            "instrument_uncertainty": {},
            # Contract-expected keys
            "uncertainty_result": {
                "error": error,
                "uncertainty_score": 0.5,
                "fragility_score": 0.5,
            },
            "sampling_uncertainty": 0.5,
            "uncertainty_thesis": thesis,
            "effective_samples": 0,
            "alternative_samples": [],
            "confidence_bounds": {"lower": 0.0, "upper": 1.0},
            "diversity_score": 0.5,
            "sampling_decision_id": None,
            "sampling_fragility": 0.5,
            "_thesis": thesis,
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE PERSISTENCE - Save/Load module state
    # ═══════════════════════════════════════════════════════════════════

    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.

        Saves:
        - current_sigma
        - uncertainty history (truncated)
        - sampling history (truncated)
        - quality metrics
        - sampling statistics
        """
        return {
            "current_sigma": self.current_sigma,
            "uncertainty_history": list(self.uncertainty_history)[-80:],
            "sampling_history": list(self.sampling_history)[-80:],
            "quality_metrics": dict(self.quality_metrics),
            "sampling_stats": dict(self.sampling_stats),
            "last_uncertainty": self._last_uncertainty,
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        """
        Restore custom state from persistence.
        """
        if not state:
            return

        try:
            self.current_sigma = float(
                state.get("current_sigma", self.base_sigma)
            )
        except Exception:
            self.current_sigma = self.base_sigma

        unc_hist = state.get("uncertainty_history", [])
        self.uncertainty_history = deque(unc_hist, maxlen=240)

        samp_hist = state.get("sampling_history", [])
        self.sampling_history = deque(samp_hist, maxlen=120)

        quality = state.get("quality_metrics", {})
        self.quality_metrics.update(quality)

        stats = state.get("sampling_stats", {})
        self.sampling_stats.update(stats)

        self._last_uncertainty = float(
            state.get("last_uncertainty", self._last_uncertainty)
        )

        self.logger.info(
            f"📂 UncertaintySampler state restored | "
            f"sigma={self.current_sigma:.4f} | "
            f"avg_uncertainty={self.sampling_stats.get('avg_uncertainty', 0.5):.3f}"
        )
