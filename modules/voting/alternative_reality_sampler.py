# modules/voting/alternative_reality_sampler.py
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Deque, Callable, ContextManager, cast
from collections import deque

from modules.contracts import module_args
import numpy as np

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities


# ─────────────────────────────────────────────────────────────
# Config Schema
# ─────────────────────────────────────────────────────────────
@dataclass
class SamplerConfig:
    dim: int = 5
    n_samples: int = 8
    sigma: float = 0.05
    adaptive_sigma: bool = True
    uncertainty_threshold: float = 0.30
    debug: bool = False
    auto_dim: bool = True
    # intelligence / bounds
    sigma_bounds: Tuple[float, float] = (0.005, 0.25)
    adaptation_rate: float = 0.12
    diversity_target: float = 0.15
    convergence_threshold: float = 0.02
    exploration_momentum: float = 0.85
    uncertainty_sensitivity: float = 0.70
    # history limits
    history_samples: int = 120
    history_uncert: int = 240
    history_effect: int = 180
    # RNG (seed=None → nondeterministic)
    seed: Optional[int] = None


# ─────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────
@module(**module_args(
    "AlternativeRealitySampler",
    description="Alternative voting outcome sampling for robustness & uncertainty quantification",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class AlternativeRealitySampler(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    PRODUCTION v3.1: hardened, deterministic sampling with auxiliary diagnostics publishing.
    Declared outputs are returned; the orchestrator publishes them. Diagnostics are written
    under 'voting/ars/*' to avoid single-writer conflicts.
    """

    # ── init ───────────────────────────────────────────────
    def _initialize(self):
        # services
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="AlternativeRealitySampler",
            log_path="logs/voting/alternative_reality_sampler.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
            info_bus_aware=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("AlternativeRealitySampler", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor(auto_start=False)

        # typed config
        self.config_t = SamplerConfig(
            dim=int(self.config.get("dim", 5)),
            n_samples=int(self.config.get("n_samples", 8)),
            sigma=float(self.config.get("sigma", 0.05)),
            adaptive_sigma=bool(self.config.get("adaptive_sigma", True)),
            uncertainty_threshold=float(self.config.get("uncertainty_threshold", 0.30)),
            debug=bool(self.config.get("debug", False)),
            auto_dim=bool(self.config.get("auto_dim", True)),
            # ensure fixed-length Tuple[float, float] for type-checkers
            sigma_bounds=(lambda sb: (float(sb[0]), float(sb[1]))
                          if isinstance(sb, (list, tuple)) and len(sb) >= 2 else (0.005, 0.25))(
                self.config.get("sigma_bounds", (0.005, 0.25))
            ),
            adaptation_rate=float(self.config.get("adaptation_rate", 0.12)),
            diversity_target=float(self.config.get("diversity_target", 0.15)),
            convergence_threshold=float(self.config.get("convergence_threshold", 0.02)),
            exploration_momentum=float(self.config.get("exploration_momentum", 0.85)),
            uncertainty_sensitivity=float(self.config.get("uncertainty_sensitivity", 0.70)),
            history_samples=int(self.config.get("history_samples", 120)),
            history_uncert=int(self.config.get("history_uncert", 240)),
            history_effect=int(self.config.get("history_effect", 180)),
            seed=self.config.get("seed"),
        )

        # core params
        self.dim: int = self.config_t.dim
        self.n_samples: int = self.config_t.n_samples
        self.base_sigma: float = self.config_t.sigma
        self.current_sigma: float = self.base_sigma
        self.adaptive_sigma: bool = self.config_t.adaptive_sigma
        self.uncertainty_threshold: float = self.config_t.uncertainty_threshold
        self.debug: bool = self.config_t.debug
        self.auto_dim: bool = self.config_t.auto_dim

        # RNG and locks
        self._rng = np.random.default_rng(self.config_t.seed)
        self._lock = threading.RLock()

        # histories
        self.sampling_history = deque(maxlen=self.config_t.history_samples)
        self.uncertainty_history = deque(maxlen=self.config_t.history_uncert)
        self.effectiveness_history = deque(maxlen=self.config_t.history_effect)

        # strategies
        self.sampling_strategies = {
            "random_gaussian": {"weight": 0.40, "active": True},
            "structured_perturbation": {"weight": 0.30, "active": True},
            "systematic_exploration": {"weight": 0.20, "active": True},
            "uncertainty_guided": {"weight": 0.10, "active": True},
        }

        self.market_adaptation = {
            "regime_multipliers": {
                "trending": 0.8,
                "ranging": 1.0,
                "volatile": 1.6,
                "breakout": 1.2,
                "reversal": 1.4,
                "unknown": 1.1,
            },
            "volatility_multipliers": {
                "very_low": 0.6,
                "low": 0.8,
                "medium": 1.0,
                "high": 1.4,
                "extreme": 2.0,
            },
        }

        self.quality_metrics = {
            "sample_diversity": 0.0,
            "coverage_efficiency": 0.0,
            "uncertainty_accuracy": 0.0,
            "adaptation_success_rate": 0.0,
            "exploration_completeness": 0.0,
            "overall_quality_score": 0.0,
        }
        self.sampling_stats = {
            "samples_generated": 0,
            "total_samples_created": 0,
            "avg_uncertainty": 0.5,
            "sigma_adaptations": 0,
            "effective_samples": 0,
            "diversity_score": 0.0,
            "convergence_rate": 0.0,
            "exploration_efficiency": 0.0,
            "best_uncertainty_estimate": 0.5,
            "session_start": dt.datetime.now().isoformat(),
        }

        self.sampling_intelligence = {
            "sigma_bounds": self.config_t.sigma_bounds,
            "adaptation_rate": self.config_t.adaptation_rate,
            "diversity_target": self.config_t.diversity_target,
            "convergence_threshold": self.config_t.convergence_threshold,
            "exploration_momentum": self.config_t.exploration_momentum,
            "uncertainty_sensitivity": self.config_t.uncertainty_sensitivity,
        }

        self.error_count, self.circuit_breaker_threshold = 0, 5
        self.is_disabled = False
        self._last_dim_logged = None

        self._generate_initialization_thesis()
        self.logger.info(
            format_operator_message(
                icon="🎲",
                message=f"ARS v{getattr(self.metadata,'version','3.1.0')} ready",
                dimensions=self.dim,
                samples=self.n_samples,
                sigma=f"{self.base_sigma:.3f}",
                adaptive=self.adaptive_sigma,
            )
        )

    # ── process (contract) ─────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Validate inputs, compute sampling, return declared provides + '_thesis'.
        Orchestrator publishes; we only write auxiliary diagnostics.
        """
        with self.performance_tracker.track("AlternativeRealitySampler", "process"):
            self.validate_inputs(inputs)

            if self.is_disabled:
                return self._generate_disabled_response()

            try:
                voting_data = await self._get_comprehensive_voting_data()
                # Soft budget/trivial-case fast path: skip heavy sampling when committee is trivial
                members = voting_data.get('strategy_arbiter_weights') or voting_data.get('voting_weights') or {}
                n_members = (len(members) if isinstance(members, dict) else (len(members) if isinstance(members, (list, tuple)) else 0))
                if n_members <= 1:
                    now = dt.datetime.now().isoformat()
                    minimal = {
                        "alternative_samples": [],
                        "sampling_uncertainty": 0.5,
                        "diversity_score": 0.0,
                        "sampling_stats": {"samples_generated": 0, "avg_uncertainty": 0.5},
                        "effective_samples": 0,
                        "confidence_bounds": {"lower": 0.0, "upper": 0.0},
                        "sampling_recommendations": ["insufficient_voters: neutral handling"],
                        "alternative_reality_sampler_initialization": self._get_ars_init_view(),
                        "decision_id": voting_data.get("decision_id"),
                        "tick_ts": voting_data.get("tick_ts") or now,
                        "_thesis": "Alternative sampling skipped (single/no voter); fast-path applied",
                    }
                    # Publish diagnostics surfaces lightly (with alias)
                    with contextlib.suppress(Exception):
                        self.smart_bus.set("sampling_uncertainty", 0.5, module="AlternativeRealitySampler", thesis="ARS sampling uncertainty")
                        self.smart_bus.set("uncertainty", 0.5, module="AlternativeRealitySampler", thesis="ARS uncertainty alias")
                        self.smart_bus.set("effective_samples", 0, module="AlternativeRealitySampler", thesis="ARS effective samples")
                    return minimal
                await self._update_sampling_parameters_comprehensive(voting_data)
                effectiveness = await self._analyze_sampling_effectiveness_comprehensive(voting_data)
                updates = await self._update_sampling_strategy_weights(effectiveness)
                quality = await self._calculate_comprehensive_quality_metrics()

                # base weights
                base_w = self._normalize_weights_input(voting_data.get("strategy_arbiter_weights", []))
                if self.auto_dim and base_w.size and base_w.size != self.dim:
                    self._adapt_dimension(int(base_w.size))
                    base_w = self._pad_or_trim(base_w, self.dim)

                samples = await self.sample_comprehensive(base_w, voting_data)

                results = {
                    "alternative_samples": samples.tolist(),
                    "sampling_uncertainty": self.sampling_stats.get("avg_uncertainty", 0.5),
                    "diversity_score": self.sampling_stats.get("diversity_score", 0.0),
                    "sampling_stats": self._get_comprehensive_sampling_stats(),
                    "effective_samples": self.sampling_stats.get("effective_samples", 0),
                    "confidence_bounds": self._calculate_confidence_bounds(),
                    "sampling_recommendations": await self._generate_intelligent_sampling_recommendations(
                        effectiveness, quality
                    ),
                    "alternative_reality_sampler_initialization": self._get_ars_init_view(),
                    "decision_id": voting_data.get("decision_id"),
                    "tick_ts": voting_data.get("tick_ts") or dt.datetime.now().isoformat(),
                }

                thesis = await self._generate_comprehensive_sampling_thesis(effectiveness, quality, updates)
                results["_thesis"] = thesis

                # Publish uncertainty surfaces for downstream readers that pull from bus directly
                try:
                    unc = float(results.get("sampling_uncertainty", 0.5))
                    eff = int(results.get("effective_samples", 0))
                    # simple fragility heuristic from diversity (lower diversity => higher fragility)
                    div = float(results.get("diversity_score", 0.0))
                    fragility = max(0.0, min(1.0, 1.0 - div))
                    self.smart_bus.set("sampling_uncertainty", unc, module="AlternativeRealitySampler", thesis="ARS sampling uncertainty")
                    # Alias for downstream consumers
                    self.smart_bus.set("uncertainty", unc, module="AlternativeRealitySampler", thesis="ARS uncertainty alias")
                    self.smart_bus.set("effective_samples", eff, module="AlternativeRealitySampler", thesis="ARS effective samples")
                    self.smart_bus.set("fragility", fragility, module="AlternativeRealitySampler", thesis="ARS fragility estimate")
                except Exception:
                    pass

                # auxiliary diagnostics (atomic if bus supports transaction)
                with contextlib.suppress(Exception):
                    txn_cm = getattr(self.smart_bus, "transaction", None)
                    if callable(txn_cm):
                        cm = cast(ContextManager[Any], txn_cm())
                        with cm:
                            self._write_aux_bus()
                    else:
                        self._write_aux_bus()

                self.error_count = 0
                return results

            except Exception as e:
                return await self._handle_processing_error(e, time.time())

    def _write_aux_bus(self) -> None:
        now = dt.datetime.now().isoformat()
        self.smart_bus.set(
            "voting/ars/status",
            {"ok": True, "updated": now},
            module="AlternativeRealitySampler",
            thesis="ARS diagnostics heartbeat",
            confidence=0.9,
        )
        self.smart_bus.set(
            "voting/ars/metrics",
            {"sigma": float(self.current_sigma), "dim": int(self.dim)},
            module="AlternativeRealitySampler",
            thesis="ARS current parameters",
            confidence=0.9,
        )

    # ── key helpers ────────────────────────────────────────
    def _normalize_weights_input(self, weights: Any) -> np.ndarray:
        try:
            if isinstance(weights, dict):
                keys = sorted(weights.keys())
                arr = np.asarray([float(weights[k]) for k in keys], dtype=np.float32)
            else:
                arr = np.asarray(weights, dtype=np.float32).reshape(-1)

            if arr.size == 0 or not np.all(np.isfinite(arr)):
                arr = np.ones(max(1, self.dim), dtype=np.float32)

            arr = np.abs(arr)
            s = float(arr.sum())
            arr = arr / s if s > 0 else np.ones(arr.size, dtype=np.float32) / float(arr.size)
            return arr.astype(np.float32)
        except Exception:
            return np.ones(max(1, self.dim), dtype=np.float32) / float(max(1, self.dim))

    def _pad_or_trim(self, v: np.ndarray, d: int) -> np.ndarray:
        if v.size == d:
            return v
        if v.size < d:
            return np.pad(v, (0, d - v.size))
        return v[:d]

    def _adapt_dimension(self, new_dim: int) -> None:
        try:
            if new_dim > 0 and new_dim != self.dim:
                old = self.dim
                self.dim = int(new_dim)
                self.sampling_stats["dimensions"] = self.dim
                if self._last_dim_logged != self.dim:
                    self.logger.info(
                        format_operator_message(
                            icon="[ADAPT]",
                            message="Sampler dimension adjusted",
                            old_dimension=old,
                            new_dimension=self.dim,
                        )
                    )
                    self._last_dim_logged = self.dim
        except Exception:
            pass

    # ── adaptive sigma ─────────────────────────────────────
    async def _calculate_adaptive_sigma(self, ctx: Optional[Dict[str, Any]]) -> float:
        s = self.current_sigma
        try:
            lower, upper = self.sampling_intelligence["sigma_bounds"]
            if not self.adaptive_sigma:
                return float(np.clip(s, lower, upper))
            if ctx:
                unc = float(ctx.get("uncertainty", 0.0))
                vol = float(ctx.get("market_context", {}).get("volatility_value", ctx.get("volatility", 0.5)))
                agree = float(ctx.get("agreement_score", 0.5))
                if unc > self.uncertainty_threshold:
                    s *= (1.0 + (unc - self.uncertainty_threshold) * 0.5)
                s *= (0.8 + 0.4 * np.clip(vol, 0.0, 1.0))
                s *= (1.5 - np.clip(agree, 0.0, 1.0))
            return float(np.clip(s, lower, upper))
        except Exception:
            return float(np.clip(s, self.sampling_intelligence["sigma_bounds"][0], self.sampling_intelligence["sigma_bounds"][1]))

    async def _update_sampling_parameters_comprehensive(self, vd: Dict[str, Any]):
        if not self.adaptive_sigma:
            return
        try:
            regime = str(vd.get("market_regime", "unknown"))
            vol_level = str(vd.get("market_context", {}).get("volatility_level", "medium"))
            agree = float(vd.get("agreement_score", 0.5))
            m_unc = self._calculate_market_uncertainty_factor(vd)

            mul = self.market_adaptation["regime_multipliers"].get(regime, 1.0)
            mul *= self.market_adaptation["volatility_multipliers"].get(vol_level, 1.0)
            mul *= (1.5 if agree < 0.3 else 0.7 if agree > 0.8 else 1.0)
            mul *= (1.0 + (m_unc - 0.5) * self.sampling_intelligence["uncertainty_sensitivity"])

            target = float(np.clip(self.base_sigma * mul, *self.sampling_intelligence["sigma_bounds"]))

            mom = self.sampling_intelligence["exploration_momentum"]
            rate = self.sampling_intelligence["adaptation_rate"]
            old = self.current_sigma
            self.current_sigma = (old * mom * (1 - rate)) + (target * rate) + (old * (1 - mom) * 0.1)

            if abs(self.current_sigma - old) > 0.005:
                self.sampling_stats["sigma_adaptations"] += 1
                self.logger.info(
                    format_operator_message(
                        icon="[STATS]",
                        message="Sigma adapted",
                        old_sigma=f"{old:.4f}",
                        new_sigma=f"{self.current_sigma:.4f}",
                        regime=regime,
                        volatility=vol_level,
                        agreement=f"{agree:.2f}",
                        uncertainty=f"{m_unc:.3f}",
                    )
                )
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "sampling_parameters_update")
            self.logger.warning(f"Sigma update failed: {ctx}")

    def _calculate_market_uncertainty_factor(self, vd: Dict[str, Any]) -> float:
        try:
            comp: List[float] = []
            comp.append(1.0 - float(vd.get("agreement_score", 0.5)))
            vol_level = str(vd.get("market_context", {}).get("volatility_level", "medium"))
            comp.append({"very_low": 0.1, "low": 0.3, "medium": 0.5, "high": 0.8, "extreme": 1.0}.get(vol_level, 0.5))
            comp.append(0.9 if vd.get("market_regime", "unknown") == "unknown" else 0.2)
            rtr = vd.get("recent_trades", [])
            if len(rtr) >= 3:
                pnls = [float(t.get("pnl", 0.0)) for t in rtr[-5:]]
                if pnls:
                    v = np.std(pnls) / (abs(np.mean(pnls)) + 10.0)
                    comp.append(float(min(1.0, float(v))))
            w = np.array([0.3, 0.3, 0.2, 0.2][: len(comp)], dtype=np.float32)
            w = w / w.sum()
            return float(np.clip(float(np.dot(comp, w)), 0.0, 1.0))
        except Exception:
            return 0.5

    # ── sampling core ───────────────────────────────────────
    async def sample_comprehensive(self, weights: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        w = self._pad_or_trim(self._normalize_weights_input(weights), self.dim)
        sigma = await self._calculate_adaptive_sigma(context)

        # allocate counts by active strategy weights
        active = [(k, v) for k, v in self.sampling_strategies.items() if v["active"]]
        if not active:
            active = [("random_gaussian", {"weight": 1.0, "active": True})]
        total = sum(max(0.0, float(v["weight"])) for _, v in active) or 1.0
        counts: List[int] = []
        acc = 0
        for i, (_, cfg) in enumerate(active):
            if i == len(active) - 1:
                counts.append(self.n_samples - acc)
            else:
                c = max(1, int(round(self.n_samples * float(cfg["weight"]) / total)))
                counts.append(c)
                acc += c

        batches: List[np.ndarray] = []
        for (name, _), n in zip(active, counts):
            if name == "random_gaussian":
                batches.append(self._gaussian_samples(w, sigma, n))
            elif name == "structured_perturbation":
                batches.append(self._structured_perturbation_samples(w, sigma, n))
            elif name == "systematic_exploration":
                batches.append(self._systematic_exploration_samples(w, sigma, n))
            elif name == "uncertainty_guided":
                batches.append(self._uncertainty_guided_samples(w, sigma, n, context))
            else:
                batches.append(self._gaussian_samples(w, sigma, n))

        samples = np.vstack(batches)[: self.n_samples]
        samples = self._post_process_samples(samples, w)
        await self._record_sampling_results(samples, w, sigma, context)
        await self._update_sampling_statistics(samples, w)
        return samples

    # fast vectorized generators
    def _gaussian_samples(self, w: np.ndarray, sigma: float, n: int) -> np.ndarray:
        noise = self._rng.standard_normal(size=(n, self.dim)).astype(np.float32) * float(sigma)
        return w[None, :] + noise

    def _structured_perturbation_samples(self, w: np.ndarray, sigma: float, n: int) -> np.ndarray:
        k = min(n, self.dim)
        samples = np.tile(w, (n, 1))
        if k > 0:
            idxs = np.arange(k)
            pert = (2.0 * self._rng.random(k).astype(np.float32) - 1.0) * (2.0 * sigma)
            samples[np.arange(k), idxs] += pert
        if n > k:
            ridx = self._rng.integers(0, self.dim, size=n - k)
            rpert = (2.0 * self._rng.random(n - k).astype(np.float32) - 1.0) * (2.0 * sigma)
            samples[np.arange(k, n), ridx] += rpert
        return samples

    def _systematic_exploration_samples(self, w: np.ndarray, sigma: float, n: int) -> np.ndarray:
        samples = np.tile(w, (n, 1))
        step = float(sigma) * 1.5
        eye = np.eye(self.dim, dtype=np.float32)
        for i in range(min(n, self.dim * 2)):
            direction = eye[i % self.dim] * (1.0 if i < self.dim else -1.0)
            samples[i] = w + step * direction
        if n > self.dim * 2:
            m = n - (self.dim * 2)
            dirs = self._rng.standard_normal(size=(m, self.dim)).astype(np.float32)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
            samples[self.dim * 2 :] = w + step * (dirs / norms)
        return samples

    def _uncertainty_guided_samples(self, w: np.ndarray, sigma: float, n: int, ctx: Optional[Dict[str, Any]]) -> np.ndarray:
        scale = 1.5
        out = np.tile(w, (n, 1))
        if ctx and isinstance(ctx.get("dimension_uncertainties"), (list, np.ndarray)):
            du = np.asarray(ctx["dimension_uncertainties"], dtype=np.float32)
            du = self._pad_or_trim(np.abs(du), self.dim)
        else:
            du = np.ones(self.dim, dtype=np.float32)
        noise = self._rng.standard_normal(size=(n, self.dim)).astype(np.float32) * sigma * scale
        out += noise * du[None, :]
        return out

    def _post_process_samples(self, samples: np.ndarray, base_w: np.ndarray) -> np.ndarray:
        s = np.clip(samples, 0.0, None)
        s_sum = s.sum(axis=1, keepdims=True) + 1e-12
        s = s / s_sum
        with self._lock:
            self.last_samples = s.copy()
            self.last_weights = base_w.copy()
            self.last_base_weights = base_w.copy()
        return s

    async def _record_sampling_results(self, samples: np.ndarray, base_w: np.ndarray, sigma: float, ctx: Optional[Dict[str, Any]]):
        try:
            with self._lock:
                eff = self._count_effective(samples, base_w)
                div = self._sample_diversity(samples)
                self.sampling_history.append(
                    {
                        "timestamp": dt.datetime.now().isoformat(),
                        "base_weights": base_w.tolist(),
                        "samples": samples.tolist(),
                        "sigma_used": float(sigma),
                        "n_samples": int(samples.shape[0]),
                        "context": (dict(ctx) if ctx else {}),
                        "strategies_used": [k for k, v in self.sampling_strategies.items() if v["active"]],
                        "diversity_score": float(div),
                        "effective_samples": int(eff),
                    }
                )
                unc = await self._estimate_uncertainty(samples, base_w)
                self.uncertainty_history.append(float(unc))
        except Exception:
            pass

    async def _estimate_uncertainty(self, samples: np.ndarray, base_w: np.ndarray) -> float:
        if samples.shape[0] < 2:
            return 0.5
        norms = np.linalg.norm(samples, axis=1)
        spread = float(np.std(norms) / (np.mean(norms) + 1e-12))
        dist = float(np.mean(np.linalg.norm(samples - base_w[None, :], axis=1)) / (self.current_sigma * math.sqrt(self.dim) + 1e-12))
        return float(np.clip(0.6 * spread + 0.4 * dist, 0.0, 1.0))

    async def _update_sampling_statistics(self, samples: np.ndarray, base_w: np.ndarray):
        with self._lock:
            self.sampling_stats["samples_generated"] += 1
            self.sampling_stats["total_samples_created"] += int(samples.shape[0])
            eff = self._count_effective(samples, base_w)
            self.sampling_stats["effective_samples"] += eff
            div = self._sample_diversity(samples)
            self.sampling_stats["diversity_score"] = float(div)
            if self.uncertainty_history:
                self.sampling_stats["avg_uncertainty"] = float(np.mean(list(self.uncertainty_history)[-10:]))
            t = self.sampling_stats["total_samples_created"]
            self.sampling_stats["exploration_efficiency"] = float(self.sampling_stats["effective_samples"] / max(t, 1))

    # vectorized metrics
    def _count_effective(self, samples: np.ndarray, base_w: np.ndarray) -> int:
        thr = float(self.current_sigma * 0.5)
        d = np.linalg.norm(samples - base_w[None, :], axis=1)
        return int(np.count_nonzero(d > thr))

    def _pairwise_mean_distance(self, samples: np.ndarray) -> float:
        x = samples.astype(np.float64)
        n = x.shape[0]
        if n < 2:
            return 0.0
        norms = (x * x).sum(axis=1)
        dist2 = norms[:, None] + norms[None, :] - 2.0 * (x @ x.T)
        m = n * (n - 1) // 2
        tri = dist2[np.triu_indices(n, k=1)]
        return float(np.mean(np.sqrt(np.maximum(tri, 0.0)))) if m else 0.0

    def _sample_diversity(self, samples: np.ndarray) -> float:
        mpd = self._pairwise_mean_distance(samples)
        expected = math.sqrt(self.dim) * max(self.current_sigma, 1e-6)
        ratio = (float(mpd) / float(expected)) if expected > 0 else 0.0
        return float(min(1.0, float(ratio)))

    # ── reporting / bounds ──────────────────────────────────
    def _calculate_confidence_bounds(self) -> Dict[str, Any]:
        with self._lock:
            s = getattr(self, "last_samples", None)
        if s is None or s.shape[0] < 3:
            return {"lower_bound": [], "upper_bound": [], "confidence_level": 0.0}
        lo = np.percentile(s, 5, axis=0)
        hi = np.percentile(s, 95, axis=0)
        width = float(np.mean(hi - lo))
        level = float(max(0.0, min(1.0, 1.0 - width * 2)))
        return {
            "lower_bound": lo.tolist(),
            "upper_bound": hi.tolist(),
            "confidence_level": level,
            "bound_width": width,
            "percentiles": {"lower": 5, "upper": 95},
        }

    def _get_comprehensive_sampling_stats(self) -> Dict[str, Any]:
        return {
            **self.sampling_stats,
            "current_sigma": float(self.current_sigma),
            "base_sigma": float(self.base_sigma),
            "adaptive_enabled": bool(self.adaptive_sigma),
            "dimensions": int(self.dim),
            "samples_per_iteration": int(self.n_samples),
            "strategy_weights": {n: float(c["weight"]) for n, c in self.sampling_strategies.items()},
            "quality_metrics": dict(self.quality_metrics),
            "recent_uncertainty_trend": (
                self._uncertainty_trend(list(self.uncertainty_history)[-10:])
                if len(self.uncertainty_history) >= 3 else "insufficient_data"
            ),
        }

    # analytics & recommendations
    async def _analyze_sampling_effectiveness_comprehensive(self, vd: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "overall_effectiveness": 0.5,
            "diversity_analysis": {},
            "convergence_analysis": {},
            "exploration_efficiency": 0.0,
            "uncertainty_accuracy": 0.0,
            "strategy_performance": {},
        }
        if len(self.sampling_history) < 5:
            out["insufficient_data"] = True
            return out

        div = await self._analyze_sampling_diversity()
        out["diversity_analysis"] = div
        conv = await self._analyze_convergence_patterns()
        out["convergence_analysis"] = conv
        out["exploration_efficiency"] = await self._calculate_exploration_efficiency()
        out["uncertainty_accuracy"] = await self._analyze_uncertainty_accuracy(vd)
        out["strategy_performance"] = await self._evaluate_strategy_performance()

        sp = out["strategy_performance"]
        sp_mean = float(np.mean(list(sp.values()))) if sp else 0.5
        out["overall_effectiveness"] = float(
            np.clip(
                0.25 * float(div.get("diversity_score", 0.5))
                + 0.25 * float(conv.get("convergence_score", 0.5))
                + 0.20 * out["exploration_efficiency"]
                + 0.15 * out["uncertainty_accuracy"]
                + 0.15 * sp_mean,
                0.0,
                1.0,
            )
        )
        with self._lock:
            self.effectiveness_history.append(
                {
                    "timestamp": dt.datetime.now().isoformat(),
                    "overall_effectiveness": out["overall_effectiveness"],
                    "diversity_score": float(div.get("diversity_score", 0.5)),
                    "exploration_efficiency": float(out["exploration_efficiency"]),
                    "uncertainty_accuracy": float(out["uncertainty_accuracy"]),
                }
            )
        return out

    async def _analyze_sampling_diversity(self) -> Dict[str, Any]:
        if len(self.sampling_history) < 3:
            return {"diversity_score": 0.5, "insufficient_data": True}
        recent = list(self.sampling_history)[-10:]
        diversities: List[float] = []
        spreads: List[float] = []
        coverages: List[float] = []
        for s in recent:
            arr = np.asarray(s.get("samples", []), dtype=np.float32)
            if arr.shape[0] > 1:
                diversities.append(self._pairwise_mean_distance(arr))
                ctr = np.mean(arr, axis=0)
                spreads.append(float(np.max(np.linalg.norm(arr - ctr[None, :], axis=1))))
                # coverage: uniformity of nearest-neighbor distances (pure NumPy; no SciPy)
                if arr.shape[0] > 2:
                    x = arr.astype(np.float64)
                    norms = (x * x).sum(axis=1)
                    dist2 = norms[:, None] + norms[None, :] - 2.0 * (x @ x.T)
                    np.fill_diagonal(dist2, np.inf)
                    nn = np.sqrt(np.maximum(np.min(dist2, axis=1), 0.0))
                    cov = float(1.0 - (np.std(nn) / (np.mean(nn) + 1e-8)))
                    coverages.append(cov)

        score = float(np.mean(diversities)) if diversities else 0.5
        target = self.sampling_intelligence["diversity_target"]
        normalized = float(min(1.0, score / max(target, 1e-6)))
        return {
            "diversity_score": score,
            "normalized_diversity": normalized,
            "spread_score": float(np.mean(spreads)) if spreads else 0.0,
            "coverage_score": float(np.mean(coverages)) if coverages else 0.0,
            "diversity_trend": self._calculate_diversity_trend(diversities),
            "sample_count": len(recent),
        }

    def _calculate_diversity_trend(self, xs: List[float]) -> str:
        if len(xs) < 3:
            return "insufficient_data"
        x = np.arange(len(xs))
        try:
            slope = float(np.polyfit(x, xs, 1)[0])
        except Exception:
            return "unknown"
        return "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"

    async def _analyze_convergence_patterns(self) -> Dict[str, Any]:
        if len(self.uncertainty_history) < 10:
            return {"convergence_score": 0.5, "insufficient_data": True}
        xs = list(self.uncertainty_history)[-20:]
        rate = self._convergence_rate(xs)
        stab = self._stability(xs)
        pattern = self._convergence_pattern(xs)
        return {
            "convergence_rate": rate,
            "stability_score": stab,
            "convergence_pattern": pattern,
            "convergence_score": float((rate + stab) / 2.0),
            "recent_uncertainty_trend": self._uncertainty_trend(xs),
        }

    def _convergence_rate(self, xs: List[float]) -> float:
        if len(xs) < 5:
            return 0.5
        recent, earlier = xs[-5:], xs[:-5] if len(xs) >= 10 else xs[: -5]
        if not earlier:
            return 0.5
        rstd, estd = float(np.std(recent)), float(np.std(earlier))
        if estd == 0:
            return 1.0 if rstd < 0.01 else 0.5
        imp = (estd - rstd) / estd
        return float(np.clip(0.5 + imp, 0.0, 1.0))

    def _stability(self, xs: List[float]) -> float:
        if len(xs) < 3:
            return 0.5
        var = float(np.var(xs))
        return float(min(1.0, max(0.0, 1.0 - var * 10.0)))

    def _slope(self, xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        x = np.arange(len(xs))
        try:
            return float(np.polyfit(x, xs, 1)[0])
        except Exception:
            return 0.0

    def _convergence_pattern(self, xs: List[float]) -> str:
        r = self._slope(xs[-5:])
        o = self._slope(xs)
        if abs(r) < 0.01 and abs(o) < 0.01:
            return "converged"
        if r < -0.05:
            return "converging"
        if r > 0.05:
            return "diverging"
        return "oscillating" if abs(r) < 0.03 else "trending"

    def _uncertainty_trend(self, xs: List[float]) -> str:
        s = self._slope(xs)
        return "increasing" if s > 0.02 else "decreasing" if s < -0.02 else "stable"

    # Backwards-compat alias (some call sites referenced the old name)
    def _calculate_uncertainty_trend(self, xs: List[float]) -> str:
        return self._uncertainty_trend(xs)

    async def _calculate_exploration_efficiency(self) -> float:
        if len(self.sampling_history) < 3:
            return 0.5
        eff = float(self.sampling_stats.get("effective_samples", 0)) / max(
            1, int(self.sampling_stats.get("total_samples_created", 1))
        )
        if len(self.effectiveness_history) >= 3:
            trend = self._slope([e.get("exploration_efficiency", 0.5) for e in list(self.effectiveness_history)[-3:]])
            eff += float(trend) * 0.1
        return float(np.clip(eff, 0.0, 1.0))

    async def _analyze_uncertainty_accuracy(self, vd: Dict[str, Any]) -> float:
        if len(self.uncertainty_history) < 5:
            return 0.5
        est = float(np.mean(list(self.uncertainty_history)[-10:]))
        exp = float(self._calculate_market_uncertainty_factor(vd))
        return float(max(0.0, min(1.0, 1.0 - abs(est - exp))))

    async def _evaluate_strategy_performance(self) -> Dict[str, float]:
        sp: Dict[str, float] = {}
        recent = [e.get("overall_effectiveness", 0.5) for e in list(self.effectiveness_history)[-3:]]
        avg = float(np.mean(recent)) if recent else 0.5
        for name, cfg in self.sampling_strategies.items():
            perf = float(cfg["weight"]) * (avg if cfg.get("active") else 0.0)
            sp[name] = float(np.clip(perf, 0.0, 1.0))
        return sp

    async def _update_sampling_strategy_weights(self, eff: Dict[str, Any]) -> Dict[str, Any]:
        updates = {"weight_changes": {}, "activation_changes": {}, "overall_improvement": 0.0}
        sp = dict(eff.get("strategy_performance", {}))
        tot = 0.0
        for name, cfg in self.sampling_strategies.items():
            w0 = float(cfg["weight"])
            p = float(sp.get(name, 0.5))
            if p > 0.7:
                w1 = min(0.5, w0 * 1.1)
            elif p < 0.3:
                w1 = max(0.05, w0 * 0.9)
            else:
                w1 = w0
            if abs(w1 - w0) > 0.01:
                updates["weight_changes"][name] = {"old_weight": w0, "new_weight": w1, "performance": p}
            cfg["weight"] = w1
            tot += w1
        if tot > 0:
            for cfg in self.sampling_strategies.values():
                cfg["weight"] = float(cfg["weight"] / tot)
        return updates

    async def _calculate_comprehensive_quality_metrics(self) -> Dict[str, Any]:
        q = dict(self.quality_metrics)
        if len(self.sampling_history) > 0:
            arr = np.asarray(self.sampling_history[-1].get("samples", []), dtype=np.float32)
            if arr.size:
                q["sample_diversity"] = float(self._sample_diversity(arr))
        if len(self.effectiveness_history) > 0:
            q["coverage_efficiency"] = float(self.effectiveness_history[-1].get("exploration_efficiency", 0.0))
        q["uncertainty_accuracy"] = float(self.sampling_stats.get("avg_uncertainty", 0.5))
        adapt = int(self.sampling_stats.get("sigma_adaptations", 0))
        q["adaptation_success_rate"] = float(
            min(1.0, self.sampling_stats.get("effective_samples", 0) / max(adapt or 1, 1))
        )
        q["exploration_completeness"] = float(min(1.0, len(self.sampling_history) / 50.0))
        vals = [
            q["sample_diversity"],
            q["coverage_efficiency"],
            q["uncertainty_accuracy"],
            q["adaptation_success_rate"],
            q["exploration_completeness"],
        ]
        w = np.array([0.25, 0.20, 0.20, 0.20, 0.15], dtype=np.float32)
        q["overall_quality_score"] = float(np.average(vals, weights=w))
        self.quality_metrics.update(q)
        return q

    async def _generate_intelligent_sampling_recommendations(self, eff: Dict[str, Any], q: Dict[str, Any]) -> List[str]:
        rec: List[str] = []
        oe = float(eff.get("overall_effectiveness", 0.5))
        if oe < 0.3:
            rec.append("Low effectiveness – increase exploration (sigma ↑)")
        elif oe > 0.8:
            rec.append("High effectiveness – keep parameters steady")

        div = eff.get("diversity_analysis", {})
        if float(div.get("diversity_score", 0.5)) < self.sampling_intelligence["diversity_target"] * 0.7:
            rec.append("Insufficient diversity – enable more exploration strategies")

        conv = eff.get("convergence_analysis", {})
        pat = conv.get("convergence_pattern", "unknown")
        if pat == "diverging":
            rec.append("Diverging – reduce sigma to stabilize")
        elif pat == "oscillating":
            rec.append("Oscillating – stabilize parameters (lower momentum)")

        if float(q.get("overall_quality_score", 0.5)) < 0.4:
            rec.append("Low quality – review strategy weights & parameters")

        for s, p in eff.get("strategy_performance", {}).items():
            if float(p) < 0.3:
                rec.append(f"Consider disabling '{s}' due to low performance")

        sa = int(self.sampling_stats.get("sigma_adaptations", 0))
        if sa > 20:
            rec.append("High adaptation frequency – reduce sensitivity")
        elif sa == 0 and self.adaptive_sigma:
            rec.append("No adaptations detected – verify adaptive path")

        return rec[:5] or ["Sampling operating within expected parameters"]

    async def _generate_comprehensive_sampling_thesis(self, eff: Dict[str, Any], q: Dict[str, Any], updates: Dict[str, Any]) -> str:
        div = eff.get("diversity_analysis", {})
        parts = [
            f"SAMPLING: eff {eff.get('overall_effectiveness',0.5):.1%}, div {float(div.get('diversity_score',0.5)):.3f}",
            f"QUALITY: {q.get('overall_quality_score',0.5):.1%} via {len(getattr(self,'sampling_strategies',{}))} methods",
            f"ADAPT: sigma_adaptations={self.sampling_stats.get('sigma_adaptations',0)}",
            f"CONFIG: n={self.n_samples}, σ={self.current_sigma:.4f}",
            f"EFFICIENCY: {self.sampling_stats.get('exploration_efficiency',0.0):.1%}",
        ]
        if updates.get("weight_changes"):
            parts.append(f"STRAT: {len(updates['weight_changes'])} weight tweaks")
        return " | ".join(parts)

    # ── health & errors ─────────────────────────────────────
    def _get_health_metrics(self) -> Dict[str, Any]:
        return {
            "module_name": "AlternativeRealitySampler",
            "status": "disabled" if self.is_disabled else "healthy",
            "error_count": int(self.error_count),
            "circuit_breaker_threshold": int(self.circuit_breaker_threshold),
            "samples_generated": int(self.sampling_stats.get("samples_generated", 0)),
            "effective_sample_ratio": float(self.sampling_stats.get("exploration_efficiency", 0.0)),
            "diversity_score": float(self.sampling_stats.get("diversity_score", 0.0)),
            "uncertainty_level": float(self.sampling_stats.get("avg_uncertainty", 0.5)),
            "sigma_adaptations": int(self.sampling_stats.get("sigma_adaptations", 0)),
            "strategy_count": int(len([s for s in self.sampling_strategies.values() if s["active"]])),
            "session_duration": (dt.datetime.now() - dt.datetime.fromisoformat(self.sampling_stats["session_start"])).total_seconds() / 3600.0,
        }

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        self.error_count += 1
        ctx = self.error_pinpointer.analyze_error(error, "AlternativeRealitySampler")
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(
                format_operator_message(
                    icon="[ALERT]",
                    message="ARS disabled due to repeated errors",
                    error_count=self.error_count,
                    threshold=self.circuit_breaker_threshold,
                )
            )
        return {
            "alternative_samples": [],
            "sampling_uncertainty": 0.5,
            "diversity_score": 0.0,
            "sampling_stats": {"error": str(ctx)},
            "effective_samples": 0,
            "confidence_bounds": {"error": str(ctx)},
            "sampling_recommendations": ["Investigate AlternativeRealitySampler errors"],
            "health_metrics": {"status": "error", "error_context": str(ctx)},
            "alternative_reality_sampler_initialization": self._get_ars_init_view(),
            "_thesis": f"AlternativeRealitySampler error: {ctx}",
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        return {
            "alternative_samples": [],
            "sampling_uncertainty": 0.5,
            "diversity_score": 0.0,
            "sampling_stats": {"status": "disabled"},
            "effective_samples": 0,
            "confidence_bounds": {"status": "disabled"},
            "sampling_recommendations": ["Restart AlternativeRealitySampler system"],
            "health_metrics": {"status": "disabled", "reason": "circuit_breaker_triggered"},
            "alternative_reality_sampler_initialization": self._get_ars_init_view(),
            "_thesis": "AlternativeRealitySampler disabled via circuit breaker",
        }

    # ── state I/O ───────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        with self._lock:
            lw = getattr(self, "last_weights", None)
            ls = getattr(self, "last_samples", None)
            base["custom_state"] = {
                "dim": self.dim,
                "n_samples": self.n_samples,
                "base_sigma": self.base_sigma,
                "current_sigma": self.current_sigma,
                "adaptive_sigma": self.adaptive_sigma,
                "uncertainty_threshold": self.uncertainty_threshold,
                "last_weights": lw.tolist() if lw is not None else None,
                "last_samples": ls.tolist() if ls is not None else None,
                "sampling_stats": dict(self.sampling_stats),
                "quality_metrics": dict(self.quality_metrics),
                "sampling_strategies": {k: dict(v) for k, v in self.sampling_strategies.items()},
                "market_adaptation": dict(self.market_adaptation),
            }
        return base

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        cs = state.get("custom_state", {})
        with self._lock:
            self.dim = int(cs.get("dim", self.dim))
            self.n_samples = int(cs.get("n_samples", self.n_samples))
            self.base_sigma = float(cs.get("base_sigma", self.base_sigma))
            self.current_sigma = float(cs.get("current_sigma", self.current_sigma))
            self.adaptive_sigma = bool(cs.get("adaptive_sigma", self.adaptive_sigma))
            self.uncertainty_threshold = float(cs.get("uncertainty_threshold", self.uncertainty_threshold))
            if cs.get("last_weights") is not None:
                self.last_weights = np.asarray(cs["last_weights"], dtype=np.float32)
            if cs.get("last_samples") is not None:
                self.last_samples = np.asarray(cs["last_samples"], dtype=np.float32)
            self.sampling_stats.update(cs.get("sampling_stats", {}))
            self.quality_metrics.update(cs.get("quality_metrics", {}))
            if cs.get("sampling_strategies"):
                for k, v in cs["sampling_strategies"].items():
                    if k in self.sampling_strategies:
                        self.sampling_strategies[k].update(v)
            if cs.get("market_adaptation"):
                self.market_adaptation.update(cs["market_adaptation"])

    # ── data access (bus-safe) ──────────────────────────────
    async def _get_comprehensive_voting_data(self) -> Dict[str, Any]:
        try:
            g = self.smart_bus.get
            return {
                "votes": g("votes", "AlternativeRealitySampler") or [],
                "voting_summary": g("voting_summary", "AlternativeRealitySampler") or {},
                "strategy_arbiter_weights": g("strategy_arbiter_weights", "AlternativeRealitySampler") or [],
                "market_context": g("market_context", "AlternativeRealitySampler") or {},
                "volatility_data": g("volatility_data", "AlternativeRealitySampler") or {},
                "consensus_direction": g("consensus_direction", "AlternativeRealitySampler") or "neutral",
                "agreement_score": g("agreement_score", "AlternativeRealitySampler") or 0.5,
                "market_regime": g("market_regime", "AlternativeRealitySampler") or "unknown",
                "recent_trades": g("recent_trades", "AlternativeRealitySampler") or [],
                "session_metrics": g("session_metrics", "AlternativeRealitySampler") or {},
                "decision_id": g("decision_id", "AlternativeRealitySampler"),
                "tick_ts": g("tick_ts", "AlternativeRealitySampler"),
            }
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "AlternativeRealitySampler")
            self.logger.warning(f"Voting data retrieval failed: {ctx}")
            return self._get_safe_voting_defaults()

    def _get_safe_voting_defaults(self) -> Dict[str, Any]:
        """Safe defaults when bus reads fail (compatibility)."""
        return {
            "votes": [],
            "voting_summary": {},
            "strategy_arbiter_weights": [],
            "market_context": {},
            "volatility_data": {},
            "consensus_direction": "neutral",
            "agreement_score": 0.5,
            "market_regime": "unknown",
            "recent_trades": [],
            "session_metrics": {},
        }

    # ── legacy compat ───────────────────────────────────────
    def sample(self, weights: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Legacy sync wrapper."""
        try:
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                pass
            if loop and loop.is_running():
                coro = self.sample_comprehensive(weights, context)
                return asyncio.run_coroutine_threadsafe(coro, loop).result()
            return asyncio.run(self.sample_comprehensive(weights, context))
        except Exception:
            # simple fallback
            weights = np.asarray(weights, dtype=np.float32).flatten()
            if weights.size != self.dim:
                weights = np.pad(weights, (0, max(0, self.dim - weights.size)))[: self.dim]
            noise = np.random.randn(self.n_samples, self.dim).astype(np.float32) * self.current_sigma
            samples = weights[None, :] + noise
            samples = np.abs(samples)
            row_sums = samples.sum(axis=1, keepdims=True)
            samples = samples / (row_sums + 1e-12)
            with self._lock:
                self.last_samples = samples
            return samples

    def get_uncertainty_estimate(self, weights: np.ndarray) -> float:
        try:
            if getattr(self, "last_samples", None) is None:
                _ = self.sample(weights)
            samples = self.last_samples
            norms = [np.linalg.norm(s) for s in samples]
            uncertainty = np.std(norms) / (np.mean(norms) + 1e-12)
            return float(np.clip(uncertainty, 0.0, 1.0))
        except Exception:
            return 0.5

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        try:
            div = float(self.sampling_stats.get("diversity_score", 0.5))
            eff = float(self.sampling_stats.get("exploration_efficiency", 0.5))
            unc = float(self.sampling_stats.get("avg_uncertainty", 0.5))
            base = (div + eff) / 2.0
            adj = 1.0 - min(unc, 0.8)
            return float(np.clip(base * adj, 0.1, 0.95))
        except Exception:
            return 0.4

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        try:
            vd = await self._get_comprehensive_voting_data()
            w = self._normalize_weights_input(vd.get("strategy_arbiter_weights", []))
            if self.auto_dim and w.size and w.size != self.dim:
                self._adapt_dimension(int(w.size))
                w = self._pad_or_trim(w, self.dim)

            samples = await self.sample_comprehensive(w, vd)
            unc = float(np.std(samples, axis=0).mean()) if samples.shape[0] > 1 else 0.3
            div = float(self._sample_diversity(samples)) if samples.shape[0] > 1 else 0.5

            if unc > self.uncertainty_threshold:
                act, strength, why = "conservative", 0.3, f"High uncertainty ({unc:.3f})"
            elif div < 0.3:
                act, strength, why = "diversify", 0.6, f"Low diversity ({div:.3f})"
            else:
                act, strength, why = "normal", 0.7, f"Balanced metrics (unc={unc:.3f}, div={div:.3f})"

            return {
                "action": act,
                "signal_strength": strength,
                "reasoning": why,
                "sampling_metrics": {
                    "uncertainty": unc,
                    "diversity": div,
                    "n_samples": int(samples.shape[0]),
                    "effective_samples": int(self.sampling_stats.get("effective_samples", int(samples.shape[0]))),
                },
                "confidence": await self.calculate_confidence({}, **inputs),
            }
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {"action": "abstain", "signal_strength": 0.0, "reasoning": f"Sampling error: {e}", "confidence": 0.1}

    # ── initialization thesis ───────────────────────────────
    def _generate_initialization_thesis(self):
        thesis = (
            f"ARS v3.1 init: dim={self.dim}, n={self.n_samples}, σ0={self.base_sigma:.3f}, "
            f"adaptive={self.adaptive_sigma}, methods={4}, bounds={self.config_t.sigma_bounds}"
        )
        with contextlib.suppress(Exception):
            self.smart_bus.set(
                "alternative_reality_sampler_initialization",
                {
                    "status": "initialized",
                    "thesis": thesis,
                    "timestamp": dt.datetime.now().isoformat(),
                    "configuration": {
                        "dimensions": self.dim,
                        "samples_per_iteration": self.n_samples,
                        "intelligence_parameters": self.sampling_intelligence,
                    },
                },
                module="AlternativeRealitySampler",
                thesis=thesis,
            )

    def _get_ars_init_view(self) -> Dict[str, Any]:
        try:
            payload = self.smart_bus.get("alternative_reality_sampler_initialization", "AlternativeRealitySampler") or {}
            if isinstance(payload, dict) and payload.get("status"):
                return payload
        except Exception:
            pass
        return {
            "status": "initialized",
            "thesis": "ARS initialization heartbeat",
            "timestamp": dt.datetime.now().isoformat(),
            "configuration": {
                "dimensions": int(getattr(self, "dim", 0)),
                "samples_per_iteration": int(getattr(self, "n_samples", 0)),
                "intelligence_parameters": dict(getattr(self, "sampling_intelligence", {})),
            },
        }
