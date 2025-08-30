# ─────────────────────────────────────────────────────────────
# File: modules/features/advanced_feature_engine.py
# Advanced Feature Engine (Contract-Clean, Single-Writer)
# Contract alignment: provides/requires per contracts.py (AdvancedFeatureEngine). :contentReference[oaicite:0]{index=0}
# ─────────────────────────────────────────────────────────────

import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import deque
from dataclasses import dataclass, asdict

# Core infrastructure
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.contracts import module_args


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass
class FeatureEngineConfig:
    window_sizes: Optional[List[int]] = None
    max_buffer_size: int = 1000
    enable_neural_processing: bool = False  # placeholder hook
    enable_health_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_error_pinpointing: bool = True
    enable_english_explanations: bool = True
    circuit_breaker_threshold: int = 5

    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [7, 14, 28, 56]


# ─────────────────────────────────────────────────────────────
# Contract declaration
# ─────────────────────────────────────────────────────────────
@module(**module_args(
    "AdvancedFeatureEngine",
    description="Deterministic multi-window feature extraction with circuit breaker, monitoring, and explainability.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=120,
))
class AdvancedFeatureEngine(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    - Provides (per registry): advanced_features, features, feature_analysis, feature_thesis,
      feature_engine_capabilities, feature_health, feature_error, market_features, price_features,
      advanced_features_H1, advanced_features_H4, advanced_features_D1
    - Requires (per registry): price_data (preferred), historical_prices, ohlcv_data, market_data,
      multi_timeframe_data (soft-checked and used if present)
    - Fallbacks via Bus: historical_prices -> ohlcv_data -> market_data
    - Fleet-wide health aggregator still elsewhere; this module publishes its own compact health. :contentReference[oaicite:1]{index=1}
    """

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────
    def __init__(self, config: Optional[Union[FeatureEngineConfig, Dict[str, Any]]] = None, **kwargs):
        # Normalize config → dataclass
        if isinstance(config, dict):
            filtered: Dict[str, Any] = {k: config[k] for k in FeatureEngineConfig.__dataclass_fields__ if k in config}
            self._cfg = FeatureEngineConfig(**filtered)
        elif isinstance(config, FeatureEngineConfig) or config is None:
            self._cfg = config or FeatureEngineConfig()
        else:
            # Defensive fallback
            self._cfg = FeatureEngineConfig()

        # Pass DI-friendly dict config to BaseModule (required by your infra)
        super().__init__(config=asdict(self._cfg), **kwargs)

        # BaseModule __init__ calls our _initialize()

    def _initialize(self):
        """
        Implement abstract hook. Do NOT call super()._initialize() (abstract).
        Prepare internals, state, and background monitoring.
        """
        # Core handles
        self.smart_bus = InfoBusManager.get_instance()

        # Feature geometry
        self.window_sizes = sorted(self._cfg.window_sizes or [7, 14, 28, 56])
        # 6 stats per window + 6 global stats = len(ws)*6 + 6
        self.out_dim = len(self.window_sizes) * 6 + 6
        self.max_buffer_size = int(self._cfg.max_buffer_size)

        # Subsystems (use BaseModule's logger already set)
        if self._cfg.enable_error_pinpointing:
            self.error_pinpointer = ErrorPinpointer()
            self.error_handler = create_error_handler("AdvancedFeatureEngine", self.error_pinpointer)
        else:
            self.error_pinpointer = None
            self.error_handler = None

        if self._cfg.enable_english_explanations:
            self.english_explainer = EnglishExplainer()
            self.system_utilities = SystemUtilities()
        else:
            self.english_explainer = None
            self.system_utilities = None

        if self._cfg.enable_performance_tracking:
            self.performance_tracker = PerformanceTracker()
        else:
            self.performance_tracker = None

        # Circuit breaker (local accounting; real breaker can be DI'd)
        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
            "threshold": int(self._cfg.circuit_breaker_threshold),
        }

        # State & monitoring
        self._initialize_feature_state()
        self._start_monitoring()

        # Multi-timeframe presence flag (for aliasing outputs like *_H1/H4/D1)
        self._mtf_present: bool = False

        # Log
        self.logger.info(
            format_operator_message(
                "[INIT]", "ADVANCED_FEATURE_ENGINE_READY",
                details=f"Windows={self.window_sizes}, OutDim={self.out_dim}, Buffer={self.max_buffer_size}",
                result="ready",
                context="feature_engine_startup"
            )
        )

    # ─────────────────────────────────────────────────────────
    # Internal state & monitoring
    # ─────────────────────────────────────────────────────────
    def _initialize_feature_state(self):
        self.price_buffer: deque = deque(maxlen=self.max_buffer_size)
        self.feature_buffer: deque = deque(maxlen=1000)
        self.last_features = np.zeros(self.out_dim, dtype=np.float32)
        self.feature_quality_score = 100.0

        self.feature_stats: Dict[str, Any] = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_extraction_time_ms": 0.0,
            "avg_feature_quality": 0.0,
            "price_points_processed": 0
        }

        self.health_metrics: Dict[str, Any] = {
            "last_health_check": time.time(),
            "health_score": 100.0,
            "issues_detected": [],
            "performance_trend": "stable"
        }

    def _start_monitoring(self):
        try:
            loop = asyncio.get_running_loop()
            if self._cfg.enable_health_monitoring:
                loop.create_task(self._health_monitoring_loop())
            if self._cfg.enable_performance_tracking:
                loop.create_task(self._performance_monitoring_loop())
        except RuntimeError:
            # No running loop; orchestrator will attach later.
            pass

    # ─────────────────────────────────────────────────────────
    # Main processing
    # ─────────────────────────────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        start = time.time()

        # Circuit breaker
        if not self._check_circuit_breaker():
            base_adv = self._build_adv_from_payload({
                "raw_features": self._get_fallback_features(),
                "quality_score": 0.0,
                "extraction_time_ms": 0.0,
                "buffer_size": len(self.price_buffer),
                "feature_count": 0,
            })
            tf_outputs = self._compute_timeframe_outputs(base_adv)
            return self._create_fallback_response_with_tf("Circuit breaker open", tf_outputs=tf_outputs)

        try:
            market_data = await self._extract_market_data(**inputs)
            features_payload = await self._process_features_with_monitoring(market_data)
            thesis = await self._generate_feature_thesis(features_payload, market_data)

            # Prepare alias/base payload for TF mirrors
            base_adv = self._build_adv_from_payload(features_payload)
            tf_outputs = self._compute_timeframe_outputs(base_adv)

            # Publish only declared keys (single-writer discipline)
            self._update_bus(features_payload, thesis, tf_outputs)

            self._record_success(time.time() - start)

            return self._format_declared_outputs(
                features_payload=features_payload,
                thesis=thesis,
                analysis={
                    "explanation": features_payload.get("explanation"),
                    "statistics": self.feature_stats,
                    "buffer_status": {
                        "current_size": len(self.price_buffer),
                        "max_size": self.max_buffer_size,
                        "utilization": len(self.price_buffer) / max(self.max_buffer_size, 1),
                    },
                },
                extra={
                    "success": True,
                    "processing_time_ms": (time.time() - start) * 1000.0,
                },
                timeframe_outputs=tf_outputs
            )

        except Exception as e:
            # Error path: return robust fallback (do not publish to bus)
            base_adv = self._build_adv_from_payload({
                "raw_features": self._get_fallback_features(),
                "quality_score": 0.0,
                "extraction_time_ms": (time.time() - start) * 1000.0,
                "buffer_size": len(self.price_buffer),
                "feature_count": int(self.out_dim),
            })
            tf_outputs = self._compute_timeframe_outputs(base_adv)
            return await self._handle_processing_error(e, start, tf_outputs=tf_outputs)

    # ─────────────────────────────────────────────────────────
    # Inputs
    # ─────────────────────────────────────────────────────────
    async def _extract_market_data(self, **inputs) -> Dict[str, Any]:
        """
        Consumes hard-required `price_data` from inputs (preferred),
        with resilient fallbacks through the Bus. Produces a flat list of prices.
        """
        market_data = {"prices": []}

        # Presence check for multi_timeframe_data (contract requires this key to exist upstream)
        try:
            mtd = self.smart_bus.get("multi_timeframe_data", self.__class__.__name__)
            self._mtf_present = isinstance(mtd, dict) and bool(mtd)
        except Exception:
            self._mtf_present = False

        # 1) Hard require: price_data from inputs
        pd_map = inputs.get("price_data")
        if isinstance(pd_map, dict):
            for _sym, entry in pd_map.items():
                close = (entry or {}).get("close")
                if isinstance(close, (int, float)):
                    market_data["prices"].append(float(close))

        # 2) Fallbacks from Bus (in order of richness)
        if not market_data["prices"]:
            hist = self.smart_bus.get("historical_prices", self.__class__.__name__)
            if isinstance(hist, dict):
                try:
                    for _instrument, tfs in hist.items():
                        if isinstance(tfs, dict):
                            for _tf, payload in tfs.items():
                                if isinstance(payload, dict):
                                    closes = payload.get("close")
                                    if isinstance(closes, (list, np.ndarray)) and len(closes) > 0:
                                        market_data["prices"].extend(np.asarray(closes).flatten().tolist())
                                    else:
                                        cb = payload.get("current_bar", {})
                                        if isinstance(cb.get("close"), (int, float)):
                                            market_data["prices"].append(float(cb["close"]))
                except Exception:
                    pass

        if not market_data["prices"]:
            ohlcv = self.smart_bus.get("ohlcv_data", self.__class__.__name__)
            if isinstance(ohlcv, dict):
                for _symbol, bar in ohlcv.items():
                    if isinstance(bar, dict) and isinstance(bar.get("close"), (int, float)):
                        market_data["prices"].append(float(bar["close"]))

        if not market_data["prices"]:
            mkt = self.smart_bus.get("market_data", self.__class__.__name__)
            if isinstance(mkt, dict):
                for key, val in mkt.items():
                    if "price" in str(key).lower() and isinstance(val, (list, np.ndarray)):
                        market_data["prices"].extend(np.asarray(val).flatten().tolist())
                    elif isinstance(val, dict) and isinstance(val.get("close"), (int, float)):
                        market_data["prices"].append(float(val["close"]))

        # 3) Loose direct inputs convenience (non-contract)
        for k in ("prices", "price", "close", "price_series"):
            v = inputs.get(k)
            if isinstance(v, (list, np.ndarray)):
                market_data["prices"].extend(np.asarray(v).flatten().tolist())
            elif isinstance(v, (int, float)):
                market_data["prices"].append(float(v))

        # Clean & validate
        market_data["prices"] = self._validate_prices(market_data["prices"])
        if not market_data["prices"]:
            raise ValueError("No valid price data available")

        return market_data

    def _validate_prices(self, prices: List[float]) -> List[float]:
        valid: List[float] = []
        for p in prices:
            if isinstance(p, (int, float)) and np.isfinite(p) and p > 0:
                valid.append(float(p))
        # Simple 3σ outlier trim
        if len(valid) > 10:
            arr = np.asarray(valid, dtype=float)
            mu, sd = float(np.mean(arr)), float(np.std(arr))
            if sd > 0:
                valid = [x for x in valid if abs(x - mu) <= 3.0 * sd]
        return valid

    # ─────────────────────────────────────────────────────────
    # Feature extraction
    # ─────────────────────────────────────────────────────────
    async def _process_features_with_monitoring(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()

        # Buffer update
        self.price_buffer.extend(market_data["prices"])

        # Extract deterministic features
        feats = self._extract_comprehensive_features(market_data["prices"])

        # Quality
        quality = self._calculate_feature_quality(feats)
        self.feature_quality_score = quality

        # Persist local last
        self.last_features = feats
        self.feature_buffer.append({
            "features": feats.copy(),
            "quality_score": quality,
            "timestamp": time.time()
        })

        # Stats
        dur_ms = (time.time() - t0) * 1000.0
        self._update_feature_stats(dur_ms, quality)

        # Explanation
        explanation = self._generate_feature_explanation(feats, quality)

        return {
            "raw_features": feats,
            "quality_score": float(quality),
            "explanation": explanation,
            "extraction_time_ms": float(dur_ms),
            "buffer_size": int(len(self.price_buffer)),
            "feature_count": int(feats.size if isinstance(feats, np.ndarray) else len(feats))
        }

    def _extract_comprehensive_features(self, prices: List[float]) -> np.ndarray:
        """
        Features: per-window [mean, std, return, range, up_ratio, n], plus global
        [last_price, global_mean, global_std, global_range, pos_step_ratio, n].
        """
        if len(prices) < max(self.window_sizes):
            return self._get_fallback_features()

        arr = np.asarray(prices[-max(self.window_sizes):], dtype=float)
        feats: List[float] = []

        # Per-window stats
        for w in self.window_sizes:
            if len(arr) >= w:
                window = arr[-w:]
                diffs = np.diff(window)
                up_ratio = float(np.mean(diffs > 0.0)) if len(diffs) > 0 else 0.0
                ret = (window[-1] - window[0]) / max(window[0], 1e-12)
                feats.extend([
                    float(np.mean(window)),
                    float(np.std(window)),
                    float(ret),
                    float(np.max(window) - np.min(window)),
                    up_ratio,
                    float(len(window))
                ])
            else:
                feats.extend([0.0] * 6)

        # Global stats (no raw timestamps)
        diffs_all = np.diff(arr)
        pos_step_ratio = float(np.mean(diffs_all > 0.0)) if len(diffs_all) > 0 else 0.0
        feats.extend([
            float(arr[-1]),                         # last price
            float(np.mean(arr)),                    # global mean
            float(np.std(arr)),                     # global volatility
            float(np.max(arr) - np.min(arr)),       # global range
            pos_step_ratio,                         # positive step ratio
            float(len(arr))                         # sample size
        ])

        return np.asarray(feats, dtype=np.float32)

    def _calculate_feature_quality(self, features: np.ndarray) -> float:
        try:
            if not isinstance(features, np.ndarray) or features.size == 0:
                return 0.0
            if np.any(~np.isfinite(features)):
                return 0.0
            # Detect low-variability/flat data: per-window std/return/range mostly ~0 and globals agree
            L = max(0, (features.size - 6) // 6)
            if L > 0:
                try:
                    perwin_std = [float(features[i * 6 + 1]) for i in range(L)]
                    perwin_ret = [float(features[i * 6 + 2]) for i in range(L)]
                    perwin_rng = [float(features[i * 6 + 3]) for i in range(L)]
                    g_idx = L * 6
                    g_std = float(features[g_idx + 2])
                    g_rng = float(features[g_idx + 3])
                    pos_step = float(features[g_idx + 4])
                    zeros_std = sum(1 for v in perwin_std if abs(v) < 1e-12)
                    zeros_ret = sum(1 for v in perwin_ret if abs(v) < 1e-12)
                    zeros_rng = sum(1 for v in perwin_rng if abs(v) < 1e-12)
                    if (
                        zeros_std >= max(1, int(0.75 * L)) and
                        zeros_ret >= max(1, int(0.75 * L)) and
                        zeros_rng >= max(1, int(0.75 * L)) and
                        abs(g_std) < 1e-12 and abs(g_rng) < 1e-12 and
                        (pos_step <= 1e-6 or abs(pos_step - 1.0) <= 1e-6)
                    ):
                        return 30.0
                except Exception:
                    pass
            if float(np.std(features)) == 0.0:
                return 25.0
            span = float(np.max(features) - np.min(features))
            if span < 1e-8:
                return 35.0

            quality = 85.0
            if np.max(np.abs(features)) > 1e6:
                quality -= 15.0

            stdv = float(np.std(features))
            if 0.05 < stdv < 250.0:
                quality += 5.0

            return max(0.0, min(100.0, quality))
        except Exception:
            return 0.0

    def _generate_feature_explanation(self, feats: np.ndarray, quality: float) -> str:
        if not self.english_explainer:
            return "Feature extraction completed."
        try:
            analysis = {
                "feature_count": int(feats.size) if isinstance(feats, np.ndarray) else 0,
                "quality_score": float(quality),
                "max_value": float(np.max(feats)),
                "min_value": float(np.min(feats)),
                "mean_value": float(np.mean(feats)),
                "std_value": float(np.std(feats)),
                "window_sizes": self.window_sizes,
                "buffer_size": len(self.price_buffer),
            }
            return self.english_explainer.explain_module_decision(
                module_name="AdvancedFeatureEngine",
                decision="feature_extraction",
                context=analysis,
                confidence=quality / 100.0
            )
        except Exception as e:
            return f"Feature extraction completed (explanation generation failed: {e})"

    async def _generate_feature_thesis(self, features: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        try:
            prices = market_data["prices"]
            latest = prices[-1] if prices else 0.0
            change = (prices[-1] - prices[0]) / max(prices[0], 1e-12) if len(prices) > 1 else 0.0
            q = float(features["quality_score"])
            n = int(features["feature_count"])
            buf_util = f"{len(self.price_buffer)}/{self.max_buffer_size}"

            conf = "High" if q > 80 else ("Medium" if q > 60 else "Low")
            data_qual = "Good" if len(prices) > 50 else ("Adequate" if len(prices) > 20 else "Limited")

            return (
                "Advanced Feature Analysis\n"
                f"- Current price: {latest:.4f}\n"
                f"- Price change: {change:.2%}\n"
                f"- Data points processed: {len(prices)}\n\n"
                "Feature Quality\n"
                f"- Quality score: {q:.1f}/100\n"
                f"- Features extracted: {n}\n"
                f"- Buffer utilization: {buf_util}\n\n"
                "Confidence\n"
                f"- Extraction confidence: {conf}\n"
                f"- Data quality: {data_qual}\n"
                "Recommendation: " + ("Continue processing" if q > 60 else "Review data quality")
            )
        except Exception as e:
            return f"Feature extraction completed. Thesis generation encountered error: {e}"

    # ─────────────────────────────────────────────────────────
    # Bus I/O (declared keys only)
    # ─────────────────────────────────────────────────────────
    def _update_bus(self, features: Dict[str, Any], thesis: str, timeframe_outputs: Dict[str, Dict[str, Any]]):
        # advanced_features
        adv_payload = {
            "raw_features": features["raw_features"].tolist() if isinstance(features["raw_features"], np.ndarray)
                             else list(features["raw_features"]),
            "quality_score": float(features["quality_score"]),
            "extraction_time_ms": float(features["extraction_time_ms"]),
            "timestamp": time.time()
        }
        self.smart_bus.set(
            "advanced_features",
            adv_payload,
            module="AdvancedFeatureEngine",
            thesis=thesis
        )

        # features (alias/back-compat)
        self.smart_bus.set(
            "features",
            {
                "raw_features": adv_payload["raw_features"],
                "quality_score": float(features["quality_score"])
            },
            module="AdvancedFeatureEngine",
            thesis="Features alias for backward compatibility."
        )

        # feature_analysis
        self.smart_bus.set(
            "feature_analysis",
            {
                "explanation": features.get("explanation"),
                "buffer_status": {
                    "current_size": len(self.price_buffer),
                    "max_size": self.max_buffer_size,
                    "utilization": len(self.price_buffer) / max(self.max_buffer_size, 1)
                },
                "statistics": self.feature_stats
            },
            module="AdvancedFeatureEngine",
            thesis=f"Feature analysis summary: {features['quality_score']:.1f}% quality"
        )

        # feature_thesis
        self.smart_bus.set(
            "feature_thesis",
            thesis,
            module="AdvancedFeatureEngine",
            thesis="Feature engine thesis"
        )

        # --- Additional provides required by contract (publish conservative payloads) ---
        # feature_engine_capabilities
        self.smart_bus.set(
            "feature_engine_capabilities",
            {
                "window_sizes": list(self.window_sizes),
                "out_dim": int(self.out_dim),
                "max_buffer_size": int(self.max_buffer_size),
                "supports_explainability": bool(self.english_explainer is not None),
                "supports_error_pinpointing": bool(self.error_pinpointer is not None),
                "supports_performance_tracking": bool(self.performance_tracker is not None),
            },
            module="AdvancedFeatureEngine",
            thesis="Capabilities snapshot"
        )

        # feature_health
        self.smart_bus.set(
            "feature_health",
            {
                "health_score": float(self.health_metrics.get("health_score", 0.0)),
                "performance_trend": self.health_metrics.get("performance_trend", "unknown"),
                "statistics": dict(self.feature_stats),
            },
            module="AdvancedFeatureEngine",
            thesis="Feature engine health"
        )

        # feature_error (None on success)
        self.smart_bus.set(
            "feature_error",
            None,
            module="AdvancedFeatureEngine",
            thesis="No errors"
        )

        # market_features / price_features (minimal, conservative snapshot)
        self.smart_bus.set(
            "market_features",
            {},
            module="AdvancedFeatureEngine",
            thesis="Conservative placeholder; downstream may enrich"
        )
        self.smart_bus.set(
            "price_features",
            {},
            module="AdvancedFeatureEngine",
            thesis="Conservative placeholder; downstream may enrich"
        )

        # timeframe outputs (true per-TF if available, otherwise safe alias)
        for tf in ("H1", "H4", "D1"):
            key = f"advanced_features_{tf}"
            payload = timeframe_outputs.get(tf, {**adv_payload, "timeframe": tf, "alias_of": "advanced_features"})
            self.smart_bus.set(key, payload, module="AdvancedFeatureEngine", thesis=f"Advanced features ({tf})")

    # ─────────────────────────────────────────────────────────
    # Stats / monitoring / errors
    # ─────────────────────────────────────────────────────────
    def _update_feature_stats(self, extraction_time_ms: float, quality_score: float):
        self.feature_stats["total_extractions"] += 1
        self.feature_stats["successful_extractions"] += 1
        total = self.feature_stats["total_extractions"]
        self.feature_stats["avg_extraction_time_ms"] = (
            (self.feature_stats["avg_extraction_time_ms"] * (total - 1) + extraction_time_ms) / total
        )
        self.feature_stats["avg_feature_quality"] = (
            (self.feature_stats["avg_feature_quality"] * (total - 1) + quality_score) / total
        )

    def _check_circuit_breaker(self) -> bool:
        if self.circuit_breaker["state"] == "OPEN":
            # Only allow HALF_OPEN after cooldown if we've actually recorded a failure time
            lf = float(self.circuit_breaker.get("last_failure", 0.0) or 0.0)
            if lf > 0.0 and (time.time() - lf > 60.0):
                self.circuit_breaker["state"] = "HALF_OPEN"
                return True
            return False
        return True

    def _record_success(self, processing_time_s: float):
        if self.circuit_breaker["state"] == "HALF_OPEN":
            self.circuit_breaker["state"] = "CLOSED"
            self.circuit_breaker["failures"] = 0
        self.health_metrics["health_score"] = min(100.0, self.health_metrics["health_score"] + 1.0)
        self.health_metrics["performance_trend"] = "improving"
        if self.performance_tracker:
            self.performance_tracker.record_metric(
                "AdvancedFeatureEngine",
                "feature_extraction",
                processing_time_s * 1000.0,
                True
            )

    async def _handle_processing_error(self, error: Exception, start_time: float, *, tf_outputs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        elapsed = time.time() - start_time
        self._record_failure(error)
        if self.error_pinpointer:
            try:
                ctx = self.error_pinpointer.analyze_error(error, "AdvancedFeatureEngine")
                _ = self.error_pinpointer.create_debugging_guide(ctx)
            except Exception:
                pass

        # Do NOT publish undeclared keys to Bus; return a formatted fallback instead.
        fallback = self._get_fallback_features()
        thesis = f"Feature extraction failed: {error}. Using fallback features."
        self.logger.error(
            format_operator_message(
                "[CRASH]", "FEATURE_EXTRACTION_ERROR",
                details=str(error),
                context="feature_processing"
            )
        )

        return self._format_declared_outputs(
            features_payload={
                "raw_features": fallback,
                "quality_score": 0.0,
                "extraction_time_ms": elapsed * 1000.0,
                "buffer_size": len(self.price_buffer),
                "feature_count": int(fallback.size if isinstance(fallback, np.ndarray) else len(fallback)),
                "explanation": None,
            },
            thesis=thesis,
            analysis={
                "explanation": None,
                "statistics": self.feature_stats,
                "buffer_status": {
                    "current_size": len(self.price_buffer),
                    "max_size": self.max_buffer_size,
                    "utilization": len(self.price_buffer) / max(self.max_buffer_size, 1),
                },
            },
            extra={
                "success": False,
                "error": str(error),
                "processing_time_ms": elapsed * 1000.0,
            },
            timeframe_outputs=tf_outputs or {}
        )

    def _record_failure(self, error: Exception):
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()
        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
            self.circuit_breaker["state"] = "OPEN"
            self.logger.error(
                format_operator_message(
                    "[ALERT]", "CIRCUIT_BREAKER_OPEN",
                    details=f"Too many failures ({self.circuit_breaker['failures']})",
                    context="circuit_breaker"
                )
            )
        self.health_metrics["health_score"] = max(0.0, self.health_metrics["health_score"] - 10.0)
        self.health_metrics["issues_detected"].append(f"{type(error).__name__}: {error}")
        self.health_metrics["performance_trend"] = "degrading"
        self.feature_stats["failed_extractions"] += 1

    def _get_fallback_features(self) -> np.ndarray:
        if len(self.feature_buffer) > 0:
            return self.feature_buffer[-1]["features"]
        return np.zeros(self.out_dim, dtype=np.float32)

    def _create_fallback_response_with_tf(self, reason: str, *, tf_outputs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        return self._format_declared_outputs(
            features_payload={
                "raw_features": self._get_fallback_features(),
                "quality_score": 0.0,
                "extraction_time_ms": 0.0,
                "buffer_size": len(self.price_buffer),
                "feature_count": 0,
                "explanation": None,
            },
            thesis=f"Feature extraction unavailable: {reason}",
            analysis={
                "explanation": None,
                "statistics": self.feature_stats,
                "buffer_status": {
                    "current_size": len(self.price_buffer),
                    "max_size": self.max_buffer_size,
                    "utilization": len(self.price_buffer) / max(self.max_buffer_size, 1),
                },
            },
            extra={"success": False, "reason": reason, "processing_time_ms": 0.0},
            timeframe_outputs=tf_outputs or {}
        )

    # ─────────────────────────────────────────────────────────
    # Public API helpers
    # ─────────────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        feature_state = {
            "config": {
                "window_sizes": self.window_sizes,
                "max_buffer_size": self.max_buffer_size,
                "out_dim": self.out_dim
            },
            "buffers": {
                "price_buffer": list(self.price_buffer),
                "feature_buffer": [fb for fb in self.feature_buffer]
            },
            "features": {
                "last_features": self.last_features.tolist(),
                "quality_score": self.feature_quality_score
            },
            "statistics": self.feature_stats,
            "health_metrics": self.health_metrics,
            "circuit_breaker": self.circuit_breaker
        }
        return {**base, **feature_state}

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "buffers" in state:
            if "price_buffer" in state["buffers"]:
                self.price_buffer = deque(state["buffers"]["price_buffer"], maxlen=self.max_buffer_size)
            if "feature_buffer" in state["buffers"]:
                self.feature_buffer = deque(state["buffers"]["feature_buffer"], maxlen=1000)
        if "features" in state:
            if "last_features" in state["features"]:
                self.last_features = np.array(state["features"]["last_features"], dtype=np.float32)
            if "quality_score" in state["features"]:
                self.feature_quality_score = float(state["features"]["quality_score"])
        if "statistics" in state:
            self.feature_stats.update(state["statistics"])
        if "health_metrics" in state:
            self.health_metrics.update(state["health_metrics"])
        if "circuit_breaker" in state:
            self.circuit_breaker.update(state["circuit_breaker"])

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "health_score": self.health_metrics["health_score"],
            "circuit_breaker_state": self.circuit_breaker["state"],
            "issues_detected": list(self.health_metrics["issues_detected"]),
            "performance_trend": self.health_metrics["performance_trend"],
            "statistics": dict(self.feature_stats),
            "buffer_status": {
                "price_buffer_size": len(self.price_buffer),
                "price_buffer_utilization": len(self.price_buffer) / max(self.max_buffer_size, 1),
                "feature_buffer_size": len(self.feature_buffer)
            }
        }

    def get_performance_report(self) -> str:
        if not self.english_explainer:
            return "Performance reporting disabled or explainer unavailable."
        try:
            return self.english_explainer.explain_performance(
                module_name="AdvancedFeatureEngine",
                metrics={
                    "total_extractions": self.feature_stats["total_extractions"],
                    "success_rate": self.feature_stats["successful_extractions"] / max(self.feature_stats["total_extractions"], 1),
                    "avg_extraction_time_ms": self.feature_stats["avg_extraction_time_ms"],
                    "avg_feature_quality": self.feature_stats["avg_feature_quality"],
                    "health_score": self.health_metrics["health_score"],
                    "buffer_utilization": len(self.price_buffer) / max(self.max_buffer_size, 1)
                }
            )
        except Exception as e:
            return f"Performance report generation failed: {e}"

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """
        Lightweight example action proposal based on freshly computed features.
        (Not a voter; downstream logic decides.)
        """
        try:
            result = await self.process(**inputs)
            if not result.get("success", False):
                return {
                    "action_type": "no_action",
                    "confidence": 0.0,
                    "reasoning": "Feature extraction failed",
                    "features_available": False
                }

            adv = result["advanced_features"]
            features = np.asarray(adv["raw_features"], dtype=float)
            quality_score = float(adv["quality_score"])

            # Simple momentum heuristic across the first few windows
            slice_len = max(1, min(5, len(self.window_sizes)))
            if features.size >= 6 * slice_len:
                per_window_stats = features[: 6 * slice_len].reshape(slice_len, 6)
                recent_returns = per_window_stats[:, 2]
                avg_momentum = float(np.mean(recent_returns))
            else:
                avg_momentum = 0.0

            if quality_score > 80:
                if avg_momentum > 0.01:
                    action_type, magnitude = "increase_position", min(abs(avg_momentum) * 10.0, 1.0)
                elif avg_momentum < -0.01:
                    action_type, magnitude = "decrease_position", min(abs(avg_momentum) * 10.0, 1.0)
                else:
                    action_type, magnitude = "hold_position", 0.0
            else:
                action_type, magnitude = "reduce_risk", 0.5

            # Confidence blended from quality + signal strength
            base_conf = quality_score / 100.0
            strength = min(abs(avg_momentum) * 50.0, 1.0)
            confidence = float(np.clip(0.6 * base_conf + 0.4 * strength, 0.0, 1.0))

            # Adjust for breaker state
            if self.circuit_breaker["state"] == "OPEN":
                confidence *= 0.1
            elif self.circuit_breaker["state"] == "HALF_OPEN":
                confidence *= 0.5

            return {
                "action_type": action_type,
                "magnitude": float(magnitude),
                "confidence": float(confidence),
                "reasoning": f"{len(features)} features @ {quality_score:.1f}% quality; avg momentum {avg_momentum:.4f}",
                "features_used": int(features.size),
                "quality_score": quality_score,
                "momentum_signal": avg_momentum
            }

        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                "action_type": "no_action",
                "confidence": 0.0,
                "reasoning": f"Action proposal error: {e}",
                "error": str(e)
            }

    async def calculate_confidence(self, action: Dict[str, Any], **_inputs) -> float:
        try:
            if not isinstance(action, dict):
                return 0.0
            base_conf = self.feature_quality_score / 100.0
            action_type = action.get("action_type", "no_action")
            magnitude = float(action.get("magnitude", 0.0))

            if action_type in ("increase_position", "decrease_position"):
                features_used = int(action.get("features_used", 0))
                denom = max(1, len(self.window_sizes) * 6)
                feature_conf = min(1.0, features_used / denom)
                mag_conf = 1.0 - min(abs(magnitude), 0.5)
                combined = 0.5 * base_conf + 0.3 * feature_conf + 0.2 * mag_conf
            elif action_type == "hold_position":
                combined = 0.8 * base_conf
            elif action_type == "reduce_risk":
                combined = max(base_conf, 0.6)
            else:
                combined = 0.1

            if self.circuit_breaker["state"] == "OPEN":
                combined *= 0.1
            elif self.circuit_breaker["state"] == "HALF_OPEN":
                combined *= 0.5

            return float(np.clip(combined, 0.0, 1.0))
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────
    # Output formatting (contract)
    # ─────────────────────────────────────────────────────────
    def _format_declared_outputs(
        self,
        *,
        features_payload: Optional[Dict[str, Any]] = None,
        thesis: Optional[str] = None,
        analysis: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeframe_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        fp = features_payload or {}
        rf = fp.get("raw_features")
        if isinstance(rf, np.ndarray):
            rf_safe = rf.tolist()
        elif isinstance(rf, list):
            rf_safe = rf
        else:
            rf_safe = []

        adv = {
            "raw_features": rf_safe,
            "quality_score": float(fp.get("quality_score", 0.0)),
            "extraction_time_ms": float(fp.get("extraction_time_ms", 0.0)),
            "buffer_size": int(fp.get("buffer_size", 0)),
            "feature_count": int(fp.get("feature_count", len(rf_safe))),
        }
        out["advanced_features"] = adv
        out["features"] = {"raw_features": rf_safe, "quality_score": adv["quality_score"]}

        ana = analysis or {}
        if not isinstance(ana, dict):
            ana = {}
        out["feature_analysis"] = {
            "explanation": ana.get("explanation", fp.get("explanation") if isinstance(fp, dict) else None),
            "statistics": ana.get("statistics", {}),
            "buffer_status": ana.get("buffer_status", {}),
        }

        # ---- CRITICAL: orchestrator expects `_thesis` top-level ----
        ft = thesis or "Advanced feature extraction completed."
        out["feature_thesis"] = ft     # declared provide
        out["_thesis"] = ft            # orchestrator-required hidden key

        # Optional: include passthrough extras (success flag, timings, etc.)
        if extra:
            try:
                out.update(extra)
            except Exception:
                pass

        # Contract: feature_engine_capabilities (static/dynamic capabilities)
        out["feature_engine_capabilities"] = {
            "window_sizes": list(self.window_sizes),
            "out_dim": int(self.out_dim),
            "max_buffer_size": int(self.max_buffer_size),
            "supports_explainability": bool(self.english_explainer is not None),
            "supports_error_pinpointing": bool(self.error_pinpointer is not None),
            "supports_performance_tracking": bool(self.performance_tracker is not None),
        }

        # Contract: feature_health (compact snapshot)
        try:
            out["feature_health"] = {
                "health_score": float(self.health_metrics.get("health_score", 0.0)),
                "performance_trend": self.health_metrics.get("performance_trend", "unknown"),
                "statistics": dict(self.feature_stats),
            }
        except Exception:
            out["feature_health"] = {"health_score": 0.0, "performance_trend": "unknown", "statistics": {}}

        # Contract: feature_error (None on success, string when error path sets it via extra)
        try:
            out["feature_error"] = (extra or {}).get("error")
        except Exception:
            out["feature_error"] = None

        # Contract: market_features / price_features — conservative placeholders or aliases
        # Keep shapes minimal to avoid fabricating signals; downstream can enrich as needed.
        out.setdefault("market_features", {})
        out.setdefault("price_features", {})

        # Contract sanity: keep your declared provides intact
        for key in ("advanced_features", "features", "feature_analysis", "feature_thesis", "feature_engine_capabilities"):
            if key not in out:
                raise ValueError(f"Critical output '{key}' missing in AdvancedFeatureEngine")

        # Timeframe mirrors (true per-TF if provided, else alias mirror)
        tf_map = timeframe_outputs or {}
        for tf in ("H1", "H4", "D1"):
            alias = tf_map.get(tf)
            if not isinstance(alias, dict):
                alias = {
                    "raw_features": list(adv["raw_features"]),
                    "quality_score": float(adv["quality_score"]),
                    "extraction_time_ms": float(adv["extraction_time_ms"]),
                    "buffer_size": int(adv["buffer_size"]),
                    "feature_count": int(adv["feature_count"]),
                    "timeframe": tf, "alias_of": "advanced_features"
                }
            out[f"advanced_features_{tf}"] = alias

        return out

    # ─────────────────────────────────────────────────────────
    # Background tasks
    # ─────────────────────────────────────────────────────────
    async def _health_monitoring_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                self._update_health_metrics()
                self._check_health_issues()
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")

    def _update_health_metrics(self):
        total = self.feature_stats["total_extractions"]
        success_rate = self.feature_stats["successful_extractions"] / max(total, 1)
        if success_rate > 0.95:
            self.health_metrics["health_score"] = min(100.0, self.health_metrics["health_score"] + 0.5)
        elif success_rate < 0.8:
            self.health_metrics["health_score"] = max(0.0, self.health_metrics["health_score"] - 1.0)
        self.health_metrics["last_health_check"] = time.time()

    def _check_health_issues(self):
        issues: List[str] = []
        if self.circuit_breaker["state"] == "OPEN":
            issues.append("Circuit breaker is open")
        buf_util = len(self.price_buffer) / max(self.max_buffer_size, 1)
        if buf_util > 0.9:
            issues.append("Price buffer nearly full")
        if self.feature_stats["avg_extraction_time_ms"] > 100.0:
            issues.append("Processing time is high")
        if self.feature_stats["avg_feature_quality"] < 60.0:
            issues.append("Feature quality is low")
        self.health_metrics["issues_detected"] = issues
        if issues:
            self.logger.warning(
                format_operator_message(
                    "[WARN]", "HEALTH_ISSUES_DETECTED",
                    details=f"{len(issues)} issues found",
                    context="health_monitoring"
                )
            )

    async def _performance_monitoring_loop(self):
        while True:
            try:
                await asyncio.sleep(60)
                if self.performance_tracker:
                    self.performance_tracker.record_metric(
                        "AdvancedFeatureEngine",
                        "periodic_metrics",
                        1.0,  # collection tick
                        True
                    )
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")

    # ─────────────────────────────────────────────────────────
    # Internal helpers (timeframe extraction & packaging)
    # ─────────────────────────────────────────────────────────
    def _build_adv_from_payload(self, fp: Dict[str, Any]) -> Dict[str, Any]:
        rf = fp.get("raw_features")
        if isinstance(rf, np.ndarray):
            rf_safe = rf.tolist()
        elif isinstance(rf, list):
            rf_safe = rf
        else:
            rf_safe = []
        return {
            "raw_features": rf_safe,
            "quality_score": float(fp.get("quality_score", 0.0)),
            "extraction_time_ms": float(fp.get("extraction_time_ms", 0.0)),
            "buffer_size": int(fp.get("buffer_size", 0)),
            "feature_count": int(fp.get("feature_count", len(rf_safe))),
        }

    def _compute_timeframe_outputs(self, base_adv: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Attempt to compute true per-TF advanced features from multi_timeframe_data.
        If not available, provide robust aliases.
        """
        outputs: Dict[str, Dict[str, Any]] = {}
        if not self._mtf_present:
            for tf in ("H1", "H4", "D1"):
                outputs[tf] = {**base_adv, "timeframe": tf, "alias_of": "advanced_features", "mtf_available": False}
            return outputs

        mtd = None
        try:
            mtd = self.smart_bus.get("multi_timeframe_data", self.__class__.__name__)
        except Exception:
            mtd = None

        for tf in ("H1", "H4", "D1"):
            prices = self._collect_mtf_prices(mtd, tf)
            if prices:
                t0 = time.time()
                feats = self._extract_comprehensive_features(prices)
                q = self._calculate_feature_quality(feats)
                dur_ms = (time.time() - t0) * 1000.0
                outputs[tf] = {
                    "raw_features": feats.tolist(),
                    "quality_score": float(q),
                    "extraction_time_ms": float(dur_ms),
                    "buffer_size": int(len(self.price_buffer)),
                    "feature_count": int(feats.size),
                    "timeframe": tf
                }
            else:
                outputs[tf] = {**base_adv, "timeframe": tf, "alias_of": "advanced_features", "mtf_available": True}

        return outputs

    def _collect_mtf_prices(self, mtd: Any, tf: str) -> List[float]:
        """
        Heuristic extractor for closes from multi_timeframe_data.
        Expected tolerant shapes:
          - {symbol: {TF: {'close': [...]} } }
          - {TF: {'close': [...]} }
          - {symbol: {TF: [{'close': x}, ...] } }
          - {TF: [ {'close': x}, ... ] }
          - {symbol: {TF: np.ndarray/ list } }
        """
        prices: List[float] = []
        if not isinstance(mtd, dict):
            return prices

        def _extend_from_payload(payload: Any):
            nonlocal prices
            if isinstance(payload, dict):
                if isinstance(payload.get("close"), (list, np.ndarray)):
                    prices.extend(np.asarray(payload["close"], dtype=float).flatten().tolist())
                elif isinstance(payload.get("current_bar", {}).get("close"), (int, float)):
                    prices.append(float(payload["current_bar"]["close"]))
                elif "bars" in payload and isinstance(payload["bars"], (list, np.ndarray)):
                    for bar in payload["bars"]:
                        if isinstance(bar, dict) and isinstance(bar.get("close"), (int, float)):
                            prices.append(float(bar["close"]))
            elif isinstance(payload, (list, np.ndarray)):
                # Maybe it's a list of closes or list of bar dicts
                if len(payload) > 0 and isinstance(payload[0], dict):
                    for bar in payload:
                        if isinstance(bar, dict) and isinstance(bar.get("close"), (int, float)):
                            prices.append(float(bar["close"]))
                else:
                    prices.extend(np.asarray(payload, dtype=float).flatten().tolist())

        # Shape 1: direct TF at top-level
        if tf in mtd:
            _extend_from_payload(mtd.get(tf))

        # Shape 2: symbols at top-level
        for _sym, per_sym in mtd.items():
            if isinstance(per_sym, dict) and tf in per_sym:
                _extend_from_payload(per_sym.get(tf))

        # Validate
        return self._validate_prices(prices)
