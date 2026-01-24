# ─────────────────────────────────────────────────────────────
# File: modules/features/advanced_feature_engine.py
# Advanced Feature Engine (XAUUSD-only, Contract-Clean, Single-Writer, Hard-Audit Logging)
#
# Fixes applied (architectural + correctness hardening):
# - Guaranteed attribute initialization order (debug / preview knobs exist before any audit path)
# - No “success” on empty-after-validate price series (all sources re-checked post-validate)
# - Incremental price ingestion to prevent buffer ballooning on “full-history every tick” feeds
# - Timeframe mirror quality correctness (no market_data.quality_score leak; uses computed q)
# - Consistent feature_error shape (dict on error, None on success) in returned outputs
# - Bus write serialization via lock (single-writer invariant even under concurrent calls)
# - Added shutdown() to stop background loop and close audit handlers (hot-reload safe)
# - Removed silent exception swallowing in non-last-resort paths (all meaningful failures audited)
# - Log sync: cycle_id/step_idx/bus_timestamp/session_id/correlation_id propagated into audit
# - Robust float matching tolerance for float32↔float64 feeds (prevents duplicate ingestion)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import json
import threading
import time
import asyncio
import hashlib
import logging
import traceback
from logging.handlers import RotatingFileHandler

import numpy as np
from typing import Dict, Any, List, Optional, Union
from collections import deque
from dataclasses import dataclass, asdict

# Core infrastructure
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.contracts import module_args


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass
class FeatureEngineConfig:
    # Feature geometry
    window_sizes: Optional[List[int]] = None
    max_buffer_size: int = 2000

    # Instrument scope (HARD)
    symbol: str = "XAUUSD"
    primary_timeframe: str = "M15"
    timeframes: Optional[List[str]] = None  # for TF mirrors

    # Robustness / sync
    ingest_match_tol: float = 1e-3  # tolerates float32 rounding vs float64 feeds

    # Monitoring knobs
    enable_neural_processing: bool = True  # placeholder hook
    enable_health_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_error_pinpointing: bool = True
    enable_english_explanations: bool = True

    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_s: float = 60.0

    # Audit logging (ONE FILE)
    audit_log_enabled: bool = True
    audit_log_path: str = "logs/audit/advanced_feature_engine.log.jsonl"
    audit_max_bytes: int = 50_000_000  # 50MB
    audit_backup_count: int = 5

    # Verbosity control (single switch)
    debug: bool = True

    # Debug controls (only used when debug=True)
    debug_preview_n: int = 8  # head/tail preview size when not dumping full arrays
    debug_full_prices: bool = True
    debug_full_features: bool = True

    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [7, 14, 28, 56]
        if self.timeframes is None:
            self.timeframes = ["M15", "H1", "H4", "D1"]


# ─────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────
@module(**module_args(
    "AdvancedFeatureEngine",
    description="XAUUSD-only deterministic multi-window feature extraction with circuit breaker, monitoring, and hard-audit logging.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class AdvancedFeatureEngine(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Contract intent:
    - Provides: advanced_features, features (alias), feature_analysis, feature_thesis,
      feature_engine_capabilities, feature_health, feature_error, market_features, price_features,
      advanced_features_{M15,H1,H4,D1} (mirrors)
    - Requires: price_data (preferred), historical_prices, ohlcv_data, market_data,
      multi_timeframe_data (soft-checked and used if present)

    Hard scope:
    - ONLY one instrument: XAUUSD (config.symbol). No multi-symbol blending.
    """

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────
    def __init__(self, config: Optional[Union[FeatureEngineConfig, Dict[str, Any]]] = None, **kwargs):
        if isinstance(config, dict):
            filtered: Dict[str, Any] = {k: config[k] for k in FeatureEngineConfig.__dataclass_fields__ if k in config}
            self._cfg = FeatureEngineConfig(**filtered)
        elif isinstance(config, FeatureEngineConfig) or config is None:
            self._cfg = config or FeatureEngineConfig()
        else:
            self._cfg = FeatureEngineConfig()

        # Ensure these exist even if an audit path triggers very early.
        self.debug = bool(self._cfg.debug)
        self._preview_n = int(self._cfg.debug_preview_n)
        self._full_prices = bool(self._cfg.debug_full_prices) if self.debug else False
        self._full_features = bool(self._cfg.debug_full_features) if self.debug else False

        # Sync context (filled per-cycle)
        self._sync_ctx: Dict[str, Any] = {}

        super().__init__(config=asdict(self._cfg), **kwargs)

    def _initialize(self):
        self.smart_bus = InfoBusManager.get_instance()
        self._bus_lock = threading.Lock()  # single-writer invariant for bus writes + tracking list

        # Hard instrument & TF scope
        self.symbol = str(self._cfg.symbol or "XAUUSD").upper()
        self.primary_timeframe = str(self._cfg.primary_timeframe or "M15").upper()
        self.timeframes = [str(x).upper() for x in (self._cfg.timeframes or ["M15", "H1", "H4", "D1"])]

        # Robustness
        self._ingest_tol = float(getattr(self._cfg, "ingest_match_tol", 1e-3) or 1e-3)

        # Feature geometry
        self.window_sizes = sorted(self._cfg.window_sizes or [7, 14, 28, 56])
        self.out_dim = len(self.window_sizes) * 6 + 6
        self.max_buffer_size = int(self._cfg.max_buffer_size)

        # Subsystems
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

        # Circuit breaker
        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
            "threshold": int(self._cfg.circuit_breaker_threshold),
            "cooldown_s": float(self._cfg.circuit_breaker_cooldown_s),
        }

        # State
        self._initialize_feature_state()

        # Audit logger (ONE FILE, JSONL)
        self._cycle_idx = 0
        self._audit_logger = self._init_audit_logger()

        # Verbosity controls (single switch) — re-assert after _initialize for safety.
        self.debug = bool(self._cfg.debug)
        self._preview_n = int(self._cfg.debug_preview_n)
        self._full_prices = bool(self._cfg.debug_full_prices) if self.debug else False
        self._full_features = bool(self._cfg.debug_full_features) if self.debug else False

        # Clean tracking buffer BEFORE any baseline _bus_set calls
        self._reset_last_bus_writes()

        # Background loops
        self._start_monitoring()

        # Operator log
        self.logger.info(
            format_operator_message(
                "[INIT]", "ADVANCED_FEATURE_ENGINE_READY",
                details=(
                    f"Symbol={self.symbol}, PrimaryTF={self.primary_timeframe}, TFs={self.timeframes}, "
                    f"Windows={self.window_sizes}, OutDim={self.out_dim}, Buffer={self.max_buffer_size}, "
                    f"Debug={self.debug}, IngestTol={self._ingest_tol}"
                ),
                result="ready",
                context="feature_engine_startup"
            )
        )

        # Publish baseline (best-effort; failures audited)
        try:
            baseline_feats = self._get_fallback_features()
            thesis = "Baseline features published at init to avoid BUS MISS."
            adv_payload = self._make_adv_payload(baseline_feats, quality=0.0, extraction_ms=0.0)
            adv_payload.update({"instrument": self.symbol, "timeframe": self.primary_timeframe, "source": "init_baseline"})

            self._bus_set("advanced_features", adv_payload, thesis=thesis)
            self._bus_set(
                "features",
                {"raw_features": adv_payload["raw_features"], "quality_score": 0.0, "instrument": self.symbol, "timeframe": self.primary_timeframe},
                thesis="Features alias (baseline)"
            )
            self._bus_set("feature_engine_capabilities", self._capabilities_snapshot(), thesis="Capabilities snapshot (baseline)")
            self._bus_set("feature_health", self._health_snapshot(), thesis="Feature engine health (baseline)")
            self._bus_set("feature_error", None, thesis="No errors (baseline)")
            for tf in self.timeframes:
                self._bus_set(
                    f"advanced_features_{tf}",
                    {**adv_payload, "timeframe": tf, "alias_of": "advanced_features", "mtf_available": (tf != self.primary_timeframe)},
                    thesis=f"Baseline mirror ({tf})"
                )
        except Exception as e:
            self._audit_event(
                level="ERROR",
                event_type="init_baseline_publish_failed",
                error=str(e),
                trace=traceback.format_exc()
            )

    def shutdown(self):
        """
        Hot-reload safe shutdown:
        - Stop background loop if we created one
        - Close audit logger handlers
        """
        try:
            loop = getattr(self, "_bg_loop", None)
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
        except Exception as e:
            self._audit_event(level="WARN", event_type="shutdown_loop_stop_failed", error=str(e), trace=traceback.format_exc())

        try:
            lg = getattr(self, "_audit_logger", None)
            if lg is not None:
                for h in list(lg.handlers):
                    try:
                        h.flush()
                        h.close()
                    except Exception:
                        pass
                lg.handlers.clear()
        except Exception as e:
            self._audit_event(level="WARN", event_type="shutdown_audit_close_failed", error=str(e), trace=traceback.format_exc())

    def _initialize_feature_state(self):
        self.price_buffer: deque = deque(maxlen=self.max_buffer_size)
        self.feature_buffer: deque = deque(maxlen=2000)
        self.last_features = np.zeros(self.out_dim, dtype=np.float32)
        self.feature_quality_score = 0.0

        # Must exist before any _bus_set/_track_bus_write is called.
        self._last_bus_writes: List[Dict[str, Any]] = []

        self.feature_stats: Dict[str, Any] = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_extraction_time_ms": 0.0,
            "avg_feature_quality": 0.0,
            "price_points_processed": 0,
            "last_success_ts": 0.0,
            "last_failure_ts": 0.0,
            "last_ingest_n": 0,
            "last_source_n": 0,
        }

        self.health_metrics: Dict[str, Any] = {
            "last_health_check": time.time(),
            "health_score": 100.0,
            "issues_detected": [],
            "performance_trend": "stable",
        }

    def _start_monitoring(self):
        # Nothing to do
        if not (self._cfg.enable_health_monitoring or self._cfg.enable_performance_tracking):
            return

        def _schedule(loop: asyncio.AbstractEventLoop) -> None:
            if self._cfg.enable_health_monitoring:
                loop.create_task(self._health_monitoring_loop())
            if self._cfg.enable_performance_tracking:
                loop.create_task(self._performance_monitoring_loop())

        # Try current running loop first
        try:
            loop = asyncio.get_running_loop()
            _schedule(loop)
            self._audit_event(level="INFO" if self.debug else "DEBUG", event_type="monitoring_tasks_scheduled", mode="running_loop")
            return
        except RuntimeError:
            pass

        # No running loop: start a dedicated background loop once
        if getattr(self, "_bg_loop", None) is not None:
            return

        self._bg_loop = asyncio.new_event_loop()

        def _runner():
            try:
                asyncio.set_event_loop(self._bg_loop)
                _schedule(self._bg_loop)
                self._bg_loop.run_forever()
            except Exception as e:
                self._audit_event(level="ERROR", event_type="monitoring_bg_loop_failed", error=str(e), trace=traceback.format_exc())

        self._bg_thread = threading.Thread(target=_runner, name="AdvancedFeatureEngineMonitoring", daemon=True)
        self._bg_thread.start()

        self._audit_event(level="INFO" if self.debug else "DEBUG", event_type="monitoring_tasks_scheduled", mode="bg_thread_loop")

    # ─────────────────────────────────────────────────────────
    # Main processing
    # ─────────────────────────────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        start_ts = time.time()

        # Capture sync context first (used in *all* audit events this cycle)
        self._sync_ctx = self._capture_sync_context(inputs)

        # Prefer external step_idx as cycle_id to keep MarketDataProvider / AFE / MultiScale aligned
        self._cycle_idx += 1
        internal_cycle = int(self._cycle_idx)
        step_idx = self._sync_ctx.get("step_idx")
        cycle_id = int(step_idx) if isinstance(step_idx, int) else internal_cycle

        bus_ts = self._sync_ctx.get("bus_timestamp")

        # Circuit breaker gating
        if not self._check_circuit_breaker():
            payload = self._make_adv_payload(self._get_fallback_features(), quality=0.0, extraction_ms=0.0)
            payload.update({
                "instrument": self.symbol,
                "timeframe": self.primary_timeframe,
                "source": "circuit_breaker_fallback",
                "bus_timestamp": bus_ts,
                "step_idx": step_idx,
            })
            tf_outputs = {tf: {**payload, "timeframe": tf, "alias_of": "advanced_features", "mtf_available": False} for tf in self.timeframes}
            thesis = "Circuit breaker OPEN: returning fallback features."
            self._audit_event(
                level="WARN",
                event_type="circuit_breaker_open",
                cycle_id=cycle_id,
                breaker=dict(self.circuit_breaker),
            )
            return self._format_declared_outputs(
                features_payload=payload,
                thesis=thesis,
                analysis={"buffer_status": self._buffer_status(), "statistics": dict(self.feature_stats), "explanation": None},
                extra={"success": False, "reason": "circuit_breaker_open", "processing_time_ms": 0.0, "feature_error": {"error": "circuit_breaker_open"}},
                timeframe_outputs=tf_outputs
            )

        market_data = None
        features_payload = None
        thesis = None
        tf_outputs: Dict[str, Dict[str, Any]] = {}

        try:
            market_data = await self._extract_market_data(**inputs)  # strict + audit
            features_payload = await self._process_features_with_monitoring(market_data)
            thesis = await self._generate_feature_thesis(features_payload, market_data)

            # TF outputs (per timeframe, still XAUUSD only)
            tf_outputs = self._compute_timeframe_outputs(market_data, base_feats=features_payload["raw_features"])

            # Publish (critical writes)
            self._update_bus(features_payload, thesis, tf_outputs)

            # Success accounting
            elapsed_ms = (time.time() - start_ts) * 1000.0
            self._update_feature_stats(
                extraction_time_ms=float(features_payload.get("extraction_time_ms", elapsed_ms)),
                quality_score=float(features_payload.get("quality_score", 0.0)),
                success=True
            )
            self.feature_stats["last_success_ts"] = time.time()
            self._record_success(processing_time_s=(time.time() - start_ts))

            # Per-cycle audit
            self._audit_cycle(
                level="INFO",
                cycle_id=cycle_id,
                step_idx=step_idx,
                bus_timestamp=bus_ts,
                start_ts=start_ts,
                market_data=market_data,
                features_payload=features_payload,
                thesis=thesis,
                tf_outputs=tf_outputs,
                bus_writes=self._last_bus_writes_snapshot(),
                success=True
            )

            adv_out = self._make_adv_payload(
                features_payload["raw_features"],
                quality=float(features_payload["quality_score"]),
                extraction_ms=float(features_payload["extraction_time_ms"])
            )
            adv_out.update({
                "instrument": self.symbol,
                "timeframe": self.primary_timeframe,
                "source": features_payload.get("source", "unknown"),
                "bus_timestamp": bus_ts,
                "step_idx": step_idx,
            })

            return self._format_declared_outputs(
                features_payload=adv_out,
                thesis=thesis,
                analysis={
                    "explanation": features_payload.get("explanation"),
                    "statistics": dict(self.feature_stats),
                    "buffer_status": self._buffer_status(),
                    "instrument": self.symbol,
                    "primary_timeframe": self.primary_timeframe,
                    "source": market_data.get("source", "unknown") if isinstance(market_data, dict) else "unknown",
                },
                extra={"success": True, "processing_time_ms": elapsed_ms, "feature_error": None},
                timeframe_outputs=tf_outputs
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_ts) * 1000.0

            # Failure accounting
            self._update_feature_stats(extraction_time_ms=float(elapsed_ms), quality_score=0.0, success=False)
            self.feature_stats["last_failure_ts"] = time.time()
            self._record_failure(e)

            # Pinpointer (best-effort, but never silent)
            pin = None
            if self.error_pinpointer:
                try:
                    ctx = self.error_pinpointer.analyze_error(e, "AdvancedFeatureEngine")
                    pin = {"context": ctx, "guide": self.error_pinpointer.create_debugging_guide(ctx)}
                except Exception as pe:
                    pin = {"pinpointer_failed": str(pe), "trace": traceback.format_exc()}

            # Audit error
            self._audit_cycle(
                level="ERROR",
                cycle_id=cycle_id,
                step_idx=self._sync_ctx.get("step_idx"),
                bus_timestamp=self._sync_ctx.get("bus_timestamp"),
                start_ts=start_ts,
                market_data=market_data,
                features_payload=features_payload,
                thesis=thesis,
                tf_outputs=tf_outputs or {},
                bus_writes=self._last_bus_writes_snapshot(),
                success=False,
                error=str(e),
                trace=traceback.format_exc(),
                pinpointer=pin
            )

            fallback = self._get_fallback_features()
            payload = self._make_adv_payload(fallback, quality=0.0, extraction_ms=float(elapsed_ms))
            payload.update({
                "instrument": self.symbol,
                "timeframe": self.primary_timeframe,
                "source": "exception_fallback",
                "bus_timestamp": self._sync_ctx.get("bus_timestamp"),
                "step_idx": self._sync_ctx.get("step_idx"),
            })
            tf_fallback = {tf: {**payload, "timeframe": tf, "alias_of": "advanced_features"} for tf in self.timeframes}

            # Publish error/health only; do NOT publish advanced_features on failure.
            err_obj = {"error": str(e), "trace": traceback.format_exc() if self.debug else None}
            try:
                self._bus_set("feature_error", err_obj, thesis="Feature extraction error (reported)")
                self._bus_set("feature_health", self._health_snapshot(), thesis="Health snapshot after failure")
            except Exception as be:
                self._audit_event(level="ERROR", event_type="bus_set_failed_during_error_reporting", error=str(be), trace=traceback.format_exc())

            return self._format_declared_outputs(
                features_payload=payload,
                thesis=f"Feature extraction failed: {e}. Returning fallback.",
                analysis={
                    "explanation": None,
                    "statistics": dict(self.feature_stats),
                    "buffer_status": self._buffer_status(),
                    "instrument": self.symbol,
                    "primary_timeframe": self.primary_timeframe,
                },
                extra={"success": False, "error": str(e), "processing_time_ms": float(elapsed_ms), "feature_error": err_obj},
                timeframe_outputs=tf_fallback
            )

    def _capture_sync_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keeps logs from MarketDataProvider / AdvancedFeatureEngine / MultiScaleFeatureEngine aligned.
        We do *not* hard-require any of these keys; missing data is fine.
        """
        ctx: Dict[str, Any] = {}
        # Prefer bus step_idx/timestamp (pipeline-owned)
        try:
            step_idx = self._bus_get("step_idx", default=None, soft=True)
            if isinstance(step_idx, (int, np.integer)):
                ctx["step_idx"] = int(step_idx)
        except Exception:
            pass

        try:
            ts = self._bus_get("timestamp", default=None, soft=True)
            if ts is not None:
                ctx["bus_timestamp"] = ts
        except Exception:
            pass

        # Optional run identifiers (from inputs or bus)
        for k in ("session_id", "correlation_id", "execution_id"):
            v = inputs.get(k)
            if v is None:
                try:
                    v = self._bus_get(k, default=None, soft=True)
                except Exception:
                    v = None
            if v is not None:
                ctx[k] = v

        return ctx

    # ─────────────────────────────────────────────────────────
    # Inputs (XAUUSD-only, strict + audited)
    # ─────────────────────────────────────────────────────────
    async def _extract_market_data(self, **inputs) -> Dict[str, Any]:
        """
        Extract a single coherent XAUUSD price series for primary timeframe.
        No multi-symbol blending. No silent fallbacks.
        """
        audit: Dict[str, Any] = {
            "instrument": self.symbol,
            "primary_timeframe": self.primary_timeframe,
            "inputs_keys": sorted(list(inputs.keys())),
            "attempts": [],
            "bus_reads": [],
        }

        def _attempt(name: str, ok: bool, details: Dict[str, Any]):
            audit["attempts"].append({"name": name, "ok": bool(ok), "details": details})

        # 1) Prefer inputs.price_data
        prices = self._extract_from_price_data(inputs.get("price_data"), symbol=self.symbol)
        if prices:
            prices = self._validate_prices(prices)
            if prices:
                _attempt("inputs.price_data", True, self._summarize_prices(prices))
                return {"instrument": self.symbol, "timeframe": self.primary_timeframe, "prices": prices, "source": "inputs.price_data", "audit": audit}
            _attempt("inputs.price_data", False, {"reason": "all_prices_invalid_after_validate"})
        else:
            _attempt("inputs.price_data", False, {"reason": "no_prices_for_symbol"})

        # 2) Try bus.historical_prices[XAUUSD][primary_tf]
        hist = self._bus_get("historical_prices", default=None, soft=True, audit=audit)
        prices = self._extract_from_historical_prices(hist, symbol=self.symbol, timeframe=self.primary_timeframe)
        if prices:
            prices = self._validate_prices(prices)
            if prices:
                _attempt("bus.historical_prices", True, self._summarize_prices(prices))
                return {
                    "instrument": self.symbol,
                    "timeframe": self.primary_timeframe,
                    "prices": prices,
                    "source": f"bus.historical_prices[{self.symbol}][{self.primary_timeframe}]",
                    "audit": audit
                }
            _attempt("bus.historical_prices", False, {"reason": "all_prices_invalid_after_validate"})
        else:
            _attempt("bus.historical_prices", False, {"reason": "no_prices_for_symbol_tf"})

        # 3) Try bus.multi_timeframe_data (XAUUSD-only)
        mtd = self._bus_get("multi_timeframe_data", default=None, soft=True, audit=audit)
        prices = self._extract_from_multi_timeframe_data(mtd, symbol=self.symbol, timeframe=self.primary_timeframe)
        if prices:
            prices = self._validate_prices(prices)
            if prices:
                _attempt("bus.multi_timeframe_data", True, self._summarize_prices(prices))
                return {
                    "instrument": self.symbol,
                    "timeframe": self.primary_timeframe,
                    "prices": prices,
                    "source": f"bus.multi_timeframe_data[{self.symbol}][{self.primary_timeframe}]",
                    "audit": audit
                }
            _attempt("bus.multi_timeframe_data", False, {"reason": "all_prices_invalid_after_validate"})
        else:
            _attempt("bus.multi_timeframe_data", False, {"reason": "no_prices_for_symbol_tf"})

        # 4) Try bus.ohlcv_data (XAUUSD latest close only; not a series)
        ohlcv = self._bus_get("ohlcv_data", default=None, soft=True, audit=audit)
        one = self._extract_latest_close(ohlcv, symbol=self.symbol)
        if one is not None:
            prices = self._validate_prices([one])
            if prices:
                _attempt("bus.ohlcv_data", True, self._summarize_prices(prices))
                return {"instrument": self.symbol, "timeframe": self.primary_timeframe, "prices": prices,
                        "source": f"bus.ohlcv_data[{self.symbol}]", "audit": audit}
            _attempt("bus.ohlcv_data", False, {"reason": "latest_close_invalid_after_validate"})
        else:
            _attempt("bus.ohlcv_data", False, {"reason": "no_close_for_symbol"})

        # 5) Try bus.market_data (XAUUSD latest close)
        mkt = self._bus_get("market_data", default=None, soft=True, audit=audit)
        one = self._extract_latest_close(mkt, symbol=self.symbol)
        if one is not None:
            prices = self._validate_prices([one])
            if prices:
                _attempt("bus.market_data", True, self._summarize_prices(prices))
                return {"instrument": self.symbol, "timeframe": self.primary_timeframe, "prices": prices,
                        "source": f"bus.market_data[{self.symbol}]", "audit": audit}
            _attempt("bus.market_data", False, {"reason": "latest_close_invalid_after_validate"})
        else:
            _attempt("bus.market_data", False, {"reason": "no_close_for_symbol"})

        raise ValueError(
            f"No valid price data found for {self.symbol} (primary TF={self.primary_timeframe}). "
            f"Attempts={len(audit['attempts'])}"
        )

    # ---- Extractors (XAUUSD-only) ----
    def _extract_from_price_data(self, pd_map: Any, symbol: str) -> List[float]:
        out: List[float] = []
        if not isinstance(pd_map, dict):
            return out

        entry = None
        for k, v in pd_map.items():
            if str(k).upper() == symbol.upper():
                entry = v
                break
        if entry is None:
            return out

        try:
            if isinstance(entry, dict):
                # Prefer series if present; otherwise use scalar close/price
                closes = entry.get("closes") or entry.get("close_series")
                if isinstance(closes, (list, np.ndarray)) and len(closes) > 0:
                    out.extend(np.asarray(closes, dtype=float).flatten().tolist())
                else:
                    close = entry.get("close")
                    if close is None:
                        close = entry.get("price")
                    if isinstance(close, (int, float, np.floating)):
                        out.append(float(close))

                # If both scalar and series exist, avoid duplication by only appending scalar
                # when it differs from series tail (tolerant for float32 rounding).
                close = entry.get("close")
                if isinstance(close, (int, float, np.floating)) and out:
                    if abs(float(close) - float(out[-1])) > self._ingest_tol:
                        out.append(float(close))

            elif isinstance(entry, (list, tuple, np.ndarray)):
                if len(entry) == 0:
                    return out
                if isinstance(entry[0], dict):
                    for bar in entry:
                        c = bar.get("close")
                        if isinstance(c, (int, float, np.floating)):
                            out.append(float(c))
                else:
                    out.extend(np.asarray(entry, dtype=float).flatten().tolist())

            elif isinstance(entry, (int, float, np.floating)):
                out.append(float(entry))

        except Exception as e:
            self._audit_event(level="ERROR", event_type="extract_from_price_data_failed", error=str(e), trace=traceback.format_exc())

        return out

    def _extract_from_historical_prices(self, hist: Any, symbol: str, timeframe: str) -> List[float]:
        out: List[float] = []
        if not isinstance(hist, dict):
            return out
        sym_block = None
        for k, v in hist.items():
            if str(k).upper() == symbol.upper():
                sym_block = v
                break
        if not isinstance(sym_block, dict):
            return out

        tf_block = None
        for k, v in sym_block.items():
            if str(k).upper() == timeframe.upper():
                tf_block = v
                break
        if not isinstance(tf_block, dict):
            return out

        try:
            closes = tf_block.get("close")
            if isinstance(closes, (list, np.ndarray)) and len(closes) > 0:
                out.extend(np.asarray(closes, dtype=float).flatten().tolist())

            cb = tf_block.get("current_bar")
            if isinstance(cb, dict) and isinstance(cb.get("close"), (int, float, np.floating)):
                if len(out) > 0:
                    out[-1] = float(cb["close"])
                else:
                    out.append(float(cb["close"]))
        except Exception as e:
            self._audit_event(level="ERROR", event_type="extract_from_historical_prices_failed", error=str(e), trace=traceback.format_exc())

        return out

    def _extract_from_multi_timeframe_data(self, mtd: Any, symbol: str, timeframe: str) -> List[float]:
        out: List[float] = []
        if not isinstance(mtd, dict):
            return out

        sym_block = None
        for k, v in mtd.items():
            if str(k).upper() == symbol.upper():
                sym_block = v
                break

        candidate = None
        if isinstance(sym_block, dict):
            for k, v in sym_block.items():
                if str(k).upper() == timeframe.upper():
                    candidate = v
                    break
        else:
            # Some producers store TF at top-level
            for k, v in mtd.items():
                if str(k).upper() == timeframe.upper():
                    candidate = v
                    break

        if candidate is None:
            return out

        try:
            if isinstance(candidate, dict):
                closes = candidate.get("close")
                if isinstance(closes, (list, np.ndarray)) and len(closes) > 0:
                    out.extend(np.asarray(closes, dtype=float).flatten().tolist())
                cb = candidate.get("current_bar")
                if isinstance(cb, dict) and isinstance(cb.get("close"), (int, float, np.floating)):
                    if len(out) > 0:
                        out[-1] = float(cb["close"])
                    else:
                        out.append(float(cb["close"]))
                bars = candidate.get("bars")
                if isinstance(bars, (list, np.ndarray)) and len(bars) > 0 and isinstance(bars[0], dict):
                    for bar in bars:
                        c = bar.get("close")
                        if isinstance(c, (int, float, np.floating)):
                            out.append(float(c))

            elif isinstance(candidate, (list, tuple, np.ndarray)):
                if len(candidate) == 0:
                    return out
                if isinstance(candidate[0], dict):
                    for bar in candidate:
                        c = bar.get("close")
                        if isinstance(c, (int, float, np.floating)):
                            out.append(float(c))
                else:
                    out.extend(np.asarray(candidate, dtype=float).flatten().tolist())

        except Exception as e:
            self._audit_event(level="ERROR", event_type="extract_from_multi_timeframe_data_failed", error=str(e), trace=traceback.format_exc())

        return out

    def _looks_like_quote(self, d: Dict[str, Any]) -> bool:
        # Heuristic: OHLC or at least close plus some structure
        if not isinstance(d, dict):
            return False
        if "close" not in d:
            return False
        c = d.get("close")
        if not isinstance(c, (int, float, np.floating)):
            return False
        # If it's a quote dict, it often has OHLC fields
        ohlc = {"open", "high", "low", "close"}
        if len(ohlc.intersection(set(d.keys()))) >= 2:
            return True
        # Or bar_state/time fields
        if "bar_state" in d or "timestamp" in d or "time" in d:
            return True
        return True

    def _extract_latest_close(self, blob: Any, symbol: str) -> Optional[float]:
        if blob is None:
            return None

        # Case A: direct quote dict (no symbol key)
        if isinstance(blob, dict) and self._looks_like_quote(blob):
            return float(blob["close"])

        # Case B: mapping keyed by symbol
        if not isinstance(blob, dict):
            return None

        for k, v in blob.items():
            if str(k).upper() == symbol.upper():
                if isinstance(v, dict) and isinstance(v.get("close"), (int, float, np.floating)):
                    return float(v["close"])
                if isinstance(v, (int, float, np.floating)):
                    return float(v)

        # Case C: shallow nested: accept only if nested dict explicitly declares symbol/instrument
        for _, v in blob.items():
            if isinstance(v, dict):
                declared = v.get("symbol") or v.get("instrument")
                if declared is not None and str(declared).upper() == symbol.upper():
                    if isinstance(v.get("close"), (int, float, np.floating)):
                        return float(v["close"])
                # Last-resort heuristic: if dict looks like quote and blob size is small (avoid random dicts)
                if len(blob) <= 3 and self._looks_like_quote(v):
                    return float(v["close"])
        return None

    def _validate_prices(self, prices: List[float]) -> List[float]:
        valid: List[float] = []
        invalid_n = 0
        for p in prices:
            try:
                if isinstance(p, (int, float, np.floating)) and np.isfinite(p) and float(p) > 0.0:
                    valid.append(float(p))
                else:
                    invalid_n += 1
            except Exception:
                invalid_n += 1

        # Outlier trim (3σ) if large enough
        trimmed = 0
        if len(valid) > 20:
            arr = np.asarray(valid, dtype=float)
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            if sd > 0:
                before = len(valid)
                valid = [x for x in valid if abs(x - mu) <= 3.0 * sd]
                trimmed = before - len(valid)

        # Audit only when something meaningful happened (avoid log spam).
        if (invalid_n > 0 or trimmed > 0) and self.debug:
            self._audit_event(
                level="DEBUG",
                event_type="validate_prices_summary",
                invalid_n=int(invalid_n),
                trimmed_outliers=int(trimmed),
                in_n=int(len(prices)),
                out_n=int(len(valid)),
            )

        return valid

    # ─────────────────────────────────────────────────────────
    # Feature extraction
    # ─────────────────────────────────────────────────────────
    async def _process_features_with_monitoring(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()

        src_prices = market_data["prices"]
        ingest_n = self._ingest_prices(src_prices)
        self.feature_stats["price_points_processed"] += int(ingest_n)
        self.feature_stats["last_ingest_n"] = int(ingest_n)
        self.feature_stats["last_source_n"] = int(len(src_prices))

        # Always compute from internal buffer to avoid “append full history every tick” blowups.
        series = list(self.price_buffer)
        feats = self._extract_comprehensive_features(series)
        quality = self._calculate_feature_quality(feats)
        self.feature_quality_score = float(quality)

        self.last_features = feats
        self.feature_buffer.append({
            "features": feats.copy(),
            "quality_score": float(quality),
            "timestamp": time.time(),
        })

        dur_ms = (time.time() - t0) * 1000.0
        explanation = self._generate_feature_explanation(feats, float(quality))

        return {
            "raw_features": feats,
            "quality_score": float(quality),
            "explanation": explanation,
            "extraction_time_ms": float(dur_ms),
            "buffer_size": int(len(self.price_buffer)),
            "feature_count": int(feats.size),
            "instrument": self.symbol,
            "timeframe": self.primary_timeframe,
            "source": market_data.get("source", "unknown"),
            "ingest_n": int(ingest_n),
            "source_n": int(len(src_prices)),
        }

    def _ingest_prices(self, prices_in: List[float]) -> int:
        """
        Incrementally ingest prices into price_buffer to prevent duplication when upstream sends
        full history each cycle. Returns number of newly ingested points.
        """
        if not prices_in:
            return 0

        try:
            if len(self.price_buffer) == 0:
                self.price_buffer.extend(prices_in)
                return int(len(prices_in))

            last = float(self.price_buffer[-1])
            idx = self._find_last_match(prices_in, last, tol=self._ingest_tol)
            if idx is None:
                start = 1 if abs(float(prices_in[0]) - last) <= self._ingest_tol else 0
                new = prices_in[start:]
            else:
                new = prices_in[idx + 1:]

            if not new:
                return 0

            self.price_buffer.extend(new)
            return int(len(new))
        except Exception as e:
            self._audit_event(level="ERROR", event_type="ingest_prices_failed", error=str(e), trace=traceback.format_exc())
            # Fail closed: do not mutate buffer further.
            return 0

    def _find_last_match(self, arr: List[float], target: float, tol: float = 1e-3) -> Optional[int]:
        try:
            for i in range(len(arr) - 1, -1, -1):
                if abs(float(arr[i]) - float(target)) <= float(tol):
                    return i
            return None
        except Exception as e:
            self._audit_event(level="WARN", event_type="find_last_match_failed", error=str(e), trace=traceback.format_exc())
            return None

    def _extract_comprehensive_features(self, prices: List[float]) -> np.ndarray:
        if len(prices) < max(self.window_sizes):
            return self._get_fallback_features()

        arr = np.asarray(prices[-max(self.window_sizes):], dtype=float)
        feats: List[float] = []

        for w in self.window_sizes:
            window = arr[-w:]
            diffs = np.diff(window)
            up_ratio = float(np.mean(diffs > 0.0)) if diffs.size > 0 else 0.0
            ret = float((window[-1] - window[0]) / max(window[0], 1e-12))
            feats.extend([
                float(np.mean(window)),
                float(np.std(window)),
                float(ret),
                float(np.max(window) - np.min(window)),
                up_ratio,
                float(len(window)),
            ])

        diffs_all = np.diff(arr)
        pos_step_ratio = float(np.mean(diffs_all > 0.0)) if diffs_all.size > 0 else 0.0
        feats.extend([
            float(arr[-1]),
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.max(arr) - np.min(arr)),
            pos_step_ratio,
            float(len(arr)),
        ])

        out = np.asarray(feats, dtype=np.float32)
        if out.size != self.out_dim:
            self._audit_event(level="ERROR", event_type="feature_dim_mismatch", expected=int(self.out_dim), got=int(out.size))
            fixed = np.zeros(self.out_dim, dtype=np.float32)
            n = min(self.out_dim, out.size)
            fixed[:n] = out[:n]
            return fixed

        return out

    def _calculate_feature_quality(self, features: np.ndarray) -> float:
        try:
            if not isinstance(features, np.ndarray) or features.size == 0:
                return 0.0
            if np.any(~np.isfinite(features)):
                return 0.0

            L = max(0, (features.size - 6) // 6)
            if L > 0:
                perwin_std = [float(features[i * 6 + 1]) for i in range(L)]
                perwin_ret = [float(features[i * 6 + 2]) for i in range(L)]
                perwin_rng = [float(features[i * 6 + 3]) for i in range(L)]
                g_idx = L * 6
                # global layout: [last, mean, std, range, pos_step, len]
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

            span = float(np.max(features) - np.min(features))
            if span < 1e-8:
                return 35.0

            q = 85.0
            if float(np.max(np.abs(features))) > 1e6:
                q -= 15.0

            stdv = float(np.std(features))
            if 0.05 < stdv < 250.0:
                q += 5.0

            return float(np.clip(q, 0.0, 100.0))
        except Exception as e:
            self._audit_event(level="ERROR", event_type="quality_calc_failed", error=str(e), trace=traceback.format_exc())
            return 0.0

    def _generate_feature_explanation(self, feats: np.ndarray, quality: float) -> str:
        if not self.english_explainer:
            return "Feature extraction completed."
        try:
            analysis = {
                "instrument": self.symbol,
                "timeframe": self.primary_timeframe,
                "feature_count": int(feats.size),
                "quality_score": float(quality),
                "max_value": float(np.max(feats)),
                "min_value": float(np.min(feats)),
                "mean_value": float(np.mean(feats)),
                "std_value": float(np.std(feats)),
                "window_sizes": list(self.window_sizes),
                "buffer_size": int(len(self.price_buffer)),
            }
            return self.english_explainer.explain_module_decision(
                module_name="AdvancedFeatureEngine",
                decision="feature_extraction",
                context=analysis,
                confidence=float(quality) / 100.0
            )
        except Exception as e:
            self._audit_event(level="WARN", event_type="explanation_failed", error=str(e), trace=traceback.format_exc())
            return f"Feature extraction completed (explanation failed: {e})"

    async def _generate_feature_thesis(self, features: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        try:
            series = list(self.price_buffer)
            prices = series[-max(self.window_sizes):] if len(series) >= 2 else market_data.get("prices", [])

            latest = float(prices[-1]) if prices else 0.0
            change = float((prices[-1] - prices[0]) / max(prices[0], 1e-12)) if len(prices) > 1 else 0.0
            q = float(features.get("quality_score", 0.0))
            n = int(features.get("feature_count", 0))
            buf_util = f"{len(self.price_buffer)}/{self.max_buffer_size}"

            conf = "High" if q > 80 else ("Medium" if q > 60 else "Low")
            data_qual = "Good" if len(series) > 200 else ("Adequate" if len(series) > 60 else "Limited")

            return (
                f"Advanced Feature Analysis ({self.symbol} {self.primary_timeframe})\n"
                f"- Source: {market_data.get('source', 'unknown')}\n"
                f"- Latest price: {latest:.4f}\n"
                f"- Change: {change:.2%}\n"
                f"- Source points: {int(features.get('source_n', 0))}\n"
                f"- Ingested points: {int(features.get('ingest_n', 0))}\n\n"
                f"Quality\n"
                f"- Quality score: {q:.1f}/100\n"
                f"- Features: {n}\n"
                f"- Buffer: {buf_util}\n\n"
                f"Confidence: {conf} | Data: {data_qual}\n"
                f"Recommendation: {'Continue' if q > 60 else 'Investigate data feed quality/consistency'}"
            )
        except Exception as e:
            self._audit_event(level="ERROR", event_type="thesis_failed", error=str(e), trace=traceback.format_exc())
            return f"Feature extraction completed. Thesis generation error: {e}"

    # ─────────────────────────────────────────────────────────
    # Timeframe mirrors (XAUUSD-only)
    # ─────────────────────────────────────────────────────────
    def _compute_timeframe_outputs(self, market_data: Dict[str, Any], base_feats: np.ndarray) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}

        hist = self._bus_get("historical_prices", default=None, soft=True)
        mtd = self._bus_get("multi_timeframe_data", default=None, soft=True)

        base_payload = self._make_adv_payload(base_feats, quality=float(self.feature_quality_score), extraction_ms=0.0)
        base_payload.update({"instrument": self.symbol, "source": market_data.get("source", "unknown")})

        for tf in self.timeframes:
            if tf == self.primary_timeframe:
                out[tf] = {
                    **self._make_adv_payload(base_feats, quality=float(self.feature_quality_score), extraction_ms=float(0.0)),
                    "instrument": self.symbol,
                    "timeframe": tf,
                    "mtf_available": True
                }
                continue

            series = self._extract_from_historical_prices(hist, self.symbol, tf)
            if not series:
                series = self._extract_from_multi_timeframe_data(mtd, self.symbol, tf)

            if series:
                series = self._validate_prices(series)
                if len(series) >= max(self.window_sizes):
                    t0 = time.time()
                    feats = self._extract_comprehensive_features(series)
                    q = self._calculate_feature_quality(feats)
                    dur_ms = (time.time() - t0) * 1000.0
                    out[tf] = {
                        **self._make_adv_payload(feats, quality=float(q), extraction_ms=float(dur_ms)),
                        "instrument": self.symbol,
                        "timeframe": tf,
                        "mtf_available": True
                    }
                else:
                    out[tf] = {
                        **base_payload,
                        "timeframe": tf,
                        "alias_of": "advanced_features",
                        "mtf_available": True,
                        "note": "insufficient_points_for_tf"
                    }
            else:
                out[tf] = {
                    **base_payload,
                    "timeframe": tf,
                    "alias_of": "advanced_features",
                    "mtf_available": False,
                    "note": "tf_series_missing"
                }

        return out

    # ─────────────────────────────────────────────────────────
    # Bus I/O (declared keys only; failures are critical)
    # ─────────────────────────────────────────────────────────
    def _update_bus(self, features: Dict[str, Any], thesis: str, timeframe_outputs: Dict[str, Dict[str, Any]]):
        self._reset_last_bus_writes()

        adv_payload = self._make_adv_payload(
            features["raw_features"],
            quality=float(features["quality_score"]),
            extraction_ms=float(features["extraction_time_ms"])
        )
        adv_payload["instrument"] = self.symbol
        adv_payload["timeframe"] = self.primary_timeframe
        adv_payload["source"] = features.get("source", "unknown")
        adv_payload["bus_timestamp"] = self._sync_ctx.get("bus_timestamp")
        adv_payload["step_idx"] = self._sync_ctx.get("step_idx")

        self._bus_set("advanced_features", adv_payload, thesis=thesis)

        self._bus_set(
            "features",
            {
                "raw_features": adv_payload["raw_features"],
                "quality_score": float(features["quality_score"]),
                "instrument": self.symbol,
                "timeframe": self.primary_timeframe
            },
            thesis="Features alias for backward compatibility."
        )

        self._bus_set(
            "feature_analysis",
            {
                "instrument": self.symbol,
                "timeframe": self.primary_timeframe,
                "source": features.get("source", "unknown"),
                "explanation": features.get("explanation"),
                "buffer_status": self._buffer_status(),
                "statistics": dict(self.feature_stats),
                "bus_timestamp": self._sync_ctx.get("bus_timestamp"),
                "step_idx": self._sync_ctx.get("step_idx"),
            },
            thesis=f"Feature analysis: {float(features['quality_score']):.1f}% quality"
        )

        self._bus_set("feature_thesis", thesis, thesis="Feature engine thesis")
        self._bus_set("feature_engine_capabilities", self._capabilities_snapshot(), thesis="Capabilities snapshot")
        self._bus_set("feature_health", self._health_snapshot(), thesis="Feature engine health")
        self._bus_set("feature_error", None, thesis="No errors")

        self._bus_set("market_features", {"instrument": self.symbol, "timeframe": self.primary_timeframe}, thesis="Conservative placeholder")
        self._bus_set("price_features", {"instrument": self.symbol, "timeframe": self.primary_timeframe}, thesis="Conservative placeholder")

        for tf, payload in timeframe_outputs.items():
            key = f"advanced_features_{tf}"
            # Ensure sync fields exist in TF payload as well
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("instrument", self.symbol)
                payload.setdefault("timeframe", tf)
                payload.setdefault("bus_timestamp", self._sync_ctx.get("bus_timestamp"))
                payload.setdefault("step_idx", self._sync_ctx.get("step_idx"))
            self._bus_set(key, payload, thesis=f"Advanced features mirror ({self.symbol} {tf})")

    # ─────────────────────────────────────────────────────────
    # Stats / monitoring / breaker
    # ─────────────────────────────────────────────────────────
    def _update_feature_stats(self, extraction_time_ms: float, quality_score: float, success: bool):
        self.feature_stats["total_extractions"] += 1
        if success:
            self.feature_stats["successful_extractions"] += 1
        else:
            self.feature_stats["failed_extractions"] += 1

        total = max(1, self.feature_stats["total_extractions"])
        self.feature_stats["avg_extraction_time_ms"] = (
            (self.feature_stats["avg_extraction_time_ms"] * (total - 1) + float(extraction_time_ms)) / total
        )
        self.feature_stats["avg_feature_quality"] = (
            (self.feature_stats["avg_feature_quality"] * (total - 1) + float(quality_score)) / total
        )

    def _check_circuit_breaker(self) -> bool:
        state = self.circuit_breaker.get("state", "CLOSED")
        if state == "OPEN":
            lf = float(self.circuit_breaker.get("last_failure", 0.0) or 0.0)
            cooldown = float(self.circuit_breaker.get("cooldown_s", 60.0))
            if lf > 0.0 and (time.time() - lf) > cooldown:
                self.circuit_breaker["state"] = "HALF_OPEN"
                return True
            return False
        return True

    def _record_success(self, processing_time_s: float):
        if self.circuit_breaker["state"] == "HALF_OPEN":
            self.circuit_breaker["state"] = "CLOSED"
            self.circuit_breaker["failures"] = 0

        self.health_metrics["health_score"] = min(100.0, float(self.health_metrics.get("health_score", 100.0)) + 1.0)
        self.health_metrics["performance_trend"] = "improving"

        if self.performance_tracker:
            try:
                self.performance_tracker.record_metric("AdvancedFeatureEngine", "feature_extraction", float(processing_time_s) * 1000.0, True)
            except Exception as e:
                self._audit_event(level="WARN", event_type="performance_tracker_failed", error=str(e), trace=traceback.format_exc())

    def _record_failure(self, error: Exception):
        self.circuit_breaker["failures"] = int(self.circuit_breaker.get("failures", 0)) + 1
        self.circuit_breaker["last_failure"] = time.time()
        if int(self.circuit_breaker["failures"]) >= int(self.circuit_breaker["threshold"]):
            self.circuit_breaker["state"] = "OPEN"
            self.logger.error(
                format_operator_message(
                    "[ALERT]", "CIRCUIT_BREAKER_OPEN",
                    details=f"Too many failures ({self.circuit_breaker['failures']})",
                    context="circuit_breaker"
                )
            )

        self.health_metrics["health_score"] = max(0.0, float(self.health_metrics.get("health_score", 100.0)) - 10.0)
        issues = list(self.health_metrics.get("issues_detected", []))
        issues.append(f"{type(error).__name__}: {error}")
        self.health_metrics["issues_detected"] = issues
        self.health_metrics["performance_trend"] = "degrading"

    def _get_fallback_features(self) -> np.ndarray:
        if len(self.feature_buffer) > 0:
            try:
                last = self.feature_buffer[-1]
                feats = last.get("features")
                if isinstance(feats, np.ndarray) and feats.size == self.out_dim:
                    return feats
            except Exception as e:
                self._audit_event(level="WARN", event_type="fallback_from_feature_buffer_failed", error=str(e), trace=traceback.format_exc())
        return np.zeros(self.out_dim, dtype=np.float32)

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

        adv = {
            "raw_features": list(fp.get("raw_features", [])),
            "quality_score": float(fp.get("quality_score", 0.0)),
            "extraction_time_ms": float(fp.get("extraction_time_ms", 0.0)),
            "timestamp": float(fp.get("timestamp", time.time())),
            "instrument": self.symbol,
            "timeframe": self.primary_timeframe,
            "source": fp.get("source", "unknown"),
        }

        # Sync fields are optional but improve cross-module timeline alignment
        if "bus_timestamp" in fp:
            adv["bus_timestamp"] = fp.get("bus_timestamp")
        if "step_idx" in fp:
            adv["step_idx"] = fp.get("step_idx")

        out["advanced_features"] = adv
        out["features"] = {"raw_features": adv["raw_features"], "quality_score": adv["quality_score"], "instrument": self.symbol, "timeframe": self.primary_timeframe}

        ana = analysis if isinstance(analysis, dict) else {}
        out["feature_analysis"] = {
            "instrument": self.symbol,
            "timeframe": self.primary_timeframe,
            "explanation": ana.get("explanation"),
            "statistics": ana.get("statistics", {}),
            "buffer_status": ana.get("buffer_status", {}),
            "source": ana.get("source", fp.get("source", "unknown")),
        }

        ft = thesis or f"Advanced feature extraction completed ({self.symbol} {self.primary_timeframe})."
        out["feature_thesis"] = ft
        out["_thesis"] = ft  # orchestrator-required

        if extra and isinstance(extra, dict):
            out.update(extra)

        out["feature_engine_capabilities"] = self._capabilities_snapshot()
        out["feature_health"] = self._health_snapshot()

        # Consistent error contract: dict or None
        out["feature_error"] = out.get("feature_error", None)

        out.setdefault("market_features", {"instrument": self.symbol, "timeframe": self.primary_timeframe})
        out.setdefault("price_features", {"instrument": self.symbol, "timeframe": self.primary_timeframe})

        tf_map = timeframe_outputs or {}
        for tf in self.timeframes:
            k = f"advanced_features_{tf}"
            payload = tf_map.get(tf)
            if not isinstance(payload, dict):
                payload = {**adv, "timeframe": tf, "alias_of": "advanced_features"}
            out[k] = payload

        for k in ("advanced_features", "features", "feature_analysis", "feature_thesis", "feature_engine_capabilities"):
            if k not in out:
                raise ValueError(f"Critical output '{k}' missing in AdvancedFeatureEngine")

        return out

    # ─────────────────────────────────────────────────────────
    # Health / performance loops
    # ─────────────────────────────────────────────────────────
    async def _health_monitoring_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                self._update_health_metrics()
                self._check_health_issues()
                self._audit_event(level="INFO" if self.debug else "DEBUG", event_type="health_tick", health=self._health_snapshot())
            except Exception as e:
                self._audit_event(level="ERROR", event_type="health_loop_failed", error=str(e), trace=traceback.format_exc())

    def _update_health_metrics(self):
        total = int(self.feature_stats.get("total_extractions", 0))
        succ = int(self.feature_stats.get("successful_extractions", 0))
        success_rate = float(succ) / max(1, total)
        if success_rate > 0.95:
            self.health_metrics["health_score"] = min(100.0, float(self.health_metrics.get("health_score", 100.0)) + 0.5)
        elif success_rate < 0.8:
            self.health_metrics["health_score"] = max(0.0, float(self.health_metrics.get("health_score", 100.0)) - 1.0)
        self.health_metrics["last_health_check"] = time.time()

    def _check_health_issues(self):
        issues: List[str] = []
        if self.circuit_breaker.get("state") == "OPEN":
            issues.append("Circuit breaker OPEN")
        buf_util = len(self.price_buffer) / max(1, self.max_buffer_size)
        if buf_util > 0.9:
            issues.append("Price buffer > 90%")
        if float(self.feature_stats.get("avg_extraction_time_ms", 0.0)) > 100.0:
            issues.append("Avg extraction time > 100ms")
        if float(self.feature_stats.get("avg_feature_quality", 0.0)) < 60.0 and int(self.feature_stats.get("total_extractions", 0)) > 20:
            issues.append("Avg feature quality < 60")
        self.health_metrics["issues_detected"] = issues
        if issues:
            self.logger.warning(format_operator_message("[WARN]", "HEALTH_ISSUES_DETECTED", details="; ".join(issues), context="health_monitoring"))

    async def _performance_monitoring_loop(self):
        while True:
            try:
                await asyncio.sleep(60)
                if self.performance_tracker:
                    self.performance_tracker.record_metric("AdvancedFeatureEngine", "periodic_metrics_tick", 1.0, True)
            except Exception as e:
                self._audit_event(level="ERROR", event_type="performance_loop_failed", error=str(e), trace=traceback.format_exc())

    # ─────────────────────────────────────────────────────────
    # State API
    # ─────────────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        return {
            **base,
            "config": {
                "symbol": self.symbol,
                "primary_timeframe": self.primary_timeframe,
                "timeframes": list(self.timeframes),
                "window_sizes": list(self.window_sizes),
                "max_buffer_size": int(self.max_buffer_size),
                "out_dim": int(self.out_dim),
                "debug": bool(self.debug),
                "ingest_match_tol": float(self._ingest_tol),
            },
            "buffers": {
                "price_buffer_size": len(self.price_buffer),
                "feature_buffer_size": len(self.feature_buffer),
            },
            "features": {
                "last_features": self.last_features.tolist(),
                "quality_score": float(self.feature_quality_score),
            },
            "statistics": dict(self.feature_stats),
            "health_metrics": dict(self.health_metrics),
            "circuit_breaker": dict(self.circuit_breaker),
        }

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)

    def get_health_status(self) -> Dict[str, Any]:
        return self._health_snapshot()

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────
    def _capabilities_snapshot(self) -> Dict[str, Any]:
        return {
            "instrument": self.symbol,
            "primary_timeframe": self.primary_timeframe,
            "timeframes": list(self.timeframes),
            "window_sizes": list(self.window_sizes),
            "out_dim": int(self.out_dim),
            "max_buffer_size": int(self.max_buffer_size),
            "supports_explainability": bool(self.english_explainer is not None),
            "supports_error_pinpointing": bool(self.error_pinpointer is not None),
            "supports_performance_tracking": bool(self.performance_tracker is not None),
            "debug": bool(self.debug),
        }

    def _health_snapshot(self) -> Dict[str, Any]:
        return {
            "instrument": self.symbol,
            "timeframe": self.primary_timeframe,
            "health_score": float(self.health_metrics.get("health_score", 0.0)),
            "performance_trend": self.health_metrics.get("performance_trend", "unknown"),
            "issues_detected": list(self.health_metrics.get("issues_detected", [])),
            "statistics": dict(self.feature_stats),
            "breaker": dict(self.circuit_breaker),
            "buffer_status": self._buffer_status(),
        }

    def _buffer_status(self) -> Dict[str, Any]:
        return {
            "current_size": int(len(self.price_buffer)),
            "max_size": int(self.max_buffer_size),
            "utilization": float(len(self.price_buffer) / max(1, self.max_buffer_size)),
        }

    def _make_adv_payload(self, feats: Union[np.ndarray, List[float]], quality: float, extraction_ms: float) -> Dict[str, Any]:
        if isinstance(feats, np.ndarray):
            raw = feats.astype(np.float32, copy=False).tolist()
            h = self._hash_bytes(feats.tobytes())
        else:
            raw = list(feats)
            h = self._hash_bytes(np.asarray(raw, dtype=np.float32).tobytes()) if raw else None
        return {
            "raw_features": raw,
            "quality_score": float(quality),
            "extraction_time_ms": float(extraction_ms),
            "timestamp": time.time(),
            "features_hash": h,
        }

    def _hash_bytes(self, b: bytes) -> str:
        return hashlib.blake2b(b, digest_size=16).hexdigest()

    def _summarize_prices(self, prices: List[float]) -> Dict[str, Any]:
        if not prices:
            return {"n": 0}
        arr = np.asarray(prices, dtype=float)
        head_n = min(self._preview_n, arr.size)
        tail_n = min(self._preview_n, arr.size)
        out = {
            "n": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if arr.size > 1 else 0.0,
            "head": arr[:head_n].tolist(),
            "tail": arr[-tail_n:].tolist(),
            "hash": self._hash_bytes(arr.astype(np.float32).tobytes()),
        }
        if self._full_prices:
            out["full"] = arr.tolist()
        return out

    def _summarize_features(self, feats: np.ndarray) -> Dict[str, Any]:
        if not isinstance(feats, np.ndarray) or feats.size == 0:
            return {"n": 0}
        arr = feats.astype(np.float32, copy=False)
        head_n = min(self._preview_n, arr.size)
        tail_n = min(self._preview_n, arr.size)
        out = {
            "n": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if arr.size > 1 else 0.0,
            "head": arr[:head_n].tolist(),
            "tail": arr[-tail_n:].tolist(),
            "hash": self._hash_bytes(arr.tobytes()),
        }
        if self._full_features:
            out["full"] = arr.tolist()
        return out

    # ─────────────────────────────────────────────────────────
    # Audit logging (ONE FILE)
    # ─────────────────────────────────────────────────────────
    def _init_audit_logger(self) -> Optional[logging.Logger]:
        if not bool(self._cfg.audit_log_enabled):
            return None
        try:
            path = str(self._cfg.audit_log_path)
            log_dir = os.path.dirname(path) or "."
            os.makedirs(log_dir, exist_ok=True)

            lg = logging.getLogger("audit.AdvancedFeatureEngine")
            lg.setLevel(logging.INFO)
            lg.propagate = False

            abs_path = os.path.abspath(path)
            have = any(
                isinstance(h, RotatingFileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == abs_path
                for h in lg.handlers
            )
            if not have:
                h = RotatingFileHandler(
                    path,
                    maxBytes=int(self._cfg.audit_max_bytes),
                    backupCount=int(self._cfg.audit_backup_count),
                    encoding="utf-8",
                )
                h.setFormatter(logging.Formatter("%(message)s"))
                lg.addHandler(h)

            return lg
        except Exception as e:
            self.logger.error(format_operator_message("[AUDIT]", "AUDIT_LOG_INIT_FAILED", details=str(e), context="audit"))
            return None

    def _audit_event(self, level: str, event_type: str, **fields: Any) -> None:
        try:
            # Merge sync context into every event for cross-module alignment
            sync = dict(getattr(self, "_sync_ctx", {}) or {})
            evt = {
                "ts": time.time(),
                "level": str(level).upper(),
                "module": "AdvancedFeatureEngine",
                "event": str(event_type),
                "instrument": getattr(self, "symbol", "XAUUSD"),
                "primary_timeframe": getattr(self, "primary_timeframe", "M15"),
                "debug": bool(getattr(self, "debug", False)),
                **sync,
                **fields,
            }
            line = json.dumps(evt, ensure_ascii=False, separators=(",", ":"))
            logger = getattr(self, "_audit_logger", None)
            if logger is not None and hasattr(logger, "info"):
                logger.info(line)
            else:
                self.logger.warning(f"[AUDIT_FALLBACK] {line}")
        except Exception:
            # Last resort: never raise from audit.
            return

    def _audit_cycle(
        self,
        *,
        level: str,
        cycle_id: int,
        step_idx: Any,
        bus_timestamp: Any,
        start_ts: float,
        market_data: Optional[Dict[str, Any]],
        features_payload: Optional[Dict[str, Any]],
        thesis: Optional[str],
        tf_outputs: Dict[str, Dict[str, Any]],
        bus_writes: List[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None,
        trace: Optional[str] = None,
        pinpointer: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            md_audit = market_data.get("audit") if isinstance(market_data, dict) else None
            prices_summary = None
            if isinstance(market_data, dict) and isinstance(market_data.get("prices"), list):
                prices_summary = self._summarize_prices(market_data["prices"])

            feats_summary = None
            if isinstance(features_payload, dict) and isinstance(features_payload.get("raw_features"), np.ndarray):
                feats_summary = self._summarize_features(features_payload["raw_features"])

            tf_summary: Dict[str, Any] = {}
            for tf, p in (tf_outputs or {}).items():
                try:
                    rf = p.get("raw_features", [])
                    q = float(p.get("quality_score", 0.0))
                    tf_summary[tf] = {
                        "quality_score": q,
                        "alias_of": p.get("alias_of"),
                        "mtf_available": p.get("mtf_available"),
                        "feature_count": int(len(rf)) if isinstance(rf, list) else None
                    }
                    if self.debug and isinstance(rf, list) and self._full_features:
                        tf_summary[tf]["raw_features_full"] = rf
                except Exception as te:
                    tf_summary[tf] = {"error": f"tf_summary_failed: {te}"}

            self._audit_event(
                level=level,
                event_type="cycle",
                cycle_id=int(cycle_id),
                step_idx=step_idx,
                bus_timestamp=bus_timestamp,
                duration_ms=float((time.time() - start_ts) * 1000.0),
                success=bool(success),
                source=(market_data.get("source") if isinstance(market_data, dict) else None),
                prices_summary=prices_summary,
                market_data_audit=md_audit,
                features_summary=feats_summary,
                quality_score=(float(features_payload.get("quality_score", 0.0)) if isinstance(features_payload, dict) else None),
                extraction_time_ms=(float(features_payload.get("extraction_time_ms", 0.0)) if isinstance(features_payload, dict) else None),
                thesis=thesis if (self.debug or not thesis) else thesis[:4000],
                timeframes=tf_summary,
                bus_writes=bus_writes,
                error=error,
                trace=(trace if self.debug else None),
                pinpointer=(pinpointer if self.debug else None),
            )
        except Exception as e:
            self._audit_event(level="ERROR", event_type="audit_cycle_failed", error=str(e), trace=traceback.format_exc())

    # ─────────────────────────────────────────────────────────
    # Bus wrappers (audited, no silent failures)
    # ─────────────────────────────────────────────────────────
    def _bus_get(self, key: str, default: Any = None, soft: bool = False, audit: Optional[Dict[str, Any]] = None) -> Any:
        try:
            v = self.smart_bus.get(key, self.__class__.__name__)
            if audit is not None:
                audit.setdefault("bus_reads", []).append(self._summarize_bus_value(key, v))
            return v
        except Exception as e:
            if audit is not None:
                audit.setdefault("bus_reads", []).append({"key": key, "ok": False, "error": str(e)})
            self._audit_event(level="WARN" if soft else "ERROR", event_type="bus_get_failed", key=key, error=str(e), trace=(traceback.format_exc() if self.debug else None))
            if soft:
                return default
            raise

    def _bus_set(self, key: str, value: Any, thesis: str) -> None:
        try:
            with self._bus_lock:
                self.smart_bus.set(key, value, module="AdvancedFeatureEngine", thesis=thesis)
                self._track_bus_write(key, value, thesis)
        except Exception as e:
            self._audit_event(level="ERROR", event_type="bus_set_failed", key=key, error=str(e), trace=traceback.format_exc())
            raise

    def _summarize_bus_value(self, key: str, v: Any) -> Dict[str, Any]:
        try:
            summary: Dict[str, Any] = {"key": key, "ok": True, "type": type(v).__name__}
            if self.debug:
                if isinstance(v, dict):
                    summary["n_keys"] = len(v)
                    summary["keys_head"] = list(v.keys())[:50]
                elif isinstance(v, (list, tuple)):
                    summary["n"] = len(v)
                    summary["head_types"] = [type(x).__name__ for x in list(v[:min(10, len(v))])]
                elif isinstance(v, np.ndarray):
                    summary["shape"] = list(v.shape)
                    summary["dtype"] = str(v.dtype)
            return summary
        except Exception as e:
            return {"key": key, "ok": True, "type": "unknown", "summary_error": str(e)}

    def _reset_last_bus_writes(self):
        self._last_bus_writes = []

    def _track_bus_write(self, key: str, value: Any, thesis: str):
        try:
            if not hasattr(self, "_last_bus_writes") or self._last_bus_writes is None:
                self._last_bus_writes = []

            item: Dict[str, Any] = {"key": key, "thesis": thesis}
            if self.debug:
                item["type"] = type(value).__name__
                if isinstance(value, dict):
                    item["n_keys"] = len(value)
                    item["keys_head"] = list(value.keys())[:50]
                elif isinstance(value, list):
                    item["n"] = len(value)

            self._last_bus_writes.append(item)
        except Exception as e:
            self._audit_event(level="WARN", event_type="track_bus_write_failed", key=key, error=str(e), trace=traceback.format_exc())

    def _last_bus_writes_snapshot(self) -> List[Dict[str, Any]]:
        try:
            return list(self._last_bus_writes or [])
        except Exception as e:
            self._audit_event(level="WARN", event_type="last_bus_writes_snapshot_failed", error=str(e), trace=traceback.format_exc())
            return []
