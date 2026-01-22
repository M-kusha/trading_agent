# ─────────────────────────────────────────────────────────────
# File: modules/features/multiscale_feature_engine.py
# MultiScale Feature Engine (XAUUSD-only, Contract-Clean, Orchestration-Safe, Hard-Audit Logging)
#
# Goals (robustness + correctness hardening):
# - HARD scope: XAUUSD only (no multi-symbol blending, no EURUSD references)
# - Contract-clean outputs + orchestration-safe bus writes (single-writer lock)
# - Deterministic fallbacks (NO random vectors) to avoid train/live drift
# - Uses real per-timeframe mirrors when available: advanced_features_{M15,H1,H4,D1}
# - Hot-swap networks if upstream feature dimension changes
# - Circuit breaker with cooldown + health/perf monitoring loops (hot-reload safe)
# - JSONL rotating audit log (single file) with cycle-level snapshots
# - No silent exception swallowing in meaningful paths
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import json
import time
import asyncio
import threading
import traceback
import logging
from logging.handlers import RotatingFileHandler

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# Core infrastructure
from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker

# Optional dependency (do NOT instantiate by default to avoid duplicate writers)
try:
    from modules.features.advanced_feature_engine import AdvancedFeatureEngine  # noqa: F401
except Exception:
    AdvancedFeatureEngine = None  # type: ignore

if TYPE_CHECKING:
    from modules.features.advanced_feature_engine import AdvancedFeatureEngine as AFEType


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass
class MultiScaleConfig:
    # HARD instrument scope
    symbol: str = "XAUUSD"
    primary_timeframe: str = "M15"
    timeframes: Optional[List[str]] = None  # ["M15","H1","H4","D1"]

    # Model geometry
    embed_dim: int = 64
    num_attention_heads: int = 4
    dropout_rate: float = 0.10
    enable_gpu: bool = True
    feature_fusion_method: str = "attention"  # "attention" | "concat" | "weighted" (attention default)

    # If AFE isn't injected and Bus doesn't yet have features, we need a deterministic dim:
    assumed_input_dim: int = 256

    # Circuit breaker
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_s: float = 120.0

    # Monitoring knobs
    enable_health_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_error_pinpointing: bool = True
    enable_english_explanations: bool = True

    # Audit logging (ONE FILE)
    audit_log_enabled: bool = True
    audit_log_path: str = "logs/audit/multiscale_feature_engine.log.jsonl"
    audit_max_bytes: int = 50_000_000  # 50MB
    audit_backup_count: int = 5

    # Verbosity control
    debug: bool = True
    debug_preview_n: int = 8
    debug_full_vectors: bool = False
    debug_full_attention: bool = False

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["M15", "H1", "H4", "D1"]


# ─────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────
class AttentionFeatureFusion(nn.Module):
    """
    Self-attention over timeframe embeddings (x: [B, T, E]) + residual MLP block.
    Returns:
      - h: [B, T, E]
      - attn_w: [B, T, T]
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(x, x, x, need_weights=True)  # [B,T,E], [B,T,T]
        h = self.norm1(x + attn_out)
        h2 = self.out(h)
        h = self.norm2(h + h2)
        return h, attn_w


# ─────────────────────────────────────────────────────────────
# Module declaration
# ─────────────────────────────────────────────────────────────
@module(**module_args(
    "MultiScaleFeatureEngine",
    description="XAUUSD-only multiscale fusion over AdvancedFeatureEngine mirrors; produces multiscale_features, embeddings, attention, neural health/capabilities.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class MultiScaleFeatureEngine(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Contract (registry):
      Provides: attention_weights, feature_fusion, multiscale_features, neural_capabilities, neural_embeddings, neural_health
      Requires: advanced_features, market_data

    Notes:
    - This module is a READER of advanced_features (and its per-TF mirrors) by default.
      It will not instantiate AdvancedFeatureEngine internally to avoid duplicate writers.
    """

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        config: Optional[Union[MultiScaleConfig, Dict[str, Any]]] = None,
        afe: Optional["AFEType"] = None,  # optional injection; no default creation
        **kwargs
    ):
        # Normalize config → dataclass
        if isinstance(config, dict):
            filtered = {k: config[k] for k in MultiScaleConfig.__dataclass_fields__ if k in config}
            self._cfg = MultiScaleConfig(**filtered)
        elif isinstance(config, MultiScaleConfig) or config is None:
            self._cfg = config or MultiScaleConfig()
        else:
            self._cfg = MultiScaleConfig()

        # Ensure debug knobs exist early (audit paths must not explode).
        self.debug = bool(self._cfg.debug)
        self._preview_n = int(self._cfg.debug_preview_n)
        self._full_vec = bool(self._cfg.debug_full_vectors) if self.debug else False
        self._full_attn = bool(self._cfg.debug_full_attention) if self.debug else False

        # Optional injected AFE
        self._afe = afe

        super().__init__(config=asdict(self._cfg), **kwargs)

    def _initialize(self):
        # Core handles
        self.smart_bus = InfoBusManager.get_instance()
        self._bus_lock = threading.Lock()
        self._cycle_idx = 0

        # HARD scope
        self.symbol = str(self._cfg.symbol or "XAUUSD").upper()
        self.primary_timeframe = str(self._cfg.primary_timeframe or "M15").upper()
        self.timeframes = [str(x).upper() for x in (self._cfg.timeframes or ["M15", "H1", "H4", "D1"])]

        # Device
        self.device = torch.device("cuda" if (torch.cuda.is_available() and self._cfg.enable_gpu) else "cpu")

        # Determine input dimension
        self.input_dim = self._discover_input_dim()
        self.output_dim = int(self._cfg.embed_dim)

        # Subsystems
        if self._cfg.enable_error_pinpointing:
            self.error_pinpointer = ErrorPinpointer()
            self.error_handler = create_error_handler("MultiScaleFeatureEngine", self.error_pinpointer)
        else:
            self.error_pinpointer = None
            self.error_handler = None

        if self._cfg.enable_english_explanations:
            self.english_explainer = EnglishExplainer()
            self.system_utilities = SystemUtilities()
        else:
            self.english_explainer = None
            self.system_utilities = None

        self.performance_tracker = PerformanceTracker() if self._cfg.enable_performance_tracking else None

        # Circuit breaker
        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",  # CLOSED | HALF_OPEN | OPEN
            "threshold": int(self._cfg.circuit_breaker_threshold),
            "cooldown_s": float(self._cfg.circuit_breaker_cooldown_s),
        }

        # State
        self._initialize_state()

        # Audit logger
        self._audit_logger = self._init_audit_logger()

        # Verbosity re-assert
        self.debug = bool(self._cfg.debug)
        self._preview_n = int(self._cfg.debug_preview_n)
        self._full_vec = bool(self._cfg.debug_full_vectors) if self.debug else False
        self._full_attn = bool(self._cfg.debug_full_attention) if self.debug else False

        # Build networks
        self._build_networks(self.input_dim)

        # Background monitoring
        self._start_monitoring()

        # Operator log
        self.logger.info(
            format_operator_message(
                "🧠", "MULTISCALE_READY",
                details=f"Symbol={self.symbol}, TFs={self.timeframes}, InputDim={self.input_dim}, EmbedDim={self.output_dim}, Device={self.device}, Debug={self.debug}",
                result="ready",
                context="multiscale_startup"
            )
        )

        # Publish baseline to avoid BUS MISS (best-effort)
        try:
            baseline = self._baseline_outputs(reason="baseline_init")
            self._update_bus_from_outputs(baseline, thesis="Baseline published at init to avoid BUS MISS.")
            self._audit_event(level="INFO" if self.debug else "DEBUG", event_type="init_baseline_published")
        except Exception as e:
            self._audit_event(level="ERROR", event_type="init_baseline_publish_failed", error=str(e), trace=traceback.format_exc())

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

    # ─────────────────────────────────────────────────────────
    # State init
    # ─────────────────────────────────────────────────────────
    def _initialize_state(self):
        self.last_embedding = np.zeros(self.output_dim, dtype=np.float32)
        self.last_attention = np.zeros((1, len(self.timeframes), len(self.timeframes)), dtype=np.float32)
        self.attention_weights_history = deque(maxlen=100)
        self.embedding_history = deque(maxlen=500)

        self._last_bus_writes: List[Dict[str, Any]] = []

        self.neural_stats: Dict[str, Any] = {
            "total_forward_passes": 0,
            "successful_passes": 0,
            "failed_passes": 0,
            "avg_forward_time_ms": 0.0,
            "avg_attention_entropy": 0.0,
            "gpu_memory_usage_mb": 0.0,
            "last_success_ts": 0.0,
            "last_failure_ts": 0.0,
        }

        self.neural_health: Dict[str, Any] = {
            "model_health_score": 100.0,
            "attention_quality": 100.0,
            "embedding_quality": 100.0,
            "performance_trend": "stable",
            "issues_detected": [],
            "last_neural_check": time.time(),
        }

    # ─────────────────────────────────────────────────────────
    # Monitoring
    # ─────────────────────────────────────────────────────────
    def _start_monitoring(self):
        if not (self._cfg.enable_health_monitoring or self._cfg.enable_performance_tracking):
            return

        def _schedule(loop: asyncio.AbstractEventLoop) -> None:
            if self._cfg.enable_health_monitoring:
                loop.create_task(self._neural_health_monitoring_loop())
            if self._cfg.enable_performance_tracking:
                loop.create_task(self._gpu_monitoring_loop())

        # Try running loop first
        try:
            loop = asyncio.get_running_loop()
            _schedule(loop)
            self._audit_event(level="INFO" if self.debug else "DEBUG", event_type="monitoring_tasks_scheduled", mode="running_loop")
            return
        except RuntimeError:
            pass

        # No running loop: start background loop once
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

        self._bg_thread = threading.Thread(target=_runner, name="MultiScaleFeatureEngineMonitoring", daemon=True)
        self._bg_thread.start()
        self._audit_event(level="INFO" if self.debug else "DEBUG", event_type="monitoring_tasks_scheduled", mode="bg_thread_loop")

    async def _neural_health_monitoring_loop(self):
        while True:
            try:
                await asyncio.sleep(60)
                self._update_health_metrics()
                self._check_health_issues()
                self.neural_health["last_neural_check"] = time.time()
                self._audit_event(level="INFO" if self.debug else "DEBUG", event_type="health_tick", health=self._health_snapshot())
            except Exception as e:
                self._audit_event(level="ERROR", event_type="health_loop_failed", error=str(e), trace=traceback.format_exc())

    async def _gpu_monitoring_loop(self):
        # GPU metrics are optional; still run loop for periodic perf stats
        while True:
            try:
                await asyncio.sleep(30)
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1024.0 / 1024.0
                    self.neural_stats["gpu_memory_usage_mb"] = float(mem_alloc)
                    if mem_alloc > 1500.0:
                        self.logger.warning(
                            format_operator_message(
                                "🖥️[WARN]", "HIGH_GPU_MEMORY_USAGE",
                                details=f"Allocated: {mem_alloc:.1f}MB",
                                context="gpu_monitoring"
                            )
                        )
            except Exception as e:
                self._audit_event(level="ERROR", event_type="gpu_loop_failed", error=str(e), trace=traceback.format_exc())

    def _update_health_metrics(self):
        total = int(self.neural_stats.get("total_forward_passes", 0))
        succ = int(self.neural_stats.get("successful_passes", 0))
        sr = float(succ) / max(1, total)

        hs = float(self.neural_health.get("model_health_score", 100.0))
        if sr > 0.95:
            hs = min(100.0, hs + 1.0)
            self.neural_health["performance_trend"] = "improving"
        elif sr < 0.8 and total > 10:
            hs = max(0.0, hs - 2.0)
            self.neural_health["performance_trend"] = "degrading"
        else:
            self.neural_health["performance_trend"] = self.neural_health.get("performance_trend", "stable")

        self.neural_health["model_health_score"] = float(hs)

    def _check_health_issues(self):
        issues: List[str] = []
        if self.circuit_breaker.get("state") == "OPEN":
            issues.append("Circuit breaker OPEN")
        if float(self.neural_stats.get("avg_forward_time_ms", 0.0)) > 50.0:
            issues.append("Avg forward time > 50ms")
        if float(self.neural_stats.get("avg_attention_entropy", 0.0)) < 0.1 and int(self.neural_stats.get("total_forward_passes", 0)) > 20:
            issues.append("Attention entropy extremely low")
        self.neural_health["issues_detected"] = issues
        if issues:
            self.logger.warning(
                format_operator_message("[WARN]", "NEURAL_HEALTH_ISSUES", details="; ".join(issues), context="neural_health")
            )

    # ─────────────────────────────────────────────────────────
    # Networks
    # ─────────────────────────────────────────────────────────
    def _build_networks(self, input_dim: int):
        tfs = list(self.timeframes)
        E = int(self._cfg.embed_dim)
        dr = float(self._cfg.dropout_rate)

        self.scale_processors = nn.ModuleDict({
            tf: nn.Sequential(
                nn.Linear(input_dim, E),
                nn.ReLU(),
                nn.LayerNorm(E),
                nn.Dropout(dr),
            ) for tf in tfs
        })

        fusion_input_dim = E * len(tfs)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, E * 2),
            nn.ReLU(),
            nn.LayerNorm(E * 2),
            nn.Dropout(dr),
            nn.Linear(E * 2, E),
            nn.ReLU(),
            nn.Linear(E, E),
        )

        self.attention_fusion = AttentionFeatureFusion(embed_dim=E, num_heads=int(self._cfg.num_attention_heads), dropout=dr)

        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.scale_processors.apply(init_layer)
        self.fusion_network.apply(init_layer)
        self.attention_fusion.apply(init_layer)

        self.scale_processors.to(self.device)
        self.fusion_network.to(self.device)
        self.attention_fusion.to(self.device)

    def _maybe_rebuild_for(self, new_dim: int):
        if int(new_dim) != int(self.input_dim) and int(new_dim) > 0:
            self.logger.info(
                format_operator_message(
                    "🧩", "INPUT_DIM_CHANGE",
                    details=f"{self.input_dim} → {new_dim}; rebuilding networks",
                    result="rebuild",
                    context="model_hot_swap"
                )
            )
            self.input_dim = int(new_dim)
            self._build_networks(self.input_dim)

    # ─────────────────────────────────────────────────────────
    # Main processing
    # ─────────────────────────────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        start_ts = time.time()
        self._cycle_idx += 1
        cycle_id = self._cycle_idx

        step_idx = self._bus_get("step_idx", default=None, soft=True)
        bus_ts = self._bus_get("timestamp", default=None, soft=True)

        # Circuit breaker gating
        if not self._check_circuit_breaker():
            out = self._baseline_outputs(reason="circuit_breaker_open")
            thesis = "Neural circuit breaker OPEN: returning fallback embeddings/attention."
            self._audit_cycle(
                level="WARN",
                cycle_id=cycle_id,
                step_idx=step_idx,
                bus_timestamp=bus_ts,
                start_ts=start_ts,
                success=False,
                thesis=thesis,
                error="circuit_breaker_open",
                extra={"breaker": dict(self.circuit_breaker)},
            )
            return self._format_declared_outputs(
                outputs=out,
                thesis=thesis,
                extra={"success": False, "reason": "circuit_breaker_open", "processing_time_ms": 0.0, "neural_error": {"error": "circuit_breaker_open"}},
            )

        try:
            market_data = self._get_market_data(**inputs)
            self._enforce_xauusd_scope(market_data=market_data, afe_payload=None)

            # Read AFE vectors (base + per-TF mirrors when present)
            afe_bundle = await self._get_afe_bundle(**inputs)
            self._enforce_xauusd_scope(market_data=market_data, afe_payload=afe_bundle.get("base_payload"))

            # Hot-swap if needed
            base_vec = np.asarray(afe_bundle["base_vec"], dtype=np.float32).reshape(-1)
            if base_vec.size > 0:
                self._maybe_rebuild_for(int(base_vec.size))

            # Build multiscale features (per-tf padded to input_dim)
            ms_result = self._build_multiscale_features(afe_bundle, market_data)

            # Neural forward
            nn_result = await self._neural_forward(ms_result)

            # Thesis
            thesis = await self._generate_neural_thesis(afe_bundle, ms_result, nn_result, market_data)

            # Update stats/health
            elapsed_ms = (time.time() - start_ts) * 1000.0
            self._record_success(forward_time_ms=float(nn_result.get("forward_time_ms", 0.0)))

            # Publish declared keys ONLY (single-writer)
            out = self._package_outputs(ms_result, nn_result)
            self._update_bus_from_outputs(out, thesis=thesis)

            # Audit cycle
            self._audit_cycle(
                level="INFO",
                cycle_id=cycle_id,
                step_idx=step_idx,
                bus_timestamp=bus_ts,
                start_ts=start_ts,
                success=True,
                thesis=thesis,
                extra={
                    "processing_time_ms": float(elapsed_ms),
                    "market_fields_seen": int(ms_result.get("market_fields_seen", 0)),
                    "base_dim": int(base_vec.size),
                    "timeframes_used": list(ms_result.get("timeframes_used", [])),
                    "bus_writes": self._last_bus_writes_snapshot(),
                },
                ms_result=ms_result,
                nn_result=nn_result,
                afe_bundle=afe_bundle,
            )

            return self._format_declared_outputs(
                outputs=out,
                thesis=thesis,
                extra={
                    "success": True,
                    "processing_time_ms": float(elapsed_ms),
                    "device_used": str(self.device),
                    "attention_entropy": float(nn_result.get("attention_entropy", 0.0)),
                    "market_fields_seen": int(ms_result.get("market_fields_seen", 0)),
                    "neural_error": None,
                },
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_ts) * 1000.0
            self._record_failure(e)

            # Pinpointer (best-effort)
            pin = None
            if self.error_pinpointer:
                try:
                    ctx = self.error_pinpointer.analyze_error(e, "MultiScaleFeatureEngine")
                    pin = {"context": ctx, "guide": self.error_pinpointer.create_debugging_guide(ctx)}
                except Exception as pe:
                    pin = {"pinpointer_failed": str(pe), "trace": traceback.format_exc()}

            thesis = f"Neural processing failed: {e}. Returning fallback embeddings/attention."
            self._audit_cycle(
                level="ERROR",
                cycle_id=cycle_id,
                step_idx=step_idx,
                bus_timestamp=bus_ts,
                start_ts=start_ts,
                success=False,
                thesis=thesis,
                error=str(e),
                trace=traceback.format_exc(),
                extra={"processing_time_ms": float(elapsed_ms), "pinpointer": pin, "bus_writes": self._last_bus_writes_snapshot()},
            )

            # Publish health only; keep last embeddings/attention stable (avoid flapping).
            try:
                with self._bus_lock:
                    self.smart_bus.set(
                        "neural_health",
                        self._health_snapshot(),
                        module="MultiScaleFeatureEngine",
                        thesis="Neural health snapshot after failure.",
                    )
                    self._track_bus_write("neural_health", self._health_snapshot(), "Neural health snapshot after failure.")
            except Exception as be:
                self._audit_event(level="ERROR", event_type="bus_set_failed_during_error_reporting", error=str(be), trace=traceback.format_exc())

            out = self._baseline_outputs(reason="exception_fallback")
            return self._format_declared_outputs(
                outputs=out,
                thesis=thesis,
                extra={"success": False, "processing_time_ms": float(elapsed_ms), "neural_error": {"error": str(e), "trace": traceback.format_exc() if self.debug else None}},
            )

    # ─────────────────────────────────────────────────────────
    # Scope enforcement
    # ─────────────────────────────────────────────────────────
    def _enforce_xauusd_scope(self, *, market_data: Optional[Dict[str, Any]], afe_payload: Optional[Dict[str, Any]]):
        # Accept missing instrument fields; enforce only when provided.
        candidates: List[str] = []

        if isinstance(market_data, dict):
            for k in ("instrument", "symbol"):
                if isinstance(market_data.get(k), str):
                    candidates.append(str(market_data[k]).upper())

        if isinstance(afe_payload, dict):
            for k in ("instrument", "symbol"):
                if isinstance(afe_payload.get(k), str):
                    candidates.append(str(afe_payload[k]).upper())

        for c in candidates:
            if c and c != self.symbol:
                raise ValueError(f"MultiScaleFeatureEngine scope violation: expected {self.symbol}, got {c}")

    # ─────────────────────────────────────────────────────────
    # Inputs
    # ─────────────────────────────────────────────────────────
    def _bus_get(self, key: str, default: Any = None, soft: bool = False) -> Any:
        try:
            return self.smart_bus.get(key, self.__class__.__name__)
        except Exception as e:
            self._audit_event(level="WARN" if soft else "ERROR", event_type="bus_get_failed", key=key, error=str(e), trace=(traceback.format_exc() if self.debug else None))
            return default if soft else default

    def _get_market_data(self, **inputs) -> Dict[str, Any]:
        md = inputs.get("market_data")
        if isinstance(md, dict):
            return md
        bus_md = self._bus_get("market_data", default={}, soft=True)
        return bus_md if isinstance(bus_md, dict) else {}

    def _discover_input_dim(self) -> int:
        """
        Deterministic input dim discovery (in descending priority):
        1) injected AFE.out_dim
        2) InfoBus 'advanced_features'.raw_features length
        3) InfoBus 'advanced_features_{TF}'.raw_features length (first match)
        4) fallback assumed_input_dim
        """
        # 1) injected AFE
        if self._afe is not None and hasattr(self._afe, "out_dim"):
            try:
                od = int(getattr(self._afe, "out_dim"))
                if od > 0:
                    return od
            except Exception:
                pass

        # 2) base advanced_features
        try:
            adv = self.smart_bus.get("advanced_features", self.__class__.__name__)
            if isinstance(adv, dict):
                rf = adv.get("raw_features")
                if isinstance(rf, (list, tuple)) and len(rf) > 0:
                    return int(len(rf))
        except Exception:
            pass

        # 3) per-tf mirrors
        for tf in (self._cfg.timeframes or ["M15", "H1", "H4", "D1"]):
            try:
                k = f"advanced_features_{str(tf).upper()}"
                adv_tf = self.smart_bus.get(k, self.__class__.__name__)
                if isinstance(adv_tf, dict):
                    rf = adv_tf.get("raw_features")
                    if isinstance(rf, (list, tuple)) and len(rf) > 0:
                        return int(len(rf))
            except Exception:
                continue

        return int(self._cfg.assumed_input_dim)

    async def _get_afe_bundle(self, **inputs) -> Dict[str, Any]:
        """
        Returns:
          {
            "base_vec": np.ndarray,
            "base_payload": dict|None,
            "tf_vecs": Dict[tf, np.ndarray],
            "tf_payloads": Dict[tf, dict],
            "source": "injected_afe"|"bus"|"fallback"
          }

        Rules:
        - No random fallback. If missing: deterministic zeros.
        - Prefer per-timeframe mirrors advanced_features_{TF} if available.
        - Optional injection: if injected AFE exists, we may read its output (reader mode).
          WARNING: if AFE.process writes to bus, orchestration must ensure single writer upstream.
        """
        base_payload = None
        tf_payloads: Dict[str, Dict[str, Any]] = {}

        # 1) injected AFE (optional)
        if self._afe is not None and hasattr(self._afe, "process"):
            try:
                afe_out = await self._afe.process(**inputs)
                adv = afe_out.get("advanced_features", {})
                if isinstance(adv, dict) and isinstance(adv.get("raw_features"), list) and len(adv["raw_features"]) > 0:
                    base_payload = adv
                    base_vec = np.asarray(adv["raw_features"], dtype=np.float32).reshape(-1)

                    # also try mirrors from injected output (if present)
                    tf_vecs: Dict[str, np.ndarray] = {}
                    for tf in self.timeframes:
                        k = f"advanced_features_{tf}"
                        p = afe_out.get(k)
                        if isinstance(p, dict) and isinstance(p.get("raw_features"), list) and len(p["raw_features"]) > 0:
                            tf_payloads[tf] = p
                            tf_vecs[tf] = np.asarray(p["raw_features"], dtype=np.float32).reshape(-1)

                    if base_vec.size > 0:
                        return {
                            "base_vec": base_vec,
                            "base_payload": base_payload,
                            "tf_vecs": tf_vecs,
                            "tf_payloads": tf_payloads,
                            "source": "injected_afe",
                        }
            except Exception as e:
                self._audit_event(level="WARN", event_type="afe_process_failed", error=str(e), trace=(traceback.format_exc() if self.debug else None))

        # 2) Bus base + mirrors
        try:
            bus_adv = self.smart_bus.get("advanced_features", self.__class__.__name__)
            if isinstance(bus_adv, dict):
                rf = bus_adv.get("raw_features")
                if isinstance(rf, (list, tuple)) and len(rf) > 0:
                    base_payload = bus_adv
                    base_vec = np.asarray(rf, dtype=np.float32).reshape(-1)

                    tf_vecs: Dict[str, np.ndarray] = {}
                    for tf in self.timeframes:
                        k = f"advanced_features_{tf}"
                        adv_tf = self._bus_get(k, default=None, soft=True)
                        if isinstance(adv_tf, dict):
                            rft = adv_tf.get("raw_features")
                            if isinstance(rft, (list, tuple)) and len(rft) > 0:
                                tf_payloads[tf] = adv_tf
                                tf_vecs[tf] = np.asarray(rft, dtype=np.float32).reshape(-1)

                    if base_vec.size > 0:
                        return {
                            "base_vec": base_vec,
                            "base_payload": base_payload,
                            "tf_vecs": tf_vecs,
                            "tf_payloads": tf_payloads,
                            "source": "bus",
                        }
        except Exception as e:
            self._audit_event(level="WARN", event_type="bus_advanced_features_unavailable", error=str(e), trace=(traceback.format_exc() if self.debug else None))

        # 3) Deterministic fallback (zeros)
        z = np.zeros(int(self.input_dim), dtype=np.float32)
        return {"base_vec": z, "base_payload": None, "tf_vecs": {}, "tf_payloads": {}, "source": "fallback"}

    # ─────────────────────────────────────────────────────────
    # Multiscale features shaping
    # ─────────────────────────────────────────────────────────
    def _pad_trunc(self, v: np.ndarray, L: int) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size < L:
            out = np.zeros(L, dtype=np.float32)
            out[:v.size] = v
            return out
        if v.size > L:
            return v[-L:].astype(np.float32, copy=False)
        return v.astype(np.float32, copy=False)

    def _build_multiscale_features(self, afe_bundle: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()

        base = np.asarray(afe_bundle.get("base_vec", []), dtype=np.float32).reshape(-1)
        base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

        target_len = int(self.input_dim)
        base_fixed = self._pad_trunc(base, target_len)

        # Use real per-tf mirrors when available; otherwise fall back to base.
        timeframe_features: Dict[str, np.ndarray] = {}
        timeframes_used: List[str] = []
        for tf in self.timeframes:
            src = afe_bundle.get("tf_vecs", {}).get(tf)
            if src is None:
                vec = base_fixed
            else:
                vec = self._pad_trunc(np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0), target_len)
                timeframes_used.append(tf)
            timeframe_features[tf] = vec

        # Safe correlation on a compact projection to avoid huge-dot instability
        # (first K dims after centering; deterministic)
        K = int(min(128, target_len))
        def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a[:K], dtype=np.float64)
            b = np.asarray(b[:K], dtype=np.float64)
            if a.size < 2 or b.size < 2 or a.size != b.size:
                return 0.0
            a = a - a.mean()
            b = b - b.mean()
            sa = a.std()
            sb = b.std()
            if not (np.isfinite(sa) and np.isfinite(sb)) or sa < 1e-12 or sb < 1e-12:
                return 0.0
            corr = float(np.dot(a, b) / (sa * sb * a.size))
            return corr if np.isfinite(corr) else 0.0

        correlations: Dict[str, float] = {}
        tf_names = list(self.timeframes)
        for i, a in enumerate(tf_names):
            for b in tf_names[i + 1:]:
                correlations[f"{a}_{b}"] = _safe_corr(timeframe_features[a], timeframe_features[b])

        md_fields = int(len(market_data) if isinstance(market_data, dict) else 0)

        return {
            "timeframe_features": timeframe_features,
            "correlations": correlations,
            "base_features": base_fixed,
            "market_fields_seen": md_fields,
            "processing_time_ms": float((time.time() - t0) * 1000.0),
            "timeframes_used": timeframes_used,
        }

    # ─────────────────────────────────────────────────────────
    # Neural forward
    # ─────────────────────────────────────────────────────────
    async def _neural_forward(self, ms_result: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()

        tf_feats = ms_result.get("timeframe_features") or {}
        if not isinstance(tf_feats, dict) or not tf_feats:
            return self._structured_fallback_nn("no timeframe features")

        processed: Dict[str, torch.Tensor] = {}
        ordered_tfs: List[str] = []
        for tf in self.timeframes:
            if tf not in tf_feats:
                continue
            try:
                vec = np.asarray(tf_feats[tf], dtype=np.float32).reshape(-1)
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                vec = self._pad_trunc(vec, int(self.input_dim))

                x = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]
                y = self.scale_processors[tf](x)  # [1, E]
                processed[tf] = y
                ordered_tfs.append(tf)
            except Exception as e:
                self._audit_event(level="WARN", event_type="tf_forward_skip", tf=tf, error=str(e), trace=(traceback.format_exc() if self.debug else None))
                continue

        if not processed:
            return self._structured_fallback_nn("no processed timeframe tensors")

        # Concat fusion: [B, E*T]
        concat = torch.cat([processed[tf] for tf in ordered_tfs], dim=-1)
        fused = self.fusion_network(concat)  # [B, E]

        # Attention fusion: [B, T, E]
        attn_in = torch.stack([processed[tf] for tf in ordered_tfs], dim=1)
        attn_out, attn_w = self.attention_fusion(attn_in)  # [B,T,E], [B,T,T]

        # Combine fused with mean attention output
        combined = 0.5 * (fused + attn_out.mean(dim=1))  # [B, E]

        embedding = combined.detach().cpu().numpy().astype(np.float32, copy=False)      # [B, E]
        attn_weights = attn_w.detach().cpu().numpy().astype(np.float32, copy=False)    # [B, T, T]
        forward_ms = float((time.time() - t0) * 1000.0)

        entropy = float(self._attention_entropy(attn_weights))

        # Persist last-good outputs
        self.last_embedding = embedding.reshape(-1).copy()
        # Expand attention into full TF grid deterministically (missing TFs -> zeros)
        full_T = len(self.timeframes)
        full = np.zeros((1, full_T, full_T), dtype=np.float32)
        idx_map = {tf: i for i, tf in enumerate(self.timeframes)}
        used_idx = [idx_map[tf] for tf in ordered_tfs if tf in idx_map]
        # attn_weights is [1, t, t] over ordered_tfs
        for i_local, i_global in enumerate(used_idx):
            for j_local, j_global in enumerate(used_idx):
                full[0, i_global, j_global] = float(attn_weights[0, i_local, j_local])
        self.last_attention = full

        self.attention_weights_history.append(attn_weights)
        self.embedding_history.append({
            "embedding": self.last_embedding.copy(),
            "timestamp": time.time(),
            "attention_entropy": entropy,
            "timeframes_used": ordered_tfs,
        })

        self._update_neural_stats(forward_ms=forward_ms, attention_entropy=entropy, success=True)

        return {
            "embeddings": embedding,                    # [B,E]
            "attention_weights": attn_weights,          # [B,t,t] (t = used timeframes)
            "attention_weights_full": full,             # [B,T,T] (T = configured timeframes)
            "processed_features": {k: v.detach().cpu().numpy().astype(np.float32, copy=False) for k, v in processed.items()},
            "forward_time_ms": forward_ms,
            "attention_entropy": entropy,
            "timeframes_used": ordered_tfs,
        }

    def _structured_fallback_nn(self, reason: str) -> Dict[str, Any]:
        T = len(self.timeframes)
        emb = self.last_embedding if isinstance(self.last_embedding, np.ndarray) and self.last_embedding.size == self.output_dim else np.zeros(self.output_dim, dtype=np.float32)
        attn = self.last_attention if isinstance(self.last_attention, np.ndarray) and self.last_attention.shape == (1, T, T) else np.zeros((1, T, T), dtype=np.float32)
        self._update_neural_stats(forward_ms=0.0, attention_entropy=float(self.neural_stats.get("avg_attention_entropy", 0.0)), success=False)
        return {
            "embeddings": emb.reshape(1, -1).astype(np.float32, copy=False),
            "attention_weights": attn.astype(np.float32, copy=False),
            "attention_weights_full": attn.astype(np.float32, copy=False),
            "processed_features": {},
            "forward_time_ms": 0.0,
            "attention_entropy": 0.0,
            "_reason": reason,
            "timeframes_used": [],
        }

    @staticmethod
    def _attention_entropy(attn_w: np.ndarray) -> float:
        try:
            w = np.asarray(attn_w, dtype=np.float64).reshape(-1)
            if w.size == 0:
                return 0.0
            # Normalize to a probability distribution
            s = float(np.sum(w)) + 1e-12
            p = w / s
            return float(-np.sum(p * np.log(p + 1e-12)))
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────
    # Thesis / explainability (returned only; not bus)
    # ─────────────────────────────────────────────────────────
    async def _generate_neural_thesis(
        self,
        afe_bundle: Dict[str, Any],
        ms_result: Dict[str, Any],
        nn_result: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> str:
        try:
            emb = np.asarray(nn_result.get("embeddings"), dtype=np.float32).reshape(-1)
            q_emb = float(self._assess_embedding_quality(emb))
            attn_entropy = float(nn_result.get("attention_entropy", 0.0))
            fwd = float(nn_result.get("forward_time_ms", 0.0))
            md_seen = int(ms_result.get("market_fields_seen", 0))
            tf_used = nn_result.get("timeframes_used", [])
            src = str(afe_bundle.get("source", "unknown"))

            return (
                f"Neural Multi-Scale Feature Analysis ({self.symbol})\n"
                f"- Source: {src}\n"
                f"- Timeframes configured: {len(self.timeframes)} | used: {len(tf_used)}\n"
                f"- Input dim: {int(self.input_dim)} | Embedding dim: {int(self.output_dim)}\n"
                f"- Attention entropy: {attn_entropy:.3f}\n"
                f"- Forward time: {fwd:.1f} ms\n\n"
                "Quality\n"
                f"- Embedding quality: {q_emb:.1f}%\n"
                f"- Correlations computed: {len(ms_result.get('correlations', {}))}\n\n"
                "Inputs\n"
                f"- market_data fields seen: {md_seen}\n"
                f"- Device: {self.device}\n"
            )
        except Exception as e:
            return f"Neural processing completed. Thesis generation failed: {e}"

    @staticmethod
    def _assess_embedding_quality(embeddings: np.ndarray) -> float:
        try:
            flat = np.asarray(embeddings, dtype=float).flatten()
            if flat.size == 0 or np.any(~np.isfinite(flat)):
                return 0.0
            std = float(np.std(flat))
            rng = float(np.max(flat) - np.min(flat))
            if std < 1e-9 or rng < 1e-9:
                return 20.0
            score = 90.0
            if np.max(np.abs(flat)) > 10.0:
                score -= 15.0
            if 0.1 < std < 5.0:
                score += 5.0
            return float(np.clip(score, 0.0, 100.0))
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────
    # Outputs + bus publishing (declared keys only)
    # ─────────────────────────────────────────────────────────
    def _package_outputs(self, ms_result: Dict[str, Any], nn_result: Dict[str, Any]) -> Dict[str, Any]:
        # multiscale_features (numpy → python)
        tf_feats = ms_result.get("timeframe_features", {})
        tf_feats_py = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in tf_feats.items()}

        out: Dict[str, Any] = {
            "multiscale_features": {
                "correlations": ms_result.get("correlations", {}),
                "timeframe_features": tf_feats_py,
                "processing_time_ms": float(ms_result.get("processing_time_ms", 0.0)),
                "timeframes_used": list(ms_result.get("timeframes_used", [])),
                "instrument": self.symbol,
                "primary_timeframe": self.primary_timeframe,
            },
            # Keep backward-compatible shapes for declared keys:
            "neural_embeddings": np.asarray(nn_result.get("embeddings"), dtype=np.float32).reshape(-1).tolist(),
            "attention_weights": np.asarray(nn_result.get("attention_weights_full"), dtype=np.float32).tolist(),  # [1,T,T]
            "feature_fusion": {
                "processed_features": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in (nn_result.get("processed_features") or {}).items()
                },
                "fusion_method": str(self._cfg.feature_fusion_method),
                "processing_time_ms": float(nn_result.get("forward_time_ms", 0.0)),
                "timeframes_used": list(nn_result.get("timeframes_used", [])),
            },
            "neural_capabilities": self._capabilities_snapshot(),
            "neural_health": self._health_snapshot(),
        }

        # Optional “payload” mirrors (helps debugging without breaking contracts)
        out["neural_embeddings_payload"] = {
            "embeddings": out["neural_embeddings"],
            "dimensions": [1, int(self.output_dim)],
            "device": str(self.device),
            "timestamp": time.time(),
        }
        out["attention_weights_payload"] = {
            "weights": out["attention_weights"],
            "entropy": float(nn_result.get("attention_entropy", 0.0)),
            "num_heads": int(self._cfg.num_attention_heads),
            "timeframes": list(self.timeframes),
        }
        return out

    def _baseline_outputs(self, reason: str) -> Dict[str, Any]:
        T = len(self.timeframes)
        emb = self.last_embedding if isinstance(self.last_embedding, np.ndarray) and self.last_embedding.size == self.output_dim else np.zeros(self.output_dim, dtype=np.float32)
        attn = self.last_attention if isinstance(self.last_attention, np.ndarray) and self.last_attention.shape == (1, T, T) else np.zeros((1, T, T), dtype=np.float32)

        out: Dict[str, Any] = {
            "multiscale_features": {
                "correlations": {},
                "timeframe_features": {},
                "processing_time_ms": 0.0,
                "timeframes_used": [],
                "instrument": self.symbol,
                "primary_timeframe": self.primary_timeframe,
                "_reason": reason,
            },
            "neural_embeddings": emb.reshape(-1).tolist(),
            "attention_weights": attn.tolist(),
            "feature_fusion": {
                "processed_features": {},
                "fusion_method": str(self._cfg.feature_fusion_method),
                "processing_time_ms": 0.0,
                "timeframes_used": [],
                "_reason": reason,
            },
            "neural_capabilities": self._capabilities_snapshot(),
            "neural_health": self._health_snapshot(),
            "neural_embeddings_payload": {
                "embeddings": emb.reshape(-1).tolist(),
                "dimensions": [1, int(self.output_dim)],
                "device": str(self.device),
                "timestamp": time.time(),
                "_reason": reason,
            },
            "attention_weights_payload": {
                "weights": attn.tolist(),
                "entropy": 0.0,
                "num_heads": int(self._cfg.num_attention_heads),
                "timeframes": list(self.timeframes),
                "_reason": reason,
            },
        }
        return out

    def _format_declared_outputs(
        self,
        *,
        outputs: Dict[str, Any],
        thesis: Optional[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        out["multiscale_features"] = outputs.get("multiscale_features", {})
        out["neural_embeddings"] = outputs.get("neural_embeddings", [])
        out["attention_weights"] = outputs.get("attention_weights", [])
        out["feature_fusion"] = outputs.get("feature_fusion", {})
        out["neural_capabilities"] = outputs.get("neural_capabilities", {})
        out["neural_health"] = outputs.get("neural_health", {})

        # Helpful extras (non-contract)
        if thesis:
            out["_thesis"] = thesis
        if extra and isinstance(extra, dict):
            out.update(extra)

        # Optional payload mirrors
        if "neural_embeddings_payload" in outputs:
            out["neural_embeddings_payload"] = outputs["neural_embeddings_payload"]
        if "attention_weights_payload" in outputs:
            out["attention_weights_payload"] = outputs["attention_weights_payload"]

        # Validate declared provides are present
        for k in ("multiscale_features", "neural_embeddings", "attention_weights", "feature_fusion", "neural_capabilities", "neural_health"):
            if k not in out:
                raise ValueError(f"Critical output '{k}' missing in MultiScaleFeatureEngine")

        return out

    def _update_bus_from_outputs(self, outputs: Dict[str, Any], thesis: str):
        self._reset_last_bus_writes()
        with self._bus_lock:
            # multiscale_features
            self.smart_bus.set(
                "multiscale_features",
                outputs.get("multiscale_features", {}),
                module="MultiScaleFeatureEngine",
                thesis="Multiscale features published."
            )
            self._track_bus_write("multiscale_features", outputs.get("multiscale_features", {}), "Multiscale features published.")

            # neural_embeddings (bus-friendly dict payload)
            nep = outputs.get("neural_embeddings_payload") or {
                "embeddings": outputs.get("neural_embeddings", []),
                "dimensions": [1, int(self.output_dim)],
                "device": str(self.device),
                "timestamp": time.time(),
            }
            self.smart_bus.set("neural_embeddings", nep, module="MultiScaleFeatureEngine", thesis="Neural embeddings published.")
            self._track_bus_write("neural_embeddings", nep, "Neural embeddings published.")

            # attention_weights (bus-friendly dict payload)
            awp = outputs.get("attention_weights_payload") or {
                "weights": outputs.get("attention_weights", []),
                "entropy": float(outputs.get("attention_entropy", 0.0)) if isinstance(outputs, dict) else 0.0,
                "num_heads": int(self._cfg.num_attention_heads),
                "timeframes": list(self.timeframes),
            }
            self.smart_bus.set("attention_weights", awp, module="MultiScaleFeatureEngine", thesis="Attention analysis.")
            self._track_bus_write("attention_weights", awp, "Attention analysis.")

            # feature_fusion
            self.smart_bus.set(
                "feature_fusion",
                outputs.get("feature_fusion", {}),
                module="MultiScaleFeatureEngine",
                thesis="Feature fusion summary."
            )
            self._track_bus_write("feature_fusion", outputs.get("feature_fusion", {}), "Feature fusion summary.")

            # neural_capabilities
            caps = outputs.get("neural_capabilities", self._capabilities_snapshot())
            self.smart_bus.set("neural_capabilities", caps, module="MultiScaleFeatureEngine", thesis="Neural capabilities.")
            self._track_bus_write("neural_capabilities", caps, "Neural capabilities.")

            # neural_health
            nh = outputs.get("neural_health", self._health_snapshot())
            self.smart_bus.set("neural_health", nh, module="MultiScaleFeatureEngine", thesis="Neural health snapshot.")
            self._track_bus_write("neural_health", nh, "Neural health snapshot.")

    # ─────────────────────────────────────────────────────────
    # Capabilities / health snapshots
    # ─────────────────────────────────────────────────────────
    def _capabilities_snapshot(self) -> Dict[str, Any]:
        return {
            "instrument": self.symbol,
            "primary_timeframe": self.primary_timeframe,
            "timeframes": list(self.timeframes),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "embed_dim": int(self._cfg.embed_dim),
            "num_attention_heads": int(self._cfg.num_attention_heads),
            "feature_fusion_method": str(self._cfg.feature_fusion_method),
            "device": str(self.device),
            "gpu_available": bool(torch.cuda.is_available()),
            "debug": bool(self.debug),
        }

    def _health_snapshot(self) -> Dict[str, Any]:
        return {
            "instrument": self.symbol,
            "timeframe": self.primary_timeframe,
            "model_health_score": float(self.neural_health.get("model_health_score", 0.0)),
            "performance_trend": self.neural_health.get("performance_trend", "unknown"),
            "issues_detected": list(self.neural_health.get("issues_detected", [])),
            "avg_forward_time_ms": float(self.neural_stats.get("avg_forward_time_ms", 0.0)),
            "avg_attention_entropy": float(self.neural_stats.get("avg_attention_entropy", 0.0)),
            "successful_passes": int(self.neural_stats.get("successful_passes", 0)),
            "total_forward_passes": int(self.neural_stats.get("total_forward_passes", 0)),
            "gpu_memory_usage_mb": float(self.neural_stats.get("gpu_memory_usage_mb", 0.0)),
            "breaker": dict(self.circuit_breaker),
        }

    # ─────────────────────────────────────────────────────────
    # Stats / breaker
    # ─────────────────────────────────────────────────────────
    def _check_circuit_breaker(self) -> bool:
        state = self.circuit_breaker.get("state", "CLOSED")
        if state == "OPEN":
            lf = float(self.circuit_breaker.get("last_failure", 0.0) or 0.0)
            cooldown = float(self.circuit_breaker.get("cooldown_s", 120.0))
            if lf > 0.0 and (time.time() - lf) > cooldown:
                self.circuit_breaker["state"] = "HALF_OPEN"
                return True
            return False
        return True

    def _record_success(self, forward_time_ms: float):
        if self.circuit_breaker["state"] == "HALF_OPEN":
            self.circuit_breaker["state"] = "CLOSED"
            self.circuit_breaker["failures"] = 0

        self.neural_stats["total_forward_passes"] += 1
        self.neural_stats["successful_passes"] += 1
        self.neural_stats["last_success_ts"] = time.time()

        n = int(self.neural_stats["total_forward_passes"])
        prev = float(self.neural_stats.get("avg_forward_time_ms", 0.0))
        self.neural_stats["avg_forward_time_ms"] = (prev * (n - 1) + float(forward_time_ms)) / max(1, n)

        # Health bump
        hs = float(self.neural_health.get("model_health_score", 100.0))
        self.neural_health["model_health_score"] = float(min(100.0, hs + 2.0))

        if self.performance_tracker:
            try:
                self.performance_tracker.record_metric("MultiScaleFeatureEngine", "neural_forward", float(forward_time_ms), True)
            except Exception as e:
                self._audit_event(level="WARN", event_type="performance_tracker_failed", error=str(e), trace=(traceback.format_exc() if self.debug else None))

    def _record_failure(self, error: Exception):
        self.circuit_breaker["failures"] = int(self.circuit_breaker.get("failures", 0)) + 1
        self.circuit_breaker["last_failure"] = time.time()
        if int(self.circuit_breaker["failures"]) >= int(self.circuit_breaker["threshold"]):
            self.circuit_breaker["state"] = "OPEN"
            self.logger.error(
                format_operator_message(
                    "[ALERT]", "NEURAL_CIRCUIT_BREAKER_OPEN",
                    details=f"Too many failures ({self.circuit_breaker['failures']})",
                    context="neural_circuit_breaker"
                )
            )

        self.neural_stats["total_forward_passes"] += 1
        self.neural_stats["failed_passes"] += 1
        self.neural_stats["last_failure_ts"] = time.time()

        # Health penalty
        hs = float(self.neural_health.get("model_health_score", 100.0))
        self.neural_health["model_health_score"] = float(max(0.0, hs - 15.0))

        issues = list(self.neural_health.get("issues_detected", []))
        issues.append(f"{type(error).__name__}: {error}")
        self.neural_health["issues_detected"] = issues
        self.neural_health["performance_trend"] = "degrading"

    def _update_neural_stats(self, *, forward_ms: float, attention_entropy: float, success: bool):
        self.neural_stats["total_forward_passes"] = int(self.neural_stats.get("total_forward_passes", 0)) + 1
        if success:
            self.neural_stats["successful_passes"] = int(self.neural_stats.get("successful_passes", 0)) + 1
        else:
            self.neural_stats["failed_passes"] = int(self.neural_stats.get("failed_passes", 0)) + 1

        n = int(self.neural_stats["total_forward_passes"])
        avg = float(self.neural_stats.get("avg_forward_time_ms", 0.0))
        self.neural_stats["avg_forward_time_ms"] = (avg * (n - 1) + float(forward_ms)) / max(1, n)

        ae = float(self.neural_stats.get("avg_attention_entropy", 0.0))
        self.neural_stats["avg_attention_entropy"] = (ae * (n - 1) + float(attention_entropy)) / max(1, n)

        if torch.cuda.is_available():
            self.neural_stats["gpu_memory_usage_mb"] = float(torch.cuda.memory_allocated() / 1024.0 / 1024.0)

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

            lg = logging.getLogger("audit.MultiScaleFeatureEngine")
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
            evt = {
                "ts": time.time(),
                "level": str(level).upper(),
                "module": "MultiScaleFeatureEngine",
                "event": str(event_type),
                "instrument": getattr(self, "symbol", "XAUUSD"),
                "primary_timeframe": getattr(self, "primary_timeframe", "M15"),
                "debug": bool(getattr(self, "debug", False)),
                **fields,
            }
            line = json.dumps(evt, ensure_ascii=False, separators=(",", ":"))
            logger = getattr(self, "_audit_logger", None)
            if logger is not None and hasattr(logger, "info"):
                logger.info(line)
            else:
                self.logger.warning(f"[AUDIT_FALLBACK] {line}")
        except Exception:
            return

    def _audit_cycle(
        self,
        *,
        level: str,
        cycle_id: int,
        step_idx: Any,
        bus_timestamp: Any,
        start_ts: float,
        success: bool,
        thesis: Optional[str],
        error: Optional[str] = None,
        trace: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        ms_result: Optional[Dict[str, Any]] = None,
        nn_result: Optional[Dict[str, Any]] = None,
        afe_bundle: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            fields: Dict[str, Any] = {
                "cycle_id": int(cycle_id),
                "step_idx": step_idx,
                "bus_timestamp": bus_timestamp,
                "duration_ms": float((time.time() - start_ts) * 1000.0),
                "success": bool(success),
                "thesis": thesis if (self.debug or not thesis) else (thesis[:4000] if thesis else None),
                "error": error,
                "trace": (trace if self.debug else None),
            }

            if isinstance(extra, dict):
                fields["extra"] = extra

            if self.debug and isinstance(ms_result, dict):
                # keep compact
                fields["ms"] = {
                    "processing_time_ms": float(ms_result.get("processing_time_ms", 0.0)),
                    "market_fields_seen": int(ms_result.get("market_fields_seen", 0)),
                    "corr_n": int(len(ms_result.get("correlations", {}) or {})),
                    "tfs_used": list(ms_result.get("timeframes_used", [])),
                }

            if self.debug and isinstance(nn_result, dict):
                emb = np.asarray(nn_result.get("embeddings", []), dtype=np.float32).reshape(-1)
                fields["nn"] = {
                    "forward_time_ms": float(nn_result.get("forward_time_ms", 0.0)),
                    "attention_entropy": float(nn_result.get("attention_entropy", 0.0)),
                    "embedding_n": int(emb.size),
                    "tfs_used": list(nn_result.get("timeframes_used", [])),
                }
                if self._full_vec:
                    fields["nn"]["embedding_full"] = emb.tolist()
                if self._full_attn:
                    fields["nn"]["attention_full"] = np.asarray(nn_result.get("attention_weights_full", [])).tolist()

            if self.debug and isinstance(afe_bundle, dict):
                base = np.asarray(afe_bundle.get("base_vec", []), dtype=np.float32).reshape(-1)
                fields["afe"] = {
                    "source": afe_bundle.get("source"),
                    "base_dim": int(base.size),
                    "tf_available": sorted(list((afe_bundle.get("tf_vecs") or {}).keys())),
                }

            self._audit_event(level=level, event_type="cycle", **fields)
        except Exception as e:
            self._audit_event(level="ERROR", event_type="audit_cycle_failed", error=str(e), trace=traceback.format_exc())

    # ─────────────────────────────────────────────────────────
    # Bus write tracking (debug)
    # ─────────────────────────────────────────────────────────
    def _reset_last_bus_writes(self):
        self._last_bus_writes = []

    def _track_bus_write(self, key: str, value: Any, thesis: str):
        try:
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
            self._audit_event(level="WARN", event_type="track_bus_write_failed", key=key, error=str(e), trace=(traceback.format_exc() if self.debug else None))

    def _last_bus_writes_snapshot(self) -> List[Dict[str, Any]]:
        try:
            return list(self._last_bus_writes or [])
        except Exception:
            return []

    # ─────────────────────────────────────────────────────────
    # State & reports
    # ─────────────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        return {
            **base,
            "config": self._capabilities_snapshot(),
            "statistics": dict(self.neural_stats),
            "health_metrics": dict(self.neural_health),
            "circuit_breaker": dict(self.circuit_breaker),
            "last_embedding": self.last_embedding.tolist() if isinstance(self.last_embedding, np.ndarray) else [],
            "last_attention": self.last_attention.tolist() if isinstance(self.last_attention, np.ndarray) else [],
        }

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        try:
            if "last_embedding" in state:
                self.last_embedding = np.asarray(state["last_embedding"], dtype=np.float32).reshape(-1)
            if "last_attention" in state:
                self.last_attention = np.asarray(state["last_attention"], dtype=np.float32)
            if "statistics" in state and isinstance(state["statistics"], dict):
                self.neural_stats.update(state["statistics"])
            if "health_metrics" in state and isinstance(state["health_metrics"], dict):
                self.neural_health.update(state["health_metrics"])
            if "circuit_breaker" in state and isinstance(state["circuit_breaker"], dict):
                self.circuit_breaker.update(state["circuit_breaker"])
            new_dim = int(state.get("config", {}).get("input_dim", self.input_dim))
            if new_dim != int(self.input_dim) and new_dim > 0:
                self.input_dim = new_dim
                self._build_networks(self.input_dim)
        except Exception as e:
            self._audit_event(level="WARN", event_type="set_state_failed", error=str(e), trace=(traceback.format_exc() if self.debug else None))
