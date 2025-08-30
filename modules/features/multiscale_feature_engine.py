# ─────────────────────────────────────────────────────────────
# File: modules/features/multiscale_feature_engine.py
# MultiScale Feature Engine (Contract-Clean, Orchestration-Safe)
# ─────────────────────────────────────────────────────────────

import time
import asyncio
from modules.contracts import module_args
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING
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

# Optional dependency (do NOT instantiate by default to avoid duplicate writers)
try:
    from modules.features.advanced_feature_engine import AdvancedFeatureEngine  # noqa
except Exception:
    AdvancedFeatureEngine = None  # type: ignore

if TYPE_CHECKING:
    from modules.features.advanced_feature_engine import AdvancedFeatureEngine as AFEType


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass
class MultiScaleConfig:
    embed_dim: int = 64
    num_attention_heads: int = 4
    dropout_rate: float = 0.10
    enable_gpu: bool = True
    feature_fusion_method: str = "attention"  # "attention" | "concat" | "weighted"
    timeframes: Optional[List[str]] = None
    # If AFE isn't injected and Bus doesn't yet have features, we need a deterministic dim:
    assumed_input_dim: int = 256

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["H1", "H4", "D1"]


# ─────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────
class AttentionFeatureFusion(nn.Module):
    """Self-attention over timeframe embeddings with residual MLP block."""
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, T, input_dim]
        h = self.input_proj(x)  # [batch, T, E]
        attn_out, attn_w = self.attn(h, h, h)  # [batch, T, E], [batch, T, T]
        h = self.norm(h + attn_out)
        h = self.out(h)  # [batch, T, E]
        return h, attn_w


# ─────────────────────────────────────────────────────────────
# Module declaration
# ─────────────────────────────────────────────────────────────
@module(**module_args(
    "MultiScaleFeatureEngine",
    description="Reads advanced_features + market_data; produces multiscale_features, embeddings, attention, and neural health/capabilities.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=120,
))
class MultiScaleFeatureEngine(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Contract (registry):
      Provides: attention_weights, feature_fusion, multiscale_features, neural_capabilities, neural_embeddings, neural_health
      Requires: advanced_features, market_data
    """

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        config: Optional[Union[MultiScaleConfig, Dict[str, Any]]] = None,
        afe: Optional["AFEType"] = None,  # optional injection; we won't create one by default
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

        # Optional AFE (to get input_dim early if orchestrator injects it)
        self._afe = afe

        # BaseModule expects a dict config and will call _initialize()
        super().__init__(config=asdict(self._cfg), **kwargs)

    def _initialize(self):
        # Core handles
        self.smart_bus = InfoBusManager.get_instance()

        # Device
        self.device = torch.device("cuda" if (torch.cuda.is_available() and self._cfg.enable_gpu) else "cpu")

        # Determine input dimension:
        self.input_dim = self._discover_input_dim()
        self.output_dim = int(self._cfg.embed_dim)

        # Subsystems
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("MultiScaleFeatureEngine", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker
        self.neural_circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",  # CLOSED | HALF_OPEN | OPEN
            "threshold": 3,
        }

        # Build networks & state
        self._build_networks(self.input_dim)
        self._initialize_state()

        # Background monitoring
        self._start_monitoring()

        # Log
        self.logger.info(
            format_operator_message(
                "🧠", "MULTISCALE_READY",
                details=f"InputDim={self.input_dim}, EmbedDim={self.output_dim}, Device={self.device}",
                result="ready",
                context="multiscale_startup"
            )
        )

    # ─────────────────────────────────────────────────────────
    # Init helpers
    # ─────────────────────────────────────────────────────────
    def _discover_input_dim(self) -> int:
        """
        Find a deterministic input dim:
        1) Injected AFE.out_dim
        2) InfoBus 'advanced_features'.raw_features length
        3) Fallback to assumed_input_dim
        """
        # 1) injected AFE
        if self._afe is not None and hasattr(self._afe, "out_dim"):
            try:
                od = int(getattr(self._afe, "out_dim"))
                if od > 0:
                    return od
            except Exception:
                pass

        # 2) InfoBus (use self as requester; avoid spoofing provider names)
        try:
            adv = self.smart_bus.get("advanced_features", self.__class__.__name__)
            if isinstance(adv, dict):
                rf = adv.get("raw_features")
                if isinstance(rf, (list, tuple)) and len(rf) > 0:
                    return int(len(rf))
        except Exception:
            pass

        # 3) fallback
        return int(self._cfg.assumed_input_dim)

    def _build_networks(self, input_dim: int):
        """(Re)build networks for a given input_dim and move to device."""
        tfs = self._cfg.timeframes or ["H1", "H4", "D1"]
        E = int(self._cfg.embed_dim)

        # Per-timeframe processor
        self.scale_processors = nn.ModuleDict({
            tf: nn.Sequential(
                nn.Linear(input_dim, E),
                nn.ReLU(),
                nn.LayerNorm(E),
                nn.Dropout(self._cfg.dropout_rate),
            ) for tf in tfs
        })

        # Fusion MLP over concatenated timeframe embeddings
        fusion_input_dim = E * len(tfs)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, E * 2),
            nn.ReLU(),
            nn.LayerNorm(E * 2),
            nn.Dropout(self._cfg.dropout_rate),
            nn.Linear(E * 2, E),
            nn.ReLU(),
            nn.Linear(E, E)  # final embedding dim = E
        )

        # Self-attention across timeframes
        self.attention_fusion = AttentionFeatureFusion(
            input_dim=E, embed_dim=E, num_heads=int(self._cfg.num_attention_heads)
        )

        # Weight init
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        self.scale_processors.apply(init_layer)
        self.fusion_network.apply(init_layer)
        self.attention_fusion.apply(init_layer)

        # To device
        self.scale_processors.to(self.device)
        self.fusion_network.to(self.device)
        self.attention_fusion.to(self.device)

    def _initialize_state(self):
        self.last_embedding = np.zeros(self.output_dim, dtype=np.float32)
        self.attention_weights_history = deque(maxlen=100)
        self.embedding_history = deque(maxlen=500)

        self.neural_stats: Dict[str, Any] = {
            "total_forward_passes": 0,
            "successful_passes": 0,
            "failed_passes": 0,
            "avg_forward_time_ms": 0.0,
            "avg_attention_entropy": 0.0,
            "gpu_memory_usage_mb": 0.0
        }

        self.neural_health: Dict[str, Any] = {
            "model_health_score": 100.0,
            "gradient_health": "unknown",
            "attention_quality": 100.0,
            "embedding_quality": 100.0,
            "last_neural_check": time.time()
        }

    def _start_monitoring(self):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._neural_health_monitoring_loop())
            loop.create_task(self._gpu_monitoring_loop())
        except RuntimeError:
            pass

    # ─────────────────────────────────────────────────────────
    # Formatting / contract
    # ─────────────────────────────────────────────────────────
    def _format_declared_outputs(
        self,
        *,
        multiscale_features: Optional[Dict[str, Any]] = None,
        neural_embeddings: Optional[Any] = None,
        attention_weights: Optional[Any] = None,
        feature_fusion: Optional[Dict[str, Any]] = None,
        thesis: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # multiscale_features
        if isinstance(multiscale_features, dict):
            # numpy → python
            msf = {}
            for k, v in multiscale_features.items():
                if isinstance(v, dict):
                    msf[k] = {
                        kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items()
                    }
                elif isinstance(v, np.ndarray):
                    msf[k] = v.tolist()
                else:
                    msf[k] = v
            out["multiscale_features"] = msf
        else:
            out["multiscale_features"] = {}

        # embeddings
        emb = neural_embeddings
        if emb is None:
            emb = getattr(self, "last_embedding", np.zeros(self.output_dim, dtype=np.float32))
        if isinstance(emb, np.ndarray):
            out["neural_embeddings"] = emb.flatten().tolist()
        elif isinstance(emb, (list, tuple)):
            out["neural_embeddings"] = [float(x) for x in emb]
        else:
            out["neural_embeddings"] = [float(emb)] if emb is not None else []

        # attention weights
        aw = attention_weights
        if aw is None:
            T = len(self._cfg.timeframes or ["H1", "H4", "D1"])
            aw = np.zeros((int(self._cfg.num_attention_heads), T, T))
        if isinstance(aw, np.ndarray):
            out["attention_weights"] = aw.tolist()
        elif isinstance(aw, (list, tuple)):
            out["attention_weights"] = list(aw)
        else:
            out["attention_weights"] = []

        # feature_fusion
        ff = feature_fusion if isinstance(feature_fusion, dict) else {}
        processed = ff.get("processed_features", {})
        if isinstance(processed, dict):
            processed = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in processed.items()}
        out["feature_fusion"] = {
            "processed_features": processed,
            "fusion_method": ff.get("fusion_method", self._cfg.feature_fusion_method),
            "processing_time_ms": float(ff.get("processing_time_ms", 0.0)),
        }

        # extras (non-bus; safe to include for orchestrator UI)
        if thesis:
            out["_thesis"] = thesis
        if extra:
            try:
                out.update(extra)
            except Exception:
                pass

        # Contract: neural_capabilities (static model capabilities)
        out["neural_capabilities"] = {
            "input_dim": int(getattr(self, "input_dim", 0)),
            "output_dim": int(getattr(self, "output_dim", 0)),
            "embed_dim": int(self._cfg.embed_dim),
            "num_attention_heads": int(self._cfg.num_attention_heads),
            "timeframes": list(self._cfg.timeframes or []),
            "feature_fusion_method": str(self._cfg.feature_fusion_method),
            "device": str(getattr(self, "device", "cpu")),
            "gpu_available": bool(torch.cuda.is_available()),
        }

        # Contract: neural_health (compact health snapshot)
        try:
            out["neural_health"] = {
                "model_health_score": float(self.neural_health.get("model_health_score", 0.0)),
                "neural_circuit_breaker_state": self.neural_circuit_breaker.get("state", "CLOSED"),
                "avg_forward_time_ms": float(self.neural_stats.get("avg_forward_time_ms", 0.0)),
                "successful_passes": int(self.neural_stats.get("successful_passes", 0)),
                "total_forward_passes": int(self.neural_stats.get("total_forward_passes", 0)),
            }
        except Exception:
            out["neural_health"] = {
                "model_health_score": 0.0,
                "neural_circuit_breaker_state": "UNKNOWN",
                "avg_forward_time_ms": 0.0,
                "successful_passes": 0,
                "total_forward_passes": 0,
            }

        # Validate declared provides are present in returned payload
        for k in ("multiscale_features", "neural_embeddings", "attention_weights", "feature_fusion", "neural_capabilities", "neural_health"):
            if k not in out:
                raise ValueError(f"Critical output '{k}' missing in MultiScaleFeatureEngine")

        return out

    # ─────────────────────────────────────────────────────────
    # Public processing
    # ─────────────────────────────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        t0 = time.time()

        if not self._check_neural_circuit_breaker():
            return self._create_neural_fallback_response("Neural circuit breaker open")

        try:
            # Read required inputs per contract
            market_data = self._get_market_data(**inputs)  # required read
            afe_payload = await self._get_advanced_features(**inputs)  # required read

            ms_result = await self._process_multiscale_features(afe_payload, market_data)
            nn_result = await self._neural_forward(ms_result)
            thesis = await self._generate_neural_thesis(afe_payload, ms_result, nn_result, market_data)

            # Publish declared keys ONLY
            self._update_bus(ms_result, nn_result)

            self._record_neural_success(time.time() - t0)

            return self._format_declared_outputs(
                multiscale_features=ms_result,
                neural_embeddings=nn_result["embeddings"],
                attention_weights=nn_result["attention_weights"],
                feature_fusion={
                    "processed_features": nn_result.get("processed_features", {}),
                    "fusion_method": self._cfg.feature_fusion_method,
                    "processing_time_ms": nn_result.get("forward_time_ms", 0.0),
                },
                thesis=thesis,
                extra={
                    "processing_time_ms": (time.time() - t0) * 1000.0,
                    "device_used": str(self.device),
                    "success": True,
                    "attention_entropy": nn_result.get("attention_entropy", 0.0),
                    "market_fields_seen": int(len(market_data)),
                },
            )

        except Exception as e:
            return await self._handle_neural_error(e, t0)

    # ─────────────────────────────────────────────────────────
    # Inputs & preparation (contract-only)
    # ─────────────────────────────────────────────────────────
    def _get_market_data(self, **inputs) -> Dict[str, Any]:
        md = inputs.get("market_data")
        if not isinstance(md, dict):
            try:
                md = self.smart_bus.get("market_data", self.__class__.__name__)
            except Exception:
                md = None
        return md if isinstance(md, dict) else {}

    async def _get_advanced_features(self, **inputs) -> Dict[str, Any]:
        """
        Returns {'raw_features': np.ndarray} from:
        1) injected AFE.process(**inputs),
        2) InfoBus 'advanced_features',
        3) fallback synthetic vector.
        Also hot-swaps networks if feature length changes.
        """
        # 1) injected AFE instance
        if self._afe is not None and hasattr(self._afe, "process"):
            try:
                afe_out = await self._afe.process(**inputs)
                adv = afe_out.get("advanced_features", {})
                rf = adv.get("raw_features", [])
                vec = np.asarray(rf, dtype=np.float32)
                if vec.size > 0:
                    self._maybe_rebuild_for(vec.size)
                    return {"raw_features": vec}
            except Exception as e:
                self.logger.warning(f"AFE.process failed, falling back: {e}")

        # 2) InfoBus (contract key)
        try:
            bus_adv = self.smart_bus.get("advanced_features", self.__class__.__name__)
            if isinstance(bus_adv, dict):
                rf = bus_adv.get("raw_features")
                if isinstance(rf, (list, tuple)) and len(rf) > 0:
                    vec = np.asarray(rf, dtype=np.float32)
                    self._maybe_rebuild_for(vec.size)
                    return {"raw_features": vec}
        except Exception as e:
            self.logger.warning(f"Bus advanced_features unavailable: {e}")

        # 3) synthetic fallback (rare; keeps pipeline alive)
        vec = np.random.normal(0, 1, int(self.input_dim)).astype(np.float32)
        return {"raw_features": vec}

    def _maybe_rebuild_for(self, new_dim: int):
        if int(new_dim) != int(self.input_dim) and new_dim > 0:
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
    # Multiscale feature shaping (no non-contract reads)
    # ─────────────────────────────────────────────────────────
    async def _process_multiscale_features(self, afe_payload: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()

        # Base vector from upstream (sanitized)
        base = np.asarray(afe_payload.get("raw_features", []), dtype=np.float32)
        base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure each timeframe sees a vector of EXACTLY self.input_dim
        target_len = int(self.input_dim)

        def pad_trunc(v: np.ndarray, L: int) -> np.ndarray:
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.size < L:
                out = np.zeros(L, dtype=np.float32)
                out[:v.size] = v
                return out
            elif v.size > L:
                return v[-L:].astype(np.float32)
            return v.astype(np.float32)

        tfs = list(self._cfg.timeframes or ["H1", "H4", "D1"])
        timeframe_features: Dict[str, np.ndarray] = {}

        # Use the same base vector per timeframe (with tiny deterministic scaling to avoid perfect duplicates)
        for i, tf in enumerate(tfs):
            vec = pad_trunc(base, target_len)
            if vec.size > 0:
                vec = (vec * (1.0 + 0.02 * i)).astype(np.float32)
            timeframe_features[tf] = vec

        # Safe correlations across timeframes
        def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            n = a.size
            if n != b.size or n < 2:
                return 0.0
            a = a - a.mean()
            b = b - b.mean()
            sa = a.std()
            sb = b.std()
            if not (np.isfinite(sa) and np.isfinite(sb)) or sa < 1e-12 or sb < 1e-12:
                return 0.0
            corr = float(np.dot(a, b) / (sa * sb * n))
            return corr if np.isfinite(corr) else 0.0

        correlations: Dict[str, float] = {}
        tf_names = list(timeframe_features.keys())
        for i, a in enumerate(tf_names):
            for b in tf_names[i + 1:]:
                correlations[f"{a}_{b}"] = _safe_corr(timeframe_features[a], timeframe_features[b])

        md_fields = int(len(market_data) if isinstance(market_data, dict) else 0)

        return {
            "timeframe_features": timeframe_features,
            "correlations": correlations,
            "base_features": pad_trunc(base, target_len),
            "market_fields_seen": md_fields,
            "processing_time_ms": (time.time() - t0) * 1000.0
        }


    # ─────────────────────────────────────────────────────────
    # Neural forward
    # ─────────────────────────────────────────────────────────
    async def _neural_forward(self, ms_result: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()

        cfg_tfs = list(self._cfg.timeframes or ["H1", "H4", "D1"])
        available = list((ms_result.get("timeframe_features") or {}).keys())
        tfs = [tf for tf in cfg_tfs if tf in available] or available

        def _structured_fallback(reason: str) -> Dict[str, Any]:
            T = max(1, len(cfg_tfs))
            return {
                "embeddings": np.zeros((1, int(self.output_dim)), dtype=np.float32),
                "attention_weights": np.zeros((1, T, T), dtype=np.float32),  # batch=1
                "processed_features": {},
                "forward_time_ms": 0.0,
                "attention_entropy": 0.0,
                "_reason": reason
            }

        if not tfs:
            return _structured_fallback("no timeframe features available")

        processed: Dict[str, torch.Tensor] = {}
        for tf in tfs:
            try:
                vec = np.asarray(ms_result["timeframe_features"][tf], dtype=np.float32)
                x = torch.tensor(vec, dtype=torch.float32, device=self.device)
                if x.dim() == 1:
                    x = x.unsqueeze(0)  # [B, D]

                # Pylance-safe: use configured input size instead of indexing Sequential
                in_features = int(self.input_dim)

                if x.shape[-1] != in_features:
                    if x.shape[-1] < in_features:
                        pad = torch.zeros((x.shape[0], in_features - x.shape[-1]),
                                        device=self.device, dtype=x.dtype)
                        x = torch.cat([x, pad], dim=-1)
                    else:
                        x = x[..., -in_features:]

                processed[tf] = self.scale_processors[tf](x)  # [B, E]
            except Exception as e:
                self.logger.warning(f"Skipping timeframe {tf} in neural forward: {e}")
                continue

        if not processed:
            return _structured_fallback("no processed timeframe tensors available")

        ordered_tfs = list(processed.keys())
        concat = torch.cat([processed[tf] for tf in ordered_tfs], dim=-1)  # [B, E*T]
        fused = self.fusion_network(concat)                                # [B, E]

        attn_in = torch.stack([processed[tf] for tf in ordered_tfs], dim=1)  # [B, T, E]
        attn_out, attn_w = self.attention_fusion(attn_in)                    # [B, T, E], [B, T, T]

        combined = (fused + attn_out.mean(dim=1)) * 0.5                      # [B, E]

        embedding = combined.detach().cpu().numpy()        # [B, E]
        attn_weights = attn_w.detach().cpu().numpy()       # [B, T, T]
        forward_ms = (time.time() - t0) * 1000.0

        self.last_embedding = embedding.flatten()
        entropy = self._attention_entropy(attn_weights)
        self.attention_weights_history.append(attn_weights)
        self.embedding_history.append({
            "embedding": self.last_embedding.copy(),
            "timestamp": time.time(),
            "attention_entropy": float(entropy)
        })
        self._update_neural_stats(forward_ms, True)

        return {
            "embeddings": embedding,
            "attention_weights": attn_weights,
            "processed_features": {k: v.detach().cpu().numpy() for k, v in processed.items()},
            "forward_time_ms": forward_ms,
            "attention_entropy": float(entropy)
        }



    @staticmethod
    def _attention_entropy(attn_w: np.ndarray) -> float:
        try:
            w = attn_w.astype(np.float64).flatten()
            s = float(np.sum(w)) + 1e-12
            w = w / s
            return float(-np.sum(w * np.log(w + 1e-12)))
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────
    # Thesis / explainability (returned only; not bus)
    # ─────────────────────────────────────────────────────────
    async def _generate_neural_thesis(
        self,
        afe_payload: Dict[str, Any],
        ms_result: Dict[str, Any],
        nn_result: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> str:
        try:
            emb = np.asarray(nn_result["embeddings"]).flatten()
            q_emb = self._assess_embedding_quality(emb)
            attn_entropy = float(nn_result.get("attention_entropy", 0.0))
            fwd = float(nn_result.get("forward_time_ms", 0.0))
            device_info = f"Device: {self.device}"
            md_seen = int(ms_result.get("market_fields_seen", 0))

            return (
                "Neural Multi-Scale Feature Analysis\n"
                f"- Timeframes: {len((self._cfg.timeframes or []))}\n"
                f"- Embedding dim: {self.output_dim}\n"
                f"- Attention entropy: {attn_entropy:.3f}\n"
                f"- Forward time: {fwd:.1f} ms\n\n"
                "Quality\n"
                f"- Embedding quality: {q_emb:.1f}%\n"
                f"- Correlations computed: {len(ms_result.get('correlations', {}))}\n\n"
                "Inputs\n"
                f"- Base feature length: {int(np.asarray(afe_payload.get('raw_features', [])).size)}\n"
                f"- market_data fields seen: {md_seen}\n\n"
                "Confidence\n"
                f"- Neural processing: {'High' if q_emb > 80 else 'Medium' if q_emb > 60 else 'Low'}\n\n"
                f"{device_info}"
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
    # Bus (declared keys only)
    # ─────────────────────────────────────────────────────────
    def _update_bus(self, ms_result: Dict[str, Any], nn_result: Dict[str, Any]):
        # multiscale_features
        self.smart_bus.set(
            "multiscale_features",
            {
                "correlations": ms_result.get("correlations", {}),
                "timeframe_features": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in ms_result.get("timeframe_features", {}).items()
                },
                "processing_time_ms": float(ms_result.get("processing_time_ms", 0.0)),
            },
            module="MultiScaleFeatureEngine",
            thesis="Multiscale features published."
        )

        # neural_embeddings
        self.smart_bus.set(
            "neural_embeddings",
            {
                "embeddings": nn_result["embeddings"].tolist() if isinstance(nn_result["embeddings"], np.ndarray)
                else list(nn_result["embeddings"]),
                "dimensions": list(np.asarray(nn_result["embeddings"]).shape),
                "device": str(self.device),
                "timestamp": time.time()
            },
            module="MultiScaleFeatureEngine",
            thesis="Neural embeddings published."
        )

        # attention_weights
        self.smart_bus.set(
            "attention_weights",
            {
                "weights": nn_result["attention_weights"].tolist() if isinstance(nn_result["attention_weights"], np.ndarray)
                else list(nn_result["attention_weights"]),
                "entropy": float(nn_result.get("attention_entropy", 0.0)),
                "num_heads": int(self._cfg.num_attention_heads),
                "timeframes": list(self._cfg.timeframes or [])
            },
            module="MultiScaleFeatureEngine",
            thesis="Attention analysis."
        )

        # feature_fusion
        self.smart_bus.set(
            "feature_fusion",
            {
                "processed_features": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in nn_result.get("processed_features", {}).items()
                },
                "fusion_method": self._cfg.feature_fusion_method,
                "processing_time_ms": float(nn_result.get("forward_time_ms", 0.0)),
            },
            module="MultiScaleFeatureEngine",
            thesis="Feature fusion summary."
        )

        # neural_capabilities (publish per contract)
        self.smart_bus.set(
            "neural_capabilities",
            {
                "input_dim": int(getattr(self, "input_dim", 0)),
                "output_dim": int(getattr(self, "output_dim", 0)),
                "embed_dim": int(self._cfg.embed_dim),
                "num_attention_heads": int(self._cfg.num_attention_heads),
                "timeframes": list(self._cfg.timeframes or []),
                "feature_fusion_method": str(self._cfg.feature_fusion_method),
                "device": str(getattr(self, "device", "cpu")),
                "gpu_available": bool(torch.cuda.is_available()),
            },
            module="MultiScaleFeatureEngine",
            thesis="Neural capabilities."
        )

        # neural_health (publish per contract)
        self.smart_bus.set(
            "neural_health",
            {
                "model_health_score": float(self.neural_health.get("model_health_score", 0.0)),
                "neural_circuit_breaker_state": self.neural_circuit_breaker.get("state", "CLOSED"),
                "avg_forward_time_ms": float(self.neural_stats.get("avg_forward_time_ms", 0.0)),
                "successful_passes": int(self.neural_stats.get("successful_passes", 0)),
                "total_forward_passes": int(self.neural_stats.get("total_forward_passes", 0)),
            },
            module="MultiScaleFeatureEngine",
            thesis="Neural health snapshot."
        )

    # ─────────────────────────────────────────────────────────
    # Health / errors
    # ─────────────────────────────────────────────────────────
    def _check_neural_circuit_breaker(self) -> bool:
        state = self.neural_circuit_breaker["state"]
        if state == "OPEN":
            if time.time() - self.neural_circuit_breaker["last_failure"] > 120.0:
                self.neural_circuit_breaker["state"] = "HALF_OPEN"
                return True
            return False
        return True

    def _record_neural_success(self, processing_time_s: float):
        if self.neural_circuit_breaker["state"] == "HALF_OPEN":
            self.neural_circuit_breaker["state"] = "CLOSED"
            self.neural_circuit_breaker["failures"] = 0
        self.neural_health["model_health_score"] = min(100.0, self.neural_health["model_health_score"] + 2.0)
        if self.performance_tracker:
            self.performance_tracker.record_metric(
                "MultiScaleFeatureEngine", "neural_processing", processing_time_s * 1000.0, True
            )

    async def _handle_neural_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        elapsed = time.time() - start_time
        self._record_neural_failure(error)
        try:
            ctx = self.error_pinpointer.analyze_error(error, "MultiScaleFeatureEngine")
            _ = self.error_pinpointer.create_debugging_guide(ctx)
        except Exception:
            pass
        self.logger.error(
            format_operator_message(
                "🧠[CRASH]", "NEURAL_PROCESSING_ERROR",
                details=str(error),
                context="neural_processing"
            )
        )
        return self._create_neural_fallback_response(f"Neural processing failed: {error}")

    def _record_neural_failure(self, error: Exception):
        self.neural_circuit_breaker["failures"] += 1
        self.neural_circuit_breaker["last_failure"] = time.time()
        if self.neural_circuit_breaker["failures"] >= self.neural_circuit_breaker["threshold"]:
            self.neural_circuit_breaker["state"] = "OPEN"
            self.logger.error(
                format_operator_message(
                    "🧠[ALERT]", "NEURAL_CIRCUIT_BREAKER_OPEN",
                    details=f"Too many neural failures ({self.neural_circuit_breaker['failures']})",
                    context="neural_circuit_breaker"
                )
            )
        self.neural_health["model_health_score"] = max(0.0, self.neural_health["model_health_score"] - 15.0)
        self.neural_stats["failed_passes"] += 1

    def _create_neural_fallback_response(self, reason: str) -> Dict[str, Any]:
        T = len(self._cfg.timeframes or ["H1", "H4", "D1"])
        fallback_embedding = self.last_embedding if isinstance(self.last_embedding, np.ndarray) and self.last_embedding.size > 0 else np.zeros(self.output_dim, dtype=np.float32)
        fallback_attention = np.zeros((1, T, T))  # batch=1 to match attention_fusion output
        return self._format_declared_outputs(
            multiscale_features={},
            neural_embeddings=fallback_embedding,
            attention_weights=fallback_attention,
            feature_fusion={
                "processed_features": {},
                "fusion_method": self._cfg.feature_fusion_method,
                "processing_time_ms": 0.0,
            },
            thesis=f"Neural processing unavailable: {reason}. Using fallback embeddings.",
            extra={
                "processing_time_ms": 0.0,
                "device_used": str(self.device),
                "success": False,
                "reason": reason,
            },
        )

    def _update_neural_stats(self, forward_time_ms: float, success: bool):
        self.neural_stats["total_forward_passes"] += 1
        if success:
            self.neural_stats["successful_passes"] += 1
        avg = self.neural_stats["avg_forward_time_ms"]
        n = self.neural_stats["total_forward_passes"]
        self.neural_stats["avg_forward_time_ms"] = (avg * (n - 1) + forward_time_ms) / max(n, 1)
        if torch.cuda.is_available():
            self.neural_stats["gpu_memory_usage_mb"] = torch.cuda.memory_allocated() / 1024.0 / 1024.0

    # ─────────────────────────────────────────────────────────
    # Background monitors
    # ─────────────────────────────────────────────────────────
    async def _neural_health_monitoring_loop(self):
        while True:
            try:
                await asyncio.sleep(60)
                total = self.neural_stats["total_forward_passes"]
                if total > 0:
                    sr = self.neural_stats["successful_passes"] / total
                    if sr > 0.95:
                        self.neural_health["model_health_score"] = min(100.0, self.neural_health["model_health_score"] + 1.0)
                    elif sr < 0.8:
                        self.neural_health["model_health_score"] = max(0.0, self.neural_health["model_health_score"] - 2.0)
                self.neural_health["last_neural_check"] = time.time()
            except Exception as e:
                self.logger.error(f"Neural health monitoring error: {e}")

    async def _gpu_monitoring_loop(self):
        if not torch.cuda.is_available():
            return
        while True:
            try:
                await asyncio.sleep(30)
                mem_alloc = torch.cuda.memory_allocated() / 1024.0 / 1024.0
                self.neural_stats["gpu_memory_usage_mb"] = mem_alloc
                if mem_alloc > 1500.0:
                    self.logger.warning(
                        format_operator_message(
                            "🖥️[WARN]", "HIGH_GPU_MEMORY_USAGE",
                            details=f"Allocated: {mem_alloc:.1f}MB",
                            context="gpu_monitoring"
                        )
                    )
            except Exception as e:
                self.logger.error(f"GPU monitoring error: {e}")

    # ─────────────────────────────────────────────────────────
    # State & reports
    # ─────────────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        neural_state = {
            "config": {
                "embed_dim": self._cfg.embed_dim,
                "num_attention_heads": self._cfg.num_attention_heads,
                "timeframes": list(self._cfg.timeframes or []),
                "device": str(self.device),
                "assumed_input_dim": self._cfg.assumed_input_dim
            },
            "neural_data": {
                "last_embedding": self.last_embedding.tolist() if isinstance(self.last_embedding, np.ndarray) else [],
                "embedding_history": [
                    {**e, "embedding": e["embedding"].tolist()} for e in list(self.embedding_history)
                ],
                "attention_weights_history": [aw.tolist() for aw in list(self.attention_weights_history)]
            },
            "statistics": dict(self.neural_stats),
            "health_metrics": dict(self.neural_health),
            "circuit_breaker": dict(self.neural_circuit_breaker),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
        }
        return {**base, **neural_state}

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "neural_data" in state:
            nd = state["neural_data"]
            if "last_embedding" in nd:
                self.last_embedding = np.asarray(nd["last_embedding"], dtype=np.float32)
            if "embedding_history" in nd:
                self.embedding_history = deque(
                    [{**e, "embedding": np.asarray(e["embedding"], dtype=np.float32)} for e in nd["embedding_history"]],
                    maxlen=500
                )
            if "attention_weights_history" in nd:
                self.attention_weights_history = deque(
                    [np.asarray(x) for x in nd["attention_weights_history"]], maxlen=100
                )
        if "statistics" in state:
            self.neural_stats.update(state["statistics"])
        if "health_metrics" in state:
            self.neural_health.update(state["health_metrics"])
        if "circuit_breaker" in state:
            self.neural_circuit_breaker.update(state["circuit_breaker"])
        # Optionally rebuild if input_dim changed
        new_dim = int(state.get("input_dim", self.input_dim))
        if new_dim != int(self.input_dim) and new_dim > 0:
            self.input_dim = new_dim
            self._build_networks(self.input_dim)

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "model_health_score": self.neural_health["model_health_score"],
            "neural_circuit_breaker_state": self.neural_circuit_breaker["state"],
            "neural_statistics": dict(self.neural_stats),
            "device": str(self.device),
            "gpu_available": bool(torch.cuda.is_available()),
            "attention_quality": self.neural_health["attention_quality"],
            "embedding_quality": self.neural_health["embedding_quality"],
        }

    def get_neural_performance_report(self) -> str:
        try:
            return self.english_explainer.explain_performance(
                module_name="MultiScaleFeatureEngine",
                metrics={
                    "total_forward_passes": self.neural_stats["total_forward_passes"],
                    "neural_success_rate": self.neural_stats["successful_passes"] / max(self.neural_stats["total_forward_passes"], 1),
                    "avg_forward_time_ms": self.neural_stats["avg_forward_time_ms"],
                    "model_health_score": self.neural_health["model_health_score"],
                    "gpu_memory_usage_mb": self.neural_stats["gpu_memory_usage_mb"],
                    "attention_entropy": self.neural_stats["avg_attention_entropy"]
                }
            )
        except Exception as e:
            return f"Neural performance report generation failed: {e}"

    # ─────────────────────────────────────────────────────────
    # Actions (optional helper)
    # ─────────────────────────────────────────────────────────
    async def propose_action(self, **inputs) -> Dict[str, Any]:
        try:
            result = await self.process(**inputs)
            if not result.get("success", False):
                return {"action_type": "no_action", "confidence": 0.0, "reasoning": "Neural processing failed", "neural_available": False}

            embeddings = np.asarray(result["neural_embeddings"], dtype=float)
            attn = np.asarray(result["attention_weights"])
            if embeddings.size == 0:
                return {"action_type": "no_action", "confidence": 0.0, "reasoning": "No neural embeddings", "neural_available": False}

            mag = float(np.linalg.norm(embeddings))
            mean = float(np.mean(embeddings))
            focus = float(np.max(attn)) if attn.size > 0 else 0.0
            entropy = float(result.get("attention_entropy", 0.0))

            if self.neural_health["model_health_score"] > 80.0:
                if mean > 0.1 and focus > 0.5:
                    action_type, magnitude = "increase_neural_exposure", min(mag * 0.5, 1.0)
                elif mean < -0.1 and focus > 0.5:
                    action_type, magnitude = "decrease_neural_exposure", min(mag * 0.5, 1.0)
                elif entropy > 2.0:
                    action_type, magnitude = "reduce_neural_risk", 0.3
                else:
                    action_type, magnitude = "hold_neural_position", 0.0
            else:
                action_type, magnitude = "reduce_neural_risk", 0.7

            conf = min(self.neural_health["model_health_score"] / 100.0, 1.0)
            return {
                "action_type": action_type,
                "magnitude": float(magnitude),
                "confidence": float(conf),
                "reasoning": f"{embeddings.size} dims, |emb|={mag:.3f}, mean={mean:.3f}, focus={focus:.3f}",
                "embedding_magnitude": mag,
                "embedding_mean": mean,
                "attention_entropy": entropy,
                "attention_focus": focus,
                "neural_health": self.neural_health["model_health_score"]
            }

        except Exception as e:
            self.logger.error(f"Neural action proposal failed: {e}")
            return {"action_type": "no_action", "confidence": 0.0, "reasoning": f"Neural action proposal error: {e}", "error": str(e)}

    async def calculate_confidence(self, action: Dict[str, Any], **_inputs) -> float:
        try:
            if not isinstance(action, dict):
                return 0.0
            base = self.neural_health["model_health_score"] / 100.0
            a_type = action.get("action_type", "no_action")
            mag = float(action.get("magnitude", 0.0))

            if a_type in ("increase_neural_exposure", "decrease_neural_exposure"):
                strength = min(float(action.get("embedding_magnitude", 0.0)) * 2.0, 1.0)
                focus = float(action.get("attention_focus", 0.0))
                entropy = float(action.get("attention_entropy", 0.0))
                attn_conf = focus if entropy < 3.0 else focus * 0.5
                mag_conf = 1.0 - (mag * 0.3)
                combined = 0.4 * base + 0.3 * strength + 0.2 * attn_conf + 0.1 * mag_conf
            elif a_type == "hold_neural_position":
                combined = 0.7 * base
            elif a_type == "reduce_neural_risk":
                combined = max(base, 0.7)
            else:
                combined = 0.1

            if self.neural_circuit_breaker["state"] == "OPEN":
                combined *= 0.1
            elif self.neural_circuit_breaker["state"] == "HALF_OPEN":
                combined *= 0.5

            # Recent performance modifier
            total = self.neural_stats["total_forward_passes"]
            if total > 0:
                sr = self.neural_stats["successful_passes"] / total
                combined *= sr

            return float(np.clip(combined, 0.0, 1.0))
        except Exception:
            return 0.0
