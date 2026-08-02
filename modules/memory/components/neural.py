# modules/memory/components/neural.py
"""
Neural Memory Component
Implements attention-based memory with importance scoring.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MemoryComponent


class NeuralComponent(MemoryComponent):
    """Neural memory with attention mechanisms."""

    # Fraction of global memory reserved for neural buffer
    _BUFFER_FRACTION: float = 0.20
    # Top-K default for retrieval
    _DEFAULT_TOPK: int = 5
    # Minimal epsilon to avoid 0/0 etc.
    _EPS: float = 1e-8
    # Gentle decay applied per update tick
    _DEFAULT_DECAY: float = 0.995

    def _initialize_component(self) -> None:
        """Initialize neural-specific resources."""
        # Configuration (with safe fallbacks)
        cfg = self.config
        self.embed_dim: int = int(getattr(cfg, "embed_dim", 32))
        self.num_heads: int = int(getattr(cfg, "num_heads", 4))
        self.memory_decay: float = float(getattr(cfg, "memory_decay", 0.95))
        self.importance_threshold: float = float(getattr(cfg, "importance_threshold", 0.3))
        self.max_buffer_size: int = int(getattr(cfg, "max_memory_size", 10_000) * self._BUFFER_FRACTION)

        # Device (match shared encoder if present, else CPU)
        self._device = self._infer_device()

        # Memory buffer & side data (global - for backward compatibility)
        self.buffer: torch.Tensor = torch.zeros((0, self.embed_dim), dtype=torch.float32, device=self._device)
        self.importance_scores: torch.Tensor = torch.zeros(0, dtype=torch.float32, device=self._device)
        self.memory_metadata: List[Dict[str, Any]] = []
        
        # NEW: Per-instrument neural buffers
        # Maps instrument -> {buffer: Tensor, importance_scores: Tensor, metadata: List}
        self.buffers_by_instrument: Dict[str, Dict[str, Any]] = {}
        
        # Track which instruments we have sufficient data for
        self.instruments_with_data: set = set()

        # Neural submodules
        self._init_neural_networks()

        # Tracking
        self.memories_stored: int = 0
        self.memories_retrieved: int = 0
        self.avg_importance: float = 0.0
        self.attention_efficiency: float = 0.0
        self.retrieval_history: deque[Dict[str, Any]] = deque(maxlen=100)
        self.importance_evolution: deque[Dict[str, Any]] = deque(maxlen=500)
        self.attention_patterns: deque[Any] = deque(maxlen=50)
        self._last_decay_ts: float = time.time()

        self._log_debug(
            "neural_initialized",
            details={
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "max_buffer_size": self.max_buffer_size,
                "device": str(self._device),
            },
        )

    # -------------------------------------------------------------------------
    # Networks / initialization
    # -------------------------------------------------------------------------

    def _infer_device(self) -> torch.device:
        """Infer an execution device (prefer encoder's device if available)."""
        if self.encoder is not None:
            try:
                p = next(self.encoder.parameters(), None)
                if p is not None:
                    return p.device
            except Exception:
                pass
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _init_neural_networks(self) -> None:
        """Initialize neural network components."""
        try:
            # Encoder (use shared if available)
            if self.encoder is not None:
                self.memory_encoder: nn.Module = self.encoder
            else:
                self.memory_encoder = self._create_encoder()

            self.memory_encoder.to(self._device)

            # Multi-head attention (batch_first=True means [B, T, E])
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.1,
                batch_first=True,
                device=self._device,
            )

            # Importance head
            self.value_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim // 2, 1),
                nn.Sigmoid(),
            ).to(self._device)

            # Context integration (kept for future use)
            self.context_net = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            ).to(self._device)

            self._init_weights()
        except Exception as e:
            self.log_error("Neural network initialization failed", e)

    def _create_encoder(self) -> nn.Module:
        """Create a simple feed-forward encoder to the embedding space."""

        class Encoder(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim * 2, dim),
                    nn.LayerNorm(dim),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        return Encoder(self.embed_dim)

    def _init_weights(self) -> None:
        """Initialize linear layers with Xavier and zero bias."""
        for module in (self.value_head, self.context_net):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process neural memory operations."""
        try:
            storage_result = await self._store_experiences(context)

            # Optional retrieval - now supports per-instrument
            query = context.get("query")
            if query is not None:
                # Extract instrument for per-instrument retrieval
                instrument = self._extract_instrument_from_context(context)
                retrieval_result = await self._perform_retrieval(
                    query, 
                    top_k=int(context.get("top_k", self._DEFAULT_TOPK)),
                    instrument=instrument
                )
                storage_result.update(retrieval_result)

            # Periodic decay and metric updates
            self._apply_importance_decay()
            self._update_neural_metrics()

            return self._format_output(storage_result)
        except Exception as e:
            self.log_error("Neural processing failed", e)
            return self._get_fallback_output()
    
    def _extract_instrument_from_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract current instrument from context for per-instrument retrieval."""
        market_ctx = context.get("market_context", {}) or {}
        instrument = (
            market_ctx.get("instrument") or
            market_ctx.get("symbol") or
            context.get("instrument") or
            context.get("symbol")
        )
        if instrument:
            return str(instrument).upper().replace("/", "").replace("_", "")
        return None

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    async def _store_experiences(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store new experiences in neural memory (both global and per-instrument)."""
        experiences = context.get("experiences", []) or []
        stored_count = 0
        stored_by_instrument: Dict[str, int] = {}

        for exp in experiences[-10:]:  # process latest batch
            if not isinstance(exp, dict):
                continue

            features = self._extract_experience_features(exp, context)
            if features is None:
                continue
            
            # Extract instrument from experience or context
            instrument = self._extract_instrument(exp, context)

            encoded = await self._encode_experience(features)  # [E]
            importance = await self._calculate_importance(encoded, exp)

            if importance > self.importance_threshold:
                # Store in global buffer (backward compatibility)
                await self._add_to_buffer(encoded, importance, exp)
                
                # Store in per-instrument buffer
                await self._add_to_instrument_buffer(instrument, encoded, importance, exp)
                
                stored_count += 1
                stored_by_instrument[instrument] = stored_by_instrument.get(instrument, 0) + 1

        return {
            "storage_performed": stored_count > 0,
            "memories_stored": stored_count,
            "buffer_size": int(self.buffer.shape[0]),
            "stored_by_instrument": stored_by_instrument,
            "instruments_tracked": list(self.instruments_with_data),
        }

    def _extract_experience_features(self, exp: Dict[str, Any], context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract a fixed-length feature vector from an experience + context."""
        try:
            feats: List[float] = []

            # Observation
            if "observation" in exp:
                obs = exp["observation"]
                if isinstance(obs, np.ndarray):
                    feats.extend(obs.flatten()[: self.embed_dim // 2].tolist())
                elif isinstance(obs, (list, tuple)):
                    feats.extend(list(obs)[: self.embed_dim // 2])
                else:
                    feats.append(float(obs))

            # Reward
            feats.append(float(exp.get("reward", 0.0)))

            # Action
            act = exp.get("action", None)
            if isinstance(act, (list, tuple, np.ndarray)):
                feats.extend(np.array(act, dtype=np.float32).flatten()[:5].tolist())
            elif act is not None:
                feats.append(float(act))

            # Market context (lightweight)
            market_context = context.get("market_context", {}) or {}
            feats.append(float(market_context.get("volatility", 0.5)))

            # Pad/trim to embed_dim
            if len(feats) < self.embed_dim:
                feats.extend([0.0] * (self.embed_dim - len(feats)))
            else:
                feats = feats[: self.embed_dim]

            arr = np.asarray(feats, dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            return arr
        except Exception:
            return None

    async def _encode_experience(self, features: np.ndarray) -> torch.Tensor:
        """Encode an experience using the encoder into embedding space."""
        try:
            with torch.no_grad():
                t = torch.from_numpy(features).to(self._device).float().unsqueeze(0)  # [1, E]
                enc = self.memory_encoder(t)  # [1, E]
                return enc.squeeze(0)  # [E]
        except Exception:
            return torch.zeros(self.embed_dim, dtype=torch.float32, device=self._device)

    async def _calculate_importance(self, encoded: torch.Tensor, exp: Dict[str, Any]) -> float:
        """Compute importance score ∈ [0,1] using value head, reward, and novelty."""
        try:
            with torch.no_grad():
                base = self.value_head(encoded.unsqueeze(0))  # [1,1]
                importance = float(base.squeeze())

                # Reward adjustment
                reward = float(exp.get("reward", 0.0))
                if reward > 0:
                    importance *= 1.2
                elif reward < 0:
                    importance *= 0.8

                # Novelty adjustment (vs. existing buffer)
                if self.buffer.shape[0] > 0:
                    sims = F.cosine_similarity(encoded.unsqueeze(0), self.buffer, dim=1)  # [N]
                    max_sim = float(torch.clamp(sims.max(), -1.0, 1.0))
                    novelty = 1.0 - (max_sim + 1.0) / 2.0  # map [-1,1] -> [0,1] similarity, then invert
                    importance *= (0.5 + 0.5 * novelty)

                return float(np.clip(importance, 0.0, 1.0))
        except Exception:
            return 0.0

    async def _add_to_buffer(self, encoded: torch.Tensor, importance: float, exp: Dict[str, Any]) -> None:
        """Append to global neural memory and prune if needed."""
        try:
            # Append vectors
            self.buffer = torch.cat([self.buffer, encoded.unsqueeze(0)], dim=0)  # [N+1, E]
            self.importance_scores = torch.cat(
                [self.importance_scores, torch.tensor([importance], dtype=torch.float32, device=self._device)]
            )

            # Metadata
            self.memory_metadata.append(
                {"timestamp": time.time(), "importance": float(importance), "type": str(exp.get("type", "unknown"))}
            )

            # Prune if beyond capacity
            if int(self.buffer.shape[0]) > self.max_buffer_size:
                await self._prune_buffer()

            # Metrics
            self.memories_stored += 1
            self._update_importance_metrics(importance)
        except Exception as e:
            self.log_error("Buffer addition failed", e)
    
    async def _add_to_instrument_buffer(
        self, instrument: str, encoded: torch.Tensor, importance: float, exp: Dict[str, Any]
    ) -> None:
        """
        Add to per-instrument neural buffer.
        
        This ensures XAUUSD and EURUSD have separate attention memories,
        preventing cross-contamination of learned patterns.
        """
        try:
            # Initialize instrument buffer if needed
            if instrument not in self.buffers_by_instrument:
                self.buffers_by_instrument[instrument] = {
                    "buffer": torch.zeros((0, self.embed_dim), dtype=torch.float32, device=self._device),
                    "importance_scores": torch.zeros(0, dtype=torch.float32, device=self._device),
                    "metadata": [],
                }
            
            inst_data = self.buffers_by_instrument[instrument]
            
            # Append to instrument buffer
            inst_data["buffer"] = torch.cat([inst_data["buffer"], encoded.unsqueeze(0)], dim=0)
            inst_data["importance_scores"] = torch.cat(
                [inst_data["importance_scores"], torch.tensor([importance], dtype=torch.float32, device=self._device)]
            )
            inst_data["metadata"].append({
                "timestamp": time.time(),
                "importance": float(importance),
                "type": str(exp.get("type", "unknown")),
                "instrument": instrument,
            })
            
            # Track instruments with sufficient data
            if inst_data["buffer"].shape[0] >= 5:
                self.instruments_with_data.add(instrument)
            
            # Prune per-instrument buffer if needed (use same fraction of max_buffer_size)
            max_per_inst = max(100, self.max_buffer_size // 2)  # At least 100, or half of global
            if inst_data["buffer"].shape[0] > max_per_inst:
                await self._prune_instrument_buffer(instrument, max_per_inst)
                
        except Exception as e:
            self.log_error(f"Instrument buffer addition failed for {instrument}", e)
    
    async def _prune_instrument_buffer(self, instrument: str, max_size: int) -> None:
        """Prune per-instrument buffer keeping most important and recent."""
        try:
            if instrument not in self.buffers_by_instrument:
                return
            
            inst_data = self.buffers_by_instrument[instrument]
            n = inst_data["buffer"].shape[0]
            if n <= max_size:
                return
            
            n_keep = int(max(1, max_size * 0.8))
            scores = inst_data["importance_scores"]
            
            # Top by importance
            top_imp = torch.topk(scores, k=min(n_keep // 2, n)).indices
            
            # Most recent
            recent_start = max(0, n - (n_keep - len(top_imp)))
            recent = torch.arange(recent_start, n, device=self._device, dtype=torch.long)
            
            keep = torch.unique(torch.cat([top_imp, recent], dim=0)).sort().values
            
            inst_data["buffer"] = inst_data["buffer"].index_select(0, keep)
            inst_data["importance_scores"] = inst_data["importance_scores"].index_select(0, keep)
            
            keep_set = set(keep.tolist())
            inst_data["metadata"] = [m for i, m in enumerate(inst_data["metadata"]) if i in keep_set]
            
        except Exception as e:
            self.log_error(f"Instrument buffer pruning failed for {instrument}", e)
    
    def _extract_instrument(self, exp: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Extract instrument from experience or context."""
        # Try experience first
        instrument = (
            exp.get("instrument") or
            exp.get("symbol") or
            exp.get("metadata", {}).get("instrument")
        )
        
        # Fall back to context
        if not instrument:
            market_ctx = context.get("market_context", {}) or {}
            instrument = (
                market_ctx.get("instrument") or
                market_ctx.get("symbol") or
                context.get("instrument") or
                "UNKNOWN"
            )
        
        # Normalize
        return str(instrument).upper().replace("/", "").replace("_", "")

    async def _prune_buffer(self) -> None:
        """Prune to capacity, keeping most important and most recent items."""
        try:
            n = int(self.buffer.shape[0])
            if n <= self.max_buffer_size:
                return

            n_keep = int(max(1, self.max_buffer_size * 0.8))

            # Top by importance
            top_imp = torch.topk(self.importance_scores, k=min(n_keep // 2, n)).indices

            # Most recent indices
            recent_start = max(0, n - (n_keep - len(top_imp)))
            recent = torch.arange(recent_start, n, device=self._device, dtype=torch.long)

            keep = torch.unique(torch.cat([top_imp, recent], dim=0))
            keep = keep.sort().values  # deterministic order

            self.buffer = self.buffer.index_select(0, keep)
            self.importance_scores = self.importance_scores.index_select(0, keep)

            keep_set = set(keep.tolist())
            self.memory_metadata = [m for i, m in enumerate(self.memory_metadata) if i in keep_set]
        except Exception as e:
            self.log_error("Buffer pruning failed", e)

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    async def _perform_retrieval(self, query: Any, *, top_k: int, instrument: Optional[str] = None) -> Dict[str, Any]:
        """
        Attention-based retrieval for a query.
        
        Per-instrument retrieval: If instrument is specified and has sufficient data,
        retrieval is performed against that instrument's buffer only. This prevents
        XAUUSD patterns from influencing EURUSD decisions.
        
        Falls back to global buffer if per-instrument data is insufficient.
        """
        try:
            # Try per-instrument retrieval first
            if instrument:
                instrument = str(instrument).upper().replace("/", "").replace("_", "")
                if instrument in self.instruments_with_data:
                    result = await self._perform_instrument_retrieval(query, instrument, top_k)
                    if result.get("retrieval_performed"):
                        result["retrieval_type"] = "per_instrument"
                        result["instrument"] = instrument
                        return result
            
            # Fall back to global buffer
            if int(self.buffer.shape[0]) == 0:
                return {"retrieval_performed": False, "reason": "empty_buffer"}

            q = self._process_query(query)  # raw -> [E]
            if q is None:
                return {"retrieval_performed": False, "reason": "invalid_query"}

            # Encode query to the same space as memory embeddings
            with torch.no_grad():
                q_enc = self.memory_encoder(q.unsqueeze(0)).squeeze(0)  # [E]

            # Shapes for attention: [B, T, E]
            query_batch = q_enc.unsqueeze(0).unsqueeze(0)  # [1, 1, E]
            memory_batch = self.buffer.unsqueeze(0)        # [1, N, E]

            with torch.no_grad():
                attn_out, attn_weights = self.attention(query_batch, memory_batch, memory_batch)
                # attn_weights: [B, Q, N] -> [N]
                weights = attn_weights.squeeze(0).squeeze(0)
                weights = torch.clamp(weights, min=0.0)  # ensure non-negative
                if float(weights.sum()) <= self._EPS:
                    weights = torch.full_like(weights, 1.0 / max(1, weights.numel()))
                else:
                    weights = weights / (weights.sum() + self._EPS)

            # Top-K selection
            k = int(min(max(1, top_k), int(self.buffer.shape[0])))
            top_vals, top_idx = torch.topk(weights, k=k, largest=True, sorted=True)

            retrieved: List[Dict[str, Any]] = []
            sim_scores: List[float] = []

            for idx, w in zip(top_idx.tolist(), top_vals.tolist()):
                item = {
                    "embedding": self.buffer[idx].detach().cpu().numpy().tolist(),
                    "importance": float(self.importance_scores[idx].item()),
                    "metadata": self.memory_metadata[idx] if idx < len(self.memory_metadata) else {},
                    "attention_weight": float(w),
                }
                retrieved.append(item)
                sim_scores.append(float(w))

            # Metrics
            self.memories_retrieved += 1
            self.retrieval_history.append(
                {"timestamp": time.time(), "retrieved_count": len(retrieved), "avg_similarity": float(np.mean(sim_scores))}
            )
            
            # === NEW: neural_risk_hint for memory_vote composition ===
            # Low max attention = model is uncertain = higher risk
            max_attention = float(weights.max().item()) if weights.numel() > 0 else 0.0
            neural_risk_hint = 1.0 - max_attention  # High when attention is diffuse

            return {
                "retrieval_performed": True,
                "retrieval_type": "global",
                "retrieved_memories": retrieved,
                "similarity_scores": sim_scores,
                "attention_weights": weights.detach().cpu().numpy().tolist(),
                # New fields for memory_vote
                "neural_risk_hint": neural_risk_hint,
                "max_attention": max_attention,
            }
        except Exception as e:
            self.log_error("Retrieval failed", e)
            return {"retrieval_performed": False, "error": str(e)}
    
    async def _perform_instrument_retrieval(
        self, query: Any, instrument: str, top_k: int
    ) -> Dict[str, Any]:
        """
        Perform retrieval using only the specified instrument's buffer.
        
        This ensures attention patterns learned from XAUUSD don't influence
        EURUSD decisions and vice versa.
        """
        try:
            if instrument not in self.buffers_by_instrument:
                return {"retrieval_performed": False, "reason": f"no_buffer_for_{instrument}"}
            
            inst_data = self.buffers_by_instrument[instrument]
            buffer = inst_data["buffer"]
            
            if buffer.shape[0] == 0:
                return {"retrieval_performed": False, "reason": "empty_instrument_buffer"}
            
            q = self._process_query(query)
            if q is None:
                return {"retrieval_performed": False, "reason": "invalid_query"}
            
            # Encode query
            with torch.no_grad():
                q_enc = self.memory_encoder(q.unsqueeze(0)).squeeze(0)
            
            # Attention computation
            query_batch = q_enc.unsqueeze(0).unsqueeze(0)  # [1, 1, E]
            memory_batch = buffer.unsqueeze(0)              # [1, N, E]
            
            with torch.no_grad():
                attn_out, attn_weights = self.attention(query_batch, memory_batch, memory_batch)
                weights = attn_weights.squeeze(0).squeeze(0)
                weights = torch.clamp(weights, min=0.0)
                if float(weights.sum()) <= self._EPS:
                    weights = torch.full_like(weights, 1.0 / max(1, weights.numel()))
                else:
                    weights = weights / (weights.sum() + self._EPS)
            
            # Top-K selection
            k = int(min(max(1, top_k), buffer.shape[0]))
            top_vals, top_idx = torch.topk(weights, k=k, largest=True, sorted=True)
            
            retrieved: List[Dict[str, Any]] = []
            sim_scores: List[float] = []
            metadata_list = inst_data["metadata"]
            importance_scores = inst_data["importance_scores"]
            
            for idx, w in zip(top_idx.tolist(), top_vals.tolist()):
                item = {
                    "embedding": buffer[idx].detach().cpu().numpy().tolist(),
                    "importance": float(importance_scores[idx].item()),
                    "metadata": metadata_list[idx] if idx < len(metadata_list) else {},
                    "attention_weight": float(w),
                    "instrument": instrument,
                }
                retrieved.append(item)
                sim_scores.append(float(w))
            
            # Neural risk hint
            max_attention = float(weights.max().item()) if weights.numel() > 0 else 0.0
            neural_risk_hint = 1.0 - max_attention
            
            # Boost confidence for per-instrument retrieval (more reliable)
            max_attention = min(1.0, max_attention * 1.1)
            neural_risk_hint = max(0.0, neural_risk_hint * 0.9)
            
            return {
                "retrieval_performed": True,
                "retrieved_memories": retrieved,
                "similarity_scores": sim_scores,
                "attention_weights": weights.detach().cpu().numpy().tolist(),
                "neural_risk_hint": neural_risk_hint,
                "max_attention": max_attention,
                "instrument_buffer_size": buffer.shape[0],
            }
            
        except Exception as e:
            self.log_error(f"Instrument retrieval failed for {instrument}", e)
            return {"retrieval_performed": False, "error": str(e)}

    def _process_query(self, query: Any) -> Optional[torch.Tensor]:
        """Convert an arbitrary query to a length-E vector on the correct device."""
        try:
            if isinstance(query, torch.Tensor):
                t = query.detach().to(self._device).float()
            elif isinstance(query, np.ndarray):
                t = torch.from_numpy(query).to(self._device).float()
            elif isinstance(query, (list, tuple)):
                t = torch.tensor(list(query), device=self._device, dtype=torch.float32)
            else:
                return None

            if t.ndim > 1:
                t = t.reshape(-1)

            if t.numel() < self.embed_dim:
                pad = torch.zeros(self.embed_dim - t.numel(), device=self._device)
                t = torch.cat([t, pad], dim=0)
            elif t.numel() > self.embed_dim:
                t = t[: self.embed_dim]

            return t
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Metrics / decay
    # -------------------------------------------------------------------------

    def _apply_importance_decay(self) -> None:
        """Apply gentle exponential decay to stored importance scores."""
        if self.importance_scores.numel() == 0:
            return
        now = time.time()
        elapsed = max(0.0, now - self._last_decay_ts)
        # Map elapsed seconds to a compounded decay; use a mild per-second factor.
        per_sec = self._DEFAULT_DECAY
        decay_factor = float(per_sec ** elapsed)
        self.importance_scores.mul_(decay_factor)
        self._last_decay_ts = now

    def _update_importance_metrics(self, importance: float) -> None:
        """Update running average and history."""
        n = max(1, self.memories_stored)
        if n == 1:
            self.avg_importance = float(importance)
        else:
            self.avg_importance = float(((self.avg_importance * (n - 1)) + importance) / n)

        self.importance_evolution.append(
            {"timestamp": time.time(), "importance": float(importance), "avg": float(self.avg_importance)}
        )

    def _update_neural_metrics(self) -> None:
        """Update neural performance metrics."""
        n = int(self.buffer.shape[0])
        if n > 0:
            high = int((self.importance_scores > 0.7).sum().item())
            self.attention_efficiency = float(high / max(1, n))
        else:
            self.attention_efficiency = 0.0

    # -------------------------------------------------------------------------
    # Output / scoring
    # -------------------------------------------------------------------------

    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements."""
        if self.importance_scores.numel() > 0:
            avg = float(self.importance_scores.mean().item())
            mx = float(self.importance_scores.max().item())
            mn = float(self.importance_scores.min().item())
            std = float(self.importance_scores.std(unbiased=False).item())
            total = int(self.importance_scores.numel())
        else:
            avg = mx = mn = std = 0.0
            total = 0
        
        # Per-instrument stats
        instrument_stats: Dict[str, Dict[str, Any]] = {}
        for inst, data in self.buffers_by_instrument.items():
            buf = data["buffer"]
            scores = data["importance_scores"]
            if scores.numel() > 0:
                instrument_stats[inst] = {
                    "buffer_size": buf.shape[0],
                    "avg_importance": float(scores.mean().item()),
                    "max_importance": float(scores.max().item()),
                }
            else:
                instrument_stats[inst] = {
                    "buffer_size": 0,
                    "avg_importance": 0.0,
                    "max_importance": 0.0,
                }

        return {
            "attention_retrieval": {
                "retrieved_count": len(result.get("retrieved_memories", [])),
                "similarity_scores": result.get("similarity_scores", []),
                "top_k": self._DEFAULT_TOPK,
                "attention_heads": int(self.num_heads),
                "retrieval_type": result.get("retrieval_type", "none"),
            },
            "importance_scoring": {
                "average_importance": avg,
                "max_importance": mx,
                "min_importance": mn,
                "std_importance": std,
                "total_scored": total,
            },
            "memory_embedding": {
                "embedding_dim": int(self.embed_dim),
                "total_embeddings": int(self.buffer.shape[0]),
                "importance_threshold": float(self.importance_threshold),
                "decay_rate": float(self.memory_decay),
                "instruments_tracked": list(self.instruments_with_data),
            },
            "neural_memory": {
                "buffer_size": int(self.buffer.shape[0]),
                "memory_utilization": float(self.buffer.shape[0] / max(1, self.max_buffer_size)),
                "average_importance": float(self.avg_importance),
                "neural_performance_score": float(self._calculate_performance_score()),
                "last_updated": time.time(),
                "per_instrument_stats": instrument_stats,
            },
            # Include retrieval signals for memory_vote composition
            "neural_risk_hint": float(result.get("neural_risk_hint", 0.5)),
            "max_attention": float(result.get("max_attention", 0.5)),
        }

    def _calculate_performance_score(self) -> float:
        """Compute a composite neural performance score ∈ [0,100]."""
        util = float(self.buffer.shape[0] / max(1, self.max_buffer_size))
        util_score = max(0.0, 1.0 - abs(util - 0.7) * 2.0)  # best near 0.7
        imp_score = float(self.avg_importance)
        eff_score = float(self.attention_efficiency)

        score = (0.3 * util_score + 0.4 * imp_score + 0.3 * eff_score) * 100.0
        score = float(np.clip(score, 0.0, 100.0))
        return score

    def _get_fallback_output(self) -> Dict[str, Any]:
        """Conservative payload on error."""
        return {
            "attention_retrieval": {
                "retrieved_count": 0,
                "similarity_scores": [],
                "top_k": self._DEFAULT_TOPK,
                "attention_heads": int(self.num_heads),
            },
            "importance_scoring": {
                "average_importance": 0.0,
                "max_importance": 0.0,
                "min_importance": 0.0,
                "std_importance": 0.0,
                "total_scored": 0,
            },
            "memory_embedding": {
                "embedding_dim": int(self.embed_dim),
                "total_embeddings": 0,
                "importance_threshold": float(self.importance_threshold),
                "decay_rate": float(self.memory_decay),
            },
            "neural_memory": {
                "buffer_size": 0,
                "memory_utilization": 0.0,
                "average_importance": 0.0,
                "neural_performance_score": 0.0,
                "last_updated": 0.0,
            },
            # Fallback values for memory_vote composition
            "neural_risk_hint": 0.5,
            "max_attention": 0.5,
        }
