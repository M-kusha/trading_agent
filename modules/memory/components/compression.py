# modules/memory/components/compression.py
"""
Memory Compression Component
Compresses experiences using PCA and (optionally) a shared neural encoder.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base import MemoryComponent


class CompressionComponent(MemoryComponent):
    """Memory compression using PCA and (optional) neural encoding."""

    # Minimum samples required to attempt PCA
    _MIN_PCA_SAMPLES: int = 2
    # Fraction of max_memory_size reserved for compression buffers
    _BUFFER_FRACTION: float = 0.10

    def _initialize_component(self) -> None:
        """Initialize compression-specific resources and state."""
        # Configuration
        self.n_components: int = int(getattr(self.config, "n_components", 8))
        self.compression_ratio: float = float(getattr(self.config, "compression_ratio", 0.7))
        self.compress_interval: int = int(getattr(self.config, "compress_interval", 10))
        self.max_memory_size: int = int(getattr(self.config, "max_memory_size", 10_000))
        self.replay_profit_threshold: float = float(
            getattr(self.config, "replay_profit_threshold", 10.0)
        )

        # Memory buffers (store tuples (features: np.ndarray, weight: float))
        self.profit_memory: List[tuple[np.ndarray, float]] = []
        self.loss_memory: List[tuple[np.ndarray, float]] = []
        
        # NEW: Per-instrument memory buffers
        # Maps instrument -> {profit: [(features, weight)], loss: [(features, weight)]}
        self.memory_by_instrument: Dict[str, Dict[str, List[tuple[np.ndarray, float]]]] = {}
        
        # NEW: Per-instrument PCA models and directions
        # Maps instrument -> {profit_pca, loss_pca, profit_direction, loss_direction, intuition_vector, ...}
        self.compression_by_instrument: Dict[str, Dict[str, Any]] = {}
        
        # Track which instruments have sufficient data for compression
        self.instruments_with_data: set = set()

        # Compressed representations (always length n_components; padded as needed)
        self.intuition_vector: np.ndarray = np.zeros(self.n_components, dtype=np.float32)
        self.profit_direction: np.ndarray = np.zeros(self.n_components, dtype=np.float32)
        self.loss_direction: np.ndarray = np.zeros(self.n_components, dtype=np.float32)

        # Separate scalers to avoid distribution leakage between profit/loss streams
        self._profit_scaler = StandardScaler()
        self._loss_scaler = StandardScaler()
        self._profit_scaler_fitted = False
        self._loss_scaler_fitted = False

        # PCA models (will be re-instantiated with effective n_components as needed)
        self.profit_pca: Optional[PCA] = None
        self.loss_pca: Optional[PCA] = None
        self._profit_pca_fitted = False
        self._loss_pca_fitted = False

        # Tracking / quality metrics
        self.compression_count: int = 0
        self.compression_quality_scores: deque[float] = deque(maxlen=50)
        self.explained_variance_history: deque[float] = deque(maxlen=50)
        self.compression_efficiency: float = 0.0

        self._log_debug(
            "compression_initialized",
            details={
                "n_components": self.n_components,
                "compress_interval": self.compress_interval,
                "buffer_fraction": self._BUFFER_FRACTION,
            },
        )

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory compression."""
        try:
            memory_data = context.get("memory_data", [])
            trades = context.get("trades", [])
            episode = int(context.get("episode", 0))

            self._update_memory_buffers(memory_data, trades)

            result: Dict[str, Any]
            if self._should_compress(episode):
                result = self._perform_compression()
            else:
                result = {"compression_performed": False}

            # Update intuition vector regardless
            result.update(self._update_intuition_vector())

            return self._format_output(result)
        except Exception as e:
            self.log_error("Compression processing failed", e)
            return self._get_fallback_output()

    # -------------------------------------------------------------------------
    # Buffer management
    # -------------------------------------------------------------------------

    def _update_memory_buffers(self, memory_data: List[Dict[str, Any]], trades: List[Dict[str, Any]]) -> None:
        """Update profit and loss buffers from store data and recent trades (global + per-instrument)."""
        # From memory_data (already structured experiences)
        for entry in memory_data:
            pnl = entry.get("pnl")
            feats = entry.get("features")
            if pnl is None or feats is None:
                continue
            features = self._to_feature_vector(feats)
            if features is None:
                continue
            
            # Extract instrument
            instrument = self._extract_instrument(entry)
            
            if pnl > self.replay_profit_threshold:
                self.profit_memory.append((features, float(pnl)))
                self._add_to_instrument_buffer(instrument, "profit", features, float(pnl))
            elif pnl < -self.replay_profit_threshold / 2.0:
                self.loss_memory.append((features, float(abs(pnl))))
                self._add_to_instrument_buffer(instrument, "loss", features, float(abs(pnl)))

        # From latest trades (derive features via extractor)
        for trade in trades[-10:]:
            if not isinstance(trade, dict) or "pnl" not in trade:
                continue
            pnl = float(trade["pnl"])
            derived = self.extractor.extract_trade_features(trade, {})
            features = self._to_feature_vector(derived)
            if features is None:
                continue
            
            # Extract instrument from trade
            instrument = self._extract_instrument(trade)
            
            if pnl > self.replay_profit_threshold:
                self.profit_memory.append((features, pnl))
                self._add_to_instrument_buffer(instrument, "profit", features, pnl)
            elif pnl < -self.replay_profit_threshold / 2.0:
                self.loss_memory.append((features, float(abs(pnl))))
                self._add_to_instrument_buffer(instrument, "loss", features, float(abs(pnl)))

        # Bound global memory sizes
        max_size = int(self.max_memory_size * self._BUFFER_FRACTION)
        if len(self.profit_memory) > max_size:
            self.profit_memory = self.profit_memory[-max_size:]
        if len(self.loss_memory) > max_size:
            self.loss_memory = self.loss_memory[-max_size:]
        
        # Bound per-instrument buffers
        self._bound_instrument_buffers(max_size // 2)  # Half of global per instrument

        self._log_debug(
            "buffers_updated",
            details={
                "profit_len": len(self.profit_memory),
                "loss_len": len(self.loss_memory),
                "max_size": max_size,
                "instruments_tracked": list(self.instruments_with_data),
            },
        )
    
    def _extract_instrument(self, entry: Dict[str, Any]) -> str:
        """Extract instrument from entry."""
        instrument = (
            entry.get("instrument") or
            entry.get("symbol") or
            entry.get("metadata", {}).get("instrument") or
            entry.get("context", {}).get("instrument") or
            "UNKNOWN"
        )
        return str(instrument).upper().replace("/", "").replace("_", "")
    
    def _add_to_instrument_buffer(
        self, instrument: str, stream: str, features: np.ndarray, weight: float
    ) -> None:
        """
        Add to per-instrument buffer.
        
        This ensures XAUUSD and EURUSD have separate PCA compression,
        learning distinct profit/loss patterns for each instrument.
        """
        if instrument not in self.memory_by_instrument:
            self.memory_by_instrument[instrument] = {"profit": [], "loss": []}
        
        self.memory_by_instrument[instrument][stream].append((features, weight))
        
        # Track instruments with enough data for compression
        inst_data = self.memory_by_instrument[instrument]
        total = len(inst_data["profit"]) + len(inst_data["loss"])
        if total >= self._MIN_PCA_SAMPLES * 2:
            self.instruments_with_data.add(instrument)
    
    def _bound_instrument_buffers(self, max_per_instrument: int) -> None:
        """Bound per-instrument buffers to prevent memory bloat."""
        for instrument, data in self.memory_by_instrument.items():
            if len(data["profit"]) > max_per_instrument:
                data["profit"] = data["profit"][-max_per_instrument:]
            if len(data["loss"]) > max_per_instrument:
                data["loss"] = data["loss"][-max_per_instrument:]

    def _first_linear_in_features(self, model) -> Optional[int]:
        """Find the first module that exposes `in_features` (typically nn.Linear)."""
        try:
            for m in model.modules():
                if hasattr(m, "in_features"):
                    return int(m.in_features)
        except Exception:
            pass
        return None

    def _warn_once(self, key: str, msg: str) -> None:
        if not hasattr(self, "_once_flags"):
            self._once_flags = set()
        if key not in self._once_flags:
            self._once_flags.add(key)
            # Use component's debug logger to avoid attribute errors
            try:
                self._log_debug("warning", details={"message": msg})
            except Exception:
                pass


    def _maybe_reset_feature_dim(self, new_dim: int) -> None:
        """
        If the incoming feature width changes, reset state so PCA/Scaler don’t mix spaces.
        This is safer than padding/trimming to an old width after an upstream schema change.
        """
        try:
            cur = getattr(self, "_feat_dim", None)
            if cur is None:
                self._feat_dim = int(new_dim)
                return
            if int(new_dim) != int(cur):
                self._feat_dim = int(new_dim)
                # clear buffers
                self.profit_memory.clear()
                self.loss_memory.clear()
                # reset scalers/PCA flags & objects
                self._profit_scaler = StandardScaler()
                self._loss_scaler = StandardScaler()
                self._profit_scaler_fitted = False
                self._loss_scaler_fitted = False
                self.profit_pca = None
                self.loss_pca = None
                self._profit_pca_fitted = False
                self._loss_pca_fitted = False
                self._warn_once("feat_dim_reset", f"Feature dimension changed → {new_dim}. Buffers & PCA reset.")
        except Exception:
            pass

    def _to_feature_vector(self, x: Any) -> Optional[np.ndarray]:
        """Convert payload into a 1D float32 vector; normalize to encoder input size; encode if available; sanitize NaNs; enforce dimension consistency."""
        try:
            import torch

            # 1) make it 1D float32 & sanitize
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim == 0:
                return None
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

            # 2) pick expected encoder input size (fallback to config.embed_dim)
            target_dim = int(getattr(self.config, "embed_dim", 32))
            expect_in = target_dim
            if self.encoder is not None:
                enc_in = self._first_linear_in_features(self.encoder)
                if enc_in and enc_in > 0:
                    expect_in = enc_in

            # 3) adjust input to encoder (pad/trim) so matmul cannot fail
            if arr.size != expect_in:
                self._warn_once("feat_pad_trim", f"Adjusted feature length {arr.size} → {expect_in}")
                arr = self._pad_or_trim(arr, expect_in)

            # 4) encode (if encoder is present)
            if self.encoder is not None:
                with torch.no_grad():
                    t = torch.from_numpy(arr).unsqueeze(0)  # [1, F]
                    # optional: route to encoder device if it has parameters
                    try:
                        params = list(self.encoder.parameters())
                        if params:
                            t = t.to(params[0].device)
                    except Exception:
                        pass
                    enc = self.encoder(t)
                    if hasattr(enc, "detach"):
                        enc = enc.detach()
                    arr = enc
                    # back to cpu numpy
                    try:
                        arr = arr.to("cpu")
                    except Exception:
                        pass
                    arr = arr.numpy().reshape(-1).astype(np.float32, copy=False)

            # 5) sanitize again post-encoder (defensive) and lock dimension for PCA stack
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            self._maybe_reset_feature_dim(arr.size)
            return arr

        except Exception as e:
            self.log_error("Feature conversion/encoding failed", e)
            return None


    # -------------------------------------------------------------------------
    # Compression
    # -------------------------------------------------------------------------

    def _should_compress(self, episode: int) -> bool:
        """Check if compression should be performed."""
        if episode <= 0 or self.compress_interval <= 0:
            return False
        if episode % self.compress_interval != 0:
            return False
        # Need at least a couple of samples to run PCA safely
        return len(self.profit_memory) >= max(self._MIN_PCA_SAMPLES, self.n_components)

    def _perform_compression(self) -> Dict[str, Any]:
        """Run PCA compression for profit and loss memories (global + per-instrument)."""
        results: Dict[str, Any] = {"compression_performed": True}

        # Global compression (backward compatibility)
        if len(self.profit_memory) >= self._MIN_PCA_SAMPLES:
            results.update(self._compress_stream(self.profit_memory, stream="profit"))

        if len(self.loss_memory) >= self._MIN_PCA_SAMPLES:
            results.update(self._compress_stream(self.loss_memory, stream="loss"))
        
        # Per-instrument compression
        per_instrument_results: Dict[str, Dict[str, Any]] = {}
        for instrument in self.instruments_with_data:
            inst_result = self._compress_instrument(instrument)
            if inst_result:
                per_instrument_results[instrument] = inst_result
        
        results["per_instrument_compression"] = per_instrument_results

        self.compression_count += 1
        return results
    
    def _compress_instrument(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Run PCA compression for a specific instrument.
        
        This creates instrument-specific intuition vectors, allowing the system
        to learn distinct patterns for XAUUSD vs EURUSD.
        """
        if instrument not in self.memory_by_instrument:
            return None
        
        inst_data = self.memory_by_instrument[instrument]
        profit_mem = inst_data["profit"]
        loss_mem = inst_data["loss"]
        
        if len(profit_mem) < self._MIN_PCA_SAMPLES and len(loss_mem) < self._MIN_PCA_SAMPLES:
            return None
        
        # Initialize compression state for this instrument if needed
        if instrument not in self.compression_by_instrument:
            self.compression_by_instrument[instrument] = {
                "profit_scaler": StandardScaler(),
                "loss_scaler": StandardScaler(),
                "profit_scaler_fitted": False,
                "loss_scaler_fitted": False,
                "profit_pca": None,
                "loss_pca": None,
                "profit_pca_fitted": False,
                "loss_pca_fitted": False,
                "profit_direction": np.zeros(self.n_components, dtype=np.float32),
                "loss_direction": np.zeros(self.n_components, dtype=np.float32),
                "intuition_vector": np.zeros(self.n_components, dtype=np.float32),
            }
        
        comp = self.compression_by_instrument[instrument]
        result: Dict[str, Any] = {"instrument": instrument}
        
        try:
            # Compress profit stream for this instrument
            if len(profit_mem) >= self._MIN_PCA_SAMPLES:
                profit_result = self._compress_stream_with_state(
                    profit_mem, 
                    comp["profit_scaler"], 
                    comp["profit_scaler_fitted"],
                    "profit"
                )
                if "direction" in profit_result:
                    comp["profit_direction"] = profit_result["direction"]
                    comp["profit_pca"] = profit_result.get("pca")
                    comp["profit_pca_fitted"] = True
                    comp["profit_scaler_fitted"] = True
                    result["profit"] = {
                        "samples": profit_result["samples"],
                        "explained_variance": profit_result["explained_variance"],
                        "strength": float(np.linalg.norm(comp["profit_direction"])),
                    }
            
            # Compress loss stream for this instrument
            if len(loss_mem) >= self._MIN_PCA_SAMPLES:
                loss_result = self._compress_stream_with_state(
                    loss_mem,
                    comp["loss_scaler"],
                    comp["loss_scaler_fitted"],
                    "loss"
                )
                if "direction" in loss_result:
                    comp["loss_direction"] = loss_result["direction"]
                    comp["loss_pca"] = loss_result.get("pca")
                    comp["loss_pca_fitted"] = True
                    comp["loss_scaler_fitted"] = True
                    result["loss"] = {
                        "samples": loss_result["samples"],
                        "explained_variance": loss_result["explained_variance"],
                        "strength": float(np.linalg.norm(comp["loss_direction"])),
                    }
            
            # Update intuition vector for this instrument
            self._update_instrument_intuition(instrument)
            result["intuition_strength"] = float(np.linalg.norm(comp["intuition_vector"]))
            
            return result
            
        except Exception as e:
            self.log_error(f"Instrument compression failed for {instrument}", e)
            return None
    
    def _compress_stream_with_state(
        self, 
        mem: List[tuple[np.ndarray, float]], 
        scaler: StandardScaler,
        scaler_fitted: bool,
        stream: str
    ) -> Dict[str, Any]:
        """
        Compress a memory stream with provided scaler state.
        Returns direction, PCA model, and stats.
        """
        try:
            feats = np.stack([m[0] for m in mem], axis=0)
            weights_raw = np.asarray([m[1] for m in mem], dtype=np.float32)
            
            w_sum = float(weights_raw.sum())
            if w_sum <= 0.0 or not np.isfinite(w_sum):
                weights = np.ones_like(weights_raw, dtype=np.float32)
            else:
                weights = weights_raw / w_sum
            
            weighted = feats * weights[:, None]
            
            if not scaler_fitted:
                standardized = scaler.fit_transform(weighted)
            else:
                standardized = scaler.transform(weighted)
            
            n_samples, n_features = standardized.shape
            n_eff = max(1, min(self.n_components, n_samples, n_features))
            
            pca = PCA(n_components=n_eff, svd_solver="auto", random_state=0)
            pca.fit(standardized)
            compressed = pca.transform(standardized)
            
            direction_eff = np.average(compressed, axis=0, weights=weights).astype(np.float32)
            direction_full = self._pad_or_trim(direction_eff, self.n_components)
            
            explained = float(np.sum(pca.explained_variance_ratio_))
            
            return {
                "direction": direction_full,
                "pca": pca,
                "samples": n_samples,
                "explained_variance": explained,
            }
            
        except Exception as e:
            self.log_error(f"{stream.capitalize()} compression with state failed", e)
            return {}
    
    def _update_instrument_intuition(self, instrument: str) -> None:
        """Update intuition vector for a specific instrument."""
        if instrument not in self.compression_by_instrument:
            return
        
        comp = self.compression_by_instrument[instrument]
        profit_dir = comp["profit_direction"]
        loss_dir = comp["loss_direction"]
        
        profit_strength = float(np.linalg.norm(profit_dir))
        loss_strength = float(np.linalg.norm(loss_dir))
        
        if profit_strength > 0.0 and loss_strength > 0.0:
            profit_component = profit_dir * 2.0
            loss_component = -loss_dir * 1.5
            combined = profit_component + loss_component
            
            learning_rate = 0.10
            current = comp["intuition_vector"]
            vec = (1.0 - learning_rate) * current + learning_rate * combined
            norm = float(np.linalg.norm(vec))
            comp["intuition_vector"] = (vec / norm).astype(np.float32) if norm > 1e-8 else vec.astype(np.float32)
        elif profit_strength > 0.0:
            comp["intuition_vector"] = profit_dir.astype(np.float32, copy=True)
    
    def get_instrument_intuition(self, instrument: str) -> Optional[np.ndarray]:
        """
        Get the intuition vector for a specific instrument.
        
        Use this for per-instrument decision making.
        """
        instrument = str(instrument).upper().replace("/", "").replace("_", "")
        if instrument in self.compression_by_instrument:
            return self.compression_by_instrument[instrument]["intuition_vector"].copy()
        return None

    def _compress_stream(self, mem: List[tuple[np.ndarray, float]], *, stream: str) -> Dict[str, Any]:
        """
        Compress a memory stream ('profit' or 'loss') with separate scaler/PCA.
        Returns a dict with {f"{stream}_compression": {...}}.
        """
        try:
            feats = np.stack([m[0] for m in mem], axis=0)  # [N, F]
            weights_raw = np.asarray([m[1] for m in mem], dtype=np.float32)  # [N]

            # Normalize weights; fall back to uniform if degenerate
            w_sum = float(weights_raw.sum())
            if w_sum <= 0.0 or not np.isfinite(w_sum):
                weights = np.ones_like(weights_raw, dtype=np.float32)
            else:
                weights = weights_raw / w_sum

            # Weighted features
            weighted = feats * weights[:, None]

            # Standardize with stream-specific scaler
            if stream == "profit":
                scaler = self._profit_scaler
                if not self._profit_scaler_fitted:
                    standardized = scaler.fit_transform(weighted)
                    self._profit_scaler_fitted = True
                else:
                    standardized = scaler.transform(weighted)
            else:
                scaler = self._loss_scaler
                if not self._loss_scaler_fitted:
                    standardized = scaler.fit_transform(weighted)
                    self._loss_scaler_fitted = True
                else:
                    standardized = scaler.transform(weighted)

            # Determine effective n_components
            n_samples, n_features = standardized.shape
            n_eff = max(1, min(self.n_components, n_samples, n_features))

            pca = PCA(n_components=n_eff, svd_solver="auto", random_state=0)
            pca.fit(standardized)
            compressed = pca.transform(standardized)  # [N, n_eff]

            # Direction: weighted average in compressed space
            direction_eff = np.average(compressed, axis=0, weights=weights).astype(np.float32)

            # Pad/resize to declared n_components for stable contract
            direction_full = self._pad_or_trim(direction_eff, self.n_components)

            # Save to correct stream slots
            explained = float(np.sum(pca.explained_variance_ratio_))
            if stream == "profit":
                self.profit_pca = pca
                self._profit_pca_fitted = True
                self.profit_direction = direction_full
                self.explained_variance_history.append(explained)
                avg = float(np.mean(weights_raw)) if weights_raw.size else 0.0
                return {
                    "profit_compression": {
                        "samples_compressed": int(n_samples),
                        "explained_variance": explained,
                        "profit_direction_strength": float(np.linalg.norm(direction_full)),
                        "avg_profit": avg,
                    }
                }
            else:
                self.loss_pca = pca
                self._loss_pca_fitted = True
                self.loss_direction = direction_full
                avg = float(np.mean(weights_raw)) if weights_raw.size else 0.0
                return {
                    "loss_compression": {
                        "samples_compressed": int(n_samples),
                        "explained_variance": explained,
                        "loss_direction_strength": float(np.linalg.norm(direction_full)),
                        "avg_loss": avg,
                    }
                }

        except Exception as e:
            self.log_error(f"{stream.capitalize()} compression failed", e)
            key = f"{stream}_compression"
            return {key: {"error": str(e)}}

    # -------------------------------------------------------------------------
    # Intuition vector
    # -------------------------------------------------------------------------

    def _update_intuition_vector(self) -> Dict[str, Any]:
        """Blend profit and loss directions into an intuition vector with smoothing."""
        profit_strength = float(np.linalg.norm(self.profit_direction))
        loss_strength = float(np.linalg.norm(self.loss_direction))

        if profit_strength > 0.0 and loss_strength > 0.0:
            profit_component = self.profit_direction * 2.0  # prefer profit patterns
            loss_component = -self.loss_direction * 1.5     # avoid loss patterns
            combined = profit_component + loss_component

            # EMA style update
            learning_rate = 0.10
            vec = (1.0 - learning_rate) * self.intuition_vector + learning_rate * combined
            norm = float(np.linalg.norm(vec))
            self.intuition_vector = (vec / norm).astype(np.float32) if norm > 1e-8 else vec.astype(np.float32)
        elif profit_strength > 0.0:
            self.intuition_vector = self.profit_direction.astype(np.float32, copy=True)

        return {
            "intuition_update": {
                "intuition_strength": float(np.linalg.norm(self.intuition_vector)),
                "profit_strength": profit_strength,
                "loss_strength": loss_strength,
            }
        }

    # -------------------------------------------------------------------------
    # Output / fallback
    # -------------------------------------------------------------------------

    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements."""
        if self.explained_variance_history:
            self.compression_efficiency = float(np.mean(list(self.explained_variance_history)))

        feature_importance: Dict[str, Any] = {}
        if self._profit_pca_fitted and self.profit_pca is not None and hasattr(self.profit_pca, "components_"):
            feature_importance = {
                "profit_components": self.profit_pca.components_.tolist(),
                "explained_variance_ratio": self.profit_pca.explained_variance_ratio_.tolist(),
                "n_features": int(getattr(self.profit_pca, "n_features_in_", 0)),
            }
        
        # Per-instrument stats
        per_instrument_stats: Dict[str, Dict[str, Any]] = {}
        for inst, comp in self.compression_by_instrument.items():
            per_instrument_stats[inst] = {
                "profit_strength": float(np.linalg.norm(comp["profit_direction"])),
                "loss_strength": float(np.linalg.norm(comp["loss_direction"])),
                "intuition_strength": float(np.linalg.norm(comp["intuition_vector"])),
                "profit_samples": len(self.memory_by_instrument.get(inst, {}).get("profit", [])),
                "loss_samples": len(self.memory_by_instrument.get(inst, {}).get("loss", [])),
            }

        return {
            "compressed_patterns": {
                "profit_direction": self.profit_direction.tolist(),
                "loss_direction": self.loss_direction.tolist(),
                "profit_strength": float(np.linalg.norm(self.profit_direction)),
                "loss_strength": float(np.linalg.norm(self.loss_direction)),
                "compression_count": int(self.compression_count),
                "instruments_tracked": list(self.instruments_with_data),
            },
            "feature_importance": feature_importance,
            "intuition_vector": {
                "vector": self.intuition_vector.tolist(),
                "strength": float(np.linalg.norm(self.intuition_vector)),
                "components": int(self.n_components),
                "last_updated": time.time(),
                "per_instrument": {
                    inst: comp["intuition_vector"].tolist() 
                    for inst, comp in self.compression_by_instrument.items()
                },
            },
            "memory_compression": {
                "total_memories": int(len(self.profit_memory) + len(self.loss_memory)),
                "profit_memories": len(self.profit_memory),
                "loss_memories": len(self.loss_memory),
                "compression_efficiency": float(self.compression_efficiency),
                "last_compression": int(self.compression_count),
                "per_instrument_stats": per_instrument_stats,
            },
        }

    def _get_fallback_output(self) -> Dict[str, Any]:
        """Get fallback output for errors."""
        return {
            "compressed_patterns": {
                "profit_direction": [],
                "loss_direction": [],
                "profit_strength": 0.0,
                "loss_strength": 0.0,
                "compression_count": 0,
            },
            "feature_importance": {},
            "intuition_vector": {
                "vector": [],
                "strength": 0.0,
                "components": int(self.n_components),
                "last_updated": 0.0,
            },
            "memory_compression": {
                "total_memories": 0,
                "profit_memories": 0,
                "loss_memories": 0,
                "compression_efficiency": 0.0,
                "last_compression": 0,
            },
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _pad_or_trim(v: np.ndarray, size: int) -> np.ndarray:
        """Pad with zeros or trim to length `size`."""
        if v.ndim != 1:
            v = v.reshape(-1)
        if v.size == size:
            return v.astype(np.float32, copy=False)
        out = np.zeros(size, dtype=np.float32)
        n = min(size, v.size)
        out[:n] = v[:n]
        return out
