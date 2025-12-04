# modules/memory/components/playbook.py
"""
Playbook Memory Component
Context-aware pattern recognition and recall.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .base import MemoryComponent
from modules.memory.shared.utils import safe_float


class PlaybookComponent(MemoryComponent):
    """Playbook memory for pattern recall."""

    # Fraction of global memory reserved for playbook entries
    _BUFFER_FRACTION: float = 0.30
    # Max recent trades to ingest per step
    _RECENT_WINDOW: int = 10
    # Numerical stability epsilon
    _EPS: float = 1e-12

    def _initialize_component(self) -> None:
        """Initialize playbook-specific resources."""
        cfg = self.config
        self.k_neighbors: int = max(1, int(getattr(cfg, "k_neighbors", 5)))
        self.similarity_threshold: float = float(getattr(cfg, "similarity_threshold", 0.7))
        self.pattern_memory_size: int = int(getattr(cfg, "pattern_memory_size", 50))
        self.max_entries: int = int(getattr(cfg, "max_memory_size", 10_000) * self._BUFFER_FRACTION)

        # Local storage (aligned by index)
        self.features: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.pnls: List[float] = []
        self.contexts: List[Dict[str, Any]] = []
        self.timestamps: List[float] = []
        self.trade_metadata: List[Dict[str, Any]] = []
        
        # Track processed trade IDs to avoid double-counting
        self._processed_trade_ids: set = set()

        # Pattern stats
        self.pattern_effectiveness: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0}
        )
        self.context_patterns: Dict[str, int] = defaultdict(int)

        # Local scaler to avoid contaminating/being contaminated by other components
        self._scaler = StandardScaler()
        self._scaler_fitted: bool = False

        # KNN model
        self.knn_model: Optional[NearestNeighbors] = None
        self.knn_fitted: bool = False

        # Metrics
        self.recall_history: deque[Dict[str, Any]] = deque(maxlen=100)
        self.memory_quality_score: float = 0.0
        self.prediction_accuracy: float = 0.0
        self.pattern_diversity: float = 0.0
        self.recall_efficiency: float = 0.0

        self._log_debug(
            "playbook_initialized",
            details={
                "k_neighbors": self.k_neighbors,
                "max_entries": self.max_entries,
                "pattern_memory_size": self.pattern_memory_size,
            },
        )

    # -------------------------------------------------------------------------
    # Main entry
    # -------------------------------------------------------------------------

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process playbook memory operations."""
        try:
            storage_result = await self._process_trades(context)

            if bool(context.get("recall_requested", False)):
                recall_result = await self._perform_recall(context)
                storage_result.update(recall_result)

            # Fit model lazily when enough samples exist
            if not self.knn_fitted and len(self.features) >= self.k_neighbors:
                await self._fit_models()

            self._update_analytics()
            return self._format_output(storage_result)
        except Exception as e:
            self.log_error("Playbook processing failed", e)
            return self._get_fallback_output()

    # -------------------------------------------------------------------------
    # Ingestion / storage
    # -------------------------------------------------------------------------

    async def _process_trades(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process and store recent trades."""
        trades: List[Dict[str, Any]] = context.get("trades", []) or []
        market_context: Dict[str, Any] = context.get("market_context", {}) or {}
        prices: Dict[str, Any] = context.get("prices", {}) or {}

        processed = 0
        skipped = 0
        for trade in trades[-self._RECENT_WINDOW :]:
            if not isinstance(trade, dict) or "pnl" not in trade:
                continue
            
            # Only process CLOSED trades with actual realized PnL
            # Skip open trades (pnl=0, action contains 'open')
            pnl_val = safe_float(trade.get("pnl"), 0.0) or safe_float(trade.get("realized_pnl"), 0.0)
            action = str(trade.get("action", "")).lower()
            if pnl_val == 0 or "open" in action:
                continue  # Skip open trades - they have no outcome yet
            
            # Generate unique trade ID to avoid double-counting
            trade_id = trade.get("id") or trade.get("trade_id") or trade.get("ticket")
            if trade_id is None:
                # Fallback: create ID from trade properties
                inst = trade.get("instrument") or trade.get("symbol") or ""
                ts = trade.get("ts") or trade.get("timestamp") or trade.get("close_time") or ""
                pnl_val = trade.get("pnl", 0)
                trade_id = f"{inst}_{ts}_{pnl_val}"
            
            # Skip if already processed
            if trade_id in self._processed_trade_ids:
                skipped += 1
                continue
            
            # Mark as processed
            self._processed_trade_ids.add(trade_id)
            
            # Limit set size to prevent memory bloat
            if len(self._processed_trade_ids) > 1000:
                self._processed_trade_ids = set(list(self._processed_trade_ids)[-500:])

            feats = self._extract_trade_features(trade, market_context, prices)
            action = self._extract_trade_action(trade)
            pnl = safe_float(trade.get("pnl", 0.0), 0.0)

            await self._store_trade(feats, action, pnl, market_context, trade)
            processed += 1

        return {"trades_processed": processed, "trades_skipped": skipped, "memory_size": len(self.features)}

    def _extract_trade_features(
        self, trade: Dict[str, Any], market_context: Dict[str, Any], prices: Dict[str, Any]
    ) -> np.ndarray:
        """Extract a compact feature vector from trade and context."""
        feats: List[float] = []

        # Market regime (one-hot-ish)
        regime_map = {"trending": [1, 0, 0], "volatile": [0, 1, 0], "ranging": [0, 0, 1]}
        regime = str(market_context.get("regime", "unknown")).lower()
        feats.extend(regime_map.get(regime, [0.33, 0.33, 0.33]))

        # Volatility level
        vol_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "extreme": 1.0}
        vol_level = str(market_context.get("volatility_level", "medium")).lower()
        feats.append(safe_float(vol_map.get(vol_level, 0.5), 0.5))

        # Risk context
        feats.extend(
            [
                safe_float(market_context.get("drawdown_pct", 0.0), 0.0) / 100.0,
                safe_float(market_context.get("exposure_pct", 0.0), 0.0) / 100.0,
                safe_float(market_context.get("position_count", 0), 0) / 10.0,
            ]
        )

        # Session
        session_map = {"asian": [1, 0, 0], "european": [0, 1, 0], "american": [0, 0, 1]}
        session = str(market_context.get("session", "unknown")).lower()
        feats.extend(session_map.get(session, [0.25, 0.25, 0.25]))

        # Trade features
        feats.extend(
            [
                safe_float(trade.get("size", 0.0), 0.0),
                safe_float(trade.get("confidence", 0.5), 0.5),
                1.0
                if str(trade.get("side", "")).lower() == "buy"
                else -1.0
                if str(trade.get("side", "")).lower() == "sell"
                else 0.0,
            ]
        )

        # Price context
        symbol = str(trade.get("symbol", "EUR_USD"))
        if symbol in prices:
            current_price = safe_float(prices[symbol], 0.0)
            entry_price = safe_float(trade.get("price", current_price), current_price)
            price_change = (current_price - entry_price) / (entry_price + self._EPS)
            feats.extend([current_price / 2.0, price_change])
        else:
            feats.extend([0.5, 0.0])

        arr = np.asarray(feats, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    def _extract_trade_action(self, trade: Dict[str, Any]) -> np.ndarray:
        """Extract a 2D action vector from trade (signed size, placeholder)."""
        size = safe_float(trade.get("size", 0.0), 0.0)
        side = str(trade.get("side", "hold")).lower()
        if side == "buy":
            action = [size, 0.0]
        elif side == "sell":
            action = [-size, 0.0]
        else:
            action = [0.0, 0.0]
        return np.asarray(action, dtype=np.float32)

    async def _store_trade(
        self,
        features: np.ndarray,
        action: np.ndarray,
        pnl: float,
        market_context: Dict[str, Any],
        trade: Dict[str, Any],
    ) -> None:
        """Store trade into local memory (with gentle pnl decay and size bounds)."""
        # Decay historical PnLs
        if self.pnls:
            self.pnls = [float(p) * 0.98 for p in self.pnls]

        # Enforce capacity
        if len(self.features) >= self.max_entries:
            self.features.pop(0)
            self.actions.pop(0)
            self.pnls.pop(0)
            self.contexts.pop(0)
            self.timestamps.pop(0)
            self.trade_metadata.pop(0)

        # Append new record
        self.features.append(features)
        self.actions.append(action)
        self.pnls.append(float(pnl))
        self.contexts.append(dict(market_context))
        now = time.time()
        self.timestamps.append(now)
        
        # Get instrument from trade or context
        instrument = trade.get("instrument") or trade.get("symbol") or "UNKNOWN"
        
        self.trade_metadata.append(
            {
                "timestamp": now,
                "pnl": float(pnl),
                "regime": market_context.get("regime"),
                "volatility": market_context.get("volatility_level"),
                "session": market_context.get("session"),
                "instrument": instrument,
            }
        )

        # Pattern tracking (includes instrument for per-instrument analysis)
        self._update_pattern_tracking(market_context, float(pnl), instrument)
        
        # Model is now stale until re-fit (we fit lazily in process)
        self.knn_fitted = False

    def _update_pattern_tracking(self, context: Dict[str, Any], pnl: float, instrument: str = "UNKNOWN") -> None:
        """Update effectiveness stats keyed by (instrument_regime_vol_session)."""
        regime = str(context.get("regime", "unknown")).lower()
        vol = str(context.get("volatility_level", "medium")).lower()
        session = str(context.get("session", "unknown")).lower()
        # Include instrument for per-instrument pattern tracking
        key = f"{instrument}_{regime}_{vol}_{session}"

        self.context_patterns[key] += 1
        data = self.pattern_effectiveness[key]
        # Only count actual wins/losses (non-zero pnl)
        if pnl > 0.0:
            data["wins"] += 1
        elif pnl < 0.0:
            data["losses"] += 1
        # Skip pnl == 0 (open trades) for win/loss counting
        data["total_pnl"] += float(pnl)

        # Keep pattern_effectiveness dict reasonably bounded
        if len(self.pattern_effectiveness) > self.pattern_memory_size:
            # Drop the stalest/least updated pattern
            # (heuristic: smallest wins+losses)
            victim = min(self.pattern_effectiveness.items(), key=lambda kv: kv[1]["wins"] + kv[1]["losses"])[0]
            if victim in self.pattern_effectiveness:
                del self.pattern_effectiveness[victim]
        
        # Keep context_patterns dict bounded as well
        if len(self.context_patterns) > self.pattern_memory_size * 2:
            # Drop lowest-count patterns
            sorted_patterns = sorted(self.context_patterns.items(), key=lambda kv: kv[1])
            to_remove = len(self.context_patterns) - self.pattern_memory_size
            for key_to_remove, _ in sorted_patterns[:to_remove]:
                del self.context_patterns[key_to_remove]

    # -------------------------------------------------------------------------
    # Modeling
    # -------------------------------------------------------------------------

    async def _fit_models(self) -> None:
        """Fit the KNN model on scaled features (if enough samples)."""
        try:
            n = len(self.features)
            if n < self.k_neighbors:
                return

            X = np.vstack(self.features)  # [N, F]

            # Fit scaler locally (fresh each fit to adapt to drift)
            X_scaled = self._scaler.fit_transform(X)
            self._scaler_fitted = True

            self.knn_model = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, n),
                metric="euclidean",
            )
            self.knn_model.fit(X_scaled)
            self.knn_fitted = True

            # Quality score: share of profitable examples
            profitable = int(np.sum(np.asarray(self.pnls) > 0.0))
            self.memory_quality_score = (profitable / max(1, n)) * 100.0

            self._log_debug(
                "knn_fitted",
                details={"n_samples": n, "k": min(self.k_neighbors, n), "quality": self.memory_quality_score},
            )
        except Exception as e:
            self.log_error("Model fitting failed", e)
            self.knn_fitted = False

    async def _perform_recall(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recall similar past trades and propose an action."""
        try:
            if not self.knn_fitted or self.knn_model is None or not self._scaler_fitted or len(self.features) == 0:
                return {"recall_performed": False, "reason": "model_not_fitted"}

            query_features = context.get("query_features")
            if query_features is None:
                market_context = context.get("market_context", {}) or {}
                prices = context.get("prices", {}) or {}
                query_features = self._create_query_features(market_context, prices)
            else:
                # Unwrap feature payloads published by AdvancedFeatureEngine / SmartInfoBus
                if isinstance(query_features, dict):
                    # Common case: {"raw_features": [...], "quality_score": float}
                    if "raw_features" in query_features and isinstance(
                        query_features.get("raw_features"), (list, tuple, np.ndarray)
                    ):
                        query_features = query_features["raw_features"]
                    # Nested container: {"features": {"raw_features": [...]}}
                    elif "features" in query_features and isinstance(query_features.get("features"), dict):
                        inner = query_features["features"]
                        if "raw_features" in inner and isinstance(
                            inner.get("raw_features"), (list, tuple, np.ndarray)
                        ):
                            query_features = inner["raw_features"]
                    else:
                        # Fallback: try first list/array-like value, else regenerate from market context
                        candidate = None
                        for v in query_features.values():
                            if isinstance(v, (list, tuple, np.ndarray)):
                                candidate = v
                                break
                        if candidate is not None:
                            query_features = candidate
                        else:
                            market_context = context.get("market_context", {}) or {}
                            prices = context.get("prices", {}) or {}
                            query_features = self._create_query_features(market_context, prices)

            q = np.asarray(query_features, dtype=np.float32).reshape(1, -1)
            q_scaled = self._scaler.transform(q)

            distances, indices = self.knn_model.kneighbors(q_scaled)
            idx = indices[0].tolist()
            dists = distances[0].astype(np.float32)

            # Be robust to any historical schema changes that might have stored
            # non-scalar objects in self.pnls.
            similar_pnls = [safe_float(self.pnls[i], 0.0) for i in idx]
            similar_actions = [self.actions[i] for i in idx]

            expected_pnl = float(np.mean(similar_pnls)) if similar_pnls else 0.0

            # Confidence from distance; handle zero variance
            d_mean = float(np.mean(dists)) if dists.size else 0.0
            confidence = float(np.exp(-d_mean))

            # Weights ~ exp(-distance) normalized
            w = np.exp(-dists)
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum <= self._EPS:
                w = np.full_like(dists, 1.0 / max(1, dists.size), dtype=np.float32)
            else:
                w = w / (w_sum + self._EPS)

            # Weighted action recommendation
            if similar_actions:
                A = np.vstack(similar_actions)  # [K, 2]
                recommended_action = (A * w.reshape(-1, 1)).sum(axis=0)
            else:
                recommended_action = np.zeros(2, dtype=np.float32)

            profitable_matches = int(np.sum(np.asarray(similar_pnls) > 0.0))
            
            # === NEW: signed_bias for memory_vote ===
            # signed_bias = tanh(expected_pnl / pnl_scale)
            pnl_scale = 20.0  # Scale factor for normalizing PnL to [-1, 1]
            signed_bias = float(np.tanh(expected_pnl / pnl_scale))
            
            # === NEW: top-K neighbors for rationale bundle ===
            top_neighbors: List[Dict[str, Any]] = []
            for i, (dist_val, pnl_val) in enumerate(zip(dists.tolist(), similar_pnls)):
                # Calculate age in hours
                neighbor_idx = idx[i]
                if neighbor_idx < len(self.timestamps):
                    age_h = (time.time() - self.timestamps[neighbor_idx]) / 3600.0
                else:
                    age_h = 0.0
                
                top_neighbors.append({
                    "sim": round(float(np.exp(-dist_val)), 3),  # Convert distance to similarity
                    "pnl": round(pnl_val, 2),
                    "age_h": round(age_h, 1),
                    "dist": round(float(dist_val), 3),
                })

            # Record recall
            self.recall_history.append(
                {
                    "timestamp": time.time(),
                    "expected_pnl": expected_pnl,
                    "confidence": confidence,
                    "similar_trades": len(idx),
                    "signed_bias": signed_bias,
                }
            )

            return {
                "recall_performed": True,
                "expected_pnl": expected_pnl,
                "confidence": confidence,
                "recommended_action": recommended_action.astype(np.float32).tolist(),
                "similar_trades": len(idx),
                "profitable_matches": profitable_matches,
                # New fields for memory_vote composition
                "signed_bias": signed_bias,
                "top_neighbors": top_neighbors,
            }
        except Exception as e:
            self.log_error("Recall failed", e)
            return {"recall_performed": False, "error": str(e)}

    def _create_query_features(self, market_context: Dict[str, Any], prices: Dict[str, Any]) -> np.ndarray:
        """Create query features from current context using a neutral 'dummy' trade."""
        dummy_trade = {
            "size": 1.0,
            "confidence": 0.5,
            "side": "hold",
            "symbol": "EUR_USD",
            "price": 1.0,
        }
        return self._extract_trade_features(dummy_trade, market_context, prices)

    # -------------------------------------------------------------------------
    # Analytics / reporting
    # -------------------------------------------------------------------------

    def _update_analytics(self) -> None:
        """Update memory analytics metrics."""
        # Diversity: #patterns normalized to 0..1 (cap at 20)
        unique_patterns = int(len(self.pattern_effectiveness))
        self.pattern_diversity = min(1.0, unique_patterns / 20.0)

        # Recall efficiency: average confidence on last 10 recalls
        if self.recall_history:
            recent = list(self.recall_history)[-10:]
            self.recall_efficiency = float(np.mean([float(r.get("confidence", 0.0)) for r in recent]))
        else:
            self.recall_efficiency = 0.0

        # Prediction accuracy: wins / (wins + losses) aggregated over patterns
        wins = sum(int(p["wins"]) for p in self.pattern_effectiveness.values())
        losses = sum(int(p["losses"]) for p in self.pattern_effectiveness.values())
        denom = max(1, wins + losses)
        self.prediction_accuracy = float(wins / denom)

    # -------------------------------------------------------------------------
    # Output / fallback
    # -------------------------------------------------------------------------

    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements."""
        # Top pattern by total PnL
        top_pattern: Optional[str] = None
        if self.pattern_effectiveness:
            top_pattern = max(self.pattern_effectiveness.items(), key=lambda kv: kv[1]["total_pnl"])[0]

        last_recall = self.recall_history[-1] if self.recall_history else None

        return {
            "memory_analytics": {
                "total_recalls": len(self.recall_history),
                "recent_performance": float(self.recall_efficiency),
                "memory_health": "healthy" if self.memory_quality_score > 50.0 else "warning",
                "circuit_breaker_state": "CLOSED",
            },
            "pattern_memory": {
                "total_patterns": len(self.pattern_effectiveness),
                "pattern_effectiveness": dict(list(self.pattern_effectiveness.items())[: self.pattern_memory_size]),
                "pattern_diversity": float(self.pattern_diversity),
                "top_pattern": top_pattern,
            },
            "playbook_quality": {
                "memory_utilization": float(len(self.features) / max(1, self.max_entries)),
                "quality_score": float(self.memory_quality_score),
                "models_fitted": bool(self.knn_fitted),
                "adaptive_k": int(min(self.k_neighbors, max(1, len(self.features)))),
            },
            "playbook_recall": {
                "memory_entries": int(len(self.features)),
                "patterns_identified": int(len(self.pattern_effectiveness)),
                "recall_efficiency": float(self.recall_efficiency),
                "prediction_accuracy": float(self.prediction_accuracy),
                "last_recall": last_recall,
                # Include recall results for memory_vote composition
                "signed_bias": float(result.get("signed_bias", 0.0)),
                "confidence": float(result.get("confidence", 0.5)),
                "expected_pnl": float(result.get("expected_pnl", 0.0)),
                "top_neighbors": result.get("top_neighbors", []),
            },
        }

    def _get_fallback_output(self) -> Dict[str, Any]:
        """Conservative payload on error."""
        return {
            "memory_analytics": {
                "total_recalls": 0,
                "recent_performance": 0.0,
                "memory_health": "unknown",
                "circuit_breaker_state": "CLOSED",
            },
            "pattern_memory": {
                "total_patterns": 0,
                "pattern_effectiveness": {},
                "pattern_diversity": 0.0,
                "top_pattern": None,
            },
            "playbook_quality": {
                "memory_utilization": 0.0,
                "quality_score": 0.0,
                "models_fitted": False,
                "adaptive_k": int(self.k_neighbors),
            },
            "playbook_recall": {
                "memory_entries": 0,
                "patterns_identified": 0,
                "recall_efficiency": 0.0,
                "prediction_accuracy": 0.0,
                "last_recall": None,
                # Fallback values for memory_vote composition
                "signed_bias": 0.0,
                "confidence": 0.5,
                "expected_pnl": 0.0,
                "top_neighbors": [],
            },
        }
