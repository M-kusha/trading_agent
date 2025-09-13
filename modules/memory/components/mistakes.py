# modules/memory/components/mistakes.py
"""
Mistake Memory Component
Identifies danger zones and provides loss-avoidance signals.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .base import MemoryComponent


class MistakeComponent(MemoryComponent):
    """Mistake detection and avoidance component."""

    # Clustering defaults / guardrails
    _DBSCAN_EPS_DEFAULT: float = 0.30
    _DBSCAN_MIN_SAMPLES: int = 5
    _BUFFER_FRACTION: float = 0.10  # max fraction of max_memory_size for buffers
    _RECENT_WINDOW: int = 20  # last N trades considered for learning pass

    def _initialize_component(self) -> None:
        """Initialize mistake-specific resources."""
        # Configuration (safe fallbacks)
        cfg = self.config
        self.n_clusters: int = int(getattr(cfg, "n_clusters", 5))
        self.danger_threshold: float = float(getattr(cfg, "danger_threshold", 0.7))
        self.avoidance_sensitivity: float = float(getattr(cfg, "avoidance_sensitivity", 1.0))
        self.profit_threshold: float = float(getattr(cfg, "mistake_profit_threshold", 10.0))
        self.max_memory_size: int = int(getattr(cfg, "max_memory_size", 10_000))

        # Memory buffers: (features: np.ndarray, magnitude: float, trade: dict)
        self.loss_buffer: List[Tuple[np.ndarray, float, Dict[str, Any]]] = []
        self.win_buffer: List[Tuple[np.ndarray, float, Dict[str, Any]]] = []

        # Separate scalers to avoid distribution leakage
        self._loss_scaler = StandardScaler()
        self._win_scaler = StandardScaler()
        self._loss_scaler_fitted: bool = False
        self._win_scaler_fitted: bool = False

        # Clustering artifacts
        self.loss_clusterer: Optional[DBSCAN] = None
        self.win_clusterer: Optional[DBSCAN] = None
        self.danger_zones: List[Dict[str, Any]] = []  # centers in LOSS-scaled space
        self.profit_zones: List[Dict[str, Any]] = []  # centers in WIN-scaled space

        # State
        self.consecutive_losses: int = 0
        self.avoidance_signal: float = 0.0

        # Pattern tracking
        self.loss_patterns: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "severity": 0.0, "last_seen": 0.0}
        )
        self.win_patterns: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "profitability": 0.0, "last_seen": 0.0}
        )

        # Metrics
        self.cluster_quality_scores: deque[float] = deque(maxlen=20)
        self.avoidance_effectiveness: float = 0.0
        self.false_positive_rate: float = 0.0
        self.true_positive_rate: float = 0.0

        self._log_debug(
            "mistakes_initialized",
            details={
                "n_clusters": self.n_clusters,
                "eps": self._DBSCAN_EPS_DEFAULT,
                "min_samples": self._DBSCAN_MIN_SAMPLES,
                "buffer_fraction": self._BUFFER_FRACTION,
            },
        )

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process mistake memory operations."""
        try:
            learning_result = self._process_learning_data(context)

            if self._should_update_clustering():
                clustering_result = self._update_clustering()
                learning_result.update(clustering_result)

            avoidance_result = self._calculate_avoidance_signals(context)
            learning_result.update(avoidance_result)

            return self._format_output(learning_result)
        except Exception as e:
            self.log_error("Mistake processing failed", e)
            return self._get_fallback_output()

    # -------------------------------------------------------------------------
    # Learning / buffers
    # -------------------------------------------------------------------------

    def _process_learning_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process recent trades to update loss/win memories."""
        trades: List[Dict[str, Any]] = context.get("trades", []) or []
        market_context: Dict[str, Any] = context.get("market_context", {}) or {}

        losses_learned = 0
        wins_learned = 0

        for trade in trades[-self._RECENT_WINDOW :]:
            if not isinstance(trade, dict) or "pnl" not in trade:
                continue

            features = self._extract_trade_features(trade, market_context)
            if features is None:
                continue

            pnl = float(trade["pnl"])
            if pnl < -self.profit_threshold / 2.0:
                self._process_loss_trade(features, abs(pnl), trade)
                losses_learned += 1
                self.consecutive_losses += 1
            elif pnl > self.profit_threshold:
                self._process_win_trade(features, pnl, trade)
                wins_learned += 1
                self.consecutive_losses = 0

        return {
            "losses_learned": losses_learned,
            "wins_learned": wins_learned,
            "total_loss_memories": len(self.loss_buffer),
            "total_win_memories": len(self.win_buffer),
        }

    def _extract_trade_features(self, trade: Dict[str, Any], market_context: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract a lightweight, robust feature vector from a trade + market context.
        NOTE: This component intentionally uses a simple handcrafted vector to
        remain independent of shared encoders; you can extend with self.extractor.
        """
        try:
            features: List[float] = []

            # Trade features
            features.append(float(trade.get("confidence", 0.5)))
            features.append(float(trade.get("volume", 1.0)))
            features.append(float(trade.get("duration", 1.0)))

            # Market context: volatility can be scalar or dict
            vol = market_context.get("volatility", 0.5)
            if isinstance(vol, dict):
                vol = (list(vol.values()) or [0.5])[0]
            features.append(float(vol))

            # Session encoding
            session_map = {"asian": 0.0, "european": 0.5, "us": 1.0}
            session = str(market_context.get("session", "unknown")).lower()
            features.append(float(session_map.get(session, 0.25)))

            # Regime encoding
            regime_map = {"trending": 1.0, "ranging": 0.0, "volatile": 0.5}
            regime = str(market_context.get("regime", "unknown")).lower()
            features.append(float(regime_map.get(regime, 0.25)))

            # Pad to a minimum stable size (10), cap at 20
            while len(features) < 10:
                features.append(0.0)
            arr = np.asarray(features[:20], dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            return arr
        except Exception:
            return None

    def _process_loss_trade(self, features: np.ndarray, loss: float, trade: Dict[str, Any]) -> None:
        """Record a loss example and track patterns."""
        self.loss_buffer.append((features, loss, trade))
        self._bound_buffer_inplace(self.loss_buffer)

        pattern = self._extract_pattern(features, trade)
        if pattern:
            data = self.loss_patterns[pattern]
            data["count"] += 1
            data["severity"] += float(loss)
            data["last_seen"] = time.time()

    def _process_win_trade(self, features: np.ndarray, profit: float, trade: Dict[str, Any]) -> None:
        """Record a win example and track patterns."""
        self.win_buffer.append((features, profit, trade))
        self._bound_buffer_inplace(self.win_buffer)

        pattern = self._extract_pattern(features, trade)
        if pattern:
            data = self.win_patterns[pattern]
            data["count"] += 1
            data["profitability"] += float(profit)
            data["last_seen"] = time.time()

    def _bound_buffer_inplace(self, buf: List[Tuple[np.ndarray, float, Dict[str, Any]]]) -> None:
        """Ensure memory buffers remain within configured fraction."""
        cap = max(1, int(self.max_memory_size * self._BUFFER_FRACTION))
        if len(buf) > cap:
            del buf[: (len(buf) - cap)]

    def _extract_pattern(self, features: np.ndarray, trade: Dict[str, Any]) -> Optional[str]:
        """Create a compact symbolic pattern label from features + action."""
        try:
            elems: List[str] = []
            head = features[:5].tolist()
            for i, feat in enumerate(head):
                if feat > 0.7:
                    elems.append(f"H{i}")
                elif feat < 0.3:
                    elems.append(f"L{i}")
                else:
                    elems.append(f"M{i}")

            action = trade.get("action")
            if isinstance(action, (list, tuple)) and len(action) > 0:
                a0 = float(action[0])
                if a0 > 0.5:
                    elems.append("BUY")
                elif a0 < -0.5:
                    elems.append("SELL")
                else:
                    elems.append("HOLD")

            return "_".join(elems) if elems else None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Clustering
    # -------------------------------------------------------------------------

    def _should_update_clustering(self) -> bool:
        """Determine whether we have enough data to (re)cluster."""
        if len(self.loss_buffer) < max(self._DBSCAN_MIN_SAMPLES, self.n_clusters):
            return False
        # Recluster every 10 new loss examples to amortize cost
        return (len(self.loss_buffer) % 10) == 0

    def _update_clustering(self) -> Dict[str, Any]:
        """Update clustering for loss/win data and derive zones."""
        results: Dict[str, Any] = {}

        if len(self.loss_buffer) >= self._DBSCAN_MIN_SAMPLES:
            results.update(self._cluster_loss_data())

        if len(self.win_buffer) >= self._DBSCAN_MIN_SAMPLES:
            results.update(self._cluster_win_data())

        results["clustering_updated"] = True
        return results

    def _cluster_loss_data(self) -> Dict[str, Any]:
        """Cluster loss data to identify danger zones (LOSS stream)."""
        try:
            feats = np.stack([e[0] for e in self.loss_buffer], axis=0)
            losses = np.asarray([e[1] for e in self.loss_buffer], dtype=np.float32)

            # Fit / transform loss scaler
            if not self._loss_scaler_fitted:
                scaled = self._loss_scaler.fit_transform(feats)
                self._loss_scaler_fitted = True
            else:
                scaled = self._loss_scaler.transform(feats)

            # DBSCAN clustering
            self.loss_clusterer = DBSCAN(
                eps=self._DBSCAN_EPS_DEFAULT, min_samples=self._DBSCAN_MIN_SAMPLES
            )
            clusters = self.loss_clusterer.fit_predict(scaled)

            # Silhouette quality (requires ≥2 clusters, and each cluster ≥2 samples)
            quality = 0.0
            labels = np.unique(clusters)
            if len(labels) > 1:
                # Check minimal cluster sizes
                counts = [np.sum(clusters == c) for c in labels if c != -1]
                if all(c >= 2 for c in counts) and sum(counts) >= 3:
                    quality = float(silhouette_score(scaled, clusters))

            # Build zones (cluster centers with severity)
            self.danger_zones = []
            for cid in labels:
                if cid == -1:
                    continue
                mask = clusters == cid
                if not np.any(mask):
                    continue
                center = np.mean(scaled[mask], axis=0)
                severity = float(np.mean(losses[mask]))
                self.danger_zones.append(
                    {"center": center.tolist(), "severity": severity, "size": int(np.sum(mask))}
                )

            if quality:
                self.cluster_quality_scores.append(quality)

            return {
                "loss_clustering": {
                    "clusters_found": len(self.danger_zones),
                    "quality_score": float(quality),
                    "total_losses": int(len(feats)),
                }
            }
        except Exception as e:
            self.log_error("Loss clustering failed", e)
            return {"loss_clustering": {"error": str(e)}}

    def _cluster_win_data(self) -> Dict[str, Any]:
        """Cluster win data to identify profit zones (WIN stream)."""
        try:
            feats = np.stack([e[0] for e in self.win_buffer], axis=0)
            profits = np.asarray([e[1] for e in self.win_buffer], dtype=np.float32)

            # Fit / transform win scaler
            if not self._win_scaler_fitted:
                scaled = self._win_scaler.fit_transform(feats)
                self._win_scaler_fitted = True
            else:
                scaled = self._win_scaler.transform(feats)

            self.win_clusterer = DBSCAN(
                eps=self._DBSCAN_EPS_DEFAULT, min_samples=self._DBSCAN_MIN_SAMPLES
            )
            clusters = self.win_clusterer.fit_predict(scaled)

            self.profit_zones = []
            for cid in np.unique(clusters):
                if cid == -1:
                    continue
                mask = clusters == cid
                if not np.any(mask):
                    continue
                center = np.mean(scaled[mask], axis=0)
                profitability = float(np.mean(profits[mask]))
                self.profit_zones.append(
                    {
                        "center": center.tolist(),
                        "profitability": profitability,
                        "size": int(np.sum(mask)),
                    }
                )

            return {
                "win_clustering": {
                    "clusters_found": len(self.profit_zones),
                    "total_wins": int(len(feats)),
                }
            }
        except Exception as e:
            self.log_error("Win clustering failed", e)
            return {"win_clustering": {"error": str(e)}}

    # -------------------------------------------------------------------------
    # Signals
    # -------------------------------------------------------------------------

    def _calculate_avoidance_signals(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute avoidance score based on proximity to danger/profit zones."""
        features = context.get("features")
        if features is None:
            return {
                "avoidance_signal": 0.0,
                "danger_similarity": 0.0,
                "profit_similarity": 0.0,
            }

        danger_similarity = self._calculate_danger_similarity(features)
        profit_similarity = self._calculate_profit_similarity(features)

        # Base avoidance proportional to danger similarity
        avoidance = float(danger_similarity) * float(self.avoidance_sensitivity)

        # If near profit zones more than danger zones, reduce avoidance
        if profit_similarity > danger_similarity:
            avoidance *= 0.5

        # Amplify based on recent loss streak
        if self.consecutive_losses > 2:
            avoidance *= 1.0 + 0.1 * float(self.consecutive_losses)

        self.avoidance_signal = float(np.clip(avoidance, 0.0, 1.0))

        return {
            "avoidance_signal": float(self.avoidance_signal),
            "danger_similarity": float(danger_similarity),
            "profit_similarity": float(profit_similarity),
            "consecutive_losses": int(self.consecutive_losses),
        }

    def _calculate_danger_similarity(self, features: np.ndarray) -> float:
        """Distance-based similarity to loss 'danger zones' in LOSS-scaled space."""
        if not self.danger_zones or not self._loss_scaler_fitted:
            return 0.0
        try:
            vec = self._prep_feature_row(features)
            scaled = self._loss_scaler.transform(vec)  # 1 x F

            min_wdist = float("inf")
            for zone in self.danger_zones:
                center = np.asarray(zone["center"], dtype=np.float32)
                dist = float(np.linalg.norm(scaled[0] - center))
                wdist = dist / (float(zone.get("severity", 0.0)) + 1.0)
                if wdist < min_wdist:
                    min_wdist = wdist

            return float(1.0 / (1.0 + min_wdist))
        except Exception:
            return 0.0

    def _calculate_profit_similarity(self, features: np.ndarray) -> float:
        """Distance-based similarity to win 'profit zones' in WIN-scaled space."""
        if not self.profit_zones or not self._win_scaler_fitted:
            return 0.0
        try:
            vec = self._prep_feature_row(features)
            scaled = self._win_scaler.transform(vec)  # 1 x F

            min_wdist = float("inf")
            for zone in self.profit_zones:
                center = np.asarray(zone["center"], dtype=np.float32)
                dist = float(np.linalg.norm(scaled[0] - center))
                wdist = dist / (float(zone.get("profitability", 0.0)) + 1.0)
                if wdist < min_wdist:
                    min_wdist = wdist

            return float(1.0 / (1.0 + min_wdist))
        except Exception:
            return 0.0

    @staticmethod
    def _prep_feature_row(x: Any) -> np.ndarray:
        """Ensure an input is a 2D float32 array shape (1, F)."""
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(1, -1)
        return arr

    # -------------------------------------------------------------------------
    # Output / fallback
    # -------------------------------------------------------------------------

    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements."""
        avg_quality = float(np.mean(list(self.cluster_quality_scores))) if self.cluster_quality_scores else 0.0

        return {
            "danger_zones": {
                "zones": self.danger_zones,
                "zone_count": int(len(self.danger_zones)),
                "avoidance_sensitivity": float(self.avoidance_sensitivity),
                "last_updated": time.time(),
            },
            "loss_prevention": {
                "avoidance_effectiveness": float(self.avoidance_effectiveness),
                "false_positive_rate": float(self.false_positive_rate),
                "true_positive_rate": float(self.true_positive_rate),
                "cluster_quality": float(avg_quality),
                "learning_samples": int(len(self.loss_buffer) + len(self.win_buffer)),
            },
            "mistake_avoidance": {
                "avoidance_signal": float(self.avoidance_signal),
                "consecutive_losses": int(self.consecutive_losses),
                "danger_zones_count": int(len(self.danger_zones)),
                "profit_zones_count": int(len(self.profit_zones)),
                "total_loss_memories": int(len(self.loss_buffer)),
                "total_win_memories": int(len(self.win_buffer)),
            },
            "mistake_memory": {
                "current_score": float(np.clip(self.avoidance_signal, 0.0, 1.0)),
                "consecutive_losses": int(self.consecutive_losses),
                "avoidance_signal": float(self.avoidance_signal),
                "last_updated": time.time(),
            },
            "pattern_recognition": {
                "loss_patterns": dict(list(self.loss_patterns.items())[:10]),
                "win_patterns": dict(list(self.win_patterns.items())[:10]),
                "total_loss_patterns": int(len(self.loss_patterns)),
                "total_win_patterns": int(len(self.win_patterns)),
            },
        }

    def _get_fallback_output(self) -> Dict[str, Any]:
        """Conservative payload on error."""
        return {
            "danger_zones": {"zones": [], "zone_count": 0},
            "loss_prevention": {
                "avoidance_effectiveness": 0.0,
                "false_positive_rate": 0.0,
                "true_positive_rate": 0.0,
                "cluster_quality": 0.0,
                "learning_samples": 0,
            },
            "mistake_avoidance": {"avoidance_signal": 0.0, "consecutive_losses": 0},
            "mistake_memory": {"current_score": 0.0, "avoidance_signal": 0.0},
            "pattern_recognition": {"loss_patterns": {}, "win_patterns": {}},
        }
