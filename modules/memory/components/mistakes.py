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
from modules.memory.shared.utils import safe_float


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

        # State - Per-instrument loss tracking
        self.consecutive_losses: int = 0  # Global counter (legacy, for compatibility)
        self.consecutive_losses_by_instrument: Dict[str, int] = {}  # Per-instrument counters
        self.avoidance_signal: float = 0.0
        self._processed_trade_ids: set = set()  # Track processed trade IDs to avoid duplicates
        self._veto_start_time: float = 0.0  # When veto was triggered
        self._veto_start_by_instrument: Dict[str, float] = {}  # Per-instrument veto timers
        self._veto_timeout_seconds: float = 60.0  # Veto expires after 60 seconds (allows recovery)
        self._ticks_since_last_trade: int = 0  # Ticks since last trade for decay

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
            
            # Compose gate snippet for UnifiedMemory consumption
            gate_snippet = self._compose_gate_snippet(
                danger_similarity=avoidance_result.get("danger_similarity", 0.0),
                profit_similarity=avoidance_result.get("profit_similarity", 0.0),
                avoidance_signal=avoidance_result.get("avoidance_signal", 0.0),
                context=context,
            )
            learning_result["gate_snippet"] = gate_snippet

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

            # Generate a unique trade ID to avoid reprocessing the same trade
            trade_id = trade.get("id") or trade.get("trade_id") or trade.get("ticket")
            if trade_id is None:
                # Fallback: create ID from trade properties
                trade_id = f"{trade.get('symbol', '')}_{trade.get('open_time', '')}_{trade.get('close_time', '')}_{trade.get('pnl', 0)}"
            
            # Skip if already processed
            if trade_id in self._processed_trade_ids:
                continue
            
            # Mark as processed
            self._processed_trade_ids.add(trade_id)
            
            # Limit set size to prevent memory bloat
            if len(self._processed_trade_ids) > 1000:
                # Remove oldest entries (convert to list, slice, convert back)
                self._processed_trade_ids = set(list(self._processed_trade_ids)[-500:])

            features = self._extract_trade_features(trade, market_context)
            if features is None:
                continue

            # Get instrument for per-instrument tracking
            instrument = trade.get("instrument") or trade.get("symbol") or "UNKNOWN"
            
            pnl = float(trade["pnl"])
            if pnl < -self.profit_threshold / 2.0:
                self._process_loss_trade(features, abs(pnl), trade)
                losses_learned += 1
                # Update both global and per-instrument counters
                self.consecutive_losses += 1
                self.consecutive_losses_by_instrument[instrument] = \
                    self.consecutive_losses_by_instrument.get(instrument, 0) + 1
            elif pnl > self.profit_threshold:
                self._process_win_trade(features, pnl, trade)
                wins_learned += 1
                # Reset both global and per-instrument counters
                self.consecutive_losses = 0
                self.consecutive_losses_by_instrument[instrument] = 0
                self._veto_start_time = 0.0  # Reset veto timer on win
                self._veto_start_by_instrument[instrument] = 0.0  # Reset per-instrument timer

        return {
            "losses_learned": losses_learned,
            "wins_learned": wins_learned,
            "total_loss_memories": len(self.loss_buffer),
            "total_win_memories": len(self.win_buffer),
            "consecutive_losses_by_instrument": dict(self.consecutive_losses_by_instrument),
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
            features.append(safe_float(trade.get("confidence", 0.5), 0.5))
            features.append(safe_float(trade.get("volume", 1.0), 1.0))
            features.append(safe_float(trade.get("duration", 1.0), 1.0))

            # Market context: volatility can be scalar or dict
            vol = market_context.get("volatility", 0.5)
            if isinstance(vol, dict):
                vol = (list(vol.values()) or [0.5])[0]
            features.append(safe_float(vol, 0.5))

            # Session encoding (consistent across all memory components)
            session_map = {"asian": 0.0, "european": 0.5, "american": 1.0, "us": 1.0}  # us is alias for american
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
            
            # Bound pattern dict to prevent memory leaks
            self._bound_pattern_dict(self.loss_patterns, max_patterns=100)

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
            
            # Bound pattern dict to prevent memory leaks
            self._bound_pattern_dict(self.win_patterns, max_patterns=100)

    def _bound_pattern_dict(self, patterns: Dict[str, Dict[str, Any]], max_patterns: int = 100) -> None:
        """Evict oldest/least-used patterns if dict exceeds max_patterns."""
        if len(patterns) <= max_patterns:
            return
        
        # Sort by last_seen (oldest first)
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda kv: kv[1].get("last_seen", 0.0)
        )
        
        # Evict oldest patterns
        to_remove = len(patterns) - max_patterns
        for key, _ in sorted_patterns[:to_remove]:
            del patterns[key]

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

    def _compose_gate_snippet(
        self,
        danger_similarity: float,
        profit_similarity: float,
        avoidance_signal: float,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compose a compact gate snippet for UnifiedMemory consumption.
        
        Returns:
            Dict containing:
            - risk_multiplier: Trade size risk adjustment (higher = riskier)
            - veto: Boolean recommending trade rejection
            - confidence: Confidence in the gate decision [0, 1]
            - reasons: List of reasons for the gate decision
            - vetoed_instruments: List of instruments currently vetoed (per-instrument)
        """
        # Risk multiplier: increases with danger, decreases with profit proximity
        # risk_multiplier = 1 + 1.5 * max(0, danger_similarity - profit_similarity)
        net_danger = max(0.0, danger_similarity - profit_similarity)
        risk_multiplier = 1.0 + 1.5 * net_danger
        
        # Veto decision based on thresholds
        # veto = (danger_similarity > 0.55 and avoidance_signal > 0.6)
        veto_threshold_danger = 0.55
        veto_threshold_avoidance = 0.6
        veto = bool(danger_similarity > veto_threshold_danger and avoidance_signal > veto_threshold_avoidance)
        
        # Per-instrument veto tracking
        vetoed_instruments: List[str] = []
        current_time = time.time()
        
        # Check per-instrument loss streaks
        for inst, losses in self.consecutive_losses_by_instrument.items():
            if losses >= 5:
                # Check if veto should expire (timeout recovery)
                veto_start = self._veto_start_by_instrument.get(inst, 0.0)
                if veto_start == 0.0:
                    self._veto_start_by_instrument[inst] = current_time
                    vetoed_instruments.append(inst)
                else:
                    elapsed = current_time - veto_start
                    if elapsed < self._veto_timeout_seconds:
                        vetoed_instruments.append(inst)
                    else:
                        # Veto expired - decay consecutive losses to allow recovery
                        self.consecutive_losses_by_instrument[inst] = max(0, losses - 2)
                        self._veto_start_by_instrument[inst] = 0.0
                        self._log_debug("veto_timeout_recovery_instrument", details={
                            "instrument": inst,
                            "new_consecutive_losses": self.consecutive_losses_by_instrument[inst],
                            "elapsed_seconds": elapsed
                        })
        
        # Legacy global veto (for backwards compatibility) - only if ALL instruments are struggling
        if self.consecutive_losses >= 5:
            if self._veto_start_time == 0.0:
                self._veto_start_time = current_time
            
            elapsed = current_time - self._veto_start_time
            if elapsed < self._veto_timeout_seconds:
                veto = True  # Global veto still in effect
            else:
                # Veto expired - decay consecutive losses to allow recovery
                self.consecutive_losses = max(0, self.consecutive_losses - 2)
                self._veto_start_time = 0.0
                self._log_debug("veto_timeout_recovery", details={
                    "new_consecutive_losses": self.consecutive_losses,
                    "elapsed_seconds": elapsed
                })
        
        # Confidence: based on clustering quality and sample count
        sample_confidence = min(1.0, (len(self.loss_buffer) + len(self.win_buffer)) / 50.0)
        avg_quality = float(np.mean(list(self.cluster_quality_scores))) if self.cluster_quality_scores else 0.5
        confidence = 0.5 * sample_confidence + 0.5 * avg_quality
        
        # Build reasons list
        reasons: List[Dict[str, Any]] = []
        
        if danger_similarity > 0.4:
            # Find closest danger zone for pattern info
            pattern_info = self._get_nearest_pattern_info(context.get("features"))
            reasons.append({
                "type": "pattern",
                "label": pattern_info.get("label", "DANGER_ZONE"),
                "regime": str(context.get("market_context", {}).get("regime", "unknown")).lower(),
                "similarity": round(danger_similarity, 3),
                "stats": {
                    "n": len(self.loss_buffer),
                    "winrate": self._calculate_zone_winrate(),
                    "avg_pnl": self._calculate_avg_loss_pnl(),
                },
            })
        
        # Add per-instrument streak reasons
        for inst, losses in self.consecutive_losses_by_instrument.items():
            if losses >= 3:
                reasons.append({
                    "type": "streak",
                    "instrument": inst,
                    "consecutive_losses": losses,
                    "message": f"Loss streak of {losses} detected for {inst}",
                })
        
        # Legacy global streak reason (if no per-instrument reasons)
        if self.consecutive_losses >= 3 and not any(r.get("type") == "streak" for r in reasons):
            reasons.append({
                "type": "streak",
                "consecutive_losses": self.consecutive_losses,
                "message": f"Loss streak of {self.consecutive_losses} detected",
            })
        
        if profit_similarity > danger_similarity and profit_similarity > 0.5:
            reasons.append({
                "type": "opportunity",
                "profit_similarity": round(profit_similarity, 3),
                "message": "Setup resembles profitable patterns",
            })
        
        return {
            "risk_multiplier": float(np.clip(risk_multiplier, 1.0, 5.0)),
            "veto": veto,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "reasons": reasons,
            "danger_similarity": float(danger_similarity),
            "profit_similarity": float(profit_similarity),
            "avoidance_signal": float(avoidance_signal),
            "vetoed_instruments": vetoed_instruments,
            "consecutive_losses_by_instrument": dict(self.consecutive_losses_by_instrument),
        }
    
    def _get_nearest_pattern_info(self, features: Optional[np.ndarray]) -> Dict[str, Any]:
        """Get info about the nearest danger zone pattern."""
        if features is None or not self.danger_zones:
            return {"label": "UNKNOWN", "severity": 0.0}
        
        try:
            # Find nearest zone
            vec = self._prep_feature_row(features)
            if not self._loss_scaler_fitted:
                return {"label": "UNKNOWN", "severity": 0.0}
            
            scaled = self._loss_scaler.transform(vec)
            
            min_dist = float("inf")
            nearest_zone = None
            
            for i, zone in enumerate(self.danger_zones):
                center = np.asarray(zone["center"], dtype=np.float32)
                dist = float(np.linalg.norm(scaled[0] - center))
                if dist < min_dist:
                    min_dist = dist
                    nearest_zone = zone
            
            if nearest_zone is None:
                return {"label": "UNKNOWN", "severity": 0.0}
            
            # Generate pattern label from zone characteristics
            size = nearest_zone.get("size", 0)
            severity = nearest_zone.get("severity", 0.0)
            
            if severity > 20:
                severity_label = "HIGH"
            elif severity > 10:
                severity_label = "MED"
            else:
                severity_label = "LOW"
            
            return {
                "label": f"DANGER_ZONE_{severity_label}_{size}",
                "severity": severity,
                "size": size,
            }
            
        except Exception:
            return {"label": "UNKNOWN", "severity": 0.0}
    
    def _calculate_zone_winrate(self) -> float:
        """Calculate win rate for patterns near danger zones."""
        if not self.loss_buffer and not self.win_buffer:
            return 0.0
        
        total = len(self.loss_buffer) + len(self.win_buffer)
        if total == 0:
            return 0.0
        
        return float(len(self.win_buffer) / total)
    
    def _calculate_avg_loss_pnl(self) -> float:
        """Calculate average PnL of loss patterns."""
        if not self.loss_buffer:
            return 0.0
        
        losses = [float(entry[1]) for entry in self.loss_buffer]
        return float(-np.mean(losses)) if losses else 0.0

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
        
        # Get gate snippet if present
        gate_snippet = result.get("gate_snippet", {})

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
            # Gate snippet for UnifiedMemory to compose memory_gate
            "gate_snippet": {
                "risk_multiplier": gate_snippet.get("risk_multiplier", 1.0),
                "veto": gate_snippet.get("veto", False),
                "confidence": gate_snippet.get("confidence", 0.0),
                "reasons": gate_snippet.get("reasons", []),
                "danger_similarity": gate_snippet.get("danger_similarity", 0.0),
                "profit_similarity": gate_snippet.get("profit_similarity", 0.0),
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
