# ─────────────────────────────────────────────────────────────
# File: modules/memory/mistake_memory.py
# [ROCKET] PRODUCTION-READY Mistake Memory System
# Advanced loss avoidance with clustering and SmartInfoBus integration
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.exceptions import NotFittedError

from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.core.mixins import (
    SmartInfoBusTradingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
)
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# --- local no-op bus so smart_bus is never None (satisfies type checkers) ---
class _DummyBus:
    def get(self, _key: str, *_args, **_kwargs):
        return None

    def set(self, _key: str, _value: Any, **_kwargs):
        return None


@dataclass
class MistakeConfig:
    """Configuration for Mistake Memory"""
    max_mistakes: int = 100
    n_clusters: int = 5
    profit_threshold: float = 10.0
    cluster_update_threshold: int = 10
    avoidance_sensitivity: float = 1.0
    pattern_memory_size: int = 50
    danger_zone_weight: float = 2.0

    # Performance thresholds
    max_processing_time_ms: float = 250
    circuit_breaker_threshold: int = 3
    min_cluster_quality: float = 0.3

    # Learning parameters
    learning_rate: float = 0.1
    false_positive_threshold: float = 0.2
    min_samples_for_clustering: int = 10


@module(
    **module_args(
        "MistakeMemory",
        description="Mistake/Pattern memory with clustering, circuit breaker, monitoring, and explainability.",
        error_handling=True,
        hot_reload=True,
        timeout_ms=120,
    )
)
class MistakeMemory(
    BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
):
    """
    Advanced mistake memory with SmartInfoBus integration.
    Learns from both losses and wins using clustering to identify danger and profit zones.

    CONTRACT (provided): mistake_memory, mistake_avoidance, danger_zones, pattern_recognition, loss_prevention
    """

    # ---------------- helpers -----------------

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _to_feature_vector(x: Any) -> Optional[np.ndarray]:
        """Try to coerce various inputs into a 1-D float vector."""
        try:
            if isinstance(x, np.ndarray):
                arr = x.astype(float, copy=False)
            else:
                arr = np.asarray(x, dtype=float)
            if arr.ndim == 0:
                return arr.reshape(1)
            if arr.ndim > 1:
                return arr.ravel()
            return arr
        except Exception:
            return None

    def _ensure_bus(self):
        try:
            self.smart_bus = InfoBusManager.get_instance()  # type: ignore[assignment]
        except Exception:
            # Never leave it as None; use no-op bus to satisfy type checkers / mixins
            self.smart_bus = _DummyBus()  # type: ignore[assignment]

    # ---------------- lifecycle ----------------

    def __init__(
        self,
        config: Optional[MistakeConfig] = None,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.mistake_config: MistakeConfig = config or MistakeConfig()

        # Make bus available before BaseModule.__init__ may call _initialize()
        self._ensure_bus()

        super().__init__(**kwargs)
        # Keep a dict-style config for any BaseModule expectations
        self.config: Dict[str, Any] = asdict(self.mistake_config)

        # systems, genome, state, monitoring
        self._initialize_advanced_systems()
        self._initialize_genome_parameters(genome)
        self._initialize_mistake_state()
        self._start_monitoring()

        self.logger.info(
            format_operator_message(
                "🧠",
                "MISTAKE_MEMORY_INITIALIZED",
                details=(
                    f"Max mistakes: {self.mistake_config.max_mistakes}, "
                    f"Clusters: {self.mistake_config.n_clusters}"
                ),
                result="Loss avoidance system ready",
                context="mistake_learning",
            )
        )

    def _initialize_advanced_systems(self):
        self.logger = RotatingLogger(
            name="MistakeMemory",
            log_path="logs/memory/mistake_memory.log",
            max_lines=3000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("MistakeMemory", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0,
            "state": "CLOSED",
            "threshold": self.mistake_config.circuit_breaker_threshold,
        }

        self._health_status = "healthy"
        self._last_health_check = time.time()

        self._ensure_bus()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        cfg = self.mistake_config
        if genome:
            self.genome = {
                "max_mistakes": int(genome.get("max_mistakes", cfg.max_mistakes)),
                "n_clusters": int(genome.get("n_clusters", cfg.n_clusters)),
                "profit_threshold": float(genome.get("profit_threshold", cfg.profit_threshold)),
                "cluster_update_threshold": int(
                    genome.get("cluster_update_threshold", cfg.cluster_update_threshold)
                ),
                "avoidance_sensitivity": float(
                    genome.get("avoidance_sensitivity", cfg.avoidance_sensitivity)
                ),
                "pattern_memory_size": int(
                    genome.get("pattern_memory_size", cfg.pattern_memory_size)
                ),
                "danger_zone_weight": float(
                    genome.get("danger_zone_weight", cfg.danger_zone_weight)
                ),
            }
        else:
            self.genome = {
                "max_mistakes": cfg.max_mistakes,
                "n_clusters": cfg.n_clusters,
                "profit_threshold": cfg.profit_threshold,
                "cluster_update_threshold": cfg.cluster_update_threshold,
                "avoidance_sensitivity": cfg.avoidance_sensitivity,
                "pattern_memory_size": cfg.pattern_memory_size,
                "danger_zone_weight": cfg.danger_zone_weight,
            }

    def _initialize_mistake_state(self):
        self._loss_buf: List[Tuple[np.ndarray, float, Dict]] = []
        self._win_buf: List[Tuple[np.ndarray, float, Dict]] = []

        self._km_loss: Optional[KMeans] = None
        self._km_win: Optional[KMeans] = None
        self._scaler = StandardScaler()

        self._mean_dist = 0.0
        self._last_dist = 0.0
        # store centers as plain python lists for JSON-ability; annotate clearly
        self._danger_zones: List[List[float]] = []
        self._profit_zones: List[List[float]] = []

        self._consecutive_losses = 0
        self._loss_patterns: Dict[str, Dict[str, Any]] = {}
        self._win_patterns: Dict[str, Dict[str, Any]] = {}
        self._avoidance_signal: float = 0.0

        self._pattern_evolution = deque(maxlen=100)
        self._danger_zone_violations = deque(maxlen=50)
        self._learning_effectiveness = deque(maxlen=200)
        self._market_context_correlations: Dict[str, float] = {}

        self._cluster_quality_scores: deque = deque(maxlen=20)
        self._prediction_accuracy = 0.0
        self._false_positive_rate = 0.0
        self._true_positive_rate = 0.0

        self._mistake_performance = {
            "total_losses_learned": 0,
            "total_wins_learned": 0,
            "avoidance_effectiveness": 0.0,
            "cluster_updates": 0,
            "pattern_discoveries": 0,
        }

    def _start_monitoring(self):
        def monitoring_loop():
            while getattr(self, "_monitoring_active", True):
                try:
                    self._update_mistake_health()
                    self._analyze_avoidance_effectiveness()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")

        self._monitoring_active = True
        threading.Thread(target=monitoring_loop, daemon=True).start()

    def _initialize(self):
        try:
            self._ensure_bus()

            # Initial publish (no-op if dummy bus)
            initial_status = {
                "danger_zones": [],
                "profit_zones": [],
                "avoidance_signal": 0.0,
                "consecutive_losses": 0,
                "pattern_count": 0,
            }
            self.smart_bus.set(
                "mistake_avoidance",
                initial_status,
                module="MistakeMemory",
                thesis="Initial mistake memory and avoidance status",
            )

            self.smart_bus.set(
                "mistake_memory",
                {
                    "current_score": 0.0,
                    "consecutive_losses": 0,
                    "avoidance_signal": 0.0,
                    "last_updated": time.time(),
                },
                module="MistakeMemory",
                thesis="Initial mistake memory summary for consumers",
            )
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    # ---------------- main processing ----------------

    async def process(self, **inputs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            learning_data = await self._extract_learning_data(**inputs)
            if not learning_data:
                return await self._handle_no_data_fallback()

            learning_result = await self._process_learning_data(learning_data)
            learning_result.update(await self._update_clustering())
            learning_result.update(await self._calculate_avoidance_signals(learning_data))

            thesis = await self._generate_mistake_thesis(learning_data, learning_result)
            await self._update_mistake_smart_bus(learning_result, thesis)

            self._record_success((time.time() - start_time) * 1000)

            current_score = self._safe_float(np.clip(abs(self._avoidance_signal), 0.0, 1.0))
            mistake_summary = {
                "current_score": current_score,
                "consecutive_losses": int(self._consecutive_losses),
                "avoidance_signal": self._safe_float(self._avoidance_signal),
                "last_updated": time.time(),
            }

            provides_payload = {
                "mistake_memory": mistake_summary,
                "mistake_avoidance": {
                    "avoidance_signal": self._safe_float(self._avoidance_signal),
                    "consecutive_losses": int(self._consecutive_losses),
                    "danger_zones_count": len(self._danger_zones),
                    "profit_zones_count": len(self._profit_zones),
                    "total_loss_memories": len(self._loss_buf),
                    "total_win_memories": len(self._win_buf),
                },
                "danger_zones": {
                    "zones": self._danger_zones,
                    "zone_count": len(self._danger_zones),
                    "avoidance_sensitivity": self.genome["avoidance_sensitivity"],
                    "last_updated": time.time(),
                },
                "pattern_recognition": {
                    "loss_patterns": dict(list(self._loss_patterns.items())[:10]),
                    "win_patterns": dict(list(self._win_patterns.items())[:10]),
                    "total_loss_patterns": len(self._loss_patterns),
                    "total_win_patterns": len(self._win_patterns),
                    "pattern_memory_size": int(self.genome["pattern_memory_size"]),
                },
                "loss_prevention": {
                    "avoidance_effectiveness": self._safe_float(
                        self._mistake_performance["avoidance_effectiveness"]
                    ),
                    "false_positive_rate": self._safe_float(self._false_positive_rate),
                    "true_positive_rate": self._safe_float(self._true_positive_rate),
                    "cluster_quality": self._safe_float(
                        np.mean(list(self._cluster_quality_scores)) if self._cluster_quality_scores else 0.0
                    ),
                    "learning_samples": len(self._loss_buf) + len(self._win_buf),
                },
                "_thesis": thesis,
            }
            learning_result.update(provides_payload)
            return learning_result

        except Exception as e:
            return await self._handle_mistake_error(e, start_time)

    async def _extract_learning_data(self, **inputs) -> Optional[Dict[str, Any]]:
        try:
            self._ensure_bus()
            trades = self.smart_bus.get("trades", "MistakeMemory") or []
            features = self.smart_bus.get("features", "MistakeMemory")
            market_context = self.smart_bus.get("market_context", "MistakeMemory") or {}

            risk_data = (
                self.smart_bus.get("time_risk_analysis", "MistakeMemory")
                or self.smart_bus.get("risk_data", "MistakeMemory")
                or {}
            )

            current_features = inputs.get("features", features)

            return {
                "trades": trades,
                "features": features,
                "current_features": current_features,
                "market_context": market_context,
                "risk_data": risk_data,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Failed to extract learning data: {e}")
            return None

    async def _process_learning_data(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            trades = learning_data.get("trades", [])
            market_context = learning_data.get("market_context", {})

            if not trades:
                return {"learning_processed": False, "reason": "no_trades"}

            losses_learned = 0
            wins_learned = 0

            for trade in trades[-20:]:
                if not isinstance(trade, dict) or "pnl" not in trade:
                    continue

                trade_features = self._extract_trade_features(trade, market_context)
                if trade_features is None:
                    continue

                pnl = self._safe_float(trade.get("pnl", 0.0))
                if pnl < -self.genome["profit_threshold"] / 2:
                    self._process_loss_trade(trade_features, abs(pnl), trade)
                    losses_learned += 1
                elif pnl > self.genome["profit_threshold"]:
                    self._process_win_trade(trade_features, pnl, trade)
                    wins_learned += 1

            self._mistake_performance["total_losses_learned"] += losses_learned
            self._mistake_performance["total_wins_learned"] += wins_learned

            return {
                "learning_processed": True,
                "losses_learned": losses_learned,
                "wins_learned": wins_learned,
                "total_loss_memories": len(self._loss_buf),
                "total_win_memories": len(self._win_buf),
            }

        except Exception as e:
            self.logger.error(f"Learning data processing failed: {e}")
            return {"learning_processed": False, "error": str(e)}

    def _extract_trade_features(self, trade: Dict[str, Any], market_context: Dict[str, Any]) -> Optional[np.ndarray]:
        try:
            features: List[float] = []

            if "confidence" in trade:
                features.append(self._safe_float(trade.get("confidence", 0.0)))
            if "volume" in trade:
                features.append(self._safe_float(trade.get("volume", 1.0)))
            if "duration" in trade:
                features.append(self._safe_float(trade.get("duration", 1.0)))

            if "volatility" in market_context:
                vol = market_context["volatility"]
                if isinstance(vol, dict):
                    vals: List[float] = []
                    for v in vol.values():
                        try:
                            vals.append(float(v))
                        except Exception:
                            vals.append(0.0)
                    features.extend(vals[:3])
                else:
                    features.append(self._safe_float(vol, 0.0))

            if "session" in market_context:
                session_map = {"asian": 0.0, "european": 0.5, "us": 1.0}
                features.append(session_map.get(str(market_context["session"]).lower(), 0.25))

            if "regime" in market_context:
                regime_map = {"trending": 1.0, "ranging": 0.0, "volatile": 0.5}
                features.append(regime_map.get(str(market_context["regime"]).lower(), 0.25))

            while len(features) < 8:
                features.append(0.0)

            return np.array(features[:20], dtype=float)

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def _process_loss_trade(self, features: np.ndarray, loss_amount: float, trade_info: Dict[str, Any]):
        try:
            self._loss_buf.append((features, float(loss_amount), dict(trade_info)))
            if len(self._loss_buf) > self.genome["max_mistakes"]:
                self._loss_buf.pop(0)

            self._consecutive_losses += 1

            pattern = self._extract_pattern(features, trade_info)
            if pattern:
                self._record_loss_pattern(pattern, float(loss_amount))

            if loss_amount > self.genome["profit_threshold"]:
                self.logger.warning(
                    format_operator_message(
                        "💸",
                        "SIGNIFICANT_LOSS_LEARNED",
                        loss_amount=f"{loss_amount:.2f}",
                        consecutive_losses=int(self._consecutive_losses),
                        pattern=(pattern[:10] if pattern else "unknown"),
                        context="loss_learning",
                    )
                )
        except Exception as e:
            self.logger.error(f"Loss trade processing failed: {e}")

    def _process_win_trade(self, features: np.ndarray, profit_amount: float, trade_info: Dict[str, Any]):
        try:
            self._win_buf.append((features, float(profit_amount), dict(trade_info)))
            if len(self._win_buf) > self.genome["max_mistakes"]:
                self._win_buf.pop(0)

            self._consecutive_losses = 0

            pattern = self._extract_pattern(features, trade_info)
            if pattern:
                self._record_win_pattern(pattern, float(profit_amount))
        except Exception as e:
            self.logger.error(f"Win trade processing failed: {e}")

    def _extract_pattern(self, features: np.ndarray, trade_info: Dict[str, Any]) -> Optional[str]:
        try:
            pattern_elements: List[str] = []

            for i, feature in enumerate(features[:5]):
                if feature > 0.7:
                    pattern_elements.append(f"H{i}")
                elif feature < 0.3:
                    pattern_elements.append(f"L{i}")
                else:
                    pattern_elements.append(f"M{i}")

            if "action" in trade_info:
                action = trade_info["action"]
                if isinstance(action, (list, tuple, np.ndarray)) and len(action) > 0:
                    a0 = self._safe_float(action[0], 0.0)
                    if a0 > 0.5:
                        pattern_elements.append("BUY")
                    elif a0 < -0.5:
                        pattern_elements.append("SELL")
                    else:
                        pattern_elements.append("HOLD")

            return "_".join(pattern_elements) if pattern_elements else None
        except Exception as e:
            self.logger.error(f"Pattern extraction failed: {e}")
            return None

    def _record_loss_pattern(self, pattern: str, loss_amount: float):
        if pattern not in self._loss_patterns:
            self._loss_patterns[pattern] = {"count": 0, "total_severity": 0.0, "last_seen": time.time()}
        self._loss_patterns[pattern]["count"] += 1
        self._loss_patterns[pattern]["total_severity"] += float(loss_amount)
        self._loss_patterns[pattern]["last_seen"] = time.time()

    def _record_win_pattern(self, pattern: str, profit_amount: float):
        if pattern not in self._win_patterns:
            self._win_patterns[pattern] = {"count": 0, "total_profitability": 0.0, "last_seen": time.time()}
        self._win_patterns[pattern]["count"] += 1
        self._win_patterns[pattern]["total_profitability"] += float(profit_amount)
        self._win_patterns[pattern]["last_seen"] = time.time()

    async def _update_clustering(self) -> Dict[str, Any]:
        try:
            total_samples = len(self._loss_buf) + len(self._win_buf)
            if total_samples > 0 and total_samples % max(1, int(self.genome["cluster_update_threshold"])) == 0:
                clustering_result = await self._perform_clustering()
                self._mistake_performance["cluster_updates"] += 1
                return {"clustering_updated": True, "clustering_result": clustering_result}
            return {"clustering_updated": False}
        except Exception as e:
            self.logger.error(f"Clustering update failed: {e}")
            return {"clustering_updated": False, "error": str(e)}

    async def _perform_clustering(self) -> Dict[str, Any]:
        try:
            results: Dict[str, Any] = {}
            if len(self._loss_buf) >= self.mistake_config.min_samples_for_clustering:
                results["loss_clustering"] = await self._cluster_loss_data()
            if len(self._win_buf) >= self.mistake_config.min_samples_for_clustering:
                results["win_clustering"] = await self._cluster_win_data()
            return results
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {"error": str(e)}

    async def _cluster_loss_data(self) -> Dict[str, Any]:
        try:
            features = np.array([entry[0] for entry in self._loss_buf])
            losses = np.array([entry[1] for entry in self._loss_buf], dtype=float)

            features_scaled = self._scaler.fit_transform(features)

            n_clusters = min(max(1, int(self.genome["n_clusters"])), len(features))
            self._km_loss = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_labels = self._km_loss.fit_predict(features_scaled)

            if len(np.unique(cluster_labels)) > 1:
                quality_score = silhouette_score(features_scaled, cluster_labels)
                self._cluster_quality_scores.append(float(quality_score))
            else:
                quality_score = 0.0

            self._danger_zones = self._km_loss.cluster_centers_.tolist()

            cluster_severities: Dict[int, float] = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_severities[i] = float(np.mean(losses[cluster_mask])) if np.any(cluster_mask) else 0.0

            return {
                "n_clusters": n_clusters,
                "quality_score": float(quality_score),
                "cluster_severities": cluster_severities,
                "danger_zones_count": len(self._danger_zones),
            }

        except Exception as e:
            self.logger.error(f"Loss clustering failed: {e}")
            return {"error": str(e)}

    async def _cluster_win_data(self) -> Dict[str, Any]:
        try:
            features = np.array([entry[0] for entry in self._win_buf])
            profits = np.array([entry[1] for entry in self._win_buf], dtype=float)

            try:
                features_scaled = self._scaler.transform(features)
            except NotFittedError:
                features_scaled = self._scaler.fit_transform(features)

            n_clusters = min(max(1, int(self.genome["n_clusters"])), len(features))
            self._km_win = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_labels = self._km_win.fit_predict(features_scaled)

            if len(np.unique(cluster_labels)) > 1:
                quality_score = silhouette_score(features_scaled, cluster_labels)
            else:
                quality_score = 0.0

            self._profit_zones = self._km_win.cluster_centers_.tolist()

            cluster_profitabilities: Dict[int, float] = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_profitabilities[i] = float(np.mean(profits[cluster_mask])) if np.any(cluster_mask) else 0.0

            return {
                "n_clusters": n_clusters,
                "quality_score": float(quality_score),
                "cluster_profitabilities": cluster_profitabilities,
                "profit_zones_count": len(self._profit_zones),
            }

        except Exception as e:
            self.logger.error(f"Win clustering failed: {e}")
            return {"error": str(e)}

    async def _calculate_avoidance_signals(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current_features_raw = learning_data.get("current_features")
            current_features = self._to_feature_vector(current_features_raw)
            if current_features is None:
                return {"avoidance_signal": 0.0, "danger_similarity": 0.0, "profit_similarity": 0.0}

            danger_similarity = self._calculate_danger_similarity(current_features)
            profit_similarity = self._calculate_profit_similarity(current_features)

            avoidance_signal = danger_similarity * self.genome["avoidance_sensitivity"]

            if profit_similarity > danger_similarity:
                avoidance_signal *= 0.5

            if self._consecutive_losses > 2:
                avoidance_signal *= (1.0 + self._consecutive_losses * 0.1)

            self._avoidance_signal = float(avoidance_signal)

            return {
                "avoidance_signal": float(avoidance_signal),
                "danger_similarity": float(danger_similarity),
                "profit_similarity": float(profit_similarity),
                "consecutive_losses": int(self._consecutive_losses),
            }

        except Exception as e:
            self.logger.error(f"Avoidance signal calculation failed: {e}")
            return {"avoidance_signal": 0.0, "danger_similarity": 0.0, "profit_similarity": 0.0, "error": str(e)}

    def _calculate_danger_similarity(self, features: np.ndarray) -> float:
        if not self._danger_zones or self._km_loss is None:
            return 0.0
        try:
            try:
                features_scaled = self._scaler.transform(features.reshape(1, -1))
            except NotFittedError:
                centers = np.asarray(self._danger_zones, dtype=float)
                if centers.size:
                    self._scaler.fit(centers)
                    features_scaled = self._scaler.transform(features.reshape(1, -1))
                else:
                    features_scaled = features.reshape(1, -1).astype(float)

            distances = []
            for center in self._danger_zones:
                c = np.asarray(center, dtype=float)
                distances.append(float(np.linalg.norm(features_scaled[0] - c)))

            if not distances:
                return 0.0

            distances_arr = np.asarray(distances, dtype=float)
            min_distance = float(distances_arr.min())
            return float(1.0 / (1.0 + min_distance))
        except Exception as e:
            self.logger.error(f"Danger similarity calculation failed: {e}")
            return 0.0

    def _calculate_profit_similarity(self, features: np.ndarray) -> float:
        if not self._profit_zones or self._km_win is None:
            return 0.0
        try:
            try:
                features_scaled = self._scaler.transform(features.reshape(1, -1))
            except NotFittedError:
                centers = np.asarray(self._profit_zones, dtype=float)
                if centers.size:
                    self._scaler.fit(centers)
                    features_scaled = self._scaler.transform(features.reshape(1, -1))
                else:
                    features_scaled = features.reshape(1, -1).astype(float)

            distances = []
            for center in self._profit_zones:
                c = np.asarray(center, dtype=float)
                distances.append(float(np.linalg.norm(features_scaled[0] - c)))

            if not distances:
                return 0.0

            distances_arr = np.asarray(distances, dtype=float)
            min_distance = float(distances_arr.min())
            return float(1.0 / (1.0 + min_distance))
        except Exception as e:
            self.logger.error(f"Profit similarity calculation failed: {e}")
            return 0.0

    async def _generate_mistake_thesis(self, _learning_data: Dict[str, Any], learning_result: Dict[str, Any]) -> str:
        try:
            total_losses = len(self._loss_buf)
            total_wins = len(self._win_buf)
            consecutive_losses = int(self._consecutive_losses)

            avoidance_signal = self._safe_float(learning_result.get("avoidance_signal", 0.0))
            danger_similarity = self._safe_float(learning_result.get("danger_similarity", 0.0))

            loss_patterns = len(self._loss_patterns)
            win_patterns = len(self._win_patterns)

            parts = [
                f"Mistake Memory Analysis: {total_losses} losses and {total_wins} wins stored for pattern recognition",
                f"Avoidance system: {avoidance_signal:.3f} signal with {consecutive_losses} consecutive losses",
                f"Pattern recognition: {loss_patterns} loss patterns and {win_patterns} win patterns identified",
            ]

            if learning_result.get("clustering_updated", False):
                cr = learning_result.get("clustering_result", {})
                dz = cr.get("loss_clustering", {}).get("danger_zones_count", 0)
                pz = cr.get("win_clustering", {}).get("profit_zones_count", 0)
                parts.append(f"Clustering updated: {dz} danger zones and {pz} profit zones")

            if danger_similarity > 0.5:
                parts.append(f"HIGH DANGER: Similar to loss patterns (sim={danger_similarity:.2f})")
            elif danger_similarity > 0.3:
                parts.append("MODERATE RISK: Some similarity to loss patterns")
            else:
                parts.append("LOW RISK: Dissimilar to known loss patterns")

            if self._cluster_quality_scores:
                avg_quality = float(np.mean(list(self._cluster_quality_scores)[-3:]))
                parts.append(f"Learning quality: {avg_quality:.2f} clustering effectiveness")

            if consecutive_losses > 3:
                parts.append(f"WARNING: {consecutive_losses} consecutive losses – elevated avoidance active")

            return " | ".join(parts)
        except Exception as e:
            return f"Mistake thesis generation failed: {str(e)} - Loss avoidance system maintaining basic functionality"

    async def _update_mistake_smart_bus(self, _learning_result: Dict[str, Any], thesis: str):
        try:
            self._ensure_bus()

            self.smart_bus.set(
                "mistake_avoidance",
                {
                    "avoidance_signal": self._safe_float(self._avoidance_signal),
                    "consecutive_losses": int(self._consecutive_losses),
                    "danger_zones_count": len(self._danger_zones),
                    "profit_zones_count": len(self._profit_zones),
                    "total_loss_memories": len(self._loss_buf),
                    "total_win_memories": len(self._win_buf),
                },
                module="MistakeMemory",
                thesis=thesis,
            )

            self.smart_bus.set(
                "danger_zones",
                {
                    "zones": self._danger_zones,
                    "zone_count": len(self._danger_zones),
                    "avoidance_sensitivity": self.genome["avoidance_sensitivity"],
                    "last_updated": time.time(),
                },
                module="MistakeMemory",
                thesis=f"Identified {len(self._danger_zones)} danger zones from clustering analysis",
            )

            self.smart_bus.set(
                "pattern_recognition",
                {
                    "loss_patterns": dict(list(self._loss_patterns.items())[:10]),
                    "win_patterns": dict(list(self._win_patterns.items())[:10]),
                    "total_loss_patterns": len(self._loss_patterns),
                    "total_win_patterns": len(self._win_patterns),
                    "pattern_memory_size": int(self.genome["pattern_memory_size"]),
                },
                module="MistakeMemory",
                thesis="Pattern recognition from trading outcomes for loss avoidance",
            )

            self.smart_bus.set(
                "loss_prevention",
                {
                    "avoidance_effectiveness": self._safe_float(self._mistake_performance["avoidance_effectiveness"]),
                    "false_positive_rate": self._safe_float(self._false_positive_rate),
                    "true_positive_rate": self._safe_float(self._true_positive_rate),
                    "cluster_quality": self._safe_float(
                        np.mean(list(self._cluster_quality_scores)) if self._cluster_quality_scores else 0.0
                    ),
                    "learning_samples": len(self._loss_buf) + len(self._win_buf),
                },
                module="MistakeMemory",
                thesis="Loss prevention effectiveness and learning metrics",
            )

            current_score = self._safe_float(np.clip(abs(self._avoidance_signal), 0.0, 1.0))
            self.smart_bus.set(
                "mistake_memory",
                {
                    "current_score": current_score,
                    "consecutive_losses": int(self._consecutive_losses),
                    "avoidance_signal": self._safe_float(self._avoidance_signal),
                    "last_updated": time.time(),
                },
                module="MistakeMemory",
                thesis="Summary score and basics for mistake memory consumers",
            )

        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    # ---------------- fallbacks & errors ----------------

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        self.logger.warning("No learning data available - using cached mistake memory")
        thesis = "No learning data available – serving cached mistake memory state"

        current_score = self._safe_float(np.clip(abs(self._avoidance_signal), 0.0, 1.0))
        mistake_summary = {
            "current_score": current_score,
            "consecutive_losses": int(self._consecutive_losses),
            "avoidance_signal": self._safe_float(self._avoidance_signal),
            "last_updated": time.time(),
        }

        return {
            "avoidance_signal": self._safe_float(self._avoidance_signal),
            "total_loss_memories": len(self._loss_buf),
            "total_win_memories": len(self._win_buf),
            "consecutive_losses": int(self._consecutive_losses),
            "fallback_reason": "no_learning_data",
            "mistake_memory": mistake_summary,
            "mistake_avoidance": {
                "avoidance_signal": self._safe_float(self._avoidance_signal),
                "consecutive_losses": int(self._consecutive_losses),
                "danger_zones_count": len(self._danger_zones),
                "profit_zones_count": len(self._profit_zones),
                "total_loss_memories": len(self._loss_buf),
                "total_win_memories": len(self._win_buf),
            },
            "danger_zones": {
                "zones": self._danger_zones,
                "zone_count": len(self._danger_zones),
                "avoidance_sensitivity": self.genome["avoidance_sensitivity"],
                "last_updated": time.time(),
            },
            "pattern_recognition": {
                "loss_patterns": dict(list(self._loss_patterns.items())[:10]),
                "win_patterns": dict(list(self._win_patterns.items())[:10]),
                "total_loss_patterns": len(self._loss_patterns),
                "total_win_patterns": len(self._win_patterns),
                "pattern_memory_size": int(self.genome["pattern_memory_size"]),
            },
            "loss_prevention": {
                "avoidance_effectiveness": self._safe_float(self._mistake_performance["avoidance_effectiveness"]),
                "false_positive_rate": self._safe_float(self._false_positive_rate),
                "true_positive_rate": self._safe_float(self._true_positive_rate),
                "cluster_quality": self._safe_float(
                    np.mean(list(self._cluster_quality_scores)) if self._cluster_quality_scores else 0.0
                ),
                "learning_samples": len(self._loss_buf) + len(self._win_buf),
            },
            "_thesis": thesis,
        }

    async def _handle_mistake_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        processing_time = (time.time() - start_time) * 1000
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()
        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
            self.circuit_breaker["state"] = "OPEN"

        explanation = self.english_explainer.explain_error("MistakeMemory", str(error), "mistake learning")
        self.logger.error(
            format_operator_message(
                "[CRASH]",
                "MISTAKE_MEMORY_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="mistake_learning",
            )
        )
        self._record_failure(error)
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        thesis = f"Mistake memory encountered issues; serving last known state ({reason})"
        return {
            "avoidance_signal": self._safe_float(self._avoidance_signal),
            "total_loss_memories": len(self._loss_buf),
            "total_win_memories": len(self._win_buf),
            "consecutive_losses": int(self._consecutive_losses),
            "circuit_breaker_state": self.circuit_breaker["state"],
            "fallback_reason": reason,
            "mistake_avoidance": {
                "avoidance_signal": self._safe_float(self._avoidance_signal),
                "consecutive_losses": int(self._consecutive_losses),
                "danger_zones_count": len(self._danger_zones),
                "profit_zones_count": len(self._profit_zones),
                "total_loss_memories": len(self._loss_buf),
                "total_win_memories": len(self._win_buf),
            },
            "danger_zones": {
                "zones": self._danger_zones,
                "zone_count": len(self._danger_zones),
                "avoidance_sensitivity": self.genome["avoidance_sensitivity"],
                "last_updated": time.time(),
            },
            "pattern_recognition": {
                "loss_patterns": dict(list(self._loss_patterns.items())[:10]),
                "win_patterns": dict(list(self._win_patterns.items())[:10]),
                "total_loss_patterns": len(self._loss_patterns),
                "total_win_patterns": len(self._win_patterns),
                "pattern_memory_size": int(self.genome["pattern_memory_size"]),
            },
            "loss_prevention": {
                "avoidance_effectiveness": self._safe_float(self._mistake_performance["avoidance_effectiveness"]),
                "false_positive_rate": self._safe_float(self._false_positive_rate),
                "true_positive_rate": self._safe_float(self._true_positive_rate),
                "cluster_quality": self._safe_float(
                    np.mean(list(self._cluster_quality_scores)) if self._cluster_quality_scores else 0.0
                ),
                "learning_samples": len(self._loss_buf) + len(self._win_buf),
            },
            "_thesis": thesis,
        }

    # ---------------- health & metrics ----------------

    def _update_mistake_health(self):
        try:
            if self._cluster_quality_scores:
                avg_quality = float(np.mean(list(self._cluster_quality_scores)[-3:]))
                self._health_status = "healthy" if avg_quality >= self.mistake_config.min_cluster_quality else "warning"

            if self._consecutive_losses > 5:
                self._health_status = "warning"

            self._last_health_check = time.time()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = "warning"

    def _analyze_avoidance_effectiveness(self):
        try:
            if len(self._learning_effectiveness) >= 10:
                recent = list(self._learning_effectiveness)[-10:]
                avg_eff = float(np.mean(recent))
                self._mistake_performance["avoidance_effectiveness"] = avg_eff
                if avg_eff > 0.7:
                    self.logger.info(
                        format_operator_message(
                            "[SAFE]",
                            "HIGH_AVOIDANCE_EFFECTIVENESS",
                            effectiveness=f"{avg_eff:.2f}",
                            danger_zones=len(self._danger_zones),
                            context="avoidance_analysis",
                        )
                    )
        except Exception as e:
            self.logger.error(f"Avoidance effectiveness analysis failed: {e}")

    def _record_success(self, processing_time: float):
        self.performance_tracker.record_metric("MistakeMemory", "learning_cycle", processing_time, True)
        if self.circuit_breaker["state"] == "OPEN":
            self.circuit_breaker["failures"] = 0
            self.circuit_breaker["state"] = "CLOSED"

    def _record_failure(self, _error: Exception):
        self.performance_tracker.record_metric("MistakeMemory", "learning_cycle", 0, False)

    # ---------------- persistence & legacy ----------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "loss_buffer": [(entry[0].tolist(), float(entry[1]), entry[2]) for entry in self._loss_buf[-50:]],
            "win_buffer": [(entry[0].tolist(), float(entry[1]), entry[2]) for entry in self._win_buf[-50:]],
            "loss_patterns": dict(self._loss_patterns),
            "win_patterns": dict(self._win_patterns),
            "danger_zones": self._danger_zones,
            "profit_zones": self._profit_zones,
            "genome": self.genome.copy(),
            "consecutive_losses": int(self._consecutive_losses),
            "avoidance_signal": self._safe_float(self._avoidance_signal),
            "mistake_performance": dict(self._mistake_performance),
            "circuit_breaker": dict(self.circuit_breaker),
            "health_status": self._health_status,
            "config": asdict(self.mistake_config),
        }

    def set_state(self, state: Dict[str, Any]):
        if "loss_buffer" in state:
            self._loss_buf = [(np.array(entry[0], dtype=float), float(entry[1]), entry[2]) for entry in state["loss_buffer"]]
        if "win_buffer" in state:
            self._win_buf = [(np.array(entry[0], dtype=float), float(entry[1]), entry[2]) for entry in state["win_buffer"]]
        if "loss_patterns" in state:
            self._loss_patterns = dict(state["loss_patterns"])
        if "win_patterns" in state:
            self._win_patterns = dict(state["win_patterns"])
        if "danger_zones" in state:
            self._danger_zones = state["danger_zones"]
        if "profit_zones" in state:
            self._profit_zones = state["profit_zones"]
        if "genome" in state:
            self.genome.update(state["genome"])
        if "consecutive_losses" in state:
            self._consecutive_losses = int(state["consecutive_losses"])
        if "avoidance_signal" in state:
            self._avoidance_signal = self._safe_float(state["avoidance_signal"])
        if "mistake_performance" in state:
            self._mistake_performance.update(state["mistake_performance"])
        if "circuit_breaker" in state:
            self.circuit_breaker.update(state["circuit_breaker"])
        if "health_status" in state:
            self._health_status = str(state["health_status"])

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": self._health_status,
            "last_check": self._last_health_check,
            "circuit_breaker": self.circuit_breaker["state"],
            "total_memories": len(self._loss_buf) + len(self._win_buf),
            "consecutive_losses": int(self._consecutive_losses),
            "avoidance_signal": self._safe_float(self._avoidance_signal),
        }

    def stop_monitoring(self):
        self._monitoring_active = False

    def check_similarity_to_mistakes(self, features: np.ndarray) -> float:
        f = self._to_feature_vector(features)
        if f is None:
            return 0.0
        return self._calculate_danger_similarity(f)

    async def propose_action(self, **_inputs) -> Dict[str, Any]:
        avoidance_action = [-self._safe_float(self._avoidance_signal), 0.0]
        confidence = max(0.0, 1.0 - self._safe_float(self._avoidance_signal))
        return {
            "action": avoidance_action,
            "confidence": float(confidence),
            "thesis": f"Mistake avoidance action with {self._safe_float(self._avoidance_signal):.3f} avoidance signal",
            "avoidance_signal": self._safe_float(self._avoidance_signal),
            "consecutive_losses": int(self._consecutive_losses),
        }

    async def calculate_confidence(self, action: Dict[str, Any], **_inputs) -> float:
        if not isinstance(action, dict):
            return 0.5
        base_confidence = max(0.0, 1.0 - self._safe_float(self._avoidance_signal))
        if self._cluster_quality_scores:
            avg_quality = float(np.mean(list(self._cluster_quality_scores)))
            quality_factor = avg_quality
        else:
            quality_factor = 0.5
        consecutive_penalty = min(0.3, int(self._consecutive_losses) * 0.05)
        memory_count = len(self._loss_buf) + len(self._win_buf)
        memory_factor = min(1.0, memory_count / 100.0)
        cb_factor = 0.5 if self.circuit_breaker["state"] == "OPEN" else 1.0
        confidence = (base_confidence * 0.4 + quality_factor * 0.3 + memory_factor * 0.2 + cb_factor * 0.1) - consecutive_penalty
        return float(max(0.0, min(1.0, float(confidence))))

    def confidence(self, obs: Any = None, **_kwargs) -> float:
        return max(0.0, 1.0 - self._safe_float(self._avoidance_signal))
