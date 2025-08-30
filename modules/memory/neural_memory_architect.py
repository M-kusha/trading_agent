# ─────────────────────────────────────────────────────────────
# File: modules/memory/neural_memory_architect.py
# [ROCKET] PRODUCTION-READY Neural Memory Architecture System
# Advanced neural memory with attention mechanisms and SmartInfoBus integration
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Deque, Union
from collections import deque

import numpy as np
import torch
import torch.nn as nn

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


@dataclass
class NeuralMemoryConfig:
    """Configuration for Neural Memory Architect"""
    embed_dim: int = 32
    num_heads: int = 4
    max_len: int = 500
    memory_decay: float = 0.95
    importance_threshold: float = 0.3
    retrieval_top_k: int = 5
    learning_rate: float = 0.001
    attention_dropout: float = 0.1

    # Performance thresholds
    max_processing_time_ms: float = 400
    circuit_breaker_threshold: int = 3
    min_memory_quality: float = 0.4

    # Neural parameters
    hidden_multiplier: int = 2
    context_features: int = 8
    quality_threshold: float = 0.5


@module(**module_args(
    "NeuralMemoryArchitect",
    description="Deterministic multi-window feature extraction with circuit breaker, monitoring, and explainability.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=120,
))
class NeuralMemoryArchitect(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced neural memory architect with SmartInfoBus integration.
    Uses attention mechanisms and neural networks for intelligent memory storage and retrieval.
    """

    # ─────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────
    def __init__(
        self,
        config: Optional[NeuralMemoryConfig] = None,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # Typed config we use internally
        self.neural_config: NeuralMemoryConfig = config or NeuralMemoryConfig()

        # Flags required before BaseModule possibly calls _initialize()
        self._monitoring_active: bool = False

        # Initialize advanced systems BEFORE BaseModule init
        self._initialize_advanced_systems()

        # Pass a dict config to BaseModule (keeps linters happy)
        base_cfg: Dict[str, Any] = asdict(self.neural_config)
        super().__init__(config=base_cfg)

        # Initialize genome and neural state/components
        self._initialize_genome_parameters(genome)
        self._initialize_neural_state()
        self._initialize_neural_components()

        # Start monitoring after everything exists
        self._start_monitoring()

        self.logger.info(
            format_operator_message(
                "🧠",
                "NEURAL_MEMORY_ARCHITECT_INITIALIZED",
                details=f"Embedding dim: {self.neural_config.embed_dim}, Heads: {self.neural_config.num_heads}",
                result="Neural memory system ready",
                context="neural_memory",
            )
        )

    # ─────────────────────────────────────────────────────────────
    # ADVANCED SYSTEMS / INIT HELPERS
    # ─────────────────────────────────────────────────────────────
    def _initialize_advanced_systems(self) -> None:
        """Initialize advanced systems for neural memory"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="NeuralMemoryArchitect",
            log_path="logs/memory/neural_memory.log",
            max_lines=3000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("NeuralMemoryArchitect", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker for neural operations
        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",
            "threshold": int(self.neural_config.circuit_breaker_threshold),
        }

        # Health monitoring
        self._health_status: str = "healthy"
        self._last_health_check: float = time.time()

        # Device (CPU only here)
        self.device = torch.device("cpu")

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]) -> None:
        """Initialize genome-based parameters"""
        cfg = self.neural_config
        g = genome or {}
        self.genome: Dict[str, Any] = {
            "embed_dim": int(g.get("embed_dim", cfg.embed_dim)),
            "num_heads": int(g.get("num_heads", cfg.num_heads)),
            "max_len": int(g.get("max_len", cfg.max_len)),
            "memory_decay": float(g.get("memory_decay", cfg.memory_decay)),
            "importance_threshold": float(g.get("importance_threshold", cfg.importance_threshold)),
            "retrieval_top_k": int(g.get("retrieval_top_k", cfg.retrieval_top_k)),
            "learning_rate": float(g.get("learning_rate", cfg.learning_rate)),
            "attention_dropout": float(g.get("attention_dropout", cfg.attention_dropout)),
        }

    def _initialize_neural_state(self) -> None:
        """Initialize neural memory state"""
        ed = int(self.genome["embed_dim"])
        self.buffer: torch.Tensor = torch.zeros((0, ed), dtype=torch.float32)
        self.importance_scores: torch.Tensor = torch.zeros(0, dtype=torch.float32)
        self.memory_metadata: List[Dict[str, Any]] = []

        # Enhanced tracking
        self._memory_usage_history: Deque[float] = deque(maxlen=200)
        self._retrieval_history: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._importance_evolution: Deque[float] = deque(maxlen=500)
        self._attention_patterns: Deque[Any] = deque(maxlen=50)

        # Performance analytics
        self._storage_efficiency: float = 0.0
        self._retrieval_accuracy: float = 0.0
        self._memory_turnover_rate: float = 0.0
        self._neural_performance_score: float = 100.0

        # Learning analytics
        self._learning_curves: Dict[str, Deque[float]] = {
            "importance_prediction": deque(maxlen=100),
            "attention_focus": deque(maxlen=100),
            "memory_utilization": deque(maxlen=100),
        }

        # Adaptive parameters
        self._adaptive_params: Dict[str, float] = {
            "importance_scaling": 1.0,
            "attention_temperature": 1.0,
            "decay_adjustment": 1.0,
            "quality_threshold": float(self.neural_config.quality_threshold),
        }

        # Performance metrics
        self._neural_performance: Dict[str, Any] = {
            "memories_stored": 0,
            "memories_retrieved": 0,
            "average_importance": 0.0,
            "attention_efficiency": 0.0,
        }

    def _initialize_neural_components(self) -> None:
        """Initialize neural network components"""
        try:
            ed = int(self.genome["embed_dim"])
            hm = int(self.neural_config.hidden_multiplier)

            # Encoder network
            self.encoder = nn.Sequential(
                nn.Linear(ed, ed * hm),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(ed * hm, ed),
                nn.LayerNorm(ed),
            ).to(self.device)

            # Multi-head attention
            self.attn = nn.MultiheadAttention(
                ed,
                int(self.genome["num_heads"]),
                dropout=float(self.genome["attention_dropout"]),
                batch_first=True,
            ).to(self.device)

            # Importance prediction head
            self.value_head = nn.Sequential(
                nn.Linear(ed, ed // 2),
                nn.ReLU(),
                nn.Linear(ed // 2, 1),
                nn.Sigmoid(),
            ).to(self.device)

            # Context integration network
            self.context_net = nn.Sequential(
                nn.Linear(ed + int(self.neural_config.context_features), ed),
                nn.ReLU(),
                nn.Linear(ed, ed),
            ).to(self.device)

            self._initialize_weights()
            self.logger.info("Neural components initialized successfully")

        except Exception as e:
            self.logger.error(f"Neural component initialization failed: {e}")
            self._health_status = "error"

    def _initialize_weights(self) -> None:
        """Initialize neural network weights"""
        for module in [self.encoder, self.value_head, self.context_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _start_monitoring(self) -> None:
        """Start background monitoring"""
        if self._monitoring_active:
            return

        def monitoring_loop() -> None:
            while getattr(self, "_monitoring_active", False):
                try:
                    self._update_neural_health()
                    self._analyze_memory_efficiency()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")

        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    # Called by BaseModule during construction
    def _initialize(self) -> None:
        """BaseModule hook: set initial SmartInfoBus snapshot."""
        try:
            if not hasattr(self, "smart_bus"):
                return
            initial_status = {
                "buffer_size": 0,
                "memory_utilization": 0.0,
                "average_importance": 0.0,
                "neural_performance": 100.0,
            }
            self.smart_bus.set(
                "neural_memory",
                initial_status,
                module="NeuralMemoryArchitect",
                thesis="Initial neural memory architecture status",
            )
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # MAIN PROCESS
    # ─────────────────────────────────────────────────────────────
    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """Process neural memory operations"""
        start_time = time.time()
        try:
            memory_data = await self._extract_memory_data(**inputs)
            if not memory_data:
                return await self._handle_no_data_fallback()

            # Process storage (if any)
            storage_result = await self._process_experience_storage(memory_data)

            # Retrieval if query provided
            if memory_data.get("query") is not None:
                retrieval_result = await self._perform_memory_retrieval(memory_data)
                storage_result.update(retrieval_result)

            # Metrics
            metrics_result = await self._update_neural_metrics()
            storage_result.update(metrics_result)

            # Thesis
            thesis = await self._generate_neural_thesis(memory_data, storage_result)

            # Bus updates
            await self._update_neural_smart_bus(storage_result, thesis)

            # Success bookkeeping
            processing_time = (time.time() - start_time) * 1000.0
            self._record_success(processing_time)

            # Contract-compliant payload
            neural_status = {
                "buffer_size": len(self.buffer),
                "memory_utilization": len(self.buffer) / max(1, int(self.genome["max_len"])),
                "average_importance": self._neural_performance["average_importance"],
                "neural_performance_score": self._neural_performance_score,
                "last_updated": time.time(),
            }

            if storage_result.get("retrieval_performed", False):
                retrieval_data = storage_result.get("retrieved_memories", {})
                attention_data = {
                    "retrieved_count": len(retrieval_data.get("memories", [])),
                    "similarity_scores": retrieval_data.get("similarity_scores", []),
                    "top_k": int(self.genome["retrieval_top_k"]),
                    "attention_heads": int(self.genome["num_heads"]),
                }
            else:
                attention_data = {
                    "retrieved_count": 0,
                    "similarity_scores": [],
                    "top_k": int(self.genome["retrieval_top_k"]),
                    "attention_heads": int(self.genome["num_heads"]),
                }

            embedding_info = {
                "embedding_dim": int(self.genome["embed_dim"]),
                "total_embeddings": len(self.buffer),
                "importance_threshold": float(self.genome["importance_threshold"]),
                "decay_rate": float(self.genome["memory_decay"]),
            }

            if len(self.importance_scores) > 0:
                importance_stats = {
                    "average_importance": float(torch.mean(self.importance_scores)),
                    "max_importance": float(torch.max(self.importance_scores)),
                    "min_importance": float(torch.min(self.importance_scores)),
                    "std_importance": float(torch.std(self.importance_scores)),
                    "total_scored": int(len(self.importance_scores)),
                }
            else:
                importance_stats = {
                    "average_importance": 0.0,
                    "max_importance": 0.0,
                    "min_importance": 0.0,
                    "std_importance": 0.0,
                    "total_scored": 0,
                }

            storage_result.update({
                "neural_memory": neural_status,
                "attention_retrieval": attention_data,
                "memory_embedding": embedding_info,
                "importance_scoring": importance_stats,
                "_thesis": thesis,
                "thesis": thesis,
            })
            return storage_result

        except Exception as e:
            return await self._handle_neural_error(e, start_time)

    # ─────────────────────────────────────────────────────────────
    # DATA EXTRACTION / FEATURE ENCODING
    # ─────────────────────────────────────────────────────────────
    async def _extract_memory_data(self, **inputs: Any) -> Optional[Dict[str, Any]]:
        """Extract memory data from SmartInfoBus"""
        try:
            observations = self.smart_bus.get("observations", "NeuralMemoryArchitect")
            rewards = self.smart_bus.get("rewards", "NeuralMemoryArchitect")
            actions = self.smart_bus.get("actions", "NeuralMemoryArchitect")
            market_context = self.smart_bus.get("market_context", "NeuralMemoryArchitect") or {}

            query = inputs.get("query")
            experience = inputs.get("experience")

            return {
                "observations": observations,
                "rewards": rewards,
                "actions": actions,
                "market_context": market_context,
                "query": query,
                "experience": experience,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Failed to extract memory data: {e}")
            return None

    async def _process_experience_storage(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process experience storage in neural memory"""
        try:
            experience = memory_data.get("experience")
            if not experience:
                return {"storage_performed": False, "reason": "no_experience"}

            # Extract features from experience
            features = self._extract_experience_features(experience, memory_data)
            if features is None:
                return {"storage_performed": False, "reason": "feature_extraction_failed"}

            # Encode experience
            encoded_experience = await self._encode_experience(features)

            # Calculate importance
            importance = await self._calculate_importance(encoded_experience, memory_data)

            # Store if important enough
            stored = False
            if importance > float(self.genome["importance_threshold"]):
                await self._store_in_buffer(encoded_experience, importance, experience)
                stored = True

            # Update performance metrics
            if stored:
                self._neural_performance["memories_stored"] = int(self._neural_performance["memories_stored"]) + 1
                ms = self._neural_performance["memories_stored"]
                prev_avg = float(self._neural_performance["average_importance"])
                self._neural_performance["average_importance"] = float((prev_avg * (ms - 1) + importance) / ms)

            return {
                "storage_performed": stored,
                "importance_score": float(importance),
                "buffer_size": len(self.buffer),
                "memory_utilization": len(self.buffer) / max(1, int(self.genome["max_len"])),
            }

        except Exception as e:
            self.logger.error(f"Experience storage failed: {e}")
            return {"storage_performed": False, "error": str(e)}

    def _extract_experience_features(self, experience: Any, memory_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from experience"""
        try:
            features: List[float] = []

            # Handle different experience types
            if isinstance(experience, dict):
                if "observation" in experience:
                    obs = experience["observation"]
                    if isinstance(obs, np.ndarray):
                        features.extend(np.asarray(obs).flatten()[:20].astype(float))
                    elif isinstance(obs, (int, float, np.number)):
                        features.append(float(obs))
                    else:
                        features.append(0.0)

                if "reward" in experience:
                    try:
                        features.append(float(experience["reward"]))
                    except Exception:
                        features.append(0.0)

                if "action" in experience:
                    action = experience["action"]
                    if isinstance(action, np.ndarray):
                        features.extend(np.asarray(action).flatten()[:5].astype(float))
                    elif isinstance(action, (int, float, np.number)):
                        features.append(float(action))
                    else:
                        features.append(0.0)

            elif isinstance(experience, np.ndarray):
                features.extend(np.asarray(experience).flatten()[:20].astype(float))
            elif isinstance(experience, (int, float, np.number)):
                features.append(float(experience))
            else:
                features.append(0.0)

            # Add market context features
            market_context = memory_data.get("market_context", {}) or {}
            context_features = self._extract_context_features(market_context)
            features.extend(context_features)

            # Ensure consistent feature length
            target_length = int(self.genome["embed_dim"])
            if len(features) < target_length:
                features.extend([0.0] * (target_length - len(features)))
            elif len(features) > target_length:
                features = features[:target_length]

            return np.array(features, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def _extract_context_features(self, market_context: Dict[str, Any]) -> List[float]:
        """Extract context features from market context"""
        features: List[float] = []

        # Volatility
        vol = market_context.get("volatility", 0.0)
        if isinstance(vol, dict):
            try:
                # Take the first numeric value if present
                val = next((float(v) for v in vol.values() if isinstance(v, (int, float, np.number))), 0.0)
                features.append(val)
            except Exception:
                features.append(0.0)
        elif isinstance(vol, (int, float, np.number)):
            features.append(float(vol))
        else:
            features.append(0.0)

        # Session
        session_val = market_context.get("session", None)
        session_map = {"asian": 0.0, "european": 0.5, "us": 1.0}
        if isinstance(session_val, str):
            features.append(float(session_map.get(session_val.lower(), 0.25)))
        else:
            features.append(0.25)

        # Regime
        regime_val = market_context.get("regime", None)
        regime_map = {"trending": 1.0, "ranging": 0.0, "volatile": 0.5}
        if isinstance(regime_val, str):
            features.append(float(regime_map.get(regime_val.lower(), 0.25)))
        else:
            features.append(0.25)

        # Pad to context_features length
        while len(features) < int(self.neural_config.context_features):
            features.append(0.0)

        return features[: int(self.neural_config.context_features)]

    async def _encode_experience(self, features: np.ndarray) -> torch.Tensor:
        """Encode experience using neural encoder"""
        try:
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                encoded = self.encoder(features_tensor)
            return encoded.squeeze(0)
        except Exception as e:
            self.logger.error(f"Experience encoding failed: {e}")
            return torch.zeros(int(self.genome["embed_dim"]), dtype=torch.float32)

    def _coerce_numeric_rewards(self, rewards: Any) -> List[float]:
        """Convert various reward structures into a flat numeric list."""
        nums: List[float] = []
        if rewards is None:
            return nums

        def maybe_add(x: Any) -> None:
            if isinstance(x, (int, float, np.number)):
                nums.append(float(x))

        try:
            if isinstance(rewards, dict):
                for v in rewards.values():
                    if isinstance(v, dict):
                        for k2 in ("pnl", "reward", "value", "score"):
                            if k2 in v:
                                maybe_add(v[k2])
                                break
                    else:
                        maybe_add(v)
            elif isinstance(rewards, (list, tuple, deque)):
                for r in rewards:
                    if isinstance(r, dict):
                        for k in ("pnl", "reward", "value", "score"):
                            if k in r:
                                maybe_add(r[k])
                                break
                    else:
                        maybe_add(r)
            else:
                maybe_add(rewards)
        except Exception:
            # Be forgiving
            pass
        return nums

    async def _calculate_importance(self, encoded_experience: torch.Tensor, memory_data: Dict[str, Any]) -> float:
        """Calculate importance score for experience"""
        try:
            with torch.no_grad():
                importance = self.value_head(encoded_experience.unsqueeze(0))
            importance_score = float(importance.squeeze())

            # Adjust based on recent numeric rewards
            rewards_raw = memory_data.get("rewards")
            numeric_rewards = self._coerce_numeric_rewards(rewards_raw)[-5:]  # last 5
            if numeric_rewards:
                avg_reward = float(np.mean(numeric_rewards))
                if avg_reward > 0:
                    importance_score *= 1.2
                elif avg_reward < 0:
                    importance_score *= 0.8

            return float(max(0.0, min(1.0, importance_score)))

        except Exception as e:
            self.logger.error(f"Importance calculation failed: {e}")
            return 0.0

    async def _store_in_buffer(self, encoded_experience: torch.Tensor, importance: float, experience: Any) -> None:
        """Store experience in neural buffer"""
        try:
            self.buffer = torch.cat([self.buffer, encoded_experience.unsqueeze(0)], dim=0)
            self.importance_scores = torch.cat(
                [self.importance_scores, torch.tensor([importance], dtype=torch.float32)]
            )

            metadata = {
                "timestamp": time.time(),
                "importance": float(importance),
                "experience_type": type(experience).__name__,
            }
            self.memory_metadata.append(metadata)

            if len(self.buffer) > int(self.genome["max_len"]):
                await self._prune_buffer()

        except Exception as e:
            self.logger.error(f"Buffer storage failed: {e}")

    async def _prune_buffer(self) -> None:
        """Prune buffer to maintain size limits"""
        try:
            n_keep = int(int(self.genome["max_len"]) * 0.8)
            if n_keep <= 0:
                # Edge guard
                self.buffer = self.buffer[-1:]
                self.importance_scores = self.importance_scores[-1:]
                self.memory_metadata = self.memory_metadata[-1:]
                return

            sorted_indices = torch.argsort(self.importance_scores, descending=True)
            top_indices = sorted_indices[: max(1, n_keep // 2)]
            recent_indices = torch.arange(max(0, len(self.buffer) - max(1, n_keep // 2)), len(self.buffer))

            keep_indices = torch.unique(torch.cat([top_indices, recent_indices]))
            self.buffer = self.buffer[keep_indices]
            self.importance_scores = self.importance_scores[keep_indices]

            keep_set = set(int(i) for i in keep_indices.tolist())
            self.memory_metadata = [m for idx, m in enumerate(self.memory_metadata) if idx in keep_set]

        except Exception as e:
            self.logger.error(f"Buffer pruning failed: {e}")

    async def _perform_memory_retrieval(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform memory retrieval using attention mechanism"""
        try:
            query = memory_data.get("query")
            if query is None:
                return {"retrieval_performed": False, "reason": "no_query"}

            if isinstance(query, np.ndarray):
                query_tensor = torch.tensor(query, dtype=torch.float32)
            elif isinstance(query, (list, tuple)):
                query_tensor = torch.tensor(list(query), dtype=torch.float32)
            elif isinstance(query, (int, float, np.number)):
                query_tensor = torch.tensor([float(query)], dtype=torch.float32)
            else:
                # Unsupported type -> zero vector
                query_tensor = torch.zeros(int(self.genome["embed_dim"]), dtype=torch.float32)

            if query_tensor.dim() == 1:
                if len(query_tensor) != int(self.genome["embed_dim"]):
                    if len(query_tensor) < int(self.genome["embed_dim"]):
                        padding = torch.zeros(int(self.genome["embed_dim"]) - len(query_tensor))
                        query_tensor = torch.cat([query_tensor, padding])
                    else:
                        query_tensor = query_tensor[: int(self.genome["embed_dim"])]
                query_tensor = query_tensor.unsqueeze(0)  # [1, embed_dim]

            if len(self.buffer) == 0:
                return {"retrieval_performed": False, "reason": "empty_buffer"}

            retrieved_memories = await self._attention_retrieval(query_tensor)

            self._neural_performance["memories_retrieved"] = int(self._neural_performance["memories_retrieved"]) + 1

            return {
                "retrieval_performed": True,
                "retrieved_memories": retrieved_memories,
                "query_similarity": retrieved_memories.get("similarity_scores", []),
                "top_k": int(self.genome["retrieval_top_k"]),
            }

        except Exception as e:
            self.logger.error(f"Memory retrieval failed: {e}")
            return {"retrieval_performed": False, "error": str(e)}

    async def _attention_retrieval(self, query: torch.Tensor) -> Dict[str, Any]:
        """Perform attention-based memory retrieval"""
        try:
            with torch.no_grad():
                # query: [1, embed_dim] -> add batch/time dims for MHA (batch_first=True)
                attn_output, attn_weights = self.attn(
                    query.unsqueeze(0),            # [1, 1, ed]
                    self.buffer.unsqueeze(0),      # [1, n, ed]
                    self.buffer.unsqueeze(0),      # [1, n, ed]
                )

            weights = attn_weights.squeeze().cpu().numpy()
            if weights.ndim > 1:
                weights = weights.reshape(-1)
            elif weights.ndim == 0:
                weights = np.array([float(weights)])

            k = int(min(int(self.genome["retrieval_top_k"]), max(1, len(weights))))
            top_indices = np.argsort(weights)[-k:][::-1]

            retrieved_memories: List[Dict[str, Any]] = []
            similarity_scores: List[float] = []

            for idx in top_indices:
                idx_int = int(idx)
                memory = {
                    "embedding": self.buffer[idx_int].cpu().numpy().tolist(),
                    "importance": float(self.importance_scores[idx_int]),
                    "metadata": self.memory_metadata[idx_int] if idx_int < len(self.memory_metadata) else {},
                }
                retrieved_memories.append(memory)
                similarity_scores.append(float(weights[idx_int]))

            return {
                "memories": retrieved_memories,
                "similarity_scores": similarity_scores,
                "attention_weights": weights.tolist(),
            }

        except Exception as e:
            self.logger.error(f"Attention retrieval failed: {e}")
            return {"memories": [], "similarity_scores": [], "attention_weights": []}

    async def _update_neural_metrics(self) -> Dict[str, Any]:
        """Update neural performance metrics"""
        try:
            if len(self.buffer) > 0:
                avg_importance = float(torch.mean(self.importance_scores))
                storage_efficiency = avg_importance / max(0.1, float(self.genome["importance_threshold"]))
            else:
                storage_efficiency = 0.0

            memory_utilization = len(self.buffer) / max(1, int(self.genome["max_len"]))
            self._learning_curves["memory_utilization"].append(memory_utilization)

            neural_score = (
                storage_efficiency * 0.5
                + memory_utilization * 0.3
                + float(self._neural_performance["average_importance"]) * 0.2
            ) * 100.0
            self._neural_performance_score = float(neural_score)

            return {
                "neural_metrics": {
                    "storage_efficiency": float(storage_efficiency),
                    "memory_utilization": float(memory_utilization),
                    "neural_performance_score": float(neural_score),
                    "buffer_size": len(self.buffer),
                    "average_importance": float(self._neural_performance["average_importance"]),
                }
            }

        except Exception as e:
            self.logger.error(f"Neural metrics update failed: {e}")
            return {"neural_metrics": {"error": str(e)}}

    async def _generate_neural_thesis(self, memory_data: Dict[str, Any], storage_result: Dict[str, Any]) -> str:
        """Generate comprehensive neural memory thesis"""
        try:
            buffer_size = len(self.buffer)
            memory_utilization = buffer_size / max(1, int(self.genome["max_len"]))
            avg_importance = float(self._neural_performance["average_importance"])
            storage_performed = bool(storage_result.get("storage_performed", False))
            retrieval_performed = bool(storage_result.get("retrieval_performed", False))

            parts: List[str] = [
                f"Neural Memory: {buffer_size} experiences, utilization {memory_utilization:.1%}",
                f"Avg importance {avg_importance:.3f} (threshold {float(self.genome['importance_threshold']):.3f})",
                f"Performance score {self._neural_performance_score:.1f}/100",
            ]

            if storage_performed:
                imp = float(storage_result.get("importance_score", 0.0))
                parts.append(f"Stored new experience (importance {imp:.3f})")

            if retrieval_performed:
                retrieved_count = len(storage_result.get("retrieved_memories", {}).get("memories", []))
                parts.append(f"Retrieved {retrieved_count} memories via attention")

            parts.append(f"Attention: {int(self.genome['num_heads'])} heads, {int(self.genome['embed_dim'])}-dim embeddings")

            if buffer_size > 0:
                high_importance = int(torch.sum(self.importance_scores > 0.7).item())
                parts.append(f"Quality: {high_importance}/{buffer_size} high-importance")

            if len(self._learning_curves["memory_utilization"]) > 10:
                recent_util = float(np.mean(list(self._learning_curves["memory_utilization"])[-5:]))
                parts.append(f"Recent utilization trend {recent_util:.1%}")

            return " | ".join(parts)

        except Exception as e:
            return f"Neural thesis generation failed: {str(e)} - core functionality intact"

    async def _update_neural_smart_bus(self, storage_result: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with neural memory results"""
        try:
            neural_status = {
                "buffer_size": len(self.buffer),
                "memory_utilization": len(self.buffer) / max(1, int(self.genome["max_len"])),
                "average_importance": float(self._neural_performance["average_importance"]),
                "neural_performance_score": float(self._neural_performance_score),
                "last_updated": time.time(),
            }
            self.smart_bus.set("neural_memory", neural_status, module="NeuralMemoryArchitect", thesis=thesis)

            if storage_result.get("retrieval_performed", False):
                retrieval_data = storage_result.get("retrieved_memories", {})
                attention_data = {
                    "retrieved_count": len(retrieval_data.get("memories", [])),
                    "similarity_scores": retrieval_data.get("similarity_scores", []),
                    "top_k": int(self.genome["retrieval_top_k"]),
                    "attention_heads": int(self.genome["num_heads"]),
                }
                self.smart_bus.set(
                    "attention_retrieval", attention_data, module="NeuralMemoryArchitect", thesis="Attention-based retrieval"
                )

            embedding_info = {
                "embedding_dim": int(self.genome["embed_dim"]),
                "total_embeddings": len(self.buffer),
                "importance_threshold": float(self.genome["importance_threshold"]),
                "decay_rate": float(self.genome["memory_decay"]),
            }
            self.smart_bus.set("memory_embedding", embedding_info, module="NeuralMemoryArchitect", thesis="Embedding stats")

            if len(self.importance_scores) > 0:
                importance_stats = {
                    "average_importance": float(torch.mean(self.importance_scores)),
                    "max_importance": float(torch.max(self.importance_scores)),
                    "min_importance": float(torch.min(self.importance_scores)),
                    "std_importance": float(torch.std(self.importance_scores)),
                    "total_scored": int(len(self.importance_scores)),
                }
                self.smart_bus.set(
                    "importance_scoring", importance_stats, module="NeuralMemoryArchitect", thesis="Importance stats"
                )

        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    # ─────────────────────────────────────────────────────────────
    # FALLBACKS / ERRORS
    # ─────────────────────────────────────────────────────────────
    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no memory data is available"""
        self.logger.warning("No memory data available - returning current neural status")

        thesis = "NeuralMemoryArchitect fallback: no memory data available"
        neural_status = {
            "buffer_size": len(self.buffer),
            "memory_utilization": len(self.buffer) / max(1, int(self.genome["max_len"])),
            "average_importance": float(self._neural_performance["average_importance"]),
            "neural_performance_score": float(self._neural_performance_score),
            "last_updated": time.time(),
        }
        attention_data = {
            "retrieved_count": 0,
            "similarity_scores": [],
            "top_k": int(self.genome["retrieval_top_k"]),
            "attention_heads": int(self.genome["num_heads"]),
        }
        embedding_info = {
            "embedding_dim": int(self.genome["embed_dim"]),
            "total_embeddings": len(self.buffer),
            "importance_threshold": float(self.genome["importance_threshold"]),
            "decay_rate": float(self.genome["memory_decay"]),
        }
        if len(self.importance_scores) > 0:
            importance_stats = {
                "average_importance": float(torch.mean(self.importance_scores)),
                "max_importance": float(torch.max(self.importance_scores)),
                "min_importance": float(torch.min(self.importance_scores)),
                "std_importance": float(torch.std(self.importance_scores)),
                "total_scored": int(len(self.importance_scores)),
            }
        else:
            importance_stats = {
                "average_importance": 0.0,
                "max_importance": 0.0,
                "min_importance": 0.0,
                "std_importance": 0.0,
                "total_scored": 0,
            }

        return {
            "neural_memory": neural_status,
            "attention_retrieval": attention_data,
            "memory_embedding": embedding_info,
            "importance_scoring": importance_stats,
            "_thesis": thesis,
            "thesis": thesis,
            "fallback_reason": "no_memory_data",
        }

    async def _handle_neural_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle neural memory errors"""
        processing_time = (time.time() - start_time) * 1000.0

        # Circuit breaker
        self.circuit_breaker["failures"] = int(self.circuit_breaker.get("failures", 0)) + 1
        self.circuit_breaker["last_failure"] = time.time()
        if self.circuit_breaker["failures"] >= int(self.circuit_breaker["threshold"]):
            self.circuit_breaker["state"] = "OPEN"

        # Log error with context
        try:
            self.error_pinpointer.analyze_error(error, "NeuralMemoryArchitect")
        except Exception:
            pass

        explanation = self.english_explainer.explain_error(
            "NeuralMemoryArchitect", str(error), "neural memory processing"
        )

        self.logger.error(
            format_operator_message(
                "[CRASH]",
                "NEURAL_MEMORY_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="neural_memory",
            )
        )

        self._record_failure(error)
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        thesis = f"NeuralMemoryArchitect fallback: {reason}"
        neural_status = {
            "buffer_size": len(self.buffer),
            "memory_utilization": len(self.buffer) / max(1, int(self.genome["max_len"])),
            "average_importance": float(self._neural_performance["average_importance"]),
            "neural_performance_score": float(self._neural_performance_score),
            "last_updated": time.time(),
        }
        attention_data = {
            "retrieved_count": 0,
            "similarity_scores": [],
            "top_k": int(self.genome["retrieval_top_k"]),
            "attention_heads": int(self.genome["num_heads"]),
        }
        embedding_info = {
            "embedding_dim": int(self.genome["embed_dim"]),
            "total_embeddings": len(self.buffer),
            "importance_threshold": float(self.genome["importance_threshold"]),
            "decay_rate": float(self.genome["memory_decay"]),
        }
        if len(self.importance_scores) > 0:
            importance_stats = {
                "average_importance": float(torch.mean(self.importance_scores)),
                "max_importance": float(torch.max(self.importance_scores)),
                "min_importance": float(torch.min(self.importance_scores)),
                "std_importance": float(torch.std(self.importance_scores)),
                "total_scored": int(len(self.importance_scores)),
            }
        else:
            importance_stats = {
                "average_importance": 0.0,
                "max_importance": 0.0,
                "min_importance": 0.0,
                "std_importance": 0.0,
                "total_scored": 0,
            }

        return {
            "neural_memory": neural_status,
            "attention_retrieval": attention_data,
            "memory_embedding": embedding_info,
            "importance_scoring": importance_stats,
            "_thesis": thesis,
            "thesis": thesis,
            "circuit_breaker_state": self.circuit_breaker["state"],
            "fallback_reason": reason,
        }

    # ─────────────────────────────────────────────────────────────
    # HEALTH / METRICS
    # ─────────────────────────────────────────────────────────────
    def _update_neural_health(self) -> None:
        """Update neural memory health metrics"""
        try:
            if not hasattr(self, "_neural_performance_score") or not hasattr(self, "buffer"):
                return

            score = float(self._neural_performance_score)
            if score < 20:
                self._health_status = "critical"
            elif score < 50:
                self._health_status = "warning"
            else:
                self._health_status = "healthy"

            utilization = len(self.buffer) / max(1, int(self.genome["max_len"]))
            if utilization > 0.95:
                self._health_status = "warning"

            self._last_health_check = time.time()

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = "warning"

    def _analyze_memory_efficiency(self) -> None:
        """Analyze memory efficiency"""
        try:
            if not hasattr(self, "buffer") or not hasattr(self, "importance_scores"):
                return

            if len(self.buffer) > 10:
                high_importance = int(torch.sum(self.importance_scores > 0.7).item())
                total_memories = int(len(self.buffer))
                efficiency = float(high_importance / max(1, total_memories))
                self._neural_performance["attention_efficiency"] = efficiency

                if efficiency > 0.6:
                    self.logger.info(
                        format_operator_message(
                            "🧠",
                            "HIGH_MEMORY_EFFICIENCY",
                            efficiency=f"{efficiency:.2f}",
                            high_importance_count=high_importance,
                            total_memories=total_memories,
                            context="efficiency_analysis",
                        )
                    )
        except Exception as e:
            self.logger.error(f"Memory efficiency analysis failed: {e}")

    def _record_success(self, processing_time_ms: float) -> None:
        """Record successful processing"""
        try:
            self.performance_tracker.record_metric(
                "NeuralMemoryArchitect", "neural_cycle", float(processing_time_ms), True
            )
        except Exception:
            pass
        # Reset circuit breaker on success
        if self.circuit_breaker.get("state") == "OPEN":
            self.circuit_breaker["failures"] = 0
            self.circuit_breaker["state"] = "CLOSED"

    def _record_failure(self, error: Exception) -> None:
        """Record processing failure"""
        try:
            self.performance_tracker.record_metric("NeuralMemoryArchitect", "neural_cycle", 0.0, False)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # STATE IO
    # ─────────────────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            "buffer": self.buffer.cpu().numpy().tolist(),
            "importance_scores": self.importance_scores.cpu().numpy().tolist(),
            "memory_metadata": list(self.memory_metadata),
            "genome": dict(self.genome),
            "neural_performance": dict(self._neural_performance),
            "neural_performance_score": float(self._neural_performance_score),
            "circuit_breaker": dict(self.circuit_breaker),
            "health_status": self._health_status,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state from persistence"""
        if "buffer" in state:
            self.buffer = torch.tensor(state["buffer"], dtype=torch.float32)
        if "importance_scores" in state:
            self.importance_scores = torch.tensor(state["importance_scores"], dtype=torch.float32)
        if "memory_metadata" in state:
            self.memory_metadata = list(state["memory_metadata"])
        if "genome" in state:
            self.genome.update(state["genome"])
        if "neural_performance" in state:
            self._neural_performance.update(state["neural_performance"])
        if "neural_performance_score" in state:
            self._neural_performance_score = float(state["neural_performance_score"])
        if "circuit_breaker" in state:
            self.circuit_breaker.update(state["circuit_breaker"])
        if "health_status" in state:
            self._health_status = str(state["health_status"])

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            "status": self._health_status,
            "last_check": self._last_health_check,
            "circuit_breaker": self.circuit_breaker.get("state", "CLOSED"),
            "buffer_size": len(self.buffer),
            "memory_utilization": len(self.buffer) / max(1, int(self.genome["max_len"])),
            "neural_performance_score": float(self._neural_performance_score),
        }

    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring_active = False

    # ─────────────────────────────────────────────────────────────
    # LEGACY / ACTION API
    # ─────────────────────────────────────────────────────────────
    def retrieve(self, query: np.ndarray, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Legacy compatibility for memory retrieval"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            memory_data = {"query": query}
            result = loop.run_until_complete(self._perform_memory_retrieval(memory_data))
            return result.get("retrieved_memories", {})
        finally:
            loop.close()

    async def propose_action(self, **inputs: Any) -> Dict[str, Any]:
        """Propose action based on neural memory (contract: include _thesis)"""
        if len(self.buffer) > 0:
            max_idx = int(torch.argmax(self.importance_scores).item())
            memory_embedding = self.buffer[max_idx].cpu().numpy()

            action_values = [
                float(memory_embedding[0]),
                float(memory_embedding[1] if len(memory_embedding) > 1 else 0.0),
            ]
            confidence = float(self._neural_performance_score / 100.0)
            thesis = f"Neural memory action from most important memory (performance: {self._neural_performance_score:.1f})"

            return {
                "action": action_values,
                "confidence": confidence,
                "thesis": thesis,
                "_thesis": thesis,
                "memory_index": int(max_idx),
                "importance_score": float(self.importance_scores[max_idx].item()),
            }

        thesis = "Neural memory action proposal - no memories available"
        return {"action": [0.0, 0.0], "confidence": 0.5, "thesis": thesis, "_thesis": thesis}

    async def calculate_confidence(self, action: Dict[str, Any], **inputs: Any) -> float:
        """Calculate confidence in neural memory decisions"""
        if not isinstance(action, dict):
            return 0.5

        base_confidence = float(self._neural_performance_score / 100.0)
        utilization = len(self.buffer) / max(1, int(self.genome["max_len"]))
        utilization_factor = min(1.0, utilization + 0.2)

        if len(self.importance_scores) > 0:
            avg_importance = float(torch.mean(self.importance_scores).item())
            importance_factor = min(1.0, avg_importance + 0.3)
        else:
            importance_factor = 0.3

        cb_factor = 0.5 if self.circuit_breaker.get("state") == "OPEN" else 1.0
        health_factor = 1.0 if self._health_status == "healthy" else (0.8 if self._health_status == "warning" else 0.6)

        confidence = (
            base_confidence * 0.4
            + utilization_factor * 0.2
            + importance_factor * 0.2
            + cb_factor * 0.1
            + health_factor * 0.1
        )
        return float(max(0.0, min(1.0, confidence)))

    def confidence(self, obs: Any = None, **kwargs: Any) -> float:
        """Legacy compatibility for confidence"""
        return float(self._neural_performance_score / 100.0)
