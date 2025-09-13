# modules/memory/unified_memory.py
"""
Unified Memory System - Production Ready Implementation
Orchestrates all memory subsystems with optimal performance and debugging
"""

from __future__ import annotations

import asyncio
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
from typing import Awaitable, Sequence, cast, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

from modules.core.module_base import BaseModule, module
from modules.contracts import module_args
from modules.core.mixins import (
    SmartInfoBusTradingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
)
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker
import torch

# Import components
from .components.replay import ReplayComponent
from .components.compression import CompressionComponent
from .components.mistakes import MistakeComponent
from .components.neural import NeuralComponent
from .components.playbook import PlaybookComponent
from .components.budget import BudgetComponent

# Import shared resources
from .shared.memory_store import UnifiedMemoryStore
from .shared.feature_extractor import UnifiedFeatureExtractor
from .shared.pattern_detector import UnifiedPatternDetector
from .shared.utils import MemoryUtils, LRUCache

# Import debug logger
from .debug.memory_logger import MemoryDebugLogger


@dataclass
class UnifiedMemoryConfig:
    """Unified configuration for all memory subsystems"""
    # Debug master switches
    debug: bool = True
    debug_level: str = "DEBUG"  # "TRACE"|"DEBUG"|"INFO"|"WARNING"|"ERROR"

    # Combined debug file (legacy default)
    debug_log_path: str = "logs/memory/unified_debug.log"
    enable_combined_log: bool = True

    # Per-level file toggles + paths (None = derive from debug_log_path)
    enable_trace_log: bool = True
    trace_file_path: Optional[str] = None

    enable_debug_log: bool = True
    debug_file_path: Optional[str] = None

    enable_info_log: bool = True
    info_file_path: Optional[str] = None

    enable_warning_log: bool = True
    warning_file_path: Optional[str] = None

    enable_error_log: bool = True
    error_file_path: Optional[str] = None

    # Core settings
    max_memory_size: int = 10000
    batch_size: int = 32
    parallel_processing: bool = True
    cache_size: int = 1000

    # Component enable flags
    enable_replay: bool = True
    enable_compression: bool = True
    enable_mistakes: bool = True
    enable_neural: bool = True
    enable_playbook: bool = True
    enable_budget: bool = True

    # Performance settings
    max_processing_time_ms: float = 500
    circuit_breaker_threshold: int = 3
    health_check_interval: int = 30

    # Replay settings
    replay_interval: int = 10
    replay_decay: float = 0.9
    sequence_len: int = 5
    replay_profit_threshold: float = 10.0

    # Compression settings
    n_components: int = 8
    compression_ratio: float = 0.7
    compress_interval: int = 10

    # Mistakes settings
    n_clusters: int = 5
    danger_threshold: float = 0.7
    avoidance_sensitivity: float = 1.0
    mistake_profit_threshold: float = 10.0

    # Neural settings
    embed_dim: int = 32
    num_heads: int = 4
    memory_decay: float = 0.95
    importance_threshold: float = 0.3

    # Playbook settings
    k_neighbors: int = 5
    similarity_threshold: float = 0.7
    pattern_memory_size: int = 50

    # Budget settings
    rebalance_interval: int = 50
    utilization_target: float = 0.8
    efficiency_weight: float = 0.7
    recency_weight: float = 0.3


@module(**module_args(
    "UnifiedMemory",
    description="Unified memory system orchestrating all memory components",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class UnifiedMemory(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Unified Memory System - Production Ready

    Orchestrates all memory subsystems with shared resources, parallel processing,
    and comprehensive debugging capabilities.
    """

    def __init__(
        self,
        config: Optional[UnifiedMemoryConfig] = None,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize unified memory system"""
        # Configuration
        self.unified_config = config or UnifiedMemoryConfig()

        # Bootstrap attributes that may be referenced during early initialization hooks
        # (e.g., BaseModule._initialize) to avoid AttributeError races.
        self.components: Dict[str, Any] = {}
        # Ensure SmartInfoBus is available for _initialize() status publish
        try:
            self.smart_bus = InfoBusManager.get_instance()
        except Exception:
            # Defer wiring to _initialize_core_systems if manager unavailable
            self.smart_bus = None  # type: ignore[assignment]

        # Initialize BaseModule (will invoke self._initialize())
        super().__init__(**kwargs)

        # Initialize core systems
        self._initialize_core_systems()

        # Initialize shared resources
        self._initialize_shared_resources()

        # Initialize components
        self._initialize_components()

        # Initialize debug logger
        self._initialize_debug_logger()

        # Initialize state BEFORE starting monitoring to avoid race conditions
        self._initialize_state()

        # Initialize monitoring (starts background thread)
        self._initialize_monitoring()

        self.logger.info(
            format_operator_message(
                "🧠",
                "UNIFIED_MEMORY_INITIALIZED",
                details=f"Debug: {self.unified_config.debug}, Components: {self._count_enabled_components()}",
                result="Unified memory system ready",
                context="memory_system",
            )
        )

    def _initialize_core_systems(self) -> None:
        """Initialize core systems"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="UnifiedMemory",
            log_path="logs/memory/unified_memory.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("UnifiedMemory", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker
        self.circuit_breaker = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",
            "threshold": self.unified_config.circuit_breaker_threshold,
        }

    def _initialize_shared_resources(self) -> None:
        """Initialize shared resources used by all components"""
        # Central memory store
        self.memory_store = UnifiedMemoryStore(
            max_size=self.unified_config.max_memory_size,
            batch_size=self.unified_config.batch_size,
        )

        # Shared feature extractor
        self.feature_extractor = UnifiedFeatureExtractor()

        # Shared pattern detector
        self.pattern_detector = UnifiedPatternDetector()

        # Shared ML models
        self.shared_scaler = StandardScaler()
        self.shared_encoder = self._create_shared_encoder()

        # Warmup encoder to avoid first-call latency spikes
        try:
            with torch.no_grad():
                _dummy = torch.zeros(1, self.unified_config.embed_dim, dtype=torch.float32)
                # Forward once to initialize weights/graph
                _ = self.shared_encoder(_dummy)
        except Exception:
            # Non-fatal; continue without warmup
            pass

        # Shared cache
        self.cache = LRUCache(maxsize=self.unified_config.cache_size)

        # Shared utilities
        self.utils = MemoryUtils()

    def _create_shared_encoder(self) -> "torch.nn.Module":
        """Create shared neural encoder"""
        import torch.nn as nn

        class SharedEncoder(nn.Module):
            def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 32):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, output_dim),
                    nn.LayerNorm(output_dim),
                )

            def forward(self, x):
                return self.encoder(x)

        return SharedEncoder(
            input_dim=self.unified_config.embed_dim,
            hidden_dim=self.unified_config.embed_dim * 2,
            output_dim=self.unified_config.embed_dim,
        )

    def _initialize_components(self) -> None:
        """Initialize all memory components"""
        shared_resources = {
            "store": self.memory_store,
            "extractor": self.feature_extractor,
            "pattern_detector": self.pattern_detector,
            "scaler": self.shared_scaler,
            "encoder": self.shared_encoder,
            "cache": self.cache,
            "utils": self.utils,
            "logger": self.logger,
        }

        self.components: Dict[str, Any] = {}

        if self.unified_config.enable_replay:
            self.components["replay"] = ReplayComponent(self.unified_config, shared_resources)

        if self.unified_config.enable_compression:
            self.components["compression"] = CompressionComponent(self.unified_config, shared_resources)

        if self.unified_config.enable_mistakes:
            self.components["mistakes"] = MistakeComponent(self.unified_config, shared_resources)

        if self.unified_config.enable_neural:
            self.components["neural"] = NeuralComponent(self.unified_config, shared_resources)

        if self.unified_config.enable_playbook:
            self.components["playbook"] = PlaybookComponent(self.unified_config, shared_resources)

        if self.unified_config.enable_budget:
            self.components["budget"] = BudgetComponent(self.unified_config, shared_resources)

    def _initialize_debug_logger(self) -> None:
        """Initialize unified debug logging (with per-level files)"""
        # Master off switch
        if not self.unified_config.debug:
            self.debug_logger = MemoryDebugLogger(enabled=False)
            return

        self.debug_logger = MemoryDebugLogger(
            enabled=True,
            level=self.unified_config.debug_level,
            # combined
            log_path=self.unified_config.debug_log_path,
            enable_combined_file=self.unified_config.enable_combined_log,
            # per-level
            enable_trace_file=self.unified_config.enable_trace_log,
            trace_file=self.unified_config.trace_file_path,
            enable_debug_file=self.unified_config.enable_debug_log,
            debug_file=self.unified_config.debug_file_path,
            enable_info_file=self.unified_config.enable_info_log,
            info_file=self.unified_config.info_file_path,
            enable_warning_file=self.unified_config.enable_warning_log,
            warning_file=self.unified_config.warning_file_path,
            enable_error_file=self.unified_config.enable_error_log,
            error_file=self.unified_config.error_file_path,
        )

    def _initialize_monitoring(self) -> None:
        """Initialize monitoring systems"""
        self._monitoring_active = False
        self._health_status = "healthy"
        self._last_health_check = time.time()

        # Performance metrics
        self._processing_times: List[float] = []
        self._component_performance: Dict[str, List[float]] = defaultdict(list)
        self._cache_stats = {"hits": 0, "misses": 0}

        # Start monitoring thread
        self._start_monitoring()

    def _initialize_state(self) -> None:
        """Initialize system state"""
        self._episode_count = 0
        self._total_memories_processed = 0
        self._last_optimization = time.time()

        # Component states
        self._component_states = {
            name: {"status": "ready", "last_run": 0, "errors": 0} for name in self.components.keys()
        }

    def _count_enabled_components(self) -> int:
        """Count enabled components"""
        return len(self.components)

    def _start_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self._monitoring_active:
            return

        def monitoring_loop():
            while self._monitoring_active:
                try:
                    self._check_health()
                    self._update_performance_metrics()
                    self._check_memory_pressure()
                    time.sleep(self.unified_config.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitor_thread.start()

    # BaseModule hook
    def _initialize(self) -> None:
        """Initialize module (called by BaseModule)"""
        try:
            # Set initial status in SmartInfoBus
            initial_status = {
                "components_enabled": self._count_enabled_components(),
                "memory_size": 0,
                "status": "initialized",
            }

            self.smart_bus.set(
                "unified_memory_status",
                initial_status,
                module="UnifiedMemory",
                thesis="Unified memory system initialized",
            )

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    async def _run_components(self, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Dispatch to the appropriate execution mode (parallel vs sequential).
        Exists primarily to satisfy static typing and keep process() clean.
        """
        if self.unified_config.parallel_processing:
            return await self._run_parallel_components(context)
        return await self._run_sequential_components(context)

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """
        Main processing loop coordinating all components

        Returns all contract-required keys with proper values
        """
        start_time = time.time()

        # Debug logging
        if self.unified_config.debug:
            self.debug_logger.log_input("PROCESS_START", inputs)

        try:
            # Check circuit breaker
            if self.circuit_breaker["state"] == "OPEN":
                if time.time() - self.circuit_breaker["last_failure"] > 60:
                    self.circuit_breaker["state"] = "HALF_OPEN"
                else:
                    return self._create_fallback_response("Circuit breaker open")

            # 1. Extract unified context
            context = await self._extract_unified_context(inputs)

            # 2. Store new experiences
            await self._store_experiences(context)

            # 3. Run components
            component_results = await self._run_components(context)

            # 4. Merge results
            unified_result = self._merge_results(component_results)

            # 5. Apply budget optimization
            if "budget" in component_results:
                unified_result = self._apply_budget_optimization(
                    unified_result,
                    component_results["budget"],
                )

            # 6. Update SmartInfoBus
            await self._update_all_bus_keys(unified_result)

            # 7. Generate thesis
            thesis = self._generate_unified_thesis(unified_result, context)
            unified_result["_thesis"] = thesis

            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)

            # Debug logging
            if self.unified_config.debug:
                self.debug_logger.log_output("PROCESS_COMPLETE", unified_result)
                self.debug_logger.log_performance(
                    {
                        "processing_time_ms": processing_time,
                        "total_memories": self.memory_store.size(),
                        "cache_hit_rate": self._calculate_cache_hit_rate(),
                        "component_health": self._assess_component_health(component_results),
                    }
                )

            return unified_result

        except Exception as e:
            return await self._handle_error(e, start_time)

    async def _extract_unified_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context once for all components"""
        cache_key = f"context_{int(time.time())}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._cache_stats["hits"] += 1
            return cached

        self._cache_stats["misses"] += 1

        context: Dict[str, Any] = {
            "trades": self.smart_bus.get("trades", "UnifiedMemory") or [],
            "actions": self.smart_bus.get("actions", "UnifiedMemory") or [],
            "market_context": self.smart_bus.get("market_context", "UnifiedMemory") or {},
            "market_data": self.smart_bus.get("market_data", "UnifiedMemory") or {},
            "features": self.smart_bus.get("features", "UnifiedMemory"),
            "episode_data": self.smart_bus.get("episode_data", "UnifiedMemory") or {},
            "observations": self.smart_bus.get("observations", "UnifiedMemory"),
            "rewards": self.smart_bus.get("rewards", "UnifiedMemory"),
            "risk_data": self.smart_bus.get("risk_data", "UnifiedMemory") or {},
            "time_risk_analysis": self.smart_bus.get("time_risk_analysis", "UnifiedMemory") or {},
            "prices": self.smart_bus.get("prices", "UnifiedMemory") or {},
            "timestamp": datetime.now(),
            "episode": inputs.get("episode", self._episode_count),
        }

        # Extract features if needed
        if context["features"] is None and context["observations"] is not None:
            context["features"] = self.feature_extractor.extract(
                observations=context["observations"],
                market_context=context["market_context"],
            )

        self.cache.put(cache_key, context, ttl=1)
        return context

    async def _store_experiences(self, context: Dict[str, Any]) -> None:
        """Store new experiences in unified store (with diagnostics)."""
        trades = context.get("trades", []) or []
        if self.unified_config.debug:
            self.debug_logger.debug(
                f"STORE_DIAG: incoming trades={len(trades)}",
                component="store",
                data={"sample_keys": list(trades[0].keys()) if trades and isinstance(trades[0], dict) else "n/a"}
            )

        invalid_type = 0
        missing_pnl = 0
        batch: List[Dict[str, Any]] = []

        for trade in trades[-20:]:
            if not isinstance(trade, dict):
                invalid_type += 1
                continue
            if "pnl" not in trade:       # ← this is the strict filter
                missing_pnl += 1
                continue

            try:
                entry = {
                    "timestamp": time.time(),
                    "features": self.feature_extractor.extract_trade_features(
                        trade, context.get("market_context", {})
                    ),
                    "action": self.utils.extract_action(trade),
                    "pnl": float(trade.get("pnl", 0.0)),
                    "context": context["market_context"],
                    "metadata": {
                        "regime": context["market_context"].get("regime"),
                        "volatility": context["market_context"].get("volatility"),
                        "session": context["market_context"].get("session"),
                        "episode": context.get("episode", self._episode_count),
                    },
                }
                batch.append(entry)
            except Exception as e:
                if self.unified_config.debug:
                    self.debug_logger.log_error("STORE_DIAG_EXTRACT_FEATURES", e)

        if self.unified_config.debug:
            self.debug_logger.debug(
                "STORE_DIAG: filtering summary",
                component="store",
                data={
                    "candidates": len(trades[-20:]),
                    "invalid_type": invalid_type,
                    "missing_pnl": missing_pnl,
                    "accepted": len(batch)
                }
            )

        if batch:
            await self.memory_store.add_batch(batch)
            self._total_memories_processed += len(batch)
            if self.unified_config.debug:
                self.debug_logger.log_memory_operation("STORE_BATCH", {
                    "count": len(batch),
                    "total_stored": self.memory_store.size(),
                })


    async def _run_parallel_components(self, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run components in parallel groups based on dependencies"""
        results: Dict[str, Dict[str, Any]] = {}

        # Group 1: Independent components
        group1_tasks: List[Awaitable[Dict[str, Any]]] = []
        group1_names: List[str] = []

        if "replay" in self.components:
            group1_tasks.append(self._run_component("replay", context))
            group1_names.append("replay")

        if "mistakes" in self.components:
            group1_tasks.append(self._run_component("mistakes", context))
            group1_names.append("mistakes")

        if "playbook" in self.components:
            group1_tasks.append(self._run_component("playbook", context))
            group1_names.append("playbook")

        if group1_tasks:
            group1_results = await asyncio.gather(*group1_tasks, return_exceptions=True)
            for name, result in zip(group1_names, group1_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Component {name} failed: {result}")
                    results[name] = self._get_component_fallback(name)
                else:
                    results[name] = cast(Dict[str, Any], result)

        # Group 2: Depends on memory data
        group2_context = {**context, "memory_data": self.memory_store.get_recent(100)}
        group2_tasks: List[Awaitable[Dict[str, Any]]] = []
        group2_names: List[str] = []

        if "compression" in self.components:
            group2_tasks.append(self._run_component("compression", group2_context))
            group2_names.append("compression")

        if "neural" in self.components:
            neural_context = {**group2_context, "experiences": self.memory_store.get_recent(50)}
            group2_tasks.append(self._run_component("neural", neural_context))
            group2_names.append("neural")

        if group2_tasks:
            group2_results = await asyncio.gather(*group2_tasks, return_exceptions=True)
            for name, result in zip(group2_names, group2_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Component {name} failed: {result}")
                    results[name] = self._get_component_fallback(name)
                else:
                    results[name] = cast(Dict[str, Any], result)

        # Group 3: Budget optimization (depends on all others)
        if "budget" in self.components:
            budget_context = {**context, "component_performance": self._calculate_component_performance(results)}
            try:
                results["budget"] = await self._run_component("budget", budget_context)
            except Exception as e:
                self.logger.error(f"Budget component failed: {e}")
                results["budget"] = self._get_component_fallback("budget")

        return results

    async def _run_sequential_components(self, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run components sequentially"""
        results: Dict[str, Dict[str, Any]] = {}

        for name, _component in self.components.items():
            try:
                component_context = context.copy()
                if name == "compression":
                    component_context["memory_data"] = self.memory_store.get_recent(100)
                elif name == "neural":
                    component_context["experiences"] = self.memory_store.get_recent(50)
                elif name == "budget":
                    component_context["component_performance"] = self._calculate_component_performance(results)

                results[name] = await self._run_component(name, component_context)

            except Exception as e:
                self.logger.error(f"Component {name} failed: {e}")
                results[name] = self._get_component_fallback(name)

        return results

    async def _run_component(self, name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single component with monitoring"""
        start_time = time.time()
        try:
            component = self.components[name]
            result = await component.process(context)

            # Update component state
            self._component_states[name]["last_run"] = time.time()
            self._component_states[name]["status"] = "success"

            # Record performance
            processing_time = (time.time() - start_time) * 1000
            self._component_performance[name].append(processing_time)

            if self.unified_config.debug:
                self.debug_logger.log_component_operation(
                    name,
                    "PROCESS",
                    {
                        "input": {"context_keys": list(context.keys())},
                        "output": {"result_keys": list(result.keys())},
                        "time_ms": processing_time,
                        "status": "SUCCESS",
                    },
                )

            return result

        except Exception as e:
            self._component_states[name]["errors"] += 1
            self._component_states[name]["status"] = "error"
            if self.unified_config.debug:
                self.debug_logger.log_error(f"COMPONENT_{name.upper()}_ERROR", e)
            raise

    def _merge_results(self, component_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Merge component results into unified output"""
        merged: Dict[str, Any] = {}

        key_mappings: Dict[str, List[str]] = {
            "replay": ["learning_progress", "pattern_analysis", "replay_sequences", "sequence_quality"],
            "budget": ["allocation_strategy", "budget_optimization", "memory_allocation", "memory_efficiency"],
            "compression": ["compressed_patterns", "feature_importance", "intuition_vector", "memory_compression"],
            "mistakes": ["danger_zones", "loss_prevention", "mistake_avoidance", "mistake_memory", "pattern_recognition"],
            "neural": ["attention_retrieval", "importance_scoring", "memory_embedding", "neural_memory"],
            "playbook": ["memory_analytics", "pattern_memory", "playbook_quality", "playbook_recall"],
        }

        for component, keys in key_mappings.items():
            result = component_results.get(component, {})
            for key in keys:
                if key in result:
                    merged[key] = result[key]
                else:
                    merged[key] = self._get_default_value(key)

        merged["unified_metrics"] = {
            "total_memories": self.memory_store.size(),
            "memory_utilization": self.memory_store.utilization(),
            "components_active": len(component_results),
            "processing_status": "complete",
            "health_status": self._health_status,
        }
        return merged

    def _apply_budget_optimization(self, result: Dict[str, Any], budget_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply budget optimization to results"""
        if "memory_allocation" in budget_result:
            allocation = budget_result["memory_allocation"]
            if "trades" in allocation and "replay_sequences" in result:
                result["replay_sequences"]["max_sequences"] = allocation["trades"]
            if "mistakes" in allocation and "mistake_memory" in result:
                result["mistake_memory"]["max_mistakes"] = allocation["mistakes"]
            if "plays" in allocation and "playbook_recall" in result:
                result["playbook_recall"]["max_entries"] = allocation["plays"]
        return result

    async def _update_all_bus_keys(self, result: Dict[str, Any]) -> None:
        """Update SmartInfoBus with all provided keys"""
        updates: List[Tuple[str, Any]] = []
        for key in [
            # Replay keys
            "replay_sequences",
            "pattern_analysis",
            "learning_progress",
            "sequence_quality",
            # Budget keys
            "memory_allocation",
            "memory_efficiency",
            "budget_optimization",
            "allocation_strategy",
            # Compression keys
            "intuition_vector",
            "compressed_patterns",
            "memory_compression",
            "feature_importance",
            # Mistake keys
            "mistake_memory",
            "mistake_avoidance",
            "danger_zones",
            "loss_prevention",
            "pattern_recognition",
            # Neural keys
            "neural_memory",
            "attention_retrieval",
            "memory_embedding",
            "importance_scoring",
            # Playbook keys
            "playbook_recall",
            "pattern_memory",
            "playbook_quality",
            "memory_analytics",
        ]:
            if key in result:
                updates.append((key, result[key]))

        for key, value in updates:
            self.smart_bus.set(key, value, module="UnifiedMemory", thesis=f"Unified memory: {key}")

        if self.unified_config.debug:
            self.debug_logger.log_bus_updates(cast(Sequence[Tuple[str, Any]], updates))

    def _generate_unified_thesis(self, result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate comprehensive unified thesis"""
        try:
            parts: List[str] = []
            total_memories = self.memory_store.size()
            utilization = self.memory_store.utilization()
            parts.append(f"Unified Memory: {total_memories} entries ({utilization:.1%} capacity)")

            if "replay_sequences" in result:
                replay = result["replay_sequences"]
                parts.append(f"Replay: {replay.get('sequences_analyzed', 0)} sequences analyzed")

            if "mistake_memory" in result:
                mistakes = result["mistake_memory"]
                parts.append(f"Mistakes: {mistakes.get('avoidance_signal', 0.0):.2f} avoidance signal")

            if "neural_memory" in result:
                neural = result["neural_memory"]
                parts.append(f"Neural: {neural.get('buffer_size', 0)} embeddings stored")

            if "playbook_recall" in result:
                playbook = result["playbook_recall"]
                parts.append(f"Playbook: {playbook.get('patterns_identified', 0)} patterns")

            if self._processing_times:
                avg_time = np.mean(self._processing_times[-10:])
                parts.append(f"Performance: {avg_time:.1f}ms avg")

            parts.append(f"Health: {self._health_status}")
            return " | ".join(parts)
        except Exception as e:
            return f"Unified memory processing (error generating thesis: {e})"

    def _calculate_component_performance(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics for each component"""
        performance: Dict[str, Any] = {}
        for name in results.keys():
            if name in self._component_performance:
                recent_times = self._component_performance[name][-10:]
                performance[name] = {
                    "avg_time_ms": float(np.mean(recent_times)) if recent_times else 0.0,
                    "max_time_ms": float(max(recent_times)) if recent_times else 0.0,
                    "errors": self._component_states[name]["errors"],
                    "status": self._component_states[name]["status"],
                }
        return performance

    def _calculate_cache_hit_rate(self) -> float:
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        if total == 0:
            return 0.0
        return self._cache_stats["hits"] / total

    def _assess_component_health(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Assess health of each component"""
        health: Dict[str, str] = {}
        for name in self.components.keys():
            state = self._component_states.get(name, {})
            if state.get("errors", 0) > 5:
                health[name] = "critical"
            elif state.get("errors", 0) > 2:
                health[name] = "warning"
            elif name in results and results[name]:
                health[name] = "healthy"
            else:
                health[name] = "unknown"
        return health

    def _get_default_value(self, key: str) -> Any:
        """Get default value for a missing key"""
        defaults: Dict[str, Any] = {
            # Replay defaults
            "learning_progress": {"episodes_processed": 0, "patterns_learned": 0},
            "pattern_analysis": {"total_patterns": 0, "profitable_patterns": 0},
            "replay_sequences": {"total_sequences": 0, "best_sequence_pnl": 0.0},
            "sequence_quality": {"current_quality": 0.0, "average_quality": 0.0},
            # Budget defaults
            "allocation_strategy": {"allocation_method": "default", "rebalance_frequency": 50},
            "budget_optimization": {"optimality_score": 0.5, "total_profit": 0.0, "optimization_count": 0},
            "memory_allocation": {"trades": 500, "mistakes": 100, "plays": 200},
            "memory_efficiency": {},
            # Compression defaults
            "compressed_patterns": {"profit_direction": [], "loss_direction": []},
            "feature_importance": {"profit_components": [], "explained_variance_ratio": []},
            "intuition_vector": {"vector": [], "strength": 0.0},
            "memory_compression": {"total_memories": 0, "compression_efficiency": 0.0},
            # Mistakes defaults
            "danger_zones": {"zones": [], "zone_count": 0},
            "loss_prevention": {"avoidance_effectiveness": 0.0, "learning_samples": 0},
            "mistake_avoidance": {"avoidance_signal": 0.0, "consecutive_losses": 0},
            "mistake_memory": {"current_score": 0.0, "avoidance_signal": 0.0},
            "pattern_recognition": {"loss_patterns": {}, "win_patterns": {}},
            # Neural defaults
            "attention_retrieval": {"retrieved_count": 0, "similarity_scores": []},
            "importance_scoring": {"average_importance": 0.0, "total_scored": 0},
            "memory_embedding": {"embedding_dim": self.unified_config.embed_dim, "total_embeddings": 0},
            "neural_memory": {"buffer_size": 0, "memory_utilization": 0.0},
            # Playbook defaults
            "memory_analytics": {"total_recalls": 0, "memory_health": "unknown"},
            "pattern_memory": {"total_patterns": 0, "pattern_effectiveness": {}},
            "playbook_quality": {"memory_utilization": 0.0, "quality_score": 0.0},
            "playbook_recall": {"memory_entries": 0, "patterns_identified": 0},
        }
        return defaults.get(key, {})

    def _get_component_fallback(self, name: str) -> Dict[str, Any]:
        """Get fallback response for failed component"""
        fallbacks = {
            "replay": {
                "learning_progress": self._get_default_value("learning_progress"),
                "pattern_analysis": self._get_default_value("pattern_analysis"),
                "replay_sequences": self._get_default_value("replay_sequences"),
                "sequence_quality": self._get_default_value("sequence_quality"),
            },
            "budget": {
                "allocation_strategy": self._get_default_value("allocation_strategy"),
                "budget_optimization": self._get_default_value("budget_optimization"),
                "memory_allocation": self._get_default_value("memory_allocation"),
                "memory_efficiency": self._get_default_value("memory_efficiency"),
            },
            "compression": {
                "compressed_patterns": self._get_default_value("compressed_patterns"),
                "feature_importance": self._get_default_value("feature_importance"),
                "intuition_vector": self._get_default_value("intuition_vector"),
                "memory_compression": self._get_default_value("memory_compression"),
            },
            "mistakes": {
                "danger_zones": self._get_default_value("danger_zones"),
                "loss_prevention": self._get_default_value("loss_prevention"),
                "mistake_avoidance": self._get_default_value("mistake_avoidance"),
                "mistake_memory": self._get_default_value("mistake_memory"),
                "pattern_recognition": self._get_default_value("pattern_recognition"),
            },
            "neural": {
                "attention_retrieval": self._get_default_value("attention_retrieval"),
                "importance_scoring": self._get_default_value("importance_scoring"),
                "memory_embedding": self._get_default_value("memory_embedding"),
                "neural_memory": self._get_default_value("neural_memory"),
            },
            "playbook": {
                "memory_analytics": self._get_default_value("memory_analytics"),
                "pattern_memory": self._get_default_value("pattern_memory"),
                "playbook_quality": self._get_default_value("playbook_quality"),
                "playbook_recall": self._get_default_value("playbook_recall"),
            },
        }
        return fallbacks.get(name, {})

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create complete fallback response with all required keys"""
        response: Dict[str, Any] = {}
        for key in [
            "learning_progress",
            "pattern_analysis",
            "replay_sequences",
            "sequence_quality",
            "allocation_strategy",
            "budget_optimization",
            "memory_allocation",
            "memory_efficiency",
            "compressed_patterns",
            "feature_importance",
            "intuition_vector",
            "memory_compression",
            "danger_zones",
            "loss_prevention",
            "mistake_avoidance",
            "mistake_memory",
            "pattern_recognition",
            "attention_retrieval",
            "importance_scoring",
            "memory_embedding",
            "neural_memory",
            "memory_analytics",
            "pattern_memory",
            "playbook_quality",
            "playbook_recall",
        ]:
            response[key] = self._get_default_value(key)

        response["_thesis"] = f"Unified memory fallback: {reason}"
        response["fallback_reason"] = reason
        return response

    async def _handle_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors"""
        processing_time = (time.time() - start_time) * 1000

        # Update circuit breaker
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()
        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
            self.circuit_breaker["state"] = "OPEN"

        # Log error
        self.logger.error(
            format_operator_message(
                "❌",
                "UNIFIED_MEMORY_ERROR",
                error=str(error),
                processing_time_ms=processing_time,
                context="unified_memory",
            )
        )

        # Record failure
        self._record_failure(error)

        # Debug logging
        if self.unified_config.debug:
            self.debug_logger.log_error("PROCESS_ERROR", error)

        return self._create_fallback_response(f"Error: {str(error)}")

    def _record_success(self, processing_time: float) -> None:
        """Record successful processing"""
        self._processing_times.append(processing_time)
        if len(self._processing_times) > 100:
            self._processing_times.pop(0)

        # Reset circuit breaker on success
        if self.circuit_breaker["state"] == "HALF_OPEN":
            self.circuit_breaker["failures"] = 0
            self.circuit_breaker["state"] = "CLOSED"

        # Update performance tracker
        self.performance_tracker.record_metric("UnifiedMemory", "process_cycle", processing_time, True)

    def _record_failure(self, error: Exception) -> None:
        """Record processing failure"""
        self.performance_tracker.record_metric("UnifiedMemory", "process_cycle", 0, False)

    def _check_health(self) -> None:
        """Check system health"""
        try:
            comp_states = getattr(self, "_component_states", {})
            if self.memory_store.utilization() > 0.95:
                self._health_status = "warning"
            elif any((state or {}).get("errors", 0) > 5 for state in comp_states.values()):
                self._health_status = "critical"
            elif self._processing_times and np.mean(self._processing_times) > self.unified_config.max_processing_time_ms:
                self._health_status = "warning"
            else:
                self._health_status = "healthy"

            self._last_health_check = time.time()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = "unknown"

    def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        try:
            for name in list(self._component_performance.keys()):
                if len(self._component_performance[name]) > 100:
                    self._component_performance[name] = self._component_performance[name][-100:]
        except Exception as e:
            self.logger.error(f"Performance update failed: {e}")

    def _check_memory_pressure(self) -> None:
        """Check and handle memory pressure"""
        try:
            utilization = self.memory_store.utilization()
            if utilization > 0.9:
                self.memory_store.cleanup(keep_ratio=0.8)
                self.logger.warning(f"Memory pressure detected: {utilization:.1%}, triggered cleanup")
        except Exception as e:
            self.logger.error(f"Memory pressure check failed: {e}")




    # Public methods

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            "config": asdict(self.unified_config),
            "episode_count": self._episode_count,
            "total_memories_processed": self._total_memories_processed,
            "component_states": dict(self._component_states),
            "health_status": self._health_status,
            "circuit_breaker": dict(self.circuit_breaker),
            "cache_stats": dict(self._cache_stats),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state from persistence"""
        if "episode_count" in state:
            self._episode_count = state["episode_count"]
        if "total_memories_processed" in state:
            self._total_memories_processed = state["total_memories_processed"]
        if "component_states" in state:
            self._component_states.update(state["component_states"])
        if "health_status" in state:
            self._health_status = state["health_status"]
        if "circuit_breaker" in state:
            self.circuit_breaker.update(state["circuit_breaker"])
        if "cache_stats" in state:
            self._cache_stats.update(state["cache_stats"])

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status snapshot"""
        return {
            "status": self._health_status,
            "last_check": self._last_health_check,
            "circuit_breaker": self.circuit_breaker["state"],
            "total_memories": self.memory_store.size(),
            "memory_utilization": self.memory_store.utilization(),
            "components": self._assess_component_health({}),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

    def stop_monitoring(self) -> None:
        """Stop monitoring thread"""
        self._monitoring_active = False
        if hasattr(self, "_monitor_thread"):
            self._monitor_thread.join(timeout=1.0)

    # Legacy compatibility

    async def propose_action(self, **inputs: Any) -> Dict[str, Any]:
        result = await self.process(**inputs)
        if "intuition_vector" in result:
            vector = result["intuition_vector"].get("vector", [])
            if vector and len(vector) >= 2:
                return {"action": [float(vector[0]), float(vector[1])], "confidence": 0.7, "thesis": "Unified memory action proposal"}
        return {"action": [0.0, 0.0], "confidence": 0.5, "thesis": "Default action"}

    async def calculate_confidence(self, action: Dict[str, Any], **inputs: Any) -> float:
        if self._health_status == "healthy":
            base = 0.7
        elif self._health_status == "warning":
            base = 0.5
        else:
            base = 0.3

        utilization = self.memory_store.utilization()
        if 0.2 < utilization < 0.8:
            base += 0.1

        cache_hit_rate = self._calculate_cache_hit_rate()
        base += cache_hit_rate * 0.2

        return min(1.0, max(0.0, base))

    def confidence(self, obs: Any = None, **kwargs: Any) -> float:
        """Legacy compatibility for confidence"""
        return 0.5
