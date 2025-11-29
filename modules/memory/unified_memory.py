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
from collections import defaultdict, deque
from typing import Awaitable, Sequence, cast, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from modules.utils.metrics_utils import sanitize_metrics

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
from .components.loss_risk_head import LossRiskHeadComponent
from .components.interventions import InterventionsComponent

# Import shared resources
from .shared.memory_store import UnifiedMemoryStore
from .shared.feature_extractor import UnifiedFeatureExtractor
from .shared.pattern_detector import UnifiedPatternDetector
from .shared.utils import MemoryUtils, LRUCache, safe_float

# Import debug logger
from .debug.memory_logger import MemoryDebugLogger


@dataclass
class UnifiedMemoryConfig:
    """Unified configuration for all memory subsystems"""
    # Debug master switches
    debug: bool = False
    debug_level: str = "INFO"  # "TRACE"|"DEBUG"|"INFO"|"WARNING"|"ERROR"

    # Combined debug file (legacy default)
    debug_log_path: str = "logs/memory/unified_debug.log"
    enable_combined_log: bool = False

    # Per-level file toggles + paths (None = derive from debug_log_path)
    enable_trace_log: bool = False
    trace_file_path: Optional[str] = None

    enable_debug_log: bool = False
    debug_file_path: Optional[str] = None

    enable_info_log: bool = False
    info_file_path: Optional[str] = None

    enable_warning_log: bool = False
    warning_file_path: Optional[str] = None

    enable_error_log: bool = False
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
    enable_loss_risk_head: bool = True
    enable_interventions: bool = True

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

    # Loss Risk Head settings (NEW: per enhance.md)
    loss_threshold: float = 0.0  # P(loss > τ) where τ is this threshold
    loss_head_lr: float = 0.001
    loss_head_hidden_dim: int = 64

    # Interventions settings (NEW: per enhance.md)
    intervention_decay: float = 0.99  # Decay factor for intervention strength
    intervention_min_samples: int = 3  # Min samples before intervention activates


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
        
        # Load persisted memories from disk
        self.load_memory_store()

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

        # NEW: Loss Risk Head component (per enhance.md)
        if self.unified_config.enable_loss_risk_head:
            self.components["loss_risk_head"] = LossRiskHeadComponent(self.unified_config, shared_resources)

        # NEW: Interventions component (per enhance.md)
        if self.unified_config.enable_interventions:
            self.components["interventions"] = InterventionsComponent(self.unified_config, shared_resources)

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

        # Performance metrics (FIX: use deques with maxlen to prevent memory leak)
        self._processing_times: deque = deque(maxlen=100)
        self._component_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._cache_stats = {"hits": 0, "misses": 0}

        # Start monitoring thread
        self._start_monitoring()

    def _initialize_state(self) -> None:
        """Initialize system state"""
        self._episode_count = 0
        self._total_memories_processed = 0
        self._last_optimization = time.time()
        
        # Step-based throttling for performance (skip expensive components most steps)
        self._step_count = 0
        self._throttle_interval = 5  # Run expensive components every 5 steps
        self._cached_component_results: Dict[str, Dict[str, Any]] = {}
        self._cached_unified_result: Optional[Dict[str, Any]] = None

        # Component states
        self._component_states = {
            name: {"status": "ready", "last_run": 0, "errors": 0} for name in self.components.keys()
        }

        # Training metrics (bounded history to prevent memory growth)
        self._training_metrics_history: deque = deque(maxlen=100)
        self._training_metrics_current: Dict[str, Any] = {}
        
        # Component performance tracking
        self._component_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Cache stats
        self._cache_stats = {"hits": 0, "misses": 0}

    def _count_enabled_components(self) -> int:
        """Count enabled components"""
        return len(self.components)

    def _start_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self._monitoring_active:
            return

        def monitoring_loop():
            save_counter = 0
            while self._monitoring_active:
                try:
                    self._check_health()
                    self._update_performance_metrics()
                    self._check_memory_pressure()
                    
                    # Save memory store periodically (every 10 health checks ~5 minutes)
                    save_counter += 1
                    if save_counter >= 10 and self.memory_store.size() > 0:
                        self.save_memory_store()
                        save_counter = 0
                    
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

            # Seed mistake_memory with a typed, empty baseline (single-writer)
            try:
                baseline_mm = {"recent": [], "stats": {"count": 0, "last_ts": None}}
                self.smart_bus.set(
                    "mistake_memory",
                    baseline_mm,
                    module="UnifiedMemory",
                    thesis="Seeded baseline mistake_memory"
                )
            except Exception:
                pass

            # ═══════════════════════════════════════════════════════════════════
            # Seed default values for ALL memory outputs so downstream modules
            # (PositionManager, Executor, VotingKernel, etc.) can run before
            # UnifiedMemory's first process() cycle completes.
            # ═══════════════════════════════════════════════════════════════════
            self._publish_default_outputs()

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    def _publish_default_outputs(self) -> None:
        """
        Publish default/neutral values for all memory outputs.
        
        This ensures downstream modules (PositionManager, Executor, etc.) can
        execute before UnifiedMemory has completed its first process() cycle.
        These values are conservative/neutral to avoid affecting trading decisions.
        """
        try:
            # memory_gate: Default to no veto, neutral risk
            default_gate = {
                "veto": False,
                "risk_multiplier": 1.0,
                "confidence": 0.5,
                "reasons": ["memory_initializing"],
                "risk_score": 0.0,
                "danger_similarity": 0.0,
                "loss_prob": 0.0,
            }
            self.smart_bus.set("memory_gate", default_gate, module="UnifiedMemory",
                               thesis="Default memory gate (pre-initialization)")

            # memory_vote: Neutral vote
            default_vote = {
                "signed_bias": 0.0,
                "confidence": 0.5,
                "expected_pnl": 0.0,
                "weight": 0.0,
            }
            self.smart_bus.set("memory_vote", default_vote, module="UnifiedMemory",
                               thesis="Default memory vote (pre-initialization)")

            # memory_rationale: Empty rationale
            default_rationale = {
                "danger_evidence": [],
                "playbook_evidence": [],
                "intervention_evidence": [],
                "summary": "Memory system initializing",
            }
            self.smart_bus.set("memory_rationale", default_rationale, module="UnifiedMemory",
                               thesis="Default memory rationale (pre-initialization)")

            # playbook_recall: Empty recall
            default_playbook = {
                "signed_bias": 0.0,
                "confidence": 0.5,
                "expected_pnl": 0.0,
                "similar_patterns": [],
                "recommendation": "neutral",
            }
            self.smart_bus.set("playbook_recall", default_playbook, module="UnifiedMemory",
                               thesis="Default playbook recall (pre-initialization)")

            # intuition_vector: Zero vector
            import numpy as np
            default_intuition = np.zeros(32, dtype=np.float32).tolist()
            self.smart_bus.set("intuition_vector", default_intuition, module="UnifiedMemory",
                               thesis="Default intuition vector (pre-initialization)")

            # danger_zones: Empty zones
            default_danger = {
                "zones": [],
                "current_risk": 0.0,
                "nearest_zone_distance": float("inf"),
            }
            self.smart_bus.set("danger_zones", default_danger, module="UnifiedMemory",
                               thesis="Default danger zones (pre-initialization)")

            # mistake_avoidance: Neutral avoidance
            default_avoidance = {
                "avoidance_signal": 0.0,
                "similar_mistakes": [],
                "recommendation": "proceed",
            }
            self.smart_bus.set("mistake_avoidance", default_avoidance, module="UnifiedMemory",
                               thesis="Default mistake avoidance (pre-initialization)")

            # neural_risk_hint: Neutral hint
            self.smart_bus.set("neural_risk_hint", 0.5, module="UnifiedMemory",
                               thesis="Default neural risk hint (pre-initialization)")

            self.logger.info("Published default memory outputs for downstream modules")

        except Exception as e:
            self.logger.error(f"Failed to publish default outputs: {e}")

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
        
        Performance optimization: Uses step-based throttling to skip expensive
        component execution most steps. Cached results are returned when throttled.
        """
        start_time = time.time()
        self._step_count += 1

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

            # Throttling: Return cached result if available and not time to refresh
            should_refresh = (self._step_count % self._throttle_interval == 0) or self._cached_unified_result is None
            
            if not should_refresh and self._cached_unified_result is not None:
                # Fast path: return cached result with updated timestamp
                cached = self._cached_unified_result.copy()
                cached["_thesis"] = f"Cached memory result (step {self._step_count}, refresh every {self._throttle_interval})"
                processing_time = (time.time() - start_time) * 1000
                self._record_success(processing_time)
                return cached

            # 1. Extract unified context
            context = await self._extract_unified_context(inputs)

            # 2. Store new experiences (lightweight, do every step)
            await self._store_experiences(context)

            # 3. Run components (expensive, throttled)
            component_results = await self._run_components(context)

            # 4. Merge results
            unified_result = self._merge_results(component_results)

            # 4a. Compose memory_gate, memory_vote, and memory_rationale
            memory_gate, memory_vote, memory_rationale = self._compose_gate_and_vote(
                component_results, context
            )
            unified_result["memory_gate"] = memory_gate
            unified_result["memory_vote"] = memory_vote
            unified_result["memory_rationale"] = memory_rationale
            
            # 4a.1. Extract neural_risk_hint as top-level key (required by contract)
            unified_result["neural_risk_hint"] = memory_vote.get("neural_risk_hint", 0.5)

            # 4b. Ingest optional training metrics (from bus or inputs) and expose consolidated progress
            try:
                tm = inputs.get("training_metrics") or self.smart_bus.get("enhanced_performance", "UnifiedMemory")
                if isinstance(tm, dict) and tm:
                    safe_tm = sanitize_metrics(tm)
                    self._training_metrics_current = safe_tm
                    self._training_metrics_history.append({
                        **{k: v for k, v in safe_tm.items() if k in ("step", "episode_reward_mean", "steps_per_second", "system_health_score", "env_balance", "circuit_breaker_active")},
                        "timestamp": safe_tm.get("timestamp", datetime.now().isoformat()),
                    })
                    unified_result["training_progress"] = {
                        "current": self._training_metrics_current,
                        "history": list(self._training_metrics_history),
                    }
            except Exception:
                # Non-fatal; training metrics are optional
                pass

            # 5. Apply budget optimization
            if "budget" in component_results:
                unified_result = self._apply_budget_optimization(
                    unified_result,
                    component_results["budget"],
                )

            # 6. Ensure required outputs are present (never return partials silently)
            unified_result = self._ensure_required_outputs(unified_result, reason=None)

            # 7. Update SmartInfoBus
            await self._update_all_bus_keys(unified_result)

            # 8. Generate thesis
            thesis = self._generate_unified_thesis(unified_result, context)
            unified_result["_thesis"] = thesis
            
            # 9. Cache result for throttled fast-path returns
            self._cached_unified_result = unified_result.copy()
            self._cached_component_results = component_results

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

        # Get trades from multiple sources for robustness
        trades = self.smart_bus.get("trades", "UnifiedMemory") or []

        # Also try current_fills and recent_trades as fallbacks
        if not trades:
            trades = self.smart_bus.get("current_fills", "UnifiedMemory") or []
        if not trades:
            trades = self.smart_bus.get("recent_trades", "UnifiedMemory") or []

        # Get position data to track unrealized P&L
        positions = self.smart_bus.get("position_data", "UnifiedMemory") or {}
        if isinstance(positions, dict):
            positions = positions.get("positions", [])

        context: Dict[str, Any] = {
            "trades": trades,
            "positions": positions,  # Add positions to context
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

    def _extract_query_from_context(self, market_context: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract a query vector from market context for neural retrieval.
        
        Creates a lightweight feature vector from available market signals.
        Returns None if insufficient data.
        """
        try:
            features: List[float] = []
            
            # Volatility
            vol = market_context.get("volatility", 0.5)
            if isinstance(vol, dict):
                vol = float(list(vol.values())[0]) if vol else 0.5
            features.append(safe_float(vol, 0.5))
            
            # Regime encoding
            regime_map = {"trending": 1.0, "ranging": 0.0, "volatile": 0.5}
            regime = str(market_context.get("regime", "unknown")).lower()
            features.append(safe_float(regime_map.get(regime, 0.25), 0.25))
            
            # Session encoding
            session_map = {"asian": 0.0, "european": 0.5, "american": 1.0}
            session = str(market_context.get("session", "unknown")).lower()
            features.append(safe_float(session_map.get(session, 0.25), 0.25))
            
            # Risk metrics
            features.append(safe_float(market_context.get("drawdown_pct", 0.0), 0.0) / 100.0)
            features.append(safe_float(market_context.get("exposure_pct", 0.0), 0.0) / 100.0)
            
            # Pad to embed_dim
            embed_dim = int(getattr(self.unified_config, "embed_dim", 32))
            while len(features) < embed_dim:
                features.append(0.0)
            
            return np.array(features[:embed_dim], dtype=np.float32)
        except Exception:
            return None

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

            # Accept trades with any P&L field (pnl, realized_pnl, unrealized_pnl)
            # This allows storing both trade fills and position snapshots
            has_pnl = any(k in trade for k in ["pnl", "realized_pnl", "realised_pnl", "unrealized_pnl"])
            if not has_pnl:
                missing_pnl += 1
                continue

            try:
                # Extract P&L from multiple possible fields
                pnl_value = 0.0
                # Priority: pnl > realized_pnl > unrealized_pnl
                for pnl_key in ["pnl", "realized_pnl", "realised_pnl", "unrealized_pnl", "unrealized_pnl_eur"]:
                    if pnl_key in trade:
                        try:
                            pnl_value = float(trade[pnl_key])
                            break
                        except (ValueError, TypeError):
                            continue

                # Extract features for pattern detection
                trade_features = self.feature_extractor.extract_trade_features(
                    trade, context.get("market_context", {})
                )
                
                # Compute pattern_label for intervention linkage
                # Uses a hash of discretized features to create stable pattern labels
                pattern_label: Optional[str] = None
                try:
                    # Create a simple pattern label from key features
                    regime = str(context["market_context"].get("regime", "unknown"))
                    
                    # Extract volatility - try numeric first, then map string level
                    volatility_raw = context["market_context"].get("volatility")
                    if volatility_raw is None:
                        # Try volatility_level (string like "low", "medium", "high")
                        vol_level = str(context["market_context"].get("volatility_level", "medium")).lower()
                        volatility = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(vol_level, 0.5)
                    else:
                        volatility = float(volatility_raw) if volatility_raw else 0.5
                    
                    vol_bucket = "high" if volatility > 0.7 else ("low" if volatility < 0.3 else "med")
                    session = str(context["market_context"].get("session", "unknown"))
                    pnl_sign = "loss" if pnl_value < 0 else "win"
                    
                    # Compose pattern label from context
                    pattern_label = f"{regime}_{vol_bucket}_{session}_{pnl_sign}"
                except Exception:
                    volatility = 0.5
                    vol_bucket = "med"

                # Calculate importance based on P&L magnitude and trade characteristics
                importance = min(1.0, 0.3 + abs(pnl_value) * 0.1)  # Base 0.3, increases with P&L
                if pnl_value != 0.0:
                    importance = min(1.0, importance + 0.2)  # Boost for trades with actual P&L

                # Get episode from bus or training metrics
                episode = context.get("episode", 0)
                if episode == 0:
                    # Try to get from episode_data on bus
                    episode_data = context.get("episode_data", {})
                    episode = episode_data.get("episode", episode_data.get("num_episodes", 0))
                if episode == 0:
                    # Try enhanced_performance from bus (published by training callback)
                    enhanced_perf = self.smart_bus.get("enhanced_performance", "UnifiedMemory") or {}
                    episode = enhanced_perf.get("episode", enhanced_perf.get("episodes", self._episode_count))

                entry = {
                    "timestamp": time.time(),
                    "features": trade_features,
                    "action": self.utils.extract_action(trade),
                    "pnl": pnl_value,
                    "importance": importance,
                    "context": context["market_context"],
                    "metadata": {
                        "regime": context["market_context"].get("regime"),
                        "volatility": volatility,
                        "volatility_level": context["market_context"].get("volatility_level", vol_bucket),
                        "session": context["market_context"].get("session"),
                        "episode": episode,
                        "pattern_label": pattern_label,  # NEW: for intervention linkage
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
                # FIX: Pass sample memory entry with actual data instead of just batch stats
                sample_entry = batch[0] if batch else {}
                self.debug_logger.log_memory_operation("STORE_BATCH", {
                    "count": len(batch),
                    "total_stored": self.memory_store.size(),
                    "timestamp": sample_entry.get("timestamp"),
                    "pnl": sample_entry.get("pnl"),
                    "importance": sample_entry.get("importance"),
                    "metadata": sample_entry.get("metadata"),
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
            # Always request recall to compute signed_bias for memory_vote
            playbook_context = {
                **context,
                "recall_requested": True,
                "query_features": context.get("features"),
            }
            group1_tasks.append(self._run_component("playbook", playbook_context))
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
            # Pass current features as query to enable retrieval and neural_risk_hint computation
            current_features = context.get("features")
            if current_features is None:
                # Fallback: try to get features from market_context
                market_ctx = context.get("market_context", {})
                if market_ctx:
                    current_features = self._extract_query_from_context(market_ctx)
            
            neural_context = {
                **group2_context,
                "experiences": self.memory_store.get_recent(50),
                "query": current_features,  # Enable retrieval for neural_risk_hint
            }
            group2_tasks.append(self._run_component("neural", neural_context))
            group2_names.append("neural")

        # NEW: Loss risk head component
        if "loss_risk_head" in self.components:
            loss_context = {**group2_context, "experiences": self.memory_store.get_recent(50)}
            group2_tasks.append(self._run_component("loss_risk_head", loss_context))
            group2_names.append("loss_risk_head")

        # NEW: Interventions component  
        if "interventions" in self.components:
            # Pass pattern_label from recent experiences and mistakes results
            intervention_context = {
                **group2_context,
                "experiences": self.memory_store.get_recent(50),
                "mistakes_result": results.get("mistakes", {}),
            }
            group2_tasks.append(self._run_component("interventions", intervention_context))
            group2_names.append("interventions")

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
                    # Pass query for retrieval and neural_risk_hint computation
                    component_context["experiences"] = self.memory_store.get_recent(50)
                    current_features = context.get("features")
                    if current_features is None:
                        market_ctx = context.get("market_context", {})
                        if market_ctx:
                            current_features = self._extract_query_from_context(market_ctx)
                    component_context["query"] = current_features
                elif name == "playbook":
                    # Request recall to compute signed_bias for memory_vote
                    component_context["recall_requested"] = True
                    component_context["query_features"] = context.get("features")
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
        
        # Also add unified_memory_status (required by contract)
        merged["unified_memory_status"] = {
            "components_enabled": self._count_enabled_components(),
            "memory_size": self.memory_store.size(),
            "status": "active" if self._health_status == "healthy" else self._health_status,
            "health_status": self._health_status,
            "total_memories": self.memory_store.size(),
            "memory_utilization": self.memory_store.utilization(),
        }
        return merged

    def _compose_gate_and_vote(
        self,
        component_results: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Compose memory_gate, memory_vote, and memory_rationale from component outputs.
        
        This is the central fusion method per enhance.md Section 4.1:
        - memory_gate: Can veto or reduce trade size based on risk signals
        - memory_vote: Adds signed bias to ensemble (confidence * signed_bias)
        - memory_rationale: Evidence bundle for explainability
        
        Returns:
            Tuple of (memory_gate, memory_vote, memory_rationale) dicts
        """
        # === Gather signals from components ===
        
        # From mistakes component
        mistakes_result = component_results.get("mistakes", {})
        gate_snippet = mistakes_result.get("gate_snippet", {})
        mistake_avoidance = mistakes_result.get("mistake_avoidance", {})
        
        # Get danger signals from gate_snippet (primary) or mistake_avoidance (fallback)
        danger_similarity = safe_float(gate_snippet.get("danger_similarity", 0.0), 0.0)
        danger_confidence = safe_float(gate_snippet.get("confidence", 0.0), 0.0)
        avoidance_signal = safe_float(mistake_avoidance.get("avoidance_signal", gate_snippet.get("avoidance_signal", 0.0)), 0.0)
        
        # From playbook component
        playbook_result = component_results.get("playbook", {})
        playbook_recall = playbook_result.get("playbook_recall", {})
        signed_bias = safe_float(playbook_recall.get("signed_bias", 0.0), 0.0)
        playbook_confidence = safe_float(playbook_recall.get("confidence", 0.5), 0.5)
        top_neighbors = playbook_recall.get("top_neighbors", [])
        expected_pnl = safe_float(playbook_recall.get("expected_pnl", 0.0), 0.0)
        
        # From neural component
        neural_result = component_results.get("neural", {})
        neural_risk_hint = safe_float(neural_result.get("neural_risk_hint", 0.5), 0.5)
        max_attention = safe_float(neural_result.get("max_attention", 0.5), 0.5)
        
        # From loss_risk_head (if available)
        loss_risk_result = component_results.get("loss_risk_head", {})
        loss_risk_assessment = loss_risk_result.get("loss_risk_assessment", {})
        loss_prob = safe_float(loss_risk_assessment.get("loss_prob", 0.0), 0.0)
        loss_uncertainty = safe_float(loss_risk_assessment.get("uncertainty", 0.5), 0.5)
        
        # From interventions (if available)
        interventions_result = component_results.get("interventions", {})
        intervention_rec = interventions_result.get("intervention_recommendation", {})
        intervention_type = intervention_rec.get("intervention", "none") if intervention_rec else "none"
        intervention_strength = safe_float(intervention_rec.get("strength", 0.0) if intervention_rec else 0.0, 0.0)
        intervention_veto = bool(intervention_rec.get("veto_recommended", False) if intervention_rec else False)
        
        # === Compute fused risk_score ===
        # risk_score = max(loss_prob, danger_similarity) as per enhance.md
        risk_score = max(loss_prob, danger_similarity)
        
        # === Compose memory_gate ===
        # Use gate_snippet if available, otherwise compute from signals
        if gate_snippet:
            memory_gate = {
                "veto": gate_snippet.get("veto", False),
                "risk_multiplier": gate_snippet.get("risk_multiplier", 1.0),
                "confidence": gate_snippet.get("confidence", 0.5),
                "reasons": gate_snippet.get("reasons", []),
                "risk_score": risk_score,
                "danger_similarity": danger_similarity,
                "loss_prob": loss_prob,
            }
        else:
            # Fallback: compute gate from raw signals
            veto = False
            risk_multiplier = 1.0
            reasons: List[str] = []
            
            # Veto conditions
            if danger_similarity > 0.8 and danger_confidence > 0.7:
                veto = True
                reasons.append(f"High danger similarity: {danger_similarity:.2f}")
            
            if loss_prob > 0.7 and loss_uncertainty < 0.3:
                veto = True
                reasons.append(f"High loss probability: {loss_prob:.2f}")
            
            if intervention_type == "avoid" and intervention_strength > 0.8:
                veto = True
                reasons.append(f"Intervention: avoid pattern")
            
            # Also check if interventions component directly recommends veto
            if intervention_veto:
                veto = True
                reasons.append(f"Intervention veto recommended")
            
            # Risk multiplier (reduce size if risky but not vetoing)
            if not veto:
                if avoidance_signal > 0.5:
                    risk_multiplier = max(0.3, 1.0 - avoidance_signal * 0.7)
                    reasons.append(f"Size reduced by avoidance signal: {avoidance_signal:.2f}")
                
                if intervention_type == "halve_size":
                    risk_multiplier = min(risk_multiplier, 0.5)
                    reasons.append("Intervention: halve_size")
                
                if risk_score > 0.5:
                    risk_multiplier = min(risk_multiplier, 1.0 - risk_score * 0.5)
                    reasons.append(f"Risk score adjustment: {risk_score:.2f}")
            
            gate_confidence = max(danger_confidence, 1.0 - loss_uncertainty)
            
            memory_gate = {
                "veto": veto,
                "risk_multiplier": round(risk_multiplier, 3),
                "confidence": round(gate_confidence, 3),
                "reasons": reasons,
                "risk_score": round(risk_score, 3),
                "danger_similarity": round(danger_similarity, 3),
                "loss_prob": round(loss_prob, 3),
            }
        
        # === Compose memory_vote ===
        # vote_value = playbook_confidence * signed_bias
        vote_value = playbook_confidence * signed_bias
        
        # Adjust vote by neural risk hint (reduce confidence if attention is diffuse)
        attention_weight = 1.0 - neural_risk_hint * 0.3  # Scale down by up to 30%
        
        memory_vote = {
            "vote_value": round(vote_value * attention_weight, 3),
            "signed_bias": round(signed_bias, 3),
            "confidence": round(playbook_confidence * attention_weight, 3),
            "expected_pnl": round(expected_pnl, 2),
            "neural_risk_hint": round(neural_risk_hint, 3),
            "max_attention": round(max_attention, 3),
            "source": "memory",
        }
        
        # === Compose memory_rationale ===
        # Evidence bundle for explainability
        memory_rationale = {
            "gate_reasons": memory_gate.get("reasons", []),
            "top_neighbors": top_neighbors[:5] if top_neighbors else [],
            "signals": {
                "danger_similarity": round(danger_similarity, 3),
                "avoidance_signal": round(avoidance_signal, 3),
                "loss_prob": round(loss_prob, 3),
                "signed_bias": round(signed_bias, 3),
                "neural_risk_hint": round(neural_risk_hint, 3),
            },
            "intervention": {
                "type": intervention_type,
                "strength": round(intervention_strength, 3),
            } if intervention_type != "none" else None,
            "verdict": "veto" if memory_gate["veto"] else (
                "caution" if memory_gate["risk_multiplier"] < 0.7 else "proceed"
            ),
            "summary": self._generate_rationale_summary(memory_gate, memory_vote),
        }
        
        return memory_gate, memory_vote, memory_rationale
    
    def _generate_rationale_summary(
        self,
        gate: Dict[str, Any],
        vote: Dict[str, Any],
    ) -> str:
        """Generate human-readable summary for rationale."""
        parts: List[str] = []
        
        if gate["veto"]:
            parts.append("VETO: Trade blocked due to high risk")
        elif gate["risk_multiplier"] < 0.5:
            parts.append(f"CAUTION: Size reduced to {gate['risk_multiplier']:.0%}")
        else:
            parts.append("OK: Proceed with trade")
        
        if vote["signed_bias"] > 0.3:
            parts.append(f"Memory favors trade (+{vote['signed_bias']:.2f})")
        elif vote["signed_bias"] < -0.3:
            parts.append(f"Memory warns against trade ({vote['signed_bias']:.2f})")
        
        if gate.get("reasons"):
            raw_reasons = gate.get("reasons")
            reasons_list: List[Any]
            if isinstance(raw_reasons, dict):
                reasons_list = [raw_reasons]
            elif isinstance(raw_reasons, (list, tuple)):
                reasons_list = list(raw_reasons)
            else:
                reasons_list = [raw_reasons]

            formatted: List[str] = []
            for r in reasons_list[:2]:
                if isinstance(r, dict):
                    text = (
                        r.get("reason")
                        or r.get("description")
                        or r.get("message")
                        or r.get("text")
                        or str(r)
                    )
                else:
                    text = str(r)
                formatted.append(str(text))

            if formatted:
                parts.append(f"Reasons: {', '.join(formatted)}")
        
        return " | ".join(parts)

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
            # Unified metrics (for frontend API)
            "unified_metrics",
            "unified_memory_status",
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
            # Training progress (optional consolidated view)
            "training_progress",
            # NEW: Memory gate/vote/rationale keys per enhance.md
            "memory_gate",     # Veto/size control signal
            "memory_vote",     # Signed ensemble contribution
            "memory_rationale", # Evidence bundle for explainability
        ]:
            if key in result:
                updates.append((key, result[key]))

        # Avoid writing keys with canonical owners in other modules
        forbidden_keys = {
            'pattern_analysis',  # PlaybookClusterer owns this
        }
        for key, value in updates:
            if key in forbidden_keys:
                continue
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
                avg_time = np.mean(list(self._processing_times)[-10:])
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
                recent_times = list(self._component_performance[name])[-10:]
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
            # Typed baseline for mistake_memory (always available)
            "mistake_memory": {"recent": [], "stats": {"count": 0, "last_ts": None}},
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
            # NEW: Loss risk head fallback
            "loss_risk_head": {
                "loss_prob": 0.0,
                "uncertainty": 0.5,
                "calibration_ece": 0.0,
                "samples_seen": 0,
            },
            # NEW: Interventions fallback
            "interventions": {
                "intervention": None,
                "intervention_count": 0,
                "active_patterns": [],
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

        # Add composite gate/vote signals (required by contract)
        response["memory_gate"] = {
            "veto": False,
            "risk_multiplier": 1.0,
            "confidence": 0.5,
            "reasons": [f"fallback: {reason}"],
            "risk_score": 0.0,
            "danger_similarity": 0.0,
            "loss_prob": 0.0,
        }
        response["memory_vote"] = {
            "vote_value": 0.0,
            "signed_bias": 0.0,
            "confidence": 0.5,
            "expected_pnl": 0.0,
            "neural_risk_hint": 0.5,
            "max_attention": 0.5,
            "source": "memory_fallback",
        }
        response["memory_rationale"] = {
            "gate_reasons": [f"fallback: {reason}"],
            "top_neighbors": [],
            "signals": {
                "danger_similarity": 0.0,
                "avoidance_signal": 0.0,
                "loss_prob": 0.0,
                "signed_bias": 0.0,
                "neural_risk_hint": 0.5,
            },
            "intervention": None,
            "verdict": "proceed",
            "summary": f"Memory fallback: {reason}",
        }
        
        # Add neural_risk_hint as top-level key (required by contract)
        response["neural_risk_hint"] = 0.5

        # Add unified metrics/status to satisfy contract even on failure
        response["unified_metrics"] = {
            "total_memories": self.memory_store.size() if hasattr(self, "memory_store") else 0,
            "memory_utilization": self.memory_store.utilization() if hasattr(self, "memory_store") else 0.0,
            "components_active": 0,
            "processing_status": "fallback",
            "health_status": "degraded",
        }
        response["unified_memory_status"] = {
            "components_enabled": 0,
            "memory_size": self.memory_store.size() if hasattr(self, "memory_store") else 0,
            "status": "fallback",
            "health_status": "degraded",
            "total_memories": self.memory_store.size() if hasattr(self, "memory_store") else 0,
            "memory_utilization": self.memory_store.utilization() if hasattr(self, "memory_store") else 0.0,
        }

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
        ctx = self.error_pinpointer.analyze_error(error, "UnifiedMemory.process")
        self.logger.error(
            format_operator_message(
                "❌",
                "UNIFIED_MEMORY_ERROR",
                error=str(ctx),
                processing_time_ms=processing_time,
                context="unified_memory",
            )
        )

        # Record failure
        self._record_failure(error)

        # Debug logging
        if self.unified_config.debug:
            self.debug_logger.log_error("PROCESS_ERROR", error)

        fallback = self._create_fallback_response(f"Error: {str(error)}")
        fallback = self._ensure_required_outputs(fallback, reason=str(error))
        return fallback

    def _record_success(self, processing_time: float) -> None:
        """Record successful processing"""
        self._processing_times.append(processing_time)
        # maxlen=100 automatically removes old entries

        # Reset circuit breaker on success
        if self.circuit_breaker["state"] == "HALF_OPEN":
            self.circuit_breaker["failures"] = 0
            self.circuit_breaker["state"] = "CLOSED"

        # Update performance tracker
        self.performance_tracker.record_metric("UnifiedMemory", "process_cycle", processing_time, True)

    def _ensure_required_outputs(self, result: Dict[str, Any], reason: Optional[str]) -> Dict[str, Any]:
        """
        Guarantee contract-required outputs exist. If missing, populate safe defaults and
        emit an explicit error so issues are not silently ignored.
        """
        required_defaults: Dict[str, Any] = {
            "unified_metrics": {
                "total_memories": self.memory_store.size() if hasattr(self, "memory_store") else 0,
                "memory_utilization": self.memory_store.utilization() if hasattr(self, "memory_store") else 0.0,
                "components_active": 0,
                "processing_status": "fallback" if reason else "complete",
                "health_status": "degraded" if reason else getattr(self, "_health_status", "unknown"),
            },
            "unified_memory_status": {
                "components_enabled": 0,
                "memory_size": self.memory_store.size() if hasattr(self, "memory_store") else 0,
                "status": "fallback" if reason else "active",
                "health_status": "degraded" if reason else getattr(self, "_health_status", "unknown"),
                "total_memories": self.memory_store.size() if hasattr(self, "memory_store") else 0,
                "memory_utilization": self.memory_store.utilization() if hasattr(self, "memory_store") else 0.0,
            },
            "memory_gate": {
                "veto": False,
                "risk_multiplier": 1.0,
                "confidence": 0.5,
                "reasons": [f"autofill: {reason}"] if reason else [],
                "risk_score": 0.0,
                "danger_similarity": 0.0,
                "loss_prob": 0.0,
            },
            "memory_vote": {
                "vote_value": 0.0,
                "signed_bias": 0.0,
                "confidence": 0.5,
                "expected_pnl": 0.0,
                "neural_risk_hint": 0.5,
                "max_attention": 0.5,
                "source": "memory_autofill",
            },
            "memory_rationale": {
                "gate_reasons": [f"autofill: {reason}"] if reason else [],
                "top_neighbors": [],
                "signals": {
                    "danger_similarity": 0.0,
                    "avoidance_signal": 0.0,
                    "loss_prob": 0.0,
                    "signed_bias": 0.0,
                    "neural_risk_hint": 0.5,
                },
                "intervention": None,
                "verdict": "proceed",
                "summary": f"Memory autofill: {reason or 'missing required outputs'}",
            },
            "neural_risk_hint": 0.5,
        }

        missing: List[str] = []
        for key, default_val in required_defaults.items():
            if key not in result:
                result[key] = default_val
                missing.append(key)

        if missing:
            self.logger.error(
                format_operator_message(
                    "�!",
                    "UNIFIED_MEMORY_MISSING_OUTPUTS",
                    missing=missing,
                    reason=reason or "unspecified",
                )
            )
        return result

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
            # Component performance deques auto-trim with maxlen=100
            pass
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

    def on_episode_end(self, episode_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each episode to update episode counter and optionally process episode data.
        Should be called by the training loop or environment.
        """
        self._episode_count += 1
        
        if self.unified_config.debug:
            self.debug_logger.debug(
                f"Episode ended: episode={self._episode_count}",
                component="lifecycle",
                data={"episode_info": episode_info or {}}
            )

    def increment_episode(self) -> int:
        """Increment and return the new episode count."""
        self._episode_count += 1
        return self._episode_count


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
        """Stop monitoring thread and save memories"""
        self._monitoring_active = False
        
        # Save memory store on shutdown
        if hasattr(self, 'memory_store') and self.memory_store.size() > 0:
            self.save_memory_store()
        
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

    # =========================================================
    # Memory Persistence - Save/Load actual memory contents
    # =========================================================
    
    def get_memory_store_path(self) -> str:
        """Get path for memory store persistence."""
        import os
        state_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "state", "memory")
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(state_dir, "memory_store.pkl")
    
    def save_memory_store(self) -> bool:
        """
        Save the memory store to disk.
        
        This should be called periodically and on shutdown to persist learned patterns.
        """
        try:
            filepath = self.get_memory_store_path()
            success = self.memory_store.save(filepath)
            if success:
                self.logger.info(
                    format_operator_message(
                        "💾",
                        "MEMORY_SAVED",
                        details=f"Saved {self.memory_store.size()} memories to {filepath}",
                    )
                )
            return success
        except Exception as e:
            self.logger.error(f"Failed to save memory store: {e}")
            return False
    
    def load_memory_store(self) -> bool:
        """
        Load the memory store from disk.
        
        This should be called on startup to restore learned patterns.
        """
        import os
        try:
            filepath = self.get_memory_store_path()
            if not os.path.exists(filepath):
                self.logger.info(
                    format_operator_message(
                        "🔍",
                        "MEMORY_NOT_FOUND",
                        details=f"No saved memory at {filepath}, starting fresh",
                    )
                )
                return False
            
            success = self.memory_store.load(filepath)
            if success:
                self.logger.info(
                    format_operator_message(
                        "📂",
                        "MEMORY_LOADED",
                        details=f"Loaded {self.memory_store.size()} memories from {filepath}",
                    )
                )
            return success
        except Exception as e:
            self.logger.error(f"Failed to load memory store: {e}")
            return False
