# ─────────────────────────────────────────────────────────────
# File: modules/utils/info_bus.py
# [ROCKET-X] PRODUCTION-READY SmartInfoBus - Zero-Wiring Architecture (XL)
# MAXED OUT: Transactions, Middleware, Waiters, Bulk Ops, Throttling, Snapshots
# VERSION: 2.0 (Hardened based on architectural audit)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import sys
import time
import asyncio
import json
import pickle
import hashlib
import copy
import threading
import uuid
import psutil
import gzip
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Set, Callable, Tuple, TypedDict
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
import numpy as np
from typing import DefaultDict, Deque, cast


# Import core dependencies
from modules.utils.audit_utils import RotatingLogger, format_operator_message, AuditSystem

# Typed structures for quality metrics (module scope for reuse in annotations)
class QualityTrend(TypedDict):
    ts: float
    score: float

class QualityEntry(TypedDict, total=False):
    score: float
    issues: List[str]
    trends: List[QualityTrend]

# Public exports
__all__ = [
    "SmartInfoBus",
    "InfoBusManager",
    "InfoBusConfig",
    "DataVersion",
    "DataRequest",
    "create_info_bus",
    "validate_info_bus",
    "InfoBusExtractor",
    "InfoBusUpdater",
    "InfoBusQuality",
    # Legacy type alias exported for backward compatibility
    "InfoBus",
]

# Backward compatibility: some modules import `InfoBus` as a type symbol.
# The legacy InfoBus is a dict-shaped container that may include a
# reference to the XL SmartInfoBus under the `_smart_bus` key.
# Providing this alias maintains compatibility without changing callers.
InfoBus = Dict[str, Any]

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InfoBusConfig:
    """
    Military-grade configuration for SmartInfoBus with comprehensive validation.
    """
    # Core settings
    enabled: bool = True
    debug_mode: bool = True
    log_level: str = "DEBUG"
    max_cache_size: int = 20000
    cache_ttl_seconds: int = 3600

    # Performance settings
    max_parallel_operations: int = 50
    default_timeout_ms: int = 5000
    health_check_interval_ms: int = 25000
    metrics_retention_hours: int = 24
    background_thread_count: int = 3

    # Data management
    max_data_age_seconds: int = 600  # 10 minutes
    max_history_versions: int = 50   # Reduced from 2000 to prevent memory accumulation
    cleanup_interval_seconds: int = 45
    integrity_validation: bool = True
    auto_cleanup: bool = True
    compression_enabled: bool = False

    # Circuit breaker
    circuit_breaker_threshold: int = 3
    recovery_time_seconds: int = 60
    failure_escalation_enabled: bool = True
    emergency_mode_enabled: bool = True

    # Event system
    max_event_log_size: int = 120000
    event_replay_enabled: bool = True
    subscription_timeout_ms: int = 1000
    async_callback_support: bool = True

    # Security & audit
    validation_enabled: bool = True
    audit_enabled: bool = True
    encryption_enabled: bool = False
    access_control_enabled: bool = False

    # Quality & analytics
    quality_monitoring_enabled: bool = True
    predictive_analytics_enabled: bool = True
    anomaly_detection_enabled: bool = True
    performance_profiling_enabled: bool = True

    # Advanced features
    dependency_tracking_enabled: bool = True
    circular_dependency_detection: bool = True
    auto_dependency_resolution: bool = True
    smart_caching_enabled: bool = True

    # NEW: control switches
    read_only_mode: bool = False
    pause_support_enabled: bool = True
    enable_transactions: bool = True
    rate_limit_writes_per_sec: int = 0  # 0 = unlimited
    default_namespace: Optional[str] = None  # e.g. "core"

    # Operating mode - affects staleness checks
    live_mode: bool = False  # When False (training), staleness warnings are suppressed

    # [FIXED] New contract enforcement flags from audit
    enforce_single_writer: bool = True
    enforce_dependency_declaration: bool = True

    # Cross-process persistence (enables frontend to see training data)
    persistence_enabled: bool = True  # Default ON for frontend visibility
    persist_write_interval_seconds: float = 1.0
    persistence_file: str = "state/infobus_data.json"
    persist_keys: Optional[List[str]] = None  # If None, persist all keys
    # Keys that should NOT be loaded from persistence on startup (memory learning data)
    # These keys accumulate incorrectly across sessions if loaded
    no_load_keys: List[str] = None  # type: ignore  # Will be set in __post_init__

    def __post_init__(self):
        # Set default no_load_keys if not provided
        # These are memory-learning keys that accumulate incorrectly across sessions
        if self.no_load_keys is None:
            self.no_load_keys = [
                # Pattern/memory learning data - must start fresh each session
                "pattern_memory",
                "pattern_effectiveness", 
                "playbook_recall",
                "playbook_quality",
                "playbook_memory",
                "memory_analytics",
                "loss_prevention",
                "danger_zones",
                "mistake_memory",
                "mistake_avoidance",
                "intervention_recommendation",
                "loss_risk_assessment",
                "neural_memory",
                "memory_embedding",
                "memory_compression",
                # Trade history - each session should track its own trades
                "recent_trades",
                "trades",
                "trade_history",
            ]
        self._validate_config()

    def _validate_config(self):
        errors = []
        if self.default_timeout_ms <= 0 or self.default_timeout_ms > 60000:
            errors.append("default_timeout_ms must be between 1 and 60000 ms")
        if self.health_check_interval_ms <= 0 or self.health_check_interval_ms > 300000:
            errors.append("health_check_interval_ms must be between 1 and 300000 ms")
        if self.max_parallel_operations <= 0 or self.max_parallel_operations > 2000:
            errors.append("max_parallel_operations must be between 1 and 2000")
        if self.cache_ttl_seconds <= 0 or self.cache_ttl_seconds > 172800:
            errors.append("cache_ttl_seconds must be between 1 s and 2 days")
        if self.max_data_age_seconds <= 0 or self.max_data_age_seconds > 172800:
            errors.append("max_data_age_seconds must be between 1 s and 2 days")
        if self.max_history_versions <= 0 or self.max_history_versions > 20000:
            errors.append("max_history_versions must be between 1 and 20000")
        if self.circuit_breaker_threshold <= 0 or self.circuit_breaker_threshold > 50:
            errors.append("circuit_breaker_threshold must be between 1 and 50")
        if self.recovery_time_seconds <= 0 or self.recovery_time_seconds > 7200:
            errors.append("recovery_time_seconds must be between 1 s and 2 hours")
        if self.max_event_log_size <= 0 or self.max_event_log_size > 2000000:
            errors.append("max_event_log_size must be between 1 and 2,000,000")
        if self.background_thread_count <= 0 or self.background_thread_count > 32:
            errors.append("background_thread_count must be between 1 and 32")
        if self.rate_limit_writes_per_sec < 0 or self.rate_limit_writes_per_sec > 100000:
            errors.append("rate_limit_writes_per_sec must be between 0 and 100000")
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            errors.append("log_level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL")
        if errors:
            raise ValueError(f"InfoBusConfig validation failed: {errors}")

    def update(self, updates: Dict[str, Any]):
        old = {}
        for k, v in updates.items():
            if hasattr(self, k):
                old[k] = getattr(self, k)
                setattr(self, k, v)
        try:
            self._validate_config()
        except Exception:
            for k, v in old.items():
                setattr(self, k, v)
            raise

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# ═══════════════════════════════════════════════════════════════════
# DATA OBJECTS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DataVersion:
    value: Any
    timestamp: float
    source_module: str
    version: int
    confidence: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    thesis: Optional[str] = None
    processing_time_ms: float = 0.0
    validation_hash: str = field(default="")
    access_count: int = field(default=0)

    # Enhanced tracking
    creation_stack_trace: Optional[str] = field(default=None)
    last_access_time: float = field(default_factory=time.time)
    access_patterns: Dict[str, int] = field(default_factory=dict)
    quality_score: float = field(default=100.0)
    anomaly_flags: List[str] = field(default_factory=list)
    compression_ratio: float = field(default=1.0)

    def __post_init__(self):
        if not self.validation_hash:
            self._calculate_validation_hash()
        if not self.creation_stack_trace:
            self.creation_stack_trace = self._capture_stack_trace()
        self._assess_data_quality()


    def _capture_stack_trace(self) -> str:
        try:
            import traceback
            return "".join(traceback.format_stack()[-6:])
        except Exception:
            return "Stack trace unavailable"

    def _assess_data_quality(self):
        score = 100.0
        if self.confidence < 0.8:
            score -= (0.8 - self.confidence) * 50
        if self.processing_time_ms > 1000:
            score -= min(self.processing_time_ms / 100, 30)
        if not self.thesis and self.confidence > 0.5:
            score -= 10
        if self.value is None:
            score -= 50
        elif isinstance(self.value, (int, float)) and hasattr(np, "isnan") and np.isnan(self.value):
            score -= 40
        self.quality_score = max(0.0, score)
        if self.quality_score < 50:
            self.anomaly_flags.append("low_quality")
        if self.processing_time_ms > 5000:
            self.anomaly_flags.append("slow_processing")
        if self.confidence < 0.3:
            self.anomaly_flags.append("low_confidence")

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

# ─────────────────────────────────────────────────────────────
# In class DataVersion (modules/utils/info_bus.py)
# ─────────────────────────────────────────────────────────────
    def _compute_validation_hash(self) -> str:
        try:
            data_str = json.dumps({
                'value': str(self.value)[:1000],
                'timestamp': self.timestamp,
                'source_module': self.source_module,
                'version': self.version,
                'confidence': self.confidence
            }, sort_keys=True)
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        except Exception:
            h = f"{self.timestamp}{self.source_module}{self.version}{self.confidence}"
            return hashlib.md5(h.encode()).hexdigest()[:16]

    def _calculate_validation_hash(self):
        self.validation_hash = self._compute_validation_hash()

    def validate_integrity(self) -> bool:
        try:
            computed = self._compute_validation_hash()
            ok = (computed == self.validation_hash)
            if not ok:
                self.anomaly_flags.append("integrity_failure")
            return ok
        except Exception:
            self.anomaly_flags.append("validation_error")
            return False

    def increment_access(self, accessor_module: str = "unknown"):
        self.access_count += 1
        self.last_access_time = time.time()
        self.access_patterns[accessor_module] = self.access_patterns.get(accessor_module, 0) + 1

    def get_access_frequency(self) -> float:
        hours = self.age_seconds() / 3600
        return self.access_count / max(hours, 1e-9)

    def is_stale(self, max_age_seconds: float) -> bool:
        return self.age_seconds() > max_age_seconds

    def to_dict(self, include_value: bool = True) -> Dict[str, Any]:
        out = {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'source_module': self.source_module,
            'version': self.version,
            'confidence': self.confidence,
            'dependencies': self.dependencies,
            'thesis': self.thesis,
            'processing_time_ms': self.processing_time_ms,
            'validation_hash': self.validation_hash,
            'access_count': self.access_count,
            'last_access_time': self.last_access_time,
            'access_patterns': self.access_patterns,
            'quality_score': self.quality_score,
            'anomaly_flags': self.anomaly_flags,
            'age_seconds': self.age_seconds(),
            'access_frequency': self.get_access_frequency(),
            'compression_ratio': self.compression_ratio
        }
        if include_value:
            out['value'] = self.value
        return out

@dataclass
class DataRequest:
    requesting_module: str
    requested_key: str
    timestamp: float
    max_age_seconds: Optional[float] = None
    min_confidence: Optional[float] = None
    priority: int = 0
    callback: Optional[Callable] = None
    timeout_seconds: float = 60.0
    # Tracking
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retry_count: int = 0
    max_retries: int = 3
    escalation_threshold: float = 30.0
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.requesting_module:
            raise ValueError("requesting_module cannot be empty")
        if not self.requested_key:
            raise ValueError("requested_key cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.timeout_seconds

    def should_escalate(self) -> bool:
        return time.time() - self.timestamp > self.escalation_threshold

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def increment_retry(self):
        self.retry_count += 1

    def matches_data(self, data: DataVersion) -> bool:
        if self.max_age_seconds and data.age_seconds() > self.max_age_seconds:
            return False
        if self.min_confidence and data.confidence < self.min_confidence:
            return False
        return True

# ═══════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    successful_calls: int = 0
    total_calls: int = 0
    last_success_time: float = 0
    failure_rate: float = 0.0
    avg_failure_interval: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    failure_history: deque = field(default_factory=lambda: deque(maxlen=100))
    success_history: deque = field(default_factory=lambda: deque(maxlen=100))
    # Diagnostics
    last_error: str = ""
    last_error_time: float = 0.0
    last_open_reason: str | None = None  # why we moved to OPEN (threshold/rate/probe)

    def record_success(self):
        self.successful_calls += 1
        self.total_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.success_history.append(self.last_success_time)
        self._update_failure_rate()
        if self.state == "HALF_OPEN" and self.consecutive_successes >= 3:
            self.state = "CLOSED"
            self.failure_count = 0
            self.consecutive_failures = 0
            setattr(self, "_half_open_trials", 0)

    def trip(self):
        self.state = "OPEN"
        self.consecutive_successes = 0
        setattr(self, "_half_open_trials", 0)

    def record_failure(self):
        self.failure_count += 1
        self.total_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()
        self.failure_history.append(self.last_failure_time)
        self._update_failure_rate()
        self._update_failure_interval()

    def _update_failure_rate(self):
        if self.total_calls > 0:
            self.failure_rate = self.failure_count / self.total_calls
        recent_failures = min(self.failure_count, len(self.failure_history))
        recent_successes = min(self.successful_calls, len(self.success_history))
        denom = (recent_failures + recent_successes)
        if denom > 0:
            recent_rate = recent_failures / denom
            self.failure_rate = (self.failure_rate * 0.7) + (recent_rate * 0.3)

    def _update_failure_interval(self):
        if len(self.failure_history) >= 2:
            intervals = [self.failure_history[i] - self.failure_history[i-1] for i in range(1, len(self.failure_history))]
            if intervals:
                self.avg_failure_interval = sum(intervals) / len(intervals)

    def should_allow_request(self, recovery_time: float, failure_threshold: int = 5) -> bool:
        now = time.time()
        if self.state == "CLOSED":
            if self.consecutive_failures >= failure_threshold:
                self.last_open_reason = "consecutive_failures_threshold"
                self.trip()
                return False
            if (self.failure_rate > 0.5 and self.total_calls > 10):
                self.last_open_reason = "failure_rate_threshold"
                self.trip()
                return False
            return True
        if self.state == "OPEN":
            if now - self.last_failure_time > recovery_time:
                self.state = "HALF_OPEN"
                setattr(self, "_half_open_trials", 0)
                return True
            return False
        # HALF_OPEN
        if (now - self.last_failure_time > recovery_time * 2 and self.consecutive_failures == 0):
            self.state = "CLOSED"
            setattr(self, "_half_open_trials", 0)
            return True
        trials = getattr(self, "_half_open_trials", 0)
        if trials >= 3:
            # Hit probe limit without enough successes
            self.last_open_reason = self.last_open_reason or "half_open_probe_limit"
            return False
        setattr(self, "_half_open_trials", trials + 1)
        return True

    def get_health_score(self) -> float:
        if self.state == "OPEN":
            return 0.0
        if self.total_calls == 0:
            return 100.0
        success_rate = self.successful_calls / max(self.total_calls, 1)
        base = success_rate * 100
        if self.consecutive_failures > 0:
            base -= min(self.consecutive_failures * 5, 30)
        if self.consecutive_successes > 5:
            base = min(100.0, base + 5)
        return max(0.0, base)

    def predict_next_failure(self) -> Optional[float]:
        if self.avg_failure_interval > 0 and len(self.failure_history) >= 3:
            return self.last_failure_time + self.avg_failure_interval
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'successful_calls': self.successful_calls,
            'total_calls': self.total_calls,
            'failure_rate': self.failure_rate,
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'avg_failure_interval': self.avg_failure_interval,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'health_score': self.get_health_score(),
            'predicted_next_failure': self.predict_next_failure(),
            'last_error': (self.last_error[:200] if isinstance(self.last_error, str) else str(self.last_error)),
            'last_error_time': self.last_error_time,
            'open_reason': self.last_open_reason,
        }

# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS (XL)
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBus:
    """
    XL Information Bus — feature-complete:
      • Thread-safe set/get with integrity, TTL, LRU, history
      • Middleware hooks (pre/post set & get)
      • Transactions (context manager) + bulk ops
      • Waiters: wait_for(key) / wait_for_many()
      • Schema validators per-key
      • Read-only & Pause modes
      • Write rate-limiting per module
      • Snapshot import/export (+ gzip), sessions, metrics
      • Circuit breaker & module health telemetry
    """

    # ──────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _load_config_from_yaml() -> Optional[InfoBusConfig]:
        """Load InfoBusConfig from system_config.yaml if available."""
        try:
            import yaml as yaml_module
            config_paths = [
                "config/system_config.yaml",
                "../config/system_config.yaml",
                os.path.join(os.path.dirname(__file__), "../../config/system_config.yaml"),
            ]
            for config_path in config_paths:
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        system_config = yaml_module.safe_load(f)
                    if system_config and 'info_bus' in system_config:
                        bus_cfg = system_config['info_bus']
                        # Map YAML keys to InfoBusConfig fields
                        return InfoBusConfig(
                            persistence_enabled=bus_cfg.get('persistence_enabled', True),
                            persist_write_interval_seconds=float(bus_cfg.get('persist_write_interval_seconds', 0.5)),
                            persistence_file=bus_cfg.get('persistence_file', 'state/infobus_data.json'),
                            persist_keys=bus_cfg.get('persist_keys'),  # None means persist all
                        )
                    break
        except Exception:
            pass  # Fall back to defaults
        return None

    def __init__(self, config: Optional[InfoBusConfig] = None):
            # Load config from system_config.yaml if not provided
            if config is None:
                config = self._load_config_from_yaml()
            self.config = config or InfoBusConfig()

            # Core data store + history
            self._data_store: Dict[str, DataVersion] = {}
            self._data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_history_versions))
            self._data_timestamps: Dict[str, float] = {}

            # Cross-process persistence - use config value
            self._persistence_file = getattr(self.config, 'persistence_file', "state/infobus_data.json")
            self._persistence_lock = threading.Lock()

            # Locks
            self._access_lock = threading.RLock()
            self._write_lock = threading.Lock()
            self._registry_lock = threading.RLock()
            self._event_lock = threading.Lock()
            self._subscription_lock = threading.Lock()
            self._performance_lock = threading.Lock()
            self._circuit_breaker_lock = threading.Lock()
            self._request_lock = threading.Lock()

            # Events log
            self._event_log: deque = deque(maxlen=self.config.max_event_log_size)

            # Registries
            self._providers: Dict[str, Set[str]] = defaultdict(set)
            self._consumers: Dict[str, Set[str]] = defaultdict(set)
            self._module_graph: Dict[str, Set[str]] = defaultdict(set)

            # Perf stats (reduced sizes to prevent memory accumulation)
            self._access_patterns = defaultdict(lambda: defaultdict(int))
            self._latency_history = defaultdict(lambda: deque(maxlen=500))      # Reduced from 5000
            self._cache_hits = 0
            self._cache_misses = 0
            self._operation_timings = defaultdict(lambda: deque(maxlen=500))    # Reduced from 2000
            self._memory_usage_history = deque(maxlen=200)
            self._cpu_usage_history = deque(maxlen=200)
            self._predictive_metrics = {}

            # Subscriptions
            self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
            self._async_subscribers: Dict[str, List[Callable]] = defaultdict(list)

            # Circuit breaker
            self._circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
            self._module_disabled: Set[str] = set()

            # Requests & waiters (BOUNDED to prevent memory leaks)
            self._pending_requests: deque = deque(maxlen=1000)  # Was unbounded List
            self._request_history: deque = deque(maxlen=1000)   # Reduced from 10000
            self._waiters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # Was unbounded List

            # Quality / Validation
            self._validation_enabled = self.config.validation_enabled

            # Typed quality ledger to satisfy static analysis
            def _default_quality_entry() -> "QualityEntry":
                return {"score": 100.0, "issues": [], "trends": []}

            self._quality_metrics: DefaultDict[str, QualityEntry] = defaultdict(_default_quality_entry)
            self._anomaly_detector = None
            self._quality_lock = threading.Lock()

            # Emergency & control
            self._emergency_mode = False
            self._emergency_triggers = 0
            self._emergency_threshold = 5
            self._degraded_operations = set()
            self._paused = threading.Event()
            self._paused.clear()

            # Cache stats
            self._cache_stats = defaultdict(int)
            self._cache_access_times = defaultdict(float)
            self._cache_priorities = defaultdict(float)

            # Thread infra
            self._maintenance_running = True
            self._maintenance_threads = []
            self._cleanup_thread: Optional[threading.Thread] = None
            self._cleanup_shutdown = threading.Event()
            self._cleanup_interval = 60
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.background_thread_count, thread_name_prefix="InfoBus")

            # Async task management (BOUNDED to prevent accumulation)
            self._pending_tasks: Set["asyncio.Task[Any]"] = set()  # Track async tasks
            # [FIXED] Corrected type hint from asyncio.Future to concurrent.futures.Future
            self._pending_async_ops: deque = deque(maxlen=500)  # Was unbounded List
            self._shutdown_event = asyncio.Event()
            # [FIXED] Changed from asyncio.Lock to a thread-safe RLock
            self._async_lock = threading.RLock()

            # Logger / Audit
            self.logger = RotatingLogger(
                name="SmartInfoBus",
                log_dir="logs/infobus",
                max_lines=15000,
                operator_mode=True,
                info_bus_aware=True
            )
            self._audit_system = AuditSystem("SmartInfoBus") if self.config.audit_enabled else None

            # Middleware & validators
            self._pre_set_hooks: List[Callable[[str, Any, Dict[str, Any]], Any]] = []
            self._post_set_hooks: List[Callable[[str, DataVersion], None]] = []
            self._pre_get_hooks: List[Callable[[str, str, Dict[str, Any]], None]] = []
            self._post_get_hooks: List[Callable[[str, str, Any, Dict[str, Any]], None]] = []
            self._validators: Dict[str, Callable[[Any], bool]] = {}

            # Single-writer policy for critical canonical keys
            self._critical_single_writer_keys: Set[str] = {
                "market_regime", "training_metrics", "performance_metrics",
                "risk_data", "sequence_quality", "trade_vote",
                # Contended keys: enforce single-writer policy explicitly
                "mode_recommendations", "member_confidences", "expert_votes",
            }

            # Ownership registry and policies (lightweight, in-memory)
            self._owners: Dict[str, str] = {}
            self._policies: Dict[str, Dict[str, Any]] = defaultdict(dict)
            self._streams: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=1000))  # Reduced from 10000

            # Default stream-like feeds (multi-writer, append-only)
            # Note: keep 'vote' as a stream; 'expert_votes' is a canonical snapshot
            # owned by the coordinator and must remain a state key to avoid BUS MISS
            # in consumers that call get('expert_votes').

            def _single_writer_guard(key: str, value: Any, meta: Dict[str, Any]) -> Any:
                try:
                    base_key = key.split(":", 1)[1] if ":" in key else key
                    if base_key not in self._critical_single_writer_keys:
                        return None

                    # If the key is configured as a stream, do not enforce single-writer
                    pol = self._policies.get(key) or self._policies.get(base_key)
                    if pol and pol.get("mode") == "stream":
                        return None

                    writer = str(meta.get("module", "unknown"))
                    existing: Set[str] = set()
                    try:
                        existing |= set(self.get_providers(base_key))
                    except Exception: pass
                    try:
                        existing |= set(self.get_providers(key))
                    except Exception: pass

                    if existing and (writer not in existing or len(existing) > 1):
                        evt = {
                            "type": "duplicate_writer", "key": key, "base_key": base_key,
                            "attempting_module": writer, "existing_providers": sorted(list(existing)),
                            "policy": "single_writer",
                        }
                        self._log_event(evt)
                        if self.config.debug_mode:
                            self.logger.warning(
                                f"[BUS][SINGLE-WRITER] Duplicate write attempt for '{base_key}' by {writer}; existing={sorted(list(existing))}"
                            )
                except Exception as _hook_exc:
                    try:
                        self.logger.error(f"[HOOK] single-writer guard failed for {key}: {_hook_exc}")
                    except Exception: pass
                return None

            self.register_pre_set_hook(_single_writer_guard)

            # Rate-limiting
            self._rate_counters: DefaultDict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=1000))  # Reduced from 10000

            # Transactions
            self._tx_local = threading.local()

            # Initialize subsystems
            self._initialize_anomaly_detection() if self.config.anomaly_detection_enabled else self._seed_anomaly_default()
            self._start_background_services()
            self._init_dependency_tracing_state()
            self._start_cleanup_thread()
            self._initialization_time = time.time()
            self._initialize_system_monitoring()

            # Emit init
            self._emit('bus_initialized', {
                'timestamp': self._initialization_time,
                'config': self.config.to_dict(),
                'features': self._get_enabled_features(),
                'system_info': self._get_system_info()
            })
            self.logger.info(format_operator_message("[ROCKET-X]", "SMARTINFOBUS XL INITIALIZED",
                                                    details=f"Features: {', '.join(self._get_enabled_features())}",
                                                    context="startup"))

            # No implicit seeding; providers are responsible for publishing their own keys.

            # Load persisted data on startup
            self._load_persisted_data()

    # ──────────────────────────────────────────────────────────────
    # Cross-Process Persistence
    # ──────────────────────────────────────────────────────────────
    def _persist_data(self, key: str, value: Any) -> None:
        """Persist key-value data to file for cross-process sharing (opt-in, debounced)."""
        # Backward-compatible config gates (won't fail if attrs are missing)
        if not getattr(self.config, 'persistence_enabled', False):
            return
        
        # Check if this key should be persisted
        persist_keys = getattr(self.config, 'persist_keys', None)
        if persist_keys is not None and key not in persist_keys:
            return  # Skip keys not in the allowed list
            
        try:
            with self._persistence_lock:
                # Debounce writes
                if not hasattr(self, "_last_persist_write"):
                    self._last_persist_write = 0.0
                now = time.time()
                interval = float(getattr(self.config, 'persist_write_interval_seconds', 1.0))
                if now - self._last_persist_write < interval:
                    # Queue the key for next batch write
                    if not hasattr(self, "_pending_persist_keys"):
                        self._pending_persist_keys: Set[str] = set()
                    self._pending_persist_keys.add(key)
                    return

                # Ensure directory exists
                persist_file = getattr(self.config, 'persistence_file', self._persistence_file)
                persist_dir = os.path.dirname(persist_file)
                if persist_dir and not os.path.exists(persist_dir):
                    os.makedirs(persist_dir, exist_ok=True)
                
                # Read existing persisted data safely (fallbacks + repair)
                persisted_data: Dict[str, Any] = self._safe_read_json_file(persist_file)

                # Collect all pending keys + current key
                keys_to_persist = getattr(self, "_pending_persist_keys", set()) | {key}
                self._pending_persist_keys = set()  # Clear pending
                
                # Batch persist all queued keys
                for k in keys_to_persist:
                    if k in self._data_store:
                        stored_val = self._data_store[k].value
                        serializable_value = self._make_serializable(stored_val)
                        version = self._data_store[k].version
                        persisted_data[k] = {
                            'value': serializable_value,
                            'timestamp': now,
                            'version': version
                        }

                # Atomic write with backup to avoid partial/corrupt files
                self._atomic_write_json(persist_file, persisted_data)
                self._last_persist_write = now

        except Exception as e:
            self.logger.warning(f"[PERSISTENCE] Failed to persist key '{key}': {e}")


    def _load_persisted_data(self) -> None:
        """Load persisted data from file on startup.
        
        Note: Keys in config.no_load_keys are skipped to prevent accumulation
        of stale memory-learning data across sessions.
        """
        try:
            persist_file = getattr(self.config, 'persistence_file', self._persistence_file)
            if os.path.exists(persist_file):
                persisted_data = self._safe_read_json_file(persist_file)
                
                # Get keys that should NOT be loaded (memory learning data)
                no_load_keys = set(getattr(self.config, 'no_load_keys', []) or [])
                
                loaded_count = 0
                skipped_count = 0
                
                # Load persisted data into memory store if not already present
                for key, data in persisted_data.items():
                    # Skip memory-learning keys that accumulate incorrectly across sessions
                    if key in no_load_keys:
                        skipped_count += 1
                        continue
                        
                    if key not in self._data_store:
                        try:
                            # Create a minimal DataVersion for persisted data
                            data_version = DataVersion(
                                value=data['value'],
                                version=data.get('version', 1),
                                timestamp=data.get('timestamp', time.time()),
                                source_module='Persistence',
                                thesis='Loaded from cross-process persistence',
                                confidence=1.0
                            )
                            self._data_store[key] = data_version
                            self._data_timestamps[key] = data_version.timestamp
                            loaded_count += 1
                        except Exception as e:
                            self.logger.warning(f"[PERSISTENCE] Failed to load persisted key '{key}': {e}")
                
                self.logger.info(f"[PERSISTENCE] Loaded {loaded_count} keys, skipped {skipped_count} memory-learning keys (fresh session)")

        except Exception as e:
            self.logger.warning(f"[PERSISTENCE] Failed to load persisted data: {e}")

    def _get_persisted_value(self, key: str) -> Any:
        """Get value from persistent storage if not in memory."""
        try:
            persist_file = getattr(self.config, 'persistence_file', self._persistence_file)
            if os.path.exists(persist_file):
                persisted_data = self._safe_read_json_file(persist_file)
                if key in persisted_data:
                    return persisted_data[key]['value']
        except Exception as e:
            self.logger.warning(f"[PERSISTENCE] Failed to get persisted value for '{key}': {e}")
        return None

    def _make_serializable(self, value: Any) -> Any:
        """Convert value to JSON-serializable format."""
        import math
        if isinstance(value, (str, int, bool, type(None))):
            return value
        elif isinstance(value, float):
            # Handle inf and NaN which are not JSON-compliant
            if math.isnan(value) or math.isinf(value):
                return 0.0
            return value
        elif isinstance(value, (list, tuple)):
            return [self._make_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._make_serializable(v) for k, v in value.items()}
        elif hasattr(value, '__dict__'):
            return self._make_serializable(value.__dict__)
        elif isinstance(value, np.ndarray):
            # Handle inf/NaN in numpy arrays
            arr = value.copy()
            arr = np.where(np.isnan(arr), 0.0, arr)
            arr = np.where(np.isinf(arr), 0.0, arr)
            return arr.tolist()
        else:
            # For other types, convert to string representation
            return str(value)

    #
    # Persistence hardening helpers
    #
    def _safe_read_json_file(self, path: str) -> Dict[str, Any]:
        """Robust JSON file reader with fallback/repair.
        - Returns parsed dict, or {} on failure.
        - Tries main file, then .bak, then salvage truncated content.
        """
        # Fast path
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e1:
            # Try backup
            bak = f"{path}.bak"
            try:
                if os.path.exists(bak):
                    with open(bak, 'r') as fb:
                        data = json.load(fb)
                        self.logger.warning(f"[PERSISTENCE] Using backup for '{os.path.basename(path)}'")
                        return data
            except Exception:
                pass

            # Try salvage
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                repaired = self._salvage_truncated_json(content)
                if repaired is not None:
                    # Backup corrupt file (best effort) and write repaired
                    try:
                        shutil.copy2(path, f"{path}.corrupt")
                    except Exception:
                        pass
                    self._atomic_write_json(path, repaired)
                    return repaired
            except Exception:
                pass

            self.logger.warning(f"[PERSISTENCE] Failed to read JSON '{path}': {e1}")
            return {}

    def _salvage_truncated_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Attempt to salvage a truncated top-level JSON object by
        truncating at the last position where braces balance.
        Returns dict on success, else None.
        """
        try:
            return json.loads(content)
        except Exception:
            pass

        depth = 0
        in_str = False
        esc = False
        last_balanced = -1
        for i, ch in enumerate(content):
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth = max(0, depth - 1)
                    if depth == 0:
                        last_balanced = i
        if last_balanced >= 0:
            snippet = content[: last_balanced + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None

    def _atomic_write_json(self, path: str, data: Dict[str, Any]) -> None:
        """Write JSON atomically with .bak backup of previous file."""
        directory = os.path.dirname(os.path.abspath(path)) or '.'
        os.makedirs(directory, exist_ok=True)
        payload = json.dumps(data, indent=2)
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(path):
                try:
                    shutil.copy2(path, f"{path}.bak")
                except Exception:
                    pass
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise

    # ──────────────────────────────────────────────────────────────
    # Helper: namespaces, pause, read-only, throttling
    # ──────────────────────────────────────────────────────────────
    def _ns_key(self, key: str, namespace: Optional[str]) -> str:
        if not namespace:
            namespace = self.config.default_namespace
        return f"{namespace}:{key}" if namespace else key

    def pause(self, reason: str = "maintenance"):
        if not self.config.pause_support_enabled:
            return
        self._paused.set()
        self._emit("bus_paused", {"reason": reason, "timestamp": time.time()})
        self.logger.warning(f"[PAUSE] InfoBus paused: {reason}")

    def resume(self):
        self._paused.clear()
        self._emit("bus_resumed", {"timestamp": time.time()})
        self.logger.info("[RESUME] InfoBus resumed")

    def _enforce_read_only(self):
        if self.config.read_only_mode:
            raise RuntimeError("SmartInfoBus is in read-only mode")

    def _enforce_rate_limit(self, module: str):
        limit = self.config.rate_limit_writes_per_sec
        if limit <= 0:
            return

        now = time.time()

        # Ensure the per-module bucket is a deque (defensive in case anything overwrote it)
        dq = self._rate_counters.get(module)
        if not isinstance(dq, deque):
            dq = deque(maxlen=10000)
            self._rate_counters[module] = dq  # type: ignore[assignment]

        dq.append(now)

        # drop entries older than 1s
        while len(dq) and (now - dq[0]) > 1.0:
            dq.popleft()

        if len(dq) > limit:
            raise RuntimeError(f"Write rate exceeded for module '{module}' ({limit}/sec)")


    # ──────────────────────────────────────────────────────────────
    # Middleware & Validators
    # ──────────────────────────────────────────────────────────────
    def register_pre_set_hook(self, fn: Callable[[str, Any, Dict[str, Any]], Any]):
        self._pre_set_hooks.append(fn)

    def register_post_set_hook(self, fn: Callable[[str, DataVersion], None]):
        self._post_set_hooks.append(fn)

    def register_pre_get_hook(self, fn: Callable[[str, str, Dict[str, Any]], None]):
        self._pre_get_hooks.append(fn)

    def register_post_get_hook(self, fn: Callable[[str, str, Any, Dict[str, Any]], None]):
        self._post_get_hooks.append(fn)

    def register_validator(self, key: str, fn: Callable[[Any], bool]):
        """Register a schema/shape validator for a key."""
        self._validators[key] = fn

    def _apply_pre_set(self, key: str, value: Any, meta: Dict[str, Any]) -> Any:
        for fn in list(self._pre_set_hooks):
            try:
                maybe = fn(key, value, meta)
                if maybe is not None:
                    value = maybe
            except Exception as e:
                self.logger.error(f"[HOOK] pre_set error for {key}: {e}")
                # [FIXED] Re-raise contract violations to enforce them
                if isinstance(e, PermissionError):
                    raise
        return value

    def _apply_post_set(self, key: str, dv: DataVersion):
        for fn in list(self._post_set_hooks):
            try:
                fn(key, dv)
            except Exception as e:
                self.logger.error(f"[HOOK] post_set error for {key}: {e}")

    def _apply_pre_get(self, key: str, module: str, meta: Dict[str, Any]):
        for fn in list(self._pre_get_hooks):
            try:
                fn(key, module, meta)
            except Exception as e:
                self.logger.error(f"[HOOK] pre_get error for {key}: {e}")

    def _apply_post_get(self, key: str, module: str, value: Any, meta: Dict[str, Any]):
        for fn in list(self._post_get_hooks):
            try:
                fn(key, module, value, meta)
            except Exception as e:
                self.logger.error(f"[HOOK] post_get error for {key}: {e}")

    # ──────────────────────────────────────────────────────────────
    # System info & initialization helpers
    # ──────────────────────────────────────────────────────────────
    def _get_enabled_features(self) -> List[str]:
        feats = ["core", "thread_safety", "performance_monitoring", "history", "events"]
        if self.config.audit_enabled:
            feats.append("audit")
        if self.config.anomaly_detection_enabled:
            feats.append("anomaly_detection")
        if self.config.predictive_analytics_enabled:
            feats.append("predictive_analytics")
        if self.config.quality_monitoring_enabled:
            feats.append("quality_monitoring")
        if self.config.event_replay_enabled:
            feats.append("event_replay")
        if self.config.smart_caching_enabled:
            feats.append("smart_caching")
        if self.config.dependency_tracking_enabled:
            feats.append("dependency_tracking")
        if self.config.emergency_mode_enabled:
            feats.append("emergency_mode")
        feats.extend(["transactions", "waiters", "validators", "pause", "throttling"])
        return feats

    def _get_system_info(self) -> Dict[str, Any]:
        try:
            return {
                'python_version': sys.version.split()[0],
                'platform': sys.platform,
                'cpu_count': os.cpu_count(),
                'memory_mb': psutil.virtual_memory().total // (1024 * 1024),
                'thread_count': threading.active_count(),
                'process_id': os.getpid()
            }
        except Exception:
            return {'error': 'System info unavailable'}

    def _initialize_anomaly_detection(self):
        try:
            self._anomaly_detector = {
                'data_access_patterns': defaultdict(list),
                'performance_baselines': defaultdict(list),
                'quality_baselines': defaultdict(list),
                'alert_thresholds': {
                    'access_frequency_multiplier': 3.0,
                    'performance_degradation_multiplier': 2.0,
                    'quality_score_threshold': 50.0
                }
            }
            self.logger.info("[ANOM] Anomaly detection initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize anomaly detection: {e}")

    def _seed_anomaly_default(self):
        try:
            self._anomaly_detector = {'anomaly_score': 1.0}
        except Exception:
            pass

    def _start_background_services(self):
        try:
            t1 = threading.Thread(target=self._background_maintenance, daemon=True, name="InfoBus-Maintenance")
            t1.start()
            self._maintenance_threads.append(t1)
            t2 = threading.Thread(target=self._background_performance_monitoring, daemon=True, name="InfoBus-Performance")
            t2.start()
            self._maintenance_threads.append(t2)
            if self.config.quality_monitoring_enabled:
                t3 = threading.Thread(target=self._background_quality_monitoring, daemon=True, name="InfoBus-Quality")
                t3.start()
                self._maintenance_threads.append(t3)
            self.logger.info(f"[OK] Background services: {len(self._maintenance_threads)}")
        except Exception as e:
            self.logger.error(f"Failed to start background services: {e}")

    def _initialize_system_monitoring(self):
        try:
            self._record_system_metrics()
            if self.config.predictive_analytics_enabled:
                self._predictive_metrics = {
                    'baseline_response_time': 10.0,
                    'baseline_throughput': 1000.0,
                    'baseline_memory_usage': 50.0,
                    'trend_window_size': 100,
                    'prediction_confidence': 0.8
                }
        except Exception as e:
            self.logger.error(f"Failed to initialize system monitoring: {e}")


    # ──────────────────────────────────────────────────────────────
    # Background workers (performance & quality)
    # ──────────────────────────────────────────────────────────────
    def _background_performance_monitoring(self) -> None:
        """
        Periodically record host metrics and derive simple performance signals.
        Emits soft alerts for degraded cache hit-rate or latency spikes.
        """
        self.logger.info("[TOOL] Background performance monitor started")
        interval = max(1.0, float(self.config.health_check_interval_ms) / 1000.0)

        while self._maintenance_running:
            try:
                # Host metrics (CPU/mem)
                self._record_system_metrics()

                # Hit rate & latency signals
                with self._performance_lock:
                    total = self._cache_hits + self._cache_misses
                    hit_rate = self._cache_hits / max(total, 1)
                    self._predictive_metrics["hit_rate"] = hit_rate

                    # Compute rolling p95 latency per module (cheap heuristic)
                    hot_modules: Dict[str, float] = {}
                    for m, timings in self._latency_history.items():
                        if timings:
                            arr = list(timings)
                            p95 = float(np.percentile(arr, 95) if len(arr) >= 10 else max(arr))
                            hot_modules[m] = p95
                    self._predictive_metrics["p95_by_module"] = hot_modules

                # Emit warnings on clear degradation
                if total > 200 and hit_rate < 0.20:
                    self._emit(
                        "performance_alert",
                        {
                            "kind": "low_cache_hit_rate",
                            "hit_rate": hit_rate,
                            "total_requests": int(total),
                            "timestamp": time.time(),
                        },
                    )

                # Very high module p95s
                slow = [(m, p95) for m, p95 in self._predictive_metrics.get("p95_by_module", {}).items() if p95 > 500]
                if slow:
                    worst = max(slow, key=lambda x: x[1])
                    self._emit(
                        "performance_alert",
                        {
                            "kind": "high_latency",
                            "module": worst[0],
                            "p95_ms": worst[1],
                            "timestamp": time.time(),
                        },
                    )

            except Exception as e:
                self.logger.debug(f"[perf-monitor] loop error: {e}")

            # pacing
            time.sleep(interval)

    def _background_quality_monitoring(self) -> None:
        """
        Periodically sweep data quality. Tracks per-key quality trends and emits warnings
        for low quality, low confidence, or excessive staleness.
        """
        self.logger.info("[TOOL] Background quality monitor started")
        # run a bit more often than cleanup; but at least 2s
        interval = max(2.0, float(self.config.cleanup_interval_seconds) / 2.0)

        while self._maintenance_running and self.config.quality_monitoring_enabled:
            try:
                issues_found = 0
                low_quality_keys: List[str] = []
                stale_keys: List[str] = []
                low_conf_keys: List[str] = []

                with self._access_lock:
                    snapshot_items = list(self._data_store.items())

                now = time.time()
                for key, dv in snapshot_items:
                    # Update local quality ledger
                    try:
                        with self._quality_lock:
                            q = self._quality_metrics[key]
                            # Keep score as float consistently
                            q['score'] = float(dv.quality_score)
                            # Work on a typed local list to avoid int|list unions
                            trends = cast(List[QualityTrend], q.get('trends', []))
                            item: QualityTrend = {'ts': now, 'score': float(dv.quality_score)}
                            trends.append(item)
                            if len(trends) > 200:
                                trends.pop(0)
                            q['trends'] = trends  # write back
                    except Exception:
                        pass

                    # Collect issues
                    if dv.quality_score < 50.0:
                        low_quality_keys.append(key)
                        issues_found += 1
                    if dv.confidence < 0.3:
                        low_conf_keys.append(key)
                        issues_found += 1
                    if dv.age_seconds() > self.config.max_data_age_seconds * 2:
                        stale_keys.append(key)
                        issues_found += 1

                if issues_found:
                    self._emit(
                        "quality_warning",
                        {
                            "low_quality": low_quality_keys[:25],
                            "low_confidence": low_conf_keys[:25],
                            "stale": stale_keys[:25],
                            "totals": {
                                "low_quality": len(low_quality_keys),
                                "low_confidence": len(low_conf_keys),
                                "stale": len(stale_keys),
                            },
                            "timestamp": now,
                        },
                    )

            except Exception as e:
                self.logger.debug(f"[quality-monitor] loop error: {e}")

            time.sleep(interval)


    def _record_system_metrics(self):
        try:
            mem = psutil.virtual_memory()
            self._memory_usage_history.append({'ts': time.time(), 'percent': mem.percent, 'available_mb': mem.available // (1024 * 1024)})
            cpu = psutil.cpu_percent(interval=0.1)
            self._cpu_usage_history.append({'ts': time.time(), 'percent': cpu})
        except Exception as e:
            self.logger.debug(f"Failed to record system metrics: {e}")

    # ──────────────────────────────────────────────────────────────
    # Cleanup / TTL / LRU
    # ──────────────────────────────────────────────────────────────
    def _start_cleanup_thread(self):
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        self._cleanup_shutdown.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, name="InfoBus-MemoryCleanup", daemon=True)
        self._cleanup_thread.start()
        self.logger.info("🧹 Memory cleanup thread started")

    def _cleanup_worker(self):
        while not self._cleanup_shutdown.wait(self._cleanup_interval):
            try:
                self._cleanup_expired_and_lru()
            except Exception as e:
                self.logger.error(f"Memory cleanup error: {e}")

    def _cleanup_expired_and_lru(self):
        with self._write_lock:
            expired = [k for k, v in self._data_store.items() if v.age_seconds() > self.config.cache_ttl_seconds]
            for k in expired:
                self._data_store.pop(k, None)
                self._data_timestamps.pop(k, None)
            removed = 0
            if len(self._data_store) > self.config.max_cache_size:
                oldest = sorted(self._data_store.items(), key=lambda kv: kv[1].last_access_time)
                to_remove = len(self._data_store) - self.config.max_cache_size
                for i in range(to_remove):
                    key = oldest[i][0]
                    self._data_store.pop(key, None)
                    self._data_timestamps.pop(key, None)
                    removed += 1
        if expired or removed:
            self.logger.debug(f"🗑️ Cleaned: {len(expired)} expired, {removed} LRU")

    # ──────────────────────────────────────────────────────────────
    # Core operations – set/get (+ bulk) with transactions & validators
    # ──────────────────────────────────────────────────────────────
    def set(self, key: str, value: Any, module: str, thesis: str | None = None,
            confidence: float = 1.0, dependencies: List[str] | None = None,
            processing_time_ms: float = 0.0, *, namespace: Optional[str] = None) -> None:
        """
        Store a value with versioning, TTL, safety, middleware, validators, and optional namespace.
        Transaction-aware: if inside a transaction, it's queued until commit.
        """
        self._enforce_read_only()
        if self._paused.is_set():
            raise RuntimeError("SmartInfoBus is paused")

        if not key or not isinstance(key, str):
            raise ValueError("Key must be a non-empty string")
        if not module or not isinstance(module, str):
            raise ValueError("Module must be a non-empty string")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

        full_key = self._ns_key(key, namespace)
        self._enforce_rate_limit(module)

        meta = {
            "module": module, "thesis": thesis, "confidence": confidence,
            "dependencies": dependencies, "processing_time_ms": processing_time_ms,
            "namespace": namespace
        }
        # Prevent Environment from writing canonical provider-owned keys to avoid owner violations
        try:
            base_key = full_key.split(":", 1)[-1]
            if module == "Environment" and base_key in {"market_data", "market_context", "step_idx", "environment_config"}:
                return
        except Exception:
            pass

        value = self._apply_pre_set(full_key, value, meta)

        # Ownership guard: block non-owner writes to canonical keys (soft-fail)
        owner = self._owners.get(full_key) or self._owners.get(key)
        if owner and owner != module:
            self._log_event({"type": "owner_violation", "key": full_key, "expected_owner": owner, "writer": module})
            try:
                self.logger.warning(f"[BUS][OWNER] Blocked write: {module} -> {full_key}; owner is {owner}")
            except Exception:
                pass
            return

        # Stream policy: append-only (no provider table churn)
        pol = self._policies.get(full_key) or self._policies.get(key)
        if pol and pol.get("mode") == "stream":
            self._streams[full_key].append({"t": time.time(), "module": module, "value": self._safe_clone(value)})
            return

        # Validators (schema/shape)
        validator = self._validators.get(full_key) or self._validators.get(key)
        if validator:
            try:
                ok = bool(validator(value))
                if not ok:
                    raise ValueError(f"Validation failed for '{full_key}'")
            except Exception as e:
                raise ValueError(f"Validator error for '{full_key}': {e}")

        # Transaction-aware
        if getattr(self._tx_local, "buffer", None) is not None:
            self._tx_local.buffer.append(("set", (full_key, value, module, thesis, confidence, dependencies, processing_time_ms)))
            return

        self._set_core(full_key, value, module, thesis, confidence, dependencies, processing_time_ms)

    # Ownership & policy helpers
    def declare_owner(self, key: str, owner: str) -> None:
        if not key or not owner:
            raise ValueError("key and owner must be non-empty")
        self._owners[key] = owner

    def set_policy(self, key: str, *, mode: str = "state", **kwargs) -> None:
        if mode not in ("state", "stream"):
            raise ValueError("Unsupported policy mode")
        self._policies[key] = {"mode": mode, **kwargs}

    def publish(self, key: str, value: Any, module: str, thesis: Optional[str] = None) -> None:
        pol = self._policies.get(key)
        if pol and pol.get("mode") == "stream":
            full_key = self._ns_key(key, None)
            try:
                stored_value = self._safe_clone(value)
            except Exception:
                stored_value = value
            self._streams[full_key].append({"t": time.time(), "module": module, "value": stored_value})
            return
        # fallback to standard set if not a stream key
        self.set(key, value, module, thesis)


    # Core application of a set (factored for transactions)
    def _set_core(self, full_key: str, value: Any, module: str, thesis: Optional[str], confidence: float,
                dependencies: Optional[List[str]], processing_time_ms: float) -> None:
        try:
            with self._write_lock:
                prev = self._data_store.get(full_key)
                version = prev.version + 1 if prev else 1

                try:
                    stored_value = self._safe_clone(value)
                except Exception:
                    stored_value = value

                data = DataVersion(
                    value=stored_value, timestamp=time.time(), source_module=module, version=version,
                    confidence=confidence, thesis=thesis, dependencies=dependencies or [],
                    processing_time_ms=max(0.0, processing_time_ms),
                )

                if self._validation_enabled and not data.validate_integrity():
                    raise RuntimeError(f"Integrity check failed for '{full_key}'")

                self._data_store[full_key] = data
                self._data_history[full_key].append(data)
                self._data_timestamps[full_key] = data.timestamp

                if len(self._data_store) > self.config.max_cache_size:
                    self._cleanup_expired_and_lru()

            # Register provider only for non-stream keys (prevents "provider flip" storms)
            with self._registry_lock:
                base_key = full_key.split(":", 1)[-1]
                pol = self._policies.get(full_key) or self._policies.get(base_key)
                if not (pol and pol.get("mode") == "stream"):
                    self._providers[full_key].add(module)

                if dependencies:
                    for dep in dependencies:
                        provider = self._get_primary_provider(dep)
                        if provider and provider != module:
                            self._module_graph[module].add(provider)

            if self.config.dependency_tracking_enabled:
                try:
                    self._set_log.append((self._now_iso(), module, full_key, version))
                except Exception:
                    pass

            if self.config.debug_mode and getattr(self, "_verbose_io", False):
                self.logger.info(f"[BUS][SET] {module} → '{full_key}' v{version} (conf={confidence:.2f}) {self._preview(value)}")

            with self._performance_lock:
                self._access_patterns[module][f"write:{full_key}"] += 1

            self._log_event({
                "type": "set", "key": full_key, "module": module,
                "timestamp": data.timestamp, "version": version,
                "has_thesis": thesis is not None, "confidence": confidence
            })
            self._emit("data_updated", {
                "key": full_key, "module": module, "version": version,
                "confidence": confidence, "has_thesis": thesis is not None
            })

            # Notify waiters
            self._notify_waiters(full_key, data)

            # Post-set hooks + pending requests
            self._apply_post_set(full_key, data)
            self._check_pending_requests(full_key)

            # Persist data for cross-process sharing
            self._persist_data(full_key, stored_value)

            self.logger.debug(f"[OK] {module} set '{full_key}' v{version} (conf={confidence:0.2f})")

        except Exception as exc:
            self.logger.error(f"[CRASH] Failed to set {full_key}: {exc}")
            self.record_module_failure(module, f"set failed: {exc}")
            raise


    def set_many(self, entries: List[Dict[str, Any]], *, atomic: bool = False, namespace: Optional[str] = None) -> int:
        """
        Bulk set; entries is a list of dicts with at minimum {key, value, module}.
        If atomic=True, performed within a transaction.
        """
        if atomic:
            with self.transaction():
                for e in entries:
                    self.set(e["key"], e["value"], e["module"],
                             thesis=e.get("thesis"), confidence=e.get("confidence", 1.0),
                             dependencies=e.get("dependencies"), processing_time_ms=e.get("processing_time_ms", 0.0),
                             namespace=e.get("namespace", namespace))
            return len(entries)
        else:
            count = 0
            for e in entries:
                self.set(e["key"], e["value"], e["module"],
                         thesis=e.get("thesis"), confidence=e.get("confidence", 1.0),
                         dependencies=e.get("dependencies"), processing_time_ms=e.get("processing_time_ms", 0.0),
                         namespace=e.get("namespace", namespace))
                count += 1
            return count

    def get(self, key: str, module: str, max_age: Optional[float] = None,
            min_confidence: float = 0.0, default: Any = None, *, namespace: Optional[str] = None,
            declared_dependencies: Optional[Set[str]] = None) -> Any:
        """
        Get value with freshness/confidence validation, middleware hooks, and dependency enforcement.
        """
        try:
            full_key = self._ns_key(key, namespace)

            # Enforce dependency declaration (contract)
            if self.config.enforce_dependency_declaration and declared_dependencies is not None:
                if full_key not in declared_dependencies and key not in declared_dependencies:
                    raise PermissionError(
                        f"[BUS-CONTRACT] Module '{module}' tried to access undeclared dependency '{key}'."
                    )

            self._apply_pre_get(full_key, module, {"max_age": max_age, "min_confidence": min_confidence})

            with self._access_lock:
                with self._registry_lock:
                    self._consumers[full_key].add(module)
                with self._performance_lock:
                    self._access_patterns[module][f'read:{full_key}'] += 1

                data = self._data_store.get(full_key)
                if not data:
                    # Check for persisted data from other processes
                    persisted_value = self._get_persisted_value(full_key)
                    if persisted_value is not None:
                        # Create a temporary DataVersion for the persisted data
                        data = DataVersion(
                            value=persisted_value,
                            timestamp=time.time(),
                            source_module='Persistence',
                            thesis='Cross-process data',
                            confidence=1.0,
                            version=1
                        )
                        # Don't store in memory to avoid conflicts, just return the value
                        self._apply_post_get(full_key, module, persisted_value, {"from_persistence": True})
                        return persisted_value

                    with self._performance_lock:
                        self._cache_misses += 1
                    self._log_miss(full_key, module)
                    self._apply_post_get(full_key, module, default, {"miss": True})
                    return default

                if self._validation_enabled and not data.validate_integrity():
                    self.logger.error(f"Data integrity check failed for {full_key}")
                    try:
                        self._emit_get_event(full_key, module, data, reason="integrity")
                    except Exception:
                        pass
                    self._apply_post_get(full_key, module, default, {"blocked": "integrity"})
                    return default

                age_seconds = data.age_seconds()
                # Defensive: callers sometimes pass default as positional 3rd arg, which binds to max_age.
                # Ensure max_age_check is numeric; otherwise fall back to configured max age and warn once.
                if isinstance(max_age, (int, float)):
                    max_age_check = float(max_age)
                else:
                    max_age_check = float(self.config.max_data_age_seconds)
                    # In training mode, use a much higher threshold (or disable)
                    if not getattr(self.config, 'live_mode', False):
                        max_age_check = max(max_age_check, 7200.0)  # 2 hours for training
                    if max_age is not None and not isinstance(max_age, (int, float)):
                        try:
                            self.logger.warning(
                                f"[BUS][GET] Non-numeric max_age for key '{full_key}' from {module};"
                                f" using default {max_age_check}s (got type {type(max_age).__name__})"
                            )
                        except Exception:
                            pass
                if age_seconds > max_age_check:
                    self._emit('stale_data_warning', {'key': full_key, 'age': age_seconds, 'module': module, 'max_age': max_age_check})
                    self.logger.warning(f"Stale data: {full_key} is {age_seconds:.1f}s old (max: {max_age_check}s)")
                    try:
                        self._emit_get_event(full_key, module, data, reason="stale")
                    except Exception:
                        pass
                    self._apply_post_get(full_key, module, default, {"blocked": "stale"})
                    return default

                if data.confidence < min_confidence:
                    self.logger.warning(f"Low confidence: {full_key} has {data.confidence:.2f} (min: {min_confidence:.2f})")
                    try:
                        self._emit_get_event(full_key, module, data, reason="low_confidence")
                    except Exception:
                        pass
                    self._apply_post_get(full_key, module, default, {"blocked": "low_confidence"})
                    return default

                data.increment_access(accessor_module=module)
                with self._performance_lock:
                    self._cache_hits += 1

                try:
                    self._emit_get_event(full_key, module, data, reason=None)
                except Exception:
                    pass

                out = self._safe_clone(data.value)

                self._apply_post_get(full_key, module, out, {"ok": True})
                return out

        except Exception as e:
            self.logger.error(f"[CRASH] Failed to get {key} for {module}: {e}")
            self.record_module_failure(module, f"Data get failed: {str(e)}")
            if isinstance(e, PermissionError):
                raise
            return default

            
    # [NEW] Added for performance-critical paths where cloning can be skipped.
    def get_readonly_ref(self, key: str, module: str, default: Any = None, *, namespace: Optional[str] = None) -> Any:
        """
        Get a direct, read-only reference to a value without cloning it.

        WARNING: This is a high-performance, unsafe operation. The caller MUST NOT
        mutate the returned object, as it is a direct reference to the cached data.
        Mutating the returned object will corrupt the InfoBus state.
        """
        full_key = self._ns_key(key, namespace)
        with self._access_lock:
            data = self._data_store.get(full_key)
            if not data:
                return default
            # Perform standard checks (age, confidence, integrity)
            if data.is_stale(self.config.max_data_age_seconds) or not data.validate_integrity():
                return default
            data.increment_access(accessor_module=f"{module}_readonly")
            return data.value

    def get_many(self, keys: List[str], module: str, *, namespace: Optional[str] = None,
                 max_age: Optional[float] = None, min_confidence: float = 0.0, default: Any = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in keys:
            out[k] = self.get(k, module, max_age=max_age, min_confidence=min_confidence, default=default, namespace=namespace)
        return out

    # ──────────────────────────────────────────────────────────────
    # Waiters (sync wait on values becoming available/matching a predicate)
    # ──────────────────────────────────────────────────────────────
    def wait_for(
        self,
        key: str,
        *,
        timeout: float = 10.0,
        predicate: Optional[Callable[[Any], bool]] = None,
        namespace: Optional[str] = None
    ) -> Optional[Any]:
        """
        Block the caller until key appears (and predicate(value) is True if provided) or timeout.
        """
        full_key = self._ns_key(key, namespace)

        # Fast-path with explicit None-guard so Pylance sees the narrow
        existing = self.get_with_metadata(full_key, "WaiterBootstrap")
        if existing is not None and (predicate is None or predicate(existing.value)):
            return existing.value

        event = threading.Event()
        waiter_info: Dict[str, Any] = {"result": None, "matched": False}
        self._waiters[full_key].append((event, predicate, waiter_info))
        signaled = event.wait(timeout)
        try:
            self._waiters[full_key].remove((event, predicate, waiter_info))
        except Exception:
            pass
        if not signaled or not waiter_info.get("matched"):
            return None
        return waiter_info.get("result")


    def wait_for_many(self, keys: List[str], *, timeout: float = 10.0,
                      namespace: Optional[str] = None) -> Dict[str, Optional[Any]]:
        deadline = time.time() + timeout
        out: Dict[str, Optional[Any]] = {}
        for k in keys:
            remaining = max(0.0, deadline - time.time())
            out[k] = self.wait_for(k, timeout=remaining, namespace=namespace)
        return out

    def _notify_waiters(self, full_key: str, data: DataVersion):
        items = list(self._waiters.get(full_key, []))
        if not items:
            return
        for event, predicate, info in items:
            try:
                if predicate and not predicate(data.value):
                    continue
                info["result"] = data.value
                info["matched"] = True
                event.set()
            except Exception:
                event.set()

    # ──────────────────────────────────────────────────────────────
    # Metadata & requests
    # ──────────────────────────────────────────────────────────────
    def get_with_metadata(self, key: str, module: str) -> Optional[DataVersion]:
        try:
            with self._access_lock:
                with self._registry_lock:
                    self._consumers[key].add(module)
                with self._performance_lock:
                    self._access_patterns[module][f'metadata:{key}'] += 1
                data = self._data_store.get(key)
                if data:
                    if self._validation_enabled and not data.validate_integrity():
                        self.logger.error(f"Data integrity check failed for {key}")
                        return None
                    data.increment_access(accessor_module=module)
                    with self._performance_lock:
                        self._cache_hits += 1
                else:
                    with self._performance_lock:
                        self._cache_misses += 1
                return data
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to get metadata for {key}: {e}")
            return None

    def get_with_thesis(self, key: str, module: str) -> Optional[Tuple[Any, str]]:
        dv = self.get_with_metadata(key, module)
        if not dv:
            return None
        return dv.value, (dv.thesis or "No explanation provided")

    def request_data(self, key: str, module: str, max_age: Optional[float] = None,
                     min_confidence: Optional[float] = None, priority: int = 0,
                     callback: Optional[Callable] = None, timeout_seconds: float = 60.0):
        try:
            req = DataRequest(
                requesting_module=module, requested_key=key, timestamp=time.time(),
                max_age_seconds=max_age, min_confidence=min_confidence,
                priority=priority, callback=callback, timeout_seconds=timeout_seconds
            )
            with self._request_lock:
                inserted = False
                for i, ex in enumerate(self._pending_requests):
                    if req.priority > ex.priority:
                        self._pending_requests.insert(i, req)
                        inserted = True
                        break
                if not inserted:
                    self._pending_requests.append(req)
            with self._registry_lock:
                self._consumers[key].add(module)
            self.logger.debug(f"📋 {module} requested '{key}' (priority: {priority})")
            return req.request_id
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to create data request: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # Registry / graph
    # ──────────────────────────────────────────────────────────────
    def register_provider(self, module: str, provides: List[str]):
        if not isinstance(provides, list):
            provides = [provides]
        try:
            with self._registry_lock:
                for key in provides:
                    if isinstance(key, str) and key:
                        self._providers[key].add(module)
                    else:
                        self.logger.warning(f"Invalid provider key: {key} for module {module}")
            self.logger.info(f"📦 Registered {module} providing: {provides}")
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to register provider {module}: {e}")

    def register_consumer(self, module: str, requires: List[str]):
        if not isinstance(requires, list):
            requires = [requires]
        try:
            with self._registry_lock:
                for key in requires:
                    if isinstance(key, str) and key:
                        self._consumers[key].add(module)
                    else:
                        self.logger.warning(f"Invalid consumer key: {key} for module {module}")
            self.logger.info(f"📨 Registered {module} requiring: {requires}")
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to register consumer {module}: {e}")

    def get_providers(self, key: str) -> Set[str]:
        with self._registry_lock:
            return self._providers.get(key, set()).copy()

    def get_consumers(self, key: str) -> Set[str]:
        with self._registry_lock:
            return self._consumers.get(key, set()).copy()

    def _get_primary_provider(self, key: str) -> Optional[str]:
        providers = self.get_providers(key)
        if providers:
            for p in providers:
                if self.is_module_enabled(p):
                    return p
        return None

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        try:
            with self._registry_lock:
                graph = {}
                for key, consumers in self._consumers.items():
                    providers = self._providers.get(key, set())
                    for provider in providers:
                        graph.setdefault(provider, []).extend(list(consumers))
                for module, deps in self._module_graph.items():
                    graph.setdefault(module, []).extend(list(deps))
                for module in graph:
                    graph[module] = list(set(graph[module]) - {module})
                return graph
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to build dependency graph: {e}")
            return {}

    def find_circular_dependencies(self) -> List[List[str]]:
        graph = self.get_dependency_graph()
        cycles = []

        def dfs(node: str, path: List[str], visited: Set[str]):
            if node in path:
                cyc = path[path.index(node):] + [node]
                cycles.append(cyc)
                return
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            for nb in graph.get(node, []):
                dfs(nb, path.copy(), visited.copy())

        try:
            for n in graph:
                dfs(n, [], set())
            uniq = []
            for c in cycles:
                s = set(c[:-1])
                if not any(set(x[:-1]) == s for x in uniq):
                    uniq.append(c)
            if uniq:
                self.logger.warning(f"Found {len(uniq)} circular dependencies")
            return uniq
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to find circular dependencies: {e}")
            return []

    # ──────────────────────────────────────────────────────────────
    # Performance / health / metrics
    # ──────────────────────────────────────────────────────────────
    def record_module_timing(self, module: str, duration_ms: float):
        if duration_ms < 0:
            self.logger.warning(f"Invalid duration for {module}: {duration_ms}ms")
            return
        try:
            with self._performance_lock:
                self._latency_history[module].append(duration_ms)
                self._access_patterns[module]['execution_count'] += 1
            if len(self._latency_history[module]) >= 10:
                recent_avg = np.mean(list(self._latency_history[module])[-10:])
                if recent_avg > 200:
                    self._emit('performance_warning', {'module': module, 'avg_latency_ms': float(recent_avg), 'threshold_ms': 200})
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to record timing for {module}: {e}")

    def record_module_failure(self, module: str, error: str):
        try:
            with self._circuit_breaker_lock:
                br = self._circuit_breakers[module]
                br.record_failure()
                if not br.should_allow_request(self.config.recovery_time_seconds, self.config.circuit_breaker_threshold):
                    self._module_disabled.add(module)
                    self._emit('module_disabled', {'module': module, 'failures': br.failure_count,
                                                   'consecutive_failures': br.consecutive_failures,
                                                   'failure_rate': br.failure_rate, 'error': error,
                                                   'timestamp': time.time(), 'circuit_breaker_state': br.to_dict()})
                    self.logger.error(format_operator_message("🚫", "MODULE DISABLED",
                                                              instrument=module,
                                                              details=f"After {br.failure_count} failures (rate: {br.failure_rate:.1%})",
                                                              context="circuit_breaker"))
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to record failure for {module}: {e}")

    def is_module_enabled(self, module: str) -> bool:
        try:
            with self._circuit_breaker_lock:
                br = self._circuit_breakers.get(module, CircuitBreakerState())
                ok = br.should_allow_request(self.config.recovery_time_seconds, self.config.circuit_breaker_threshold)
                if ok:
                    self._module_disabled.discard(module)
                else:
                    self._module_disabled.add(module)
                return ok
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to check module status for {module}: {e}")
            return True


    def reset_module_failures(self, module: str):
        try:
            with self._circuit_breaker_lock:
                self._circuit_breakers[module] = CircuitBreakerState()
                self._module_disabled.discard(module)
            self._emit('module_enabled', {'module': module, 'timestamp': time.time(), 'circuit_breaker_reset': True})
            self.logger.info(format_operator_message("[OK]", "MODULE ENABLED", instrument=module,
                                                     details="Circuit breaker reset", context="circuit_breaker_recovery"))
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to reset failures for {module}: {e}")

# ─────────────────────────────────────────────────────────────
# In class SmartInfoBus (modules/utils/info_bus.py)
# ─────────────────────────────────────────────────────────────
    def get_module_health(self, module: str) -> Dict[str, Any]:
        try:
            # Snapshot breaker state first (no nested locking into is_module_enabled)
            with self._circuit_breaker_lock:
                br = self._circuit_breakers.get(module, CircuitBreakerState())

            enabled = self.is_module_enabled(module)

            # Read perf stats
            with self._performance_lock:
                lat = list(self._latency_history.get(module, []))
                ap = dict(self._access_patterns.get(module, {}))

            # Read registry info
            with self._registry_lock:
                provides = [k for k, ps in self._providers.items() if module in ps]
                consumes = [k for k, cs in self._consumers.items() if module in cs]

            avg = float(np.mean(lat)) if lat else 0.0
            return {
                'enabled': enabled,
                'circuit_breaker_state': br.state,
                'failures': br.failure_count,
                'successful_calls': br.successful_calls,
                'total_calls': br.total_calls,
                'failure_rate': br.failure_rate,
                'consecutive_failures': br.consecutive_failures,
                'consecutive_successes': br.consecutive_successes,
                'last_failure_time': br.last_failure_time,
                'last_success_time': br.last_success_time,
                'avg_latency_ms': avg,
                'max_latency_ms': (max(lat) if lat else 0),
                'total_executions': len(lat),
                'provides': provides,
                'consumes': consumes,
                'access_patterns': ap,
                'health_score': br.get_health_score(),
                'predicted_next_failure': br.predict_next_failure(),
                'circuit_breaker_details': br.to_dict()
            }
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to get health for {module}: {e}")
            return {'error': str(e)}


    def get_performance_metrics(self) -> Dict[str, Any]:
        try:
            # Canonical lock order: access -> registry -> performance -> circuit -> event -> request
            with self._access_lock:
                active_keys = len(self._data_store)
                total_versions = sum(len(h) for h in self._data_history.values())

            with self._registry_lock:
                pass  # reserved for coupled reads if needed

            with self._performance_lock:
                total = self._cache_hits + self._cache_misses
                hit_rate = self._cache_hits / max(total, 1)

                module_lat = {}
                for m, timings in self._latency_history.items():
                    if timings:
                        module_lat[m] = {
                            'avg_ms': float(np.mean(timings)),
                            'max_ms': max(timings),
                            'min_ms': min(timings),
                            'p95_ms': float(np.percentile(timings, 95) if len(timings) > 10 else max(timings)),
                            'count': len(timings)
                        }

            with self._circuit_breaker_lock:
                disabled = list(self._module_disabled)
                total_fail = sum(b.failure_count for b in self._circuit_breakers.values())

            with self._event_lock:
                pass  # reserved

            with self._request_lock:
                pend = len(self._pending_requests)

            return {
                'cache_hit_rate': hit_rate,
                'total_requests': int(total),
                'cache_hits': int(self._cache_hits),
                'cache_misses': int(self._cache_misses),
                'active_data_keys': active_keys,
                'total_data_versions': total_versions,
                'total_events': len(self._event_log),
                'disabled_modules': disabled,
                'total_module_failures': int(total_fail),
                'pending_requests': int(pend),
                'module_latencies': module_lat,
                'data_store_size_mb': self._estimate_data_size() / (1024 * 1024),
                'uptime_seconds': time.time() - self._get_initialization_time()
            }
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to get performance metrics: {e}")
            return {'error': str(e)}


    def export_metrics_text(self) -> str:
        """Simple text exposition, Prometheus-style (no server)."""
        m = self.get_performance_metrics()
        lines = [
            "# HELP infobus_cache_hit_rate Cache hit rate.",
            f"infobus_cache_hit_rate {m.get('cache_hit_rate', 0.0)}",
            f"infobus_total_requests {m.get('total_requests', 0)}",
            f"infobus_active_keys {m.get('active_data_keys', 0)}",
            f"infobus_uptime_seconds {int(m.get('uptime_seconds', 0))}",
        ]
        return "\n".join(lines)

    def _estimate_data_size(self) -> float:
        try:
            total = 0
            for dv in self._data_store.values():
                try:
                    serialized = pickle.dumps(dv.value)
                    total += len(serialized)
                except Exception:
                    total += len(str(dv.value)) * 2
            return float(total)
        except Exception:
            return 0.0

    def _get_initialization_time(self) -> float:
        with self._event_lock:
            events = list(self._event_log)
        if events:
            for e in events:
                if isinstance(e, dict) and e.get('type') == 'bus_initialized':
                    return e.get('timestamp', time.time())
        return time.time()

    # ──────────────────────────────────────────────────────────────
    # Cloning & trace helpers
    # ──────────────────────────────────────────────────────────────
    def _safe_clone(self, obj: Any) -> Any:
        # [FIXED] Optimized to avoid deepcopy for immutable types
        if isinstance(obj, (int, float, str, bool, tuple, type(None))):
            return obj
        try:
            return copy.deepcopy(obj)
        except Exception:
            try:
                return json.loads(json.dumps(obj))
            except Exception:
                return obj

    def _init_dependency_tracing_state(self) -> None:
        self._capabilities: Dict[str, Dict[str, Set[str]]] = {}
        self._get_hit_log: deque = deque(maxlen=8000)
        self._get_miss_log: deque = deque(maxlen=8000)
        self._set_log: deque = deque(maxlen=8000)
        self._verbose_io = True

    def _preview(self, value: Any, limit: int = 80) -> str:
        try:
            if isinstance(value, (int, float, bool, type(None))):
                s = repr(value)
            elif isinstance(value, str):
                s = value
            elif isinstance(value, dict):
                keys = list(value.keys())[:5]
                kv = []
                for k in keys:
                    v = value[k]
                    if k in ("regime", "market_regime", "session", "instrument", "intensity"):
                        kv.append(f"{k}={v!r}")
                    else:
                        kv.append(f"{k}={type(v).__name__}")
                s = "{" + ", ".join(kv) + (" ...}" if len(value) > 5 else "}")
            elif isinstance(value, (list, tuple)):
                s = f"{type(value).__name__}[{len(value)}]"
            else:
                s = f"{type(value).__name__}"
        except Exception:
            s = "<unprintable>"
        s = s.replace("\n", " ")
        return (s[:limit] + "…") if len(s) > limit else s

    def _now_iso(self) -> str:
        try:
            return datetime.now(timezone.utc).isoformat()
        except Exception:
            return str(time.time())

    def register_capabilities(self, module_name: str, provides: List[str] | None = None, requires: List[str] | None = None) -> None:
        provides = [k for k in (provides or []) if isinstance(k, str) and k]
        requires = [k for k in (requires or []) if isinstance(k, str) and k]
        try:
            with self._registry_lock:
                entry = self._capabilities.get(module_name) or {'provides': set(), 'requires': set()}
                entry['provides'].update(provides)
                entry['requires'].update(requires)
                self._capabilities[module_name] = entry
            if provides:
                self.register_provider(module_name, provides)
            if requires:
                self.register_consumer(module_name, requires)
            self.logger.debug(f"[GRAPH] Capabilities updated for {module_name}: provides={provides}, requires={requires}")
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to register capabilities for {module_name}: {e}")

    def dump_dependency_report(self, title: str = "Dependency Report") -> str:
        try:
            lines: List[str] = [title, "=" * max(10, len(title)), f"Generated: {self._now_iso()}", ""]
            with self._registry_lock:
                modules = sorted(self._capabilities.keys())
                known_keys = set(self._providers.keys()) | set(self._consumers.keys())
                lines.append(f"Modules: {len(modules)}")
                lines.append(f"Known data keys: {len(known_keys)}")
                lines.append("")
                if modules:
                    lines.append("Capabilities:")
                    for m in modules:
                        caps = self._capabilities.get(m, {'provides': set(), 'requires': set()})
                        prov = sorted(caps.get('provides', set()))
                        req = sorted(caps.get('requires', set()))
                        lines.append(f"  - {m}: provides={prov or ['∅']}, requires={req or ['∅']}")
            lines.append("")
            graph = self.get_dependency_graph()
            if graph:
                lines.append("Dependency Graph (provider -> consumers):")
                for provider, consumers in sorted(graph.items()):
                    lines.append(f"  {provider} -> [{', '.join(sorted(consumers)) if consumers else '∅'}]")
            else:
                lines.append("Dependency Graph: ∅")
            cycles = self.find_circular_dependencies()
            if cycles:
                lines.append("")
                lines.append(f"[ALERT] Circular dependencies detected ({len(cycles)}):")
                for c in cycles:
                    lines.append("  - " + " -> ".join(c))
            rep = "\n".join(lines)
            self.logger.info(f"[GRAPH] Dependency report generated: {len(graph)} nodes, {len(cycles)} cycles")
            return rep
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to build dependency report: {e}")
            return f"{title}\nERROR: {e}"

    # ──────────────────────────────────────────────────────────────
    # Events
    # ──────────────────────────────────────────────────────────────
    def subscribe(self, event_type: str, callback: Callable):
        if not callable(callback):
            raise ValueError("Callback must be callable")
        try:
            with self._subscription_lock:
                self._subscribers[event_type].append(callback)
            self.logger.debug(f"📡 Subscribed to '{event_type}'")
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to subscribe: {e}")

    def unsubscribe(self, event_type: str, callback: Callable):
        try:
            with self._subscription_lock:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to unsubscribe: {e}")

    async def _track_task_completion(self, task: "asyncio.Task[Any]") -> None:
            """Track and remove completed tasks from the set."""
            try:
                await task
            except asyncio.CancelledError:
                pass  # Expected when cancelled
            finally:
                # [FIXED] Use standard `with` on the thread-safe lock
                with self._async_lock:
                    self._pending_tasks.discard(task)

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        try:
            with self._subscription_lock:
                callbacks = list(self._subscribers.get(event_type, []))
        except Exception as exc:
            self.logger.error(f"Emit failed ({event_type}): {exc}")
            return

        for cb in callbacks:
            try:
                # For core event_logged notifications, invoke synchronously to avoid flakiness in tests
                if event_type == 'event_logged':
                    if asyncio.iscoroutinefunction(cb):
                        try:
                            loop = asyncio.get_running_loop()
                            task = loop.create_task(cb(data))
                            with self._async_lock:
                                self._pending_tasks.add(task)
                            asyncio.create_task(self._track_task_completion(task))
                        except RuntimeError:
                            # No running loop; run in thread pool
                            result = self._thread_pool.submit(lambda: asyncio.run(cb(data)))
                            with self._async_lock:
                                self._pending_async_ops.append(result)
                    else:
                        cb(data)
                else:
                    if asyncio.iscoroutinefunction(cb):
                        try:
                            loop = asyncio.get_running_loop()
                            task = loop.create_task(cb(data))
                            with self._async_lock:
                                self._pending_tasks.add(task)
                            asyncio.create_task(self._track_task_completion(task))
                        except RuntimeError:
                            result = self._thread_pool.submit(lambda: asyncio.run(cb(data)))
                            with self._async_lock:
                                self._pending_async_ops.append(result)
                    else:
                        self._thread_pool.submit(cb, data)
            except Exception as exc:
                self.logger.error(f"[CRASH] Event callback error for '{event_type}': {exc}")

    def _log_event(self, event: Dict[str, Any]):
        try:
            event['timestamp'] = event.get('timestamp', time.time())
            with self._event_lock:
                self._event_log.append(event)
            self._emit('event_logged', event)
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to log event: {e}")

    # Bootstrap canonical owners (call once during orchestrator start or first get_instance)
    # NOTE: bootstrap_canonical_owners removed to avoid implicit ownership coupling.


    def _log_miss(self, key: str, module: str):
        try:
            providers = self.get_providers(key)
            self._log_event({'type': 'miss', 'key': key, 'module': module,
                             'providers': list(providers), 'timestamp': time.time()})
            self._emit('data_miss', {'key': key, 'module': module, 'providers': list(providers)})
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to log miss: {e}")

    def _emit_get_event(self, key: str, module: str, data: "DataVersion", reason: str | None = None):
        def _preview(value, max_len=160):
            try:
                s = repr(value)
            except Exception:
                s = str(value)
            return s[:max_len] + "…" if len(s) > max_len else s
        evt = {
            "key": key, "module": module, "version": data.version,
            "source_module": data.source_module, "confidence": data.confidence,
            "age_seconds": data.age_seconds(), "quality_score": data.quality_score,
            "reason": reason or "ok", "preview": _preview(data.value), "timestamp": time.time(),
        }
        self._log_event({"type": "get", **evt})
        self._emit("data_get" if reason is None else "data_get_blocked", evt)

    def _check_pending_requests(self, key: str):
        try:
            fulfilled: List[int] = []
            with self._request_lock:
                for i, req in enumerate(self._pending_requests):
                    if req.requested_key != key:
                        continue

                    data = self._data_store.get(key)
                    if data is None or not isinstance(data, DataVersion):
                        continue

                    dv = cast(DataVersion, data)  # helps the type checker

                    if req.matches_data(dv):
                        payload = {
                            'key': key,
                            'requesting_module': req.requesting_module,
                            'value': dv.value,
                            'metadata': dv.to_dict(),
                        }
                        self._emit('data_available', payload)

                        if req.callback:
                            cb = req.callback
                            try:
                                if asyncio.iscoroutinefunction(cb):
                                    try:
                                        asyncio.get_running_loop().create_task(cb(dv.value))
                                    except RuntimeError:
                                        self._thread_pool.submit(lambda: asyncio.run(cb(dv.value)))
                                else:
                                    self._thread_pool.submit(cb, dv.value)
                            except Exception as e:
                                self.logger.error(f"Request callback error: {e}")

                        fulfilled.append(i)

                # Convert to list for indexed removal, then back to deque
                req_list = list(self._pending_requests)
                for i in reversed(fulfilled):
                    req_list.pop(i)
                self._pending_requests.clear()
                self._pending_requests.extend(req_list)

        except Exception as e:
            self.logger.error(f"[CRASH] Failed to check pending requests: {e}")


    # ──────────────────────────────────────────────────────────────
    # Maintenance
    # ──────────────────────────────────────────────────────────────
    def _background_maintenance(self):
        self.logger.info("[TOOL] Background maintenance started")
        while self._maintenance_running:
            try:
                if self.config.auto_cleanup:
                    self._cleanup_old_data()
                self._cleanup_expired_requests()
                if self._validation_enabled:
                    self._validate_data_integrity()
                time.sleep(self.config.cleanup_interval_seconds)
            except Exception as e:
                self.logger.error(f"[CRASH] Background maintenance error: {e}")
                time.sleep(10)

    def _cleanup_old_data(self):
        max_age = self.config.max_data_age_seconds
        now = time.time()
        removed: List[str] = []
        with self._write_lock:
            for key, dv in list(self._data_store.items()):
                if dv.age_seconds() > max_age and dv.access_count < 5:
                    self._data_store.pop(key, None)
                    removed.append(key)
                    hist = self._data_history.get(key)
                    if hist:
                        recent = [ver for ver in hist if now - ver.timestamp < max_age * 2]
                        if recent:
                            self._data_history[key] = deque(recent, maxlen=self.config.max_history_versions)
                        else:
                            self._data_history.pop(key, None)
        if removed:
            self.logger.debug(f"🧹 Removed {len(removed)} aged keys")

    def _cleanup_expired_requests(self):
        try:
            with self._request_lock:
                active = []
                expired = 0
                for req in self._pending_requests:
                    if not req.is_expired():
                        active.append(req)
                    else:
                        expired += 1
                self._pending_requests.clear()
                self._pending_requests.extend(active)
                if expired > 0:
                    self.logger.debug(f"🕒 Removed {expired} expired requests")
        except Exception as e:
            self.logger.error(f"[CRASH] Request cleanup failed: {e}")

    def _validate_data_integrity(self):
        try:
            bad = 0
            with self._write_lock:
                for key, data in list(self._data_store.items()):
                    if not data.validate_integrity():
                        self.logger.error(f"Data corruption detected: {key}")
                        del self._data_store[key]
                        bad += 1
            if bad > 0:
                self.logger.warning(f"[ALERT] Removed {bad} corrupted data entries")
                self._emit('data_corruption_detected', {'corrupted_count': bad, 'timestamp': time.time()})
        except Exception as e:
            self.logger.error(f"[CRASH] Data integrity validation failed: {e}")

    # ──────────────────────────────────────────────────────────────
    # Analysis / reporting / snapshots
    # ──────────────────────────────────────────────────────────────
    def get_data_freshness_report(self) -> Dict[str, Dict[str, Any]]:
        try:
            report = {}
            with self._access_lock:
                for key, data in self._data_store.items():
                    report[key] = {
                        'age_seconds': data.age_seconds(),
                        'version': data.version,
                        'source': data.source_module,
                        'confidence': data.confidence,
                        'access_count': data.access_count,
                        'has_thesis': data.thesis is not None,
                        'dependencies': len(data.dependencies),
                        'validation_hash': (data.validation_hash[:8] + "…") if data.validation_hash else ""
                    }
            return report
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to generate freshness report: {e}")
            return {}

    def explain_data_flow(self, key: str) -> str:
        try:
            providers = list(self.get_providers(key))
            consumers = list(self.get_consumers(key))
            data = self.get_with_metadata(key, "SystemAnalyzer")
            lines = [f"DATA FLOW ANALYSIS: '{key}'", "=" * 60, ""]
            if not providers and not consumers:
                lines.append(f"[FAIL] No modules interact with '{key}'")
                return "\n".join(lines)
            if providers:
                lines.append(f"📤 PROVIDERS ({len(providers)}):")
                for p in providers:
                    enabled = "[OK]" if self.is_module_enabled(p) else "[FAIL]"
                    health = self.get_module_health(p).get('health_score', 0)
                    lines.append(f"  {enabled} {p} (health: {health:.0f}%)")
            if consumers:
                lines.append("")
                lines.append(f"📥 CONSUMERS ({len(consumers)}):")
                for c in consumers:
                    enabled = "[OK]" if self.is_module_enabled(c) else "[FAIL]"
                    lines.append(f"  {enabled} {c}")
            if data:
                lines.extend(["", "[STATS] CURRENT STATE:",
                              f"  Version: {data.version}",
                              f"  Age: {data.age_seconds():.1f} seconds",
                              f"  Source: {data.source_module}",
                              f"  Confidence: {data.confidence:.1%}",
                              f"  Access Count: {data.access_count}",
                              f"  Dependencies: {len(data.dependencies)}"])
                if data.thesis:
                    lines.extend(["", "💭 EXPLANATION:", f"  {data.thesis}"])
            else:
                lines.extend(["", "[FAIL] NO DATA AVAILABLE"])
            total_reads = 0
            with self._performance_lock:
                for consumer in consumers:
                    total_reads += self._access_patterns[consumer].get(f"read:{key}", 0)
            if total_reads > 0:
                lines.extend(["", "[CHART] ACCESS STATISTICS:", f"  Total Reads: {total_reads}"])
            return "\n".join(lines)
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to explain data flow for {key}: {e}")
            return f"Error explaining data flow: {e}"

    def export_session(self, filepath: str):
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'events': list(self._event_log),
                'final_state': {k: dv.to_dict() for k, dv in self._data_store.items()},
                'performance_metrics': self.get_performance_metrics(),
                'data_freshness': self.get_data_freshness_report(),
                'dependency_graph': self.get_dependency_graph()
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"📁 Session exported to {filepath}")
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to export session: {e}")
            raise

    def import_session(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session = json.load(f)
            with self._event_lock:
                self._event_log = deque(session.get('events', []), maxlen=self.config.max_event_log_size)
            self.logger.info(f"[FOLDER] Imported session with {len(self._event_log)} events")
        except Exception as e:
            self.logger.error(f"[CRASH] Failed to import session: {e}")
            raise

    def export_snapshot(self, filepath: Optional[str] = None, *,
                        include_values: bool = True, include_history: bool = False,
                        include_events: bool = False, include_metrics: bool = True,
                        compress: bool = False) -> Dict[str, Any]:
        """
        Concurrency-safe snapshotting with consistent lock order.
        Holds locks briefly to copy, then serializes unlocked.
        """
        try:
            # Step 1: Acquire locks in canonical order and copy minimal state
            with self._access_lock, self._registry_lock, self._performance_lock, self._circuit_breaker_lock, self._event_lock:
                data_state = copy.deepcopy({k: dv.to_dict(include_value=include_values) for k, dv in self._data_store.items()})
                history = {k: [ver.to_dict(include_value=False) for ver in list(hist)[-5:]]
                        for k, hist in self._data_history.items()} if include_history else {}
                providers = copy.deepcopy(self._providers)
                consumers = copy.deepcopy(self._consumers)
                circuit_breakers = copy.deepcopy(self._circuit_breakers)
                disabled_modules = copy.deepcopy(self._module_disabled)
                events_tail = list(self._event_log)[-1500:] if include_events else []

            # Step 2: Build snapshot object unlocked
            snap = {
                "meta": {"generated_at": datetime.now(timezone.utc).isoformat(),
                        "python": sys.version.split()[0], "platform": sys.platform,
                        "pid": os.getpid(), "features": self._get_enabled_features()},
                "config": self.config.to_dict(),
                "data": data_state,
                "history_tail": history,
                "providers": {k: sorted(list(v)) for k, v in providers.items()},
                "consumers": {k: sorted(list(v)) for k, v in consumers.items()},
                "circuit_breakers": {m: cb.to_dict() for m, cb in circuit_breakers.items()},
                "disabled_modules": sorted(list(disabled_modules)),
            }
            if include_events:
                snap["events_tail"] = events_tail
            if include_metrics:
                snap["metrics"] = self.get_performance_metrics()
                snap["cache_stats"] = self.get_cache_stats()

            # Step 3: Write to file if requested
            if filepath:
                if compress or filepath.endswith(".gz"):
                    with gzip.open(filepath, "wt", encoding="utf-8") as f:
                        json.dump(snap, f, indent=2, default=str)
                else:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(snap, f, indent=2, default=str)
                self.logger.info(f"📦 Snapshot exported to {filepath}")
            return snap
        except Exception as e:
            self.logger.error(f"[CRASH] export_snapshot failed: {e}")
            raise


    def import_snapshot(self, data_or_path: Any, *, replace_existing: bool = False, apply_values: bool = True) -> int:
        try:
            if isinstance(data_or_path, str):
                path = data_or_path
                if path.endswith(".gz"):
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        snapshot = json.load(f)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        snapshot = json.load(f)
            else:
                snapshot = data_or_path
            if not isinstance(snapshot, dict):
                raise ValueError("Invalid snapshot format: expected dict")
            applied = 0
            if replace_existing:
                with self._write_lock:
                    self._data_store.clear()
                    self._data_history.clear()
                    self._data_timestamps.clear()
            block = snapshot.get("data", {})
            for key, payload in block.items():
                if not apply_values:
                    continue
                try:
                    self._set_core(key, payload.get("value"),
                                   payload.get("source_module", "Snapshot"),
                                   payload.get("thesis"), float(payload.get("confidence", 1.0)),
                                   payload.get("dependencies", []), float(payload.get("processing_time_ms", 0.0)))
                    applied += 1
                except Exception as e:
                    self.logger.error(f"[CRASH] Snapshot apply failed for {key}: {e}")
            self.logger.info(f"[OK] Snapshot imported: applied {applied} keys")
            return applied
        except Exception as e:
            self.logger.error(f"[CRASH] import_snapshot failed: {e}")
            raise

    def get_cache_stats(self) -> Dict[str, Any]:
        try:
            with self._access_lock:
                size = len(self._data_store)
            with self._performance_lock:
                hits = self._cache_hits
                misses = self._cache_misses
            return {"size": size, "hits": int(hits), "misses": int(misses),
                    "hit_rate": hits / max(hits + misses, 1), "history_keys": len(self._data_history)}
        except Exception as e:
            self.logger.error(f"[CRASH] get_cache_stats failed: {e}")
            return {"error": str(e)}

    def clear_caches(self, preserve_critical: bool = True) -> int:
        preserved: Set[str] = {"anomaly_detector", "compliance"}
        removed = 0
        now = time.time()
        try:
            with self._write_lock:
                for key in list(self._data_store.keys()):
                    if preserve_critical and key in preserved:
                        continue
                    dv = self._data_store.get(key)
                    if dv is None:
                        continue
                    if preserve_critical:
                        if (now - dv.timestamp) < 10:
                            continue
                        if dv.access_count >= 10:
                            continue
                        if (now - dv.last_access_time) < 5:
                            continue
                    self._data_store.pop(key, None)
                    self._data_timestamps.pop(key, None)
                    hist = self._data_history.get(key)
                    if hist:
                        self._data_history[key] = deque(list(hist)[-1:], maxlen=self.config.max_history_versions)
                    removed += 1
            if removed:
                self._log_event({"type": "cache_cleared", "removed": removed,
                                 "preserve_critical": preserve_critical, "timestamp": time.time()})
                self._emit("cache_cleared", {"removed": removed, "preserve_critical": preserve_critical})
                self.logger.info(f"🧽 Cleared {removed} cached keys (preserve_critical={preserve_critical})")
        except Exception as e:
            self.logger.error(f"[CRASH] clear_caches failed: {e}")
        return removed

    # ──────────────────────────────────────────────────────────────
    # Transactions (context manager)
    # ──────────────────────────────────────────────────────────────
    class _TxContext:
        def __init__(self, bus: "SmartInfoBus"):
            self.bus = bus
            self.ok = False

        def __enter__(self):
            if not self.bus.config.enable_transactions:
                raise RuntimeError("Transactions disabled")
            if getattr(self.bus._tx_local, "buffer", None) is not None:
                raise RuntimeError("Nested transactions not supported")
            self.bus._tx_local.buffer = []
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                if exc:
                    # rollback — simply drop buffer
                    return False
                # commit
                ops = getattr(self.bus._tx_local, "buffer", [])
                for op, args in ops:
                    if op == "set":
                        self.bus._set_core(*args)
                self.ok = True
                return False
            finally:
                self.bus._tx_local.buffer = None

    def transaction(self) -> "SmartInfoBus._TxContext":
        """Usage: with bus.transaction(): bus.set(...); bus.set(...)."""
        return SmartInfoBus._TxContext(self)

    # ──────────────────────────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────────────────────────
    def shutdown(self) -> None:
            """
            Enhanced shutdown with comprehensive async task cleanup and resource management.
            Ensures no pending tasks cause "Task was destroyed but it is pending!" errors.
            """
            self.logger.info("[STOP] Shutting down SmartInfoBus …")

            # Phase 1: Stop background processing immediately
            self._maintenance_running = False
            self._shutdown_event.set()
            if getattr(self, "_cleanup_thread", None) and self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_shutdown.set()
                self.logger.debug("[STOP] Waiting for cleanup thread to finish ...")
                self._cleanup_thread.join(timeout=5.0)

            # Phase 2: Cancel all pending async tasks gracefully
            cancelled_tasks = 0
            try:
                # [FIXED] Use standard `with` on the thread-safe lock
                with self._async_lock:
                    pending_count = len(self._pending_tasks)
                    self.logger.debug(f"[STOP] Cancelling {pending_count} pending async tasks ...")

                    # Cancel all tracked tasks
                    tasks_to_cancel = list(self._pending_tasks)
                    self._pending_tasks.clear()

                    for task in tasks_to_cancel:
                        if not task.done():
                            task.cancel()
                            cancelled_tasks += 1

                    # Cancel related operations in thread pool
                    pending_ops_count = len(self._pending_async_ops)
                    self.logger.debug(f"[STOP] Cancelling {pending_ops_count} pending async operations ...")
                    for future in self._pending_async_ops:
                        if not future.done():
                            future.cancel()
                    self._pending_async_ops.clear()

            except Exception as e:
                self.logger.warning(f"[STOP] Async task cleanup error: {e}")

            # Phase 3: Wait for threads to finish with generous timeout
            try:
                all_threads = []
                if hasattr(self, "_maintenance_threads"):
                    all_threads.extend(self._maintenance_threads or [])

                for t in all_threads:
                    if t.is_alive():
                        self.logger.debug(f"[STOP] Waiting for thread {t.name} ...")
                        t.join(timeout=10.0)
                        if t.is_alive():
                            self.logger.warning(f"[STOP] Thread {t.name} did not finish gracefully")

            except Exception as e:
                self.logger.warning(f"[STOP] Thread cleanup error: {e}")

            # Phase 4: Shutdown thread pool (this will wait for active work)
            try:
                if hasattr(self, "_thread_pool") and self._thread_pool:
                    self.logger.debug("[STOP] Shutting down thread pool ...")
                    self._thread_pool.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                self.logger.warning(f"[STOP] Thread pool shutdown error: {e}")

            # Phase 5: Final cleanup - clear all data structures
            try:
                # Core stores
                with self._write_lock:
                    self._data_store.clear()
                    self._data_history.clear()
                    self._data_timestamps.clear()
                    self._waiters.clear()

                # Registries
                with self._registry_lock:
                    self._providers.clear()
                    self._consumers.clear()
                    self._module_graph.clear()
                    self._capabilities.clear()

                # Event system
                with self._event_lock:
                    self._event_log.clear()

                # Circuit breakers
                with self._circuit_breaker_lock:
                    self._circuit_breakers.clear()
                    self._module_disabled.clear()

                # Requests
                with self._request_lock:
                    self._pending_requests.clear()
                    self._request_history.clear()

                # Performance & metrics
                with self._performance_lock:
                    self._access_patterns.clear()
                    self._operation_timings.clear()
                    self._predictive_metrics.clear()

                # Subscriptions
                with self._subscription_lock:
                    self._subscribers.clear()
                    self._async_subscribers.clear()

            except Exception as e:
                self.logger.warning(f"[STOP] Data cleanup error: {e}")

            # Phase 6: Final statistics (best-effort)
            try:
                m = self.get_performance_metrics()
                self.logger.info(f"[STATS] Final: {m['total_requests']} req, {m['cache_hit_rate']*100:.0f}% hits, "
                                f"{m['active_data_keys']} keys, {cancelled_tasks} tasks cancelled")
            except Exception:
                self.logger.debug("[STOP] Could not generate final stats")

            self.logger.info("[SUCCESS] SmartInfoBus shutdown complete - all resources cleaned up")

# ═══════════════════════════════════════════════════════════════════
# HELPER: ASYNC TASK CLEANUP UTILITIES (for testing/debugging)
# ═══════════════════════════════════════════════════════════════════

    def get_async_task_status(self) -> Dict[str, Any]:
        """Debug helper to check async cleanup status."""
        try:
            tasks = self._pending_tasks.copy()
            pending_tasks = len(tasks)
            active_tasks = len([t for t in tasks if not t.done()])
            done_tasks = pending_tasks - active_tasks
            cancelled_tasks = len([t for t in tasks if t.cancelled()])
            exception_tasks = len([t for t in tasks if t.exception() is not None])

            return {
                'pending_tasks': pending_tasks,
                'active_tasks': active_tasks,
                'done_tasks': done_tasks,
                'cancelled_tasks': cancelled_tasks,
                'exception_tasks': exception_tasks,
                'maintenance_running': self._maintenance_running,
                'thread_pool_alive': self._check_thread_pool_status(),
                'cleanup_thread_alive': getattr(self._cleanup_thread, 'is_alive', False)
            }
        except Exception:
            return {'error': 'Status unavailable'}

    def _check_thread_pool_status(self) -> bool:
        """Check if thread pool is still active."""
        try:
            if hasattr(self, '_thread_pool') and self._thread_pool:
                return not self._thread_pool._shutdown
            return True
        except Exception:
            return False

    def force_cleanup_async_tasks(self) -> int:
        """Force cleanup of any lingering async tasks (emergency only)."""
        try:
            cancelled = 0
            tasks_to_cancel = list(self._pending_tasks)
            self._pending_tasks.clear()

            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
                    cancelled += 1

            return cancelled
        except Exception:
            return -1


# ═══════════════════════════════════════════════════════════════════
# SINGLETON MANAGER
# ═══════════════════════════════════════════════════════════════════

class InfoBusManager:
    """
    Thread-safe singleton manager for SmartInfoBus (XL).
    Provides global access to the unified bus without wiring.
    """
    _instance: Optional[SmartInfoBus] = None
    _lock = threading.RLock()  # allow re-entrant access during nested calls

    
    @classmethod
    def get_instance(cls) -> SmartInfoBus:
        """Get (or lazily create) the SmartInfoBus singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SmartInfoBus()
                    try:
                        # Default policies for stream-like keys (prevents "provider changed" churn)
                        cls._instance.set_policy("thesis_stream", mode="stream")
                        cls._instance.set_policy("vote", mode="stream")
                    except Exception:
                        pass
        return cls._instance

    @classmethod
    def register_module_capabilities(
        cls,
        module_name: str,
        provides: List[str] | None = None,
        requires: List[str] | None = None,
    ):
        """
        Convenience helper so modules can declare what they provide/require
        without importing the bus directly.
        """
        try:
            bus = cls.get_instance()
            bus.register_capabilities(module_name, provides=provides, requires=requires)
        except Exception as e:
            # Never let capability registration crash the process
            try:
                bus = cls.get_instance()
                bus.logger.warning(f"register_module_capabilities failed for {module_name}: {e}")
            except Exception:
                pass

    @classmethod
    def dependency_report(cls, title: str = "Dependency Report") -> str:
        """Generate a human-readable dependency report."""
        bus = cls.get_instance()
        return bus.dump_dependency_report(title)

    @classmethod
    def create_info_bus(cls, env: Any, step: int = 0) -> Dict[str, Any]:
        """
        Create a legacy InfoBus-shaped dict backed by SmartInfoBus (XL).
        Mirrors previous helpers while writing prices into the SmartInfoBus.
        """
        smart_bus = cls.get_instance()

        # Legacy/compat shape
        info_bus = {
            'timestamp': datetime.now().isoformat(),
            'step_idx': step,
            'episode_idx': getattr(env, 'episode_count', 0),
            '_smart_bus': smart_bus,  # reference to the XL bus
            'prices': {},
            'positions': [],
            'risk': {'risk_score': 0.0}
        }

        # Extract price data from the environment, if available
        if hasattr(env, 'data') and hasattr(env, 'instruments'):
            for instrument in env.instruments:
                try:
                    if instrument in env.data and 'D1' in env.data[instrument]:
                        df = env.data[instrument]['D1']
                        if step < len(df):
                            price = float(df['close'].iloc[step])
                            info_bus['prices'][instrument] = price
                            # Reflect into XL bus (namespaced: market)
                            smart_bus.set(
                                f'price_{instrument}',
                                price,
                                module='Environment',
                                thesis=f"Market price for {instrument} at step {step}",
                                confidence=1.0,
                                dependencies=[],
                                processing_time_ms=0.0,
                                namespace="market"
                            )
                except Exception as e:
                    smart_bus.logger.debug(f"[create_info_bus] skip {instrument}: {e}")

        return info_bus

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (useful in tests)."""
        with cls._lock:
            if cls._instance:
                try:
                    cls._instance.shutdown()
                except Exception:
                    pass
            cls._instance = None


# ═══════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY (helpers/extractors/updaters)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InfoBusQuality:
    """Quality assessment for InfoBus validation."""
    score: float
    is_valid: bool
    missing_fields: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


def create_info_bus(env: Any, step: int = 0) -> Dict[str, Any]:
    """Legacy shim — delegates to InfoBusManager."""
    return InfoBusManager.create_info_bus(env, step)


def validate_info_bus(info_bus: Dict[str, Any]) -> InfoBusQuality:
    """
    Legacy validation with enhanced scoring.
    Ensures presence of required fields and basic freshness.
    """
    required = ['timestamp', 'step_idx']
    missing = [f for f in required if f not in info_bus]
    issues: List[str] = []

    # Check for SmartInfoBus integration
    if '_smart_bus' not in info_bus:
        issues.append("Missing SmartInfoBus integration")

    # Timestamp freshness (<= 10 minutes)
    if 'timestamp' in info_bus:
        try:
            ts = datetime.fromisoformat(info_bus['timestamp'].replace('Z', '+00:00'))
            age = (datetime.now() - ts.replace(tzinfo=None)).total_seconds()
            if age > 600:
                issues.append(f"Stale data: {age:.0f}s old")
        except Exception:
            issues.append("Invalid timestamp format")

    # Compute score
    score = 100.0
    score -= len(missing) * 25      # 25 points per missing field
    score -= len(issues) * 10       # 10 points per issue
    score = max(0.0, score)

    return InfoBusQuality(
        score=score,
        is_valid=score >= 50,
        missing_fields=missing,
        issues=issues
    )


class InfoBusExtractor:
    """Legacy extractor that delegates to SmartInfoBus (XL)."""

    @staticmethod
    def get_risk_score(info_bus: Dict[str, Any]) -> float:
        """
        Supports:
          • top-level: info_bus['risk_score']
          • nested:    info_bus['risk']['risk_score']
          • XL bus:    bus.get('risk_score', 'InfoBusExtractor')
        """
        # direct top-level
        if 'risk_score' in info_bus:
            try:
                return float(info_bus['risk_score'])
            except Exception:
                pass

        # legacy nested dict
        risk = info_bus.get('risk')
        if isinstance(risk, dict) and 'risk_score' in risk:
            try:
                return float(risk['risk_score'])
            except Exception:
                pass

        # SmartInfoBus fallback
        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            dv = smart_bus.get('risk_score', 'InfoBusExtractor', namespace=None)
            if dv is not None:
                try:
                    return float(dv)
                except Exception:
                    return 0.0

        return 0.0

    @staticmethod
    def get_market_regime(info_bus: Dict[str, Any]) -> str:
        """Return market regime from legacy dict or XL bus."""
        if 'market_regime' in info_bus:
            return str(info_bus['market_regime'])
        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            dv = smart_bus.get('market_regime', 'InfoBusExtractor', namespace="market")
            if dv is not None:
                return str(dv)
        return 'unknown'

    @staticmethod
    def has_fresh_data(info_bus: Dict[str, Any], max_age_seconds: float = 1.0) -> bool:
        """
        A loose heuristic: if the bus has observed hits recently, consider it fresh.
        (Retains legacy behavior while being resilient.)
        """
        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            metrics = smart_bus.get_performance_metrics()
            return metrics.get('cache_hits', 0) > 0
        return True

    @staticmethod
    def extract_risk_context(info_bus: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate a compact risk context from both legacy and XL bus sources."""
        ctx = {
            'risk_score': InfoBusExtractor.get_risk_score(info_bus),
            'drawdown_pct': info_bus.get('drawdown_pct', 0.0),
            'exposure_pct': info_bus.get('exposure_pct', 0.0),
            'position_count': len(info_bus.get('positions', [])),
            'market_regime': InfoBusExtractor.get_market_regime(info_bus),
        }

        # Enrich from XL bus (optional keys)
        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            for key in ['volatility', 'correlation_risk', 'liquidity_risk']:
                val = smart_bus.get(key, 'InfoBusExtractor', namespace="risk")
                if val is not None:
                    ctx[key] = val

        return ctx


class InfoBusUpdater:
    """Legacy updater that writes through to SmartInfoBus (XL)."""

    @staticmethod
    def add_vote(info_bus: Dict[str, Any], vote: Dict[str, Any]) -> None:
        """
        Append a vote to the legacy list and mirror as a versioned bus key.
        """
        votes = info_bus.get('votes', [])
        votes.append(vote)
        info_bus['votes'] = votes

        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            smart_bus.set(
                f"vote_{len(votes)}",
                vote,
                module='InfoBusUpdater',
                thesis=f"Vote from {vote.get('module', 'unknown')} module",
                confidence=float(vote.get('confidence', 1.0)),
                namespace="votes"
            )

    @staticmethod
    def set_risk_score(info_bus: Dict[str, Any], score: float) -> None:
        """Set risk score in both legacy and XL shapes."""
        info_bus['risk_score'] = score
        info_bus.setdefault('risk', {})['risk_score'] = score

        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            smart_bus.set(
                'risk_score',
                score,
                module='InfoBusUpdater',
                thesis=f"Risk score updated to {score:.4f}",
                confidence=1.0,
                namespace=None
            )

    @staticmethod
    def set_market_regime(info_bus: Dict[str, Any], regime: str) -> None:
        """Set market regime in legacy shape and XL bus (market namespace)."""
        info_bus['market_regime'] = regime

        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            # Single-writer policy: do not publish canonical market_regime from updater.
            # Leave legacy dict updated for backward compatibility; rely on
            # dedicated market module (e.g., FractalRegimeConfirmation) to publish.
            providers = set(smart_bus.get_providers('market_regime'))
            if not providers or providers == {'InfoBusUpdater'}:
                smart_bus.set(
                    'market_regime',
                    regime,
                    module='InfoBusUpdater',
                    thesis=f"Market regime identified as {regime}",
                    confidence=0.95,
                    namespace="market"
                )

    @staticmethod
    def add_alert(info_bus: Dict[str, Any], message: str, *, severity: str = "info", module: str = "InfoBusUpdater", code: Optional[str] = None) -> None:
        """Append an alert to the legacy InfoBus and mirror to SmartInfoBus if available."""
        alert = {
            'timestamp': now_utc(),
            'severity': severity.upper(),
            'module': module,
            'message': message,
        }
        if code:
            alert['code'] = code

        alerts = info_bus.get('alerts')
        if not isinstance(alerts, list):
            alerts = []
        alerts.append(alert)
        info_bus['alerts'] = alerts

        # Mirror to SmartInfoBus as a stream entry when available
        if '_smart_bus' in info_bus:
            try:
                smart_bus: SmartInfoBus = info_bus['_smart_bus']
                # Use publish() for append-only stream semantics
                smart_bus.set(
                    'last_alert',
                    alert,
                    module='InfoBusUpdater',
                    thesis=f"{severity.upper()} alert from {module}",
                    confidence=1.0,
                    namespace='alerts'
                )
            except Exception:
                pass

    @staticmethod
    def add_module_data(info_bus: Dict[str, Any], module_name: str, data: Dict[str, Any]) -> None:
        """Record module-scoped data in legacy shape and mirror to SmartInfoBus."""
        md = info_bus.get('module_data')
        if not isinstance(md, dict):
            md = {}
        md[module_name] = data
        info_bus['module_data'] = md

        if '_smart_bus' in info_bus:
            try:
                smart_bus: SmartInfoBus = info_bus['_smart_bus']
                smart_bus.set(
                    module_name,
                    data,
                    module='InfoBusUpdater',
                    thesis=f"Module data update for {module_name}",
                    confidence=0.9,
                    namespace='modules'
                )
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
# TINY UTILITIES
# ═══════════════════════════════════════════════════════════════════

def now_utc() -> str:
    """Current UTC timestamp (ISO8601, timezone-aware)."""
    return datetime.now(timezone.utc).isoformat()


def extract_standard_context(info_bus: Dict[str, Any]) -> Dict[str, Any]:
    """Small helper to derive a standard snapshot of context for modules."""
    return {
        'regime': InfoBusExtractor.get_market_regime(info_bus),
        'risk_score': InfoBusExtractor.get_risk_score(info_bus),
        'position_count': len(info_bus.get('positions', [])),
        'has_fresh_data': InfoBusExtractor.has_fresh_data(info_bus),
        'timestamp': info_bus.get('timestamp'),
        'step_idx': info_bus.get('step_idx', 0),
    }
