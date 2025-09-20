# ─────────────────────────────────────────────────────────────
# File: modules/core/module_system.py
# SmartInfoBus Module System & Orchestrator (V1.4, "Navigator+")
# - Deterministic stage orchestration with lifecycle telemetry
# - Stage heartbeats & pre-stage readiness scans
# - Explicit timeout attribution + inputs-not-ready reporting
# - Safe bus publishing & richer execution summaries
# - Circuit-breaker hygiene (HALF_OPEN single-probe)
# - Adaptive per-module/stage timeouts (p95+EWMA)
# - Dynamic configuration (async, thread fallback, bus-driven)
# - Recursive module discovery
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import importlib
import inspect
import time
import threading
import yaml
import weakref
import os
from pathlib import Path
from typing import Dict, List, Set, Type, Optional, Any, Callable, Tuple
from typing import get_origin, get_args
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, is_dataclass, fields
import traceback

# Optional deps (graceful fallback)
try:
    import numpy as _np
    _HAVE_NP = True
except Exception:
    _HAVE_NP = False
    _np = None  # type: ignore

try:
    import psutil as _ps
    _HAVE_PS = True
except Exception:
    _HAVE_PS = False
    _ps = None  # type: ignore

from modules.core.module_base import BaseModule, ModuleMetadata
from modules.utils.info_bus import SmartInfoBus, InfoBusManager
from modules.utils.system_utilities import EnglishExplainer
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.core.error_pinpointer import ErrorPinpointer


# ─────────────────────────────────────────────────────────────
# Helpers: math fallbacks + latency predictors
# ─────────────────────────────────────────────────────────────
def _mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    # Guard against optional numpy for static type checkers
    if _np is not None:
        return float(_np.mean(xs))
    return float(sum(xs) / len(xs))


def _percentile_ms(samples: List[float], q: float) -> float:
    """Return the q (0..1) percentile from samples (milliseconds)."""
    if not samples:
        return 0.0
    # Guard against optional numpy for static type checkers
    if _np is not None:
        return float(_np.percentile(samples, q * 100.0))
    xs = sorted(samples)
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * q))))
    return float(xs[k])


def _predict_timeout_ms(perf: Dict[str, Any], default_ms: float, cfg: 'ModuleConfig') -> int:
    """Adaptive timeout = EWMA(pctl(recent)), clamped to [floor, ceiling]."""
    recent = list(perf.get("recent_times", []))
    if not recent:
        return int(default_ms)
    target = float(getattr(cfg, "timeout_target_pctl", 0.95))
    pctl = _percentile_ms(recent, target)
    prev = max(1.0, float(default_ms))
    alpha = 0.30  # EWMA smoothing
    blended = alpha * pctl + (1.0 - alpha) * prev
    lo = float(getattr(cfg, "timeout_floor_ms", 150))
    hi = float(getattr(cfg, "timeout_ceiling_ms", 10000))
    return int(min(hi, max(lo, blended)))


# ─────────────────────────────────────────────────────────────
# Circuit Breaker (thread-safe) with HALF_OPEN single-probe
# ─────────────────────────────────────────────────────────────
@dataclass
class CircuitBreakerState:
    """Thread-safe circuit breaker state per module."""

    def __init__(self):
        self._lock = threading.RLock()
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self.state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.successful_calls: int = 0
        self.total_calls: int = 0
        self.last_success_time: float = 0.0
        self._probe_inflight: bool = False  # allow only a single probe in HALF_OPEN

    def record_success(self):
        with self._lock:
            self.successful_calls += 1
            self.total_calls += 1
            self.last_success_time = time.time()
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self._probe_inflight = False

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.total_calls += 1
            self.last_failure_time = time.time()
            # If probing failed, go back to OPEN and free probe flag
            if self.state == "HALF_OPEN":
                self.state = "OPEN"
                self._probe_inflight = False

    def should_allow_request(self, recovery_time: float, single_probe: bool = True) -> bool:
        with self._lock:
            if self.state == "CLOSED":
                return True
            if self.state == "OPEN":
                # move to HALF_OPEN only after cooldown
                if time.time() - self.last_failure_time > recovery_time:
                    self.state = "HALF_OPEN"
                    if not single_probe:
                        return True
                    if not self._probe_inflight:
                        self._probe_inflight = True
                        return True
                    return False
                return False
            # HALF_OPEN
            if not single_probe:
                return True
            # allow only one in-flight probe
            if not self._probe_inflight:
                self._probe_inflight = True
                return True
            return False

    def trip(self):
        with self._lock:
            self.state = "OPEN"
            self._probe_inflight = False

    def get_state(self) -> str:
        with self._lock:
            return self.state

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'successful_calls': self.successful_calls,
                'total_calls': self.total_calls,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time
            }


# ─────────────────────────────────────────────────────────────
# Stage/Module Lifecycle Telemetry
# ─────────────────────────────────────────────────────────────
@dataclass
class StageStats:
    executions: int = 0
    timeouts: int = 0
    avg_time_ms: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_timeout_at: float = 0.0
    last_missing_inputs: Dict[str, List[str]] = field(default_factory=dict)  # module -> missing keys

    def record(self, dur_ms: float):
        self.executions += 1
        self.recent_times.append(dur_ms)
        self.avg_time_ms = _mean(list(self.recent_times))

    def record_timeout(self):
        self.timeouts += 1
        self.last_timeout_at = time.time()


@dataclass
class ModuleRunStatus:
    status: str  # SUCCESS | ERROR | TIMEOUT | INPUTS_NOT_READY | SKIPPED
    error: Optional[str] = None
    missing: Optional[List[str]] = None
    timeout: bool = False
    dur_ms: Optional[float] = None


# ─────────────────────────────────────────────────────────────
# ModuleConfig (thread-safe, hot-reloadable, dynamic)
# ─────────────────────────────────────────────────────────────
class ModuleConfig:
    """Production-grade configuration with validation + hot-reload."""

    def __init__(self, **kwargs):
        self._lock = threading.RLock()

        # Core system defaults
        self.debug = kwargs.get('debug', True)
        self.max_history = kwargs.get('max_history', 1000)
        self.audit_enabled = kwargs.get('audit_enabled', True)
        self.log_rotation_lines = kwargs.get('log_rotation_lines', 5000)
        self.health_check_interval = kwargs.get('health_check_interval', 100)
        self.performance_tracking = kwargs.get('performance_tracking', True)
        self.cache_enabled = kwargs.get('cache_enabled', True)
        self.explainable = kwargs.get('explainable', True)
        self.hot_reload = kwargs.get('hot_reload', True)

        # Execution parameters
        self.max_parallel_modules = kwargs.get('max_parallel_modules', 10)
        self.default_timeout_ms = kwargs.get('default_timeout_ms', 1000)
        self.circuit_breaker_threshold = kwargs.get('circuit_breaker_threshold', 3)
        self.recovery_time_s = kwargs.get('recovery_time_s', 60)

        # Stage-level controls
        self.stage_timeout_overhead_s = kwargs.get('stage_timeout_overhead_s', 5.0)
        self.stage_timeout_pct_padding = kwargs.get('stage_timeout_pct_padding', 0.25)
        self.soft_schedule_on_missing_inputs = kwargs.get('soft_schedule_on_missing_inputs', False)
        self.pre_stage_readiness_preview = kwargs.get('pre_stage_readiness_preview', True)
        self.stage_heartbeat_interval_s = kwargs.get('stage_heartbeat_interval_s', 6.0)
        self.stage_report_to_bus = kwargs.get('stage_report_to_bus', True)

        # NEW: Dynamic behavior knobs
        self.auto_tune_timeouts = kwargs.get('auto_tune_timeouts', True)
        self.timeout_target_pctl = kwargs.get('timeout_target_pctl', 0.95)
        self.timeout_floor_ms = kwargs.get('timeout_floor_ms', 150)
        self.timeout_ceiling_ms = kwargs.get('timeout_ceiling_ms', 10000)
        self.readiness_grace_s = kwargs.get('readiness_grace_s', 0.75)
        self.half_open_single_probe = kwargs.get('half_open_single_probe', True)
        self.dynamic_config_bus_key = kwargs.get('dynamic_config_bus_key', 'config_update')
        self.stale_warn_s = kwargs.get('stale_warn_s', 60.0)

        # Error handling
        self.max_retries = kwargs.get('max_retries', 3)
        self.error_escalation = kwargs.get('error_escalation', True)
        self.emergency_shutdown_threshold = kwargs.get('emergency_shutdown_threshold', 5)

        # Performance thresholds
        self.latency_warning_ms = kwargs.get('latency_warning_ms', 550)
        self.latency_critical_ms = kwargs.get('latency_critical_ms', 900)
        self.memory_warning_mb = kwargs.get('memory_warning_mb', 1000)
        self.memory_critical_mb = kwargs.get('memory_critical_mb', 2000)

        # Emergency parameters
        self.emergency_mode_enabled = kwargs.get('emergency_mode_enabled', True)
        self.emergency_cooldown_s = kwargs.get('emergency_cooldown_s', 300)
        self.emergency_health_threshold = kwargs.get('emergency_health_threshold', 0.7)

        # Bounded windows
        self.perf_window_size = kwargs.get('perf_window_size', 1000)
        self.stage_timing_window = kwargs.get('stage_timing_window', 100)

        # Discovery: modernized modules
        self.module_paths = kwargs.get('module_paths', [
            'modules/auditing',
            'modules/core',
            'modules/executor',
            'modules/external',
            'modules/features',
            'modules/market_1',
            'modules/memory',
            'modules/meta',
            'modules/models',
            'modules/position',
            'modules/reward',
            'modules/risk',
            'modules/strategy',
            'modules/trading_modes',
            'modules/visualization',
            'modules/voting',
            'modules/simulation',  # simulation modules (ShadowSimulator modernized)
        ])

        self.legacy_modules = {
            'modules/simulation': [
                'OpponentSimulator',
                'RoleCoach',
                'ShadowSimulator'
            ]
        }

        # dynamic updates
        self._config_watchers: List[Callable] = []
        self._config_file_path: Optional[Path] = None
        self._last_config_update = time.time()

        self._validate_config()

    def _validate_config(self):
        errors = []
        if self.max_parallel_modules <= 0:
            errors.append("max_parallel_modules must be positive")
        if self.default_timeout_ms <= 0:
            errors.append("default_timeout_ms must be positive")
        if self.circuit_breaker_threshold <= 0:
            errors.append("circuit_breaker_threshold must be positive")
        if not 0 < self.recovery_time_s <= 3600:
            errors.append("recovery_time_s must be within 1..3600")
        if not 0 < self.emergency_health_threshold <= 1:
            errors.append("emergency_health_threshold must be within 0..1")
        if self.stage_timeout_overhead_s < 0:
            errors.append("stage_timeout_overhead_s must be >= 0")
        if not 0.0 <= self.stage_timeout_pct_padding <= 1.0:
            errors.append("stage_timeout_pct_padding must be in [0,1]")
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")

    def update_config(self, updates: Dict[str, Any], notify: bool = True):
        with self._lock:
            old_values = {}
            for k, v in updates.items():
                if hasattr(self, k):
                    old_values[k] = getattr(self, k)
                    setattr(self, k, v)
            try:
                self._validate_config()
            except Exception as e:
                # rollback
                for k, old in old_values.items():
                    setattr(self, k, old)
                raise e
            self._last_config_update = time.time()
            if notify:
                for watcher in list(self._config_watchers):
                    try:
                        watcher(updates, old_values)
                    except Exception as e:
                        print(f"Config watcher error: {e}")

    def add_config_watcher(self, callback: Callable):
        with self._lock:
            self._config_watchers.append(callback)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def load_from_file(self, config_path: Path):
        self._config_file_path = config_path
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f)
            config_data: Dict[str, Any] = raw if isinstance(raw, dict) else {}
            module_cfg = config_data.get('module_config')
            if isinstance(module_cfg, dict):
                self.update_config(module_cfg)


# ─────────────────────────────────────────────────────────────
# Helper: attribute-access dict wrapper for module configs
# ─────────────────────────────────────────────────────────────
class _AttrDict(dict):
    """A dict that also supports attribute access and exposes __dict__."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        base = dict(*args, **kwargs)
        for k, v in base.items():
            super().__setitem__(k, self._wrap(v))

    @staticmethod
    def _wrap(value):
        if isinstance(value, dict):
            return _AttrDict(value)
        if isinstance(value, list):
            return [_AttrDict._wrap(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_AttrDict._wrap(v) for v in value)
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        if name in ("__class__",):
            return super().__setattr__(name, value)
        self[name] = self._wrap(value)

    @property
    def __dict__(self):  # type: ignore[override]
        return self


# ─────────────────────────────────────────────────────────────
# Module Orchestrator
# ─────────────────────────────────────────────────────────────
class ModuleOrchestrator:
    """
    Central orchestrator for SmartInfoBus modules.
    - Single source of truth for circuit breakers
    - Async execution by stages with dependency awareness
    - Emergency-mode management
    - Health monitor integration
    - Stage heartbeats, readiness scans, and explicit timeout attribution
    - Adaptive timeout prediction; dynamic configs (file/bus)
    """

    _instance: Optional['ModuleOrchestrator'] = None
    _registered_classes: Dict[str, Type[BaseModule]] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        smart_bus: Optional[SmartInfoBus] = None,
        config: Optional[ModuleConfig] = None
    ):
        self.smart_bus = smart_bus or InfoBusManager.get_instance()
        self.config = config or ModuleConfig()
        self.explainer = EnglishExplainer()
        self.error_pinpointer = ErrorPinpointer(self)

        # Registries
        self.modules: Dict[str, BaseModule] = {}
        self.metadata: Dict[str, ModuleMetadata] = {}
        self.module_classes: Dict[str, Type[BaseModule]] = {}

        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self._circuit_breaker_lock = threading.RLock()

        # Execution planning
        self.execution_order: List[str] = []
        self.execution_stages: List[List[str]] = []
        self.voting_members: List[str] = []
        self.critical_modules: Set[str] = set()

        # Dependencies
        self.module_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.circular_dependencies: List[List[str]] = []

        # Performance tracking (bounded)
        self.execution_history: deque = deque(maxlen=self.config.perf_window_size)
        self.stage_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.stage_timing_window))
        self.module_performance: Dict[str, Dict[str, Any]] = {}
        self._perf_lock = threading.RLock()

        # Stage stats & heartbeat
        self._stage_stats: Dict[int, StageStats] = defaultdict(StageStats)
        self._stage_heartbeat_stop = threading.Event()
        self._stage_heartbeat_thread: Optional[threading.Thread] = None

        # Emergency mode
        self.emergency_mode = False
        self.emergency_mode_reason = ""
        self.emergency_activation_time = 0.0
        self.emergency_activation_count = 0
        self.last_emergency_check = 0.0

        # Health monitoring
        self.health_monitor = None
        self.health_check_interval = 10
        self.last_health_check = 0.0

        # System state & errors
        self.module_errors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.consecutive_system_failures = 0
        self.last_successful_execution = time.time()

        # ThreadPool (reserved for potential blocking modules; async path dominates)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()
        self._pending_futures = weakref.WeakSet()
        self._create_executor()

        # Locks
        self.execution_lock = threading.RLock()
        self._async_execution_lock: Optional[asyncio.Lock] = None

        # Config monitor
        self.config_monitor_task: Optional[asyncio.Task] = None
        self.config.add_config_watcher(self._on_config_change)

        # Optional external configuration manager (set in _load_system_configuration)
        self.config_manager: Optional[Any] = None

        # Logging
        self.logger = RotatingLogger(
            name="ModuleOrchestrator",
            log_path="logs/orchestrator/orchestrator.log",
            max_lines=10000,
            operator_mode=True,
            info_bus_aware=True,
            plain_english=True
        )

        # State & helpers
        from modules.core.persistence import StateManager
        self.state_manager = StateManager()

        from modules.monitoring.dependency_visualizer import DependencyVisualizer
        self.dependency_visualizer = DependencyVisualizer(self)

        self._initialized = False
        self._shutdown_requested = False
        ModuleOrchestrator._instance = self

        self.logger.info(
            format_operator_message(
                "[ROCKET]", "ORCHESTRATOR INITIALIZED",
                details=f"Config: {len(self.config.module_paths)} paths",
                context="startup"
            )
        )

    # ───── Executor (reserved for blocking workloads) ─────
    def _create_executor(self):
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.config.max_parallel_modules,
                    thread_name_prefix="ModuleExec"
                )

    @property
    def executor(self):
        if self._executor is None:
            self._create_executor()
        return self._executor

    def set_health_monitor(self, health_monitor):
        self.health_monitor = health_monitor
        self.logger.info("[OK] Health monitor integrated with orchestrator")

    def _on_config_change(self, updates: Dict[str, Any], old_values: Dict[str, Any]):
        self.logger.info(f"[LOG] Configuration updated: {list(updates.keys())}")

        if 'max_parallel_modules' in updates:
            with self._executor_lock:
                if self._executor:
                    old_executor = self._executor
                    self._executor = None
                    try:
                        old_executor.shutdown(wait=True)
                    except Exception as e:
                        self.logger.warning(f"Executor shutdown warning: {e}")
                        old_executor.shutdown(wait=False)
                    self._create_executor()

        if 'circuit_breaker_threshold' in updates:
            with self._circuit_breaker_lock:
                new_thr = updates['circuit_breaker_threshold']
                old_thr = old_values.get('circuit_breaker_threshold', new_thr)
                if new_thr < old_thr:
                    for cb in self.circuit_breakers.values():
                        cb.failure_count = min(cb.failure_count, new_thr - 1)

    # ───── Initialization ─────
    def initialize(self):
        if self._initialized:
            return
        try:
            self.logger.info(
                format_operator_message(
                    "[SEARCH]", "STARTING MODULE DISCOVERY",
                    details="Scanning module directories",
                    context="initialization"
                )
            )

            self._load_system_configuration()

            # Discover & register modules (recursive)
            self.discover_all_modules()

            # IMPORTANT: Re-apply execution config *after* discovery so timeout overrides land
            try:
                cm = getattr(self, "config_manager", None)
                if cm is not None:
                    exec_cfg = cm.get_execution_config()
                    if exec_cfg:
                        self._apply_execution_configuration(exec_cfg)
                        self.logger.info("[OK] Re-applied execution configuration after discovery")
            except Exception as e:
                self.logger.warning(f"Post-discovery execution config apply failed: {e}")

            self._initialize_circuit_breakers()
            self.build_execution_plan()
            self._initialize_emergency_monitoring()
            self._restore_system_state()
            self._start_config_monitoring()
            self._start_stage_heartbeat()
            self._start_bus_config_listener()  # NEW: bus-driven dynamic config

            # Persist initial snapshot
            try:
                results = self.state_manager.save_all_module_states(self)
                saved = sum(1 for ok in results.values() if ok)
                self.logger.info(
                    format_operator_message(
                        "[SAVE]", "INITIAL STATE SNAPSHOT SAVED",
                        details=f"Saved {saved}/{len(results)} modules",
                        context="state_management"
                    )
                )
            except Exception as e:
                self.logger.warning(f"Initial state snapshot failed: {e}")

            self._initialized = True

            self.logger.info(
                format_operator_message(
                    "[OK]", "ORCHESTRATOR READY",
                    details=f"{len(self.modules)} modules, {len(self.execution_stages)} stages",
                    context="initialization"
                )
            )
        except Exception as e:
            self.logger.error(f"[CRASH] INITIALIZATION FAILED: {e}\n{traceback.format_exc()}")
            self.error_pinpointer.analyze_error(e, "ModuleOrchestrator")
            raise

    def _initialize_circuit_breakers(self):
        with self._circuit_breaker_lock:
            for module_name in self.modules:
                self.circuit_breakers[module_name] = CircuitBreakerState()
                self.logger.info(f"[FAST] Initialized circuit breaker for {module_name}")
        self.logger.info(f"[FAST] Initialized {len(self.circuit_breakers)} circuit breakers")

    def _initialize_emergency_monitoring(self):
        self.emergency_mode = False
        self.emergency_mode_reason = ""
        self.emergency_activation_time = 0.0
        self.emergency_triggers = {
            'system_failure_rate': 0.5,
            'critical_module_failure': True,
            'memory_critical': 0.9,
            'consecutive_failures': 3,
            'health_score_threshold': 0.3
        }
        self.logger.info("[ALERT] Emergency monitoring systems initialized")

    def _restore_system_state(self) -> None:
        """Best-effort restoration of module states from disk using StateManager."""
        try:
            results = self.state_manager.restore_all_states(self)
            if not results:
                self.logger.info("[FOLDER] No module states found to restore")
                return
            restored = sum(1 for ok in results.values() if ok)
            total = len(results)
            self.logger.info(f"[FOLDER] Restored {restored}/{total} module states")
        except Exception as e:
            self.logger.warning(f"State restoration skipped: {e}")

    # ───── Checkpoint helpers (manual/opt-in) ─────
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        try:
            return self.state_manager.list_checkpoints()
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            return []

    def restore_latest_checkpoint(self, name_filter: Optional[str] = None) -> bool:
        try:
            checkpoints = self.list_checkpoints()
            if not checkpoints:
                self.logger.info("[FOLDER] No checkpoints available to restore")
                return False

            selected: Optional[Dict[str, Any]] = None
            if name_filter:
                for ck in checkpoints:
                    if str(ck.get('name', '')).startswith(name_filter):
                        selected = ck
                        break
            if selected is None:
                selected = checkpoints[0]

            checkpoint_id = str(selected.get('checkpoint_id') or selected.get('id') or '')
            if not checkpoint_id:
                self.logger.warning("Checkpoint metadata missing checkpoint_id; skipping restore")
                return False

            self.logger.info(f"[RELOAD] Restoring latest checkpoint: {checkpoint_id} ({selected.get('timestamp','')})")
            ok = self.state_manager.restore_checkpoint(self, checkpoint_id)
            if ok:
                try:
                    self.build_execution_plan()
                except Exception:
                    pass
            return ok
        except Exception as e:
            self.logger.error(f"Failed to restore latest checkpoint: {e}")
            return False

    # ───── Perf tracking ─────
    def _update_perf_stats(self, module_name: str, dur_ms: float):
        with self._perf_lock:
            perf = self.module_performance.setdefault(
                module_name,
                {
                    "total_executions": 0,
                    "total_time_ms": 0.0,
                    "failures": 0,
                    "avg_time_ms": 0.0,
                    "recent_times": deque(maxlen=100),
                },
            )
            perf["total_executions"] += 1
            perf["recent_times"].append(dur_ms)
            perf["avg_time_ms"] = _mean(list(perf["recent_times"]))
            if perf["total_executions"] > 10000:
                perf["total_executions"] = len(perf["recent_times"])
                perf["total_time_ms"] = sum(perf["recent_times"])
    # ───── Failure bookkeeping ─────
    def _handle_module_failure(
        self,
        module: BaseModule,
        module_name: str,
        cb: CircuitBreakerState,
        dur_ms: float,
        error_msg: str,
        execution_id: str,
        tag: str = "CRASH",
    ):
        try:
            self.logger.error(f"[FAIL] {module_name} failed ({tag}) after {dur_ms:.1f}ms: {error_msg}")
        except Exception:
            pass
        self.smart_bus.record_module_failure(module_name, error_msg)
        module.record_execution(dur_ms, False, error_msg)

        cb.record_failure()
        if cb.failure_count >= self.config.circuit_breaker_threshold:
            cb.trip()
            self.logger.error(f"[FAST] Circuit breaker TRIPPED for {module_name}")

        with self._perf_lock:
            perf = self.module_performance.setdefault(
                module_name,
                {
                    "total_executions": 0,
                    "total_time_ms": 0.0,
                    "failures": 0,
                    "avg_time_ms": 0.0,
                    "recent_times": deque(maxlen=100),
                },
            )
            perf["failures"] += 1

        self.module_errors[module_name].append({
            'timestamp': time.time(),
            'error': error_msg[:500],
            'execution_id': execution_id
        })

        self.logger.error(
            format_operator_message(
                f"[{tag}]", "MODULE FAILED",
                instrument=module_name,
                details=error_msg[:200],
                context=execution_id,
            )
        )

    # ───── Emergency mode checks ─────
    def _check_emergency_conditions(self) -> Tuple[bool, str]:
        if time.time() - self.last_emergency_check < 1:
            return False, ""
        self.last_emergency_check = time.time()

        if self.execution_history:
            recent = list(self.execution_history)[-10:]
            if recent:
                failure_rate = sum(
                    1 for e in recent if e.get('failure_count', 0) > e.get('success_count', 1)
                ) / len(recent)
                if failure_rate >= self.emergency_triggers['system_failure_rate']:
                    return True, f"System failure rate {failure_rate:.1%} exceeds threshold"

        with self._circuit_breaker_lock:
            for module_name in self.critical_modules:
                cb = self.circuit_breakers.get(module_name)
                if cb and cb.get_state() == "OPEN":
                    return True, f"Critical module '{module_name}' circuit breaker is open"

        if _HAVE_PS:
            try:
                memory_percent = _ps.virtual_memory().percent / 100.0  # type: ignore
                if memory_percent >= self.emergency_triggers['memory_critical']:
                    return True, f"Memory usage {memory_percent:.1%} is critical"
            except Exception:
                pass

        if self.consecutive_system_failures >= self.emergency_triggers['consecutive_failures']:
            return True, f"System had {self.consecutive_system_failures} consecutive failures"

        if self.health_monitor:
            try:
                report = self.health_monitor.generate_health_report()
                score = self._health_score_from_status(getattr(report, 'overall_status', 'unknown'))
                if score < self.emergency_triggers['health_score_threshold']:
                    return True, f"System health score {score:.2f} below critical threshold"
            except Exception:
                pass

        return False, ""

    @staticmethod
    def _health_score_from_status(status: str) -> float:
        s = (status or "").lower()
        if s in ("healthy",):
            return 1.0
        if s in ("warning",):
            return 0.7
        if s in ("critical",):
            return 0.3
        if s in ("error", "unknown"):
            return 0.1
        return 0.5

    def _enter_emergency_mode(self, reason: str):
        if self.emergency_mode:
            return
        self.emergency_mode = True
        self.emergency_mode_reason = reason
        self.emergency_activation_time = time.time()
        self.emergency_activation_count += 1

        self.logger.critical(
            format_operator_message(
                "[ALERT]", "EMERGENCY MODE ACTIVATED",
                details=reason,
                context="emergency"
            )
        )

        disabled_count = 0
        for module_name, metadata in self.metadata.items():
            if not metadata.critical:
                self.smart_bus.record_module_failure(module_name, "Emergency mode - non-critical disabled")
                disabled_count += 1

        with self._executor_lock:
            if self._executor:
                self._executor._max_workers = max(1, self.config.max_parallel_modules // 2)

        if self.health_monitor:
            try:
                self.health_monitor.trigger_emergency_alert(reason)
            except Exception:
                pass

        self._safe_bus_set(
            'emergency_mode_event',
            {
                'activated': True,
                'reason': reason,
                'timestamp': self.emergency_activation_time,
                'disabled_modules': disabled_count,
                'activation_count': self.emergency_activation_count
            },
            module='Orchestrator',
            thesis=f"Emergency mode activated due to: {reason}",
            confidence=0.9
        )

        self.logger.info(f"[ALERT] Disabled {disabled_count} non-critical modules")

    def disable_module(self, module_name: str, reason: str = "Manual disable") -> bool:
        if module_name in self.modules:
            self.smart_bus.record_module_failure(module_name, f"DISABLED: {reason}")
            self.logger.warning(f"⛔ Module disabled: {module_name} - {reason}")
            return True
        return False

    def enable_module(self, module_name: str) -> bool:
        if module_name in self.modules:
            self.smart_bus.reset_module_failures(module_name)
            with self._circuit_breaker_lock:
                self.circuit_breakers[module_name] = CircuitBreakerState()
            self.logger.info(f"[OK] Module enabled: {module_name}")
            return True
        return False

    def exit_emergency_mode(self) -> bool:
        if not self.emergency_mode:
            return True

        time_in_emergency = time.time() - self.emergency_activation_time
        if time_in_emergency < self.config.emergency_cooldown_s:
            remaining = self.config.emergency_cooldown_s - time_in_emergency
            self.logger.info(f"[WAIT] Emergency cooldown: {remaining:.0f}s remaining")
            return False

        checks = {
            'circuit_breakers': self._validate_circuit_breakers(),
            'memory': self._validate_memory_usage(),
            'module_health': self._validate_module_health(),
            'execution_success': self._validate_recent_executions()
        }

        if self.health_monitor:
            try:
                report = self.health_monitor.generate_health_report()
                score = self._health_score_from_status(getattr(report, 'overall_status', 'unknown'))
                checks['overall_health'] = (score >= self.config.emergency_health_threshold)
            except Exception:
                pass

        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            self.logger.warning(f"[FAIL] Cannot exit emergency mode. Failed checks: {failed}")
            return False

        self.emergency_mode = False
        self.consecutive_system_failures = 0

        enabled_count = 0
        for module_name in self.modules:
            if not self.smart_bus.is_module_enabled(module_name):
                self.smart_bus.reset_module_failures(module_name)
                with self._circuit_breaker_lock:
                    self.circuit_breakers[module_name] = CircuitBreakerState()
                enabled_count += 1

        with self._executor_lock:
            if self._executor:
                self._executor._max_workers = self.config.max_parallel_modules

        if self.health_monitor:
            try:
                self.health_monitor.clear_emergency_alert()
            except Exception:
                pass

        self._safe_bus_set(
            'emergency_mode_recovery',
            {
                'recovered': True,
                'duration_seconds': time_in_emergency,
                'timestamp': time.time(),
                'enabled_modules': enabled_count,
                'health_checks': checks
            },
            module='Orchestrator',
            thesis=f"System recovered from emergency mode after {time_in_emergency:.0f}s",
            confidence=0.9
        )

        self.logger.info(
            format_operator_message(
                "[OK]", "EMERGENCY MODE DEACTIVATED",
                details=f"Re-enabled {enabled_count} modules",
                context="recovery"
            )
        )
        return True

    def _validate_circuit_breakers(self) -> bool:
        with self._circuit_breaker_lock:
            open_breakers = [n for n, cb in self.circuit_breakers.items() if cb.get_state() == "OPEN"]
            critical_open = [n for n in open_breakers if n in self.critical_modules]
            return len(critical_open) == 0

    def _validate_memory_usage(self) -> bool:
        if not _HAVE_PS:
            return True
        try:
            return (_ps.virtual_memory().percent / 100.0) < 0.8  # type: ignore
        except Exception:
            return True

    def _validate_module_health(self) -> bool:
        if not self.modules:
            return False
        healthy = sum(1 for m in self.modules.values() if m.is_healthy)
        return (healthy / len(self.modules)) >= 0.7

    def _validate_recent_executions(self) -> bool:
        if not self.execution_history:
            return True
        recent = list(self.execution_history)[-20:]
        if not recent:
            return True
        success_rate = _mean([
            (r['success_count'] / max(r['module_count'], 1)) for r in recent
        ]) if recent else 0.0
        return success_rate >= 0.8

    # ───── Config monitoring ─────
    def _start_config_monitoring(self):
        try:
            loop = asyncio.get_running_loop()
            self.config_monitor_task = loop.create_task(self._monitor_config())
            self.logger.info("[OK] Configuration monitoring started (async)")
        except RuntimeError:
            # Fallback: background thread polling
            def _poll():
                try:
                    config_dir = Path("config")
                    mtimes: Dict[str, float] = {}
                    while not self._shutdown_requested:
                        if config_dir.exists():
                            for cf in config_dir.glob("*.yaml"):
                                try:
                                    m = os.path.getmtime(cf)
                                    if m > mtimes.get(str(cf), 0):
                                        mtimes[str(cf)] = m
                                        if hasattr(self.config, 'load_from_file'):
                                            self.config.load_from_file(cf)
                                            self.logger.info(f"[OK] Config reloaded (thread): {cf}")
                                except Exception:
                                    pass
                        time.sleep(5.0)
                except Exception as e:
                    self.logger.warning(f"Thread config monitor error: {e}")
            t = threading.Thread(target=_poll, name="ConfigMonitor", daemon=True)
            t.start()
            self.logger.info("[OK] Configuration monitoring started (thread)")
        except Exception as e:
            self.logger.warning(f"[WARN] Config monitoring setup failed: {e}")

    async def _monitor_config(self):
        config_files: List[Path] = []
        config_mtimes: Dict[str, float] = {}

        try:
            config_dir = Path("config")
            if config_dir.exists():
                for config_file in config_dir.glob("*.yaml"):
                    config_files.append(config_file)
                    try:
                        config_mtimes[str(config_file)] = os.path.getmtime(config_file)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning(f"Config monitoring setup failed: {e}")
            return

        self.logger.info(f"[LOG] Monitoring {len(config_files)} config files")

        while not self._shutdown_requested:
            try:
                await asyncio.sleep(5)
                for config_file in config_files:
                    try:
                        current_mtime = os.path.getmtime(config_file)
                        if current_mtime > config_mtimes.get(str(config_file), 0):
                            self.logger.info(f"[CHANGE] Config file changed: {config_file}")
                            config_mtimes[str(config_file)] = current_mtime
                            try:
                                if hasattr(self.config, 'load_from_file'):
                                    self.config.load_from_file(config_file)
                                    self.logger.info(f"[OK] Config reloaded: {config_file}")
                            except Exception as reload_error:
                                self.logger.warning(f"Config reload failed: {reload_error}")
                    except (OSError, FileNotFoundError):
                        continue
            except asyncio.CancelledError:
                self.logger.info("[LOG] Config monitoring stopped")
                break
            except Exception as e:
                self.logger.error(f"Config monitoring error: {e}")
                await asyncio.sleep(10)

    # ───── Bus-driven config (polling-friendly) ─────
    def _start_bus_config_listener(self):
        try:
            bus_key = getattr(self.config, "dynamic_config_bus_key", "config_update")

            def _loop():
                # light polling to avoid tight loop; replace with bus subscription if available
                while not self._shutdown_requested:
                    try:
                        payload = self.smart_bus.get(bus_key, "Orchestrator")
                        if isinstance(payload, dict):
                            exec_cfg = payload.get("execution") or {}
                            mod_reg = payload.get("modules") or {}
                            # clear the key to avoid re-applying forever if your bus is persistent
                            if exec_cfg or mod_reg:
                                if exec_cfg:
                                    self._apply_execution_configuration(exec_cfg)
                                if mod_reg:
                                    self._apply_module_registry(mod_reg)
                                self.build_execution_plan()
                                self.logger.info("[OK] Applied config from bus")
                                # Optionally clear: self.smart_bus.set(bus_key, None, module="Orchestrator")
                    except Exception:
                        pass
                    time.sleep(2.0)

            t = threading.Thread(target=_loop, name="BusConfigListener", daemon=True)
            t.start()
            self.logger.info("[OK] Bus config listener started")
        except Exception as e:
            self.logger.warning(f"Bus config listener skipped: {e}")

    # ───── Stage heartbeat (periodic ops telemetry) ─────
    def _start_stage_heartbeat(self):
        if self._stage_heartbeat_thread and self._stage_heartbeat_thread.is_alive():
            return

        def loop():
            interval = max(2.0, float(self.config.stage_heartbeat_interval_s))
            while not self._stage_heartbeat_stop.is_set():
                try:
                    self._emit_stage_heartbeat()
                except Exception as e:
                    self.logger.error(f"[HEARTBEAT] error: {e}")
                self._stage_heartbeat_stop.wait(interval)

        self._stage_heartbeat_stop.clear()
        self._stage_heartbeat_thread = threading.Thread(target=loop, name="StageHeartbeat", daemon=True)
        self._stage_heartbeat_thread.start()
        self.logger.info("[OK] Stage heartbeat loop started")

    def _stop_stage_heartbeat(self):
        self._stage_heartbeat_stop.set()
        if self._stage_heartbeat_thread and self._stage_heartbeat_thread.is_alive():
            self._stage_heartbeat_thread.join(timeout=2.0)
        self.logger.info("[OK] Stage heartbeat loop stopped")

    def _emit_stage_heartbeat(self):
        try:
            cb = self.get_circuit_breaker_status()
            stages = {
                f"stage_{i}": {
                    "executions": s.executions,
                    "timeouts": s.timeouts,
                    "avg_ms": round(s.avg_time_ms, 1),
                    "last_timeout": s.last_timeout_at,
                    "missing_inputs_modules": list(s.last_missing_inputs.keys())[:8],
                }
                for i, s in self._stage_stats.items()
            }
            payload = {
                "timestamp": time.time(),
                "stages": stages,
                "open_breakers": [m for m, d in cb.items() if d.get('state') == 'OPEN'],
            }
            self._safe_bus_set("stage_heartbeat", payload, module="Orchestrator", thesis="Periodic stage summary", confidence=0.7)
        except Exception as e:
            self.logger.debug(f"Heartbeat bus push failed: {e}")

    # ───── Async step execution ─────
    async def execute_step(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")

        if self._async_execution_lock is None:
            self._async_execution_lock = asyncio.Lock()

        should_enter, reason = self._check_emergency_conditions()
        if should_enter and self.config.emergency_mode_enabled:
            self._enter_emergency_mode(reason)

        if self.emergency_mode:
            self.exit_emergency_mode()
            if self.emergency_mode:
                return await self._execute_emergency_mode(market_data)

        start_time = time.time()
        execution_id = f"exec_{int(start_time)}"

        try:
            async with self._async_execution_lock:
                self.logger.debug(f"[ROCKET] STARTING EXECUTION: {execution_id}")

                self._store_market_data(market_data, execution_id)

                if time.time() - self.last_health_check > self.health_check_interval:
                    self._perform_health_check()
                    self.last_health_check = time.time()

                results: Dict[str, Any] = {}
                stage_results: List[Dict[str, Any]] = []

                for stage_idx, stage_modules in enumerate(self.execution_stages):
                    stage_start = time.time()

                    try:
                        allowed = [m for m in stage_modules if self._check_circuit_breaker(m)]

                        if not allowed and self._is_critical_stage(stage_modules):
                            raise RuntimeError(f"All modules in critical stage {stage_idx} are circuit-broken")

                        preview_missing = self._pre_stage_readiness_preview(allowed) if self.config.pre_stage_readiness_preview else {}

                        stage_result = await self._execute_stage(allowed, stage_idx, results, execution_id, preview_missing)
                        results.update(stage_result)
                        stage_results.append(stage_result)

                        stage_dur = (time.time() - stage_start) * 1000.0
                        self.stage_timings[f"stage_{stage_idx}"].append(stage_dur)
                        self._stage_stats[stage_idx].record(stage_dur)

                        if hasattr(self, 'dependency_visualizer'):
                            self.dependency_visualizer.update_performance_metrics(
                                f"stage_{stage_idx}",
                                {"avg_latency_ms": stage_dur, "error_rate": 0.0, "success_rate": 1.0}
                            )

                        if self._check_critical_failures(stage_result):
                            self.logger.error(f"Critical failures in stage {stage_idx}")
                            self.consecutive_system_failures += 1
                            break

                    except Exception as e:
                        self.logger.error(f"Stage {stage_idx} execution failed: {e}")
                        self.error_pinpointer.analyze_error(e, f"Stage{stage_idx}")
                        self.consecutive_system_failures += 1
                        if stage_idx == 0 or self._is_critical_stage(stage_modules):
                            raise

                aggregated = self._aggregate_results(results, execution_id)

                exec_time_ms = (time.time() - start_time) * 1000.0
                self._record_execution(execution_id, exec_time_ms, results, aggregated)

                if len(aggregated.get('successful_modules', [])) > len(aggregated.get('failed_modules', [])):
                    self.consecutive_system_failures = 0
                    self.last_successful_execution = time.time()

                summary = self._generate_execution_summary(
                    execution_id, exec_time_ms, stage_results, aggregated
                )
                self.logger.info(summary)

                return aggregated

        except Exception as e:
            exec_time_ms = (time.time() - start_time) * 1000.0
            self.logger.error(f"[CRASH] EXECUTION FAILED: {execution_id} ({exec_time_ms:.0f}ms)")
            self.logger.error(traceback.format_exc())
            self.error_pinpointer.analyze_error(e, "ModuleOrchestrator")

            self.consecutive_system_failures += 1

            should_enter, reason = self._check_emergency_conditions()
            if should_enter and self.config.emergency_mode_enabled:
                self._enter_emergency_mode(reason)

            raise

    def _check_circuit_breaker(self, module_name: str) -> bool:
        with self._circuit_breaker_lock:
            cb = self.circuit_breakers.get(module_name)
            if not cb:
                return True
            return cb.should_allow_request(
                self.config.recovery_time_s,
                single_probe=getattr(self.config, "half_open_single_probe", True)
            )

    def _perform_health_check(self):
        if not self.health_monitor:
            return
        try:
            report = self.health_monitor.generate_health_report()
            status = getattr(report, 'overall_status', 'unknown')
            alerts = getattr(report, 'alerts', []) or []

            for alert in alerts:
                level = (alert.get('status') or alert.get('severity') or '').lower()
                a_type = alert.get('type', '')
                if level in ('critical',) or ('error' in level):
                    self.logger.critical(f"[ALERT] Health Alert: {a_type or alert}")
                    if 'memory' in a_type:
                        self._handle_memory_alert(alert)
                    elif 'latency' in a_type:
                        self._handle_latency_alert(alert)
                    elif 'error_rate' in a_type:
                        self._handle_error_rate_alert(alert)

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    def _handle_memory_alert(self, alert):
        import gc
        gc.collect()
        with self._executor_lock:
            if self._executor and self._executor._max_workers > 2:
                self._executor._max_workers = max(2, self._executor._max_workers // 2)
                self.logger.warning(f"Reduced parallel execution to {self._executor._max_workers} workers")

    def _handle_latency_alert(self, alert):
        slow_modules = []
        with self._perf_lock:
            for module_name, perf in self.module_performance.items():
                if perf.get('avg_time_ms', 0) > self.config.latency_critical_ms:
                    slow_modules.append(module_name)
        for module_name in slow_modules:
            if module_name not in self.critical_modules:
                self.logger.warning(f"Temporarily disabling slow module: {module_name}")
                self.smart_bus.record_module_failure(module_name, "Disabled due to high latency")

    def _handle_error_rate_alert(self, alert):
        with self._circuit_breaker_lock:
            for module_name, cb in self.circuit_breakers.items():
                if cb.get_state() == "HALF_OPEN" and cb.successful_calls > 5:
                    cb.state = "CLOSED"
                    cb.failure_count = 0
                    self.logger.info(f"Reset circuit breaker for {module_name}")

    # ───── Safe single-module execution ─────
    async def _execute_module_safe(
        self,
        module: BaseModule,
        module_name: str,
        inputs: Dict[str, Any],
        metadata: ModuleMetadata,
        execution_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._circuit_breaker_lock:
            cb = self.circuit_breakers.setdefault(module_name, CircuitBreakerState())
            can_execute = cb.should_allow_request(
                self.config.recovery_time_s,
                single_probe=getattr(self.config, "half_open_single_probe", True)
            )
        if not can_execute:
            self.logger.warning(f"[FAST] Circuit breaker OPEN for {module_name}")
            return {'error': 'Circuit breaker open', '_circuit_breaker': True}

        start_t = time.perf_counter()

        # Adaptive timeout prediction
        with self._perf_lock:
            perf = self.module_performance.get(module_name, {})
        pred_ms = _predict_timeout_ms(perf, metadata.timeout_ms, self.config) \
                  if getattr(self.config, "auto_tune_timeouts", True) else metadata.timeout_ms
        per_mod_timeout = max(0.1, pred_ms / 1000.0)

        async def _run_entire_module() -> Dict[str, Any] | None:
            # 1) Input validation
            if hasattr(module, "validate_inputs"):
                try:
                    if inspect.iscoroutinefunction(module.validate_inputs):
                        await module.validate_inputs(inputs)  # type: ignore
                    else:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, module.validate_inputs, inputs)  # type: ignore
                except Exception:
                    raise

            # 2) Process
            result = await module.process(**inputs)

            # 3) Output validation
            if isinstance(result, dict) and hasattr(module, "validate_outputs"):
                if inspect.iscoroutinefunction(module.validate_outputs):
                    await module.validate_outputs(result)  # type: ignore
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, module.validate_outputs, result)  # type: ignore

            # 4) Publish declared outputs to bus
            if isinstance(result, dict):
                for key in metadata.provides:
                    if key in result:
                        self._safe_bus_set(
                            key,
                            result[key],
                            module=module_name,
                            thesis=result.get("_thesis", f"{module_name} output"),
                            confidence=result.get("_confidence", 0.8),
                        )

            # 5) Optional hooks: confidence + voting
            bm = BaseModule
            if module.__class__.calculate_confidence is not bm.calculate_confidence:
                try:
                    conf_res = module.calculate_confidence(result, **inputs)  # type: ignore
                    conf = await conf_res if asyncio.iscoroutine(conf_res) else conf_res
                    if conf is not None and isinstance(result, dict):
                        result["_confidence"] = float(conf)
                except Exception as e:
                    self.logger.warning(f"{module_name}: confidence error – {e}")

            if module.__class__.propose_action is not bm.propose_action:
                try:
                    voting_context = {"bus": self.smart_bus, **inputs, **(result or {})}
                    ballot_res = module.propose_action(**voting_context)  # type: ignore
                    ballot = await ballot_res if asyncio.iscoroutine(ballot_res) else ballot_res
                    if ballot:
                        self._safe_bus_set(
                            "vote",
                            ballot,
                            module=module_name,
                            thesis=(result or {}).get("_thesis", ""),
                            confidence=(result or {}).get("_confidence", 0.0),
                        )
                except Exception as e:
                    self.logger.warning(f"{module_name}: voting error – {e}")

            return result  # type: ignore

        try:
            result = await asyncio.wait_for(_run_entire_module(), timeout=per_mod_timeout)

            dur_ms = (time.perf_counter() - start_t) * 1000.0
            module.record_execution(dur_ms, True)
            cb.record_success()
            self._update_perf_stats(module_name, dur_ms)
            return result

        except asyncio.TimeoutError:
            dur_ms = float(pred_ms)
            msg = f"Timeout after {dur_ms:.0f} ms (entire module)"
            self._handle_module_failure(module, module_name, cb, dur_ms, msg, execution_id, "TIME")
            raise TimeoutError(msg)

        except Exception as e:
            dur_ms = (time.perf_counter() - start_t) * 1000.0
            self._handle_module_failure(module, module_name, cb, dur_ms, str(e), execution_id, "CRASH")
            raise

    # ───── Emergency execution ─────
    async def _execute_emergency_mode(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("[ALERT] Executing in EMERGENCY MODE - critical modules only")

        execution_id = f"emergency_{int(time.time())}"
        start_time = time.time()
        self._store_market_data(market_data, execution_id)

        results: Dict[str, Any] = {}
        successful = 0
        failed = 0

        for module_name in sorted(self.critical_modules):
            if module_name not in self.modules:
                continue
            module = self.modules[module_name]
            metadata = self.metadata[module_name]
            try:
                with self._circuit_breaker_lock:
                    cb = self.circuit_breakers.get(module_name)

                inputs = self._prepare_module_inputs(module_name, metadata, execution_id)
                # Be generous in emergency
                timeout_s = max(2.0, (metadata.timeout_ms * 2) / 1000.0)
                result = await asyncio.wait_for(module.process(**inputs), timeout=timeout_s)
                results[module_name] = result
                successful += 1
                if cb:
                    cb.record_success()
            except Exception as e:
                self.logger.error(f"Emergency execution failed for {module_name}: {e}")
                results[module_name] = {'error': str(e), '_emergency_mode': True}
                failed += 1
                with self._circuit_breaker_lock:
                    if module_name in self.circuit_breakers:
                        self.circuit_breakers[module_name].record_failure()

        exec_time_ms = (time.time() - start_time) * 1000.0

        if successful > failed:
            self.exit_emergency_mode()

        return {
            'emergency_mode': True,
            'execution_id': execution_id,
            'results': results,
            'successful_modules': successful,
            'failed_modules': failed,
            'execution_time_ms': exec_time_ms,
            'timestamp': time.time(),
            'emergency_reason': self.emergency_mode_reason
        }

    # ───── Utilities ─────
    def get_module_by_name(self, name: str) -> Optional[BaseModule]:
        return self.modules.get(name)

    def get_dependency_graph(self) -> Dict[str, Any]:
        if hasattr(self, 'dependency_visualizer'):
            return self.dependency_visualizer.get_graph_data()
        return {
            'nodes': list(self.modules.keys()),
            'edges': [
                {'from': module, 'to': dep}
                for module, deps in self.module_dependencies.items()
                for dep in deps
            ]
        }

    def _cb_can_execute_peek(self, cb: CircuitBreakerState) -> bool:
        state = cb.get_state()
        if state == "CLOSED":
            return True
        if state == "OPEN":
            return (time.time() - cb.last_failure_time) > self.config.recovery_time_s
        return True  # HALF_OPEN probe may be allowed

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        status = {}
        with self._circuit_breaker_lock:
            for module_name, cb in self.circuit_breakers.items():
                stats = cb.get_stats()
                status[module_name] = {
                    'state': stats['state'],
                    'failure_count': stats['failure_count'],
                    'success_rate': (stats['successful_calls'] / max(stats['total_calls'], 1)) if stats['total_calls'] else None,
                    'last_failure': stats['last_failure_time'],
                    'can_execute': self._cb_can_execute_peek(cb)
                }
        return status

    def reset_circuit_breaker(self, module_name: str) -> bool:
        with self._circuit_breaker_lock:
            if module_name in self.circuit_breakers:
                self.circuit_breakers[module_name] = CircuitBreakerState()
                self.logger.info(f"[FAST] Circuit breaker reset for {module_name}")
                return True
        return False

    def can_exit_emergency_mode(self) -> bool:
        if not self.emergency_mode:
            return True
        t = time.time() - self.emergency_activation_time
        if t < self.config.emergency_cooldown_s:
            return False
        checks = {
            'circuit_breakers': self._validate_circuit_breakers(),
            'memory': self._validate_memory_usage(),
            'module_health': self._validate_module_health(),
            'execution_success': self._validate_recent_executions(),
        }
        if self.health_monitor:
            try:
                report = self.health_monitor.generate_health_report()
                score = self._health_score_from_status(getattr(report, 'overall_status', 'unknown'))
                checks['overall_health'] = (score >= self.config.emergency_health_threshold)
            except Exception:
                pass
        return all(checks.values())

    def get_emergency_mode_status(self) -> Dict[str, Any]:
        return {
            'active': self.emergency_mode,
            'reason': self.emergency_mode_reason,
            'activation_time': self.emergency_activation_time,
            'duration_seconds': time.time() - self.emergency_activation_time if self.emergency_mode else 0,
            'activation_count': self.emergency_activation_count,
            'can_exit': self.can_exit_emergency_mode(),
            'triggers': getattr(self, 'emergency_triggers', {})
        }

    def trigger_emergency_mode_manually(self, reason: str = "Manual trigger"):
        self._enter_emergency_mode(f"MANUAL: {reason}")

    def shutdown(self):
        self.logger.info("[STOP] Initiating system shutdown...")

        self._shutdown_requested = True

        try:
            if self.config_monitor_task:
                self.config_monitor_task.cancel()

            self._stop_stage_heartbeat()

            try:
                results = self.state_manager.save_all_module_states(self)
                saved = sum(1 for ok in results.values() if ok)
                self.logger.info(
                    format_operator_message(
                        "[SAVE]", "FINAL STATE SNAPSHOT SAVED",
                        details=f"Saved {saved}/{len(results)} modules",
                        context="state_management"
                    )
                )
            except Exception as e:
                self.logger.warning(f"Final state snapshot failed: {e}")

            self.state_manager.create_checkpoint(self, "shutdown")

            with self._circuit_breaker_lock:
                cb_summary = {name: cb.get_state() for name, cb in self.circuit_breakers.items()}
            self.logger.info(f"Final circuit breaker states: {cb_summary}")

            with self._executor_lock:
                if self._executor:
                    try:
                        self._executor.shutdown(wait=True)
                    except Exception as e:
                        self.logger.warning(f"Executor shutdown warning: {e}")
                        self._executor.shutdown(wait=False)
                    self._executor = None

            self._registered_classes.clear()
            self.modules.clear()
            self.circuit_breakers.clear()

            self.logger.info("[OK] System shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    # ───── Discovery & registration ─────
    def discover_modules(self) -> Dict[str, Type[BaseModule]]:
        discovered: Dict[str, Type[BaseModule]] = {}
        for path_str in self.config.module_paths:
            path = Path(path_str)
            if not path.exists():
                self.logger.warning(f"Module path does not exist: {path}")
                continue
            for py_file in path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                # support nested packages: build module import path from relative path
                try:
                    rel = py_file.with_suffix("").relative_to(Path("."))
                    module_name = ".".join(rel.parts)
                except Exception:
                    # fallback to legacy behavior
                    module_name = f"{path_str.replace('/', '.')}.{py_file.stem}"
                try:
                    mod = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(mod):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, BaseModule)
                            and obj is not BaseModule
                            and hasattr(obj, '__module_metadata__')
                        ):
                            discovered[name] = obj
                            self.logger.debug(f"Discovered module: {name}")
                except ImportError as e:
                    self.logger.error(f"Failed to import {module_name}: {e}")
                except Exception as e:
                    self.logger.error(f"Error discovering modules in {py_file}: {e}")
        return discovered

    def discover_all_modules(self):
        discovered = self.discover_modules()

        for name, module_class in discovered.items():
            if name not in self.modules:
                self.register_module(name, module_class)

        for name, cls in self._registered_classes.items():
            if name not in self.modules:
                self.register_module(name, cls)

        self.logger.info(f"Module discovery complete: {len(self.modules)} modules registered")

    # ---------------------- Config normalization ----------------------
    @staticmethod
    def _extract_config_dataclass_type(module_class: Type[BaseModule]) -> Optional[Type[Any]]:
        try:
            sig = inspect.signature(module_class.__init__)
            param = sig.parameters.get('config')
            if not param or param.annotation is inspect._empty:
                return None
            ann = param.annotation
            origin = get_origin(ann)
            if origin is None:
                if isinstance(ann, type) and is_dataclass(ann):
                    return ann
                return None
            args = [a for a in get_args(ann) if a is not type(None)]  # noqa: E721
            if not args:
                return None
            t = args[0]
            if isinstance(t, type) and is_dataclass(t):
                return t
            return None
        except Exception:
            return None

    def _normalize_module_config(self, module_class: Type[BaseModule], module_config: Any) -> Optional[Any]:
        if not module_config:
            return None

        dc_type = self._extract_config_dataclass_type(module_class)
        if dc_type is not None:
            if isinstance(module_config, dc_type):
                return module_config
            if isinstance(module_config, dict):
                try:
                    field_names = {f.name for f in fields(dc_type)}
                    filtered = {k: v for k, v in module_config.items() if k in field_names}
                    return dc_type(**filtered)
                except Exception:
                    try:
                        return dc_type()
                    except Exception:
                        pass
            return module_config

        if isinstance(module_config, dict):
            cfg = dict(module_config)
            cfg.setdefault('circuit_breaker_threshold', self.config.circuit_breaker_threshold)
            return _AttrDict(cfg)

        return module_config

    def register_module(self, name: str, module_class: Type[BaseModule]):
        try:
            if not hasattr(module_class, '__module_metadata__'):
                raise ValueError(f"Module {name} missing metadata")

            metadata = getattr(module_class, '__module_metadata__', None)
            if not metadata:
                raise ValueError(f"Module {name} missing metadata")

            if name in self.modules:
                self.logger.warning(f"Module {name} already registered")
                return

            module_config = {}
            if hasattr(self, 'config_manager') and self.config_manager:
                module_config = self.config_manager.get_module_config(name)

            try:
                config_obj: Optional[Any] = self._normalize_module_config(module_class, module_config)
                instance = module_class(config=config_obj) if config_obj is not None else module_class()
            except TypeError:
                instance = module_class()
                if hasattr(instance, 'set_config') and module_config:
                    instance.set_config(module_config)

            self.modules[name] = instance
            self.metadata[name] = metadata
            self.module_classes[name] = module_class

            with self._circuit_breaker_lock:
                self.circuit_breakers[name] = CircuitBreakerState()

            self.smart_bus.register_provider(name, metadata.provides)
            self.smart_bus.register_consumer(name, metadata.requires)

            if metadata.is_voting_member:
                self.voting_members.append(name)
            if metadata.critical:
                self.critical_modules.add(name)

            self.logger.info(f"[OK] Registered module: {name}")

        except Exception as e:
            self.logger.error(f"Failed to register {name}: {e}")

    def _stages_respect_dependencies(self, stages: List[List[str]]) -> bool:
        pos: Dict[str, int] = {}
        for i, stage in enumerate(stages):
            for m in stage:
                if m in pos:
                    return False
                pos[m] = i

        if set(pos.keys()) != set(self.modules.keys()):
            return False

        for consumer, deps in self.module_dependencies.items():
            ci = pos.get(consumer, None)
            if ci is None:
                return False
            for dep in deps:
                di = pos.get(dep, None)
                if di is None:
                    return False
                if ci <= di:
                    return False
        return True

    def _normalize_and_validate_manual_stages(self, manual: List[List[str]]) -> Optional[List[List[str]]]:
        if not manual:
            return None

        listed = [m for stage in manual for m in stage]
        unknown = [m for m in listed if m not in self.modules]
        if unknown:
            self.logger.warning(f"Manual stages contain unknown modules: {unknown}")
            return None

        missing = [m for m in self.modules.keys() if m not in listed]
        if missing:
            self.logger.warning(f"Manual stages are missing modules: {missing}")
            return None

        if not self._stages_respect_dependencies(manual):
            self.logger.warning("Manual stages violate dependency ordering; ignoring manual plan")
            return None

        return manual

    # ───── Planning ─────
    def build_execution_plan(self):
        try:
            self.module_dependencies = defaultdict(set)
            self.reverse_dependencies = defaultdict(set)

            for name, metadata in self.metadata.items():
                self._build_module_dependencies(name, metadata)

            self.circular_dependencies = self._find_circular_dependencies_efficient()
            if self.circular_dependencies:
                self.logger.warning(f"Found circular dependencies: {self.circular_dependencies}")
                self._break_circular_dependencies()

            self.execution_order = self._topological_sort()
            base_stages = self._build_parallel_stages()

            manual = getattr(self, "execution_stages_config", None)
            chosen_stages: Optional[List[List[str]]] = None
            if isinstance(manual, list) and manual and isinstance(manual[0], list):
                normalized = self._normalize_and_validate_manual_stages(manual)
                if normalized:
                    chosen_stages = normalized
                else:
                    self.logger.warning("Ignoring manual parallel_stages due to validation failure")

            if chosen_stages is None:
                chosen_stages = base_stages
                if hasattr(self, 'dependency_visualizer') and self.dependency_visualizer:
                    try:
                        optimized = self.dependency_visualizer.optimize_execution_stages()
                        if optimized and self._stages_respect_dependencies(optimized):
                            chosen_stages = optimized
                        elif optimized:
                            self.logger.warning("Optimizer produced invalid stage ordering; keeping base plan")
                    except Exception as e:
                        self.logger.warning(f"Optimizer error; keeping base plan: {e}")

            self.execution_stages = chosen_stages
            self._log_execution_plan()

        except Exception as e:
            self.logger.error(f"Failed to build execution plan: {e}")
            raise

    def _build_module_dependencies(self, module_name: str, metadata: ModuleMetadata):
        deps: Set[str] = set()

        for required_key in metadata.requires:
            providers = self.smart_bus.get_providers(required_key)
            for provider in providers:
                if provider != module_name and provider in self.modules:
                    deps.add(provider)

        req_keys = set(metadata.requires)
        if req_keys:
            for provider_name, provider_meta in self.metadata.items():
                if provider_name == module_name or provider_name not in self.modules:
                    continue
                if req_keys.intersection(set(provider_meta.provides)):
                    deps.add(provider_name)

        self.module_dependencies[module_name].clear()
        self.module_dependencies[module_name].update(deps)
        for p in deps:
            self.reverse_dependencies[p].add(module_name)

    def _find_circular_dependencies_efficient(self) -> List[List[str]]:
        idx_ctr = [0]
        stack: List[str] = []
        low: Dict[str, int] = {}
        idx: Dict[str, int] = {}
        on_stack: Dict[str, bool] = {}
        cycles: List[List[str]] = []

        def strongconnect(v: str):
            idx[v] = idx_ctr[0]
            low[v] = idx_ctr[0]
            idx_ctr[0] += 1
            on_stack[v] = True
            stack.append(v)

            for w in self.module_dependencies.get(v, []):
                if w not in idx:
                    strongconnect(w)
                    low[v] = min(low[v], low[w])
                elif on_stack.get(w, False):
                    low[v] = min(low[v], idx[w])

            if low[v] == idx[v]:
                comp: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    cycles.append(comp)

        for v in self.modules:
            if v not in idx:
                strongconnect(v)

        return cycles

    def _break_circular_dependencies(self):
        for cycle in self.circular_dependencies:
            if len(cycle) < 2:
                continue
            scc = set(cycle)
            victim = min(scc, key=lambda m: self.metadata[m].priority)
            candidate = next((dep for dep in self.module_dependencies.get(victim, set()) if dep in scc), None)
            if candidate:
                self.module_dependencies[victim].discard(candidate)
                self.reverse_dependencies[candidate].discard(victim)
                self.logger.warning(f"Broke circular dependency: {victim} -> {candidate}")

    def _topological_sort(self) -> List[str]:
        in_degree = {m: 0 for m in self.modules}
        for m, deps in self.module_dependencies.items():
            in_degree[m] += len(deps)

        available = sorted(
            (m for m, d in in_degree.items() if d == 0),
            key=lambda m: self.metadata[m].priority,
            reverse=True
        )

        result: List[str] = []
        while available:
            m = available.pop(0)
            result.append(m)
            for consumer in self.reverse_dependencies.get(m, []):
                in_degree[consumer] -= 1
                if in_degree[consumer] == 0 and consumer not in result and consumer not in available:
                    pri = self.metadata[consumer].priority
                    inserted = False
                    for i, ex in enumerate(available):
                        if self.metadata[ex].priority < pri:
                            available.insert(i, consumer)
                            inserted = True
                            break
                    if not inserted:
                        available.append(consumer)

        remaining = set(self.modules) - set(result)
        if remaining:
            self.logger.warning(f"Orphaned modules: {remaining}")
            result.extend(sorted(remaining, key=lambda m: self.metadata[m].priority, reverse=True))
        return result

    def _build_parallel_stages(self) -> List[List[str]]:
        stages: List[List[str]] = []
        remaining = set(self.modules.keys())
        completed: Set[str] = set()

        while remaining:
            stage: List[str] = []
            for module in list(remaining):
                deps = self.module_dependencies.get(module, set())
                if deps.issubset(completed):
                    stage.append(module)

            if not stage:
                stage = sorted(
                    list(remaining)[:self.config.max_parallel_modules],
                    key=lambda m: self.metadata[m].priority,
                    reverse=True
                )
                self.logger.warning(f"Forced stage: {stage}")

            stages.append(stage)
            completed.update(stage)
            remaining -= set(stage)
        return stages

    def _log_execution_plan(self):
        lines = [
            "EXECUTION PLAN",
            "=" * 50,
            f"Modules: {len(self.modules)}",
            f"Stages: {len(self.execution_stages)}",
            f"Critical: {len(self.critical_modules)}",
            ""
        ]
        for i, stage in enumerate(self.execution_stages, 1):
            lines.append(f"Stage {i}: {len(stage)} modules")
            for module in stage:
                meta = self.metadata[module]
                tags = []
                if meta.critical:
                    tags.append("CRITICAL")
                if meta.is_voting_member:
                    tags.append("VOTER")
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                lines.append(f"  • {module}{tag_str}")
        self.logger.info("\n".join(lines))

    # ───── Stage execution ─────
    async def _execute_stage(
            self,
            module_names: List[str],
            stage_idx: int,
            previous_results: Dict[str, Any],
            execution_id: str,
            preview_missing: Optional[Dict[str, List[str]]] = None
        ) -> Dict[str, Any]:
        self.logger.debug(f"[SEARCH] Debug: Executing stage {stage_idx}: {module_names}")
        if not module_names:
            return {}

        tasks: List[Tuple[str, asyncio.Task]] = []
        enriched: List[Tuple[str, BaseModule, ModuleMetadata]] = []
        results: Dict[str, Any] = {}
        missing_inputs_by_module: Dict[str, List[str]] = {}

        grace = float(getattr(self.config, "readiness_grace_s", 0.0))
        retry_budget = grace if grace > 0 else 0.0

        # Build tasks (or record inputs-not-ready), with micro grace for late providers
        for module_name in module_names:
            module = self.modules[module_name]
            metadata = self.metadata[module_name]

            if not self.smart_bus.is_module_enabled(module_name):
                if getattr(metadata, 'critical', False):
                    try:
                        self.logger.warning(f"[WARN] Critical module disabled by circuit breaker: {module_name}; resetting failures and proceeding")
                        self.smart_bus.reset_module_failures(module_name)
                    except Exception:
                        pass
                else:
                    self.logger.warning(f"[WARN] Warning: Skipping disabled module: {module_name}")
                    results[module_name] = {'error': 'Module disabled', 'status': 'SKIPPED'}
                    continue

            start_grace = time.perf_counter()
            while True:
                try:
                    inputs = self._prepare_module_inputs(module_name, metadata, execution_id)
                    task = asyncio.create_task(
                        self._execute_module_safe(module, module_name, inputs, metadata, execution_id),
                        name=f"{execution_id}_{module_name}"
                    )
                    tasks.append((module_name, task))
                    enriched.append((module_name, module, metadata))
                    break
                except RuntimeError as e:
                    # Inputs-not-ready path: retry briefly if within grace window
                    if "Inputs not ready:" in str(e) and (time.perf_counter() - start_grace) < retry_budget:
                        await asyncio.sleep(0.02)
                        continue
                    msg = str(e)
                    missing_keys = self._parse_missing_keys_from_error(msg)
                    missing_inputs_by_module[module_name] = missing_keys
                    results[module_name] = {
                        'error': 'Inputs not ready',
                        'missing': missing_keys,
                        'status': 'INPUTS_NOT_READY'
                    }
                    self._record_stage_missing(stage_idx, module_name, missing_keys)

                    if self.config.soft_schedule_on_missing_inputs:
                        try:
                            partial_inputs = {'execution_id': execution_id}  # minimal
                            task = asyncio.create_task(
                                self._execute_module_safe(module, module_name, partial_inputs, metadata, execution_id),
                                name=f"{execution_id}_{module_name}_soft"
                            )
                            tasks.append((module_name, task))
                            enriched.append((module_name, module, metadata))
                            results.pop(module_name, None)
                        except Exception:
                            pass
                    break
                except Exception as e:
                    results[module_name] = {'error': f'Prepare inputs failed: {e}', 'status': 'ERROR'}
                    break

        if not tasks:
            self._publish_stage_report(stage_idx, execution_id, results, missing_inputs_by_module, timed_out=[])
            return results

        # Deterministic stage timeout from predicted module budgets
        predicted = []
        with self._perf_lock:
            for name, _ in tasks:
                perf = self.module_performance.get(name, {})
                ms = _predict_timeout_ms(perf, self.metadata[name].timeout_ms, self.config) \
                     if getattr(self.config, "auto_tune_timeouts", True) else self.metadata[name].timeout_ms
                predicted.append(ms)

        stage_timeout = (max(predicted) / 1000.0) * (1.0 + float(self.config.stage_timeout_pct_padding)) \
                        + float(self.config.stage_timeout_overhead_s)

        # Execute with hard deadline
        timed_out_modules: List[str] = []
        stage_start = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.gather(*[t for _, t in tasks], return_exceptions=True),
                timeout=stage_timeout
            )
        except asyncio.TimeoutError:
            self.logger.error(f"[FAIL] Error: Stage {stage_idx} timeout")
            self._stage_stats[stage_idx].record_timeout()
            for name, task in tasks:
                if not task.done():
                    task.cancel()
                    timed_out_modules.append(name)
            for name, module, metadata in enriched:
                if name in timed_out_modules:
                    with self._circuit_breaker_lock:
                        cb = self.circuit_breakers.setdefault(name, CircuitBreakerState())
                    try:
                        self._handle_module_failure(module, name, cb, float(metadata.timeout_ms),
                                                    f"Stage {stage_idx} timeout (>{stage_timeout:.1f}s)", execution_id, "TIME")
                    except Exception:
                        pass

        # Collect results
        for module_name, task in tasks:
            try:
                if task.done() and not task.cancelled():
                    res = task.result()
                    if isinstance(res, Exception):
                        results[module_name] = {'error': str(res), 'status': 'ERROR'}
                    else:
                        if isinstance(res, dict) and 'error' in res:
                            results[module_name] = {'error': res.get('error', 'Unknown error'), 'status': 'ERROR', **res}
                        else:
                            results[module_name] = res if isinstance(res, dict) else {'result': res}
                            results[module_name]['status'] = 'SUCCESS'
                else:
                    results[module_name] = {'error': 'Task canceled due to stage timeout', 'status': 'TIMEOUT'}
            except Exception as e:
                self.logger.error(f"Error collecting result from {module_name}: {e}")
                results[module_name] = {'error': str(e), 'status': 'ERROR'}

        self._publish_stage_report(stage_idx, execution_id, results, missing_inputs_by_module, timed_out=timed_out_modules)

        stage_dur_ms = (time.perf_counter() - stage_start) * 1000.0
        self._stage_stats[stage_idx].record(stage_dur_ms)

        return results

    def _parse_missing_keys_from_error(self, msg: str) -> List[str]:
        if "Inputs not ready:" in msg:
            tail = msg.split("Inputs not ready:", 1)[-1].strip()
            if tail.startswith("[") and tail.endswith("]"):
                raw = tail.strip("[] \t")
                if not raw:
                    return []
                parts = [p.strip().strip("'").strip('"') for p in raw.split(",")]
                return [p for p in parts if p]
        return []

    def _record_stage_missing(self, stage_idx: int, module_name: str, keys: List[str]):
        try:
            s = self._stage_stats[stage_idx]
            s.last_missing_inputs[module_name] = keys
        except Exception:
            pass

    def _publish_stage_report(
        self,
        stage_idx: int,
        execution_id: str,
        results: Dict[str, Any],
        missing_inputs: Dict[str, List[str]],
        timed_out: List[str]
    ):
        if not self.config.stage_report_to_bus:
            return
        try:
            classification = {
                m: (r.get('status') or ('ERROR' if r.get('error') else 'SUCCESS'))
                for m, r in results.items()
            }
            payload = {
                "execution_id": execution_id,
                "stage_index": stage_idx,
                "timestamp": time.time(),
                "modules": list(results.keys()),
                "status_by_module": classification,
                "missing_inputs": missing_inputs,
                "timed_out": timed_out,
            }
            self._safe_bus_set(
                f"stage_report_{stage_idx}",
                payload,
                module="Orchestrator",
                thesis=f"Stage {stage_idx} report for {execution_id}",
                confidence=0.8
            )
        except Exception as e:
            self.logger.debug(f"Stage report bus push failed: {e}")

    def _pre_stage_readiness_preview(self, module_names: List[str]) -> Dict[str, List[str]]:
        preview: Dict[str, List[str]] = {}
        try:
            for module_name in module_names:
                meta = self.metadata[module_name]
                missing: List[str] = []
                for key in meta.requires:
                    try:
                        md = self.smart_bus.get_with_metadata(key, module_name)
                        if md is None or md.value is None:
                            val = self.smart_bus.get(key, module_name)
                            if val is None:
                                missing.append(key)
                    except Exception:
                        missing.append(key)
                if missing:
                    preview[module_name] = missing
            if preview:
                self.logger.debug(f"[PREVIEW] Stage missing preview: {preview}")
        except Exception:
            pass
        return preview

    def _safe_bus_set(self, key: str, value: Any, *, module: str, thesis: str = "", confidence: float = 0.8):
        try:
            self.smart_bus.set(key, value, module=module, thesis=thesis, confidence=confidence)
        except Exception as e:
            self.logger.debug(f"Bus set failed for {key}: {e}")

    def _prepare_module_inputs(
        self,
        module_name: str,
        metadata: ModuleMetadata,
        execution_id: str
    ) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {'execution_id': execution_id}
        missing: List[str] = []

        stale_warn_s = float(getattr(self.config, "stale_warn_s", 60.0))

        for required_key in metadata.requires:
            data = self.smart_bus.get_with_metadata(required_key, module_name)
            if data:
                inputs[required_key] = data.value
                age = data.age_seconds()
                if age > stale_warn_s:
                    self.logger.warning(f"Stale data for {module_name}: {required_key} ({age:.1f}s old)")
            else:
                value = self.smart_bus.get(required_key, module_name)
                if value is not None:
                    inputs[required_key] = value
                else:
                    missing.append(required_key)

        if missing:
            for key in missing:
                self.smart_bus.request_data(key, module_name)
            raise RuntimeError(f"Inputs not ready: {missing}")

        return inputs

    def _store_market_data(self, market_data: Dict[str, Any] | None, execution_id: str):
        try:
            payload = market_data or {}
            if not isinstance(payload, dict):
                try:
                    payload = dict(payload)
                except Exception:
                    self.logger.warning("Non-dict market_data provided; storing under 'raw_market_payload'")
                    self._safe_bus_set(
                        'raw_market_payload',
                        market_data,
                        module="Environment",
                        thesis=f"Raw market payload for {execution_id}",
                        confidence=0.5
                    )
                    payload = {}

            for key, value in payload.items():
                if not str(key).startswith('_'):
                    self._safe_bus_set(
                        key,
                        value,
                        module="Environment",
                        thesis=f"Market data for {execution_id}",
                        confidence=1.0
                    )

            self._safe_bus_set(
                'execution_metadata',
                {
                    'execution_id': execution_id,
                    'timestamp': time.time(),
                    'data_keys': list(payload.keys())
                },
                module='Orchestrator',
                thesis=f"Execution metadata for {execution_id}",
                confidence=0.9
            )
        except Exception as e:
            self.logger.error(f"Failed to store market data: {e}")
            raise

    def _check_critical_failures(self, stage_result: Dict[str, Any]) -> bool:
        for module_name, result in stage_result.items():
            if module_name in self.critical_modules:
                if isinstance(result, dict) and 'error' in result and (result.get('status') != 'SUCCESS'):
                    return True
        return False

    def _is_critical_stage(self, stage_modules: List[str]) -> bool:
        return any(m in self.critical_modules for m in stage_modules)

    def _aggregate_results(self, results: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        aggregated: Dict[str, Any] = {
            'execution_id': execution_id,
            'timestamp': time.time(),
            'module_count': len(results),
            'successful_modules': [],
            'failed_modules': [],
            'votes': {},
            'signals': {},
            'analysis': {},
            'theses': {},
            'performance_metrics': {}
        }

        for module_name, result in results.items():
            if isinstance(result, dict) and (result.get('status') == 'SUCCESS') and ('error' not in result):
                aggregated['successful_modules'].append(module_name)
                for key, value in result.items():
                    if key == '_thesis':
                        aggregated['theses'][module_name] = value
                    elif key == 'vote':
                        aggregated['votes'][module_name] = value
                    elif key == 'trading_signal':
                        aggregated['signals'][module_name] = value
                    elif not key.startswith('_') and key not in ('status',):
                        aggregated['analysis'].setdefault(key, {})[module_name] = value
            else:
                reason = 'Unknown error'
                if isinstance(result, dict):
                    reason = result.get('error', reason)
                    status = result.get('status', 'ERROR')
                else:
                    status = 'ERROR'
                aggregated['failed_modules'].append({
                    'module': module_name,
                    'error': reason,
                    'status': status
                })

        with self._perf_lock:
            for module_name in results:
                if module_name in self.modules:
                    try:
                        health = self.modules[module_name].get_health_status() or {}
                    except Exception:
                        health = {}
                    perf = health.get('performance') if isinstance(health, dict) else None
                    if perf is None:
                        perf = self.module_performance.get(module_name, health if isinstance(health, dict) else {})
                    aggregated['performance_metrics'][module_name] = perf

        self._safe_bus_set(
            'execution_results',
            aggregated,
            module='Orchestrator',
            thesis=self.explainer.explain_execution_results(
                aggregated,
                execution_time=0.0,
                module_count=len(results),
                success_count=len(aggregated['successful_modules'])
            ),
            confidence=0.9
        )

        return aggregated

    def _record_execution(
        self,
        execution_id: str,
        execution_time: float,
        results: Dict[str, Any],
        aggregated: Dict[str, Any]
    ):
        record = {
            'execution_id': execution_id,
            'timestamp': time.time(),
            'execution_time_ms': execution_time,
            'module_count': len(results),
            'success_count': len(aggregated['successful_modules']),
            'failure_count': len(aggregated['failed_modules']),
            'emergency_mode': self.emergency_mode
        }
        self.execution_history.append(record)

    def _generate_execution_summary(
        self,
        execution_id: str,
        execution_time: float,
        stage_results: List[Dict[str, Any]],
        aggregated: Dict[str, Any]
    ) -> str:
        mode = "[ALERT] EMERGENCY" if self.emergency_mode else "[OK] NORMAL"
        failures = aggregated['failed_modules']
        timeouts = [f for f in failures if f.get('status') == 'TIMEOUT']
        inputs_wait = [f for f in failures if f.get('status') == 'INPUTS_NOT_READY']
        errors = [f for f in failures if f.get('status') not in ('TIMEOUT', 'INPUTS_NOT_READY')]

        lines = [
            f"EXECUTION COMPLETE: {execution_id} [{mode}]",
            "=" * 50,
            f"Time: {execution_time:.0f}ms",
            f"Success: {len(aggregated['successful_modules'])}/{aggregated['module_count']}",
        ]
        if failures:
            lines.append("FAILURES:")
            for bucket, title in ((timeouts, "Timeout"), (inputs_wait, "Inputs not ready"), (errors, "Error")):
                if bucket:
                    lines.append(f"  • {title}: " + ", ".join(f["module"] for f in bucket[:5]))
        return "\n".join(lines)

    # ───── Config/Registry loaders ─────
    def _load_system_configuration(self):
        try:
            from modules.core.configuration_manager import ConfigurationManager
            config_manager = ConfigurationManager.get_instance()

            execution_config = config_manager.get_execution_config()
            if execution_config:
                self._apply_execution_configuration(execution_config)

            module_registry = config_manager.get_module_registry()
            if module_registry:
                self._apply_module_registry(module_registry)

            self.config_manager = config_manager

            try:
                def _cm_watcher(name: str, old: Dict[str, Any], new: Dict[str, Any]):
                    if name == 'system':
                        try:
                            exec_cfg = self.config_manager.get_execution_config() if self.config_manager else {}
                            if exec_cfg:
                                self._apply_execution_configuration(exec_cfg)
                                self.logger.info("[OK] Re-applied execution configuration (hot)")
                                if 'parallel_stages' in exec_cfg:
                                    try:
                                        self.build_execution_plan()
                                    except Exception as e:
                                        self.logger.warning(f"Execution plan rebuild failed after config change: {e}")
                        except Exception as e:
                            self.logger.warning(f"Execution config hot-apply failed: {e}")
                config_manager.add_config_watcher(_cm_watcher)
            except Exception as e:
                self.logger.debug(f"Config watcher registration skipped: {e}")
            self.logger.info("[OK] System configuration loaded from ConfigurationManager")

        except Exception as e:
            self.logger.error(f"Failed to load system configuration: {e}")
            self.config_manager = None

    def _apply_execution_configuration(self, execution_config: Dict[str, Any]):
        try:
            if 'timeouts' in execution_config:
                timeouts = execution_config['timeouts'] or {}
                if 'default_ms' in timeouts:
                    self.config.default_timeout_ms = timeouts['default_ms']
                if 'by_module' in timeouts:
                    self.module_timeouts = timeouts['by_module']
                if 'by_category' in timeouts:
                    self.category_timeouts = timeouts['by_category']

                try:
                    overrides_applied = 0
                    for m_name, meta in self.metadata.items():
                        override_ms = None
                        try:
                            override_ms = (self.module_timeouts or {}).get(m_name)
                            if override_ms is None:
                                override_ms = (self.category_timeouts or {}).get(getattr(meta, 'category', ''), None)
                        except Exception:
                            override_ms = None

                        if override_ms:
                            new_ms = int(max(1, float(override_ms)))
                            if getattr(meta, 'timeout_ms', None) != new_ms:
                                meta.timeout_ms = new_ms  # type: ignore[attr-defined]
                                overrides_applied += 1
                    if overrides_applied:
                        self.logger.info(f"[OK] Applied timeout overrides to {overrides_applied} modules")
                except Exception as e:
                    self.logger.warning(f"Timeout override application failed: {e}")

            if 'circuit_breakers' in execution_config:
                cb = execution_config['circuit_breakers'] or {}
                if 'failure_threshold' in cb:
                    self.config.circuit_breaker_threshold = cb['failure_threshold']
                if 'recovery_time_s' in cb:
                    self.config.recovery_time_s = cb['recovery_time_s']

            if 'performance' in execution_config:
                perf = execution_config['performance'] or {}
                if 'enable_caching' in perf:
                    self.config.cache_enabled = perf['enable_caching']
                if 'memory_limits' in perf:
                    mem = perf['memory_limits'] or {}
                    if 'per_module_mb' in mem:
                        self.config.memory_warning_mb = mem['per_module_mb']
                if 'cpu_limits' in perf:
                    cpu = perf['cpu_limits'] or {}
                    if 'total_worker_threads' in cpu:
                        self.config.max_parallel_modules = cpu['total_worker_threads']

            if 'parallel_stages' in execution_config:
                self.execution_stages_config = execution_config['parallel_stages']

            if 'stage' in execution_config:
                stg = execution_config['stage'] or {}
                self.config.stage_timeout_overhead_s = stg.get('timeout_overhead_s', self.config.stage_timeout_overhead_s)
                self.config.stage_timeout_pct_padding = stg.get('timeout_pct_padding', self.config.stage_timeout_pct_padding)
                self.config.pre_stage_readiness_preview = stg.get('pre_stage_readiness_preview', self.config.pre_stage_readiness_preview)
                self.config.soft_schedule_on_missing_inputs = stg.get('soft_schedule_on_missing_inputs', self.config.soft_schedule_on_missing_inputs)
                self.config.stage_heartbeat_interval_s = stg.get('heartbeat_interval_s', self.config.stage_heartbeat_interval_s)
                self.config.stage_report_to_bus = stg.get('report_to_bus', self.config.stage_report_to_bus)

            # Optional: enable/disable adaptive features from exec config
            if 'adaptive' in execution_config:
                ad = execution_config['adaptive'] or {}
                for k in ('auto_tune_timeouts', 'timeout_target_pctl', 'timeout_floor_ms',
                          'timeout_ceiling_ms', 'readiness_grace_s', 'half_open_single_probe', 'stale_warn_s'):
                    if k in ad:
                        setattr(self.config, k, ad[k])

            self.logger.info("[OK] Applied execution configuration")

        except Exception as e:
            self.logger.error(f"Failed to apply execution configuration: {e}")

    def _apply_module_registry(self, module_registry: Dict[str, Any]):
        try:
            self.module_registry_config = module_registry
            for module_name, module_config in module_registry.items():
                if isinstance(module_config, dict):
                    if not hasattr(self, 'pending_module_configs'):
                        self.pending_module_configs = {}
                    self.pending_module_configs[module_name] = module_config
            self.logger.info(f"[OK] Applied module registry: {len(module_registry)} modules")
        except Exception as e:
            self.logger.error(f"Failed to apply module registry: {e}")

    # ───── Public reports ─────
    def get_execution_metrics(self) -> Dict[str, Any]:
        if not self.execution_history:
            return {}
        recent = list(self.execution_history)[-100:]
        with self._circuit_breaker_lock:
            cb_status = self.get_circuit_breaker_status()
        with self._perf_lock:
            perf_metrics = dict(self.module_performance)

        avg_exec_ms = _mean([float(r.get('execution_time_ms', 0.0)) for r in recent]) if recent else 0.0
        success_rate = _mean([
            (r['success_count'] / max(r['module_count'], 1)) for r in recent
        ]) if recent else 0.0

        return {
            'total_executions': len(self.execution_history),
            'avg_execution_time_ms': avg_exec_ms,
            'success_rate': success_rate,
            'emergency_mode': self.emergency_mode,
            'circuit_breakers': cb_status,
            'module_performance': perf_metrics
        }

    def get_system_status_report(self) -> str:
        metrics = self.get_execution_metrics()
        emergency_status = self.get_emergency_mode_status()
        with self._circuit_breaker_lock:
            open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.get_state() == 'OPEN')

        lines = [
            "SMARTINFOBUS SYSTEM STATUS",
            "=" * 50,
            f"Mode: {'[ALERT] EMERGENCY' if self.emergency_mode else '[OK] NORMAL'}",
            f"Modules: {len(self.modules)} ({len(self.critical_modules)} critical)",
            f"Circuit Breakers: {open_breakers} open",
        ]
        if self.emergency_mode:
            lines.extend([
                "",
                "EMERGENCY MODE DETAILS:",
                f"  Reason: {emergency_status['reason']}",
                f"  Duration: {emergency_status['duration_seconds']:.0f}s",
                f"  Can Exit: {emergency_status['can_exit']}"
            ])
        return "\n".join(lines)

    def get_legacy_module_status(self) -> Dict[str, Any]:
        legacy_status: Dict[str, Any] = {}
        for path, modules in self.config.legacy_modules.items():
            legacy_status[path] = {
                'total_modules': len(modules),
                'modernized_count': 0,
                'instructions': []
            }
            for module_name in modules:
                if module_name in self.modules:
                    legacy_status[path]['modernized_count'] += 1
                else:
                    legacy_status[path]['instructions'].append({
                        'module_name': module_name,
                        'path': path,
                        'reason': 'Module not found in current modules list. Please ensure it is registered.'
                    })
        return legacy_status

    # ───── Singleton API ─────
    @classmethod
    def get_instance(cls) -> 'ModuleOrchestrator':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def register_class(cls, module_class: Type[BaseModule]) -> None:
        """
        Idempotent registration that prefers the @module metadata name over the
        Python class name and avoids duplicate instance construction.
        """
        try:
            meta = getattr(module_class, "__module_metadata__", None)
            module_name = meta.name if meta and getattr(meta, "name", None) else module_class.__name__
        except Exception:
            module_name = module_class.__name__

        with cls._lock:
            existing = cls._registered_classes.get(module_name)

            if existing is None:
                cls._registered_classes[module_name] = module_class
                if cls._instance and module_name not in cls._instance.modules:
                    cls._instance.register_module(module_name, module_class)
                return

            if existing is module_class:
                if cls._instance and module_name not in cls._instance.modules:
                    cls._instance.register_module(module_name, module_class)
                return

            if cls._instance:
                try:
                    cls._instance.logger.warning(
                        f"register_class: name conflict for '{module_name}', "
                        f"keeping {existing} and ignoring {module_class}"
                    )
                except Exception:
                    pass
            return
