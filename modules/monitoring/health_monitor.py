# ─────────────────────────────────────────────────────────────
# File: modules/monitoring/health_monitor.py
# [ROCKET] Production-Grade Health Monitor for SmartInfoBus
# v2.6 — type-hinted net I/O, no direct attr access, robust math
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
import threading
import json
import os
import sys
import traceback
import hashlib
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Deque, Iterable, cast, Protocol
from collections import deque, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
import warnings
import weakref
import uuid
from functools import wraps

# Suppress psutil warnings
warnings.filterwarnings('ignore', module='psutil')

# Optional deps: psutil, numpy
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore[assignment]

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

# Contract registry (optional, contract-first weighting)
try:
    # try common locations; these imports are optional
    from modules.core.contracts_registry import ContractsRegistry  # type: ignore
except Exception:
    try:
        from modules.utils.contracts_registry import ContractsRegistry  # type: ignore
    except Exception:
        ContractsRegistry = None  # type: ignore

# Configuration manager (optional, with graceful fallback)
try:
    from modules.core.configuration_manager import ConfigurationManager  # type: ignore
except Exception:
    ConfigurationManager = None  # type: ignore

from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager


# ─────────────────────────────────────────────────────────────
# Health data models
# ─────────────────────────────────────────────────────────────

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class HealthMetric:
    timestamp: float
    metric_type: str
    value: float
    threshold: float
    status: str
    details: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthReport:
    timestamp: datetime
    overall_status: str
    system_metrics: Dict[str, Any]
    module_health: Dict[str, str]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


# ─────────────────────────────────────────────────────────────
# Utilities: math safety, counters, breaker, rate limiter
# ─────────────────────────────────────────────────────────────

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _safe_mean(values: Iterable[Any]) -> float:
    total = 0.0
    n = 0
    for v in values:
        try:
            total += float(v)
            n += 1
        except Exception:
            continue
    return total / n if n else 0.0

def _safe_percentile(sorted_values: List[float], pct: float) -> float:
    """pct in [0, 100]; expects pre-sorted list."""
    if not sorted_values:
        return 0.0
    pct = max(0.0, min(100.0, pct))
    if NUMPY_AVAILABLE and isinstance(np, object):  # type: ignore[truthy-function]
        try:
            return float(np.percentile(sorted_values, pct))  # type: ignore
        except Exception:
            pass
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return sorted_values[idx]

class ThreadSafeCounter:
    def __init__(self, initial: int = 0):
        self._v = initial
        self._lock = threading.Lock()
    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._v += amount
            return self._v
    def get(self) -> int:
        with self._lock:
            return self._v
    def reset(self) -> None:
        with self._lock:
            self._v = 0


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, timeout: float = 60.0):
        self.failure_threshold = int(failure_threshold)
        self.timeout = float(timeout)
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    def record_success(self) -> None:
        with self._lock:
            self.failure_count = 0
            self.state = "closed"
    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
    def is_open(self) -> bool:
        with self._lock:
            if self.state == "open":
                if self.last_failure_time is not None and (time.time() - self.last_failure_time > self.timeout):
                    self.state = "half-open"
                    return False
                return True
            return False
    def reset(self) -> None:
        with self._lock:
            self.failure_count = 0
            self.state = "closed"
            self.last_failure_time = None


class RateLimiter:
    def __init__(self, max_calls: int = 10, window_seconds: float = 1.0):
        self.max_calls = int(max_calls)
        self.window_seconds = float(window_seconds)
        self.calls: Deque[float] = deque()
        self._lock = threading.Lock()
    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            while self.calls and self.calls[0] < now - self.window_seconds:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False


def validate_input(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, str) and len(arg) > 1000:
                raise ValueError(f"Input too long in {func.__name__}")
        for key, value in kwargs.items():
            if isinstance(value, str) and len(value) > 1000:
                raise ValueError(f"Input {key} too long in {func.__name__}")
        return func(self, *args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────────────────────
# Net I/O typing helper for Pylance
# ─────────────────────────────────────────────────────────────

class _NetCounters(Protocol):
    bytes_sent: int
    bytes_recv: int


# ─────────────────────────────────────────────────────────────
# Health Monitor
# ─────────────────────────────────────────────────────────────

class HealthMonitor:
    """
    Production-grade health monitor with:
    • Config-driven thresholds (hot-reload via ConfigurationManager watcher)
    • Contract-aware module scoring (critical modules weighted higher)
    • Unified operator logging (RotatingLogger) + SmartInfoBus publishing
    • Non-blocking psutil snapshots, safe fallbacks when deps missing
    """

    # Singleton
    _instance: Optional['HealthMonitor'] = None
    _instance_lock = threading.Lock()

    # Lightweight cache for expensive calls
    _cache: Dict[str, Tuple[Any, float]] = {}
    _cache_lock = threading.Lock()
    CACHE_TTL = 5.0

    # Defaults (overridden by config)
    _DEFAULTS: Dict[str, Any] = {
        'thresholds': {
            'cpu_percent': {'warning': 70.0, 'critical': 90.0},
            'memory_percent': {'warning': 75.0, 'critical': 90.0},
            'disk_percent': {'warning': 80.0, 'critical': 95.0},
            'error_rate': {'warning': 0.05, 'critical': 0.10},
            'latency_ms': {'warning': 150.0, 'critical': 300.0},
            'queue_size': {'warning': 1000.0, 'critical': 5000.0}
        },
        'bus_namespace': 'health',
        'publish_interval_s': 15,
        'alert_cooldown_s': 15,
    }

    def __init__(self,
                 orchestrator: Optional[Any] = None,
                 check_interval: int = 30,
                 auto_start: bool = False,
                 config: Optional[Dict[str, Any]] = None):
        self.orchestrator = orchestrator
        self.check_interval = max(1, int(check_interval))
        self._initialized = False
        self._started = False
        self._start_time: Optional[float] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._startup_lock = threading.Lock()

        # SmartInfoBus
        self._smart_bus: Optional[Any] = None

        # Operator logger (RotatingLogger)
        self._oplog = RotatingLogger(
            name="HealthMonitor",
            log_path="logs/monitoring/health.log",
            max_lines=8000,
            operator_mode=True,
            plain_english=True,
            info_bus_aware=True  # safely attaches after bus is alive
        )

        # Process handle (optional)
        self._process: Optional[Any] = None

        # Thread-safety
        self._metrics_lock = threading.RLock()
        self.metrics: Dict[str, Deque[HealthMetric]] = defaultdict(lambda: deque(maxlen=1000))

        # Net I/O trending
        self._last_net_io: Optional[Dict[str, float]] = None
        self._last_net_io_time: Optional[float] = None

        # Config & thresholds (merge defaults + runtime config)
        self._config = self._load_runtime_config(config or {})
        self.thresholds: Dict[str, Dict[str, float]] = dict(self._config['thresholds'])
        self._bus_ns: str = str(self._config.get('bus_namespace', 'health')).strip('/')
        self._alert_cooldown_s: float = float(self._config.get('alert_cooldown_s', 15))
        self._publish_interval_s: int = max(5, int(self._config.get('publish_interval_s', 15)))

        # Alerts
        self._alerts_lock = threading.RLock()
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self._alert_callbacks: weakref.WeakSet = weakref.WeakSet()
        self._cooldowns: Dict[Tuple[str, str], float] = {}

        # Module health scoring
        self._module_health_lock = threading.RLock()
        self.module_health_scores: Dict[str, float] = {}
        self.unhealthy_modules: Set[str] = set()

        # Perf meta
        self._check_count = ThreadSafeCounter()
        self._error_count = ThreadSafeCounter()
        self._last_check_duration = 0.0

        self._circuit_breakers: Dict[str, CircuitBreaker] = defaultdict(
            lambda: CircuitBreaker(failure_threshold=3, timeout=60.0)
        )
        self._rate_limiter = RateLimiter(max_calls=2, window_seconds=1.0)

        # Meta-monitoring
        self._meta_metrics: Dict[str, Deque[float]] = {
            'monitor_cpu_usage': deque(maxlen=100),
            'monitor_memory_usage': deque(maxlen=100),
            'check_durations': deque(maxlen=100)
        }

        # Background publisher
        self._publisher_shutdown = False
        self._publisher_thread: Optional[threading.Thread] = None

        # Auto-wire config hot-reloader (if CM exists)
        self._attach_config_watcher()

        if auto_start:
            self.start()

    # ─────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────
    @property
    def logger(self) -> RotatingLogger:
        # for compatibility with your previous code that used `.logger`
        return self._oplog

    @property
    def smart_bus(self) -> Any:
        if self._smart_bus is None:
            try:
                self._smart_bus = InfoBusManager.get_instance()
                # register provider (best-effort)
                try:
                    if hasattr(self._smart_bus, "register_provider"):
                        self._smart_bus.register_provider(
                            "HealthMonitor",
                            [
                                f"{self._bus_ns}/summary",
                                f"{self._bus_ns}/system",
                                f"{self._bus_ns}/modules",
                                "system_health",
                            ]
                        )
                except Exception:
                    pass
            except Exception as e:
                # Fallback: minimal dummy bus
                self._oplog.warning(format_operator_message(
                    "[WARN]", "SmartInfoBus not available",
                    details=str(e), context="health_monitor"
                ))
                self._smart_bus = self._create_dummy_bus()
        return self._smart_bus

    # ─────────────────────────────────────────────────────────
    # Config
    # ─────────────────────────────────────────────────────────
    def _load_runtime_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = dict(self._DEFAULTS)
        if ConfigurationManager is not None:
            try:
                cm = ConfigurationManager.get_instance()
                # Prefer a dedicated monitoring section if available
                mon: Dict[str, Any] = {}
                if hasattr(cm, "get_monitoring_config"):
                    mon = cm.get_monitoring_config() or {}
                else:
                    syscfg = cm.get_system_config() or {}
                    mon = cast(Dict[str, Any], syscfg.get('monitoring', {}))
                # overlay shallow keys
                for k in ('thresholds', 'bus_namespace', 'publish_interval_s', 'alert_cooldown_s'):
                    if k in mon:
                        cfg[k] = mon[k]
            except Exception:
                pass
        # explicit overrides last
        for k, v in overrides.items():
            cfg[k] = v
        # ensure sub-maps exist
        cfg.setdefault('thresholds', dict(self._DEFAULTS['thresholds']))
        return cfg

    def _attach_config_watcher(self):
        if ConfigurationManager is None:
            return
        try:
            cm = ConfigurationManager.get_instance()
            def _on_cfg_change(name: str, old: Dict[str, Any], new: Dict[str, Any]):
                try:
                    # rebuild thresholds from new config
                    mon: Dict[str, Any] = {}
                    if hasattr(cm, "get_monitoring_config"):
                        mon = cm.get_monitoring_config() or {}
                    else:
                        mon = (new or {}).get('monitoring', {})
                    if mon:
                        self.thresholds = dict(mon.get('thresholds', self.thresholds))
                        self._bus_ns = str(mon.get('bus_namespace', self._bus_ns)).strip('/')
                        self._alert_cooldown_s = float(mon.get('alert_cooldown_s', self._alert_cooldown_s))
                        self._publish_interval_s = max(5, int(mon.get('publish_interval_s', self._publish_interval_s)))
                        self.logger.info(format_operator_message(
                            "[LOG]", "HealthMonitor config hot-reloaded",
                            details=f"ns={self._bus_ns} interval={self._publish_interval_s}s",
                            context="config"
                        ))
                except Exception as e:
                    self.logger.error(f"Config watcher error: {e}")
            cm.add_config_watcher(_on_cfg_change)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, **kwargs) -> 'HealthMonitor':
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    def start(self) -> bool:
        with self._startup_lock:
            if self._started:
                self.logger.info("Health monitor already started")
                return True
            try:
                self._initialize_components()
                self._start_time = time.time()
                self._shutdown_event.clear()
                self._monitor_thread = threading.Thread(
                    target=self._monitoring_loop, name="HealthMonitor", daemon=True
                )
                self._monitor_thread.start()

                # publisher thread
                self._publisher_shutdown = False
                self._publisher_thread = threading.Thread(
                    target=self._publisher_loop, name="HealthPublisher", daemon=True
                )
                self._publisher_thread.start()

                self._started = True
                self.logger.info("Health monitor started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start health monitor: {e}")
                self.logger.error(traceback.format_exc())
                return False

    def stop(self, timeout: float = 5.0) -> bool:
        if not self._started:
            return True
        self.logger.info("[STOP] Stopping health monitor...")
        self._shutdown_event.set()

        # stop publisher
        self._publisher_shutdown = True
        if self._publisher_thread and self._publisher_thread.is_alive():
            self._publisher_thread.join(timeout=1.0)

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout)
            if self._monitor_thread.is_alive():
                self.logger.critical("[ALERT] THREAD LEAK DETECTED: Health monitor thread did not stop")
                if self.orchestrator and hasattr(self.orchestrator, '_report_thread_leak'):
                    try:
                        self.orchestrator._report_thread_leak('HealthMonitor', self._monitor_thread)
                    except Exception as e:
                        self.logger.error(f"Failed to report thread leak: {e}")
                self._monitor_thread = None
                self._started = False
                with self._alerts_lock:
                    alert = {
                        'type': 'thread_leak', 'component': 'HealthMonitor', 'severity': 'critical',
                        'timestamp': time.time(), 'message': f'Monitor thread survived {timeout}s'
                    }
                    self.active_alerts['thread_leak_health_monitor'] = alert
                    self.alert_history.append(alert)
                return False

        self._started = False
        self.logger.info("[OK] Health monitor stopped gracefully")
        return True

    def force_shutdown(self) -> bool:
        self.logger.warning("[WARN] Forcing health monitor shutdown...")
        self._shutdown_event.set()
        self._publisher_shutdown = True
        self._started = False
        self._monitor_thread = None
        self._publisher_thread = None
        with self._alerts_lock:
            self.active_alerts.clear()
        with self._module_health_lock:
            self.module_health_scores.clear()
            self.unhealthy_modules.clear()
        self.logger.warning("[WARN] Health monitor force shutdown complete")
        return True

    def _initialize_components(self) -> None:
        if self._initialized:
            return
        # psutil process and prime CPU meter for non-blocking snapshots
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                self._process = psutil.Process()  # type: ignore
                try:
                    psutil.cpu_percent(interval=None)  # type: ignore
                except Exception:
                    pass
            except Exception as e:
                self.logger.warning(f"Could not initialize process handle: {e}")
                self._process = None
        self._log_system_info()
        self._initialized = True

    def _log_system_info(self) -> None:
        info = {
            'platform': sys.platform,
            'python_version': sys.version.split()[0],
            'psutil_available': PSUTIL_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
        }
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                info.update({
                    'cpu_count': psutil.cpu_count(),  # type: ignore
                    'memory_total_gb': round(cast(float, psutil.virtual_memory().total) / (1024**3), 2),  # type: ignore
                })
            except Exception:
                pass
        self.logger.info(f"System info: {json.dumps(info)}")

    # ─────────────────────────────────────────────────────────
    # Main monitoring loop
    # ─────────────────────────────────────────────────────────
    def _monitoring_loop(self) -> None:
        self.logger.info("Health monitoring loop started")
        time.sleep(2)
        consecutive_errors = 0
        base_interval = self.check_interval

        while not self._shutdown_event.is_set():
            try:
                t0 = time.time()
                self._record_meta_metrics()
                self.check_system_health()
                dur = time.time() - t0
                self._last_check_duration = dur
                self._meta_metrics['check_durations'].append(float(dur))
                consecutive_errors = 0
                self._shutdown_event.wait(base_interval)
            except Exception as e:
                self._error_count.increment()
                consecutive_errors += 1
                self.logger.error(f"Error in monitoring loop: {e}")
                self.logger.error(traceback.format_exc())
                backoff = min(base_interval * (2 ** consecutive_errors), 300)
                self.logger.info(f"Backing off for {backoff}s after {consecutive_errors} errors")
                self._shutdown_event.wait(backoff)
        self.logger.info("Health monitoring loop stopped")

    def _record_meta_metrics(self) -> None:
        if PSUTIL_AVAILABLE and psutil is not None and self._process is not None:
            try:
                cpu = float(self._process.cpu_percent())
                mem = float(self._process.memory_info().rss) / (1024**2)
                self._meta_metrics['monitor_cpu_usage'].append(cpu)
                self._meta_metrics['monitor_memory_usage'].append(mem)
            except Exception:
                self._process = None

    # ─────────────────────────────────────────────────────────
    # Public checks & reports
    # ─────────────────────────────────────────────────────────
    @validate_input
    def check_system_health(self) -> Dict[str, Any]:
        trace_id = str(uuid.uuid4())
        self._check_count.increment()
        health = {
            'timestamp': time.time(),
            'trace_id': trace_id,
            'check_number': self._check_count.get(),
            'system': self._check_system_resources(),
            'modules': self._check_module_health(),
            'infobus': self._check_infobus_health(),
            'performance': self._check_performance_health()
        }
        health['overall_status'] = self._calculate_overall_status(health)
        with self._metrics_lock:
            self._record_health_metrics(health)
        with self._alerts_lock:
            self._check_for_alerts(health)
        self._cleanup_old_metrics()
        return health

    def generate_health_report(self) -> HealthReport:
        health = self.check_system_health()
        recs = self._generate_recommendations(health)
        with self._alerts_lock:
            alerts = list(self.active_alerts.values())
        return HealthReport(
            timestamp=datetime.now(),
            overall_status=health['overall_status'],
            system_metrics=health.get('system', {}),
            module_health={m: info['status'] for m, info in health.get('modules', {}).get('module_details', {}).items()},
            alerts=alerts,
            recommendations=recs,
            trace_id=health.get('trace_id', str(uuid.uuid4()))
        )

    # ─────────────────────────────────────────────────────────
    # Sub-checks
    # ─────────────────────────────────────────────────────────
    def _get_cached(self, key: str, generator: Callable[[], Any], ttl: Optional[float] = None) -> Any:
        ttl_val = float(ttl if ttl is not None else self.CACHE_TTL)
        with self._cache_lock:
            if key in self._cache:
                value, ts = self._cache[key]
                if time.time() - ts < ttl_val:
                    return value
            val = generator()
            self._cache[key] = (val, time.time())
            return val

    def _check_system_resources(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE or psutil is None:
            return {'error': 'psutil not available'}

        breaker = self._circuit_breakers['system_resources']
        if breaker.is_open():
            return {'error': 'Circuit breaker open', 'status': 'degraded'}

        def generate() -> Dict[str, Any]:
            try:
                cpu_percent = float(psutil.cpu_percent(interval=None))  # type: ignore
                if not (0.0 <= cpu_percent <= 100.0):
                    cpu_percent = 0.0

                mem = psutil.virtual_memory()  # type: ignore
                root_path = os.path.abspath(os.sep)
                disk = psutil.disk_usage(root_path)  # type: ignore

                # Net may be None or have unexpected structure; treat via safe getattr
                net: Optional[_NetCounters] = None
                try:
                    net = cast(Optional[_NetCounters], psutil.net_io_counters())  # type: ignore
                except Exception:
                    net = None

                # pull counters safely into locals
                net_sent = float(getattr(net, 'bytes_sent', 0.0)) if net is not None else 0.0
                net_recv = float(getattr(net, 'bytes_recv', 0.0)) if net is not None else 0.0

                rates = {'network_send_rate_mbps': 0.0, 'network_recv_rate_mbps': 0.0}
                now = time.time()
                if net is not None:
                    if self._last_net_io is not None and self._last_net_io_time is not None:
                        dt = now - self._last_net_io_time
                        if dt > 0:
                            prev_sent = float(self._last_net_io.get('bytes_sent', 0.0))
                            prev_recv = float(self._last_net_io.get('bytes_recv', 0.0))
                            ds = net_sent - prev_sent
                            dr = net_recv - prev_recv
                            rates['network_send_rate_mbps'] = round((ds * 8.0) / (dt * 1024.0 * 1024.0), 2)
                            rates['network_recv_rate_mbps'] = round((dr * 8.0) / (dt * 1024.0 * 1024.0), 2)
                    # update last snapshot
                    self._last_net_io = {'bytes_sent': net_sent, 'bytes_recv': net_recv}
                    self._last_net_io_time = now
                else:
                    # If no net, clear previous to avoid misleading deltas later
                    self._last_net_io = None
                    self._last_net_io_time = None

                proc_mem = 0.0
                threads = threading.active_count()
                if self._process is not None:
                    try:
                        proc_mem = float(self._process.memory_info().rss) / (1024.0**2)
                    except Exception:
                        self._process = None

                out: Dict[str, Any] = {
                    'cpu_percent': round(cpu_percent, 2),
                    'memory_percent': round(float(mem.percent), 2),
                    'memory_available_gb': round(float(mem.available) / (1024.0**3), 2),
                    'disk_percent': round(float(disk.percent), 2),
                    'disk_free_gb': round(float(disk.free) / (1024.0**3), 2),
                    'process_memory_mb': round(proc_mem, 2),
                    'thread_count': int(threads),
                    **rates
                }

                if net is not None:
                    out['network_sent_mb'] = round(net_sent / (1024.0**2), 2)
                    out['network_recv_mb'] = round(net_recv / (1024.0**2), 2)

                return out
            except Exception as e:
                self.logger.error(f"Failed to check system resources: {e}")
                return {'error': str(e)}

        try:
            res = cast(Dict[str, Any], self._get_cached("system_resources", generate, ttl=2.0))
            breaker.record_success()
            return res
        except Exception:
            breaker.record_failure()
            return {'error': 'system resource check failed'}

    def _check_module_health(self) -> Dict[str, Any]:
        details: Dict[str, Any] = {}
        unhealthy = 0
        with self._module_health_lock:
            if self.orchestrator and hasattr(self.orchestrator, 'modules'):
                try:
                    modules_items = list(getattr(self.orchestrator, 'modules').items())
                except Exception:
                    modules_items = []
                for name, module in modules_items:
                    info = self._check_single_module_health(str(name), module)
                    details[str(name)] = info
                    if info.get('status') in ['critical', 'error', 'disabled']:
                        unhealthy += 1
                        self.unhealthy_modules.add(str(name))
                    else:
                        self.unhealthy_modules.discard(str(name))
        return {
            'total_modules': len(details),
            'healthy_modules': len(details) - unhealthy,
            'unhealthy_modules': unhealthy,
            'module_details': details
        }

    def _module_criticality_weight(self, module_name: str) -> float:
        # contract-first: try to read criticality from registry
        try:
            if ContractsRegistry:
                contract = ContractsRegistry.get(module_name)  # type: ignore
                if contract:
                    if contract.get('critical', False):
                        return 1.5
                    if contract.get('category') in ('risk', 'trading'):
                        return 1.25
        except Exception:
            pass
        return 1.0

    def _check_single_module_health(self, module_name: str, module: Any) -> Dict[str, Any]:
        try:
            enabled = True
            failures = 0
            status: str = 'unknown'

            # enabled?
            try:
                enabled = bool(self.smart_bus.is_module_enabled(module_name))
            except Exception:
                enabled = True

            # failures?
            try:
                br = getattr(self.smart_bus, "_circuit_breakers", {}).get(module_name)
                if br and hasattr(br, 'failure_count'):
                    failures = int(br.failure_count)
            except Exception:
                pass

            # ask module
            if hasattr(module, 'get_health_status'):
                try:
                    mod_status = module.get_health_status()
                    if isinstance(mod_status, dict):
                        status = str(mod_status.get('status', status))
                except Exception:
                    status = 'error'

            # latency
            avg_latency: Optional[float] = None
            try:
                raw = getattr(self.smart_bus, "_latency_history", {}).get(module_name, [])
                latencies: List[float] = []
                for v in list(raw)[-10:]:
                    if isinstance(v, (int, float)):
                        latencies.append(float(v))
                if latencies:
                    if NUMPY_AVAILABLE and isinstance(np, object):  # type: ignore[truthy-function]
                        avg_latency = float(np.mean(latencies))  # type: ignore
                    else:
                        avg_latency = sum(latencies) / len(latencies)
            except Exception:
                pass

            # compute score
            if not enabled:
                status = 'disabled'
                score = 0.0
            elif failures >= 3:
                status = 'critical'
                score = 0.0
            elif failures > 0:
                status = 'warning'
                score = 0.5
            else:
                status = status if status != 'unknown' else 'healthy'
                score = 1.0

            # apply latency thresholds
            if avg_latency is not None:
                thr = self.thresholds.get(f'module.{module_name}', self.thresholds.get('latency_ms', {'warning':150.0,'critical':300.0}))
                crit = float(thr.get('critical', 300.0))
                warn = float(thr.get('warning', 150.0))
                if avg_latency > crit:
                    status = 'critical'
                    score = min(score, 0.3)
                elif avg_latency > warn:
                    status = 'warning'
                    score = min(score, 0.7)

            # weight by contract criticality
            weight = self._module_criticality_weight(module_name)
            if weight <= 0:
                weight = 1.0
            score = max(0.0, min(1.0, score / weight))  # heavier modules penalize more quickly

            with self._module_health_lock:
                self.module_health_scores[module_name] = float(score)

            out: Dict[str, Any] = {'enabled': enabled, 'failures': failures, 'status': status, 'score': float(score)}
            if avg_latency is not None:
                out['avg_latency_ms'] = round(float(avg_latency), 2)
            return out

        except Exception as e:
            self.logger.error(f"Error checking module {module_name}: {e}")
            return {'status': 'error', 'score': 0.0, 'error': str(e)}

    def _check_infobus_health(self) -> Dict[str, Any]:
        try:
            perf = self.smart_bus.get_performance_metrics()
            cache_hit_rate = _to_float(perf.get('cache_hit_rate', 0.0), 0.0) if isinstance(perf, dict) else 0.0
            active_modules = int(perf.get('active_modules', 0)) if isinstance(perf, dict) else 0
            disabled_list = perf.get('disabled_modules', []) if isinstance(perf, dict) else []
            disabled = int(len(disabled_list)) if isinstance(disabled_list, (list, tuple, set)) else 0
            data_keys = len(getattr(self.smart_bus, "_data_store", {}))
            event_log_size = int(perf.get('total_events', 0)) if isinstance(perf, dict) else 0
            status = self._assess_infobus_status(perf if isinstance(perf, dict) else {})
            return {
                'cache_hit_rate': round(cache_hit_rate, 3),
                'active_modules': active_modules,
                'disabled_modules': disabled,
                'event_log_size': event_log_size,
                'data_keys': data_keys,
                'status': status
            }
        except Exception as e:
            self.logger.error(f"Failed to check InfoBus health: {e}")
            return {'status': 'error', 'error': str(e)}

    def _check_performance_health(self) -> Dict[str, Any]:
        try:
            latencies: List[float] = []
            recent_errors = 0
            if self.orchestrator and hasattr(self.orchestrator, 'modules'):
                for name in list(getattr(self.orchestrator, 'modules').keys()):
                    try:
                        raw = getattr(self.smart_bus, "_latency_history", {}).get(name, [])
                        # last 10; coerce to float and filter
                        tail: List[float] = []
                        for v in list(raw)[-10:]:
                            if isinstance(v, (int, float)):
                                tail.append(float(v))
                        if tail:
                            latencies.extend(tail)
                        br = getattr(self.smart_bus, "_circuit_breakers", {}).get(name)
                        if br and hasattr(br, 'failure_count'):
                            recent_errors += int(br.failure_count)
                    except Exception:
                        pass

            total = len(latencies)
            if total:
                if NUMPY_AVAILABLE and isinstance(np, object):  # type: ignore[truthy-function]
                    avg = float(np.mean(latencies))  # type: ignore
                    mx = float(np.max(latencies))    # type: ignore
                    p95 = float(np.percentile(latencies, 95))  # type: ignore
                else:
                    avg = sum(latencies) / total
                    mx = max(latencies)
                    s = sorted(latencies)
                    p95 = _safe_percentile(s, 95.0)
            else:
                avg = mx = p95 = 0.0

            # meta metrics
            meta: Dict[str, Any] = {}
            if self._meta_metrics['monitor_cpu_usage']:
                meta['monitor_cpu_percent'] = round(_safe_mean(self._meta_metrics['monitor_cpu_usage']), 2)
            if self._meta_metrics['monitor_memory_usage']:
                meta['monitor_memory_mb'] = round(_safe_mean(self._meta_metrics['monitor_memory_usage']), 2)

            uptime = (time.time() - self._start_time) if self._start_time else 0.0
            return {
                'avg_latency_ms': round(avg, 2),
                'max_latency_ms': round(mx, 2),
                'p95_latency_ms': round(p95, 2),
                'error_rate': round(float(recent_errors) / float(max(total, 1)), 4),
                'throughput_per_min': int(total * 2),  # ~ checks per 30s window x2
                'monitor_uptime_seconds': round(float(uptime), 2),
                'checks_performed': self._check_count.get(),
                'monitor_errors': self._error_count.get(),
                **meta
            }
        except Exception as e:
            self.logger.error(f"Failed to check performance health: {e}")
            return {'error': str(e)}

    # ─────────────────────────────────────────────────────────
    # Status computation & metrics recording
    # ─────────────────────────────────────────────────────────
    def _calculate_overall_status(self, health: Dict[str, Any]) -> str:
        statuses: List[str] = []

        system = health.get('system', {})
        if isinstance(system, dict):
            for metric, thr in [('cpu_percent', self.thresholds.get('cpu_percent', {})),
                                ('memory_percent', self.thresholds.get('memory_percent', {})),
                                ('disk_percent', self.thresholds.get('disk_percent', {}))]:
                v = _to_float(system.get(metric, 0.0), 0.0)
                crit = _to_float(thr.get('critical', 1e9), 1e9)
                warn = _to_float(thr.get('warning', 1e9), 1e9)
                if v >= crit:
                    statuses.append('critical')
                elif v >= warn:
                    statuses.append('warning')

        modules = health.get('modules', {})
        if isinstance(modules, dict):
            total = int(modules.get('total_modules', 1) or 1)
            unhealthy = int(modules.get('unhealthy_modules', 0) or 0)
            ratio = float(unhealthy) / float(max(total, 1))
            if ratio > 0.3:
                statuses.append('critical')
            elif ratio > 0.1:
                statuses.append('warning')

        perf = health.get('performance', {})
        if isinstance(perf, dict):
            er = _to_float(perf.get('error_rate', 0.0), 0.0)
            thr = self.thresholds.get('error_rate', {'warning': 0.05, 'critical': 0.1})
            if er >= _to_float(thr.get('critical', 0.1), 0.1):
                statuses.append('critical')
            elif er >= _to_float(thr.get('warning', 0.05), 0.05):
                statuses.append('warning')

        if 'critical' in statuses:
            return HealthStatus.CRITICAL.value
        if 'warning' in statuses:
            return HealthStatus.WARNING.value
        return HealthStatus.HEALTHY.value

    def _record_health_metrics(self, health: Dict[str, Any]) -> None:
        ts = _to_float(health.get('timestamp', time.time()), time.time())
        trace = health.get('trace_id')

        system = health.get('system', {})
        if isinstance(system, dict):
            for name, value in system.items():
                if isinstance(value, (int, float)):
                    valf = float(value)
                    self.metrics[f'system.{name}'].append(
                        HealthMetric(
                            timestamp=ts,
                            metric_type=name,
                            value=valf,
                            threshold=_to_float(self.thresholds.get(name, {}).get('critical', float('inf')), float('inf')),
                            status=self._metric_status(name, valf),
                            trace_id=trace
                        )
                    )

        with self._module_health_lock:
            snapshot = list(self.module_health_scores.items())
        for mod, score in snapshot:
            scoref = float(score)
            self.metrics[f'module.{mod}.score'].append(
                HealthMetric(
                    timestamp=ts,
                    metric_type='health_score',
                    value=scoref,
                    threshold=0.5,
                    status='healthy' if scoref > 0.7 else 'warning' if scoref > 0.3 else 'critical',
                    trace_id=trace
                )
            )

    def _metric_status(self, metric: str, value: float) -> str:
        thr = self.thresholds.get(metric)
        if not thr:
            return HealthStatus.HEALTHY.value
        crit = _to_float(thr.get('critical', 1e9), 1e9)
        warn = _to_float(thr.get('warning', 1e9), 1e9)
        if value >= crit:
            return HealthStatus.CRITICAL.value
        if value >= warn:
            return HealthStatus.WARNING.value
        return HealthStatus.HEALTHY.value

    def _cleanup_old_metrics(self, max_age_seconds: int = 24 * 3600) -> None:
        """
        Prune old in-memory metrics and stale cooldown entries to keep memory bounded.
        """
        cutoff = time.time() - float(max_age_seconds)

        # Remove old HealthMetric entries and empty metric keys
        with self._metrics_lock:
            empty_keys: List[str] = []
            for key, dq in list(self.metrics.items()):
                while dq and dq[0].timestamp < cutoff:
                    dq.popleft()
                if not dq:
                    empty_keys.append(key)
            for key in empty_keys:
                try:
                    del self.metrics[key]
                except Exception:
                    pass

        # Clean up stale cooldown entries
        with self._alerts_lock:
            try:
                now = time.time()
                ttl = max(2 * self._alert_cooldown_s, 3600.0)  # at least 1 hour
                stale = [k for k, ts in list(self._cooldowns.items()) if (now - ts) > ttl]
                for k in stale:
                    self._cooldowns.pop(k, None)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────
    # Alerts
    # ─────────────────────────────────────────────────────────
    def _cooldown_allows(self, key: Tuple[str, str]) -> bool:
        now = time.time()
        last = _to_float(self._cooldowns.get(key, 0.0), 0.0)
        if now - last >= self._alert_cooldown_s:
            self._cooldowns[key] = now
            return True
        return False

    def _check_for_alerts(self, health: Dict[str, Any]) -> None:
        to_trigger: List[Dict[str, Any]] = []

        system = health.get('system', {})
        if isinstance(system, dict):
            for metric, value in system.items():
                if metric in self.thresholds and isinstance(value, (int, float)):
                    status = self._metric_status(metric, float(value))
                    if status != HealthStatus.HEALTHY.value:
                        key = f'system.{metric}'
                        if key not in self.active_alerts:
                            thr_map = self.thresholds.get(metric, {})
                            threshold_val = _to_float(thr_map.get(status, thr_map.get('warning', 0.0)), 0.0)
                            alert = {
                                'type': 'system_resource',
                                'metric': metric,
                                'value': float(value),
                                'threshold': threshold_val,
                                'status': status,
                                'timestamp': time.time(),
                                'trace_id': health.get('trace_id')
                            }
                            self.active_alerts[key] = alert
                            to_trigger.append(alert)

        modules = health.get('modules', {})
        if isinstance(modules, dict):
            details = modules.get('module_details', {})
            if isinstance(details, dict):
                for mod, info in details.items():
                    s = info.get('status')
                    if s in ['critical', 'error']:
                        key = f'module.{mod}'
                        if key not in self.active_alerts:
                            alert = {
                                'type': 'module_health',
                                'module': mod,
                                'status': s,
                                'failures': int(info.get('failures', 0) or 0),
                                'timestamp': time.time(),
                                'trace_id': health.get('trace_id')
                            }
                            self.active_alerts[key] = alert
                            to_trigger.append(alert)

        for a in to_trigger:
            self._trigger_alert(a)
        self._clear_resolved_alerts(health)

    def _clear_resolved_alerts(self, health: Dict[str, Any]) -> None:
        resolved: List[str] = []

        system = health.get('system', {})
        if isinstance(system, dict):
            for metric in self.thresholds:
                key = f'system.{metric}'
                if key in self.active_alerts:
                    v = system.get(metric)
                    if isinstance(v, (int, float)) and self._metric_status(metric, float(v)) == HealthStatus.HEALTHY.value:
                        resolved.append(key)

        modules = health.get('modules', {})
        if isinstance(modules, dict):
            details = modules.get('module_details', {})
            if isinstance(details, dict):
                for mod, info in details.items():
                    key = f'module.{mod}'
                    if key in self.active_alerts and info.get('status') not in ['critical', 'error']:
                        resolved.append(key)

        for k in resolved:
            self.active_alerts.pop(k, None)
            self.logger.info(f"Alert resolved: {k}")

    def _trigger_alert(self, alert: Dict[str, Any]) -> None:
        self.alert_history.append(alert)
        key = ('alert', str(alert.get('type', 'unknown')))
        if not self._cooldown_allows(key):
            return
        # operator log
        msg = f"[ALERT] {alert.get('type')} - {alert.get('metric') or alert.get('module', 'unknown')} - {alert.get('status','unknown')}"
        self.logger.warning(msg)
        # callbacks
        for cb in list(self._alert_callbacks):
            try:
                cb(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
        # publish to bus (best-effort)
        try:
            self.smart_bus.set(
                f"{self._bus_ns}/alerts",
                list(self.alert_history)[-20:],
                module="HealthMonitor",
                thesis="Recent health alerts"
            )
        except Exception:
            pass

    @validate_input
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        if not callable(callback):
            raise ValueError("Callback must be callable")
        import inspect
        if len(inspect.signature(callback).parameters) != 1:
            raise ValueError("Callback must accept exactly one parameter")
        self._alert_callbacks.add(callback)

    # ─────────────────────────────────────────────────────────
    # Trends, export, status
    # ─────────────────────────────────────────────────────────
    @validate_input
    def get_health_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        # Defensive coerce to int to satisfy the type checker and callers who send floats/strings
        try:
            hours_val = int(hours)
        except Exception:
            raise ValueError("Hours must be an integer")
        if hours_val < 0 or hours_val > 168:
            raise ValueError("Hours must be between 0 and 168")

        cutoff = time.time() - (hours_val * 3600.0)
        with self._metrics_lock:
            if metric_name not in self.metrics:
                return {'error': 'Metric not found'}
            ms = [m for m in self.metrics[metric_name] if m.timestamp > cutoff]
        if not ms:
            return {'error': 'No data in time range'}

        vals: List[float] = [float(m.value) for m in ms]
        if NUMPY_AVAILABLE and isinstance(np, object):  # type: ignore[truthy-function]
            try:
                avg = float(np.mean(vals))  # type: ignore
                minimum = float(np.min(vals))  # type: ignore
                maximum = float(np.max(vals))  # type: ignore
                std = float(np.std(vals))  # type: ignore
            except Exception:
                avg = _safe_mean(vals); minimum = min(vals); maximum = max(vals)
                var = _safe_mean([(x - avg) ** 2 for x in vals]); std = var ** 0.5
        else:
            avg = _safe_mean(vals)
            minimum = min(vals); maximum = max(vals)
            var = _safe_mean([(x - avg) ** 2 for x in vals]); std = var ** 0.5

        return {
            'metric': metric_name,
            'period_hours': hours_val,
            'data_points': len(vals),
            'current': vals[-1],
            'average': round(avg, 3),
            'minimum': round(minimum, 3),
            'maximum': round(maximum, 3),
            'std_deviation': round(std, 3),
            'trend': self._calc_trend(vals)
        }

    def _calc_trend(self, values: List[float]) -> str:
        if len(values) < 10:
            return 'insufficient_data'
        third = max(1, len(values)//3)
        a = values[:third]; b = values[-third:]
        if NUMPY_AVAILABLE and isinstance(np, object):  # type: ignore[truthy-function]
            try:
                fa = float(np.mean(a)); fb = float(np.mean(b))  # type: ignore
            except Exception:
                fa = _safe_mean(a); fb = _safe_mean(b)
        else:
            fa = _safe_mean(a); fb = _safe_mean(b)
        base = max(abs(fa), 1.0)
        change = (fb - fa) / base * 100.0
        if change > 10.0: return 'increasing'
        if change < -10.0: return 'decreasing'
        return 'stable'

    @validate_input
    def export_health_data(self, filepath: str) -> bool:
        tmp_name: Optional[str] = None
        try:
            if not filepath or '..' in filepath:
                raise ValueError("Invalid filepath")
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            current = self.check_system_health()
            with self._alerts_lock:
                alerts = list(self.active_alerts.values())
                history = list(self.alert_history)[-100:]
            with self._module_health_lock:
                scores = dict(self.module_health_scores)
            uptime = (time.time() - self._start_time) if self._start_time else 0.0
            avg_check_ms = _safe_mean(self._meta_metrics['check_durations']) * 1000.0

            data: Dict[str, Any] = {
                'export_time': datetime.now().isoformat(),
                'current_health': current,
                'active_alerts': alerts,
                'alert_history': history,
                'module_scores': scores,
                'recommendations': self._generate_recommendations(current),
                'monitor_stats': {
                    'checks_performed': self._check_count.get(),
                    'errors_encountered': self._error_count.get(),
                    'uptime_seconds': round(float(uptime), 2),
                    'is_running': self._started,
                    'meta_metrics': {
                        'avg_check_duration_ms': round(avg_check_ms, 3)
                    }
                }
            }
            js = json.dumps(data, indent=2, default=str)
            checksum = hashlib.sha256(js.encode()).hexdigest()
            data['checksum'] = checksum
            with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(filepath) or '.', delete=False) as tmp:
                json.dump(data, tmp, indent=2, default=str)
                tmp_name = tmp.name
            shutil.move(tmp_name, filepath)
            self.logger.info(f"Health data exported to {filepath} (checksum: {checksum[:8]}...)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export health data: {e}")
            if tmp_name:
                try: os.unlink(tmp_name)
                except Exception: pass
            return False

    def get_status(self) -> Dict[str, Any]:
        with self._metrics_lock:
            tracked = int(len(self.metrics))
        uptime = (time.time() - self._start_time) if self._start_time else 0.0
        return {
            'initialized': self._initialized,
            'running': self._started,
            'checks_performed': self._check_count.get(),
            'errors_encountered': self._error_count.get(),
            'last_check_duration_ms': float(self._last_check_duration) * 1000.0,
            'active_alerts': len(self.active_alerts),
            'unhealthy_modules': len(self.unhealthy_modules),
            'thread_alive': (self._monitor_thread.is_alive() if self._monitor_thread else False),
            'uptime_seconds': round(float(uptime), 2),
            'circuit_breakers': {name: br.state for name, br in self._circuit_breakers.items()},
            'cache_size': len(self._cache),
            'metrics_tracked': tracked
        }

    def __del__(self):
        try:
            if self._started:
                self.stop(timeout=1.0)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # Recommendations & bus helpers
    # ─────────────────────────────────────────────────────────
    def _generate_recommendations(self, health: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        system = health.get('system', {})
        if isinstance(system, dict):
            cpu = _to_float(system.get('cpu_percent', 0.0), 0.0)
            if cpu >= _to_float(self.thresholds['cpu_percent']['warning'], 70.0):
                recs.append(f"High CPU usage ({cpu:.1f}%) - optimize compute-heavy modules")
            mem = _to_float(system.get('memory_percent', 0.0), 0.0)
            if mem >= _to_float(self.thresholds['memory_percent']['warning'], 75.0):
                recs.append(f"High memory usage ({mem:.1f}%) - check for leaks")
        with self._module_health_lock:
            if self.unhealthy_modules:
                u = list(self.unhealthy_modules)[:5]
                recs.append(f"Unhealthy modules detected: {', '.join(u)}")
                recs.append("Consider restarting or investigating these modules")
        perf = health.get('performance', {})
        if isinstance(perf, dict):
            avg = _to_float(perf.get('avg_latency_ms', 0.0), 0.0)
            if avg > 100.0:
                recs.append(f"High average latency ({avg:.0f}ms) - review module performance")
            er = _to_float(perf.get('error_rate', 0.0), 0.0)
            if er > 0.05:
                recs.append(f"High error rate ({er:.1%}) - investigate failing modules")
        bus = health.get('infobus', {})
        if isinstance(bus, dict) and _to_float(bus.get('cache_hit_rate', 1.0), 1.0) < 0.7:
            recs.append("Low cache hit rate - consider increasing cache size/TTL")
        if _to_float(perf.get('monitor_cpu_percent', 0.0), 0.0) > 5.0:
            recs.append("Health monitor using excessive CPU - increase check interval")
        return recs

    def _assess_infobus_status(self, metrics: Dict[str, Any]) -> str:
        if _to_float(metrics.get('cache_hit_rate', 0.0), 0.0) < 0.5:
            return HealthStatus.WARNING.value
        disabled = metrics.get('disabled_modules', [])
        if isinstance(disabled, (list, tuple, set)) and len(disabled) > 3:
            return HealthStatus.CRITICAL.value
        return HealthStatus.HEALTHY.value

    def _create_dummy_bus(self) -> Any:
        class DummyBus:
            def set(self, *_, **__): pass
            def get(self, *_, **__): return None
            def get_performance_metrics(self): return {'dummy_bus': True}
            def is_module_enabled(self, module: str) -> bool: return True
        return DummyBus()

    def _publisher_loop(self):
        # periodic lightweight publish for dashboards
        while not self._publisher_shutdown:
            try:
                snap = self.get_status()
                health = self.check_system_health()
                # summary
                self.smart_bus.set(
                    f"{self._bus_ns}/summary",
                    {'timestamp': time.time(), 'overall': health.get('overall_status'),
                     'checks': snap['checks_performed'], 'errors': snap['errors_encountered']},
                    module="HealthMonitor",
                    thesis="System health summary"
                )
                # system
                self.smart_bus.set(
                    f"{self._bus_ns}/system",
                    health.get('system', {}),
                    module="HealthMonitor",
                    thesis="System resource snapshot"
                )
                # modules (trim to 100 for safety)
                mods = health.get('modules', {}).get('module_details', {})
                if isinstance(mods, dict):
                    self.smart_bus.set(
                        f"{self._bus_ns}/modules",
                        {k: v for k, v in list(mods.items())[:100]},
                        module="HealthMonitor",
                        thesis="Module health snapshot"
                    )
                # canonical consolidated health surface for consumers
                try:
                    consolidated = {
                        'overall_status': health.get('overall_status'),
                        'system': health.get('system', {}),
                        'modules': mods if isinstance(mods, dict) else {},
                        'performance': health.get('performance', {}),
                        'checks_performed': snap.get('checks_performed'),
                        'errors_encountered': snap.get('errors_encountered'),
                        'timestamp': time.time(),
                    }
                    self.smart_bus.set(
                        'system_health',
                        consolidated,
                        module='HealthMonitor',
                        thesis='Canonical system health snapshot'
                    )
                except Exception:
                    pass
            except Exception:
                pass
            time.sleep(self._publish_interval_s)
