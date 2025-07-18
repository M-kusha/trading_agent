# ─────────────────────────────────────────────────────────────
# File: modules/monitoring/health_monitor.py
# [ROCKET] Production-Grade Lazy Health Monitor for SmartInfoBus
# ─────────────────────────────────────────────────────────────

import time
import threading
import logging
import json
import os
import sys
import traceback
from typing import Dict, List, Any, Optional, Callable, Set, Union
from collections import deque, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
import warnings

# Suppress psutil warnings
warnings.filterwarnings('ignore', module='psutil')

# Try to import psutil, but don't fail if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore

# Try to import numpy for calculations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class HealthMetric:
    """Single health measurement"""
    timestamp: float
    metric_type: str
    value: float
    threshold: float
    status: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class HealthReport:
    """Comprehensive health report"""
    timestamp: datetime
    overall_status: str
    system_metrics: Dict[str, float]
    module_health: Dict[str, str]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ThreadSafeCounter:
    """Thread-safe counter implementation"""
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value
    
    def get(self) -> int:
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        with self._lock:
            self._value = 0


class HealthMonitor:
    """
    Production-grade health monitor with lazy initialization.
    No blocking operations during import or initialization.
    """
    
    # Class-level singleton instance
    _instance: Optional['HealthMonitor'] = None
    _instance_lock = threading.Lock()
    
    def __init__(self, orchestrator: Optional[Any] = None,
                 check_interval: int = 30,
                 auto_start: bool = False):
        """
        Initialize health monitor without blocking operations.
        
        Args:
            orchestrator: Optional orchestrator instance
            check_interval: Seconds between health checks
            auto_start: Whether to start monitoring automatically (default: False)
        """
        # Basic configuration
        self.orchestrator = orchestrator
        self.check_interval = max(1, check_interval)  # Minimum 1 second
        self._initialized = False
        self._started = False
        
        # Thread management
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._startup_lock = threading.Lock()
        
        # Lazy-loaded components
        self._smart_bus: Optional[Any] = None
        self._logger: Optional[logging.Logger] = None
        self._process: Optional[Any] = None
        
        # Metrics storage (thread-safe)
        self._metrics_lock = threading.Lock()
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Thresholds configuration
        self.thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 75, 'critical': 90},
            'disk_percent': {'warning': 80, 'critical': 95},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'latency_ms': {'warning': 150, 'critical': 300},
            'queue_size': {'warning': 1000, 'critical': 5000}
        }
        
        # Alert management (thread-safe)
        self._alerts_lock = threading.Lock()
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Module health tracking (thread-safe)
        self._module_health_lock = threading.Lock()
        self.module_health_scores: Dict[str, float] = {}
        self.unhealthy_modules: Set[str] = set()
        
        # Performance tracking
        self._check_count = ThreadSafeCounter()
        self._error_count = ThreadSafeCounter()
        self._last_check_duration = 0.0
        
        # Initialize if auto_start is True
        if auto_start:
            self.start()
    
    @classmethod
    def get_instance(cls, **kwargs) -> 'HealthMonitor':
        """Get or create singleton instance"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance
    
    @property
    def logger(self) -> logging.Logger:
        """Lazy-load logger"""
        if self._logger is None:
            self._logger = self._create_logger()
        return self._logger
    
    @property
    def smart_bus(self) -> Any:
        """Lazy-load SmartInfoBus"""
        if self._smart_bus is None:
            try:
                from modules.utils.info_bus import InfoBusManager
                self._smart_bus = InfoBusManager.get_instance()
            except Exception as e:
                self.logger.warning(f"SmartInfoBus not available: {e}")
                # Create a dummy bus
                self._smart_bus = self._create_dummy_bus()
        return self._smart_bus
    
    def _create_logger(self) -> logging.Logger:
        """Create a proper logger"""
        logger = logging.getLogger('HealthMonitor')
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # File handler with rotation
            try:
                os.makedirs('logs/monitoring', exist_ok=True)
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    'logs/monitoring/health.log',
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
                file_handler.setLevel(logging.DEBUG)
                
                # Formatter
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(formatter)
                file_handler.setFormatter(formatter)
                
                logger.addHandler(console_handler)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not set up file logging: {e}")
                logger.addHandler(console_handler)
            
            logger.setLevel(logging.DEBUG)
        
        return logger
    
    def _create_dummy_bus(self) -> Any:
        """Create a dummy SmartInfoBus for fallback with critical logging"""
        
        class DummyBus:
            def __init__(self):
                self._logger = logging.getLogger("DummyBus")
                self._logger.critical("[ALERT] USING DUMMY BUS - SMARTINFOBUS INSTRUMENTATION FAILED")
                self._call_count = 0
                self._logged_critical = False
            
            def get(self, key: str, module: Optional[str] = None) -> Any:
                self._call_count += 1
                if self._call_count == 1:  # Log only first call
                    self._logger.error(f"[FAIL] DummyBus.get() called - data unavailable for '{key}' (module: {module})")
                return None
            
            def set(self, key: str, value: Any, module: Optional[str] = None, thesis: Optional[str] = None) -> None:
                self._call_count += 1
                if self._call_count <= 5:  # Log first 5 calls
                    self._logger.warning(f"[WARN] DummyBus.set() called - data LOST for '{key}' (module: {module})")
                elif self._call_count == 6:
                    self._logger.error("[ALERT] DummyBus receiving more calls - further data loss not logged")
            
            def get_performance_metrics(self) -> Dict[str, Any]:
                if not self._logged_critical:
                    self._logger.critical("[CRASH] Performance metrics unavailable - DummyBus active")
                    self._logged_critical = True
                return {
                    'dummy_bus_active': True,
                    'data_loss_calls': self._call_count,
                    'status': 'INSTRUMENTATION_FAILED'
                }
            
            @property
            def _circuit_breakers(self) -> Dict:
                return {}
            
            @property
            def _latency_history(self) -> Dict:
                return {}
            
            @property
            def _module_disabled(self) -> Set:
                return set()
            
            @property
            def _data_store(self) -> Dict:
                return {}
            
            def is_module_enabled(self, module: str) -> bool:
                return True
        
        return DummyBus()
    
    def start(self) -> bool:
        """
        Start health monitoring in a separate thread.
        Safe to call multiple times.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        with self._startup_lock:
            if self._started:
                self.logger.info("Health monitor already started")
                return True
            
            try:
                # Initialize components
                self._initialize_components()
                
                # Clear shutdown event
                self._shutdown_event.clear()
                
                # Start monitoring thread
                self._monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    name="HealthMonitor",
                    daemon=True
                )
                self._monitor_thread.start()
                
                self._started = True
                self.logger.info("Health monitor started successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to start health monitor: {e}")
                self.logger.debug(traceback.format_exc())
                return False
    def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop health monitoring with guaranteed thread termination.
        
        Args:
            timeout: Maximum time to wait for thread to stop
            
        Returns:
            bool: True if stopped successfully, False if thread leaked
        """
        if not self._started:
            return True
        
        self.logger.info("[STOP] Stopping health monitor...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for thread to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout)
            
            if self._monitor_thread.is_alive():
                # ENHANCED: Escalate thread leak to orchestrator
                self.logger.critical(
                    f"[ALERT] THREAD LEAK DETECTED: Health monitor thread survived {timeout}s timeout"
                )
                
                # Report to orchestrator if available
                if self.orchestrator and hasattr(self.orchestrator, '_report_thread_leak'):
                    try:
                        self.orchestrator._report_thread_leak('HealthMonitor', self._monitor_thread)
                    except Exception as e:
                        self.logger.error(f"Failed to report thread leak: {e}")
                
                # Mark thread as zombie
                self._monitor_thread = None
                self._started = False
                
                # Set alert for operators
                with self._alerts_lock:
                    leak_alert = {
                        'type': 'thread_leak',
                        'component': 'HealthMonitor',
                        'severity': 'critical',
                        'timestamp': time.time(),
                        'message': f'Monitor thread survived {timeout}s shutdown timeout'
                    }
                    self.active_alerts['thread_leak_health_monitor'] = leak_alert
                    self.alert_history.append(leak_alert)
                
                return False
        
        self._started = False
        self.logger.info("[OK] Health monitor stopped gracefully")
        return True

    def force_shutdown(self) -> bool:
        """
        Force shutdown with aggressive thread termination.
        Use only as last resort when normal stop() fails.
        
        Returns:
            bool: True if forced shutdown completed
        """
        self.logger.warning("[WARN] Forcing health monitor shutdown...")
        
        # Set shutdown flag
        self._shutdown_event.set()
        self._started = False
        
        # Abandon the thread (mark as zombie)
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.critical(
                f"🧟 Abandoning zombie thread: {self._monitor_thread.name} "
                f"(ID: {self._monitor_thread.ident})"
            )
            self._monitor_thread = None
        
        # Clear all state
        with self._alerts_lock:
            self.active_alerts.clear()
        
        with self._module_health_lock:
            self.module_health_scores.clear()
            self.unhealthy_modules.clear()
        
        self.logger.warning("[WARN] Health monitor force shutdown complete")
        return True
    
    def _initialize_components(self) -> None:
        """Initialize lazy components"""
        if self._initialized:
            return
        
        # Initialize process handle for psutil
        if PSUTIL_AVAILABLE:
            try:
                self._process = psutil.Process()  # type: ignore
            except Exception as e:
                self.logger.warning(f"Could not initialize process handle: {e}")
                self._process = None
        
        # Log system info
        self._log_system_info()
        
        self._initialized = True
    
    def _log_system_info(self) -> None:
        """Log system information"""
        try:
            info = {
                'platform': sys.platform,
                'python_version': sys.version.split()[0],
                'psutil_available': PSUTIL_AVAILABLE,
                'numpy_available': NUMPY_AVAILABLE,
            }
            
            if PSUTIL_AVAILABLE:
                info.update({
                    'cpu_count': psutil.cpu_count(),  # type: ignore
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),  # type: ignore
                })
            
            self.logger.info(f"System info: {json.dumps(info, indent=2)}")
            
        except Exception as e:
            self.logger.warning(f"Could not log system info: {e}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop - runs in separate thread"""
        self.logger.info("Health monitoring loop started")
        
        # Initial delay to let system stabilize
        time.sleep(2)
        
        while not self._shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Perform health check
                self.check_system_health()
                
                # Record duration
                self._last_check_duration = time.time() - start_time
                
                # Wait for next check or shutdown
                self._shutdown_event.wait(self.check_interval)
                
            except Exception as e:
                self._error_count.increment()
                self.logger.error(f"Error in monitoring loop: {e}")
                self.logger.debug(traceback.format_exc())
                
                # Back off on errors
                self._shutdown_event.wait(min(self.check_interval * 2, 60))
        
        self.logger.info("Health monitoring loop stopped")
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        Non-blocking and thread-safe.
        """
        self._check_count.increment()
        
        health_data = {
            'timestamp': time.time(),
            'check_number': self._check_count.get(),
            'system': self._check_system_resources(),
            'modules': self._check_module_health(),
            'infobus': self._check_infobus_health(),
            'performance': self._check_performance_health()
        }
        
        # Calculate overall status
        health_data['overall_status'] = self._calculate_overall_status(health_data)
        
        # Record metrics
        with self._metrics_lock:
            self._record_health_metrics(health_data)
        
        # Check for alerts
        with self._alerts_lock:
            self._check_for_alerts(health_data)
        
        return health_data
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage (non-blocking)"""
        if not PSUTIL_AVAILABLE:
            return {'error': 'psutil not available'}
        
        try:
            if not PSUTIL_AVAILABLE or psutil is None:
                return {'error': 'psutil not available'}
            
            # FIX: CPU check with proper interval
            cpu_percent = psutil.cpu_percent(interval=0.1)  # 0.1s interval instead of 0
            
            memory = psutil.virtual_memory()  # type: ignore
            disk = psutil.disk_usage('/')  # type: ignore
            
            # Network I/O
            net_io = psutil.net_io_counters()  # type: ignore
            
            # Process-specific metrics
            process_memory = 0
            thread_count = threading.active_count()
            
            if self._process:
                try:
                    process_memory = self._process.memory_info().rss / (1024**2)
                except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore
                    # Process might have been terminated
                    self._process = None
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'process_memory_mb': process_memory,
                'network_sent_mb': net_io.bytes_sent / (1024**2),
                'network_recv_mb': net_io.bytes_recv / (1024**2),
                'thread_count': thread_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check system resources: {e}")
            return {'error': str(e)}
    
    def _check_module_health(self) -> Dict[str, Any]:
        """Check health of all modules"""
        with self._module_health_lock:
            module_health = {}
            unhealthy_count = 0
            
            if self.orchestrator and hasattr(self.orchestrator, 'modules'):
                for module_name, module in self.orchestrator.modules.items():
                    health_info = self._check_single_module_health(module_name, module)
                    module_health[module_name] = health_info
                    
                    if health_info['status'] in ['critical', 'error', 'disabled']:
                        unhealthy_count += 1
                        self.unhealthy_modules.add(module_name)
                    else:
                        self.unhealthy_modules.discard(module_name)
            
            return {
                'total_modules': len(module_health),
                'healthy_modules': len(module_health) - unhealthy_count,
                'unhealthy_modules': unhealthy_count,
                'module_details': module_health
            }
    
    def _check_single_module_health(self, module_name: str, module: Any) -> Dict[str, Any]:
        """Check health of a single module"""
        try:
            health_info = {
                'enabled': self.smart_bus.is_module_enabled(module_name),
                'failures': 0,
                'status': 'unknown'
            }
            
            # Check circuit breaker failures
            if hasattr(self.smart_bus, '_circuit_breakers'):
                breaker = self.smart_bus._circuit_breakers.get(module_name)
                if breaker and hasattr(breaker, 'failure_count'):
                    health_info['failures'] = breaker.failure_count
            
            # Get module-specific health if available
            if hasattr(module, 'get_health_status'):
                try:
                    module_status = module.get_health_status()
                    health_info.update(module_status)
                except Exception:
                    health_info['status'] = 'error'
            
            # Calculate health score
            if not health_info['enabled']:
                health_info['status'] = 'disabled'
                health_info['score'] = 0
            elif health_info['failures'] >= 3:
                health_info['status'] = 'critical'
                health_info['score'] = 0
            elif health_info['failures'] > 0:
                health_info['status'] = 'warning'
                health_info['score'] = 0.5
            else:
                health_info['status'] = health_info.get('status', 'healthy')
                health_info['score'] = 1.0
            
            # Check latency
            if hasattr(self.smart_bus, '_latency_history'):
                latencies = list(self.smart_bus._latency_history.get(module_name, []))
                if latencies:
                    if NUMPY_AVAILABLE:
                        avg_latency = np.mean(latencies[-10:])  # type: ignore
                    else:
                        avg_latency = sum(latencies[-10:]) / len(latencies[-10:])
                    
                    health_info['avg_latency_ms'] = avg_latency
                    
                    if avg_latency > self.thresholds['latency_ms']['critical']:
                        health_info['status'] = 'critical'
                        health_info['score'] = min(health_info['score'], 0.3)
                    elif avg_latency > self.thresholds['latency_ms']['warning']:
                        health_info['status'] = 'warning'
                        health_info['score'] = min(health_info['score'], 0.7)
            
            self.module_health_scores[module_name] = health_info['score']
            return health_info
            
        except Exception as e:
            self.logger.error(f"Error checking module {module_name}: {e}")
            return {'status': 'error', 'score': 0, 'error': str(e)}
    
    def _check_infobus_health(self) -> Dict[str, Any]:
        """Check SmartInfoBus health"""
        try:
            perf_metrics = self.smart_bus.get_performance_metrics()
            
            # Calculate queue sizes
            event_log_size = perf_metrics.get('total_events', 0)
            
            return {
                'cache_hit_rate': perf_metrics.get('cache_hit_rate', 0),
                'active_modules': perf_metrics.get('active_modules', 0),
                'disabled_modules': len(perf_metrics.get('disabled_modules', [])),
                'event_log_size': event_log_size,
                'data_keys': len(self.smart_bus._data_store),
                'status': self._assess_infobus_status(perf_metrics)
            }
        except Exception as e:
            self.logger.error(f"Failed to check InfoBus health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _check_performance_health(self) -> Dict[str, Any]:
        """Check overall performance health"""
        try:
            recent_latencies = []
            recent_errors = 0
            
            if self.orchestrator and hasattr(self.orchestrator, 'modules'):
                for module_name in self.orchestrator.modules:
                    # Latencies
                    if hasattr(self.smart_bus, '_latency_history'):
                        latencies = list(self.smart_bus._latency_history.get(module_name, []))
                        if latencies:
                            recent_latencies.extend(latencies[-10:])
                    
                    # Errors
                    if hasattr(self.smart_bus, '_circuit_breakers'):
                        breaker = self.smart_bus._circuit_breakers.get(module_name)
                        if breaker and hasattr(breaker, 'failure_count'):
                            recent_errors += breaker.failure_count
            
            total_executions = len(recent_latencies)
            
            # Calculate metrics
            if NUMPY_AVAILABLE and recent_latencies:
                avg_latency = np.mean(recent_latencies)  # type: ignore
                max_latency = np.max(recent_latencies)  # type: ignore
                p95_latency = np.percentile(recent_latencies, 95)  # type: ignore
            elif recent_latencies:
                avg_latency = sum(recent_latencies) / len(recent_latencies)
                max_latency = max(recent_latencies)
                sorted_latencies = sorted(recent_latencies)
                p95_index = int(len(sorted_latencies) * 0.95)
                p95_latency = sorted_latencies[p95_index]
            else:
                avg_latency = max_latency = p95_latency = 0
            
            return {
                'avg_latency_ms': avg_latency,
                'max_latency_ms': max_latency,
                'p95_latency_ms': p95_latency,
                'error_rate': recent_errors / max(total_executions, 1),
                'throughput_per_min': total_executions * 2,  # Rough estimate
                'monitor_uptime_seconds': time.time() - (self._check_count.get() * self.check_interval),
                'checks_performed': self._check_count.get(),
                'monitor_errors': self._error_count.get()
            }
        except Exception as e:
            self.logger.error(f"Failed to check performance health: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_status(self, health_data: Dict[str, Any]) -> str:
        """Calculate overall system health status"""
        statuses = []
        
        # System resources
        system = health_data.get('system', {})
        if not isinstance(system, dict):
            statuses.append('error')
        else:
            for metric, thresholds in [
                ('cpu_percent', self.thresholds['cpu_percent']),
                ('memory_percent', self.thresholds['memory_percent'])
            ]:
                value = system.get(metric, 0)
                if value > thresholds['critical']:
                    statuses.append('critical')
                elif value > thresholds['warning']:
                    statuses.append('warning')
        
        # Module health
        modules = health_data.get('modules', {})
        if isinstance(modules, dict):
            total = modules.get('total_modules', 1)
            unhealthy = modules.get('unhealthy_modules', 0)
            unhealthy_ratio = unhealthy / max(total, 1)
            
            if unhealthy_ratio > 0.3:
                statuses.append('critical')
            elif unhealthy_ratio > 0.1:
                statuses.append('warning')
        
        # Performance
        perf = health_data.get('performance', {})
        if isinstance(perf, dict):
            error_rate = perf.get('error_rate', 0)
            if error_rate > self.thresholds['error_rate']['critical']:
                statuses.append('critical')
            elif error_rate > self.thresholds['error_rate']['warning']:
                statuses.append('warning')
        
        # Determine overall status
        if 'error' in statuses:
            return HealthStatus.ERROR.value
        elif 'critical' in statuses:
            return HealthStatus.CRITICAL.value
        elif 'warning' in statuses:
            return HealthStatus.WARNING.value
        else:
            return HealthStatus.HEALTHY.value
    
    def _assess_infobus_status(self, metrics: Dict[str, Any]) -> str:
        """Assess InfoBus health status"""
        if metrics.get('cache_hit_rate', 0) < 0.5:
            return HealthStatus.WARNING.value
        
        if len(metrics.get('disabled_modules', [])) > 3:
            return HealthStatus.CRITICAL.value
        
        return HealthStatus.HEALTHY.value
    
    def _record_health_metrics(self, health_data: Dict[str, Any]) -> None:
        """Record health metrics for trending"""
        timestamp = health_data['timestamp']
        
        # System metrics
        system = health_data.get('system', {})
        if isinstance(system, dict):
            for metric_name, value in system.items():
                if isinstance(value, (int, float)):
                    self.metrics[f'system.{metric_name}'].append(
                        HealthMetric(
                            timestamp=timestamp,
                            metric_type=metric_name,
                            value=value,
                            threshold=self.thresholds.get(metric_name, {}).get('critical', float('inf')),
                            status=self._get_metric_status(metric_name, value)
                        )
                    )
        
        # Module health scores
        for module_name, score in self.module_health_scores.items():
            self.metrics[f'module.{module_name}.score'].append(
                HealthMetric(
                    timestamp=timestamp,
                    metric_type='health_score',
                    value=score,
                    threshold=0.5,
                    status='healthy' if score > 0.7 else 'warning' if score > 0.3 else 'critical'
                )
            )
    
    def _get_metric_status(self, metric_name: str, value: float) -> str:
        """Get status for a metric value"""
        if metric_name not in self.thresholds:
            return HealthStatus.HEALTHY.value
        
        thresholds = self.thresholds[metric_name]
        
        if value >= thresholds.get('critical', float('inf')):
            return HealthStatus.CRITICAL.value
        elif value >= thresholds.get('warning', float('inf')):
            return HealthStatus.WARNING.value
        else:
            return HealthStatus.HEALTHY.value
    
    def _check_for_alerts(self, health_data: Dict[str, Any]) -> None:
        """Check for alert conditions"""
        alerts_to_trigger = []
        
        # System resource alerts
        system = health_data.get('system', {})
        if isinstance(system, dict):
            for metric, value in system.items():
                if metric in self.thresholds and isinstance(value, (int, float)):
                    status = self._get_metric_status(metric, value)
                    if status != HealthStatus.HEALTHY.value:
                        alert_key = f'system.{metric}'
                        
                        if alert_key not in self.active_alerts:
                            alert = {
                                'type': 'system_resource',
                                'metric': metric,
                                'value': value,
                                'threshold': self.thresholds[metric][status],
                                'status': status,
                                'timestamp': time.time()
                            }
                            self.active_alerts[alert_key] = alert
                            alerts_to_trigger.append(alert)
        
        # Module health alerts
        modules = health_data.get('modules', {})
        if isinstance(modules, dict):
            module_details = modules.get('module_details', {})
            for module_name, health_info in module_details.items():
                if health_info['status'] in ['critical', 'error']:
                    alert_key = f'module.{module_name}'
                    
                    if alert_key not in self.active_alerts:
                        alert = {
                            'type': 'module_health',
                            'module': module_name,
                            'status': health_info['status'],
                            'failures': health_info.get('failures', 0),
                            'timestamp': time.time()
                        }
                        self.active_alerts[alert_key] = alert
                        alerts_to_trigger.append(alert)
        
        # Trigger alerts
        for alert in alerts_to_trigger:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]) -> None:
        """Trigger an alert"""
        # Add to history
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(
            f"[ALERT] HEALTH ALERT: {alert['type']} - "
            f"{alert.get('metric') or alert.get('module', 'unknown')} - "
            f"Status: {alert.get('status', 'unknown')}"
        )
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for health alerts"""
        if callable(callback):
            self.alert_callbacks.append(callback)
    
    def generate_health_report(self) -> HealthReport:
        """Generate comprehensive health report"""
        health_data = self.check_system_health()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(health_data)
        
        # Convert alerts to list
        with self._alerts_lock:
            alerts = list(self.active_alerts.values())
        
        return HealthReport(
            timestamp=datetime.now(),
            overall_status=health_data['overall_status'],
            system_metrics=health_data.get('system', {}),
            module_health={
                m: info['status'] 
                for m, info in health_data.get('modules', {}).get('module_details', {}).items()
            },
            alerts=alerts,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, health_data: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        # System resource recommendations
        system = health_data.get('system', {})
        if isinstance(system, dict):
            cpu = system.get('cpu_percent', 0)
            if cpu > self.thresholds['cpu_percent']['warning']:
                recommendations.append(
                    f"High CPU usage ({cpu:.1f}%) - consider optimizing compute-intensive modules"
                )
            
            memory = system.get('memory_percent', 0)
            if memory > self.thresholds['memory_percent']['warning']:
                recommendations.append(
                    f"High memory usage ({memory:.1f}%) - check for memory leaks"
                )
        
        # Module recommendations
        if self.unhealthy_modules:
            unhealthy_list = list(self.unhealthy_modules)[:5]
            recommendations.append(
                f"Unhealthy modules detected: {', '.join(unhealthy_list)}"
            )
            recommendations.append("Consider restarting or investigating these modules")
        
        # Performance recommendations
        perf = health_data.get('performance', {})
        if isinstance(perf, dict):
            avg_latency = perf.get('avg_latency_ms', 0)
            if avg_latency > 100:
                recommendations.append(
                    f"High average latency ({avg_latency:.0f}ms) - review module performance"
                )
            
            error_rate = perf.get('error_rate', 0)
            if error_rate > 0.05:
                recommendations.append(
                    f"High error rate ({error_rate:.1%}) - investigate failing modules"
                )
        
        # InfoBus recommendations
        infobus = health_data.get('infobus', {})
        if isinstance(infobus, dict):
            cache_hit_rate = infobus.get('cache_hit_rate', 1)
            if cache_hit_rate < 0.7:
                recommendations.append(
                    "Low cache hit rate - consider increasing cache size or TTL"
                )
        
        return recommendations
    
    def get_health_trends(self, metric_name: str, 
                         hours: int = 24) -> Dict[str, Any]:
        """Get health metric trends"""
        cutoff = time.time() - (hours * 3600)
        
        with self._metrics_lock:
            if metric_name not in self.metrics:
                return {'error': 'Metric not found'}
            
            metrics = [m for m in self.metrics[metric_name] if m.timestamp > cutoff]
        
        if not metrics:
            return {'error': 'No data in time range'}
        
        values = [m.value for m in metrics]
        
        # Calculate statistics
        if NUMPY_AVAILABLE:
            avg = float(np.mean(values))  # type: ignore
            minimum = float(np.min(values))  # type: ignore
            maximum = float(np.max(values))  # type: ignore
        else:
            avg = sum(values) / len(values)
            minimum = min(values)
            maximum = max(values)
        
        return {
            'metric': metric_name,
            'period_hours': hours,
            'data_points': len(values),
            'current': values[-1] if values else 0,
            'average': avg,
            'minimum': minimum,
            'maximum': maximum,
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Compare first third to last third
        third = len(values) // 3
        first_third = values[:third]
        last_third = values[-third:]
        
        if NUMPY_AVAILABLE:
            first_avg = float(np.mean(first_third))  # type: ignore
            last_avg = float(np.mean(last_third))  # type: ignore
        else:
            first_avg = sum(first_third) / len(first_third)
            last_avg = sum(last_third) / len(last_third)
        
        change = (last_avg - first_avg) / max(abs(first_avg), 1) * 100
        
        if change > 10:
            return 'increasing'
        elif change < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def export_health_data(self, filepath: str) -> bool:
        """Export health data for analysis"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Gather data
            with self._alerts_lock:
                alerts = list(self.active_alerts.values())
                alert_history = list(self.alert_history)[-100:]
            
            data = {
                'export_time': datetime.now().isoformat(),
                'current_health': self.check_system_health(),
                'active_alerts': alerts,
                'alert_history': alert_history,
                'module_scores': dict(self.module_health_scores),
                'recommendations': self._generate_recommendations(
                    self.check_system_health()
                ),
                'monitor_stats': {
                    'checks_performed': self._check_count.get(),
                    'errors_encountered': self._error_count.get(),
                    'uptime_seconds': self._check_count.get() * self.check_interval,
                    'is_running': self._started
                }
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Health data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export health data: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status"""
        return {
            'initialized': self._initialized,
            'running': self._started,
            'checks_performed': self._check_count.get(),
            'errors_encountered': self._error_count.get(),
            'last_check_duration_ms': self._last_check_duration * 1000,
            'active_alerts': len(self.active_alerts),
            'unhealthy_modules': len(self.unhealthy_modules),
            'thread_alive': self._monitor_thread.is_alive() if self._monitor_thread else False
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self._started:
                self.stop(timeout=1.0)
        except Exception:
            pass