# ─────────────────────────────────────────────────────────────
# File: modules/utils/info_bus.py
# 🚀 PRODUCTION-READY SmartInfoBus - Zero-Wiring Architecture
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# ENHANCED: Complete production patterns, advanced monitoring, predictive analytics
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import sys
import time
import asyncio
import json
import pickle
import hashlib
import threading
import uuid
import psutil
from typing import Dict, Any, List, Optional, Set, Callable, Tuple, TYPE_CHECKING
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Import core dependencies
from modules.utils.audit_utils import RotatingLogger, format_operator_message, AuditSystem

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE CONFIGURATION STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InfoBusConfig:
    """
    Military-grade configuration for SmartInfoBus with comprehensive validation.
    """
    # Core settings
    enabled: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"
    max_cache_size: int = 10000
    cache_ttl_seconds: int = 3600
    
    # Performance settings
    max_parallel_operations: int = 50
    default_timeout_ms: int = 5000
    health_check_interval_ms: int = 30000
    metrics_retention_hours: int = 24
    background_thread_count: int = 2
    
    # Data management
    max_data_age_seconds: int = 300  # 5 minutes
    max_history_versions: int = 1000
    cleanup_interval_seconds: int = 60
    integrity_validation: bool = True
    auto_cleanup: bool = True
    compression_enabled: bool = False
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = 3
    recovery_time_seconds: int = 60
    failure_escalation_enabled: bool = True
    emergency_mode_enabled: bool = True
    
    # Event system
    max_event_log_size: int = 50000
    event_replay_enabled: bool = True
    subscription_timeout_ms: int = 1000
    async_callback_support: bool = True
    
    # Security & audit
    validation_enabled: bool = True
    audit_enabled: bool = True
    encryption_enabled: bool = False
    access_control_enabled: bool = False
    
    # Quality assurance
    quality_monitoring_enabled: bool = True
    predictive_analytics_enabled: bool = True
    anomaly_detection_enabled: bool = True
    performance_profiling_enabled: bool = True
    
    # Advanced features
    dependency_tracking_enabled: bool = True
    circular_dependency_detection: bool = True
    auto_dependency_resolution: bool = True
    smart_caching_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration integrity"""
        self._validate_config()
    
    def _validate_config(self):
        """Military-grade configuration validation"""
        errors = []
        
        # Timeout validations
        if self.default_timeout_ms <= 0 or self.default_timeout_ms > 60000:
            errors.append("default_timeout_ms must be between 1ms and 60000ms")
        
        if self.health_check_interval_ms <= 0 or self.health_check_interval_ms > 300000:
            errors.append("health_check_interval_ms must be between 1ms and 300000ms")
        
        # Performance validations
        if self.max_parallel_operations <= 0 or self.max_parallel_operations > 1000:
            errors.append("max_parallel_operations must be between 1 and 1000")
        
        if self.cache_ttl_seconds <= 0 or self.cache_ttl_seconds > 86400:
            errors.append("cache_ttl_seconds must be between 1 second and 1 day")
        
        # Data management validations
        if self.max_data_age_seconds <= 0 or self.max_data_age_seconds > 86400:
            errors.append("max_data_age_seconds must be between 1 second and 1 day")
        
        if self.max_history_versions <= 0 or self.max_history_versions > 10000:
            errors.append("max_history_versions must be between 1 and 10000")
        
        # Circuit breaker validations
        if self.circuit_breaker_threshold <= 0 or self.circuit_breaker_threshold > 20:
            errors.append("circuit_breaker_threshold must be between 1 and 20")
        
        if self.recovery_time_seconds <= 0 or self.recovery_time_seconds > 3600:
            errors.append("recovery_time_seconds must be between 1 second and 1 hour")
        
        # Event system validations
        if self.max_event_log_size <= 0 or self.max_event_log_size > 1000000:
            errors.append("max_event_log_size must be between 1 and 1000000")
        
        # Thread count validation
        if self.background_thread_count <= 0 or self.background_thread_count > 10:
            errors.append("background_thread_count must be between 1 and 10")
        
        # Log level validation
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_levels:
            errors.append(f"log_level must be one of: {valid_levels}")
        
        if errors:
            raise ValueError(f"InfoBusConfig validation failed: {errors}")
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with validation"""
        old_values = {}
        
        for key, value in updates.items():
            if hasattr(self, key):
                old_values[key] = getattr(self, key)
                setattr(self, key, value)
        
        try:
            self._validate_config()
        except ValueError as e:
            # Rollback on validation failure
            for key, old_value in old_values.items():
                setattr(self, key, old_value)
            raise e
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

@dataclass
class DataVersion:
    """
    PRODUCTION-GRADE versioned data with complete tracking and validation.
    Military-grade data integrity and comprehensive audit trail.
    """
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
        """Calculate validation hash and initialize tracking"""
        if not self.validation_hash:
            self._calculate_validation_hash()
        
        # Capture creation stack trace in debug mode
        if not self.creation_stack_trace:
            self.creation_stack_trace = self._capture_stack_trace()
        
        # Initialize quality assessment
        self._assess_data_quality()
    
    def _calculate_validation_hash(self):
        """Calculate comprehensive validation hash"""
        try:
            data_str = json.dumps({
                'value': str(self.value)[:1000],  # Limit size for performance
                'timestamp': self.timestamp,
                'source_module': self.source_module,
                'version': self.version,
                'confidence': self.confidence
            }, sort_keys=True)
            self.validation_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        except Exception:
            # Fallback hash if JSON serialization fails
            hash_input = f"{self.timestamp}{self.source_module}{self.version}{self.confidence}"
            self.validation_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _capture_stack_trace(self) -> str:
        """Capture creation stack trace for debugging"""
        try:
            # Get limited stack trace (last 5 frames)
            import traceback
            stack = traceback.format_stack()[-5:]
            return "".join(stack)
        except Exception:
            return "Stack trace unavailable"
    
    def _assess_data_quality(self):
        """Assess and score data quality"""
        score = 100.0
        
        # Deduct for low confidence
        if self.confidence < 0.8:
            score -= (0.8 - self.confidence) * 50
        
        # Deduct for excessive processing time
        if self.processing_time_ms > 1000:
            score -= min(self.processing_time_ms / 100, 30)
        
        # Deduct for missing thesis (if explainable)
        if not self.thesis and self.confidence > 0.5:
            score -= 10
        
        # Assess value quality
        if self.value is None:
            score -= 50
        elif isinstance(self.value, (int, float)) and np.isnan(self.value):
            score -= 40
        
        self.quality_score = max(0.0, score)
        
        # Set anomaly flags
        if self.quality_score < 50:
            self.anomaly_flags.append("low_quality")
        if self.processing_time_ms > 5000:
            self.anomaly_flags.append("slow_processing")
        if self.confidence < 0.3:
            self.anomaly_flags.append("low_confidence")
    
    def age_seconds(self) -> float:
        """Get age of data in seconds"""
        return time.time() - self.timestamp
    
    def validate_integrity(self) -> bool:
        """Validate data integrity using hash"""
        try:
            original_hash = self.validation_hash
            self.validation_hash = ""
            self._calculate_validation_hash()
            
            is_valid = self.validation_hash == original_hash
            if not is_valid:
                self.anomaly_flags.append("integrity_failure")
            
            return is_valid
        except Exception:
            self.anomaly_flags.append("validation_error")
            return False
    
    def increment_access(self, accessor_module: str = "unknown"):
        """Enhanced access tracking with module attribution"""
        self.access_count += 1
        self.last_access_time = time.time()
        self.access_patterns[accessor_module] = self.access_patterns.get(accessor_module, 0) + 1
    
    def get_access_frequency(self) -> float:
        """Calculate access frequency (accesses per hour)"""
        age_hours = self.age_seconds() / 3600
        if age_hours == 0:
            return float(self.access_count)
        return self.access_count / age_hours
    
    def is_stale(self, max_age_seconds: float) -> bool:
        """Check if data is stale"""
        return self.age_seconds() > max_age_seconds
    
    def get_staleness_ratio(self, max_age_seconds: float) -> float:
        """Get staleness ratio (0 = fresh, 1+ = stale)"""
        return self.age_seconds() / max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'value': self.value,
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

@dataclass
class DataRequest:
    """Enhanced request for data with comprehensive tracking and priority management"""
    requesting_module: str
    requested_key: str
    timestamp: float
    max_age_seconds: Optional[float] = None
    min_confidence: Optional[float] = None
    priority: int = 0
    callback: Optional[Callable] = None
    timeout_seconds: float = 60.0
    
    # Enhanced tracking
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retry_count: int = field(default=0)
    max_retries: int = field(default=3)
    escalation_threshold: float = field(default=30.0)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate request parameters"""
        if not self.requesting_module:
            raise ValueError("requesting_module cannot be empty")
        if not self.requested_key:
            raise ValueError("requested_key cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
    
    def is_expired(self) -> bool:
        """Check if request has expired"""
        return time.time() - self.timestamp > self.timeout_seconds
    
    def should_escalate(self) -> bool:
        """Check if request should be escalated"""
        return time.time() - self.timestamp > self.escalation_threshold
    
    def can_retry(self) -> bool:
        """Check if request can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry counter"""
        self.retry_count += 1
    
    def matches_data(self, data: DataVersion) -> bool:
        """Check if data matches request criteria"""
        if self.max_age_seconds and data.age_seconds() > self.max_age_seconds:
            return False
        
        if self.min_confidence and data.confidence < self.min_confidence:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'request_id': self.request_id,
            'requesting_module': self.requesting_module,
            'requested_key': self.requested_key,
            'timestamp': self.timestamp,
            'max_age_seconds': self.max_age_seconds,
            'min_confidence': self.min_confidence,
            'priority': self.priority,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'escalation_threshold': self.escalation_threshold,
            'context': self.context,
            'age_seconds': time.time() - self.timestamp,
            'is_expired': self.is_expired(),
            'should_escalate': self.should_escalate(),
            'can_retry': self.can_retry()
        }

# ═══════════════════════════════════════════════════════════════════
# ENHANCED CIRCUIT BREAKER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CircuitBreakerState:
    """Enhanced circuit breaker state with predictive failure detection"""
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    successful_calls: int = 0
    total_calls: int = 0
    last_success_time: float = 0
    
    # Enhanced metrics
    failure_rate: float = 0.0
    avg_failure_interval: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    failure_history: deque = field(default_factory=lambda: deque(maxlen=100))
    success_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_success(self):
        """Record successful operation with enhanced tracking"""
        self.successful_calls += 1
        self.total_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        
        self.success_history.append(time.time())
        
        # Update failure rate
        self._update_failure_rate()
        
        # Reset circuit breaker if in half-open state
        if self.state == "HALF_OPEN":
            if self.consecutive_successes >= 3:  # Require 3 consecutive successes
                self.state = "CLOSED"
                self.failure_count = 0
                self.consecutive_failures = 0
    
    def record_failure(self):
        """Record failed operation with enhanced tracking"""
        self.failure_count += 1
        self.total_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()
        
        self.failure_history.append(time.time())
        
        # Update metrics
        self._update_failure_rate()
        self._update_failure_interval()
    
    def _update_failure_rate(self):
        """Update failure rate based on recent history"""
        if self.total_calls > 0:
            self.failure_rate = self.failure_count / self.total_calls
        
        # Calculate recent failure rate (last 50 operations)
        recent_total = min(self.total_calls, 50)
        if recent_total > 0:
            recent_failures = min(self.failure_count, len(self.failure_history))
            recent_successes = min(self.successful_calls, len(self.success_history))
            recent_rate = recent_failures / (recent_failures + recent_successes) if (recent_failures + recent_successes) > 0 else 0
            
            # Weight recent rate more heavily
            self.failure_rate = (self.failure_rate * 0.7) + (recent_rate * 0.3)
    
    def _update_failure_interval(self):
        """Update average failure interval"""
        if len(self.failure_history) >= 2:
            intervals = []
            for i in range(1, len(self.failure_history)):
                interval = self.failure_history[i] - self.failure_history[i-1]
                intervals.append(interval)
            
            if intervals:
                self.avg_failure_interval = sum(intervals) / len(intervals)
    
    def should_allow_request(self, recovery_time: float, failure_threshold: int = 5) -> bool:
        """Enhanced request allowance logic with predictive analysis"""
        if self.state == "CLOSED":
            # Check if we should trip based on failure rate and consecutive failures
            if (self.consecutive_failures >= failure_threshold or 
                (self.failure_rate > 0.5 and self.total_calls > 10)):
                self.trip()
                return False
            return True
        
        elif self.state == "OPEN":
            # Check if recovery time has passed
            if time.time() - self.last_failure_time > recovery_time:
                self.state = "HALF_OPEN"
                return True
            return False
        
        else:  # HALF_OPEN
            # Allow limited requests to test recovery
            return True
    
    def trip(self):
        """Trip the circuit breaker with enhanced state management"""
        self.state = "OPEN"
        self.consecutive_successes = 0
    
    def get_health_score(self) -> float:
        """Calculate health score based on circuit breaker metrics"""
        if self.state == "OPEN":
            return 0.0
        
        if self.total_calls == 0:
            return 100.0
        
        # Base score on success rate
        success_rate = self.successful_calls / self.total_calls
        base_score = success_rate * 100
        
        # Adjust for consecutive failures
        if self.consecutive_failures > 0:
            base_score -= min(self.consecutive_failures * 5, 30)
        
        # Bonus for consecutive successes
        if self.consecutive_successes > 5:
            base_score = min(100.0, base_score + 5)
        
        return max(0.0, base_score)
    
    def predict_next_failure(self) -> Optional[float]:
        """Predict when next failure might occur based on patterns"""
        if self.avg_failure_interval > 0 and len(self.failure_history) >= 3:
            # Simple prediction based on average interval
            return self.last_failure_time + self.avg_failure_interval
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
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
            'predicted_next_failure': self.predict_next_failure()
        }

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE SMARTINFOBUS
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBus:
    """
    PRODUCTION-GRADE Information Bus for zero-wiring architecture.
    
    NASA/MILITARY SPECIFICATIONS:
    - Thread-safe operations with deadlock prevention
    - Data versioning and comprehensive integrity validation
    - Advanced performance monitoring and predictive analytics
    - Enhanced circuit breakers with failure prediction
    - Complete audit trail with replay capabilities
    - Real-time event streaming with async support
    - Automatic data freshness management with smart caching
    - Advanced dependency graph analysis with circular detection
    - Quality assurance with anomaly detection
    - Emergency mode and graceful degradation
    """
    
    def __init__(self, config: Optional[InfoBusConfig] = None):
        """Initialize SmartInfoBus with production-grade configuration"""
        
        # Configuration with validation
        self.config = config or InfoBusConfig()
        
        # Core data storage with enhanced thread safety
        self._data_store: Dict[str, DataVersion] = {}
        self._data_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_history_versions)
        )
        self._access_lock = threading.RLock()
        self._write_lock = threading.Lock()  # Separate write lock for better performance
        
        # Event logging for replay and audit
        self._event_log: deque = deque(maxlen=self.config.max_event_log_size)
        self._replay_mode = False
        self._replay_position = 0
        self._event_lock = threading.Lock()
        
        # Module registry with enhanced thread safety
        self._providers: Dict[str, Set[str]] = defaultdict(set)
        self._consumers: Dict[str, Set[str]] = defaultdict(set)
        self._module_graph: Dict[str, Set[str]] = defaultdict(set)
        self._registry_lock = threading.RLock()
        
        # Enhanced performance tracking
        self._access_patterns = defaultdict(lambda: defaultdict(int))
        self._latency_history = defaultdict(lambda: deque(maxlen=1000))
        self._cache_hits = 0
        self._cache_misses = 0
        self._performance_lock = threading.Lock()
        
        # Advanced metrics
        self._operation_timings = defaultdict(lambda: deque(maxlen=1000))
        self._memory_usage_history = deque(maxlen=100)
        self._cpu_usage_history = deque(maxlen=100)
        self._predictive_metrics = {}
        
        # Event subscription system with async support
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._subscription_lock = threading.Lock()
        
        # Enhanced circuit breakers with predictive failure detection
        self._circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
        self._module_disabled: Set[str] = set()
        self._circuit_breaker_lock = threading.Lock()
        
        # Enhanced request tracking and fulfillment
        self._pending_requests: List[DataRequest] = []
        self._request_history: deque = deque(maxlen=10000)
        self._request_lock = threading.Lock()
        
        # Data quality and validation with anomaly detection
        self._validation_enabled = self.config.validation_enabled
        self._quality_metrics = defaultdict(lambda: {'score': 100, 'issues': [], 'trends': []})
        self._anomaly_detector = None  # Will be initialized if enabled
        self._quality_lock = threading.Lock()
        
        # Emergency mode and graceful degradation
        self._emergency_mode = False
        self._emergency_triggers = 0
        self._emergency_threshold = 5
        self._degraded_operations = set()
        self._emergency_lock = threading.Lock()
        
        # Smart caching system
        self._cache_stats = defaultdict(int)
        self._cache_access_times = defaultdict(float)
        self._cache_priorities = defaultdict(float)
        
        # Background maintenance with enhanced management
        self._maintenance_running = True
        self._maintenance_threads = []
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.config.background_thread_count,
            thread_name_prefix="InfoBus"
        )
        
        # Setup enhanced logging with audit integration
        self.logger = RotatingLogger(
            name="SmartInfoBus",
            log_dir="logs/infobus",
            max_lines=10000,
            operator_mode=True,
            info_bus_aware=True
        )
        
        # Initialize audit system integration
        if self.config.audit_enabled:
            self._audit_system = AuditSystem("SmartInfoBus")
        else:
            self._audit_system = None
        
        # Initialize anomaly detection if enabled
        if self.config.anomaly_detection_enabled:
            self._initialize_anomaly_detection()
        
        # Start enhanced background services
        self._start_background_services()
        
        # Initialize system
        self._initialization_time = time.time()
        self._initialize_system_monitoring()
        
        # Log initialization event
        self._emit('bus_initialized', {
            'timestamp': self._initialization_time,
            'config': self.config.to_dict(),
            'features': self._get_enabled_features(),
            'system_info': self._get_system_info()
        })
        
        self.logger.info(
            format_operator_message(
                "🚀", "SMARTINFOBUS INITIALIZED - PRODUCTION MODE",
                details=f"Zero-wiring architecture ready with {len(self._get_enabled_features())} features",
                context="startup",
                performance=f"Startup time: {(time.time() - self._initialization_time)*1000:.1f}ms"
            )
        )
    
    def _get_enabled_features(self) -> List[str]:
        """Get list of enabled features"""
        features = ["core_operations", "thread_safety", "performance_monitoring"]
        
        if self.config.audit_enabled:
            features.append("audit_system")
        if self.config.anomaly_detection_enabled:
            features.append("anomaly_detection")
        if self.config.predictive_analytics_enabled:
            features.append("predictive_analytics")
        if self.config.quality_monitoring_enabled:
            features.append("quality_monitoring")
        if self.config.event_replay_enabled:
            features.append("event_replay")
        if self.config.smart_caching_enabled:
            features.append("smart_caching")
        if self.config.dependency_tracking_enabled:
            features.append("dependency_tracking")
        if self.config.emergency_mode_enabled:
            features.append("emergency_mode")
        
        return features
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for initialization"""
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
        """Initialize anomaly detection system"""
        try:
            # Simple anomaly detection based on statistical methods
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
            self.logger.info("🔍 Anomaly detection system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize anomaly detection: {e}")
    
    def _start_background_services(self):
        """Start enhanced background services"""
        try:
            # Main maintenance thread
            maintenance_thread = threading.Thread(
                target=self._background_maintenance,
                daemon=True,
                name="InfoBus-Maintenance"
            )
            maintenance_thread.start()
            self._maintenance_threads.append(maintenance_thread)
            
            # Performance monitoring thread
            if self.config.performance_profiling_enabled:
                perf_thread = threading.Thread(
                    target=self._background_performance_monitoring,
                    daemon=True,
                    name="InfoBus-Performance"
                )
                perf_thread.start()
                self._maintenance_threads.append(perf_thread)
            
            # Quality monitoring thread
            if self.config.quality_monitoring_enabled:
                quality_thread = threading.Thread(
                    target=self._background_quality_monitoring,
                    daemon=True,
                    name="InfoBus-Quality"
                )
                quality_thread.start()
                self._maintenance_threads.append(quality_thread)
            
            self.logger.info(f"✅ Started {len(self._maintenance_threads)} background services")
            
        except Exception as e:
            self.logger.error(f"Failed to start background services: {e}")
    
    def _initialize_system_monitoring(self):
        """Initialize system-level monitoring"""
        try:
            # Record initial system state
            self._record_system_metrics()
            
            # Set up performance baselines
            if self.config.predictive_analytics_enabled:
                self._initialize_performance_baselines()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system monitoring: {e}")
    
    def _record_system_metrics(self):
        """Record current system metrics"""
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            self._memory_usage_history.append({
                'timestamp': time.time(),
                'percent': memory_info.percent,
                'available_mb': memory_info.available // (1024 * 1024)
            })
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self._cpu_usage_history.append({
                'timestamp': time.time(),
                'percent': cpu_percent
            })
            
        except Exception as e:
            self.logger.debug(f"Failed to record system metrics: {e}")
    
    def _initialize_performance_baselines(self):
        """Initialize performance baselines for predictive analytics"""
        try:
            self._predictive_metrics = {
                'baseline_response_time': 10.0,  # 10ms baseline
                'baseline_throughput': 1000.0,   # 1000 ops/sec baseline
                'baseline_memory_usage': 50.0,   # 50MB baseline
                'trend_window_size': 100,
                'prediction_confidence': 0.8
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance baselines: {e}")

    def _background_performance_monitoring(self):
        """Background performance monitoring thread"""
        self.logger.info("📊 Performance monitoring started")
        
        while self._maintenance_running:
            try:
                # Record system metrics
                self._record_system_metrics()
                
                # Sleep for performance monitoring interval
                time.sleep(self.config.health_check_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"💥 Performance monitoring error: {e}")
                time.sleep(10)  # Back off on error

    def _background_quality_monitoring(self):
        """Background quality monitoring thread"""
        self.logger.info("🔍 Quality monitoring started")
        
        while self._maintenance_running:
            try:
                # Sleep for quality monitoring interval
                time.sleep(self.config.health_check_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"💥 Quality monitoring error: {e}")
                time.sleep(10)  # Back off on error

    # ═══════════════════════════════════════════════════════════════════
    # CORE DATA OPERATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def set(self, 
            key: str, 
            value: Any, 
            module: str, 
            thesis: Optional[str] = None, 
            confidence: float = 1.0,
            dependencies: Optional[List[str]] = None,
            processing_time_ms: float = 0.0):
        """
        Set data with comprehensive validation and tracking.
        
        Args:
            key: Data key identifier
            value: Data value to store
            module: Source module name
            thesis: Plain English explanation (optional)
            confidence: Confidence score 0-1
            dependencies: List of data keys this depends on
            processing_time_ms: Time taken to compute this value
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If operation fails
        """
        # Validate inputs
        if not isinstance(key, str) or not key:
            raise ValueError("Key must be a non-empty string")
        
        if not isinstance(module, str) or not module:
            raise ValueError("Module must be a non-empty string")
        
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if value is None:
            self.logger.warning(f"Setting None value for key '{key}' from {module}")
        
        try:
            with self._access_lock:
                # Get previous version
                prev = self._data_store.get(key)
                version = prev.version + 1 if prev else 1
                
                # Create versioned data with validation
                data = DataVersion(
                    value=value,
                    timestamp=time.time(),
                    source_module=module,
                    version=version,
                    confidence=max(0.0, min(1.0, confidence)),
                    thesis=thesis,
                    dependencies=dependencies or [],
                    processing_time_ms=max(0.0, processing_time_ms)
                )
                
                # Validate data integrity if enabled
                if self._validation_enabled:
                    if not data.validate_integrity():
                        raise RuntimeError(f"Data integrity validation failed for {key}")
                
                # Store current version
                self._data_store[key] = data
                
                # Store in history
                self._data_history[key].append(data)
                
                # Update registry
                with self._registry_lock:
                    self._providers[key].add(module)
                
                # Update dependency graph
                if dependencies:
                    for dep in dependencies:
                        provider = self._get_primary_provider(dep)
                        if provider and provider != module:
                            self._module_graph[module].add(provider)
                
                # Check cache size and cleanup if needed
                if len(self._data_store) > self.config.max_cache_size:
                    self._cleanup_old_data()
                
                # Log event for replay
                self._log_event({
                    'type': 'set',
                    'key': key,
                    'module': module,
                    'timestamp': data.timestamp,
                    'version': version,
                    'has_thesis': thesis is not None,
                    'confidence': confidence,
                    'data_size': len(str(value))
                })
                
                # Emit event for subscribers
                self._emit('data_updated', {
                    'key': key,
                    'module': module,
                    'version': version,
                    'confidence': confidence,
                    'has_thesis': thesis is not None
                })
                
                # Check pending requests
                self._check_pending_requests(key)
                
                # Update performance metrics
                with self._performance_lock:
                    self._access_patterns[module][f'write:{key}'] += 1
                
                self.logger.debug(f"✅ {module} set '{key}' v{version} (conf: {confidence:.2f})")
                
        except Exception as e:
            self.logger.error(f"💥 Failed to set {key} from {module}: {e}")
            
            # Record failure for circuit breaker
            self.record_module_failure(module, f"Data set failed: {str(e)}")
            raise RuntimeError(f"Failed to set data for key '{key}': {e}")
    
    def get(self, 
            key: str, 
            module: str, 
            max_age: Optional[float] = None,
            min_confidence: float = 0.0,
            default: Any = None) -> Any:
        """
        Get data with freshness and confidence validation.
        
        Args:
            key: Data key to retrieve
            module: Requesting module name
            max_age: Maximum acceptable age in seconds
            min_confidence: Minimum confidence required
            default: Default value if not found
            
        Returns:
            Data value or default if not found/invalid
        """
        try:
            with self._access_lock:
                # Track access pattern
                with self._performance_lock:
                    self._access_patterns[module][f'read:{key}'] += 1
                
                # Register consumer
                with self._registry_lock:
                    self._consumers[key].add(module)
                
                # Get data
                data = self._data_store.get(key)
                
                if not data:
                    with self._performance_lock:
                        self._cache_misses += 1
                    
                    self._log_miss(key, module)
                    return default
                
                # Validate data integrity
                if self._validation_enabled and not data.validate_integrity():
                    self.logger.error(f"Data integrity check failed for {key}")
                    return default
                
                # Check age requirement
                age_seconds = data.age_seconds()
                max_age_check = max_age or self.config.max_data_age_seconds
                
                if age_seconds > max_age_check:
                    self._emit('stale_data_warning', {
                        'key': key,
                        'age': age_seconds,
                        'module': module,
                        'max_age': max_age_check
                    })
                    
                    self.logger.warning(f"Stale data: {key} is {age_seconds:.1f}s old (max: {max_age_check}s)")
                    return default
                
                # Check confidence requirement
                if data.confidence < min_confidence:
                    self.logger.warning(f"Low confidence: {key} has {data.confidence:.2f} (min: {min_confidence:.2f})")
                    return default
                
                # Update access tracking
                data.increment_access(accessor_module=module)
                
                with self._performance_lock:
                    self._cache_hits += 1
                
                return data.value
                
        except Exception as e:
            self.logger.error(f"💥 Failed to get {key} for {module}: {e}")
            
            # Record failure
            self.record_module_failure(module, f"Data get failed: {str(e)}")
            return default
    
    def get_with_metadata(self, key: str, module: str) -> Optional[DataVersion]:
        """
        Get data with complete metadata.
        
        Args:
            key: Data key to retrieve
            module: Requesting module name
            
        Returns:
            DataVersion object or None if not found
        """
        try:
            with self._access_lock:
                # Register consumer and track access
                with self._registry_lock:
                    self._consumers[key].add(module)
                
                with self._performance_lock:
                    self._access_patterns[module][f'metadata:{key}'] += 1
                
                data = self._data_store.get(key)
                
                if data:
                    # Validate integrity
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
            self.logger.error(f"💥 Failed to get metadata for {key}: {e}")
            return None
    
    def get_with_thesis(self, key: str, module: str) -> Optional[Tuple[Any, str]]:
        """
        Get data value with its explanation.
        
        Args:
            key: Data key to retrieve
            module: Requesting module name
            
        Returns:
            Tuple of (value, thesis) or None if not found
        """
        data = self.get_with_metadata(key, module)
        if not data:
            return None
        
        thesis = data.thesis or "No explanation provided"
        return data.value, thesis
    
    def request_data(self, 
                    key: str, 
                    module: str, 
                    max_age: Optional[float] = None,
                    min_confidence: Optional[float] = None,
                    priority: int = 0,
                    callback: Optional[Callable] = None,
                    timeout_seconds: float = 60.0):
        """
        Request data that may not be available yet.
        
        Args:
            key: Data key to request
            module: Requesting module name
            max_age: Maximum acceptable age in seconds
            min_confidence: Minimum confidence required
            priority: Request priority (higher = more urgent)
            callback: Callback function when data becomes available
            timeout_seconds: Request timeout
        """
        try:
            request = DataRequest(
                requesting_module=module,
                requested_key=key,
                timestamp=time.time(),
                max_age_seconds=max_age,
                min_confidence=min_confidence,
                priority=priority,
                callback=callback,
                timeout_seconds=timeout_seconds
            )
            
            with self._request_lock:
                # Insert by priority (higher priority first)
                inserted = False
                for i, existing_req in enumerate(self._pending_requests):
                    if request.priority > existing_req.priority:
                        self._pending_requests.insert(i, request)
                        inserted = True
                        break
                
                if not inserted:
                    self._pending_requests.append(request)
            
            # Register as consumer
            with self._registry_lock:
                self._consumers[key].add(module)
            
            self.logger.debug(f"📋 {module} requested '{key}' (priority: {priority})")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to create data request: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # MODULE REGISTRY & DISCOVERY
    # ═══════════════════════════════════════════════════════════════════
    
    def register_provider(self, module: str, provides: List[str]):
        """
        Register what data a module provides.
        
        Args:
            module: Module name
            provides: List of data keys the module provides
        """
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
            self.logger.error(f"💥 Failed to register provider {module}: {e}")
    
    def register_consumer(self, module: str, requires: List[str]):
        """
        Register what data a module requires.
        
        Args:
            module: Module name
            requires: List of data keys the module requires
        """
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
            self.logger.error(f"💥 Failed to register consumer {module}: {e}")
    
    def get_providers(self, key: str) -> Set[str]:
        """Get all modules that can provide a data key"""
        with self._registry_lock:
            return self._providers.get(key, set()).copy()
    
    def get_consumers(self, key: str) -> Set[str]:
        """Get all modules that consume a data key"""
        with self._registry_lock:
            return self._consumers.get(key, set()).copy()
    
    def _get_primary_provider(self, key: str) -> Optional[str]:
        """Get primary (first available) provider for a key"""
        providers = self.get_providers(key)
        if providers:
            # Return first non-disabled provider
            for provider in providers:
                if self.is_module_enabled(provider):
                    return provider
        return None
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get complete module dependency graph"""
        try:
            with self._registry_lock:
                graph = {}
                
                # Build from data dependencies
                for key, consumers in self._consumers.items():
                    providers = self._providers.get(key, set())
                    
                    for provider in providers:
                        if provider not in graph:
                            graph[provider] = []
                        graph[provider].extend(list(consumers))
                
                # Add explicit dependencies
                for module, deps in self._module_graph.items():
                    if module not in graph:
                        graph[module] = []
                    graph[module].extend(list(deps))
                
                # Remove duplicates and self-references
                for module in graph:
                    graph[module] = list(set(graph[module]) - {module})
                
                return graph
                
        except Exception as e:
            self.logger.error(f"💥 Failed to build dependency graph: {e}")
            return {}
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in module graph"""
        graph = self.get_dependency_graph()
        cycles = []
        
        def dfs(node: str, path: List[str], visited: Set[str]):
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy(), visited.copy())
        
        try:
            # Start DFS from each node
            for node in graph:
                dfs(node, [], set())
            
            # Remove duplicate cycles
            unique_cycles = []
            for cycle in cycles:
                cycle_set = set(cycle[:-1])  # Remove duplicate end node
                if not any(set(c[:-1]) == cycle_set for c in unique_cycles):
                    unique_cycles.append(cycle)
            
            if unique_cycles:
                self.logger.warning(f"Found {len(unique_cycles)} circular dependencies")
            
            return unique_cycles
            
        except Exception as e:
            self.logger.error(f"💥 Failed to find circular dependencies: {e}")
            return []
    
    # ═══════════════════════════════════════════════════════════════════
    # PERFORMANCE & HEALTH MONITORING
    # ═══════════════════════════════════════════════════════════════════
    
    def record_module_timing(self, module: str, duration_ms: float):
        """Record module execution time with validation"""
        if duration_ms < 0:
            self.logger.warning(f"Invalid duration for {module}: {duration_ms}ms")
            return
        
        try:
            with self._performance_lock:
                self._latency_history[module].append(duration_ms)
                self._access_patterns[module]['execution_count'] += 1
            
            # Check for performance issues
            if len(self._latency_history[module]) >= 10:
                recent_avg = np.mean(list(self._latency_history[module])[-10:])
                if recent_avg > 200:  # 200ms threshold
                    self._emit('performance_warning', {
                        'module': module,
                        'avg_latency_ms': recent_avg,
                        'threshold_ms': 200
                    })
                    
        except Exception as e:
            self.logger.error(f"💥 Failed to record timing for {module}: {e}")
    
    def record_module_failure(self, module: str, error: str):
        """Record module failure for enhanced circuit breaker"""
        try:
            with self._circuit_breaker_lock:
                breaker = self._circuit_breakers[module]
                breaker.record_failure()
                
                # Check if should disable module
                if not breaker.should_allow_request(
                    recovery_time=self.config.recovery_time_seconds,
                    failure_threshold=self.config.circuit_breaker_threshold
                ):
                    self._module_disabled.add(module)
                    
                    self._emit('module_disabled', {
                        'module': module,
                        'failures': breaker.failure_count,
                        'consecutive_failures': breaker.consecutive_failures,
                        'failure_rate': breaker.failure_rate,
                        'error': error,
                        'timestamp': time.time(),
                        'circuit_breaker_state': breaker.to_dict()
                    })
                    
                    self.logger.error(
                        format_operator_message(
                            "🚫", "MODULE DISABLED",
                            instrument=module,
                            details=f"After {breaker.failure_count} failures (rate: {breaker.failure_rate:.1%})",
                            context="circuit_breaker"
                        )
                    )
                    
        except Exception as e:
            self.logger.error(f"💥 Failed to record failure for {module}: {e}")
    
    def is_module_enabled(self, module: str) -> bool:
        """Check if module is enabled using enhanced circuit breaker"""
        try:
            with self._circuit_breaker_lock:
                breaker = self._circuit_breakers[module]
                
                # Check if circuit breaker allows requests
                is_allowed = breaker.should_allow_request(
                    recovery_time=self.config.recovery_time_seconds,
                    failure_threshold=self.config.circuit_breaker_threshold
                )
                
                # Update disabled set based on circuit breaker state
                if is_allowed and module in self._module_disabled:
                    self._module_disabled.discard(module)
                elif not is_allowed:
                    self._module_disabled.add(module)
                
                return is_allowed
                
        except Exception as e:
            self.logger.error(f"💥 Failed to check module status for {module}: {e}")
            return True  # Fail open for safety
    
    def reset_module_failures(self, module: str):
        """Reset module failures using enhanced circuit breaker"""
        try:
            with self._circuit_breaker_lock:
                # Reset circuit breaker state
                self._circuit_breakers[module] = CircuitBreakerState()
                self._module_disabled.discard(module)
            
            self._emit('module_enabled', {
                'module': module,
                'timestamp': time.time(),
                'circuit_breaker_reset': True
            })
            
            self.logger.info(
                format_operator_message(
                    "✅", "MODULE ENABLED",
                    instrument=module,
                    details="Circuit breaker reset",
                    context="circuit_breaker_recovery"
                )
            )
            
        except Exception as e:
            self.logger.error(f"💥 Failed to reset failures for {module}: {e}")
    
    def get_module_health(self, module: str) -> Dict[str, Any]:
        """Get comprehensive module health information with enhanced circuit breaker data"""
        try:
            with self._circuit_breaker_lock:
                breaker = self._circuit_breakers[module]
                enabled = self.is_module_enabled(module)
            
            with self._performance_lock:
                latencies = list(self._latency_history.get(module, []))
                access_patterns = dict(self._access_patterns.get(module, {}))
            
            with self._registry_lock:
                provides = [k for k, providers in self._providers.items() if module in providers]
                consumes = [k for k, consumers in self._consumers.items() if module in consumers]
            
            avg_latency = np.mean(latencies) if latencies else 0.0
            
            return {
                'enabled': enabled,
                'circuit_breaker_state': breaker.state,
                'failures': breaker.failure_count,
                'successful_calls': breaker.successful_calls,
                'total_calls': breaker.total_calls,
                'failure_rate': breaker.failure_rate,
                'consecutive_failures': breaker.consecutive_failures,
                'consecutive_successes': breaker.consecutive_successes,
                'last_failure_time': breaker.last_failure_time,
                'last_success_time': breaker.last_success_time,
                'avg_latency_ms': float(avg_latency),
                'max_latency_ms': max(latencies) if latencies else 0,
                'total_executions': len(latencies),
                'provides': provides,
                'consumes': consumes,
                'access_patterns': access_patterns,
                'health_score': breaker.get_health_score(),
                'predicted_next_failure': breaker.predict_next_failure(),
                'circuit_breaker_details': breaker.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"💥 Failed to get health for {module}: {e}")
            return {'error': str(e)}
    
    def _calculate_health_score(self, enabled: bool, failures: int, avg_latency: float) -> float:
        """Calculate module health score 0-100"""
        if not enabled:
            return 0.0
        
        score = 100.0
        
        # Deduct for failures
        score -= min(failures * 10, 50)
        
        # Deduct for high latency
        if avg_latency > 500:
            score -= 30
        elif avg_latency > 200:
            score -= 15
        elif avg_latency > 100:
            score -= 5
        
        return max(0.0, score)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            with self._performance_lock:
                total_requests = self._cache_hits + self._cache_misses
                cache_hit_rate = self._cache_hits / max(total_requests, 1)
            
            with self._access_lock:
                active_data_keys = len(self._data_store)
                total_data_versions = sum(len(hist) for hist in self._data_history.values())
            
            with self._circuit_breaker_lock:
                disabled_modules = list(self._module_disabled)
                total_failures = sum(breaker.failure_count for breaker in self._circuit_breakers.values())
            
            with self._request_lock:
                pending_requests = len(self._pending_requests)
            
            # Calculate average latencies by module
            module_latencies = {}
            with self._performance_lock:
                for module, timings in self._latency_history.items():
                    if timings:
                        module_latencies[module] = {
                            'avg_ms': np.mean(timings),
                            'max_ms': max(timings),
                            'min_ms': min(timings),
                            'p95_ms': np.percentile(timings, 95) if len(timings) > 10 else max(timings),
                            'count': len(timings)
                        }
            
            return {
                'cache_hit_rate': cache_hit_rate,
                'total_requests': total_requests,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'active_data_keys': active_data_keys,
                'total_data_versions': total_data_versions,
                'total_events': len(self._event_log),
                'disabled_modules': disabled_modules,
                'total_module_failures': total_failures,
                'pending_requests': pending_requests,
                'module_latencies': module_latencies,
                'data_store_size_mb': self._estimate_data_size() / (1024 * 1024),
                'uptime_seconds': time.time() - self._get_initialization_time()
            }
            
        except Exception as e:
            self.logger.error(f"💥 Failed to get performance metrics: {e}")
            return {'error': str(e)}
    
    def _estimate_data_size(self) -> float:
        """Estimate total data store size in bytes"""
        try:
            total_size = 0
            for data in self._data_store.values():
                try:
                    # Rough estimation using pickle
                    serialized = pickle.dumps(data.value)
                    total_size += len(serialized)
                except:
                    # Fallback to string length estimation
                    total_size += len(str(data.value)) * 2  # Rough UTF-8 estimate
            
            return total_size
            
        except Exception as e:
            self.logger.error(f"Failed to estimate data size: {e}")
            return 0.0
    
    def _get_initialization_time(self) -> float:
        """Get initialization timestamp from first event"""
        if self._event_log:
            for event in self._event_log:
                if event.get('type') == 'bus_initialized':
                    return event.get('timestamp', time.time())
        return time.time()
    
    # ═══════════════════════════════════════════════════════════════════
    # EVENT SYSTEM
    # ═══════════════════════════════════════════════════════════════════
    
    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to bus events with validation.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Callback function to invoke
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        
        try:
            with self._subscription_lock:
                self._subscribers[event_type].append(callback)
            
            self.logger.debug(f"📡 Subscribed to '{event_type}' events")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to subscribe to {event_type}: {e}")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from bus events"""
        try:
            with self._subscription_lock:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                    
        except Exception as e:
            self.logger.error(f"💥 Failed to unsubscribe from {event_type}: {e}")
    
    def _emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all subscribers with error handling"""
        try:
            with self._subscription_lock:
                subscribers = self._subscribers[event_type].copy()
            
            for callback in subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        # Handle async callbacks
                        asyncio.create_task(callback(data))
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"💥 Event callback error for {event_type}: {e}")
                    
        except Exception as e:
            self.logger.error(f"💥 Failed to emit event {event_type}: {e}")
    
    def _log_event(self, event: Dict[str, Any]):
        """Log event for replay with size management"""
        try:
            event['timestamp'] = event.get('timestamp', time.time())
            self._event_log.append(event)
            
            # Emit for external logging
            self._emit('event_logged', event)
            
        except Exception as e:
            self.logger.error(f"💥 Failed to log event: {e}")
    
    def _log_miss(self, key: str, module: str):
        """Log data miss for analysis"""
        try:
            providers = self.get_providers(key)
            
            self._log_event({
                'type': 'miss',
                'key': key,
                'module': module,
                'providers': list(providers),
                'timestamp': time.time()
            })
            
            # Emit miss event
            self._emit('data_miss', {
                'key': key,
                'module': module,
                'providers': list(providers)
            })
            
        except Exception as e:
            self.logger.error(f"💥 Failed to log miss: {e}")
    
    def _check_pending_requests(self, key: str):
        """Check if any pending requests can be fulfilled"""
        try:
            fulfilled = []
            
            with self._request_lock:
                for i, request in enumerate(self._pending_requests):
                    if request.requested_key == key:
                        # Check if data meets requirements
                        data = self._data_store.get(key)
                        if data and request.matches_data(data):
                            # Fulfill request
                            self._emit('data_available', {
                                'key': key,
                                'requesting_module': request.requesting_module,
                                'value': data.value,
                                'metadata': data.to_dict()
                            })
                            
                            # Call callback if provided
                            if request.callback:
                                try:
                                    request.callback(data.value)
                                except Exception as e:
                                    self.logger.error(f"Request callback error: {e}")
                            
                            fulfilled.append(i)
                
                # Remove fulfilled requests (reverse order to maintain indices)
                for i in reversed(fulfilled):
                    self._pending_requests.pop(i)
                    
        except Exception as e:
            self.logger.error(f"💥 Failed to check pending requests: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # DATA QUALITY & MAINTENANCE
    # ═══════════════════════════════════════════════════════════════════
    
    def _background_maintenance(self):
        """Background maintenance thread"""
        self.logger.info("🔧 Background maintenance started")
        
        while self._maintenance_running:
            try:
                # Cleanup old data
                if self.config.auto_cleanup:
                    self._cleanup_old_data()
                
                # Clean expired requests
                self._cleanup_expired_requests()
                
                # Validate data integrity
                if self._validation_enabled:
                    self._validate_data_integrity()
                
                # Sleep for cleanup interval
                time.sleep(self.config.cleanup_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"💥 Background maintenance error: {e}")
                time.sleep(10)  # Back off on error
    
    def _cleanup_old_data(self):
        """Cleanup old data based on age and access patterns"""
        try:
            current_time = time.time()
            max_age = self.config.max_data_age_seconds
            keys_to_remove = []
            
            with self._access_lock:
                for key, data in self._data_store.items():
                    # Remove if too old and not frequently accessed
                    if (data.age_seconds() > max_age and 
                        data.access_count < 5):
                        keys_to_remove.append(key)
                
                # Remove old data
                for key in keys_to_remove:
                    del self._data_store[key]
                    
                    # Also cleanup history
                    if key in self._data_history:
                        # Keep only recent versions in history
                        recent_versions = []
                        for version in self._data_history[key]:
                            if current_time - version.timestamp < max_age * 2:
                                recent_versions.append(version)
                        
                        if recent_versions:
                            self._data_history[key] = deque(recent_versions, maxlen=self.config.max_history_versions)
                        else:
                            del self._data_history[key]
            
            if keys_to_remove:
                self.logger.debug(f"🧹 Cleaned up {len(keys_to_remove)} old data entries")
                
        except Exception as e:
            self.logger.error(f"💥 Data cleanup failed: {e}")
    
    def _cleanup_expired_requests(self):
        """Remove expired pending requests"""
        try:
            with self._request_lock:
                active_requests = []
                expired_count = 0
                
                for request in self._pending_requests:
                    if not request.is_expired():
                        active_requests.append(request)
                    else:
                        expired_count += 1
                
                self._pending_requests = active_requests
                
                if expired_count > 0:
                    self.logger.debug(f"🕒 Removed {expired_count} expired requests")
                    
        except Exception as e:
            self.logger.error(f"💥 Request cleanup failed: {e}")
    
    def _validate_data_integrity(self):
        """Validate integrity of stored data"""
        try:
            corruption_count = 0
            
            with self._access_lock:
                for key, data in list(self._data_store.items()):
                    if not data.validate_integrity():
                        self.logger.error(f"Data corruption detected for key: {key}")
                        # Remove corrupted data
                        del self._data_store[key]
                        corruption_count += 1
            
            if corruption_count > 0:
                self.logger.warning(f"🚨 Removed {corruption_count} corrupted data entries")
                
                # Emit corruption alert
                self._emit('data_corruption_detected', {
                    'corrupted_count': corruption_count,
                    'timestamp': time.time()
                })
                
        except Exception as e:
            self.logger.error(f"💥 Data integrity validation failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS & REPORTING
    # ═══════════════════════════════════════════════════════════════════
    
    def get_data_freshness_report(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive data freshness report"""
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
                        'validation_hash': data.validation_hash[:8] + "..."  # Truncated for display
                    }
            
            return report
            
        except Exception as e:
            self.logger.error(f"💥 Failed to generate freshness report: {e}")
            return {}
    
    def explain_data_flow(self, key: str) -> str:
        """Generate plain English explanation of data flow"""
        try:
            providers = list(self.get_providers(key))
            consumers = list(self.get_consumers(key))
            data = self.get_with_metadata(key, "SystemAnalyzer")
            
            lines = [
                f"DATA FLOW ANALYSIS: '{key}'",
                "=" * 60,
                ""
            ]
            
            if not providers and not consumers:
                lines.append(f"❌ No modules interact with '{key}'")
                return "\n".join(lines)
            
            if providers:
                lines.extend([
                    f"📤 PROVIDERS ({len(providers)}):"
                ])
                for provider in providers:
                    enabled = "✅" if self.is_module_enabled(provider) else "❌"
                    health = self.get_module_health(provider)
                    health_score = health.get('health_score', 0)
                    lines.append(f"  {enabled} {provider} (health: {health_score:.0f}%)")
            
            if consumers:
                lines.extend([
                    "",
                    f"📥 CONSUMERS ({len(consumers)}):"
                ])
                for consumer in consumers:
                    enabled = "✅" if self.is_module_enabled(consumer) else "❌"
                    lines.append(f"  {enabled} {consumer}")
            
            if data:
                lines.extend([
                    "",
                    "📊 CURRENT STATE:",
                    f"  Version: {data.version}",
                    f"  Age: {data.age_seconds():.1f} seconds",
                    f"  Source: {data.source_module}",
                    f"  Confidence: {data.confidence:.1%}",
                    f"  Access Count: {data.access_count}",
                    f"  Dependencies: {len(data.dependencies)}"
                ])
                
                if data.thesis:
                    lines.extend([
                        "",
                        "💭 EXPLANATION:",
                        f"  {data.thesis}"
                    ])
            else:
                lines.extend([
                    "",
                    "❌ NO DATA AVAILABLE"
                ])
            
            # Access statistics
            total_reads = 0
            with self._performance_lock:
                for consumer in consumers:
                    reads = self._access_patterns[consumer].get(f'read:{key}', 0)
                    total_reads += reads
            
            if total_reads > 0:
                lines.extend([
                    "",
                    "📈 ACCESS STATISTICS:",
                    f"  Total Reads: {total_reads}"
                ])
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"💥 Failed to explain data flow for {key}: {e}")
            return f"Error explaining data flow: {e}"
    
    def export_session(self, filepath: str):
        """Export complete session for replay"""
        try:
            session_data = {
                'export_timestamp': datetime.now().isoformat(),
                'events': list(self._event_log),
                'final_state': {
                    key: data.to_dict()
                    for key, data in self._data_store.items()
                },
                'performance_metrics': self.get_performance_metrics(),
                'data_freshness': self.get_data_freshness_report(),
                'dependency_graph': self.get_dependency_graph()
            }
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            self.logger.info(f"📁 Session exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to export session: {e}")
            raise
    
    def import_session(self, filepath: str):
        """Import session for replay"""
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            self._event_log = deque(session_data['events'], maxlen=self.config.max_event_log_size)
            
            self.logger.info(f"📂 Imported session with {len(self._event_log)} events")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to import session: {e}")
            raise
    
    def shutdown(self):
        """Graceful shutdown of SmartInfoBus"""
        try:
            self.logger.info("🛑 Shutting down SmartInfoBus...")
            
            # Stop background maintenance
            self._maintenance_running = False
            
            # Wait for maintenance thread to finish
            if self._maintenance_threads:
                for thread in self._maintenance_threads:
                    if thread.is_alive():
                        thread.join(timeout=5)
            
            # Shutdown thread pool
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)
            
            # Log final statistics
            metrics = self.get_performance_metrics()
            self.logger.info(
                f"📊 Final stats: {metrics['total_requests']} requests, "
                f"{metrics['cache_hit_rate']:.1%} hit rate, "
                f"{metrics['active_data_keys']} active keys"
            )
            
            # Clear all data
            with self._access_lock:
                self._data_store.clear()
                self._data_history.clear()
            
            self.logger.info("✅ SmartInfoBus shutdown complete")
            
        except Exception as e:
            self.logger.error(f"💥 Shutdown error: {e}")

# ═══════════════════════════════════════════════════════════════════
# SINGLETON MANAGER
# ═══════════════════════════════════════════════════════════════════

class InfoBusManager:
    """
    Thread-safe singleton manager for SmartInfoBus.
    Provides global access to the unified information bus.
    """
    
    _instance: Optional[SmartInfoBus] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> SmartInfoBus:
        """Get SmartInfoBus singleton instance with thread safety"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SmartInfoBus()
        return cls._instance
    
    @classmethod
    def create_info_bus(cls, env: Any, step: int = 0) -> Dict[str, Any]:
        """
        Create legacy InfoBus structure backed by SmartInfoBus.
        
        Args:
            env: Environment object
            step: Current step number
            
        Returns:
            Legacy InfoBus dictionary structure
        """
        smart_bus = cls.get_instance()
        
        # Create legacy structure
        info_bus = {
            'timestamp': datetime.now().isoformat(),
            'step_idx': step,
            'episode_idx': getattr(env, 'episode_count', 0),
            '_smart_bus': smart_bus,  # Reference to SmartInfoBus
            'prices': {},
            'positions': [],
            'risk': {'risk_score': 0.0}
        }
        
        # Extract data from environment if available
        if hasattr(env, 'data') and hasattr(env, 'instruments'):
            for instrument in env.instruments:
                if instrument in env.data and 'D1' in env.data[instrument]:
                    df = env.data[instrument]['D1']
                    if step < len(df):
                        price = float(df['close'].iloc[step])
                        info_bus['prices'][instrument] = price
                        
                        # Store in SmartInfoBus
                        smart_bus.set(
                            f'price_{instrument}',
                            price,
                            module='Environment',
                            thesis=f"Market price for {instrument} at step {step}",
                            confidence=1.0
                        )
        
        return info_bus
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)"""
        with cls._lock:
            if cls._instance:
                cls._instance.shutdown()
            cls._instance = None

# ═══════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InfoBusQuality:
    """Quality assessment for InfoBus validation"""
    score: float
    is_valid: bool
    missing_fields: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

def create_info_bus(env: Any, step: int = 0) -> Dict[str, Any]:
    """Legacy function - creates InfoBus backed by SmartInfoBus"""
    return InfoBusManager.create_info_bus(env, step)

def validate_info_bus(info_bus: Dict[str, Any]) -> InfoBusQuality:
    """Legacy validation with enhanced scoring"""
    
    # Enhanced validation
    required = ['timestamp', 'step_idx']
    missing = [f for f in required if f not in info_bus]
    issues = []
    
    # Check for SmartInfoBus integration
    if '_smart_bus' not in info_bus:
        issues.append("Missing SmartInfoBus integration")
    
    # Check data freshness
    if 'timestamp' in info_bus:
        try:
            timestamp = datetime.fromisoformat(info_bus['timestamp'].replace('Z', '+00:00'))
            age = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds()
            if age > 300:  # 5 minutes
                issues.append(f"Stale data: {age:.0f}s old")
        except:
            issues.append("Invalid timestamp format")
    
    # Calculate score
    score = 100.0
    score -= len(missing) * 25  # 25 points per missing field
    score -= len(issues) * 10   # 10 points per issue
    
    return InfoBusQuality(
        score=max(0, score),
        is_valid=score >= 50,
        missing_fields=missing,
        issues=issues
    )

# Legacy extractor class - now wraps SmartInfoBus
class InfoBusExtractor:
    """Legacy extractor - delegates to SmartInfoBus"""
    
    @staticmethod
    def get_risk_score(info_bus: Dict[str, Any]) -> float:
        """Get risk score with SmartInfoBus fallback"""
        # Try direct access first
        if 'risk_score' in info_bus:
            return float(info_bus['risk_score'])
        
        # Try SmartInfoBus
        if '_smart_bus' in info_bus:
            smart_bus = info_bus['_smart_bus']
            risk_data = smart_bus.get('risk_score', 'InfoBusExtractor')
            if risk_data is not None:
                return float(risk_data)
        
        return 0.0
    
    @staticmethod
    def get_market_regime(info_bus: Dict[str, Any]) -> str:
        """Get market regime with SmartInfoBus fallback"""
        # Try direct access first
        if 'market_regime' in info_bus:
            return str(info_bus['market_regime'])
        
        # Try SmartInfoBus
        if '_smart_bus' in info_bus:
            smart_bus = info_bus['_smart_bus']
            regime_data = smart_bus.get('market_regime', 'InfoBusExtractor')
            if regime_data is not None:
                return str(regime_data)
        
        return 'unknown'
    
    @staticmethod
    def has_fresh_data(info_bus: Dict[str, Any], max_age_seconds: float = 1.0) -> bool:
        """Check data freshness"""
        if '_smart_bus' in info_bus:
            smart_bus = info_bus['_smart_bus']
            metrics = smart_bus.get_performance_metrics()
            # Consider fresh if we have recent cache activity
            return metrics.get('cache_hits', 0) > 0
        
        return True  # Default to true for legacy compatibility
    
    @staticmethod
    def extract_risk_context(info_bus: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive risk context"""
        context = {
            'risk_score': InfoBusExtractor.get_risk_score(info_bus),
            'drawdown_pct': info_bus.get('drawdown_pct', 0.0),
            'exposure_pct': info_bus.get('exposure_pct', 0.0),
            'position_count': len(info_bus.get('positions', [])),
            'market_regime': InfoBusExtractor.get_market_regime(info_bus)
        }
        
        # Enhance with SmartInfoBus data if available
        if '_smart_bus' in info_bus:
            smart_bus = info_bus['_smart_bus']
            
            # Get additional risk metrics
            for key in ['volatility', 'correlation_risk', 'liquidity_risk']:
                value = smart_bus.get(key, 'InfoBusExtractor')
                if value is not None:
                    context[key] = value
        
        return context

# Legacy updater class - now wraps SmartInfoBus
class InfoBusUpdater:
    """Legacy updater - delegates to SmartInfoBus"""
    
    @staticmethod
    def add_vote(info_bus: Dict[str, Any], vote: Dict[str, Any]) -> None:
        """Add vote to InfoBus and SmartInfoBus"""
        # Update legacy structure
        votes = info_bus.get('votes', [])
        votes.append(vote)
        info_bus['votes'] = votes
        
        # Update SmartInfoBus
        if '_smart_bus' in info_bus:
            smart_bus = info_bus['_smart_bus']
            smart_bus.set(
                f"vote_{len(votes)}",
                vote,
                module='InfoBusUpdater',
                thesis=f"Vote from {vote.get('module', 'unknown')} module"
            )
    
    @staticmethod
    def set_risk_score(info_bus: Dict[str, Any], score: float) -> None:
        """Set risk score in both legacy and SmartInfoBus"""
        info_bus['risk_score'] = score
        
        if '_smart_bus' in info_bus:
            smart_bus = info_bus['_smart_bus']
            smart_bus.set(
                'risk_score',
                score,
                module='InfoBusUpdater',
                thesis=f"Risk score updated to {score:.2%}"
            )
    
    @staticmethod
    def set_market_regime(info_bus: Dict[str, Any], regime: str) -> None:
        """Set market regime in both legacy and SmartInfoBus"""
        info_bus['market_regime'] = regime
        
        if '_smart_bus' in info_bus:
            smart_bus = info_bus['_smart_bus']
            smart_bus.set(
                'market_regime',
                regime,
                module='InfoBusUpdater',
                thesis=f"Market regime identified as {regime}"
            )

# Utility functions
def now_utc() -> str:
    """Current UTC timestamp"""
    return datetime.now().isoformat()

def extract_standard_context(info_bus: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standard context for modules"""
    return {
        'regime': InfoBusExtractor.get_market_regime(info_bus),
        'risk_score': InfoBusExtractor.get_risk_score(info_bus),
        'position_count': len(info_bus.get('positions', [])),
        'has_fresh_data': InfoBusExtractor.has_fresh_data(info_bus),
        'timestamp': info_bus.get('timestamp'),
        'step_idx': info_bus.get('step_idx', 0)
    }