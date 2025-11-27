# modules/voting/core/base.py
"""
Unified base class for all voting modules.
Eliminates ~500+ lines of duplicated boilerplate across voting files.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Any, cast
from pathlib import Path

import numpy as np

# Core module system
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin

# Utilities
try:
    from modules.utils.info_bus import InfoBusManager
    from modules.utils.audit_utils import RotatingLogger, format_operator_message
    SMARTINFOBUS_AVAILABLE = True
except ImportError:
    InfoBusManager = None
    RotatingLogger = None
    format_operator_message = lambda **kw: str(kw.get("message", ""))
    SMARTINFOBUS_AVAILABLE = False

try:
    from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ErrorPinpointer = None
    create_error_handler = None
    ERROR_HANDLING_AVAILABLE = False

try:
    from modules.monitoring.performance_tracker import PerformanceTracker
    PERFORMANCE_TRACKING_AVAILABLE = True
except ImportError:
    PerformanceTracker = None
    PERFORMANCE_TRACKING_AVAILABLE = False

from .constants import VOTING_DEFAULTS, VotingBusKeys

if TYPE_CHECKING:
    from modules.utils.info_bus import SmartInfoBus  # pragma: no cover
    from modules.utils.audit_utils import RotatingLogger as RotLoggerType  # pragma: no cover


class VotingModuleBase(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Unified base class for all voting system modules.
    
    Provides:
    - SmartInfoBus integration (get/set with thesis)
    - Rotating logger setup
    - Error handling with pinpointing
    - Performance tracking
    - Standard initialization pattern
    - Decision ID coordination helpers
    
    Subclasses only need to implement:
    - _module_specific_init(): One-time setup
    - process(): Main processing logic
    """
    
    # ═══════════════════════════════════════════════════════════════
    # Initialization
    # ═══════════════════════════════════════════════════════════════
    
    smart_bus: "SmartInfoBus"
    logger: Any
    error_pinpointer: Optional[Any]
    error_handler: Optional[Any]
    performance_tracker: Any

    def _initialize(self) -> None:
        """
        Standard initialization - called by BaseModule.
        Sets up all common infrastructure, then delegates to subclass.
        """
        self._init_start_time = time.perf_counter()
        
        # Core references
        self._module_name = self.__class__.__name__
        self._voting_defaults = VOTING_DEFAULTS.copy()
        
        # Setup infrastructure in order
        self._setup_smart_bus()
        self._setup_logging()
        self._setup_error_handling()
        self._setup_performance_tracking()
        self._setup_state_tracking()
        
        # Delegate to subclass for module-specific init
        self._module_specific_init()
        
        # Log successful initialization
        init_time_ms = (time.perf_counter() - self._init_start_time) * 1000
        self._log_info(f"Initialized in {init_time_ms:.1f}ms")
    
    def _setup_smart_bus(self) -> None:
        """Initialize SmartInfoBus connection."""
        self.smart_bus_enabled = False
        
        if SMARTINFOBUS_AVAILABLE and InfoBusManager is not None:
            try:
                self.smart_bus = InfoBusManager.get_instance()
                self.smart_bus_enabled = True
            except Exception as e:
                self._fallback_log(f"SmartInfoBus init failed: {e}")
                self.smart_bus = cast("SmartInfoBus", _UnavailableSmartBus())
        else:
            # Keep type stable but surface failure loudly if used
            self.smart_bus = cast("SmartInfoBus", _UnavailableSmartBus())
    
    def _setup_logging(self) -> None:
        """Initialize rotating logger."""
        self.logger = None
        
        log_dir = Path("logs/voting")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self._module_name.lower()}.log"
        
        if SMARTINFOBUS_AVAILABLE and RotatingLogger is not None:
            try:
                self.logger = RotatingLogger(
                    name=self._module_name,
                    log_path=str(log_file),
                    max_lines=2000,
                    operator_mode=True,
                    plain_english=True,
                )
            except Exception as e:
                self._fallback_log(f"RotatingLogger init failed: {e}")
        
        # Fallback to standard logging if needed
        if self.logger is None:
            import logging
            self.logger = logging.getLogger(self._module_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
    
    def _setup_error_handling(self) -> None:
        """Initialize error pinpointer and handler."""
        self.error_pinpointer = None
        self.error_handler = None
        
        if ERROR_HANDLING_AVAILABLE and ErrorPinpointer is not None:
            try:
                if create_error_handler is None:
                    raise RuntimeError("create_error_handler unavailable")
                self.error_pinpointer = ErrorPinpointer()
                self.error_handler = create_error_handler(self._module_name, self.error_pinpointer)
            except Exception as e:
                self._fallback_log(f"Error handler init failed: {e}")
    
    def _setup_performance_tracking(self) -> None:
        """Initialize performance tracker."""
        self.performance_tracker = None
        
        if PERFORMANCE_TRACKING_AVAILABLE and PerformanceTracker is not None:
            try:
                self.performance_tracker = PerformanceTracker(orchestrator=None)
            except Exception as e:
                self._fallback_log(f"Performance tracker init failed: {e}")
        
        # Fallback stub if performance tracker unavailable
        if self.performance_tracker is None:
            self.performance_tracker = _NullPerformanceTracker()
    
    def _setup_state_tracking(self) -> None:
        """Initialize internal state tracking."""
        self._process_count = 0
        self._last_process_time = None
        self._consecutive_failures = 0
        self._last_decision_id = None
        self._health_status = "ok"
        self._cached_data: Dict[str, Any] = {}
    
    @abstractmethod
    def _module_specific_init(self) -> None:
        """
        Override in subclasses for module-specific initialization.
        Called after all base infrastructure is set up.
        """
        pass
    
    # ═══════════════════════════════════════════════════════════════
    # Logging Helpers
    # ═══════════════════════════════════════════════════════════════
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        self._log_info(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_debug(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_warning(message, **kwargs)
    
    def log_error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message with optional exception."""
        self._log_error(message, error, **kwargs)
    
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        try:
            logger = self.logger
            if logger and hasattr(logger, 'info'):
                if SMARTINFOBUS_AVAILABLE and format_operator_message:
                    formatted = format_operator_message(
                        message=message,
                        icon="[INFO]",
                        **kwargs
                    )
                    logger.info(formatted)
                else:
                    logger.info(f"{message} {kwargs}" if kwargs else message)
        except Exception:
            self._fallback_log(f"INFO: {message}")
    
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        try:
            logger = self.logger
            if logger and hasattr(logger, 'warning'):
                if SMARTINFOBUS_AVAILABLE and format_operator_message:
                    formatted = format_operator_message(
                        message=message,
                        icon="[WARN]",
                        **kwargs
                    )
                    logger.warning(formatted)
                else:
                    logger.warning(f"{message} {kwargs}" if kwargs else message)
        except Exception:
            self._fallback_log(f"WARNING: {message}")
    
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        try:
            logger = self.logger
            if logger and hasattr(logger, 'debug'):
                logger.debug(f"{message} {kwargs}" if kwargs else message)
        except Exception:
            pass  # Debug logs can be silently dropped
    
    def _log_error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message with optional exception."""
        try:
            if error and self.error_pinpointer:
                pinpointed = self.error_pinpointer.analyze_error(error, self._module_name)
                message = f"{message}: {pinpointed}"
            
            logger = self.logger
            if logger and hasattr(logger, 'error'):
                if SMARTINFOBUS_AVAILABLE and format_operator_message:
                    formatted = format_operator_message(
                        message=message,
                        icon="[ERROR]",
                        **kwargs
                    )
                    logger.error(formatted)
                else:
                    logger.error(f"{message} {kwargs}" if kwargs else message)
        except Exception:
            self._fallback_log(f"ERROR: {message}")
    
    def _fallback_log(self, message: str) -> None:
        """Fallback logging when logger unavailable."""
        print(f"[{self._module_name}] {message}")
    
    # ═══════════════════════════════════════════════════════════════
    # SmartInfoBus Helpers
    # ═══════════════════════════════════════════════════════════════
    
    def bus_get(self, key: str, default: Any = None) -> Any:
        """Get value from SmartInfoBus with fallback (public alias)."""
        return self._bus_get(key, default)
    
    def bus_set(self, key: str, value: Any, thesis: str = "") -> bool:
        """Set value on SmartInfoBus with thesis (public alias)."""
        return self._bus_set(key, value, thesis)
    
    def _bus_get(self, key: str, default: Any = None) -> Any:
        """Get value from SmartInfoBus with fallback."""
        if not self.smart_bus_enabled or not self.smart_bus:
            return default
        try:
            value = self.smart_bus.get(key, self._module_name, default=default)
            return value if value is not None else default
        except Exception:
            return default
    
    def _bus_set(self, key: str, value: Any, thesis: str = "") -> bool:
        """Set value on SmartInfoBus with thesis."""
        if not self.smart_bus_enabled or not self.smart_bus:
            return False
        try:
            self.smart_bus.set(
                key, 
                value, 
                module=self._module_name,
                thesis=thesis or f"{self._module_name} output"
            )
            return True
        except Exception as e:
            self._log_warning(f"Bus set failed for {key}: {e}")
            return False
    
    def _bus_get_multi(self, keys: List[str], default: Any = None) -> Dict[str, Any]:
        """Get multiple values from bus."""
        result = {}
        for key in keys:
            result[key] = self._bus_get(key, default)
        return result
    
    # ═══════════════════════════════════════════════════════════════
    # Decision Coordination Helpers
    # ═══════════════════════════════════════════════════════════════
    
    def _get_current_decision_id(self) -> Optional[str]:
        """Get current decision ID from bus."""
        return self._bus_get(VotingBusKeys.DECISION_ID)
    
    def _get_current_tick_ts(self) -> Optional[str]:
        """Get current tick timestamp from bus."""
        return self._bus_get(VotingBusKeys.TICK_TS)
    
    def _is_same_decision_cycle(self, decision_id: str) -> bool:
        """Check if we're in the same decision cycle."""
        return decision_id == self._last_decision_id
    
    def _check_data_freshness(self, timestamp: str, max_age_seconds: Optional[float] = None) -> bool:
        """Check if data is fresh enough."""
        if not timestamp:
            return False
        
        max_age = max_age_seconds or self._voting_defaults.get("max_staleness_seconds", 15.0)
        
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            age = (datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()) - ts
            return age.total_seconds() < max_age
        except Exception:
            return True  # Be permissive on parse failure
    
    # ═══════════════════════════════════════════════════════════════
    # Performance Tracking
    # ═══════════════════════════════════════════════════════════════
    
    def _record_performance(self, metric_name: str, value: float, success: bool = True) -> None:
        """Record performance metric."""
        if self.performance_tracker:
            try:
                self.performance_tracker.record_metric(
                    self._module_name,
                    metric_name,
                    value,
                    success
                )
            except Exception:
                pass
    
    def _time_operation(self, operation_name: str) -> "_OperationTimer":
        """Context manager for timing operations."""
        return _OperationTimer(self, operation_name)
    
    # ═══════════════════════════════════════════════════════════════
    # Health & Status
    # ═══════════════════════════════════════════════════════════════
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "module": self._module_name,
            "status": self._health_status,
            "process_count": self._process_count,
            "consecutive_failures": self._consecutive_failures,
            "last_process_time": self._last_process_time,
            "smart_bus_enabled": self.smart_bus_enabled,
        }
    
    def _mark_success(self) -> None:
        """Mark successful processing."""
        self._consecutive_failures = 0
        self._health_status = "ok"
        self._process_count += 1
        self._last_process_time = datetime.now().isoformat()
    
    def _mark_failure(self, error: Optional[str] = None) -> None:
        """Mark failed processing."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            self._health_status = "degraded"
        if self._consecutive_failures >= 10:
            self._health_status = "critical"
    
    # ═══════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════
    
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert to float."""
        try:
            f = float(value)
            return f if np.isfinite(f) else default
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def _safe_clip(value: float, low: float, high: float) -> float:
        """Safely clip value to range."""
        try:
            return float(np.clip(value, low, high))
        except Exception:
            return (low + high) / 2
    
    @staticmethod
    def _utcnow() -> str:
        """Get current UTC timestamp as ISO string."""
        return datetime.utcnow().isoformat() + "Z"


class _NullPerformanceTracker:
    """Stub performance tracker when the real one is unavailable."""
    
    def record_metric(self, module: str, metric: str, value: float, success: bool = True) -> None:
        """No-op metric recording."""
        pass
    
    def get_metrics(self, module: Optional[str] = None) -> Dict[str, Any]:
        """Return empty metrics."""
        return {}


class _UnavailableSmartBus:
    """
    Sentinel SmartInfoBus replacement when the real bus is unavailable.
    Raises on use to avoid silent fallback data.
    """

    def __getattr__(self, item: str) -> Any:
        raise RuntimeError("SmartInfoBus is unavailable in this context")

class _OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, module: VotingModuleBase, operation_name: str):
        self.module = module
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.module._record_performance(self.operation_name, elapsed_ms, exc_type is None)
        return False  # Don't suppress exceptions
