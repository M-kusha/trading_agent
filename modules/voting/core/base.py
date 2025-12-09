"""
Unified base class for all voting modules.
Eliminates large amounts of duplicated boilerplate across voting files.

Responsibilities:
- SmartInfoBus integration (get/set with thesis)
- Rotating logger setup
- Error handling with pinpointing
- Performance tracking
- Standard initialization pattern
- Decision ID coordination helpers
- Basic health & circuit-breaker semantics
- Thin convenience layer for instrument-aware thresholds
"""

from __future__ import annotations

import time
from abc import abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING, cast
from pathlib import Path

import numpy as np

# Core module system
from modules.core.module_base import BaseModule, module  # noqa: F401
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin

# Utilities (optional, degrade gracefully if missing)
try:
    from modules.utils.info_bus import InfoBusManager
    from modules.utils.audit_utils import RotatingLogger, format_operator_message
    SMARTINFOBUS_AVAILABLE = True
except ImportError:
    InfoBusManager = None  # type: ignore[assignment]
    RotatingLogger = None  # type: ignore[assignment]

    def format_operator_message(**kw: Any) -> str:  # type: ignore[no-redef]
        return str(kw.get("message", ""))

    SMARTINFOBUS_AVAILABLE = False

try:
    from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ErrorPinpointer = None  # type: ignore[assignment]
    create_error_handler = None  # type: ignore[assignment]
    ERROR_HANDLING_AVAILABLE = False

try:
    from modules.monitoring.performance_tracker import PerformanceTracker
    PERFORMANCE_TRACKING_AVAILABLE = True
except ImportError:
    PerformanceTracker = None  # type: ignore[assignment]
    PERFORMANCE_TRACKING_AVAILABLE = False

from .constants import (
    VOTING_DEFAULTS,
    VotingBusKeys,
    MAX_PROCESSING_TIME_MS,
    CIRCUIT_BREAKER_THRESHOLD,
    get_adaptive_thresholds_for_instrument,
    get_instrument_threshold,
)

if TYPE_CHECKING:  # pragma: no cover
    from modules.utils.info_bus import SmartInfoBus
    from modules.utils.audit_utils import RotatingLogger as RotLoggerType


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
    - Basic health & circuit-breaker semantics
    - Convenience wrappers for instrument-aware thresholds

    Subclasses must implement:
    - _module_specific_init(): one-time setup
    - process(): main processing logic
    """

    # Smart bus & infrastructure
    smart_bus: "SmartInfoBus"   # set in _setup_smart_bus
    logger: Any                 # RotatingLogger or std logging.Logger
    error_pinpointer: Optional[Any]
    error_handler: Optional[Any]
    performance_tracker: Any

    # Internal state
    _module_name: str
    _voting_defaults: Dict[str, Any]

    # ═══════════════════════════════════════════════════════════════
    # Initialization
    # ═══════════════════════════════════════════════════════════════

    def _initialize(self) -> None:
        """
        Standard initialization – called by BaseModule.
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
        init_time_ms = (time.perf_counter() - self._init_start_time) * 1000.0
        self._log_info(f"Initialized in {init_time_ms:.1f}ms")

    def _setup_smart_bus(self) -> None:
        """Initialize SmartInfoBus connection (or a safe sentinel)."""
        self.smart_bus_enabled = False

        if SMARTINFOBUS_AVAILABLE and InfoBusManager is not None:
            try:
                self.smart_bus = InfoBusManager.get_instance()
                self.smart_bus_enabled = True
            except Exception as e:
                self._fallback_log(f"SmartInfoBus init failed: {e}")
                self.smart_bus = cast("SmartInfoBus", _UnavailableSmartBus())
        else:
            # Keep type stable but surface failure loudly on use
            self.smart_bus = cast("SmartInfoBus", _UnavailableSmartBus())

    def _setup_logging(self) -> None:
        """Initialize rotating logger (with console fallback)."""
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

        if self.logger is None:
            # Fallback to plain stdlib logging
            import logging

            self.logger = logging.getLogger(self._module_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

    def _setup_error_handling(self) -> None:
        """Initialize error pinpointer and error handler callback."""
        self.error_pinpointer = None
        self.error_handler = None

        if ERROR_HANDLING_AVAILABLE and ErrorPinpointer is not None:
            try:
                if create_error_handler is None:
                    raise RuntimeError("create_error_handler unavailable")
                self.error_pinpointer = ErrorPinpointer()
                self.error_handler = create_error_handler(
                    self._module_name, self.error_pinpointer
                )
            except Exception as e:
                self._fallback_log(f"Error handler init failed: {e}")

    def _setup_performance_tracking(self) -> None:
        """Initialize performance tracker (or a no-op stub)."""
        self.performance_tracker = None

        if PERFORMANCE_TRACKING_AVAILABLE and PerformanceTracker is not None:
            try:
                self.performance_tracker = PerformanceTracker(orchestrator=None)
            except Exception as e:
                self._fallback_log(f"Performance tracker init failed: {e}")

        if self.performance_tracker is None:
            self.performance_tracker = _NullPerformanceTracker()

    def _setup_state_tracking(self) -> None:
        """Initialize internal health and state tracking."""
        self._process_count: int = 0
        self._last_process_time: Optional[str] = None
        self._consecutive_failures: int = 0
        self._last_decision_id: Optional[str] = None
        self._health_status: str = "ok"
        self._cached_data: Dict[str, Any] = {}
        self._last_error: Optional[str] = None

    @abstractmethod
    def _module_specific_init(self) -> None:
        """
        Override in subclasses for module-specific initialization.
        Called after all base infrastructure is set up.
        """
        raise NotImplementedError

    # ═══════════════════════════════════════════════════════════════
    # Logging Helpers
    # ═══════════════════════════════════════════════════════════════

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Public info logger."""
        self._log_info(message, **kwargs)

    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Public debug logger."""
        self._log_debug(message, **kwargs)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Public warning logger."""
        self._log_warning(message, **kwargs)

    def log_error(
        self, message: str, error: Optional[Exception] = None, **kwargs: Any
    ) -> None:
        """Public error logger."""
        self._log_error(message, error, **kwargs)

    def _log_info(self, message: str, **kwargs: Any) -> None:
        """Internal info logger with optional operator formatting."""
        try:
            logger = self.logger
            if logger and hasattr(logger, "info"):
                if SMARTINFOBUS_AVAILABLE and format_operator_message:
                    formatted = format_operator_message(
                        message=message,
                        icon="[INFO]",
                        **kwargs,
                    )
                    logger.info(formatted)
                else:
                    logger.info(f"{message} {kwargs}" if kwargs else message)
        except Exception:
            self._fallback_log(f"INFO: {message}")

    def _log_warning(self, message: str, **kwargs: Any) -> None:
        """Internal warning logger."""
        try:
            logger = self.logger
            if logger and hasattr(logger, "warning"):
                if SMARTINFOBUS_AVAILABLE and format_operator_message:
                    formatted = format_operator_message(
                        message=message,
                        icon="[WARN]",
                        **kwargs,
                    )
                    logger.warning(formatted)
                else:
                    logger.warning(f"{message} {kwargs}" if kwargs else message)
        except Exception:
            self._fallback_log(f"WARNING: {message}")

    def _log_debug(self, message: str, **kwargs: Any) -> None:
        """Internal debug logger."""
        try:
            logger = self.logger
            if logger and hasattr(logger, "debug"):
                logger.debug(f"{message} {kwargs}" if kwargs else message)
        except Exception:
            # Debug logs can be safely dropped
            pass

    def _log_error(
        self, message: str, error: Optional[Exception] = None, **kwargs: Any
    ) -> None:
        """Internal error logger with optional pinpointed stack info."""
        try:
            if error and self.error_pinpointer:
                pinpointed = self.error_pinpointer.analyze_error(
                    error, self._module_name
                )
                message = f"{message}: {pinpointed}"

            logger = self.logger
            if logger and hasattr(logger, "error"):
                if SMARTINFOBUS_AVAILABLE and format_operator_message:
                    formatted = format_operator_message(
                        message=message,
                        icon="[ERROR]",
                        **kwargs,
                    )
                    logger.error(formatted)
                else:
                    logger.error(f"{message} {kwargs}" if kwargs else message)
        except Exception:
            self._fallback_log(f"ERROR: {message}")

    def _fallback_log(self, message: str) -> None:
        """Fallback logging when logger infrastructure is unavailable."""
        print(f"[{self._module_name}] {message}")

    # ═══════════════════════════════════════════════════════════════
    # SmartInfoBus Helpers
    # ═══════════════════════════════════════════════════════════════

    def bus_get(self, key: str, default: Any = None) -> Any:
        """Public wrapper: get value from SmartInfoBus with fallback."""
        return self._bus_get(key, default)

    def bus_set(self, key: str, value: Any, thesis: str = "") -> bool:
        """Public wrapper: set value on SmartInfoBus with thesis."""
        return self._bus_set(key, value, thesis)

    def _bus_get(self, key: str, default: Any = None) -> Any:
        """Internal SmartInfoBus get with defensive fallback."""
        if not self.smart_bus_enabled or not self.smart_bus:
            return default
        try:
            value = self.smart_bus.get(key, self._module_name, default=default)
            return value if value is not None else default
        except Exception:
            return default

    def _bus_set(self, key: str, value: Any, thesis: str = "") -> bool:
        """Internal SmartInfoBus set with defensive logging."""
        if not self.smart_bus_enabled or not self.smart_bus:
            return False
        try:
            self.smart_bus.set(
                key,
                value,
                module=self._module_name,
                thesis=thesis or f"{self._module_name} output",
            )
            return True
        except Exception as e:
            self._log_warning(f"Bus set failed for {key}: {e}")
            return False

    def _bus_get_multi(
        self, keys: List[str], default: Any = None
    ) -> Dict[str, Any]:
        """Get multiple values from bus as a {key: value} map."""
        return {key: self._bus_get(key, default) for key in keys}

    # ═══════════════════════════════════════════════════════════════
    # Instrument Threshold Convenience
    # ═══════════════════════════════════════════════════════════════

    def get_instrument_thresholds(self, instrument: str) -> Dict[str, Any]:
        """
        Get mode-aware, per-instrument thresholds.

        Wraps constants.get_adaptive_thresholds_for_instrument() so that
        modules do not need to talk to the dynamic threshold manager directly.
        """
        try:
            return get_adaptive_thresholds_for_instrument(instrument)
        except Exception:
            # Fallback to static/defaults
            return VOTING_DEFAULTS.copy()

    def get_confidence_threshold(self, instrument: Optional[str] = None) -> float:
        """
        Get the effective confidence threshold.

        - If instrument is provided, returns instrument-specific adaptive
          threshold (LIVE) or training threshold (TRAINING).
        - Otherwise, falls back to global VOTING_DEFAULTS.
        """
        if instrument:
            try:
                return float(get_instrument_threshold(instrument, "confidence"))
            except Exception:
                pass
        return float(self._voting_defaults.get("confidence_threshold", 0.5))

    # ═══════════════════════════════════════════════════════════════
    # Decision Coordination Helpers
    # ═══════════════════════════════════════════════════════════════

    def _get_current_decision_id(self) -> Optional[str]:
        """
        Get current decision ID from bus.

        Priority:
        1) DECISION_ID (global orchestrator)
        2) KERNEL_DECISION_ID (SlimVotingKernel)
        3) raw 'decision_id' key (legacy)
        """
        decision_id = self._bus_get(VotingBusKeys.DECISION_ID)
        if decision_id:
            return decision_id
        decision_id = self._bus_get(VotingBusKeys.KERNEL_DECISION_ID)
        if decision_id:
            return decision_id
        return self._bus_get("decision_id")

    def _get_current_tick_ts(self) -> Optional[str]:
        """
        Get current tick timestamp from bus.

        Priority:
        1) TICK_TS (global orchestrator)
        2) KERNEL_TICK_TS (SlimVotingKernel)
        """
        ts = self._bus_get(VotingBusKeys.TICK_TS)
        if ts:
            return ts
        return self._bus_get(VotingBusKeys.KERNEL_TICK_TS)

    def _is_same_decision_cycle(self, decision_id: str) -> bool:
        """Check if we are still in the same decision cycle."""
        return decision_id == self._last_decision_id

    def _check_data_freshness(
        self, timestamp: str, max_age_seconds: Optional[float] = None
    ) -> bool:
        """
        Check if data is fresh enough based on timestamp.

        Returns False on missing/invalid timestamps; returns True on parse
        failure (permissive) but logs a warning.
        """
        if not timestamp:
            return False

        max_age = max_age_seconds or self._voting_defaults.get(
            "max_staleness_seconds", 15.0
        )

        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
            age = now - ts
            return age.total_seconds() < max_age
        except Exception:
            self._log_warning(
                "Failed to parse timestamp for freshness check",
                ts=timestamp,
            )
            return True

    def get_decision_context(self) -> Dict[str, Any]:
        """
        Convenience helper: fetch decision_id and tick_ts from the bus,
        check staleness, and update internal last_decision_id.

        Returned dict:
            {
                "decision_id": str | None,
                "tick_ts": str | None,
                "is_fresh": bool | None,
            }
        """
        decision_id = self._get_current_decision_id()
        tick_ts = self._get_current_tick_ts()

        context: Dict[str, Any] = {
            "decision_id": decision_id,
            "tick_ts": tick_ts,
            "is_fresh": None,
        }

        if decision_id is None or tick_ts is None:
            self._log_warning(
                "Missing decision context on bus",
                decision_id=decision_id,
                tick_ts=tick_ts,
            )
        else:
            is_fresh = self._check_data_freshness(
                tick_ts,
                self._voting_defaults.get("max_staleness_seconds", 15.0),
            )
            context["is_fresh"] = is_fresh
            if not is_fresh:
                self._log_warning(
                    "Stale decision context detected",
                    tick_ts=tick_ts,
                    decision_id=decision_id,
                )

        self._last_decision_id = decision_id
        return context

    # ═══════════════════════════════════════════════════════════════
    # Performance Tracking
    # ═══════════════════════════════════════════════════════════════

    def _record_performance(
        self, metric_name: str, value: float, success: bool = True
    ) -> None:
        """Record performance metric (no-op if tracker is disabled)."""
        if self.performance_tracker:
            try:
                self.performance_tracker.record_metric(
                    self._module_name,
                    metric_name,
                    value,
                    success,
                )
            except Exception:
                # Performance metrics must never break trading logic
                pass

    def _time_operation(self, operation_name: str) -> "_OperationTimer":
        """Context manager for timing operations."""
        return _OperationTimer(self, operation_name)

    # ═══════════════════════════════════════════════════════════════
    # Health & Status / Circuit Breaker
    # ═══════════════════════════════════════════════════════════════

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status snapshot."""
        return {
            "module": self._module_name,
            "status": self._health_status,
            "process_count": self._process_count,
            "consecutive_failures": self._consecutive_failures,
            "last_process_time": self._last_process_time,
            "smart_bus_enabled": self.smart_bus_enabled,
            "circuit_open": self.circuit_open,
            "last_error": self._last_error,
        }

    @property
    def circuit_open(self) -> bool:
        """
        Returns True if this module has reached the circuit breaker threshold
        of consecutive failures and should be considered unsafe to trade
        until reset.
        """
        return self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD

    def _mark_success(self) -> None:
        """Mark successful processing and reset failure counter."""
        self._consecutive_failures = 0
        self._health_status = "ok"
        self._last_error = None
        self._process_count += 1
        self._last_process_time = datetime.now().isoformat()

    def _mark_failure(self, error: Optional[str] = None) -> None:
        """Mark failed processing and update circuit-breaker state."""
        self._consecutive_failures += 1

        if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            self._health_status = "critical"
        elif self._consecutive_failures >= 5:
            self._health_status = "degraded"

        if error:
            self._last_error = error
            self._log_warning(
                "Processing failure",
                error=error,
                consecutive_failures=self._consecutive_failures,
            )

    # ═══════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert arbitrary input to float."""
        try:
            f = float(value)
            return f if np.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_clip(value: float, low: float, high: float) -> float:
        """Safely clip value to [low, high]."""
        try:
            return float(np.clip(value, low, high))
        except Exception:
            return (low + high) / 2.0

    @staticmethod
    def _utcnow() -> str:
        """Get current UTC timestamp as ISO string."""
        return datetime.utcnow().isoformat() + "Z"


class _NullPerformanceTracker:
    """Stub performance tracker when the real one is unavailable."""

    def record_metric(
        self, module: str, metric: str, value: float, success: bool = True
    ) -> None:
        """No-op metric recording."""
        return

    def get_metrics(self, module: Optional[str] = None) -> Dict[str, Any]:
        """Return empty metrics."""
        return {}


class _UnavailableSmartBus:
    """
    Sentinel SmartInfoBus replacement when the real bus is unavailable.

    Any attempt to use it raises loudly to avoid silent data corruption.
    """

    def __getattr__(self, item: str) -> Any:
        raise RuntimeError("SmartInfoBus is unavailable in this context")


class _OperationTimer:
    """Context manager for timing operations in a module."""

    def __init__(self, module: VotingModuleBase, operation_name: str):
        self.module = module
        self.operation_name = operation_name
        self.start_time: Optional[float] = None

    def __enter__(self) -> "_OperationTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.start_time is not None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            self.module._record_performance(
                self.operation_name, elapsed_ms, exc_type is None
            )
            # Warn if processing is too slow for voting timeliness
            if elapsed_ms > MAX_PROCESSING_TIME_MS:
                self.module._log_warning(
                    "Operation exceeded max processing time",
                    operation=self.operation_name,
                    elapsed_ms=elapsed_ms,
                    max_ms=MAX_PROCESSING_TIME_MS,
                )
        # Do not suppress exceptions
        return False
