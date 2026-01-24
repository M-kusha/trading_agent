"""
Unified base class for all voting modules.
Eliminates duplicated boilerplate across voting files.

Responsibilities:
- SmartInfoBus integration (get/set with thesis)
- Rotating logger setup
- Error handling with pinpointing
- Performance tracking
- Standard initialization pattern
- Decision ID coordination helpers
- Basic health & circuit-breaker semantics
- Thin convenience layer for instrument-aware thresholds
- Typed config accessors (clamp/validate once, reuse everywhere)
- Optional structured forensic debug (JSONL), safe-by-default
- Canonical instrument normalization helpers (single-source behavior)
"""

from __future__ import annotations

import json
import os
import time
import traceback
from abc import abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, TYPE_CHECKING, cast, Iterable, Tuple

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
    normalize_instrument,
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
    - Typed config accessors (safe defaults + clamping)
    - Canonical instrument normalization helpers
    - Optional structured debug trace (JSONL)

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

    # Debug trace
    _debug_enabled: bool
    _debug_level: str
    _debug_logger_fp: Optional[Any]
    _debug_path: Optional[str]
    _debug_flush: bool

    # Circuit breaker (enhanced)
    _consecutive_failures: int
    _health_status: str
    _last_error: Optional[str]
    _circuit_open_until: Optional[datetime]
    _circuit_open_reason: Optional[str]

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
        self._setup_debug_trace()

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

        self._consecutive_failures = 0
        self._health_status = "ok"
        self._last_error = None

        self._last_decision_id: Optional[str] = None
        self._cached_data: Dict[str, Any] = {}

        # Enhanced circuit semantics
        self._circuit_open_until = None
        self._circuit_open_reason = None

    def _setup_debug_trace(self) -> None:
        """
        Optional structured JSONL debug trace.

        Config keys (all optional):
          - debug.enabled: bool (default False)
          - debug.level: "light"|"standard"|"forensic" (default "standard")
          - debug.dir: str (default "logs/voting_debug")
          - debug.flush: bool (default True)
          - debug.max_bytes: int (default 10MB)  [best-effort, rotation only if stdlib RotatingFileHandler is available]
          - debug.backups: int (default 5)
        """
        cfg = self.config if isinstance(getattr(self, "config", None), dict) else {}

        debug_block = cfg.get("debug", cfg.get("debug_config", {}))
        if not isinstance(debug_block, dict):
            debug_block = {}

        self._debug_enabled = bool(debug_block.get("enabled", False))
        self._debug_level = str(debug_block.get("level", "standard") or "standard").lower().strip()
        self._debug_flush = bool(debug_block.get("flush", True))

        self._debug_logger_fp = None
        self._debug_path = None

        if not self._debug_enabled:
            return

        debug_dir = str(debug_block.get("dir", "logs/voting_debug") or "logs/voting_debug")
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(debug_dir, f"{self._module_name}.trace.jsonl")
        self._debug_path = filename

        # Prefer stdlib rotating handler if available; otherwise simple append.
        try:
            import logging
            from logging.handlers import RotatingFileHandler

            logger = logging.getLogger(f"{__name__}.{self._module_name}.trace")
            logger.setLevel(logging.DEBUG)
            logger.propagate = False

            max_bytes = int(debug_block.get("max_bytes", 10 * 1024 * 1024))
            backups = int(debug_block.get("backups", 5))

            # avoid duplicate handlers
            if not any(
                isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == filename
                for h in logger.handlers
            ):
                h = RotatingFileHandler(
                    filename, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
                )
                h.setLevel(logging.DEBUG)
                h.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(h)

            self._debug_logger_fp = logger
            self._debug_log(event="debug_trace_initialized", path=filename, level=self._debug_level)
        except Exception:
            # Last resort: raw file append on each event (safe, but no rotation)
            try:
                self._debug_logger_fp = open(filename, "a", encoding="utf-8")
                self._debug_log(event="debug_trace_initialized_fallback", path=filename, level=self._debug_level)
            except Exception:
                self._debug_enabled = False
                self._debug_logger_fp = None
                self._debug_path = None

    @abstractmethod
    def _module_specific_init(self) -> None:
        """
        Override in subclasses for module-specific initialization.
        Called after all base infrastructure is set up.
        """
        raise NotImplementedError

    # ═══════════════════════════════════════════════════════════════
    # Typed Config Helpers (centralized validation/clamping)
    # ═══════════════════════════════════════════════════════════════

    def conf_bool(self, key: str, default: bool = False) -> bool:
        cfg = self.config if isinstance(getattr(self, "config", None), dict) else {}
        v = cfg.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        return bool(default)

    def conf_str(self, key: str, default: str = "") -> str:
        cfg = self.config if isinstance(getattr(self, "config", None), dict) else {}
        v = cfg.get(key, default)
        try:
            s = str(v)
        except Exception:
            s = str(default)
        return s

    def conf_int(self, key: str, default: int = 0, min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
        cfg = self.config if isinstance(getattr(self, "config", None), dict) else {}
        v = cfg.get(key, default)
        try:
            i = int(float(v))
        except Exception:
            i = int(default)
        if min_v is not None and i < min_v:
            i = min_v
        if max_v is not None and i > max_v:
            i = max_v
        return i

    def conf_float(
        self,
        key: str,
        default: float = 0.0,
        min_v: Optional[float] = None,
        max_v: Optional[float] = None,
    ) -> float:
        cfg = self.config if isinstance(getattr(self, "config", None), dict) else {}
        v = cfg.get(key, default)
        f = self._safe_float(v, default=float(default))
        if min_v is not None and f < min_v:
            f = float(min_v)
        if max_v is not None and f > max_v:
            f = float(max_v)
        return f

    def conf_list(self, key: str, default: Optional[List[Any]] = None) -> List[Any]:
        cfg = self.config if isinstance(getattr(self, "config", None), dict) else {}
        v = cfg.get(key, default if default is not None else [])
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, tuple):
            return list(v)
        return [v]

    # ═══════════════════════════════════════════════════════════════
    # Logging Helpers
    # ═══════════════════════════════════════════════════════════════

    def log_info(self, message: str, **kwargs: Any) -> None:
        self._log_info(message, **kwargs)

    def log_debug(self, message: str, **kwargs: Any) -> None:
        self._log_debug(message, **kwargs)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        self._log_warning(message, **kwargs)

    def log_error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        self._log_error(message, error, **kwargs)

    def _log_info(self, message: str, **kwargs: Any) -> None:
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
        try:
            logger = self.logger
            if logger and hasattr(logger, "debug"):
                logger.debug(f"{message} {kwargs}" if kwargs else message)
        except Exception:
            pass

    def _log_error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        try:
            if error and self.error_pinpointer:
                pinpointed = self.error_pinpointer.analyze_error(error, self._module_name)
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
        print(f"[{self._module_name}] {message}")

    # ═══════════════════════════════════════════════════════════════
    # Structured Debug Trace (JSONL)
    # ═══════════════════════════════════════════════════════════════

    def _debug_log(self, event: str, **payload: Any) -> None:
        if not getattr(self, "_debug_enabled", False):
            return

        rec = {
            "ts": time.time(),
            "iso": datetime.now().isoformat(),
            "module": self._module_name,
            "event": str(event),
            "level": self._debug_level,
            "payload": payload,
        }

        try:
            line = json.dumps(rec, ensure_ascii=False, default=str)
        except Exception:
            line = json.dumps(
                {
                    "ts": rec.get("ts"),
                    "iso": rec.get("iso"),
                    "module": rec.get("module"),
                    "event": rec.get("event"),
                    "level": rec.get("level"),
                    "payload": str(payload),
                },
                ensure_ascii=False,
                default=str,
            )

        fp = getattr(self, "_debug_logger_fp", None)
        if fp is None:
            return

        try:
            # logging.Logger path
            if hasattr(fp, "debug"):
                fp.debug(line)
                if self._debug_flush and hasattr(fp, "handlers"):
                    for h in fp.handlers:
                        try:
                            h.flush()
                        except Exception:
                            pass
                return

            # raw file handle path
            fp.write(line + "\n")
            if self._debug_flush:
                fp.flush()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    # SmartInfoBus Helpers
    # ═══════════════════════════════════════════════════════════════

    def bus_get(self, key: str, default: Any = None) -> Any:
        return self._bus_get(key, default)

    def bus_set(self, key: str, value: Any, thesis: str = "", **kwargs: Any) -> bool:
        return self._bus_set(key, value, thesis, **kwargs)

    def _bus_get(self, key: str, default: Any = None) -> Any:
        if not self.smart_bus_enabled or not self.smart_bus:
            return default
        try:
            value = self.smart_bus.get(key, self._module_name, default=default)
            return value if value is not None else default
        except Exception as e:
            self._debug_log(event="bus_get_failed", key=key, error=str(e))
            return default

    def _bus_set(self, key: str, value: Any, thesis: str = "", **kwargs: Any) -> bool:
        if not self.smart_bus_enabled or not self.smart_bus:
            return False
        try:
            self.smart_bus.set(
                key,
                value,
                module=self._module_name,
                thesis=thesis or f"{self._module_name} output",
                **kwargs,
            )
            return True
        except Exception as e:
            self._debug_log(event="bus_set_failed", key=key, error=str(e))
            self._log_warning(f"Bus set failed for {key}: {e}")
            return False

    def _bus_get_multi(self, keys: List[str], default: Any = None) -> Dict[str, Any]:
        return {key: self._bus_get(key, default) for key in keys}

    # ═══════════════════════════════════════════════════════════════
    # Instrument Helpers (canonicalization + convenience)
    # ═══════════════════════════════════════════════════════════════

    def canon(self, instrument: str) -> str:
        """Canonical symbol normalization (single behavior)."""
        return normalize_instrument(instrument)

    def canon_list(self, instruments: Iterable[Any]) -> List[str]:
        out: List[str] = []
        for x in instruments:
            if x is None:
                continue
            s = normalize_instrument(str(x))
            if s and s not in out:
                out.append(s)
        return out

    def get_active_instruments(self, default: Optional[List[str]] = None) -> List[str]:
        """
        Prefer bus (watched_instruments) then config ('instruments'), else default.
        """
        default = default or ["XAUUSD"]
        bus_list = self._bus_get(VotingBusKeys.ACTIVE_INSTRUMENTS, default=None)
        if isinstance(bus_list, list) and bus_list:
            return self.canon_list(bus_list)

        cfg_list = self.conf_list("instruments", default=default)
        return self.canon_list(cfg_list) if cfg_list else self.canon_list(default)

    # ═══════════════════════════════════════════════════════════════
    # Instrument Threshold Convenience
    # ═══════════════════════════════════════════════════════════════

    def get_instrument_thresholds(self, instrument: str) -> Dict[str, Any]:
        try:
            return get_adaptive_thresholds_for_instrument(self.canon(instrument))
        except Exception:
            return VOTING_DEFAULTS.copy()

    def get_confidence_threshold(self, instrument: Optional[str] = None) -> float:
        if instrument:
            try:
                return float(get_instrument_threshold(self.canon(instrument), "confidence"))
            except Exception:
                pass
        return float(self._voting_defaults.get("confidence_threshold", 0.5))

    # ═══════════════════════════════════════════════════════════════
    # Decision Coordination Helpers
    # ═══════════════════════════════════════════════════════════════

    def _get_current_decision_id(self) -> Optional[str]:
        decision_id = self._bus_get(VotingBusKeys.DECISION_ID)
        if decision_id:
            return decision_id
        decision_id = self._bus_get(VotingBusKeys.KERNEL_DECISION_ID)
        if decision_id:
            return decision_id
        return self._bus_get("decision_id")

    def _get_current_tick_ts(self) -> Optional[str]:
        ts = self._bus_get(VotingBusKeys.TICK_TS)
        if ts:
            return ts
        return self._bus_get(VotingBusKeys.KERNEL_TICK_TS)

    def _is_same_decision_cycle(self, decision_id: str) -> bool:
        return decision_id == self._last_decision_id

    def _check_data_freshness(self, timestamp: str, max_age_seconds: Optional[float] = None) -> bool:
        if not timestamp:
            return False

        max_age = max_age_seconds or self._voting_defaults.get("max_staleness_seconds", 15.0)

        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
            age = now - ts
            return age.total_seconds() < float(max_age)
        except Exception:
            self._log_warning("Failed to parse timestamp for freshness check", ts=timestamp)
            return True

    def get_decision_context(self) -> Dict[str, Any]:
        decision_id = self._get_current_decision_id()
        tick_ts = self._get_current_tick_ts()

        context: Dict[str, Any] = {
            "decision_id": decision_id,
            "tick_ts": tick_ts,
            "is_fresh": None,
        }

        if decision_id is None or tick_ts is None:
            self._log_warning("Missing decision context on bus", decision_id=decision_id, tick_ts=tick_ts)
        else:
            is_fresh = self._check_data_freshness(
                tick_ts,
                self._voting_defaults.get("max_staleness_seconds", 15.0),
            )
            context["is_fresh"] = is_fresh
            if not is_fresh:
                self._log_warning("Stale decision context detected", tick_ts=tick_ts, decision_id=decision_id)

        self._last_decision_id = decision_id
        self._debug_log(event="decision_context", **context)
        return context

    # ═══════════════════════════════════════════════════════════════
    # Performance Tracking
    # ═══════════════════════════════════════════════════════════════

    def _record_performance(self, metric_name: str, value: float, success: bool = True) -> None:
        if self.performance_tracker:
            try:
                self.performance_tracker.record_metric(self._module_name, metric_name, value, success)
            except Exception:
                pass

    def _time_operation(self, operation_name: str) -> "_OperationTimer":
        return _OperationTimer(self, operation_name)

    # ═══════════════════════════════════════════════════════════════
    # Health & Status / Circuit Breaker
    # ═══════════════════════════════════════════════════════════════

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "module": self._module_name,
            "status": self._health_status,
            "process_count": self._process_count,
            "consecutive_failures": self._consecutive_failures,
            "last_process_time": self._last_process_time,
            "smart_bus_enabled": self.smart_bus_enabled,
            "circuit_open": self.circuit_open,
            "circuit_open_until": self._circuit_open_until.isoformat() if self._circuit_open_until else None,
            "circuit_open_reason": self._circuit_open_reason,
            "last_error": self._last_error,
        }

    @property
    def circuit_open(self) -> bool:
        """
        Circuit breaker considers:
        - consecutive failure threshold
        - optional cooldown window (_circuit_open_until)
        """
        if self._circuit_open_until is not None:
            if datetime.now() < self._circuit_open_until:
                return True
            # cooldown expired: clear it
            self._circuit_open_until = None
            self._circuit_open_reason = None
        return self._consecutive_failures >= int(CIRCUIT_BREAKER_THRESHOLD)

    def _mark_success(self) -> None:
        self._consecutive_failures = 0
        self._health_status = "ok"
        self._last_error = None
        self._circuit_open_until = None
        self._circuit_open_reason = None

        self._process_count += 1
        self._last_process_time = datetime.now().isoformat()
        self._debug_log(event="mark_success", process_count=self._process_count)

    def _mark_failure(self, error: Optional[str] = None, cooldown_seconds: Optional[float] = None) -> None:
        self._consecutive_failures += 1

        if self._consecutive_failures >= int(CIRCUIT_BREAKER_THRESHOLD):
            self._health_status = "critical"
        elif self._consecutive_failures >= 5:
            self._health_status = "degraded"

        if error:
            self._last_error = error
            self._circuit_open_reason = error
            self._log_warning(
                "Processing failure",
                error=error,
                consecutive_failures=self._consecutive_failures,
            )

        if cooldown_seconds is not None and cooldown_seconds > 0:
            self._circuit_open_until = datetime.now() + timedelta(seconds=float(cooldown_seconds))

        self._debug_log(
            event="mark_failure",
            error=error,
            consecutive_failures=self._consecutive_failures,
            status=self._health_status,
            cooldown_seconds=cooldown_seconds,
        )

    # ═══════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            f = float(value)
            return f if np.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_clip(value: float, low: float, high: float) -> float:
        try:
            return float(np.clip(value, low, high))
        except Exception:
            return (low + high) / 2.0

    @staticmethod
    def _utcnow() -> str:
        return datetime.utcnow().isoformat() + "Z"

    # ═══════════════════════════════════════════════════════════════
    # Safe wrapper for subclass process implementations (optional use)
    # ═══════════════════════════════════════════════════════════════

    def _safe_process_wrapper(self, fn_name: str, exc: Exception) -> str:
        """
        Produce a stable, operator-friendly error message while capturing forensic info.
        """
        msg = str(exc)
        try:
            if self.error_pinpointer is not None:
                msg = str(self.error_pinpointer.analyze_error(exc, fn_name))
        except Exception:
            pass

        self._debug_log(
            event="process_exception",
            function=fn_name,
            error=str(exc),
            message=msg,
            traceback=traceback.format_exc(),
        )
        return msg


class _NullPerformanceTracker:
    """Stub performance tracker when the real one is unavailable."""

    def record_metric(self, module: str, metric: str, value: float, success: bool = True) -> None:
        return

    def get_metrics(self, module: Optional[str] = None) -> Dict[str, Any]:
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
            self.module._record_performance(self.operation_name, elapsed_ms, exc_type is None)

            if elapsed_ms > float(MAX_PROCESSING_TIME_MS):
                self.module._log_warning(
                    "Operation exceeded max processing time",
                    operation=self.operation_name,
                    elapsed_ms=elapsed_ms,
                    max_ms=MAX_PROCESSING_TIME_MS,
                )

        return False
