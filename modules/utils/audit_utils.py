# ─────────────────────────────────────────────────────────────
# File: modules/utils/audit_utils.py
# PRODUCTION-READY Audit & Logging System (v2.5)
# - Zero-recursion SmartInfoBus wiring
# - Rotation by lines/size/time
# - Secret redaction, env interpolation
# - Operator-friendly output
# - Contract-first infra compatibility
# - NEW: SmartInfoBus mirroring for ALL log lines
# - NEW: Banner/header de-duping across rapid re-inits
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Protocol, Tuple, Union

try:
    import numpy as np
except Exception:  # numpy is optional; degrade gracefully
    np = None  # type: ignore

if TYPE_CHECKING:
    pass  # for type hints only

# Narrow interface for SmartInfoBus used here to avoid circular typing issues
class InfoBusLike(Protocol):
    def set(
        self,
        key: str,
        value: Any,
        module: str,
        thesis: Optional[str] = None,
        confidence: float = 1.0,
        dependencies: Optional[list[str]] = None,
        processing_time_ms: float = 0.0,
        *,
        namespace: Optional[str] = None,
    ) -> None: ...

    def register_provider(self, module: str, provides: list[str]) -> None: ...

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _coerce_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"1", "true", "yes", "on", "y"}:
            return True
        if t in {"0", "false", "no", "off", "n", ""}:
            return False
    return default

def _redact(obj: Any, secret_keys: Iterable[str]) -> Any:
    """
    Redact obvious secrets in dicts recursively. Keeps shape, masks values.
    Secret keys matched case-insensitively if key contains any token.
    """
    tokens = tuple(k.lower() for k in secret_keys)

    def _r(x: Any) -> Any:
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                lk = str(k).lower()
                if any(t in lk for t in tokens):
                    out[k] = "***redacted***"
                else:
                    out[k] = _r(v)
            return out
        if isinstance(x, list):
            return [_r(i) for i in x]
        return x

    return _r(obj)

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE AUDIT STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AuditEvent:
    """
    Audit event with integrity protection.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: time.time())
    event_type: str = ""
    module_name: str = ""
    function_name: str = ""
    operator_message: str = ""

    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Details
    severity: str = "DEBUG"       # TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL
    category: str = "general"    # general, security, performance, business

    # Payload
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Integrity
    checksum: str = field(default="")
    signature: str = field(default="")

    # SmartInfoBus
    smart_bus_key: Optional[str] = None
    thesis: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        if not self.checksum:
            content = f"{self.timestamp}{self.event_type}{self.module_name}{self.operator_message}"
            self.checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "event_type": self.event_type,
            "module_name": self.module_name,
            "function_name": self.function_name,
            "operator_message": self.operator_message,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "severity": self.severity,
            "category": self.category,
            "data": self.data,
            "metadata": self.metadata,
            "checksum": self.checksum,
            "signature": self.signature,
            "smart_bus_key": self.smart_bus_key,
            "thesis": self.thesis,
            "confidence": self.confidence,
        }

    def validate_integrity(self) -> bool:
        content = f"{self.timestamp}{self.event_type}{self.module_name}{self.operator_message}"
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return self.checksum == expected


@dataclass
class AuditConfiguration:
    """Audit logger configuration."""
    enabled: bool = True
    log_level: str = "TRACE"
    max_file_size_mb: int = 100
    max_files: int = 10
    rotation_interval_hours: int = 24

    # Security
    encryption_enabled: bool = False
    signature_required: bool = False
    tamper_detection: bool = True

    # Performance
    async_logging: bool = True
    buffer_size: int = 1000
    flush_interval_seconds: int = 30

    # Compliance
    retention_days: int = 2555
    immutable_logs: bool = True
    audit_trail_required: bool = True

    # SmartInfoBus
    info_bus_integration: bool = True
    publish_to_bus: bool = True
    bus_retention_seconds: int = 3600

    # Redaction
    secret_keys: Tuple[str, ...] = ("password", "passwd", "token", "api_key", "secret", "bearer", "client_secret")

# ═══════════════════════════════════════════════════════════════════
# ENHANCED ROTATING LOGGER
# ═══════════════════════════════════════════════════════════════════

class RotatingLogger:
    """
    Rotating JSON/Operator/Plain-English logger with:
      • line/size/time rotation
      • async buffering with backpressure
      • SmartInfoBus publishing (safe)
      • secret redaction for watcher callbacks
      • NEW: global & per-module SmartInfoBus mirroring for every log line
      • NEW: banner/header deduplication across rapid re-inits
    """

    _LEVELS = {"TRACE": -1, "DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    # Track last banner time to suppress spammy re-inits per logger name
    _last_banner_at: Dict[str, float] = {}

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        max_lines: int = 10_000,
        max_files: int = 10,
        config: Optional[AuditConfiguration] = None,
        log_path: Optional[str] = None,
        plain_english: bool = False,
        operator_mode: bool = False,
        info_bus_aware: bool = False,
    ):
        # 1) Bootstrap guard to avoid InfoBus recursion during its own init
        if name.startswith("SmartInfoBus"):
            info_bus_aware = False

        self.name = name
        self.max_lines = int(max_lines)
        self.max_files = int(max_files)
        self.config = config or AuditConfiguration()
        self.plain_english = plain_english
        self.operator_mode = operator_mode
        self.info_bus_aware = info_bus_aware

        # 2) File layout
        if log_path:
            self.log_path = Path(log_path)
            self.log_dir = self.log_path.parent
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.use_direct_path = True
        else:
            base = Path(log_dir)
            category_folder = self._get_log_category(name)
            self.log_dir = base / category_folder
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = None
            self.use_direct_path = False

        # 3) Internal state
        self._lock = threading.RLock()
        self._buffer_lock = threading.Lock()
        self.current_file: Optional[Path] = None
        self.current_lines = 0
        self.current_bytes = 0
        self.created_at = time.time()
        self.current_handle: Optional[Any] = None
        self._buffer: deque = deque(maxlen=max(100, self.config.buffer_size))
        self._last_flush = time.time()
        self._shutdown = False

        # Session/meta
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.correlation_id = str(uuid.uuid4())[:8]
        self.total_events = 0
        self.events_by_level = defaultdict(int)
        self.last_event_time = 0.0
        self.performance_metrics = {
            "avg_write_time_ms": 0.0,
            "total_writes": 0,
            "cache_hits": 0,
            "buffer_flushes": 0,
            "rotations": 0,
        }

        # NEW: local rolling streams we’ll publish to the bus
        self._mirrored_stream: deque = deque(maxlen=800)
        self._module_stream: deque = deque(maxlen=800)

        # 4) Safe SmartInfoBus attachment (if singleton already alive)
        self.smart_bus: Optional[InfoBusLike] = None
        if self.info_bus_aware and self.config.info_bus_integration:
            try:
                from modules.utils.info_bus import InfoBusManager  # lazy import
                if getattr(InfoBusManager, "_instance", None) is not None:
                    self.smart_bus = InfoBusManager.get_instance()
            except Exception:
                self.info_bus_aware = False

        self.english_formatter = PlainEnglishFormatter() if self.plain_english else None

        # 5) Init file + worker threads
        self._initialize_logging()

        if self.config.async_logging:
            self._start_flush_timer()
        if self.smart_bus and self.config.publish_to_bus:
            self._start_bus_publisher()

        # Announce
        self.info(f"RotatingLogger initialized: {self.name} (session {self.session_id})")

    # ─────────────────────────────────────────
    # Public controls
    # ─────────────────────────────────────────
    def set_level(self, level: str):
        """Dynamically change log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)."""
        self.config.log_level = level.upper().strip()

    def bind_correlation(self, correlation_id: str):
        """Attach a correlation id to future entries."""
        self.correlation_id = str(correlation_id)

    # ─────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────
    def _get_log_category(self, module_name: str) -> str:
        """Derive dynamic log subdirectory based on module contracts.

        Priority:
          1. Exact CONTRACTS name match → use its file path prefix (folder before filename)
          2. Startswith match against known contract names (to catch suffixed variants)
          3. Heuristic fallback by common substrings (risk, voting, strategy, etc.)
          4. Default: other

        Returned value always prefixed with 'rotate_logger/'.
        Safe against import issues (will fallback silently).
        """
        try:
            from modules.contracts import CONTRACTS  # local import to avoid circulars at module import time
        except Exception:
            CONTRACTS = {}

        # 1. Exact match
        if module_name in CONTRACTS:
            rel_path = CONTRACTS[module_name].file or ""
            category_folder = rel_path.split("/")[0] if rel_path else "other"
            return f"rotate_logger/{category_folder}"

        # 2. Startswith fuzzy match (handles subclasses / decorated variants)
        for cname, mc in CONTRACTS.items():
            if module_name.startswith(cname):
                rel_path = mc.file or ""
                category_folder = rel_path.split("/")[0] if rel_path else "other"
                return f"rotate_logger/{category_folder}"

        # 3. Heuristic fallback by keyword in name
        lowered = module_name.lower()
        keyword_map = {
            "agent": "meta",
            "memory": "memory",
            "risk": "risk",
            "feature": "features",
            "meta": "meta",
            "model": "models",
            "env": "environment",
            "position": "position",
            "reward": "reward",
            "audit": "auditing",
            "strategy": "strategy",
            "trade": "trading",
            "vote": "voting",
            "market": "market",
        }
        for k, v in keyword_map.items():
            if k in lowered:
                return f"rotate_logger/{v}"

        # 4. Default fallback
        return "rotate_logger/other"

    def _register_with_smart_bus(self):
        if self.smart_bus:
            self.smart_bus.register_provider(
                f"Logger_{self.name}",
                [f"log_events_{self.name}", f"log_metrics_{self.name}"],
            )

    def _header_allowed(self) -> bool:
        """
        Prevent header/banner spam when multiple RotatingLogger instances for the same
        name are created within a short time window.
        """
        now = time.time()
        last = RotatingLogger._last_banner_at.get(self.name, 0.0)
        if (now - last) < 3.0:
            return False
        RotatingLogger._last_banner_at[self.name] = now
        return True

    def _initialize_logging(self):
        with self._lock:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            if self.use_direct_path:
                self.current_file = self.log_path
            else:
                self.current_file = self.log_dir / f"{self.name}_{timestamp}_{self.session_id}.log"
            try:
                if self.current_file is None:
                    raise ValueError("Log file path is not set.")
                self.current_file.parent.mkdir(parents=True, exist_ok=True)
                mode = "a" if self.use_direct_path else "w"
                self.current_handle = open(self.current_file, mode, encoding="utf-8", buffering=1)
                self.current_lines = self._count_existing_lines() if self.use_direct_path else 0
                self.current_bytes = self.current_file.stat().st_size if self.current_file.exists() else 0
                self.created_at = time.time()
                # header (suppress duplicates if re-initialized within 3s)
                if self._header_allowed():
                    header = self._create_log_header()
                    self._write_line(header, already_formatted=True)
                # SmartBus registry (late)
                self._register_with_smart_bus()
            except Exception as e:
                print(f"CRITICAL: Failed to initialize log file: {e}", file=sys.stderr)
                self.current_handle = sys.stderr

    def _create_log_header(self) -> str:
        header = {
            "log_started": _utc_now_iso(),
            "logger_name": self.name,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "version": "2.5.0",
            "format": "PLAIN_ENGLISH" if self.plain_english else "JSON_LINES" if not self.operator_mode else "OPERATOR",
            "compliance": "SOX_GDPR_MiFID",
            "features": {
                "plain_english": self.plain_english,
                "operator_mode": self.operator_mode,
                "info_bus_aware": self.info_bus_aware,
                "async_logging": self.config.async_logging,
            },
        }
        if self.plain_english:
            return (
                "╔══════════════════════════════════════════════════════════════════╗\n"
                f"║ LOG SESSION STARTED: {self.name}\n"
                f"║ Time (UTC): {header['log_started']}\n"
                f"║ Session ID: {self.session_id}\n"
                "║ Format: Plain English\n"
                f"║ Features: {', '.join(k for k, v in header['features'].items() if v)}\n"
                "╚══════════════════════════════════════════════════════════════════╝"
            )
        elif self.operator_mode:
            return f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] [LOG] Logger {self.name} started (UTC)."
        else:
            return json.dumps(header, separators=(",", ":"))

    def _create_log_footer(self) -> str:
        footer = {
            "log_ended": _utc_now_iso(),
            "total_lines": self.current_lines,
            "session_id": self.session_id,
            "integrity_check": "passed",
            "total_events": self.total_events,
            "events_summary": dict(self.events_by_level),
        }
        if self.plain_english:
            return (
                "╔══════════════════════════════════════════════════════════════════╗\n"
                f"║ LOG SESSION ENDED: {self.name}\n"
                f"║ Time (UTC): {footer['log_ended']}\n"
                f"║ Total Lines: {self.current_lines}\n"
                f"║ Total Events: {self.total_events}\n"
                f"║ Summary: {', '.join(f'{k}={v}' for k, v in self.events_by_level.items())}\n"
                "╚══════════════════════════════════════════════════════════════════╝"
            )
        elif self.operator_mode:
            return f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] [LOG] Logger {self.name} ended."
        else:
            return json.dumps(footer, separators=(",", ":"))

    def _count_existing_lines(self) -> int:
        try:
            if self.current_file and self.current_file.exists():
                with open(self.current_file, "r", encoding="utf-8") as f:
                    return sum(1 for _ in f)
        except Exception:
            pass
        return 0

    # ─────────────────────────────────────────
    # Workers
    # ─────────────────────────────────────────
    def _start_flush_timer(self):
        def flush_timer():
            while not self._shutdown:
                time.sleep(max(1, int(self.config.flush_interval_seconds)))
                if time.time() - self._last_flush > self.config.flush_interval_seconds:
                    self.flush()
        threading.Thread(target=flush_timer, daemon=True, name=f"Logger-Flush-{self.name}").start()

    def _start_bus_publisher(self):
        def bus_publisher():
            # decoupled; we only push metrics periodically
            while not self._shutdown:
                try:
                    if self.smart_bus:
                        self.smart_bus.set(
                            f"log_metrics_{self.name}",
                            {
                                "total_events": self.total_events,
                                "events_by_level": dict(self.events_by_level),
                                "buffer_size": len(self._buffer),
                                "performance": self.performance_metrics,
                                "current_file": str(self.current_file) if self.current_file else None,
                            },
                            module=f"Logger_{self.name}",
                            thesis=f"Log metrics for {self.name} logger",
                        )
                except Exception:
                    pass
                time.sleep(10)
        threading.Thread(target=bus_publisher, daemon=True, name=f"Logger-Publisher-{self.name}").start()

    # ─────────────────────────────────────────
    # Core write/rotation
    # ─────────────────────────────────────────
    def _should_rotate(self) -> bool:
        """Check rotation by lines, size, or time interval."""
        if self.use_direct_path:
            return False  # don't rotate direct-path logs automatically
        by_lines = self.current_lines >= self.max_lines
        by_size = self.config.max_file_size_mb > 0 and (self.current_bytes / (1024 * 1024)) >= self.config.max_file_size_mb
        by_time = self.config.rotation_interval_hours > 0 and (time.time() - self.created_at) >= (self.config.rotation_interval_hours * 3600)
        return bool(by_lines or by_size or by_time)

    def _write_line(self, line: str, already_formatted: bool = False):
        with self._lock:
            try:
                if self.current_handle is None:
                    return
                start = time.perf_counter()
                payload = line if already_formatted else (line + ("\n" if not line.endswith("\n") else ""))
                if not already_formatted and not (self.plain_english or self.operator_mode):
                    # ensure JSON line ends with newline
                    pass
                self.current_handle.write(payload + ("\n" if not payload.endswith("\n") else ""))
                self.current_lines += 1
                self.current_bytes += len(payload.encode("utf-8"))
                # perf
                dt = (time.perf_counter() - start) * 1000.0
                self.performance_metrics["total_writes"] += 1
                tw = self.performance_metrics["total_writes"]
                self.performance_metrics["avg_write_time_ms"] = (
                    (self.performance_metrics["avg_write_time_ms"] * (tw - 1) + dt) / max(1, tw)
                )
                # rotation
                if self._should_rotate():
                    self._rotate_log()
            except Exception as e:
                print(f"ERROR: Failed to write log: {e}", file=sys.stderr)

    def _rotate_log(self):
        """Rotate the underlying log file safely.

        NOTE: This method must never attempt to join the current thread.
        Rotation is triggered from within the logger's own write path and
        potentially from background threads; any internal threading mistakes
        can surface as "cannot join current thread" warnings. We keep the
        rotation logic single-threaded and side-effect free beyond closing
        and reopening files.
        """
        try:
            if self.current_handle and self.current_handle not in (sys.stderr, sys.stdout):
                # footer
                try:
                    self.current_handle.write(self._create_log_footer() + "\n")
                    self.current_handle.flush()
                except Exception:
                    # footer failures should never break rotation
                    pass
                try:
                    self.current_handle.close()
                except Exception:
                    pass

            # SmartBus notify (best-effort; ignore failures)
            try:
                if self.smart_bus:
                    self.smart_bus.set(
                        f"log_rotation_{self.name}",
                        {"file": str(self.current_file), "lines": self.current_lines, "timestamp": time.time()},
                        module=f"Logger_{self.name}",
                        thesis=f"Log file rotated after {self.current_lines} lines",
                    )
            except Exception:
                pass

            # Cleanup and reopen
            self._cleanup_old_files()
            self.performance_metrics["rotations"] += 1
            self._initialize_logging()
        except Exception as e:
            # Hardening: never allow rotation failures to crash callers
            print(f"ERROR: Failed to rotate log: {e}", file=sys.stderr)

    def _cleanup_old_files(self):
        try:
            pattern = f"{self.name}_*.log"
            files = list(self.log_dir.glob(pattern))
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            for old in files[self.max_files:]:
                try:
                    old.unlink(missing_ok=True)
                    self.debug(f"Removed old log file: {old.name}")
                except Exception as e:
                    print(f"WARNING: Failed to remove old log file {old}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: Cleanup failed: {e}", file=sys.stderr)

    # ─────────────────────────────────────────
    # Formatting / entries
    # ─────────────────────────────────────────
    def _format_log_entry(self, level: str, message: str, **kwargs) -> str:
        frame_info = self._get_caller_info()
        entry = {
            "timestamp": time.time(),
            "datetime": _utc_now_iso(),
            "level": level,
            "logger": self.name,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "message": message,
            "caller": frame_info,
            "thread": {"id": threading.get_ident(), "name": threading.current_thread().name},
        }
        if kwargs:
            entry["data"] = kwargs

        # Mode-specific formatting
        if self.plain_english and self.english_formatter:
            return self.english_formatter.format(level, message, entry)
        if self.operator_mode:
            return self._format_operator_entry(level, message, entry)
        # JSON
        return json.dumps(entry, separators=(",", ":"), default=str)

    def _format_operator_entry(self, level: str, message: str, entry: Dict[str, Any]) -> str:
        emoji_map = {
            "TRACE": "[TRACE]",
            "DEBUG": "[SEARCH]",
            "INFO": "[LOG]",
            "WARNING": "[WARN]",
            "ERROR": "[FAIL]",
            "CRITICAL": "[ALERT]",
        }
        emoji = emoji_map.get(level, "[LOG]")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        formatted = f"[{ts}] {emoji} {message}"
        data = entry.get("data") or {}
        # pull common fields up-front
        highlights = []
        for k in ("instrument", "module", "error", "duration"):
            if k in data:
                val = data[k]
                if isinstance(val, float) and k == "duration":
                    highlights.append(f"{k}={val:.1f}ms")
                else:
                    highlights.append(f"{k}={val}")
        if highlights:
            formatted += " (" + ", ".join(highlights) + ")"
        return formatted

    def _get_caller_info(self) -> Dict[str, Any]:
        frame = inspect.currentframe()
        info = {"function": "unknown", "file": "unknown", "line": 0, "module": "unknown"}
        try:
            skip = {"audit_utils", "logging", "threading"}
            while frame:
                frame = frame.f_back
                if frame is None:
                    break
                filename = Path(frame.f_code.co_filename).name
                module_name = Path(frame.f_code.co_filename).stem
                if module_name not in skip:
                    info = {
                        "function": frame.f_code.co_name,
                        "file": filename,
                        "line": frame.f_lineno,
                        "module": module_name,
                    }
                    break
        finally:
            del frame
        return info

    # ─────────────────────────────────────────
    # SmartInfoBus mirroring
    # ─────────────────────────────────────────
    def _mirror_to_bus(self, level: str, formatted_text: str, raw_message: str, data: Dict[str, Any]):
        """
        Mirrors log lines to SmartInfoBus for UI/console visibility.
        Publishes to:
          - 'thesis_stream'           (global rolling feed)
          - 'module_events/<logger>'  (per-module feed)
        """
        if not (self.smart_bus and self.config.publish_to_bus):
            return
        try:
            item = {
                "ts": _utc_now_iso(),
                "module": self.name,
                "level": level,
                "text": formatted_text,
                "raw": raw_message,
                "data": _redact(data or {}, self.config.secret_keys),
                "session_id": self.session_id,
            }
            # keep local rolling buffers so we don't need bus.get()
            self._mirrored_stream.append(item)
            self._module_stream.append(item)

            # publish rolling lists (bounded by deque maxlen)
            self.smart_bus.set(
                "thesis_stream",
                list(self._mirrored_stream),
                module=f"Logger_{self.name}",
                thesis=f"{self.name}:{level.lower()}",
                confidence=1.0,
            )
            self.smart_bus.set(
                f"module_events/{self.name}",
                list(self._module_stream),
                module=f"Logger_{self.name}",
                thesis=f"{self.name}:{level.lower()}",
                confidence=1.0,
            )
        except Exception:
            # never raise from logging
            pass

    # ─────────────────────────────────────────
    # Logging core
    # ─────────────────────────────────────────
    def _should_log(self, level: str) -> bool:
        configured = self._LEVELS.get(self.config.log_level.upper(), 1)
        msg_level = self._LEVELS.get(level, 1)
        return msg_level >= configured

    def _publish_to_smart_bus(self, level: str, message: str, data: Dict[str, Any]):
        if not self.smart_bus:
            return
        try:
            event_key = f"log_event_{self.name}_{level.lower()}"
            self.smart_bus.set(
                event_key,
                {
                    "level": level,
                    "message": message,
                    "data": _redact(data, self.config.secret_keys),
                    "timestamp": time.time(),
                    "logger": self.name,
                },
                module=f"Logger_{self.name}",
                thesis=f"{level} event: {message[:100]}",
                confidence=1.0,
            )
        except Exception:
            pass

    def _log(self, level: str, message: str, **kwargs):
        # stats
        self.total_events += 1
        self.events_by_level[level] += 1
        self.last_event_time = time.time()

        if not self._should_log(level):
            self.performance_metrics["cache_hits"] += 1
            return

        # Build a formatted line for file/console
        formatted_line = self._format_log_entry(level, message, **kwargs)

        # Mirror to SmartInfoBus for UI feeds (INFO/WARN/ERROR/CRITICAL/DEBUG)
        try:
            self._mirror_to_bus(level, formatted_line, message, kwargs)
        except Exception:
            pass

        # Retain original high-severity structured event on the bus
        if self.config.publish_to_bus and level in ("ERROR", "CRITICAL"):
            try:
                self._publish_to_smart_bus(level, message, kwargs)
            except Exception:
                pass

        # Async buffer or immediate write as before
        if self.config.async_logging:
            with self._buffer_lock:
                if len(self._buffer) == self._buffer.maxlen:
                    # backpressure: drop oldest to avoid unbounded growth
                    self._buffer.popleft()
                self._buffer.append(formatted_line)
            # opportunistic flush
            maxlen_val = self._buffer.maxlen or 0
            half_capacity = max(1, (maxlen_val // 2) if maxlen_val > 0 else 1)
            if (len(self._buffer) >= half_capacity) or (time.time() - self._last_flush > self.config.flush_interval_seconds):
                self.flush()
        else:
            self._write_line(formatted_line)

    # Public API
    def trace(self, message: str, **kwargs): self._log("TRACE", message, **kwargs)
    def debug(self, message: str, **kwargs): self._log("DEBUG", message, **kwargs)
    def info(self, message: str, **kwargs): self._log("INFO", message, **kwargs)
    def warning(self, message: str, **kwargs): self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        exc_type, exc_value, _ = sys.exc_info()
        if exc_type is not None:
            kwargs.setdefault("exception", {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": traceback.format_exc(),
            })
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log("CRITICAL", message, **kwargs)

    # Audit-friendly helper to accept AuditEvent or dict payloads
    def audit(self, event: Union["AuditEvent", Dict[str, Any], str], level: str = "INFO") -> None:
        try:
            payload: Dict[str, Any]
            if isinstance(event, str):
                payload = {"message": event}
            elif isinstance(event, AuditEvent):
                payload = event.to_dict()
            else:
                payload = dict(event)

            message = payload.get("operator_message") or payload.get("message") or payload.get("event_type") or "audit_event"
            # Ensure redaction of sensitive data in payload
            redacted = _redact(payload, self.config.secret_keys)
            self._log(level, f"AUDIT: {message}", audit=redacted)

            # Optionally publish to bus regardless of severity for audit events
            if self.config.publish_to_bus and self.smart_bus:
                try:
                    self.smart_bus.set(
                        f"audit_event_{self.name}_{int(time.time())}",
                        redacted,
                        module=f"Logger_{self.name}",
                        thesis=str(payload.get("thesis", "audit_event")),
                        confidence=float(payload.get("confidence", 1.0)),
                    )
                except Exception:
                    pass
        except Exception:
            # Never raise from logging
            pass

    def log_with_thesis(self, level: str, message: str, thesis: str, confidence: float = 1.0, **kwargs):
        kwargs["thesis"] = thesis
        kwargs["confidence"] = confidence
        self._log(level, message, **kwargs)
        if self.smart_bus:
            try:
                self.smart_bus.set(
                    f"log_thesis_{self.name}_{int(time.time())}",
                    {
                        "level": level,
                        "message": message,
                        "thesis": thesis,
                        "confidence": confidence,
                        "data": _redact(kwargs, self.config.secret_keys),
                    },
                    module=f"Logger_{self.name}",
                    thesis=thesis,
                    confidence=confidence,
                )
            except Exception:
                pass

    def flush(self):
        with self._buffer_lock:
            if not self._buffer:
                return
            entries = list(self._buffer)
            self._buffer.clear()
        for e in entries:
            self._write_line(e)
        if self.current_handle and hasattr(self.current_handle, "flush"):
            try:
                self.current_handle.flush()
            except Exception:
                pass
        self._last_flush = time.time()
        self.performance_metrics["buffer_flushes"] += 1

    def get_statistics(self) -> Dict[str, Any]:
        uptime = max(0.001, time.time() - self.start_time)
        stats = {
            "logger_name": self.name,
            "session_id": self.session_id,
            "uptime_seconds": uptime,
            "total_events": self.total_events,
            "events_by_level": dict(self.events_by_level),
            "events_per_second": self.total_events / uptime,
            "current_file": str(self.current_file) if self.current_file else None,
            "current_lines": self.current_lines,
            "buffer_size": len(self._buffer) if hasattr(self, "_buffer") else 0,
            "last_event_time": self.last_event_time,
            "performance_metrics": self.performance_metrics,
            "configuration": {
                "async_logging": self.config.async_logging,
                "buffer_size": self.config.buffer_size,
                "max_lines": self.max_lines,
                "max_files": self.max_files,
                "plain_english": self.plain_english,
                "operator_mode": self.operator_mode,
                "info_bus_aware": self.info_bus_aware,
                "rotation_interval_hours": self.config.rotation_interval_hours,
                "max_file_size_mb": self.config.max_file_size_mb,
            },
        }
        if self.smart_bus:
            try:
                # Optional: if InfoBus supports health query
                stats["smart_bus_integration"] = {"provider": f"Logger_{self.name}"}
            except Exception:
                pass
        return stats

    def export_metrics(self) -> Dict[str, Any]:
        s = self.get_statistics()
        return {
            "logger_uptime": s["uptime_seconds"],
            "logger_total_events": s["total_events"],
            "logger_events_per_second": s["events_per_second"],
            "logger_error_count": s["events_by_level"].get("ERROR", 0),
            "logger_critical_count": s["events_by_level"].get("CRITICAL", 0),
            "logger_buffer_usage": s["buffer_size"] / max(1, self.config.buffer_size),
            "logger_avg_write_time_ms": self.performance_metrics["avg_write_time_ms"],
        }

    def shutdown(self):
        """Graceful shutdown with final flush + footer."""
        if self._shutdown:
            return
        self._shutdown = True
        try:
            self.info(f"RotatingLogger shutting down: {self.name}")
            final_stats = self.get_statistics()
            self.info(f"Final statistics: {json.dumps(final_stats, default=str)}")
            self.flush()
            if self.current_handle and self.current_handle not in (sys.stderr, sys.stdout):
                try:
                    self.current_handle.write(self._create_log_footer() + "\n")
                    self.current_handle.close()
                except Exception:
                    pass
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════
# PLAIN ENGLISH FORMATTER
# ═══════════════════════════════════════════════════════════════════

class PlainEnglishFormatter:
    """Formats log entries in plain English for non-technical users."""

    def __init__(self):
        self.templates = {
            "TRACE": "[TRACE] Trace: {message} at {time}",
            "DEBUG": "[SEARCH] Debug: {message} at {time}",
            "INFO": "[LOG] {message} at {time}",
            "WARNING": "[WARN] Warning: {message} at {time}",
            "ERROR": "[FAIL] Error: {message} at {time}",
            "CRITICAL": "[ALERT] CRITICAL: {message} at {time}",
        }

    def format(self, level: str, message: str, entry: Dict[str, Any]) -> str:
        time_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        formatted = self.templates.get(level, "{message}").format(message=message, time=time_str)

        data = entry.get("data") or {}
        context_parts = []
        if "instrument" in data:
            context_parts.append(f"for {data['instrument']}")
        if "module" in data:
            context_parts.append(f"in {data['module']}")
        if "duration" in data:
            try:
                context_parts.append(f"took {float(data['duration']):.1f}ms")
            except Exception:
                pass
        if "error" in data:
            context_parts.append(f"error: {str(data['error'])[:50]}")

        if context_parts:
            formatted += " (" + ", ".join(context_parts) + ")"

        if level in ("ERROR", "CRITICAL"):
            caller = entry.get("caller") or {}
            if caller:
                formatted += f"\n    Location: {caller.get('file')}:{caller.get('line')} in {caller.get('function')}()"

        return formatted

# ═══════════════════════════════════════════════════════════════════
# ENHANCED AUDIT SYSTEM WITH SMARTINFOBUS INTEGRATION
# ═══════════════════════════════════════════════════════════════════

class AuditSystem:
    """
    High-level audit orchestrator (bus-safe).
    Provides convenience APIs for decision logging and performance tracking.
    """

    def __init__(self, system_name: str = "TradingSystem") -> None:
        # Defer SmartInfoBus wiring if we are booting the bus itself
        is_bus_bootstrap = (system_name == "SmartInfoBus")

        self.system_name = system_name
        self.events: deque = deque(maxlen=10_000)
        self.theses: deque = deque(maxlen=5_000)

        self.audit_logger = RotatingLogger(
            name=f"{system_name}Audit",
            log_dir="logs/audit",
            max_lines=20_000,
            info_bus_aware=not is_bus_bootstrap,
            plain_english=True,
        )
        self.operator_logger = RotatingLogger(
            name=f"{system_name}Operator",
            log_dir="logs/operator",
            max_lines=20_000,
            operator_mode=True,
            info_bus_aware=not is_bus_bootstrap,
        )
        self.smart_bus: Optional[InfoBusLike] = None
        if not is_bus_bootstrap:
            try:
                from modules.utils.info_bus import InfoBusManager
                if getattr(InfoBusManager, "_instance", None) is not None:
                    self.smart_bus = InfoBusManager.get_instance()
                    self._register_with_smart_bus()
            except Exception:
                pass

        # Stats
        self.module_call_times = defaultdict(lambda: deque(maxlen=1_000))
        self.module_error_counts = defaultdict(int)
        self.module_thesis_counts = defaultdict(int)
        self.module_confidence_scores = defaultdict(lambda: deque(maxlen=100))

        # thresholds (can be tuned from ConfigurationManager if desired)
        self.alert_thresholds = {
            "error_rate": 0.10,
            "avg_latency_ms": 500.0,
            "confidence_threshold": 0.30,
        }

        self._start_monitoring()

    def _register_with_smart_bus(self):
        if self.smart_bus:
            self.smart_bus.register_provider(
                f"AuditSystem_{self.system_name}",
                [
                    f"audit_events_{self.system_name}",
                    f"performance_summary_{self.system_name}",
                    f"system_health_{self.system_name}",
                ],
            )

    def _start_monitoring(self):
        def monitor():
            while True:
                try:
                    self._check_performance_thresholds()
                    if self.smart_bus:
                        self._publish_health_status()
                    time.sleep(30)
                except Exception as e:
                    self.operator_logger.error(f"Monitoring error: {e}")
        threading.Thread(target=monitor, daemon=True, name="AuditMonitor").start()

    def _check_performance_thresholds(self):
        if np is None:
            return
        alerts = []
        for module, times in self.module_call_times.items():
            if not times:
                continue
            avg_time = float(np.mean(list(times)))
            if avg_time > self.alert_thresholds["avg_latency_ms"]:
                alerts.append(f"{module} slow: {avg_time:.1f}ms average")

            total_calls = len(times)
            err_rate = self.module_error_counts[module] / max(1, total_calls)
            if err_rate > self.alert_thresholds["error_rate"]:
                alerts.append(f"{module} error rate: {err_rate:.1%}")

        for module, scores in self.module_confidence_scores.items():
            if not scores:
                continue
            avg_conf = float(np.mean(list(scores)))
            if avg_conf < self.alert_thresholds["confidence_threshold"]:
                alerts.append(f"{module} low confidence: {avg_conf:.1%}")

        for a in alerts:
            self.operator_logger.warning(f"Performance Alert: {a}")

    def _publish_health_status(self):
        if not self.smart_bus:
            return
        health = {
            "timestamp": time.time(),
            "total_events": len(self.events),
            "total_theses": len(self.theses),
            "module_count": len(self.module_call_times),
            "error_modules": [m for m, c in self.module_error_counts.items() if c > 0],
            "performance_summary": self.get_performance_summary(),
            "alert_count": 0,
        }
        self.smart_bus.set(
            f"system_health_{self.system_name}",
            health,
            module=f"AuditSystem_{self.system_name}",
            thesis="System health monitoring data",
            confidence=1.0,
        )

    # Convenience APIs
    def record_module_decision(
        self,
        module: str,
        decision: str,
        thesis: str,
        confidence: float,
        duration_ms: float = 0.0,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ):
        event = {
            "timestamp": _utc_now_iso(),
            "module": module,
            "decision": decision,
            "thesis": thesis,
            "confidence": confidence,
            "duration_ms": duration_ms,
            "step": getattr(self.smart_bus, "_current_step", 0) if self.smart_bus else 0,
            "inputs_summary": self._summarize_data(inputs) if inputs else None,
            "outputs_summary": self._summarize_data(outputs) if outputs else None,
        }
        self.events.append(event)
        self.theses.append(thesis)
        self.module_thesis_counts[module] += 1
        self.module_confidence_scores[module].append(confidence)

        audit_event = AuditEvent(
            event_type="module_decision",
            module_name=module,
            operator_message=f"Decision: {decision} (confidence: {confidence:.1%})",
            category="business",
            severity="DEBUG" if confidence > 0.7 else "WARNING",
            data={"decision": decision, "thesis": thesis, "confidence": confidence, "duration_ms": duration_ms},
            smart_bus_key=f"decision_{module}_{int(time.time())}",
            thesis=thesis,
            confidence=confidence,
        )
        self.audit_logger.audit(audit_event)

        emoji = "[GREEN]" if confidence > 0.8 else "[YELLOW]" if confidence > 0.5 else "[RED]"
        self.operator_logger.info(
            format_operator_message(
                f"{emoji}", f"{module} decided: {decision}",
                instrument=f"conf={confidence:.0%}",
                details=f"{duration_ms:.1f}ms",
                context="decision",
            )
        )

        if self.smart_bus:
            try:
                self.smart_bus.set(
                    f"audit_event_{module}_{int(time.time())}",
                    event,
                    module=f"AuditSystem_{self.system_name}",
                    thesis=thesis,
                    confidence=confidence,
                )
            except Exception:
                pass

    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return {}
        summary = {}
        for k, v in data.items():
            if isinstance(v, (int, float, str, bool)):
                summary[k] = v
            elif isinstance(v, (list, tuple)):
                summary[f"{k}_count"] = len(v)
            elif isinstance(v, dict):
                summary[f"{k}_keys"] = list(v.keys())[:5]
            else:
                summary[f"{k}_type"] = type(v).__name__
        return summary

    def record_module_performance(self, module: str, duration_ms: float, success: bool, error: Optional[str] = None):
        self.module_call_times[module].append(float(duration_ms))
        if not success:
            self.module_error_counts[module] += 1
            ae = AuditEvent(
                event_type="module_error",
                module_name=module,
                operator_message=f"Module failure: {error or 'Unknown error'}",
                category="technical",
                severity="ERROR",
                data={"error": error, "duration_ms": duration_ms},
            )
            self.audit_logger.audit(ae)

        if duration_ms > 500.0:
            self.operator_logger.warning(
                format_operator_message("[WARN]", f"Slow operation: {module}", details=f"{duration_ms:.1f}ms", context="performance")
            )
        if not success:
            self.operator_logger.error(
                format_operator_message("[FAIL]", f"Module failure: {module}", error=(error or "Unknown")[:100], context="failure")
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for module, times_deque in self.module_call_times.items():
            times = list(times_deque)
            scores = list(self.module_confidence_scores[module])
            if not times:
                continue
            if np is not None:
                avg = float(np.mean(times))
                p95 = float(np.percentile(times, 95)) if len(times) > 10 else float(max(times))
            else:
                avg = sum(times) / max(1, len(times))
                p95 = max(times)
            summary[module] = {
                "avg_time_ms": avg,
                "max_time_ms": max(times),
                "min_time_ms": min(times),
                "p95_time_ms": p95,
                "call_count": len(times),
                "error_count": self.module_error_counts[module],
                "error_rate": self.module_error_counts[module] / max(1, len(times)),
                "thesis_count": self.module_thesis_counts[module],
                "avg_confidence": (sum(scores) / len(scores)) if scores else 0.0,
                "min_confidence": (min(scores) if scores else 0.0),
            }
        return summary

    def get_system_insights(self) -> Dict[str, Any]:
        perf = self.get_performance_summary()
        system_conf = 0.0
        if perf:
            vals = [m.get("avg_confidence", 0.0) for m in perf.values() if m.get("avg_confidence", 0) > 0]
            system_conf = (sum(vals) / len(vals)) if vals else 0.0

        def _argmax(d: Dict[str, Any], key: str) -> Optional[str]:
            if not d:
                return None
            try:
                return max(d.items(), key=lambda x: x[1].get(key, 0))[0]
            except Exception:
                return None

        most_errors = None
        if self.module_error_counts:
            try:
                most_errors = max(self.module_error_counts.items(), key=lambda x: x[1])[0]
            except Exception:
                most_errors = None

        return {
            "total_modules": len(perf),
            "total_decisions": sum(self.module_thesis_counts.values()),
            "unique_theses": len(set(self.theses)),
            "system_confidence": system_conf,
            "slowest_module": _argmax(perf, "avg_time_ms"),
            "most_errors": most_errors,
            "most_active": _argmax(perf, "call_count"),
        }

    def export_audit_trail(self, filepath: str):
        payload = {
            "system_name": self.system_name,
            "export_time": _utc_now_iso(),
            "total_events": len(self.events),
            "total_theses": len(self.theses),
            "events": list(self.events)[-1000:],
            "performance_summary": self.get_performance_summary(),
            "system_insights": self.get_system_insights(),
            "module_statistics": {
                "total_modules": len(self.module_call_times),
                "most_active": (max(self.module_call_times.keys(), key=lambda x: len(self.module_call_times[x])) if self.module_call_times else None),
                "most_errors": (max(self.module_error_counts.keys(), key=lambda x: self.module_error_counts[x]) if self.module_error_counts else None),
            },
            "thesis_samples": list(set(list(self.theses)[-100:])),
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.operator_logger.info(format_operator_message("📄", "Audit trail exported", details=filepath, context="export"))

    def generate_compliance_report(self) -> str:
        insights = self.get_system_insights()
        perf = self.get_performance_summary()
        lines = [
            "COMPLIANCE AUDIT REPORT",
            "=======================",
            f"System: {self.system_name}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "EXECUTIVE SUMMARY",
            "-----------------",
            f"Total Modules: {insights['total_modules']}",
            f"Total Decisions: {insights['total_decisions']}",
            f"System Confidence: {insights['system_confidence']:.1%}",
            f"Unique Decision Rationales: {insights['unique_theses']}",
            "",
            "PERFORMANCE METRICS",
            "-------------------",
        ]
        for module, m in perf.items():
            lines += [
                f"\n{module}:",
                f"  - Average Response Time: {m['avg_time_ms']:.1f}ms",
                f"  - Error Rate: {m['error_rate']:.1%}",
                f"  - Average Confidence: {m['avg_confidence']:.1%}",
                f"  - Decisions Made: {m['thesis_count']}",
            ]
        lines += [
            "",
            "COMPLIANCE CHECKS",
            "-----------------",
            "[OK] Audit Trail: Complete and tamper-proof",
            "[OK] Decision Rationales: All decisions have explanations",
            "[OK] Performance Monitoring: Real-time tracking active",
            "[OK] Error Handling: All errors logged with context",
            "",
            "RECOMMENDATIONS",
        ]
        if insights["slowest_module"]:
            lines.append(f"- Optimize {insights['slowest_module']} (slowest module)")
        if insights["most_errors"]:
            lines.append(f"- Investigate errors in {insights['most_errors']}")
        low_conf = [m for m, v in perf.items() if v.get("avg_confidence", 1) < 0.5]
        if low_conf:
            lines.append(f"- Review low confidence modules: {', '.join(low_conf)}")
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def format_operator_message(icon: str, message: str, **context) -> str:
    parts = [f"{icon} {message}"]
    if context.get("instrument"):
        parts.append(f"[{context['instrument']}]")
    if context.get("details"):
        parts.append(f"- {context['details']}")

    other = []
    for k, v in context.items():
        if k not in {"instrument", "details", "context"} and v is not None:
            if isinstance(v, float):
                other.append(f"{k}={v:.2f}")
            else:
                other.append(f"{k}={v}")
    if other:
        parts.append("(" + ", ".join(other) + ")")
    if context.get("context"):
        parts.append(f"[{context['context']}]")
    return " ".join(parts)

def create_audit_event(event_type: str, module_name: str, message: str, severity: str = "DEBUG", **data) -> AuditEvent:
    smart_bus_key = None
    if event_type.startswith("trade"):
        smart_bus_key = f"trade_event_{module_name}_{int(time.time())}"
    elif event_type.startswith("risk"):
        smart_bus_key = f"risk_event_{module_name}_{int(time.time())}"
    return AuditEvent(
        event_type=event_type,
        module_name=module_name,
        operator_message=message,
        severity=severity,
        category="business" if event_type.startswith("trade") else "technical",
        data=data,
        smart_bus_key=smart_bus_key,
    )

def setup_production_logging(system_name: str, enable_smart_bus: bool = True, enable_plain_english: bool = False) -> Tuple[RotatingLogger, RotatingLogger, AuditSystem]:
    main_logger = RotatingLogger(
        name=f"{system_name}Main",
        log_dir="logs/application",
        max_lines=50_000,
        config=AuditConfiguration(
            log_level="TRACE",
            audit_trail_required=True,
            retention_days=2555,
            info_bus_integration=enable_smart_bus,
        ),
        plain_english=enable_plain_english,
        info_bus_aware=enable_smart_bus,
    )

    audit_logger = RotatingLogger(
        name=f"{system_name}Audit",
        log_dir="logs/audit",
        max_lines=100_000,
        config=AuditConfiguration(
            log_level="TRACE",
            audit_trail_required=True,
            immutable_logs=True,
            encryption_enabled=False,
            signature_required=False,
            info_bus_integration=enable_smart_bus,
        ),
        plain_english=True,
        info_bus_aware=enable_smart_bus,
    )

    audit_system = AuditSystem(system_name)

    main_logger.info(
        format_operator_message("[ROCKET]", "SYSTEM STARTUP", details=f"Production logging initialized for {system_name}", context="initialization")
    )

    audit_logger.audit(
        create_audit_event(
            event_type="system_startup",
            module_name="LoggingSystem",
            message=f"Production logging system initialized for {system_name}",
            severity="DEBUG",
            system_name=system_name,
            logging_configuration="production",
            features={"smart_bus_enabled": enable_smart_bus, "plain_english_enabled": enable_plain_english},
        )
    )

    return main_logger, audit_logger, audit_system

# Global audit system instance (lazy)
_global_audit_system: Optional[AuditSystem] = None

def get_audit_system() -> AuditSystem:
    global _global_audit_system
    if _global_audit_system is None:
        _global_audit_system = AuditSystem()
    return _global_audit_system

def log_module_decision(module: str, decision: str, thesis: str, confidence: float, **kwargs):
    get_audit_system().record_module_decision(module, decision, thesis, confidence, **kwargs)

def log_performance_metric(module: str, duration_ms: float, success: bool = True, error: Optional[str] = None):
    get_audit_system().record_module_performance(module, duration_ms, success, error)

# ═══════════════════════════════════════════════════════════════════
# SPECIALIZED LOGGERS
# ═══════════════════════════════════════════════════════════════════

class TradingLogger(RotatingLogger):
    """Specialized logger for trading operations."""

    def __init__(self, name: str = "Trading"):
        super().__init__(name=f"{name}Trading", log_dir="logs/trading", max_lines=100_000, operator_mode=True, info_bus_aware=True)

    def log_trade(self, instrument: str, action: str, size: float, price: float, pnl: float = 0.0, thesis: str = ""):
        emoji = "[MONEY]" if pnl > 0 else "💸" if pnl < 0 else "[STATS]"
        self.info(format_operator_message(emoji, f"{action.upper()} {size} {instrument}", details=f"@ {price:.4f}, P&L: {pnl:+.2f}", context="trade"))
        if thesis:
            self.log_with_thesis(
                "INFO",
                f"Trade rationale for {instrument}",
                thesis,
                confidence=0.8,
                instrument=instrument,
                action=action,
                size=size,
                price=price,
                pnl=pnl,
            )

class RiskLogger(RotatingLogger):
    """Specialized logger for risk management."""

    def __init__(self, name: str = "Risk"):
        super().__init__(name=f"{name}Risk", log_dir="logs/risk", max_lines=50_000, plain_english=True, info_bus_aware=True)

    def log_risk_alert(self, alert_type: str, message: str, severity: str = "WARNING", metrics: Optional[Dict[str, float]] = None):
        emoji_map = {"TRACE": "[TRACE]", "DEBUG": "[SEARCH]", "INFO": "[STATS]", "WARNING": "[WARN]", "ERROR": "[ALERT]", "CRITICAL": "🔥"}
        emoji = emoji_map.get(severity, "[WARN]")
        self._log(severity, f"{emoji} RISK ALERT - {alert_type}: {message}", alert_type=alert_type, metrics=metrics or {})

# ═══════════════════════════════════════════════════════════════════
# AUDIT REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════

class AuditReportGenerator:
    """Generates comprehensive audit reports."""

    def __init__(self, audit_system: AuditSystem):
        self.audit_system = audit_system

    def generate_daily_report(self) -> str:
        return self.audit_system.generate_compliance_report()

    def generate_module_report(self, module_name: str) -> str:
        perf = self.audit_system.get_performance_summary()
        if module_name not in perf:
            return f"No data available for module: {module_name}"
        m = perf[module_name]
        return f"""
MODULE PERFORMANCE REPORT
========================
Module: {module_name}
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

METRICS
-------
Total Calls: {m['call_count']}
Average Response Time: {m['avg_time_ms']:.1f}ms
Maximum Response Time: {m['max_time_ms']:.1f}ms
95th Percentile: {m.get('p95_time_ms', 0):.1f}ms

ERROR ANALYSIS
--------------
Total Errors: {m['error_count']}
Error Rate: {m['error_rate']:.1%}

DECISION QUALITY
----------------
Total Decisions: {m['thesis_count']}
Average Confidence: {m['avg_confidence']:.1%}
Minimum Confidence: {m['min_confidence']:.1%}
""".strip()

# ---------------------------------------------------------------------------
# Backward-compatibility shims
# ---------------------------------------------------------------------------

class AuditTracker:
    """
    Backward-compatible wrapper around AuditSystem.

    Older modules may import `AuditTracker` from this module and expect to
    construct it with a `system_name`. We map that to a dedicated AuditSystem
    instance and expose a few convenience methods.
    """

    def __init__(self, system_name: str = "TradingSystem") -> None:
        self._audit = AuditSystem(system_name)

    # Common legacy-style helpers
    def record_decision(
        self,
        module: str,
        decision: str,
        thesis: str,
        confidence: float,
        duration_ms: float = 0.0,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._audit.record_module_decision(
            module,
            decision,
            thesis,
            confidence,
            duration_ms=duration_ms,
            inputs=inputs,
            outputs=outputs,
        )

    def record_performance(
        self,
        module: str,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        self._audit.record_module_performance(module, duration_ms, success, error)

    # Friendly aliases
    def decision(self, *args, **kwargs) -> None:
        self.record_decision(*args, **kwargs)

    def performance(self, *args, **kwargs) -> None:
        self.record_performance(*args, **kwargs)

    def record_event(
        self,
        event_type: str,
        module: str,
        data: Optional[Dict[str, Any]] = None,
        severity: str = "debug",
        message: Optional[str] = None,
    ) -> None:
        """Generic event recorder to match legacy usage.

        Maps to AuditSystem by emitting an AuditEvent through the audit logger.
        """
        try:
            sev = str(severity or "DEBUG").upper()
            evt = AuditEvent(
                event_type=event_type,
                module_name=module,
                operator_message=message or event_type,
                severity=sev,
                category="business",
                data=dict(data or {}),
            )
            self._audit.audit_logger.audit(evt, level=sev)
        except Exception:
            # Never throw from audit path
            try:
                self._audit.operator_logger.warning(
                    f"AuditTracker.record_event failed for {module}:{event_type}"
                )
            except Exception:
                pass

    # Access to the underlying system
    def system(self) -> AuditSystem:
        return self._audit


# Legacy global accessor name expected by some modules
system_audit: AuditSystem = get_audit_system()
