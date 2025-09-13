# ─────────────────────────────────────────────────────────────
# File: modules/market/debug/trace_logger.py
# Advanced trace logging system with hierarchical levels — Production Upgrade
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, Any, List, Deque
from datetime import datetime, timezone
import traceback
import inspect
import json
from collections import deque
from pathlib import Path
import threading
import queue
import sys
import os
import atexit
import contextvars
import time
import itertools
import copy


# Context variables for correlation
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)
_span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "span_id", default=None
)


class TraceLevel(Enum):
    """Hierarchical trace levels"""
    TRACE = 0    # Most detailed - every variable state
    DEBUG = 1    # Component method calls
    INFO = 2     # Major events
    WARNING = 3  # Potential issues
    ERROR = 4    # Errors and exceptions

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented


def _utc_iso() -> str:
    """Millisecond precision, always UTC with Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TraceLogger:
    """
    Advanced trace logging with hierarchical levels, async I/O, rotation,
    structured JSONL, de-dup/sampling, context correlation, and redaction.

    Backward compatible with existing usage.
    """

    def __init__(
        self,
        name: str,
        trace_level: TraceLevel = TraceLevel.INFO,
        enabled: bool = True,
        max_entries: int = 10000,
        output_file: Optional[str] = None,
        *,
        async_write: bool = True,
        rotate_size_mb: Optional[int] = 10,     # set None to disable rotation
        rotate_backups: int = 3,
        console: bool = True,
        color: Optional[bool] = None,            # auto if None
        redact_keys: Optional[List[str]] = None, # keys to redact in `data`/locals
        sample_by_level: Optional[Dict[str, float]] = None,  # e.g., {"TRACE":0.1}
        dedup_window_sec: float = 1.0,           # collapse duplicates within window
    ):
        self.name = name
        self.trace_level = trace_level
        self.enabled = enabled
        self.max_entries = max_entries

        # In-memory storage
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=max_entries)
        self._error_entries: Deque[Dict[str, Any]] = deque(maxlen=min(1000, max_entries))
        self._level_counts: Dict[TraceLevel, int] = {level: 0 for level in TraceLevel}
        self._component_counts: Dict[str, int] = {}

        # Output config
        self.console = console
        self._color = sys.stdout.isatty() if color is None else bool(color)
        self._output_path: Optional[Path] = Path(output_file) if output_file else None
        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # Async writer
        self._async = bool(async_write)
        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_stop = threading.Event()

        # Rotation
        self._rotate_size_bytes = None if rotate_size_mb is None else int(rotate_size_mb * 1024 * 1024)
        self._rotate_backups = int(max(0, rotate_backups))

        # Redaction/sampling/dedup
        self._redact_keys = {k.lower() for k in (redact_keys or [])}
        self._sample_by_level = {k.upper(): float(v) for k, v in (sample_by_level or {}).items()}
        self._dedup_window_sec = float(dedup_window_sec)
        self._recent_signatures: Dict[str, Dict[str, Any]] = {}  # sig -> {'time': float, 'count': int}

        # Locks & counters
        self._lock = threading.RLock()
        self._counter = itertools.count(1)

        if self._async and self._output_path:
            self._writer_thread = threading.Thread(
                target=self._writer_loop, name=f"TraceWriter-{self.name}", daemon=True
            )
            self._writer_thread.start()

        # Ensure clean shutdown
        atexit.register(self._shutdown)

    # ------------------------- Public API -------------------------

    def trace(
        self,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        component: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """
        Log a trace message with detailed context.
        """
        if not self.enabled:
            return

        # Allow string level at runtime; keep type-safe call sites in this file.
        if isinstance(level, str):  # runtime leniency
            try:
                level = TraceLevel[level.upper()]
            except Exception:
                level = TraceLevel.INFO

        # Fast threshold check
        if level.value < self.trace_level.value:
            return

        # Sampling (probabilistic) for verbose levels if configured
        if self._should_drop_by_sampling(level):
            return

        # Build entry with minimal overhead
        ts = _utc_iso()

        # Caller info (guarded to avoid heavy overhead)
        caller_info: Optional[Dict[str, Any]] = None
        local_vars: Optional[Dict[str, Any]] = None
        try:
            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame else None
            if caller_frame:
                caller_info = {
                    "file": os.path.basename(caller_frame.f_code.co_filename),
                    "function": caller_frame.f_code.co_name,
                    "line": caller_frame.f_lineno,
                }

                # Only capture locals at TRACE when logger level is TRACE
                if level == TraceLevel.TRACE and self.trace_level == TraceLevel.TRACE:
                    local_vars = {
                        k: self._serialize_value(v)
                        for k, v in caller_frame.f_locals.items()
                        if not k.startswith("_") and k != "self"
                    }
        except Exception:
            caller_info = None
            local_vars = None

        # Correlation context
        request_id = _request_id_var.get()
        trace_id = _trace_id_var.get()
        span_id = _span_id_var.get()

        # Prepare entry
        comp = component or self.name
        payload = copy.deepcopy(data) if isinstance(data, dict) else (data if data is None else {"value": str(data)})
        payload = self._redact(payload)

        entry: Dict[str, Any] = {
            "timestamp": ts,
            "level": level.name,
            "message": message,
            "component": comp,
            "caller": caller_info,
            "data": payload,
            "local_vars": self._redact(local_vars) if local_vars is not None else None,
            "request_id": request_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "seq": next(self._counter),
        }

        if exc_info:
            entry["exception"] = traceback.format_exc()

        # De-duplication (collapse bursts of identical messages)
        if self._should_collapse(entry):
            return

        # Update in-memory & stats
        with self._lock:
            self._entries.append(entry)
            if level == TraceLevel.ERROR:
                self._error_entries.append(entry)
            self._level_counts[level] += 1
            self._component_counts[comp] = self._component_counts.get(comp, 0) + 1

        # Console
        if self.console:
            self._print_console(entry)

        # Output file (JSONL)
        if self._output_path:
            if self._async:
                try:
                    self._q.put_nowait(entry)
                except queue.Full:
                    # Drop on overflow to protect caller
                    pass
            else:
                self._write_jsonl(entry)

    # -------------- Compatibility helpers (logging-like API) --------------

    def log(
        self,
        level: "TraceLevel | str | int",
        message: str,
        *,
        component: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ):
        lvl: TraceLevel
        if isinstance(level, TraceLevel):
            lvl = level
        elif isinstance(level, str):
            try:
                lvl = TraceLevel[level.upper()]
            except Exception:
                lvl = TraceLevel.INFO
        elif isinstance(level, int):
            # Map roughly: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR
            if level <= 10:
                lvl = TraceLevel.DEBUG
            elif level <= 20:
                lvl = TraceLevel.INFO
            elif level <= 30:
                lvl = TraceLevel.WARNING
            else:
                lvl = TraceLevel.ERROR
        else:
            lvl = TraceLevel.INFO
        self.trace(message, level=lvl, component=component, data=data, exc_info=exc_info)

    def debug(self, message: str, *, component: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        self.trace(message, level=TraceLevel.DEBUG, component=component, data=data)

    def info(self, message: str, *, component: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        self.trace(message, level=TraceLevel.INFO, component=component, data=data)

    def warning(self, message: str, *, component: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        self.trace(message, level=TraceLevel.WARNING, component=component, data=data)

    # Deprecated alias used by some callers
    warn = warning

    def error(
        self,
        message: str,
        *,
        component: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ):
        self.trace(message, level=TraceLevel.ERROR, component=component, data=data, exc_info=exc_info)

    def exception(self, message: str, *, component: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        # Match logging.exception() behavior: logs ERROR with exc_info=True
        self.trace(message, level=TraceLevel.ERROR, component=component, data=data, exc_info=True)

    # ------------------------- Context helpers -------------------------

    def set_context(
        self,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None
    ):
        if request_id is not None:
            _request_id_var.set(request_id)
        if trace_id is not None:
            _trace_id_var.set(trace_id)
        if span_id is not None:
            _span_id_var.set(span_id)

    def clear_context(self):
        _request_id_var.set(None)
        _trace_id_var.set(None)
        _span_id_var.set(None)

    class span:
        """Lightweight span context manager: with logger.span('load_data'): ..."""

        def __init__(self, logger: "TraceLogger", name: str, data: Optional[Dict[str, Any]] = None):
            self.logger = logger
            self.name = name
            self.data = data or {}
            self._start: Optional[float] = None

        def __enter__(self):
            self._start = time.time()
            self.logger.trace(f"Span start: {self.name}", level=TraceLevel.DEBUG, data=self.data)
            return self

        def __exit__(self, exc_type, exc, tb):
            dur_ms = (time.time() - (self._start or time.time())) * 1000.0
            data = dict(self.data, duration_ms=dur_ms)
            if exc_type:
                self.logger.trace(
                    f"Span error: {self.name}",
                    level=TraceLevel.ERROR,
                    data=data,
                    exc_info=True
                )
            else:
                self.logger.trace(f"Span end: {self.name}", level=TraceLevel.DEBUG, data=data)

    # ------------------------- Admin / queries -------------------------

    def set_level(self, level: TraceLevel):
        self.trace_level = level
        self.trace(f"Trace level changed to {level.name}", TraceLevel.INFO)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def get_entries(
        self,
        level: Optional[TraceLevel] = None,
        component: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get filtered log entries"""
        entries = list(self._entries)
        if level:
            entries = [e for e in entries if e["level"] == level.name]
        if component:
            entries = [e for e in entries if e["component"] == component]
        return entries[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "total_entries": len(self._entries),
            "error_entries": len(self._error_entries),
            "level_counts": {level.name: count for level, count in self._level_counts.items()},
            "component_counts": dict(self._component_counts),
            "current_level": self.trace_level.name,
            "enabled": self.enabled,
        }

    def clear(self):
        """Clear all log entries and counters"""
        with self._lock:
            self._entries.clear()
            self._error_entries.clear()
            self._level_counts = {level: 0 for level in TraceLevel}
            self._component_counts.clear()
            self._recent_signatures.clear()

    # ------------------------- Internals -------------------------

    def _should_drop_by_sampling(self, level: TraceLevel) -> bool:
        p = self._sample_by_level.get(level.name)
        if p is None or p >= 1.0:
            return False
        # inexpensive PRNG using time
        return (hash((time.time_ns(), id(self))) % 1_000_000) / 1_000_000.0 > p

    def _signature(self, entry: Dict[str, Any]) -> str:
        # signature for de-dup: level|component|message|caller-file|caller-line
        caller = entry.get("caller") or {}
        return f"{entry['level']}|{entry['component']}|{entry['message']}|{caller.get('file')}|{caller.get('line')}"

    def _should_collapse(self, entry: Dict[str, Any]) -> bool:
        if self._dedup_window_sec <= 0:
            return False
        sig = self._signature(entry)
        now = time.time()
        meta = self._recent_signatures.get(sig)
        if meta and (now - meta["time"]) <= self._dedup_window_sec:
            meta["count"] += 1
            meta["time"] = now
            # We keep a compact dedup record in memory but don't re-print to console/file
            return True
        self._recent_signatures[sig] = {"time": now, "count": 1}
        return False

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for logging (bounded & safe)."""
        try:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [self._serialize_value(v) for v in list(value)[:10]]  # Limit to 10 items
            if isinstance(value, dict):
                return {str(k): self._serialize_value(v) for k, v in list(value.items())[:10]}
            if hasattr(value, "__dict__"):
                return f"<{value.__class__.__name__}>"
            return str(value)
        except Exception:
            return f"<unserializable:{type(value).__name__}>"

    def _redact(self, obj: Any) -> Any:
        """Redact configured keys recursively."""
        if not self._redact_keys or obj is None:
            return obj
        try:
            if isinstance(obj, dict):
                out: Dict[str, Any] = {}
                for k, v in obj.items():
                    if str(k).lower() in self._redact_keys:
                        out[k] = "***"
                    else:
                        out[k] = self._redact(v)
                return out
            if isinstance(obj, list):
                return [self._redact(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(self._redact(v) for v in obj)
            return obj
        except Exception:
            return obj

    # ------------------------- Console / File Output -------------------------

    def _print_console(self, entry: Dict[str, Any]):
        formatted = self._format_entry_console(entry)
        lvl = entry["level"]
        if self._color:
            if lvl == "ERROR":
                formatted = f"\033[91m{formatted}\033[0m"  # Red
            elif lvl == "WARNING":
                formatted = f"\033[93m{formatted}\033[0m"  # Yellow
            elif lvl == "DEBUG":
                formatted = f"\033[94m{formatted}\033[0m"  # Blue
            elif lvl == "TRACE":
                formatted = f"\033[90m{formatted}\033[0m"  # Gray
        print(formatted)

    def _format_entry_console(self, entry: Dict[str, Any]) -> str:
        ts = entry["timestamp"].split("T")[-1]
        level = f"[{entry['level']:<7}]"
        comp = f"[{entry['component']:<20}]"
        parts = [ts, level, comp]
        caller = entry.get("caller")
        if caller:
            parts.append(f"{caller.get('function')}:{caller.get('line')}")
        parts.append(entry["message"])
        if entry.get("data") is not None:
            try:
                data_snip = json.dumps(entry["data"], default=str)
                if len(data_snip) > 120:
                    data_snip = data_snip[:120] + "..."
                parts.append(f"| data={data_snip}")
            except Exception:
                pass
        if entry.get("local_vars") is not None and self.trace_level == TraceLevel.TRACE:
            try:
                vars_str = json.dumps(entry["local_vars"], default=str)
                if len(vars_str) > 200:
                    vars_str = vars_str[:200] + "..."
                parts.append(f"| vars={vars_str}")
            except Exception:
                pass
        return " ".join(parts)

    def _writer_loop(self):
        last_flush = time.time()
        fp = None
        try:
            if self._output_path:
                fp = open(self._output_path, "a", encoding="utf-8")
            while not self._writer_stop.is_set():
                try:
                    item = self._q.get(timeout=0.25)
                except queue.Empty:
                    item = None

                if item:
                    self._write_jsonl(item, fp)
                # periodic flush
                if fp and (time.time() - last_flush) > 0.5:
                    fp.flush()
                    last_flush = time.time()
        except Exception as e:
            # Fallback to synchronous prints if writer dies
            print(f"[TraceLogger] Writer thread error: {e}", file=sys.stderr)
        finally:
            try:
                if fp:
                    fp.flush()
                    fp.close()
            except Exception:
                pass

    def _write_jsonl(self, entry: Dict[str, Any], fp=None):
        if not self._output_path:
            return
        # Rotate if needed
        try:
            if self._rotate_size_bytes and self._output_path.exists():
                if self._output_path.stat().st_size >= self._rotate_size_bytes:
                    self._rotate_files()
            line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
            if fp:
                fp.write(line)
            else:
                with open(self._output_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            # Don't crash callers for I/O issues
            print(f"[TraceLogger] Failed to write log: {e}", file=sys.stderr)

    def _rotate_files(self):
        # Guard: nothing to rotate without an output file
        if not self._output_path:
            return
        try:
            # Close async file handle by signaling thread to reopen on next loop
            if self._writer_thread and self._writer_thread.is_alive():
                self._writer_stop.set()
                self._writer_thread.join(timeout=1.0)
                self._writer_stop.clear()
                self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
                self._writer_thread.start()

            # Rotate: file -> .1, .1 -> .2, ...
            for i in range(self._rotate_backups, 0, -1):
                src = self._output_path.with_suffix(self._output_path.suffix + f".{i}")
                dst = self._output_path.with_suffix(self._output_path.suffix + f".{i+1}")
                if src.exists():
                    if i == self._rotate_backups:
                        try:
                            src.unlink(missing_ok=True)  # Python 3.8+: missing_ok kw may not exist; fallback next
                        except TypeError:
                            if src.exists():
                                src.unlink()
                    else:
                        src.rename(dst)

            # Move current to .1
            if self._output_path.exists():
                self._output_path.rename(self._output_path.with_suffix(self._output_path.suffix + ".1"))
        except Exception as e:
            print(f"[TraceLogger] Rotation failed: {e}", file=sys.stderr)

    # ------------------------- Shutdown -------------------------

    def _shutdown(self):
        # drain queue and stop thread
        try:
            if self._writer_thread and self._writer_thread.is_alive():
                self._writer_stop.set()
                self._writer_thread.join(timeout=2.0)
        except Exception:
            pass
