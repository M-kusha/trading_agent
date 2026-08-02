# modules/memory/debug/memory_logger.py
"""
Unified Debug Logger for Memory System
- Master enable via `enabled`
- Minimum verbosity via `level` ("TRACE"|"DEBUG"|"INFO"|"WARNING"|"ERROR")
- Per-level file toggles + per-level file paths
- Optional combined file
- Rotation per file
- Thread safe
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np  # used for simple stats in profiler and logger

LevelName = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]


class MemoryDebugLogger:
    """
    Comprehensive debug logging for unified memory system

    Master switches:
      - enabled: turn all logging on/off
      - level: minimum verbosity ("TRACE" < "DEBUG" < "INFO" < "WARNING" < "ERROR")

    Outputs:
      - Combined file (optional)
      - Per-level files (optional, each with its own toggle & path)

    Notes:
      - Console echo happens only when level <= DEBUG (i.e., DEBUG/TRACE modes)
      - File rotation is per-file
    """

    def __init__(
        self,
        *,
        # Master controls
        enabled: bool = False,
        level: str = "INFO",

        # Combined file
        log_path: str = "logs/memory/unified_debug.log",
        enable_combined_file: bool = True,

        # Per-level toggles + paths (if path is None, a sane default is derived from log_path)
        enable_trace_file: bool = False,
        trace_file: Optional[str] = None,

        enable_debug_file: bool = False,
        debug_file: Optional[str] = None,

        enable_info_file: bool = False,
        info_file: Optional[str] = None,

        enable_warning_file: bool = False,
        warning_file: Optional[str] = None,

        enable_error_file: bool = False,
        error_file: Optional[str] = None,

        # Rotation
        max_size_mb: int = 100,
        rotation_count: int = 5,
    ):
        self.enabled = enabled
        self.level = level.upper()

        # Severity thresholds
        self.levels: Dict[str, int] = {
            "TRACE": 0,
            "DEBUG": 1,
            "INFO": 2,
            "WARNING": 3,
            "ERROR": 4,
        }
        self.current_level = self.levels.get(self.level, 2)

        # Rotation config
        self.max_size_mb = max(1, int(max_size_mb))
        self.rotation_count = max(1, int(rotation_count))

        # Console colors (best effort)
        self.colors: Dict[str, str] = {
            "replay": "\033[94m",
            "compression": "\033[92m",
            "mistakes": "\033[91m",
            "neural": "\033[95m",
            "playbook": "\033[93m",
            "budget": "\033[96m",
            "unified": "\033[97m",
            "store": "\033[90m",
            "bus": "\033[36m",
            "performance": "\033[32m",
            "error": "\033[31m",
            "profiler": "\033[35m",
            "reset": "\033[0m",
            "bold": "\033[1m",
            "underline": "\033[4m",
        }

        # Paths + file handles
        self.enable_combined_file = bool(enable_combined_file)
        self.combined_path = str(log_path)

        # Derive per-level defaults from combined path when missing
        base_dir = Path(self.combined_path).parent
        base_stem = Path(self.combined_path).stem
        # strip common suffixes so unified_debug -> unified
        for suf in ("_debug", "_trace", "_info", "_warning", "_error", "_combined"):
            if base_stem.endswith(suf):
                base_stem = base_stem[: -len(suf)]

        def default_path(suffix: str) -> str:
            return str(base_dir / f"{base_stem}_{suffix}.log")

        self.enable_trace_file = bool(enable_trace_file)
        self.trace_path = trace_file or default_path("trace")

        self.enable_debug_file = bool(enable_debug_file)
        self.debug_path = debug_file or default_path("debug")

        self.enable_info_file = bool(enable_info_file)
        self.info_path = info_file or default_path("info")

        self.enable_warning_file = bool(enable_warning_file)
        self.warning_path = warning_file or default_path("warning")

        self.enable_error_file = bool(enable_error_file)
        self.error_path = error_file or default_path("error")

        # Ensure directories exist for every potential path
        for p in [self.combined_path, self.trace_path, self.debug_path, self.info_path, self.warning_path, self.error_path]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

        # Open handles lazily on first write
        self._file_handles: Dict[str, Any] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Performance tracking
        self.operation_times: Deque[float] = deque(maxlen=1000)
        self.component_times: Dict[str, Deque[float]] = {
            "replay": deque(maxlen=100),
            "compression": deque(maxlen=100),
            "mistakes": deque(maxlen=100),
            "neural": deque(maxlen=100),
            "playbook": deque(maxlen=100),
            "budget": deque(maxlen=100),
        }

        # Memory snapshots
        self.memory_snapshots: Deque[Dict[str, Any]] = deque(maxlen=100)

        # Stats
        self.log_counts: Dict[str, int] = {
            "TRACE": 0,
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
        }

    # ---------- public controls ----------

    def set_level(self, level: str) -> None:
        with self.lock:
            self.level = level.upper()
            self.current_level = self.levels.get(self.level, self.current_level)

    def enable(self) -> None:
        with self.lock:
            self.enabled = True

    def disable(self) -> None:
        with self.lock:
            self.enabled = False

    # ---------- core logging ----------

    def log(
        self,
        level: str,
        message: str,
        component: Optional[str] = None,
        data: Optional[Any] = None,
    ) -> None:
        if not self.enabled:
            return

        lvl = self.levels.get(level.upper(), 2)
        if lvl < self.current_level:
            return

        timestamp = datetime.now()
        entry = self._format_entry(timestamp, level, message, component, data)

        with self.lock:
            # Combined file
            if self.enable_combined_file:
                self._write_to_file(self.combined_path, entry)

            # Level-specific file
            level = level.upper()
            if level == "TRACE" and self.enable_trace_file:
                self._write_to_file(self.trace_path, entry)
            elif level == "DEBUG" and self.enable_debug_file:
                self._write_to_file(self.debug_path, entry)
            elif level == "INFO" and self.enable_info_file:
                self._write_to_file(self.info_path, entry)
            elif level == "WARNING" and self.enable_warning_file:
                self._write_to_file(self.warning_path, entry)
            elif level == "ERROR" and self.enable_error_file:
                self._write_to_file(self.error_path, entry)

            # Console echo disabled - use beautiful visualizer instead
            # if self.current_level <= self.levels["DEBUG"]:
            #     self._print_colored(timestamp, level, message, component, data)

            self.log_counts[level] = self.log_counts.get(level, 0) + 1

    # Convenience wrappers
    def trace(self, message: str, **kwargs: Any) -> None:
        self.log("TRACE", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self.log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.log("WARNING", message, **kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        if exception is not None:
            tb = traceback.format_exc()
            extra = kwargs.get("data") or {}
            if isinstance(extra, dict):
                extra = {**extra, "traceback": tb}
            else:
                extra = {"traceback": tb, "data": str(extra)}
            kwargs["data"] = extra
        self.log("ERROR", message, **kwargs)

    # ---------- helpers used by UnifiedMemory ----------

    def log_input(self, operation: str, data: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "operation": operation,
            "input_keys": list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            "timestamp": time.time(),
        }
        self.debug(f"Operation Input: {operation}", component="unified", data=payload)

    def log_output(self, operation: str, data: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "operation": operation,
            "output_keys": list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            "timestamp": time.time(),
        }
        self.debug(f"Operation Output: {operation}", component="unified", data=payload)

    def log_component_operation(self, component: str, operation: str, data: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if "time_ms" in data:
            try:
                self.component_times.setdefault(component, deque(maxlen=100)).append(float(data["time_ms"]))
            except Exception:
                pass
        self.debug(f"[{component.upper()}] {operation}", component=component, data=data)

    def log_memory_operation(self, operation: str, entry: Dict[str, Any]) -> None:
        if not self.enabled or self.current_level > self.levels["DEBUG"]:
            return
        payload = {
            "operation": operation,
            "timestamp": entry.get("timestamp"),
            "pnl": entry.get("pnl"),
            "importance": entry.get("importance"),
            "metadata": entry.get("metadata"),
        }
        self.debug(f"Memory Operation: {operation}", component="store", data=payload)

    def log_bus_update(self, key: str, value: Any) -> None:
        if not self.enabled or self.current_level > self.levels["TRACE"]:
            return
        payload = {
            "key": key,
            "value_type": type(value).__name__,
            "value_sample": self._truncate(value, 100),
        }
        self.trace(f"Bus Update: {key}", component="bus", data=payload)

    def log_bus_updates(self, updates: Sequence[Tuple[str, Any]]) -> None:
        if not self.enabled or self.current_level > self.levels["TRACE"]:
            return
        updates_list = list(updates)
        update_summary = {key: type(value).__name__ for key, value in updates_list[:10]}
        self.trace(
            f"Batch Bus Update: {len(updates_list)} keys",
            component="bus",
            data={"updates": update_summary},
        )

    def log_performance(self, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        pt = float(metrics.get("processing_time_ms", 0.0))
        self.operation_times.append(pt)
        if "total_memories" in metrics:
            self.memory_snapshots.append(
                {
                    "timestamp": time.time(),
                    "total_memories": metrics["total_memories"],
                    "utilization": metrics.get("utilization", 0.0),
                }
            )
        self.info("Performance Metrics", component="performance", data=metrics)

    def log_error(self, context: str, error: Exception) -> None:
        payload = {
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        self.error(f"Error in {context}", exception=error, component="error", data=payload)

    # ---------- stats & lifecycle ----------

    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            avg_times: Dict[str, Dict[str, float]] = {}
            for component, times in self.component_times.items():
                if times:
                    arr = np.asarray(times, dtype=float)
                    avg_times[component] = {
                        "avg_ms": float(np.mean(arr)),
                        "max_ms": float(np.max(arr)),
                        "min_ms": float(np.min(arr)),
                        "p95_ms": float(np.percentile(arr, 95)),
                    }

            file_sizes = {}
            for p in self._all_active_paths():
                try:
                    file_sizes[p] = os.path.getsize(p) / (1024 * 1024)
                except Exception:
                    file_sizes[p] = 0.0

            return {
                "enabled": self.enabled,
                "level": self.level,
                "log_counts": dict(self.log_counts),
                "total_logs": int(sum(self.log_counts.values())),
                "component_performance": avg_times,
                "memory_snapshots": len(self.memory_snapshots),
                "files": file_sizes,
            }

    def flush(self) -> None:
        with self.lock:
            for fh in list(self._file_handles.values()):
                try:
                    fh.flush()
                except Exception:
                    pass

    def close(self) -> None:
        with self.lock:
            for path, fh in list(self._file_handles.items()):
                try:
                    fh.close()
                except Exception:
                    pass
                finally:
                    self._file_handles.pop(path, None)

    def __enter__(self) -> "MemoryDebugLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ---------- private helpers ----------

    def _all_active_paths(self) -> List[str]:
        paths: List[str] = []
        if self.enable_combined_file:
            paths.append(self.combined_path)
        if self.enable_trace_file:
            paths.append(self.trace_path)
        if self.enable_debug_file:
            paths.append(self.debug_path)
        if self.enable_info_file:
            paths.append(self.info_path)
        if self.enable_warning_file:
            paths.append(self.warning_path)
        if self.enable_error_file:
            paths.append(self.error_path)
        return paths

    def _ensure_open(self, path: str) -> None:
        if path not in self._file_handles:
            try:
                self._file_handles[path] = open(path, "a", encoding="utf-8")
            except Exception as e:
                print(f"[MemoryDebugLogger] Failed to open log file '{path}': {e}")

    def _write_to_file(self, path: str, entry: str) -> None:
        self._ensure_open(path)
        fh = self._file_handles.get(path)
        if not fh:
            return
        try:
            fh.write(entry + "\n")
            fh.flush()
        except Exception:
            # If write fails, try to reopen next time; still echo on console in DEBUG/TRACE via caller
            try:
                fh.close()
            except Exception:
                pass
            self._file_handles.pop(path, None)
            return

        # Rotation
        self._check_rotation_and_rotate(path)

    def _check_rotation_and_rotate(self, path: str) -> None:
        try:
            if not os.path.exists(path):
                return
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb <= self.max_size_mb:
                return
        except Exception:
            return

        # rotate: .N -> .N+1, ..., .1 -> .2, current -> .1
        try:
            # Close current
            fh = self._file_handles.get(path)
            if fh:
                try:
                    fh.flush()
                except Exception:
                    pass
                fh.close()
        except Exception:
            pass
        finally:
            self._file_handles.pop(path, None)

        try:
            # Shift older files
            for i in range(self.rotation_count - 1, 0, -1):
                src = f"{path}.{i}"
                dst = f"{path}.{i + 1}"
                if os.path.exists(src):
                    # ensure last slot is free
                    try:
                        if os.path.exists(dst):
                            os.remove(dst)
                    except Exception:
                        pass
                    try:
                        os.replace(src, dst)
                    except Exception:
                        pass

            # Move current -> .1
            if os.path.exists(path):
                try:
                    os.replace(path, f"{path}.1")
                except Exception:
                    pass
        finally:
            # Reopen fresh
            self._ensure_open(path)

    def _format_entry(
        self,
        timestamp: datetime,
        level: str,
        message: str,
        component: Optional[str],
        data: Optional[Any],
    ) -> str:
        entry_parts: List[str] = [timestamp.isoformat(), f"[{level.upper():8}]"]
        if component:
            entry_parts.append(f"[{component:12}]")
        entry_parts.append(message)

        if data is not None:
            if isinstance(data, dict):
                try:
                    data_str = json.dumps(data, indent=2, default=str)
                except Exception:
                    data_str = self._truncate(data, 1000)
            else:
                data_str = self._truncate(data, 1000)
            entry_parts.append(f"\n  DATA: {data_str}")

        return " ".join(entry_parts)

    def _print_colored(
        self,
        timestamp: datetime,
        level: str,
        message: str,
        component: Optional[str],
        data: Optional[Any],
    ) -> None:
        color = self.colors.get(component or "", "")
        reset = self.colors["reset"]
        level_colors = {
            "TRACE": "\033[90m",
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
        }
        level_color = level_colors.get(level.upper(), "")
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
        if component:
            print(f"{level_color}[{level:5}] {time_str} {color}[{component}] {message}{reset}")
        else:
            print(f"{level_color}[{level:5}] {time_str} {message}{reset}")

        # Only echo data in TRACE to avoid spam
        if data is not None and self.current_level == self.levels["TRACE"]:
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"  {k}: {self._truncate(v, 200)}")
            else:
                print(f"  {self._truncate(data, 200)}")

    def _truncate(self, data: Any, max_length: int = 200) -> str:
        try:
            if isinstance(data, (dict, list, tuple)):
                s = json.dumps(data, default=str)
            else:
                s = str(data)
        except Exception:
            s = repr(data)
        if len(s) > max_length:
            return s[:max_length] + "..."
        return s


class PerformanceProfiler:
    """Lightweight timing profiler (optionally emits TRACE logs)."""

    def __init__(self, logger: Optional[MemoryDebugLogger] = None):
        self.logger = logger
        self.timers: Dict[str, float] = {}
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()

    def start(self, name: str) -> None:
        with self._lock:
            self.timers[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        with self._lock:
            start = self.timers.pop(name, None)
            if start is None:
                return 0.0
            duration = (time.perf_counter() - start) * 1000.0
            self.profiles[name].append(duration)
        if self.logger:
            self.logger.trace(f"Operation '{name}' took {duration:.2f}ms", component="profiler")
        return duration

    def get_profile(self, name: str) -> Dict[str, float]:
        with self._lock:
            times = self.profiles.get(name, [])
            if not times:
                return {"count": 0}
            arr = np.asarray(times, dtype=float)
            return {
                "count": float(len(arr)),
                "total_ms": float(np.sum(arr)),
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
            }

    def get_all_profiles(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {name: self.get_profile(name) for name in self.profiles.keys()}

    def reset(self) -> None:
        with self._lock:
            self.timers.clear()
            self.profiles.clear()
