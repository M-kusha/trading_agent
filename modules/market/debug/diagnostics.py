# ─────────────────────────────────────────────────────────────
# File: modules/market/debug/diagnostics.py
# Real-time diagnostics and performance profiling — Production Upgrade
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

# Optional deps
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # graceful degrade

try:
    import resource  # Unix-only
except Exception:  # pragma: no cover
    resource = None  # graceful degrade


class DiagnosticsEngine:
    """
    Real-time diagnostics and performance profiling for the market module.

    Backward-compatible API:
      - start_profiling(operation, metadata=None)
      - stop_profiling(operation_or_id) -> Optional[Dict[str, Any]]
      - get_current_diagnostics() -> Dict[str, Any]
      - get_memory_profile() -> Dict[str, Any]
      - stop_monitoring()

    New capabilities:
      • Thread-safe (internal RLock)
      • Non-blocking CPU sampling + EMA smoothing
      • Concurrent profiling of the same operation (unique IDs)
      • Context manager & decorator for easy instrumentation
      • JSONL export of completed profiles
      • Memory leak & slow-exec detection with debounced alerts
      • Tunable sampling interval and retention sizes
      • Graceful degrade if psutil/resource are unavailable
    """

    def __init__(
        self,
        profiling: bool = False,
        memory_tracking: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None,
        sample_interval_sec: float = 1.0,
        retention_seconds: int = 300,
        completed_profiles_max: int = 2000,
        alerts_max: int = 200,
        ema_alpha: float = 0.2,  # smoothing for CPU/memory
        logger: Optional[Any] = None,
    ):
        # Config
        self.profiling = bool(profiling)
        self.memory_tracking = bool(memory_tracking)
        self.logger = logger

        self.alert_thresholds = {
            'cpu_percent': 80.0,          # system CPU percent
            'proc_cpu_percent': 200.0,    # per-process can exceed 100 on multi-core (N cores * 100)
            'memory_percent': 80.0,       # system memory usage
            'execution_time_ms': 1000.0,  # single operation
            'mem_leak_mb_per_min': 50.0,  # observed slope over window
            **(alert_thresholds or {})
        }

        # Sampling
        self.sample_interval_sec = max(0.2, float(sample_interval_sec))
        self._ema_alpha = float(ema_alpha)

        # Retention
        self._resource_snapshots: Deque[Dict[str, Any]] = deque(maxlen=max(10, int(retention_seconds / self.sample_interval_sec)))
        self._memory_snapshots: Deque[Tuple[float, float]] = deque(maxlen=max(10, int(retention_seconds / self.sample_interval_sec)))
        self._completed_profiles: Deque[Dict[str, Any]] = deque(maxlen=int(completed_profiles_max))
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=int(alerts_max))

        # Profiling state
        self._lock = threading.RLock()
        self._id_seq = 0
        self._profile_stack: List[Dict[str, Any]] = []  # nested stack (for context manager depth)
        self._active_by_id: Dict[str, Dict[str, Any]] = {}
        self._active_by_name: Dict[str, List[str]] = defaultdict(list)  # op -> [ids]

        # Debounce for alerts
        self._last_alert_time: Dict[str, float] = {}

        # Process handles
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self._last_proc_cpu_percent_initialized = False
        if self._proc and hasattr(self._proc, "cpu_percent"):
            # Initialize to create baseline; non-blocking if interval=None
            try:
                self._proc.cpu_percent(interval=None)
                self._last_proc_cpu_percent_initialized = True
            except Exception:
                pass

        # EMA state
        self._ema_cpu: Optional[float] = None
        self._ema_proc_cpu: Optional[float] = None
        self._ema_mem_mb: Optional[float] = None

        # Background monitor
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, name="DiagnosticsMonitor", daemon=True)
        self._monitor_thread.start()

        self._trace("DiagnosticsEngine initialized", "DEBUG")

    # ------------------------- Public API -------------------------

    def start_profiling(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start profiling an operation. Returns a unique profile id (optional for callers)."""
        if not self.profiling:
            return None
        now = time.time()
        profile = {
            'id': self._next_id(),
            'operation': operation,
            'start_time': now,
            'metadata': dict(metadata or {}),
            'depth': len(self._profile_stack),
        }
        if self.memory_tracking:
            profile['start_memory_mb'] = self._get_memory_usage_mb()

        with self._lock:
            self._profile_stack.append(profile)
            self._active_by_id[profile['id']] = profile
            self._active_by_name[operation].append(profile['id'])

        return profile['id']

    def stop_profiling(self, operation_or_id: str) -> Optional[Dict[str, Any]]:
        """Stop profiling an operation by name or by returned id."""
        if not self.profiling:
            return None

        with self._lock:
            profile: Optional[Dict[str, Any]] = None

            # Prefer ID match
            if operation_or_id in self._active_by_id:
                profile = self._active_by_id.pop(operation_or_id)
                op = profile['operation']
                if self._active_by_name.get(op):
                    ids = self._active_by_name[op]
                    if operation_or_id in ids:
                        ids.remove(operation_or_id)
                        if not ids:
                            self._active_by_name.pop(op, None)
            else:
                # Pop the most recent profile for this operation name
                op = operation_or_id
                ids = self._active_by_name.get(op) or []
                if ids:
                    pid = ids.pop()  # last started for this name
                    if not ids:
                        self._active_by_name.pop(op, None)
                    profile = self._active_by_id.pop(pid, None)

            if profile is None:
                return None

            # Finalize metrics
            end_time = time.time()
            profile['end_time'] = end_time
            profile['execution_time_ms'] = (end_time - profile['start_time']) * 1000.0
            if self.memory_tracking:
                end_mem = self._get_memory_usage_mb()
                profile['end_memory_mb'] = end_mem
                start_mem = profile.get('start_memory_mb', end_mem)
                profile['memory_delta_mb'] = (end_mem - start_mem)

            # Remove from stack (if present)
            if self._profile_stack and self._profile_stack[-1]['id'] == profile['id']:
                self._profile_stack.pop()
            else:
                # Fallback: remove by search
                try:
                    self._profile_stack.remove(profile)
                except ValueError:
                    pass

            self._completed_profiles.append(profile)

        # Alerts outside lock
        self._check_performance_alerts(profile)
        return profile

    def get_current_diagnostics(self) -> Dict[str, Any]:
        """Current snapshot: smoothed CPU, memory, active profiles, alerts, perf summary."""
        with self._lock:
            latest = self._resource_snapshots[-1] if self._resource_snapshots else {}
            active_ops = [self._active_by_id[pid]['operation'] for pid in self._active_by_id]
            profile_stack_depth = len(self._profile_stack)
            recent_alerts = list(self._alerts)[-5:]

        return {
            'resource_usage': latest,
            'active_profiles': active_ops,
            'profile_stack_depth': profile_stack_depth,
            'recent_alerts': recent_alerts,
            'performance_summary': self._get_performance_summary(),
        }

    def get_memory_profile(self) -> Dict[str, Any]:
        """Detailed memory view (graceful if psutil/resource not available)."""
        if not self.memory_tracking:
            return {'enabled': False}

        current_mb = self._get_memory_usage_mb()
        available_mb: Optional[float] = None
        percent_used: Optional[float] = None
        peak_mb: Optional[float] = None

        if psutil:
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 * 1024)
            percent_used = vm.percent

        # Track observed peak from memory snapshots as cross-platform fallback
        with self._lock:
            if self._memory_snapshots:
                peak_mb = max(mb for _, mb in self._memory_snapshots)
            else:
                peak_mb = current_mb

        # Best effort OS peak via resource (if available)
        if (resource is not None) and hasattr(resource, "getrusage") and hasattr(resource, "RUSAGE_SELF"):
            try:
                ru = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
                # ru_maxrss: Linux: KB, macOS: bytes
                ru_val = float(getattr(ru, "ru_maxrss", 0.0))
                # Heuristic: consider values > 1e8 as bytes (macOS)
                peak_os_mb = (ru_val / (1024 * 1024)) if ru_val > 1e8 else (ru_val / 1024.0)
                peak_mb = max(peak_mb or 0.0, peak_os_mb)
            except Exception:
                pass

        return {
            'enabled': True,
            'current_mb': current_mb,
            'peak_mb': peak_mb,
            'available_mb': available_mb,
            'percent_used': percent_used,
            'gc_stats': self._get_gc_stats(),
        }

    def stop_monitoring(self, join_timeout: float = 5.0):
        """Stop resource monitoring and join thread."""
        self._monitoring = False
        try:
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=join_timeout)
        except Exception:
            pass

    # ---------------------- Convenience APIs ----------------------

    def profile(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager: with diag.profile('op'): ..."""
        class _Ctx:
            def __init__(self, outer: DiagnosticsEngine, op: str, md: Optional[Dict[str, Any]]):
                self.outer = outer
                self.op = op
                self.md = md
                self.pid: Optional[str] = None

            def __enter__(self):
                self.pid = self.outer.start_profiling(self.op, self.md)
                return self

            def __exit__(self, exc_type, exc, tb):
                self.outer.stop_profiling(self.pid or self.op)
        return _Ctx(self, operation, metadata)

    def profiled(self, operation: Optional[str] = None, metadata_fn: Optional[Callable[..., Dict[str, Any]]] = None):
        """Decorator to auto-profile a function."""
        def decorator(fn: Callable):
            name = operation or f"{fn.__module__}.{fn.__qualname__}"
            def wrapper(*args, **kwargs):
                md = metadata_fn(*args, **kwargs) if metadata_fn else None
                pid = self.start_profiling(name, md)
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.stop_profiling(pid or name)
            return wrapper
        return decorator

    def export_profiles_jsonl(self, filepath: str) -> int:
        """Export completed profiles to a JSONL file. Returns count written."""
        with self._lock:
            items = list(self._completed_profiles)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for p in items:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            return len(items)
        except Exception as e:
            self._trace(f"Failed to export profiles: {e}", "ERROR")
            return 0

    # ------------------------- Internals --------------------------

    def _monitor_resources(self):
        """Background resource sampler (non-blocking, smoothed, debounced alerts)."""
        while self._monitoring:
            t0 = time.time()
            try:
                cpu_pct = self._safe_cpu_percent_system()
                proc_cpu_pct = self._safe_cpu_percent_process()
                mem_mb = self._get_memory_usage_mb()
                mem_pct = self._safe_memory_percent()

                # EMA smoothing
                self._ema_cpu = self._ema(cpu_pct, self._ema_cpu)
                self._ema_proc_cpu = self._ema(proc_cpu_pct, self._ema_proc_cpu)
                self._ema_mem_mb = self._ema(mem_mb, self._ema_mem_mb)

                snapshot = {
                    'timestamp': t0,
                    'cpu_percent': cpu_pct,
                    'cpu_percent_ema': self._ema_cpu,
                    'proc_cpu_percent': proc_cpu_pct,
                    'proc_cpu_percent_ema': self._ema_proc_cpu,
                    'memory_mb': mem_mb,
                    'memory_mb_ema': self._ema_mem_mb,
                    'memory_percent': mem_pct,
                    'active_profiles': self._active_profiles_count(),
                    'thread_count': threading.active_count(),
                }

                with self._lock:
                    self._resource_snapshots.append(snapshot)
                    self._memory_snapshots.append((t0, mem_mb))

                # Alerts: CPU + mem usage
                if cpu_pct is not None and cpu_pct > self.alert_thresholds['cpu_percent']:
                    self._add_alert('HIGH_CPU', f"System CPU: {cpu_pct:.1f}%")
                if proc_cpu_pct is not None and proc_cpu_pct > self.alert_thresholds['proc_cpu_percent']:
                    self._add_alert('HIGH_PROC_CPU', f"Process CPU: {proc_cpu_pct:.1f}%")
                if mem_pct is not None and mem_pct > self.alert_thresholds['memory_percent']:
                    self._add_alert('HIGH_MEMORY', f"System memory: {mem_pct:.1f}%")

                # Memory leak detection via slope on last ~60 samples (1 min @ 1Hz)
                self._maybe_alert_memory_leak()

                # Sleep until next tick (account for sampling work)
                elapsed = time.time() - t0
                delay = max(0.05, self.sample_interval_sec - elapsed)
                time.sleep(delay)

            except Exception as e:
                # Never kill the monitor; back off briefly
                self._trace(f"Resource monitoring error: {e}", "WARNING")
                time.sleep(min(5.0, self.sample_interval_sec * 2))

    # ---- Safe probes ----

    def _safe_cpu_percent_system(self) -> Optional[float]:
        if not psutil:
            return None
        # Non-blocking snapshot; psutil.cpu_percent(interval=None) uses last interval
        try:
            val = psutil.cpu_percent(interval=None, percpu=False)
            # Some stubs suggest list return; guard defensively
            if isinstance(val, (list, tuple)):
                if not val:
                    return 0.0
                return float(sum(float(v) for v in val) / len(val))
            return float(val)
        except Exception:
            return None

    def _safe_cpu_percent_process(self) -> Optional[float]:
        if not self._proc:
            return None
        try:
            # Non-blocking; relies on previous call for baseline
            if not self._last_proc_cpu_percent_initialized:
                self._proc.cpu_percent(interval=None)
                self._last_proc_cpu_percent_initialized = True
                return 0.0
            return float(self._proc.cpu_percent(interval=None))
        except Exception:
            return None

    def _safe_memory_percent(self) -> Optional[float]:
        if not psutil:
            return None
        try:
            return float(psutil.virtual_memory().percent)
        except Exception:
            return None

    def _get_memory_usage_mb(self) -> float:
        if self._proc:
            try:
                return float(self._proc.memory_info().rss) / (1024 * 1024)
            except Exception:
                pass
        # Fallback: approximate with process-wide
        try:
            import tracemalloc  # lightweight fallback if enabled upstream
            if tracemalloc.is_tracing():
                current, _peak = tracemalloc.get_traced_memory()
                return float(current) / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    # ---- Analytics & alerts ----

    def _ema(self, x: Optional[float], prev: Optional[float]) -> Optional[float]:
        if x is None:
            return prev
        if prev is None:
            return x
        a = self._ema_alpha
        return a * x + (1 - a) * prev

    def _maybe_alert_memory_leak(self):
        # Require enough samples
        with self._lock:
            if len(self._memory_snapshots) < 30:
                return
            # Use last ~60 seconds
            samples = list(self._memory_snapshots)[-60:]
        # Linear regression slope (MB / sec) => convert to MB/min
        xs = [s[0] - samples[0][0] for s in samples]
        ys = [s[1] for s in samples]
        n = len(xs)
        if n < 2 or (max(xs) - min(xs)) < 15.0:
            return
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom <= 1e-9:
            return
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        slope_per_min = slope * 60.0
        if slope_per_min > self.alert_thresholds['mem_leak_mb_per_min']:
            self._add_alert('MEMORY_LEAK', f"Observed growth ~{slope_per_min:.1f} MB/min over last minute")

    def _check_performance_alerts(self, profile: Dict[str, Any]):
        t = float(profile.get('execution_time_ms') or 0.0)
        if t > self.alert_thresholds['execution_time_ms']:
            self._add_alert('SLOW_EXECUTION', f"{profile['operation']} took {t:.1f}ms")

        if self.memory_tracking:
            md = float(profile.get('memory_delta_mb') or 0.0)
            if md > 100.0:
                self._add_alert('HIGH_MEMORY_USAGE', f"{profile['operation']} used {md:.1f}MB")

    def _add_alert(self, alert_type: str, message: str, min_interval_sec: float = 10.0):
        now = time.time()
        last = self._last_alert_time.get(alert_type, 0.0)
        if (now - last) < min_interval_sec:
            return  # debounce similar alerts
        self._last_alert_time[alert_type] = now
        alert = {'type': alert_type, 'message': message, 'timestamp': now}
        with self._lock:
            self._alerts.append(alert)

    # ---- Summaries ----

    def _get_performance_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._completed_profiles:
                return {}

            per_op: Dict[str, List[float]] = defaultdict(list)
            for p in self._completed_profiles:
                per_op[p['operation']].append(float(p.get('execution_time_ms') or 0.0))

        def stats(vals: List[float]) -> Dict[str, Any]:
            arr = sorted(vals)
            n = len(arr)
            if n == 0:
                return {
                    'count': 0,
                    'avg_ms': 0.0,
                    'min_ms': 0.0,
                    'max_ms': 0.0,
                    'p50_ms': 0.0,
                    'p95_ms': 0.0,
                    'p99_ms': 0.0
                }

            def pct(p: float) -> float:
                # index based percentile (inclusive, 0..n-1)
                idx = max(0, min(n - 1, int(round(p * (n - 1)))))
                return arr[idx]

            return {
                'count': n,
                'avg_ms': sum(arr) / n,
                'min_ms': arr[0],
                'max_ms': arr[-1],
                'p50_ms': pct(0.50),
                'p95_ms': pct(0.95),
                'p99_ms': pct(0.99),
            }

        return {op: stats(tms) for op, tms in per_op.items()}

    # ---- Helpers ----

    def _active_profiles_count(self) -> int:
        with self._lock:
            return len(self._active_by_id)

    def _get_gc_stats(self) -> Dict[str, Any]:
        try:
            import gc
            return {
                'counts': gc.get_count(),
                'threshold': gc.get_threshold(),
                # Note: avoid gc.collect() in diagnostics; too intrusive
            }
        except Exception:
            return {}

    def _next_id(self) -> str:
        with self._lock:
            self._id_seq += 1
            return f"prof-{self._id_seq}"

    def _trace(self, msg: str, level: str = "INFO"):
        if self.logger:
            try:
                self.logger.trace(f"[Diagnostics] {msg}", level=level)
            except Exception:
                pass
