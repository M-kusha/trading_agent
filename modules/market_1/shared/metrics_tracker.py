# ─────────────────────────────────────────────────────────────
# File: modules/market/shared/metrics_tracker.py
# Performance metrics tracking for market module — Production Upgrade
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Dict, Any, Optional, List, Deque, Tuple, Callable, Awaitable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
import threading
import time
import math
import numpy as np

# Minimal, dependency-free metrics tracker with:
# - Rolling window stats (time-based + sample-based)
# - P50/P95/P99 from rolling reservoir (bounded)
# - Latency histograms (Prometheus-style buckets)
# - SLO checks per metric (success-rate + p95 latency)
# - Thread-safe counters (locks)
# - Context managers & decorators to time code paths (sync & async)
# - Backward-compatible methods retained from your original class


@dataclass
class _RollingStats:
    """Bounded rolling stats for one metric."""
    max_samples: int = 1000
    window_seconds: float = 300.0  # 5-minute recency window
    # Execution times reservoir (ms) with timestamps
    times_ms: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=1000))
    # Success/failure counters (all-time and windowed)
    success: int = 0
    failure: int = 0
    # Recent errors (type/message) with timestamps
    recent_errors: Deque[Tuple[float, str, str]] = field(default_factory=lambda: deque(maxlen=50))
    # Exponential moving average latency (ms)
    ema_latency_ms: float = 0.0
    ema_alpha: float = 0.2
    # Latency histogram buckets (ms)
    buckets: Tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
    bucket_counts: List[int] = field(default_factory=list)
    # Track last aggregation time for pruning
    _last_prune: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.bucket_counts:
            self.bucket_counts = [0 for _ in self.buckets] + [0]  # +Inf

    def _prune_old(self, now: float) -> None:
        """Remove samples outside window_seconds from times_ms."""
        if not self.times_ms:
            return
        if now - self._last_prune < 5.0:  # prune at most every 5s
            return
        cutoff = now - self.window_seconds
        while self.times_ms and self.times_ms[0][0] < cutoff:
            self.times_ms.popleft()
        # we do not roll back histogram; it's intended as all-time since start
        self._last_prune = now

    def add_success(self, duration_ms: float, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.success += 1
        # Rolling reservoir + time window prune
        self.times_ms.append((now, float(duration_ms)))
        self._prune_old(now)
        # EMA latency
        if self.ema_latency_ms == 0.0:
            self.ema_latency_ms = float(duration_ms)
        else:
            self.ema_latency_ms = (
                self.ema_alpha * float(duration_ms) + (1 - self.ema_alpha) * self.ema_latency_ms
            )
        # Histogram
        self._add_to_histogram(float(duration_ms))

    def add_failure(self, err_type: str, err_msg: str, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.failure += 1
        if err_msg is None:
            err_msg = ""
        # Keep last errors short to avoid memory blowup
        msg = err_msg if len(err_msg) <= 240 else (err_msg[:237] + "...")
        self.recent_errors.append((now, err_type, msg))
        self._prune_old(now)

    def _add_to_histogram(self, v: float) -> None:
        # Find first bucket >= v
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.bucket_counts[i] += 1
                return
        self.bucket_counts[-1] += 1  # +Inf

    # Snapshot calculations on current window
    def window_times(self) -> List[float]:
        return [t for _, t in self.times_ms]

    def window_success_rate(self) -> float:
        total_window = len(self.times_ms) + self._window_failures_estimate()
        if total_window <= 0:
            # fall back to all-time if window empty
            total_all = self.success + self.failure
            return (self.success / total_all) if total_all > 0 else 0.0
        # approximate window success rate using successes implied by times_ms
        # (failures lack durations, we approximate by proportional split)
        successes_window = len(self.times_ms)
        return successes_window / total_window

    def _window_failures_estimate(self) -> int:
        """We don't store per-failure timestamps; approximate by ratio of all-time failure share."""
        total_all = self.success + self.failure
        if total_all == 0:
            return 0
        failure_share = self.failure / total_all
        # number of events represented in window = successes_in_window / (1 - failure_share)
        successes_window = len(self.times_ms)
        if failure_share >= 0.9999:
            return successes_window  # degenerate
        est_total_window = successes_window / max(1e-6, (1.0 - failure_share))
        return max(0, int(round(est_total_window - successes_window)))

    def percentiles(self, ps: Tuple[float, ...] = (50, 95, 99)) -> Dict[str, float]:
        arr = self.window_times()
        if not arr:
            return {f"p{int(p)}": 0.0 for p in ps}
        a = np.asarray(arr, dtype=np.float64)
        out = {}
        for p in ps:
            out[f"p{int(p)}"] = float(np.percentile(a, p))
        return out

    def average_ms(self) -> float:
        arr = self.window_times()
        return float(np.mean(arr)) if arr else 0.0

    def median_ms(self) -> float:
        arr = self.window_times()
        return float(np.median(arr)) if arr else 0.0


@dataclass
class _SLO:
    """Service Level Objective per metric."""
    target_success_rate: float = 0.99  # 99%
    target_p95_ms: float = 200.0
    enforced: bool = False  # informational by default


class MetricsTracker:
    """
    Unified metrics tracking for all market components.
    Production-ready upgrade:
      • Rolling time window & bounded memory
      • P50/P95/P99, avg, EMA latency, histograms
      • Per-metric SLOs and evaluation
      • Thread-safe recording
      • Context managers & decorators for auto timing (sync/async)
      • Prometheus-style text export (optional)
    Backward-compatible methods retained.
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        detailed_tracking: bool = False,
        # New knobs:
        max_samples_per_metric: int = 1000,
        window_seconds: float = 300.0,
        histogram_buckets_ms: Tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000),
    ):
        self.logger = logger
        self.detailed_tracking = detailed_tracking

        self._lock = threading.RLock()
        self._stats: Dict[str, _RollingStats] = defaultdict(
            lambda: _RollingStats(max_samples=max_samples_per_metric, window_seconds=window_seconds,
                                  buckets=histogram_buckets_ms)
        )
        # Kept for backward compatibility (backed by _stats)
        self._metrics: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=1000))
        self._success_counts: Dict[str, int] = defaultdict(int)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._execution_times: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))

        # Aggregated snapshots (informational)
        self._aggregated_stats: Dict[str, Dict[str, float]] = {}
        self._last_aggregation = time.time()

        # SLOs per metric
        self._slos: Dict[str, _SLO] = {}

        self.trace("MetricsTracker initialized", level="DEBUG")

    # ----------------------------- Logging -----------------------------

    def trace(self, message: str, level: str = "INFO"):
        if self.logger:
            self.logger.trace(f"[MetricsTracker] {message}", level=level)

    # ----------------------- Back-compat API ---------------------------

    def record_success(self, metric_name: str, execution_time_ms: float):
        """Record successful execution (backward-compatible)."""
        with self._lock:
            self._success_counts[metric_name] += 1
            self._execution_times[metric_name].append(float(execution_time_ms))
            if self.detailed_tracking:
                self._metrics[metric_name].append({
                    'status': 'success',
                    'execution_time_ms': float(execution_time_ms),
                    'timestamp': time.time()
                })
            self._stats[metric_name].add_success(float(execution_time_ms))
        self.trace(f"Success recorded for {metric_name}: {execution_time_ms:.2f}ms", level="TRACE")

    def record_failure(self, metric_name: str, error: str):
        """Record failed execution (backward-compatible)."""
        err_type = type(error).__name__ if not isinstance(error, str) else "Error"
        err_msg = str(error)
        with self._lock:
            self._failure_counts[metric_name] += 1
            if self.detailed_tracking:
                self._metrics[metric_name].append({
                    'status': 'failure',
                    'error': err_msg,
                    'timestamp': time.time()
                })
            self._stats[metric_name].add_failure(err_type, err_msg)
        self.trace(f"Failure recorded for {metric_name}: {err_msg}", level="DEBUG")

    def get_success_rate(self, metric_name: str) -> float:
        """All-time success rate for a metric."""
        with self._lock:
            total = self._success_counts[metric_name] + self._failure_counts[metric_name]
            return (self._success_counts[metric_name] / total) if total > 0 else 0.0

    def get_average_execution_time(self, metric_name: str) -> float:
        """Average execution time from bounded recent deque (back-compat)."""
        with self._lock:
            times = self._execution_times.get(metric_name)
            return float(np.mean(times)) if times else 0.0

    def get_percentile_execution_time(self, metric_name: str, percentile: float = 95) -> float:
        """Percentile execution time over rolling window."""
        with self._lock:
            pmap = self._stats[metric_name].percentiles((percentile,))
            return pmap.get(f"p{int(percentile)}", 0.0)

    def get_metrics_summary(self, metric_name: str) -> Dict[str, Any]:
        """Comprehensive summary on current rolling window (plus all-time counters)."""
        with self._lock:
            st = self._stats[metric_name]
            pmap = st.percentiles((50, 95, 99))
            times = list(self._execution_times.get(metric_name, []))
            all_success = self._success_counts[metric_name]
            all_failure = self._failure_counts[metric_name]
            summary = {
                'success_count': all_success,
                'failure_count': all_failure,
                'success_rate': self.get_success_rate(metric_name),
                'window_success_rate': st.window_success_rate(),
                'average_time_ms': float(np.mean(times)) if times else 0.0,  # back-compat small window
                'median_time_ms': pmap['p50'],
                'p95_time_ms': pmap['p95'],
                'p99_time_ms': pmap['p99'],
                'ema_latency_ms': st.ema_latency_ms,
                'histogram_buckets_ms': st.buckets,
                'histogram_counts': list(st.bucket_counts),
                'recent_errors': list(st.recent_errors),
            }
            # SLO status (if configured)
            slo = self._slos.get(metric_name)
            if slo:
                summary['slo'] = {
                    'target_success_rate': slo.target_success_rate,
                    'target_p95_ms': slo.target_p95_ms,
                    'enforced': slo.enforced,
                    'met': (summary['window_success_rate'] >= slo.target_success_rate and
                            pmap['p95'] <= slo.target_p95_ms),
                }
            return summary

    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary across all metrics."""
        with self._lock:
            names = set(list(self._success_counts.keys()) + list(self._failure_counts.keys()) + list(self._stats.keys()))
        return {name: self.get_metrics_summary(name) for name in names}

    def aggregate_stats(self):
        """Periodic aggregation snapshot (kept for API parity)."""
        with self._lock:
            current_time = time.time()
            if current_time - self._last_aggregation < 60:
                return
            self.trace("Aggregating statistics", level="TRACE")
            for metric_name, d in self.get_summary().items():
                self._aggregated_stats[metric_name] = {
                    'success_rate': d['success_rate'],
                    'window_success_rate': d['window_success_rate'],
                    'avg_execution_time': d['average_time_ms'],
                    'p95_time_ms': d['p95_time_ms'],
                    'last_aggregation': current_time
                }
            self._last_aggregation = current_time

    def reset_metrics(self, metric_name: Optional[str] = None):
        """Reset metrics for specific metric or all."""
        with self._lock:
            if metric_name:
                self._success_counts.pop(metric_name, None)
                self._failure_counts.pop(metric_name, None)
                if metric_name in self._execution_times:
                    self._execution_times[metric_name].clear()
                if metric_name in self._metrics:
                    self._metrics[metric_name].clear()
                self._stats.pop(metric_name, None)
            else:
                self._success_counts.clear()
                self._failure_counts.clear()
                self._execution_times.clear()
                self._metrics.clear()
                self._stats.clear()
        self.trace(f"Metrics reset: {metric_name or 'all'}", level="INFO")

    # ---------------------------- New API --------------------------------

    def set_slo(self, metric_name: str, target_success_rate: float = 0.99, target_p95_ms: float = 200.0, enforced: bool = False):
        """Configure an SLO for a metric (success rate + p95 latency)."""
        with self._lock:
            self._slos[metric_name] = _SLO(target_success_rate, target_p95_ms, enforced)
        self.trace(f"SLO set for {metric_name}: sr≥{target_success_rate:.3f}, p95≤{target_p95_ms:.1f}ms", level="DEBUG")

    def evaluate_slo(self, metric_name: str) -> Dict[str, Any]:
        """Evaluate current SLO status for metric (rolling window)."""
        s = self.get_metrics_summary(metric_name)  # uses lock internally
        slo = s.get('slo')
        if not slo:
            return {'configured': False}
        return {'configured': True, **slo}

    @contextmanager
    def time_block(self, metric_name: str):
        """Context manager to time a code block (sync)."""
        start_ns = time.perf_counter_ns()
        try:
            yield
        except Exception as e:
            self.record_failure(metric_name, str(e))
            raise
        else:
            dur_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self.record_success(metric_name, dur_ms)

    @asynccontextmanager
    async def time_block_async(self, metric_name: str):
        """Context manager to time an async code block."""
        start_ns = time.perf_counter_ns()
        try:
            yield
        except Exception as e:
            self.record_failure(metric_name, str(e))
            raise
        else:
            dur_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self.record_success(metric_name, dur_ms)

    def timed(self, metric_name: str):
        """Decorator to time a sync function."""
        def deco(fn: Callable):
            def wrapper(*args, **kwargs):
                start_ns = time.perf_counter_ns()
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    self.record_failure(metric_name, str(e))
                    raise
                finally:
                    end_ns = time.perf_counter_ns()
                    dur_ms = (end_ns - start_ns) / 1e6
                    # Only record success if no exception
                    # (we still recorded failure above on exception)
                    if end_ns >= start_ns and 'e' not in locals():
                        self.record_success(metric_name, dur_ms)
            return wrapper
        return deco

    def timed_async(self, metric_name: str):
        """Decorator to time an async function."""
        def deco(fn: Callable[..., Awaitable]):
            async def wrapper(*args, **kwargs):
                start_ns = time.perf_counter_ns()
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    self.record_failure(metric_name, str(e))
                    raise
                finally:
                    end_ns = time.perf_counter_ns()
                    dur_ms = (end_ns - start_ns) / 1e6
                    if end_ns >= start_ns and 'e' not in locals():
                        self.record_success(metric_name, dur_ms)
            return wrapper
        return deco

    # ------------------------ Prometheus Export ---------------------------

    def export_prometheus(self, prefix: str = "market") -> str:
        """
        Render a text exposition of counters and histograms (Prometheus format).
        Intended for quick scraping via an HTTP debug endpoint.
        """
        lines: List[str] = []
        with self._lock:
            for name, st in self._stats.items():
                label = f'metric="{name}"'
                # Counters
                lines.append(f'# HELP {prefix}_success_total Total successes')
                lines.append(f'# TYPE {prefix}_success_total counter')
                lines.append(f'{prefix}_success_total{{{label}}} {st.success}')
                lines.append(f'# HELP {prefix}_failure_total Total failures')
                lines.append(f'# TYPE {prefix}_failure_total counter')
                lines.append(f'{prefix}_failure_total{{{label}}} {st.failure}')
                # Histogram
                lines.append(f'# HELP {prefix}_latency_milliseconds Latency histogram (ms)')
                lines.append(f'# TYPE {prefix}_latency_milliseconds histogram')
                cum = 0
                for b, c in zip(st.buckets, st.bucket_counts[:-1]):
                    cum += c
                    lines.append(f'{prefix}_latency_milliseconds_bucket{{{label},le="{b}"}} {cum}')
                cum += st.bucket_counts[-1]
                lines.append(f'{prefix}_latency_milliseconds_bucket{{{label},le="+Inf"}} {cum}')
                lines.append(f'{prefix}_latency_milliseconds_count{{{label}}} {cum}')
                # We approximate sum by EMA * count for brevity
                lines.append(f'{prefix}_latency_milliseconds_sum{{{label}}} {st.ema_latency_ms * max(cum,1)}')
        return "\n".join(lines)
