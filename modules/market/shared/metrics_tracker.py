

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _RollingStats:
    max_samples: int = 1000
    window_seconds: float = 300.0

    times_ms: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=1000))

    success: int = 0
    failure: int = 0

    recent_errors: Deque[Tuple[float, str, str]] = field(default_factory=lambda: deque(maxlen=50))

    ema_latency_ms: float = 0.0
    ema_alpha: float = 0.2

    buckets: Tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
    bucket_counts: List[int] = field(default_factory=list)

    _last_prune: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.bucket_counts:
            self.bucket_counts = [0 for _ in self.buckets] + [0]

    def _prune_old(self, now: float) -> None:
        if not self.times_ms:
            return
        if now - self._last_prune < 5.0:
            return
        cutoff = now - self.window_seconds
        while self.times_ms and self.times_ms[0][0] < cutoff:
            self.times_ms.popleft()

        self._last_prune = now

    def add_success(self, duration_ms: float, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.success += 1

        self.times_ms.append((now, float(duration_ms)))
        self._prune_old(now)

        if self.ema_latency_ms == 0.0:
            self.ema_latency_ms = float(duration_ms)
        else:
            self.ema_latency_ms = (
                self.ema_alpha * float(duration_ms) + (1 - self.ema_alpha) * self.ema_latency_ms
            )

        self._add_to_histogram(float(duration_ms))

    def add_failure(self, err_type: str, err_msg: str, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.failure += 1
        if err_msg is None:
            err_msg = ""

        msg = err_msg if len(err_msg) <= 240 else (err_msg[:237] + "...")
        self.recent_errors.append((now, err_type, msg))
        self._prune_old(now)

    def _add_to_histogram(self, v: float) -> None:

        for i, b in enumerate(self.buckets):
            if v <= b:
                self.bucket_counts[i] += 1
                return
        self.bucket_counts[-1] += 1


    def window_times(self) -> List[float]:
        return [t for _, t in self.times_ms]

    def window_success_rate(self) -> float:
        total_window = len(self.times_ms) + self._window_failures_estimate()
        if total_window <= 0:

            total_all = self.success + self.failure
            return (self.success / total_all) if total_all > 0 else 0.0


        successes_window = len(self.times_ms)
        return successes_window / total_window

    def _window_failures_estimate(self) -> int:
        total_all = self.success + self.failure
        if total_all == 0:
            return 0
        failure_share = self.failure / total_all

        successes_window = len(self.times_ms)
        if failure_share >= 0.9999:
            return successes_window
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
    target_success_rate: float = 0.99
    target_p95_ms: float = 200.0
    enforced: bool = False


class MetricsTracker:

    def __init__(
        self,
        logger: Optional[Any] = None,
        detailed_tracking: bool = False,

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

        self._metrics: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=1000))
        self._success_counts: Dict[str, int] = defaultdict(int)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._execution_times: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))


        self._aggregated_stats: Dict[str, Dict[str, float]] = {}
        self._last_aggregation = time.time()


        self._slos: Dict[str, _SLO] = {}

        self.trace("MetricsTracker initialized", level="DEBUG")


    def trace(self, message: str, level: str = "INFO"):
        if self.logger:
            self.logger.trace(f"[MetricsTracker] {message}", level=level)


    def record_success(self, metric_name: str, execution_time_ms: float):
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
        with self._lock:
            total = self._success_counts[metric_name] + self._failure_counts[metric_name]
            return (self._success_counts[metric_name] / total) if total > 0 else 0.0

    def get_average_execution_time(self, metric_name: str) -> float:
        with self._lock:
            times = self._execution_times.get(metric_name)
            return float(np.mean(times)) if times else 0.0

    def get_percentile_execution_time(self, metric_name: str, percentile: float = 95) -> float:
        with self._lock:
            pmap = self._stats[metric_name].percentiles((percentile,))
            return pmap.get(f"p{int(percentile)}", 0.0)

    def get_metrics_summary(self, metric_name: str) -> Dict[str, Any]:
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
                'average_time_ms': float(np.mean(times)) if times else 0.0,
                'median_time_ms': pmap['p50'],
                'p95_time_ms': pmap['p95'],
                'p99_time_ms': pmap['p99'],
                'ema_latency_ms': st.ema_latency_ms,
                'histogram_buckets_ms': st.buckets,
                'histogram_counts': list(st.bucket_counts),
                'recent_errors': list(st.recent_errors),
            }

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
        with self._lock:
            names = set(list(self._success_counts.keys()) + list(self._failure_counts.keys()) + list(self._stats.keys()))
        return {name: self.get_metrics_summary(name) for name in names}

    def aggregate_stats(self):
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


    def set_slo(self, metric_name: str, target_success_rate: float = 0.99, target_p95_ms: float = 200.0, enforced: bool = False):
        with self._lock:
            self._slos[metric_name] = _SLO(target_success_rate, target_p95_ms, enforced)
        self.trace(f"SLO set for {metric_name}: sr≥{target_success_rate:.3f}, p95≤{target_p95_ms:.1f}ms", level="DEBUG")

    def evaluate_slo(self, metric_name: str) -> Dict[str, Any]:
        s = self.get_metrics_summary(metric_name)
        slo = s.get('slo')
        if not slo:
            return {'configured': False}
        return {'configured': True, **slo}

    @contextmanager
    def time_block(self, metric_name: str):
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


                    if end_ns >= start_ns and 'e' not in locals():
                        self.record_success(metric_name, dur_ms)
            return wrapper
        return deco

    def timed_async(self, metric_name: str):
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


    def export_prometheus(self, prefix: str = "market") -> str:
        lines: List[str] = []
        with self._lock:
            for name, st in self._stats.items():
                label = f'metric="{name}"'

                lines.append(f'# HELP {prefix}_success_total Total successes')
                lines.append(f'# TYPE {prefix}_success_total counter')
                lines.append(f'{prefix}_success_total{{{label}}} {st.success}')
                lines.append(f'# HELP {prefix}_failure_total Total failures')
                lines.append(f'# TYPE {prefix}_failure_total counter')
                lines.append(f'{prefix}_failure_total{{{label}}} {st.failure}')

                lines.append(f'# HELP {prefix}_latency_milliseconds Latency histogram (ms)')
                lines.append(f'# TYPE {prefix}_latency_milliseconds histogram')
                cum = 0
                for b, c in zip(st.buckets, st.bucket_counts[:-1]):
                    cum += c
                    lines.append(f'{prefix}_latency_milliseconds_bucket{{{label},le="{b}"}} {cum}')
                cum += st.bucket_counts[-1]
                lines.append(f'{prefix}_latency_milliseconds_bucket{{{label},le="+Inf"}} {cum}')
                lines.append(f'{prefix}_latency_milliseconds_count{{{label}}} {cum}')

                lines.append(f'{prefix}_latency_milliseconds_sum{{{label}}} {st.ema_latency_ms * max(cum,1)}')
        return "\n".join(lines)
