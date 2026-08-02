

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from modules.core.configuration_manager import ConfigurationManager
from modules.core.module_system import ModuleOrchestrator
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager
from modules.utils.system_utilities import EnglishExplainer


@dataclass
class PerformanceMetric:
    timestamp: float
    monotonic_ns: int
    module: str
    operation: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'monotonic_ns': self.monotonic_ns,
            'module': self.module,
            'operation': self.operation,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'error': self.error,
            'memory_mb': self.memory_mb
        }


@dataclass
class PerformanceReport:
    period_start: datetime
    period_end: datetime
    module_metrics: Dict[str, Dict[str, float]]
    bottlenecks: List[str]
    trends: Dict[str, str]
    recommendations: List[str]
    summary: str


class PerformanceTracker:

    _singleton: Optional["PerformanceTracker"] = None
    _singleton_lock = threading.Lock()


    _DEFAULTS = {
        'response_time_ms': {'excellent': 50, 'good': 100, 'acceptable': 200, 'poor': 500},
        'error_rate': {'excellent': 0.001, 'good': 0.01, 'acceptable': 0.05, 'poor': 0.10},
        'throughput_per_min': {'excellent': 100, 'good': 50, 'acceptable': 20, 'poor': 10},
        'trend_window': 100,
        'anomaly_threshold': 3.0,
        'min_ms_for_anomaly': 100.0,
        'publish_interval_s': 15,
        'alert_cooldown_s': 15,
        'bus_namespace': 'perf',
        'max_metrics': 10_000,
        'per_module_max': 1_000,
    }

    def __new__(cls, *args, **kwargs):
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(self, orchestrator: Optional[ModuleOrchestrator] = None):

        if getattr(self, "_initialized", False):
            return
        self._initialized = True


        self.orchestrator = orchestrator
        self.smart_bus = InfoBusManager.get_instance()
        self.explainer = EnglishExplainer()


        self._config = self._load_runtime_config()


        self._lock = threading.RLock()


        self.metrics: Deque[PerformanceMetric] = deque(maxlen=int(self._config['max_metrics']))
        self.module_metrics: Dict[str, Deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=int(self._config['per_module_max']))
        )


        self.hourly_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.daily_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)


        self.thresholds = {
            'response_time_ms': dict(self._config['response_time_ms']),
            'error_rate': dict(self._config['error_rate']),
            'throughput_per_min': dict(self._config['throughput_per_min'])
        }
        self.trend_window: int = int(self._config['trend_window'])
        self.anomaly_threshold: float = float(self._config['anomaly_threshold'])
        self._min_ms_for_anomaly: float = float(self._config['min_ms_for_anomaly'])


        self.logger = RotatingLogger(
            name="PerformanceTracker",
            log_path="logs/monitoring/performance.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )


        self._subscribed = False
        self._event_cooldowns: Dict[Tuple[str, str], float] = {}
        self._event_cooldown_seconds = float(self._config['alert_cooldown_s'])
        self._subscribe_to_events()


        self._publisher_shutdown = False
        self._publisher_thread = threading.Thread(
            target=self._publisher_loop, daemon=True, name="PerfPublisher"
        )
        self._publisher_thread.start()


    def _load_runtime_config(self) -> Dict[str, Any]:
        cfg = dict(self._DEFAULTS)
        if ConfigurationManager is not None:
            try:
                cm = ConfigurationManager.get_instance()
                mon = cm.get_monitoring_config()

                for k in ('response_time_ms', 'error_rate', 'throughput_per_min',
                          'trend_window', 'anomaly_threshold', 'min_ms_for_anomaly',
                          'publish_interval_s', 'alert_cooldown_s',
                          'bus_namespace', 'max_metrics', 'per_module_max'):
                    if k in mon:
                        cfg[k] = mon[k]
            except Exception:
                pass
        return cfg


    def _snapshot_deque(self, dq: Deque) -> List[Any]:
        with self._lock:
            return list(dq)

    def _snapshot_module_keys(self) -> List[str]:
        with self._lock:
            return list(self.module_metrics.keys())

    def _safe_percentile(self, values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        try:
            if np is not None:
                return float(np.percentile(values, pct))
        except Exception:
            pass

        sorted_vals = sorted(values)
        idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
        return float(sorted_vals[max(0, min(idx, len(sorted_vals) - 1))])


    @contextlib.contextmanager
    def track(self, module: str, operation: str, *, capture_memory: bool = False):
        t0 = time.perf_counter()
        mem0 = self._memory_mb() if (capture_memory and psutil) else None
        success = True
        err: Optional[str] = None
        try:
            yield
        except Exception as e:
            success = False
            err = str(e)
            raise
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            mem_mb = None
            if capture_memory and psutil:
                mem1 = self._memory_mb()
                if mem0 is not None and mem1 is not None:
                    mem_mb = max(0.0, mem1 - mem0)
            self.record_metric(module, operation, dt_ms, success=success, error=err, memory_mb=mem_mb)

    def wrap(self, module: str, operation: str, *, capture_memory: bool = False) -> Callable:
        def decorator(fn: Callable):
            def inner(*args, **kwargs):
                with self.track(module, operation, capture_memory=capture_memory):
                    return fn(*args, **kwargs)
            inner.__name__ = getattr(fn, "__name__", "wrapped")
            inner.__doc__ = getattr(fn, "__doc__", "")
            return inner
        return decorator

    def _memory_mb(self) -> Optional[float]:
        try:
            if psutil is None:
                return None
            proc = psutil.Process()
            return float(proc.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return None


    def record_metric(
        self,
        module: str,
        operation: str,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        memory_mb: Optional[float] = None
    ):
        try:
            d = float(duration_ms)
        except Exception:
            d = 0.0
        if -1.0 < d < 0.0:
            d = 0.0

        metric = PerformanceMetric(
            timestamp=time.time(),
            monotonic_ns=time.perf_counter_ns(),
            module=str(module),
            operation=str(operation),
            duration_ms=d,
            success=bool(success),
            error=error,
            memory_mb=memory_mb
        )

        with self._lock:
            self.metrics.append(metric)
            self.module_metrics[module].append(metric)


        try:
            if hasattr(self.smart_bus, "record_module_timing"):
                self.smart_bus.record_module_timing(module, d)
        except Exception:
            pass

        self._check_performance_issues(module, metric)
        self._update_aggregated_stats(metric)

    def get_module_performance(self, module: str, window_minutes: int = 60) -> Dict[str, Any]:
        cutoff = time.time() - (window_minutes * 60)
        with self._lock:
            module_deque = self.module_metrics.get(module, deque())
            metrics_list = [m for m in module_deque if m.timestamp > cutoff]

        if not metrics_list:
            return {
                'avg_time_ms': 0.0, 'max_time_ms': 0.0, 'min_time_ms': 0.0, 'p95_time_ms': 0.0,
                'error_rate': 0.0, 'success_count': 0, 'error_count': 0,
                'throughput_per_min': 0.0, 'trend': 'insufficient_data'
            }

        durations = [m.duration_ms for m in metrics_list]
        total = len(metrics_list)
        errors = sum(1 for m in metrics_list if not m.success)

        span_min = max((metrics_list[-1].timestamp - metrics_list[0].timestamp) / 60.0, 1.0)
        throughput = total / span_min

        return {
            'avg_time_ms': (float(np.mean(durations)) if (np and durations) else (sum(durations)/len(durations))),
            'max_time_ms': max(durations),
            'min_time_ms': min(durations),
            'p95_time_ms': self._safe_percentile(durations, 95),
            'error_rate': (errors / total),
            'success_count': total - errors,
            'error_count': errors,
            'throughput_per_min': throughput,
            'trend': self._calculate_trend(durations)
        }

    def generate_performance_report(self, period_hours: int = 24) -> PerformanceReport:
        period_start = datetime.now() - timedelta(hours=period_hours)
        period_end = datetime.now()
        modules = self._snapshot_module_keys()

        module_metrics: Dict[str, Dict[str, float]] = {}
        for module in modules:
            metrics = self.get_module_performance(module, period_hours * 60)
            if metrics['success_count'] > 0 or metrics['error_count'] > 0:
                module_metrics[module] = metrics

        bottlenecks = self._identify_bottlenecks(module_metrics)
        trends = self._analyze_trends(module_metrics)
        recommendations = self._generate_recommendations(module_metrics, bottlenecks)
        summary = self._create_performance_summary(module_metrics, period_hours)

        return PerformanceReport(
            period_start=period_start,
            period_end=period_end,
            module_metrics=module_metrics,
            bottlenecks=bottlenecks,
            trends=trends,
            recommendations=recommendations,
            summary=summary
        )

    def get_plain_english_report(self, period_hours: int = 24) -> str:
        report = self.generate_performance_report(period_hours)
        return self.explainer.explain_performance(
            module_name="System Overall",
            metrics={
                'avg_time_ms': self._calculate_system_average(report.module_metrics),
                'error_rate': self._calculate_system_error_rate(report.module_metrics),
                'throughput_per_min': self._calculate_system_throughput(report.module_metrics),
                'bottleneck_count': len(report.bottlenecks),
                'module_count': len(report.module_metrics)
            },
            period=f"Last {period_hours} hours"
        ) + self._format_detailed_findings(report)

    def export_metrics(self, filepath: str, period_hours: int = 24):
        cutoff = time.time() - (period_hours * 3600)
        with self._lock:
            metrics_snapshot = [m.to_dict() for m in self.metrics if m.timestamp > cutoff]

        metrics_data = {
            'export_time': datetime.now().isoformat(),
            'period_hours': period_hours,
            'metrics': metrics_snapshot,
            'summary': asdict(self.generate_performance_report(period_hours))
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=str)

    def get_realtime_dashboard_data(self) -> Dict[str, Any]:
        cutoff = time.time() - 300
        with self._lock:
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff]
            module_keys = list(self.module_metrics.keys())
            module_current: Dict[str, Any] = {}
            for module in module_keys:
                dq = self.module_metrics[module]
                module_recent = [m for m in dq if m.timestamp > cutoff]
                if module_recent:
                    durations = [m.duration_ms for m in module_recent]
                    module_current[module] = {
                        'avg_time': (float(np.mean(durations)) if (np and durations) else (sum(durations)/len(durations) if durations else 0.0)),
                        'count': len(module_recent),
                        'errors': sum(1 for m in module_recent if not m.success)
                    }

        if recent_metrics:
            current_throughput = len(recent_metrics) / 5.0
            current_error_rate = sum(1 for m in recent_metrics if not m.success) / len(recent_metrics)
            avg_list = [m.duration_ms for m in recent_metrics]
            current_avg_time = (float(np.mean(avg_list)) if (np and avg_list) else (sum(avg_list)/len(avg_list) if avg_list else 0.0))
        else:
            current_throughput = 0.0
            current_error_rate = 0.0
            current_avg_time = 0.0

        recent_errors = [
            {'module': m.module, 'operation': m.operation, 'error': m.error or '<no details>', 'timestamp': m.timestamp}
            for m in recent_metrics if not m.success
        ][-10:]

        return {
            'timestamp': time.time(),
            'current_throughput': current_throughput,
            'current_error_rate': current_error_rate,
            'current_avg_time': current_avg_time,
            'module_performance': module_current,
            'recent_errors': recent_errors
        }


    def _subscribe_to_events(self):
        if self._subscribed:
            return
        try:
            self.smart_bus.subscribe('performance_warning', self._handle_performance_warning)
            self.smart_bus.subscribe('module_disabled', self._handle_module_disabled)
            self._subscribed = True
        except Exception:

            pass

    def _cooldown_allows(self, key: Tuple[str, str]) -> bool:
        now = time.time()
        last = self._event_cooldowns.get(key, 0.0)
        if now - last >= self._event_cooldown_seconds:
            self._event_cooldowns[key] = now
            return True
        return False

    def _handle_performance_warning(self, data: Dict[str, Any]):
        module = str(data.get('module', 'Unknown'))
        avg_latency = float(data.get('avg_latency_ms', 0.0))
        key = ('performance_warning', module)
        if not self._cooldown_allows(key):
            return
        self.logger.warning(
            format_operator_message(
                "[WARN]", "PERFORMANCE WARNING",
                instrument=module,
                details=f"Average latency {avg_latency:.0f}ms",
                context="performance"
            )
        )

    def _handle_module_disabled(self, data: Dict[str, Any]):
        module = str(data.get('module', 'Unknown'))
        failures = int(data.get('failures', 0))
        key = ('module_disabled', module)
        if not self._cooldown_allows(key):
            return
        self.logger.error(
            format_operator_message(
                "🚫", "MODULE DISABLED",
                instrument=module,
                details=f"After {failures} failures",
                context="circuit_breaker"
            )
        )


    def _check_performance_issues(self, module: str, metric: PerformanceMetric):
        if metric.duration_ms > self.thresholds['response_time_ms']['poor']:
            self.logger.warning(
                f"Slow operation: {module}.{metric.operation} took {metric.duration_ms:.0f}ms"
            )

        if not metric.success:
            self.logger.error(
                f"Operation failed: {module}.{metric.operation} - {metric.error or '<no error details>'}"
            )

        if self._is_anomaly(module, metric.duration_ms):
            self.logger.warning(
                f"Performance anomaly detected: {module}.{metric.operation} ({metric.duration_ms:.0f}ms is unusual)"
            )

    def _is_anomaly(self, module: str, duration_ms: float) -> bool:

        if duration_ms < self._min_ms_for_anomaly:
            return False
        with self._lock:
            recent = [m.duration_ms for m in list(self.module_metrics[module])[-100:]]
        if len(recent) < 10:
            return False
        baseline = recent[:-1] if len(recent) > 1 else recent
        if not baseline:
            return False

        if np is not None:
            mean = float(np.mean(baseline))
            std = float(np.std(baseline))
        else:
            mean = sum(baseline) / len(baseline)
            var = sum((x - mean) ** 2 for x in baseline) / max(1, len(baseline))
            std = var ** 0.5
        if std == 0.0:
            return False


        if duration_ms <= mean:
            return False
        z = (duration_ms - mean) / std
        return bool(z > self.anomaly_threshold)

    def _update_aggregated_stats(self, metric: PerformanceMetric):
        dt = datetime.fromtimestamp(metric.timestamp)
        hour_key = dt.strftime('%Y-%m-%d %H:00')
        day_key = dt.strftime('%Y-%m-%d')
        with self._lock:
            h = self.hourly_stats[metric.module].setdefault(
                hour_key, {'count': 0, 'errors': 0, 'total_time': 0.0, 'max_time': 0.0}
            )
            h['count'] += 1
            h['total_time'] += metric.duration_ms
            h['max_time'] = max(h['max_time'], metric.duration_ms)
            if not metric.success:
                h['errors'] += 1

            d = self.daily_stats[metric.module].setdefault(
                day_key, {'count': 0, 'errors': 0, 'total_time': 0.0, 'max_time': 0.0}
            )
            d['count'] += 1
            d['total_time'] += metric.duration_ms
            d['max_time'] = max(d['max_time'], metric.duration_ms)
            if not metric.success:
                d['errors'] += 1

    def _identify_bottlenecks(self, module_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        if not module_metrics:
            return []
        sorted_modules = sorted(
            module_metrics.items(),
            key=lambda x: x[1].get('avg_time_ms', 0.0),
            reverse=True
        )
        out: List[str] = []
        for module, m in sorted_modules:
            avg_ms = m.get('avg_time_ms', 0.0)
            err_rate = m.get('error_rate', 0.0)
            max_ms = m.get('max_time_ms', 0.0)
            if avg_ms > self.thresholds['response_time_ms']['acceptable']:
                out.append(f"{module}: Slow average response time ({avg_ms:.0f}ms)")
            if err_rate > self.thresholds['error_rate']['acceptable']:
                out.append(f"{module}: High error rate ({err_rate:.1%})")
            if max_ms > self.thresholds['response_time_ms']['poor'] * 2:
                out.append(f"{module}: Extreme outliers (max {max_ms:.0f}ms)")
        return out[:10]

    def _analyze_trends(self, module_metrics: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        trends: Dict[str, str] = {}
        for module, m in module_metrics.items():
            trend = m.get('trend', 'stable')
            if trend and trend != 'stable':
                trends[module] = str(trend)
        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        if len(values) < 10:
            return 'insufficient_data'
        window = values[-self.trend_window:] if len(values) > self.trend_window else values
        x = list(range(len(window)))
        try:
            if np is not None:
                slope, _ = np.polyfit(np.arange(len(window), dtype=float), window, 1)
            else:

                n = float(len(window))
                sum_x = sum(x)
                sum_y = sum(window)
                sum_xx = sum(i * i for i in x)
                sum_xy = sum(i * y for i, y in zip(x, window))
                denom = (n * sum_xx - sum_x * sum_x) or 1.0
                slope = (n * sum_xy - sum_x * sum_y) / denom
        except Exception:
            return 'stable'
        mean_val = (float(np.mean(window)) if (np and window) else (sum(window)/len(window)))
        if mean_val == 0.0:
            return 'stable'
        change = slope / mean_val
        if change > 0.01:
            return 'degrading'
        elif change < -0.01:
            return 'improving'
        return 'stable'

    def _generate_recommendations(self, module_metrics: Dict[str, Dict[str, float]], bottlenecks: List[str]) -> List[str]:
        recs: List[str] = []
        avg_response = self._calculate_system_average(module_metrics)
        err_rate = self._calculate_system_error_rate(module_metrics)

        if avg_response > self.thresholds['response_time_ms']['good']:
            recs.append("System response times are higher than optimal. Profile slow modules and optimize algorithms.")
        if err_rate > self.thresholds['error_rate']['good']:
            recs.append(f"Error rate ({err_rate:.1%}) exceeds target. Investigate error patterns and add better handling.")

        for module, m in module_metrics.items():
            if m['avg_time_ms'] > self.thresholds['response_time_ms']['acceptable']:
                recs.append(f"Optimize {module}: consider caching/parallelism/algorithmic improvements (~{m['avg_time_ms']:.0f}ms avg).")
            if m['error_rate'] > 0.10:
                recs.append(f"Fix {module}: critical error rate {m['error_rate']:.1%} indicates serious issues.")

        if len(bottlenecks) > 5:
            recs.append("Multiple bottlenecks detected. Consider an architectural review to improve system design.")

        return recs[:10]

    def _create_performance_summary(self, module_metrics: Dict[str, Dict[str, float]], period_hours: int) -> str:
        total_ops = sum(m.get('success_count', 0) + m.get('error_count', 0) for m in module_metrics.values())
        avg_response = self._calculate_system_average(module_metrics)
        err_rate = self._calculate_system_error_rate(module_metrics)

        if err_rate > 0.10 or avg_response > 500:
            status = "Critical - Immediate attention required"
        elif err_rate > 0.05 or avg_response > 200:
            status = "Degraded - Performance issues detected"
        elif err_rate > 0.01 or avg_response > 100:
            status = "Fair - Room for improvement"
        else:
            status = "Good - System performing well"

        return (
            f"\nPerformance Summary ({period_hours} hours)\n"
            f"Status: {status}\n"
            f"Total Operations: {total_ops:,}\n"
            f"Active Modules: {len(module_metrics)}\n"
            f"Average Response Time: {avg_response:.1f}ms\n"
            f"System Error Rate: {err_rate:.2%}\n"
        )

    def _calculate_system_average(self, module_metrics: Dict[str, Dict[str, float]]) -> float:
        total_time = 0.0
        total_ops = 0
        for m in module_metrics.values():
            ops = m.get('success_count', 0) + m.get('error_count', 0)
            total_time += m.get('avg_time_ms', 0.0) * ops
            total_ops += ops
        return (total_time / total_ops) if total_ops > 0 else 0.0

    def _calculate_system_error_rate(self, module_metrics: Dict[str, Dict[str, float]]) -> float:
        total_errors = sum(m.get('error_count', 0) for m in module_metrics.values())
        total_ops = sum(m.get('success_count', 0) + m.get('error_count', 0) for m in module_metrics.values())
        return (total_errors / total_ops) if total_ops > 0 else 0.0

    def _calculate_system_throughput(self, module_metrics: Dict[str, Dict[str, float]]) -> float:
        return float(sum(m.get('throughput_per_min', 0.0) for m in module_metrics.values()))

    def _format_detailed_findings(self, report: PerformanceReport) -> str:
        lines = ["\n\nDETAILED FINDINGS:", "=" * 50]
        fastest = sorted(report.module_metrics.items(), key=lambda x: x[1].get('avg_time_ms', 0.0))
        if fastest:
            lines += ["\nFastest Modules:", "-" * 20]
            for module, m in fastest[:3]:
                lines.append(f"• {module}: {m.get('avg_time_ms', 0.0):.1f}ms average")

        problems = [
            (mod, m) for mod, m in report.module_metrics.items()
            if m.get('error_rate', 0.0) > 0.05 or m.get('avg_time_ms', 0.0) > 200.0
        ]
        if problems:
            lines += ["\nModules Needing Attention:", "-" * 30]
            for module, m in problems:
                issues = []
                if m.get('error_rate', 0.0) > 0.05:
                    issues.append(f"{m['error_rate']:.1%} errors")
                if m.get('avg_time_ms', 0.0) > 200.0:
                    issues.append(f"{m['avg_time_ms']:.0f}ms avg response")
                lines.append(f"• {module}: {', '.join(issues)}")

        if report.trends:
            lines += ["\nPerformance Trends:", "-" * 20]
            for module, trend in report.trends.items():
                symbol = "↗" if trend == "improving" else "↘" if trend == "degrading" else "→"
                lines.append(f"• {module}: {trend} {symbol}")
        return "\n".join(lines)


    def _publisher_loop(self):
        ns = str(self._config.get('bus_namespace', 'perf')).strip('/')
        interval = max(5, int(self._config.get('publish_interval_s', 15)))
        while not self._publisher_shutdown:
            try:
                dashboard = self.get_realtime_dashboard_data()

                self.smart_bus.set(
                    f"{ns}/summary",
                    {
                        'timestamp': dashboard['timestamp'],
                        'throughput_per_min': dashboard['current_throughput'],
                        'error_rate': dashboard['current_error_rate'],
                        'avg_time_ms': dashboard['current_avg_time'],
                    },
                    module="PerformanceTracker",
                    thesis="Realtime performance summary"
                )

                per_mod = dashboard.get('module_performance', {})
                for mod, m in list(per_mod.items())[:50]:
                    self.smart_bus.set(
                        f"{ns}/modules/{mod}",
                        m,
                        module="PerformanceTracker",
                        thesis=f"Realtime performance for {mod}"
                    )

                recent_errors = dashboard.get('recent_errors', [])
                if recent_errors:
                    self.smart_bus.set(
                        f"{ns}/recent_errors",
                        recent_errors[-10:],
                        module="PerformanceTracker",
                        thesis="Recent performance errors"
                    )
            except Exception:

                pass
            time.sleep(interval)


    def shutdown(self):
        self._publisher_shutdown = True
        try:
            if self._publisher_thread.is_alive():
                self._publisher_thread.join(timeout=1.0)
        except Exception:
            pass


    def get_health_status(self) -> Dict[str, Any]:
        try:

            dashboard = self.get_realtime_dashboard_data()


            with self._lock:
                total_metrics = len(self.metrics)
                module_count = len(self.module_metrics)


            current_error_rate = dashboard.get('current_error_rate', 0.0)
            current_throughput = dashboard.get('current_throughput', 0.0)

            status = 'OK'
            is_healthy = True
            issues = []


            if current_error_rate > 0.10:
                status = 'DEGRADED'
                is_healthy = False
                issues.append(f"High system error rate: {current_error_rate:.1%}")


            if not self._publisher_thread or not self._publisher_thread.is_alive():
                status = 'DEGRADED'
                is_healthy = False
                issues.append("Publisher thread not running")


            if module_count == 0:
                status = 'WARNING'
                issues.append("No modules being tracked")

            return {
                'status': status,
                'module': 'PerformanceTracker',
                'version': '2.3',
                'is_healthy': is_healthy,
                'metrics_tracked': total_metrics,
                'modules_tracked': module_count,
                'current_throughput': current_throughput,
                'current_error_rate': current_error_rate,
                'current_avg_time_ms': dashboard.get('current_avg_time', 0.0),
                'publisher_active': bool(self._publisher_thread and self._publisher_thread.is_alive()),
                'issues': issues if issues else None,
                'performance': {
                    'total_metrics': total_metrics,
                    'modules_tracked': module_count,
                    'bus_publishing': self._subscribed
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'module': 'PerformanceTracker',
                'version': '2.3',
                'is_healthy': False,
                'error': str(e),
                'last_error': str(e)
            }
