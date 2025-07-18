# ─────────────────────────────────────────────────────────────
# File: modules/monitoring/performance_tracker.py
# [ROCKET] Performance tracking for SmartInfoBus with plain English reports
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from modules.utils.info_bus import  InfoBusManager
from modules.utils.audit_utils import format_operator_message, RotatingLogger
from modules.utils.system_utilities import EnglishExplainer

if TYPE_CHECKING:
    from modules.core.module_system import ModuleOrchestrator


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: float
    module: str
    operation: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'module': self.module,
            'operation': self.operation,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'error': self.error,
            'memory_mb': self.memory_mb
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    period_start: datetime
    period_end: datetime
    module_metrics: Dict[str, Dict[str, float]]
    bottlenecks: List[str]
    trends: Dict[str, str]
    recommendations: List[str]
    summary: str


class PerformanceTracker:
    """
    Tracks and analyzes system performance with plain English reporting.
    Identifies bottlenecks and provides optimization recommendations.
    """
    
    def __init__(self, orchestrator: Optional[ModuleOrchestrator] = None):
        self.orchestrator = orchestrator
        self.smart_bus = InfoBusManager.get_instance()
        self.explainer = EnglishExplainer()
        
        # Performance metrics storage
        self.metrics: deque = deque(maxlen=10000)
        self.module_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Aggregated statistics
        self.hourly_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.daily_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        
        # Performance thresholds
        self.thresholds = {
            'response_time_ms': {
                'excellent': 50,
                'good': 100,
                'acceptable': 200,
                'poor': 500
            },
            'error_rate': {
                'excellent': 0.001,
                'good': 0.01,
                'acceptable': 0.05,
                'poor': 0.1
            },
            'throughput_per_min': {
                'excellent': 100,
                'good': 50,
                'acceptable': 20,
                'poor': 10
            }
        }
        
        # Trend analysis
        self.trend_window = 100  # Number of samples for trend
        self.anomaly_threshold = 3  # Standard deviations for anomaly
        
        # Setup logging
        self.logger = RotatingLogger(
            name="PerformanceTracker",
            log_path="logs/monitoring/performance.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        
        # Subscribe to SmartInfoBus events
        self._subscribe_to_events()
    
    def record_metric(self, module: str, operation: str, duration_ms: float,
                     success: bool = True, error: Optional[str] = None,
                     memory_mb: Optional[float] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            timestamp=time.time(),
            module=module,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error=error,
            memory_mb=memory_mb
        )
        
        self.metrics.append(metric)
        self.module_metrics[module].append(metric)
        
        # Update SmartInfoBus
        self.smart_bus.record_module_timing(module, duration_ms)
        
        # Check for performance issues
        self._check_performance_issues(module, metric)
        
        # Update aggregated stats
        self._update_aggregated_stats(metric)
    
    def get_module_performance(self, module: str, 
                              window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics for a specific module"""
        cutoff = time.time() - (window_minutes * 60)
        module_metrics = [m for m in self.module_metrics[module] 
                         if m.timestamp > cutoff]
        
        if not module_metrics:
            return {
                'avg_time_ms': 0,
                'max_time_ms': 0,
                'min_time_ms': 0,
                'p95_time_ms': 0,
                'error_rate': 0,
                'success_count': 0,
                'error_count': 0,
                'throughput_per_min': 0
            }
        
        durations = [m.duration_ms for m in module_metrics]
        errors = sum(1 for m in module_metrics if not m.success)
        total = len(module_metrics)
        
        # Calculate time range in minutes
        time_range = (module_metrics[-1].timestamp - module_metrics[0].timestamp) / 60
        throughput = total / max(time_range, 1)
        
        return {
            'avg_time_ms': np.mean(durations),
            'max_time_ms': max(durations),
            'min_time_ms': min(durations),
            'p95_time_ms': np.percentile(durations, 95),
            'error_rate': errors / total,
            'success_count': total - errors,
            'error_count': errors,
            'throughput_per_min': throughput,
            'trend': self._calculate_trend(durations)
        }
    
    def generate_performance_report(self, period_hours: int = 24) -> PerformanceReport:
        """Generate comprehensive performance report"""
        period_start = datetime.now() - timedelta(hours=period_hours)
        period_end = datetime.now()
        cutoff = period_start.timestamp()
        
        # Collect metrics for all modules
        module_metrics = {}
        for module in self.module_metrics:
            metrics = self.get_module_performance(module, period_hours * 60)
            if metrics['success_count'] > 0:  # Only include active modules
                module_metrics[module] = metrics
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(module_metrics)
        
        # Calculate trends
        trends = self._analyze_trends(module_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(module_metrics, bottlenecks)
        
        # Create summary
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
        """Generate plain English performance report"""
        report = self.generate_performance_report(period_hours)
        
        # Use explainer to format the report
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
    
    def _subscribe_to_events(self):
        """Subscribe to SmartInfoBus performance events"""
        self.smart_bus.subscribe('performance_warning', self._handle_performance_warning)
        self.smart_bus.subscribe('module_disabled', self._handle_module_disabled)
    
    def _handle_performance_warning(self, data: Dict[str, Any]):
        """Handle performance warning from SmartInfoBus"""
        module = data.get('module', 'Unknown')
        avg_latency = data.get('avg_latency_ms', 0)
        
        self.logger.warning(
            format_operator_message(
                "[WARN]", "PERFORMANCE WARNING",
                instrument=module,
                details=f"Average latency {avg_latency:.0f}ms",
                context="performance"
            )
        )
    
    def _handle_module_disabled(self, data: Dict[str, Any]):
        """Handle module disabled event"""
        module = data.get('module', 'Unknown')
        failures = data.get('failures', 0)
        
        self.logger.error(
            format_operator_message(
                "🚫", "MODULE DISABLED",
                instrument=module,
                details=f"After {failures} failures",
                context="circuit_breaker"
            )
        )
    
    def _check_performance_issues(self, module: str, metric: PerformanceMetric):
        """Check for performance issues in real-time"""
        # Check response time
        if metric.duration_ms > self.thresholds['response_time_ms']['poor']:
            self.logger.warning(
                f"Slow operation: {module}.{metric.operation} "
                f"took {metric.duration_ms:.0f}ms"
            )
        
        # Check for errors
        if not metric.success:
            self.logger.error(
                f"Operation failed: {module}.{metric.operation} - {metric.error}"
            )
        
        # Check for anomalies
        if self._is_anomaly(module, metric.duration_ms):
            self.logger.warning(
                f"Performance anomaly detected: {module}.{metric.operation} "
                f"({metric.duration_ms:.0f}ms is unusual)"
            )
    
    def _is_anomaly(self, module: str, duration_ms: float) -> bool:
        """Detect performance anomalies"""
        recent = [m.duration_ms for m in list(self.module_metrics[module])[-100:]]
        
        if len(recent) < 10:
            return False
        
        mean = np.mean(recent[:-1])  # Exclude current
        std = np.std(recent[:-1])
        
        if std == 0:
            return False
        
        z_score = abs((duration_ms - mean) / std)
        return bool(z_score > self.anomaly_threshold)
    
    def _update_aggregated_stats(self, metric: PerformanceMetric):
        """Update hourly and daily aggregated statistics"""
        # Get hour and day keys
        dt = datetime.fromtimestamp(metric.timestamp)
        hour_key = dt.strftime('%Y-%m-%d %H:00')
        day_key = dt.strftime('%Y-%m-%d')
        
        # Update hourly stats
        if hour_key not in self.hourly_stats[metric.module]:
            self.hourly_stats[metric.module][hour_key] = {
                'count': 0,
                'errors': 0,
                'total_time': 0,
                'max_time': 0
            }
        
        stats = self.hourly_stats[metric.module][hour_key]
        stats['count'] += 1
        stats['total_time'] += metric.duration_ms
        stats['max_time'] = max(stats['max_time'], metric.duration_ms)
        if not metric.success:
            stats['errors'] += 1
    
    def _identify_bottlenecks(self, module_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Sort by average response time
        sorted_modules = sorted(
            module_metrics.items(),
            key=lambda x: x[1]['avg_time_ms'],
            reverse=True
        )
        
        for module, metrics in sorted_modules:
            # Check various bottleneck conditions
            if metrics['avg_time_ms'] > self.thresholds['response_time_ms']['acceptable']:
                bottlenecks.append(
                    f"{module}: Slow average response time ({metrics['avg_time_ms']:.0f}ms)"
                )
            
            if metrics['error_rate'] > self.thresholds['error_rate']['acceptable']:
                bottlenecks.append(
                    f"{module}: High error rate ({metrics['error_rate']:.1%})"
                )
            
            if metrics['max_time_ms'] > self.thresholds['response_time_ms']['poor'] * 2:
                bottlenecks.append(
                    f"{module}: Extreme outliers (max {metrics['max_time_ms']:.0f}ms)"
                )
        
        return bottlenecks[:10]  # Top 10 bottlenecks
    
    def _analyze_trends(self, module_metrics: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Analyze performance trends"""
        trends = {}
        
        for module, metrics in module_metrics.items():
            trend = metrics.get('trend', 'stable')
            if trend != 'stable':
                trends[module] = trend
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Use linear regression on recent values
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Calculate percentage change
        mean_value = np.mean(values)
        if mean_value == 0:
            return 'stable'
        
        change_per_sample = slope / mean_value
        
        if change_per_sample > 0.01:  # 1% increase per sample
            return 'degrading'
        elif change_per_sample < -0.01:  # 1% decrease per sample
            return 'improving'
        else:
            return 'stable'
    
    def _generate_recommendations(self, module_metrics: Dict[str, Dict[str, float]], 
                                bottlenecks: List[str]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Analyze overall system performance
        avg_response = self._calculate_system_average(module_metrics)
        error_rate = self._calculate_system_error_rate(module_metrics)
        
        if avg_response > self.thresholds['response_time_ms']['good']:
            recommendations.append(
                "System response times are higher than optimal. "
                "Consider profiling slow modules and optimizing algorithms."
            )
        
        if error_rate > self.thresholds['error_rate']['good']:
            recommendations.append(
                f"Error rate ({error_rate:.1%}) exceeds target. "
                "Investigate error patterns and add better error handling."
            )
        
        # Module-specific recommendations
        for module, metrics in module_metrics.items():
            if metrics['avg_time_ms'] > self.thresholds['response_time_ms']['acceptable']:
                recommendations.append(
                    f"Optimize {module}: Consider caching, parallel processing, "
                    f"or algorithm improvements (currently {metrics['avg_time_ms']:.0f}ms avg)"
                )
            
            if metrics['error_rate'] > 0.1:
                recommendations.append(
                    f"Fix {module}: Critical error rate {metrics['error_rate']:.1%} "
                    "indicates serious issues"
                )
        
        # Bottleneck-specific recommendations
        if len(bottlenecks) > 5:
            recommendations.append(
                "Multiple bottlenecks detected. Consider architectural review "
                "to improve system design."
            )
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _create_performance_summary(self, module_metrics: Dict[str, Dict[str, float]], 
                                  period_hours: int) -> str:
        """Create performance summary"""
        total_operations = sum(
            m['success_count'] + m['error_count'] 
            for m in module_metrics.values()
        )
        
        avg_response = self._calculate_system_average(module_metrics)
        error_rate = self._calculate_system_error_rate(module_metrics)
        
        # Determine overall status
        if error_rate > 0.1 or avg_response > 500:
            status = "Critical - Immediate attention required"
        elif error_rate > 0.05 or avg_response > 200:
            status = "Degraded - Performance issues detected"
        elif error_rate > 0.01 or avg_response > 100:
            status = "Fair - Room for improvement"
        else:
            status = "Good - System performing well"
        
        return f"""
Performance Summary ({period_hours} hours)
Status: {status}
Total Operations: {total_operations:,}
Active Modules: {len(module_metrics)}
Average Response Time: {avg_response:.1f}ms
System Error Rate: {error_rate:.2%}
"""
    
    def _calculate_system_average(self, module_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate system-wide average response time"""
        total_time = sum(
            m['avg_time_ms'] * (m['success_count'] + m['error_count'])
            for m in module_metrics.values()
        )
        total_ops = sum(
            m['success_count'] + m['error_count']
            for m in module_metrics.values()
        )
        
        return total_time / max(total_ops, 1)
    
    def _calculate_system_error_rate(self, module_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate system-wide error rate"""
        total_errors = sum(m['error_count'] for m in module_metrics.values())
        total_ops = sum(
            m['success_count'] + m['error_count']
            for m in module_metrics.values()
        )
        
        return total_errors / max(total_ops, 1)
    
    def _calculate_system_throughput(self, module_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate system-wide throughput"""
        return sum(m['throughput_per_min'] for m in module_metrics.values())
    
    def _format_detailed_findings(self, report: PerformanceReport) -> str:
        """Format detailed findings section"""
        lines = [
            "\n\nDETAILED FINDINGS:",
            "=" * 50
        ]
        
        # Top performers
        sorted_by_speed = sorted(
            report.module_metrics.items(),
            key=lambda x: x[1]['avg_time_ms']
        )
        
        if sorted_by_speed:
            lines.extend([
                "\nFastest Modules:",
                "-" * 20
            ])
            for module, metrics in sorted_by_speed[:3]:
                lines.append(
                    f"• {module}: {metrics['avg_time_ms']:.1f}ms average"
                )
        
        # Problem modules
        problem_modules = [
            (m, metrics) for m, metrics in report.module_metrics.items()
            if metrics['error_rate'] > 0.05 or metrics['avg_time_ms'] > 200
        ]
        
        if problem_modules:
            lines.extend([
                "\nModules Needing Attention:",
                "-" * 30
            ])
            for module, metrics in problem_modules:
                issues = []
                if metrics['error_rate'] > 0.05:
                    issues.append(f"{metrics['error_rate']:.1%} errors")
                if metrics['avg_time_ms'] > 200:
                    issues.append(f"{metrics['avg_time_ms']:.0f}ms avg response")
                lines.append(f"• {module}: {', '.join(issues)}")
        
        # Trends
        if report.trends:
            lines.extend([
                "\nPerformance Trends:",
                "-" * 20
            ])
            for module, trend in report.trends.items():
                trend_symbol = "↗" if trend == "improving" else "↘" if trend == "degrading" else "→"
                lines.append(f"• {module}: {trend} {trend_symbol}")
        
        return "\n".join(lines)
    
    def export_metrics(self, filepath: str, period_hours: int = 24):
        """Export performance metrics for analysis"""
        cutoff = time.time() - (period_hours * 3600)
        
        metrics_data = {
            'export_time': datetime.now().isoformat(),
            'period_hours': period_hours,
            'metrics': [
                m.to_dict() for m in self.metrics 
                if m.timestamp > cutoff
            ],
            'summary': self.generate_performance_report(period_hours).__dict__
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
    
    def get_realtime_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time performance dashboard"""
        # Last 5 minutes of data
        cutoff = time.time() - 300
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff]
        
        # Calculate real-time stats
        if recent_metrics:
            current_throughput = len(recent_metrics) / 5  # per minute
            current_error_rate = sum(1 for m in recent_metrics if not m.success) / len(recent_metrics)
            current_avg_time = np.mean([m.duration_ms for m in recent_metrics])
        else:
            current_throughput = 0
            current_error_rate = 0
            current_avg_time = 0
        
        # Module-specific current performance
        module_current = {}
        for module in self.module_metrics:
            module_recent = [m for m in self.module_metrics[module] if m.timestamp > cutoff]
            if module_recent:
                module_current[module] = {
                    'avg_time': np.mean([m.duration_ms for m in module_recent]),
                    'count': len(module_recent),
                    'errors': sum(1 for m in module_recent if not m.success)
                }
        
        return {
            'timestamp': time.time(),
            'current_throughput': current_throughput,
            'current_error_rate': current_error_rate,
            'current_avg_time': current_avg_time,
            'module_performance': module_current,
            'recent_errors': [
                {
                    'module': m.module,
                    'operation': m.operation,
                    'error': m.error,
                    'timestamp': m.timestamp
                }
                for m in recent_metrics 
                if not m.success
            ][-10:]  # Last 10 errors
        }