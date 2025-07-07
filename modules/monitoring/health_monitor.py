# ─────────────────────────────────────────────────────────────
# File: modules/monitoring/health_monitor.py
# 🚀 System health monitoring for SmartInfoBus
# ─────────────────────────────────────────────────────────────

import time
import numpy as np
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from collections import deque, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from modules.utils.info_bus import SmartInfoBus, InfoBusManager
from modules.utils.audit_utils import format_operator_message, RotatingLogger
from modules.core.module_orchestrator import ModuleOrchestrator


@dataclass
class HealthMetric:
    """Single health measurement"""
    timestamp: float
    metric_type: str
    value: float
    threshold: float
    status: str  # 'healthy', 'warning', 'critical'
    details: Optional[Dict[str, Any]] = None

@dataclass
class HealthReport:
    """Comprehensive health report"""
    timestamp: datetime
    overall_status: str
    system_metrics: Dict[str, float]
    module_health: Dict[str, str]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]


class HealthMonitor:
    """
    Monitors system health, module status, and resource usage.
    Provides early warning for potential issues.
    """
    
    def __init__(self, orchestrator: ModuleOrchestrator,
                 check_interval: int = 30):
        self.orchestrator = orchestrator
        self.smart_bus = InfoBusManager.get_instance()
        self.check_interval = check_interval
        
        # Health metrics storage
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # Thresholds
        self.thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 75, 'critical': 90},
            'disk_percent': {'warning': 80, 'critical': 95},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'latency_ms': {'warning': 150, 'critical': 300},
            'queue_size': {'warning': 1000, 'critical': 5000}
        }
        
        # Alert management
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Module health tracking
        self.module_health_scores: Dict[str, float] = {}
        self.unhealthy_modules: Set[str] = set()
        
        # Setup logging
        self.logger = RotatingLogger(
            name="HealthMonitor",
            log_path="logs/monitoring/health.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        
        # Start monitoring
        self.monitoring_active = True
        self._start_monitoring()
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_data = {
            'timestamp': time.time(),
            'system': self._check_system_resources(),
            'modules': self._check_module_health(),
            'infobus': self._check_infobus_health(),
            'performance': self._check_performance_health()
        }
        
        # Calculate overall status
        health_data['overall_status'] = self._calculate_overall_status(health_data)
        
        # Record metrics
        self._record_health_metrics(health_data)
        
        # Check for alerts
        self._check_for_alerts(health_data)
        
        return health_data
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'process_memory_mb': process_memory.rss / (1024**2),
                'network_sent_mb': net_io.bytes_sent / (1024**2),
                'network_recv_mb': net_io.bytes_recv / (1024**2),
                'open_files': len(process.open_files()),
                'thread_count': threading.active_count()
            }
        except Exception as e:
            self.logger.error(f"Failed to check system resources: {e}")
            return {}
    
    def _check_module_health(self) -> Dict[str, Any]:
        """Check health of all modules"""
        module_health = {}
        unhealthy_count = 0
        
        for module_name, module in self.orchestrator.modules.items():
            health_info = {
                'enabled': self.smart_bus.is_module_enabled(module_name),
                'failures': self.smart_bus._module_failures.get(module_name, 0),
                'status': 'unknown'
            }
            
            # Get module-specific health if available
            if hasattr(module, 'get_health_status'):
                try:
                    module_status = module.get_health_status()
                    health_info.update(module_status)
                except:
                    health_info['status'] = 'error'
            
            # Calculate health score
            if not health_info['enabled']:
                health_info['status'] = 'disabled'
                health_info['score'] = 0
            elif health_info['failures'] >= 3:
                health_info['status'] = 'critical'
                health_info['score'] = 0
            elif health_info['failures'] > 0:
                health_info['status'] = 'warning'
                health_info['score'] = 0.5
            else:
                health_info['status'] = health_info.get('status', 'healthy')
                health_info['score'] = 1.0
            
            # Check latency
            latencies = list(self.smart_bus._latency_history.get(module_name, []))
            if latencies:
                avg_latency = np.mean(latencies[-10:])
                health_info['avg_latency_ms'] = avg_latency
                
                if avg_latency > self.thresholds['latency_ms']['critical']:
                    health_info['status'] = 'critical'
                    health_info['score'] = min(health_info['score'], 0.3)
                elif avg_latency > self.thresholds['latency_ms']['warning']:
                    health_info['status'] = 'warning'
                    health_info['score'] = min(health_info['score'], 0.7)
            
            module_health[module_name] = health_info
            self.module_health_scores[module_name] = health_info['score']
            
            if health_info['status'] in ['critical', 'error', 'disabled']:
                unhealthy_count += 1
                self.unhealthy_modules.add(module_name)
            else:
                self.unhealthy_modules.discard(module_name)
        
        return {
            'total_modules': len(module_health),
            'healthy_modules': len(module_health) - unhealthy_count,
            'unhealthy_modules': unhealthy_count,
            'module_details': module_health
        }
    
    def _check_infobus_health(self) -> Dict[str, Any]:
        """Check SmartInfoBus health"""
        perf_metrics = self.smart_bus.get_performance_metrics()
        
        # Calculate queue sizes (approximation from event log)
        event_log_size = perf_metrics.get('total_events', 0)
        
        return {
            'cache_hit_rate': perf_metrics.get('cache_hit_rate', 0),
            'active_modules': perf_metrics.get('active_modules', 0),
            'disabled_modules': len(perf_metrics.get('disabled_modules', [])),
            'event_log_size': event_log_size,
            'data_keys': len(self.smart_bus._data_store),
            'status': self._assess_infobus_status(perf_metrics)
        }
    
    def _check_performance_health(self) -> Dict[str, Any]:
        """Check overall performance health"""
        # Get recent performance data
        recent_latencies = []
        recent_errors = 0
        
        for module_name in self.orchestrator.modules:
            # Latencies
            latencies = list(self.smart_bus._latency_history.get(module_name, []))
            if latencies:
                recent_latencies.extend(latencies[-10:])
            
            # Errors
            failures = self.smart_bus._module_failures.get(module_name, 0)
            recent_errors += failures
        
        total_executions = len(recent_latencies)
        
        return {
            'avg_latency_ms': np.mean(recent_latencies) if recent_latencies else 0,
            'max_latency_ms': max(recent_latencies) if recent_latencies else 0,
            'p95_latency_ms': np.percentile(recent_latencies, 95) if recent_latencies else 0,
            'error_rate': recent_errors / max(total_executions, 1),
            'throughput_per_min': total_executions * 2  # Rough estimate
        }
    
    def _calculate_overall_status(self, health_data: Dict[str, Any]) -> str:
        """Calculate overall system health status"""
        statuses = []
        
        # System resources
        system = health_data.get('system', {})
        if system.get('cpu_percent', 0) > self.thresholds['cpu_percent']['critical']:
            statuses.append('critical')
        elif system.get('cpu_percent', 0) > self.thresholds['cpu_percent']['warning']:
            statuses.append('warning')
        
        if system.get('memory_percent', 0) > self.thresholds['memory_percent']['critical']:
            statuses.append('critical')
        elif system.get('memory_percent', 0) > self.thresholds['memory_percent']['warning']:
            statuses.append('warning')
        
        # Module health
        modules = health_data.get('modules', {})
        unhealthy_ratio = modules.get('unhealthy_modules', 0) / max(modules.get('total_modules', 1), 1)
        
        if unhealthy_ratio > 0.3:
            statuses.append('critical')
        elif unhealthy_ratio > 0.1:
            statuses.append('warning')
        
        # Performance
        perf = health_data.get('performance', {})
        if perf.get('error_rate', 0) > self.thresholds['error_rate']['critical']:
            statuses.append('critical')
        elif perf.get('error_rate', 0) > self.thresholds['error_rate']['warning']:
            statuses.append('warning')
        
        # Determine overall status
        if 'critical' in statuses:
            return 'critical'
        elif 'warning' in statuses:
            return 'warning'
        else:
            return 'healthy'
    
    def _assess_infobus_status(self, metrics: Dict[str, Any]) -> str:
        """Assess InfoBus health status"""
        if metrics.get('cache_hit_rate', 0) < 0.5:
            return 'warning'
        
        if len(metrics.get('disabled_modules', [])) > 3:
            return 'critical'
        
        return 'healthy'
    
    def _record_health_metrics(self, health_data: Dict[str, Any]):
        """Record health metrics for trending"""
        timestamp = health_data['timestamp']
        
        # System metrics
        system = health_data.get('system', {})
        for metric_name, value in system.items():
            if isinstance(value, (int, float)):
                self.metrics[f'system.{metric_name}'].append(
                    HealthMetric(
                        timestamp=timestamp,
                        metric_type=metric_name,
                        value=value,
                        threshold=self.thresholds.get(metric_name, {}).get('critical', float('inf')),
                        status=self._get_metric_status(metric_name, value)
                    )
                )
        
        # Module health scores
        for module_name, score in self.module_health_scores.items():
            self.metrics[f'module.{module_name}.score'].append(
                HealthMetric(
                    timestamp=timestamp,
                    metric_type='health_score',
                    value=score,
                    threshold=0.5,
                    status='healthy' if score > 0.7 else 'warning' if score > 0.3 else 'critical'
                )
            )
    
    def _get_metric_status(self, metric_name: str, value: float) -> str:
        """Get status for a metric value"""
        if metric_name not in self.thresholds:
            return 'healthy'
        
        thresholds = self.thresholds[metric_name]
        
        if value >= thresholds.get('critical', float('inf')):
            return 'critical'
        elif value >= thresholds.get('warning', float('inf')):
            return 'warning'
        else:
            return 'healthy'
    
    def _check_for_alerts(self, health_data: Dict[str, Any]):
        """Check for alert conditions"""
        alerts_to_trigger = []
        
        # System resource alerts
        system = health_data.get('system', {})
        for metric, value in system.items():
            if metric in self.thresholds:
                status = self._get_metric_status(metric, value)
                if status != 'healthy':
                    alert_key = f'system.{metric}'
                    
                    if alert_key not in self.active_alerts:
                        alert = {
                            'type': 'system_resource',
                            'metric': metric,
                            'value': value,
                            'threshold': self.thresholds[metric][status],
                            'status': status,
                            'timestamp': time.time()
                        }
                        self.active_alerts[alert_key] = alert
                        alerts_to_trigger.append(alert)
        
        # Module health alerts
        for module_name, health_info in health_data.get('modules', {}).get('module_details', {}).items():
            if health_info['status'] in ['critical', 'error']:
                alert_key = f'module.{module_name}'
                
                if alert_key not in self.active_alerts:
                    alert = {
                        'type': 'module_health',
                        'module': module_name,
                        'status': health_info['status'],
                        'failures': health_info.get('failures', 0),
                        'timestamp': time.time()
                    }
                    self.active_alerts[alert_key] = alert
                    alerts_to_trigger.append(alert)
        
        # Trigger alerts
        for alert in alerts_to_trigger:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger an alert"""
        # Add to history
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(
            format_operator_message(
                "🚨", "HEALTH ALERT",
                instrument=alert.get('metric') or alert.get('module', ''),
                details=f"{alert['type']}: {alert.get('status', 'unknown')}",
                context="health_monitoring"
            )
        )
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for health alerts"""
        self.alert_callbacks.append(callback)
    
    def generate_health_report(self) -> HealthReport:
        """Generate comprehensive health report"""
        health_data = self.check_system_health()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(health_data)
        
        # Convert alerts to list
        alerts = list(self.active_alerts.values())
        
        return HealthReport(
            timestamp=datetime.now(),
            overall_status=health_data['overall_status'],
            system_metrics=health_data.get('system', {}),
            module_health={
                m: info['status'] 
                for m, info in health_data.get('modules', {}).get('module_details', {}).items()
            },
            alerts=alerts,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, health_data: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        # System resource recommendations
        system = health_data.get('system', {})
        
        if system.get('cpu_percent', 0) > self.thresholds['cpu_percent']['warning']:
            recommendations.append(
                f"High CPU usage ({system['cpu_percent']:.1f}%) - consider optimizing compute-intensive modules"
            )
        
        if system.get('memory_percent', 0) > self.thresholds['memory_percent']['warning']:
            recommendations.append(
                f"High memory usage ({system['memory_percent']:.1f}%) - check for memory leaks"
            )
        
        # Module recommendations
        if self.unhealthy_modules:
            recommendations.append(
                f"Unhealthy modules detected: {', '.join(list(self.unhealthy_modules)[:5])}"
            )
            recommendations.append("Consider restarting or investigating these modules")
        
        # Performance recommendations
        perf = health_data.get('performance', {})
        if perf.get('avg_latency_ms', 0) > 100:
            recommendations.append(
                f"High average latency ({perf['avg_latency_ms']:.0f}ms) - review module performance"
            )
        
        if perf.get('error_rate', 0) > 0.05:
            recommendations.append(
                f"High error rate ({perf['error_rate']:.1%}) - investigate failing modules"
            )
        
        # InfoBus recommendations
        infobus = health_data.get('infobus', {})
        if infobus.get('cache_hit_rate', 1) < 0.7:
            recommendations.append(
                "Low cache hit rate - consider increasing cache size or TTL"
            )
        
        return recommendations
    
    def get_health_trends(self, metric_name: str, 
                         hours: int = 24) -> Dict[str, Any]:
        """Get health metric trends"""
        cutoff = time.time() - (hours * 3600)
        
        if metric_name not in self.metrics:
            return {'error': 'Metric not found'}
        
        metrics = [m for m in self.metrics[metric_name] if m.timestamp > cutoff]
        
        if not metrics:
            return {'error': 'No data in time range'}
        
        values = [m.value for m in metrics]
        
        return {
            'metric': metric_name,
            'period_hours': hours,
            'data_points': len(values),
            'current': values[-1] if values else 0,
            'average': np.mean(values),
            'minimum': min(values),
            'maximum': max(values),
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Compare first third to last third
        third = len(values) // 3
        first_avg = np.mean(values[:third])
        last_avg = np.mean(values[-third:])
        
        change = (last_avg - first_avg) / max(abs(first_avg), 1) * 100
        
        if change > 10:
            return 'increasing'
        elif change < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def export_health_data(self, filepath: str):
        """Export health data for analysis"""
        data = {
            'export_time': datetime.now().isoformat(),
            'current_health': self.check_system_health(),
            'active_alerts': list(self.active_alerts.values()),
            'alert_history': list(self.alert_history)[-100:],  # Last 100
            'module_scores': self.module_health_scores,
            'recommendations': self._generate_recommendations(
                self.check_system_health()
            )
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _start_monitoring(self):
        """Start background health monitoring"""
        def monitor():
            while self.monitoring_active:
                try:
                    self.check_system_health()
                except Exception as e:
                    self.logger.error(f"Health check error: {e}")
                
                time.sleep(self.check_interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False