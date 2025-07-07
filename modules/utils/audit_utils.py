# ─────────────────────────────────────────────────────────────
# File: modules/utils/audit_utils.py
# 🚀 ENHANCED SmartInfoBus audit logging with plain English
# ─────────────────────────────────────────────────────────────

import os
import json
import logging
import datetime
from typing import Any, Dict, Optional, Union, List, Callable
from pathlib import Path
from collections import defaultdict, deque
from functools import wraps
import time
import traceback

import numpy as np
from modules.utils.info_bus import InfoBus, SmartInfoBus, InfoBusExtractor, InfoBusUpdater, InfoBusManager

# ═══════════════════════════════════════════════════════════════════
# PLAIN ENGLISH MESSAGE FORMATTING
# ═══════════════════════════════════════════════════════════════════

def format_operator_message(
    emoji: str, 
    action: str, 
    instrument: str = "", 
    details: str = "", 
    result: str = "", 
    context: str = "",
    risk_score: Optional[float] = None,
    thesis: Optional[str] = None
) -> str:
    """
    Format operator-centric messages with plain English clarity.
    Enhanced with thesis support for SmartInfoBus.
    """
    parts = [emoji, action.upper()]
    
    if instrument:
        parts.append(instrument)
        
    if details:
        parts.append(f"| {details}")
        
    if result:
        parts.append(f"| {result}")
        
    if context:
        parts.append(f"| {context}")
        
    # Add risk score if provided (0-1 range)
    if risk_score is not None:
        risk_level = "🟢 Low" if risk_score < 0.3 else "🟡 Med" if risk_score < 0.7 else "🔴 High"
        parts.append(f"| Risk: {risk_score:.2f} {risk_level}")
    
    # Add thesis summary if provided
    if thesis:
        # Extract first line of thesis for summary
        thesis_summary = thesis.split('\n')[0][:50] + "..."
        parts.append(f"| Reason: {thesis_summary}")
        
    return " ".join(parts)

def format_plain_english_report(title: str, sections: Dict[str, Any], 
                              recommendations: List[str] = None) -> str:
    """Format a plain English report for operators"""
    lines = [
        f"{'='*60}",
        f"{title.upper()}",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*60}",
        ""
    ]
    
    # Add sections
    for section_name, content in sections.items():
        lines.append(f"{section_name}:")
        lines.append("-" * len(section_name))
        
        if isinstance(content, dict):
            for key, value in content.items():
                lines.append(f"  • {key}: {value}")
        elif isinstance(content, list):
            for item in content:
                lines.append(f"  • {item}")
        else:
            lines.append(f"  {content}")
        lines.append("")
    
    # Add recommendations
    if recommendations:
        lines.extend([
            "RECOMMENDATIONS:",
            "---------------"
        ])
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
    
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════
# ENHANCED ROTATING LOGGER WITH SMARTINFOBUS
# ═══════════════════════════════════════════════════════════════════

class RotatingLogger:
    """
    Enhanced rotating logger with SmartInfoBus integration.
    Includes plain English formatting and thesis tracking.
    """
    
    def __init__(
        self, 
        name: str, 
        log_path: str, 
        max_lines: int = 2000,
        operator_mode: bool = False, 
        json_mode: bool = False,
        info_bus_aware: bool = True,
        plain_english: bool = True
    ):
        self.name = name
        self.log_path = Path(log_path)
        self.max_lines = max_lines
        self.operator_mode = operator_mode
        self.json_mode = json_mode
        self.info_bus_aware = info_bus_aware
        self.plain_english = plain_english
        
        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file
        if not self.log_path.exists():
            self.log_path.touch()
            
        # Setup logging
        self.logger = logging.getLogger(f"{name}_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG if operator_mode else logging.INFO)
        self.logger.propagate = False
        
        # Performance tracking
        self._log_count = 0
        self._rotation_count = 0
        self._thesis_count = 0
        
        self._setup_handler()
        
    def _setup_handler(self):
        """Setup handler with custom formatter"""
        self.logger.handlers.clear()
        
        # Console handler
        console = logging.StreamHandler()
        
        if self.operator_mode:
            # Operator-friendly format
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            # Technical format
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
        console.setFormatter(formatter)
        self.logger.addHandler(console)
    
    def _extract_context(self, info_bus: Optional[Union[InfoBus, SmartInfoBus]] = None) -> Dict[str, Any]:
        """Extract context from InfoBus or SmartInfoBus"""
        if not self.info_bus_aware or info_bus is None:
            return {}
        
        try:
            # Check if it's a SmartInfoBus reference
            if isinstance(info_bus, dict) and '_smart_bus' in info_bus:
                smart_bus = info_bus['_smart_bus']
                
                # Get latest module theses
                theses = []
                for key in ['risk_assessment', 'trading_summary', 'vote_consensus']:
                    data_with_thesis = smart_bus.get_with_thesis(key, self.name)
                    if data_with_thesis:
                        _, thesis = data_with_thesis
                        if thesis and thesis != "No thesis provided":
                            theses.append(thesis.split('\n')[0])  # First line
                
                return {
                    'step': info_bus.get('step_idx', 0),
                    'risk_score': InfoBusExtractor.get_risk_score(info_bus),
                    'regime': InfoBusExtractor.get_market_regime(info_bus),
                    'position_count': len(info_bus.get('positions', [])),
                    'has_theses': len(theses) > 0,
                    'thesis_summary': theses[0] if theses else None
                }
            else:
                # Legacy InfoBus
                return {
                    'step': info_bus.get('step_idx', 0),
                    'risk_score': InfoBusExtractor.get_risk_score(info_bus),
                    'regime': InfoBusExtractor.get_market_regime(info_bus),
                    'position_count': len(info_bus.get('positions', []))
                }
        except:
            return {}
    
    def log_with_thesis(self, level: str, message: str, thesis: str, 
                       info_bus: Optional[InfoBus] = None):
        """Log message with associated thesis"""
        self._thesis_count += 1
        
        if self.plain_english:
            # Format as plain English
            formatted = format_operator_message(
                "📝", level.upper(),
                details=message,
                thesis=thesis
            )
            self.logger.log(getattr(logging, level.upper()), formatted)
        else:
            self.logger.log(getattr(logging, level.upper()), f"{message} | Thesis: {thesis[:100]}...")
        
        # Write full thesis to file
        context = self._extract_context(info_bus)
        context['thesis'] = thesis
        self._write_to_file(message, level.upper(), context)
    
    def log_module_performance(self, module: str, metrics: Dict[str, float], 
                             recommendations: List[str] = None):
        """Log module performance in plain English"""
        if not self.plain_english:
            self.info(f"Module {module} performance: {metrics}")
            return
        
        # Create plain English report
        report = format_plain_english_report(
            f"{module} Performance Report",
            {
                "Metrics": {
                    f"Average execution time": f"{metrics.get('avg_time_ms', 0):.1f}ms",
                    f"Maximum execution time": f"{metrics.get('max_time_ms', 0):.1f}ms",
                    f"Success rate": f"{(1 - metrics.get('error_rate', 0)):.1%}",
                    f"Total calls": metrics.get('call_count', 0)
                },
                "Status": self._get_performance_status(metrics)
            },
            recommendations=recommendations
        )
        
        self.info(report)
    
    def _get_performance_status(self, metrics: Dict[str, float]) -> str:
        """Get plain English performance status"""
        avg_time = metrics.get('avg_time_ms', 0)
        error_rate = metrics.get('error_rate', 0)
        
        if error_rate > 0.1:
            return "⚠️ Module experiencing errors - investigation needed"
        elif avg_time > 100:
            return "🐌 Module running slowly - optimization recommended"
        elif avg_time > 50:
            return "🟡 Module performance acceptable but could be improved"
        else:
            return "🟢 Module performing well within parameters"
    
    # Original methods remain with enhancements
    def _rotate_if_needed(self):
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > self.max_lines:
                keep_lines = lines[-self.max_lines:]
                
                rotation_marker = f"\n--- LOG ROTATED at {datetime.datetime.now().isoformat()} ---\n"
                
                with open(self.log_path, 'w', encoding='utf-8') as f:
                    f.write(rotation_marker)
                    f.writelines(keep_lines)
                
                self._rotation_count += 1
                
        except Exception as e:
            print(f"Log rotation error: {e}")
    
    def _write_to_file(self, message: str, level: str = "INFO", 
                      context: Optional[Dict[str, Any]] = None):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        if self.json_mode:
            log_entry = {
                'timestamp': timestamp,
                'level': level,
                'logger': self.name,
                'message': message,
                'log_count': self._log_count,
                'thesis_count': self._thesis_count
            }
            
            if context:
                log_entry['context'] = context
            
            line = json.dumps(log_entry) + '\n'
        else:
            line = f"{timestamp} [{level}] {message}\n"
            if context and context.get('thesis'):
                line += f"    THESIS: {context['thesis']}\n"
            
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line)
        
        self._log_count += 1
        
        if self._log_count % 100 == 0:
            self._rotate_if_needed()
    
    # Standard logging methods remain the same
    def debug(self, message: str, info_bus: Optional[InfoBus] = None):
        self.logger.debug(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "DEBUG", context)
    
    def info(self, message: str, info_bus: Optional[InfoBus] = None):
        self.logger.info(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "INFO", context)
    
    def warning(self, message: str, info_bus: Optional[InfoBus] = None):
        self.logger.warning(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "WARNING", context)
    
    def error(self, message: str, info_bus: Optional[InfoBus] = None):
        self.logger.error(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "ERROR", context)
    
    def critical(self, message: str, info_bus: Optional[InfoBus] = None):
        self.logger.critical(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "CRITICAL", context)
    
    def exception(self, message: str, info_bus: Optional[InfoBus] = None):
        self.logger.exception(message)
        tb = traceback.format_exc()
        context = self._extract_context(info_bus)
        context['traceback'] = tb
        self._write_to_file(f"{message}\n{tb}", "EXCEPTION", context)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_logs': self._log_count,
            'rotations': self._rotation_count,
            'thesis_logs': self._thesis_count,
            'file_size': self.log_path.stat().st_size if self.log_path.exists() else 0,
            'max_lines': self.max_lines
        }

# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS AUDIT TRACKER
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusAuditTracker:
    """
    Enhanced audit tracker with SmartInfoBus thesis tracking.
    Provides plain English explanations of system behavior.
    """
    
    def __init__(self, system_name: str = "TradingSystem"):
        self.system_name = system_name
        self.events = deque(maxlen=1000)
        self.theses = deque(maxlen=500)
        
        # Setup specialized loggers
        self.audit_logger = RotatingLogger(
            name=f"{system_name}Audit",
            log_path=f"logs/audit/{system_name.lower()}_audit.jsonl",
            max_lines=2000,
            json_mode=True,
            info_bus_aware=True,
            plain_english=False
        )
        
        self.operator_logger = RotatingLogger(
            name=f"{system_name}Operator",
            log_path=f"logs/operator/{system_name.lower()}_operator.log",
            max_lines=2000,
            operator_mode=True,
            info_bus_aware=True,
            plain_english=True
        )
        
        # Performance tracking
        self.module_call_times = defaultdict(lambda: deque(maxlen=100))
        self.module_error_counts = defaultdict(int)
        self.module_thesis_counts = defaultdict(int)
        
        # SmartInfoBus reference
        self.smart_bus = InfoBusManager.get_instance()
    
    def record_module_decision(
        self, 
        module: str, 
        decision: str,
        thesis: str,
        confidence: float,
        info_bus: InfoBus,
        duration_ms: float = 0
    ):
        """Record module decision with thesis"""
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'module': module,
            'decision': decision,
            'thesis': thesis,
            'confidence': confidence,
            'duration_ms': duration_ms,
            'step': info_bus.get('step_idx', 0),
            'risk_score': InfoBusExtractor.get_risk_score(info_bus)
        }
        
        self.events.append(event)
        self.theses.append(thesis)
        self.module_thesis_counts[module] += 1
        
        # Log to audit
        self.audit_logger.log_with_thesis("info", f"{module} decision: {decision}", thesis, info_bus)
        
        # Store in SmartInfoBus
        self.smart_bus.set(
            f'audit_decision_{module}_{self.module_thesis_counts[module]}',
            event,
            module='AuditTracker',
            thesis=thesis,
            confidence=confidence
        )
        
        # Alert operator for important decisions
        if confidence > 0.9:
            self.operator_logger.info(
                format_operator_message(
                    "🎯", "HIGH CONFIDENCE DECISION",
                    instrument=module,
                    details=decision,
                    thesis=thesis,
                    risk_score=InfoBusExtractor.get_risk_score(info_bus)
                ),
                info_bus
            )
    
    def generate_plain_english_summary(self, info_bus: InfoBus) -> str:
        """Generate plain English summary of recent activity"""
        recent_events = list(self.events)[-20:]  # Last 20 events
        
        # Group by module
        module_activity = defaultdict(list)
        for event in recent_events:
            module_activity[event['module']].append(event)
        
        # Build summary
        sections = {}
        
        # Activity summary
        activity_lines = []
        for module, events in module_activity.items():
            avg_confidence = np.mean([e['confidence'] for e in events])
            activity_lines.append(
                f"{module}: {len(events)} decisions, "
                f"avg confidence {avg_confidence:.1%}"
            )
        sections["Recent Activity"] = activity_lines
        
        # Risk assessment
        current_risk = InfoBusExtractor.get_risk_score(info_bus)
        risk_trend = self._calculate_risk_trend()
        sections["Risk Status"] = [
            f"Current risk score: {current_risk:.1%}",
            f"Risk trend: {risk_trend}",
            f"Open positions: {len(info_bus.get('positions', []))}"
        ]
        
        # Performance metrics
        perf_metrics = self.get_module_performance()
        slow_modules = [
            m for m, stats in perf_metrics.items() 
            if stats['avg_time_ms'] > 50
        ]
        sections["Performance"] = [
            f"Slow modules: {', '.join(slow_modules) if slow_modules else 'None'}",
            f"Total decisions tracked: {sum(self.module_thesis_counts.values())}",
            f"Unique theses generated: {len(set(self.theses))}"
        ]
        
        # Recommendations
        recommendations = []
        if current_risk > 0.7:
            recommendations.append("Consider reducing exposure due to high risk")
        if slow_modules:
            recommendations.append(f"Investigate performance of: {', '.join(slow_modules)}")
        if risk_trend == "deteriorating":
            recommendations.append("Risk is increasing - monitor closely")
        
        return format_plain_english_report(
            "System Activity Summary",
            sections,
            recommendations
        )
    
    def _calculate_risk_trend(self) -> str:
        """Calculate risk trend from recent events"""
        recent_risks = [
            e['risk_score'] for e in list(self.events)[-50:]
            if 'risk_score' in e
        ]
        
        if len(recent_risks) < 10:
            return "insufficient data"
        
        # Compare first half to second half
        mid = len(recent_risks) // 2
        first_half = np.mean(recent_risks[:mid])
        second_half = np.mean(recent_risks[mid:])
        
        if second_half > first_half * 1.2:
            return "deteriorating"
        elif second_half < first_half * 0.8:
            return "improving"
        else:
            return "stable"
    
    def record_module_call(
        self, 
        module: str, 
        method: str, 
        info_bus: InfoBus,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record module method call (backward compatible)"""
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'module': module,
            'method': method,
            'duration_ms': duration_ms,
            'success': success,
            'error': error,
            'step': info_bus.get('step_idx', 0),
            'risk_score': InfoBusExtractor.get_risk_score(info_bus)
        }
        
        self.events.append(event)
        self.module_call_times[module].append(duration_ms)
        
        if not success:
            self.module_error_counts[module] += 1
            
        # Log to audit
        self.audit_logger.debug(f"{module}.{method} completed in {duration_ms:.1f}ms", info_bus)
        
        # Alert operator for slow operations
        if duration_ms > 100:
            self.operator_logger.warning(
                format_operator_message(
                    "⏱️", "SLOW OPERATION",
                    instrument=module,
                    details=f"{method} took {duration_ms:.0f}ms",
                    context="performance"
                ),
                info_bus
            )
        
        # Alert for errors
        if not success:
            self.operator_logger.error(
                format_operator_message(
                    "💥", "MODULE ERROR",
                    instrument=module,
                    details=f"{method} failed: {error}",
                    context="error"
                ),
                info_bus
            )
    
    def get_module_performance(self) -> Dict[str, Dict[str, float]]:
        """Get module performance statistics"""
        stats = {}
        
        for module, times in self.module_call_times.items():
            if times:
                stats[module] = {
                    'avg_time_ms': np.mean(times),
                    'max_time_ms': max(times),
                    'min_time_ms': min(times),
                    'call_count': len(times),
                    'error_count': self.module_error_counts[module],
                    'error_rate': self.module_error_counts[module] / len(times),
                    'thesis_count': self.module_thesis_counts[module]
                }
        
        return stats

# ═══════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════

# Legacy class names
InfoBusAuditTracker = SmartInfoBusAuditTracker
AuditTracker = SmartInfoBusAuditTracker

# Global audit tracker instance
system_audit = SmartInfoBusAuditTracker("TradingSystem")

# Audit decorators remain the same
def audit_module_call(tracker: Optional[SmartInfoBusAuditTracker] = None):
    """Decorator to audit module method calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, info_bus: InfoBus, *args, **kwargs):
            audit_tracker = tracker or getattr(self, '_audit_tracker', None)
            if not audit_tracker:
                return func(self, info_bus, *args, **kwargs)
            
            start_time = time.perf_counter()
            error = None
            success = True
            
            try:
                result = func(self, info_bus, *args, **kwargs)
                return result
                
            except Exception as e:
                error = str(e)
                success = False
                raise
                
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                audit_tracker.record_module_call(
                    module=self.__class__.__name__,
                    method=func.__name__,
                    info_bus=info_bus,
                    duration_ms=duration_ms,
                    success=success,
                    error=error
                )
        
        return wrapper
    return decorator

# Convenience functions remain the same
def audit_trade_execution(trade: Dict[str, Any], info_bus: InfoBus):
    system_audit.operator_logger.log_trade(trade, info_bus)

def audit_risk_event(event_type: str, current: float, limit: float, 
                    action: str, info_bus: InfoBus):
    system_audit.operator_logger.log_risk_event(
        event_type, current, limit, action, info_bus
    )

def get_performance_report(info_bus: InfoBus) -> str:
    return system_audit.generate_plain_english_summary(info_bus)