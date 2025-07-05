# ─────────────────────────────────────────────────────────────
# File: modules/utils/audit_utils.py
# 🚀 ENHANCED InfoBus-centric audit logging with strict rotation
# ─────────────────────────────────────────────────────────────

import os
import json
import logging
import datetime
from typing import Any, Dict, Optional, Union, List, Callable
from pathlib import Path
from collections import deque
from functools import wraps
import time
import traceback
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater

# ═══════════════════════════════════════════════════════════════════
# OPERATOR-CENTRIC MESSAGE FORMATTING
# ═══════════════════════════════════════════════════════════════════

def format_operator_message(
    emoji: str, 
    action: str, 
    instrument: str = "", 
    details: str = "", 
    result: str = "", 
    context: str = "",
    risk_score: Optional[float] = None
) -> str:
    """
    Format operator-centric messages for maximum clarity.
    
    Example: "📈 TRADE EUR/USD | BUY 0.5 lots @ 1.0850 | P&L: +$125.50 | Risk: 0.45"
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
        
    return " ".join(parts)

def format_trade_message(trade: Dict[str, Any], risk_score: float = 0.0) -> str:
    """Format trade execution message"""
    symbol = trade.get('symbol', 'UNKNOWN')
    side = trade.get('side', 'buy').upper()
    size = trade.get('size', 0)
    price = trade.get('price', 0)
    pnl = trade.get('pnl', 0)
    
    emoji = "📈" if side == "BUY" else "📉"
    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    
    return format_operator_message(
        emoji=emoji,
        action="TRADE",
        instrument=symbol,
        details=f"{side} {size:.2f} @ {price:.5f}",
        result=f"P&L: {pnl_str}",
        risk_score=risk_score
    )

def format_risk_alert(alert_type: str, current_value: float, 
                     limit: float, action: str = "blocked") -> str:
    """Format risk alert message"""
    percentage = (current_value / limit * 100) if limit > 0 else 0
    emoji = "🚨" if percentage > 100 else "⚠️"
    
    return format_operator_message(
        emoji=emoji,
        action=f"RISK {alert_type.upper()}",
        details=f"{current_value:.2f}/{limit:.2f} ({percentage:.0f}%)",
        result=f"Action: {action}",
        context="risk_management"
    )

# ═══════════════════════════════════════════════════════════════════
# ENHANCED ROTATING LOGGER
# ═══════════════════════════════════════════════════════════════════

class RotatingLogger:
    """
    Enhanced rotating logger with InfoBus integration.
    Enforces 2000-line limit with automatic rotation.
    """
    
    def __init__(
        self, 
        name: str, 
        log_path: str, 
        max_lines: int = 2000,
        operator_mode: bool = False, 
        json_mode: bool = False,
        info_bus_aware: bool = True
    ):
        self.name = name
        self.log_path = Path(log_path)
        self.max_lines = max_lines
        self.operator_mode = operator_mode
        self.json_mode = json_mode
        self.info_bus_aware = info_bus_aware
        
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
        
    def _rotate_if_needed(self):
        """Enforce line limit by keeping only recent entries"""
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > self.max_lines:
                # Keep most recent lines
                keep_lines = lines[-self.max_lines:]
                
                # Add rotation marker
                rotation_marker = f"\n--- LOG ROTATED at {datetime.datetime.now().isoformat()} ---\n"
                
                with open(self.log_path, 'w', encoding='utf-8') as f:
                    f.write(rotation_marker)
                    f.writelines(keep_lines)
                
                self._rotation_count += 1
                
        except Exception as e:
            # Don't break logging if rotation fails
            print(f"Log rotation error: {e}")
    
    def _write_to_file(self, message: str, level: str = "INFO", 
                      context: Optional[Dict[str, Any]] = None):
        """Write message to file with context"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        if self.json_mode:
            # Structured JSON logging
            log_entry = {
                'timestamp': timestamp,
                'level': level,
                'logger': self.name,
                'message': message,
                'log_count': self._log_count,
            }
            
            # Add InfoBus context if available
            if context:
                log_entry['context'] = context
            
            line = json.dumps(log_entry) + '\n'
        else:
            # Standard text format
            line = f"{timestamp} [{level}] {message}\n"
            
        # Write to file
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line)
        
        self._log_count += 1
        
        # Rotate if needed
        if self._log_count % 100 == 0:  # Check every 100 logs
            self._rotate_if_needed()
    
    def _extract_context(self, info_bus: Optional[InfoBus] = None) -> Dict[str, Any]:
        """Extract context from InfoBus if available"""
        if not self.info_bus_aware or info_bus is None:
            return {}
        
        try:
            return {
                'step': info_bus.get('step_idx', 0),
                'risk_score': InfoBusExtractor.get_risk_score(info_bus),
                'regime': InfoBusExtractor.get_market_regime(info_bus),
                'position_count': len(info_bus.get('positions', []))
            }
        except:
            return {}
    
    # Enhanced logging methods with InfoBus awareness
    
    def debug(self, message: str, info_bus: Optional[InfoBus] = None):
        """Log debug message with optional InfoBus context"""
        self.logger.debug(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "DEBUG", context)
    
    def info(self, message: str, info_bus: Optional[InfoBus] = None):
        """Log info message with optional InfoBus context"""
        self.logger.info(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "INFO", context)
    
    def warning(self, message: str, info_bus: Optional[InfoBus] = None):
        """Log warning message with optional InfoBus context"""
        self.logger.warning(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "WARNING", context)
    
    def error(self, message: str, info_bus: Optional[InfoBus] = None):
        """Log error message with optional InfoBus context"""
        self.logger.error(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "ERROR", context)
    
    def critical(self, message: str, info_bus: Optional[InfoBus] = None):
        """Log critical message with optional InfoBus context"""
        self.logger.critical(message)
        context = self._extract_context(info_bus)
        self._write_to_file(message, "CRITICAL", context)
    
    def exception(self, message: str, info_bus: Optional[InfoBus] = None):
        """Log exception with traceback"""
        self.logger.exception(message)
        tb = traceback.format_exc()
        context = self._extract_context(info_bus)
        context['traceback'] = tb
        self._write_to_file(f"{message}\n{tb}", "EXCEPTION", context)
    
    def log_trade(self, trade: Dict[str, Any], info_bus: Optional[InfoBus] = None):
        """Log trade execution with full context"""
        risk_score = InfoBusExtractor.get_risk_score(info_bus) if info_bus else 0.0
        message = format_trade_message(trade, risk_score)
        self.info(message, info_bus)
    
    def log_risk_event(self, event_type: str, value: float, limit: float, 
                      action: str, info_bus: Optional[InfoBus] = None):
        """Log risk event with context"""
        message = format_risk_alert(event_type, value, limit, action)
        
        if value > limit:
            self.warning(message, info_bus)
        else:
            self.info(message, info_bus)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        return {
            'total_logs': self._log_count,
            'rotations': self._rotation_count,
            'file_size': self.log_path.stat().st_size if self.log_path.exists() else 0,
            'max_lines': self.max_lines
        }

# ═══════════════════════════════════════════════════════════════════
# INFOBUS AUDIT TRACKER
# ═══════════════════════════════════════════════════════════════════

class InfoBusAuditTracker:
    """
    Enhanced audit tracker with full InfoBus integration.
    Tracks all module interactions through InfoBus.
    """
    
    def __init__(self, system_name: str = "TradingSystem"):
        self.system_name = system_name
        self.events = deque(maxlen=1000)
        
        # Setup specialized loggers
        self.audit_logger = RotatingLogger(
            name=f"{system_name}Audit",
            log_path=f"logs/audit/{system_name.lower()}_audit.jsonl",
            max_lines=2000,
            json_mode=True,
            info_bus_aware=True
        )
        
        self.operator_logger = RotatingLogger(
            name=f"{system_name}Operator",
            log_path=f"logs/operator/{system_name.lower()}_operator.log",
            max_lines=2000,
            operator_mode=True,
            info_bus_aware=True
        )
        
        # Performance tracking
        self.module_call_times = defaultdict(list)
        self.module_error_counts = defaultdict(int)
        
    def record_module_call(
        self, 
        module: str, 
        method: str, 
        info_bus: InfoBus,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record module method call with InfoBus context"""
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
        self.audit_logger.log(event, info_bus=info_bus)
        
        # Alert operator for slow operations
        if duration_ms > 100:  # 100ms threshold
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
    
    def record_info_bus_update(
        self,
        field: str,
        module: str,
        info_bus: InfoBus,
        old_value: Any = None,
        new_value: Any = None
    ):
        """Record InfoBus field update"""
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': 'info_bus_update',
            'field': field,
            'module': module,
            'step': info_bus.get('step_idx', 0),
            'has_old_value': old_value is not None,
            'has_new_value': new_value is not None
        }
        
        self.audit_logger.log(event, info_bus=info_bus)
    
    def get_module_performance(self) -> Dict[str, Dict[str, float]]:
        """Get module performance statistics"""
        stats = {}
        
        for module, times in self.module_call_times.items():
            if times:
                stats[module] = {
                    'avg_time_ms': sum(times) / len(times),
                    'max_time_ms': max(times),
                    'min_time_ms': min(times),
                    'call_count': len(times),
                    'error_count': self.module_error_counts[module],
                    'error_rate': self.module_error_counts[module] / len(times)
                }
        
        return stats
    
    def generate_performance_report(self, info_bus: InfoBus) -> str:
        """Generate operator-friendly performance report"""
        perf = self.get_module_performance()
        
        report_lines = [
            "📊 MODULE PERFORMANCE REPORT",
            "=" * 50
        ]
        
        # Sort by average time
        sorted_modules = sorted(
            perf.items(), 
            key=lambda x: x[1]['avg_time_ms'], 
            reverse=True
        )
        
        for module, stats in sorted_modules[:10]:  # Top 10 slowest
            status = "🔴" if stats['error_rate'] > 0.1 else "🟡" if stats['avg_time_ms'] > 50 else "🟢"
            
            report_lines.append(
                f"{status} {module}: "
                f"avg={stats['avg_time_ms']:.1f}ms, "
                f"calls={stats['call_count']}, "
                f"errors={stats['error_rate']:.1%}"
            )
        
        # Add risk context
        risk_score = InfoBusExtractor.get_risk_score(info_bus)
        risk_emoji = "🟢" if risk_score < 0.3 else "🟡" if risk_score < 0.7 else "🔴"
        
        report_lines.extend([
            "",
            f"{risk_emoji} Current Risk Score: {risk_score:.2f}",
            f"📍 Step: {info_bus.get('step_idx', 0)}"
        ])
        
        return "\n".join(report_lines)

# ═══════════════════════════════════════════════════════════════════
# AUDIT DECORATORS
# ═══════════════════════════════════════════════════════════════════

def audit_module_call(tracker: Optional[InfoBusAuditTracker] = None):
    """Decorator to audit module method calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, info_bus: InfoBus, *args, **kwargs):
            # Get tracker
            audit_tracker = tracker or getattr(self, '_audit_tracker', None)
            if not audit_tracker:
                return func(self, info_bus, *args, **kwargs)
            
            # Time the call
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
                # Record the call
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

def audit_info_bus_update(field: str):
    """Decorator to audit InfoBus updates"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, info_bus: InfoBus, *args, **kwargs):
            # Get old value
            old_value = info_bus.get(field)
            
            # Execute update
            result = func(self, info_bus, *args, **kwargs)
            
            # Get new value
            new_value = info_bus.get(field)
            
            # Record if changed
            if old_value != new_value and hasattr(self, '_audit_tracker'):
                self._audit_tracker.record_info_bus_update(
                    field=field,
                    module=self.__class__.__name__,
                    info_bus=info_bus,
                    old_value=old_value,
                    new_value=new_value
                )
            
            return result
        
        return wrapper
    return decorator

# ═══════════════════════════════════════════════════════════════════
# GLOBAL AUDIT SYSTEM
# ═══════════════════════════════════════════════════════════════════

# Global audit tracker instance
system_audit = InfoBusAuditTracker("TradingSystem")

# Convenience functions
def audit_trade_execution(trade: Dict[str, Any], info_bus: InfoBus):
    """Audit trade execution with InfoBus context"""
    system_audit.operator_logger.log_trade(trade, info_bus)

def audit_risk_event(event_type: str, current: float, limit: float, 
                    action: str, info_bus: InfoBus):
    """Audit risk event with InfoBus context"""
    system_audit.operator_logger.log_risk_event(
        event_type, current, limit, action, info_bus
    )

def get_performance_report(info_bus: InfoBus) -> str:
    """Get system performance report"""
    return system_audit.generate_performance_report(info_bus)