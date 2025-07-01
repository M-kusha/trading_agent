# ─────────────────────────────────────────────────────────────
# File: modules/utils/audit_utils.py
# Central utilities for audit logging with mandatory 2000-line rotation
# ─────────────────────────────────────────────────────────────

import os
import json
import logging
import datetime
from typing import Any, Dict, Optional, Union
from pathlib import Path
from collections import deque


class RotatingLogger:
    """
    Enforced log rotation - prevents logs from exceeding 2000 lines.
    Overwrites oldest entries automatically to prevent disk clogging.
    """
    
    def __init__(self, name: str, log_path: str, max_lines: int = 2000, 
                 operator_mode: bool = False, json_mode: bool = False):
        self.name = name
        self.log_path = Path(log_path)
        self.max_lines = max_lines
        self.operator_mode = operator_mode
        self.json_mode = json_mode
        
        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if not exists
        if not self.log_path.exists():
            self.log_path.touch()
            
        # Setup logging
        self.logger = logging.getLogger(f"{name}_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Create custom handler
        self._setup_handler()
        
    def _setup_handler(self):
        """Setup handler with custom formatter"""
        handler = logging.StreamHandler()
        
        if self.operator_mode:
            # Human-readable format for operators
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            # Standard format for technical logs
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def _rotate_if_needed(self):
        """Enforce line limit by deleting oldest entries"""
        if not self.log_path.exists():
            return
            
        # Read all lines
        with open(self.log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # If over limit, keep only the most recent max_lines
        if len(lines) > self.max_lines:
            keep_lines = lines[-self.max_lines:]
            
            # Overwrite file with recent lines only
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.writelines(keep_lines)
                
    def _write_to_file(self, message: str, level: str = "INFO"):
        """Write message directly to file with rotation"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if self.json_mode:
            # JSON format for structured logging
            log_entry = {
                'timestamp': timestamp,
                'level': level,
                'message': message if isinstance(message, str) else json.dumps(message, default=str)
            }
            line = json.dumps(log_entry) + '\n'
        else:
            # Standard text format
            line = f"{timestamp} [{level}] {message}\n"
            
        # Append to file
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line)
            
        # Rotate if needed
        self._rotate_if_needed()
        
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
        self._write_to_file(message, "INFO")
        
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
        self._write_to_file(message, "WARNING")
        
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
        self._write_to_file(message, "ERROR")
        
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
        self._write_to_file(message, "CRITICAL")
        
    def log(self, data: Union[str, Dict[str, Any]], level: str = "INFO"):
        """Log structured data (for JSON mode)"""
        if self.json_mode and isinstance(data, dict):
            self._write_to_file(data, level)
        else:
            self._write_to_file(str(data), level)


def format_operator_message(emoji: str, action: str, instrument: str = "", 
                           details: str = "", result: str = "", context: str = "") -> str:
    """
    Format operator-centric messages for maximum clarity and actionability.
    
    Example: "📈 TRADE BUY EUR/USD 1.25 @ 1.0850 | P&L: $125.50 | Regime: trending"
    """
    parts = [emoji, action]
    
    if instrument:
        parts.append(instrument)
        
    if details:
        parts.append(details)
        
    if result:
        parts.append(f"| {result}")
        
    if context:
        parts.append(f"| {context}")
        
    return " ".join(parts)


def create_audit_summary(title: str, data: Dict[str, Any], 
                        highlight_keys: Optional[list] = None) -> str:
    """
    Create human-readable audit summary for operators.
    Highlights key metrics and provides actionable insights.
    """
    summary = f"\n{'='*50}\n{title.upper()}\n{'='*50}\n"
    
    if highlight_keys:
        summary += "🎯 KEY METRICS:\n"
        for key in highlight_keys:
            if key in data:
                value = data[key]
                if isinstance(value, float):
                    if 'pnl' in key.lower() or 'profit' in key.lower():
                        summary += f"• {key.replace('_', ' ').title()}: ${value:.2f}\n"
                    elif 'rate' in key.lower() or 'ratio' in key.lower():
                        summary += f"• {key.replace('_', ' ').title()}: {value:.1%}\n"
                    else:
                        summary += f"• {key.replace('_', ' ').title()}: {value:.2f}\n"
                else:
                    summary += f"• {key.replace('_', ' ').title()}: {value}\n"
        summary += "\n"
    
    # Add remaining data
    summary += "📊 DETAILED DATA:\n"
    for key, value in data.items():
        if highlight_keys and key in highlight_keys:
            continue
            
        if isinstance(value, dict):
            summary += f"• {key.replace('_', ' ').title()}:\n"
            for sub_key, sub_value in value.items():
                summary += f"  - {sub_key}: {sub_value}\n"
        elif isinstance(value, list):
            summary += f"• {key.replace('_', ' ').title()}: {len(value)} items\n"
        else:
            summary += f"• {key.replace('_', ' ').title()}: {value}\n"
    
    return summary


class AuditTracker:
    """
    Centralized audit tracker for module interactions and system events.
    Maintains audit trail with automatic rotation and operator alerts.
    """
    
    def __init__(self, system_name: str = "TradingSystem"):
        self.system_name = system_name
        self.events = deque(maxlen=1000)  # Keep last 1000 events in memory
        
        # Setup audit loggers
        self.audit_logger = RotatingLogger(
            name=f"{system_name}Audit",
            log_path=f"logs/audit/{system_name.lower()}_audit.jsonl",
            max_lines=2000,
            json_mode=True
        )
        
        self.operator_logger = RotatingLogger(
            name=f"{system_name}Operator",
            log_path=f"logs/audit/{system_name.lower()}_operator.log",
            max_lines=2000,
            operator_mode=True
        )
        
    def record_event(self, event_type: str, module: str, details: Dict[str, Any], 
                    severity: str = "info"):
        """Record system event with full context"""
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': event_type,
            'module': module,
            'severity': severity,
            'details': details,
            'system': self.system_name
        }
        
        self.events.append(event)
        self.audit_logger.log(event)
        
        # Generate operator alert for important events
        if severity in ['warning', 'error', 'critical']:
            self._generate_operator_alert(event)
            
    def _generate_operator_alert(self, event: Dict[str, Any]):
        """Generate operator-friendly alert"""
        severity_emoji = {
            'warning': '⚠️',
            'error': '❌', 
            'critical': '🚨'
        }
        
        emoji = severity_emoji.get(event['severity'], 'ℹ️')
        message = format_operator_message(
            emoji=emoji,
            action=f"{event['event_type']} - {event['module']}",
            details=str(event['details'].get('summary', '')),
            context=f"Severity: {event['severity']}"
        )
        
        if event['severity'] == 'critical':
            self.operator_logger.critical(message)
        elif event['severity'] == 'error':
            self.operator_logger.error(message)
        else:
            self.operator_logger.warning(message)
            
    def get_recent_events(self, count: int = 10, severity_filter: Optional[str] = None) -> list:
        """Get recent events, optionally filtered by severity"""
        events = list(self.events)
        
        if severity_filter:
            events = [e for e in events if e['severity'] == severity_filter]
            
        return events[-count:]
        
    def get_event_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of events in the last N hours"""
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=hours)
        
        recent_events = [
            e for e in self.events 
            if datetime.datetime.fromisoformat(e['timestamp']) > cutoff
        ]
        
        summary = {
            'total_events': len(recent_events),
            'by_severity': {},
            'by_module': {},
            'by_type': {}
        }
        
        for event in recent_events:
            # Count by severity
            severity = event['severity']
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Count by module
            module = event['module']
            summary['by_module'][module] = summary['by_module'].get(module, 0) + 1
            
            # Count by type
            event_type = event['event_type']
            summary['by_type'][event_type] = summary['by_type'].get(event_type, 0) + 1
            
        return summary


# Global audit tracker instance
system_audit = AuditTracker("TradingSystem")


def audit_module_interaction(from_module: str, to_module: str, interaction_type: str, 
                           data: Optional[Dict[str, Any]] = None):
    """Audit inter-module communication"""
    system_audit.record_event(
        event_type="module_interaction",
        module=f"{from_module}->{to_module}",
        details={
            'interaction_type': interaction_type,
            'data_size': len(str(data)) if data else 0,
            'summary': f"{from_module} {interaction_type} {to_module}"
        }
    )


def audit_trade_execution(symbol: str, side: str, size: float, price: float, 
                         result: str, pnl: float):
    """Audit trade execution with full context"""
    system_audit.record_event(
        event_type="trade_execution",
        module="TradingEngine",
        details={
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'result': result,
            'pnl': pnl,
            'summary': f"{side} {size} {symbol} @ {price} -> {result}"
        },
        severity='warning' if result == 'failed' else 'info'
    )


def audit_risk_event(risk_type: str, current_value: float, limit: float, 
                    action_taken: str):
    """Audit risk management events"""
    severity = 'critical' if current_value > limit else 'warning'
    
    system_audit.record_event(
        event_type="risk_event",
        module="RiskManager",
        details={
            'risk_type': risk_type,
            'current_value': current_value,
            'limit': limit,
            'breach_ratio': current_value / limit if limit > 0 else 0,
            'action_taken': action_taken,
            'summary': f"{risk_type}: {current_value:.2f}/{limit:.2f} - {action_taken}"
        },
        severity=severity
    )


def audit_system_health(module: str, health_score: float, issues: list):
    """Audit system health checks"""
    severity = 'critical' if health_score < 0.5 else ('warning' if health_score < 0.8 else 'info')
    
    system_audit.record_event(
        event_type="health_check",
        module=module,
        details={
            'health_score': health_score,
            'issues': issues,
            'issue_count': len(issues),
            'summary': f"Health: {health_score:.1%} - {len(issues)} issues"
        },
        severity=severity
    )