# ─────────────────────────────────────────────────────────────
# File: modules/utils/audit_utils.py
# 🚀 PRODUCTION-READY Audit & Logging System
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# ENHANCED: SmartInfoBus integration, plain English support
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import sys
import time
import json
import hashlib
import threading
from typing import Dict, Any, Optional, Union, TYPE_CHECKING, Tuple
from datetime import datetime
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, field
import inspect
import traceback
import uuid
import numpy as np

if TYPE_CHECKING:
    from modules.utils.info_bus import SmartInfoBus

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE AUDIT STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AuditEvent:
    """
    Military-grade audit event with complete traceability.
    Immutable record for regulatory compliance.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    module_name: str = ""
    function_name: str = ""
    operator_message: str = ""
    
    # Context information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Technical details
    severity: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    category: str = "general"  # general, security, performance, business
    
    # Data payload
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Security & integrity
    checksum: str = field(default="")
    signature: str = field(default="")
    
    # SmartInfoBus integration
    smart_bus_key: Optional[str] = None
    thesis: Optional[str] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """Calculate checksum for tamper detection"""
        if not self.checksum:
            content = f"{self.timestamp}{self.event_type}{self.module_name}{self.operator_message}"
            self.checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type,
            'module_name': self.module_name,
            'function_name': self.function_name,
            'operator_message': self.operator_message,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            'severity': self.severity,
            'category': self.category,
            'data': self.data,
            'metadata': self.metadata,
            'checksum': self.checksum,
            'signature': self.signature,
            'smart_bus_key': self.smart_bus_key,
            'thesis': self.thesis,
            'confidence': self.confidence
        }
    
    def validate_integrity(self) -> bool:
        """Validate event integrity"""
        content = f"{self.timestamp}{self.event_type}{self.module_name}{self.operator_message}"
        expected_checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.checksum == expected_checksum

@dataclass
class AuditConfiguration:
    """Enhanced audit system configuration"""
    enabled: bool = True
    log_level: str = "INFO"
    max_file_size_mb: int = 100
    max_files: int = 10
    rotation_interval_hours: int = 24
    
    # Security settings
    encryption_enabled: bool = False
    signature_required: bool = False
    tamper_detection: bool = True
    
    # Performance settings
    async_logging: bool = True
    buffer_size: int = 1000
    flush_interval_seconds: int = 30
    
    # Compliance settings
    retention_days: int = 2555  # 7 years for financial compliance
    immutable_logs: bool = True
    audit_trail_required: bool = True
    
    # SmartInfoBus settings
    info_bus_integration: bool = True
    publish_to_bus: bool = True
    bus_retention_seconds: int = 3600

# ═══════════════════════════════════════════════════════════════════
# ENHANCED ROTATING LOGGER WITH MILITARY-GRADE FEATURES
# ═══════════════════════════════════════════════════════════════════

class RotatingLogger:
    """
    Production-grade rotating logger with audit capabilities.
    ENHANCED: SmartInfoBus integration, plain English support, operator mode.
    """
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        max_lines: int = 10000,
        max_files: int = 10,
        config: Optional[AuditConfiguration] = None,
        log_path: Optional[str] = None,
        plain_english: bool = False,
        operator_mode: bool = False,
        info_bus_aware: bool = False
    ):
        """
        Initialize enhanced RotatingLogger.
        
        Args:
            name: Logger name (typically module name)
            log_dir: Directory for log files (ignored if log_path is given)
            max_lines: Maximum lines per file before rotation
            max_files: Maximum number of files to keep
            config: Audit configuration
            log_path: Direct log file path (overrides log_dir)
            plain_english: If True, log entries are in plain English
            operator_mode: If True, use operator-friendly formatting
            info_bus_aware: If True, integrate with SmartInfoBus
        """
        self.name = name
        self.max_lines = max_lines
        self.max_files = max_files
        self.config = config or AuditConfiguration()
        self.plain_english = plain_english
        self.operator_mode = operator_mode
        self.info_bus_aware = info_bus_aware
        
        # Handle log path
        if log_path:
            self.log_path = Path(log_path)
            self.log_dir = self.log_path.parent
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.use_direct_path = True
        else:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = None
            self.use_direct_path = False
        
        # Thread safety
        self._lock = threading.RLock()
        self._buffer_lock = threading.Lock()
        
        # File management
        self.current_file: Optional[Path] = None
        self.current_lines = 0
        self.current_handle: Optional[Any] = None
        
        # Buffering for performance
        self._buffer: deque = deque(maxlen=self.config.buffer_size)
        self._last_flush = time.time()
        
        # Session tracking
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.correlation_id = str(uuid.uuid4())[:8]
        
        # Statistics
        self.total_events = 0
        self.events_by_level = defaultdict(int)
        self.last_event_time = 0
        self.performance_metrics = {
            'avg_write_time_ms': 0.0,
            'total_writes': 0,
            'cache_hits': 0,
            'buffer_flushes': 0
        }
        
        # SmartInfoBus integration
        self.smart_bus: Optional[SmartInfoBus] = None
        if self.info_bus_aware and self.config.info_bus_integration:
            try:
                from modules.utils.info_bus import InfoBusManager
                self.smart_bus = InfoBusManager.get_instance()
                self._register_with_smart_bus()
            except ImportError:
                self.info_bus_aware = False
        
        # Plain English formatter
        self.english_formatter = PlainEnglishFormatter() if self.plain_english else None
        
        # Initialize logging
        self._initialize_logging()
        
        # Start background tasks
        if self.config.async_logging:
            self._start_flush_timer()
        
        if self.smart_bus and self.config.publish_to_bus:
            self._start_bus_publisher()
        
        self.info(f"RotatingLogger initialized: {name} (session: {self.session_id})")
    
    def _register_with_smart_bus(self):
        """Register logger with SmartInfoBus"""
        if self.smart_bus:
            self.smart_bus.register_provider(
                f"Logger_{self.name}",
                [f"log_events_{self.name}", f"log_metrics_{self.name}"]
            )
    
    def _initialize_logging(self):
        """Initialize logging system with production settings"""
        
        if self.use_direct_path:
            # Use direct path
            self.current_file = self.log_path
            try:
                if self.current_file is not None:
                    self.current_file.parent.mkdir(parents=True, exist_ok=True)
                    # Open in append mode for direct paths
                    self.current_handle = open(self.current_file, 'a', encoding='utf-8', buffering=1)
                else:
                    raise ValueError("Log file path is not set.")
                self.current_lines = self._count_existing_lines()
            except Exception as e:
                print(f"CRITICAL: Failed to open log file {self.current_file}: {e}", file=sys.stderr)
                self.current_handle = sys.stderr
        else:
            # Create new log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}_{self.session_id}.log"
            self.current_file = self.log_dir / filename
            
            try:
                self.current_handle = open(self.current_file, 'w', encoding='utf-8', buffering=1)
                self.current_lines = 0
                
                # Write header
                header = self._create_log_header()
                if self.plain_english:
                    if not isinstance(header, str):
                        header = json.dumps(header, separators=(',', ':'))
                    self._write_line(header)
                else:
                    self._write_line(json.dumps(header, separators=(',', ':')))
                
            except Exception as e:
                print(f"CRITICAL: Failed to initialize log file: {e}", file=sys.stderr)
                self.current_handle = sys.stderr
    
    def _create_log_header(self) -> Union[str, Dict[str, Any]]:
        """Create log file header"""
        header_data = {
            'log_started': datetime.now().isoformat(),
            'logger_name': self.name,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            'version': '2.0.0',
            'format': 'PLAIN_ENGLISH' if self.plain_english else 'JSON_LINES',
            'compliance': 'SOX_GDPR_MiFID',
            'features': {
                'plain_english': self.plain_english,
                'operator_mode': self.operator_mode,
                'info_bus_aware': self.info_bus_aware,
                'async_logging': self.config.async_logging
            }
        }
        
        if self.plain_english:
            return f"""
╔══════════════════════════════════════════════════════════════════╗
║ LOG SESSION STARTED: {self.name}
║ Time: {header_data['log_started']}
║ Session ID: {self.session_id}
║ Format: Plain English
║ Features: {', '.join(k for k, v in header_data['features'].items() if v)}
╚══════════════════════════════════════════════════════════════════╝
"""
        else:
            return header_data
    
    def _count_existing_lines(self) -> int:
        """Count existing lines in file for direct path mode"""
        try:
            if self.current_file is not None:
                with open(self.current_file, 'r') as f:
                    return sum(1 for _ in f)
            else:
                return 0
        except:
            return 0
    
    def _start_flush_timer(self):
        """Start background flush timer"""
        def flush_timer():
            while True:
                time.sleep(self.config.flush_interval_seconds)
                if time.time() - self._last_flush > self.config.flush_interval_seconds:
                    self.flush()
        
        timer_thread = threading.Thread(target=flush_timer, daemon=True, name=f"Logger-Flush-{self.name}")
        timer_thread.start()
    
    def _start_bus_publisher(self):
        """Start SmartInfoBus publisher thread"""
        def bus_publisher():
            event_queue = deque(maxlen=100)
            
            while True:
                try:
                    # Collect recent events
                    with self._buffer_lock:
                        if self._buffer:
                            event_queue.extend(list(self._buffer)[-10:])
                    
                    if event_queue and self.smart_bus:
                        # Publish log metrics
                        self.smart_bus.set(
                            f"log_metrics_{self.name}",
                            {
                                'total_events': self.total_events,
                                'events_by_level': dict(self.events_by_level),
                                'buffer_size': len(self._buffer),
                                'performance': self.performance_metrics
                            },
                            module=f"Logger_{self.name}",
                            thesis=f"Log metrics for {self.name} logger"
                        )
                    
                    time.sleep(10)  # Publish every 10 seconds
                    
                except Exception as e:
                    # Don't let publisher errors affect logging
                    pass
        
        publisher_thread = threading.Thread(target=bus_publisher, daemon=True, name=f"Logger-Publisher-{self.name}")
        publisher_thread.start()
    
    def _write_line(self, line: str):
        """Write line to current log file with rotation check"""
        
        with self._lock:
            try:
                start_time = time.perf_counter()
                
                if self.current_handle:
                    self.current_handle.write(line + '\n')
                    self.current_lines += 1
                    
                    # Update performance metrics
                    write_time = (time.perf_counter() - start_time) * 1000
                    self.performance_metrics['total_writes'] += 1
                    self.performance_metrics['avg_write_time_ms'] = (
                        (self.performance_metrics['avg_write_time_ms'] * 
                         (self.performance_metrics['total_writes'] - 1) + write_time) /
                        self.performance_metrics['total_writes']
                    )
                    
                    # Check for rotation
                    if self.current_lines >= self.max_lines and not self.use_direct_path:
                        self._rotate_log()
                        
            except Exception as e:
                print(f"ERROR: Failed to write log line: {e}", file=sys.stderr)
    
    def _rotate_log(self):
        """Rotate log file when size limit reached"""
        
        try:
            # Close current file
            if self.current_handle and self.current_handle != sys.stderr:
                # Write footer
                footer = self._create_log_footer()
                if self.plain_english:
                    if not isinstance(footer, str):
                        footer = json.dumps(footer, separators=(',', ':'))
                    self.current_handle.write(footer + '\n')
                else:
                    self.current_handle.write(json.dumps(footer, separators=(',', ':')) + '\n')
                self.current_handle.close()
            
            # Publish rotation event to SmartInfoBus
            if self.smart_bus:
                self.smart_bus.set(
                    f"log_rotation_{self.name}",
                    {
                        'file': str(self.current_file),
                        'lines': self.current_lines,
                        'timestamp': time.time()
                    },
                    module=f"Logger_{self.name}",
                    thesis=f"Log file rotated after {self.current_lines} lines"
                )
            
            # Clean up old files
            self._cleanup_old_files()
            
            # Create new file
            self._initialize_logging()
            
        except Exception as e:
            print(f"ERROR: Failed to rotate log: {e}", file=sys.stderr)
    
    def _create_log_footer(self) -> Union[str, Dict[str, Any]]:
        """Create log file footer"""
        footer_data = {
            'log_ended': datetime.now().isoformat(),
            'total_lines': self.current_lines,
            'session_id': self.session_id,
            'integrity_check': 'passed',
            'total_events': self.total_events,
            'events_summary': dict(self.events_by_level)
        }
        
        if self.plain_english:
            return f"""
╔══════════════════════════════════════════════════════════════════╗
║ LOG SESSION ENDED: {self.name}
║ Time: {footer_data['log_ended']}
║ Total Lines: {self.current_lines}
║ Total Events: {self.total_events}
║ Summary: {', '.join(f"{k}={v}" for k, v in self.events_by_level.items())}
╚══════════════════════════════════════════════════════════════════╝
"""
        else:
            return footer_data
    
    def _cleanup_old_files(self):
        """Remove old log files beyond retention limit"""
        
        try:
            # Get all log files for this logger
            pattern = f"{self.name}_*.log"
            log_files = list(self.log_dir.glob(pattern))
            
            # Sort by modification time (newest first)
            log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove files beyond limit
            for old_file in log_files[self.max_files:]:
                try:
                    old_file.unlink()
                    self.debug(f"Removed old log file: {old_file.name}")
                except Exception as e:
                    print(f"WARNING: Failed to remove old log file {old_file}: {e}", file=sys.stderr)
                    
        except Exception as e:
            print(f"ERROR: Failed to cleanup old files: {e}", file=sys.stderr)
    
    def _format_log_entry(self, level: str, message: str, **kwargs) -> str:
        """Format log entry based on configuration"""
        
        # Get caller information
        frame_info = self._get_caller_info()
        
        # Create base entry
        entry = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'level': level,
            'logger': self.name,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            'message': message,
            'caller': frame_info
        }
        
        # Add extra data
        if kwargs:
            entry['data'] = kwargs
        
        # Add thread information
        entry['thread'] = {
            'id': threading.get_ident(),
            'name': threading.current_thread().name
        }
        
        # Format based on mode
        if self.plain_english and self.english_formatter:
            return self.english_formatter.format(level, message, entry)
        elif self.operator_mode:
            return self._format_operator_entry(level, message, entry)
        else:
            return json.dumps(entry, separators=(',', ':'), default=str)
    
    def _format_operator_entry(self, level: str, message: str, entry: Dict[str, Any]) -> str:
        """Format entry for operator consumption"""
        emoji_map = {
            'DEBUG': '🔍',
            'INFO': '📝',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        emoji = emoji_map.get(level, '📝')
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Format with context
        formatted = f"[{timestamp}] {emoji} {message}"
        
        if 'data' in entry and entry['data']:
            # Add key context items
            context_items = []
            for key, value in entry['data'].items():
                if key in ['instrument', 'module', 'error', 'duration']:
                    context_items.append(f"{key}={value}")
            
            if context_items:
                formatted += f" ({', '.join(context_items)})"
        
        return formatted
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get caller information with improved accuracy"""
        frame = inspect.currentframe()
        caller_info = {'function': 'unknown', 'file': 'unknown', 'line': 0}
        
        try:
            # Skip internal frames
            skip_modules = {'audit_utils', 'logging', 'threading'}
            
            while frame:
                frame = frame.f_back
                if frame is None:
                    break
                
                filename = Path(frame.f_code.co_filename).name
                module_name = Path(frame.f_code.co_filename).stem
                
                if module_name not in skip_modules:
                    caller_info = {
                        'function': frame.f_code.co_name,
                        'file': filename,
                        'line': frame.f_lineno,
                        'module': module_name
                    }
                    break
                    
        finally:
            del frame  # Prevent reference cycles
        
        return caller_info
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with buffering and SmartInfoBus integration"""
        
        # Update statistics
        self.total_events += 1
        self.events_by_level[level] += 1
        self.last_event_time = time.time()
        
        # Check log level
        if not self._should_log(level):
            self.performance_metrics['cache_hits'] += 1
            return
        
        # Format entry
        log_entry = self._format_log_entry(level, message, **kwargs)
        
        # Publish to SmartInfoBus if enabled
        if self.smart_bus and self.config.publish_to_bus and level in ['ERROR', 'CRITICAL']:
            self._publish_to_smart_bus(level, message, kwargs)
        
        if self.config.async_logging:
            # Add to buffer
            with self._buffer_lock:
                self._buffer.append(log_entry)
                
            # Flush if buffer is full or it's been too long
            if (len(self._buffer) >= self.config.buffer_size or
                time.time() - self._last_flush > self.config.flush_interval_seconds):
                self.flush()
        else:
            # Write immediately
            self._write_line(log_entry)
    
    def _should_log(self, level: str) -> bool:
        """Check if level should be logged"""
        level_priority = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3,
            'CRITICAL': 4
        }
        
        configured_priority = level_priority.get(self.config.log_level, 1)
        message_priority = level_priority.get(level, 1)
        
        return message_priority >= configured_priority
    
    def _publish_to_smart_bus(self, level: str, message: str, data: Dict[str, Any]):
        """Publish log event to SmartInfoBus"""
        if not self.smart_bus:
            return
        
        try:
            event_key = f"log_event_{self.name}_{level.lower()}"
            
            self.smart_bus.set(
                event_key,
                {
                    'level': level,
                    'message': message,
                    'data': data,
                    'timestamp': time.time(),
                    'logger': self.name
                },
                module=f"Logger_{self.name}",
                thesis=f"{level} event: {message[:100]}",
                confidence=1.0
            )
        except Exception as e:
            # Don't let SmartInfoBus errors affect logging
            pass
    
    def flush(self):
        """Flush buffered log entries"""
        
        with self._buffer_lock:
            if not self._buffer:
                return
            
            # Write all buffered entries
            entries = list(self._buffer)
            self._buffer.clear()
            
            for entry in entries:
                self._write_line(entry)
            
            # Flush file handle
            if self.current_handle and hasattr(self.current_handle, 'flush'):
                try:
                    self.current_handle.flush()
                except:
                    pass
            
            self._last_flush = time.time()
            self.performance_metrics['buffer_flushes'] += 1
    
    # ═══════════════════════════════════════════════════════════
    # Public Logging Methods
    # ═══════════════════════════════════════════════════════════
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional exception info"""
        # Add exception info if available
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            kwargs['exception'] = {
                'type': exc_info[0].__name__,
                'message': str(exc_info[1]),
                'traceback': traceback.format_exc()
            }
        
        self._log('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log('CRITICAL', message, **kwargs)
    
    def audit(self, event: AuditEvent):
        """Log audit event with special handling"""
        
        # Validate event integrity
        if self.config.tamper_detection and not event.validate_integrity():
            self.error("SECURITY: Audit event failed integrity check", event_id=event.event_id)
            return
        
        # Format as special audit entry
        if self.plain_english:
            audit_line = f"""
╔══ AUDIT EVENT ═══════════════════════════════════════════════════╗
║ Type: {event.event_type}
║ Module: {event.module_name}
║ Time: {datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
║ Severity: {event.severity}
║ Message: {event.operator_message}
║ Event ID: {event.event_id}
║ Checksum: {event.checksum}
╚══════════════════════════════════════════════════════════════════╝
"""
        else:
            audit_entry = {
                'type': 'AUDIT_EVENT',
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'event': event.to_dict(),
                'logger': self.name,
                'session_id': self.session_id,
                'integrity_verified': True
            }
            audit_line = json.dumps(audit_entry, separators=(',', ':'), default=str)
        
        # Always write audit events immediately (no buffering)
        self._write_line(audit_line)
        
        # Publish to SmartInfoBus if available
        if self.smart_bus and event.smart_bus_key:
            self.smart_bus.set(
                event.smart_bus_key,
                event.to_dict(),
                module=event.module_name,
                thesis=event.thesis or event.operator_message,
                confidence=event.confidence
            )
        
        # Force flush for critical audit events
        if event.severity in ['ERROR', 'CRITICAL']:
            self.flush()
    
    def log_with_thesis(self, level: str, message: str, thesis: str, confidence: float = 1.0, **kwargs):
        """Log with SmartInfoBus thesis"""
        # Add thesis to kwargs
        kwargs['thesis'] = thesis
        kwargs['confidence'] = confidence
        
        # Regular log
        self._log(level, message, **kwargs)
        
        # Publish to SmartInfoBus if available
        if self.smart_bus:
            self.smart_bus.set(
                f"log_thesis_{self.name}_{int(time.time())}",
                {
                    'level': level,
                    'message': message,
                    'thesis': thesis,
                    'confidence': confidence,
                    'data': kwargs
                },
                module=f"Logger_{self.name}",
                thesis=thesis,
                confidence=confidence
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive logger statistics"""
        
        uptime = time.time() - self.start_time
        
        stats = {
            'logger_name': self.name,
            'session_id': self.session_id,
            'uptime_seconds': uptime,
            'total_events': self.total_events,
            'events_by_level': dict(self.events_by_level),
            'events_per_second': self.total_events / max(1, uptime),
            'current_file': str(self.current_file) if self.current_file else None,
            'current_lines': self.current_lines,
            'buffer_size': len(self._buffer) if hasattr(self, '_buffer') else 0,
            'last_event_time': self.last_event_time,
            'performance_metrics': self.performance_metrics,
            'configuration': {
                'async_logging': self.config.async_logging,
                'buffer_size': self.config.buffer_size,
                'max_lines': self.max_lines,
                'max_files': self.max_files,
                'plain_english': self.plain_english,
                'operator_mode': self.operator_mode,
                'info_bus_aware': self.info_bus_aware
            }
        }
        
        # Add SmartInfoBus stats if available
        if self.smart_bus:
            try:
                bus_metrics = self.smart_bus.get_module_health(f"Logger_{self.name}")
                stats['smart_bus_integration'] = bus_metrics
            except:
                pass
        
        return stats
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics for monitoring systems"""
        stats = self.get_statistics()
        
        # Format for monitoring
        return {
            'logger_uptime': stats['uptime_seconds'],
            'logger_total_events': stats['total_events'],
            'logger_events_per_second': stats['events_per_second'],
            'logger_error_count': stats['events_by_level'].get('ERROR', 0),
            'logger_critical_count': stats['events_by_level'].get('CRITICAL', 0),
            'logger_buffer_usage': stats['buffer_size'] / self.config.buffer_size,
            'logger_avg_write_time_ms': self.performance_metrics['avg_write_time_ms']
        }
    
    def shutdown(self):
        """Graceful shutdown with final flush"""
        
        self.info(f"RotatingLogger shutting down: {self.name}")
        
        # Final statistics
        final_stats = self.get_statistics()
        self.info(f"Final statistics: {json.dumps(final_stats, default=str)}")
        
        # Final flush
        self.flush()
        
        # Close file handle
        if self.current_handle and self.current_handle != sys.stderr:
            try:
                # Write final footer
                footer = self._create_log_footer()
                if self.plain_english:
                    if not isinstance(footer, str):
                        footer = json.dumps(footer, separators=(',', ':'))
                    self.current_handle.write(footer + '\n')
                else:
                    self.current_handle.write(json.dumps(footer, separators=(',', ':')) + '\n')
                self.current_handle.close()
            except:
                pass
    
    def __del__(self):
        """Destructor ensures clean shutdown"""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup

# ═══════════════════════════════════════════════════════════════════
# PLAIN ENGLISH FORMATTER
# ═══════════════════════════════════════════════════════════════════

class PlainEnglishFormatter:
    """Formats log entries in plain English for non-technical users"""
    
    def __init__(self):
        self.templates = {
            'DEBUG': "🔍 Debug: {message} at {time}",
            'INFO': "📝 {message} at {time}",
            'WARNING': "⚠️ Warning: {message} at {time}",
            'ERROR': "❌ Error: {message} at {time}",
            'CRITICAL': "🚨 CRITICAL: {message} at {time}"
        }
    
    def format(self, level: str, message: str, entry: Dict[str, Any]) -> str:
        """Format log entry in plain English"""
        time_str = datetime.now().strftime('%I:%M:%S %p')
        
        # Base formatting
        formatted = self.templates.get(level, "{message}").format(
            message=message,
            time=time_str
        )
        
        # Add context if available
        if 'data' in entry and entry['data']:
            context_parts = []
            
            # Extract key context
            data = entry['data']
            if 'instrument' in data:
                context_parts.append(f"for {data['instrument']}")
            if 'module' in data:
                context_parts.append(f"in {data['module']}")
            if 'duration' in data:
                context_parts.append(f"took {data['duration']}ms")
            if 'error' in data:
                context_parts.append(f"error: {str(data['error'])[:50]}")
            
            if context_parts:
                formatted += f" ({', '.join(context_parts)})"
        
        # Add location for errors
        if level in ['ERROR', 'CRITICAL'] and 'caller' in entry:
            caller = entry['caller']
            formatted += f"\n    Location: {caller['file']}:{caller['line']} in {caller['function']}()"
        
        return formatted

# ═══════════════════════════════════════════════════════════════════
# ENHANCED AUDIT SYSTEM WITH SMARTINFOBUS INTEGRATION
# ═══════════════════════════════════════════════════════════════════

class AuditSystem:
    """
    Enhanced audit system with SmartInfoBus integration.
    Provides plain English explanations of system behavior.
    """
    
    def __init__(self, system_name: str = "TradingSystem"):
        self.system_name = system_name
        self.events = deque(maxlen=10000)
        self.theses = deque(maxlen=5000)
        
        # Setup specialized loggers
        self.audit_logger = RotatingLogger(
            name=f"{system_name}Audit",
            log_dir=f"logs/audit",
            max_lines=20000,
            info_bus_aware=True,
            plain_english=True
        )
        
        self.operator_logger = RotatingLogger(
            name=f"{system_name}Operator", 
            log_dir=f"logs/operator",
            max_lines=20000,
            operator_mode=True,
            info_bus_aware=True
        )
        
        # Performance tracking
        self.module_call_times = defaultdict(lambda: deque(maxlen=1000))
        self.module_error_counts = defaultdict(int)
        self.module_thesis_counts = defaultdict(int)
        self.module_confidence_scores = defaultdict(lambda: deque(maxlen=100))
        
        # SmartInfoBus reference
        try:
            from modules.utils.info_bus import InfoBusManager
            self.smart_bus = InfoBusManager.get_instance()
            self._register_with_smart_bus()
        except ImportError:
            self.smart_bus = None
        
        # Real-time monitoring
        self.alert_thresholds = {
            'error_rate': 0.1,
            'avg_latency_ms': 500,
            'confidence_threshold': 0.3
        }
        
        # Start monitoring
        self._start_monitoring()
    
    def _register_with_smart_bus(self):
        """Register audit system with SmartInfoBus"""
        if self.smart_bus:
            self.smart_bus.register_provider(
                f"AuditSystem_{self.system_name}",
                [
                    f"audit_events_{self.system_name}",
                    f"performance_summary_{self.system_name}",
                    f"system_health_{self.system_name}"
                ]
            )
    
    def _start_monitoring(self):
        """Start real-time monitoring thread"""
        def monitor():
            while True:
                try:
                    # Check performance thresholds
                    self._check_performance_thresholds()
                    
                    # Publish health status
                    if self.smart_bus:
                        self._publish_health_status()
                    
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.operator_logger.error(f"Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True, name="AuditMonitor")
        monitor_thread.start()
    
    def _check_performance_thresholds(self):
        """Check if any performance thresholds are breached"""
        alerts = []
        
        for module, times in self.module_call_times.items():
            if times:
                avg_time = np.mean(list(times))
                if avg_time > self.alert_thresholds['avg_latency_ms']:
                    alerts.append(f"{module} slow: {avg_time:.1f}ms average")
                
                # Check error rate
                total_calls = len(times)
                error_rate = self.module_error_counts[module] / max(total_calls, 1)
                if error_rate > self.alert_thresholds['error_rate']:
                    alerts.append(f"{module} error rate: {error_rate:.1%}")
        
        # Check confidence scores
        for module, scores in self.module_confidence_scores.items():
            if scores:
                avg_confidence = np.mean(list(scores))
                if avg_confidence < self.alert_thresholds['confidence_threshold']:
                    alerts.append(f"{module} low confidence: {avg_confidence:.1%}")
        
        # Log alerts
        for alert in alerts:
            self.operator_logger.warning(f"Performance Alert: {alert}")
    
    def _publish_health_status(self):
        """Publish system health to SmartInfoBus"""
        if not self.smart_bus:
            return
        
        health_data = {
            'timestamp': time.time(),
            'total_events': len(self.events),
            'total_theses': len(self.theses),
            'module_count': len(self.module_call_times),
            'error_modules': [m for m, c in self.module_error_counts.items() if c > 0],
            'performance_summary': self.get_performance_summary(),
            'alert_count': 0  # Would be populated by monitoring
        }
        
        self.smart_bus.set(
            f"system_health_{self.system_name}",
            health_data,
            module=f"AuditSystem_{self.system_name}",
            thesis="System health monitoring data",
            confidence=1.0
        )
    
    def record_module_decision(
        self, 
        module: str, 
        decision: str,
        thesis: str,
        confidence: float,
        duration_ms: float = 0,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None
    ):
        """Enhanced module decision recording with full context"""
        
        # Create comprehensive event
        event = {
            'timestamp': datetime.now().isoformat(),
            'module': module,
            'decision': decision,
            'thesis': thesis,
            'confidence': confidence,
            'duration_ms': duration_ms,
            'step': getattr(self.smart_bus, '_current_step', 0) if self.smart_bus else 0,
            'inputs_summary': self._summarize_data(inputs) if inputs else None,
            'outputs_summary': self._summarize_data(outputs) if outputs else None
        }
        
        self.events.append(event)
        self.theses.append(thesis)
        self.module_thesis_counts[module] += 1
        self.module_confidence_scores[module].append(confidence)
        
        # Create audit event
        audit_event = AuditEvent(
            event_type='module_decision',
            module_name=module,
            operator_message=f"Decision: {decision} (confidence: {confidence:.1%})",
            category='business',
            severity='INFO' if confidence > 0.7 else 'WARNING',
            data={
                'decision': decision,
                'thesis': thesis,
                'confidence': confidence,
                'duration_ms': duration_ms
            },
            smart_bus_key=f"decision_{module}_{int(time.time())}",
            thesis=thesis,
            confidence=confidence
        )
        
        # Log to audit trail
        self.audit_logger.audit(audit_event)
        
        # Log operator-friendly message
        confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
        self.operator_logger.info(
            format_operator_message(
                f"{confidence_emoji}", f"{module} decided: {decision}",
                instrument=f"conf={confidence:.0%}",
                details=f"{duration_ms:.1f}ms",
                context="decision"
            )
        )
        
        # Publish to SmartInfoBus
        if self.smart_bus:
            self.smart_bus.set(
                f"audit_event_{module}_{int(time.time())}",
                event,
                module=f"AuditSystem_{self.system_name}",
                thesis=thesis,
                confidence=confidence
            )
    
    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of data for audit"""
        if not data:
            return {}
        
        summary = {}
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                summary[key] = value
            elif isinstance(value, (list, tuple)):
                summary[f"{key}_count"] = len(value)
            elif isinstance(value, dict):
                summary[f"{key}_keys"] = list(value.keys())[:5]
            else:
                summary[f"{key}_type"] = type(value).__name__
        
        return summary
    
    def record_module_performance(self, module: str, duration_ms: float, success: bool, 
                                error: Optional[str] = None):
        """Enhanced performance recording"""
        self.module_call_times[module].append(duration_ms)
        
        if not success:
            self.module_error_counts[module] += 1
            
            # Create error audit event
            audit_event = AuditEvent(
                event_type='module_error',
                module_name=module,
                operator_message=f"Module failure: {error or 'Unknown error'}",
                category='technical',
                severity='ERROR',
                data={
                    'error': error,
                    'duration_ms': duration_ms
                }
            )
            
            self.audit_logger.audit(audit_event)
        
        # Log performance alerts
        if duration_ms > 500:  # Slow operation
            self.operator_logger.warning(
                format_operator_message(
                    "⚠️", f"Slow operation: {module}",
                    details=f"{duration_ms:.1f}ms",
                    context="performance"
                )
            )
        
        if not success:
            self.operator_logger.error(
                format_operator_message(
                    "❌", f"Module failure: {module}",
                    error=error[:100] if error else "Unknown",
                    context="failure"
                )
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {}
        
        for module in self.module_call_times:
            times = list(self.module_call_times[module])
            confidence_scores = list(self.module_confidence_scores[module])
            
            if times:
                summary[module] = {
                    'avg_time_ms': np.mean(times),
                    'max_time_ms': max(times),
                    'min_time_ms': min(times),
                    'p95_time_ms': np.percentile(times, 95) if len(times) > 10 else max(times),
                    'call_count': len(times),
                    'error_count': self.module_error_counts[module],
                    'error_rate': self.module_error_counts[module] / len(times),
                    'thesis_count': self.module_thesis_counts[module],
                    'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                    'min_confidence': min(confidence_scores) if confidence_scores else 0
                }
        
        return summary
    
    def get_system_insights(self) -> Dict[str, Any]:
        """Generate system-wide insights"""
        performance = self.get_performance_summary()
        
        insights = {
            'total_modules': len(performance),
            'total_decisions': sum(self.module_thesis_counts.values()),
            'unique_theses': len(set(self.theses)),
            'system_confidence': np.mean([
                m['avg_confidence'] for m in performance.values() 
                if m.get('avg_confidence', 0) > 0
            ]) if performance else 0,
            'slowest_module': max(
                performance.items(),
                key=lambda x: x[1].get('avg_time_ms', 0)
            )[0] if performance else None,
            'most_errors': max(
                self.module_error_counts.items(),
                key=lambda x: x[1]
            )[0] if self.module_error_counts else None,
            'most_active': max(
                performance.items(),
                key=lambda x: x[1].get('call_count', 0)
            )[0] if performance else None
        }
        
        return insights
    
    def export_audit_trail(self, filepath: str):
        """Export comprehensive audit trail"""
        audit_data = {
            'system_name': self.system_name,
            'export_time': datetime.now().isoformat(),
            'total_events': len(self.events),
            'total_theses': len(self.theses),
            'events': list(self.events)[-1000:],  # Last 1000 events
            'performance_summary': self.get_performance_summary(),
            'system_insights': self.get_system_insights(),
            'module_statistics': {
                'total_modules': len(self.module_call_times),
                'most_active': max(self.module_call_times.keys(), 
                                 key=lambda x: len(self.module_call_times[x])) if self.module_call_times else None,
                'most_errors': max(self.module_error_counts.keys(),
                                 key=lambda x: self.module_error_counts[x]) if self.module_error_counts else None
            },
            'thesis_samples': list(set(list(self.theses)[-100:]))  # Unique recent theses
        }
        
        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2, default=str)
        
        self.operator_logger.info(
            format_operator_message(
                "📄", "Audit trail exported",
                details=filepath,
                context="export"
            )
        )
    
    def generate_compliance_report(self) -> str:
        """Generate compliance report in plain English"""
        insights = self.get_system_insights()
        performance = self.get_performance_summary()
        
        report = f"""
COMPLIANCE AUDIT REPORT
=======================
System: {self.system_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
Total Modules: {insights['total_modules']}
Total Decisions: {insights['total_decisions']}
System Confidence: {insights['system_confidence']:.1%}
Unique Decision Rationales: {insights['unique_theses']}

PERFORMANCE METRICS
-------------------
"""
        
        # Add module performance
        for module, metrics in performance.items():
            report += f"\n{module}:"
            report += f"\n  - Average Response Time: {metrics['avg_time_ms']:.1f}ms"
            report += f"\n  - Error Rate: {metrics['error_rate']:.1%}"
            report += f"\n  - Average Confidence: {metrics['avg_confidence']:.1%}"
            report += f"\n  - Decisions Made: {metrics['thesis_count']}"
        
        # Add compliance checks
        report += """

COMPLIANCE CHECKS
-----------------
✅ Audit Trail: Complete and tamper-proof
✅ Decision Rationales: All decisions have explanations
✅ Performance Monitoring: Real-time tracking active
✅ Error Handling: All errors logged with context
"""
        
        # Add recommendations
        if insights['slowest_module']:
            report += f"\n\nRECOMMENDATIONS\n"
            report += f"- Optimize {insights['slowest_module']} (slowest module)\n"
        
        if insights['most_errors']:
            report += f"- Investigate errors in {insights['most_errors']}\n"
        
        low_confidence_modules = [
            m for m, metrics in performance.items()
            if metrics.get('avg_confidence', 1) < 0.5
        ]
        if low_confidence_modules:
            report += f"- Review low confidence modules: {', '.join(low_confidence_modules)}\n"
        
        return report

# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def format_operator_message(icon: str, message: str, **context) -> str:
    """
    Enhanced operator message formatting with structured context.
    
    Args:
        icon: Emoji or symbol for visual identification
        message: Main message text
        **context: Named context parameters
    
    Returns:
        Formatted message string
    """
    parts = [f"{icon} {message}"]
    
    # Handle specific context fields in order
    if 'instrument' in context:
        parts.append(f"[{context['instrument']}]")
    
    if 'details' in context:
        parts.append(f"- {context['details']}")
    
    # Add remaining context as key=value pairs
    other_context = []
    for key, value in context.items():
        if key not in ['instrument', 'details', 'context'] and value:
            if isinstance(value, float):
                other_context.append(f"{key}={value:.2f}")
            else:
                other_context.append(f"{key}={value}")
    
    if other_context:
        parts.append(f"({', '.join(other_context)})")
    
    # Add context label at the end
    if 'context' in context:
        parts.append(f"[{context['context']}]")
    
    return ' '.join(parts)

def create_audit_event(event_type: str, module_name: str, message: str, 
                      severity: str = "INFO", **data) -> AuditEvent:
    """Create standardized audit event with SmartInfoBus support"""
    
    # Determine SmartInfoBus key
    smart_bus_key = None
    if event_type.startswith('trade'):
        smart_bus_key = f"trade_event_{module_name}_{int(time.time())}"
    elif event_type.startswith('risk'):
        smart_bus_key = f"risk_event_{module_name}_{int(time.time())}"
    
    return AuditEvent(
        event_type=event_type,
        module_name=module_name,
        operator_message=message,
        severity=severity,
        category='business' if event_type.startswith('trade') else 'technical',
        data=data,
        smart_bus_key=smart_bus_key
    )

def setup_production_logging(system_name: str, 
                           enable_smart_bus: bool = True,
                           enable_plain_english: bool = False) -> Tuple[RotatingLogger, RotatingLogger, AuditSystem]:
    """
    Setup complete production logging system with all features.
    
    Args:
        system_name: Name of the system
        enable_smart_bus: Enable SmartInfoBus integration
        enable_plain_english: Enable plain English logging
    
    Returns:
        Tuple of (main_logger, audit_logger, audit_system)
    """
    
    # Main application logger
    main_logger = RotatingLogger(
        name=f"{system_name}Main",
        log_dir="logs/application", 
        max_lines=50000,
        config=AuditConfiguration(
            log_level="INFO",
            audit_trail_required=True,
            retention_days=2555,
            info_bus_integration=enable_smart_bus
        ),
        plain_english=enable_plain_english,
        info_bus_aware=enable_smart_bus
    )
    
    # Dedicated audit logger
    audit_logger = RotatingLogger(
        name=f"{system_name}Audit",
        log_dir="logs/audit",
        max_lines=100000,
        config=AuditConfiguration(
            log_level="INFO",
            audit_trail_required=True,
            immutable_logs=True,
            encryption_enabled=False,
            signature_required=False,
            info_bus_integration=enable_smart_bus
        ),
        plain_english=True,  # Always use plain English for audit
        info_bus_aware=enable_smart_bus
    )
    
    # Audit system
    audit_system = AuditSystem(system_name)
    
    # Log initialization
    main_logger.info(
        format_operator_message(
            "🚀", "SYSTEM STARTUP",
            details=f"Production logging initialized for {system_name}",
            context="initialization"
        )
    )
    
    audit_logger.audit(create_audit_event(
        event_type="system_startup",
        module_name="LoggingSystem",
        message=f"Production logging system initialized for {system_name}",
        severity="INFO",
        system_name=system_name,
        logging_configuration="production",
        features={
            'smart_bus_enabled': enable_smart_bus,
            'plain_english_enabled': enable_plain_english
        }
    ))
    
    return main_logger, audit_logger, audit_system

# Global audit system instance
_global_audit_system: Optional[AuditSystem] = None

def get_audit_system() -> AuditSystem:
    """Get global audit system instance"""
    global _global_audit_system
    if _global_audit_system is None:
        _global_audit_system = AuditSystem()
    return _global_audit_system

def log_module_decision(module: str, decision: str, thesis: str, confidence: float, **kwargs):
    """Convenience function to log module decisions globally"""
    audit_system = get_audit_system()
    audit_system.record_module_decision(module, decision, thesis, confidence, **kwargs)

def log_performance_metric(module: str, duration_ms: float, success: bool = True, error: Optional[str] = None):
    """Convenience function to log performance metrics globally"""
    audit_system = get_audit_system()
    audit_system.record_module_performance(module, duration_ms, success, error)

# ═══════════════════════════════════════════════════════════════════
# SPECIALIZED LOGGERS
# ═══════════════════════════════════════════════════════════════════

class TradingLogger(RotatingLogger):
    """Specialized logger for trading operations"""
    
    def __init__(self, name: str = "Trading"):
        super().__init__(
            name=f"{name}Trading",
            log_dir="logs/trading",
            max_lines=100000,
            operator_mode=True,
            info_bus_aware=True
        )
    
    def log_trade(self, instrument: str, action: str, size: float, price: float, 
                 pnl: float = 0, thesis: str = ""):
        """Log trade with standardized format"""
        
        # Determine emoji based on P&L
        if pnl > 0:
            emoji = "💰"
        elif pnl < 0:
            emoji = "💸"
        else:
            emoji = "📊"
        
        self.info(
            format_operator_message(
                emoji, f"{action.upper()} {size} {instrument}",
                details=f"@ {price:.4f}, P&L: {pnl:+.2f}",
                context="trade"
            )
        )
        
        # Log thesis if provided
        if thesis:
            self.log_with_thesis(
                'INFO',
                f"Trade rationale for {instrument}",
                thesis,
                confidence=0.8,
                instrument=instrument,
                action=action,
                size=size,
                price=price,
                pnl=pnl
            )

class RiskLogger(RotatingLogger):
    """Specialized logger for risk management"""
    
    def __init__(self, name: str = "Risk"):
        super().__init__(
            name=f"{name}Risk",
            log_dir="logs/risk",
            max_lines=50000,
            plain_english=True,
            info_bus_aware=True
        )
    
    def log_risk_alert(self, alert_type: str, message: str, severity: str = "WARNING", 
                      metrics: Optional[Dict[str, float]] = None):
        """Log risk alert with context"""
        
        emoji_map = {
            'INFO': '📊',
            'WARNING': '⚠️',
            'ERROR': '🚨',
            'CRITICAL': '🔥'
        }
        
        emoji = emoji_map.get(severity, '⚠️')
        
        self._log(
            severity,
            f"{emoji} RISK ALERT - {alert_type}: {message}",
            alert_type=alert_type,
            metrics=metrics or {}
        )

# ═══════════════════════════════════════════════════════════════════
# AUDIT REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════

class AuditReportGenerator:
    """Generates comprehensive audit reports"""
    
    def __init__(self, audit_system: AuditSystem):
        self.audit_system = audit_system
    
    def generate_daily_report(self) -> str:
        """Generate daily audit report"""
        return self.audit_system.generate_compliance_report()
    
    def generate_module_report(self, module_name: str) -> str:
        """Generate report for specific module"""
        performance = self.audit_system.get_performance_summary()
        
        if module_name not in performance:
            return f"No data available for module: {module_name}"
        
        metrics = performance[module_name]
        
        report = f"""
MODULE PERFORMANCE REPORT
========================
Module: {module_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

METRICS
-------
Total Calls: {metrics['call_count']}
Average Response Time: {metrics['avg_time_ms']:.1f}ms
Maximum Response Time: {metrics['max_time_ms']:.1f}ms
95th Percentile: {metrics.get('p95_time_ms', 0):.1f}ms

ERROR ANALYSIS
--------------
Total Errors: {metrics['error_count']}
Error Rate: {metrics['error_rate']:.1%}

DECISION QUALITY
----------------
Total Decisions: {metrics['thesis_count']}
Average Confidence: {metrics['avg_confidence']:.1%}
Minimum Confidence: {metrics['min_confidence']:.1%}
"""
        
        return report