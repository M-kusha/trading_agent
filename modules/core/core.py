# ─────────────────────────────────────────────────────────────
# File: modules/core/core.py
# Windows-Compatible Enhanced base module with ASCII logging
# ─────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
import logging
import numpy as np
import datetime
import json
import os
import sys
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from collections import deque
from functools import wraps

if TYPE_CHECKING:
    from modules.utils.info_bus import InfoBus


class ModuleConfig:
    """Standard configuration for all modules"""
    def __init__(self, **kwargs):
        # Standard defaults that every module needs
        self.debug = kwargs.get('debug', True)
        self.max_history = kwargs.get('max_history', 100)
        self.audit_enabled = kwargs.get('audit_enabled', True)
        self.log_rotation_lines = kwargs.get('log_rotation_lines', 2000)
        self.health_check_interval = kwargs.get('health_check_interval', 100)
        
        # Allow custom config
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class ModuleLogger:
    """Standardized rotating logger for all modules with Windows compatibility"""
    
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, module_name: str, config: ModuleConfig) -> logging.Logger:
        """Get or create standardized logger with rotation and Windows compatibility"""
        
        if module_name in cls._loggers:
            return cls._loggers[module_name]
        
        # Create logger
        logger = logging.getLogger(module_name)
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
        logger.propagate = False
        
        # Setup log directory
        log_dir = Path("logs") / module_name.lower().replace("_", "/")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler with UTF-8 encoding
        log_path = log_dir / f"{module_name.lower()}.log"
        handler = WindowsCompatibleRotatingHandler(
            str(log_path), 
            maxLines=config.log_rotation_lines
        )
        
        # Windows-compatible format (no emojis)
        if config.debug:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Add console handler with proper encoding for Windows
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Set proper encoding for Windows console
        if platform.system() == "Windows":
            if hasattr(console_handler.stream, 'reconfigure'):
                try:
                    console_handler.stream.reconfigure(encoding='utf-8')
                except:
                    pass  # Fallback to default encoding
        
        logger.addHandler(console_handler)
        
        cls._loggers[module_name] = logger
        return logger


class WindowsCompatibleRotatingHandler(logging.FileHandler):
    """File handler that enforces line limits with Windows compatibility"""
    
    def __init__(self, filename, maxLines=2000, **kwargs):
        self.maxLines = maxLines
        # Force UTF-8 encoding for consistency
        kwargs['encoding'] = 'utf-8'
        super().__init__(filename, **kwargs)
        
    def emit(self, record):
        try:
            super().emit(record)
            self._rotate_if_needed()
        except UnicodeEncodeError:
            # Fallback: strip non-ASCII characters and retry
            if hasattr(record, 'msg'):
                record.msg = self._strip_non_ascii(str(record.msg))
            super().emit(record)
            self._rotate_if_needed()
        
    def _strip_non_ascii(self, text):
        """Strip non-ASCII characters for Windows compatibility"""
        return ''.join(char if ord(char) < 128 else '?' for char in text)
        
    def _rotate_if_needed(self):
        """Enforce line limit by deleting oldest entries"""
        try:
            if not os.path.exists(self.baseFilename):
                return
                
            with open(self.baseFilename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if len(lines) > self.maxLines:
                # Keep only most recent lines
                keep_lines = lines[-self.maxLines:]
                with open(self.baseFilename, 'w', encoding='utf-8') as f:
                    f.writelines(keep_lines)
        except Exception:
            pass  # Don't break logging if rotation fails


def audit_step(func):
    """Decorator to automatically audit module steps"""
    @wraps(func)
    def wrapper(self, info_bus: Optional['InfoBus'] = None, **kwargs):
        if not hasattr(self, '_audit_enabled') or not self._audit_enabled:
            return func(self, info_bus, **kwargs)
            
        # Pre-step audit
        start_time = datetime.datetime.now()
        
        try:
            # Execute step
            result = func(self, info_bus, **kwargs)
            
            # Post-step audit
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self._record_step_audit({
                'timestamp': start_time.isoformat(),
                'module': self.__class__.__name__,
                'duration_ms': duration * 1000,
                'success': True,
                'has_info_bus': info_bus is not None,
                'step_count': getattr(self, '_step_count', 0)
            })
            
            return result
            
        except Exception as e:
            # Error audit
            self._record_step_audit({
                'timestamp': start_time.isoformat(),
                'module': self.__class__.__name__,
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            self._update_health_status("ERROR", str(e))
            raise
            
    return wrapper


def validate_observation(func):
    """Decorator to validate observation components"""
    @wraps(func)
    def wrapper(self):
        obs = func(self)
        
        # Validate observation
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
            
        # Check for non-finite values
        if not np.all(np.isfinite(obs)):
            self.logger.error(f"[ERROR] Non-finite observation: {obs}")
            # Replace with safe default
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Ensure correct dtype
        obs = obs.astype(np.float32)
        
        return obs
        
    return wrapper


def format_operator_message_windows(prefix: str, action: str, instrument: str = "", 
                                   details: str = "", result: str = "", context: str = "") -> str:
    """
    Format operator-centric messages for Windows compatibility (no emojis).
    
    Example: "[TRADE] BUY EUR/USD 1.25 @ 1.0850 | P&L: $125.50 | Regime: trending"
    """
    parts = [f"[{prefix}]", action]
    
    if instrument:
        parts.append(instrument)
        
    if details:
        parts.append(details)
        
    if result:
        parts.append(f"| {result}")
        
    if context:
        parts.append(f"| {context}")
        
    return " ".join(parts)


class Module(ABC):
    """Enhanced base module with Windows compatibility"""
    
    def __init__(self, config: Optional[Union[Dict, ModuleConfig]] = None, **kwargs):
        # Handle config
        if isinstance(config, dict):
            self.config = ModuleConfig(**config, **kwargs)
        elif isinstance(config, ModuleConfig):
            self.config = config
        else:
            self.config = ModuleConfig(**kwargs)
            
        # Standard state
        self._initialized = True
        self._health_status = "OK"
        self._last_error = None
        self._step_count = 0
        self._audit_enabled = self.config.audit_enabled
        
        # Standard collections with automatic size management
        self._audit_trail = deque(maxlen=self.config.max_history)
        self._health_history = deque(maxlen=50)
        self._performance_metrics = {}
        
        # Setup logger with Windows compatibility
        self.logger = ModuleLogger.get_logger(
            self.__class__.__name__, 
            self.config
        )
        
        # Initialize module-specific state
        self._initialize_module_state()
        
        # Windows-compatible success message
        self.logger.info(f"[SUCCESS] {self.__class__.__name__} initialized")

    def _initialize_module_state(self):
        """Override this to initialize module-specific state"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset module to initial state"""
        self._step_count = 0
        self._audit_trail.clear()
        self._health_history.clear()
        self._performance_metrics.clear()
        self._update_health_status("OK")

    @audit_step
    def step(self, info_bus: Optional['InfoBus'] = None, **kwargs) -> None:
        """Enhanced step with automatic audit and health tracking"""
        self._step_count += 1
        
        # Call implementation-specific step
        self._step_impl(info_bus, **kwargs)
        
        # Periodic health check
        if self._step_count % self.config.health_check_interval == 0:
            self._perform_health_check()

    @abstractmethod
    def _step_impl(self, info_bus: Optional['InfoBus'] = None, **kwargs) -> None:
        """Implement this instead of step() - automatic audit/health included"""
        pass

    @validate_observation
    def get_observation_components(self) -> np.ndarray:
        """
        Universal getter for observation features.
        1. If subclass overrides this, use the override.
        2. Else, call _get_observation_impl (modern standard).
        3. If neither, raise an explicit error—NO fallback data.
        """
        # If the subclass provides its own get_observation_components, Python will call that.
        # If not, call _get_observation_impl.
        # If not implemented, raise.
        base_method = Module.get_observation_components
        sub_method = self.__class__.__dict__.get('get_observation_components', None)
        if sub_method and sub_method is not base_method:
            return sub_method(self)
        # fallback to _get_observation_impl (modern style)
        method = getattr(self, '_get_observation_impl', None)
        if callable(method):
            return method()
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either get_observation_components() or _get_observation_impl()."
        )


    
    def _get_observation_impl(self) -> np.ndarray:
        """
        Must be implemented by subclass if get_observation_components is not overridden.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_observation_impl() or get_observation_components()."
        )

    def _update_health_status(self, status: str, error: Optional[str] = None):
        """Update health status with automatic history"""
        self._health_status = status
        self._last_error = error
        
        health_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'status': status,
            'error': error,
            'step_count': self._step_count
        }
        
        self._health_history.append(health_record)
        
        if status != "OK":
            self.logger.warning(f"[WARNING] Health status: {status} - {error}")

    def _perform_health_check(self):
        """Automatic health check - override for custom checks"""
        try:
            # Basic health checks
            checks = {
                'memory_usage': self._check_memory_usage(),
                'state_integrity': self._check_state_integrity(),
                'performance': self._check_performance()
            }
            
            failed_checks = [k for k, v in checks.items() if not v]
            
            if failed_checks:
                self._update_health_status("DEGRADED", f"Failed: {', '.join(failed_checks)}")
            else:
                self._update_health_status("OK")
                
        except Exception as e:
            self._update_health_status("ERROR", f"Health check failed: {e}")

    def _check_memory_usage(self) -> bool:
        """Check if collections are within reasonable size"""
        return (len(self._audit_trail) < self.config.max_history * 1.1 and
                len(self._health_history) < 55)

    def _check_state_integrity(self) -> bool:
        """Override for module-specific state checks"""
        return True

    def _check_performance(self) -> bool:
        """Check if module performance is acceptable"""
        if len(self._audit_trail) < 10:
            return True
            
        # Check if step duration is reasonable (< 100ms avg)
        recent_audits = list(self._audit_trail)[-10:]
        avg_duration = np.mean([
            a.get('duration_ms', 0) for a in recent_audits 
            if 'duration_ms' in a
        ])
        
        return avg_duration < 100  # 100ms threshold

    def _record_step_audit(self, audit_data: Dict[str, Any]):
        """Record step audit with automatic management"""
        if self._audit_enabled:
            self._audit_trail.append(audit_data)

    def get_health_status(self) -> Dict[str, Any]:
        """Enhanced health status with history and metrics"""
        recent_errors = [
            h for h in self._health_history 
            if h['status'] != 'OK'
        ]
        
        return {
            "status": self._health_status,
            "module": self.__class__.__name__,
            "initialized": self._initialized,
            "last_error": self._last_error,
            "step_count": self._step_count,
            "recent_errors": len(recent_errors),
            "health_trend": self._calculate_health_trend(),
            "performance_metrics": self._performance_metrics.copy(),
            "details": self._get_health_details()
        }

    def _calculate_health_trend(self) -> str:
        """Calculate health trend from recent history"""
        if len(self._health_history) < 5:
            return "insufficient_data"
            
        recent = list(self._health_history)[-5:]
        error_count = sum(1 for h in recent if h['status'] != 'OK')
        
        if error_count == 0:
            return "stable"
        elif error_count >= 3:
            return "degrading"
        else:
            return "unstable"

    def _get_health_details(self) -> Optional[Dict[str, Any]]:
        """Override to provide module-specific health details"""
        return {
            'audit_trail_size': len(self._audit_trail),
            'health_history_size': len(self._health_history),
            'config': {
                'debug': self.config.debug,
                'max_history': self.config.max_history,
                'audit_enabled': self.config.audit_enabled
            }
        }

    def get_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        base_state = {
            'class_name': self.__class__.__name__,
            'config': self.config.__dict__,
            'step_count': self._step_count,
            'health_status': self._health_status,
            'last_error': self._last_error,
            'performance_metrics': self._performance_metrics.copy()
        }
        
        # Add module-specific state
        module_state = self._get_module_state()
        if module_state:
            base_state['module_state'] = module_state
            
        return base_state

    def _get_module_state(self) -> Optional[Dict[str, Any]]:
        """Override to provide module-specific state"""
        return None

    def set_state(self, state: Dict[str, Any]) -> None:
        """Enhanced state restoration"""
        self._step_count = state.get('step_count', 0)
        self._health_status = state.get('health_status', 'OK')
        self._last_error = state.get('last_error')
        self._performance_metrics = state.get('performance_metrics', {}).copy()
        
        # Restore module-specific state
        if 'module_state' in state:
            self._set_module_state(state['module_state'])
            
        self.logger.info(f"[SUCCESS] State restored at step {self._step_count}")

    def _set_module_state(self, module_state: Dict[str, Any]) -> None:
        """Override to restore module-specific state"""
        pass

    # Standard interface methods with defaults
    def propose_action(self, obs: Any, info_bus: Optional['InfoBus'] = None) -> np.ndarray:
        """Default action proposal - override for voting modules"""
        if hasattr(obs, 'shape'):
            action_dim = obs.shape[0] // 2 if len(obs.shape) > 0 else 2
        else:
            action_dim = 2
        return np.zeros(action_dim, dtype=np.float32)

    def confidence(self, obs: Any, info_bus: Optional['InfoBus'] = None) -> float:
        """Default confidence - override for voting modules"""
        return 0.5

    # Performance tracking helpers
    def _update_performance_metric(self, name: str, value: float):
        """Update a performance metric with history"""
        if name not in self._performance_metrics:
            self._performance_metrics[name] = deque(maxlen=100)
        self._performance_metrics[name].append(value)

    def _get_performance_metric(self, name: str, default: float = 0.0) -> float:
        """Get latest value of a performance metric"""
        if name in self._performance_metrics and self._performance_metrics[name]:
            return self._performance_metrics[name][-1]
        return default

    def _get_performance_trend(self, name: str, window: int = 10) -> str:
        """Get trend for a performance metric"""
        if (name not in self._performance_metrics or 
            len(self._performance_metrics[name]) < window):
            return "insufficient_data"
            
        values = list(self._performance_metrics[name])[-window:]
        recent_avg = np.mean(values[-window//2:])
        earlier_avg = np.mean(values[:window//2])
        
        if recent_avg > earlier_avg * 1.05:
            return "improving"
        elif recent_avg < earlier_avg * 0.95:
            return "declining"
        else:
            return "stable"

    # Windows-compatible logging helpers
    def log_operator_info(self, message: str, **context):
        """Log operator-friendly info message (Windows-compatible)"""
        formatted_msg = self._format_operator_message_windows("INFO", message, **context)
        self.logger.info(formatted_msg)

    def log_operator_warning(self, message: str, **context):
        """Log operator-friendly warning message (Windows-compatible)"""
        formatted_msg = self._format_operator_message_windows("WARNING", message, **context)
        self.logger.warning(formatted_msg)

    def log_operator_error(self, message: str, **context):
        """Log operator-friendly error message (Windows-compatible)"""
        formatted_msg = self._format_operator_message_windows("ERROR", message, **context)
        self.logger.error(formatted_msg)

    def _format_operator_message_windows(self, level: str, message: str, **context) -> str:
        """Format message for operator readability (Windows-compatible)"""
        parts = [f"[{level}]", message]
        
        if context:
            context_str = " | ".join(f"{k}: {v}" for k, v in context.items())
            parts.append(f"| {context_str}")
            
        return " ".join(parts)

    # Evolution support (kept from original)
    def mutate(self, noise_std: float = 0.01) -> None:
        """Mutate module parameters for evolution"""
        pass

    def crossover(self, other: 'Module') -> 'Module':
        """Create offspring through crossover"""
        return self

    def fitness(self) -> float:
        """Return fitness score for evolution"""
        return 0.0