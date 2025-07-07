# ─────────────────────────────────────────────────────────────
# File: modules/core/core.py
# 🚀 ENHANCED SmartInfoBus base module with @module decorator
# ─────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
import logging
import numpy as np
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from collections import deque
from functools import wraps
from dataclasses import dataclass
import importlib
import inspect

from modules.utils.info_bus import (
    SmartInfoBus, InfoBus, InfoBusManager, InfoBusExtractor, InfoBusUpdater,
    require_info_bus, cache_computation, validate_info_bus
)
from modules.utils.audit_utils import (
    RotatingLogger, InfoBusAuditTracker, audit_module_call,
    format_operator_message, system_audit
)

# ═══════════════════════════════════════════════════════════════════
# MODULE METADATA & DECORATOR SYSTEM
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModuleMetadata:
    """Metadata for self-registering modules"""
    name: str
    provides: List[str]
    requires: List[str]
    version: str = "1.0.0"
    category: str = "general"
    is_voting_member: bool = False
    hot_reload: bool = True
    explainable: bool = True
    timeout_ms: int = 100
    priority: int = 0

def module(**kwargs):
    """
    Decorator for self-registering SmartInfoBus modules.
    Automatically registers with orchestrator on import.
    """
    def decorator(cls):
        # Extract metadata
        cls.__module_metadata__ = ModuleMetadata(
            name=cls.__name__,
            **kwargs
        )
        
        # Auto-register with orchestrator on import
        try:
            from modules.core.module_orchestrator import ModuleOrchestrator
            ModuleOrchestrator.register_class(cls)
        except ImportError:
            # Orchestrator not yet available - will be picked up during discovery
            pass
        
        # Add required methods if not present
        if not hasattr(cls, 'get_state'):
            def get_state(self) -> Dict[str, Any]:
                """Default state getter"""
                return {
                    'class_name': self.__class__.__name__,
                    'step_count': getattr(self, '_step_count', 0)
                }
            cls.get_state = get_state
            
        if not hasattr(cls, 'set_state'):
            def set_state(self, state: Dict[str, Any]):
                """Default state setter"""
                self._step_count = state.get('step_count', 0)
            cls.set_state = set_state
        
        # Register with SmartInfoBus
        smart_bus = InfoBusManager.get_instance()
        smart_bus.register_provider(cls.__name__, kwargs.get('provides', []))
        smart_bus.register_consumer(cls.__name__, kwargs.get('requires', []))
        
        return cls
    return decorator

# ═══════════════════════════════════════════════════════════════════
# ENHANCED MODULE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class ModuleConfig:
    """Enhanced configuration for SmartInfoBus modules"""
    def __init__(self, **kwargs):
        # Standard defaults
        self.debug = kwargs.get('debug', True)
        self.max_history = kwargs.get('max_history', 100)
        self.audit_enabled = kwargs.get('audit_enabled', True)
        self.log_rotation_lines = kwargs.get('log_rotation_lines', 2000)
        self.health_check_interval = kwargs.get('health_check_interval', 100)
        self.performance_tracking = kwargs.get('performance_tracking', True)
        self.cache_enabled = kwargs.get('cache_enabled', True)
        self.explainable = kwargs.get('explainable', True)
        self.hot_reload = kwargs.get('hot_reload', True)
        
        # Allow custom config
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS DECORATORS
# ═══════════════════════════════════════════════════════════════════

def smart_info_bus_step(func):
    """Decorator for SmartInfoBus-based step methods with timing"""
    @wraps(func)
    def wrapper(self, info_bus: Union[InfoBus, SmartInfoBus], **kwargs):
        # Get SmartInfoBus reference
        if isinstance(info_bus, dict) and '_smart_bus' in info_bus:
            smart_bus = info_bus['_smart_bus']
        else:
            smart_bus = InfoBusManager.get_instance()
        
        # Check if module is enabled (circuit breaker)
        if not smart_bus.is_module_enabled(self.__class__.__name__):
            self.logger.warning(f"Module {self.__class__.__name__} is disabled")
            return None
        
        # Execute with timing
        start_time = time.perf_counter()
        error = None
        
        try:
            result = func(self, info_bus, **kwargs)
            return result
            
        except Exception as e:
            error = str(e)
            smart_bus.record_module_failure(self.__class__.__name__, error)
            raise
            
        finally:
            # Record timing
            duration_ms = (time.perf_counter() - start_time) * 1000
            smart_bus.record_module_timing(self.__class__.__name__, duration_ms)
            
            # Update performance metrics
            if hasattr(self, '_update_performance_metric'):
                self._update_performance_metric('avg_duration_ms', duration_ms)
            
            # Health check
            self._step_count += 1
            if self._step_count % self.config.health_check_interval == 0:
                self._perform_health_check(info_bus)
    
    return wrapper

# ═══════════════════════════════════════════════════════════════════
# BASE MODULE CLASS
# ═══════════════════════════════════════════════════════════════════

class BaseModule(ABC):
    """
    🚀 Enhanced SmartInfoBus base module.
    ALL modules MUST inherit from this and use @module decorator.
    """
    
    def __init__(self, config: Optional[Union[Dict, ModuleConfig]] = None, **kwargs):
        # Handle config
        if isinstance(config, dict):
            self.config = ModuleConfig(**config, **kwargs)
        elif isinstance(config, ModuleConfig):
            self.config = config
        else:
            self.config = ModuleConfig(**kwargs)
        
        # Get metadata if decorated
        if hasattr(self.__class__, '__module_metadata__'):
            self.metadata = self.__class__.__module_metadata__
        else:
            # Create default metadata
            self.metadata = ModuleMetadata(
                name=self.__class__.__name__,
                provides=[],
                requires=[]
            )
        
        # Core state
        self._initialized = True
        self._health_status = "OK"
        self._last_error = None
        self._step_count = 0
        self._last_thesis = ""
        
        # Performance tracking
        self.version = self.metadata.version
        self.execution_time = 0
        
        # Collections with size limits
        self._health_history = deque(maxlen=50)
        self._performance_metrics = {}
        
        # Setup logging
        self.logger = RotatingLogger(
            name=self.__class__.__name__,
            log_path=f"logs/modules/{self.__class__.__name__.lower()}.log",
            max_lines=self.config.log_rotation_lines,
            operator_mode=self.config.debug,
            info_bus_aware=True
        )
        
        # Setup audit tracking
        if self.config.audit_enabled:
            self._audit_tracker = InfoBusAuditTracker(self.__class__.__name__)
        
        # Initialize module-specific state
        self._initialize_module_state()
        
        self.logger.info(
            format_operator_message(
                "✅", "MODULE INITIALIZED",
                instrument=self.__class__.__name__,
                details=f"v{self.version}",
                context="startup"
            )
        )
    
    # ─────────────────────────────────────────────────────────────
    # SmartInfoBus Process Method (replaces step)
    # ─────────────────────────────────────────────────────────────
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Main processing method for SmartInfoBus modules.
        Extract inputs, process, return outputs with thesis.
        """
        # Default implementation delegates to step for compatibility
        info_bus = inputs.get('info_bus') or InfoBusManager.get_current()
        if info_bus:
            self.step(info_bus)
        
        # Return empty dict by default
        return {}
    
    def explain_decision(self, decision: Any, context: Dict) -> str:
        """Generate human-readable explanation"""
        return f"Decision: {decision} based on {len(context)} factors"
    
    def validate_inputs(self, inputs: Dict) -> bool:
        """Validate all required inputs are present"""
        for req in self.metadata.requires:
            if req not in inputs or inputs[req] is None:
                raise ValueError(f"Missing required input: {req}")
        return True
    
    # ─────────────────────────────────────────────────────────────
    # Legacy Step Method (for backward compatibility)
    # ─────────────────────────────────────────────────────────────
    
    @smart_info_bus_step
    def step(self, info_bus: InfoBus) -> None:
        """
        Legacy step method - wraps to SmartInfoBus process.
        """
        # Get SmartInfoBus
        smart_bus = InfoBusManager.get_instance()
        
        # Collect inputs based on requirements
        inputs = {'info_bus': info_bus}
        for req in self.metadata.requires:
            value = smart_bus.get(req, self.__class__.__name__)
            if value is not None:
                inputs[req] = value
        
        # Call module implementation
        self._step_impl(info_bus)
        
        # Record any outputs
        for output in self.metadata.provides:
            # Module should have set these in SmartInfoBus during _step_impl
            pass
    
    @abstractmethod
    def _step_impl(self, info_bus: InfoBus) -> None:
        """Legacy step implementation"""
        pass
    
    # ─────────────────────────────────────────────────────────────
    # State Management for Hot-Reload
    # ─────────────────────────────────────────────────────────────
    
    def get_state(self) -> Dict[str, Any]:
        """Get module state for hot-reload persistence"""
        base_state = {
            'class_name': self.__class__.__name__,
            'version': self.metadata.version,
            'config': self.config.__dict__,
            'step_count': self._step_count,
            'health_status': self._health_status,
            'last_error': self._last_error,
            'last_thesis': self._last_thesis
        }
        
        # Add module-specific state
        module_state = self._get_module_state()
        if module_state:
            base_state['module_state'] = module_state
            
        return base_state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore module state after hot-reload"""
        self._step_count = state.get('step_count', 0)
        self._health_status = state.get('health_status', 'OK')
        self._last_error = state.get('last_error')
        self._last_thesis = state.get('last_thesis', '')
        
        if 'module_state' in state:
            self._set_module_state(state['module_state'])
            
        self.logger.info(
            format_operator_message(
                "📥", "STATE RESTORED",
                instrument=self.__class__.__name__,
                details=f"Step {self._step_count}",
                context="hot_reload"
            )
        )
    
    def _get_module_state(self) -> Optional[Dict[str, Any]]:
        """Override to provide module-specific state"""
        return None
    
    def _set_module_state(self, module_state: Dict[str, Any]) -> None:
        """Override to restore module-specific state"""
        pass
    
    # ─────────────────────────────────────────────────────────────
    # Other methods remain the same for compatibility
    # ─────────────────────────────────────────────────────────────
    
    def _initialize_module_state(self):
        """Override to initialize module-specific state"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset module to initial state"""
        self._step_count = 0
        self._health_history.clear()
        self._performance_metrics.clear()
        self._update_health_status("OK")
        
        self.logger.info(
            format_operator_message(
                "🔄", "MODULE RESET",
                instrument=self.__class__.__name__,
                context="state_management"
            )
        )
    
    @abstractmethod
    def get_observation_components(self, info_bus: InfoBus) -> np.ndarray:
        """Get observation from InfoBus data"""
        return self._get_observation_impl(info_bus)
    
    @abstractmethod
    def _get_observation_impl(self, info_bus: InfoBus) -> np.ndarray:
        """Extract observation features from InfoBus"""
        pass
    
    # Health and performance methods remain the same
    def _update_health_status(self, status: str, error: Optional[str] = None):
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
            self.logger.warning(
                format_operator_message(
                    "⚠️", "HEALTH DEGRADED",
                    instrument=self.__class__.__name__,
                    details=f"{status}: {error}",
                    context="health_monitoring"
                )
            )
    
    def _perform_health_check(self, info_bus: InfoBus):
        try:
            checks = {
                'data_quality': self._check_data_quality(info_bus),
                'performance': self._check_performance(),
                'state_integrity': self._check_state_integrity()
            }
            
            failed = [k for k, v in checks.items() if not v]
            
            if failed:
                self._update_health_status("DEGRADED", f"Failed: {', '.join(failed)}")
            else:
                self._update_health_status("OK")
                
        except Exception as e:
            self._update_health_status("ERROR", str(e))
    
    def _check_data_quality(self, info_bus: InfoBus) -> bool:
        quality = validate_info_bus(info_bus)
        return quality.score >= 70
    
    def _check_performance(self) -> bool:
        if self.config.performance_tracking and self._audit_tracker:
            perf = self._audit_tracker.get_module_performance()
            module_perf = perf.get(self.__class__.__name__, {})
            avg_time = module_perf.get('avg_time_ms', 0)
            return avg_time < self.metadata.timeout_ms
        return True
    
    def _check_state_integrity(self) -> bool:
        return True
    
    def _update_performance_metric(self, name: str, value: float):
        if name not in self._performance_metrics:
            self._performance_metrics[name] = deque(maxlen=100)
        self._performance_metrics[name].append(value)
    
    def get_health_status(self) -> Dict[str, Any]:
        recent_errors = [
            h for h in self._health_history 
            if h['status'] != 'OK'
        ]
        
        performance_summary = {}
        if self._audit_tracker:
            perf = self._audit_tracker.get_module_performance()
            performance_summary = perf.get(self.__class__.__name__, {})
        
        return {
            "status": self._health_status,
            "module": self.__class__.__name__,
            "version": self.metadata.version,
            "step_count": self._step_count,
            "recent_errors": len(recent_errors),
            "last_error": self._last_error,
            "performance": performance_summary
        }

# ═══════════════════════════════════════════════════════════════════
# LEGACY MODULE CLASS (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════

class Module(BaseModule):
    """Legacy Module class - redirects to BaseModule"""
    pass