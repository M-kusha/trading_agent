# ─────────────────────────────────────────────────────────────
# File: modules/core/core.py
# 🚀 ENHANCED InfoBus-only base module (NO legacy interfaces)
# ─────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
import logging
import numpy as np
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import deque
from functools import wraps

from modules.utils.info_bus import (
    InfoBus, InfoBusManager, InfoBusExtractor, InfoBusUpdater,
    require_info_bus, cache_computation, validate_info_bus
)
from modules.utils.audit_utils import (
    RotatingLogger, InfoBusAuditTracker, audit_module_call,
    format_operator_message, system_audit
)


class ModuleConfig:
    """Standard configuration for all InfoBus modules"""
    def __init__(self, **kwargs):
        # Standard defaults
        self.debug = kwargs.get('debug', True)
        self.max_history = kwargs.get('max_history', 100)
        self.audit_enabled = kwargs.get('audit_enabled', True)
        self.log_rotation_lines = kwargs.get('log_rotation_lines', 2000)
        self.health_check_interval = kwargs.get('health_check_interval', 100)
        self.performance_tracking = kwargs.get('performance_tracking', True)
        self.cache_enabled = kwargs.get('cache_enabled', True)
        
        # Allow custom config
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def info_bus_step(func):
    """Decorator for InfoBus-based step methods"""
    @wraps(func)
    @require_info_bus
    @audit_module_call()
    def wrapper(self, info_bus: InfoBus, **kwargs):
        # Pre-step validation
        if self.config.debug:
            quality = validate_info_bus(info_bus)
            if not quality.is_valid:
                self.logger.warning(
                    f"Step called with invalid InfoBus: {quality.score:.1f}%"
                )
        
        # Check data freshness
        if not InfoBusExtractor.has_fresh_data(info_bus, max_age_seconds=2.0):
            self.logger.warning("InfoBus data is stale")
        
        # Execute step
        start_time = time.perf_counter()
        result = func(self, info_bus, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Update performance metrics
        InfoBusUpdater.add_performance_timing(
            info_bus, self.__class__.__name__, duration_ms
        )
        
        # Post-step health check
        self._step_count += 1
        if self._step_count % self.config.health_check_interval == 0:
            self._perform_health_check(info_bus)
        
        return result
    
    return wrapper


def validate_observation(func):
    """Decorator to validate observation components"""
    @wraps(func)
    def wrapper(self, info_bus: InfoBus):
        obs = func(self, info_bus)
        
        # Ensure numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        # Validate
        if not np.all(np.isfinite(obs)):
            non_finite = np.sum(~np.isfinite(obs))
            self.logger.warning(
                format_operator_message(
                    "🧹", "SANITIZING OBSERVATION",
                    instrument=self.__class__.__name__,
                    details=f"{non_finite} non-finite values",
                    context="data_quality"
                )
            )
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs.astype(np.float32)
    
    return wrapper


class Module(ABC):
    """
    🚀 ENHANCED InfoBus-only base module.
    ALL modules MUST use InfoBus for data exchange.
    NO direct module-to-module communication allowed.
    """
    
    def __init__(self, config: Optional[Union[Dict, ModuleConfig]] = None, **kwargs):
        # Handle config
        if isinstance(config, dict):
            self.config = ModuleConfig(**config, **kwargs)
        elif isinstance(config, ModuleConfig):
            self.config = config
        else:
            self.config = ModuleConfig(**kwargs)
        
        # Core state
        self._initialized = True
        self._health_status = "OK"
        self._last_error = None
        self._step_count = 0
        
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
                context="startup"
            )
        )
    
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
    
    @info_bus_step
    def step(self, info_bus: InfoBus) -> None:
        """
        InfoBus-only step method.
        ALL data MUST come from and go to InfoBus.
        """
        self._step_impl(info_bus)
    
    @abstractmethod
    def _step_impl(self, info_bus: InfoBus) -> None:
        """
        Implement module logic here.
        Extract data from InfoBus, process, write results back.
        """
        pass
    
    @validate_observation
    def get_observation_components(self, info_bus: InfoBus) -> np.ndarray:
        """
        Get observation from InfoBus data.
        ALL modules MUST implement this.
        """
        return self._get_observation_impl(info_bus)
    
    @abstractmethod
    def _get_observation_impl(self, info_bus: InfoBus) -> np.ndarray:
        """
        Extract observation features from InfoBus.
        NO external data access allowed.
        """
        pass
    
    # ─────────────────────────────────────────────────────────────
    # Health & Performance Management
    # ─────────────────────────────────────────────────────────────
    
    def _update_health_status(self, status: str, error: Optional[str] = None):
        """Update module health status"""
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
        """Perform module health check"""
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
        """Check InfoBus data quality"""
        quality = validate_info_bus(info_bus)
        return quality.score >= 70
    
    def _check_performance(self) -> bool:
        """Check module performance"""
        if self.config.performance_tracking and self._audit_tracker:
            perf = self._audit_tracker.get_module_performance()
            module_perf = perf.get(self.__class__.__name__, {})
            
            # Check if average time is reasonable
            avg_time = module_perf.get('avg_time_ms', 0)
            return avg_time < 100  # 100ms threshold
        
        return True
    
    def _check_state_integrity(self) -> bool:
        """Override for module-specific state checks"""
        return True
    
    # ─────────────────────────────────────────────────────────────
    # InfoBus Data Access Helpers
    # ─────────────────────────────────────────────────────────────
    
    def _get_market_data(self, info_bus: InfoBus, instrument: str, 
                        timeframe: str = 'D1') -> Optional[Dict[str, np.ndarray]]:
        """Get market data from InfoBus"""
        return InfoBusExtractor.get_market_data(info_bus, instrument, timeframe)
    
    def _get_cached_feature(self, info_bus: InfoBus, feature_name: str) -> Optional[np.ndarray]:
        """Get cached feature from InfoBus"""
        cached = InfoBusExtractor.get_cached_features(info_bus, feature_name)
        
        if cached is not None:
            self.logger.debug(f"Using cached feature: {feature_name}")
        
        return cached
    
    def _update_feature(self, info_bus: InfoBus, feature_name: str, 
                       feature_data: np.ndarray):
        """Update feature in InfoBus with caching"""
        InfoBusUpdater.update_feature(
            info_bus, feature_name, feature_data, 
            module=self.__class__.__name__
        )
    
    def _add_module_data(self, info_bus: InfoBus, data: Dict[str, Any]):
        """Add module-specific data to InfoBus"""
        InfoBusUpdater.add_module_data(
            info_bus, self.__class__.__name__, data
        )
    
    # ─────────────────────────────────────────────────────────────
    # State Management
    # ─────────────────────────────────────────────────────────────
    
    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        base_state = {
            'class_name': self.__class__.__name__,
            'config': self.config.__dict__,
            'step_count': self._step_count,
            'health_status': self._health_status,
            'last_error': self._last_error
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
        """Restore module state"""
        self._step_count = state.get('step_count', 0)
        self._health_status = state.get('health_status', 'OK')
        self._last_error = state.get('last_error')
        
        if 'module_state' in state:
            self._set_module_state(state['module_state'])
            
        self.logger.info(
            format_operator_message(
                "📥", "STATE RESTORED",
                instrument=self.__class__.__name__,
                details=f"Step {self._step_count}",
                context="state_management"
            )
        )
    
    def _set_module_state(self, module_state: Dict[str, Any]) -> None:
        """Override to restore module-specific state"""
        pass
    
    # ─────────────────────────────────────────────────────────────
    # Committee Voting Interface (if module participates)
    # ─────────────────────────────────────────────────────────────
    
    def propose_action(self, info_bus: InfoBus) -> np.ndarray:
        """
        Propose action based on InfoBus data.
        Override for voting modules.
        """
        action_dim = len(info_bus.get('raw_actions', [0, 0]))
        return np.zeros(action_dim, dtype=np.float32)
    
    def confidence(self, info_bus: InfoBus) -> float:
        """
        Return confidence in proposed action.
        Override for voting modules.
        """
        return 0.5
    
    # ─────────────────────────────────────────────────────────────
    # Performance Helpers
    # ─────────────────────────────────────────────────────────────
    
    def _update_performance_metric(self, name: str, value: float):
        """Update a performance metric"""
        if name not in self._performance_metrics:
            self._performance_metrics[name] = deque(maxlen=100)
        self._performance_metrics[name].append(value)
    
    def _get_performance_metric(self, name: str, default: float = 0.0) -> float:
        """Get latest performance metric"""
        if name in self._performance_metrics and self._performance_metrics[name]:
            return self._performance_metrics[name][-1]
        return default
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
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
            "step_count": self._step_count,
            "recent_errors": len(recent_errors),
            "last_error": self._last_error,
            "performance": performance_summary
        }