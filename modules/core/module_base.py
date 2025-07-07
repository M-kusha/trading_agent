# ─────────────────────────────────────────────────────────────
# File: modules/core/module_base.py
# 🚀 SmartInfoBus Base Module with Enhanced Decorators
# ─────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from functools import wraps
import inspect
import yaml
from pathlib import Path

from modules.utils.smart_info_bus import SmartInfoBus
from modules.utils.english_explainer import EnglishExplainer


@dataclass
class ModuleMetadata:
    """Enhanced metadata for SmartInfoBus modules"""
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
    min_confidence: float = 0.0
    max_retries: int = 3
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate metadata completeness"""
        errors = []
        
        if not self.name:
            errors.append("Module name is required")
        if not self.provides:
            errors.append("Module must provide at least one output")
        if self.timeout_ms <= 0:
            errors.append("Timeout must be positive")
        if not 0 <= self.min_confidence <= 1:
            errors.append("Min confidence must be between 0 and 1")
            
        return len(errors) == 0, errors


def module(**kwargs):
    """
    Enhanced decorator for SmartInfoBus modules.
    Automatically registers module and adds required functionality.
    """
    def decorator(cls):
        # Create metadata
        cls.__module_metadata__ = ModuleMetadata(
            name=kwargs.get('name', cls.__name__),
            **kwargs
        )
        
        # Validate metadata
        is_valid, errors = cls.__module_metadata__.validate()
        if not is_valid:
            raise ValueError(f"Invalid module metadata for {cls.__name__}: {errors}")
        
        # Add marker for discovery
        cls.__is_smartinfobus_module__ = True
        
        # Enhance class with required methods
        _enhance_module_class(cls)
        
        # Auto-register if orchestrator available
        try:
            from modules.core.module_orchestrator import ModuleOrchestrator
            ModuleOrchestrator.register_class(cls)
        except ImportError:
            pass  # Orchestrator not yet available
        
        return cls
    return decorator


def _enhance_module_class(cls):
    """Add required methods and functionality to module class"""
    
    # Ensure it inherits from BaseModule
    if not issubclass(cls, BaseModule):
        # Inject BaseModule as parent
        cls.__bases__ = (BaseModule,) + cls.__bases__
    
    # Add state management if not present
    if not hasattr(cls, 'get_state'):
        def get_state(self) -> Dict[str, Any]:
            """Default state getter"""
            return {
                'class_name': self.__class__.__name__,
                'version': self.__module_metadata__.version,
                'step_count': getattr(self, '_step_count', 0),
                'health_status': getattr(self, '_health_status', 'OK')
            }
        cls.get_state = get_state
    
    if not hasattr(cls, 'set_state'):
        def set_state(self, state: Dict[str, Any]):
            """Default state setter"""
            self._step_count = state.get('step_count', 0)
            self._health_status = state.get('health_status', 'OK')
        cls.set_state = set_state
    
    # Add validation if not present
    if not hasattr(cls, 'validate_inputs'):
        def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
            """Default input validation"""
            for req in self.__module_metadata__.requires:
                if req not in inputs or inputs[req] is None:
                    raise ValueError(f"Missing required input: {req}")
            return True
        cls.validate_inputs = validate_inputs
    
    # Add explain_decision if explainable but not present
    if cls.__module_metadata__.explainable and not hasattr(cls, 'explain_decision'):
        def explain_decision(self, decision: Any, context: Dict[str, Any]) -> str:
            """Default decision explanation"""
            explainer = EnglishExplainer()
            return explainer.explain_module_decision(
                module_name=self.__class__.__name__,
                decision=decision,
                context=context,
                confidence=context.get('confidence', 0.5)
            )
        cls.explain_decision = explain_decision


def requires(*fields):
    """Decorator to validate required inputs"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, inputs: Dict[str, Any], *args, **kwargs):
            missing = [f for f in fields if f not in inputs]
            if missing:
                raise ValueError(f"Missing required inputs: {missing}")
            return func(self, inputs, *args, **kwargs)
        return wrapper
    return decorator


def provides(*fields):
    """Decorator to validate provided outputs"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if isinstance(result, dict):
                missing = [f for f in fields if f not in result]
                if missing:
                    raise ValueError(f"Missing required outputs: {missing}")
            return result
        return wrapper
    return decorator


def with_timeout(timeout_ms: Optional[int] = None):
    """Decorator to enforce timeout on module execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            timeout = timeout_ms or self.__module_metadata__.timeout_ms
            timeout_sec = timeout / 1000.0
            
            try:
                return await asyncio.wait_for(
                    func(self, *args, **kwargs),
                    timeout=timeout_sec
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"{self.__class__.__name__}.{func.__name__} "
                    f"exceeded timeout of {timeout}ms"
                )
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # For synchronous functions, we can't easily enforce timeout
            # Log warning if execution takes too long
            start = time.time()
            result = func(self, *args, **kwargs)
            duration = (time.time() - start) * 1000
            
            timeout = timeout_ms or self.__module_metadata__.timeout_ms
            if duration > timeout:
                logging.warning(
                    f"{self.__class__.__name__}.{func.__name__} "
                    f"took {duration:.0f}ms (timeout: {timeout}ms)"
                )
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_confidence_threshold(min_confidence: Optional[float] = None):
    """Decorator to skip execution if confidence too low"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, inputs: Dict[str, Any], *args, **kwargs):
            threshold = min_confidence or self.__module_metadata__.min_confidence
            confidence = inputs.get('confidence', 1.0)
            
            if confidence < threshold:
                return {
                    'skipped': True,
                    'reason': f'Confidence {confidence:.2f} below threshold {threshold:.2f}'
                }
            
            return func(self, inputs, *args, **kwargs)
        return wrapper
    return decorator


class BaseModule(ABC):
    """
    Enhanced base class for all SmartInfoBus modules.
    Provides common functionality and enforces standards.
    """
    
    def __init__(self):
        # Verify module has metadata
        if not hasattr(self.__class__, '__module_metadata__'):
            raise TypeError(
                f"{self.__class__.__name__} must be decorated with @module"
            )
        
        self.metadata = self.__class__.__module_metadata__
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # State tracking
        self._step_count = 0
        self._health_status = "OK"
        self._last_error = None
        self._performance_history = []
        
        # Explainability
        self.explainer = EnglishExplainer() if self.metadata.explainable else None
        
        # Initialize module
        self._initialize()
        
    @abstractmethod
    def _initialize(self):
        """Initialize module-specific state"""
        pass
    
    @abstractmethod
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Main processing method for the module.
        Must be implemented by all modules.
        
        Args:
            **inputs: Required inputs as specified in metadata
            
        Returns:
            Dict containing outputs specified in metadata
            Must include '_thesis' for explainable modules
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate all required inputs are present"""
        for req in self.metadata.requires:
            if req not in inputs or inputs[req] is None:
                raise ValueError(f"Missing required input: {req}")
        return True
    
    def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
        """Validate all promised outputs are present"""
        for prov in self.metadata.provides:
            if prov not in outputs:
                raise ValueError(f"Missing required output: {prov}")
        
        # Check for thesis if explainable
        if self.metadata.explainable and '_thesis' not in outputs:
            raise ValueError("Explainable modules must provide '_thesis'")
        
        return True
    
    def explain_decision(self, decision: Any, context: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for decision.
        Override for custom explanations.
        """
        if not self.explainer:
            return "Module is not configured as explainable"
        
        return self.explainer.explain_module_decision(
            module_name=self.__class__.__name__,
            decision=decision,
            context=context,
            confidence=context.get('confidence', 0.5)
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current module state for persistence"""
        return {
            'class_name': self.__class__.__name__,
            'version': self.metadata.version,
            'step_count': self._step_count,
            'health_status': self._health_status,
            'last_error': self._last_error,
            'performance_history': self._performance_history[-100:]  # Last 100
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore module state"""
        self._step_count = state.get('step_count', 0)
        self._health_status = state.get('health_status', 'OK')
        self._last_error = state.get('last_error')
        self._performance_history = state.get('performance_history', [])
    
    def reset(self):
        """Reset module to initial state"""
        self._step_count = 0
        self._health_status = "OK"
        self._last_error = None
        self._performance_history.clear()
        self._initialize()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get module health information"""
        return {
            'status': self._health_status,
            'step_count': self._step_count,
            'last_error': self._last_error,
            'performance': {
                'avg_time_ms': np.mean([p['duration_ms'] for p in self._performance_history[-100:]])
                if self._performance_history else 0,
                'error_rate': sum(1 for p in self._performance_history[-100:] if not p['success']) / 
                            max(len(self._performance_history[-100:]), 1)
            }
        }
    
    def record_execution(self, duration_ms: float, success: bool, error: Optional[str] = None):
        """Record execution metrics"""
        self._step_count += 1
        
        record = {
            'step': self._step_count,
            'duration_ms': duration_ms,
            'success': success,
            'error': error,
            'timestamp': time.time()
        }
        
        self._performance_history.append(record)
        
        # Update health status
        if not success:
            self._last_error = error
            self._health_status = "DEGRADED"
        elif self._health_status == "DEGRADED" and success:
            # Recover after successful execution
            self._health_status = "OK"
    
    @property
    def is_healthy(self) -> bool:
        """Check if module is healthy"""
        return self._health_status == "OK"
    
    @property
    def error_rate(self) -> float:
        """Calculate recent error rate"""
        if not self._performance_history:
            return 0.0
        
        recent = self._performance_history[-100:]
        errors = sum(1 for p in recent if not p['success'])
        return errors / len(recent)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"version={self.metadata.version}, "
            f"category={self.metadata.category}, "
            f"health={self._health_status})"
        )


# Example module implementation
@module(
    provides=['example_output', 'example_confidence'],
    requires=['market_data', 'risk_score'],
    category='example',
    explainable=True,
    timeout_ms=100
)
class ExampleModule(BaseModule):
    """Example SmartInfoBus module implementation"""
    
    def _initialize(self):
        """Initialize module state"""
        self.processing_history = []
    
    @with_timeout()
    @requires('market_data', 'risk_score')
    @provides('example_output', 'example_confidence')
    async def process(self, **inputs) -> Dict[str, Any]:
        """Process inputs and generate outputs"""
        market_data = inputs['market_data']
        risk_score = inputs['risk_score']
        
        # Simulate processing
        await asyncio.sleep(0.01)
        
        # Generate decision
        decision = self._make_decision(market_data, risk_score)
        confidence = self._calculate_confidence(market_data, risk_score)
        
        # Generate explanation
        thesis = self.explain_decision(decision, {
            'market_data': market_data,
            'risk_score': risk_score,
            'confidence': confidence
        })
        
        return {
            'example_output': decision,
            'example_confidence': confidence,
            '_thesis': thesis
        }
    
    def _make_decision(self, market_data: Any, risk_score: float) -> Any:
        """Make example decision"""
        return {'action': 'hold', 'reason': 'example'}
    
    def _calculate_confidence(self, market_data: Any, risk_score: float) -> float:
        """Calculate confidence in decision"""
        return max(0.0, min(1.0, 1.0 - risk_score))