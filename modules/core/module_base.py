# ─────────────────────────────────────────────────────────────
# File: modules/core/module_base.py
# [ROCKET] PRODUCTION-READY SmartInfoBus Module Base System
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# FIXED: Abstract method enforcement, complete docstrings, state management
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
import time
import logging
import numpy as np
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast
from collections import deque
from functools import wraps
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE MODULE METADATA
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModuleMetadata:
    """
    PRODUCTION-GRADE module metadata with comprehensive validation.
    Military-grade specifications for zero-failure tolerance.
    """
    name: str
    provides: List[str]
    requires: List[str]
    version: str = "1.0.0"
    category: str = "general"
    description: str = ""
    is_voting_member: bool = False
    hot_reload: bool = True
    explainable: bool = True
    timeout_ms: int = 100
    priority: int = 0
    min_confidence: float = 0.0
    max_retries: int = 3
    critical: bool = False
    dependencies: List[str] = field(default_factory=list)
    thesis_required: bool = False
    health_monitoring: bool = False
    performance_tracking: bool = False
    error_handling: bool = False
    
    # Valid categories for modules
    VALID_CATEGORIES = [
        'auditing',    # Auditing and compliance
        'core',        # Core infrastructure
        'external',    # External integrations
        'features',    # Feature engineering
        'market',      # Market data processing
        'memory',      # Memory/state management
        'meta',        # Meta-analysis
        'models',      # Model implementations
        'monitoring',  # System monitoring
        'position',    # Position management
        'reward',      # Reward calculation
        'risk',        # Risk management
        'simulation',  # Simulation modules
        'strategy',    # Trading strategy
        'trading_modes', # Trading mode implementations
        'utils',       # Utility functions
        'visualization', # Visualization modules
        'voting',      # Voting/consensus
        'general'      # General purpose
    ]
    
    def __post_init__(self):
        """CRITICAL: Validate metadata integrity"""
        errors = []
        
        # Name validation
        if not self.name or not isinstance(self.name, str):
            errors.append("Module name must be non-empty string")
        
        if not self.name.replace('_', '').replace('-', '').isalnum():
            errors.append("Module name must be alphanumeric with underscores/hyphens only")
        
        # Provides validation
        if not self.provides or not isinstance(self.provides, list):
            errors.append("Module must provide at least one output")
        
        for output in self.provides:
            if not isinstance(output, str) or not output:
                errors.append(f"Invalid output specification: {output}")
        
        # Requires validation
        if not isinstance(self.requires, list):
            errors.append("Requires must be a list")
        
        for req in self.requires:
            if not isinstance(req, str) or not req:
                errors.append(f"Invalid requirement specification: {req}")
        
        # Timeout validation
        if self.timeout_ms <= 0 or self.timeout_ms > 30000:
            errors.append("Timeout must be between 1ms and 30000ms")
        
        # Confidence validation
        if not 0 <= self.min_confidence <= 1:
            errors.append("Min confidence must be between 0 and 1")
        
        # Category validation
        if self.category not in self.VALID_CATEGORIES:
            errors.append(f"Category must be one of: {self.VALID_CATEGORIES}")
        
        # Version validation
        if not self._validate_version(self.version):
            errors.append(f"Invalid version format: {self.version} (use semantic versioning)")
        
        if errors:
            raise ValueError(f"Module metadata validation failed for {self.name}: {errors}")
    
    def _validate_version(self, version: str) -> bool:
        """Validate semantic version format"""
        try:
            parts = version.split('.')
            if len(parts) != 3:
                return False
            return all(part.isdigit() for part in parts)
        except:
            return False
    
    def validate_compatibility(self, other: 'ModuleMetadata') -> List[str]:
        """Validate compatibility with another module"""
        issues = []
        
        # Check for output conflicts
        common_outputs = set(self.provides) & set(other.provides)
        if common_outputs:
            issues.append(f"Output conflict with {other.name}: {common_outputs}")
        
        # Check circular dependencies
        if (self.name in other.requires and other.name in self.requires):
            issues.append(f"Circular dependency with {other.name}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'provides': self.provides,
            'requires': self.requires,
            'version': self.version,
            'category': self.category,
            'description': self.description,
            'is_voting_member': self.is_voting_member,
            'hot_reload': self.hot_reload,
            'explainable': self.explainable,
            'timeout_ms': self.timeout_ms,
            'priority': self.priority,
            'min_confidence': self.min_confidence,
            'max_retries': self.max_retries,
            'critical': self.critical,
            'dependencies': self.dependencies,
            'thesis_required': self.thesis_required,
            'health_monitoring': self.health_monitoring,
            'performance_tracking': self.performance_tracking,
            'error_handling': self.error_handling
        }

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE MODULE DECORATOR
# ═══════════════════════════════════════════════════════════════════

def module(**kwargs):
    """
    PRODUCTION-GRADE decorator for self-registering SmartInfoBus modules.
    
    Usage:
        @module(
            provides=['output1', 'output2'],
            requires=['input1', 'input2'],
            category='market',
            explainable=True,
            hot_reload=True,
            timeout_ms=100,
            priority=10
        )
        class MyModule(BaseModule):
            pass
    
    Args:
        provides: List of data keys this module produces
        requires: List of data keys this module needs
        category: Module category (market, strategy, risk, etc.)
        explainable: Whether module must provide thesis
        hot_reload: Whether module supports hot-reload
        timeout_ms: Maximum execution time
        priority: Execution priority (higher = earlier)
        critical: Whether module is critical to system operation

        thesis_required: Whether module requires thesis generation
        health_monitoring: Whether module supports health monitoring
        performance_tracking: Whether module supports performance tracking
        error_handling: Whether module supports error handling
    """
    def decorator(cls):
        # Validate class inheritance
        if not issubclass(cls, BaseModule):
            raise TypeError(f"Module {cls.__name__} must inherit from BaseModule")
        
        # Create and validate metadata
        try:
            # Filter out decorator-specific arguments that aren't metadata fields
            metadata_kwargs = kwargs.copy()
            name = metadata_kwargs.pop('name', cls.__name__)
            
            # Remove decorator-specific arguments that aren't part of ModuleMetadata
            decorator_args = ['thesis_required', 'health_monitoring', 'performance_tracking', 'error_handling']
            for arg in decorator_args:
                metadata_kwargs.pop(arg, None)
            
            metadata = ModuleMetadata(
                name=name,
                **metadata_kwargs
            )
        except ValueError as e:
            raise ValueError(f"Module {cls.__name__} metadata validation failed: {e}")
        
        # Attach metadata
        setattr(cls, '__module_metadata__', metadata)
        setattr(cls, '__is_smartinfobus_module__', True)
        
        # Validate required methods are properly implemented
        _validate_module_implementation(cls, metadata)
        
        # Add enhanced state management if not present
        _enhance_state_management(cls)
        
        # Add validation methods if not present
        _enhance_validation_methods(cls)
        
        # Add explanation capability if explainable
        if metadata.explainable:
            _enhance_explanation_capability(cls)
        
        # Auto-register with orchestrator
        try:
            from modules.core.module_system import ModuleOrchestrator
            ModuleOrchestrator.register_class(cls)
        except ImportError:
            # Orchestrator not available yet - will be registered during discovery
            pass
        
        # Register with SmartInfoBus
        try:
            from modules.utils.info_bus import InfoBusManager
            smart_bus = InfoBusManager.get_instance()
            smart_bus.register_provider(cls.__name__, metadata.provides)
            smart_bus.register_consumer(cls.__name__, metadata.requires)
        except ImportError:
            # InfoBus not available yet - will be registered during orchestration
            pass
        
        return cls
    
    return decorator

def _validate_module_implementation(cls, metadata: ModuleMetadata):
    """Validate that module properly implements required methods"""
    # Check abstract methods are implemented
    abstract_methods = []
    
    # Check all methods in MRO for abstract methods
    for name in dir(cls):
        try:
            attr = getattr(cls, name)
            if hasattr(attr, '__isabstractmethod__') and attr.__isabstractmethod__:
                abstract_methods.append(name)
        except AttributeError:
            continue
    
    if abstract_methods:
        raise TypeError(
            f"Module {cls.__name__} must implement abstract methods: {abstract_methods}"
        )

def _enhance_state_management(cls):
    """Enhance module with robust state management"""
    if not hasattr(cls, 'get_state'):
        def get_state(self) -> Dict[str, Any]:
            """Get complete module state for persistence"""
            state = {
                'class_name': self.__class__.__name__,
                'module_path': self.__class__.__module__,
                'version': self.__module_metadata__.version,
                'step_count': getattr(self, '_step_count', 0),
                'health_status': getattr(self, '_health_status', 'OK'),
                'last_execution': getattr(self, '_last_execution', 0),
                'error_count': getattr(self, '_error_count', 0),
                'success_count': getattr(self, '_success_count', 0),
                'failure_count': getattr(self, '_failure_count', 0),
                'performance_history': list(getattr(self, '_performance_history', []))[-100:],
                'custom_state': {}
            }
            
            # Allow modules to add custom state
            if hasattr(self, '_get_custom_state'):
                state['custom_state'] = self._get_custom_state()
            
            return state
        cls.get_state = get_state
    
    if not hasattr(cls, 'set_state'):
        def set_state(self, state: Dict[str, Any]):
            """Restore module state from persistence"""
            self._step_count = state.get('step_count', 0)
            self._health_status = state.get('health_status', 'OK')
            self._last_execution = state.get('last_execution', 0)
            self._error_count = state.get('error_count', 0)
            self._success_count = state.get('success_count', 0)
            self._failure_count = state.get('failure_count', 0)
            
            performance = state.get('performance_history', [])
            self._performance_history = deque(performance, maxlen=1000)
            
            # Restore custom state if available
            if 'custom_state' in state and hasattr(self, '_set_custom_state'):
                self._set_custom_state(state['custom_state'])
            
            self.logger.info(
                f"📥 STATE RESTORED: {self.__class__.__name__} "
                f"step {self._step_count}, health {self._health_status}"
            )
        cls.set_state = set_state

def _enhance_validation_methods(cls):
    """Enhance module with comprehensive validation"""
    if not hasattr(cls, 'validate_inputs'):
        def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
            """Validate required inputs are present and valid"""
            for req in self.__module_metadata__.requires:
                if req not in inputs:
                    raise ValueError(f"Missing required input: {req}")
                if inputs[req] is None:
                    raise ValueError(f"Required input {req} cannot be None")
            return True
        cls.validate_inputs = validate_inputs
    
    if not hasattr(cls, 'validate_outputs'):
        def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
            """Validate outputs meet module contract"""
            metadata = self.__module_metadata__
            
            for prov in metadata.provides:
                if prov not in outputs:
                    raise ValueError(f"Missing required output: {prov}")
            
            # Check for thesis if explainable
            if metadata.explainable and '_thesis' not in outputs:
                raise ValueError("Explainable modules must provide '_thesis'")
            
            # Validate confidence if provided
            if '_confidence' in outputs:
                conf = outputs['_confidence']
                if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
                    raise ValueError(f"Invalid confidence value: {conf}")
            
            return True
        cls.validate_outputs = validate_outputs




def _enhance_explanation_capability(cls):
    """Enhance module with explanation generation"""
    if not hasattr(cls, 'explain_decision'):
        def explain_decision(self, decision: Any, context: Dict[str, Any]) -> str:
            """Generate human-readable explanation for decision"""
            from modules.utils.system_utilities import EnglishExplainer
            explainer = EnglishExplainer()
            return explainer.explain_module_decision(
                module_name=self.__class__.__name__,
                decision=decision,
                context=context,
                confidence=context.get('confidence', 0.5)
            )
        cls.explain_decision = explain_decision

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE DECORATORS
# ═══════════════════════════════════════════════════════════════════

def requires(*fields):
    """
    Decorator to validate required inputs.
    
    Args:
        *fields: Required input field names
        
    Example:
        @requires('market_data', 'risk_score')
        async def process(self, inputs):
            # inputs guaranteed to have market_data and risk_score
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, inputs: Dict[str, Any], *args, **kwargs):
            missing = [f for f in fields if f not in inputs or inputs[f] is None]
            if missing:
                from modules.core.error_pinpointer import ErrorPinpointer
                error = ValueError(f"Missing required inputs: {missing}")
                if hasattr(self, 'error_pinpointer'):
                    self.error_pinpointer.analyze_error(error, self.__class__.__name__)
                raise error
            return await func(self, inputs, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(self, inputs: Dict[str, Any], *args, **kwargs):
            missing = [f for f in fields if f not in inputs or inputs[f] is None]
            if missing:
                from modules.core.error_pinpointer import ErrorPinpointer
                error = ValueError(f"Missing required inputs: {missing}")
                if hasattr(self, 'error_pinpointer'):
                    self.error_pinpointer.analyze_error(error, self.__class__.__name__)
                raise error
            return func(self, inputs, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def provides(*fields):
    """
    Decorator to validate provided outputs.
    
    Args:
        *fields: Required output field names
        
    Example:
        @provides('trading_signal', 'confidence')
        async def process(self, inputs):
            # Must return dict with trading_signal and confidence
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            result = await func(self, *args, **kwargs)
            if isinstance(result, dict):
                missing = [f for f in fields if f not in result]
                if missing:
                    from modules.core.error_pinpointer import ErrorPinpointer
                    error = ValueError(f"Missing required outputs: {missing}")
                    if hasattr(self, 'error_pinpointer'):
                        self.error_pinpointer.analyze_error(error, self.__class__.__name__)
                    raise error
            return result
        
        @wraps(func)  
        def sync_wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if isinstance(result, dict):
                missing = [f for f in fields if f not in result]
                if missing:
                    from modules.core.error_pinpointer import ErrorPinpointer
                    error = ValueError(f"Missing required outputs: {missing}")
                    if hasattr(self, 'error_pinpointer'):
                        self.error_pinpointer.analyze_error(error, self.__class__.__name__)
                    raise error
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def with_timeout(timeout_ms: Optional[int] = None):
    """
    Decorator to enforce timeout on module execution.
    
    Args:
        timeout_ms: Timeout in milliseconds (uses module default if None)
        
    Example:
        @with_timeout(500)  # 500ms timeout
        async def process(self, inputs):
            # Must complete within 500ms
    """
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
                from modules.core.error_pinpointer import ErrorPinpointer
                error = TimeoutError(
                    f"{self.__class__.__name__}.{func.__name__} "
                    f"exceeded timeout of {timeout}ms"
                )
                if hasattr(self, 'error_pinpointer'):
                    self.error_pinpointer.analyze_error(error, self.__class__.__name__)
                raise error
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            duration = (time.time() - start) * 1000
            
            timeout = timeout_ms or self.__module_metadata__.timeout_ms
            if duration > timeout:
                self.logger.warning(
                    f"{self.__class__.__name__}.{func.__name__} "
                    f"took {duration:.0f}ms (timeout: {timeout}ms)"
                )
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def with_confidence_threshold(min_confidence: Optional[float] = None):
    """
    Decorator to skip execution if confidence too low.
    
    Args:
        min_confidence: Minimum confidence required (0-1)
        
    Example:
        @with_confidence_threshold(0.7)
        async def process(self, inputs):
            # Only executes if confidence >= 0.7
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, inputs: Dict[str, Any], *args, **kwargs):
            threshold = min_confidence or self.__module_metadata__.min_confidence
            confidence = inputs.get('confidence', 1.0)
            
            if confidence < threshold:
                return {
                    'skipped': True,
                    'reason': f'Confidence {confidence:.2f} below threshold {threshold:.2f}',
                    '_thesis': f'Execution skipped due to low confidence ({confidence:.1%} < {threshold:.1%})'
                }
            
            return await func(self, inputs, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(self, inputs: Dict[str, Any], *args, **kwargs):
            threshold = min_confidence or self.__module_metadata__.min_confidence
            confidence = inputs.get('confidence', 1.0)
            
            if confidence < threshold:
                return {
                    'skipped': True,
                    'reason': f'Confidence {confidence:.2f} below threshold {threshold:.2f}',
                    '_thesis': f'Execution skipped due to low confidence ({confidence:.1%} < {threshold:.1%})'
                }
            
            return func(self, inputs, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def with_retry(max_retries: Optional[int] = None):
    """
    Decorator to retry on failure with exponential backoff.
    
    Args:
        max_retries: Maximum retry attempts
        
    Example:
        @with_retry(3)
        async def process(self, inputs):
            # Will retry up to 3 times on failure
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            retries = max_retries or self.__module_metadata__.max_retries
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries:
                        await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                        self.logger.warning(
                            f"Retry {attempt + 1}/{retries} for {self.__class__.__name__}.{func.__name__}"
                        )
                    else:
                        from modules.core.error_pinpointer import ErrorPinpointer
                        if hasattr(self, 'error_pinpointer'):
                            self.error_pinpointer.analyze_error(e, self.__class__.__name__)
                        raise last_exception
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            retries = max_retries or self.__module_metadata__.max_retries
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries:
                        time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                        self.logger.warning(
                            f"Retry {attempt + 1}/{retries} for {self.__class__.__name__}.{func.__name__}"
                        )
                    else:
                        from modules.core.error_pinpointer import ErrorPinpointer
                        if hasattr(self, 'error_pinpointer'):
                            self.error_pinpointer.analyze_error(e, self.__class__.__name__)
                        raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE BASE MODULE CLASS
# ═══════════════════════════════════════════════════════════════════

class BaseModule(ABC):
    """
    PRODUCTION-GRADE base class for all SmartInfoBus modules.
    
    NASA/MILITARY SPECIFICATIONS:
    - Zero tolerance for errors
    - Comprehensive validation
    - Complete audit trail
    - Hot-reload capability
    - Explainable decisions
    - Performance monitoring
    - Error recovery
    - State persistence
    
    All modules MUST:
    1. Implement _initialize() for setup
    2. Implement async process() for execution
    3. Call validate_inputs() before processing
    4. Call validate_outputs() after processing
    5. Provide '_thesis' in outputs if explainable
    6. Handle errors gracefully with recovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base module with production-grade defaults"""
        # Verify module has metadata
        if not hasattr(self.__class__, '__module_metadata__'):
            raise TypeError(
                f"{self.__class__.__name__} must be decorated with @module"
            )

        # Core references
        self.metadata = getattr(self.__class__, '__module_metadata__', None)
        if self.metadata is None:
            raise AttributeError(
                f"{self.__class__.__name__} is missing required '__module_metadata__' attribute."
            )
        
        # Store configuration
        self.config = config or {}
        
        self.logger = self._setup_logger()

        # State tracking
        self._step_count = 0
        self._health_status = "OK"
        self._last_error = None
        self._last_execution = 0
        self._error_count = 0
        self._performance_history = deque(maxlen=1000)
        
        # Explainability
        if self.metadata.explainable:
            from modules.utils.system_utilities import EnglishExplainer
            self.explainer = EnglishExplainer()
        else:
            self.explainer = None
        
        # Performance tracking
        self._execution_times = deque(maxlen=100)
        self._success_count = 0
        self._failure_count = 0
        
        # Error analysis
        self.error_pinpointer = None
        try:
            from modules.core.error_pinpointer import ErrorPinpointer
            self.error_pinpointer = ErrorPinpointer()
        except ImportError:
            pass
        
        # Initialize module-specific state
        self._initialize()
        
        # Log initialization
        self.logger.info(
            f"[OK] MODULE INITIALIZED: {self.__class__.__name__} "
            f"v{self.metadata.version} ({self.metadata.category})"
        )
    
    def set_config(self, config: Dict[str, Any]):
        """Set module configuration"""
        self.config.update(config)
        self.logger.info(f"Configuration updated for {self.__class__.__name__}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up module logger with rotation support.
        
        Returns:
            Logger instance configured for the module
        """
        try:
            from modules.utils.audit_utils import RotatingLogger
            return cast(logging.Logger, RotatingLogger(
                name=self.__class__.__name__,
                max_lines=5000
            ))
        except Exception:
            # Fallback to standard logger
            logger = logging.getLogger(self.__class__.__name__)
            logger.setLevel(logging.INFO)
            return logger
    
    @abstractmethod
    def _initialize(self):
        """
        Initialize module-specific state.
        
        This method MUST be implemented by all modules to set up:
        - Internal data structures
        - Configuration parameters
        - Connections to external systems
        - Initial state
        
        Example:
            def _initialize(self):
                self.buffer = deque(maxlen=100)
                self.threshold = 0.5
                self.model = self._load_model()
        """
        pass
    
    @abstractmethod
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Main processing method for the module.
        
        This method MUST be implemented by all modules to:
        1. Validate inputs using self.validate_inputs()
        2. Process the data according to module logic
        3. Generate outputs as specified in metadata.provides
        4. Include '_thesis' if module is explainable
        5. Validate outputs using self.validate_outputs()
        
        Args:
            **inputs: Required inputs as specified in metadata.requires
            
        Returns:
            Dict containing:
            - All outputs specified in metadata.provides
            - '_thesis': Human-readable explanation (if explainable)
            - '_confidence': Confidence score 0-1 (optional)
            - Additional metadata prefixed with '_'
            
        Raises:
            ValueError: If inputs are invalid
            TimeoutError: If execution exceeds timeout
            RuntimeError: If processing fails
            
        Example:
            async def process(self, **inputs) -> Dict[str, Any]:
                # Validate inputs
                self.validate_inputs(inputs)
                
                # Process data
                market_data = inputs['market_data']
                signal = self._calculate_signal(market_data)
                
                # Create outputs
                outputs = {
                    'trading_signal': signal,
                    '_confidence': 0.85,
                    '_thesis': self.explain_decision(signal, inputs)
                }
                
                # Validate outputs
                self.validate_outputs(outputs)
                
                return outputs
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate that all required inputs are present and valid.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails with specific reason
        """
        if self.metadata is None:
            raise AttributeError(f"{self.__class__.__name__} is missing required 'metadata' attribute.")
        
        for req in self.metadata.requires:
            if req not in inputs:
                raise ValueError(f"Missing required input: {req}")
            if inputs[req] is None:
                raise ValueError(f"Required input {req} cannot be None")
        
        return True
    
    def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
        """
        Validate that all required outputs are present and valid.
        
        Args:
            outputs: Dictionary of output values
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails with specific reason
        """
        if self.metadata is None:
            raise AttributeError(f"{self.__class__.__name__} is missing required 'metadata' attribute.")
        
        for prov in self.metadata.provides:
            if prov not in outputs:
                raise ValueError(f"Missing required output: {prov}")
        
        # Check for thesis if explainable
        if self.metadata.explainable and '_thesis' not in outputs:
            raise ValueError("Explainable modules must provide '_thesis'")
        
        # Validate confidence if provided
        if '_confidence' in outputs:
            conf = outputs['_confidence']
            if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
                raise ValueError(f"Invalid confidence value: {conf}")
        
        return True

     # ──────────────────────────────────────────────────────────────
    # OPTIONAL CAPABILITIES – confidence scoring & voting
    #   • A module overrides one (or both) of these to opt-in.
    #   • Leaving the stub unchanged means “I don't support it”.
    # ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """
        ASYNC trading action proposal - must be implemented by modules using trading mixin.
        """
        pass

    @abstractmethod
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """
        ASYNC confidence calculation - must be implemented by modules using trading mixin.
        """
        pass
    
    def explain_decision(self, decision: Any, context: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for decision.
        
        Args:
            decision: The decision made by the module
            context: Context information for the decision
            
        Returns:
            Plain English explanation of the decision
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
        """
        Get complete module state for persistence.
        
        Returns:
            Dictionary containing all state information
        """
        if self.metadata is None:
            raise AttributeError(f"{self.__class__.__name__} is missing required 'metadata' attribute.")
        
        state = {
            'class_name': self.__class__.__name__,
            'module_path': self.__class__.__module__,
            'version': self.metadata.version,
            'step_count': self._step_count,
            'health_status': self._health_status,
            'last_error': self._last_error,
            'last_execution': self._last_execution,
            'error_count': self._error_count,
            'success_count': self._success_count,
            'failure_count': self._failure_count,
            'performance_history': list(self._performance_history)[-100:],
            'execution_times': list(self._execution_times),
            'custom_state': {}
        }
        
        # Allow modules to add custom state
        if hasattr(self, '_get_custom_state'):
            state['custom_state'] = self._get_custom_state()
        
        return state
    
    def set_state(self, state: Dict[str, Any]):
        """
        Restore module state from persistence.
        
        Args:
            state: State dictionary to restore
        """
        self._step_count = state.get('step_count', 0)
        self._health_status = state.get('health_status', 'OK')
        self._last_error = state.get('last_error')
        self._last_execution = state.get('last_execution', 0)
        self._error_count = state.get('error_count', 0)
        self._success_count = state.get('success_count', 0)
        self._failure_count = state.get('failure_count', 0)
        
        performance = state.get('performance_history', [])
        self._performance_history = deque(performance, maxlen=1000)
        
        execution_times = state.get('execution_times', [])
        self._execution_times = deque(execution_times, maxlen=100)
        
        # Restore custom state if available
        if 'custom_state' in state and hasattr(self, '_set_custom_state'):
            self._set_custom_state(state['custom_state'])
        
        self.logger.info(
            f"📥 STATE RESTORED: {self.__class__.__name__} "
            f"step {self._step_count}, health {self._health_status}"
        )
    
    def reset(self):
        """Reset module to initial state"""
        self._step_count = 0
        self._health_status = "OK"
        self._last_error = None
        self._last_execution = 0
        self._error_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._performance_history.clear()
        self._execution_times.clear()
        
        # Re-initialize module-specific state
        self._initialize()
        
        self.logger.info(f"[RELOAD] MODULE RESET: {self.__class__.__name__}")

    def _get_custom_state(self) -> Dict[str, Any]:
        """Get custom module state for persistence. Override in subclasses."""
        return {}

    def _set_custom_state(self, state: Dict[str, Any]):
        """Set custom module state from persistence. Override in subclasses."""
        pass

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate state structure. Override for custom validation."""
        # Basic validation - can be overridden by subclasses
        if not isinstance(state, dict):
            return False
        
        # Check for critical fields
        required_fields = ['class_name', 'version']
        for field in required_fields:
            if field not in state:
                return False
        
        return True

    def validate_state_compatibility(self, state: Dict[str, Any]) -> bool:
        """Check if state is compatible with current module version. Override for custom checks."""
        # Basic compatibility check - can be overridden by subclasses
        saved_version = state.get('version', '1.0.0')
        current_version = self.metadata.version if self.metadata else '1.0.0'
        
        # Simple version check - major version must match
        try:
            saved_major = int(saved_version.split('.')[0])
            current_major = int(current_version.split('.')[0])
            return saved_major == current_major
        except:
            return saved_version == current_version

    def reduce_load(self, factor: float = 0.5):
        """Reduce module load/processing. Override in subclasses that support load reduction."""
        self.logger.info(f"Load reduction requested for {self.__class__.__name__} (factor: {factor})")
        # Subclasses should implement actual load reduction logic
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of the module.
        
        Returns:
            Dictionary containing health metrics and status
        """
        if self.metadata is None:
            raise AttributeError(f"{self.__class__.__name__} is missing required 'metadata' attribute.")
        
        avg_time = np.mean(list(self._execution_times)) if self._execution_times else 0
        total_executions = self._success_count + self._failure_count
        error_rate = self._failure_count / max(total_executions, 1)
        
        return {
            'status': self._health_status,
            'module': self.__class__.__name__,
            'version': self.metadata.version,
            'step_count': self._step_count,
            'error_count': self._error_count,
            'last_error': self._last_error,
            'last_execution': self._last_execution,
            'performance': {
                'avg_time_ms': avg_time,
                'max_time_ms': max(self._execution_times) if self._execution_times else 0,
                'min_time_ms': min(self._execution_times) if self._execution_times else 0,
                'error_rate': error_rate,
                'total_executions': total_executions,
                'success_rate': 1 - error_rate
            },
            'is_healthy': self.is_healthy
        }
    
    @property
    def is_healthy(self) -> bool:
        """Check if module is healthy based on multiple criteria"""
        # Health criteria
        is_status_ok = self._health_status == "OK"
        is_error_rate_low = self._error_count < 10  # Less than 10 total errors
        
        total_executions = self._success_count + self._failure_count
        if total_executions > 0:
            error_rate = self._failure_count / total_executions
            is_failure_rate_acceptable = error_rate < 0.1  # Less than 10% failure rate
        else:
            is_failure_rate_acceptable = True
        
        # Check recent performance
        if self._execution_times and len(self._execution_times) > 5:
            recent_times = list(self._execution_times)[-5:]
            avg_recent = np.mean(recent_times)
            timeout_threshold = self.metadata.timeout_ms * 0.8 if self.metadata else 100 * 0.8
            is_performance_acceptable = avg_recent < timeout_threshold
        else:
            is_performance_acceptable = True
        
        return all([
            is_status_ok,
            is_error_rate_low,
            is_failure_rate_acceptable,
            is_performance_acceptable
        ])
    
    def record_execution(self, duration_ms: float, success: bool, error: Optional[str] = None):
        """
        Record execution metrics for monitoring.
        
        Args:
            duration_ms: Execution time in milliseconds
            success: Whether execution was successful
            error: Error message if failed
        """
        self._step_count += 1
        self._last_execution = time.time()
        self._execution_times.append(duration_ms)
        
        record = {
            'step': self._step_count,
            'duration_ms': duration_ms,
            'success': success,
            'error': error,
            'timestamp': self._last_execution
        }
        
        self._performance_history.append(record)
        
        if success:
            self._success_count += 1
            if self._health_status == "DEGRADED":
                # Check if we should recover
                recent_success_rate = self._calculate_recent_success_rate()
                if recent_success_rate > 0.8:  # 80% recent success rate
                    self._health_status = "OK"
                    self.logger.info(f"[OK] Module health recovered: {self.__class__.__name__}")
        else:
            self._failure_count += 1
            self._error_count += 1
            self._last_error = error
            
            # Check if we should degrade health
            recent_success_rate = self._calculate_recent_success_rate()
            if recent_success_rate < 0.5:  # Less than 50% recent success
                self._health_status = "DEGRADED"
                self.logger.warning(f"[WARN] Module health degraded: {self.__class__.__name__}")
            
            # Log error with pinpointer if available
            if error and self.error_pinpointer:
                try:
                    self.error_pinpointer.analyze_error(
                        Exception(error), 
                        self.__class__.__name__
                    )
                except:
                    pass
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate for recent executions"""
        if not self._performance_history:
            return 1.0
        
        recent = list(self._performance_history)[-20:]  # Last 20 executions
        successes = sum(1 for r in recent if r['success'])
        return successes / len(recent)
    
    def __repr__(self):
        if self.metadata is None:
            return f"{self.__class__.__name__}(no metadata)"
        
        return (
            f"{self.__class__.__name__}("
            f"version={self.metadata.version}, "
            f"category={self.metadata.category}, "
            f"health={self._health_status}, "
            f"steps={self._step_count})"
        )