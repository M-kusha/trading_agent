# ─────────────────────────────────────────────────────────────
# File: modules/core/module_base.py
# SmartInfoBus Module Base (V1.1)
# - Single-source-of-truth circuit breaking (delegates via DI)
# - Memory-safe deques (no list slicing leaks)
# - Input sanitization + consistent errors + exception chaining
# - Async lifecycle hooks + timeout cleanup
# - Dependency Injection hooks (breaker, metrics, etc.)
# - Minimal/no-op fallbacks so tests/dev don’t choke
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import logging
import re
import time
import inspect
import hashlib
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast, Callable

try:
    import numpy as _np
    _HAVE_NP = True
except Exception:  # numpy is optional
    _HAVE_NP = False
    _np = None  # type: ignore

__all__ = [
    "ModuleMetadata",
    "module",
    "BaseModule",
]

# Module-level cache for rotating loggers (keyed by name:pid)
# Accept either stdlib Logger or custom RotatingLogger; keep typing flexible.
_ROTATING_LOGGER_CACHE: Dict[str, Any] = {}

# ─────────────────────────────────────────────────────────────
# Tunables / constants (no magic numbers)
# ─────────────────────────────────────────────────────────────
PERF_HISTORY_LIMIT = 100            # bounded history kept in state
EXEC_TIMES_LIMIT = 100              # per-module execution window
RECENT_SUCCESS_SAMPLE_SIZE = 20     # used for health trend
RECENT_LATENCY_SAMPLE = 5           # quick perf sanity
MAX_INPUT_KEY_LEN = 255
VERSION_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
NAME_VALID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# ─────────────────────────────────────────────────────────────
# Module Metadata
# ─────────────────────────────────────────────────────────────
@dataclass
class ModuleMetadata:
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

    VALID_CATEGORIES = [
        'auditing', 'core', 'external', 'features', 'market', 'memory', 'meta',
        'models', 'monitoring', 'position', 'reward', 'risk', 'simulation',
        'strategy', 'trading_modes', 'utils', 'visualization', 'voting', 'general'
    ]

    def __post_init__(self):
        errors: List[str] = []

        if not self.name or not isinstance(self.name, str):
            errors.append("module name must be a non-empty string")
        elif not NAME_VALID_RE.match(self.name):
            errors.append("module name must match ^[A-Za-z0-9_-]+$")

        if not self.provides or not isinstance(self.provides, list):
            errors.append("module must provide at least one output")
        else:
            for o in self.provides:
                if not isinstance(o, str) or not o:
                    errors.append(f"invalid provides entry: {o!r}")

        if not isinstance(self.requires, list):
            errors.append("requires must be a list")
        else:
            for r in self.requires:
                if not isinstance(r, str) or not r:
                    errors.append(f"invalid requires entry: {r!r}")

        # de-dupe (preserve order)
        self.provides = list(dict.fromkeys(self.provides))
        self.requires = list(dict.fromkeys(self.requires))
        self.dependencies = list(dict.fromkeys(self.dependencies))

        if not (1 <= self.timeout_ms <= 30000):
            errors.append("timeout_ms must be between 1 and 30000")

        if not (0.0 <= float(self.min_confidence) <= 1.0):
            errors.append("min_confidence must be between 0 and 1")

        if self.category not in self.VALID_CATEGORIES:
            errors.append(f"category must be one of {self.VALID_CATEGORIES}")

        if not VERSION_SEMVER_RE.match(self.version):
            errors.append(f"version must be semantic (e.g., 1.2.3), got {self.version!r}")

        if errors:
            raise ValueError(f"Module metadata validation failed for {self.name}: {errors}")

    def validate_compatibility(self, other: "ModuleMetadata") -> List[str]:
        issues: List[str] = []
        overlap = set(self.provides) & set(other.provides)
        if overlap:
            issues.append(f"output conflict with {other.name}: {sorted(overlap)}")
        if (self.name in other.dependencies) and (other.name in self.dependencies):
            issues.append(f"circular dependency between {self.name} and {other.name}")
        return issues

    def to_dict(self) -> Dict[str, Any]:
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


# ─────────────────────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────────────────────
def module(**kwargs):
    """
    Decorator to mark a class as a SmartInfoBus module and attach metadata.
    Requires: provides=[...], requires=[...]
    """
    def _decorator(cls):
        if not issubclass(cls, BaseModule):
            raise TypeError(f"Module {cls.__name__} must inherit from BaseModule")

        meta_kwargs = kwargs.copy()
        name = meta_kwargs.pop('name', cls.__name__)
        if 'provides' not in meta_kwargs or 'requires' not in meta_kwargs:
            raise ValueError("@module requires 'provides' and 'requires' lists")

        # build metadata
        metadata = ModuleMetadata(name=name, **meta_kwargs)
        setattr(cls, '__module_metadata__', metadata)
        setattr(cls, '__is_smartinfobus_module__', True)

        # attach integrity signature (source hash)
        try:
            src = inspect.getsource(cls)
            setattr(cls, '__module_signature__', hashlib.sha256(src.encode('utf-8')).hexdigest())
        except Exception:
            setattr(cls, '__module_signature__', None)

        # validate implementation (ensure no abstract leftovers)
        _validate_module_implementation(cls)

        # auto-enhancements (only if missing)
        _enhance_state_management(cls)
        _enhance_validation_methods(cls)
        if metadata.explainable:
            _enhance_explanation_capability(cls)

        # best-effort registration (no hard dependency)
        try:
            from modules.core.module_system import ModuleOrchestrator
            ModuleOrchestrator.register_class(cls)
        except (ImportError, AttributeError):
            pass
        except Exception:
            # avoid killing import-time on unexpected envs
            pass

        try:
            from modules.utils.info_bus import InfoBusManager
            bus = InfoBusManager.get_instance()
            bus.register_provider(cls.__name__, metadata.provides)
            bus.register_consumer(cls.__name__, metadata.requires)
        except (ImportError, AttributeError):
            pass
        except Exception:
            pass

        return cls
    return _decorator


def _validate_module_implementation(cls, metadata: ModuleMetadata | None = None) -> None:
    abstract_methods: List[str] = []
    for name in dir(cls):
        try:
            attr = getattr(cls, name)
            if getattr(attr, "__isabstractmethod__", False):
                abstract_methods.append(name)
        except Exception:
            continue
    if abstract_methods:
        raise TypeError(f"Module {cls.__name__} must implement abstract methods: {abstract_methods}")


def _enhance_state_management(cls):
    if not hasattr(cls, 'get_state'):
        def get_state(self) -> Dict[str, Any]:
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
                'performance_history': list(getattr(self, '_performance_history', deque(maxlen=PERF_HISTORY_LIMIT))),
                'execution_times': list(getattr(self, '_execution_times', deque(maxlen=EXEC_TIMES_LIMIT))),
                'custom_state': {}
            }
            if hasattr(self, '_get_custom_state'):
                state['custom_state'] = self._get_custom_state()
            return state
        cls.get_state = get_state  # type: ignore[attr-defined]

    if not hasattr(cls, 'set_state'):
        def set_state(self, state: Dict[str, Any]):
            self._step_count = state.get('step_count', 0)
            self._health_status = state.get('health_status', 'OK')
            self._last_execution = state.get('last_execution', 0)
            self._error_count = state.get('error_count', 0)
            self._success_count = state.get('success_count', 0)
            self._failure_count = state.get('failure_count', 0)

            # bounded deques (no leak)
            self._performance_history = deque(state.get('performance_history', []), maxlen=PERF_HISTORY_LIMIT)
            self._execution_times = deque(state.get('execution_times', []), maxlen=EXEC_TIMES_LIMIT)

            if 'custom_state' in state and hasattr(self, '_set_custom_state'):
                self._set_custom_state(state['custom_state'])
            if hasattr(self, 'logger'):
                self.logger.info(f"📥 STATE RESTORED: {self.__class__.__name__} step {self._step_count}, health {self._health_status}")
        cls.set_state = set_state  # type: ignore[attr-defined]


def _enhance_validation_methods(cls):
    if not hasattr(cls, 'validate_inputs'):
        def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
            # key sanity (security hardening)
            for k in inputs.keys():
                if not isinstance(k, str) or len(k) > MAX_INPUT_KEY_LEN or k.startswith("__"):
                    raise ValueError(f"invalid input key: {k!r}")
            for req in self.__module_metadata__.requires:
                if req not in inputs:
                    raise ValueError(f"missing required input: {req}")
                if inputs[req] is None:
                    raise ValueError(f"required input {req} cannot be None")
            return True
        cls.validate_inputs = validate_inputs  # type: ignore[attr-defined]

    if not hasattr(cls, 'validate_outputs'):
        def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
            md = self.__module_metadata__
            for prov in md.provides:
                if prov not in outputs:
                    raise ValueError(f"missing required output: {prov}")
            if getattr(md, 'thesis_required', False) and '_thesis' not in outputs:
                raise ValueError("explainable modules must provide '_thesis'")
            if '_confidence' in outputs:
                c = outputs['_confidence']
                if not isinstance(c, (int, float)) or not 0 <= c <= 1:
                    raise ValueError(f"invalid confidence value: {c!r}")
            return True
        cls.validate_outputs = validate_outputs  # type: ignore[attr-defined]


def _enhance_explanation_capability(cls):
    if not hasattr(cls, 'explain_decision'):
        def explain_decision(self, decision: Any, context: Dict[str, Any]) -> str:
            try:
                from modules.utils.system_utilities import EnglishExplainer
                explainer = EnglishExplainer()
                return explainer.explain_module_decision(
                    module_name=self.__class__.__name__,
                    decision=decision,
                    context=context,
                    confidence=context.get('confidence', 0.5)
                )
            except Exception:
                return "Explanation unavailable."
        cls.explain_decision = explain_decision  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────
# Decorators: requires / provides / timeout / retry / confidence
# ─────────────────────────────────────────────────────────────
def requires(*fields: str):
    def deco(func: Callable):
        async def _async(self, *args, **kwargs):
            # normalize inputs without mutating call
            if 'inputs' in kwargs and isinstance(kwargs['inputs'], dict):
                ctx = kwargs['inputs']
            elif args and isinstance(args[0], dict):
                ctx = args[0]
            else:
                ctx = kwargs
            missing = [f for f in fields if f not in ctx or ctx[f] is None]
            if missing:
                err = ValueError(f"missing required inputs: {missing}")
                if getattr(self, 'error_pinpointer', None):
                    try:
                        self.error_pinpointer.analyze_error(err, self.__class__.__name__)
                    except Exception:
                        pass
                raise err
            return await func(self, *args, **kwargs)

        def _sync(self, *args, **kwargs):
            if 'inputs' in kwargs and isinstance(kwargs['inputs'], dict):
                ctx = kwargs['inputs']
            elif args and isinstance(args[0], dict):
                ctx = args[0]
            else:
                ctx = kwargs
            missing = [f for f in fields if f not in ctx or ctx[f] is None]
            if missing:
                err = ValueError(f"missing required inputs: {missing}")
                if getattr(self, 'error_pinpointer', None):
                    try:
                        self.error_pinpointer.analyze_error(err, self.__class__.__name__)
                    except Exception:
                        pass
                raise err
            return func(self, *args, **kwargs)

        return _async if asyncio.iscoroutinefunction(func) else _sync
    return deco


def provides(*fields: str):
    def deco(func: Callable):
        async def _async(self, *args, **kwargs):
            try:
                result = await func(self, *args, **kwargs)
            except Exception as e:
                raise e
            if isinstance(result, dict):
                missing = [f for f in fields if f not in result]
                if missing:
                    err = ValueError(f"missing required outputs: {missing}")
                    if getattr(self, 'error_pinpointer', None):
                        try:
                            self.error_pinpointer.analyze_error(err, self.__class__.__name__)
                        except Exception:
                            pass
                    raise err
            return result

        def _sync(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                raise e
            if isinstance(result, dict):
                missing = [f for f in fields if f not in result]
                if missing:
                    err = ValueError(f"missing required outputs: {missing}")
                    if getattr(self, 'error_pinpointer', None):
                        try:
                            self.error_pinpointer.analyze_error(err, self.__class__.__name__)
                        except Exception:
                            pass
                    raise err
            return result

        return _async if asyncio.iscoroutinefunction(func) else _sync
    return deco


def with_timeout(timeout_ms: Optional[int] = None):
    def deco(func: Callable):
        async def _async(self, *args, **kwargs):
            to = (timeout_ms or self.__module_metadata__.timeout_ms) / 1000.0
            try:
                return await asyncio.wait_for(func(self, *args, **kwargs), timeout=to)
            except asyncio.TimeoutError as e:
                # best-effort cleanup hook
                try:
                    maybe = getattr(self, "cleanup_after_timeout", None)
                    if maybe:
                        if asyncio.iscoroutinefunction(maybe):
                            await maybe()
                        else:
                            maybe()
                except Exception:
                    pass
                raise e

        def _sync(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            duration_ms = (time.time() - start) * 1000.0
            to_ms = timeout_ms or self.__module_metadata__.timeout_ms
            if duration_ms > to_ms and getattr(self, 'logger', None):
                self.logger.warning(
                    f"{self.__class__.__name__}.{func.__name__} took {duration_ms:.0f}ms (timeout: {to_ms}ms)"
                )
            return result

        return _async if asyncio.iscoroutinefunction(func) else _sync
    return deco


def with_confidence_threshold(min_confidence: Optional[float] = None):
    def deco(func: Callable):
        async def _async(self, *args, **kwargs):
            thr = float(min_confidence or getattr(self.__module_metadata__, 'min_confidence', 0.0))
            if 'inputs' in kwargs and isinstance(kwargs['inputs'], dict):
                ctx = kwargs['inputs']
            elif args and isinstance(args[0], dict):
                ctx = args[0]
            else:
                ctx = kwargs
            conf = float(ctx.get('confidence', 1.0))
            if conf < thr:
                return {
                    'skipped': True,
                    'reason': f'confidence {conf:.2f} below threshold {thr:.2f}',
                    '_thesis': f'Execution skipped due to low confidence ({conf:.1%} < {thr:.1%})'
                }
            return await func(self, *args, **kwargs)

        def _sync(self, *args, **kwargs):
            thr = float(min_confidence or getattr(self.__module_metadata__, 'min_confidence', 0.0))
            if 'inputs' in kwargs and isinstance(kwargs['inputs'], dict):
                ctx = kwargs['inputs']
            elif args and isinstance(args[0], dict):
                ctx = args[0]
            else:
                ctx = kwargs
            conf = float(ctx.get('confidence', 1.0))
            if conf < thr:
                return {
                    'skipped': True,
                    'reason': f'confidence {conf:.2f} below threshold {thr:.2f}',
                    '_thesis': f'Execution skipped due to low confidence ({conf:.1%} < {thr:.1%})'
                }
            return func(self, *args, **kwargs)

        return _async if asyncio.iscoroutinefunction(func) else _sync
    return deco


def with_retry(max_retries: Optional[int] = None):
    def deco(func: Callable):
        async def _async(self, *args, **kwargs):
            retries = int(max_retries or self.__module_metadata__.max_retries)
            last_exc: Optional[BaseException] = None
            for attempt in range(retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:  # normal exception chain
                    last_exc = e
                    if attempt < retries:
                        if getattr(self, 'logger', None):
                            self.logger.warning(f"Retry {attempt + 1}/{retries} for {self.__class__.__name__}.{func.__name__}")
                        await asyncio.sleep(0.1 * (2 ** attempt))
                    else:
                        if getattr(self, 'error_pinpointer', None):
                            try:
                                self.error_pinpointer.analyze_error(e, self.__class__.__name__)
                            except Exception:
                                pass
                        raise last_exc
            # should never reach
            raise RuntimeError("unreachable")
        def _sync(self, *args, **kwargs):
            retries = int(max_retries or self.__module_metadata__.max_retries)
            last_exc: Optional[BaseException] = None
            for attempt in range(retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < retries:
                        if getattr(self, 'logger', None):
                            self.logger.warning(f"Retry {attempt + 1}/{retries} for {self.__class__.__name__}.{func.__name__}")
                        time.sleep(0.1 * (2 ** attempt))
                    else:
                        if getattr(self, 'error_pinpointer', None):
                            try:
                                self.error_pinpointer.analyze_error(e, self.__class__.__name__)
                            except Exception:
                                pass
                        raise last_exc
            raise RuntimeError("unreachable")
        return _async if asyncio.iscoroutinefunction(func) else _sync
    return deco


# ─────────────────────────────────────────────────────────────
# Base Module
# ─────────────────────────────────────────────────────────────
class BaseModule(ABC):
    """
    SmartInfoBus Base Module:
    - No local circuit breaker; delegates to injected breaker (DI) to avoid duplication.
    - Safe, bounded state (deques) + explicit sanitization/validation.
    - Async lifecycle hooks (__aenter__/__aexit__) and timeout cleanup.
    - Pluggable dependencies: {"circuit_breaker": <obj>, "metrics": <obj>, ...}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, dependencies: Optional[Dict[str, Any]] = None):
        if not hasattr(self.__class__, '__module_metadata__'):
            raise TypeError(f"{self.__class__.__name__} must be decorated with @module")

        self.metadata: ModuleMetadata = getattr(self.__class__, '__module_metadata__')
        self.config: Dict[str, Any] = dict(config or {})
        self.dependencies: Dict[str, Any] = dict(dependencies or {})

        # logging
        self.logger = self._setup_logger()

        # observability helpers (optional DI)
        self.metrics = self.dependencies.get("metrics")  # e.g., Prometheus/OpenTelemetry registry
        self.breaker = self.dependencies.get("circuit_breaker")  # central breaker instance (optional)

        # compute helpers (cache numpy vs pure python)
        self._mean: Callable[[List[float]], float]
        self._percentile: Callable[[List[float], float], float]
        if _HAVE_NP:
            self._mean = lambda xs: float(_np.mean(xs)) if xs else 0.0  # type: ignore
            self._percentile = lambda xs, p: float(_np.percentile(xs, p)) if xs else 0.0  # type: ignore
        else:
            self._mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
            def _p(xs: List[float], pct: float) -> float:
                if not xs: return 0.0
                s = sorted(xs)
                idx = min(max(int(round((pct/100.0)* (len(s)-1))), 0), len(s)-1)
                return float(s[idx])
            self._percentile = _p

        # state
        self._step_count = 0
        self._health_status = "OK"
        self._last_error: Optional[str] = None
        self._last_execution = 0.0
        self._error_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._performance_history: deque = deque(maxlen=PERF_HISTORY_LIMIT)
        self._execution_times: deque = deque(maxlen=EXEC_TIMES_LIMIT)

        # explainability
        if self.metadata.explainable:
            try:
                from modules.utils.system_utilities import EnglishExplainer
                self.explainer = EnglishExplainer()
            except Exception:
                self.explainer = None
        else:
            self.explainer = None

        # pinpointer (optional)
        self.error_pinpointer = None
        try:
            from modules.core.error_pinpointer import ErrorPinpointer
            self.error_pinpointer = ErrorPinpointer()
        except Exception:
            pass

        # resolve optional deps (no hard coupling)
        self._resolve_dependencies()

        # module-specific init
        self._initialize()

        self.logger.info(f"[OK] MODULE INITIALIZED: {self.__class__.__name__} v{self.metadata.version} ({self.metadata.category})")

    # ───── lifecycle (async context) ─────
    async def __aenter__(self):
        await self.initialize_async_resources()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.cleanup_async_resources()

    async def initialize_async_resources(self):
        """Override if module opens async resources (sockets, pools)."""
        return None

    async def cleanup_async_resources(self):
        """Override to cleanup async resources."""
        return None

    def cleanup_after_timeout(self):
        """Optional: called by with_timeout() when async task times out."""
        return None

    def _cleanup(self) -> None:
        """Optional finalizer hook for subclasses. Called from __del__."""
        return None

    def __del__(self):
        """Best-effort finalizer; cannot rely on ordering at interpreter shutdown."""
        try:
            self._cleanup()
        except Exception:
            pass

    # ───── DI breaker adapters (no local breaker) ─────
    def breaker_allow(self) -> bool:
        try:
            return bool(self.breaker.allow()) if self.breaker else True
        except Exception:
            return True

    def breaker_on_success(self) -> None:
        try:
            if self.breaker:
                self.breaker.on_success()
        except Exception:
            pass

    def breaker_on_failure(self) -> None:
        try:
            if self.breaker:
                self.breaker.on_failure()
        except Exception:
            pass

    # ───── config helpers ─────
    def set_config(self, config: Dict[str, Any]):
        self.config.update(config)
        self.logger.info(f"Configuration updated for {self.__class__.__name__}")

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    

    # ───── logger ─────
    def _setup_logger(self) -> logging.Logger:
        """
        Use a process-scoped cached RotatingLogger so repeated instantiations
        of the same module don't start new log sessions / banners.
        """
        try:
            return cast(logging.Logger, BaseModule._get_rotating_logger_cached(self.__class__.__name__))
        except Exception:
            logger = logging.getLogger(self.__class__.__name__)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.propagate = False
            return logger


    @staticmethod
    def _get_rotating_logger_cached(name: str) -> Any:
        """
        Return a cached RotatingLogger (or stdlib logger fallback) so we don't
        re-initialize the same log session repeatedly. Keyed by (name, pid).
        """
        try:
            from modules.utils.audit_utils import RotatingLogger  # type: ignore
        except Exception:
            import logging, os
            logger = logging.getLogger(f"{name}:{os.getpid()}")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
                logger.addHandler(h)
            logger.propagate = False
            return logger

        import os
        cache_key = f"{name}:{os.getpid()}"
        cache = _ROTATING_LOGGER_CACHE

        if cache_key not in cache:
            # Handle both RotatingLogger signatures (log_path vs log_dir)
            try:
                cache[cache_key] = RotatingLogger(
                    name=name,
                    log_path=f"logs/modules/{name}.log",
                    max_lines=5000,
                    operator_mode=True,
                    plain_english=True,
                )
            except TypeError:
                cache[cache_key] = RotatingLogger(
                    name=name,
                    log_dir="logs/modules",
                    max_lines=5000,
                    operator_mode=True,
                    plain_english=True,
                )
        return cache[cache_key]

    # ───── abstract API ─────
    @abstractmethod
    def _initialize(self):
        """Set up internal data structures, configuration, external connections."""
        raise NotImplementedError

    @abstractmethod
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Must:
          1) self.validate_inputs(inputs)
          2) compute outputs matching metadata.provides
          3) include '_thesis' if explainable/thesis_required
          4) self.validate_outputs(outputs)
        """
        raise NotImplementedError

    # ───── validation (base implementations kept for clarity + custom overrides) ─────
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        for k in inputs.keys():
            if not isinstance(k, str) or len(k) > MAX_INPUT_KEY_LEN or k.startswith("__"):
                raise ValueError(f"invalid input key: {k!r}")
        for req in self.metadata.requires:
            if req not in inputs:
                raise ValueError(f"missing required input: {req}")
            if inputs[req] is None:
                raise ValueError(f"required input {req} cannot be None")
        return True

    def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
        for prov in self.metadata.provides:
            if prov not in outputs:
                raise ValueError(f"missing required output: {prov}")
        if getattr(self.metadata, 'thesis_required', False) and '_thesis' not in outputs:
            raise ValueError("explainable modules must provide '_thesis'")
        if '_confidence' in outputs:
            c = outputs['_confidence']
            if not isinstance(c, (int, float)) or not 0 <= c <= 1:
                raise ValueError(f"invalid confidence value: {c!r}")
        return True

    # ───── optional capabilities (no-op defaults) ─────
    async def propose_action(self, **inputs) -> Optional[Dict[str, Any]]:
        return None

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> Optional[float]:
        return None

    def explain_decision(self, decision: Any, context: Dict[str, Any]) -> str:
        if not self.explainer:
            return "Module is not configured as explainable."
        try:
            return self.explainer.explain_module_decision(
                module_name=self.__class__.__name__,
                decision=decision,
                context=context,
                confidence=context.get('confidence', 0.5)
            )
        except Exception:
            return "Explanation unavailable."

    # ───── state persistence ─────
    def get_state(self) -> Dict[str, Any]:
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
            'performance_history': list(self._performance_history),
            'execution_times': list(self._execution_times),
            'custom_state': {}
        }
        if hasattr(self, '_get_custom_state'):
            state['custom_state'] = self._get_custom_state()
        return state

    def set_state(self, state: Dict[str, Any]):
        self._step_count = state.get('step_count', 0)
        self._health_status = state.get('health_status', 'OK')
        self._last_error = state.get('last_error')
        self._last_execution = state.get('last_execution', 0.0)
        self._error_count = state.get('error_count', 0)
        self._success_count = state.get('success_count', 0)
        self._failure_count = state.get('failure_count', 0)
        self._performance_history = deque(state.get('performance_history', []), maxlen=PERF_HISTORY_LIMIT)
        self._execution_times = deque(state.get('execution_times', []), maxlen=EXEC_TIMES_LIMIT)
        if 'custom_state' in state and hasattr(self, '_set_custom_state'):
            self._set_custom_state(state['custom_state'])
        self.logger.info(f"📥 STATE RESTORED: {self.__class__.__name__} step {self._step_count}, health {self._health_status}")

    def validate_state(self, state: Dict[str, Any]) -> bool:
        if not isinstance(state, dict):
            return False
        for f in ('class_name', 'version'):
            if f not in state:
                return False
        # naive: major version compatibility
        try:
            saved_major = int(str(state.get('version', '1.0.0')).split('.')[0])
            current_major = int(self.metadata.version.split('.')[0])
            if saved_major != current_major:
                return False
        except Exception:
            pass
        return True

    def validate_state_compatibility(self, state: Dict[str, Any]) -> bool:
        try:
            saved_major = int(str(state.get('version', '1.0.0')).split('.')[0])
            current_major = int(self.metadata.version.split('.')[0])
            return saved_major == current_major
        except Exception:
            return True

    def reset(self):
        self._step_count = 0
        self._health_status = "OK"
        self._last_error = None
        self._last_execution = 0.0
        self._error_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._performance_history = deque(maxlen=PERF_HISTORY_LIMIT)
        self._execution_times = deque(maxlen=EXEC_TIMES_LIMIT)
        self._initialize()
        self.logger.info(f"[RELOAD] MODULE RESET: {self.__class__.__name__}")

    # hooks for custom state
    def _get_custom_state(self) -> Dict[str, Any]:
        return {}

    def _set_custom_state(self, state: Dict[str, Any]):
        return None

    # ───── health & metrics ─────
    @property
    def is_healthy(self) -> bool:
        is_status_ok = (self._health_status == "OK")
        total_execs = self._success_count + self._failure_count
        error_rate_ok = (self._error_count < 10)
        failure_rate_ok = True
        if total_execs > 0:
            failure_rate_ok = (self._failure_count / total_execs) < 0.10

        perf_ok = True
        if len(self._execution_times) >= RECENT_LATENCY_SAMPLE:
            recent = list(self._execution_times)[-RECENT_LATENCY_SAMPLE:]
            avg_recent = self._mean(recent)
            perf_ok = avg_recent < (self.metadata.timeout_ms * 0.8)

        return all([is_status_ok, error_rate_ok, failure_rate_ok, perf_ok])

    def get_health_status(self) -> Dict[str, Any]:
        avg_time = self._mean(list(self._execution_times))
        total = self._success_count + self._failure_count
        error_rate = (self._failure_count / total) if total else 0.0
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
                'max_time_ms': max(self._execution_times) if self._execution_times else 0.0,
                'min_time_ms': min(self._execution_times) if self._execution_times else 0.0,
                'error_rate': error_rate,
                'total_executions': total,
                'success_rate': 1 - error_rate if total else 1.0
            },
            'is_healthy': self.is_healthy
        }

    def record_execution(self, duration_ms: float, success: bool, error: Optional[str] = None):
        self._step_count += 1
        self._last_execution = time.time()
        self._execution_times.append(float(duration_ms))
        self._performance_history.append({
            'step': self._step_count,
            'duration_ms': float(duration_ms),
            'success': bool(success),
            'error': error,
            'timestamp': self._last_execution
        })

        if success:
            self._success_count += 1
            if self._health_status == "DEGRADED":
                # quick recovery heuristic
                recent = list(self._performance_history)[-RECENT_SUCCESS_SAMPLE_SIZE:]
                if recent:
                    succ = sum(1 for r in recent if r['success'])
                    if succ / len(recent) > 0.8:
                        self._health_status = "OK"
                        self.logger.info(f"[OK] Module health recovered: {self.__class__.__name__}")
            self.breaker_on_success()
        else:
            self._failure_count += 1
            self._error_count += 1
            self._last_error = error
            recent = list(self._performance_history)[-RECENT_SUCCESS_SAMPLE_SIZE:]
            if recent:
                succ = sum(1 for r in recent if r['success'])
                if succ / len(recent) < 0.5:
                    self._health_status = "DEGRADED"
                    self.logger.warning(f"[WARN] Module health degraded: {self.__class__.__name__}")
            self.breaker_on_failure()

    # ───── load shedding (override in heavy modules) ─────
    def reduce_load(self, factor: float = 0.5):
        self.logger.info(f"Load reduction requested for {self.__class__.__name__} (factor={factor})")

    # ───── DI resolution (best-effort, no hard coupling) ─────
    def _resolve_dependencies(self):
        # Optionally auto-wire from InfoBus if running inside system
        if 'circuit_breaker' not in self.dependencies:
            try:
                from modules.utils.info_bus import InfoBusManager
                bus = InfoBusManager.get_instance()
                # Expect orchestrator to set: bus._circuit_breakers[name] → instance
                # We don’t know our registry name here; leave for orchestrator to inject.
                # This method intentionally does not guess to avoid coupling.
            except Exception:
                pass
