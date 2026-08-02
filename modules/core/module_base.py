# ─────────────────────────────────────────────────────────────
# File: modules/core/module_base.py
# SmartInfoBus Module Base (V1.2, "Navigator+ Startup") — FIXED
#
# Key fixes vs your pasted version:
# - Removed duplicated DI assignments (metrics/breaker/bus/orchestrator were set twice)
# - Kept circuit breaker adapters ONLY in BaseModule (no duplicated/hidden variants)
# - Made decorators safer (wraps, consistent missing-input handling, optional pinpointer hook)
# - Reduced import-time coupling (decorator auto-registration is now explicitly gated)
# - Tightened state/version compatibility checks while remaining tolerant
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, cast

try:
    import numpy as _np

    _HAVE_NP = True
except Exception:
    _HAVE_NP = False
    _np = None  # type: ignore

__all__ = [
    "BaseModule",
    "ModuleMetadata",
    "module",
    "provides",
    "requires",
    "with_confidence_threshold",
    "with_retry",
    "with_timeout",
]

# Module-level cache for rotating loggers (keyed by name:pid)
_ROTATING_LOGGER_CACHE: Dict[str, Any] = {}

# ─────────────────────────────────────────────────────────────
# Tunables / constants (avoid magic numbers)
# ─────────────────────────────────────────────────────────────
PERF_HISTORY_LIMIT = 100
EXEC_TIMES_LIMIT = 100
RECENT_SUCCESS_SAMPLE_SIZE = 20
RECENT_LATENCY_SAMPLE = 5
MAX_INPUT_KEY_LEN = 255

VERSION_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
NAME_VALID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# ─────────────────────────────────────────────────────────────
# Optional exception imports (best-effort, no hard coupling)
# ─────────────────────────────────────────────────────────────
def _InputsNotReady_exc() -> type[Exception]:
    try:
        from modules.core.exceptions import InputsNotReady  # type: ignore

        return InputsNotReady  # type: ignore[return-value]
    except Exception:
        return ValueError


def _ExecutionSkipped_exc() -> type[Exception]:
    try:
        from modules.core.exceptions import ExecutionSkipped  # type: ignore

        return ExecutionSkipped  # type: ignore[return-value]
    except Exception:
        return RuntimeError


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
    # Optional role flags
    is_final_arbiter: bool = False
    readiness_grace_s: float | None = None  # Optional per-module grace window

    VALID_CATEGORIES = [
        "core",
        "executor",
        "external",
        "features",
        "market",
        "memory",
        "meta",
        "models",
        "monitoring",
        "position",
        "reward",
        "risk",
        "strategy",
        "trading_modes",
        "utils",
        "visualization",
        "voting",
        "general",
    ]

    def __post_init__(self) -> None:
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

        if not (1 <= int(self.timeout_ms) <= 30000):
            errors.append("timeout_ms must be between 1 and 30000")

        try:
            mc = float(self.min_confidence)
            if not (0.0 <= mc <= 1.0):
                errors.append("min_confidence must be between 0 and 1")
        except Exception:
            errors.append("min_confidence must be a number between 0 and 1")

        if self.category not in self.VALID_CATEGORIES:
            errors.append(f"category must be one of {self.VALID_CATEGORIES}")

        if not VERSION_SEMVER_RE.match(str(self.version)):
            errors.append(f"version must be semantic (e.g., 1.2.3), got {self.version!r}")

        if self.readiness_grace_s is not None:
            try:
                rg = float(self.readiness_grace_s)
                if rg < 0 or rg > 30:
                    errors.append("readiness_grace_s must be within [0, 30] seconds if set")
                self.readiness_grace_s = rg
            except Exception:
                errors.append("readiness_grace_s must be a number if set")

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
            "name": self.name,
            "provides": self.provides,
            "requires": self.requires,
            "version": self.version,
            "category": self.category,
            "description": self.description,
            "is_voting_member": self.is_voting_member,
            "hot_reload": self.hot_reload,
            "explainable": self.explainable,
            "timeout_ms": self.timeout_ms,
            "priority": self.priority,
            "min_confidence": self.min_confidence,
            "max_retries": self.max_retries,
            "critical": self.critical,
            "dependencies": self.dependencies,
            "thesis_required": self.thesis_required,
            "health_monitoring": self.health_monitoring,
            "performance_tracking": self.performance_tracking,
            "error_handling": self.error_handling,
            "readiness_grace_s": self.readiness_grace_s,
        }


# ─────────────────────────────────────────────────────────────
# Decorator: @module(...)
# ─────────────────────────────────────────────────────────────
def module(**kwargs: Any):
    """
    Decorator to mark a class as a SmartInfoBus module and attach metadata.
    Requires: provides=[...], requires=[...]
    """

    def _decorator(cls: type):
        if not issubclass(cls, BaseModule):
            raise TypeError(f"Module {cls.__name__} must inherit from BaseModule")

        meta_kwargs = dict(kwargs)
        name = meta_kwargs.pop("name", cls.__name__)
        if "provides" not in meta_kwargs or "requires" not in meta_kwargs:
            raise ValueError("@module requires 'provides' and 'requires' lists")

        metadata = ModuleMetadata(name=name, **meta_kwargs)
        cls.__module_metadata__ = metadata
        cls.__is_smartinfobus_module__ = True

        # attach integrity signature (source hash)
        try:
            src = inspect.getsource(cls)
            cls.__module_signature__ = hashlib.sha256(src.encode("utf-8")).hexdigest()
        except Exception:
            cls.__module_signature__ = None

        _validate_module_implementation(cls)

        # Optional import-time auto-registration is explicitly gated to avoid circular imports.
        # Enable only if you truly want “import side effects”:
        #   SMARTINFOBUS_DECORATOR_AUTOREGISTER=1
        if os.getenv("SMARTINFOBUS_DECORATOR_AUTOREGISTER", "0").strip().lower() in {"1", "true", "yes", "on"}:
            # Best-effort: register with InfoBus for discovery
            try:
                from modules.utils.info_bus import InfoBusManager  # type: ignore

                bus = InfoBusManager.get_instance()
                bus.register_capabilities(cls.__name__, provides=metadata.provides, requires=metadata.requires)
                if metadata.name != cls.__name__:
                    bus.register_capabilities(metadata.name, provides=metadata.provides, requires=metadata.requires)
            except Exception:
                pass

        return cls

    return _decorator


def _validate_module_implementation(cls: type) -> None:
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


# ─────────────────────────────────────────────────────────────
# Decorators: requires / provides / timeout / retry / confidence
# ─────────────────────────────────────────────────────────────
def _extract_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort extraction of the input context from either:
    - process(inputs=...)
    - process({..})
    - process(**inputs)
    """
    if "inputs" in kwargs and isinstance(kwargs["inputs"], dict):
        return cast(Dict[str, Any], kwargs["inputs"])
    if args and isinstance(args[0], dict):
        return cast(Dict[str, Any], args[0])
    return cast(Dict[str, Any], kwargs)


def requires(*fields: str):
    def deco(func: Callable):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def _async(self, *args, **kwargs):
                ctx = _extract_context(args, kwargs)
                missing = [f for f in fields if f not in ctx or ctx[f] is None]
                if missing:
                    Err = _InputsNotReady_exc()
                    err = Err(f"missing required inputs: {missing}")
                    _maybe_pinpoint(self, err)
                    raise err
                return await func(self, *args, **kwargs)

            return _async

        @wraps(func)
        def _sync(self, *args, **kwargs):
            ctx = _extract_context(args, kwargs)
            missing = [f for f in fields if f not in ctx or ctx[f] is None]
            if missing:
                Err = _InputsNotReady_exc()
                err = Err(f"missing required inputs: {missing}")
                _maybe_pinpoint(self, err)
                raise err
            return func(self, *args, **kwargs)

        return _sync

    return deco


def provides(*fields: str):
    def deco(func: Callable):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def _async(self, *args, **kwargs):
                result = await func(self, *args, **kwargs)
                _validate_required_outputs(self, result, fields)
                return result

            return _async

        @wraps(func)
        def _sync(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            _validate_required_outputs(self, result, fields)
            return result

        return _sync

    return deco


def _validate_required_outputs(self: Any, result: Any, fields: tuple[str, ...]) -> None:
    if not isinstance(result, dict):
        return
    missing = [f for f in fields if f not in result]
    if missing:
        err = ValueError(f"missing required outputs: {missing}")
        _maybe_pinpoint(self, err)
        raise err


def _maybe_pinpoint(self: Any, err: BaseException) -> None:
    pp = getattr(self, "error_pinpointer", None)
    if not pp:
        return
    try:
        pp.analyze_error(err, self.__class__.__name__)
    except Exception:
        return


def with_timeout(timeout_ms: Optional[int] = None):
    def deco(func: Callable):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def _async(self, *args, **kwargs):
                to = float((timeout_ms or self.metadata.timeout_ms)) / 1000.0
                try:
                    return await asyncio.wait_for(func(self, *args, **kwargs), timeout=to)
                except asyncio.TimeoutError:
                    # optional cleanup hook
                    try:
                        maybe = getattr(self, "cleanup_after_timeout", None)
                        if maybe:
                            if asyncio.iscoroutinefunction(maybe):
                                await maybe()
                            elif callable(maybe):
                                maybe()
                    except Exception:
                        pass
                    raise

            return _async

        @wraps(func)
        def _sync(self, *args, **kwargs):
            # Portable “best-effort” hard timeout using a worker thread.
            import concurrent.futures as _f

            to_sec = float((timeout_ms or self.metadata.timeout_ms)) / 1000.0
            with _f.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(lambda: func(self, *args, **kwargs))
                try:
                    return fut.result(timeout=to_sec)
                except _f.TimeoutError:
                    try:
                        maybe = getattr(self, "cleanup_after_timeout", None)
                        if callable(maybe):
                            maybe()
                    except Exception:
                        pass
                    raise TimeoutError(f"Timeout after {to_sec:.2f}s")

        return _sync

    return deco


def with_confidence_threshold(min_confidence: Optional[float] = None):
    def deco(func: Callable):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def _async(self, *args, **kwargs):
                thr = float(min_confidence if min_confidence is not None else self.metadata.min_confidence)
                ctx = _extract_context(args, kwargs)
                conf = float(ctx.get("confidence", 1.0))
                if conf < thr:
                    Err = _ExecutionSkipped_exc()
                    raise Err(f"confidence {conf:.2f} below threshold {thr:.2f}")
                return await func(self, *args, **kwargs)

            return _async

        @wraps(func)
        def _sync(self, *args, **kwargs):
            thr = float(min_confidence if min_confidence is not None else self.metadata.min_confidence)
            ctx = _extract_context(args, kwargs)
            conf = float(ctx.get("confidence", 1.0))
            if conf < thr:
                Err = _ExecutionSkipped_exc()
                raise Err(f"confidence {conf:.2f} below threshold {thr:.2f}")
            return func(self, *args, **kwargs)

        return _sync

    return deco


def with_retry(max_retries: Optional[int] = None):
    def deco(func: Callable):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def _async(self, *args, **kwargs):
                retries = int(max_retries if max_retries is not None else self.metadata.max_retries)
                last_exc: Optional[BaseException] = None

                for attempt in range(retries + 1):
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as e:
                        last_exc = e
                        if attempt < retries:
                            _log_retry(self, attempt + 1, retries, func.__name__)
                            await asyncio.sleep(0.1 * (2**attempt))
                        else:
                            _maybe_pinpoint(self, e)
                            raise
                raise RuntimeError("unreachable") from last_exc

            return _async

        @wraps(func)
        def _sync(self, *args, **kwargs):
            retries = int(max_retries if max_retries is not None else self.metadata.max_retries)
            last_exc: Optional[BaseException] = None

            for attempt in range(retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < retries:
                        _log_retry(self, attempt + 1, retries, func.__name__)
                        time.sleep(0.1 * (2**attempt))
                    else:
                        _maybe_pinpoint(self, e)
                        raise
            raise RuntimeError("unreachable") from last_exc

        return _sync

    return deco


def _log_retry(self: Any, attempt: int, retries: int, fn_name: str) -> None:
    lg = getattr(self, "logger", None)
    if not lg:
        return
    try:
        lg.warning(f"Retry {attempt}/{retries} for {self.__class__.__name__}.{fn_name}")
    except Exception:
        return


# ─────────────────────────────────────────────────────────────
# Base Module
# ─────────────────────────────────────────────────────────────
class BaseModule(ABC):
    """
    SmartInfoBus Base Module:
    - Delegates to injected breaker (DI) to avoid duplication.
    - Safe, bounded state (deques) + explicit sanitization/validation.
    - Async lifecycle hooks (__aenter__/__aexit__) and timeout cleanup hooks.
    - Pluggable dependencies: {"circuit_breaker": <obj>, "metrics": <obj>, "bus": <bus>, "orchestrator": <obj>}
    - Startup-friendly: warmup(), probe(), self_test() are no-ops by default.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, dependencies: Optional[Dict[str, Any]] = None):
        if not hasattr(self.__class__, "__module_metadata__"):
            raise TypeError(f"{self.__class__.__name__} must be decorated with @module")

        self.metadata: ModuleMetadata = self.__class__.__module_metadata__
        self.config: Dict[str, Any] = dict(config or {})
        self.dependencies: Dict[str, Any] = dict(dependencies or {})

        # logging
        self.logger = self._setup_logger()

        # DI (set exactly once)
        self.metrics: Any = self.dependencies.get("metrics")
        self.breaker: Any = self.dependencies.get("circuit_breaker")
        self.bus: Any = self.dependencies.get("bus")
        self.orchestrator: Any = self.dependencies.get("orchestrator")

        # compute helpers
        self._mean: Callable[[List[float]], float]
        self._percentile: Callable[[List[float], float], float]
        if _HAVE_NP:
            self._mean = lambda xs: float(_np.mean(xs)) if xs else 0.0  # type: ignore
            self._percentile = lambda xs, p: float(_np.percentile(xs, p)) if xs else 0.0  # type: ignore
        else:
            self._mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0

            def _p(xs: List[float], pct: float) -> float:
                if not xs:
                    return 0.0
                s = sorted(xs)
                idx = min(max(int(round((pct / 100.0) * (len(s) - 1))), 0), len(s) - 1)
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

        # explainability helper (optional)
        self.explainer: Any = None
        if self.metadata.explainable:
            try:
                from modules.utils.system_utilities import EnglishExplainer  # type: ignore

                self.explainer = EnglishExplainer()
            except Exception:
                self.explainer = None

        # pinpointer (optional)
        self.error_pinpointer: Any = None
        try:
            from modules.core.error_pinpointer import ErrorPinpointer  # type: ignore

            self.error_pinpointer = ErrorPinpointer(self.orchestrator) if self.orchestrator else ErrorPinpointer()
        except Exception:
            self.error_pinpointer = None

        # best-effort autowiring (optional)
        self._resolve_dependencies()

        # module-specific init
        self._initialize()

        self.logger.info(f"[OK] MODULE INITIALIZED: {self.__class__.__name__} v{self.metadata.version} ({self.metadata.category})")

    # ───── async lifecycle ─────
    async def __aenter__(self):
        await self.initialize_async_resources()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.cleanup_async_resources()

    async def initialize_async_resources(self) -> None:
        return None

    async def cleanup_async_resources(self) -> None:
        return None

    def cleanup_after_timeout(self) -> None:
        return None

    def _cleanup(self) -> None:
        return None

    def __del__(self):
        try:
            self._cleanup()
        except Exception:
            pass

    # ───── DI breaker adapters (supports multiple breaker API shapes) ─────
    def breaker_allow(self) -> bool:
        b = self.breaker
        if not b:
            return True
        try:
            if hasattr(b, "allow") and callable(b.allow):
                return bool(b.allow())
            if hasattr(b, "should_allow_request") and callable(b.should_allow_request):
                return bool(b.should_allow_request())
        except Exception:
            return True
        return True

    def breaker_on_success(self) -> None:
        b = self.breaker
        if not b:
            return
        try:
            if hasattr(b, "on_success") and callable(b.on_success):
                b.on_success()
                return
            if hasattr(b, "record_success") and callable(b.record_success):
                b.record_success()
                return
        except Exception:
            return

    def breaker_on_failure(self) -> None:
        b = self.breaker
        if not b:
            return
        try:
            if hasattr(b, "on_failure") and callable(b.on_failure):
                b.on_failure()
                return
            if hasattr(b, "record_failure") and callable(b.record_failure):
                b.record_failure()
                return
        except Exception:
            return

    # ───── config helpers ─────
    def set_config(self, config: Dict[str, Any]) -> None:
        self.config.update(config)
        self.logger.info(f"Configuration updated for {self.__class__.__name__}")

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    # ───── logger ─────
    def _setup_logger(self) -> logging.Logger:
        try:
            return cast(logging.Logger, BaseModule._get_rotating_logger_cached(self.__class__.__name__))
        except Exception:
            logger = logging.getLogger(self.__class__.__name__)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.propagate = False
            return logger

    @staticmethod
    def _get_rotating_logger_cached(name: str) -> Any:
        try:
            from modules.utils.audit_utils import RotatingLogger  # type: ignore
        except Exception:
            logger = logging.getLogger(f"{name}:{os.getpid()}")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
                logger.addHandler(h)
            logger.propagate = False
            return logger

        cache_key = f"{name}:{os.getpid()}"
        if cache_key not in _ROTATING_LOGGER_CACHE:
            try:
                _ROTATING_LOGGER_CACHE[cache_key] = RotatingLogger(
                    name=name,
                    log_path=f"logs/modules/{name}.log",
                    max_lines=5000,
                    operator_mode=True,
                    plain_english=True,
                )
            except TypeError:
                _ROTATING_LOGGER_CACHE[cache_key] = RotatingLogger(
                    name=name,
                    log_dir="logs/modules",
                    max_lines=5000,
                    operator_mode=True,
                    plain_english=True,
                )
        return _ROTATING_LOGGER_CACHE[cache_key]

    # ───── abstract API ─────
    @abstractmethod
    def _initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def process(self, **inputs) -> Dict[str, Any]:
        raise NotImplementedError

    # ───── validation ─────
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
        if getattr(self.metadata, "thesis_required", False) and "_thesis" not in outputs:
            raise ValueError("explainable modules must provide '_thesis'")
        if "_confidence" in outputs:
            c = outputs["_confidence"]
            if not isinstance(c, (int, float)) or not 0 <= c <= 1:
                raise ValueError(f"invalid confidence value: {c!r}")
        return True

    # ───── optional capabilities ─────
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
                confidence=context.get("confidence", 0.5),
            )
        except Exception:
            return "Explanation unavailable."

    # Startup hooks
    async def warmup(self) -> None:
        return None

    async def probe(self) -> None:
        return None

    async def self_test(self) -> None:
        return None

    async def run_self_test(self) -> None:
        return await self.self_test()

    # ───── state persistence ─────
    def get_state(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "module_path": self.__class__.__module__,
            "version": self.metadata.version,
            "step_count": self._step_count,
            "health_status": self._health_status,
            "last_error": self._last_error,
            "last_execution": self._last_execution,
            "error_count": self._error_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "performance_history": list(self._performance_history),
            "execution_times": list(self._execution_times),
            "custom_state": self._get_custom_state(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._step_count = int(state.get("step_count", 0))
        self._health_status = str(state.get("health_status", "OK"))
        self._last_error = state.get("last_error")
        self._last_execution = float(state.get("last_execution", 0.0))
        self._error_count = int(state.get("error_count", 0))
        self._success_count = int(state.get("success_count", 0))
        self._failure_count = int(state.get("failure_count", 0))
        self._performance_history = deque(state.get("performance_history", []), maxlen=PERF_HISTORY_LIMIT)
        self._execution_times = deque(state.get("execution_times", []), maxlen=EXEC_TIMES_LIMIT)

        cs = state.get("custom_state")
        if isinstance(cs, dict):
            self._set_custom_state(cs)

        self.logger.info(f"STATE RESTORED: {self.__class__.__name__} step={self._step_count} health={self._health_status}")

    def validate_state_compatibility(self, state: Dict[str, Any]) -> bool:
        """
        Best-effort compatibility check:
        - If a version exists, enforce MAJOR match.
        - If no version exists, allow (outer persistence layer may enforce).
        """
        try:
            version_value: Optional[str] = None
            if isinstance(state, dict):
                v = state.get("version")
                if isinstance(v, (str, int)):
                    version_value = str(v)
                else:
                    mi = state.get("module_info")
                    if isinstance(mi, dict):
                        mv = mi.get("version")
                        if isinstance(mv, (str, int)):
                            version_value = str(mv)

            if not version_value:
                return True

            saved_major = int(version_value.split(".")[0])
            current_major = int(self.metadata.version.split(".")[0])
            return saved_major == current_major
        except Exception:
            return True

    def reset(self) -> None:
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
        self.logger.info(f"MODULE RESET: {self.__class__.__name__}")

    # hooks for custom state
    def _get_custom_state(self) -> Dict[str, Any]:
        return {}

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        return None

    # ───── health & metrics ─────
    @property
    def is_healthy(self) -> bool:
        is_status_ok = self._health_status == "OK"
        total_execs = self._success_count + self._failure_count
        error_rate_ok = self._error_count < 10

        failure_rate_ok = True
        if total_execs > 0:
            failure_rate_ok = (self._failure_count / total_execs) < 0.10

        perf_ok = True
        if len(self._execution_times) >= RECENT_LATENCY_SAMPLE:
            recent = list(self._execution_times)[-RECENT_LATENCY_SAMPLE:]
            avg_recent = self._mean(recent)
            perf_ok = avg_recent < (self.metadata.timeout_ms * 0.8)

        return bool(is_status_ok and error_rate_ok and failure_rate_ok and perf_ok)

    def get_health_status(self) -> Dict[str, Any]:
        avg_time = self._mean(list(self._execution_times))
        total = self._success_count + self._failure_count
        error_rate = (self._failure_count / total) if total else 0.0

        return {
            "status": self._health_status,
            "module": self.__class__.__name__,
            "version": self.metadata.version,
            "step_count": self._step_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "last_execution": self._last_execution,
            "performance": {
                "avg_time_ms": avg_time,
                "max_time_ms": max(self._execution_times) if self._execution_times else 0.0,
                "min_time_ms": min(self._execution_times) if self._execution_times else 0.0,
                "error_rate": error_rate,
                "total_executions": total,
                "success_rate": (1.0 - error_rate) if total else 1.0,
            },
            "is_healthy": self.is_healthy,
        }

    def record_execution(self, duration_ms: float, success: bool, error: Optional[str] = None) -> None:
        self._step_count += 1
        self._last_execution = time.time()
        self._execution_times.append(float(duration_ms))
        self._performance_history.append(
            {
                "step": self._step_count,
                "duration_ms": float(duration_ms),
                "success": bool(success),
                "error": error,
                "timestamp": self._last_execution,
            }
        )

        if success:
            self._success_count += 1
            if self._health_status == "DEGRADED":
                recent = list(self._performance_history)[-RECENT_SUCCESS_SAMPLE_SIZE:]
                if recent:
                    succ = sum(1 for r in recent if r.get("success"))
                    if (succ / len(recent)) > 0.8:
                        self._health_status = "OK"
                        self.logger.info(f"Module health recovered: {self.__class__.__name__}")
            self.breaker_on_success()
        else:
            self._failure_count += 1
            self._error_count += 1
            self._last_error = error
            recent = list(self._performance_history)[-RECENT_SUCCESS_SAMPLE_SIZE:]
            if recent:
                succ = sum(1 for r in recent if r.get("success"))
                if (succ / len(recent)) < 0.5:
                    self._health_status = "DEGRADED"
                    self.logger.warning(f"Module health degraded: {self.__class__.__name__}")
            self.breaker_on_failure()

    # ───── load shedding ─────
    def reduce_load(self, factor: float = 0.5) -> None:
        self.logger.info(f"Load reduction requested for {self.__class__.__name__} (factor={factor})")

    # ─────────────────────────────────────────────────────────
    # Dependency resolution (best-effort, no hard coupling)
    # ─────────────────────────────────────────────────────────
    def _resolve_dependencies(self) -> None:
        """
        Light-touch autowiring (optional):
         - bus: modules.utils.info_bus.InfoBusManager instance (if available)
         - orchestrator: modules.core.module_system.ModuleOrchestrator.get_instance() (if available)
         - circuit_breaker: if orchestrator has a registry, adopt breaker for this module
         - error_pinpointer: prefer orchestrator-aware instance
        """
        if os.getenv("SMARTINFOBUS_AUTOWIRE", "1").strip().lower() in {"0", "false", "no", "off"}:
            return

        # Bus
        if self.bus is None:
            try:
                from modules.utils.info_bus import InfoBusManager  # type: ignore

                self.bus = InfoBusManager.get_instance()
            except Exception:
                self.bus = None

        # Orchestrator
        if self.orchestrator is None:
            try:
                from modules.core.module_system import ModuleOrchestrator  # type: ignore

                self.orchestrator = ModuleOrchestrator.get_instance()
            except Exception:
                self.orchestrator = None

        # Circuit breaker from orchestrator registry (if exposed)
        if self.breaker is None and self.orchestrator is not None:
            try:
                name = self.__class__.__name__
                reg = getattr(self.orchestrator, "circuit_breakers", None)
                if isinstance(reg, dict):
                    cb = reg.get(name)
                    if cb is not None:
                        self.breaker = cb
            except Exception:
                pass

        # Error pinpointer with orchestrator context if possible
        if self.error_pinpointer is None:
            try:
                from modules.core.error_pinpointer import ErrorPinpointer  # type: ignore

                self.error_pinpointer = ErrorPinpointer(self.orchestrator) if self.orchestrator else ErrorPinpointer()
            except Exception:
                self.error_pinpointer = None
