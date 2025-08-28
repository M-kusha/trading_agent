"""
Enhanced Utilities for Modern SmartInfoBus Trading Environment
Zero-legacy, production-ready implementations (hardened)
"""

from __future__ import annotations

import time
import logging
import threading
import importlib
import platform
from logging.handlers import RotatingFileHandler
from functools import wraps, lru_cache
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- helpers

def _now() -> float:
    return time.time()

def _perf_ns() -> int:
    return time.perf_counter_ns()

def _as_float_mb(bytes_val: int) -> float:
    return float(bytes_val) / (1024.0 * 1024.0)

def _as_float_gb(bytes_val: int) -> float:
    return float(bytes_val) / (1024.0 * 1024.0 * 1024.0)

def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


# ---------- system health model

@dataclass
class SystemHealth:
    """System health status"""
    smartinfobus_active: bool = False
    modules_active: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_space_gb: float = 0.0
    last_update: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Optional extras
    total_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_count: int = 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- decorators

def profile_method(func: Callable):
    """
    Performance profiling decorator for environment methods.

    - Works for both sync and async callables
    - Logs only if elapsed > threshold (default 100 ms, or self.profile_slow_ms if present)
    - Stores metrics in SmartInfoBus if available
    """
    is_coro = hasattr(func, "__await__") or getattr(func, "__is_coroutine__", False)

    @wraps(func)
    def _sync_wrapper(self, *args, **kwargs):
        threshold_ms = float(_safe_getattr(self, "profile_slow_ms", 100.0))
        start_ns = _perf_ns()
        try:
            result = func(self, *args, **kwargs)
            elapsed_ms = ( _perf_ns() - start_ns ) / 1_000_000.0

            # Rate-limit spammy logs: once per 1s per function if slow
            if elapsed_ms > threshold_ms and hasattr(self, "logger") and self.logger:
                last_log_ts = getattr(self, f"__pm_last_{func.__name__}", 0.0)
                now = _now()
                if now - last_log_ts > 1.0:
                    self.logger.debug(f"⏱️ {func.__name__} took {elapsed_ms:.1f}ms")
                    setattr(self, f"__pm_last_{func.__name__}", now)

            # Store perf metric (lightweight payload) if SmartInfoBus is available
            sb = _safe_getattr(self, "smart_bus", None)
            if sb:
                try:
                    sb.set(
                        f"performance_{func.__name__}",
                        {"duration_ms": float(elapsed_ms), "timestamp": _now()},
                        module=_safe_getattr(self, "__class__", type("X", (), {})).__name__,
                        thesis=f"Performance metric for {func.__name__}"
                    )
                except Exception:
                    pass

            return result
        except Exception as e:
            elapsed_ms = ( _perf_ns() - start_ns ) / 1_000_000.0
            if hasattr(self, "logger") and self.logger:
                self.logger.error(f"❌ {func.__name__} failed after {elapsed_ms:.1f}ms: {e}")
            raise

    async def _async_wrapper(self, *args, **kwargs):
        # Only used if wrapping an async function
        threshold_ms = float(_safe_getattr(args[0], "profile_slow_ms", 100.0))
        start_ns = _perf_ns()
        try:
            result = await func(*args, **kwargs)
            elapsed_ms = ( _perf_ns() - start_ns ) / 1_000_000.0
            self = args[0]
            if elapsed_ms > threshold_ms and hasattr(self, "logger") and self.logger:
                last_log_ts = getattr(self, f"__pm_last_{func.__name__}", 0.0)
                now = _now()
                if now - last_log_ts > 1.0:
                    self.logger.debug(f"⏱️ {func.__name__} took {elapsed_ms:.1f}ms")
                    setattr(self, f"__pm_last_{func.__name__}", now)

            sb = _safe_getattr(self, "smart_bus", None)
            if sb:
                try:
                    sb.set(
                        f"performance_{func.__name__}",
                        {"duration_ms": float(elapsed_ms), "timestamp": _now()},
                        module=_safe_getattr(self, "__class__", type("X", (), {})).__name__,
                        thesis=f"Performance metric for {func.__name__}"
                    )
                except Exception:
                    pass
            return result
        except Exception as e:
            elapsed_ms = ( _perf_ns() - start_ns ) / 1_000_000.0
            self = args[0]
            if hasattr(self, "logger") and self.logger:
                self.logger.error(f"❌ {func.__name__} failed after {elapsed_ms:.1f}ms: {e}")
            raise

    return _async_wrapper if is_coro else _sync_wrapper


# ---------- imports

@lru_cache(maxsize=64)
def _import_module(module_name: str):
    return importlib.import_module(module_name)

def safe_import(module_name: str, fallback=None):
    """
    Safely import modules or selected attributes with fallback.

    Supports "pkg.module:AttrName" syntax.
    Returns the module/attribute if available, otherwise `fallback`.
    """
    try:
        if ":" in module_name:
            mod_name, attr = module_name.split(":", 1)
            mod = _import_module(mod_name)
            return getattr(mod, attr)
        else:
            # Special-cased returns to keep compatibility with your code
            if module_name == "modules.utils.info_bus":
                try:
                    mod = _import_module(module_name)
                    return getattr(mod, "InfoBusManager", mod)
                except Exception:
                    return fallback
            elif module_name == "modules.utils.audit_utils":
                try:
                    mod = _import_module(module_name)
                    return getattr(mod, "RotatingLogger", mod)
                except Exception:
                    return fallback
            elif module_name == "modules.core.module_system":
                try:
                    mod = _import_module(module_name)
                    return getattr(mod, "ModuleOrchestrator", mod)
                except Exception:
                    return fallback
            else:
                return _import_module(module_name)
    except Exception:
        return fallback


# ---------- logging

def create_enhanced_logger(name: str, log_path: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create enhanced logger with proper formatting and no duplicate handlers.
    - Console handler (human-readable)
    - Optional rotating file handler (safe on Windows & Linux)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.propagate = False

        # Console
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File (rotating) if path provided
        if log_path:
            try:
                path = Path(log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
                file_formatter = logging.Formatter(
                    '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                # Avoid raising; log to console
                logger.warning(f"Could not create file handler: {e}")

        logger.setLevel(level)

    return logger


# ---------- config validation

def validate_trading_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Enhanced trading configuration validation.
    Returns (ok, issues) where issues contains warnings (⚠️) and critical errors (❌).
    """
    issues: List[str] = []

    # Required
    for field in ['initial_balance', 'max_steps', 'instruments', 'max_drawdown']:
        if config.get(field) in (None, "", []):
            issues.append(f"❌ Missing required field: {field}")

    # Types & sanity
    if not isinstance(config.get('instruments', []), (list, tuple)) or not config.get('instruments'):
        issues.append("❌ 'instruments' must be a non-empty list")
    if isinstance(config.get('max_steps', None), (int, float)) and config['max_steps'] <= 0:
        issues.append("❌ 'max_steps' must be > 0")
    if isinstance(config.get('initial_balance', None), (int, float)) and config['initial_balance'] <= 0:
        issues.append("❌ 'initial_balance' must be > 0")

    # Risk
    md = float(config.get('max_drawdown', 0) or 0.0)
    if md > 0.5:
        issues.append("⚠️ Max drawdown > 50% is extremely risky")
    elif md > 0.3:
        issues.append("⚠️ Max drawdown > 30% is very risky")

    mpp = float(config.get('max_position_pct', 0) or 0.0)
    if mpp > 0.3:
        issues.append("⚠️ Position size > 30% per trade is very risky")
    elif mpp > 0.15:
        issues.append("⚠️ Position size > 15% per trade is risky")

    te = float(config.get('max_total_exposure', 0) or 0.0)
    if te > 1.0:
        issues.append("❌ Total exposure > 100% is invalid")
    elif te > 0.5:
        issues.append("⚠️ Total exposure > 50% is very risky")

    # Live trading
    if bool(config.get('live_mode', False)):
        if not bool(config.get('info_bus_enabled', True)):
            issues.append("⚠️ InfoBus strongly recommended for live trading")
        if bool(config.get('debug', False)) and mpp > 0.05:
            issues.append("⚠️ Large positions in live debug mode")
        if float(config.get('initial_balance', 0) or 0.0) < 1000:
            issues.append("⚠️ Very small balance for live trading")

    # Performance/logging
    lrl = int(config.get('log_rotation_lines', 2000) or 2000)
    if lrl > 10000:
        issues.append("⚠️ Very high log rotation may impact performance")

    ms = int(config.get('max_steps', 200) or 200)
    if ms > 1000:
        issues.append("⚠️ Very long episodes may be slow")

    ok = not any(s.startswith("❌") for s in issues)
    if not issues:
        issues.append("✅ Trading config validation passed")
    return ok, issues


# ---------- system status

def get_system_status() -> SystemHealth:
    """Get comprehensive system status (psutil optional, torch optional)."""
    health = SystemHealth(last_update=_now())

    # SmartInfoBus
    try:
        InfoBusManager = safe_import("modules.utils.info_bus")
        if InfoBusManager and hasattr(InfoBusManager, 'get_instance'):
            try:
                smart_bus = InfoBusManager.get_instance()  # type: ignore[attr-defined]
                health.smartinfobus_active = True
                health.modules_active = len(getattr(smart_bus, '_data_store', {}))
            except Exception as e:
                health.add_error(f"SmartInfoBus error: {e}")
    except Exception as e:
        health.add_error(f"SmartInfoBus import failed: {e}")

    # psutil metrics
    try:
        import psutil  # type: ignore
        process = psutil.Process()
        with process.oneshot():
            mem = process.memory_info().rss
            health.memory_usage_mb = _as_float_mb(mem)

        # Quick non-blocking sample for CPU (psutil may return float or list[float])
        _cpu = psutil.cpu_percent(interval=0.05)
        if isinstance(_cpu, (list, tuple)):
            try:
                # Average across cores if a list is returned
                health.cpu_usage_percent = float(np.mean([float(x) for x in _cpu])) if _cpu else 0.0
            except Exception:
                health.cpu_usage_percent = 0.0
        else:
            health.cpu_usage_percent = float(_cpu)
        health.disk_space_gb = _as_float_gb(psutil.disk_usage('.').free)

        # System total memory
        try:
            health.total_memory_mb = _as_float_mb(psutil.virtual_memory().total)
        except Exception:
            pass
    except ImportError:
        health.add_warning("psutil not available for system monitoring")
    except Exception as e:
        health.add_warning(f"System monitoring error: {e}")

    # GPU (optional)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            health.gpu_count = torch.cuda.device_count()
            try:
                # sum of reserved memory across devices (best-effort)
                gpu_mem = 0.0
                for i in range(health.gpu_count):
                    torch.cuda.set_device(i)
                    gpu_mem += _as_float_mb(torch.cuda.memory_reserved(i))
                health.gpu_memory_mb = gpu_mem
            except Exception:
                pass
    except Exception:
        # no warnings if torch isn't present; keep silent
        pass

    return health


# ---------- fallbacks

def create_fallback_systems():
    """Create fallback systems when SmartInfoBus is unavailable"""

    class FallbackSmartBus:
        """Minimal SmartInfoBus implementation (thread-safe)"""
        def __init__(self):
            self._data_store: Dict[str, Dict[str, Any]] = {}
            self._module_disabled = set()
            self._lock = threading.Lock()

        def set(self, key: str, value: Any, module: Optional[str] = None, thesis: Optional[str] = None):
            with self._lock:
                self._data_store[key] = {
                    'value': value,
                    'module': module,
                    'thesis': thesis,
                    'timestamp': _now()
                }

        def get(self, key: str, module: Optional[str] = None):
            with self._lock:
                data = self._data_store.get(key)
                return data['value'] if data else None

        def keys(self) -> List[str]:
            with self._lock:
                return list(self._data_store.keys())

        def register_provider(self, module: str, keys: List[str]):
            # no-op but compatible
            return True

        def register_consumer(self, module: str, keys: List[str]):
            # no-op but compatible
            return True

        def disable_module(self, module: str):
            with self._lock:
                self._module_disabled.add(module)

        def enable_module(self, module: str):
            with self._lock:
                self._module_disabled.discard(module)

        def get_performance_metrics(self):
            with self._lock:
                return {
                    'data_keys': len(self._data_store),
                    'disabled_modules': len(self._module_disabled),
                    'active': True
                }

    class FallbackOrchestrator:
        """Minimal orchestrator implementation (thread-safe, compatible)"""
        def __init__(self):
            self.modules: List[Any] = []
            self.enabled: bool = False
            self._lock = threading.Lock()

        def initialize(self):
            with self._lock:
                self.enabled = True

        def add_module(self, module: Any):
            with self._lock:
                self.modules.append(module)

        async def execute_step(self, inputs: Dict):
            # Async signature for compatibility; returns empty outputs
            return {}

    return FallbackSmartBus(), FallbackOrchestrator()


# ---------- market data validation

def validate_market_data(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[bool, List[str]]:
    """
    Validate market data structure and quality.
    Checks:
      - presence of instruments/timeframes
      - OHLC columns
      - nulls / non-positive prices
      - high >= low and OHLC within bounds
      - monotonically increasing, unique index
    """
    issues: List[str] = []

    if not data_dict:
        issues.append("❌ No market data provided")
        return False, issues

    required_columns = {'open', 'high', 'low', 'close'}

    for instrument, timeframes in data_dict.items():
        if not timeframes:
            issues.append(f"❌ No timeframes for {instrument}")
            continue

        for timeframe, df in timeframes.items():
            # Basic presence
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                issues.append(f"⚠️ Empty or invalid DataFrame for {instrument}/{timeframe}")
                continue

            # Index checks
            try:
                if not df.index.is_monotonic_increasing:
                    issues.append(f"⚠️ Index not monotonic increasing for {instrument}/{timeframe}")
                if df.index.has_duplicates:
                    dup_count = int(df.index.duplicated().sum())
                    issues.append(f"⚠️ {dup_count} duplicate index entries in {instrument}/{timeframe}")
            except Exception:
                # Don't fail if index is unusual
                pass

            # Required columns
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                issues.append(f"❌ Missing columns in {instrument}/{timeframe}: {missing_cols}")

            # Column quality
            for col in ('open', 'high', 'low', 'close'):
                if col in df.columns:
                    # NaNs
                    if bool(df[col].isnull().any()):
                        null_count = int(df[col].isnull().sum())
                        issues.append(f"⚠️ {null_count} null values in {instrument}/{timeframe}.{col}")
                    # Non-positive
                    try:
                        non_pos = int((df[col] <= 0).sum())
                        if non_pos > 0:
                            issues.append(f"⚠️ {non_pos} non-positive values in {instrument}/{timeframe}.{col}")
                    except Exception:
                        pass

            # OHLC logic
            if all(c in df.columns for c in ('open', 'high', 'low', 'close')):
                try:
                    high_low_issues = int((df['high'] < df['low']).sum())
                    if high_low_issues > 0:
                        issues.append(f"❌ {high_low_issues} bars where high < low in {instrument}/{timeframe}")

                    price_issues = int((
                        (df['open'] > df['high']) |
                        (df['open'] < df['low'])  |
                        (df['close'] > df['high'])|
                        (df['close'] < df['low'])
                    ).sum())
                    if price_issues > 0:
                        issues.append(f"❌ {price_issues} OHLC logic violations in {instrument}/{timeframe}")
                except Exception:
                    pass

    if not issues:
        issues.append("✅ Market data validation passed")

    has_critical = any(i.startswith("❌") for i in issues)
    return not has_critical, issues


# ---------- performance tuning

def optimize_environment_performance(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize environment configuration for performance without changing semantics.
    Keeps keys you set; only tweaks/adds safe defaults.
    """
    optimized = dict(config)  # shallow copy

    # Reduce logging overhead in production
    if not optimized.get('debug', False):
        optimized['log_rotation_lines'] = min(int(optimized.get('log_rotation_lines', 2000) or 2000), 1000)
        optimized['info_bus_audit_level'] = optimized.get('info_bus_audit_level', 'WARNING')

    # OS-specific tuning
    sys = platform.system()
    if sys == "Windows":
        # Windows process spawning can be expensive; keep it simple
        optimized['num_envs'] = int(optimized.get('num_envs', 1))
        optimized['enable_parallel_processing'] = bool(optimized.get('enable_parallel_processing', False))
    else:
        # Non-Windows: allow parallelism but cap based on CPU count
        try:
            import os
            cpu = max(1, os.cpu_count() or 1)
            default_envs = min(cpu, int(optimized.get('num_envs', 2)))
            optimized['num_envs'] = default_envs
            optimized['enable_parallel_processing'] = bool(optimized.get('enable_parallel_processing', default_envs > 1))
        except Exception:
            pass

    # Memory optimizations
    if int(optimized.get('max_steps', 200) or 200) > 500:
        optimized['max_history'] = int(optimized.get('max_history', 50))

    # Live trading
    if bool(optimized.get('live_mode', False)):
        optimized['risk_check_frequency'] = int(optimized.get('risk_check_frequency', 1))
        optimized['max_concurrent_alerts'] = int(optimized.get('max_concurrent_alerts', 5))
        optimized['enable_shadow_sim'] = bool(optimized.get('enable_shadow_sim', False))

    # Profiling threshold (used by @profile_method)
    optimized['profile_slow_ms'] = float(optimized.get('profile_slow_ms', 100.0))

    return optimized


# ---------- environment diagnostics

def create_environment_diagnostics(env) -> Dict[str, Any]:
    """Create comprehensive environment diagnostics (defensive)."""
    diagnostics: Dict[str, Any] = {
        'timestamp': _now(),
        'environment_type': type(env).__name__,
        'configuration': {},
        'system_status': {},
        'performance_metrics': {},
        'health_status': 'unknown'
    }

    try:
        # Config snapshot (works whether env.config is a dataclass or dict)
        if hasattr(env, 'config'):
            cfg = env.config
            get_cfg = (cfg.get if isinstance(cfg, dict) else lambda k, d=None: getattr(cfg, k, d))
            diagnostics['configuration'] = {
                'instruments': get_cfg('instruments', []),
                'max_steps': get_cfg('max_steps', None),
                'live_mode': bool(get_cfg('live_mode', False)),
                'info_bus_enabled': bool(get_cfg('info_bus_enabled', False))
            }

        # SmartInfoBus status
        smart_bus = _safe_getattr(env, 'smart_bus', None)
        if smart_bus:
            try:
                diagnostics['system_status']['smartinfobus'] = smart_bus.get_performance_metrics()
            except Exception:
                diagnostics['system_status']['smartinfobus'] = 'error'

        # Module system status
        orchestrator = _safe_getattr(env, 'orchestrator', None)
        if orchestrator:
            try:
                diagnostics['system_status']['modules'] = len(getattr(orchestrator, 'modules', []))
            except Exception:
                diagnostics['system_status']['modules'] = 'error'

        # Environment-specific metrics
        market_state = _safe_getattr(env, 'market_state', None)
        if market_state:
            diagnostics['performance_metrics'] = {
                'current_step': _safe_getattr(market_state, 'current_step', 0),
                'balance': _safe_getattr(market_state, 'balance', 0.0),
                'drawdown': _safe_getattr(market_state, 'current_drawdown', 0.0),
                'episode_count': _safe_getattr(env, 'episode_count', 0)
            }

        # Logger level (useful for debugging)
        logger = _safe_getattr(env, 'logger', None)
        if isinstance(logger, logging.Logger):
            diagnostics['system_status']['logger_level'] = logging.getLevelName(logger.level)

        # Overall health assessment
        has_smartinfobus = diagnostics['system_status'].get('smartinfobus') not in (None, 'error')
        modules_val = diagnostics['system_status'].get('modules', 0)
        has_modules = isinstance(modules_val, int) and modules_val > 0

        if has_smartinfobus and has_modules:
            diagnostics['health_status'] = 'excellent'
        elif has_smartinfobus:
            diagnostics['health_status'] = 'good'
        else:
            diagnostics['health_status'] = 'basic'

    except Exception as e:
        diagnostics['error'] = str(e)
        diagnostics['health_status'] = 'error'

    return diagnostics


# ---------- public exports

__all__ = [
    'SystemHealth',
    'profile_method',
    'safe_import',
    'create_enhanced_logger',
    'validate_trading_config',
    'get_system_status',
    'create_fallback_systems',
    'validate_market_data',
    'optimize_environment_performance',
    'create_environment_diagnostics'
]
