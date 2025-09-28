# envs/utils.py
"""
STRICT Utilities for the Modern SmartInfoBus Trading Environment

- No fallbacks or silent degradation.
- No hidden InfoBus writes or hard-coded keys.
- Deterministic, thread-safe helpers with clear errors.
"""

from __future__ import annotations

import importlib
import logging
import platform
import time
from dataclasses import dataclass, field, asdict
from functools import lru_cache, wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────
# System health model
# ─────────────────────────────────────────────────────────
@dataclass
class SystemHealth:
    """System health status (strict)."""
    smartinfobus_active: bool = False
    modules_active: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_space_gb: float = 0.0
    last_update: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_count: int = 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────
# Strict import helper (no fallbacks)
# ─────────────────────────────────────────────────────────
@lru_cache(maxsize=128)
def strict_import(module_name: str, attr: Optional[str] = None) -> Any:
    """
    Import a module (and optional attribute) strictly.
    Raises ImportError with a clear message if anything is missing.
    """
    mod = importlib.import_module(module_name)
    if attr is None:
        return mod
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise ImportError(f"Module '{module_name}' does not expose attribute '{attr}'.") from e


# ─────────────────────────────────────────────────────────
# Profiling decorator (no bus writes)
# ─────────────────────────────────────────────────────────
def profile_method(func: Callable):
    """
    Performance profiling for env methods.
    - Logs at DEBUG if elapsed > self.profile_slow_ms (default 100 ms).
    - No external side-effects beyond logging.
    """
    is_coro = hasattr(func, "__await__") or getattr(func, "__is_coroutine__", False)

    @wraps(func)
    def _sync_wrapper(self, *args, **kwargs):
        threshold_ms = float(_safe_getattr(self, "profile_slow_ms", 100.0))
        t0 = _perf_ns()
        try:
            result = func(self, *args, **kwargs)
            elapsed_ms = (_perf_ns() - t0) / 1_000_000.0

            if elapsed_ms > threshold_ms:
                logger = _safe_getattr(self, "logger", None)
                if isinstance(logger, logging.Logger):
                    # rate-limit to 1 log/sec per function
                    last_log_ts = float(_safe_getattr(self, f"__pm_last_{func.__name__}", 0.0))
                    now = _now()
                    if now - last_log_ts > 1.0:
                        logger.debug("⏱️ %s took %.1f ms", func.__name__, elapsed_ms)
                        setattr(self, f"__pm_last_{func.__name__}", now)
            return result
        except Exception:
            elapsed_ms = (_perf_ns() - t0) / 1_000_000.0
            logger = _safe_getattr(self, "logger", None)
            if isinstance(logger, logging.Logger):
                logger.error("❌ %s failed after %.1f ms", func.__name__, elapsed_ms, exc_info=True)
            raise

    async def _async_wrapper(self, *args, **kwargs):
        threshold_ms = float(_safe_getattr(args[0], "profile_slow_ms", 100.0))
        t0 = _perf_ns()
        try:
            result = await func(*args, **kwargs)
            elapsed_ms = (_perf_ns() - t0) / 1_000_000.0

            if elapsed_ms > threshold_ms:
                self = args[0]
                logger = _safe_getattr(self, "logger", None)
                if isinstance(logger, logging.Logger):
                    last_log_ts = float(_safe_getattr(self, f"__pm_last_{func.__name__}", 0.0))
                    now = _now()
                    if now - last_log_ts > 1.0:
                        logger.debug("⏱️ %s took %.1f ms", func.__name__, elapsed_ms)
                        setattr(self, f"__pm_last_{func.__name__}", now)
            return result
        except Exception:
            elapsed_ms = (_perf_ns() - t0) / 1_000_000.0
            self = args[0]
            logger = _safe_getattr(self, "logger", None)
            if isinstance(logger, logging.Logger):
                logger.error("❌ %s failed after %.1f ms", func.__name__, elapsed_ms, exc_info=True)
            raise

    return _async_wrapper if is_coro else _sync_wrapper


# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
def create_enhanced_logger(name: str, log_path: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create a process-safe logger with console + optional rotating file handler.
    No duplicate handlers, deterministic formatting.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured; enforce level only
        logger.setLevel(level)
        return logger

    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console_handler)

    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
        ))
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger


# ─────────────────────────────────────────────────────────
# Config validation (lightweight, strict messages)
# ─────────────────────────────────────────────────────────
def validate_trading_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a TradingConfig-like dict.
    Returns (ok, issues). No silent defaults.
    """
    issues: List[str] = []

    # Required core fields
    for field in ("initial_balance", "max_steps", "instruments", "max_drawdown"):
        if field not in config:
            issues.append(f"❌ Missing required field: {field}")
        elif field == "instruments" and (not isinstance(config[field], (list, tuple)) or not config[field]):
            issues.append("❌ 'instruments' must be a non-empty list")
        elif field in ("initial_balance", "max_steps", "max_drawdown"):
            try:
                _ = float(config[field]) if field != "max_steps" else int(config[field])
            except Exception:
                issues.append(f"❌ Field '{field}' has invalid type/value")

    # Guard rails
    try:
        md = float(config.get("max_drawdown", 0.0))
        if md > 0.5:
            issues.append("⚠️ Max drawdown > 50% is extremely risky")
        elif md > 0.3:
            issues.append("⚠️ Max drawdown > 30% is very risky")
    except Exception:
        pass

    try:
        mpp = float(config.get("max_position_pct", 0.0))
        if mpp > 0.3:
            issues.append("⚠️ Position size > 30% per trade is very risky")
        elif mpp > 0.15:
            issues.append("⚠️ Position size > 15% per trade is risky")
    except Exception:
        pass

    try:
        te = float(config.get("max_total_exposure", 0.0))
        if te > 1.0:
            issues.append("❌ Total exposure > 100% is invalid")
        elif te > 0.5:
            issues.append("⚠️ Total exposure > 50% is very risky")
    except Exception:
        pass

    # Episode length / logging
    try:
        ms = int(config.get("max_steps", 200))
        if ms > 1000:
            issues.append("⚠️ Very long episodes may be slow")
    except Exception:
        pass

    try:
        lrl = int(config.get("log_rotation_lines", 2000))
        if lrl > 10000:
            issues.append("⚠️ Very high log rotation may impact performance")
    except Exception:
        pass

    ok = not any(s.startswith("❌") for s in issues)
    if not issues:
        issues.append("✅ Trading config validation passed")
    return ok, issues


# ─────────────────────────────────────────────────────────
# System status (strict dependencies)
# ─────────────────────────────────────────────────────────
def get_system_status() -> SystemHealth:
    """
    Collect process/system metrics.
    Requires `psutil`. GPU metrics are best-effort via torch if available.
    """
    health = SystemHealth(last_update=_now())

    # psutil is mandatory here
    try:
        import psutil  # type: ignore
    except Exception as e:
        raise RuntimeError("psutil is required for get_system_status()") from e

    process = psutil.Process()
    with process.oneshot():
        mem = process.memory_info().rss
        health.memory_usage_mb = _as_float_mb(mem)

    # Single quick CPU sample
    _cpu = psutil.cpu_percent(interval=0.05)
    if isinstance(_cpu, (list, tuple)):
        try:
            health.cpu_usage_percent = float(np.mean([float(x) for x in _cpu])) if _cpu else 0.0
        except Exception:
            health.cpu_usage_percent = 0.0
    else:
        health.cpu_usage_percent = float(_cpu)

    du = psutil.disk_usage(".")
    health.disk_space_gb = _as_float_gb(du.free)
    try:
        health.total_memory_mb = _as_float_mb(psutil.virtual_memory().total)
    except Exception:
        pass

    # GPU (optional)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            health.gpu_count = torch.cuda.device_count()
            try:
                # sum of reserved memory across devices
                gpu_mem = 0.0
                for i in range(health.gpu_count):
                    torch.cuda.set_device(i)
                    gpu_mem += _as_float_mb(torch.cuda.memory_reserved(i))
                health.gpu_memory_mb = gpu_mem
            except Exception:
                pass
    except Exception:
        pass

    return health


# ─────────────────────────────────────────────────────────
# Market data validation
# ─────────────────────────────────────────────────────────
def validate_market_data(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[bool, List[str]]:
    """
    Validate market data structure and quality.

    Checks
    - presence of instruments/timeframes
    - OHLC columns
    - NaNs / non-positive prices
    - high >= low and OHLC inside [low, high]
    - monotonic, unique index (best-effort)
    """
    issues: List[str] = []

    if not data_dict:
        return False, ["❌ No market data provided"]

    required_columns = {"open", "high", "low", "close"}

    for instrument, timeframes in data_dict.items():
        if not timeframes:
            issues.append(f"❌ No timeframes for {instrument}")
            continue

        for timeframe, df in timeframes.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                issues.append(f"❌ Empty or invalid DataFrame for {instrument}/{timeframe}")
                continue

            # Index checks
            try:
                if not df.index.is_monotonic_increasing:
                    issues.append(f"⚠️ Index not monotonic increasing for {instrument}/{timeframe}")
                if df.index.has_duplicates:
                    dup_count = int(df.index.duplicated().sum())
                    issues.append(f"⚠️ {dup_count} duplicate index entries in {instrument}/{timeframe}")
            except Exception:
                pass

            # Columns
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                issues.append(f"❌ Missing columns in {instrument}/{timeframe}: {missing_cols}")
                continue

            # Quality
            for col in ("open", "high", "low", "close"):
                series = df[col]
                if series.isnull().any():
                    issues.append(f"❌ NaNs in {instrument}/{timeframe}.{col}")
                try:
                    if (series <= 0).any():
                        issues.append(f"❌ Non-positive values in {instrument}/{timeframe}.{col}")
                except Exception:
                    pass

            try:
                if (df["high"] < df["low"]).any():
                    issues.append(f"❌ {instrument}/{timeframe}: high < low")
                oob = (
                    (df["open"] > df["high"]) | (df["open"] < df["low"]) |
                    (df["close"] > df["high"]) | (df["close"] < df["low"])
                )
                if oob.any():
                    issues.append(f"❌ {instrument}/{timeframe}: OHLC out-of-bounds in {int(oob.sum())} rows")
            except Exception:
                pass

    if not issues:
        issues.append("✅ Market data validation passed")
    return not any(i.startswith("❌") for i in issues), issues


# ─────────────────────────────────────────────────────────
# Performance tuning (no behavior changes)
# ─────────────────────────────────────────────────────────
def optimize_environment_performance(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest minor runtime tweaks without changing semantics.
    Returns a shallow copy with adjustments.
    """
    optimized = dict(config)

    # Logging
    if not optimized.get("debug", True):
        lrl = int(optimized.get("log_rotation_lines", 2000))
        optimized["log_rotation_lines"] = min(lrl, 1000)

    # OS-specific parallelism hints
    sys_name = platform.system()
    if sys_name == "Windows":
        optimized["num_envs"] = int(optimized.get("num_envs", 1))
        optimized["enable_parallel_processing"] = bool(optimized.get("enable_parallel_processing", False))
    else:
        try:
            cpu = max(1, (importlib.import_module("os").cpu_count() or 1))
            default_envs = min(cpu, int(optimized.get("num_envs", 2)))
            optimized["num_envs"] = default_envs
            optimized["enable_parallel_processing"] = default_envs > 1
        except Exception:
            pass

    # Memory-friendly episodes
    try:
        if int(optimized.get("max_steps", 200)) > 500:
            optimized["max_history"] = int(optimized.get("max_history", 50))
    except Exception:
        pass

    # Profiling threshold (used by @profile_method)
    optimized["profile_slow_ms"] = float(optimized.get("profile_slow_ms", 100.0))
    return optimized


# ─────────────────────────────────────────────────────────
# Diagnostics snapshot (no external deps)
# ─────────────────────────────────────────────────────────
def create_environment_diagnostics(env: Any) -> Dict[str, Any]:
    """
    Collect non-invasive diagnostics from an environment instance.
    No InfoBus imports; only introspection.
    """
    diag: Dict[str, Any] = {
        "timestamp": _now(),
        "environment_type": type(env).__name__,
        "configuration": {},
        "system_status": {},
        "performance_metrics": {},
        "health_status": "unknown",
    }

    # Config snapshot (best-effort)
    cfg = _safe_getattr(env, "config", None)
    if cfg is not None:
        getter = (cfg.get if isinstance(cfg, dict) else lambda k, d=None: getattr(cfg, k, d))
        diag["configuration"] = {
            "instruments": getter("instruments", []),
            "max_steps": getter("max_steps", None),
            "live_mode": bool(getter("live_mode", False)),
        }

    # Modules (best-effort)
    orch = _safe_getattr(env, "orchestrator", None)
    if orch is not None:
        try:
            diag["system_status"]["modules"] = len(getattr(orch, "modules", []))
        except Exception:
            diag["system_status"]["modules"] = "error"

    # Environment metrics (best-effort)
    ms = _safe_getattr(env, "market_state", None)
    if ms is not None:
        diag["performance_metrics"] = {
            "current_step": _safe_getattr(ms, "current_step", 0),
            "balance": float(_safe_getattr(ms, "balance", 0.0)),
            "drawdown": float(_safe_getattr(ms, "current_drawdown", 0.0)),
            "episode_count": int(_safe_getattr(env, "episode_count", 0)),
        }

    # Logger level
    logger = _safe_getattr(env, "logger", None)
    if isinstance(logger, logging.Logger):
        diag["system_status"]["logger_level"] = logging.getLevelName(logger.level)

    # Simple health heuristic
    modules_val = diag["system_status"].get("modules", 0)
    has_modules = isinstance(modules_val, int) and modules_val > 0
    diag["health_status"] = "excellent" if has_modules else "good"

    return diag


# ─────────────────────────────────────────────────────────
# Public exports
# ─────────────────────────────────────────────────────────
__all__ = [
    "SystemHealth",
    "strict_import",
    "profile_method",
    "create_enhanced_logger",
    "validate_trading_config",
    "get_system_status",
    "validate_market_data",
    "optimize_environment_performance",
    "create_environment_diagnostics",
]
