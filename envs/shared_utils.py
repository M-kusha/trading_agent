# envs/shared_utils.py
"""
Shared utility functions for envs module.

This module consolidates common helper functions that were previously duplicated
across curriculum_env_wrapper.py, curriculum_manager.py, and prop_firm_env.py.

Audit Reference: DUP-2 - Consolidated from Week 3-4 fixes.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Tuple, Optional

import numpy as np


# -----------------------------------------------------------------------------
# Safe Type Conversions
# -----------------------------------------------------------------------------

def safe_float(x: Any, default: float = 0.0) -> float:
    """
    Safely convert any value to float.
    
    Handles None, NaN, inf, and non-numeric types gracefully.
    
    Args:
        x: Value to convert
        default: Default value if conversion fails or result is invalid
        
    Returns:
        Float value or default
    """
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def safe_int(x: Any, default: int = 0) -> int:
    """
    Safely convert any value to int.
    
    Args:
        x: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Int value or default
    """
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def clamp(v: float, lo: float, hi: float) -> float:
    """
    Clamp value to range [lo, hi], handling NaN/inf safely.
    
    Args:
        v: Value to clamp
        lo: Lower bound
        hi: Upper bound
        
    Returns:
        Clamped value, or midpoint if v is NaN/inf
    """
    if math.isnan(v) or math.isinf(v):
        return (lo + hi) / 2.0
    return max(lo, min(hi, v))


# -----------------------------------------------------------------------------
# Statistical Utilities
# -----------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a Bernoulli proportion.
    
    Useful for determining statistically significant win rates.
    For promotion decisions, typically use the LOWER bound.
    
    Args:
        k: Number of successes
        n: Total trials
        z: Z-score for confidence level (default 1.96 = 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n <= 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    radius = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    return clamp(center - radius, 0.0, 1.0), clamp(center + radius, 0.0, 1.0)


def mean_ci_normal(mean: float, std: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Normal distribution confidence interval for a mean.
    
    Args:
        mean: Sample mean
        std: Sample standard deviation
        n: Sample size
        z: Z-score for confidence level
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n <= 1:
        return mean, mean
    se = std / math.sqrt(max(n, 1))
    return mean - z * se, mean + z * se


# -----------------------------------------------------------------------------
# Timeframe Constants
# -----------------------------------------------------------------------------

# Standard timeframe hierarchy (minutes per bar)
TIMEFRAME_MINUTES = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M5": 5,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}

# Default primary timeframe for trading
DEFAULT_PRIMARY_TIMEFRAME = "M15"

# Bars per day by timeframe (for approximate calculations)
BARS_PER_DAY = {
    "M1": 1440,
    "M5": 288,
    "M15": 96,
    "M30": 48,
    "H1": 24,
    "H4": 6,
    "D1": 1,
}


def timeframe_to_minutes(tf: str) -> int:
    """
    Convert timeframe string to minutes.
    
    Args:
        tf: Timeframe string (e.g., "M15", "H1", "D1")
        
    Returns:
        Minutes per bar, defaults to 15 (M15) if unknown
    """
    tf_upper = tf.upper().strip() if tf else DEFAULT_PRIMARY_TIMEFRAME
    return TIMEFRAME_MINUTES.get(tf_upper, 15)


def bars_per_day_for_timeframe(tf: str) -> int:
    """
    Get approximate bars per trading day for a timeframe.
    
    Args:
        tf: Timeframe string
        
    Returns:
        Bars per day (24-hour day), defaults to 96 (M15) if unknown
    """
    tf_upper = tf.upper().strip() if tf else DEFAULT_PRIMARY_TIMEFRAME
    return BARS_PER_DAY.get(tf_upper, 96)


# -----------------------------------------------------------------------------
# Logging Utilities
# -----------------------------------------------------------------------------

def get_envs_logger(name: str) -> logging.Logger:
    """
    Get a standardized logger for envs module components.
    
    All envs loggers use consistent naming: "envs.{name}"
    
    Args:
        name: Component name (e.g., "prop_firm_env", "curriculum_manager")
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"envs.{name}")


# -----------------------------------------------------------------------------
# Direction Utilities
# -----------------------------------------------------------------------------

def direction_sign(direction: str) -> float:
    """
    Convert direction string to numeric sign.
    
    Args:
        direction: Direction string ("long", "buy", "short", "sell", "neutral")
        
    Returns:
        +1.0 for long/buy, -1.0 for short/sell, 0.0 for neutral/unknown
    """
    d = direction.lower().strip() if direction else "neutral"
    if d in ("long", "buy", "bullish", "up"):
        return 1.0
    elif d in ("short", "sell", "bearish", "down"):
        return -1.0
    return 0.0


def normalize_direction(direction: str) -> str:
    """
    Normalize direction string to canonical form.
    
    Args:
        direction: Direction string in various formats
        
    Returns:
        "buy", "sell", or "neutral"
    """
    d = direction.lower().strip() if direction else "neutral"
    if d in ("long", "buy", "bullish", "up"):
        return "buy"
    elif d in ("short", "sell", "bearish", "down"):
        return "sell"
    return "neutral"


# -----------------------------------------------------------------------------
# Timestamp Utilities
# -----------------------------------------------------------------------------

def iso_timestamp() -> str:
    """
    Get current ISO-format timestamp string.
    
    Returns:
        Current datetime in ISO format
    """
    from datetime import datetime
    return datetime.now().isoformat()
