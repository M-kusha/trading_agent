

from __future__ import annotations

import logging
import math
from typing import Any, Tuple


def safe_float(x: Any, default: float = 0.0) -> float:
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
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def clamp(v: float, lo: float, hi: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return (lo + hi) / 2.0
    return max(lo, min(hi, v))


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    radius = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    return clamp(center - radius, 0.0, 1.0), clamp(center + radius, 0.0, 1.0)


def mean_ci_normal(mean: float, std: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 1:
        return mean, mean
    se = std / math.sqrt(max(n, 1))
    return mean - z * se, mean + z * se


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


DEFAULT_PRIMARY_TIMEFRAME = "M15"


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
    tf_upper = tf.upper().strip() if tf else DEFAULT_PRIMARY_TIMEFRAME
    return TIMEFRAME_MINUTES.get(tf_upper, 15)


def bars_per_day_for_timeframe(tf: str) -> int:
    tf_upper = tf.upper().strip() if tf else DEFAULT_PRIMARY_TIMEFRAME
    return BARS_PER_DAY.get(tf_upper, 96)


def get_envs_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"envs.{name}")


def direction_sign(direction: str) -> float:
    d = direction.lower().strip() if direction else "neutral"
    if d in ("long", "buy", "bullish", "up"):
        return 1.0
    elif d in ("short", "sell", "bearish", "down"):
        return -1.0
    return 0.0


def normalize_direction(direction: str) -> str:
    d = direction.lower().strip() if direction else "neutral"
    if d in ("long", "buy", "bullish", "up"):
        return "buy"
    elif d in ("short", "sell", "bearish", "down"):
        return "sell"
    return "neutral"


def iso_timestamp() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
