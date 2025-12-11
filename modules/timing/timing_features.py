#!/usr/bin/env python3
"""
Timing Features - Pure Utility Module
=====================================

Computes entry timing features for PPO observation vectors.
This module has NO dependencies on SmartInfoBus or the module system.

Can be used identically in:
- Training environment (ModernTradingEnv)
- Live system (via EntryTimingController)

Features computed:
- entry_allowed: Whether entry is allowed based on time/spacing rules
- entry_quality_long/short: Quality score for long/short entries (0-1)
- zone_distance_norm: ATR-normalized distance to key levels
- zone_type: Classification ("hot", "good", "bad")
- vol_state: Volatility state ("low", "normal", "high", "extreme")
- micro_trend_dir: Short-term trend direction (-1 to 1)

Version: 1.1.0 (robust ATR, session open/close logic, configurable gating)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Timing feature vector dimension (must stay in sync with PPO obs builder)
TIMING_FEATURE_DIM = 8

_EPS_ATR = 1e-8  # Minimum ATR scale to avoid numeric issues


@dataclass
class TimingConfig:
    """
    Configuration for timing features computation.

    Canonical trading hours come from config/risk_policy.yaml (session_management).
    Timing-specific settings (spacing, zones) come from config/timing_policy.yaml.

    All hours are in local time (Europe/Berlin), matching risk_policy.yaml.
    """

    # Session windows (LOCAL hours - Europe/Berlin)
    # Canonical source: risk_policy.yaml → session_management
    trading_start_hour: int = 9      # 09:00 local (no_new_trades_end)
    trading_end_hour: int = 18       # 18:00 local (no_new_trades_start)
    prime_start_hour: int = 14       # 14:00 local (prime_hours_start)
    prime_end_hour: int = 17         # 17:00 local (prime_hours_end)
    hard_close_hour: int = 22        # 22:00 local (hard_close_hour)

    # No-trade windows (computed from trading_start/end, not hardcoded)
    # Will be auto-generated in load_timing_config() from risk_policy.yaml
    no_trade_windows: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 9),    # Overnight until trading_start
        (18, 24),  # After trading_end until midnight
    ])

    # Trade spacing rules
    min_minutes_between_entries: float = 5.0
    min_minutes_after_loss: float = 15.0
    max_trades_per_session: int = 10

    # Zone classification thresholds (ATR multipliers)
    hot_zone_atr_max: float = 0.3      # Within 0.3 ATR of key level = hot
    good_zone_atr_max: float = 1.0     # Within 1.0 ATR = good
    bad_zone_atr_min: float = 2.0      # Beyond 2.0 ATR = bad

    # Volatility classification thresholds (normalized ATR vs avg range)
    vol_low_threshold: float = 0.5     # Normalized vol < 0.5 = low
    vol_high_threshold: float = 1.5    # Normalized vol > 1.5 = high
    vol_extreme_threshold: float = 2.5 # Normalized vol > 2.5 = extreme

    # Micro-trend parameters
    micro_trend_bars: int = 5          # Bars to compute micro trend
    rejection_wick_ratio: float = 2.0  # Wick > 2x body = rejection

    # Session control
    allow_weekend_trading: bool = False
    session_open_cooldown_minutes: float = 5.0   # Avoid entries in first N minutes after open
    min_minutes_to_close_for_entry: float = 15.0 # Avoid entries very close to session end

    # Volatility gating
    extreme_volatility_blocks_entries: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "TimingConfig":
        """
        Create config from dictionary, using defaults for missing keys.

        Unknown keys are ignored. This makes the config forward-compatible
        with extra fields in YAML.
        """
        if not d:
            return cls()

        d = dict(d)  # work on a shallow copy

        # Handle no_trade_windows specially (list of tuples)
        ntw = d.get("no_trade_windows")
        if ntw is not None and isinstance(ntw, list):
            d["no_trade_windows"] = [tuple(w) for w in ntw]

        # Filter to only known fields
        known_fields = {f_name for f_name in cls.__dataclass_fields__.keys()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class TimingFeatures:
    """
    Computed timing features for a single instrument.

    These features are used:
    1. As PPO observation inputs (via timing_features_to_array)
    2. As optional safety gates in PositionManager
    """

    # Primary gate
    entry_allowed: bool = True

    # Entry quality scores (0-1, where 1 = best)
    entry_quality_long: float = 0.5
    entry_quality_short: float = 0.5

    # Zone information
    zone_distance_norm: float = 1.0  # ATR-normalized distance to nearest key level
    zone_type: str = "good"          # "hot", "good", "bad"

    # Volatility state
    vol_state: str = "normal"        # "low", "normal", "high", "extreme"
    vol_normalized: float = 1.0      # Current ATR / average ATR

    # Micro-structure
    micro_trend_dir: float = 0.0     # -1 to 1, short-term trend
    rejection_detected: bool = False # True if rejection wick detected

    # Session info
    in_trading_window: bool = True
    in_prime_window: bool = False
    minutes_to_close: float = 0.0

    # Trade spacing
    can_enter_spacing: bool = True
    minutes_since_last_entry: float = 999.0
    minutes_since_last_loss: float = 999.0
    trades_this_session: int = 0

    # Blocking reasons (for logging / debugging in bus)
    block_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for bus publishing or logging."""
        return asdict(self)


def _safe_atr_from_window(ohlc_window: np.ndarray, fallback_atr: float) -> float:
    """
    Compute a safe ATR-like scale from the OHLC window if ATR is missing/invalid.

    Uses a simple mean of high-low ranges over recent bars as a proxy ATR.
    """
    if ohlc_window is None or len(ohlc_window) == 0:
        return max(float(fallback_atr), _EPS_ATR)

    try:
        # Use up to the last 14 bars as a simple ATR proxy
        window = ohlc_window[-min(len(ohlc_window), 14):]
        ranges = window[:, 1] - window[:, 2]  # high - low
        avg_range = float(np.mean(ranges))
        if avg_range > 0:
            return max(avg_range, _EPS_ATR)
    except Exception:
        pass

    return max(float(fallback_atr), _EPS_ATR)


def compute_timing_features(
    instrument: str,
    ohlc_window: np.ndarray,           # Shape: (N, 4) = [open, high, low, close]
    atr_value: float,
    session_info: Dict[str, Any],      # {"hour": int, "minute": int, "weekday": int}
    position_state: Dict[str, Any],    # {"side": 0/1/-1, "minutes_since_entry": float, ...}
    timing_config: Optional[TimingConfig] = None,
) -> TimingFeatures:
    """
    Compute timing features for a single instrument.

    Args:
        instrument: Instrument symbol (e.g., "XAUUSD"). Currently informational only.
        ohlc_window: Recent OHLC data, shape (N, 4). Most recent bar is last.
        atr_value: Current ATR value for the instrument (may be 0/invalid; we sanitize it).
        session_info: Current session information ("hour", "minute", "weekday").
        position_state: Current position state for spacing rules.
        timing_config: Configuration (uses defaults if None).

    Returns:
        TimingFeatures dataclass with all computed features.
    """
    config = timing_config or TimingConfig()
    features = TimingFeatures()

    # Validate OHLC data
    if ohlc_window is None or len(ohlc_window) < 2:
        features.entry_allowed = False
        features.block_reasons.append("INSUFFICIENT_DATA")
        return features

    # Ensure ATR is usable
    atr_value = _safe_atr_from_window(ohlc_window, float(atr_value))
    atr_scale = max(float(atr_value), _EPS_ATR)

    # Extract session info
    hour = int(session_info.get("hour", 12))
    minute = int(session_info.get("minute", 0))
    weekday = int(session_info.get("weekday", 0))  # 0=Monday

    # ═══════════════════════════════════════════════════════════════════
    # 1. SESSION/TIME CHECKS
    # ═══════════════════════════════════════════════════════════════════

    # Base trading window check
    in_trading_window = config.trading_start_hour <= hour < config.trading_end_hour

    # No-trade hard windows
    for start_h, end_h in config.no_trade_windows:
        if start_h <= hour < end_h:
            in_trading_window = False
            features.block_reasons.append(f"NO_TRADE_WINDOW_{start_h}_{end_h}")
            break

    # Weekend
    if weekday >= 5:  # Saturday or Sunday
        if not config.allow_weekend_trading:
            in_trading_window = False
            features.block_reasons.append("WEEKEND")

    features.in_trading_window = in_trading_window

    # Prime trading window (for quality boost)
    features.in_prime_window = (
        config.prime_start_hour <= hour < config.prime_end_hour
    )

    # Minutes to close and since open
    features.minutes_to_close = max(0.0, float((config.trading_end_hour - hour) * 60 - minute))
    minutes_since_open = max(0.0, float((hour - config.trading_start_hour) * 60 + minute))

    # Session open cooldown
    if minutes_since_open < config.session_open_cooldown_minutes:
        features.block_reasons.append(
            f"SESSION_OPEN_COOLDOWN_{minutes_since_open:.0f}min"
        )

    # Close proximity
    if features.minutes_to_close < config.min_minutes_to_close_for_entry:
        features.block_reasons.append(
            f"NEAR_SESSION_CLOSE_{features.minutes_to_close:.0f}min"
        )

    # ═══════════════════════════════════════════════════════════════════
    # 2. TRADE SPACING CHECKS
    # ═══════════════════════════════════════════════════════════════════

    minutes_since_last = float(position_state.get("minutes_since_entry", 999.0))
    minutes_since_loss = float(position_state.get("minutes_since_loss", 999.0))
    trades_today = int(position_state.get("trades_this_session", 0))
    had_recent_loss = bool(position_state.get("had_recent_loss", False))

    features.minutes_since_last_entry = minutes_since_last
    features.minutes_since_last_loss = minutes_since_loss
    features.trades_this_session = trades_today

    # Base spacing
    min_spacing = config.min_minutes_between_entries
    if had_recent_loss:
        min_spacing = max(min_spacing, config.min_minutes_after_loss)

    if minutes_since_last < min_spacing:
        features.can_enter_spacing = False
        features.block_reasons.append(
            f"SPACING_{minutes_since_last:.0f}min<{min_spacing:.0f}min"
        )

    # Session trade limit
    if trades_today >= config.max_trades_per_session:
        features.can_enter_spacing = False
        features.block_reasons.append(f"MAX_TRADES_{trades_today}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. PRICE STRUCTURE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    closes = ohlc_window[:, 3]
    highs = ohlc_window[:, 1]
    lows = ohlc_window[:, 2]
    opens = ohlc_window[:, 0]

    current_close = float(closes[-1])

    # Key levels: recent high/low over last ~20 bars
    if len(highs) >= 20:
        recent_high = float(np.max(highs[-20:]))
        recent_low = float(np.min(lows[-20:]))
    else:
        recent_high = float(np.max(highs))
        recent_low = float(np.min(lows))

    # Distance to key levels (ATR normalized)
    dist_to_high = abs(current_close - recent_high) / atr_scale
    dist_to_low = abs(current_close - recent_low) / atr_scale
    features.zone_distance_norm = float(min(dist_to_high, dist_to_low))

    # Zone classification (simple 3-bucket)
    if features.zone_distance_norm < config.hot_zone_atr_max:
        features.zone_type = "hot"
    elif features.zone_distance_norm < config.good_zone_atr_max:
        features.zone_type = "good"
    elif features.zone_distance_norm > config.bad_zone_atr_min:
        features.zone_type = "bad"
    else:
        # Middle ground between "good" and "bad" is treated as "good" for now
        features.zone_type = "good"

    # ═══════════════════════════════════════════════════════════════════
    # 4. VOLATILITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    # Compute normalized volatility (current ATR vs recent average range)
    if len(ohlc_window) >= 20:
        recent_ranges = highs[-20:] - lows[-20:]
        avg_range = float(np.mean(recent_ranges))
        if avg_range > _EPS_ATR:
            features.vol_normalized = float(atr_scale / avg_range)
        else:
            features.vol_normalized = 1.0
    else:
        features.vol_normalized = 1.0

    vn = features.vol_normalized
    if vn < config.vol_low_threshold:
        features.vol_state = "low"
    elif vn > config.vol_extreme_threshold:
        features.vol_state = "extreme"
    elif vn > config.vol_high_threshold:
        features.vol_state = "high"
    else:
        features.vol_state = "normal"

    # Extreme volatility marker (blocking is controlled by config)
    if features.vol_state == "extreme":
        features.block_reasons.append("EXTREME_VOLATILITY")

    # ═══════════════════════════════════════════════════════════════════
    # 5. MICRO-TREND ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    n_bars = min(config.micro_trend_bars, len(closes) - 1)
    if n_bars >= 2:
        x = np.arange(n_bars)
        y = closes[-n_bars:]

        # Normalize by ATR scale for instrument independence
        y_norm = (y - y[0]) / atr_scale

        try:
            slope = float(np.polyfit(x, y_norm, 1)[0])
            # A mild scaling then clipping into [-1, 1]
            features.micro_trend_dir = float(np.clip(slope * 2.0, -1.0, 1.0))
        except Exception:
            features.micro_trend_dir = 0.0

    # Rejection wick detection on last bar
    last_bar = ohlc_window[-1]
    o, h, l, c = map(float, last_bar)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if body > 0.0:
        if upper_wick > body * config.rejection_wick_ratio:
            features.rejection_detected = True
        elif lower_wick > body * config.rejection_wick_ratio:
            features.rejection_detected = True

    # ═══════════════════════════════════════════════════════════════════
    # 6. ENTRY QUALITY SCORES
    # ═══════════════════════════════════════════════════════════════════

    # Base quality from zone
    zone_quality = {
        "hot": 0.9,
        "good": 0.7,
        "bad": 0.3,
    }.get(features.zone_type, 0.5)

    # Volatility adjustment
    vol_adj = {
        "low": 0.8,      # Low vol = less opportunity
        "normal": 1.0,
        "high": 1.1,     # Higher vol = more opportunity (but more risk)
        "extreme": 0.5,  # Extreme = dangerous
    }.get(features.vol_state, 1.0)

    # Prime hours bonus
    prime_bonus = 1.15 if features.in_prime_window else 1.0

    # Session open/close penalties (soft, applied to base quality)
    session_penalty = 1.0
    if minutes_since_open < config.session_open_cooldown_minutes:
        session_penalty *= 0.7
    if features.minutes_to_close < config.min_minutes_to_close_for_entry:
        session_penalty *= 0.6

    # Compute directional base quality
    base_quality = zone_quality * vol_adj * prime_bonus * session_penalty

    # Long quality: boosted near lows, penalized near highs
    if dist_to_low < dist_to_high:
        long_q = base_quality * 1.1
        short_q = base_quality * 0.9
    else:
        long_q = base_quality * 0.9
        short_q = base_quality * 1.1

    # Micro-trend adjustment: follow the micro-trend
    if features.micro_trend_dir > 0.3:
        long_q *= 1.1
        short_q *= 0.9
    elif features.micro_trend_dir < -0.3:
        long_q *= 0.9
        short_q *= 1.1

    # Rejection wick adjustment: lean into the rejection direction
    if features.rejection_detected:
        if dist_to_high < dist_to_low:
            # Rejection at/near highs: favour shorts
            short_q *= 1.2
            long_q *= 0.8
        else:
            # Rejection at/near lows: favour longs
            long_q *= 1.2
            short_q *= 0.8

    # Final clipping
    features.entry_quality_long = float(np.clip(long_q, 0.0, 1.0))
    features.entry_quality_short = float(np.clip(short_q, 0.0, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # 7. FINAL ENTRY ALLOWED CHECK
    # ═══════════════════════════════════════════════════════════════════

    # Base gates: time window + spacing
    allowed = (
        features.in_trading_window
        and features.can_enter_spacing
    )

    # Volatility gate (configurable)
    if config.extreme_volatility_blocks_entries and features.vol_state == "extreme":
        allowed = False

    # Session open/close gating (hard)
    if minutes_since_open < config.session_open_cooldown_minutes:
        allowed = False
    if features.minutes_to_close < config.min_minutes_to_close_for_entry:
        allowed = False

    features.entry_allowed = bool(allowed)

    return features


def timing_features_to_array(features: TimingFeatures) -> np.ndarray:
    """
    Convert TimingFeatures to a fixed-size array for PPO observation.

    Returns:
        np.ndarray of shape (TIMING_FEATURE_DIM,) = (8,)

    Feature layout:
        [0] entry_allowed (0/1)
        [1] entry_quality_long (0-1)
        [2] entry_quality_short (0-1)
        [3] zone_distance_norm (0-1+, clipped and normalized to [0,1])
        [4] zone_is_hot (0/1)
        [5] zone_is_good (0/1)
        [6] vol_state_encoded (0=low, 0.33=normal, 0.66=high, 1=extreme)
        [7] micro_trend_dir (-1 to 1)
    """
    arr = np.zeros(TIMING_FEATURE_DIM, dtype=np.float32)

    arr[0] = 1.0 if features.entry_allowed else 0.0
    arr[1] = float(np.clip(features.entry_quality_long, 0.0, 1.0))
    arr[2] = float(np.clip(features.entry_quality_short, 0.0, 1.0))

    # Zone distance: clip to [0, 3 ATR] and normalize to [0, 1]
    arr[3] = float(np.clip(features.zone_distance_norm, 0.0, 3.0) / 3.0)

    # Zone type flags
    arr[4] = 1.0 if features.zone_type == "hot" else 0.0
    arr[5] = 1.0 if features.zone_type == "good" else 0.0

    # Volatility state encoding
    vol_map = {"low": 0.0, "normal": 0.33, "high": 0.66, "extreme": 1.0}
    arr[6] = float(vol_map.get(features.vol_state, 0.33))

    # Micro-trend direction (already in [-1, 1])
    arr[7] = float(np.clip(features.micro_trend_dir, -1.0, 1.0))

    return arr


def load_timing_config(
    timing_path: str = "config/timing_policy.yaml",
    risk_path: str = "config/risk_policy.yaml",
) -> TimingConfig:
    """
    Load timing configuration by merging:
    1. Canonical trading hours from risk_policy.yaml (session_management)
    2. Timing-specific settings from timing_policy.yaml

    This avoids duplication - trading hours are defined ONLY in risk_policy.yaml.
    """
    import yaml
    from pathlib import Path

    merged_data = {}

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Load canonical session windows from risk_policy.yaml
    # ─────────────────────────────────────────────────────────────────
    risk_file = Path(risk_path)
    if risk_file.exists():
        try:
            with open(risk_file, "r", encoding="utf-8") as f:
                risk_data = yaml.safe_load(f) or {}

            session = risk_data.get("session_management", {})

            # Map risk_policy keys → TimingConfig keys
            if "no_new_trades_end" in session:
                merged_data["trading_start_hour"] = session["no_new_trades_end"]
            if "no_new_trades_start" in session:
                merged_data["trading_end_hour"] = session["no_new_trades_start"]
            if "prime_hours_start" in session:
                merged_data["prime_start_hour"] = session["prime_hours_start"]
            if "prime_hours_end" in session:
                merged_data["prime_end_hour"] = session["prime_hours_end"]
            if "hard_close_hour" in session:
                merged_data["hard_close_hour"] = session["hard_close_hour"]

            # Auto-generate no_trade_windows from trading hours
            start_h = merged_data.get("trading_start_hour", 9)
            end_h = merged_data.get("trading_end_hour", 18)
            merged_data["no_trade_windows"] = [
                (0, start_h),     # Overnight until trading start
                (end_h, 24),      # After trading end until midnight
            ]
        except Exception:
            pass  # Fall back to defaults

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Overlay timing-specific settings from timing_policy.yaml
    # ─────────────────────────────────────────────────────────────────
    timing_file = Path(timing_path)
    if timing_file.exists():
        try:
            with open(timing_file, "r", encoding="utf-8") as f:
                timing_data = yaml.safe_load(f) or {}

            # Only merge timing-specific keys (not session hours - those are canonical from risk_policy)
            timing_specific_keys = {
                "min_minutes_between_entries",
                "min_minutes_after_loss",
                "max_trades_per_session",
                "hot_zone_atr_max",
                "good_zone_atr_max",
                "bad_zone_atr_min",
                "vol_low_threshold",
                "vol_high_threshold",
                "vol_extreme_threshold",
                "micro_trend_bars",
                "rejection_wick_ratio",
                "allow_weekend_trading",
                "session_open_cooldown_minutes",
                "min_minutes_to_close_for_entry",
                "extreme_volatility_blocks_entries",
            }

            for key in timing_specific_keys:
                if key in timing_data:
                    merged_data[key] = timing_data[key]
        except Exception:
            pass  # Fall back to defaults/risk_policy values

    return TimingConfig.from_dict(merged_data)
