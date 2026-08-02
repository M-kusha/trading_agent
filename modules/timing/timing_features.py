#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TIMING_FEATURE_DIM = 8

_EPS_ATR = 1e-8


@dataclass
class TimingConfig:


    trading_start_hour: int = 9
    trading_end_hour: int = 18
    prime_start_hour: int = 14
    prime_end_hour: int = 17
    hard_close_hour: int = 22


    no_trade_windows: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 9),
        (18, 24),
    ])


    min_minutes_between_entries: float = 5.0
    min_minutes_after_loss: float = 15.0
    max_trades_per_session: int = 10


    hot_zone_atr_max: float = 0.3
    good_zone_atr_max: float = 1.0
    bad_zone_atr_min: float = 2.0


    vol_low_threshold: float = 0.5
    vol_high_threshold: float = 1.5
    vol_extreme_threshold: float = 2.5


    micro_trend_bars: int = 5
    rejection_wick_ratio: float = 2.0


    allow_weekend_trading: bool = False
    session_open_cooldown_minutes: float = 5.0
    min_minutes_to_close_for_entry: float = 15.0


    extreme_volatility_blocks_entries: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "TimingConfig":
        if not d:
            return cls()

        d = dict(d)


        ntw = d.get("no_trade_windows")
        if ntw is not None and isinstance(ntw, list):
            d["no_trade_windows"] = [tuple(w) for w in ntw]


        known_fields = {f_name for f_name in cls.__dataclass_fields__.keys()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class TimingFeatures:


    entry_allowed: bool = True


    entry_quality_long: float = 0.5
    entry_quality_short: float = 0.5


    zone_distance_norm: float = 1.0
    zone_type: str = "good"


    vol_state: str = "normal"
    vol_normalized: float = 1.0


    micro_trend_dir: float = 0.0
    rejection_detected: bool = False


    in_trading_window: bool = True
    in_prime_window: bool = False
    minutes_to_close: float = 0.0


    can_enter_spacing: bool = True
    minutes_since_last_entry: float = 999.0
    minutes_since_last_loss: float = 999.0
    trades_this_session: int = 0


    block_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_atr_from_window(ohlc_window: np.ndarray, fallback_atr: float) -> float:
    if ohlc_window is None or len(ohlc_window) == 0:
        return max(float(fallback_atr), _EPS_ATR)

    try:

        window = ohlc_window[-min(len(ohlc_window), 14):]
        ranges = window[:, 1] - window[:, 2]
        avg_range = float(np.mean(ranges))
        if avg_range > 0:
            return max(avg_range, _EPS_ATR)
    except Exception:
        pass

    return max(float(fallback_atr), _EPS_ATR)


def compute_timing_features(
    instrument: str,
    ohlc_window: np.ndarray,
    atr_value: float,
    session_info: Dict[str, Any],
    position_state: Dict[str, Any],
    timing_config: Optional[TimingConfig] = None,
) -> TimingFeatures:
    config = timing_config or TimingConfig()
    features = TimingFeatures()


    if ohlc_window is None or len(ohlc_window) < 2:
        features.entry_allowed = False
        features.block_reasons.append("INSUFFICIENT_DATA")
        return features


    atr_value = _safe_atr_from_window(ohlc_window, float(atr_value))
    atr_scale = max(float(atr_value), _EPS_ATR)


    hour = int(session_info.get("hour", 12))
    minute = int(session_info.get("minute", 0))
    weekday = int(session_info.get("weekday", 0))


    in_trading_window = config.trading_start_hour <= hour < config.trading_end_hour


    for start_h, end_h in config.no_trade_windows:
        if start_h <= hour < end_h:
            in_trading_window = False
            features.block_reasons.append(f"NO_TRADE_WINDOW_{start_h}_{end_h}")
            break


    if weekday >= 5:
        if not config.allow_weekend_trading:
            in_trading_window = False
            features.block_reasons.append("WEEKEND")

    features.in_trading_window = in_trading_window


    features.in_prime_window = (
        config.prime_start_hour <= hour < config.prime_end_hour
    )


    features.minutes_to_close = max(0.0, float((config.trading_end_hour - hour) * 60 - minute))
    minutes_since_open = max(0.0, float((hour - config.trading_start_hour) * 60 + minute))


    if minutes_since_open < config.session_open_cooldown_minutes:
        features.block_reasons.append(
            f"SESSION_OPEN_COOLDOWN_{minutes_since_open:.0f}min"
        )


    if features.minutes_to_close < config.min_minutes_to_close_for_entry:
        features.block_reasons.append(
            f"NEAR_SESSION_CLOSE_{features.minutes_to_close:.0f}min"
        )


    minutes_since_last = float(position_state.get("minutes_since_entry", 999.0))
    minutes_since_loss = float(position_state.get("minutes_since_loss", 999.0))
    trades_today = int(position_state.get("trades_this_session", 0))
    had_recent_loss = bool(position_state.get("had_recent_loss", False))

    features.minutes_since_last_entry = minutes_since_last
    features.minutes_since_last_loss = minutes_since_loss
    features.trades_this_session = trades_today


    min_spacing = config.min_minutes_between_entries
    if had_recent_loss:
        min_spacing = max(min_spacing, config.min_minutes_after_loss)

    if minutes_since_last < min_spacing:
        features.can_enter_spacing = False
        features.block_reasons.append(
            f"SPACING_{minutes_since_last:.0f}min<{min_spacing:.0f}min"
        )


    if trades_today >= config.max_trades_per_session:
        features.can_enter_spacing = False
        features.block_reasons.append(f"MAX_TRADES_{trades_today}")


    closes = ohlc_window[:, 3]
    highs = ohlc_window[:, 1]
    lows = ohlc_window[:, 2]
    opens = ohlc_window[:, 0]

    current_close = float(closes[-1])


    if len(highs) >= 20:
        recent_high = float(np.max(highs[-20:]))
        recent_low = float(np.min(lows[-20:]))
    else:
        recent_high = float(np.max(highs))
        recent_low = float(np.min(lows))


    dist_to_high = abs(current_close - recent_high) / atr_scale
    dist_to_low = abs(current_close - recent_low) / atr_scale
    features.zone_distance_norm = float(min(dist_to_high, dist_to_low))


    if features.zone_distance_norm < config.hot_zone_atr_max:
        features.zone_type = "hot"
    elif features.zone_distance_norm < config.good_zone_atr_max:
        features.zone_type = "good"
    elif features.zone_distance_norm > config.bad_zone_atr_min:
        features.zone_type = "bad"
    else:

        features.zone_type = "good"


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


    if features.vol_state == "extreme":
        features.block_reasons.append("EXTREME_VOLATILITY")


    n_bars = min(config.micro_trend_bars, len(closes) - 1)
    if n_bars >= 2:
        x = np.arange(n_bars)
        y = closes[-n_bars:]


        y_norm = (y - y[0]) / atr_scale

        try:
            slope = float(np.polyfit(x, y_norm, 1)[0])

            features.micro_trend_dir = float(np.clip(slope * 2.0, -1.0, 1.0))
        except Exception:
            features.micro_trend_dir = 0.0


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


    zone_quality = {
        "hot": 0.9,
        "good": 0.7,
        "bad": 0.3,
    }.get(features.zone_type, 0.5)


    vol_adj = {
        "low": 0.8,
        "normal": 1.0,
        "high": 1.1,
        "extreme": 0.5,
    }.get(features.vol_state, 1.0)


    prime_bonus = 1.15 if features.in_prime_window else 1.0


    session_penalty = 1.0
    if minutes_since_open < config.session_open_cooldown_minutes:
        session_penalty *= 0.7
    if features.minutes_to_close < config.min_minutes_to_close_for_entry:
        session_penalty *= 0.6


    base_quality = zone_quality * vol_adj * prime_bonus * session_penalty


    if dist_to_low < dist_to_high:
        long_q = base_quality * 1.1
        short_q = base_quality * 0.9
    else:
        long_q = base_quality * 0.9
        short_q = base_quality * 1.1


    if features.micro_trend_dir > 0.3:
        long_q *= 1.1
        short_q *= 0.9
    elif features.micro_trend_dir < -0.3:
        long_q *= 0.9
        short_q *= 1.1


    if features.rejection_detected:
        if dist_to_high < dist_to_low:

            short_q *= 1.2
            long_q *= 0.8
        else:

            long_q *= 1.2
            short_q *= 0.8


    features.entry_quality_long = float(np.clip(long_q, 0.0, 1.0))
    features.entry_quality_short = float(np.clip(short_q, 0.0, 1.0))


    allowed = (
        features.in_trading_window
        and features.can_enter_spacing
    )


    if config.extreme_volatility_blocks_entries and features.vol_state == "extreme":
        allowed = False


    if minutes_since_open < config.session_open_cooldown_minutes:
        allowed = False
    if features.minutes_to_close < config.min_minutes_to_close_for_entry:
        allowed = False

    features.entry_allowed = bool(allowed)

    return features


def timing_features_to_array(features: TimingFeatures) -> np.ndarray:
    arr = np.zeros(TIMING_FEATURE_DIM, dtype=np.float32)

    arr[0] = 1.0 if features.entry_allowed else 0.0
    arr[1] = float(np.clip(features.entry_quality_long, 0.0, 1.0))
    arr[2] = float(np.clip(features.entry_quality_short, 0.0, 1.0))


    arr[3] = float(np.clip(features.zone_distance_norm, 0.0, 3.0) / 3.0)


    arr[4] = 1.0 if features.zone_type == "hot" else 0.0
    arr[5] = 1.0 if features.zone_type == "good" else 0.0


    vol_map = {"low": 0.0, "normal": 0.33, "high": 0.66, "extreme": 1.0}
    arr[6] = float(vol_map.get(features.vol_state, 0.33))


    arr[7] = float(np.clip(features.micro_trend_dir, -1.0, 1.0))

    return arr


def load_timing_config(
    timing_path: str = "config/timing_policy.yaml",
    risk_path: str = "config/risk_policy.yaml",
) -> TimingConfig:
    from pathlib import Path

    import yaml

    merged_data = {}


    risk_file = Path(risk_path)
    if risk_file.exists():
        try:
            with open(risk_file, "r", encoding="utf-8") as f:
                risk_data = yaml.safe_load(f) or {}

            session = risk_data.get("session_management", {})


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


            start_h = merged_data.get("trading_start_hour", 9)
            end_h = merged_data.get("trading_end_hour", 18)
            merged_data["no_trade_windows"] = [
                (0, start_h),
                (end_h, 24),
            ]
        except Exception:
            pass


    timing_file = Path(timing_path)
    if timing_file.exists():
        try:
            with open(timing_file, "r", encoding="utf-8") as f:
                timing_data = yaml.safe_load(f) or {}


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
            pass

    return TimingConfig.from_dict(merged_data)
