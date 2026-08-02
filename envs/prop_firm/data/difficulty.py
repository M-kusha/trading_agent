
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from envs.core.env_types import PropFirmConfig
    from numpy.random import Generator

import logging
logger = logging.getLogger(__name__)


class DataDifficultyMixin:


    config: "PropFirmConfig"
    data: Dict[str, Dict[str, pd.DataFrame]]
    instruments: List[str]
    _min_data_len: int
    _data_difficulty: Optional[Any]
    _valid_start_indices: Optional[np.ndarray]
    _volatility_percentiles: Optional[np.ndarray]
    _difficulty_cache_hash: Optional[int]
    _regime_weights: Optional[np.ndarray]

    def set_data_difficulty(self, difficulty: Any) -> None:
        self._data_difficulty = difficulty
        self._valid_start_indices = None
        self._volatility_percentiles = None
        self._difficulty_cache_hash: Optional[int] = None
        self._regime_weights = None

        if difficulty is not None:
            self._precompute_data_difficulty_indices()

    def _precompute_data_difficulty_indices(self) -> None:

        if self._data_difficulty is not None:
            d = self._data_difficulty
            config_hash = hash((
                getattr(d, 'volatility_percentile_range', (0.0, 1.0)),
                getattr(d, 'min_trend_clarity', 0.0),
                getattr(d, 'max_trend_clarity', 1.0),
                getattr(d, 'include_asian_session', True),
                getattr(d, 'include_london_session', True),
                getattr(d, 'include_ny_session', True),
                getattr(d, 'avoid_session_boundaries', False),
                tuple((getattr(d, "allowed_regimes", None) or [])),
                tuple(sorted((getattr(d, "regime_sampling_weights", {}) or {}).items())),
                getattr(d, "trend_clarity_threshold", 0.40),
                getattr(d, "high_volatility_threshold", 0.70),
                getattr(d, "low_volatility_threshold", 0.30),
                getattr(d, "news_volatility_threshold", 0.90),
            ))


            if self._difficulty_cache_hash == config_hash and self._valid_start_indices is not None:
                return

            self._difficulty_cache_hash = config_hash
        if self._data_difficulty is None:
            self._valid_start_indices = None
            return

        difficulty = self._data_difficulty
        inst = self.instruments[0]
        primary_tf = self.config.primary_timeframe
        df = self.data.get(inst, {}).get(primary_tf)

        if df is None or len(df) < 200:
            self._valid_start_indices = None
            return

        n_bars = len(df)
        buffer = 120
        max_end = n_bars - self.config.max_steps_per_episode - buffer

        if max_end <= buffer:
            self._valid_start_indices = None
            return


        valid_mask = np.ones(n_bars, dtype=bool)


        vol_range = getattr(difficulty, "volatility_percentile_range", (0.0, 1.0))
        if vol_range != (0.0, 1.0):
            self._compute_volatility_percentiles(df)
            if self._volatility_percentiles is not None:
                valid_mask &= (self._volatility_percentiles >= vol_range[0])
                valid_mask &= (self._volatility_percentiles <= vol_range[1])


        min_trend = float(getattr(difficulty, "min_trend_clarity", 0.0) or 0.0)
        max_trend = float(getattr(difficulty, "max_trend_clarity", 1.0) or 1.0)
        min_trend = float(np.clip(min_trend, 0.0, 1.0))
        max_trend = float(np.clip(max_trend, 0.0, 1.0))
        if min_trend > 0.0 or max_trend < 1.0:
            trend_clarity = self._compute_trend_clarity(df)
            if min_trend > 0.0:
                valid_mask &= (trend_clarity >= min_trend)
            if max_trend < 1.0:
                valid_mask &= (trend_clarity <= max_trend)


        allowed_regimes = getattr(difficulty, "allowed_regimes", None)
        if allowed_regimes:

            norm = set()
            for r in allowed_regimes:
                s = str(r).strip().lower()
                if "." in s:
                    s = s.split(".")[-1]
                norm.add(s)


            if self._volatility_percentiles is None:
                self._compute_volatility_percentiles(df)
            trend_clarity = self._compute_trend_clarity(df)
            trend_slope = self._compute_trend_slope(df)
            vol_pct = self._volatility_percentiles if self._volatility_percentiles is not None else None

            trend_th = float(getattr(difficulty, "trend_clarity_threshold", 0.40))
            high_vol = float(getattr(difficulty, "high_volatility_threshold", 0.70))
            low_vol = float(getattr(difficulty, "low_volatility_threshold", 0.30))
            news_vol = float(getattr(difficulty, "news_volatility_threshold", 0.90))

            regime_mask = np.zeros(n_bars, dtype=bool)
            if "trending_up" in norm:
                regime_mask |= (trend_clarity >= trend_th) & (trend_slope > 0.0)
            if "trending_down" in norm:
                regime_mask |= (trend_clarity >= trend_th) & (trend_slope < 0.0)
            if "ranging" in norm:
                regime_mask |= (trend_clarity < trend_th)
            if vol_pct is not None:
                if "high_volatility" in norm:
                    regime_mask |= (vol_pct >= high_vol)
                if "low_volatility" in norm:
                    regime_mask |= (vol_pct <= low_vol)
                if "news_volatility" in norm:
                    regime_mask |= (vol_pct >= news_vol)

            if regime_mask.any():
                valid_mask &= regime_mask


        if isinstance(df.index, pd.DatetimeIndex):
            try:
                hours = np.asarray(df.index.hour, dtype=np.int32)
                if hours is not None:
                    session_mask = np.zeros(n_bars, dtype=bool)


                    if getattr(difficulty, "include_asian_session", True):
                        session_mask |= (hours < 8)
                    if getattr(difficulty, "include_london_session", True):
                        session_mask |= ((hours >= 8) & (hours < 16))
                    if getattr(difficulty, "include_ny_session", True):
                        session_mask |= ((hours >= 14) & (hours < 22))
                    if getattr(difficulty, "include_overlap_sessions", True):
                        session_mask |= ((hours >= 14) & (hours < 16))


                    if session_mask.any():
                        valid_mask &= session_mask
            except Exception as e:
                logger.debug(f"Skip session filtering: {e}")


        if getattr(difficulty, "exclude_market_open_close", False):
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    hours = np.asarray(df.index.hour, dtype=np.int32)

                    open_close_mask = ~(
                        (hours == 0) | (hours == 8) | (hours == 14) |
                        (hours == 7) | (hours == 15) | (hours == 21)
                    )
                    valid_mask &= open_close_mask
            except Exception as e:
                logger.debug(f"Skip market open/close filter: {e}")


        range_mask = np.zeros(n_bars, dtype=bool)
        range_mask[buffer:max_end] = True
        valid_mask &= range_mask


        valid_indices = np.where(valid_mask)[0]


        self._regime_weights = None
        weights_cfg = getattr(self._data_difficulty, "regime_sampling_weights", None) if self._data_difficulty is not None else None
        if isinstance(weights_cfg, dict) and weights_cfg and len(valid_indices) > 0:

            weights_norm: Dict[str, float] = {}
            for k, v in weights_cfg.items():
                key = str(k).strip().lower()
                if "." in key:
                    key = key.split(".")[-1]
                weights_norm[key] = float(v)

            if self._volatility_percentiles is None:
                self._compute_volatility_percentiles(df)
            trend_clarity = self._compute_trend_clarity(df)
            trend_slope = self._compute_trend_slope(df)
            vol_pct = self._volatility_percentiles if self._volatility_percentiles is not None else None

            trend_th = float(getattr(difficulty, "trend_clarity_threshold", 0.40))
            high_vol = float(getattr(difficulty, "high_volatility_threshold", 0.70))
            low_vol = float(getattr(difficulty, "low_volatility_threshold", 0.30))
            news_vol = float(getattr(difficulty, "news_volatility_threshold", 0.90))

            weights = np.ones(len(valid_indices), dtype=np.float64)

            def _apply_weight(mask: np.ndarray, weight: float) -> None:
                if weight <= 0:
                    return
                idx_mask = mask[valid_indices]
                if idx_mask.any():
                    weights[idx_mask] = np.maximum(weights[idx_mask], weight)

            if "trending_up" in weights_norm:
                _apply_weight((trend_clarity >= trend_th) & (trend_slope > 0.0), weights_norm["trending_up"])
            if "trending_down" in weights_norm:
                _apply_weight((trend_clarity >= trend_th) & (trend_slope < 0.0), weights_norm["trending_down"])
            if "ranging" in weights_norm:
                _apply_weight((trend_clarity < trend_th), weights_norm["ranging"])
            if vol_pct is not None:
                if "high_volatility" in weights_norm:
                    _apply_weight(vol_pct >= high_vol, weights_norm["high_volatility"])
                if "low_volatility" in weights_norm:
                    _apply_weight(vol_pct <= low_vol, weights_norm["low_volatility"])
                if "news_volatility" in weights_norm:
                    _apply_weight(vol_pct >= news_vol, weights_norm["news_volatility"])

            self._regime_weights = weights

        if len(valid_indices) == 0:

            # _data_difficulty is Optional; this branch can be reached before it
            # is set, so read through a local rather than dereferencing it twice.
            diff = self._data_difficulty
            logger.warning(
                f"DataDifficulty filter found 0 valid indices with settings: "
                f"volatility_range={getattr(diff, 'volatility_percentile_range', None)}, "
                f"min_trend_clarity={getattr(diff, 'min_trend_clarity', None)}, "
                f"sessions=(asia={getattr(diff, 'include_asian_session', True)}, "
                f"london={getattr(diff, 'include_london_session', True)}, "
                f"ny={getattr(diff, 'include_ny_session', True)}). "
                f"Falling back to full dataset ({max_end - buffer} bars)."
            )
            self._valid_start_indices = np.arange(buffer, max_end)
        else:
            self._valid_start_indices = valid_indices
            logger.debug(f"DataDifficulty filter: {len(valid_indices)} valid start indices out of {max_end - buffer}")

    def _compute_volatility_percentiles(self, df: pd.DataFrame) -> None:
        try:
            if "close" not in df.columns and "Close" not in df.columns:
                self._volatility_percentiles = None
                return

            close_col = "close" if "close" in df.columns else "Close"
            close = np.asarray(df[close_col].values, dtype=np.float64)


            window = 20
            if len(close) < window + 1:
                self._volatility_percentiles = None
                return

            returns = np.abs(np.diff(close) / (close[:-1] + 1e-10))


            vol_series = pd.Series(returns).rolling(window=window, min_periods=1).std().fillna(0.0)
            vol = np.concatenate([[0.0], vol_series.to_numpy(dtype=np.float64)])


            percentiles = pd.Series(vol).rank(pct=True, method="average").to_numpy(dtype=np.float64)


            if len(percentiles) > 0:
                percentiles[0] = 0.5

            self._volatility_percentiles = percentiles
        except Exception:
            self._volatility_percentiles = None

    def _compute_trend_clarity(self, df: pd.DataFrame) -> np.ndarray:
        try:
            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns:
                return np.ones(len(df))

            close = np.asarray(df[close_col].values, dtype=np.float64)
            n = len(close)
            clarity = np.zeros(n)

            window = 20
            for i in range(window, n):
                segment = close[i-window:i]
                seg_mean = float(np.mean(segment))
                slope = (segment[-1] - segment[0]) / (window * (seg_mean + 1e-10))
                vol = float(np.std(np.diff(segment))) / (seg_mean + 1e-10)


                clarity[i] = min(1.0, abs(slope) / (vol + 1e-10))


            clarity[:window] = np.median(clarity[window:]) if n > window else 0.5

            return clarity
        except Exception:
            return np.ones(len(df))

    def _compute_trend_slope(self, df: pd.DataFrame) -> np.ndarray:
        try:
            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns:
                return np.zeros(len(df))
            close = np.asarray(df[close_col].values, dtype=np.float64)
            n = len(close)
            slope = np.zeros(n)
            window = 20
            for i in range(window, n):
                segment = close[i-window:i]
                seg_mean = float(np.mean(segment))
                slope[i] = (segment[-1] - segment[0]) / (window * (seg_mean + 1e-10))
            slope[:window] = np.median(slope[window:]) if n > window else 0.0
            return slope
        except Exception:
            return np.zeros(len(df))

    def _sample_episode_start_with_difficulty(self, buffer: int, max_start: int) -> int:
        if self._valid_start_indices is None or len(self._valid_start_indices) == 0:

            if max_start > buffer:
                return int(self.np_random.integers(buffer, max_start))
            return min(buffer, max(self._min_data_len - 2, 0))


        valid_in_range = self._valid_start_indices[
            (self._valid_start_indices >= buffer) &
            (self._valid_start_indices < max_start)
        ]

        if len(valid_in_range) == 0:

            if len(self._valid_start_indices) > 0:
                return int(self.np_random.choice(self._valid_start_indices))
            if max_start > buffer:
                return int(self.np_random.integers(buffer, max_start))
            return min(buffer, max(self._min_data_len - 2, 0))


        if (self._data_difficulty is not None and
            getattr(self._data_difficulty, "prefer_recent_data", False)):

            weight = getattr(self._data_difficulty, "recent_data_weight", 1.0)
            if weight > 1.0:

                positions = np.arange(len(valid_in_range))
                weights = np.exp(weight * positions / len(positions))
                weights /= weights.sum()
                idx = self.np_random.choice(len(valid_in_range), p=weights)
                return int(valid_in_range[idx])


        if self._regime_weights is not None and len(self._regime_weights) == len(self._valid_start_indices):
            try:
                mask = np.isin(self._valid_start_indices, valid_in_range)
                weights = self._regime_weights[mask]
                if weights.size == len(valid_in_range) and weights.sum() > 0:
                    weights = weights / weights.sum()
                    idx = self.np_random.choice(len(valid_in_range), p=weights)
                    return int(valid_in_range[idx])
            except Exception:
                pass


        return int(self.np_random.choice(valid_in_range))
