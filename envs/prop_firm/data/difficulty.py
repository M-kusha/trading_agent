
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Callable, Any, Dict, List, Optional, TYPE_CHECKING

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
            raise ValueError(f"DataDifficulty requires at least 200 primary bars; got {0 if df is None else len(df)}")

        n_bars = len(df)
        # This mixin is always composed into PropFirmTradingEnv, which defines
        # _episode_start_buffer. The getattr is a fallback for standalone use in
        # tests; naming the return type keeps the contract explicit rather than
        # leaving the checker to infer `object` from getattr.
        history_fn: Optional[Callable[[], int]] = getattr(self, "_episode_start_buffer", None)
        buffer: int = history_fn() if callable(history_fn) else 120
        max_end = n_bars - int(self.config.max_steps_per_episode)

        if max_end <= buffer:
            raise ValueError(
                f"DataDifficulty has no capacity after history/episode bounds: "
                f"bars={n_bars}, history={buffer}, max_steps={self.config.max_steps_per_episode}"
            )


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

            supported_regimes = {
                "trending_up",
                "trending_down",
                "ranging",
                "high_volatility",
                "low_volatility",
                "news_volatility",
            }
            unknown_regimes = sorted(norm - supported_regimes)
            if unknown_regimes:
                raise ValueError(
                    "unsupported DataDifficulty regimes: "
                    + ", ".join(unknown_regimes)
                )


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

            # An empty requested regime is an empty eligible set, not an excuse
            # to silently fall back to all bars.
            valid_mask &= regime_mask


        timestamps: Optional[pd.DatetimeIndex] = None
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce", utc=True))
        elif "time" in df.columns:
            parsed = pd.to_datetime(df["time"], errors="coerce", utc=True)
            if parsed.notna().all():
                timestamps = pd.DatetimeIndex(parsed)

        if timestamps is None or len(timestamps) != n_bars:
            raise ValueError(
                "DataDifficulty session filtering requires one valid timestamp per bar"
            )

        hours = np.asarray(timestamps.hour, dtype=np.int32)
        session_mask = np.zeros(n_bars, dtype=bool)

        if getattr(difficulty, "include_asian_session", True):
            session_mask |= (hours < 8)
        if getattr(difficulty, "include_london_session", True):
            session_mask |= ((hours >= 8) & (hours < 16))
        if getattr(difficulty, "include_ny_session", True):
            session_mask |= ((hours >= 14) & (hours < 22))
        if getattr(difficulty, "include_overlap_sessions", True):
            session_mask |= ((hours >= 14) & (hours < 16))

        # Apply the requested session policy even when it selects nothing.
        # Treating an all-false mask as "no filter" made an invalid
        # configuration silently sample every session.
        valid_mask &= session_mask


        if getattr(difficulty, "exclude_market_open_close", False):
            open_close_mask = ~(
                (hours == 0) | (hours == 8) | (hours == 14) |
                (hours == 7) | (hours == 15) | (hours == 21)
            )
            valid_mask &= open_close_mask


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

            weights = np.zeros(len(valid_indices), dtype=np.float64)

            def _apply_weight(mask: np.ndarray, weight: float) -> None:
                if weight <= 0:
                    return
                idx_mask = mask[valid_indices]
                if idx_mask.any():
                    # Config values describe target mixture mass, not a raw
                    # per-row multiplier. Divide by bucket prevalence so a rare
                    # regime can receive its declared share.
                    weights[idx_mask] += float(weight) / float(idx_mask.sum())

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

            self._regime_weights = weights if np.any(weights > 0.0) else None

        if len(valid_indices) == 0:

            # _data_difficulty is Optional; this branch can be reached before it
            # is set, so read through a local rather than dereferencing it twice.
            diff = self._data_difficulty
            raise ValueError(
                f"DataDifficulty filter found 0 valid indices with settings: "
                f"volatility_range={getattr(diff, 'volatility_percentile_range', None)}, "
                f"min_trend_clarity={getattr(diff, 'min_trend_clarity', None)}, "
                f"sessions=(asia={getattr(diff, 'include_asian_session', True)}, "
                f"london={getattr(diff, 'include_london_session', True)}, "
                f"ny={getattr(diff, 'include_ny_session', True)})."
            )
        else:
            self._valid_start_indices = valid_indices
            logger.debug(f"DataDifficulty filter: {len(valid_indices)} valid start indices out of {max_end - buffer}")

    def _compute_volatility_percentiles(self, df: pd.DataFrame) -> None:
        try:
            if "close" not in df.columns and "Close" not in df.columns:
                raise ValueError("volatility filtering requires a close column")

            close_col = "close" if "close" in df.columns else "Close"
            close = np.asarray(df[close_col].values, dtype=np.float64)


            window = 20
            if len(close) < window + 1:
                raise ValueError(
                    f"volatility filtering requires at least {window + 1} bars"
                )

            returns = np.abs(np.diff(close) / (close[:-1] + 1e-10))


            vol_series = pd.Series(returns).rolling(window=window, min_periods=1).std().fillna(0.0)
            vol = np.concatenate([[0.0], vol_series.to_numpy(dtype=np.float64)])


            # pandas may expose a read-only NumPy view.  We deliberately mutate
            # the first undefined-return percentile below, so request owned,
            # writable storage instead of silently losing every volatility
            # filter on an assignment error.
            percentiles = (
                pd.Series(vol)
                .rank(pct=True, method="average")
                .to_numpy(dtype=np.float64, copy=True)
            )


            if len(percentiles) > 0:
                percentiles[0] = 0.5

            self._volatility_percentiles = percentiles
        except Exception as exc:
            self._volatility_percentiles = None
            raise ValueError("could not compute volatility percentiles") from exc

    def _compute_trend_clarity(self, df: pd.DataFrame) -> np.ndarray:
        try:
            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns:
                raise ValueError("trend filtering requires a close column")

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

            if clarity.size != len(df) or not np.all(np.isfinite(clarity)):
                raise ValueError("trend clarity computation produced invalid values")
            return clarity
        except Exception as exc:
            raise ValueError("could not compute trend clarity") from exc

    def _compute_trend_slope(self, df: pd.DataFrame) -> np.ndarray:
        try:
            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns:
                raise ValueError("trend filtering requires a close column")
            close = np.asarray(df[close_col].values, dtype=np.float64)
            n = len(close)
            slope = np.zeros(n)
            window = 20
            for i in range(window, n):
                segment = close[i-window:i]
                seg_mean = float(np.mean(segment))
                slope[i] = (segment[-1] - segment[0]) / (window * (seg_mean + 1e-10))
            slope[:window] = np.median(slope[window:]) if n > window else 0.0
            if slope.size != len(df) or not np.all(np.isfinite(slope)):
                raise ValueError("trend slope computation produced invalid values")
            return slope
        except Exception as exc:
            raise ValueError("could not compute trend slope") from exc

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
            raise ValueError(
                f"DataDifficulty has no eligible start in [{buffer}, {max_start}); "
                "refusing an out-of-window fallback"
            )


        weights = np.ones(len(valid_in_range), dtype=np.float64)
        if self._regime_weights is not None and len(self._regime_weights) == len(self._valid_start_indices):
            try:
                mask = np.isin(self._valid_start_indices, valid_in_range)
                regime_weights = self._regime_weights[mask]
                if regime_weights.size == len(valid_in_range) and regime_weights.sum() > 0:
                    weights *= regime_weights
            except Exception:
                pass

        if (self._data_difficulty is not None and
            getattr(self._data_difficulty, "prefer_recent_data", False)):
            recent_weight = float(getattr(self._data_difficulty, "recent_data_weight", 1.0) or 1.0)
            if recent_weight > 1.0:
                positions = np.arange(len(valid_in_range), dtype=np.float64)
                weights *= np.exp((recent_weight - 1.0) * positions / max(len(positions), 1))

        regime_start = getattr(self.config, "recent_regime_start_time", None)
        regime_share = float(getattr(self.config, "recent_regime_target_share", 0.0) or 0.0)
        if not np.isfinite(regime_share) or not (0.0 <= regime_share <= 1.0):
            raise ValueError(f"recent_regime_target_share must be in [0, 1], got {regime_share!r}")
        if regime_share > 0.0:
            if regime_start is None:
                raise ValueError("recent_regime_target_share requires recent_regime_start_time")
            primary = self.data.get(self.instruments[0], {}).get(self.config.primary_timeframe)
            time_fn = getattr(self, "_df_time_ns", None)
            time_ns_raw = (
                time_fn(primary)
                if callable(time_fn) and primary is not None
                else None
            )
            if time_ns_raw is None:
                raise ValueError("temporal regime sampling requires primary timestamps")
            time_ns = np.asarray(time_ns_raw)
            if time_ns.ndim != 1 or time_ns.size == 0:
                raise ValueError(
                    "temporal regime sampling requires a non-empty, one-dimensional "
                    "timestamp array"
                )
            if primary is None or time_ns.size != len(primary):
                raise ValueError(
                    "temporal regime timestamps must align one-for-one with primary bars"
                )
            if not np.issubdtype(time_ns.dtype, np.integer):
                raise ValueError(
                    "temporal regime timestamps must use integer nanoseconds"
                )
            cut = pd.Timestamp(regime_start)
            cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
            recent_mask = np.asarray(time_ns[valid_in_range] >= int(cut.value), dtype=bool)
            old_mask = ~recent_mask
            if not recent_mask.any():
                raise ValueError(
                    f"no eligible starts exist on/after recent regime boundary {cut.isoformat()}"
                )
            if regime_share < 1.0 and not old_mask.any():
                raise ValueError(
                    f"no eligible pre-regime starts exist before {cut.isoformat()} for target share {regime_share}"
                )

            recent_mass = float(weights[recent_mask].sum())
            old_mass = float(weights[old_mask].sum())
            if recent_mass <= 0.0 or (regime_share < 1.0 and old_mass <= 0.0):
                raise ValueError("temporal regime buckets have no positive sampling weight")
            weights[recent_mask] *= regime_share / recent_mass
            if old_mask.any():
                weights[old_mask] *= (1.0 - regime_share) / old_mass

        if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
            raise ValueError("DataDifficulty produced invalid sampling weights")
        weights /= weights.sum()
        return int(self.np_random.choice(valid_in_range, p=weights))
