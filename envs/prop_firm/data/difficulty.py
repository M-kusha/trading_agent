# envs/prop_firm/data/difficulty.py
# pyright: reportAttributeAccessIssue=false
"""
Data difficulty mixin for PropFirmTradingEnv.

Contains methods for curriculum-based data filtering.
"""

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
    """Mixin providing data difficulty methods for curriculum learning.
    
    Expected attributes from PropFirmTradingEnv:
    - config: PropFirmConfig
    - data: Dict[str, Dict[str, pd.DataFrame]]
    - instruments: List[str]
    - np_random: Generator (inherited from gym.Env)
    - _min_data_len: int
    """
    
    # Type hints for attributes provided by PropFirmTradingEnv
    # NOTE: np_random is NOT declared here - it's a property from gym.Env
    # Declaring it here causes incompatible override errors
    config: "PropFirmConfig"
    data: Dict[str, Dict[str, pd.DataFrame]]
    instruments: List[str]
    _min_data_len: int
    _data_difficulty: Optional[Any]
    _valid_start_indices: Optional[np.ndarray]
    _volatility_percentiles: Optional[np.ndarray]
    _difficulty_cache_hash: Optional[int]

    def set_data_difficulty(self, difficulty: Any) -> None:
        """
        Set data difficulty filtering for curriculum-based learning.
        
        Args:
            difficulty: DataDifficulty config specifying which market conditions to train on.
                        Early stages use easier conditions (clear trends, lower volatility).
        """
        self._data_difficulty = difficulty
        self._valid_start_indices = None  # Force recomputation
        self._volatility_percentiles = None
        self._difficulty_cache_hash: Optional[int] = None
        
        if difficulty is not None:
            self._precompute_data_difficulty_indices()

    def _precompute_data_difficulty_indices(self) -> None:
        """
        Pre-compute valid episode starting indices based on data difficulty settings.
        
        This avoids expensive per-reset filtering by caching valid positions.
        Uses hash-based caching to skip redundant computation when settings unchanged.
        """
        # Compute config hash to detect if recomputation needed
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
            ))
            
            # Skip if already computed with same settings
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

        # Initialize all indices as valid
        valid_mask = np.ones(n_bars, dtype=bool)

        # Apply volatility filter
        vol_range = getattr(difficulty, "volatility_percentile_range", (0.0, 1.0))
        if vol_range != (0.0, 1.0):
            self._compute_volatility_percentiles(df)
            if self._volatility_percentiles is not None:
                valid_mask &= (self._volatility_percentiles >= vol_range[0])
                valid_mask &= (self._volatility_percentiles <= vol_range[1])

        # Apply trend clarity filter
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

        # Apply session filters
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                hours = np.asarray(df.index.hour, dtype=np.int32)
                if hours is not None:
                    session_mask = np.zeros(n_bars, dtype=bool)
                    
                    # Session hours (approximate, Europe/Berlin perspective)
                    # Asian: 00:00 - 08:00
                    # London: 08:00 - 16:00
                    # NY: 14:00 - 22:00
                    # Overlap (London/NY): 14:00 - 16:00
                    
                    if getattr(difficulty, "include_asian_session", True):
                        session_mask |= (hours < 8)
                    if getattr(difficulty, "include_london_session", True):
                        session_mask |= ((hours >= 8) & (hours < 16))
                    if getattr(difficulty, "include_ny_session", True):
                        session_mask |= ((hours >= 14) & (hours < 22))
                    if getattr(difficulty, "include_overlap_sessions", True):
                        session_mask |= ((hours >= 14) & (hours < 16))
                    
                    # If at least one session enabled, apply filter
                    if session_mask.any():
                        valid_mask &= session_mask
            except Exception as e:
                logger.debug(f"Skip session filtering: {e}")

        # Apply market open/close filter
        if getattr(difficulty, "exclude_market_open_close", False):
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    hours = np.asarray(df.index.hour, dtype=np.int32)
                    # Exclude first/last hour of major sessions
                    open_close_mask = ~(
                        (hours == 0) | (hours == 8) | (hours == 14) |  # Opens
                        (hours == 7) | (hours == 15) | (hours == 21)   # Closes
                    )
                    valid_mask &= open_close_mask
            except Exception as e:
                logger.debug(f"Skip market open/close filter: {e}")

        # Restrict to valid start range
        range_mask = np.zeros(n_bars, dtype=bool)
        range_mask[buffer:max_end] = True
        valid_mask &= range_mask

        # Get valid indices
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # Fallback: use all indices in valid range
            logger.warning(
                f"DataDifficulty filter found 0 valid indices with settings: "
                f"volatility_range={self._data_difficulty.volatility_percentile_range}, "
                f"min_trend_clarity={self._data_difficulty.min_trend_clarity}, "
                f"sessions=(asia={getattr(self._data_difficulty, 'include_asian_session', True)}, "
                f"london={getattr(self._data_difficulty, 'include_london_session', True)}, "
                f"ny={getattr(self._data_difficulty, 'include_ny_session', True)}). "
                f"Falling back to full dataset ({max_end - buffer} bars)."
            )
            self._valid_start_indices = np.arange(buffer, max_end)
        else:
            self._valid_start_indices = valid_indices
            logger.debug(f"DataDifficulty filter: {len(valid_indices)} valid start indices out of {max_end - buffer}")

    def _compute_volatility_percentiles(self, df: pd.DataFrame) -> None:
        """
        Compute volatility percentile for each bar.
        
        Uses O(n log n) rank-based algorithm instead of O(n²) expanding window.
        """
        try:
            if "close" not in df.columns and "Close" not in df.columns:
                self._volatility_percentiles = None
                return
            
            close_col = "close" if "close" in df.columns else "Close"
            close = np.asarray(df[close_col].values, dtype=np.float64)
            
            # Rolling ATR-like volatility (20-bar)
            window = 20
            if len(close) < window + 1:
                self._volatility_percentiles = None
                return
            
            returns = np.abs(np.diff(close) / (close[:-1] + 1e-10))
            
            # Use pandas rolling + rank instead of manual loop + scipy.rankdata
            vol_series = pd.Series(returns).rolling(window=window, min_periods=1).std().fillna(0.0)
            vol = np.concatenate([[0.0], vol_series.to_numpy(dtype=np.float64)])
            
            # Percentile ranks in [0,1] using pandas
            percentiles = pd.Series(vol).rank(pct=True, method="average").to_numpy(dtype=np.float64)
            
            # First bar gets default 0.5 (median assumption for unknown)
            if len(percentiles) > 0:
                percentiles[0] = 0.5
            
            self._volatility_percentiles = percentiles
        except Exception:
            self._volatility_percentiles = None

    def _compute_trend_clarity(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute trend clarity for each bar.
        
        Uses a simple measure: abs(SMA slope) normalized by volatility.
        High values = clear trend, low values = choppy/ranging.
        """
        try:
            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns:
                return np.ones(len(df))  # Default: all clear
            
            close = np.asarray(df[close_col].values, dtype=np.float64)
            n = len(close)
            clarity = np.zeros(n)
            
            window = 20
            for i in range(window, n):
                segment = close[i-window:i]
                seg_mean = float(np.mean(segment))
                slope = (segment[-1] - segment[0]) / (window * (seg_mean + 1e-10))
                vol = float(np.std(np.diff(segment))) / (seg_mean + 1e-10)
                
                # Clarity = trend strength / noise
                clarity[i] = min(1.0, abs(slope) / (vol + 1e-10))
            
            # First bars get median clarity
            clarity[:window] = np.median(clarity[window:]) if n > window else 0.5
            
            return clarity
        except Exception:
            return np.ones(len(df))

    def _sample_episode_start_with_difficulty(self, buffer: int, max_start: int) -> int:
        """
        Sample episode starting position respecting data difficulty settings.
        
        Args:
            buffer: Minimum starting index (lookback buffer)
            max_start: Maximum starting index
            
        Returns:
            Starting bar index for this episode
        """
        if self._valid_start_indices is None or len(self._valid_start_indices) == 0:
            # No difficulty filtering - use uniform random
            if max_start > buffer:
                return int(self.np_random.integers(buffer, max_start))
            return min(buffer, max(self._min_data_len - 2, 0))

        # Filter to valid range
        valid_in_range = self._valid_start_indices[
            (self._valid_start_indices >= buffer) & 
            (self._valid_start_indices < max_start)
        ]
        
        if len(valid_in_range) == 0:
            # Fallback to any valid index
            if len(self._valid_start_indices) > 0:
                return int(self.np_random.choice(self._valid_start_indices))
            if max_start > buffer:
                return int(self.np_random.integers(buffer, max_start))
            return min(buffer, max(self._min_data_len - 2, 0))

        # Apply recency weighting if configured
        if (self._data_difficulty is not None and 
            getattr(self._data_difficulty, "prefer_recent_data", False)):
            
            weight = getattr(self._data_difficulty, "recent_data_weight", 1.0)
            if weight > 1.0:
                # Exponential weighting toward recent data
                positions = np.arange(len(valid_in_range))
                weights = np.exp(weight * positions / len(positions))
                weights /= weights.sum()
                idx = self.np_random.choice(len(valid_in_range), p=weights)
                return int(valid_in_range[idx])

        # Uniform random from valid indices
        return int(self.np_random.choice(valid_in_range))
