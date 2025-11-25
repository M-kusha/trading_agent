# modules/memory/shared/bar_signature.py
"""
Bar Signature Helper
Transforms short OHLCV windows into compact shape features for pattern recognition.

Features extracted (8-12 dims):
- slope_10: 10-period price slope
- slope_30: 30-period price slope  
- atr_jump: ATR change relative to mean
- compression_z: Bollinger Band compression z-score
- breakout_dist: Distance to recent breakout level
- wick_body_ratio: Candle wick to body ratio
- range_frac: Current range vs historical range
- shape_code: Encoded candle pattern shape
- trend_strength: Trend strength indicator
- momentum_divergence: Price-momentum divergence
- volume_profile: Volume relative to average
- price_position: Price position within range
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class BarSignature:
    """
    Computes compact bar signature features from OHLCV data.
    
    These features capture the "shape" of recent price action in a way
    that's useful for pattern recognition and similarity matching.
    """
    
    # Target output dimension
    OUTPUT_DIM: int = 12
    
    # Numerical stability
    _EPS: float = 1e-10
    
    def __init__(
        self,
        lookback_short: int = 10,
        lookback_long: int = 30,
        atr_period: int = 14,
    ) -> None:
        """
        Initialize bar signature calculator.
        
        Args:
            lookback_short: Short-term lookback for slope calculation
            lookback_long: Long-term lookback for slope calculation
            atr_period: Period for ATR calculation
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.atr_period = atr_period
    
    def extract(
        self,
        ohlcv: Union[np.ndarray, List[List[float]], Dict[str, np.ndarray]],
        min_bars: int = 5,
    ) -> np.ndarray:
        """
        Extract bar signature features from OHLCV data.
        
        Args:
            ohlcv: OHLCV data in one of these formats:
                - np.ndarray of shape [N, 5] where columns are [open, high, low, close, volume]
                - List of [open, high, low, close, volume] lists
                - Dict with keys 'open', 'high', 'low', 'close', 'volume' as arrays
            min_bars: Minimum number of bars required
            
        Returns:
            Feature vector of shape [OUTPUT_DIM]
        """
        # Normalize input format
        open_arr, high_arr, low_arr, close_arr, volume_arr = self._normalize_ohlcv(ohlcv)
        
        n_bars = len(close_arr)
        
        # Return zeros if insufficient data
        if n_bars < min_bars:
            return np.zeros(self.OUTPUT_DIM, dtype=np.float32)
        
        features: List[float] = []
        
        # 1. slope_10: Short-term price slope (normalized)
        slope_10 = self._calculate_slope(close_arr, min(self.lookback_short, n_bars))
        features.append(float(np.clip(slope_10, -1.0, 1.0)))
        
        # 2. slope_30: Long-term price slope (normalized)
        slope_30 = self._calculate_slope(close_arr, min(self.lookback_long, n_bars))
        features.append(float(np.clip(slope_30, -1.0, 1.0)))
        
        # 3. atr_jump: ATR change relative to mean
        atr_jump = self._calculate_atr_jump(high_arr, low_arr, close_arr)
        features.append(float(np.clip(atr_jump, -2.0, 2.0)))
        
        # 4. compression_z: Bollinger Band compression (z-score)
        compression_z = self._calculate_compression(close_arr)
        features.append(float(np.clip(compression_z, -3.0, 3.0)))
        
        # 5. breakout_dist: Distance to recent breakout level
        breakout_dist = self._calculate_breakout_distance(high_arr, low_arr, close_arr)
        features.append(float(np.clip(breakout_dist, -1.0, 1.0)))
        
        # 6. wick_body_ratio: Recent candle wick to body ratio
        wick_body_ratio = self._calculate_wick_body_ratio(open_arr, high_arr, low_arr, close_arr)
        features.append(float(np.clip(wick_body_ratio, 0.0, 5.0) / 5.0))  # Normalize to [0,1]
        
        # 7. range_frac: Current range vs historical range
        range_frac = self._calculate_range_fraction(high_arr, low_arr)
        features.append(float(np.clip(range_frac, 0.0, 2.0) / 2.0))  # Normalize to [0,1]
        
        # 8. shape_code: Encoded candle pattern (one-hot-ish)
        shape_code = self._calculate_shape_code(open_arr, high_arr, low_arr, close_arr)
        features.append(float(shape_code))
        
        # 9. trend_strength: Trend strength indicator
        trend_strength = self._calculate_trend_strength(close_arr, high_arr, low_arr)
        features.append(float(np.clip(trend_strength, 0.0, 1.0)))
        
        # 10. momentum_divergence: Price-momentum divergence
        momentum_div = self._calculate_momentum_divergence(close_arr)
        features.append(float(np.clip(momentum_div, -1.0, 1.0)))
        
        # 11. volume_profile: Volume relative to average
        volume_profile = self._calculate_volume_profile(volume_arr)
        features.append(float(np.clip(volume_profile, 0.0, 3.0) / 3.0))  # Normalize to [0,1]
        
        # 12. price_position: Price position within recent range
        price_position = self._calculate_price_position(high_arr, low_arr, close_arr)
        features.append(float(np.clip(price_position, 0.0, 1.0)))
        
        result = np.asarray(features, dtype=np.float32)
        assert result.shape[0] == self.OUTPUT_DIM, f"Expected {self.OUTPUT_DIM} features, got {result.shape[0]}"
        
        return result
    
    def _normalize_ohlcv(
        self,
        ohlcv: Union[np.ndarray, List[List[float]], Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normalize OHLCV input to separate arrays."""
        if isinstance(ohlcv, dict):
            open_arr = np.asarray(ohlcv.get("open", []), dtype=np.float64).reshape(-1)
            high_arr = np.asarray(ohlcv.get("high", []), dtype=np.float64).reshape(-1)
            low_arr = np.asarray(ohlcv.get("low", []), dtype=np.float64).reshape(-1)
            close_arr = np.asarray(ohlcv.get("close", []), dtype=np.float64).reshape(-1)
            volume_arr = np.asarray(ohlcv.get("volume", []), dtype=np.float64).reshape(-1)
        else:
            arr = np.asarray(ohlcv, dtype=np.float64)
            if arr.ndim == 1:
                # Single row: assume [open, high, low, close, volume]
                arr = arr.reshape(1, -1)
            
            # Ensure we have 5 columns
            if arr.shape[1] >= 5:
                open_arr = arr[:, 0]
                high_arr = arr[:, 1]
                low_arr = arr[:, 2]
                close_arr = arr[:, 3]
                volume_arr = arr[:, 4]
            else:
                # Fall back: use what we have
                n = arr.shape[1]
                open_arr = arr[:, 0] if n > 0 else np.array([0.0])
                high_arr = arr[:, 1] if n > 1 else open_arr
                low_arr = arr[:, 2] if n > 2 else open_arr
                close_arr = arr[:, 3] if n > 3 else open_arr
                volume_arr = arr[:, 4] if n > 4 else np.ones_like(open_arr)
        
        # Ensure minimum length
        min_len = min(len(open_arr), len(high_arr), len(low_arr), len(close_arr), len(volume_arr))
        if min_len == 0:
            return (
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([1.0]),
            )
        
        return (
            open_arr[:min_len],
            high_arr[:min_len],
            low_arr[:min_len],
            close_arr[:min_len],
            volume_arr[:min_len],
        )
    
    def _calculate_slope(self, close: np.ndarray, period: int) -> float:
        """Calculate normalized price slope over period."""
        if len(close) < 2:
            return 0.0
        
        period = min(period, len(close))
        prices = close[-period:]
        
        if len(prices) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(prices))
        x_mean = np.mean(x)
        y_mean = np.mean(prices)
        
        numerator = np.sum((x - x_mean) * (prices - y_mean))
        denominator = np.sum((x - x_mean) ** 2) + self._EPS
        
        slope = numerator / denominator
        
        # Normalize by price level
        normalized = slope / (np.mean(np.abs(prices)) + self._EPS)
        
        return float(normalized)
    
    def _calculate_atr_jump(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate ATR change relative to mean ATR."""
        if len(high) < 2:
            return 0.0
        
        period = min(self.atr_period, len(high))
        
        # True Range
        high_p = high[-period:]
        low_p = low[-period:]
        close_p = close[-period:]
        
        prev_close = np.roll(close_p, 1)
        prev_close[0] = close_p[0]
        
        tr1 = high_p - low_p
        tr2 = np.abs(high_p - prev_close)
        tr3 = np.abs(low_p - prev_close)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR
        atr = np.mean(true_range)
        current_tr = true_range[-1] if len(true_range) > 0 else 0.0
        
        # Jump relative to mean
        jump = (current_tr - atr) / (atr + self._EPS)
        
        return float(jump)
    
    def _calculate_compression(self, close: np.ndarray) -> float:
        """Calculate Bollinger Band compression as z-score."""
        period = min(20, len(close))
        if period < 3:
            return 0.0
        
        prices = close[-period:]
        
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std < self._EPS:
            return 0.0
        
        # Historical std for comparison
        if len(close) > period:
            hist_std = np.std(close[-period * 2:-period]) if len(close) >= period * 2 else std
        else:
            hist_std = std
        
        # Compression ratio (low std = compression)
        compression = (hist_std - std) / (hist_std + self._EPS)
        
        return float(compression)
    
    def _calculate_breakout_distance(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate distance from recent breakout level."""
        if len(close) < 5:
            return 0.0
        
        period = min(20, len(close))
        recent_high = np.max(high[-period:])
        recent_low = np.min(low[-period:])
        current = close[-1]
        
        range_size = recent_high - recent_low
        if range_size < self._EPS:
            return 0.0
        
        # Distance from high as fraction of range (positive = near high)
        dist_from_high = (recent_high - current) / range_size
        dist_from_low = (current - recent_low) / range_size
        
        # Signed distance: positive if closer to high, negative if closer to low
        if dist_from_high < dist_from_low:
            return float(-dist_from_high)  # Near high
        else:
            return float(dist_from_low - 1)  # Near low
    
    def _calculate_wick_body_ratio(
        self,
        open_arr: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """Calculate wick to body ratio for recent candle."""
        if len(open_arr) < 1:
            return 0.0
        
        o = open_arr[-1]
        h = high[-1]
        l = low[-1]
        c = close[-1]
        
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_wick = upper_wick + lower_wick
        
        if body < self._EPS:
            return 5.0  # Very large ratio for doji
        
        return float(total_wick / body)
    
    def _calculate_range_fraction(self, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate current range vs historical range."""
        if len(high) < 2:
            return 1.0
        
        period = min(20, len(high))
        
        current_range = high[-1] - low[-1]
        hist_ranges = high[-period:-1] - low[-period:-1] if len(high) > 1 else np.array([current_range])
        
        mean_range = np.mean(hist_ranges) if len(hist_ranges) > 0 else current_range
        
        if mean_range < self._EPS:
            return 1.0
        
        return float(current_range / mean_range)
    
    def _calculate_shape_code(
        self,
        open_arr: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """
        Calculate encoded shape of recent candle pattern.
        
        Returns value in [0, 1] encoding pattern type:
        - 0.0-0.2: Strong bearish (long body down)
        - 0.2-0.4: Weak bearish (small body down)
        - 0.4-0.6: Doji/indecision
        - 0.6-0.8: Weak bullish (small body up)
        - 0.8-1.0: Strong bullish (long body up)
        """
        if len(open_arr) < 1:
            return 0.5
        
        o = open_arr[-1]
        h = high[-1]
        l = low[-1]
        c = close[-1]
        
        range_size = h - l
        if range_size < self._EPS:
            return 0.5  # No range = indecision
        
        body = c - o
        body_ratio = abs(body) / range_size
        
        # Classify
        if abs(body_ratio) < 0.2:
            return 0.5  # Doji
        elif body > 0:
            # Bullish
            return 0.6 + 0.4 * min(1.0, body_ratio)
        else:
            # Bearish
            return 0.4 - 0.4 * min(1.0, body_ratio)
    
    def _calculate_trend_strength(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate trend strength using ADX-like logic."""
        if len(close) < 3:
            return 0.0
        
        period = min(14, len(close))
        
        # Simplified directional movement
        close_p = close[-period:]
        high_p = high[-period:]
        low_p = low[-period:]
        
        # +DM and -DM
        high_diff = np.diff(high_p)
        low_diff = -np.diff(low_p)
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # True range for normalization
        tr = np.maximum(
            high_p[1:] - low_p[1:],
            np.maximum(
                np.abs(high_p[1:] - close_p[:-1]),
                np.abs(low_p[1:] - close_p[:-1])
            )
        )
        
        atr = np.mean(tr) + self._EPS
        
        # Smoothed DI
        plus_di = np.mean(plus_dm) / atr
        minus_di = np.mean(minus_dm) / atr
        
        # DX
        di_sum = plus_di + minus_di + self._EPS
        dx = abs(plus_di - minus_di) / di_sum
        
        return float(dx)
    
    def _calculate_momentum_divergence(self, close: np.ndarray) -> float:
        """Calculate price-momentum divergence."""
        if len(close) < 5:
            return 0.0
        
        period = min(14, len(close))
        prices = close[-period:]
        
        # Price direction
        price_change = prices[-1] - prices[0]
        price_dir = 1 if price_change > 0 else -1 if price_change < 0 else 0
        
        # Momentum (rate of change)
        momentum = (prices[-1] - prices[-5]) / (prices[-5] + self._EPS) if len(prices) >= 5 else 0
        momentum_dir = 1 if momentum > 0 else -1 if momentum < 0 else 0
        
        # Divergence: opposite signs indicate divergence
        if price_dir != momentum_dir and price_dir != 0 and momentum_dir != 0:
            # Bearish divergence: price up, momentum down
            if price_dir > 0:
                return -abs(momentum)
            else:
                return abs(momentum)
        
        return 0.0
    
    def _calculate_volume_profile(self, volume: np.ndarray) -> float:
        """Calculate current volume relative to average."""
        if len(volume) < 2:
            return 1.0
        
        period = min(20, len(volume))
        avg_volume = np.mean(volume[-period:-1]) if len(volume) > 1 else volume[-1]
        
        if avg_volume < self._EPS:
            return 1.0
        
        return float(volume[-1] / avg_volume)
    
    def _calculate_price_position(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate price position within recent range [0=low, 1=high]."""
        if len(close) < 2:
            return 0.5
        
        period = min(20, len(close))
        recent_high = np.max(high[-period:])
        recent_low = np.min(low[-period:])
        current = close[-1]
        
        range_size = recent_high - recent_low
        if range_size < self._EPS:
            return 0.5
        
        position = (current - recent_low) / range_size
        
        return float(np.clip(position, 0.0, 1.0))


# Convenience function
def extract_bar_signature(
    ohlcv: Union[np.ndarray, List[List[float]], Dict[str, np.ndarray]],
    lookback_short: int = 10,
    lookback_long: int = 30,
) -> np.ndarray:
    """
    Extract bar signature features from OHLCV data.
    
    Convenience wrapper around BarSignature.extract().
    
    Args:
        ohlcv: OHLCV data
        lookback_short: Short-term lookback period
        lookback_long: Long-term lookback period
        
    Returns:
        Feature vector of shape [12]
    """
    extractor = BarSignature(lookback_short=lookback_short, lookback_long=lookback_long)
    return extractor.extract(ohlcv)
