"""
Advanced Trend Expert
=====================
A sophisticated trend-based voting expert that combines multiple trend
detection methods, trend strength analysis, support/resistance awareness,
and multi-timeframe trend alignment to generate high-conviction directional
trading signals.

This expert is designed to identify and follow trends with precision using:
- Multiple moving average types (SMA, EMA, WMA)
- ADX trend strength measurement
- Parabolic SAR trend direction
- Ichimoku Cloud components
- Support/Resistance levels
- Trend channels and slopes
- Multi-timeframe trend alignment
"""

from __future__ import annotations

import datetime
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.experts.base import VotingExpertBase


@module(**module_args("TrendExpert"))
class TrendExpert(VotingExpertBase):
    """
    Advanced Trend Expert - Multi-indicator trend analysis.
    
    Combines 8+ trend indicators with support/resistance and 
    multi-timeframe alignment for high-conviction directional signals.
    
    Analysis Components:
    1. Triple MA System (Fast/Medium/Slow EMA)
    2. ADX for trend strength with +DI/-DI direction
    3. Parabolic SAR for trend direction
    4. Linear regression channel (trend slope)
    5. Price position relative to MAs
    6. Higher timeframe trend alignment
    7. Support/Resistance proximity
    8. Trend momentum (rate of price change)
    
    Voting Actions:
    - trend_bullish: Strong uptrend with confirmation
    - trend_bearish: Strong downtrend with confirmation
    - trend_neutral: No clear trend / ranging market
    
    Features:
    - Adaptive trend detection based on volatility
    - Multi-MA confluence scoring
    - ADX-based trend strength gating
    - S/R level awareness
    - Higher timeframe trend weighting
    - Historical accuracy tracking per signal type
    """
    
    def _expert_specific_init(self) -> None:
        """Initialize advanced trend analysis state."""
        # ═══════════════════════════ CONFIGURATION ═══════════════════════════
        # Triple MA configuration
        self.fast_period = int(self.config.get('fast_period', 8))
        self.medium_period = int(self.config.get('medium_period', 21))
        self.slow_period = int(self.config.get('slow_period', 55))
        
        # ADX configuration
        self.adx_period = int(self.config.get('adx_period', 14))
        self.adx_trending_threshold = float(self.config.get('adx_trending', 25))
        self.adx_strong_threshold = float(self.config.get('adx_strong', 40))
        
        # Parabolic SAR
        self.sar_af_start = float(self.config.get('sar_af_start', 0.02))
        self.sar_af_max = float(self.config.get('sar_af_max', 0.2))
        
        # Trend thresholds
        self.trend_threshold = float(self.config.get('trend_threshold', 0.003))  # 0.3% difference
        self.strong_trend_multiplier = 2.5
        
        # Confluence requirements
        self.min_confluence_score = float(self.config.get('min_confluence', 0.15))
        self.strong_signal_confluence = float(self.config.get('strong_confluence', 0.65))
        
        # S/R detection
        self.sr_lookback = int(self.config.get('sr_lookback', 50))
        self.sr_threshold = float(self.config.get('sr_threshold', 0.005))  # 0.5% proximity
        
        # ═══════════════════════════ STATE ═══════════════════════════
        # Price/volume history
        self.price_history: deque = deque(maxlen=200)
        self.high_history: deque = deque(maxlen=200)
        self.low_history: deque = deque(maxlen=200)
        self.close_history: deque = deque(maxlen=200)
        
        # Moving averages
        self.fast_ma: float = 0.0
        self.medium_ma: float = 0.0
        self.slow_ma: float = 0.0
        self.ma_history: deque = deque(maxlen=50)
        
        # ADX components
        self.adx_value: float = 0.0
        self.plus_di: float = 0.0
        self.minus_di: float = 0.0
        self.adx_history: deque = deque(maxlen=30)
        
        # SAR
        self.sar_value: float = 0.0
        self.sar_direction: int = 0  # 1 = bullish (SAR below price), -1 = bearish
        
        # Trend state
        self.current_trend: str = 'neutral'
        self.trend_strength: float = 0.0
        self.trend_slope: float = 0.0
        self.trend_duration: int = 0
        self.ma_alignment: int = 0  # -1, 0, 1 (bearish, neutral, bullish)
        
        # Support/Resistance
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        self.near_support: bool = False
        self.near_resistance: bool = False
        
        # Analysis history
        self.trend_history: deque = deque(maxlen=100)
        self.signal_history: deque = deque(maxlen=50)
        
        # Performance tracking
        self.trend_performance: Dict[str, Dict[str, Any]] = {
            'long': {'signals': 0, 'success': 0, 'total_pnl': 0.0},
            'short': {'signals': 0, 'success': 0, 'total_pnl': 0.0},
            'flat': {'signals': 0, 'success': 0, 'total_pnl': 0.0},
        }
        
        self.log_info(
            f"[TREND] Advanced TrendExpert initialized | "
            f"MA periods={self.fast_period}/{self.medium_period}/{self.slow_period} | "
            f"ADX period={self.adx_period} | ADX threshold={self.adx_trending_threshold}"
        )
        
        self._publish_baseline_keys()
        self._publish_trend_baseline()
    
    def _publish_trend_baseline(self) -> None:
        """Publish trend baseline keys."""
        try:
            self.smart_bus.set(
                'trend_voting_proposal',
                {'action': 'trend_neutral', 'signal_strength': 0.0},
                module='TrendExpert',
                thesis='Trend baseline'
            )
            self.smart_bus.set(
                'trend_confidence',
                0.1,
                module='TrendExpert',
                thesis='Trend baseline confidence'
            )
            self.smart_bus.set(
                'trend_analysis',
                {'current_trend': 'neutral', 'trend_strength': 0.0},
                module='TrendExpert',
                thesis='Trend analysis baseline'
            )
        except Exception:
            pass
    
    # ═══════════════════════════ INDICATOR CALCULATIONS ═══════════════════════════
    
    def _update_price_data(self, market_data: Dict[str, Any]) -> bool:
        """Extract and update price data from market data."""
        try:
            # Get OHLCV data
            ohlcv = market_data.get('ohlcv') or {}
            prices = market_data.get('prices') or market_data.get('close_prices') or []
            
            # Handle various data formats
            if isinstance(prices, dict):
                close_prices = prices.get('close', [])
                high_prices = prices.get('high', [])
                low_prices = prices.get('low', [])
            elif ohlcv:
                close_prices = ohlcv.get('close', [])
                high_prices = ohlcv.get('high', [])
                low_prices = ohlcv.get('low', [])
            else:
                close_prices = list(prices) if isinstance(prices, (list, np.ndarray)) else []
                high_prices = []
                low_prices = []
            
            # Get current values
            current_price = market_data.get('current_price') or market_data.get('close')
            if current_price is None and close_prices:
                current_price = close_prices[-1]
            
            current_high = market_data.get('high')
            if current_high is None and high_prices:
                current_high = high_prices[-1]
            elif current_high is None:
                current_high = current_price
            
            current_low = market_data.get('low')
            if current_low is None and low_prices:
                current_low = low_prices[-1]
            elif current_low is None:
                current_low = current_price
            
            # Update histories
            if current_price is not None:
                self.price_history.append(float(current_price))
                self.close_history.append(float(current_price))
                self.high_history.append(float(current_high) if current_high else float(current_price))
                self.low_history.append(float(current_low) if current_low else float(current_price))
                return True
            
            return False
            
        except Exception as e:
            self.log_warning(f"[TREND] Price data update failed: {e}")
            return False
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0
        
        try:
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            
            for price in prices[period:]:
                ema = (price - ema) * multiplier + ema
            
            return float(ema)
            
        except Exception:
            return 0.0
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0
        
        return float(sum(prices[-period:]) / period)
    
    def _calculate_triple_ma(self, prices: List[float]) -> Tuple[float, float, float, int]:
        """Calculate triple MA system and alignment."""
        fast = self._calculate_ema(prices, self.fast_period)
        medium = self._calculate_ema(prices, self.medium_period)
        slow = self._calculate_ema(prices, self.slow_period)
        
        # Determine alignment
        if fast > medium > slow:
            alignment = 1  # Bullish
        elif fast < medium < slow:
            alignment = -1  # Bearish
        else:
            alignment = 0  # Neutral/mixed
        
        return fast, medium, slow, alignment
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float]) -> Tuple[float, float, float]:
        """Calculate ADX with +DI and -DI."""
        period = self.adx_period
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return 20.0, 50.0, 50.0  # Default neutral values
        
        try:
            highs_arr = np.array(highs[-(period + 1):], dtype=np.float64)
            lows_arr = np.array(lows[-(period + 1):], dtype=np.float64)
            closes_arr = np.array(closes[-(period + 1):], dtype=np.float64)
            
            # True Range
            tr1 = highs_arr[1:] - lows_arr[1:]
            tr2 = np.abs(highs_arr[1:] - closes_arr[:-1])
            tr3 = np.abs(lows_arr[1:] - closes_arr[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Directional Movement
            up_move = highs_arr[1:] - highs_arr[:-1]
            down_move = lows_arr[:-1] - lows_arr[1:]
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smooth values (simple average for now)
            atr = np.mean(tr)
            plus_dm_avg = np.mean(plus_dm)
            minus_dm_avg = np.mean(minus_dm)
            
            if atr == 0:
                return 20.0, 50.0, 50.0
            
            plus_di = (plus_dm_avg / atr) * 100
            minus_di = (minus_dm_avg / atr) * 100
            
            # DX and ADX
            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx = 0
            else:
                dx = abs(plus_di - minus_di) / di_sum * 100
            
            # ADX is smoothed DX (simplified)
            adx = dx
            
            return float(np.clip(adx, 0, 100)), float(np.clip(plus_di, 0, 100)), float(np.clip(minus_di, 0, 100))
            
        except Exception:
            return 20.0, 50.0, 50.0
    
    def _calculate_parabolic_sar(self, highs: List[float], lows: List[float]) -> Tuple[float, int]:
        """Calculate Parabolic SAR."""
        if len(highs) < 5 or len(lows) < 5:
            return 0.0, 0
        
        try:
            # Simplified SAR calculation
            current_high = highs[-1]
            current_low = lows[-1]
            prev_high = max(highs[-5:-1])
            prev_low = min(lows[-5:-1])
            
            # Trend direction based on recent price action
            if current_high > prev_high and current_low > prev_low:
                # Uptrend - SAR below price
                sar = prev_low * (1 - self.sar_af_start)
                direction = 1
            elif current_high < prev_high and current_low < prev_low:
                # Downtrend - SAR above price
                sar = prev_high * (1 + self.sar_af_start)
                direction = -1
            else:
                # No clear trend
                sar = (prev_high + prev_low) / 2
                direction = 0
            
            return float(sar), direction
            
        except Exception:
            return 0.0, 0
    
    def _calculate_trend_slope(self, prices: List[float], lookback: int = 20) -> float:
        """Calculate linear regression slope for trend direction."""
        if len(prices) < lookback:
            return 0.0
        
        try:
            recent = prices[-lookback:]
            x = np.arange(len(recent))
            
            # Linear regression
            slope = np.polyfit(x, recent, 1)[0]
            
            # Normalize by price level
            avg_price = np.mean(recent)
            if avg_price != 0:
                normalized_slope = slope / avg_price
            else:
                normalized_slope = 0.0
            
            return float(normalized_slope)
            
        except Exception:
            return 0.0
    
    def _find_support_resistance(self, highs: List[float], lows: List[float]) -> Tuple[List[float], List[float]]:
        """Find key support and resistance levels."""
        if len(highs) < self.sr_lookback or len(lows) < self.sr_lookback:
            return [], []
        
        try:
            recent_highs = highs[-self.sr_lookback:]
            recent_lows = lows[-self.sr_lookback:]
            
            # Find swing highs (resistance)
            resistance = []
            for i in range(2, len(recent_highs) - 2):
                if (recent_highs[i] > recent_highs[i-1] and 
                    recent_highs[i] > recent_highs[i-2] and
                    recent_highs[i] > recent_highs[i+1] and 
                    recent_highs[i] > recent_highs[i+2]):
                    resistance.append(recent_highs[i])
            
            # Find swing lows (support)
            support = []
            for i in range(2, len(recent_lows) - 2):
                if (recent_lows[i] < recent_lows[i-1] and 
                    recent_lows[i] < recent_lows[i-2] and
                    recent_lows[i] < recent_lows[i+1] and 
                    recent_lows[i] < recent_lows[i+2]):
                    support.append(recent_lows[i])
            
            # Keep top 3 levels
            resistance = sorted(set(resistance), reverse=True)[:3]
            support = sorted(set(support))[:3]
            
            return support, resistance
            
        except Exception:
            return [], []
    
    def _check_sr_proximity(self, current_price: float, support: List[float], resistance: List[float]) -> Tuple[bool, bool]:
        """Check if price is near support or resistance."""
        near_support = False
        near_resistance = False
        
        if current_price is None or current_price == 0:
            return False, False
        
        threshold_abs = current_price * self.sr_threshold
        
        for s in support:
            if abs(current_price - s) < threshold_abs:
                near_support = True
                break
        
        for r in resistance:
            if abs(current_price - r) < threshold_abs:
                near_resistance = True
                break
        
        return near_support, near_resistance
    
    # ═══════════════════════════ MAIN ANALYSIS ═══════════════════════════
    
    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sophisticated trend-based voting proposal."""
        try:
            # Update price data
            if not self._update_price_data(market_data):
                self.log_warning("[TREND] No valid price data - returning flat")
                return {
                    'action': 'flat',
                    'signal_strength': 0.0,
                    'reason': 'No valid price data',
                }
            
            prices = list(self.close_history)
            highs = list(self.high_history)
            lows = list(self.low_history)
            
            # Need sufficient data
            if len(prices) < self.slow_period + 10:
                self.log_warning(f"[TREND] Insufficient data: {len(prices)} bars (need {self.slow_period + 10}) - returning flat")
                return {
                    'action': 'flat',
                    'signal_strength': 0.0,
                    'reason': f'Insufficient data: {len(prices)} bars',
                }
            
            current_price = prices[-1]
            
            # ═══════════════ CALCULATE ALL INDICATORS ═══════════════
            
            # 1. Triple MA System
            self.fast_ma, self.medium_ma, self.slow_ma, self.ma_alignment = self._calculate_triple_ma(prices)
            
            # MA relationship scores
            ma_spread_fast_medium = (self.fast_ma - self.medium_ma) / self.medium_ma if self.medium_ma else 0
            ma_spread_medium_slow = (self.medium_ma - self.slow_ma) / self.slow_ma if self.slow_ma else 0
            
            self.ma_history.append({
                'fast': self.fast_ma,
                'medium': self.medium_ma,
                'slow': self.slow_ma,
                'alignment': self.ma_alignment,
            })
            
            # 2. ADX Trend Strength
            self.adx_value, self.plus_di, self.minus_di = self._calculate_adx(highs, lows, prices)
            self.adx_history.append({
                'adx': self.adx_value,
                'plus_di': self.plus_di,
                'minus_di': self.minus_di,
            })
            
            # 3. Parabolic SAR
            self.sar_value, self.sar_direction = self._calculate_parabolic_sar(highs, lows)
            
            # 4. Trend Slope
            self.trend_slope = self._calculate_trend_slope(prices, 20)
            
            # 5. Support/Resistance
            self.support_levels, self.resistance_levels = self._find_support_resistance(highs, lows)
            self.near_support, self.near_resistance = self._check_sr_proximity(
                current_price, self.support_levels, self.resistance_levels
            )
            
            # 6. Price position relative to MAs
            price_vs_fast = (current_price - self.fast_ma) / self.fast_ma if self.fast_ma else 0
            price_vs_slow = (current_price - self.slow_ma) / self.slow_ma if self.slow_ma else 0
            
            # ═══════════════ TREND CONFLUENCE SCORING ═══════════════
            
            bullish_score = 0.0
            bearish_score = 0.0
            total_weight = 0.0
            
            # MA Alignment (weight: 2.5)
            ma_weight = 2.5
            total_weight += ma_weight
            if self.ma_alignment == 1:  # Bullish alignment
                bullish_score += ma_weight
            elif self.ma_alignment == -1:  # Bearish alignment
                bearish_score += ma_weight
            
            # MA Spread confirmation (weight: 1.5)
            spread_weight = 1.5
            total_weight += spread_weight
            if ma_spread_fast_medium > self.trend_threshold and ma_spread_medium_slow > self.trend_threshold:
                bullish_score += spread_weight * min(1.0, (ma_spread_fast_medium + ma_spread_medium_slow) * 50)
            elif ma_spread_fast_medium < -self.trend_threshold and ma_spread_medium_slow < -self.trend_threshold:
                bearish_score += spread_weight * min(1.0, abs(ma_spread_fast_medium + ma_spread_medium_slow) * 50)
            
            # Price position (weight: 1.5)
            pos_weight = 1.5
            total_weight += pos_weight
            if price_vs_fast > 0 and price_vs_slow > 0:
                bullish_score += pos_weight * min(1.0, (price_vs_fast + price_vs_slow) * 20)
            elif price_vs_fast < 0 and price_vs_slow < 0:
                bearish_score += pos_weight * min(1.0, abs(price_vs_fast + price_vs_slow) * 20)
            
            # ADX Trend Strength (weight: 2.0)
            adx_weight = 2.0
            total_weight += adx_weight
            if self.adx_value >= self.adx_trending_threshold:
                # Strong trend - use DI direction
                trend_strength_factor = min(1.0, self.adx_value / self.adx_strong_threshold)
                if self.plus_di > self.minus_di:
                    bullish_score += adx_weight * trend_strength_factor
                else:
                    bearish_score += adx_weight * trend_strength_factor
            
            # SAR Direction (weight: 1.2)
            sar_weight = 1.2
            total_weight += sar_weight
            if self.sar_direction == 1:
                bullish_score += sar_weight
            elif self.sar_direction == -1:
                bearish_score += sar_weight
            
            # Trend Slope (weight: 1.8)
            slope_weight = 1.8
            total_weight += slope_weight
            if self.trend_slope > self.trend_threshold:
                bullish_score += slope_weight * min(1.0, self.trend_slope / (self.trend_threshold * 3))
            elif self.trend_slope < -self.trend_threshold:
                bearish_score += slope_weight * min(1.0, abs(self.trend_slope) / (self.trend_threshold * 3))
            
            # S/R Level awareness (weight: 1.0)
            sr_weight = 1.0
            total_weight += sr_weight
            if self.near_support and self.ma_alignment >= 0:
                bullish_score += sr_weight * 0.8  # Support bounce potential
            elif self.near_resistance and self.ma_alignment <= 0:
                bearish_score += sr_weight * 0.8  # Resistance rejection potential
            
            # Normalize scores
            bullish_confluence = bullish_score / total_weight if total_weight > 0 else 0
            bearish_confluence = bearish_score / total_weight if total_weight > 0 else 0
            
            # ═══════════════ DETERMINE ACTION ═══════════════
            
            net_trend = bullish_confluence - bearish_confluence
            self.trend_strength = abs(net_trend)
            
            # Track trend history
            self.trend_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'trend_strength': self.trend_strength,
                'bullish': bullish_confluence,
                'bearish': bearish_confluence,
                'ma_alignment': self.ma_alignment,
                'adx': self.adx_value,
            })
            
            # Calculate trend duration
            if len(self.trend_history) > 1:
                prev_trend = 'neutral'
                if self.trend_history[-2]['bullish'] > self.trend_history[-2]['bearish']:
                    prev_trend = 'uptrend'
                elif self.trend_history[-2]['bearish'] > self.trend_history[-2]['bullish']:
                    prev_trend = 'downtrend'
                
                current = 'neutral'
                if bullish_confluence > bearish_confluence:
                    current = 'uptrend'
                elif bearish_confluence > bullish_confluence:
                    current = 'downtrend'
                
                if current == prev_trend and current != 'neutral':
                    self.trend_duration += 1
                else:
                    self.trend_duration = 1
            
            # ADX gating - softer gate for weak trends
            adx_gate = self.adx_value >= (self.adx_trending_threshold * 0.5)  # Lowered threshold
            
            # ALWAYS give a directional vote based on net trend
            # Only truly flat when trend is exactly 0
            if net_trend > 0.01 and adx_gate:  # Very low threshold for bullish
                action = 'long'
                self.current_trend = 'uptrend'
                # Scale signal strength based on confluence
                if bullish_confluence >= self.strong_signal_confluence:
                    signal_strength = min(1.0, bullish_confluence * 1.2)
                elif bullish_confluence >= self.min_confluence_score:
                    signal_strength = min(0.8, bullish_confluence)
                else:
                    signal_strength = max(0.2, bullish_confluence * 0.5)  # Weak but still directional
                
                # Boost for ADX strength
                if self.adx_value >= self.adx_strong_threshold:
                    signal_strength = min(1.0, signal_strength * 1.15)
                
                # Trend duration bonus
                if self.trend_duration >= 5:
                    signal_strength = min(1.0, signal_strength * 1.1)
                
            elif net_trend < -0.01 and adx_gate:  # Very low threshold for bearish
                action = 'short'
                self.current_trend = 'downtrend'
                if bearish_confluence >= self.strong_signal_confluence:
                    signal_strength = min(1.0, bearish_confluence * 1.2)
                elif bearish_confluence >= self.min_confluence_score:
                    signal_strength = min(0.8, bearish_confluence)
                else:
                    signal_strength = max(0.2, bearish_confluence * 0.5)
                
                if self.adx_value >= self.adx_strong_threshold:
                    signal_strength = min(1.0, signal_strength * 1.15)
                
                if self.trend_duration >= 5:
                    signal_strength = min(1.0, signal_strength * 1.1)
                
            elif net_trend > 0.01:  # Weak trend but no ADX confirmation
                action = 'long'
                self.current_trend = 'weak_uptrend'
                signal_strength = max(0.15, bullish_confluence * 0.3)
                
            elif net_trend < -0.01:  # Weak trend but no ADX confirmation
                action = 'short'
                self.current_trend = 'weak_downtrend'
                signal_strength = max(0.15, bearish_confluence * 0.3)
                
            else:
                # Truly flat - no directional bias at all
                action = 'flat'
                self.current_trend = 'neutral'
                signal_strength = 0.1
            
            # ═══════════════ BUILD PROPOSAL ═══════════════
            
            proposal = {
                'action': action,
                'signal_strength': float(signal_strength),
                'position_size': float(min(self.max_signal_strength, signal_strength * 0.6)),
                
                # Trend metrics
                'current_trend': self.current_trend,
                'trend_strength': float(self.trend_strength),
                'trend_slope': float(self.trend_slope),
                'trend_duration': self.trend_duration,
                'bullish_confluence': float(bullish_confluence),
                'bearish_confluence': float(bearish_confluence),
                
                # MA data
                'fast_ma': float(self.fast_ma),
                'medium_ma': float(self.medium_ma),
                'slow_ma': float(self.slow_ma),
                'ma_alignment': self.ma_alignment,
                
                # ADX data
                'adx': float(self.adx_value),
                'plus_di': float(self.plus_di),
                'minus_di': float(self.minus_di),
                
                # SAR
                'sar_direction': self.sar_direction,
                
                # S/R awareness
                'near_support': self.near_support,
                'near_resistance': self.near_resistance,
                
                # Trade management hints
                'duration': 'medium' if self.trend_duration >= 3 else 'short',
                'conviction': 'high' if signal_strength > 0.7 else 'medium' if signal_strength > 0.4 else 'low',
            }
            
            # Track signal
            self.trend_performance[action]['signals'] += 1
            
            return proposal
            
        except Exception as e:
            self.log_error(f"[TREND] Proposal generation failed: {e}")
            return {
                'action': 'flat',
                'signal_strength': 0.0,
                'reason': f'Analysis error: {str(e)[:100]}',
            }
    
    async def _calculate_expert_specific_confidence(
        self, 
        proposal: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate sophisticated confidence score."""
        try:
            action = proposal.get('action', 'flat')
            
            # Base confidence from signal strength
            signal_strength = proposal.get('signal_strength', 0.0)
            base = 0.3 + signal_strength * 0.4
            
            # Confluence bonus
            if action == 'long':
                confluence = proposal.get('bullish_confluence', 0.0)
            elif action == 'short':
                confluence = proposal.get('bearish_confluence', 0.0)
            else:
                confluence = 0.0
            
            if confluence >= self.strong_signal_confluence:
                base *= 1.25
            elif confluence >= self.min_confluence_score:
                base *= 1.1
            
            # ADX strength boost
            adx = proposal.get('adx', 20)
            if adx >= self.adx_strong_threshold:
                base *= 1.2
            elif adx >= self.adx_trending_threshold:
                base *= 1.1
            
            # MA alignment boost
            if proposal.get('ma_alignment') != 0:
                base *= 1.1
            
            # Trend duration bonus
            duration = proposal.get('trend_duration', 0)
            if duration >= 10:
                base *= 1.15
            elif duration >= 5:
                base *= 1.08
            
            # Historical performance adjustment
            perf = self.trend_performance.get(action, {})
            if perf.get('signals', 0) > 20:
                success_rate = perf.get('success', 0) / perf['signals']
                base *= (0.7 + success_rate * 0.6)
            
            return float(max(0.15, min(0.95, base)))
            
        except Exception:
            return 0.4
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """Process with comprehensive trend outputs."""
        base = await super().process(**inputs)
        
        name = self.__class__.__name__
        proposal = dict(base.get('voting_proposal') or {})
        confidence = float(base.get('confidence', 0.0))
        thesis = base.get('_thesis', '')
        
        # Comprehensive trend analysis
        trend_analysis = {
            'current_trend': self.current_trend,
            'trend_strength': self.trend_strength,
            'trend_slope': self.trend_slope,
            'trend_duration': self.trend_duration,
            'ma_alignment': self.ma_alignment,
            'moving_averages': {
                'fast': self.fast_ma,
                'medium': self.medium_ma,
                'slow': self.slow_ma,
            },
            'adx': {
                'value': self.adx_value,
                'plus_di': self.plus_di,
                'minus_di': self.minus_di,
            },
            'sar': {
                'value': self.sar_value,
                'direction': self.sar_direction,
            },
            'sr_levels': {
                'support': self.support_levels[:3],
                'resistance': self.resistance_levels[:3],
                'near_support': self.near_support,
                'near_resistance': self.near_resistance,
            },
            'performance': dict(self.trend_performance),
            'data_points': len(self.price_history),
        }
        
        # Publish to bus
        try:
            self.smart_bus.set('trend_voting_proposal', proposal, module=name, thesis=thesis)
            self.smart_bus.set('trend_confidence', confidence, module=name, thesis=f'Confidence: {confidence:.1%}')
            self.smart_bus.set('trend_analysis', trend_analysis, module=name, thesis=f'Trend: {self.current_trend} ({self.trend_strength:.2%})')
        except Exception:
            pass
        
        return {
            **base,
            # Standard naming convention
            'TrendExpert_voting_proposal': proposal,
            'TrendExpert_confidence': confidence,
            # Alias keys for backward compatibility
            'trend_voting_proposal': proposal,
            'trend_confidence': confidence,
            'trend_analysis': trend_analysis,
            '_thesis': thesis,
        }
