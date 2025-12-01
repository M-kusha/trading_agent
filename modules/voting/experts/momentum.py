"""
Advanced Momentum Expert
========================
A sophisticated momentum-based voting expert that combines multiple momentum
indicators, divergence analysis, volume confirmation, and multi-timeframe
momentum alignment to generate high-conviction directional trading signals.

This expert is designed to capture momentum-driven moves with high precision
by analyzing:
- Multi-period Rate of Change (ROC)
- RSI with divergence detection
- MACD histogram momentum
- Stochastic oscillator
- Volume-weighted momentum
- Multi-timeframe alignment
- Momentum divergence patterns

Per-Instrument Voting:
- Analyzes each instrument separately to produce per-instrument votes
- Each instrument gets its own action/confidence based on its own data
"""

from __future__ import annotations

import datetime
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.experts.base import VotingExpertBase
from modules.voting.core.per_instrument import PerInstrumentVote, InstrumentProposal


def normalize_instrument(symbol: str) -> str:
    """Normalize instrument symbol to standard format."""
    if not symbol:
        return ""
    s = symbol.upper().replace("/", "").replace("_", "").replace("-", "").strip()
    mapping = {"EURUSD": "EURUSD", "XAUUSD": "XAUUSD", "GOLDUSD": "XAUUSD"}
    return mapping.get(s, s)


@module(**module_args("MomentumExpert"))
class MomentumExpert(VotingExpertBase):
    """
    Advanced Momentum Expert - Multi-indicator momentum analysis.
    
    Combines 7+ momentum indicators with divergence detection and 
    volume confirmation for high-conviction directional signals.
    
    Analysis Components:
    1. Multi-period ROC (5, 10, 20, 50 bars)
    2. RSI with overbought/oversold and divergence
    3. MACD histogram momentum and crossovers
    4. Stochastic oscillator with confirmation
    5. Volume-weighted momentum (OBV derivative)
    6. Multi-timeframe momentum alignment
    7. Momentum divergence patterns (bullish/bearish)
    8. Momentum acceleration/deceleration
    
    Voting Actions (canonical):
    - long: Strong bullish momentum confluence
    - short: Strong bearish momentum confluence
    - flat: No clear momentum direction / no trade
    
    Features:
    - Adaptive thresholds based on volatility
    - Divergence detection (price vs momentum)
    - Volume confirmation weighting
    - Multi-timeframe alignment scoring
    - Historical accuracy tracking per signal type
    """
    
    def _expert_specific_init(self) -> None:
        """Initialize advanced momentum analysis state."""
        # ═══════════════════════════ CONFIGURATION ═══════════════════════════
        # Instruments to analyze (from config or default)
        self.instruments = self.config.get('instruments', ['EURUSD', 'XAUUSD'])
        
        # ROC periods for multi-scale analysis
        self.roc_periods = [5, 10, 20, 50]
        self.roc_weights = [0.35, 0.30, 0.20, 0.15]  # Short-term weighted higher
        
        # RSI configuration
        self.rsi_period = int(self.config.get('rsi_period', 14))
        self.rsi_overbought = float(self.config.get('rsi_overbought', 70))
        self.rsi_oversold = float(self.config.get('rsi_oversold', 30))
        self.rsi_extreme_overbought = 80
        self.rsi_extreme_oversold = 20
        
        # MACD configuration
        self.macd_fast = int(self.config.get('macd_fast', 12))
        self.macd_slow = int(self.config.get('macd_slow', 26))
        self.macd_signal = int(self.config.get('macd_signal', 9))
        
        # Stochastic configuration
        self.stoch_k_period = int(self.config.get('stoch_k', 14))
        self.stoch_d_period = int(self.config.get('stoch_d', 3))
        self.stoch_overbought = 80
        self.stoch_oversold = 20
        
        # Momentum thresholds
        self.momentum_threshold = float(self.config.get('momentum_threshold', 0.015))
        self.strong_momentum_multiplier = 2.5
        self.weak_momentum_multiplier = 0.5
        
        # Divergence detection
        self.divergence_lookback = int(self.config.get('divergence_lookback', 20))
        self.divergence_significance = float(self.config.get('divergence_significance', 0.02))
        
        # Confluence requirements - lowered for more signals during training
        self.min_confluence_score = float(self.config.get('min_confluence', 0.15))
        self.strong_signal_confluence = float(self.config.get('strong_confluence', 0.5))
        
        # ═══════════════════════════ PER-INSTRUMENT STATE ═══════════════════════════
        # Each instrument gets its own history and indicators
        self.instrument_state: Dict[str, Dict[str, Any]] = {}
        for inst in self.instruments:
            inst_norm = normalize_instrument(inst)
            self.instrument_state[inst_norm] = {
                'price_history': deque(maxlen=200),
                'high_history': deque(maxlen=200),
                'low_history': deque(maxlen=200),
                'volume_history': deque(maxlen=200),
                'close_history': deque(maxlen=200),
                'rsi_value': 50.0,
                'rsi_history': deque(maxlen=50),
                'macd_line': 0.0,
                'macd_signal_line': 0.0,
                'macd_histogram': 0.0,
                'macd_histogram_history': deque(maxlen=30),
                'stoch_k': 50.0,
                'stoch_d': 50.0,
                'obv': 0.0,
                'obv_history': deque(maxlen=50),
                'roc_values': {p: 0.0 for p in self.roc_periods},
                'composite_momentum': 0.0,
                'momentum_direction': 0,
                'momentum_acceleration': 0.0,
                'divergence_signal': None,
                'volume_confirmation': 0.5,
                'momentum_history': deque(maxlen=100),
                'bullish_divergence_count': 0,
                'bearish_divergence_count': 0,
            }
        
        # Legacy single-instrument state (for backward compat)
        self.price_history: deque = deque(maxlen=200)
        self.high_history: deque = deque(maxlen=200)
        self.low_history: deque = deque(maxlen=200)
        self.volume_history: deque = deque(maxlen=200)
        self.close_history: deque = deque(maxlen=200)
        
        # Indicator values (legacy)
        self.rsi_value: float = 50.0
        self.rsi_history: deque = deque(maxlen=50)
        self.macd_line: float = 0.0
        self.macd_signal_line: float = 0.0
        self.macd_histogram: float = 0.0
        self.macd_histogram_history: deque = deque(maxlen=30)
        self.stoch_k: float = 50.0
        self.stoch_d: float = 50.0
        self.obv: float = 0.0
        self.obv_history: deque = deque(maxlen=50)
        
        # Multi-period ROC values
        self.roc_values: Dict[int, float] = {p: 0.0 for p in self.roc_periods}
        
        # Momentum state
        self.composite_momentum: float = 0.0
        self.momentum_direction: int = 0  # -1, 0, 1
        self.momentum_acceleration: float = 0.0
        self.divergence_signal: Optional[str] = None  # 'bullish', 'bearish', None
        self.volume_confirmation: float = 0.5
        
        # Analysis history
        self.momentum_history: deque = deque(maxlen=100)
        self.signal_history: deque = deque(maxlen=50)
        
        # Performance tracking
        self.momentum_performance: Dict[str, Dict[str, Any]] = {
            'long': {'signals': 0, 'success': 0, 'total_pnl': 0.0},
            'short': {'signals': 0, 'success': 0, 'total_pnl': 0.0},
            'flat': {'signals': 0, 'success': 0, 'total_pnl': 0.0},
        }
        
        # Divergence tracking
        self.bullish_divergence_count: int = 0
        self.bearish_divergence_count: int = 0
        
        self.log_info(
            f"[MOMENTUM] Advanced MomentumExpert initialized | "
            f"instruments={self.instruments} | ROC periods={self.roc_periods} | "
            f"RSI={self.rsi_period} | MACD={self.macd_fast}/{self.macd_slow}/{self.macd_signal}"
        )
        
        self._publish_baseline_keys()
        self._publish_momentum_baseline()
    
    def _publish_momentum_baseline(self) -> None:
        """Publish momentum baseline keys."""
        try:
            self.smart_bus.set(
                'momentum_voting_proposal',
                {
                    'action': 'flat',
                    'signal_strength': 0.0,
                    'reason': 'Momentum baseline',
                },
                module='MomentumExpert',
                thesis='Momentum baseline'
            )
            self.smart_bus.set(
                'momentum_confidence',
                0.1,
                module='MomentumExpert',
                thesis='Momentum baseline confidence'
            )
            self.smart_bus.set(
                'momentum_analysis',
                {'composite_momentum': 0.0, 'indicators': {}},
                module='MomentumExpert',
                thesis='Momentum analysis baseline'
            )
        except Exception:
            pass
    
    # ═══════════════════════════ INDICATOR CALCULATIONS ═══════════════════════════
    
    def _update_price_data(self, market_data: Dict[str, Any]) -> bool:
        """Extract and update price/volume data from market data."""
        try:
            # Get OHLCV data
            ohlcv = market_data.get('ohlcv') or {}
            prices = market_data.get('prices') or market_data.get('close_prices') or []
            
            # Handle various data formats
            if isinstance(prices, dict):
                close_prices = prices.get('close', [])
                high_prices = prices.get('high', [])
                low_prices = prices.get('low', [])
                volumes = prices.get('volume', [])
            elif ohlcv:
                close_prices = ohlcv.get('close', [])
                high_prices = ohlcv.get('high', [])
                low_prices = ohlcv.get('low', [])
                volumes = ohlcv.get('volume', [])
            else:
                close_prices = list(prices) if isinstance(prices, (list, np.ndarray)) else []
                high_prices = []
                low_prices = []
                volumes = []
            
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
            
            current_volume = market_data.get('volume', 1.0)
            if current_volume is None or current_volume == 0:
                current_volume = 1.0
            
            # Update histories
            if current_price is not None:
                self.price_history.append(float(current_price))
                self.close_history.append(float(current_price))
                self.high_history.append(float(current_high) if current_high else float(current_price))
                self.low_history.append(float(current_low) if current_low else float(current_price))
                self.volume_history.append(float(current_volume))
                return True
            
            return False
            
        except Exception as e:
            self.log_warning(f"[MOMENTUM] Price data update failed: {e}")
            return False
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate RSI with Wilder's smoothing."""
        if len(prices) < period + 1:
            return 50.0
        
        try:
            deltas = np.diff(prices[-(period + 1):])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Wilder's smoothing for remaining values
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(np.clip(rsi, 0, 100))
            
        except Exception:
            return 50.0
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram."""
        if len(prices) < self.macd_slow + self.macd_signal:
            return 0.0, 0.0, 0.0
        
        try:
            # EMA calculation helper
            def ema(data: List[float], period: int) -> float:
                if len(data) < period:
                    return sum(data) / len(data) if data else 0.0
                multiplier = 2 / (period + 1)
                ema_val = sum(data[:period]) / period
                for price in data[period:]:
                    ema_val = (price - ema_val) * multiplier + ema_val
                return ema_val
            
            fast_ema = ema(prices, self.macd_fast)
            slow_ema = ema(prices, self.macd_slow)
            macd_line = fast_ema - slow_ema
            
            # For signal line, we need MACD history
            self.macd_histogram_history.append(macd_line)
            macd_values = list(self.macd_histogram_history)
            
            if len(macd_values) >= self.macd_signal:
                signal_line = ema(macd_values, self.macd_signal)
            else:
                signal_line = sum(macd_values) / len(macd_values) if macd_values else 0.0
            
            histogram = macd_line - signal_line
            
            return float(macd_line), float(signal_line), float(histogram)
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _calculate_stochastic(self, closes: List[float], highs: List[float], lows: List[float]) -> Tuple[float, float]:
        """Calculate Stochastic %K and %D."""
        if len(closes) < self.stoch_k_period:
            return 50.0, 50.0
        
        try:
            recent_closes = closes[-self.stoch_k_period:]
            recent_highs = highs[-self.stoch_k_period:] if len(highs) >= self.stoch_k_period else recent_closes
            recent_lows = lows[-self.stoch_k_period:] if len(lows) >= self.stoch_k_period else recent_closes
            
            highest_high = max(recent_highs)
            lowest_low = min(recent_lows)
            current_close = closes[-1]
            
            if highest_high == lowest_low:
                stoch_k = 50.0
            else:
                stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            
            # %D is SMA of %K
            stoch_d = stoch_k  # Simplified
            
            return float(np.clip(stoch_k, 0, 100)), float(np.clip(stoch_d, 0, 100))
            
        except Exception:
            return 50.0, 50.0
    
    def _calculate_obv(self, closes: List[float], volumes: List[float]) -> float:
        """Calculate On-Balance Volume."""
        if len(closes) < 2 or len(volumes) < 2:
            return 0.0
        
        try:
            obv = 0.0
            min_len = min(len(closes), len(volumes))
            
            for i in range(1, min_len):
                if closes[i] > closes[i-1]:
                    obv += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv -= volumes[i]
            
            return float(obv)
            
        except Exception:
            return 0.0
    
    def _calculate_multi_period_roc(self, prices: List[float]) -> Dict[int, float]:
        """Calculate ROC for multiple periods."""
        roc_values = {}
        
        for period in self.roc_periods:
            if len(prices) >= period + 1:
                current = prices[-1]
                past = prices[-(period + 1)]
                if past != 0:
                    roc_values[period] = (current - past) / past
                else:
                    roc_values[period] = 0.0
            else:
                roc_values[period] = 0.0
        
        return roc_values
    
    def _detect_divergence(self, prices: List[float], indicator_values: List[float]) -> Optional[str]:
        """Detect bullish or bearish divergence between price and indicator."""
        if len(prices) < self.divergence_lookback or len(indicator_values) < self.divergence_lookback:
            return None
        
        try:
            recent_prices = prices[-self.divergence_lookback:]
            recent_indicator = indicator_values[-self.divergence_lookback:]
            
            # Find price swing lows and highs
            price_min_idx = np.argmin(recent_prices)
            price_max_idx = np.argmax(recent_prices)
            
            # Bullish divergence: lower price low, higher indicator low
            if price_min_idx > len(recent_prices) // 2:  # Recent low
                prev_low_idx = np.argmin(recent_prices[:len(recent_prices)//2])
                if (recent_prices[price_min_idx] < recent_prices[prev_low_idx] and
                    recent_indicator[price_min_idx] > recent_indicator[prev_low_idx]):
                    return 'bullish'
            
            # Bearish divergence: higher price high, lower indicator high
            if price_max_idx > len(recent_prices) // 2:  # Recent high
                prev_high_idx = np.argmax(recent_prices[:len(recent_prices)//2])
                if (recent_prices[price_max_idx] > recent_prices[prev_high_idx] and
                    recent_indicator[price_max_idx] < recent_indicator[prev_high_idx]):
                    return 'bearish'
            
            return None
            
        except Exception:
            return None
    
    def _calculate_volume_confirmation(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate volume confirmation score for current move."""
        if len(prices) < 10 or len(volumes) < 10:
            return 0.5
        
        try:
            # Compare recent volume to average
            recent_vol = np.mean(volumes[-5:])
            avg_vol = np.mean(volumes[-20:])
            
            if avg_vol == 0:
                return 0.5
            
            vol_ratio = recent_vol / avg_vol
            
            # Check if volume confirms price direction
            price_change = prices[-1] - prices[-5]
            vol_change = volumes[-1] - np.mean(volumes[-5:])
            
            # Volume should increase in direction of price move
            if (price_change > 0 and vol_change > 0) or (price_change < 0 and vol_change > 0):
                confirmation = min(1.0, 0.5 + vol_ratio * 0.25)
            else:
                confirmation = max(0.2, 0.5 - vol_ratio * 0.15)
            
            return float(confirmation)
            
        except Exception:
            return 0.5
    
    # ═══════════════════════════ MAIN ANALYSIS ═══════════════════════════
    
    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sophisticated momentum-based voting proposal."""
        try:
            # Update price data
            if not self._update_price_data(market_data):
                self.log_warning("[MOMENTUM] No valid price data - returning flat")
                return {
                    'action': 'flat',
                    'signal_strength': 0.0,
                    'reason': 'No valid price data',
                }
            
            prices = list(self.close_history)
            highs = list(self.high_history)
            lows = list(self.low_history)
            volumes = list(self.volume_history)
            
            # Need sufficient data
            if len(prices) < max(self.roc_periods) + 5:
                self.log_warning(f"[MOMENTUM] Insufficient data: {len(prices)} bars (need {max(self.roc_periods) + 5}) - returning flat")
                return {
                    'action': 'flat',
                    'signal_strength': 0.0,
                    'reason': f'Insufficient data: {len(prices)} bars',
                }
            
            # ═══════════════ CALCULATE ALL INDICATORS ═══════════════
            
            # 1. Multi-period ROC
            self.roc_values = self._calculate_multi_period_roc(prices)
            weighted_roc = sum(
                self.roc_values.get(p, 0) * w 
                for p, w in zip(self.roc_periods, self.roc_weights)
            )
            
            # 2. RSI
            self.rsi_value = self._calculate_rsi(prices, self.rsi_period)
            self.rsi_history.append(self.rsi_value)
            
            # 3. MACD
            self.macd_line, self.macd_signal_line, self.macd_histogram = self._calculate_macd(prices)
            
            # 4. Stochastic
            self.stoch_k, self.stoch_d = self._calculate_stochastic(prices, highs, lows)
            
            # 5. OBV momentum
            self.obv = self._calculate_obv(prices, volumes)
            self.obv_history.append(self.obv)
            obv_momentum = 0.0
            if len(self.obv_history) > 5:
                obv_recent = list(self.obv_history)[-5:]
                obv_momentum = (obv_recent[-1] - obv_recent[0]) / (abs(obv_recent[0]) + 1e-10)
            
            # 6. Volume confirmation
            self.volume_confirmation = self._calculate_volume_confirmation(prices, volumes)
            
            # 7. Divergence detection (RSI divergence)
            rsi_list = list(self.rsi_history)
            self.divergence_signal = self._detect_divergence(prices[-len(rsi_list):], rsi_list)
            if self.divergence_signal == 'bullish':
                self.bullish_divergence_count += 1
            elif self.divergence_signal == 'bearish':
                self.bearish_divergence_count += 1
            
            # ═══════════════ MOMENTUM CONFLUENCE SCORING ═══════════════
            
            bullish_score = 0.0
            bearish_score = 0.0
            total_weight = 0.0
            
            # ROC score (weight: 2.0)
            roc_weight = 2.0
            total_weight += roc_weight
            if weighted_roc > self.momentum_threshold:
                bullish_score += roc_weight * min(1.0, weighted_roc / (self.momentum_threshold * 3))
            elif weighted_roc < -self.momentum_threshold:
                bearish_score += roc_weight * min(1.0, abs(weighted_roc) / (self.momentum_threshold * 3))
            
            # RSI score (weight: 1.5)
            rsi_weight = 1.5
            total_weight += rsi_weight
            if self.rsi_value > 50:
                if self.rsi_value < self.rsi_overbought:
                    bullish_score += rsi_weight * ((self.rsi_value - 50) / 30)
                elif self.rsi_value >= self.rsi_extreme_overbought:
                    # Extreme overbought - potential reversal
                    bearish_score += rsi_weight * 0.3
            else:
                if self.rsi_value > self.rsi_oversold:
                    bearish_score += rsi_weight * ((50 - self.rsi_value) / 30)
                elif self.rsi_value <= self.rsi_extreme_oversold:
                    # Extreme oversold - potential reversal
                    bullish_score += rsi_weight * 0.3
            
            # MACD score (weight: 1.8)
            macd_weight = 1.8
            total_weight += macd_weight
            if self.macd_histogram > 0:
                macd_strength = min(1.0, abs(self.macd_histogram) * 100)
                if self.macd_line > self.macd_signal_line:
                    bullish_score += macd_weight * macd_strength
            else:
                macd_strength = min(1.0, abs(self.macd_histogram) * 100)
                if self.macd_line < self.macd_signal_line:
                    bearish_score += macd_weight * macd_strength
            
            # Stochastic score (weight: 1.2)
            stoch_weight = 1.2
            total_weight += stoch_weight
            if self.stoch_k > 50 and self.stoch_k < self.stoch_overbought:
                bullish_score += stoch_weight * ((self.stoch_k - 50) / 50)
            elif self.stoch_k < 50 and self.stoch_k > self.stoch_oversold:
                bearish_score += stoch_weight * ((50 - self.stoch_k) / 50)
            # Oversold/overbought reversals
            elif self.stoch_k >= self.stoch_overbought:
                bearish_score += stoch_weight * 0.4
            elif self.stoch_k <= self.stoch_oversold:
                bullish_score += stoch_weight * 0.4
            
            # OBV momentum score (weight: 1.0)
            obv_weight = 1.0
            total_weight += obv_weight
            if obv_momentum > 0.01:
                bullish_score += obv_weight * min(1.0, obv_momentum * 10)
            elif obv_momentum < -0.01:
                bearish_score += obv_weight * min(1.0, abs(obv_momentum) * 10)
            
            # Divergence bonus (weight: 1.5)
            div_weight = 1.5
            total_weight += div_weight
            if self.divergence_signal == 'bullish':
                bullish_score += div_weight
            elif self.divergence_signal == 'bearish':
                bearish_score += div_weight
            
            # Volume confirmation adjustment
            bullish_score *= (0.7 + self.volume_confirmation * 0.6)
            bearish_score *= (0.7 + self.volume_confirmation * 0.6)
            
            # Normalize scores
            bullish_confluence = bullish_score / total_weight if total_weight > 0 else 0
            bearish_confluence = bearish_score / total_weight if total_weight > 0 else 0
            
            # ═══════════════ DETERMINE ACTION ═══════════════
            
            net_momentum = bullish_confluence - bearish_confluence
            self.composite_momentum = net_momentum
            
            # Calculate momentum acceleration
            self.momentum_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'momentum': net_momentum,
                'bullish': bullish_confluence,
                'bearish': bearish_confluence,
            })
            
            if len(self.momentum_history) > 5:
                recent_mom = [h['momentum'] for h in list(self.momentum_history)[-5:]]
                self.momentum_acceleration = (recent_mom[-1] - recent_mom[0]) / 5
            else:
                self.momentum_acceleration = 0.0
            
            # ALWAYS give a directional vote based on net momentum
            # Only truly flat when momentum is exactly 0
            if net_momentum > 0.01:  # Very low threshold for bullish
                action = 'long'
                self.momentum_direction = 1
                # Scale signal strength based on confluence
                if bullish_confluence >= self.strong_signal_confluence:
                    signal_strength = min(1.0, bullish_confluence * 1.2)
                elif bullish_confluence >= self.min_confluence_score:
                    signal_strength = min(0.8, bullish_confluence)
                else:
                    signal_strength = max(0.2, bullish_confluence * 0.5)  # Weak but still directional
                
                # Boost for momentum acceleration
                if self.momentum_acceleration > 0.02:
                    signal_strength = min(1.0, signal_strength * 1.1)
                
            elif net_momentum < -0.01:  # Very low threshold for bearish
                action = 'short'
                self.momentum_direction = -1
                if bearish_confluence >= self.strong_signal_confluence:
                    signal_strength = min(1.0, bearish_confluence * 1.2)
                elif bearish_confluence >= self.min_confluence_score:
                    signal_strength = min(0.8, bearish_confluence)
                else:
                    signal_strength = max(0.2, bearish_confluence * 0.5)
                
                if self.momentum_acceleration < -0.02:
                    signal_strength = min(1.0, signal_strength * 1.1)
                
            else:
                # Truly flat - no directional bias at all
                action = 'flat'
                self.momentum_direction = 0
                signal_strength = 0.1
            
            # ═══════════════ BUILD PROPOSAL ═══════════════
            
            proposal = {
                'action': action,
                'signal_strength': float(signal_strength),
                'position_size': float(min(self.max_signal_strength, signal_strength * 0.6)),
                
                # Momentum metrics
                'composite_momentum': float(net_momentum),
                'bullish_confluence': float(bullish_confluence),
                'bearish_confluence': float(bearish_confluence),
                'momentum_acceleration': float(self.momentum_acceleration),
                
                # Individual indicators
                'roc_weighted': float(weighted_roc),
                'rsi': float(self.rsi_value),
                'macd_histogram': float(self.macd_histogram),
                'stochastic_k': float(self.stoch_k),
                'obv_momentum': float(obv_momentum),
                
                # Confirmations
                'divergence': self.divergence_signal,
                'volume_confirmation': float(self.volume_confirmation),
                
                # Trade management hints
                'duration': 'short' if abs(net_momentum) < 0.5 else 'medium',
                'conviction': 'high' if signal_strength > 0.7 else 'medium' if signal_strength > 0.4 else 'low',
            }
            
            # Track signal
            self.momentum_performance[action]['signals'] += 1
            
            return proposal
            
        except Exception as e:
            self.log_error(f"[MOMENTUM] Proposal generation failed: {e}")
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
            
            # Divergence boost
            if proposal.get('divergence'):
                base *= 1.15
            
            # Volume confirmation
            vol_conf = proposal.get('volume_confirmation', 0.5)
            base *= (0.85 + vol_conf * 0.3)
            
            # Momentum acceleration alignment
            accel = proposal.get('momentum_acceleration', 0.0)
            if (action == 'long' and accel > 0) or (action == 'short' and accel < 0):
                base *= 1.08
            
            # Historical performance adjustment
            perf = self.momentum_performance.get(action, {})
            if perf.get('signals', 0) > 20:
                success_rate = perf.get('success', 0) / perf['signals']
                base *= (0.7 + success_rate * 0.6)
            
            return float(max(0.15, min(0.95, base)))
            
        except Exception:
            return 0.4
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Process market data to determine momentum signals PER INSTRUMENT.
        
        NEW: Analyzes each instrument separately to produce per-instrument votes.
        Each instrument gets its own action/confidence based on its own momentum analysis.
        """
        # Get market data from InfoBus
        market_data = self.smart_bus.get("market_data", self.__class__.__name__, default={})
        features = self.smart_bus.get("features", self.__class__.__name__, default={})
        
        if not market_data and not features:
            self.log_warning("[MOMENTUM] No market data or features available")
            return self._neutral_output("No market data available")
        
        # ========== Per-Instrument Analysis ==========
        per_instrument_vote = PerInstrumentVote(member=self.__class__.__name__)
        per_instrument_analysis: Dict[str, Dict] = {}
        
        for inst in self.instruments:
            inst_norm = normalize_instrument(inst)
            
            # Extract instrument-specific data
            inst_market = self._extract_instrument_data(market_data, inst)
            inst_features = self._extract_instrument_data(features, inst)
            
            # Get price arrays for this instrument (pass instrument for lookup)
            close_prices = self._extract_prices(inst_market, inst_features, 'close', inst)
            high_prices = self._extract_prices(inst_market, inst_features, 'high', inst)
            low_prices = self._extract_prices(inst_market, inst_features, 'low', inst)
            volumes = self._extract_prices(inst_market, inst_features, 'volume', inst)
            
            # Get instrument state
            state = self.instrument_state.get(inst_norm, {})
            if not state:
                state = self.instrument_state.setdefault(inst_norm, {
                    'price_history': deque(maxlen=200),
                    'rsi_history': deque(maxlen=50),
                    'macd_histogram_history': deque(maxlen=30),
                    'obv_history': deque(maxlen=50),
                    'momentum_history': deque(maxlen=100),
                })
            
            # Update price history for this instrument
            if len(close_prices) > 0:
                state['price_history'] = deque(close_prices[-200:], maxlen=200)
                state['high_history'] = deque(high_prices[-200:] if len(high_prices) > 0 else close_prices[-200:], maxlen=200)
                state['low_history'] = deque(low_prices[-200:] if len(low_prices) > 0 else close_prices[-200:], maxlen=200)
                state['volume_history'] = deque(volumes[-200:] if len(volumes) > 0 else [1.0] * min(200, len(close_prices)), maxlen=200)
            
            prices = list(state.get('price_history', []))
            
            if len(prices) < max(self.roc_periods) + 5:
                self.log_debug(f"[MOMENTUM] Insufficient data for {inst}: {len(prices)} bars")
                # Create neutral proposal for this instrument
                per_instrument_vote.set_proposal(InstrumentProposal(
                    instrument=inst_norm,
                    action='flat',
                    confidence=0.1,
                    magnitude=0.0,
                    rationale=f"Insufficient data for {inst}: {len(prices)} bars",
                ))
                per_instrument_analysis[inst_norm] = {
                    'composite_momentum': 0.0,
                    'direction': 0,
                    'action': 'flat',
                    'confidence': 0.1,
                }
                continue
            
            # Calculate momentum indicators for this instrument
            highs = list(state.get('high_history', prices))
            lows = list(state.get('low_history', prices))
            vols = list(state.get('volume_history', [1.0] * len(prices)))
            
            # Calculate all indicators
            roc_values = self._calculate_multi_period_roc(prices)
            weighted_roc = sum(roc_values.get(p, 0) * w for p, w in zip(self.roc_periods, self.roc_weights))
            
            rsi_value = self._calculate_rsi(prices, self.rsi_period)
            macd_line, macd_signal_line, macd_histogram = self._calculate_macd_for_instrument(prices, state)
            stoch_k, stoch_d = self._calculate_stochastic(prices, highs, lows)
            obv = self._calculate_obv(prices, vols)
            
            # OBV momentum
            obv_history = state.setdefault('obv_history', deque(maxlen=50))
            obv_history.append(obv)
            obv_momentum = 0.0
            if len(obv_history) > 5:
                obv_recent = list(obv_history)[-5:]
                obv_momentum = (obv_recent[-1] - obv_recent[0]) / (abs(obv_recent[0]) + 1e-10)
            
            volume_confirmation = self._calculate_volume_confirmation(prices, vols)
            
            # Divergence detection
            rsi_history = state.setdefault('rsi_history', deque(maxlen=50))
            rsi_history.append(rsi_value)
            divergence_signal = self._detect_divergence(prices[-len(rsi_history):], list(rsi_history))
            
            # Calculate confluence scores
            bullish_score, bearish_score, total_weight = self._calculate_confluence_scores(
                weighted_roc, rsi_value, macd_line, macd_signal_line, macd_histogram,
                stoch_k, obv_momentum, divergence_signal, volume_confirmation
            )
            
            bullish_confluence = bullish_score / total_weight if total_weight > 0 else 0
            bearish_confluence = bearish_score / total_weight if total_weight > 0 else 0
            net_momentum = bullish_confluence - bearish_confluence
            
            # Track momentum history for acceleration
            momentum_history = state.setdefault('momentum_history', deque(maxlen=100))
            momentum_history.append({'momentum': net_momentum, 'timestamp': datetime.datetime.now().isoformat()})
            
            momentum_acceleration = 0.0
            if len(momentum_history) > 5:
                recent_mom = [h['momentum'] for h in list(momentum_history)[-5:]]
                momentum_acceleration = (recent_mom[-1] - recent_mom[0]) / 5
            
            # Determine action for this instrument
            action, confidence, signal_strength = self._determine_momentum_action(
                net_momentum, bullish_confluence, bearish_confluence, momentum_acceleration
            )
            
            thesis = f"{inst_norm}: Momentum {action} (net={net_momentum:.2%}, conf={confidence:.1%})"
            
            # Create per-instrument proposal
            per_instrument_vote.set_proposal(InstrumentProposal(
                instrument=inst_norm,
                action=action,
                confidence=confidence,
                magnitude=signal_strength,
                rationale=thesis,
            ))
            
            # Store analysis
            per_instrument_analysis[inst_norm] = {
                'composite_momentum': net_momentum,
                'bullish_confluence': bullish_confluence,
                'bearish_confluence': bearish_confluence,
                'momentum_acceleration': momentum_acceleration,
                'roc_weighted': weighted_roc,
                'rsi': rsi_value,
                'macd_histogram': macd_histogram,
                'stochastic_k': stoch_k,
                'obv_momentum': obv_momentum,
                'divergence': divergence_signal,
                'volume_confirmation': volume_confirmation,
                'action': action,
                'confidence': confidence,
            }
            
            self.log_debug(f"[MOMENTUM] {inst}: action={action}, conf={confidence:.2f}, momentum={net_momentum:.2%}")
        
        # ========== Calculate Global Summary (backward compat) ==========
        if per_instrument_vote.proposals:
            best_proposal = max(per_instrument_vote.proposals.values(), key=lambda p: p.confidence)
            global_action = best_proposal.action
            global_confidence = best_proposal.confidence
            global_thesis = best_proposal.rationale
        else:
            global_action = 'flat'
            global_confidence = 0.1
            global_thesis = "No instrument data available"
        
        # Build proposal dict with per-instrument proposals
        proposals_dict = {}
        for inst, proposal in per_instrument_vote.proposals.items():
            proposals_dict[inst] = {
                'action': proposal.action,
                'confidence': proposal.confidence,
                'magnitude': proposal.magnitude,
                'rationale': proposal.rationale,
            }
        
        proposal = {
            "action": global_action,
            "signal_strength": global_confidence,
            "reason": global_thesis,
            "proposals": proposals_dict,
        }
        
        name = self.__class__.__name__
        
        # Publish to SmartInfoBus
        try:
            self.smart_bus.set('MomentumExpert_voting_proposal', proposal, module=name, thesis=global_thesis)
            self.smart_bus.set('MomentumExpert_confidence', global_confidence, module=name, thesis=f'Confidence: {global_confidence:.1%}')
            self.smart_bus.set('momentum_voting_proposal', proposal, module=name, thesis=global_thesis)
            self.smart_bus.set('momentum_confidence', global_confidence, module=name, thesis=f'Momentum confidence: {global_confidence:.1%}')
            
            # NEW: Publish per-instrument votes for CommitteeCoordinator
            per_inst_votes_dict = {inst: prop.to_dict() for inst, prop in per_instrument_vote.proposals.items()}
            self.smart_bus.set(
                'MomentumExpert_per_instrument_votes',
                per_inst_votes_dict,
                module=name,
                thesis=f'Per-instrument momentum votes: {list(per_inst_votes_dict.keys())}'
            )
        except Exception as e:
            self.log_warning(f"[MOMENTUM] Failed to publish to bus: {e}")
        
        # Comprehensive momentum analysis
        momentum_analysis = {
            'composite_momentum': self.composite_momentum,
            'direction': self.momentum_direction,
            'acceleration': self.momentum_acceleration,
            'per_instrument': per_instrument_analysis,
            'performance': dict(self.momentum_performance),
        }
        
        return {
            'MomentumExpert_voting_proposal': proposal,
            'MomentumExpert_confidence': global_confidence,
            'MomentumExpert_per_instrument_votes': {inst: prop.to_dict() for inst, prop in per_instrument_vote.proposals.items()},
            'per_instrument_votes': per_instrument_vote,
            'momentum_voting_proposal': proposal,
            'momentum_confidence': global_confidence,
            'momentum_analysis': momentum_analysis,
            'voting_proposal': proposal,
            'confidence': global_confidence,
            '_thesis': global_thesis,
        }
    
    def _extract_instrument_data(self, data: Dict, instrument: str) -> Dict:
        """Extract data for a specific instrument from nested market data."""
        if not isinstance(data, dict):
            return {}
        
        inst_norm = normalize_instrument(instrument)
        
        # Try direct instrument key
        for key in [instrument, inst_norm, instrument.upper(), instrument.lower()]:
            if key in data:
                return data[key] if isinstance(data[key], dict) else data
        
        # Try with separators
        for sep in ['_', '/', '-', '']:
            for pair in [f"EUR{sep}USD", f"XAU{sep}USD"]:
                norm_pair = normalize_instrument(pair)
                if norm_pair == inst_norm and pair in data:
                    return data[pair] if isinstance(data[pair], dict) else data
        
        # Return full data if no instrument-specific found (legacy format)
        return data
    
    def _extract_prices(self, market_data: Dict, features: Dict, price_type: str, instrument: str = '') -> np.ndarray:
        """Extract price array from market data or features for a specific instrument."""
        # Normalize instrument name for matching
        inst_norm = normalize_instrument(instrument) if instrument else ''
        inst_variations = [instrument, inst_norm, f"{inst_norm[:3]}_{inst_norm[3:]}" if len(inst_norm) >= 6 else inst_norm]
        
        # Try market_data first
        if isinstance(market_data, dict):
            if price_type in market_data:
                data = market_data[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)
            
            # Try nested structure
            for tf in ['H1', 'H4', 'D1']:
                if tf in market_data and isinstance(market_data[tf], dict):
                    if price_type in market_data[tf]:
                        data = market_data[tf][price_type]
                        if isinstance(data, (list, np.ndarray)):
                            return np.array(data, dtype=float)
        
        # Try features
        if isinstance(features, dict):
            if price_type in features:
                data = features[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)
        
        # Try InfoBus historical_prices - look for specific instrument
        try:
            historical = self.smart_bus.get("historical_prices", self.__class__.__name__, default=None)
        except Exception:
            historical = None
        
        if isinstance(historical, dict):
            # Find the matching symbol key (handles XAU_USD vs XAUUSD)
            matched_symbol = None
            for sym in historical.keys():
                sym_norm = normalize_instrument(sym)
                if sym_norm == inst_norm or sym in inst_variations:
                    matched_symbol = sym
                    break
            
            if matched_symbol and matched_symbol in historical:
                sym_block = historical[matched_symbol]
                if isinstance(sym_block, dict):
                    # Try each timeframe
                    for tf in ["H4", "H1", "D1"]:
                        tf_rec = sym_block.get(tf)
                        if isinstance(tf_rec, dict):
                            seq = tf_rec.get(price_type)
                            if isinstance(seq, (list, np.ndarray)) and len(seq) > 0:
                                return np.array(seq, dtype=float)
        
        return np.array([])
    
    def _calculate_macd_for_instrument(self, prices: List[float], state: Dict) -> Tuple[float, float, float]:
        """Calculate MACD for a specific instrument using its state."""
        if len(prices) < self.macd_slow + self.macd_signal:
            return 0.0, 0.0, 0.0
        
        try:
            def ema(data: List[float], period: int) -> float:
                if len(data) < period:
                    return sum(data) / len(data) if data else 0.0
                multiplier = 2 / (period + 1)
                ema_val = sum(data[:period]) / period
                for price in data[period:]:
                    ema_val = (price - ema_val) * multiplier + ema_val
                return ema_val
            
            fast_ema = ema(prices, self.macd_fast)
            slow_ema = ema(prices, self.macd_slow)
            macd_line = fast_ema - slow_ema
            
            # Use instrument-specific MACD history
            macd_history = state.setdefault('macd_histogram_history', deque(maxlen=30))
            macd_history.append(macd_line)
            macd_values = list(macd_history)
            
            if len(macd_values) >= self.macd_signal:
                signal_line = ema(macd_values, self.macd_signal)
            else:
                signal_line = sum(macd_values) / len(macd_values) if macd_values else 0.0
            
            histogram = macd_line - signal_line
            return float(macd_line), float(signal_line), float(histogram)
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _calculate_confluence_scores(
        self, weighted_roc: float, rsi_value: float, macd_line: float,
        macd_signal_line: float, macd_histogram: float, stoch_k: float,
        obv_momentum: float, divergence_signal: Optional[str], volume_confirmation: float
    ) -> Tuple[float, float, float]:
        """Calculate bullish and bearish confluence scores."""
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        
        # ROC score (weight: 2.0)
        roc_weight = 2.0
        total_weight += roc_weight
        if weighted_roc > self.momentum_threshold:
            bullish_score += roc_weight * min(1.0, weighted_roc / (self.momentum_threshold * 3))
        elif weighted_roc < -self.momentum_threshold:
            bearish_score += roc_weight * min(1.0, abs(weighted_roc) / (self.momentum_threshold * 3))
        
        # RSI score (weight: 1.5)
        rsi_weight = 1.5
        total_weight += rsi_weight
        if rsi_value > 50:
            if rsi_value < self.rsi_overbought:
                bullish_score += rsi_weight * ((rsi_value - 50) / 30)
            elif rsi_value >= self.rsi_extreme_overbought:
                bearish_score += rsi_weight * 0.3
        else:
            if rsi_value > self.rsi_oversold:
                bearish_score += rsi_weight * ((50 - rsi_value) / 30)
            elif rsi_value <= self.rsi_extreme_oversold:
                bullish_score += rsi_weight * 0.3
        
        # MACD score (weight: 1.8)
        macd_weight = 1.8
        total_weight += macd_weight
        if macd_histogram > 0:
            macd_strength = min(1.0, abs(macd_histogram) * 100)
            if macd_line > macd_signal_line:
                bullish_score += macd_weight * macd_strength
        else:
            macd_strength = min(1.0, abs(macd_histogram) * 100)
            if macd_line < macd_signal_line:
                bearish_score += macd_weight * macd_strength
        
        # Stochastic score (weight: 1.2)
        stoch_weight = 1.2
        total_weight += stoch_weight
        if stoch_k > 50 and stoch_k < self.stoch_overbought:
            bullish_score += stoch_weight * ((stoch_k - 50) / 50)
        elif stoch_k < 50 and stoch_k > self.stoch_oversold:
            bearish_score += stoch_weight * ((50 - stoch_k) / 50)
        elif stoch_k >= self.stoch_overbought:
            bearish_score += stoch_weight * 0.4
        elif stoch_k <= self.stoch_oversold:
            bullish_score += stoch_weight * 0.4
        
        # OBV momentum score (weight: 1.0)
        obv_weight = 1.0
        total_weight += obv_weight
        if obv_momentum > 0.01:
            bullish_score += obv_weight * min(1.0, obv_momentum * 10)
        elif obv_momentum < -0.01:
            bearish_score += obv_weight * min(1.0, abs(obv_momentum) * 10)
        
        # Divergence bonus (weight: 1.5)
        div_weight = 1.5
        total_weight += div_weight
        if divergence_signal == 'bullish':
            bullish_score += div_weight
        elif divergence_signal == 'bearish':
            bearish_score += div_weight
        
        # Volume confirmation adjustment
        bullish_score *= (0.7 + volume_confirmation * 0.6)
        bearish_score *= (0.7 + volume_confirmation * 0.6)
        
        return bullish_score, bearish_score, total_weight
    
    def _determine_momentum_action(
        self, net_momentum: float, bullish_confluence: float,
        bearish_confluence: float, momentum_acceleration: float
    ) -> Tuple[str, float, float]:
        """Determine action, confidence and signal strength from momentum analysis."""
        if net_momentum > 0.01:
            action = 'long'
            if bullish_confluence >= self.strong_signal_confluence:
                signal_strength = min(1.0, bullish_confluence * 1.2)
            elif bullish_confluence >= self.min_confluence_score:
                signal_strength = min(0.8, bullish_confluence)
            else:
                signal_strength = max(0.2, bullish_confluence * 0.5)
            
            if momentum_acceleration > 0.02:
                signal_strength = min(1.0, signal_strength * 1.1)
            
            confidence = 0.3 + signal_strength * 0.5
            
        elif net_momentum < -0.01:
            action = 'short'
            if bearish_confluence >= self.strong_signal_confluence:
                signal_strength = min(1.0, bearish_confluence * 1.2)
            elif bearish_confluence >= self.min_confluence_score:
                signal_strength = min(0.8, bearish_confluence)
            else:
                signal_strength = max(0.2, bearish_confluence * 0.5)
            
            if momentum_acceleration < -0.02:
                signal_strength = min(1.0, signal_strength * 1.1)
            
            confidence = 0.3 + signal_strength * 0.5
            
        else:
            action = 'flat'
            signal_strength = 0.1
            confidence = 0.2
        
        return action, float(np.clip(confidence, 0.15, 0.95)), float(signal_strength)
    
    def _neutral_output(self, reason: str) -> Dict[str, Any]:
        """Generate neutral output with explanation."""
        thesis = f"Momentum flat: {reason}"
        name = self.__class__.__name__
        
        proposal = {
            "action": "flat",
            "signal_strength": 0.1,
            "reason": thesis,
            "proposals": {},
        }
        
        try:
            self.smart_bus.set('MomentumExpert_voting_proposal', proposal, module=name, thesis=thesis)
            self.smart_bus.set('MomentumExpert_confidence', 0.1, module=name, thesis=f'Confidence: 10%')
            self.smart_bus.set('momentum_voting_proposal', proposal, module=name, thesis=thesis)
            self.smart_bus.set('momentum_confidence', 0.1, module=name, thesis=f'Momentum confidence: 10%')
        except Exception:
            pass
        
        return {
            'MomentumExpert_voting_proposal': proposal,
            'MomentumExpert_confidence': 0.1,
            'MomentumExpert_per_instrument_votes': {},
            'momentum_voting_proposal': proposal,
            'momentum_confidence': 0.1,
            'momentum_analysis': {'composite_momentum': 0.0, 'per_instrument': {}},
            'voting_proposal': proposal,
            'confidence': 0.1,
            '_thesis': thesis,
        }
