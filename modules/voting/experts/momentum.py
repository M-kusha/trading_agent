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
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.experts.base import VotingExpertBase
from modules.voting.core.per_instrument import PerInstrumentVote, InstrumentProposal
from modules.voting.core.constants import (
    PRIMARY_TIMEFRAME,
    CONTEXT_TIMEFRAMES,
)


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
    """

    # ═══════════════════════════ INIT ═══════════════════════════

    def _expert_specific_init(self) -> None:
        """Initialize advanced momentum analysis state."""
        # Instruments to analyze (from config or default)
        self.instruments = self.config.get("instruments", ["EURUSD", "XAUUSD"])
        
        # ROC periods for multi-scale analysis
        # Keep 50-bar ROC but treat it as a true long-term window; combined with the
        # per-instrument min-length gate (max_roc_period + 5), this means MomentumExpert
        # only becomes active once we have comfortably more than 50 bars of history.
        self.roc_periods = [5, 10, 20, 50]
        self.roc_weights = [0.35, 0.30, 0.20, 0.15]  # Short-term weighted higher
        
        # RSI configuration
        self.rsi_period = int(self.config.get("rsi_period", 14))
        self.rsi_overbought = float(self.config.get("rsi_overbought", 70))
        self.rsi_oversold = float(self.config.get("rsi_oversold", 30))
        self.rsi_extreme_overbought = 80
        self.rsi_extreme_oversold = 20
        
        # MACD configuration
        self.macd_fast = int(self.config.get("macd_fast", 12))
        self.macd_slow = int(self.config.get("macd_slow", 26))
        self.macd_signal = int(self.config.get("macd_signal", 9))
        
        # Stochastic configuration
        self.stoch_k_period = int(self.config.get("stoch_k", 14))
        self.stoch_d_period = int(self.config.get("stoch_d", 3))
        self.stoch_overbought = 80
        self.stoch_oversold = 20
        
        # Momentum thresholds
        self.momentum_threshold = float(self.config.get("momentum_threshold", 0.015))
        self.strong_momentum_multiplier = 2.5
        self.weak_momentum_multiplier = 0.5
        
        # Divergence detection
        self.divergence_lookback = int(self.config.get("divergence_lookback", 20))
        self.divergence_significance = float(
            self.config.get("divergence_significance", 0.02)
        )
        
        # Confluence requirements - relatively low to give PPO material,
        # final filtering happens in VotingExpertBase._postprocess_proposal_for_voting
        self.min_confluence_score = float(self.config.get("min_confluence", 0.15))
        self.strong_signal_confluence = float(self.config.get("strong_confluence", 0.5))
        
        # Multi-timeframe configuration:
        # M15 is the PRIMARY trading timeframe (100% of signal generation).
        # H1/H4/D1 are CONTEXT timeframes (confidence modifiers ONLY, never override direction).
        # This is because ExitManager closes trades early with tight TP, so H1/H4/D1 trends
        # rarely have time to play out.
        self.use_mtf_confirmation = bool(self.config.get("use_mtf_confirmation", True))
        self.mtf_timeframes: List[str] = self.config.get(
            "mtf_timeframes", [PRIMARY_TIMEFRAME] + list(CONTEXT_TIMEFRAMES)
        )
        # M15-PRIMARY: M15 generates direction, context TFs only adjust confidence
        self.mtf_weights = {
            "M15": 1.00,  # PRIMARY: M15 is the SOLE signal generator
            "H1": 0.00,   # CONTEXT ONLY: modifies confidence, not direction
            "H4": 0.00,   # CONTEXT ONLY: modifies confidence, not direction
            "D1": 0.00,   # CONTEXT ONLY: modifies confidence, not direction
        }
        self.mtf_agreement_bonus = 0.15  # Confidence boost when context TFs agree with M15
        self.mtf_disagreement_penalty = 0.20  # Confidence penalty when context TFs disagree with M15

        # Optional performance feedback toggle (defaults off to avoid bias)
        self.use_performance_feedback: bool = bool(
            self.config.get("use_performance_feedback", False)
        )
        
        # Per-instrument state
        self.instrument_state: Dict[str, Dict[str, Any]] = {}
        for inst in self.instruments:
            inst_norm = normalize_instrument(inst)
            self.instrument_state[inst_norm] = {
                "price_history": deque(maxlen=200),
                "high_history": deque(maxlen=200),
                "low_history": deque(maxlen=200),
                "volume_history": deque(maxlen=200),
                "rsi_history": deque(maxlen=50),
                "macd_histogram_history": deque(maxlen=30),
                "obv_history": deque(maxlen=50),
                "momentum_history": deque(maxlen=100),
            }
        
        # Legacy single-instrument state (kept for compatibility, not used in per-instrument path)
        self.price_history: deque = deque(maxlen=200)
        self.high_history: deque = deque(maxlen=200)
        self.low_history: deque = deque(maxlen=200)
        self.volume_history: deque = deque(maxlen=200)
        self.close_history: deque = deque(maxlen=200)
        
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
        
        self.roc_values: Dict[int, float] = {p: 0.0 for p in self.roc_periods}
        
        # Aggregated momentum state
        self.composite_momentum: float = 0.0
        self.momentum_direction: int = 0  # -1, 0, 1
        self.momentum_acceleration: float = 0.0
        self.divergence_signal: Optional[str] = None
        self.volume_confirmation: float = 0.5
        
        # History / performance
        self.momentum_history: deque = deque(maxlen=100)
        self.signal_history: deque = deque(maxlen=50)
        
        self.momentum_performance: Dict[str, Dict[str, Any]] = {
            "long": {"signals": 0, "success": 0, "total_pnl": 0.0},
            "short": {"signals": 0, "success": 0, "total_pnl": 0.0},
            "flat": {"signals": 0, "success": 0, "total_pnl": 0.0},
        }
        
        self.bullish_divergence_count: int = 0
        self.bearish_divergence_count: int = 0
        
        self.log_info(
            f"[MOMENTUM] Advanced MomentumExpert initialized | "
            f"instruments={self.instruments} | ROC periods={self.roc_periods} | "
            f"RSI={self.rsi_period} | MACD={self.macd_fast}/{self.macd_slow}/{self.macd_signal}"
        )

        # Debug throttle per instrument
        self._debug_last_log: Dict[str, float] = {}

        # Baseline keys from VotingExpertBase prevent BUS MISS for {name}_voting_proposal.
        # Here we only add a baseline analysis blob.
        self._publish_momentum_baseline()
    
    def _publish_momentum_baseline(self) -> None:
        """Publish baseline momentum analysis so downstream consumers have structure."""
        try:
            self.smart_bus.set(
                "momentum_analysis",
                {
                    "composite_momentum": 0.0,
                    "direction": 0,
                    "acceleration": 0.0,
                    "per_instrument": {},
                    "performance": dict(self.momentum_performance),
                },
                module="MomentumExpert",
                thesis="Momentum analysis baseline",
            )
        except Exception:
            pass

    # ═══════════════════════════ INDICATORS ═══════════════════════════
    # (Single-instrument helpers kept but not used directly in per-instrument path)

    def _update_price_data(self, market_data: Dict[str, Any]) -> bool:
        """Legacy: update single-instrument price/volume data from market_data."""
        try:
            ohlcv = market_data.get("ohlcv") or {}
            prices = market_data.get("prices") or market_data.get("close_prices") or []
            
            if isinstance(prices, dict):
                close_prices = prices.get("close", [])
                high_prices = prices.get("high", [])
                low_prices = prices.get("low", [])
                volumes = prices.get("volume", [])
            elif ohlcv:
                close_prices = ohlcv.get("close", [])
                high_prices = ohlcv.get("high", [])
                low_prices = ohlcv.get("low", [])
                volumes = ohlcv.get("volume", [])
            else:
                close_prices = list(prices) if isinstance(prices, (list, np.ndarray)) else []
                high_prices = []
                low_prices = []
                volumes = []
            
            current_price = (
                market_data.get("current_price") or market_data.get("close")
            )
            if current_price is None and close_prices:
                current_price = close_prices[-1]
            
            current_high = market_data.get("high")
            if current_high is None and high_prices:
                current_high = high_prices[-1]
            elif current_high is None:
                current_high = current_price
            
            current_low = market_data.get("low")
            if current_low is None and low_prices:
                current_low = low_prices[-1]
            elif current_low is None:
                current_low = current_price
            
            current_volume = market_data.get("volume", 1.0)
            if current_volume is None or current_volume == 0:
                current_volume = 1.0
            
            if current_price is not None:
                self.price_history.append(float(current_price))
                self.close_history.append(float(current_price))
                self.high_history.append(
                    float(current_high) if current_high else float(current_price)
                )
                self.low_history.append(
                    float(current_low) if current_low else float(current_price)
                )
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
            deltas = np.diff(prices[-(period + 1) :])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
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
    
    def _calculate_macd_for_instrument(
        self, prices: List[float], state: Dict
    ) -> Tuple[float, float, float]:
        """Calculate MACD for a specific instrument using its state history."""
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
            
            macd_history = state.setdefault(
                "macd_histogram_history", deque(maxlen=30)
            )
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
    
    def _calculate_stochastic(
        self, closes: List[float], highs: List[float], lows: List[float]
    ) -> Tuple[float, float]:
        """Calculate Stochastic %K and %D."""
        if len(closes) < self.stoch_k_period:
            return 50.0, 50.0
        
        try:
            recent_closes = closes[-self.stoch_k_period :]
            recent_highs = (
                highs[-self.stoch_k_period :]
                if len(highs) >= self.stoch_k_period
                else recent_closes
            )
            recent_lows = (
                lows[-self.stoch_k_period :]
                if len(lows) >= self.stoch_k_period
                else recent_closes
            )
            
            highest_high = max(recent_highs)
            lowest_low = min(recent_lows)
            current_close = closes[-1]
            
            if highest_high == lowest_low:
                stoch_k = 50.0
            else:
                stoch_k = (
                    (current_close - lowest_low)
                    / (highest_high - lowest_low)
                    * 100
                )
            
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
                if closes[i] > closes[i - 1]:
                    obv += volumes[i]
                elif closes[i] < closes[i - 1]:
                    obv -= volumes[i]
            return float(obv)
        except Exception:
            return 0.0
    
    def _calculate_multi_period_roc(self, prices: List[float]) -> Dict[int, float]:
        """Calculate ROC for multiple periods."""
        roc_values: Dict[int, float] = {}
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
    
    def _detect_divergence(
        self, prices: List[float], indicator_values: List[float]
    ) -> Optional[str]:
        """Detect bullish or bearish divergence between price and indicator."""
        if (
            len(prices) < self.divergence_lookback
            or len(indicator_values) < self.divergence_lookback
        ):
            return None
        
        try:
            recent_prices = prices[-self.divergence_lookback :]
            recent_indicator = indicator_values[-self.divergence_lookback :]
            
            price_min_idx = int(np.argmin(recent_prices))
            price_max_idx = int(np.argmax(recent_prices))
            
            # Bullish divergence: lower price low, higher indicator low
            if price_min_idx > len(recent_prices) // 2:
                prev_low_idx = int(
                    np.argmin(recent_prices[: len(recent_prices) // 2])
                )
                if (
                    recent_prices[price_min_idx] < recent_prices[prev_low_idx]
                    and recent_indicator[price_min_idx]
                    > recent_indicator[prev_low_idx]
                ):
                    return "bullish"
            
            # Bearish divergence: higher price high, lower indicator high
            if price_max_idx > len(recent_prices) // 2:
                prev_high_idx = int(
                    np.argmax(recent_prices[: len(recent_prices) // 2])
                )
                if (
                    recent_prices[price_max_idx] > recent_prices[prev_high_idx]
                    and recent_indicator[price_max_idx]
                    < recent_indicator[prev_high_idx]
                ):
                    return "bearish"
            return None
        except Exception:
            return None
    
    def _calculate_volume_confirmation(
        self, prices: List[float], volumes: List[float]
    ) -> float:
        """Calculate volume confirmation score for current move."""
        if len(prices) < 10 or len(volumes) < 10:
            return 0.5
        
        try:
            recent_vol = np.mean(volumes[-5:])
            avg_vol = np.mean(volumes[-20:])
            if avg_vol == 0:
                return 0.5
            
            vol_ratio = recent_vol / avg_vol
            price_change = prices[-1] - prices[-5]
            vol_change = volumes[-1] - np.mean(volumes[-5:])
            
            if (price_change > 0 and vol_change > 0) or (
                price_change < 0 and vol_change > 0
            ):
                confirmation = min(1.0, 0.5 + vol_ratio * 0.25)
            else:
                confirmation = max(0.2, 0.5 - vol_ratio * 0.15)
            return float(confirmation)
        except Exception:
            return 0.5

    # ═══════════════════════════ PER-INSTRUMENT HELPERS ═══════════════════════════

    def _extract_instrument_data(self, data: Dict, instrument: str) -> Dict:
        """Extract data for a specific instrument from nested market data."""
        if not isinstance(data, dict):
            return {}
        
        inst_norm = normalize_instrument(instrument)
        
        for key in [instrument, inst_norm, instrument.upper(), instrument.lower()]:
            if key in data:
                return data[key] if isinstance(data[key], dict) else data
        
        # Simple canonical mapping for main pairs
        for sep in ["_", "/", "-", ""]:
            for pair in [f"EUR{sep}USD", f"XAU{sep}USD"]:
                norm_pair = normalize_instrument(pair)
                if norm_pair == inst_norm and pair in data:
                    return data[pair] if isinstance(data[pair], dict) else data
        
        # Legacy flat format
        return data
    
    def _extract_prices(
        self,
        market_data: Dict,
        features: Dict,
        price_type: str,
        instrument: str = "",
    ) -> np.ndarray:
        """Extract price array from market data or features for a specific instrument."""
        inst_norm = normalize_instrument(instrument) if instrument else ""
        inst_variations = [
            instrument,
            inst_norm,
            f"{inst_norm[:3]}_{inst_norm[3:]}" if len(inst_norm) >= 6 else inst_norm,
        ]
        
        # direct in market_data
        if isinstance(market_data, dict):
            if price_type in market_data:
                data = market_data[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)
            # nested by timeframe
            tfs = getattr(self, "mtf_timeframes", ["M15", "H1", "H4", "D1"])
            for tf in tfs:
                if tf in market_data and isinstance(market_data[tf], dict):
                    if price_type in market_data[tf]:
                        data = market_data[tf][price_type]
                        if isinstance(data, (list, np.ndarray)):
                            return np.array(data, dtype=float)
        
        # features fallback
        if isinstance(features, dict):
            if price_type in features:
                data = features[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)
        
        # InfoBus historical_prices fallback
        try:
            historical = self.smart_bus.get(
                "historical_prices", self.__class__.__name__, default=None
            )
        except Exception:
            historical = None
        
        if isinstance(historical, dict):
            matched_symbol = None
            for sym in historical.keys():
                sym_norm = normalize_instrument(sym)
                if sym_norm == inst_norm or sym in inst_variations:
                    matched_symbol = sym
                    break
            
            if matched_symbol and matched_symbol in historical:
                sym_block = historical[matched_symbol]
                if isinstance(sym_block, dict):
                    tfs_hist = (
                        getattr(self, "mtf_timeframes", None)
                        or ["M15", "H1", "H4", "D1"]
                    )
                    for tf in tfs_hist:
                        tf_rec = sym_block.get(tf)
                        if isinstance(tf_rec, dict):
                            seq = tf_rec.get(price_type)
                            if isinstance(seq, (list, np.ndarray)) and len(seq) > 0:
                                return np.array(seq, dtype=float)
        
        return np.array([])
    
    def _calculate_confluence_scores(
        self,
        weighted_roc: float,
        rsi_value: float,
        macd_line: float,
        macd_signal_line: float,
        macd_histogram: float,
        stoch_k: float,
        obv_momentum: float,
        divergence_signal: Optional[str],
        volume_confirmation: float,
    ) -> Tuple[float, float, float]:
        """Calculate bullish and bearish confluence scores."""
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        
        # ROC
        roc_weight = 2.0
        total_weight += roc_weight
        if weighted_roc > self.momentum_threshold:
            bullish_score += roc_weight * min(
                1.0, weighted_roc / (self.momentum_threshold * 3)
            )
        elif weighted_roc < -self.momentum_threshold:
            bearish_score += roc_weight * min(
                1.0, abs(weighted_roc) / (self.momentum_threshold * 3)
            )
        
        # RSI
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
        
        # MACD
        macd_weight = 1.8
        total_weight += macd_weight
        macd_strength = min(1.0, abs(macd_histogram) * 100)
        if macd_histogram > 0 and macd_line > macd_signal_line:
            bullish_score += macd_weight * macd_strength
        elif macd_histogram < 0 and macd_line < macd_signal_line:
            bearish_score += macd_weight * macd_strength
        
        # Stochastic
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
        
        # OBV momentum
        obv_weight = 1.0
        total_weight += obv_weight
        if obv_momentum > 0.01:
            bullish_score += obv_weight * min(1.0, obv_momentum * 10)
        elif obv_momentum < -0.01:
            bearish_score += obv_weight * min(1.0, abs(obv_momentum) * 10)
        
        # Divergence
        div_weight = 1.5
        total_weight += div_weight
        if divergence_signal == "bullish":
            bullish_score += div_weight
        elif divergence_signal == "bearish":
            bearish_score += div_weight
        
        # Volume confirmation
        bullish_score *= 0.7 + volume_confirmation * 0.6
        bearish_score *= 0.7 + volume_confirmation * 0.6
        
        return bullish_score, bearish_score, total_weight
    
    def _determine_momentum_action(
        self,
        net_momentum: float,
        bullish_confluence: float,
        bearish_confluence: float,
        momentum_acceleration: float,
    ) -> Tuple[str, float, float]:
        """Determine action, confidence and signal strength from momentum analysis."""
        if net_momentum > 0.01:
            action = "long"
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
            action = "short"
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
            action = "flat"
            signal_strength = 0.1
            confidence = 0.2
        
        return action, float(np.clip(confidence, 0.15, 0.95)), float(signal_strength)

    def _analyze_multi_timeframe_momentum(
        self,
        instrument: str,
    ) -> Dict[str, Any]:
        """
        Analyze momentum direction across multiple timeframes.
        
        Returns momentum direction for each timeframe, plus alignment score.
        This helps confirm lower timeframe signals (e.g. M15/H1) with higher timeframes.
        """
        name = self.__class__.__name__
        inst_norm = normalize_instrument(instrument)
        
        mtf_momentum: Dict[str, Dict[str, Any]] = {}
        
        try:
            historical = self.smart_bus.get("historical_prices", name, default=None)
        except Exception:
            historical = None
        
        if not isinstance(historical, dict):
            return {
                "available": False,
                "momentum": {},
                "alignment_score": 0.5,
                "dominant_direction": "neutral",
            }
        
        # Find instrument in historical data
        matched_symbol = None
        for sym in historical.keys():
            if normalize_instrument(sym) == inst_norm:
                matched_symbol = sym
                break
        
        if not matched_symbol or matched_symbol not in historical:
            return {
                "available": False,
                "momentum": {},
                "alignment_score": 0.5,
                "dominant_direction": "neutral",
            }
        
        sym_block = historical[matched_symbol]
        if not isinstance(sym_block, dict):
            return {
                "available": False,
                "momentum": {},
                "alignment_score": 0.5,
                "dominant_direction": "neutral",
            }
        
        # Analyze momentum for each timeframe
        for tf in self.mtf_timeframes:
            tf_data = sym_block.get(tf)
            if not isinstance(tf_data, dict):
                continue
            
            close_arr = tf_data.get("close")
            if not isinstance(close_arr, (list, np.ndarray)) or len(close_arr) < 20:
                continue
            
            prices = np.array(close_arr, dtype=float)
            
            # Calculate ROC for this timeframe (10-period and 20-period)
            roc_10 = 0.0
            roc_20 = 0.0
            if len(prices) > 10 and prices[-11] != 0:
                roc_10 = (prices[-1] - prices[-11]) / prices[-11]
            if len(prices) > 20 and prices[-21] != 0:
                roc_20 = (prices[-1] - prices[-21]) / prices[-21]
            
            # Calculate simple RSI
            rsi = self._calculate_rsi(list(prices), min(14, len(prices) - 1))
            
            # Determine momentum direction
            bullish_signals = 0
            bearish_signals = 0
            
            if roc_10 > 0.003:  # 0.3% positive momentum
                bullish_signals += 1
            elif roc_10 < -0.003:
                bearish_signals += 1
            
            if roc_20 > 0.005:  # 0.5% positive momentum
                bullish_signals += 1
            elif roc_20 < -0.005:
                bearish_signals += 1
            
            if rsi > 55:  # Bullish RSI
                bullish_signals += 1
            elif rsi < 45:
                bearish_signals += 1
            
            # Determine direction
            if bullish_signals >= 2 and bullish_signals > bearish_signals:
                direction = "bullish"
                strength = min(1.0, (abs(roc_10) + abs(roc_20)) * 10)
            elif bearish_signals >= 2 and bearish_signals > bullish_signals:
                direction = "bearish"
                strength = min(1.0, (abs(roc_10) + abs(roc_20)) * 10)
            else:
                direction = "neutral"
                strength = 0.2
            
            mtf_momentum[tf] = {
                "direction": direction,
                "strength": strength,
                "roc_10": roc_10,
                "roc_20": roc_20,
                "rsi": rsi,
            }
        
        if not mtf_momentum:
            return {
                "available": False,
                "momentum": {},
                "alignment_score": 0.5,
                "dominant_direction": "neutral",
            }
        
        # Calculate alignment score
        directions = [m["direction"] for m in mtf_momentum.values()]
        bullish_count = directions.count("bullish")
        bearish_count = directions.count("bearish")
        total_tf = len(directions)
        
        # Weighted alignment
        weighted_bullish = sum(
            self.mtf_weights.get(tf, 0.33)
            for tf, mom in mtf_momentum.items()
            if mom["direction"] == "bullish"
        )
        weighted_bearish = sum(
            self.mtf_weights.get(tf, 0.33)
            for tf, mom in mtf_momentum.items()
            if mom["direction"] == "bearish"
        )
        
        if bullish_count == total_tf:
            alignment_score = 1.0
            dominant = "bullish"
        elif bearish_count == total_tf:
            alignment_score = 1.0
            dominant = "bearish"
        elif bullish_count > bearish_count:
            alignment_score = bullish_count / total_tf
            dominant = "bullish" if weighted_bullish > weighted_bearish else "neutral"
        elif bearish_count > bullish_count:
            alignment_score = bearish_count / total_tf
            dominant = "bearish" if weighted_bearish > weighted_bullish else "neutral"
        else:
            alignment_score = 0.33
            dominant = "neutral"
        
        self.log_debug(
            f"[MOMENTUM MTF] {inst_norm}: "
            + ", ".join(f"{tf}={m['direction']}" for tf, m in mtf_momentum.items())
            + f" | alignment={alignment_score:.2f}, dominant={dominant}"
        )
        
        return {
            "available": True,
            "momentum": mtf_momentum,
            "alignment_score": alignment_score,
            "dominant_direction": dominant,
            "weighted_bullish": weighted_bullish,
            "weighted_bearish": weighted_bearish,
        }

    # ═══════════════════════════ CORE VOTING HOOKS ═══════════════════════════

    async def _generate_expert_specific_proposal(
        self,
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Per-instrument momentum analysis.
        
        Returns a single aggregated proposal dict; rich per-instrument data is
        also published to the bus for the CommitteeCoordinator.
        """
        name = self.__class__.__name__
        
        try:
            features = self.smart_bus.get("features", name, default={})
        except Exception:
            features = {}
        
        if not market_data and not features:
            self.log_debug(
                f"[MOMENTUM][BUS] empty fetch: market_data_keys={list(market_data.keys()) if isinstance(market_data, dict) else market_data}, "
                f"features_keys={list(features.keys()) if isinstance(features, dict) else features}"
            )
            self.log_warning("[MOMENTUM] No market data or features available")
            self.composite_momentum = 0.0
            self.momentum_direction = 0
            self.momentum_acceleration = 0.0
            return self._neutral_proposal("No market data available")
        
        per_instrument_vote = PerInstrumentVote(member=name)
        per_instrument_analysis: Dict[str, Dict[str, Any]] = {}
        
        max_roc_period = max(self.roc_periods) if self.roc_periods else 0
        
        for inst in self.instruments:
            inst_norm = normalize_instrument(inst)
            
            inst_market = self._extract_instrument_data(market_data, inst)
            inst_features = self._extract_instrument_data(features, inst)
            
            close_prices = self._extract_prices(inst_market, inst_features, "close", inst)
            high_prices = self._extract_prices(inst_market, inst_features, "high", inst)
            low_prices = self._extract_prices(inst_market, inst_features, "low", inst)
            volumes = self._extract_prices(inst_market, inst_features, "volume", inst)
            
            state = self.instrument_state.get(inst_norm)
            if state is None:
                state = self.instrument_state.setdefault(
                    inst_norm,
                    {
                        "price_history": deque(maxlen=200),
                        "high_history": deque(maxlen=200),
                        "low_history": deque(maxlen=200),
                        "volume_history": deque(maxlen=200),
                        "rsi_history": deque(maxlen=50),
                        "macd_histogram_history": deque(maxlen=30),
                        "obv_history": deque(maxlen=50),
                        "momentum_history": deque(maxlen=100),
                    },
                )
            
            if len(close_prices) > 0:
                state["price_history"] = deque(close_prices[-200:], maxlen=200)
                state["high_history"] = deque(
                    high_prices[-200:]
                    if len(high_prices) > 0
                    else close_prices[-200:],
                    maxlen=200,
                )
                state["low_history"] = deque(
                    low_prices[-200:]
                    if len(low_prices) > 0
                    else close_prices[-200:],
                    maxlen=200,
                )
                if len(volumes) > 0:
                    state["volume_history"] = deque(volumes[-200:], maxlen=200)
                else:
                    state["volume_history"] = deque(
                        [1.0] * min(200, len(close_prices)), maxlen=200
                    )
            
            prices = list(state.get("price_history", []))
            if len(prices) < max_roc_period + 5:
                tf_meta = {}
                # Fetch historical_prices from bus for MTF debug info
                try:
                    historical = self.smart_bus.get("historical_prices", name, default=None)
                except Exception:
                    historical = None
                if isinstance(historical, dict):
                    # Find matching symbol in historical data
                    matched_sym = None
                    for sym in historical.keys():
                        if normalize_instrument(sym) == inst_norm:
                            matched_sym = sym
                            break
                    if matched_sym and isinstance(historical.get(matched_sym), dict):
                        sym_block = historical[matched_sym]
                        for tf in ("M15", "H1", "H4", "D1"):
                            rec = sym_block.get(tf)
                            if isinstance(rec, dict):
                                bars_avail = rec.get("bars_available")
                                cur_bar = rec.get("current_bar") if isinstance(rec.get("current_bar"), dict) else {}
                                last_ts = cur_bar.get("timestamp") if isinstance(cur_bar, dict) else None
                                close_len = 0
                                seq = rec.get("close")
                                try:
                                    close_len = len(seq) if seq is not None else 0
                                except Exception:
                                    close_len = 0
                                tf_meta[tf] = {"close_len": close_len, "bars_available": bars_avail, "last_ts": last_ts}
                self.log_debug(
                    f"[MOMENTUM][DATA] {inst_norm} insufficient: price_len={len(prices)}, max_roc_period={max_roc_period}, tf_meta={tf_meta}"
                )
                self.log_debug(
                    f"[MOMENTUM] Insufficient data for {inst}: {len(prices)} bars"
                )
                per_instrument_vote.set_proposal(
                    InstrumentProposal(
                        instrument=inst_norm,
                        action="flat",
                        confidence=0.1,
                        magnitude=0.0,
                        rationale=f"Insufficient data for {inst}: {len(prices)} bars",
                    )
                )
                per_instrument_analysis[inst_norm] = {
                    "composite_momentum": 0.0,
                    "direction": 0,
                    "bullish_confluence": 0.0,
                    "bearish_confluence": 0.0,
                    "momentum_acceleration": 0.0,
                    "roc_weighted": 0.0,
                    "rsi": 50.0,
                    "macd_histogram": 0.0,
                    "stochastic_k": 50.0,
                    "obv_momentum": 0.0,
                    "divergence": None,
                    "volume_confirmation": 0.5,
                    "action": "flat",
                    "confidence": 0.1,
                }
                continue
            # Periodic debug snapshot of data freshness/lengths
            now_ts = time.time()
            last_log = self._debug_last_log.get(inst_norm, 0.0)
            if now_ts - last_log > 15.0:
                tf_meta = {}
                # Fetch historical_prices from bus for MTF debug info
                try:
                    historical = self.smart_bus.get("historical_prices", name, default=None)
                except Exception:
                    historical = None
                if isinstance(inst_market, dict):
                    try:
                        self.log_debug(f"[MOMENTUM][BUS] {inst_norm} inst_market keys={list(inst_market.keys())}")
                    except Exception:
                        pass
                if isinstance(historical, dict):
                    # Find matching symbol in historical data
                    matched_sym = None
                    for sym in historical.keys():
                        if normalize_instrument(sym) == inst_norm:
                            matched_sym = sym
                            break
                    if matched_sym and isinstance(historical.get(matched_sym), dict):
                        sym_block = historical[matched_sym]
                        for tf in ("M15", "H1", "H4", "D1"):
                            rec = sym_block.get(tf)
                            if isinstance(rec, dict):
                                bars_avail = rec.get("bars_available")
                                cur_bar = rec.get("current_bar") if isinstance(rec.get("current_bar"), dict) else {}
                                last_ts = cur_bar.get("timestamp") if isinstance(cur_bar, dict) else None
                                close_len = 0
                                seq = rec.get("close")
                                try:
                                    close_len = len(seq) if seq is not None else 0
                                except Exception:
                                    close_len = 0
                                tf_meta[tf] = {"close_len": close_len, "bars_available": bars_avail, "last_ts": last_ts}
                self.log_debug(
                    f"[MOMENTUM][DATA] {inst_norm}: price_len={len(prices)}, "
                    f"latest={prices[-1] if prices else None}, tf_meta={tf_meta}"
                )
                self._debug_last_log[inst_norm] = now_ts
            
            highs = list(state.get("high_history", prices))
            lows = list(state.get("low_history", prices))
            vols = list(state.get("volume_history", [1.0] * len(prices)))
            
            roc_values = self._calculate_multi_period_roc(prices)
            weighted_roc = sum(
                roc_values.get(p, 0.0) * w
                for p, w in zip(self.roc_periods, self.roc_weights)
            )
            
            rsi_value = self._calculate_rsi(prices, self.rsi_period)
            macd_line, macd_signal_line, macd_histogram = (
                self._calculate_macd_for_instrument(prices, state)
            )
            stoch_k, stoch_d = self._calculate_stochastic(prices, highs, lows)
            obv = self._calculate_obv(prices, vols)
            
            obv_history = state.setdefault("obv_history", deque(maxlen=50))
            obv_history.append(obv)
            obv_momentum = 0.0
            if len(obv_history) > 5:
                obv_recent = list(obv_history)[-5:]
                obv_momentum = (obv_recent[-1] - obv_recent[0]) / (
                    abs(obv_recent[0]) + 1e-10
                )
            
            volume_confirmation = self._calculate_volume_confirmation(prices, vols)
            
            rsi_history = state.setdefault("rsi_history", deque(maxlen=50))
            rsi_history.append(rsi_value)
            divergence_signal = self._detect_divergence(
                prices[-len(rsi_history) :], list(rsi_history)
            )
            if divergence_signal == "bullish":
                self.bullish_divergence_count += 1
            elif divergence_signal == "bearish":
                self.bearish_divergence_count += 1
            
            bullish_score, bearish_score, total_weight = (
                self._calculate_confluence_scores(
                    weighted_roc,
                    rsi_value,
                    macd_line,
                    macd_signal_line,
                    macd_histogram,
                    stoch_k,
                    obv_momentum,
                    divergence_signal,
                    volume_confirmation,
                )
            )
            
            bullish_confluence = (
                bullish_score / total_weight if total_weight > 0 else 0.0
            )
            bearish_confluence = (
                bearish_score / total_weight if total_weight > 0 else 0.0
            )
            net_momentum = bullish_confluence - bearish_confluence
            
            momentum_history = state.setdefault(
                "momentum_history", deque(maxlen=100)
            )
            momentum_history.append(
                {
                    "momentum": net_momentum,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )
            momentum_acceleration = 0.0
            if len(momentum_history) > 5:
                recent_mom = [h["momentum"] for h in list(momentum_history)[-5:]]
                momentum_acceleration = (recent_mom[-1] - recent_mom[0]) / 5.0
            
            action, inst_confidence, signal_strength = (
                self._determine_momentum_action(
                    net_momentum,
                    bullish_confluence,
                    bearish_confluence,
                    momentum_acceleration,
                )
            )
            
            # ═══════════════════════════════════════════════════════════════
            # M15-PRIMARY MULTI-TIMEFRAME CONFIRMATION
            # M15 is the SOLE signal generator - H1/H4/D1 ONLY modify confidence
            # Context TFs NEVER override M15 direction (ExitManager closes early)
            # ═══════════════════════════════════════════════════════════════
            mtf_adjustment = 0.0
            mtf_info = ""
            
            if self.use_mtf_confirmation and action in ("long", "short"):
                mtf_analysis = self._analyze_multi_timeframe_momentum(inst)
                
                if mtf_analysis.get("available"):
                    dominant = mtf_analysis.get("dominant_direction", "neutral")
                    alignment = mtf_analysis.get("alignment_score", 0.5)
                    
                    signal_is_bullish = action == "long"
                    mtf_is_bullish = dominant == "bullish"
                    mtf_is_bearish = dominant == "bearish"
                    
                    if signal_is_bullish and mtf_is_bullish:
                        mtf_adjustment = self.mtf_agreement_bonus * alignment
                        mtf_info = f"Context TFs AGREE ↑ (align={alignment:.2f})"
                    elif (not signal_is_bullish) and mtf_is_bearish:
                        mtf_adjustment = self.mtf_agreement_bonus * alignment
                        mtf_info = f"Context TFs AGREE ↓ (align={alignment:.2f})"
                    elif (signal_is_bullish and mtf_is_bearish) or (
                        not signal_is_bullish and mtf_is_bullish
                    ):
                        # M15-PRIMARY: Context TFs ONLY penalize confidence, NEVER override direction
                        mtf_adjustment = -self.mtf_disagreement_penalty * alignment
                        mtf_info = f"Context TFs DISAGREE (align={alignment:.2f}, conf penalty applied)"
                        # NOTE: We do NOT override action to flat - M15 is the decision maker
                    else:
                        mtf_adjustment = -0.03
                        mtf_info = "Context TFs neutral"
                    
                    inst_confidence = max(
                        0.1, min(0.95, inst_confidence + mtf_adjustment)
                    )
                    inst_analysis = per_instrument_analysis.get(inst_norm, {})
                    inst_analysis["mtf_analysis"] = mtf_analysis
                    inst_analysis["mtf_adjustment"] = mtf_adjustment
                    per_instrument_analysis[inst_norm] = inst_analysis
            
            # Track signals count only
            self.momentum_performance[action]["signals"] += 1
            
            thesis = (
                f"{inst_norm}: Momentum {action} "
                f"(net={net_momentum:.2%}, conf={inst_confidence:.1%}) {mtf_info}"
            )
            
            per_instrument_vote.set_proposal(
                InstrumentProposal(
                    instrument=inst_norm,
                    action=action,
                    confidence=inst_confidence,
                    magnitude=signal_strength,
                    rationale=thesis,
                )
            )
            
            # Merge with any earlier MTF metadata instead of overwriting
            inst_analysis = per_instrument_analysis.get(inst_norm, {})
            inst_analysis.update(
                {
                    "composite_momentum": net_momentum,
                    "direction": 1
                    if action == "long"
                    else -1
                    if action == "short"
                    else 0,
                    "bullish_confluence": bullish_confluence,
                    "bearish_confluence": bearish_confluence,
                    "momentum_acceleration": momentum_acceleration,
                    "roc_weighted": weighted_roc,
                    "rsi": rsi_value,
                    "macd_histogram": macd_histogram,
                    "stochastic_k": stoch_k,
                    "obv_momentum": obv_momentum,
                    "divergence": divergence_signal,
                    "volume_confirmation": volume_confirmation,
                    "action": action,
                    "confidence": inst_confidence,
                }
            )
            per_instrument_analysis[inst_norm] = inst_analysis
            
            self.log_debug(
                f"[MOMENTUM] {inst}: action={action}, conf={inst_confidence:.2f}, "
                f"momentum={net_momentum:.2%}"
            )
        
        if not per_instrument_vote.proposals:
            self.composite_momentum = 0.0
            self.momentum_direction = 0
            self.momentum_acceleration = 0.0
            return self._neutral_proposal("No instrument data available")
        
        # Leader instrument = highest confidence
        best_proposal = max(
            per_instrument_vote.proposals.values(),
            key=lambda p: p.confidence,
        )
        leader_inst = best_proposal.instrument
        leader_analysis = per_instrument_analysis.get(leader_inst, {})
        
        global_action = best_proposal.action
        global_signal_strength = best_proposal.magnitude
        thesis = best_proposal.rationale
        
        self.composite_momentum = float(
            leader_analysis.get("composite_momentum", 0.0)
        )
        self.momentum_direction = int(leader_analysis.get("direction", 0))
        self.momentum_acceleration = float(
            leader_analysis.get("momentum_acceleration", 0.0)
        )
        
        proposals_dict = {
            inst: prop.to_dict()
            for inst, prop in per_instrument_vote.proposals.items()
        }

        max_strength = getattr(self, "max_signal_strength", 1.0)
        
        proposal: Dict[str, Any] = {
            "action": global_action,
            "signal_strength": float(global_signal_strength),
            "position_size": float(
                min(max_strength, global_signal_strength * 0.6)
            ),
            "leader_instrument": leader_inst,
            "reason": thesis,
            "proposals": proposals_dict,
            "per_instrument": per_instrument_analysis,
            # aggregated metrics (used by _calculate_expert_specific_confidence)
            "composite_momentum": self.composite_momentum,
            "bullish_confluence": float(
                leader_analysis.get("bullish_confluence", 0.0)
            ),
            "bearish_confluence": float(
                leader_analysis.get("bearish_confluence", 0.0)
            ),
            "momentum_acceleration": self.momentum_acceleration,
            "divergence": leader_analysis.get("divergence"),
            "volume_confirmation": float(
                leader_analysis.get("volume_confirmation", 0.5)
            ),
        }
        
        # Publish rich per-instrument votes & analysis (does not replace canonical expert keys)
        try:
            per_inst_votes_dict = {
                inst: prop.to_dict()
                for inst, prop in per_instrument_vote.proposals.items()
            }
            self.smart_bus.set(
                "MomentumExpert_per_instrument_votes",
                per_inst_votes_dict,
                module=name,
                thesis=f"Per-instrument momentum votes: {list(per_inst_votes_dict.keys())}",
            )
            momentum_analysis = {
                "composite_momentum": self.composite_momentum,
                "direction": self.momentum_direction,
                "acceleration": self.momentum_acceleration,
                "per_instrument": per_instrument_analysis,
                "performance": dict(self.momentum_performance),
            }
            self.smart_bus.set(
                "momentum_analysis",
                momentum_analysis,
                module=name,
                thesis=f"Momentum analysis leader={leader_inst}",
            )
        except Exception as e:
            self.log_warning(
                f"[MOMENTUM] Failed to publish rich analysis to bus: {e}"
            )
        
        return proposal
    
    async def _calculate_expert_specific_confidence(
        self,
        proposal: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> float:
        """Global confidence derived from aggregated metrics (leader instrument)."""
        try:
            action = proposal.get("action", "flat")
            
            signal_strength = float(proposal.get("signal_strength", 0.0) or 0.0)
            base = 0.3 + signal_strength * 0.4
            
            if action == "long":
                confluence = float(
                    proposal.get("bullish_confluence", 0.0) or 0.0
                )
            elif action == "short":
                confluence = float(
                    proposal.get("bearish_confluence", 0.0) or 0.0
                )
            else:
                confluence = 0.0
            
            if confluence >= self.strong_signal_confluence:
                base *= 1.25
            elif confluence >= self.min_confluence_score:
                base *= 1.1
            
            if proposal.get("divergence"):
                base *= 1.15
            
            vol_conf = float(
                proposal.get("volume_confirmation", 0.5) or 0.5
            )
            base *= 0.85 + vol_conf * 0.3
            
            accel = float(
                proposal.get("momentum_acceleration", 0.0) or 0.0
            )
            if (action == "long" and accel > 0) or (
                action == "short" and accel < 0
            ):
                base *= 1.08
            
            # Optional performance feedback, off by default
            perf = self.momentum_performance.get(action, {})
            if self.use_performance_feedback and perf.get("signals", 0) > 20:
                success_rate = (
                    perf.get("success", 0) / perf["signals"]
                )
                base *= 0.7 + success_rate * 0.6
            
            return float(max(0.15, min(0.95, base)))
        except Exception:
            return 0.4

    # ═══════════════════════════ NEUTRAL HELPERS ═══════════════════════════

    def _neutral_proposal(self, reason: str) -> Dict[str, Any]:
        """Lightweight neutral proposal used when data is missing."""
        name = self.__class__.__name__
        self.composite_momentum = 0.0
        self.momentum_direction = 0
        self.momentum_acceleration = 0.0
        # Always publish per-instrument votes to satisfy contract requirements
        try:
            self.smart_bus.set(
                "MomentumExpert_per_instrument_votes",
                {},
                module=name,
                thesis=f"Neutral: {reason}",
            )
            self.smart_bus.set(
                "momentum_analysis",
                {
                    "composite_momentum": 0.0,
                    "direction": 0,
                    "acceleration": 0.0,
                    "per_instrument": {},
                },
                module=name,
                thesis=f"Neutral: {reason}",
            )
        except Exception:
            pass
        return {
            "action": "flat",
            "signal_strength": 0.1,
            "position_size": 0.0,
            "reason": f"Momentum flat: {reason}",
            "proposals": {},
            "per_instrument": {},
        }

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Override base process to add legacy momentum_* outputs required by the contract.

        This keeps the unified VotingExpertBase pipeline intact while ensuring that
        contract-provided keys like 'momentum_voting_proposal', 'momentum_confidence',
        and 'momentum_analysis' are always present in the returned dict so that
        validation and downstream consumers do not see missing outputs.
        """
        base_outputs = await super().process(**inputs)

        name = self.__class__.__name__

        # Canonical expert outputs from VotingExpertBase
        proposal = base_outputs.get("voting_proposal") or base_outputs.get(
            f"{name}_voting_proposal", {}
        )
        confidence = base_outputs.get(
            "confidence", base_outputs.get(f"{name}_confidence", 0.0)
        )

        # Momentum analysis: prefer an explicit field, then SmartInfoBus snapshot,
        # then a minimal synthetic structure based on current state.
        analysis = base_outputs.get("momentum_analysis")
        if analysis is None:
            try:
                analysis = self.smart_bus.get(
                    "momentum_analysis", name, default=None
                )
            except Exception:
                analysis = None
            if analysis is None:
                if not isinstance(proposal, dict):
                    proposal = {}
                analysis = {
                    "composite_momentum": float(
                        getattr(self, "composite_momentum", 0.0)
                    ),
                    "direction": int(
                        getattr(self, "momentum_direction", 0)
                    ),
                    "acceleration": float(
                        getattr(self, "momentum_acceleration", 0.0)
                    ),
                    "per_instrument": proposal.get("per_instrument", {}),
                    "performance": dict(
                        getattr(self, "momentum_performance", {})
                    ),
                }

        # Attach legacy alias keys expected by the contract; keep existing values if present.
        base_outputs.setdefault("momentum_voting_proposal", proposal)
        base_outputs.setdefault("momentum_confidence", confidence)
        base_outputs.setdefault("momentum_analysis", analysis)

        return base_outputs

    # ═══════════════════════════════════════════════════════════════════
    # STATE PERSISTENCE - Save/Load module state
    # ═══════════════════════════════════════════════════════════════════

    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.
        
        Saves per-instrument momentum state:
        - RSI values and history
        - MACD values
        - Momentum indicators
        - Divergence patterns
        """
        instrument_states = {}
        for inst, state in self.instrument_state.items():
            instrument_states[inst] = {
                "rsi_value": float(state.get("rsi_value", 50.0)),
                "macd_value": float(state.get("macd_value", 0.0)),
                "macd_signal": float(state.get("macd_signal", 0.0)),
                "macd_histogram": float(state.get("macd_histogram", 0.0)),
                "stoch_k": float(state.get("stoch_k", 50.0)),
                "stoch_d": float(state.get("stoch_d", 50.0)),
                "composite_momentum": float(state.get("composite_momentum", 0.0)),
                "momentum_direction": int(state.get("momentum_direction", 0)),
                "momentum_acceleration": float(state.get("momentum_acceleration", 0.0)),
                "rsi_history": list(state.get("rsi_history", []))[-20:],
                "macd_history": list(state.get("macd_history", []))[-20:],
                "momentum_history": list(state.get("momentum_history", []))[-20:],
            }
        
        return {
            "instrument_state": instrument_states,
            "composite_momentum": float(getattr(self, "composite_momentum", 0.0)),
            "momentum_direction": int(getattr(self, "momentum_direction", 0)),
            "momentum_acceleration": float(getattr(self, "momentum_acceleration", 0.0)),
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        """
        Restore custom state from persistence.
        """
        if not state:
            return
        
        # Restore per-instrument state
        inst_states = state.get("instrument_state", {})
        for inst, saved in inst_states.items():
            if inst in self.instrument_state:
                self.instrument_state[inst].update({
                    "rsi_value": float(saved.get("rsi_value", 50.0)),
                    "macd_value": float(saved.get("macd_value", 0.0)),
                    "macd_signal": float(saved.get("macd_signal", 0.0)),
                    "macd_histogram": float(saved.get("macd_histogram", 0.0)),
                    "stoch_k": float(saved.get("stoch_k", 50.0)),
                    "stoch_d": float(saved.get("stoch_d", 50.0)),
                    "composite_momentum": float(saved.get("composite_momentum", 0.0)),
                    "momentum_direction": int(saved.get("momentum_direction", 0)),
                    "momentum_acceleration": float(saved.get("momentum_acceleration", 0.0)),
                })
                # Restore histories as deques
                for key in ["rsi_history", "macd_history", "momentum_history"]:
                    hist = saved.get(key, [])
                    if key in self.instrument_state[inst]:
                        self.instrument_state[inst][key] = deque(hist, maxlen=30)
        
        # Restore legacy single-instrument state
        self.composite_momentum = float(state.get("composite_momentum", 0.0))
        self.momentum_direction = int(state.get("momentum_direction", 0))
        self.momentum_acceleration = float(state.get("momentum_acceleration", 0.0))
        
        self.log_info(
            f"📂 MomentumExpert state restored | "
            f"instruments={len(inst_states)}"
        )
