"""
Advanced Trend Expert
=====================
A sophisticated trend-based voting expert that combines multiple trend
detection methods, trend strength analysis, support/resistance awareness,
and multi-timeframe trend alignment to generate high-conviction directional
trading signals.

Per-Instrument Voting:
- Analyzes each instrument separately to produce per-instrument votes
- Each instrument gets its own action/confidence based on its own trend analysis
"""

from __future__ import annotations

import datetime
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.experts.base import VotingExpertBase
from modules.voting.core.per_instrument import (
    PerInstrumentVote,
    InstrumentProposal,
    normalize_instrument,
)
from modules.voting.core.constants import (
    CONFIDENCE_THRESHOLD_F,
    MIN_SIGNAL_STRENGTH_F,
    HIGH_CONFIDENCE_THRESHOLD_F,
    PRIMARY_TIMEFRAME,
    CONTEXT_TIMEFRAMES,
)


@module(**module_args("TrendExpert"))
class TrendExpert(VotingExpertBase):
    """
    Advanced Trend Expert - Multi-indicator trend analysis.

    Combines 8+ trend indicators with support/resistance and
    multi-timeframe alignment for high-conviction directional signals.
    """

    # ═══════════════════════════ INIT ═══════════════════════════

    def _expert_specific_init(self) -> None:
        """Initialize advanced trend analysis state."""
        # Instruments to analyze (from config or default)
        self.instruments = self.config.get("instruments", ["EURUSD", "XAUUSD"])

        # Triple MA configuration
        # Use a slower long MA so TrendExpert only becomes active once
        # we have substantially more than 50 bars of history (via MT5/MarketDataProvider).
        self.fast_period = int(self.config.get("fast_period", 8))
        self.medium_period = int(self.config.get("medium_period", 21))
        # Default was 55; bump to 90 so the effective minimum bars (slow_period + 10)
        # is ~100, matching the broader architecture's 100-bar windows.
        self.slow_period = int(self.config.get("slow_period", 90))

        # ADX configuration
        self.adx_period = int(self.config.get("adx_period", 14))
        self.adx_trending_threshold = float(self.config.get("adx_trending", 25))
        self.adx_strong_threshold = float(self.config.get("adx_strong", 40))

        # Parabolic SAR
        self.sar_af_start = float(self.config.get("sar_af_start", 0.02))
        self.sar_af_max = float(self.config.get("sar_af_max", 0.2))

        # Trend thresholds
        # 0.3% difference by default
        self.trend_threshold = float(self.config.get("trend_threshold", 0.003))
        # Net-trend gating (relative difference between bull/bear confluence)
        self.min_net_trend = float(self.config.get("min_net_trend", 0.02))
        self.weak_net_trend = float(self.config.get("weak_net_trend", 0.01))
        self.strong_trend_multiplier = 2.5

        # Confluence requirements
        self.min_confluence_score = float(self.config.get("min_confluence", 0.15))
        self.strong_signal_confluence = float(
            self.config.get("strong_confluence", 0.65)
        )

        # S/R detection
        self.sr_lookback = int(self.config.get("sr_lookback", 50))
        # 0.5% proximity
        self.sr_threshold = float(self.config.get("sr_threshold", 0.005))

        # Risk / signal caps
        self.max_signal_strength = float(self.config.get("max_signal_strength", 1.0))

        # Multi-timeframe configuration:
        # M15 is the PRIMARY trading timeframe (100% of signal generation).
        # H1/H4/D1 are CONTEXT timeframes (confidence modifiers ONLY, never override direction).
        # This is because ExitManager closes trades early with tight TP, so H1/H4/D1 trends
        # rarely have time to play out.
        self.use_mtf_confirmation = bool(self.config.get("use_mtf_confirmation", True))
        self.mtf_timeframes = [PRIMARY_TIMEFRAME] + list(CONTEXT_TIMEFRAMES)  # M15 primary, H1/H4/D1 context
        # M15-PRIMARY: M15 generates direction, context TFs only adjust confidence
        self.mtf_weights = {
            "M15": 1.00,  # PRIMARY: M15 is the SOLE signal generator
            "H1": 0.00,   # CONTEXT ONLY: modifies confidence, not direction
            "H4": 0.00,   # CONTEXT ONLY: modifies confidence, not direction
            "D1": 0.00,   # CONTEXT ONLY: modifies confidence, not direction
        }
        self.mtf_agreement_bonus = 0.15  # Confidence boost when context TFs agree with M15
        self.mtf_disagreement_penalty = 0.20  # Confidence penalty when context TFs disagree with M15

        # ═══════════════════════════ PER-INSTRUMENT STATE ═══════════════════════════
        self.instrument_state: Dict[str, Dict[str, Any]] = {}
        for inst in self.instruments:
            inst_norm = normalize_instrument(inst)
            self.instrument_state[inst_norm] = {
                "price_history": deque(maxlen=200),
                "high_history": deque(maxlen=200),
                "low_history": deque(maxlen=200),
                "close_history": deque(maxlen=200),
                "fast_ma": 0.0,
                "medium_ma": 0.0,
                "slow_ma": 0.0,
                "ma_history": deque(maxlen=50),
                "adx_value": 0.0,
                "plus_di": 0.0,
                "minus_di": 0.0,
                "adx_history": deque(maxlen=30),
                "sar_value": 0.0,
                "sar_direction": 0,
                "current_trend": "neutral",
                "trend_strength": 0.0,
                "trend_slope": 0.0,
                "trend_duration": 0,
                "ma_alignment": 0,
                "support_levels": [],
                "resistance_levels": [],
                "trend_history": deque(maxlen=100),
            }

        # Legacy single-instrument state (kept for compatibility with base hooks)
        self.price_history: deque = deque(maxlen=200)
        self.high_history: deque = deque(maxlen=200)
        self.low_history: deque = deque(maxlen=200)
        self.close_history: deque = deque(maxlen=200)

        self.fast_ma: float = 0.0
        self.medium_ma: float = 0.0
        self.slow_ma: float = 0.0
        self.ma_history: deque = deque(maxlen=50)

        self.adx_value: float = 0.0
        self.plus_di: float = 0.0
        self.minus_di: float = 0.0
        self.adx_history: deque = deque(maxlen=30)

        self.sar_value: float = 0.0
        self.sar_direction: int = 0  # 1 = bullish, -1 = bearish

        self.current_trend: str = "neutral"
        self.trend_strength: float = 0.0
        self.trend_slope: float = 0.0
        self.trend_duration: int = 0
        self.ma_alignment: int = 0

        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        self.near_support: bool = False
        self.near_resistance: bool = False

        self.trend_history: deque = deque(maxlen=100)
        self.signal_history: deque = deque(maxlen=50)

        # Performance tracking
        self.trend_performance: Dict[str, Dict[str, Any]] = {
            "long": {"signals": 0, "success": 0, "total_pnl": 0.0},
            "short": {"signals": 0, "success": 0, "total_pnl": 0.0},
            "flat": {"signals": 0, "success": 0, "total_pnl": 0.0},
        }
        # Debug throttle per instrument
        self._debug_last_log: Dict[str, float] = {}

        self.log_info(
            f"[TREND] Advanced TrendExpert initialized | "
            f"instruments={self.instruments} | MA periods={self.fast_period}/{self.medium_period}/{self.slow_period} | "
            f"ADX period={self.adx_period} | ADX threshold={self.adx_trending_threshold}"
        )

        self._publish_baseline_keys()
        self._publish_trend_baseline()

    def _publish_trend_baseline(self) -> None:
        """Publish trend baseline keys."""
        try:
            self.smart_bus.set(
                "trend_voting_proposal",
                {
                    "action": "flat",
                    "signal_strength": 0.0,
                    "reason": "Trend baseline",
                },
                module="TrendExpert",
                thesis="Trend baseline",
            )
            self.smart_bus.set(
                "trend_confidence",
                0.1,
                module="TrendExpert",
                thesis="Trend baseline confidence",
            )
            self.smart_bus.set(
                "trend_analysis",
                {"current_trend": "neutral", "trend_strength": 0.0},
                module="TrendExpert",
                thesis="Trend analysis baseline",
            )
        except Exception:
            pass

    # ═══════════════════════════ INDICATOR CALCULATIONS ═══════════════════════════

    def _update_price_data(self, market_data: Dict[str, Any]) -> bool:
        """Extract and update price data (legacy single-instrument path)."""
        try:
            ohlcv = market_data.get("ohlcv") or {}
            prices = (
                market_data.get("prices")
                or market_data.get("close_prices")
                or []
            )

            if isinstance(prices, dict):
                close_prices = prices.get("close", [])
                high_prices = prices.get("high", [])
                low_prices = prices.get("low", [])
            elif ohlcv:
                close_prices = ohlcv.get("close", [])
                high_prices = ohlcv.get("high", [])
                low_prices = ohlcv.get("low", [])
            else:
                close_prices = (
                    list(prices) if isinstance(prices, (list, np.ndarray)) else []
                )
                high_prices = []
                low_prices = []

            current_price = market_data.get("current_price") or market_data.get("close")
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

            if current_price is not None:
                cp = float(current_price)
                self.price_history.append(cp)
                self.close_history.append(cp)
                self.high_history.append(
                    float(current_high) if current_high else cp
                )
                self.low_history.append(float(current_low) if current_low else cp)
                return True

            return False

        except Exception as e:
            self.log_warning(f"[TREND] Price data update failed: {e}")
            return False

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return float(sum(prices) / len(prices)) if prices else 0.0

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
            return float(sum(prices) / len(prices)) if prices else 0.0
        return float(sum(prices[-period:]) / period)

    def _calculate_triple_ma(
        self, prices: List[float]
    ) -> Tuple[float, float, float, int]:
        """Calculate triple MA system and alignment."""
        fast = self._calculate_ema(prices, self.fast_period)
        medium = self._calculate_ema(prices, self.medium_period)
        slow = self._calculate_ema(prices, self.slow_period)

        if fast > medium > slow:
            alignment = 1
        elif fast < medium < slow:
            alignment = -1
        else:
            alignment = 0

        return fast, medium, slow, alignment

    def _calculate_adx(
        self, highs: List[float], lows: List[float], closes: List[float]
    ) -> Tuple[float, float, float]:
        """Calculate ADX with +DI and -DI."""
        period = self.adx_period
        if (
            len(highs) < period + 1
            or len(lows) < period + 1
            or len(closes) < period + 1
        ):
            # Neutral defaults
            return 20.0, 50.0, 50.0

        try:
            highs_arr = np.array(highs[-(period + 1):], dtype=np.float64)
            lows_arr = np.array(lows[-(period + 1):], dtype=np.float64)
            closes_arr = np.array(closes[-(period + 1):], dtype=np.float64)

            tr1 = highs_arr[1:] - lows_arr[1:]
            tr2 = np.abs(highs_arr[1:] - closes_arr[:-1])
            tr3 = np.abs(lows_arr[1:] - closes_arr[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))

            up_move = highs_arr[1:] - highs_arr[:-1]
            down_move = lows_arr[:-1] - lows_arr[1:]

            plus_dm = np.where(
                (up_move > down_move) & (up_move > 0), up_move, 0
            )
            minus_dm = np.where(
                (down_move > up_move) & (down_move > 0), down_move, 0
            )

            atr = np.mean(tr)
            plus_dm_avg = np.mean(plus_dm)
            minus_dm_avg = np.mean(minus_dm)

            if atr == 0:
                return 20.0, 50.0, 50.0

            plus_di = (plus_dm_avg / atr) * 100
            minus_di = (minus_dm_avg / atr) * 100

            di_sum = plus_di + minus_di
            dx = 0.0 if di_sum == 0 else abs(plus_di - minus_di) / di_sum * 100
            adx = dx  # simplified

            return (
                float(np.clip(adx, 0, 100)),
                float(np.clip(plus_di, 0, 100)),
                float(np.clip(minus_di, 0, 100)),
            )

        except Exception:
            return 20.0, 50.0, 50.0

    def _calculate_parabolic_sar(
        self, highs: List[float], lows: List[float]
    ) -> Tuple[float, int]:
        """Calculate Parabolic SAR (simplified)."""
        if len(highs) < 5 or len(lows) < 5:
            return 0.0, 0

        try:
            current_high = highs[-1]
            current_low = lows[-1]
            prev_high = max(highs[-5:-1])
            prev_low = min(lows[-5:-1])

            if current_high > prev_high and current_low > prev_low:
                sar = prev_low * (1 - self.sar_af_start)
                direction = 1
            elif current_high < prev_high and current_low < prev_low:
                sar = prev_high * (1 + self.sar_af_start)
                direction = -1
            else:
                sar = (prev_high + prev_low) / 2
                direction = 0

            return float(sar), direction

        except Exception:
            return 0.0, 0

    def _calculate_trend_slope(
        self, prices: List[float], lookback: int = 20
    ) -> float:
        """Calculate linear regression slope for trend direction."""
        if len(prices) < lookback:
            return 0.0

        try:
            recent = prices[-lookback:]
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]
            avg_price = float(np.mean(recent))
            if avg_price == 0:
                return 0.0
            return float(slope / avg_price)
        except Exception:
            return 0.0

    def _find_support_resistance(
        self, highs: List[float], lows: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Find key support and resistance levels."""
        if len(highs) < self.sr_lookback or len(lows) < self.sr_lookback:
            return [], []

        try:
            recent_highs = highs[-self.sr_lookback:]
            recent_lows = lows[-self.sr_lookback:]

            resistance = []
            for i in range(2, len(recent_highs) - 2):
                if (
                    recent_highs[i] > recent_highs[i - 1]
                    and recent_highs[i] > recent_highs[i - 2]
                    and recent_highs[i] > recent_highs[i + 1]
                    and recent_highs[i] > recent_highs[i + 2]
                ):
                    resistance.append(recent_highs[i])

            support = []
            for i in range(2, len(recent_lows) - 2):
                if (
                    recent_lows[i] < recent_lows[i - 1]
                    and recent_lows[i] < recent_lows[i - 2]
                    and recent_lows[i] < recent_lows[i + 1]
                    and recent_lows[i] < recent_lows[i + 2]
                ):
                    support.append(recent_lows[i])

            resistance = sorted(set(resistance), reverse=True)[:3]
            support = sorted(set(support))[:3]
            return support, resistance

        except Exception:
            return [], []

    def _check_sr_proximity(
        self, current_price: float, support: List[float], resistance: List[float]
    ) -> Tuple[bool, bool]:
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

    # ═══════════════════════════ LEGACY SINGLE-PATH (for base hooks) ═══════════════════════════

    async def _generate_expert_specific_proposal(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Legacy single-instrument proposal for VotingExpertBase.

        Uses the same confluence logic as the per-instrument path, but
        operates on the legacy histories. This keeps base-class behaviour
        alive while the main system consumes per-instrument outputs.
        """
        try:
            if not self._update_price_data(market_data):
                self.log_warning("[TREND] No valid price data - returning flat")
                return {
                    "action": "flat",
                    "signal_strength": 0.0,
                    "reason": "No valid price data",
                }

            prices = list(self.close_history)
            highs = list(self.high_history)
            lows = list(self.low_history)

            if len(prices) < self.slow_period + 10:
                self.log_warning(
                    f"[TREND] Insufficient data: {len(prices)} bars (need {self.slow_period + 10}) - returning flat"
                )
                return {
                    "action": "flat",
                    "signal_strength": 0.0,
                    "reason": f"Insufficient data: {len(prices)} bars",
                }

            current_price = prices[-1]

            # Indicators
            self.fast_ma, self.medium_ma, self.slow_ma, self.ma_alignment = (
                self._calculate_triple_ma(prices)
            )

            ma_spread_fast_medium = (
                (self.fast_ma - self.medium_ma) / self.medium_ma
                if self.medium_ma
                else 0
            )
            ma_spread_medium_slow = (
                (self.medium_ma - self.slow_ma) / self.slow_ma
                if self.slow_ma
                else 0
            )

            self.ma_history.append(
                {
                    "fast": self.fast_ma,
                    "medium": self.medium_ma,
                    "slow": self.slow_ma,
                    "alignment": self.ma_alignment,
                }
            )

            self.adx_value, self.plus_di, self.minus_di = self._calculate_adx(
                highs, lows, prices
            )
            self.adx_history.append(
                {
                    "adx": self.adx_value,
                    "plus_di": self.plus_di,
                    "minus_di": self.minus_di,
                }
            )

            self.sar_value, self.sar_direction = self._calculate_parabolic_sar(
                highs, lows
            )
            self.trend_slope = self._calculate_trend_slope(prices, 20)

            self.support_levels, self.resistance_levels = (
                self._find_support_resistance(highs, lows)
            )
            self.near_support, self.near_resistance = self._check_sr_proximity(
                current_price, self.support_levels, self.resistance_levels
            )

            price_vs_fast = (
                (current_price - self.fast_ma) / self.fast_ma if self.fast_ma else 0
            )
            price_vs_slow = (
                (current_price - self.slow_ma) / self.slow_ma if self.slow_ma else 0
            )

            bullish_score, bearish_score, total_weight = (
                self._calculate_trend_confluence(
                    self.ma_alignment,
                    ma_spread_fast_medium,
                    ma_spread_medium_slow,
                    price_vs_fast,
                    price_vs_slow,
                    self.adx_value,
                    self.plus_di,
                    self.minus_di,
                    self.sar_direction,
                    self.trend_slope,
                    self.near_support,
                    self.near_resistance,
                )
            )

            bullish_confluence = (
                bullish_score / total_weight if total_weight > 0 else 0.0
            )
            bearish_confluence = (
                bearish_score / total_weight if total_weight > 0 else 0.0
            )

            net_trend = bullish_confluence - bearish_confluence
            self.trend_strength = abs(net_trend)

            self.trend_history.append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "trend_strength": self.trend_strength,
                    "bullish": bullish_confluence,
                    "bearish": bearish_confluence,
                    "ma_alignment": self.ma_alignment,
                    "adx": self.adx_value,
                }
            )

            if len(self.trend_history) > 1:
                prev_rec = self.trend_history[-2]
                prev_trend = "neutral"
                if prev_rec["bullish"] > prev_rec["bearish"]:
                    prev_trend = "uptrend"
                elif prev_rec["bearish"] > prev_rec["bullish"]:
                    prev_trend = "downtrend"

                current = "neutral"
                if bullish_confluence > bearish_confluence:
                    current = "uptrend"
                elif bearish_confluence > bullish_confluence:
                    current = "downtrend"

                if current == prev_trend and current != "neutral":
                    self.trend_duration += 1
                else:
                    self.trend_duration = 1

            # Smart gating via shared helper
            action, base_conf, signal_strength, trend_label = (
                self._determine_trend_action(
                    net_trend,
                    bullish_confluence,
                    bearish_confluence,
                    self.adx_value,
                    self.trend_duration,
                )
            )
            self.current_trend = trend_label

            reason = (
                f"{trend_label} | ADX={self.adx_value:.1f} | "
                f"bull={bullish_confluence:.2f} bear={bearish_confluence:.2f}"
            )

            # Track signal performance counter
            self.trend_performance.setdefault(action, {}).setdefault(
                "signals", 0
            )
            self.trend_performance[action]["signals"] += 1

            proposal = {
                "action": action,
                "signal_strength": float(signal_strength),
                "position_size": float(
                    min(self.max_signal_strength, signal_strength * 0.6)
                ),
                "reason": reason,
                "current_trend": self.current_trend,
                "trend_strength": float(self.trend_strength),
                "trend_slope": float(self.trend_slope),
                "trend_duration": self.trend_duration,
                "bullish_confluence": float(bullish_confluence),
                "bearish_confluence": float(bearish_confluence),
                "fast_ma": float(self.fast_ma),
                "medium_ma": float(self.medium_ma),
                "slow_ma": float(self.slow_ma),
                "ma_alignment": self.ma_alignment,
                "adx": float(self.adx_value),
                "plus_di": float(self.plus_di),
                "minus_di": float(self.minus_di),
                "sar_direction": self.sar_direction,
                "near_support": self.near_support,
                "near_resistance": self.near_resistance,
                "duration": "medium"
                if self.trend_duration >= 3
                else "short",
                "conviction": "high"
                if signal_strength > 0.7
                else "medium"
                if signal_strength > 0.4
                else "low",
            }

            return proposal

        except Exception as e:
            self.log_error(f"[TREND] Proposal generation failed: {e}")
            return {
                "action": "flat",
                "signal_strength": 0.0,
                "reason": f"Analysis error: {str(e)[:100]}",
            }

    async def _calculate_expert_specific_confidence(
        self, proposal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> float:
        """
        Legacy confidence function for base class.

        Kept but less critical now that per-instrument path provides its own
        calibrated confidences.
        """
        try:
            action = proposal.get("action", "flat")
            signal_strength = float(proposal.get("signal_strength", 0.0))

            base = 0.3 + signal_strength * 0.4

            if action == "long":
                confluence = float(proposal.get("bullish_confluence", 0.0))
            elif action == "short":
                confluence = float(proposal.get("bearish_confluence", 0.0))
            else:
                confluence = 0.0

            if confluence >= self.strong_signal_confluence:
                base *= 1.25
            elif confluence >= self.min_confluence_score:
                base *= 1.1

            adx = float(proposal.get("adx", 20.0))
            if adx >= self.adx_strong_threshold:
                base *= 1.2
            elif adx >= self.adx_trending_threshold:
                base *= 1.1

            if proposal.get("ma_alignment") != 0:
                base *= 1.1

            duration = int(proposal.get("trend_duration", 0))
            if duration >= 10:
                base *= 1.15
            elif duration >= 5:
                base *= 1.08

            perf = self.trend_performance.get(action, {})
            if perf.get("signals", 0) > 20:
                success_rate = perf.get("success", 0) / perf["signals"]
                base *= 0.7 + success_rate * 0.6

            # Mode-aware clipping
            conf_floor = CONFIDENCE_THRESHOLD_F()
            high_conf = HIGH_CONFIDENCE_THRESHOLD_F()
            base = max(conf_floor * 0.5, base)
            base = min(high_conf, base)

            return float(max(0.15, base))

        except Exception:
            return 0.4

    # ═══════════════════════════ MAIN PER-INSTRUMENT PROCESS ═══════════════════════════

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Process market data to determine trend signals PER INSTRUMENT.

        Each instrument receives its own InstrumentProposal with calibrated
        (action, confidence, magnitude). Weak/noisy cases are down-weighted
        so the committee/arbiter can filter them cleanly.

        Includes:
        - Circuit breaker integration
        - Degraded mode on exceptions
        - Performance tracking
        """
        name = self.__class__.__name__
        start = time.time()

        try:
            # Circuit breaker check
            if self._check_circuit_breaker():
                return self._degraded_output("circuit_breaker_open")

            # Pull data from InfoBus
            market_data = self.smart_bus.get("market_data", name, default={})
            features = self.smart_bus.get("features", name, default={})

            if not market_data and not features:
                self.log_debug(
                    f"[TREND][BUS] empty fetch: market_data_keys={list(market_data.keys()) if isinstance(market_data, dict) else market_data}, "
                    f"features_keys={list(features.keys()) if isinstance(features, dict) else features}"
                )
                self.log_warning("[TREND] No market data or features available")
                output = self._neutral_output("No market data available")
            else:
                per_instrument_vote = PerInstrumentVote(member=name)
                per_instrument_analysis: Dict[str, Dict[str, Any]] = {}

                for inst in self.instruments:
                    inst_norm = normalize_instrument(inst)

                    inst_market = self._extract_instrument_data(market_data, inst)
                    inst_features = self._extract_instrument_data(features, inst)

                    close_prices = self._extract_prices(
                        inst_market, inst_features, "close", inst
                    )
                    high_prices = self._extract_prices(
                        inst_market, inst_features, "high", inst
                    )
                    low_prices = self._extract_prices(
                        inst_market, inst_features, "low", inst
                    )

                    state = self.instrument_state.get(inst_norm, {})
                    if not state:
                        state = self.instrument_state.setdefault(
                            inst_norm,
                            {
                                "price_history": deque(maxlen=200),
                                "high_history": deque(maxlen=200),
                                "low_history": deque(maxlen=200),
                                "trend_history": deque(maxlen=100),
                                "trend_duration": 0,
                            },
                        )

                    if len(close_prices) > 0:
                        state["price_history"] = deque(
                            close_prices[-200:], maxlen=200
                        )
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

                    prices = list(state.get("price_history", []))

                    if len(prices) < self.slow_period + 10:
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
                            f"[TREND][DATA] {inst_norm} insufficient: price_len={len(prices)}, slow_period={self.slow_period}, tf_meta={tf_meta}"
                        )
                        self.log_debug(
                            f"[TREND] Insufficient data for {inst}: {len(prices)} bars"
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
                            "current_trend": "unknown",
                            "trend_strength": 0.0,
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
                            f"[TREND][DATA] {inst_norm}: price_len={len(prices)}, "
                            f"latest={prices[-1] if prices else None}, tf_meta={tf_meta}"
                        )
                        self._debug_last_log[inst_norm] = now_ts

                    current_price = float(prices[-1])
                    highs = list(state.get("high_history", prices))
                    lows = list(state.get("low_history", prices))

                    fast_ma, medium_ma, slow_ma, ma_alignment = self._calculate_triple_ma(
                        prices
                    )
                    adx_value, plus_di, minus_di = self._calculate_adx(
                        highs, lows, prices
                    )
                    sar_value, sar_direction = self._calculate_parabolic_sar(
                        highs, lows
                    )
                    trend_slope = self._calculate_trend_slope(prices, 20)
                    support_levels, resistance_levels = self._find_support_resistance(
                        highs, lows
                    )
                    near_support, near_resistance = self._check_sr_proximity(
                        current_price, support_levels, resistance_levels
                    )

                    ma_spread_fast_medium = (
                        (fast_ma - medium_ma) / medium_ma if medium_ma else 0.0
                    )
                    ma_spread_medium_slow = (
                        (medium_ma - slow_ma) / slow_ma if slow_ma else 0.0
                    )
                    price_vs_fast = (
                        (current_price - fast_ma) / fast_ma if fast_ma else 0.0
                    )
                    price_vs_slow = (
                        (current_price - slow_ma) / slow_ma if slow_ma else 0.0
                    )

                    bullish_score, bearish_score, total_weight = (
                        self._calculate_trend_confluence(
                            ma_alignment,
                            ma_spread_fast_medium,
                            ma_spread_medium_slow,
                            price_vs_fast,
                            price_vs_slow,
                            adx_value,
                            plus_di,
                            minus_di,
                            sar_direction,
                            trend_slope,
                            near_support,
                            near_resistance,
                        )
                    )

                    bullish_confluence = (
                        bullish_score / total_weight if total_weight > 0 else 0.0
                    )
                    bearish_confluence = (
                        bearish_score / total_weight if total_weight > 0 else 0.0
                    )
                    net_trend = bullish_confluence - bearish_confluence

                    trend_history = state.setdefault(
                        "trend_history", deque(maxlen=100)
                    )
                    trend_history.append(
                        {
                            "bullish": bullish_confluence,
                            "bearish": bearish_confluence,
                            "timestamp": datetime.datetime.now().isoformat(),
                        }
                    )

                    trend_duration = int(state.get("trend_duration", 0))
                    if len(trend_history) > 1:
                        prev = trend_history[-2]
                        prev_dir = (
                            "up"
                            if prev["bullish"] > prev["bearish"]
                            else "down"
                            if prev["bearish"] > prev["bullish"]
                            else "flat"
                        )
                        curr_dir = (
                            "up"
                            if bullish_confluence > bearish_confluence
                            else "down"
                            if bearish_confluence > bullish_confluence
                            else "flat"
                        )
                        if curr_dir == prev_dir and curr_dir != "flat":
                            trend_duration += 1
                        else:
                            trend_duration = 1
                        state["trend_duration"] = trend_duration

                    action, confidence, signal_strength, current_trend = (
                        self._determine_trend_action(
                            net_trend,
                            bullish_confluence,
                            bearish_confluence,
                            adx_value,
                            trend_duration,
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
                        mtf_analysis = self._analyze_multi_timeframe_trend(inst)
                        
                        if mtf_analysis.get("available"):
                            dominant = mtf_analysis.get("dominant_direction", "neutral")
                            alignment = mtf_analysis.get("alignment_score", 0.5)
                            
                            # Check if M15 signal direction matches context TF direction
                            signal_is_bullish = action == "long"
                            mtf_is_bullish = dominant == "bullish"
                            mtf_is_bearish = dominant == "bearish"
                            
                            if signal_is_bullish and mtf_is_bullish:
                                # M15 LONG confirmed by context TFs - boost confidence
                                mtf_adjustment = self.mtf_agreement_bonus * alignment
                                mtf_info = f"Context TFs AGREE ↑ (align={alignment:.2f})"
                            elif not signal_is_bullish and mtf_is_bearish:
                                # M15 SHORT confirmed by context TFs - boost confidence
                                mtf_adjustment = self.mtf_agreement_bonus * alignment
                                mtf_info = f"Context TFs AGREE ↓ (align={alignment:.2f})"
                            elif (signal_is_bullish and mtf_is_bearish) or (not signal_is_bullish and mtf_is_bullish):
                                # M15 signal CONTRADICTS context TFs - penalize confidence ONLY
                                # M15-PRIMARY: Never override direction, only reduce confidence
                                mtf_adjustment = -self.mtf_disagreement_penalty * alignment
                                mtf_info = f"Context TFs DISAGREE (align={alignment:.2f}, conf penalty applied)"
                                # NOTE: We do NOT override action to flat - M15 is the decision maker
                            else:
                                # Neutral context TFs - slight reduction
                                mtf_adjustment = -0.05
                                mtf_info = f"Context TFs neutral (no confirmation)"
                            
                            # Apply MTF adjustment to confidence (NEVER change action)
                            confidence = max(0.1, min(0.95, confidence + mtf_adjustment))
                            
                            # Store MTF info in analysis
                            per_instrument_analysis[inst_norm] = per_instrument_analysis.get(inst_norm, {})
                            per_instrument_analysis[inst_norm]["mtf_analysis"] = mtf_analysis
                            per_instrument_analysis[inst_norm]["mtf_adjustment"] = mtf_adjustment

                    thesis = (
                        f"{inst_norm}: {current_trend} → {action} "
                        f"(ADX={adx_value:.1f}, conf={confidence:.2f}) {mtf_info}"
                    )

                    per_instrument_vote.set_proposal(
                        InstrumentProposal(
                            instrument=inst_norm,
                            action=action,
                            confidence=confidence,
                            magnitude=signal_strength,
                            rationale=thesis,
                        )
                    )

                    per_instrument_analysis[inst_norm] = {
                        "current_trend": current_trend,
                        "trend_strength": abs(net_trend),
                        "trend_slope": trend_slope,
                        "trend_duration": trend_duration,
                        "bullish_confluence": bullish_confluence,
                        "bearish_confluence": bearish_confluence,
                        "ma_alignment": ma_alignment,
                        "adx": adx_value,
                        "plus_di": plus_di,
                        "minus_di": minus_di,
                        "sar_direction": sar_direction,
                        "near_support": near_support,
                        "near_resistance": near_resistance,
                        "action": action,
                        "confidence": confidence,
                    }

                    self.log_debug(
                        f"[TREND] {inst_norm}: action={action}, conf={confidence:.2f}, "
                        f"trend={current_trend}, net={net_trend:.3f}"
                    )

                # Global summary (backward compat)
                if per_instrument_vote.proposals:
                    best_proposal = max(
                        per_instrument_vote.proposals.values(),
                        key=lambda p: p.confidence,
                    )
                    global_action = best_proposal.action
                    global_confidence = best_proposal.confidence
                    global_thesis = best_proposal.rationale
                else:
                    global_action = "flat"
                    global_confidence = 0.1
                    global_thesis = "No instrument data available"

                proposals_dict = {
                    inst: {
                        "action": prop.action,
                        "confidence": prop.confidence,
                        "magnitude": prop.magnitude,
                        "rationale": prop.rationale,
                    }
                    for inst, prop in per_instrument_vote.proposals.items()
                }

                proposal = {
                    "action": global_action,
                    "signal_strength": global_confidence,
                    "reason": global_thesis,
                    "proposals": proposals_dict,
                }

                # Publish to SmartInfoBus
                try:
                    self.smart_bus.set(
                        "TrendExpert_voting_proposal",
                        proposal,
                        module=name,
                        thesis=global_thesis,
                    )
                    self.smart_bus.set(
                        "TrendExpert_confidence",
                        global_confidence,
                        module=name,
                        thesis=f"Confidence: {global_confidence:.1%}",
                    )
                    self.smart_bus.set(
                        "trend_voting_proposal",
                        proposal,
                        module=name,
                        thesis=global_thesis,
                    )
                    self.smart_bus.set(
                        "trend_confidence",
                        global_confidence,
                        module=name,
                        thesis=f"Trend confidence: {global_confidence:.1%}",
                    )

                    per_inst_votes_dict = {
                        inst: prop.to_dict()
                        for inst, prop in per_instrument_vote.proposals.items()
                    }
                    self.smart_bus.set(
                        "TrendExpert_per_instrument_votes",
                        per_inst_votes_dict,
                        module=name,
                        thesis=(
                            f"Per-instrument trend votes: "
                            f"{list(per_inst_votes_dict.keys())}"
                        ),
                    )
                except Exception as e:
                    self.log_warning(f"[TREND] Failed to publish to bus: {e}")

                trend_analysis = {
                    "current_trend": self.current_trend,
                    "trend_strength": self.trend_strength,
                    "per_instrument": per_instrument_analysis,
                    "performance": dict(self.trend_performance),
                }

                output = {
                    "TrendExpert_voting_proposal": proposal,
                    "TrendExpert_confidence": global_confidence,
                    "TrendExpert_per_instrument_votes": {
                        inst: prop.to_dict()
                        for inst, prop in per_instrument_vote.proposals.items()
                    },
                    "per_instrument_votes": per_instrument_vote,
                    "trend_voting_proposal": proposal,
                    "trend_confidence": global_confidence,
                    "trend_analysis": trend_analysis,
                    "voting_proposal": proposal,
                    "confidence": global_confidence,
                    "_thesis": global_thesis,
                }

            # Success path: record perf + reset error count
            elapsed_ms = (time.time() - start) * 1000.0
            try:
                self.performance_tracker.record_metric(
                    name, "process", elapsed_ms, True
                )
            except Exception:
                # Do not let perf tracking kill the expert
                pass
            self._record_success()
            return output

        except Exception as e:
            # Error path: record + degraded output
            self._record_error(e)
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(
                    e, f"{name}_process"
                )
                msg = str(error_context)
            else:
                msg = str(e)

            self.log_error(f"[{name}] Process error: {msg}")

            elapsed_ms = (time.time() - start) * 1000.0
            try:
                self.performance_tracker.record_metric(
                    name, "process", elapsed_ms, False
                )
            except Exception:
                pass

            return self._degraded_output(msg)

    # ═══════════════════════════ HELPERS ═══════════════════════════

    def _extract_instrument_data(self, data: Dict, instrument: str) -> Dict:
        """Extract data for a specific instrument from nested market data."""
        if not isinstance(data, dict):
            return {}

        inst_norm = normalize_instrument(instrument)

        for key in [instrument, inst_norm, instrument.upper(), instrument.lower()]:
            if key in data:
                return data[key] if isinstance(data[key], dict) else data

        for sep in ["_", "/", "-", ""]:
            for pair in [f"EUR{sep}USD", f"XAU{sep}USD"]:
                norm_pair = normalize_instrument(pair)
                if norm_pair == inst_norm and pair in data:
                    return data[pair] if isinstance(data[pair], dict) else data

        return data

    def _extract_prices(
        self, market_data: Dict, features: Dict, price_type: str, instrument: str = ""
    ) -> np.ndarray:
        """Extract price array from market data or features for a specific instrument."""
        inst_norm = normalize_instrument(instrument) if instrument else ""
        inst_variations = [
            instrument,
            inst_norm,
            f"{inst_norm[:3]}_{inst_norm[3:]}" if len(inst_norm) >= 6 else inst_norm,
        ]

        if isinstance(market_data, dict):
            if price_type in market_data:
                data = market_data[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)

            for tf in ["M15", "H1", "H4", "D1"]:
                if tf in market_data and isinstance(market_data[tf], dict):
                    if price_type in market_data[tf]:
                        data = market_data[tf][price_type]
                        if isinstance(data, (list, np.ndarray)):
                            return np.array(data, dtype=float)

        if isinstance(features, dict):
            if price_type in features:
                data = features[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)

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
                    # M15 is primary, H1/H4/D1 are context (ordered by granularity)
                    for tf in ["M15", "H1", "H4", "D1"]:
                        tf_rec = sym_block.get(tf)
                        if isinstance(tf_rec, dict):
                            seq = tf_rec.get(price_type)
                            if isinstance(seq, (list, np.ndarray)) and len(seq) > 0:
                                return np.array(seq, dtype=float)

        return np.array([])

    def _calculate_trend_confluence(
        self,
        ma_alignment: int,
        ma_spread_fm: float,
        ma_spread_ms: float,
        price_vs_fast: float,
        price_vs_slow: float,
        adx_value: float,
        plus_di: float,
        minus_di: float,
        sar_direction: int,
        trend_slope: float,
        near_support: bool,
        near_resistance: bool,
    ) -> Tuple[float, float, float]:
        """Calculate bullish and bearish trend confluence scores."""
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0

        # MA Alignment
        ma_weight = 2.5
        total_weight += ma_weight
        if ma_alignment == 1:
            bullish_score += ma_weight
        elif ma_alignment == -1:
            bearish_score += ma_weight

        # MA Spread
        spread_weight = 1.5
        total_weight += spread_weight
        if ma_spread_fm > self.trend_threshold and ma_spread_ms > self.trend_threshold:
            bullish_score += spread_weight * min(
                1.0, (ma_spread_fm + ma_spread_ms) * 50
            )
        elif ma_spread_fm < -self.trend_threshold and ma_spread_ms < -self.trend_threshold:
            bearish_score += spread_weight * min(
                1.0, abs(ma_spread_fm + ma_spread_ms) * 50
            )

        # Price position
        pos_weight = 1.5
        total_weight += pos_weight
        if price_vs_fast > 0 and price_vs_slow > 0:
            bullish_score += pos_weight * min(
                1.0, (price_vs_fast + price_vs_slow) * 20
            )
        elif price_vs_fast < 0 and price_vs_slow < 0:
            bearish_score += pos_weight * min(
                1.0, abs(price_vs_fast + price_vs_slow) * 20
            )

        # ADX
        adx_weight = 2.0
        total_weight += adx_weight
        if adx_value >= self.adx_trending_threshold:
            trend_strength_factor = min(1.0, adx_value / self.adx_strong_threshold)
            if plus_di > minus_di:
                bullish_score += adx_weight * trend_strength_factor
            else:
                bearish_score += adx_weight * trend_strength_factor

        # SAR
        sar_weight = 1.2
        total_weight += sar_weight
        if sar_direction == 1:
            bullish_score += sar_weight
        elif sar_direction == -1:
            bearish_score += sar_weight

        # Slope
        slope_weight = 1.8
        total_weight += slope_weight
        if trend_slope > self.trend_threshold:
            bullish_score += slope_weight * min(
                1.0, trend_slope / (self.trend_threshold * 3)
            )
        elif trend_slope < -self.trend_threshold:
            bearish_score += slope_weight * min(
                1.0, abs(trend_slope) / (self.trend_threshold * 3)
            )

        # S/R awareness
        sr_weight = 1.0
        total_weight += sr_weight
        if near_support and ma_alignment >= 0:
            bullish_score += sr_weight * 0.8
        elif near_resistance and ma_alignment <= 0:
            bearish_score += sr_weight * 0.8

        return bullish_score, bearish_score, total_weight

    def _analyze_multi_timeframe_trend(
        self,
        instrument: str,
    ) -> Dict[str, Any]:
        """
        Analyze trend direction across multiple timeframes (H1, H4, D1).
        
        Returns trend direction and strength for each timeframe, plus
        an alignment score indicating how well the timeframes agree.
        
        This is critical for quality signals:
        - If H1 says LONG but H4/D1 say SHORT, don't trust the H1 signal
        - If all timeframes agree, boost confidence significantly
        """
        name = self.__class__.__name__
        inst_norm = normalize_instrument(instrument)
        
        mtf_trends: Dict[str, Dict[str, Any]] = {}
        
        try:
            historical = self.smart_bus.get("historical_prices", name, default=None)
        except Exception:
            historical = None
        
        if not isinstance(historical, dict):
            return {
                "available": False,
                "trends": {},
                "alignment_score": 0.5,
                "dominant_direction": "neutral",
            }
        
        # Find the instrument in historical data
        matched_symbol = None
        for sym in historical.keys():
            if normalize_instrument(sym) == inst_norm:
                matched_symbol = sym
                break
        
        if not matched_symbol or matched_symbol not in historical:
            return {
                "available": False,
                "trends": {},
                "alignment_score": 0.5,
                "dominant_direction": "neutral",
            }
        
        sym_block = historical[matched_symbol]
        if not isinstance(sym_block, dict):
            return {
                "available": False,
                "trends": {},
                "alignment_score": 0.5,
                "dominant_direction": "neutral",
            }
        
        # Analyze each timeframe
        for tf in self.mtf_timeframes:
            tf_data = sym_block.get(tf)
            if not isinstance(tf_data, dict):
                continue
            
            close_arr = tf_data.get("close")
            if not isinstance(close_arr, (list, np.ndarray)) or len(close_arr) < self.slow_period + 5:
                continue
            
            prices = np.array(close_arr, dtype=float)
            
            # Calculate simple trend indicators for this timeframe
            # Use EMA-based trend detection
            fast_ma = self._ema(prices, min(self.fast_period, len(prices) - 1))
            slow_ma = self._ema(prices, min(self.slow_period, len(prices) - 1))
            
            if fast_ma <= 0 or slow_ma <= 0:
                continue
            
            # Trend direction: fast MA vs slow MA
            ma_spread = (fast_ma - slow_ma) / slow_ma
            current_price = float(prices[-1])
            price_vs_slow = (current_price - slow_ma) / slow_ma if slow_ma > 0 else 0
            
            # Simple slope (last 10 bars or available)
            slope_period = min(10, len(prices) - 1)
            if slope_period > 1:
                slope = (prices[-1] - prices[-slope_period]) / prices[-slope_period]
            else:
                slope = 0.0
            
            # Determine direction: bullish, bearish, or neutral
            bullish_signals = 0
            bearish_signals = 0
            
            if ma_spread > 0.001:  # Fast above slow
                bullish_signals += 1
            elif ma_spread < -0.001:
                bearish_signals += 1
            
            if price_vs_slow > 0.002:  # Price above slow MA
                bullish_signals += 1
            elif price_vs_slow < -0.002:
                bearish_signals += 1
            
            if slope > 0.001:
                bullish_signals += 1
            elif slope < -0.001:
                bearish_signals += 1
            
            # Determine direction
            if bullish_signals >= 2 and bullish_signals > bearish_signals:
                direction = "bullish"
                strength = min(1.0, abs(ma_spread) * 20 + abs(slope) * 10)
            elif bearish_signals >= 2 and bearish_signals > bullish_signals:
                direction = "bearish"
                strength = min(1.0, abs(ma_spread) * 20 + abs(slope) * 10)
            else:
                direction = "neutral"
                strength = 0.2
            
            mtf_trends[tf] = {
                "direction": direction,
                "strength": strength,
                "ma_spread": ma_spread,
                "slope": slope,
                "price_vs_slow": price_vs_slow,
            }
        
        if not mtf_trends:
            return {
                "available": False,
                "trends": {},
                "alignment_score": 0.5,
                "dominant_direction": "neutral",
            }
        
        # Calculate alignment score
        directions = [t["direction"] for t in mtf_trends.values()]
        bullish_count = directions.count("bullish")
        bearish_count = directions.count("bearish")
        total_tf = len(directions)
        
        # Weighted alignment calculation
        weighted_bullish = sum(
            self.mtf_weights.get(tf, 0.33) 
            for tf, trend in mtf_trends.items() 
            if trend["direction"] == "bullish"
        )
        weighted_bearish = sum(
            self.mtf_weights.get(tf, 0.33) 
            for tf, trend in mtf_trends.items() 
            if trend["direction"] == "bearish"
        )
        
        # Alignment score: how much the timeframes agree
        # 1.0 = all agree, 0.0 = completely mixed
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
        
        # Log MTF analysis
        self.log_debug(
            f"[TREND MTF] {inst_norm}: " +
            ", ".join(f"{tf}={t['direction']}" for tf, t in mtf_trends.items()) +
            f" | alignment={alignment_score:.2f}, dominant={dominant}"
        )
        
        return {
            "available": True,
            "trends": mtf_trends,
            "alignment_score": alignment_score,
            "dominant_direction": dominant,
            "weighted_bullish": weighted_bullish,
            "weighted_bearish": weighted_bearish,
        }
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA for the given period."""
        if len(prices) < period or period < 1:
            return float(prices[-1]) if len(prices) > 0 else 0.0
        
        multiplier = 2.0 / (period + 1)
        ema = float(prices[0])
        for price in prices[1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    def _determine_trend_action(
        self,
        net_trend: float,
        bullish_confluence: float,
        bearish_confluence: float,
        adx_value: float,
        trend_duration: int,
    ) -> Tuple[str, float, float, str]:
        """
        Determine action, confidence, signal strength and trend label.

        Smart / stable filtering:
        - Requires confluence + ADX + net trend for strong signals.
        - Produces weak directional bias with low magnitude so the
          committee can treat it as neutral when thresholds are not met.
        """
        min_strength = MIN_SIGNAL_STRENGTH_F()
        conf_floor = CONFIDENCE_THRESHOLD_F()
        high_conf = HIGH_CONFIDENCE_THRESHOLD_F()

        max_confluence = max(bullish_confluence, bearish_confluence)
        abs_net = abs(net_trend)

        # Very weak or noisy → neutral
        if abs_net < self.weak_net_trend or max_confluence < self.min_confluence_score * 0.5:
            return "flat", 0.2, 0.05, "neutral"

        bias_long = net_trend > 0
        adx_strong_gate = adx_value >= self.adx_trending_threshold
        adx_weak_gate = adx_value >= self.adx_trending_threshold * 0.6

        # ── Strong directional signal ─────────────────────────────────
        if (
            abs_net >= self.min_net_trend
            and adx_strong_gate
            and max_confluence >= self.min_confluence_score
        ):
            action = "long" if bias_long else "short"
            current_trend = "uptrend" if bias_long else "downtrend"

            base_strength = max_confluence
            signal_strength = max(
                min_strength,
                min(1.0, base_strength * 1.2),
            )

            if max_confluence >= self.strong_signal_confluence:
                signal_strength = min(1.0, signal_strength * 1.1)
            if adx_value >= self.adx_strong_threshold:
                signal_strength = min(1.0, signal_strength * 1.05)
            if trend_duration >= 5:
                signal_strength = min(1.0, signal_strength * 1.05)

            base_conf = 0.35 + signal_strength * 0.5
            if max_confluence >= self.strong_signal_confluence:
                base_conf *= 1.1
            if trend_duration >= 10:
                base_conf *= 1.05

            base_conf = max(conf_floor, base_conf)
            base_conf = min(high_conf, base_conf)

            return action, float(base_conf), float(signal_strength), current_trend

        # ── Moderate directional bias (mode-aware, can pass in TRAINING) ──
        if (
            abs_net >= self.weak_net_trend
            and adx_weak_gate
            and max_confluence >= self.min_confluence_score
        ):
            action = "long" if bias_long else "short"
            current_trend = "weak_uptrend" if bias_long else "weak_downtrend"

            signal_strength = min(
                min_strength * 1.1,
                max_confluence * 0.8,
            )
            signal_strength = max(min_strength * 0.5, signal_strength)

            base_conf = 0.28 + signal_strength * 0.4
            base_conf = max(conf_floor * 0.8, base_conf)
            base_conf = min(high_conf * 0.9, base_conf)

            return action, float(base_conf), float(signal_strength), current_trend

        # ── Weak directional hint (below arbiter thresholds) ──────────────
        action = "long" if bias_long else "short"
        current_trend = "weak_uptrend" if bias_long else "weak_downtrend"

        signal_strength = max(0.02, max_confluence * 0.3)
        # Intentionally below min_strength so committee treats as neutral
        signal_strength = min(min_strength * 0.6, signal_strength)

        base_conf = max(0.15, conf_floor * 0.5)

        return action, float(base_conf), float(signal_strength), current_trend

    def _neutral_output(self, reason: str) -> Dict[str, Any]:
        """Generate neutral output with explanation."""
        thesis = f"Trend flat: {reason}"
        name = self.__class__.__name__

        proposal = {
            "action": "flat",
            "signal_strength": 0.1,
            "reason": thesis,
            "proposals": {},
        }

        try:
            self.smart_bus.set(
                "TrendExpert_voting_proposal", proposal, module=name, thesis=thesis
            )
            self.smart_bus.set(
                "TrendExpert_confidence", 0.1, module=name, thesis="Confidence: 10%"
            )
            self.smart_bus.set(
                "trend_voting_proposal", proposal, module=name, thesis=thesis
            )
            self.smart_bus.set(
                "trend_confidence",
                0.1,
                module=name,
                thesis="Trend confidence: 10%",
            )
        except Exception:
            pass

        return {
            "TrendExpert_voting_proposal": proposal,
            "TrendExpert_confidence": 0.1,
            "TrendExpert_per_instrument_votes": {},
            "trend_voting_proposal": proposal,
            "trend_confidence": 0.1,
            "trend_analysis": {"current_trend": "unknown", "per_instrument": {}},
            "voting_proposal": proposal,
            "confidence": 0.1,
            "_thesis": thesis,
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE PERSISTENCE - Save/Load module state
    # ═══════════════════════════════════════════════════════════════════

    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.
        
        Saves per-instrument computed state:
        - MA values and history
        - ADX values and history
        - Trend state and history
        - Support/resistance levels
        """
        instrument_states = {}
        for inst, state in self.instrument_state.items():
            instrument_states[inst] = {
                "fast_ma": float(state.get("fast_ma", 0.0)),
                "medium_ma": float(state.get("medium_ma", 0.0)),
                "slow_ma": float(state.get("slow_ma", 0.0)),
                "adx_value": float(state.get("adx_value", 0.0)),
                "plus_di": float(state.get("plus_di", 0.0)),
                "minus_di": float(state.get("minus_di", 0.0)),
                "sar_value": float(state.get("sar_value", 0.0)),
                "sar_direction": int(state.get("sar_direction", 0)),
                "current_trend": state.get("current_trend", "neutral"),
                "trend_strength": float(state.get("trend_strength", 0.0)),
                "trend_duration": int(state.get("trend_duration", 0)),
                "ma_alignment": int(state.get("ma_alignment", 0)),
                "support_levels": list(state.get("support_levels", []))[-5:],
                "resistance_levels": list(state.get("resistance_levels", []))[-5:],
                "trend_history": list(state.get("trend_history", []))[-20:],
            }
        
        return {
            "instrument_state": instrument_states,
            "fast_ma": float(self.fast_ma),
            "medium_ma": float(self.medium_ma),
            "slow_ma": float(self.slow_ma),
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
                    "fast_ma": float(saved.get("fast_ma", 0.0)),
                    "medium_ma": float(saved.get("medium_ma", 0.0)),
                    "slow_ma": float(saved.get("slow_ma", 0.0)),
                    "adx_value": float(saved.get("adx_value", 0.0)),
                    "plus_di": float(saved.get("plus_di", 0.0)),
                    "minus_di": float(saved.get("minus_di", 0.0)),
                    "sar_value": float(saved.get("sar_value", 0.0)),
                    "sar_direction": int(saved.get("sar_direction", 0)),
                    "current_trend": saved.get("current_trend", "neutral"),
                    "trend_strength": float(saved.get("trend_strength", 0.0)),
                    "trend_duration": int(saved.get("trend_duration", 0)),
                    "ma_alignment": int(saved.get("ma_alignment", 0)),
                    "support_levels": list(saved.get("support_levels", [])),
                    "resistance_levels": list(saved.get("resistance_levels", [])),
                })
                # Restore trend history as deque
                hist = saved.get("trend_history", [])
                self.instrument_state[inst]["trend_history"] = deque(hist, maxlen=100)
        
        # Restore legacy single-instrument state
        self.fast_ma = float(state.get("fast_ma", 0.0))
        self.medium_ma = float(state.get("medium_ma", 0.0))
        self.slow_ma = float(state.get("slow_ma", 0.0))
        
        self.log_info(
            f"📂 TrendExpert state restored | "
            f"instruments={len(inst_states)}"
        )
