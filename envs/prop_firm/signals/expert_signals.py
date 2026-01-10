# envs/prop_firm/signals/expert_signals.py
# pyright: reportAttributeAccessIssue=false
"""
Expert signal computation mixin for PropFirmTradingEnv.

UPGRADED (Jan 2026+):
- TrendExpert: Triple EMA (8/21/90), ADX (+DI/-DI) Wilder, Parabolic SAR, regression slope,
               and institutional market-structure integration (S/R, BOS, liquidity, OB)
- MomentumExpert: Multi-period ROC, RSI + pivot divergence, MACD histogram, Stochastic %K/%D,
                  acceleration + volume confirmation
- ThemeExpert: ATR volatility percentile (efficient), breadth score, risk regime, composite scoring

Key architectural rule:
- Market structure is computed ONLY via MarketStructureMixin to avoid drift/duplication.

Expected attributes from PropFirmTradingEnv:
- config: PropFirmConfig
- data: Dict[str, Dict[str, pd.DataFrame]]
- current_step: int
- _get_ohlcv(instrument: str, lookback: int, timeframe: Optional[str] = None) -> Dict[str, np.ndarray]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from envs.prop_firm.signals.market_structure import MarketStructureMixin

if TYPE_CHECKING:
    from envs.core.env_types import PropFirmConfig
    import pandas as pd


class ExpertSignalsMixin(MarketStructureMixin):
    """Sophisticated expert signal computation with training/live parity."""

    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════

    # Triple EMA periods
    _FAST_MA_PERIOD = 8
    _MEDIUM_MA_PERIOD = 21
    _SLOW_MA_PERIOD = 90

    # ADX configuration
    _ADX_PERIOD = 14
    _ADX_TRENDING_THRESHOLD = 25.0
    _ADX_STRONG_THRESHOLD = 40.0

    # SAR configuration
    _SAR_AF_START = 0.02
    _SAR_AF_STEP = 0.02
    _SAR_AF_MAX = 0.2
    _SAR_LOOKBACK = 140  # bound compute cost

    # RSI configuration
    _RSI_PERIOD = 14
    _RSI_OVERBOUGHT = 70.0
    _RSI_OVERSOLD = 30.0
    _RSI_EXTREME_OVERBOUGHT = 80.0
    _RSI_EXTREME_OVERSOLD = 20.0

    # MACD configuration
    _MACD_FAST = 12
    _MACD_SLOW = 26
    _MACD_SIGNAL = 9

    # Stochastic configuration
    _STOCH_K_PERIOD = 14
    _STOCH_D_PERIOD = 3
    _STOCH_OVERBOUGHT = 80.0
    _STOCH_OVERSOLD = 20.0

    # ROC configuration
    _ROC_PERIODS = [5, 10, 20, 50]
    _ROC_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

    # Divergence configuration (pivot-based)
    _DIVERGENCE_LOOKBACK = 80
    _DIVERGENCE_MIN_RSI_DELTA = 5.0
    _DIVERGENCE_SIGNIFICANCE = 0.02
    _PIVOT_LEFT = 3
    _PIVOT_RIGHT = 3

    # Theme / volatility configuration
    _ATR_PERIOD = 20
    _VOL_LOOKBACK = 80
    _VOL_THRESHOLD_LOW = 0.30
    _VOL_THRESHOLD_HIGH = 0.70

    # Confluence thresholds
    _MIN_NET_TREND = 0.02
    _MIN_CONFLUENCE = 0.15
    _STRONG_CONFLUENCE = 0.65

    # Cache sizing
    _MIN_LOOKBACK = 260  # enough for slow EMA + MACD/ATR windows

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def _prepare_expert_signals(self, instrument: str) -> Dict[str, Any]:
        """
        Prepare expert signals with per-step, per-instrument caching.

        Correctness rule:
        - Cache MUST reset when current_step changes (prevents stale signals).
        - Cache MUST be keyed by instrument (prevents cross-instrument bleed).
        """
        step = int(getattr(self, "current_step", -1))

        # Reset cache per step (critical)
        if getattr(self, "_expert_cache_step", None) != step:
            self._step_expert_signals_cache = {}
            self._expert_cache_step = step

        cache = getattr(self, "_step_expert_signals_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._step_expert_signals_cache = cache

        if instrument in cache:
            return cache[instrument]

        o = self._get_ohlcv(instrument, lookback=max(self._MIN_LOOKBACK, self._SLOW_MA_PERIOD + 60))
        if not o or len(o.get("close", [])) < self._SLOW_MA_PERIOD + 5:
            result = {"experts": {}, "market": {"regime": "unknown", "regime_strength": 0.5}}
            cache[instrument] = result
            return result

        close = np.asarray(o["close"], dtype=np.float64)
        high = np.asarray(o.get("high", close), dtype=np.float64)
        low = np.asarray(o.get("low", close), dtype=np.float64)

        if "open" in o and len(o["open"]) == len(close):
            open_ = np.asarray(o["open"], dtype=np.float64)
        else:
            open_ = np.concatenate([[close[0]], close[:-1]])

        volume = np.asarray(o.get("volume", np.ones(len(close))), dtype=np.float64)

        if close.size < 30:
            result = {"experts": {}, "market": {"regime": "unknown", "regime_strength": 0.5}}
            cache[instrument] = result
            return result

        current_price = float(close[-1])

        # Market structure (single source of truth)
        near_support, near_resistance = self._compute_market_structure_signals(high, low, close)
        adv_struct = self._compute_advanced_market_structure(high, low, close, open_)

        # Experts
        trend_result = self._compute_trend_signals(high, low, close, current_price)
        momentum_result = self._compute_momentum_signals_advanced(high, low, close, volume)
        theme_result = self._compute_theme_signals_advanced(high, low, close, trend_result, momentum_result)

        result: Dict[str, Any] = {
            "experts": {
                "trend": {
                    "direction": trend_result["direction"],
                    "score": trend_result["strength"],
                    "confidence": trend_result["confidence"],
                    "proposal": {
                        # S/R + structure (from MarketStructureMixin)
                        "near_support": float(near_support),
                        "near_resistance": float(near_resistance),
                        "structure_trend": float(adv_struct["structure_trend"]),
                        "structure_strength": float(adv_struct["structure_strength"]),
                        "bos_signal": float(adv_struct["bos_signal"]),
                        "liquidity_above": float(adv_struct["liquidity_above"]),
                        "liquidity_below": float(adv_struct["liquidity_below"]),
                        "order_block_bull": float(adv_struct["order_block_bull"]),
                        "order_block_bear": float(adv_struct["order_block_bear"]),
                        # Trend indicators
                        "adx_value": float(trend_result["adx"]),
                        "plus_di": float(trend_result["plus_di"]),
                        "minus_di": float(trend_result["minus_di"]),
                        "sar_direction": int(trend_result["sar_direction"]),
                        "ma_alignment": int(trend_result["ma_alignment"]),
                        "trend_slope": float(trend_result["trend_slope"]),
                        "confluence_score": float(trend_result["confluence"]),
                    },
                },
                "momentum": {
                    "direction": momentum_result["direction"],
                    "score": momentum_result["strength"],
                    "confidence": momentum_result["confidence"],
                    "proposal": {
                        "divergence_signal": momentum_result["divergence_signal"],
                        "overbought": momentum_result["overbought"],
                        "oversold": momentum_result["oversold"],
                        "rsi_value": momentum_result["rsi"],
                        "macd_histogram": momentum_result["macd_histogram"],
                        "macd_direction": momentum_result["macd_direction"],
                        "stoch_k": momentum_result["stoch_k"],
                        "stoch_d": momentum_result["stoch_d"],
                        "roc_composite": momentum_result["roc_composite"],
                        "momentum_acceleration": momentum_result["acceleration"],
                        "volume_confirmation": momentum_result["volume_confirmation"],
                    },
                },
                "theme": {
                    "direction": theme_result["direction"],
                    "score": theme_result["strength"],
                    "confidence": theme_result["confidence"],
                    "proposal": {
                        "volatility_regime": theme_result["volatility_regime"],
                        "risk_regime": theme_result["risk_regime"],
                        "vol_score": theme_result["vol_percentile"],
                        "trend_regime": theme_result["trend_regime"],
                        "breadth_score": theme_result["breadth_score"],
                        "composite_score": theme_result["composite_score"],
                    },
                },
                "seasonality": {"direction": "neutral", "score": 0.0, "confidence": 0.5},
            },
            "market": {
                "regime": theme_result["volatility_regime"],
                "regime_strength": theme_result["vol_percentile"],
            },
        }

        cache[instrument] = result
        return result

    # ═══════════════════════════════════════════════════════════════════
    # TREND EXPERT
    # ═══════════════════════════════════════════════════════════════════

    def _compute_trend_signals(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, current_price: float
    ) -> Dict[str, Any]:
        prices = np.asarray(close, dtype=np.float64)

        fast, medium, slow, ma_alignment = self._calculate_triple_ema(prices)
        adx, plus_di, minus_di = self._calculate_adx_wilder(high, low, close, period=self._ADX_PERIOD)
        _, sar_direction = self._calculate_parabolic_sar(high, low, close)

        trend_slope = self._calculate_trend_slope(prices, lookback=30)

        direction, strength, confluence = self._determine_trend(
            fast, medium, slow, ma_alignment,
            adx, plus_di, minus_di,
            sar_direction, trend_slope, current_price
        )
        confidence = self._calculate_trend_confidence(confluence, adx, ma_alignment, strength)

        return {
            "direction": direction,
            "strength": float(strength),
            "confidence": float(confidence),
            "adx": float(adx),
            "plus_di": float(plus_di),
            "minus_di": float(minus_di),
            "sar_direction": int(sar_direction),
            "ma_alignment": int(ma_alignment),
            "trend_slope": float(trend_slope),
            "confluence": float(confluence),
        }

    def _ema_last(self, x: np.ndarray, period: int) -> float:
        """Last EMA value (stable + fast)."""
        x = np.asarray(x, dtype=np.float64)
        n = int(x.size)
        p = int(period)
        if n == 0:
            return 0.0
        if p <= 1:
            return float(x[-1])
        if n < p:
            return float(np.mean(x))

        alpha = 2.0 / (p + 1.0)
        ema = float(np.mean(x[:p]))
        for v in x[p:]:
            ema = (float(v) - ema) * alpha + ema
        return float(ema)

    def _ema_series(self, x: np.ndarray, period: int) -> np.ndarray:
        """EMA series with SMA seed placed at (p-1) to avoid early-series distortion."""
        x = np.asarray(x, dtype=np.float64)
        n = int(x.size)
        p = int(period)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        if p <= 1:
            return x.copy()

        out = np.empty(n, dtype=np.float64)

        if n < p:
            out[:] = float(np.mean(x))
            return out

        seed = float(np.mean(x[:p]))
        out[:p] = seed
        alpha = 2.0 / (p + 1.0)
        for i in range(p, n):
            out[i] = (x[i] - out[i - 1]) * alpha + out[i - 1]
        return out

    def _calculate_triple_ema(self, prices: np.ndarray) -> Tuple[float, float, float, int]:
        fast = self._ema_last(prices, self._FAST_MA_PERIOD)
        medium = self._ema_last(prices, self._MEDIUM_MA_PERIOD)
        slow = self._ema_last(prices, self._SLOW_MA_PERIOD)

        if fast > medium > slow:
            alignment = 1
        elif fast < medium < slow:
            alignment = -1
        else:
            alignment = 0

        return float(fast), float(medium), float(slow), int(alignment)

    def _calculate_adx_wilder(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> Tuple[float, float, float]:
        """
        True ADX (+DI/-DI) using Wilder methodology.

        Key fix:
        - ATR/DM can be represented as Wilder "sums" (common implementation),
          but ADX must be Wilder-smoothed *average* of DX, not a sum.
        """
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)
        c = np.asarray(close, dtype=np.float64)

        n = int(min(h.size, l.size, c.size))
        p = int(period)
        if n < p + 2:
            return 20.0, 50.0, 50.0

        h = h[-n:]
        l = l[-n:]
        c = c[-n:]

        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))

        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

        if tr.size < p:
            return 20.0, 50.0, 50.0

        def wilder_sum(x: np.ndarray, p_: int) -> np.ndarray:
            out = np.zeros_like(x, dtype=np.float64)
            out[p_ - 1] = float(np.sum(x[:p_]))
            for i in range(p_, x.size):
                out[i] = out[i - 1] - (out[i - 1] / p_) + x[i]
            out[: p_ - 1] = out[p_ - 1]  # avoid meaningless early zeros
            return out

        atr_s = wilder_sum(tr, p)
        p_dm_s = wilder_sum(plus_dm, p)
        m_dm_s = wilder_sum(minus_dm, p)

        eps = 1e-12
        atr_safe = np.maximum(atr_s, eps)
        plus_di = 100.0 * (p_dm_s / atr_safe)
        minus_di = 100.0 * (m_dm_s / atr_safe)

        di_sum = plus_di + minus_di
        dx = 100.0 * (np.abs(plus_di - minus_di) / np.maximum(di_sum, eps))

        # ADX = Wilder-smoothed average of DX (NOT sum)
        adx = np.zeros_like(dx, dtype=np.float64)
        if dx.size < p:
            adx[:] = float(np.mean(dx)) if dx.size else 20.0
        else:
            adx[p - 1] = float(np.mean(dx[:p]))
            for i in range(p, dx.size):
                adx[i] = (adx[i - 1] * (p - 1) + dx[i]) / p
            adx[: p - 1] = adx[p - 1]

        adx_last = float(np.clip(adx[-1], 0.0, 100.0))
        pdi_last = float(np.clip(plus_di[-1], 0.0, 100.0))
        mdi_last = float(np.clip(minus_di[-1], 0.0, 100.0))
        return adx_last, pdi_last, mdi_last

    def _calculate_parabolic_sar(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[float, int]:
        """
        Parabolic SAR (classic), bounded by lookback to control cost.
        Returns (sar_value, direction): 1 bullish, -1 bearish, 0 unknown.
        """
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)
        c = np.asarray(close, dtype=np.float64)

        n = int(min(h.size, l.size, c.size, int(self._SAR_LOOKBACK)))
        if n < 10:
            return float(c[-1]) if c.size else 0.0, 0

        h = h[-n:]
        l = l[-n:]
        c = c[-n:]

        rising = bool(c[-1] >= c[max(0, n - 6)])
        sar = float(l[0] if rising else h[0])
        ep = float(h[0] if rising else l[0])
        af = float(self._SAR_AF_START)

        for i in range(1, n):
            prev_sar = sar
            sar = sar + af * (ep - sar)

            if rising:
                sar = min(sar, float(l[max(i - 2, 0)]), float(l[max(i - 1, 0)]))
            else:
                sar = max(sar, float(h[max(i - 2, 0)]), float(h[max(i - 1, 0)]))

            if rising:
                if float(l[i]) < sar:
                    rising = False
                    sar = ep
                    ep = float(l[i])
                    af = float(self._SAR_AF_START)
                else:
                    if float(h[i]) > ep:
                        ep = float(h[i])
                        af = min(float(self._SAR_AF_MAX), af + float(self._SAR_AF_STEP))
            else:
                if float(h[i]) > sar:
                    rising = True
                    sar = ep
                    ep = float(h[i])
                    af = float(self._SAR_AF_START)
                else:
                    if float(l[i]) < ep:
                        ep = float(l[i])
                        af = min(float(self._SAR_AF_MAX), af + float(self._SAR_AF_STEP))

            if not np.isfinite(sar):
                sar = prev_sar

        return float(sar), (1 if rising else -1)

    def _calculate_trend_slope(self, prices: np.ndarray, lookback: int = 20) -> float:
        prices = np.asarray(prices, dtype=np.float64)
        lb = int(lookback)
        if prices.size < lb:
            return 0.0
        y = prices[-lb:]
        x = np.arange(lb, dtype=np.float64)
        try:
            slope = float(np.polyfit(x, y, 1)[0])
            avg = float(np.mean(y))
            if not np.isfinite(avg) or abs(avg) < 1e-12:
                return 0.0
            return float(slope / avg)
        except Exception:
            return 0.0

    def _determine_trend(
        self,
        fast_ma: float, medium_ma: float, slow_ma: float, ma_alignment: int,
        adx: float, plus_di: float, minus_di: float,
        sar_direction: int, trend_slope: float, current_price: float
    ) -> Tuple[str, float, float]:
        bull = 0.0
        bear = 0.0
        total = 0.0

        # MA alignment (0.25)
        if ma_alignment == 1:
            bull += 0.25
        elif ma_alignment == -1:
            bear += 0.25
        total += 0.25

        # Price vs slow MA (0.15)
        if current_price > slow_ma:
            bull += 0.15
        elif current_price < slow_ma:
            bear += 0.15
        total += 0.15

        # ADX + DI (0.20)
        if adx > self._ADX_TRENDING_THRESHOLD:
            scale = float(np.clip(adx / 50.0, 0.0, 1.0))
            if plus_di > minus_di:
                bull += 0.20 * scale
            elif minus_di > plus_di:
                bear += 0.20 * scale
        total += 0.20

        # SAR (0.15)
        if sar_direction == 1:
            bull += 0.15
        elif sar_direction == -1:
            bear += 0.15
        total += 0.15

        # Slope (0.25)
        if trend_slope > 0.001:
            bull += 0.25 * min(abs(trend_slope) * 50.0, 1.0)
        elif trend_slope < -0.001:
            bear += 0.25 * min(abs(trend_slope) * 50.0, 1.0)
        total += 0.25

        net = bull - bear
        confluence = max(bull, bear) / max(total, 1e-6)

        if abs(net) < self._MIN_NET_TREND:
            return "neutral", 0.0, float(np.clip(confluence, 0.0, 1.0))
        if net > 0:
            return "bullish", float(np.clip(bull / max(total, 1e-6), 0.0, 1.0)), float(np.clip(confluence, 0.0, 1.0))
        return "bearish", float(np.clip(bear / max(total, 1e-6), 0.0, 1.0)), float(np.clip(confluence, 0.0, 1.0))

    def _calculate_trend_confidence(
        self, confluence: float, adx: float, ma_alignment: int, strength: float
    ) -> float:
        base = 0.40

        if adx > self._ADX_STRONG_THRESHOLD:
            base += 0.25
        elif adx > self._ADX_TRENDING_THRESHOLD:
            base += 0.15

        if ma_alignment != 0:
            base += 0.15

        if confluence > self._STRONG_CONFLUENCE:
            base += 0.15
        elif confluence > self._MIN_CONFLUENCE:
            base += 0.08

        base += 0.05 * float(np.clip(strength, 0.0, 1.0))
        return float(np.clip(base, 0.10, 0.95))

    # ═══════════════════════════════════════════════════════════════════
    # MOMENTUM EXPERT
    # ═══════════════════════════════════════════════════════════════════

    def _compute_momentum_signals_advanced(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> Dict[str, Any]:
        prices = np.asarray(close, dtype=np.float64)
        highs = np.asarray(high, dtype=np.float64)
        lows = np.asarray(low, dtype=np.float64)
        vols = np.asarray(volume, dtype=np.float64)

        rsi_series = self._rsi_series(prices, self._RSI_PERIOD)
        rsi = float(rsi_series[-1]) if rsi_series.size else 50.0

        _, _, hist = self._calculate_macd(prices)
        macd_direction = 1 if hist > 0 else (-1 if hist < 0 else 0)

        stoch_k, stoch_d = self._calculate_stochastic(prices, highs, lows)
        roc_comp = self._calculate_roc_composite(prices)
        acceleration = self._calculate_momentum_acceleration(prices, lookback=10)
        vol_confirm = self._calculate_volume_confirmation(prices, vols)

        divergence = self._detect_divergence_pivot(prices, rsi_series)

        overbought = 0.0
        oversold = 0.0
        if rsi > self._RSI_OVERBOUGHT:
            overbought = (rsi - self._RSI_OVERBOUGHT) / (100.0 - self._RSI_OVERBOUGHT)
        if rsi < self._RSI_OVERSOLD:
            oversold = (self._RSI_OVERSOLD - rsi) / self._RSI_OVERSOLD

        direction, strength, confidence = self._determine_momentum(
            rsi, macd_direction, hist, stoch_k, stoch_d, roc_comp, acceleration, vol_confirm, divergence
        )

        return {
            "direction": direction,
            "strength": float(strength),
            "confidence": float(confidence),
            "rsi": float(np.clip(rsi, 0.0, 100.0)),
            "divergence_signal": divergence,
            "overbought": float(np.clip(overbought, 0.0, 1.0)),
            "oversold": float(np.clip(oversold, 0.0, 1.0)),
            "macd_histogram": float(hist),
            "macd_direction": int(macd_direction),
            "stoch_k": float(stoch_k),
            "stoch_d": float(stoch_d),
            "roc_composite": float(roc_comp),
            "acceleration": float(acceleration),
            "volume_confirmation": float(vol_confirm),
        }

    def _rsi_series(self, prices: np.ndarray, period: int) -> np.ndarray:
        """RSI series using Wilder smoothing (fast enough for per-step)."""
        c = np.asarray(prices, dtype=np.float64)
        p = int(period)
        if c.size < p + 2:
            return np.full(c.shape, 50.0, dtype=np.float64)

        delta = np.diff(c)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        rsi = np.empty(c.size, dtype=np.float64)
        rsi[: p + 1] = 50.0

        avg_gain = float(np.mean(gains[:p]))
        avg_loss = float(np.mean(losses[:p]))

        for i in range(p, gains.size):
            avg_gain = (avg_gain * (p - 1) + float(gains[i])) / p
            avg_loss = (avg_loss * (p - 1) + float(losses[i])) / p
            if avg_loss <= 1e-12:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return np.clip(rsi, 0.0, 100.0)

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        prices = np.asarray(prices, dtype=np.float64)
        if prices.size < self._MACD_SLOW + self._MACD_SIGNAL + 5:
            return 0.0, 0.0, 0.0

        fast = self._ema_series(prices, self._MACD_FAST)
        slow = self._ema_series(prices, self._MACD_SLOW)
        macd = fast - slow
        signal = self._ema_series(macd, self._MACD_SIGNAL)
        hist = macd - signal
        return float(macd[-1]), float(signal[-1]), float(hist[-1])

    def _calculate_stochastic(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Tuple[float, float]:
        close = np.asarray(close, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)

        k = int(self._STOCH_K_PERIOD)
        d = int(self._STOCH_D_PERIOD)

        if close.size < k + d + 2:
            return 50.0, 50.0

        ks: List[float] = []
        for i in range(d):
            end = close.size - i
            start = max(0, end - k)
            hh = float(np.max(high[start:end]))
            ll = float(np.min(low[start:end]))
            cc = float(close[end - 1])
            if abs(hh - ll) < 1e-12:
                ks.append(50.0)
            else:
                ks.append(100.0 * (cc - ll) / (hh - ll))

        k_now = float(np.clip(ks[0], 0.0, 100.0))
        d_now = float(np.clip(np.mean(ks), 0.0, 100.0))
        return k_now, d_now

    def _calculate_roc_composite(self, prices: np.ndarray) -> float:
        prices = np.asarray(prices, dtype=np.float64)
        if prices.size < max(self._ROC_PERIODS) + 2:
            return 0.0

        cur = float(prices[-1])
        comp = 0.0
        for p, w in zip(self._ROC_PERIODS, self._ROC_WEIGHTS):
            past = float(prices[-(int(p) + 1)])
            denom = max(abs(past), 1e-12)
            roc = (cur - past) / denom
            roc_b = float(np.clip(roc * 5.0, -1.0, 1.0))  # bound per-horizon
            comp += float(w) * roc_b

        return float(np.clip(comp, -1.0, 1.0))

    def _calculate_momentum_acceleration(self, prices: np.ndarray, lookback: int = 10) -> float:
        prices = np.asarray(prices, dtype=np.float64)
        lb = int(lookback)
        if prices.size < lb * 2 + 2:
            return 0.0
        a = float(prices[-1])
        b = float(prices[-(lb + 1)])
        c = float(prices[-(2 * lb + 1)])

        recent = (a - b) / max(abs(b), 1e-12)
        prior = (b - c) / max(abs(c), 1e-12)
        return float(np.clip((recent - prior) * 2.5, -1.0, 1.0))

    def _calculate_volume_confirmation(self, prices: np.ndarray, volume: np.ndarray, lookback: int = 20) -> float:
        prices = np.asarray(prices, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)
        n = int(min(prices.size, volume.size))
        lb = int(lookback)
        if n < lb + 2:
            return 0.5

        p = prices[-(lb + 1):]
        v = volume[-(lb + 1):]

        ret = np.diff(p)
        sign = np.sign(ret)
        obv = np.cumsum(sign * v[1:])

        if obv.size < 8 or ret.size < 8:
            return 0.5

        obv_slope = float(np.polyfit(np.arange(obv.size, dtype=np.float64), obv, 1)[0])
        px_slope = float(np.polyfit(np.arange(ret.size, dtype=np.float64), ret, 1)[0])

        if (obv_slope > 0 and px_slope > 0) or (obv_slope < 0 and px_slope < 0):
            return 0.80
        if abs(obv_slope) < 1e-12 or abs(px_slope) < 1e-12:
            return 0.50
        return 0.20

    def _pivot_lows(self, x: np.ndarray, left: int, right: int) -> List[int]:
        """Vectorized pivot low detection."""
        n = int(x.size)
        if n < left + right + 1:
            return []
        
        # Use stride tricks for rolling windows
        window_size = left + right + 1
        shape = (n - window_size + 1, window_size)
        strides = (x.strides[0], x.strides[0])
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        
        center = windows[:, left]
        is_min = center == np.min(windows, axis=1)
        
        # Check strict less than on both sides
        left_ok = np.all(windows[:, :left] > center[:, None], axis=1)
        right_ok = np.all(windows[:, left+1:] > center[:, None], axis=1)
        
        valid = is_min & left_ok & right_ok
        return (np.where(valid)[0] + left).tolist()

    def _pivot_highs(self, x: np.ndarray, left: int, right: int) -> List[int]:
        """Vectorized pivot high detection."""
        n = int(x.size)
        if n < left + right + 1:
            return []
        
        window_size = left + right + 1
        shape = (n - window_size + 1, window_size)
        strides = (x.strides[0], x.strides[0])
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        
        center = windows[:, left]
        is_max = center == np.max(windows, axis=1)
        
        left_ok = np.all(windows[:, :left] < center[:, None], axis=1)
        right_ok = np.all(windows[:, left+1:] < center[:, None], axis=1)
        
        valid = is_max & left_ok & right_ok
        return (np.where(valid)[0] + left).tolist()

    def _detect_divergence_pivot(self, prices: np.ndarray, rsi_series: np.ndarray) -> Optional[str]:
        """
        Pivot-based divergence (stable + efficient):
        - Bullish: price lower low, RSI higher low
        - Bearish: price higher high, RSI lower high
        """
        c = np.asarray(prices, dtype=np.float64)
        r = np.asarray(rsi_series, dtype=np.float64)
        n = int(min(c.size, r.size, int(self._DIVERGENCE_LOOKBACK)))
        if n < 40:
            return None

        c = c[-n:]
        r = r[-n:]

        left = int(self._PIVOT_LEFT)
        right = int(self._PIVOT_RIGHT)

        lows = self._pivot_lows(c, left, right)
        highs = self._pivot_highs(c, left, right)

        if len(lows) >= 2:
            i1, i2 = lows[-2], lows[-1]
            p1, p2 = float(c[i1]), float(c[i2])
            r1, r2 = float(r[i1]), float(r[i2])
            if p2 < p1 * (1.0 - float(self._DIVERGENCE_SIGNIFICANCE)) and r2 > r1 + float(self._DIVERGENCE_MIN_RSI_DELTA):
                return "bullish"

        if len(highs) >= 2:
            i1, i2 = highs[-2], highs[-1]
            p1, p2 = float(c[i1]), float(c[i2])
            r1, r2 = float(r[i1]), float(r[i2])
            if p2 > p1 * (1.0 + float(self._DIVERGENCE_SIGNIFICANCE)) and r2 < r1 - float(self._DIVERGENCE_MIN_RSI_DELTA):
                return "bearish"

        return None

    def _determine_momentum(
        self,
        rsi: float, macd_direction: int, macd_histogram: float,
        stoch_k: float, stoch_d: float,
        roc_composite: float, acceleration: float,
        volume_confirmation: float, divergence: Optional[str]
    ) -> Tuple[str, float, float]:
        bull = 0.0
        bear = 0.0

        # RSI (0.20)
        if rsi >= 50:
            bull += 0.20 * ((rsi - 50.0) / 50.0)
        else:
            bear += 0.20 * ((50.0 - rsi) / 50.0)

        # MACD histogram (0.20)
        hist_mag = min(abs(macd_histogram) * 50.0, 1.0)
        if macd_direction > 0:
            bull += 0.20 * hist_mag
        elif macd_direction < 0:
            bear += 0.20 * hist_mag

        # Stochastic (0.15)
        st = (stoch_k - 50.0) / 50.0
        if st >= 0:
            bull += 0.15 * st
        else:
            bear += 0.15 * (-st)

        # ROC composite (0.25)
        if roc_composite >= 0:
            bull += 0.25 * min(abs(roc_composite) * 1.5, 1.0)
        else:
            bear += 0.25 * min(abs(roc_composite) * 1.5, 1.0)

        # Acceleration (0.10)
        if acceleration >= 0:
            bull += 0.10 * min(abs(acceleration) * 1.2, 1.0)
        else:
            bear += 0.10 * min(abs(acceleration) * 1.2, 1.0)

        # Volume confirmation (0.10 as bias)
        vol_bias = (float(volume_confirmation) - 0.5) * 0.10
        if bull >= bear:
            bull += vol_bias
        else:
            bear += vol_bias

        # Divergence bonus (context)
        if divergence == "bullish":
            bull += 0.15
        elif divergence == "bearish":
            bear += 0.15

        net = bull - bear
        if abs(net) < 0.05:
            direction = "neutral"
            strength = 0.0
        elif net > 0:
            direction = "bullish"
            strength = float(np.clip(bull, 0.0, 1.0))
        else:
            direction = "bearish"
            strength = float(np.clip(bear, 0.0, 1.0))

        agrees = 0
        agrees += 1 if (rsi > 50) else 0
        agrees += 1 if (macd_direction > 0) else 0
        agrees += 1 if (stoch_k > stoch_d) else 0
        agrees += 1 if (roc_composite > 0) else 0
        if direction == "bearish":
            agrees = 4 - agrees

        confidence = 0.30 + (agrees / 4.0) * 0.50 + (float(volume_confirmation) - 0.5) * 0.20
        return direction, strength, float(np.clip(confidence, 0.10, 0.95))

    # ═══════════════════════════════════════════════════════════════════
    # THEME EXPERT
    # ═══════════════════════════════════════════════════════════════════

    def _compute_theme_signals_advanced(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        trend_result: Dict[str, Any],
        momentum_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        vol_percentile = self._calculate_volatility_percentile(high, low, close)

        if vol_percentile < self._VOL_THRESHOLD_LOW:
            volatility_regime = "low"
        elif vol_percentile > self._VOL_THRESHOLD_HIGH:
            volatility_regime = "high"
        else:
            volatility_regime = "normal"

        adx = float(trend_result.get("adx", 20.0))
        if adx > self._ADX_STRONG_THRESHOLD:
            trend_regime = "strong"
        elif adx > self._ADX_TRENDING_THRESHOLD:
            trend_regime = "trending"
        else:
            trend_regime = "ranging"

        breadth = self._calculate_breadth_score(high, low, close)

        trend_dir = str(trend_result.get("direction", "neutral"))
        risk_regime = self._determine_risk_regime(volatility_regime, trend_dir, trend_regime, breadth)

        composite = self._calculate_theme_composite(
            vol_percentile=vol_percentile,
            adx=adx,
            breadth=breadth,
            trend_strength=float(trend_result.get("strength", 0.0)),
            momentum_strength=float(momentum_result.get("strength", 0.0)),
        )

        direction, strength = self._determine_theme_direction(volatility_regime, risk_regime, trend_dir, composite)

        # Confidence increases with clarity and ADX
        clarity = abs(composite - 0.5) * 2.0
        confidence = 0.40 + 0.30 * float(np.clip(clarity, 0.0, 1.0)) + 0.20 * float(np.clip(adx / 50.0, 0.0, 1.0))

        return {
            "direction": direction,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.10, 0.95)),
            "volatility_regime": volatility_regime,
            "trend_regime": trend_regime,
            "risk_regime": risk_regime,
            "vol_percentile": float(np.clip(vol_percentile, 0.0, 1.0)),
            "breadth_score": float(np.clip(breadth, 0.0, 1.0)),
            "composite_score": float(np.clip(composite, 0.0, 1.0)),
        }

    def _atr_series(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """ATR series (Wilder average) for percentile work."""
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)
        c = np.asarray(close, dtype=np.float64)
        if min(h.size, l.size, c.size) < 3:
            return np.zeros_like(c)

        prev_c = np.roll(c, 1)
        prev_c[0] = c[0]
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

        p = int(period)
        atr = np.empty_like(tr)
        if tr.size < p:
            atr[:] = float(np.mean(tr)) if tr.size else 0.0
            return atr

        atr[:p] = float(np.mean(tr[:p]))
        for i in range(p, tr.size):
            atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p
        return atr

    def _calculate_volatility_percentile(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        need = int(self._VOL_LOOKBACK) + int(self._ATR_PERIOD) + 5
        if len(close) < need:
            return 0.5

        h = np.asarray(high[-need:], dtype=np.float64)
        l = np.asarray(low[-need:], dtype=np.float64)
        c = np.asarray(close[-need:], dtype=np.float64)

        atr = self._atr_series(h, l, c, int(self._ATR_PERIOD))
        window = atr[-int(self._VOL_LOOKBACK):]
        window = window[np.isfinite(window)]
        if window.size < 10:
            return 0.5

        cur = float(window[-1])
        return float(np.mean(window <= cur))

    def _calculate_breadth_score(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)

        period = min(int(self._ATR_PERIOD), int(close.size))
        if period < 10:
            return 0.5

        hh = float(np.max(high[-period:]))
        ll = float(np.min(low[-period:]))
        cur = float(close[-1])

        range_pos = 0.5 if abs(hh - ll) < 1e-12 else (cur - ll) / (hh - ll)
        changes = np.diff(close[-period:])
        up_ratio = float(np.sum(changes > 0)) / float(changes.size if changes.size else 1.0)

        breadth = 0.55 * float(range_pos) + 0.45 * float(up_ratio)
        return float(np.clip(breadth, 0.0, 1.0))

    def _determine_risk_regime(self, vol_regime: str, trend_dir: str, trend_regime: str, breadth: float) -> str:
        score = 0.5

        if vol_regime == "low":
            score += 0.15
        elif vol_regime == "high":
            score -= 0.20

        if trend_dir == "bullish":
            score += 0.15
        elif trend_dir == "bearish":
            score -= 0.15

        if trend_regime == "strong":
            score += 0.08 if trend_dir == "bullish" else -0.08

        score += (float(breadth) - 0.5) * 0.20

        if score > 0.6:
            return "risk_on"
        if score < 0.4:
            return "risk_off"
        return "neutral"

    def _calculate_theme_composite(
        self,
        *,
        vol_percentile: float,
        adx: float,
        breadth: float,
        trend_strength: float,
        momentum_strength: float,
    ) -> float:
        vol_score = 1.0 - float(np.clip(vol_percentile, 0.0, 1.0))  # lower vol => higher score
        trend_score = float(np.clip(adx / 50.0, 0.0, 1.0))
        br = float(np.clip(breadth, 0.0, 1.0))
        ts = float(np.clip(trend_strength, 0.0, 1.0))
        ms = float(np.clip(momentum_strength, 0.0, 1.0))

        composite = 0.25 * vol_score + 0.25 * trend_score + 0.25 * br + 0.15 * ts + 0.10 * ms
        return float(np.clip(composite, 0.0, 1.0))

    def _determine_theme_direction(self, vol_regime: str, risk_regime: str, trend_dir: str, composite: float) -> Tuple[str, float]:
        comp = float(np.clip(composite, 0.0, 1.0))
        if risk_regime == "risk_on" and trend_dir == "bullish":
            return "bullish", comp
        if risk_regime == "risk_off" or (vol_regime == "high" and trend_dir == "bearish"):
            return "bearish", 1.0 - comp
        return "neutral", float(np.clip(0.5 - abs(comp - 0.5), 0.0, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════

    def _compute_momentum_signals(self, close: np.ndarray, rsi: float) -> Tuple[Optional[str], float, float]:
        # Maintain previous signature: divergence, overbought, oversold
        prices = np.asarray(close, dtype=np.float64)
        rsi_series = self._rsi_series(prices, int(self._RSI_PERIOD))
        divergence = self._detect_divergence_pivot(prices, rsi_series)

        overbought = 0.0
        oversold = 0.0
        if rsi > self._RSI_OVERBOUGHT:
            overbought = (rsi - self._RSI_OVERBOUGHT) / (100.0 - self._RSI_OVERBOUGHT)
        if rsi < self._RSI_OVERSOLD:
            oversold = (self._RSI_OVERSOLD - rsi) / self._RSI_OVERSOLD

        return divergence, float(np.clip(overbought, 0.0, 1.0)), float(np.clip(oversold, 0.0, 1.0))

    def _compute_theme_signals(self, vol_proxy: float, trend_dir: str) -> Tuple[str, str]:
        if vol_proxy < self._VOL_THRESHOLD_LOW:
            volatility_regime = "low"
        elif vol_proxy > self._VOL_THRESHOLD_HIGH:
            volatility_regime = "high"
        else:
            volatility_regime = "normal"

        if volatility_regime == "low" and trend_dir == "bullish":
            risk_regime = "risk_on"
        elif volatility_regime == "high" or trend_dir == "bearish":
            risk_regime = "risk_off"
        else:
            risk_regime = "neutral"

        return volatility_regime, risk_regime
