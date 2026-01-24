#!/usr/bin/env python3
"""
MomentumExpert v3.2 — Institutional-Grade Momentum (Base-Aligned, Train/Live Safe)
=================================================================================

Core contract:
- Emit ONLY per-instrument directional votes: long / short / flat.
- VotingExpertBase owns:
  - position-focus reframing (hold/exit/tighten)
  - gating/normalization and canonical publishing
  - contract-compliant output envelope
- This expert focuses strictly on momentum signal quality, vetoes, and diagnostics.

Design principles:
- XAUUSD-only (hard enforced to avoid cross-symbol leakage).
- Multi-layer defense against live overtrading:
  - CHOP veto (range filter)
  - Exhaustion filter (climax avoidance)
  - Deceleration penalty (thrust fading)
  - Trend-context penalty (counter-trend throttle)
  - Whipsaw / persistence memory (anti-flip)
  - Optional Squeeze + Fisher + MFI divergence (alpha layer, gated)
- Robust debug tracing (JSONL) with component-level forensics.

Notes:
- `use_forming_bar` defaults False for train/live parity (closed bars only).
- Any missing data must fail open to "flat" without crashing the tick.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.experts.base import VotingExpertBase
from modules.voting.core.constants import PRIMARY_TIMEFRAME, CONTEXT_TIMEFRAMES, normalize_instrument


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


# Debug default: keep OFF for production; enable via config keys:
# - momentum_debug_enabled: true
# - debug_momentum: true
MOMENTUM_DEBUG_DEFAULT: bool = True


# ──────────────────────────────────────────────────────────────
# Momentum Expert
# ──────────────────────────────────────────────────────────────

@module(**module_args("MomentumExpert"))
class MomentumExpert(VotingExpertBase):
    """
    Institutional Momentum Expert — per-instrument directional voting (XAUUSD-only).

    Emits ONLY:
      - long / short / flat per instrument
    """

    # ═══════════════════════════ INIT ═══════════════════════════

    def _expert_specific_init(self) -> None:
        # Hard enforce XAUUSD only (prevents cross-symbol leakage by design)
        self.instruments: List[str] = ["XAUUSD"]

        # Timeframes
        self.primary_tf: str = str(self.config.get("primary_timeframe", PRIMARY_TIMEFRAME) or PRIMARY_TIMEFRAME)
        self.context_tfs: List[str] = list(self.config.get("context_timeframes", list(CONTEXT_TIMEFRAMES)))
        self.mtf_timeframes: List[str] = [self.primary_tf] + [tf for tf in self.context_tfs if tf != self.primary_tf]

        # Train/live parity default
        self.use_forming_bar: bool = bool(self.config.get("use_forming_bar", False))

        # Base momentum threshold (used to normalize ROC-like components)
        self.base_momentum_threshold: float = float(self.config.get("momentum_threshold", 0.015))
        self.vol_lookback: int = int(self.config.get("vol_lookback", 30))
        self.vol_ref: float = float(self.config.get("vol_ref", 0.0025))  # anchor scale
        self.atr_period: int = int(self.config.get("atr_period", 14))

        # ROC periods (kept, but enhanced with volume weighting)
        self.roc_periods: List[int] = list(self.config.get("roc_periods", [5, 10, 20, 50]))
        self.roc_weights: List[float] = list(self.config.get("roc_weights", [0.35, 0.30, 0.20, 0.15]))
        if len(self.roc_weights) != len(self.roc_periods):
            self.roc_weights = [1.0 / max(1, len(self.roc_periods))] * len(self.roc_periods)

        # RSI / MACD / Stoch (kept but used as components, not as primary truth)
        self.rsi_period: int = int(self.config.get("rsi_period", 14))
        self.macd_fast: int = int(self.config.get("macd_fast", 12))
        self.macd_slow: int = int(self.config.get("macd_slow", 26))
        self.macd_signal: int = int(self.config.get("macd_signal", 9))

        self.stoch_k_period: int = int(self.config.get("stoch_k", 14))
        self.stoch_d_period: int = int(self.config.get("stoch_d", 3))

        # Fast trigger line (DeepSeek) — HMA default
        self.fast_line_mode: str = str(self.config.get("fast_line_mode", "hma") or "hma").lower()
        self.fast_period: int = int(self.config.get("fast_period", 8))
        self.fast_slope_lookback: int = int(self.config.get("fast_slope_lookback", 12))

        # Trend filter (confidence only)
        self.use_trend_filter: bool = bool(self.config.get("use_trend_filter", True))
        self.trend_ema_fast: int = int(self.config.get("trend_ema_fast", 20))
        self.trend_ema_slow: int = int(self.config.get("trend_ema_slow", 50))
        self.counter_trend_penalty: float = float(self.config.get("counter_trend_penalty", 0.18))

        # CHOP veto (DeepSeek)
        self.use_chop_filter: bool = bool(self.config.get("use_chop_filter", True))
        self.chop_period: int = int(self.config.get("chop_period", 14))
        self.chop_thr: float = float(self.config.get("chop_thr", 61.8))
        self.chop_conf_penalty: float = float(self.config.get("chop_conf_penalty", 0.18))
        self.chop_strength_penalty: float = float(self.config.get("chop_strength_penalty", 0.30))

        # Exhaustion filter (DeepSeek)
        self.use_exhaustion_filter: bool = bool(self.config.get("use_exhaustion_filter", True))
        self.exhaustion_lookback: int = int(self.config.get("exhaustion_lookback", 40))
        self.exhaustion_force_flat_thr: float = float(self.config.get("exhaustion_force_flat_thr", 0.75))
        self.exhaustion_conf_mult: float = float(self.config.get("exhaustion_conf_mult", 0.55))

        # Deceleration detection (DeepSeek)
        self.use_deceleration: bool = bool(self.config.get("use_deceleration", True))
        self.decel_penalty: float = float(self.config.get("decel_penalty", 0.25))
        self.decel_thr: float = float(self.config.get("decel_thr", -0.15))

        # MTF alignment (DeepSeek) — confidence only
        self.use_mtf_confirmation: bool = bool(self.config.get("use_mtf_confirmation", True))
        self.mtf_agreement_bonus: float = float(self.config.get("mtf_agreement_bonus", 0.12))
        self.mtf_disagreement_penalty: float = float(self.config.get("mtf_disagreement_penalty", 0.18))
        self.mtf_tf_weights: Dict[str, float] = dict(self.config.get("mtf_tf_weights", {"M15": 1.0, "H1": 0.8, "H4": 0.6, "D1": 0.4}))

        # Structure context (DeepSeek) — optional but ON by default (lightweight)
        self.use_structure_context: bool = bool(self.config.get("use_structure_context", True))
        self.sr_lookback: int = int(self.config.get("sr_lookback", 40))
        self.sr_proximity_atr: float = float(self.config.get("sr_proximity_atr", 0.75))
        self.structure_weight: float = float(self.config.get("structure_weight", 0.08))

        # Gemini alpha layer (optional)
        self.use_fisher: bool = bool(self.config.get("use_fisher", True))
        self.fisher_period: int = int(self.config.get("fisher_period", 10))
        self.fisher_extreme: float = float(self.config.get("fisher_extreme", 2.5))
        self.fisher_weight: float = float(self.config.get("fisher_weight", 0.10))

        self.use_squeeze: bool = bool(self.config.get("use_squeeze", True))
        self.squeeze_bb_period: int = int(self.config.get("squeeze_bb_period", 20))
        self.squeeze_bb_mult: float = float(self.config.get("squeeze_bb_mult", 2.0))
        self.squeeze_kc_period: int = int(self.config.get("squeeze_kc_period", 20))
        self.squeeze_kc_mult: float = float(self.config.get("squeeze_kc_mult", 1.5))
        self.squeeze_weight: float = float(self.config.get("squeeze_weight", 0.10))

        self.use_mfi: bool = bool(self.config.get("use_mfi", True))
        self.mfi_period: int = int(self.config.get("mfi_period", 14))
        self.mfi_weight: float = float(self.config.get("mfi_weight", 0.10))
        self.use_mfi_divergence_veto: bool = bool(self.config.get("use_mfi_divergence_veto", True))
        self.mfi_div_lookback: int = int(self.config.get("mfi_div_lookback", 20))

        # Cross-asset confirmation (optional; OFF by default)
        self.use_cross_asset: bool = bool(self.config.get("use_cross_asset", False))
        self.cross_asset_tf: str = str(self.config.get("cross_asset_tf", "H1"))
        self.cross_asset_symbols: Dict[str, str] = dict(self.config.get("cross_asset_symbols", {"dxy": "DXY", "us10y": "US10Y"}))
        self.cross_asset_conf_scale: float = float(self.config.get("cross_asset_conf_scale", 0.12))

        # Whipsaw / persistence (kept)
        self.signal_hist_len: int = int(self.config.get("signal_history_len", 12))
        self.persistence_window: int = int(self.config.get("persistence_window", 5))
        self.whipsaw_penalty: float = float(self.config.get("whipsaw_penalty", 0.15))
        self.persistence_bonus: float = float(self.config.get("persistence_bonus", 0.08))

        # Decision thresholds
        self.net_band: float = float(self.config.get("net_band", 0.08))  # require net momentum beyond this to trade
        self.min_confidence: float = float(self.config.get("min_confidence", 0.10))
        self.max_confidence: float = float(self.config.get("max_confidence", 0.95))

        # Debug trace
        cfg_val = self.config.get("momentum_debug_enabled")
        if cfg_val is None:
            cfg_val = self.config.get("debug_momentum")
        if cfg_val is None:
            cfg_val = MOMENTUM_DEBUG_DEFAULT
        self.debug_enabled: bool = bool(cfg_val)
        self.debug_dump_bars: int = int(self.config.get("momentum_debug_dump_bars", 200))
        self.debug_path: str = str(self.config.get("momentum_debug_path", "logs/voting/momentum_expert_debug.jsonl"))

        # Per-instrument rolling state
        self.instrument_state: Dict[str, Dict[str, Any]] = {}
        self._signal_history: Dict[str, Deque[int]] = {}

        for inst in self.instruments:
            self.instrument_state[inst] = {
                "rsi_history": deque(maxlen=80),
                "macd_line_history": deque(maxlen=120),
                "macd_hist_history": deque(maxlen=120),
                "mfi_history": deque(maxlen=80),
                "obv_history": deque(maxlen=80),
                "net_history": deque(maxlen=200),  # net momentum values
                "squeeze_on_prev": False,
                "last_analysis": {},
            }
            self._signal_history[inst] = deque(maxlen=self.signal_hist_len)

        self._publish_momentum_baseline()
        self.log_info(
            f"[MomentumExpert v3.2] init | instruments={self.instruments} | tf={self.primary_tf} | "
            f"use_forming_bar={self.use_forming_bar} | debug={self.debug_enabled}"
        )

    def _publish_momentum_baseline(self) -> None:
        try:
            self.smart_bus.set(
                "momentum_analysis",
                {
                    "composite_momentum": 0.0,
                    "direction": 0,
                    "acceleration": 0.0,
                    "per_instrument": {},
                    "timestamp": _dt.datetime.now().isoformat(),
                },
                module=self.__class__.__name__,
                thesis="Momentum analysis baseline",
            )
        except Exception:
            pass

    # ═══════════════════════════ DEBUG TRACE ═══════════════════════════

    def _debug_write(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.debug_enabled:
            return
        try:
            os.makedirs(os.path.dirname(self.debug_path) or ".", exist_ok=True)
            row = {"ts": _dt.datetime.now().isoformat(), "module": self.__class__.__name__, "event": event, **payload}
            with open(self.debug_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            return

    # ═══════════════════════════ BUS DATA ACCESS ═══════════════════════════

    def _get_historical(self) -> Optional[Dict[str, Any]]:
        try:
            hist = self.smart_bus.get("historical_prices", self.__class__.__name__, default=None)
            return hist if isinstance(hist, dict) else None
        except Exception:
            return None

    def _get_tf_rec(self, instrument: str, tf: str) -> Optional[Dict[str, Any]]:
        hist = self._get_historical()
        if not isinstance(hist, dict):
            return None

        inst_norm = normalize_instrument(instrument)
        matched_sym: Optional[str] = None
        for sym in hist.keys():
            if normalize_instrument(sym) == inst_norm:
                matched_sym = sym
                break
        if not matched_sym:
            return None

        sym_block = hist.get(matched_sym)
        if not isinstance(sym_block, dict):
            return None

        rec = sym_block.get(tf)
        return rec if isinstance(rec, dict) else None

    def _get_ohlcv_series(self, instrument: str, tf: str) -> Dict[str, List[float]]:
        rec = self._get_tf_rec(instrument, tf)
        if not isinstance(rec, dict):
            return {"open": [], "high": [], "low": [], "close": [], "volume": []}

        def _seq(name: str) -> List[float]:
            x = rec.get(name)
            if isinstance(x, (list, tuple, np.ndarray)):
                out = []
                for v in list(x):
                    fv = _safe_float(v, default=np.nan)
                    if not np.isnan(fv) and not np.isinf(fv):
                        out.append(float(fv))
                return out
            return []

        return {
            "open": _seq("open"),
            "high": _seq("high"),
            "low": _seq("low"),
            "close": _seq("close"),
            "volume": _seq("volume"),
        }

    def _maybe_apply_forming_bar(self, instrument: str, tf: str, ohlcv: Dict[str, List[float]]) -> Dict[str, List[float]]:
        if not self.use_forming_bar:
            return ohlcv
        rec = self._get_tf_rec(instrument, tf)
        if not isinstance(rec, dict):
            return ohlcv
        cur_bar = rec.get("current_bar")
        if not isinstance(cur_bar, dict):
            return ohlcv
        close_f = cur_bar.get("close")
        if close_f is None:
            return ohlcv

        out = {k: list(v) for k, v in ohlcv.items()}
        if out["close"]:
            out["close"][-1] = _safe_float(close_f, out["close"][-1])
        if out["high"] and cur_bar.get("high") is not None:
            out["high"][-1] = _safe_float(cur_bar.get("high"), out["high"][-1])
        if out["low"] and cur_bar.get("low") is not None:
            out["low"][-1] = _safe_float(cur_bar.get("low"), out["low"][-1])
        if out["volume"] and cur_bar.get("volume") is not None:
            out["volume"][-1] = max(0.0, _safe_float(cur_bar.get("volume"), out["volume"][-1]))
        return out

    # ═══════════════════════════ INDICATORS ═══════════════════════════

    def _ema_series(self, data: List[float], period: int) -> List[float]:
        if period <= 1 or len(data) < 2:
            return list(data)
        alpha = 2.0 / (period + 1.0)
        out = [float(data[0])]
        ema = float(data[0])
        for x in data[1:]:
            ema = ema + alpha * (float(x) - ema)
            out.append(float(ema))
        return out

    def _wma_series(self, data: List[float], period: int) -> List[float]:
        if period <= 1 or len(data) < period:
            return list(data)
        out: List[float] = []
        w = np.arange(1, period + 1, dtype=float)
        ws = float(np.sum(w))
        arr = np.array(data, dtype=float)
        for i in range(len(arr)):
            if i + 1 < period:
                out.append(float(arr[i]))
                continue
            window = arr[i + 1 - period : i + 1]
            out.append(float(np.dot(window, w) / ws))
        return out

    def _hma_series(self, data: List[float], period: int) -> List[float]:
        """Hull Moving Average: HMA(n) = WMA( 2*WMA(n/2) - WMA(n), sqrt(n) )."""
        n = int(max(1, period))
        if len(data) < max(4, n):
            return list(data)
        half = max(1, n // 2)
        sqrt_n = max(1, int(np.sqrt(n)))

        wma_full = self._wma_series(data, n)
        wma_half = self._wma_series(data, half)

        # Align lengths
        m = min(len(wma_full), len(wma_half))
        diff = [2.0 * wma_half[-m + i] - wma_full[-m + i] for i in range(m)]
        # Prepend to maintain original length
        prefix_len = len(data) - len(diff)
        series = (list(data[:prefix_len]) + diff) if prefix_len > 0 else diff
        return self._wma_series(series, sqrt_n)

    def _linreg_slope_norm(self, series: List[float], lookback: int) -> float:
        """Normalized regression slope over last N points into [-1, 1]."""
        n = int(max(5, lookback))
        if len(series) < n:
            return 0.0
        y = np.array(series[-n:], dtype=float)
        x = np.arange(n, dtype=float)
        x = x - float(np.mean(x))
        y_mean = float(np.mean(y))
        y0 = y - y_mean
        denom = float(np.sum(x * x))
        if denom <= 1e-12:
            return 0.0
        slope = float(np.sum(x * y0) / denom)
        scale = float(np.std(y)) + 1e-12
        s = slope / scale
        return _clip(s * 0.85, -1.0, 1.0)

    def _rsi_wilder(self, prices: List[float], period: int) -> float:
        if len(prices) < period + 2:
            return 50.0
        deltas = np.diff(np.array(prices[-(period + 2):], dtype=float))
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
            avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
        if avg_loss <= 1e-12:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(np.clip(rsi, 0.0, 100.0))

    def _macd(self, prices: List[float]) -> Tuple[float, float, float]:
        if len(prices) < max(self.macd_slow, self.macd_fast) + self.macd_signal + 5:
            return 0.0, 0.0, 0.0
        ema_fast = self._ema_series(prices, self.macd_fast)
        ema_slow = self._ema_series(prices, self.macd_slow)
        m = min(len(ema_fast), len(ema_slow))
        macd_line_series = [ema_fast[-m + i] - ema_slow[-m + i] for i in range(m)]
        signal_series = self._ema_series(macd_line_series, self.macd_signal)
        macd_line = float(macd_line_series[-1]) if macd_line_series else 0.0
        signal_line = float(signal_series[-1]) if signal_series else 0.0
        hist = float(macd_line - signal_line)
        return macd_line, signal_line, hist

    def _stochastic(self, closes: List[float], highs: List[float], lows: List[float]) -> Tuple[float, float]:
        n = self.stoch_k_period
        if len(closes) < n or not closes:
            return 50.0, 50.0
        h = highs[-n:] if len(highs) >= n else closes[-n:]
        l = lows[-n:] if len(lows) >= n else closes[-n:]
        hh = max(h) if h else closes[-1]
        ll = min(l) if l else closes[-1]
        if abs(hh - ll) < 1e-12:
            k = 50.0
        else:
            k = (closes[-1] - ll) / (hh - ll) * 100.0
        d = k
        return float(np.clip(k, 0.0, 100.0)), float(np.clip(d, 0.0, 100.0))

    def _obv(self, closes: List[float], volumes: List[float]) -> float:
        if len(closes) < 2 or len(volumes) < 2:
            return 0.0
        m = min(len(closes), len(volumes))
        obv = 0.0
        for i in range(1, m):
            if closes[i] > closes[i - 1]:
                obv += volumes[i]
            elif closes[i] < closes[i - 1]:
                obv -= volumes[i]
        return float(obv)

    def _atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        if len(closes) < period + 2 or len(highs) < period + 2 or len(lows) < period + 2:
            return 0.0
        h = np.array(highs[-(period + 1):], dtype=float)
        l = np.array(lows[-(period + 1):], dtype=float)
        c = np.array(closes[-(period + 1):], dtype=float)
        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        return float(np.mean(tr)) if len(tr) > 0 else 0.0

    def _atr_zscore(self, highs: List[float], lows: List[float], closes: List[float], lookback: int) -> float:
        n = int(max(20, lookback))
        if len(closes) < n + 2 or len(highs) < n + 2 or len(lows) < n + 2:
            return 0.0
        # TR series over lookback
        h = np.array(highs[-(n + 1):], dtype=float)
        l = np.array(lows[-(n + 1):], dtype=float)
        c = np.array(closes[-(n + 1):], dtype=float)
        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        if len(tr) < 5:
            return 0.0
        last = float(tr[-1])
        mu = float(np.mean(tr))
        sd = float(np.std(tr)) + 1e-12
        return (last - mu) / sd

    def _realized_vol(self, prices: List[float], lookback: int) -> float:
        if len(prices) < lookback + 2:
            return 0.0
        arr = np.array(prices[-(lookback + 1):], dtype=float)
        r = np.diff(np.log(np.clip(arr, 1e-12, np.inf)))
        return float(np.std(r)) if len(r) > 2 else 0.0

    def _volume_weighted_roc(self, closes: List[float], volumes: List[float], period: int) -> float:
        if len(closes) < period + 1:
            return 0.0
        past = closes[-(period + 1)]
        cur = closes[-1]
        if abs(past) <= 1e-12:
            return 0.0
        price_roc = (cur - past) / past

        if len(volumes) < period + 1:
            return float(price_roc)
        avgv = float(np.mean(volumes[-period:])) if period > 1 else float(volumes[-1])
        if avgv <= 1e-12:
            return float(price_roc)
        vol_ratio = float(volumes[-1] / avgv)
        vol_weight = _clip(0.5 + vol_ratio, 0.5, 1.5)
        return float(price_roc * vol_weight)

    def _roc_multi_vw(self, closes: List[float], volumes: List[float]) -> Dict[int, float]:
        return {p: self._volume_weighted_roc(closes, volumes, p) for p in self.roc_periods}

    def _choppiness_index(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        n = int(max(10, period))
        if len(closes) < n + 2 or len(highs) < n + 2 or len(lows) < n + 2:
            return 50.0
        h = np.array(highs[-(n + 1):], dtype=float)
        l = np.array(lows[-(n + 1):], dtype=float)
        c = np.array(closes[-(n + 1):], dtype=float)
        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        sum_tr = float(np.sum(tr)) + 1e-12
        hh = float(np.max(h[1:]))
        ll = float(np.min(l[1:]))
        rng = float(hh - ll) + 1e-12
        chop = 100.0 * (np.log10(sum_tr / rng) / np.log10(float(n)))
        return float(np.clip(chop, 0.0, 100.0))

    # ───────── Gemini: Fisher Transform ─────────

    def _fisher_transform(self, closes: List[float], period: int) -> Tuple[float, float]:
        """
        Ehlers Fisher Transform:
        - Normalize price into [-1, +1] by rolling min/max
        - Apply Fisher log transform
        Returns: (fisher, trigger)
        """
        n = int(max(5, period))
        if len(closes) < n + 2:
            return 0.0, 0.0

        # Build normalized x series
        xvals: List[float] = []
        for i in range(-n, 0):
            window = closes[i - n + 1 : i + 1] if i - n + 1 >= -len(closes) else closes[: i + 1]
            if len(window) < 3:
                xvals.append(0.0)
                continue
            hi = max(window)
            lo = min(window)
            if abs(hi - lo) < 1e-12:
                xvals.append(0.0)
            else:
                v = 2.0 * ((closes[i] - lo) / (hi - lo)) - 1.0
                xvals.append(_clip(v, -0.999, 0.999))

        # Smooth and transform
        fisher_series: List[float] = []
        v = 0.0
        for x in xvals:
            v = 0.33 * x + 0.67 * v
            v = _clip(v, -0.999, 0.999)
            f = 0.5 * np.log((1.0 + v) / (1.0 - v))
            fisher_series.append(float(f))
        fisher = float(fisher_series[-1]) if fisher_series else 0.0
        trigger = float(fisher_series[-2]) if len(fisher_series) >= 2 else fisher
        return fisher, trigger

    # ───────── Gemini: TTM Squeeze ─────────

    def _sma(self, data: List[float], period: int) -> float:
        n = int(max(2, period))
        if len(data) < n:
            return float(np.mean(data)) if data else 0.0
        return float(np.mean(np.array(data[-n:], dtype=float)))

    def _std(self, data: List[float], period: int) -> float:
        n = int(max(2, period))
        if len(data) < n:
            return float(np.std(data)) if data else 0.0
        return float(np.std(np.array(data[-n:], dtype=float)))

    def _check_squeeze(self, closes: List[float], highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """
        Squeeze ON if Bollinger Bands are inside Keltner Channels.
        Returns squeeze_on, squeeze_release, squeeze_mom (normalized).
        """
        if len(closes) < max(self.squeeze_bb_period, self.squeeze_kc_period) + 5:
            return {"available": False, "squeeze_on": False, "squeeze_release": False, "squeeze_mom": 0.0}

        bb_mid = self._sma(closes, self.squeeze_bb_period)
        bb_std = self._std(closes, self.squeeze_bb_period)
        bb_u = bb_mid + self.squeeze_bb_mult * bb_std
        bb_l = bb_mid - self.squeeze_bb_mult * bb_std

        kc_mid_series = self._ema_series(closes, self.squeeze_kc_period)
        kc_mid = float(kc_mid_series[-1]) if kc_mid_series else bb_mid
        atr = self._atr(highs, lows, closes, self.atr_period)
        kc_u = kc_mid + self.squeeze_kc_mult * atr
        kc_l = kc_mid - self.squeeze_kc_mult * atr

        squeeze_on = bool(bb_u < kc_u and bb_l > kc_l)

        # simple momentum proxy: deviation from mean normalized by ATR
        atr_n = max(1e-12, atr)
        mom = float(closes[-1] - self._sma(closes, self.squeeze_bb_period))
        squeeze_mom = float(np.tanh(mom / (atr_n * 2.0)))

        return {
            "available": True,
            "squeeze_on": squeeze_on,
            "squeeze_release": False,  # filled by caller using prior state
            "squeeze_mom": _clip(squeeze_mom, -1.0, 1.0),
            "bb": {"mid": bb_mid, "u": bb_u, "l": bb_l},
            "kc": {"mid": kc_mid, "u": kc_u, "l": kc_l},
        }

    # ───────── Gemini: MFI ─────────

    def _mfi(self, highs: List[float], lows: List[float], closes: List[float], volumes: List[float], period: int) -> float:
        n = int(max(5, period))
        m = min(len(closes), len(highs), len(lows), len(volumes))
        if m < n + 2:
            return 50.0
        tp = (np.array(highs[-(n + 1):], dtype=float) + np.array(lows[-(n + 1):], dtype=float) + np.array(closes[-(n + 1):], dtype=float)) / 3.0
        vol = np.array(volumes[-(n + 1):], dtype=float)
        mf = tp * vol

        pos = 0.0
        neg = 0.0
        for i in range(1, len(tp)):
            if tp[i] > tp[i - 1]:
                pos += float(mf[i])
            elif tp[i] < tp[i - 1]:
                neg += float(mf[i])
        if neg <= 1e-12:
            return 100.0
        mr = pos / neg
        mfi = 100.0 - (100.0 / (1.0 + mr))
        return float(np.clip(mfi, 0.0, 100.0))

    def _detect_divergence_simple(self, prices: List[float], indicator: List[float], lookback: int) -> Optional[str]:
        n = int(max(12, lookback))
        if len(prices) < n or len(indicator) < n:
            return None
        p = np.array(prices[-n:], dtype=float)
        i = np.array(indicator[-n:], dtype=float)
        half = n // 2
        if half < 5:
            return None

        p_low1_idx = int(np.argmin(p[:half]))
        p_high1_idx = int(np.argmax(p[:half]))
        p_low2_idx = int(np.argmin(p[half:])) + half
        p_high2_idx = int(np.argmax(p[half:])) + half

        # bullish divergence: lower low in price, higher low in indicator
        if p[p_low2_idx] < p[p_low1_idx] and i[p_low2_idx] > i[p_low1_idx]:
            return "bullish"
        # bearish divergence: higher high in price, lower high in indicator
        if p[p_high2_idx] > p[p_high1_idx] and i[p_high2_idx] < i[p_high1_idx]:
            return "bearish"
        return None

    # ───────── Structure context (lightweight) ─────────

    def _find_support_resistance(self, highs: List[float], lows: List[float], lookback: int) -> Tuple[List[float], List[float]]:
        n = int(max(20, lookback))
        if len(highs) < n or len(lows) < n:
            return [], []
        h = highs[-n:]
        l = lows[-n:]
        supports: List[float] = []
        resistances: List[float] = []

        # simple swing-point detection
        for i in range(2, n - 2):
            if l[i] < l[i - 1] and l[i] < l[i + 1] and l[i] < l[i - 2] and l[i] < l[i + 2]:
                supports.append(float(l[i]))
            if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > h[i - 2] and h[i] > h[i + 2]:
                resistances.append(float(h[i]))

        # Deduplicate nearby levels
        def _dedup(levels: List[float], tol: float) -> List[float]:
            out: List[float] = []
            for lv in sorted(levels):
                if not out or abs(lv - out[-1]) > tol:
                    out.append(lv)
            return out

        atr = self._atr(highs, lows, (highs if highs else lows), self.atr_period)
        tol = max(1e-6, atr * 0.35)
        return _dedup(supports, tol)[:8], _dedup(resistances, tol)[:8]

    def _structure_context(self, price: float, highs: List[float], lows: List[float], closes: List[float]) -> Dict[str, Any]:
        if len(closes) < max(30, self.sr_lookback):
            return {"available": False}
        atr = self._atr(highs, lows, closes, self.atr_period)
        supports, resistances = self._find_support_resistance(highs, lows, self.sr_lookback)
        if atr <= 1e-12:
            return {"available": True, "support": supports[:3], "resistance": resistances[:3], "breakout_score": 0.0, "near_support": False, "near_resistance": False}

        prox = self.sr_proximity_atr * atr
        near_sup = any(abs(price - s) <= prox for s in supports)
        near_res = any(abs(price - r) <= prox for r in resistances)

        breakout_score = 0.0
        if resistances and price > max(resistances) + 0.5 * atr:
            breakout_score = 1.0
        elif supports and price < min(supports) - 0.5 * atr:
            breakout_score = -1.0

        in_range = bool((near_sup or near_res) and breakout_score == 0.0)
        return {
            "available": True,
            "support": supports[:3],
            "resistance": resistances[:3],
            "near_support": bool(near_sup),
            "near_resistance": bool(near_res),
            "breakout_score": float(breakout_score),
            "in_range": bool(in_range),
        }

    # ───────── Trend context (confidence only) ─────────

    def _trend_context(self, closes: List[float]) -> Dict[str, Any]:
        if len(closes) < max(self.trend_ema_fast, self.trend_ema_slow) + 3:
            return {"available": False}
        ema_fast = self._ema_series(closes, self.trend_ema_fast)
        ema_slow = self._ema_series(closes, self.trend_ema_slow)
        fast = float(ema_fast[-1])
        slow = float(ema_slow[-1])
        slope = float(ema_fast[-1] - ema_fast[-3]) if len(ema_fast) >= 3 else 0.0
        regime = "bull" if fast > slow else "bear" if fast < slow else "flat"
        return {"available": True, "ema_fast": fast, "ema_slow": slow, "slope": slope, "regime": regime}

    # ───────── MTF momentum alignment ─────────

    def _mtf_momentum_alignment(self, instrument: str) -> Dict[str, Any]:
        """
        Weighted momentum alignment across configured timeframes.
        Returns alignment in [-1, +1] where sign is direction and magnitude is agreement strength.
        """
        scores: Dict[str, Dict[str, Any]] = {}
        weighted_sum = 0.0
        total_w = 0.0

        for tf in self.mtf_timeframes:
            rec = self._get_tf_rec(instrument, tf)
            if not isinstance(rec, dict):
                continue
            closes = rec.get("close")
            highs = rec.get("high")
            lows = rec.get("low")
            if not isinstance(closes, (list, tuple, np.ndarray)) or len(closes) < 60:
                continue
            closes_f = [float(x) for x in closes[-200:]]
            highs_f = [float(x) for x in highs[-200:]] if isinstance(highs, (list, tuple, np.ndarray)) else closes_f
            lows_f = [float(x) for x in lows[-200:]] if isinstance(lows, (list, tuple, np.ndarray)) else closes_f

            # HMA slope
            fast_series = self._hma_series(closes_f, self.fast_period)
            slope = self._linreg_slope_norm(fast_series, lookback=min(20, max(8, self.fast_slope_lookback)))

            # RSI bias
            rsi = self._rsi_wilder(closes_f, 14)
            rsi_bias = _clip((rsi - 50.0) / 50.0, -1.0, 1.0)

            # MACD histogram strength
            _, _, macd_hist = self._macd(closes_f)
            macd_s = _clip(macd_hist * 120.0, -1.0, 1.0)

            tf_mom = 0.50 * slope + 0.25 * rsi_bias + 0.25 * macd_s
            tf_mom = _clip(tf_mom, -1.0, 1.0)

            w = float(self.mtf_tf_weights.get(tf, 0.5))
            weighted_sum += tf_mom * w
            total_w += w
            scores[tf] = {"tf_momentum": tf_mom, "weight": w, "slope": slope, "rsi": rsi, "macd_hist": macd_hist}

        if total_w <= 1e-12 or not scores:
            return {"available": False, "alignment": 0.0, "details": {}}

        aligned = weighted_sum / total_w
        return {"available": True, "alignment": float(_clip(aligned, -1.0, 1.0)), "details": scores}

    # ───────── Cross asset (optional, graceful) ─────────

    def _asset_momentum_score(self, symbol: str, tf: str) -> Optional[float]:
        sym = normalize_instrument(symbol)
        rec = self._get_tf_rec(sym, tf)
        if not isinstance(rec, dict):
            return None
        closes = rec.get("close")
        if not isinstance(closes, (list, tuple, np.ndarray)) or len(closes) < 80:
            return None
        closes_f = [float(x) for x in closes[-200:]]
        series = self._hma_series(closes_f, self.fast_period)
        slope = self._linreg_slope_norm(series, lookback=min(24, max(10, self.fast_slope_lookback)))
        rsi = self._rsi_wilder(closes_f, 14)
        rsi_bias = _clip((rsi - 50.0) / 50.0, -1.0, 1.0)
        _, _, macd_hist = self._macd(closes_f)
        macd_s = _clip(macd_hist * 120.0, -1.0, 1.0)
        score = _clip(0.50 * slope + 0.25 * rsi_bias + 0.25 * macd_s, -1.0, 1.0)
        return float(score)

    def _cross_asset_alignment(self, primary_action: str) -> Dict[str, Any]:
        if not self.use_cross_asset or primary_action not in ("long", "short"):
            return {"available": False}

        dxy_sym = str(self.cross_asset_symbols.get("dxy", "DXY"))
        us10y_sym = str(self.cross_asset_symbols.get("us10y", "US10Y"))

        dxy = self._asset_momentum_score(dxy_sym, self.cross_asset_tf)
        y10 = self._asset_momentum_score(us10y_sym, self.cross_asset_tf)

        if dxy is None and y10 is None:
            return {"available": False}

        # Gold often inverse to USD; yields relationship can vary, but commonly rising yields pressure gold.
        alignment = 0.0
        if primary_action == "long":
            if dxy is not None:
                alignment += (-dxy) * 0.6
            if y10 is not None:
                alignment += (-y10) * 0.4
        else:  # short
            if dxy is not None:
                alignment += (dxy) * 0.6
            if y10 is not None:
                alignment += (y10) * 0.4

        alignment = float(_clip(alignment, -1.0, 1.0))
        conf_delta = float(_clip(alignment * self.cross_asset_conf_scale, -0.20, 0.20))
        return {"available": True, "alignment": alignment, "dxy_momentum": dxy, "us10y_momentum": y10, "conf_delta": conf_delta}

    # ───────── Whipsaw / persistence ─────────

    def _whipsaw_and_persistence_adjustment(self, instrument: str, direction: int) -> Tuple[float, Dict[str, Any]]:
        hist = self._signal_history.get(instrument)
        if hist is None:
            return 0.0, {"whipsaw": 0.0, "persistence": 0.0}

        nz = [d for d in hist if d != 0]
        flips = 0
        for i in range(1, len(nz)):
            if nz[i] != nz[i - 1]:
                flips += 1

        whipsaw = 0.0
        if len(nz) >= 4:
            flip_rate = flips / max(1, len(nz) - 1)
            whipsaw = -self.whipsaw_penalty * min(1.0, flip_rate * 1.5)

        persistence = 0.0
        window = list(hist)[-self.persistence_window:] if len(hist) >= 1 else []
        if direction != 0 and window:
            same = sum(1 for d in window if d == direction)
            ratio = same / max(1, len(window))
            if ratio >= 0.8:
                persistence = self.persistence_bonus * ratio

        return whipsaw + persistence, {"whipsaw": whipsaw, "persistence": persistence, "flip_count": flips, "nonzero_len": len(nz)}

    # ───────── Exhaustion ─────────

    def _momentum_exhaustion(self, rsi: float, stoch_k: float, atr_z: float, volume_ratio: float, range_ratio: float) -> float:
        scores: List[float] = []

        # RSI extremes
        if rsi > 85 or rsi < 15:
            scores.append(0.85)
        elif rsi > 80 or rsi < 20:
            scores.append(0.65)

        # Stoch extremes
        if stoch_k > 90 or stoch_k < 10:
            scores.append(0.75)
        elif stoch_k > 85 or stoch_k < 15:
            scores.append(0.55)

        # ATR z-score spike
        if atr_z > 2.0:
            scores.append(0.65)
        elif atr_z > 1.5:
            scores.append(0.45)

        # Volume spike
        if volume_ratio > 2.0:
            scores.append(0.55)
        elif volume_ratio > 1.6:
            scores.append(0.35)

        # Range spike
        if range_ratio > 1.8:
            scores.append(0.45)
        elif range_ratio > 1.5:
            scores.append(0.30)

        return float(max(scores) if scores else 0.0)

    # ───────── Composite scoring ─────────

    def _componentize(
        self,
        *,
        hma_slope: float,
        roc_weighted: float,
        adaptive_thr: float,
        rsi: float,
        macd_hist: float,
        stoch_k: float,
        obv_mom: float,
        mfi: float,
        fisher: float,
        fisher_trigger: float,
        squeeze_mom: float,
        structure_score: float,
        vol_conf: float,
    ) -> Dict[str, float]:
        """
        Map raw indicators into roughly normalized components in [-1, +1].
        """
        # Directional factors
        c_hma = _clip(hma_slope, -1.0, 1.0)

        # Normalize roc by adaptive threshold
        denom = max(1e-12, adaptive_thr * 3.0)
        c_roc = _clip(roc_weighted / denom, -1.0, 1.0)

        c_rsi = _clip((rsi - 50.0) / 50.0, -1.0, 1.0)
        c_macd = _clip(macd_hist * 120.0, -1.0, 1.0)
        c_stoch = _clip((stoch_k - 50.0) / 50.0, -1.0, 1.0)
        c_obv = _clip(obv_mom, -1.0, 1.0)  # already ratio-like

        c_mfi = _clip((mfi - 50.0) / 50.0, -1.0, 1.0)

        # Fisher: sign of fisher - trigger, scaled by fisher magnitude
        fisher_delta = fisher - fisher_trigger
        c_fisher = _clip(np.tanh(fisher_delta * 1.5), -1.0, 1.0)

        # Squeeze momentum already normalized
        c_squeeze = _clip(squeeze_mom, -1.0, 1.0)

        c_struct = _clip(structure_score, -1.0, 1.0)

        # Volume confirmation is not directional: treat as multiplier later
        c_vol = _clip((vol_conf - 0.6) / 0.4, -1.0, 1.0)  # maps ~0.2..1.0

        return {
            "hma_slope": float(c_hma),
            "roc_weighted": float(c_roc),
            "rsi_momentum": float(c_rsi),
            "macd_strength": float(c_macd),
            "stoch_position": float(c_stoch),
            "obv_trend": float(c_obv),
            "mfi_momentum": float(c_mfi),
            "fisher_signal": float(c_fisher),
            "squeeze_momentum": float(c_squeeze),
            "structure": float(c_struct),
            "volume_conf": float(c_vol),
        }

    def _composite_momentum_score(self, comp: Dict[str, float]) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Returns (bull_norm, bear_norm, net, weights_used).
        """
        weights: Dict[str, float] = {
            "hma_slope": 0.25,
            "roc_weighted": 0.18,
            "rsi_momentum": 0.12,
            "macd_strength": 0.12,
            "stoch_position": 0.08,
            "obv_trend": 0.06,
            "mfi_momentum": 0.07 if self.use_mfi else 0.0,
            "fisher_signal": self.fisher_weight if self.use_fisher else 0.0,
            "squeeze_momentum": self.squeeze_weight if self.use_squeeze else 0.0,
            "structure": self.structure_weight if self.use_structure_context else 0.0,
        }

        # Normalize total weight (ignore zeros)
        tw = float(sum(w for w in weights.values() if w > 0.0)) + 1e-12

        bull = 0.0
        bear = 0.0
        for k, w in weights.items():
            if w <= 0.0:
                continue
            v = float(comp.get(k, 0.0))
            if v > 0:
                bull += v * w
            elif v < 0:
                bear += abs(v) * w

        bull_n = bull / tw
        bear_n = bear / tw
        net = bull_n - bear_n
        return float(_clip(bull_n, 0.0, 1.0)), float(_clip(bear_n, 0.0, 1.0)), float(_clip(net, -1.0, 1.0)), weights

    def _momentum_deceleration(self, net_series: Deque[float]) -> Dict[str, Any]:
        """
        Detect momentum deceleration based on net momentum slope change.
        Returns decel in [-1,1], where negative means decelerating.
        """
        arr = list(net_series)
        if len(arr) < 20:
            return {"available": False, "decel": 0.0}
        recent = arr[-6:]
        earlier = arr[-12:-6]
        if len(recent) < 6 or len(earlier) < 6:
            return {"available": False, "decel": 0.0}

        slope_now = self._linreg_slope_norm(recent, lookback=6)
        slope_prev = self._linreg_slope_norm(earlier, lookback=6)
        decel = float(_clip(slope_now - slope_prev, -1.0, 1.0))
        return {"available": True, "decel": decel, "slope_now": slope_now, "slope_prev": slope_prev}

    # ═══════════════════════════ DECISION ═══════════════════════════

    def _determine_action(self, net: float, bull: float, bear: float, accel: float) -> Tuple[str, float, float]:
        """
        action in {long, short, flat}
        confidence and strength in [0..1]
        """
        if net > self.net_band:
            action = "long"
            strength = _clip(0.20 + bull * 0.95, 0.0, 1.0)
            conf = _clip(0.25 + bull * 0.85, self.min_confidence, self.max_confidence)
            if accel > 0.10:
                conf = _clip(conf * 1.03, self.min_confidence, self.max_confidence)
        elif net < -self.net_band:
            action = "short"
            strength = _clip(0.20 + bear * 0.95, 0.0, 1.0)
            conf = _clip(0.25 + bear * 0.85, self.min_confidence, self.max_confidence)
            if accel < -0.10:
                conf = _clip(conf * 1.03, self.min_confidence, self.max_confidence)
        else:
            action = "flat"
            strength = 0.10
            conf = 0.20

        return action, float(conf), float(strength)

    # ═══════════════════════════ CORE VOTING HOOK ═══════════════════════════

    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        name = self.__class__.__name__
        t0 = time.time()

        votes: Dict[str, Dict[str, Any]] = {}
        analysis_by_inst: Dict[str, Dict[str, Any]] = {}

        if self.debug_enabled:
            hist = self._get_historical()
            self._debug_write(
                "tick_start",
                {
                    "market_data_keys": list(market_data.keys()) if isinstance(market_data, dict) else str(type(market_data)),
                    "historical_symbols": list(hist.keys())[:25] if isinstance(hist, dict) else None,
                    "primary_tf": self.primary_tf,
                    "use_forming_bar": self.use_forming_bar,
                },
            )

        inst = "XAUUSD"
        try:
            ohlcv = self._get_ohlcv_series(inst, self.primary_tf)
            ohlcv = self._maybe_apply_forming_bar(inst, self.primary_tf, ohlcv)

            closes = (ohlcv.get("close") or [])[-300:]
            highs = (ohlcv.get("high") or closes)[-300:]
            lows = (ohlcv.get("low") or closes)[-300:]
            vols = (ohlcv.get("volume") or ([1.0] * len(closes)))[-300:]

            if len(closes) < max(80, max(self.roc_periods) + 10):
                v = {
                    "instrument": inst,
                    "action": "flat",
                    "confidence": 0.10,
                    "magnitude": 0.0,
                    "signal_strength": 0.0,
                    "rationale": f"Insufficient data: closes={len(closes)}",
                }
                votes[inst] = v
                analysis_by_inst[inst] = {"status": "insufficient_data", "closes_len": len(closes)}
                self._signal_history[inst].append(0)
                return self._neutral_or_single(votes, analysis_by_inst, leader_inst=inst, t0=t0)

            # Cache validation extends beyond closes (include last vol/high/low)
            cached = self._get_cached_indicators(inst, closes)
            cache_ok = False
            if isinstance(cached, dict):
                if (
                    _safe_float(cached.get("vol_last"), -999) == _safe_float(vols[-1], -998)
                    and _safe_float(cached.get("high_last"), -999) == _safe_float(highs[-1], -998)
                    and _safe_float(cached.get("low_last"), -999) == _safe_float(lows[-1], -998)
                ):
                    cache_ok = True

            if cache_ok:
                pack = cached
            else:
                # Volatility scaling
                rvol = self._realized_vol(closes, self.vol_lookback)
                atr = self._atr(highs, lows, closes, self.atr_period)
                vol_scale = (rvol / self.vol_ref) if self.vol_ref > 1e-12 else 1.0
                vol_scale = float(np.clip(vol_scale, 0.6, 2.5))
                adaptive_thr = float(self.base_momentum_threshold * vol_scale)

                # Fast trigger slope (HMA)
                fast_series = self._hma_series(closes, self.fast_period) if self.fast_line_mode == "hma" else self._ema_series(closes, self.fast_period)
                hma_slope = float(self._linreg_slope_norm(fast_series, self.fast_slope_lookback))

                # Primary indicators
                roc = self._roc_multi_vw(closes, vols)
                roc_weighted = float(sum(roc.get(p, 0.0) * w for p, w in zip(self.roc_periods, self.roc_weights)))

                rsi = float(self._rsi_wilder(closes, self.rsi_period))
                macd_line, macd_sig, macd_hist = self._macd(closes)
                st_k, st_d = self._stochastic(closes, highs, lows)

                obv = self._obv(closes, vols)

                st = self.instrument_state[inst]
                st["rsi_history"].append(rsi)
                st["macd_line_history"].append(macd_line)
                st["macd_hist_history"].append(macd_hist)
                st["obv_history"].append(obv)

                obv_hist = list(st["obv_history"])
                obv_mom = 0.0
                if len(obv_hist) >= 6:
                    obv_mom = float((obv_hist[-1] - obv_hist[-6]) / (abs(obv_hist[-6]) + 1e-10))
                    obv_mom = _clip(obv_mom, -1.0, 1.0)

                # Volume confirmation (quality multiplier only)
                vol_conf = self._volume_confirmation(closes, vols)

                # Fisher
                fisher, fisher_tr = (0.0, 0.0)
                if self.use_fisher:
                    fisher, fisher_tr = self._fisher_transform(closes, self.fisher_period)

                # Squeeze
                squeeze = {"available": False, "squeeze_on": False, "squeeze_release": False, "squeeze_mom": 0.0}
                if self.use_squeeze:
                    squeeze = self._check_squeeze(closes, highs, lows)
                    prev_on = bool(st.get("squeeze_on_prev", False))
                    now_on = bool(squeeze.get("squeeze_on", False))
                    squeeze["squeeze_release"] = bool(prev_on and not now_on)
                    st["squeeze_on_prev"] = now_on

                # MFI
                mfi = 50.0
                if self.use_mfi:
                    mfi = float(self._mfi(highs, lows, closes, vols, self.mfi_period))
                    st["mfi_history"].append(mfi)

                # Structure
                struct = {"available": False, "breakout_score": 0.0}
                if self.use_structure_context:
                    struct = self._structure_context(closes[-1], highs, lows, closes)

                # CHOP
                chop = float(self._choppiness_index(highs, lows, closes, self.chop_period)) if self.use_chop_filter else 50.0

                # Exhaustion components
                atr_z = float(self._atr_zscore(highs, lows, closes, self.exhaustion_lookback))
                # volume ratio
                vol_ratio = 1.0
                if len(vols) >= 21:
                    vol_ratio = float(vols[-1] / (float(np.mean(vols[-20:])) + 1e-12))
                # range ratio
                range_ratio = 1.0
                if len(highs) >= 21 and len(lows) >= 21:
                    last_rng = float(highs[-1] - lows[-1])
                    avg_rng = float(np.mean([(highs[-i] - lows[-i]) for i in range(2, 21)])) + 1e-12
                    range_ratio = float(last_rng / avg_rng)

                exhaustion = 0.0
                if self.use_exhaustion_filter:
                    exhaustion = float(self._momentum_exhaustion(rsi, st_k, atr_z, vol_ratio, range_ratio))

                pack = {
                    "roc": roc,
                    "roc_weighted": roc_weighted,
                    "hma_slope": hma_slope,
                    "rsi": rsi,
                    "macd_line": macd_line,
                    "macd_signal": macd_sig,
                    "macd_hist": macd_hist,
                    "stoch_k": st_k,
                    "stoch_d": st_d,
                    "obv": obv,
                    "obv_momentum": obv_mom,
                    "volume_confirmation": vol_conf,
                    "realized_vol": rvol,
                    "atr": atr,
                    "vol_scale": vol_scale,
                    "adaptive_threshold": adaptive_thr,
                    "fisher": fisher,
                    "fisher_trigger": fisher_tr,
                    "squeeze": squeeze,
                    "mfi": mfi,
                    "structure": struct,
                    "chop": chop,
                    "atr_z": atr_z,
                    "vol_ratio": vol_ratio,
                    "range_ratio": range_ratio,
                    "exhaustion": exhaustion,
                    # for cache validation:
                    "vol_last": _safe_float(vols[-1], 0.0),
                    "high_last": _safe_float(highs[-1], 0.0),
                    "low_last": _safe_float(lows[-1], 0.0),
                }
                self._set_cached_indicators(inst, closes, pack)

            pack = pack if isinstance(pack, dict) else {}

            # Build components and composite scores
            adaptive_thr = float(pack.get("adaptive_threshold", self.base_momentum_threshold))
            comp = self._componentize(
                hma_slope=float(pack.get("hma_slope", 0.0)),
                roc_weighted=float(pack.get("roc_weighted", 0.0)),
                adaptive_thr=adaptive_thr,
                rsi=float(pack.get("rsi", 50.0)),
                macd_hist=float(pack.get("macd_hist", 0.0)),
                stoch_k=float(pack.get("stoch_k", 50.0)),
                obv_mom=float(pack.get("obv_momentum", 0.0)),
                mfi=float(pack.get("mfi", 50.0)),
                fisher=float(pack.get("fisher", 0.0)),
                fisher_trigger=float(pack.get("fisher_trigger", 0.0)),
                squeeze_mom=float((pack.get("squeeze") or {}).get("squeeze_mom", 0.0)) if isinstance(pack.get("squeeze"), dict) else 0.0,
                structure_score=float((pack.get("structure") or {}).get("breakout_score", 0.0)) if isinstance(pack.get("structure"), dict) else 0.0,
                vol_conf=float(pack.get("volume_confirmation", 0.5)),
            )
            bull, bear, net, weights_used = self._composite_momentum_score(comp)

            # Save net history and compute deceleration (and "acceleration" proxy)
            st = self.instrument_state[inst]
            st["net_history"].append(float(net))
            decel_meta = self._momentum_deceleration(st["net_history"])
            decel = float(decel_meta.get("decel", 0.0)) if decel_meta.get("available") else 0.0
            accel_proxy = float(self._linreg_slope_norm(list(st["net_history"])[-10:], lookback=10)) if len(st["net_history"]) >= 12 else 0.0

            # Apply deceleration penalty (reduce both sides)
            decel_pen = 0.0
            if self.use_deceleration and decel_meta.get("available") and decel <= self.decel_thr:
                scale = max(0.35, 1.0 - abs(decel) * self.decel_penalty)
                bull *= scale
                bear *= scale
                net = bull - bear
                decel_pen = 1.0 - scale

            # Volume confirmation multiplier (quality)
            vol_conf = float(pack.get("volume_confirmation", 0.5))
            quality_mult = _clip(0.85 + vol_conf * 0.30, 0.75, 1.15)
            bull = _clip(bull * quality_mult, 0.0, 1.0)
            bear = _clip(bear * quality_mult, 0.0, 1.0)
            net = _clip(bull - bear, -1.0, 1.0)

            # Initial action/conf/strength from net + bull/bear
            action, conf, strength = self._determine_action(net, bull, bear, accel_proxy)
            direction = 1 if action == "long" else -1 if action == "short" else 0

            # MFI divergence veto (optional)
            mfi_div = None
            if self.use_mfi and self.use_mfi_divergence_veto and action in ("long", "short"):
                mfi_hist = list(st.get("mfi_history", []))
                if len(mfi_hist) >= self.mfi_div_lookback and len(closes) >= self.mfi_div_lookback:
                    mfi_div = self._detect_divergence_simple(closes, mfi_hist, self.mfi_div_lookback)
                    if action == "long" and mfi_div == "bearish":
                        action = "flat"
                        direction = 0
                        conf *= 0.55
                        strength *= 0.45
                    elif action == "short" and mfi_div == "bullish":
                        action = "flat"
                        direction = 0
                        conf *= 0.55
                        strength *= 0.45

            # Fisher extreme (exhaustion-style cap)
            fisher = float(pack.get("fisher", 0.0))
            fisher_extreme_hit = False
            if self.use_fisher and abs(fisher) >= self.fisher_extreme and action in ("long", "short"):
                fisher_extreme_hit = True
                conf *= 0.70
                strength *= 0.80

            # Squeeze context (prefer trades on release)
            squeeze = pack.get("squeeze", {})
            squeeze_release = bool(squeeze.get("squeeze_release", False)) if isinstance(squeeze, dict) else False
            squeeze_on = bool(squeeze.get("squeeze_on", False)) if isinstance(squeeze, dict) else False
            if self.use_squeeze and action in ("long", "short"):
                if squeeze_on:
                    # during squeeze, reduce aggressiveness (avoid chop inside compression)
                    conf *= 0.85
                    strength *= 0.85
                elif squeeze_release:
                    # just released -> small boost
                    conf = _clip(conf + 0.04, self.min_confidence, self.max_confidence)
                    strength = _clip(strength * 1.05, 0.0, 1.0)

            # CHOP veto (hard)
            chop = float(pack.get("chop", 50.0))
            chop_veto = False
            if self.use_chop_filter and chop >= self.chop_thr:
                chop_veto = True
                action = "flat"
                direction = 0
                strength = max(0.0, strength * (1.0 - self.chop_strength_penalty))
                conf = max(self.min_confidence, conf - self.chop_conf_penalty)

            # Exhaustion filter (hard at high exhaustion)
            exhaustion = float(pack.get("exhaustion", 0.0))
            exhaustion_forced = False
            if self.use_exhaustion_filter and action in ("long", "short"):
                if exhaustion >= self.exhaustion_force_flat_thr:
                    exhaustion_forced = True
                    action = "flat"
                    direction = 0
                    conf *= self.exhaustion_conf_mult
                    strength *= 0.55
                elif exhaustion >= 0.55:
                    conf *= (0.80 + (1.0 - exhaustion) * 0.25)
                    strength *= (0.85 + (1.0 - exhaustion) * 0.15)

            # MTF confirmation (confidence only)
            mtf_adj = 0.0
            mtf_meta = {"available": False}
            if self.use_mtf_confirmation and action in ("long", "short"):
                mtf_meta = self._mtf_momentum_alignment(inst)
                if mtf_meta.get("available"):
                    align = float(mtf_meta.get("alignment", 0.0))
                    if action == "long":
                        mtf_adj = (self.mtf_agreement_bonus * abs(align)) if align > 0 else (-self.mtf_disagreement_penalty * abs(align))
                    else:
                        mtf_adj = (self.mtf_agreement_bonus * abs(align)) if align < 0 else (-self.mtf_disagreement_penalty * abs(align))
                    conf = _clip(conf + mtf_adj, self.min_confidence, self.max_confidence)

            # Trend filter (confidence only)
            trend_pen = 0.0
            trend_meta = {"available": False}
            if self.use_trend_filter and action in ("long", "short"):
                trend_meta = self._trend_context(closes)
                if trend_meta.get("available"):
                    regime = trend_meta.get("regime", "flat")
                    slope = _safe_float(trend_meta.get("slope"), 0.0)
                    if action == "long" and (regime == "bear" or slope < 0.0):
                        trend_pen = -self.counter_trend_penalty
                    elif action == "short" and (regime == "bull" or slope > 0.0):
                        trend_pen = -self.counter_trend_penalty
                    conf = _clip(conf + trend_pen, self.min_confidence, self.max_confidence)

            # Cross-asset (optional; confidence only)
            xasset = {"available": False}
            if self.use_cross_asset and action in ("long", "short"):
                xasset = self._cross_asset_alignment(action)
                if xasset.get("available"):
                    conf = _clip(conf + float(xasset.get("conf_delta", 0.0)), self.min_confidence, self.max_confidence)

            # Whipsaw / persistence
            wh_adj, wh_meta = self._whipsaw_and_persistence_adjustment(inst, direction)
            conf = _clip(conf + wh_adj, self.min_confidence, self.max_confidence)

            # Update signal history
            self._signal_history[inst].append(direction)

            # Final thesis
            thesis = (
                f"{inst}: {action} | net={net:+.2f} bull={bull:.2f} bear={bear:.2f} | "
                f"conf={conf:.2f} str={strength:.2f} | "
                f"CHOP={chop:.1f}{' veto' if chop_veto else ''} | "
                f"exh={exhaustion:.2f}{' flat' if exhaustion_forced else ''} | "
                f"decel={decel:.2f} pen={decel_pen:.2f} | "
                f"mtf_adj={mtf_adj:+.2f} trend_pen={trend_pen:+.2f} wh={wh_adj:+.2f}"
            )

            votes[inst] = {
                "instrument": inst,
                "action": action,
                "confidence": float(conf),
                "magnitude": float(strength),
                "signal_strength": float(strength),
                "rationale": thesis,
            }

            analysis_by_inst[inst] = {
                "status": "ok",
                "tf": self.primary_tf,
                "action": action,
                "direction": direction,
                "confidence": float(conf),
                "signal_strength": float(strength),
                "thesis": thesis,
                "bullish_confluence": float(bull),
                "bearish_confluence": float(bear),
                "net_momentum": float(net),
                "acceleration": float(accel_proxy),
                "deceleration": decel_meta,
                "decel_pen": float(decel_pen),
                "components": comp,
                "weights": weights_used,
                "chop": float(chop),
                "exhaustion": float(exhaustion),
                "atr_z": float(pack.get("atr_z", 0.0)),
                "vol_ratio": float(pack.get("vol_ratio", 1.0)),
                "range_ratio": float(pack.get("range_ratio", 1.0)),
                "rsi": float(pack.get("rsi", 50.0)),
                "macd_hist": float(pack.get("macd_hist", 0.0)),
                "stoch_k": float(pack.get("stoch_k", 50.0)),
                "mfi": float(pack.get("mfi", 50.0)),
                "mfi_divergence": mfi_div,
                "fisher": {"val": fisher, "trigger": float(pack.get("fisher_trigger", 0.0)), "extreme": fisher_extreme_hit},
                "squeeze": pack.get("squeeze", {"available": False}),
                "structure": pack.get("structure", {"available": False}),
                "mtf": mtf_meta,
                "trend": trend_meta,
                "xasset": xasset,
                "whipsaw": wh_meta,
            }

            # Forensic debug
            if self.debug_enabled:
                dump_n = max(20, min(self.debug_dump_bars, 200))
                self._debug_write(
                    "momentum_forensic",
                    {
                        "instrument": inst,
                        "tf": self.primary_tf,
                        "prices_len": len(closes),
                        "prices_tail": closes[-dump_n:],
                        "high_tail": highs[-dump_n:],
                        "low_tail": lows[-dump_n:],
                        "vol_tail": vols[-dump_n:],
                        "analysis": analysis_by_inst[inst],
                    },
                )

        except Exception as e:
            votes[inst] = {
                "instrument": inst,
                "action": "flat",
                "confidence": 0.10,
                "magnitude": 0.0,
                "signal_strength": 0.0,
                "rationale": f"{inst}: flat (error: {type(e).__name__}: {e})",
            }
            analysis_by_inst[inst] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            self._signal_history[inst].append(0)
            self._debug_write("instrument_error", {"instrument": inst, "error": f"{type(e).__name__}: {e}"})

        return self._neutral_or_single(votes, analysis_by_inst, leader_inst="XAUUSD", t0=t0)

    def _neutral_or_single(self, votes: Dict[str, Dict[str, Any]], analysis_by_inst: Dict[str, Dict[str, Any]], leader_inst: str, t0: float) -> Dict[str, Any]:
        name = self.__class__.__name__

        if not votes:
            return self._neutral_proposal("No votes produced")

        leader = leader_inst if leader_inst in votes else next(iter(votes.keys()))
        leader_vote = votes[leader]
        global_action = str(leader_vote.get("action", "flat"))
        global_strength = _safe_float(leader_vote.get("signal_strength"), 0.0)
        thesis = str(leader_vote.get("rationale", f"leader={leader}"))

        # PPOObservationBuilder strict mode requires `momentum_analysis.per_instrument[XAUUSD].rsi`.
        sanitized_analysis: Dict[str, Dict[str, Any]] = {}
        for inst, blob in (analysis_by_inst.items() if isinstance(analysis_by_inst, dict) else []):
            inst_norm = normalize_instrument(inst)
            b = blob if isinstance(blob, dict) else {}
            out_b = dict(b)
            try:
                out_b["rsi"] = float(out_b.get("rsi", 50.0))
            except Exception:
                out_b["rsi"] = 50.0
            if "divergence_signal" not in out_b and "divergence" in out_b:
                out_b["divergence_signal"] = out_b.get("divergence")
            out_b.setdefault("divergence_signal", "neutral")
            sanitized_analysis[inst_norm] = out_b

        if "XAUUSD" not in sanitized_analysis:
            sanitized_analysis["XAUUSD"] = {
                "rsi": 50.0,
                "divergence_signal": "neutral",
                "net_momentum": 0.0,
                "direction": 0,
                "acceleration": 0.0,
            }
        analysis_by_inst = sanitized_analysis

        # Publish side-channel (per-instrument votes are published by VotingExpertBase)
        try:
            leader_analysis = analysis_by_inst.get(leader, {}) if isinstance(analysis_by_inst, dict) else {}
            self.smart_bus.set(
                "momentum_analysis",
                {
                    "composite_momentum": float(leader_analysis.get("net_momentum", 0.0)),
                    "direction": int(leader_analysis.get("direction", 0)),
                    "acceleration": float(leader_analysis.get("acceleration", 0.0)),
                    "leader_instrument": leader,
                    "per_instrument": analysis_by_inst,
                    "timestamp": _dt.datetime.now().isoformat(),
                },
                module=name,
                thesis=f"Momentum analysis leader={leader}",
            )
        except Exception as e:
            self._debug_write("bus_publish_error", {"error": f"{type(e).__name__}: {e}"})

        elapsed_ms = (time.time() - t0) * 1000.0
        self._debug_write(
            "tick_end",
            {"leader": leader, "global_action": global_action, "global_signal_strength": global_strength, "elapsed_ms": elapsed_ms},
        )

        max_strength = float(getattr(self, "max_signal_strength", 1.0))
        return {
            "action": global_action,
            "signal_strength": float(global_strength),
            "position_size": float(min(max_strength, global_strength * 0.6)),
            "leader_instrument": leader,
            "reason": thesis,
            "proposals": votes,            # votes only (base consumes)
            "per_instrument": votes,       # votes only
            "analysis": analysis_by_inst,  # rich side-channel
        }

    async def _calculate_expert_specific_confidence(self, proposal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """
        Base class still applies final gating; this is a sensible leader-based estimate.
        """
        try:
            action = str(proposal.get("action", "flat"))
            sig = _safe_float(proposal.get("signal_strength"), 0.0)
            base = 0.25 + sig * 0.55

            leader = str(proposal.get("leader_instrument", "XAUUSD"))
            analysis = proposal.get("analysis", {})
            leader_a = analysis.get(leader, {}) if isinstance(analysis, dict) else {}

            bull = _safe_float(leader_a.get("bullish_confluence"), 0.0)
            bear = _safe_float(leader_a.get("bearish_confluence"), 0.0)
            chop = _safe_float(leader_a.get("chop"), 50.0)
            exhaustion = _safe_float(leader_a.get("exhaustion"), 0.0)

            confluence = bull if action == "long" else bear if action == "short" else 0.0
            base *= 0.90 + _clip(confluence, 0.0, 1.0) * 0.35

            if self.use_chop_filter and chop >= self.chop_thr:
                base *= 0.65
            if self.use_exhaustion_filter and exhaustion >= 0.65:
                base *= 0.80

            return float(_clip(base, 0.15, 0.95))
        except Exception:
            return 0.40

    # ═══════════════════════════ NEUTRAL HELPERS ═══════════════════════════

    def _neutral_proposal(self, reason: str) -> Dict[str, Any]:
        name = self.__class__.__name__
        inst = "XAUUSD"
        per_inst_votes = {
            inst: {
                "instrument": inst,
                "action": "flat",
                "confidence": 0.10,
                "magnitude": 0.0,
                "signal_strength": 0.0,
                "rationale": f"{inst}: flat (neutral: {reason})",
            }
        }
        per_inst_analysis = {
            inst: {
                "rsi": 50.0,
                "divergence_signal": "neutral",
                "net_momentum": 0.0,
                "direction": 0,
                "acceleration": 0.0,
                "action": "flat",
                "confidence": 0.10,
            }
        }
        try:
            self.smart_bus.set(
                f"{name}_per_instrument_votes",
                per_inst_votes,
                module=name,
                thesis=f"Neutral: {reason}",
            )
            self.smart_bus.set(
                "momentum_analysis",
                {
                    "composite_momentum": 0.0,
                    "direction": 0,
                    "acceleration": 0.0,
                    "leader_instrument": inst,
                    "per_instrument": per_inst_analysis,
                    "timestamp": _dt.datetime.now().isoformat(),
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
            "proposals": per_inst_votes,
            "per_instrument": per_inst_votes,
            "analysis": per_inst_analysis,
        }

    def _volume_confirmation(self, prices: List[float], volumes: List[float]) -> float:
        if len(prices) < 25 or len(volumes) < 25:
            return 0.5
        recent = float(np.mean(volumes[-5:]))
        avg = float(np.mean(volumes[-20:]))
        if avg <= 1e-12:
            return 0.5
        vol_ratio = recent / avg
        price_change = prices[-1] - prices[-5]
        if (price_change > 0 and volumes[-1] >= recent) or (price_change < 0 and volumes[-1] >= recent):
            score = 0.5 + min(0.5, 0.20 * vol_ratio)
        else:
            score = 0.5 - min(0.3, 0.15 * vol_ratio)
        return float(np.clip(score, 0.2, 1.0))

    # ═══════════════════════════ process wrapper ═══════════════════════════

    async def process(self, **inputs) -> Dict[str, Any]:
        out = await super().process(**inputs)
        name = self.__class__.__name__

        proposal = out.get("voting_proposal") or out.get(f"{name}_voting_proposal", {})
        confidence = out.get("confidence", out.get(f"{name}_confidence", 0.0))

        analysis = out.get("momentum_analysis")
        if analysis is None:
            try:
                analysis = self.smart_bus.get("momentum_analysis", name, default=None)
            except Exception:
                analysis = None
        if analysis is None and isinstance(proposal, dict):
            analysis = {
                "composite_momentum": 0.0,
                "direction": 0,
                "acceleration": 0.0,
                "per_instrument": proposal.get("analysis", {}),
                "timestamp": _dt.datetime.now().isoformat(),
            }

        out.setdefault("momentum_voting_proposal", proposal)
        out.setdefault("momentum_confidence", confidence)
        out.setdefault("momentum_analysis", analysis)

        if self.debug_enabled:
            out.setdefault("momentum_debug", {"enabled": True, "path": self.debug_path})

        return out

    # ═══════════════════════════ STATE PERSISTENCE ═══════════════════════════

    def _get_custom_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"instrument_state": {}, "signal_history": {}}
        for inst, st in self.instrument_state.items():
            state["instrument_state"][inst] = {
                "rsi_history": list(st.get("rsi_history", []))[-60:],
                "macd_line_history": list(st.get("macd_line_history", []))[-80:],
                "macd_hist_history": list(st.get("macd_hist_history", []))[-80:],
                "mfi_history": list(st.get("mfi_history", []))[-60:],
                "obv_history": list(st.get("obv_history", []))[-60:],
                "net_history": list(st.get("net_history", []))[-120:],
                "squeeze_on_prev": bool(st.get("squeeze_on_prev", False)),
            }
        for inst, h in self._signal_history.items():
            state["signal_history"][inst] = list(h)[-self.signal_hist_len:]
        return state

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        inst_state = state.get("instrument_state", {})
        if isinstance(inst_state, dict):
            for inst, saved in inst_state.items():
                inst_norm = normalize_instrument(inst)
                if inst_norm in self.instrument_state and isinstance(saved, dict):
                    st = self.instrument_state[inst_norm]
                    st["rsi_history"] = deque(saved.get("rsi_history", []), maxlen=80)
                    st["macd_line_history"] = deque(saved.get("macd_line_history", []), maxlen=120)
                    st["macd_hist_history"] = deque(saved.get("macd_hist_history", []), maxlen=120)
                    st["mfi_history"] = deque(saved.get("mfi_history", []), maxlen=80)
                    st["obv_history"] = deque(saved.get("obv_history", []), maxlen=80)
                    st["net_history"] = deque(saved.get("net_history", []), maxlen=200)
                    st["squeeze_on_prev"] = bool(saved.get("squeeze_on_prev", False))

        sig_hist = state.get("signal_history", {})
        if isinstance(sig_hist, dict):
            for inst, arr in sig_hist.items():
                inst_norm = normalize_instrument(inst)
                if inst_norm in self._signal_history and isinstance(arr, list):
                    self._signal_history[inst_norm] = deque([int(x) for x in arr[-self.signal_hist_len:]], maxlen=self.signal_hist_len)

        self.log_info("[MomentumExpert v3.2] state restored")
