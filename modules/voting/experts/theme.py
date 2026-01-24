#!/usr/bin/env python3
"""
Institutional ThemeExpert (v3.2 - Base-Aligned, Regime-Classifier, Debug-Traceable)
==================================================================================

ROLE (by design):
- ThemeExpert is NOT a TrendExpert and NOT a MomentumExpert.
- It classifies "market texture" / regime and emits ONLY per-instrument directional votes:
    long / short / flat
  so that:
    - VotingExpertBase handles position-focus reframing (hold/exit/tighten)
    - Committee decides how to combine Theme with Trend/Momentum.

Core Regime Axes:
1) Persistence (Hurst exponent, R/S multi-scale regression)
   - H > 0.55 : persistent / trend-friendly
   - H < 0.45 : mean-reverting / chop-friendly
2) Volatility (Parkinson volatility, High/Low efficient estimator)
   - uses a vol_ratio vs baseline window

Quadrant Regimes:
- Q1: High Hurst + Low Vol  -> "goldilocks" (trend follow, higher confidence)
- Q2: High Hurst + High Vol -> "panic_mania" (trend but defensive sizing/conf)
- Q3: Low Hurst  + High Vol -> "chop_whipsaw" (KILL SWITCH -> flat)
- Q4: Low Hurst  + Low Vol  -> "drift_noise" (low edge -> mostly flat)

Institutional Filters / Upgrades:
- CHOP veto (choppiness index)
- HMA fast slope (early texture shift)
- Volume-confirmed regime shifts (avoid fake flips)
- MTF regime alignment (M15/H1/H4/D1 weighted)
- Optional cross-asset economic proxy (DXY/SPX/US10Y) if present on bus
- Dedicated JSONL forensic debug trace

Train/Live Parity:
- use_forming_bar defaults to False (closed bars only).
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


THEME_DEBUG_DEFAULT: bool = True  # Developer-level default for ThemeExpert debug trace


@module(**module_args("ThemeExpert"))
class ThemeExpert(VotingExpertBase):
    """
    ThemeExpert v3.2: Institutional regime classifier (base-aligned).

    Emits ONLY:
      - long / short / flat per instrument
    """

    # ──────────────────────────────────────────────────────────────
    # INIT
    # ──────────────────────────────────────────────────────────────

    def _expert_specific_init(self) -> None:
        raw_insts = self.config.get("instruments", ["XAUUSD"])
        if isinstance(raw_insts, str):
            raw_insts = [raw_insts]
        insts = [normalize_instrument(x) for x in (raw_insts or []) if normalize_instrument(x)]

        # De-dup while preserving order
        seen = set()
        self.instruments: List[str] = []
        for x in insts:
            if x not in seen:
                self.instruments.append(x)
                seen.add(x)

        if not self.instruments:
            self.instruments = ["XAUUSD"]

        # XAUUSD-first bias; if someone configures multiple, this stays safe but tuned for XAU behavior.
        self.primary_instrument: str = normalize_instrument(self.config.get("primary_instrument", "XAUUSD") or "XAUUSD")

        # Timeframes
        self.primary_tf: str = str(self.config.get("primary_timeframe", PRIMARY_TIMEFRAME) or PRIMARY_TIMEFRAME)
        self.context_tfs: List[str] = list(self.config.get("context_timeframes", list(CONTEXT_TIMEFRAMES)))
        self.mtf_timeframes: List[str] = [self.primary_tf] + [tf for tf in self.context_tfs if tf != self.primary_tf]
        self.mtf_default_set: List[str] = list(self.config.get("mtf_set", ["M15", "H1", "H4", "D1"]))

        # Forming bar toggle (default off for parity)
        self.use_forming_bar: bool = bool(self.config.get("use_forming_bar", False))

        # Lookbacks
        self.vol_lookback: int = int(self.config.get("vol_lookback", 20))
        self.vol_baseline_mult: int = int(self.config.get("vol_baseline_mult", 3))  # baseline_n = vol_lookback * mult
        self.hurst_lookback: int = int(self.config.get("hurst_lookback", 48))
        self.chop_period: int = int(self.config.get("chop_period", 14))
        self.atr_period: int = int(self.config.get("atr_period", 14))

        # Thresholds (quadrants)
        self.hurst_hi: float = float(self.config.get("hurst_hi", 0.55))
        self.hurst_lo: float = float(self.config.get("hurst_lo", 0.45))
        self.vol_ratio_hi: float = float(self.config.get("vol_ratio_hi", 1.35))
        self.vol_ratio_lo: float = float(self.config.get("vol_ratio_lo", 0.85))

        # CHOP veto
        self.use_chop_filter: bool = bool(self.config.get("use_chop_filter", True))
        self.chop_thr: float = float(self.config.get("chop_thr", 61.8))
        self.chop_chaos_thr: float = float(self.config.get("chop_chaos_thr", 65.0))
        self.chop_conf_mult_transition: float = float(self.config.get("chop_conf_mult_transition", 0.70))
        self.chop_conf_mult_trending: float = float(self.config.get("chop_conf_mult_trending", 1.10))

        # Fast trigger (HMA slope)
        self.fast_period: int = int(self.config.get("fast_period", 8))
        self.slope_lookback: int = int(self.config.get("slope_lookback", 12))

        # Volume confirmation
        self.use_volume_confirmation: bool = bool(self.config.get("use_volume_confirmation", True))
        self.regime_shift_vol_mult: float = float(self.config.get("regime_shift_vol_mult", 1.30))

        # MTF alignment
        self.use_mtf_alignment: bool = bool(self.config.get("use_mtf_alignment", True))
        self.mtf_agreement_bonus: float = float(self.config.get("mtf_agreement_bonus", 0.10))
        self.mtf_disagreement_penalty: float = float(self.config.get("mtf_disagreement_penalty", 0.15))
        self.mtf_weights: Dict[str, float] = dict(self.config.get("mtf_weights", {"M15": 1.0, "H1": 0.8, "H4": 0.6, "D1": 0.4}))

        # Optional cross-asset economic proxy
        self.use_cross_asset: bool = bool(self.config.get("use_cross_asset", True))
        self.cross_assets: Dict[str, str] = dict(self.config.get("cross_assets", {"DXY": "DXY", "SPX": "SPX", "US10Y": "US10Y"}))
        self.cross_asset_conf_weight: float = float(self.config.get("cross_asset_conf_weight", 0.12))

        # Persistence / hysteresis
        self.min_regime_persistence: int = int(self.config.get("min_regime_persistence", 3))
        self.max_regime_history: int = int(self.config.get("max_regime_history", 30))

        # Decision band
        self.bias_band: float = float(self.config.get("bias_band", 0.12))

        # Debug trace
        cfg_val = self.config.get("theme_debug_enabled")
        if cfg_val is None:
            cfg_val = self.config.get("debug_theme")
        if cfg_val is None:
            cfg_val = THEME_DEBUG_DEFAULT
        self.debug_enabled: bool = bool(cfg_val)
        self.debug_dump_bars: int = int(self.config.get("theme_debug_dump_bars", 200))
        self.debug_path: str = str(self.config.get("theme_debug_path", "logs/voting/theme_expert_debug.jsonl"))

        # Per-instrument rolling state
        self._state: Dict[str, Dict[str, Any]] = {}
        self._regime_hist: Dict[str, Deque[str]] = {}
        self._action_hist: Dict[str, Deque[str]] = {}

        for inst in self.instruments:
            self._state[inst] = {
                "last_quadrant": "unknown",
                "last_action": "flat",
                "last_conf": 0.10,
                "last_vol_ratio": 1.0,
                "last_hurst": 0.50,
                "last_chop": 50.0,
                "last_volume_ratio": 1.0,
                "last_ts": None,
            }
            self._regime_hist[inst] = deque(maxlen=self.max_regime_history)
            self._action_hist[inst] = deque(maxlen=self.max_regime_history)

        self._publish_theme_baseline()

        self.log_info(
            f"[ThemeExpert] init | instruments={self.instruments} | primary_tf={self.primary_tf} | "
            f"use_forming_bar={self.use_forming_bar} | debug={self.debug_enabled}"
        )

    def _publish_theme_baseline(self) -> None:
        try:
            thesis = "Theme baseline"
            self.smart_bus.set(
                "theme_analysis",
                {
                    "leader_instrument": normalize_instrument(self.primary_instrument),
                    "regime": "unknown",
                    "quadrant": "unknown",
                    "bias": 0.0,
                    "confidence": 0.10,
                    "per_instrument": {},
                    "timestamp": _dt.datetime.now().isoformat(),
                },
                module=self.__class__.__name__,
                thesis=thesis,
            )
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # DEBUG TRACE
    # ──────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────
    # DATA ACCESS
    # ──────────────────────────────────────────────────────────────

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
                out: List[float] = []
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

    # ──────────────────────────────────────────────────────────────
    # INDICATORS / MATH
    # ──────────────────────────────────────────────────────────────

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
        weights = np.arange(1, period + 1, dtype=float)
        denom = float(np.sum(weights))
        out: List[float] = []
        for i in range(len(data)):
            if i + 1 < period:
                out.append(float(data[i]))
            else:
                window = np.array(data[i + 1 - period : i + 1], dtype=float)
                out.append(float(np.sum(window * weights) / (denom + 1e-12)))
        return out

    def _hma_series(self, data: List[float], period: int) -> List[float]:
        # HMA(n) = WMA(2*WMA(price, n/2) - WMA(price, n), sqrt(n))
        if period <= 1 or len(data) < max(10, period):
            return list(data)
        n = int(period)
        half = max(1, n // 2)
        sqrt_n = max(1, int(np.sqrt(n)))

        wma_n = self._wma_series(data, n)
        wma_half = self._wma_series(data, half)
        diff = [2.0 * a - b for a, b in zip(wma_half, wma_n)]
        return self._wma_series(diff, sqrt_n)

    def _window_slope_norm(self, series: List[float]) -> float:
        if not series or len(series) < 6:
            return 0.0
        y = np.array(series, dtype=float)
        x = np.arange(len(y), dtype=float)
        x = x - np.mean(x)
        y = y - np.mean(y)
        denom = float(np.sum(x * x))
        if denom <= 1e-12:
            return 0.0
        slope = float(np.sum(x * y) / denom)
        scale = float(np.std(series)) + 1e-12
        return float(np.clip(slope / scale, -2.0, 2.0) / 2.0)

    def _parkinson_vol(self, highs: List[float], lows: List[float], lookback: int) -> float:
        n = min(len(highs), len(lows), lookback)
        if n < 5:
            return 0.0
        hs = np.array(highs[-n:], dtype=float)
        ls = np.array(lows[-n:], dtype=float)
        ls = np.clip(ls, 1e-12, np.inf)
        hs = np.clip(hs, 1e-12, np.inf)
        r = np.log(hs / ls)
        r2 = r * r
        denom = 4.0 * n * np.log(2.0)
        return float(np.sqrt(np.sum(r2) / (denom + 1e-12)))

    def _realized_vol(self, closes: List[float], lookback: int) -> float:
        if len(closes) < lookback + 2:
            return 0.0
        arr = np.array(closes[-(lookback + 1):], dtype=float)
        r = np.diff(np.log(np.clip(arr, 1e-12, np.inf)))
        return float(np.std(r)) if len(r) > 2 else 0.0

    def _atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        if len(closes) < period + 2 or len(highs) < period + 2 or len(lows) < period + 2:
            return 0.0
        h = np.array(highs[-(period + 1):], dtype=float)
        l = np.array(lows[-(period + 1):], dtype=float)
        c = np.array(closes[-(period + 1):], dtype=float)
        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        return float(np.mean(tr)) if len(tr) > 0 else 0.0

    def _choppiness_index(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        # CHOP = 100 * log10( sum(TR, n) / (max(high,n)-min(low,n)) ) / log10(n)
        n = int(period)
        if n <= 2:
            return 50.0
        if len(closes) < n + 2 or len(highs) < n + 2 or len(lows) < n + 2:
            return 50.0

        hs = np.array(highs[-n:], dtype=float)
        ls = np.array(lows[-n:], dtype=float)
        cs = np.array(closes[-(n + 1):], dtype=float)

        prev_c = cs[:-1]
        tr = np.maximum(hs - ls, np.maximum(np.abs(hs - prev_c[-n:]), np.abs(ls - prev_c[-n:])))
        tr_sum = float(np.sum(tr))

        denom_range = float(np.max(hs) - np.min(ls)) + 1e-12
        val = 100.0 * (np.log10((tr_sum / denom_range) + 1e-12) / (np.log10(float(n)) + 1e-12))
        return float(np.clip(val, 0.0, 100.0))

    def _hurst_exponent_rs(self, prices: List[float], lookback: int) -> float:
        # Multi-scale R/S regression for stability.
        n = min(len(prices), int(lookback))
        if n < 20:
            return 0.50
        p = np.array(prices[-n:], dtype=float)
        p = np.log(np.clip(p, 1e-12, np.inf))
        r = np.diff(p)
        if len(r) < 20:
            return 0.50

        sizes = []
        m = len(r)
        for k in [2, 3, 4, 6]:
            seg = max(10, m // k)
            if seg >= 10 and seg <= m:
                sizes.append(seg)
        sizes = sorted(list(set(sizes)))
        if len(sizes) < 2:
            sizes = [max(10, m // 2), max(12, m // 3)]
            sizes = sorted(list(set([s for s in sizes if s >= 10])))

        rs_vals = []
        x_vals = []
        for seg_len in sizes:
            k = m // seg_len
            if k <= 1:
                continue
            rs_list = []
            for i in range(k):
                seg = r[i * seg_len : (i + 1) * seg_len]
                mu = float(np.mean(seg))
                dev = seg - mu
                cum = np.cumsum(dev)
                R = float(np.max(cum) - np.min(cum))
                S = float(np.std(seg)) + 1e-12
                rs_list.append(R / S)
            if rs_list:
                rs_mean = float(np.mean(rs_list))
                rs_vals.append(np.log(rs_mean + 1e-12))
                x_vals.append(np.log(float(seg_len)))

        if len(rs_vals) < 2:
            return 0.50

        x = np.array(x_vals, dtype=float)
        y = np.array(rs_vals, dtype=float)
        x = x - np.mean(x)
        denom = float(np.sum(x * x))
        if denom <= 1e-12:
            return 0.50
        slope = float(np.sum(x * (y - np.mean(y))) / denom)
        return float(np.clip(slope, 0.0, 1.0))

    def _volume_ratio(self, volumes: List[float], fast: int = 5, slow: int = 20) -> float:
        if not volumes or len(volumes) < slow + 2:
            return 1.0
        v = np.array(volumes, dtype=float)
        v_fast = float(np.mean(v[-fast:]))
        v_slow = float(np.mean(v[-slow:])) + 1e-12
        return float(np.clip(v_fast / v_slow, 0.2, 5.0))

    def _volume_profile_regime(self, closes: List[float], volumes: List[float], window: int = 80, bins: int = 12) -> Dict[str, Any]:
        if len(closes) < max(30, window) or len(volumes) < max(30, window):
            return {"available": False, "regime": "unknown", "vpoc_strength": 0.0}
        c = np.array(closes[-window:], dtype=float)
        v = np.array(volumes[-window:], dtype=float)
        lo = float(np.min(c))
        hi = float(np.max(c))
        if hi - lo <= 1e-12:
            return {"available": True, "regime": "unknown", "vpoc_strength": 0.0, "vpoc_price": float(c[-1])}

        edges = np.linspace(lo, hi, bins + 1)
        prof = np.zeros(bins, dtype=float)

        idxs = np.digitize(c, edges) - 1
        idxs = np.clip(idxs, 0, bins - 1)

        for i in range(len(c)):
            prof[int(idxs[i])] += float(max(0.0, v[i]))

        total = float(np.sum(prof)) + 1e-12
        vpoc_i = int(np.argmax(prof))
        vpoc_strength = float(prof[vpoc_i] / total)
        vpoc_price = float((edges[vpoc_i] + edges[vpoc_i + 1]) / 2.0)

        if vpoc_strength > 0.30:
            regime = "institutional"
        elif vpoc_strength < 0.15:
            regime = "retail"
        else:
            regime = "mixed"

        return {
            "available": True,
            "regime": regime,
            "vpoc_strength": float(np.clip(vpoc_strength, 0.0, 1.0)),
            "vpoc_price": vpoc_price,
        }

    def _fractal_pivots(self, highs: List[float], lows: List[float], left: int = 5, right: int = 3) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        n = min(len(highs), len(lows))
        if n < left + right + 5:
            return [], []
        piv_hi: List[Tuple[int, float]] = []
        piv_lo: List[Tuple[int, float]] = []
        hs = np.array(highs, dtype=float)
        ls = np.array(lows, dtype=float)

        for i in range(left, n - right):
            h = hs[i]
            l = ls[i]
            if h >= np.max(hs[i - left : i + right + 1]):
                piv_hi.append((i, float(h)))
            if l <= np.min(ls[i - left : i + right + 1]):
                piv_lo.append((i, float(l)))

        # Keep only most recent pivots
        return piv_hi[-12:], piv_lo[-12:]

    def _structural_regime(self, highs: List[float], lows: List[float], closes: List[float]) -> Dict[str, Any]:
        if len(closes) < 60:
            return {"available": False, "structure": "unknown", "strength": 0.0, "breakout_score": 0.0}

        piv_hi, piv_lo = self._fractal_pivots(highs, lows, left=5, right=3)

        structure = "transition"
        strength = 0.4
        breakout_score = 0.0

        # Trend structure via recent HH/HL or LL/LH
        if len(piv_hi) >= 3 and len(piv_lo) >= 3:
            last_hi = [p[1] for p in piv_hi[-3:]]
            last_lo = [p[1] for p in piv_lo[-3:]]
            hh = all(last_hi[i] > last_hi[i - 1] for i in range(1, len(last_hi)))
            ll = all(last_lo[i] < last_lo[i - 1] for i in range(1, len(last_lo)))

            if hh and not ll:
                structure = "uptrend"
                strength = 0.8
            elif ll and not hh:
                structure = "downtrend"
                strength = 0.8

        # Range contraction/expansion proxy
        rng_50 = float(max(highs[-50:]) - min(lows[-50:])) + 1e-12
        rng_10 = float(max(highs[-10:]) - min(lows[-10:])) + 1e-12
        contraction = rng_10 < rng_50 * 0.35
        expansion = rng_10 > rng_50 * 0.80

        if contraction and structure == "transition":
            structure = "consolidation"
            strength = 0.6
        elif expansion and structure == "transition":
            structure = "expanding"
            strength = 0.6

        # Breakout score vs recent pivot bands
        atr = self._atr(highs, lows, closes, self.atr_period)
        if piv_hi:
            top = float(max([p[1] for p in piv_hi[-5:]]))
            if closes[-1] > top + 0.5 * atr:
                breakout_score = 1.0
        if piv_lo:
            bot = float(min([p[1] for p in piv_lo[-5:]]))
            if closes[-1] < bot - 0.5 * atr:
                breakout_score = -1.0

        return {
            "available": True,
            "structure": structure,
            "strength": float(_clip(strength, 0.0, 1.0)),
            "breakout_score": float(_clip(breakout_score, -1.0, 1.0)),
            "contraction": bool(contraction),
            "expansion": bool(expansion),
        }

    def _fast_slope(self, closes: List[float]) -> float:
        if len(closes) < max(30, self.fast_period + 10):
            return 0.0
        hma = self._hma_series(closes, self.fast_period)
        look = min(self.slope_lookback, max(6, len(hma) // 3))
        return self._window_slope_norm(hma[-look:])

    def _cross_asset_proxy(self, primary_action_hint: str = "flat") -> Dict[str, Any]:
        # Best-effort only: relies on historical_prices containing these symbols.
        if not self.use_cross_asset:
            return {"available": False}

        def _asset_momentum(sym: str) -> Optional[float]:
            sym_n = normalize_instrument(sym)
            # Attempt exact key match first (DXY/SPX/US10Y might not normalize well)
            hist = self._get_historical()
            if not isinstance(hist, dict) or not hist:
                return None

            # Find matching symbol by "normalize_instrument" if possible; else raw key.
            matched = None
            for k in hist.keys():
                if k == sym or normalize_instrument(k) == sym_n:
                    matched = k
                    break
            if matched is None:
                # Some feeds might use raw codes like "DXY" and normalization returns same
                if sym in hist:
                    matched = sym
                else:
                    return None

            rec = hist.get(matched, {})
            if not isinstance(rec, dict):
                return None
            tf_rec = rec.get(self.primary_tf) or rec.get("M15") or rec.get("H1") or rec.get("D1")
            if not isinstance(tf_rec, dict):
                return None
            closes = tf_rec.get("close")
            if not isinstance(closes, (list, tuple, np.ndarray)) or len(closes) < 60:
                return None
            arr = [float(x) for x in closes[-180:]]
            slope = self._fast_slope(arr)
            return float(_clip(slope, -1.0, 1.0))

        dxy = _asset_momentum(self.cross_assets.get("DXY", "DXY"))
        spx = _asset_momentum(self.cross_assets.get("SPX", "SPX"))
        us10y = _asset_momentum(self.cross_assets.get("US10Y", "US10Y"))

        available = (dxy is not None) or (spx is not None) or (us10y is not None)
        if not available:
            return {"available": False}

        # Economic proxy mapping:
        # - Risk-on: SPX up, DXY down (rough heuristic)
        # - Risk-off: SPX down, DXY up
        risk_on = 0.0
        risk_off = 0.0
        if spx is not None:
            risk_on += max(0.0, spx) * 0.55
            risk_off += max(0.0, -spx) * 0.55
        if dxy is not None:
            risk_off += max(0.0, dxy) * 0.30
            risk_on += max(0.0, -dxy) * 0.30
        if us10y is not None:
            # Rising yields often pressure gold; falling yields often support gold (rough heuristic)
            risk_off += max(0.0, -us10y) * 0.15
            risk_on += max(0.0, us10y) * 0.15

        # For XAUUSD bias: risk-off tends to be supportive; risk-on tends to be less supportive.
        xau_alignment = float(_clip(risk_off - risk_on, -1.0, 1.0))
        conf_delta = float(_clip(xau_alignment * self.cross_asset_conf_weight, -0.15, 0.15))

        return {
            "available": True,
            "dxy_momentum": dxy,
            "spx_momentum": spx,
            "us10y_momentum": us10y,
            "risk_on": float(_clip(risk_on, 0.0, 1.0)),
            "risk_off": float(_clip(risk_off, 0.0, 1.0)),
            "xau_alignment": xau_alignment,
            "conf_delta": conf_delta,
        }

    # ──────────────────────────────────────────────────────────────
    # REGIME / SCORING
    # ──────────────────────────────────────────────────────────────

    def _classify_quadrant(self, hurst: float, vol_ratio: float) -> str:
        if hurst >= self.hurst_hi and vol_ratio <= self.vol_ratio_hi and vol_ratio >= self.vol_ratio_lo:
            return "goldilocks"
        if hurst >= self.hurst_hi and vol_ratio > self.vol_ratio_hi:
            return "panic_mania"
        if hurst <= self.hurst_lo and vol_ratio > self.vol_ratio_hi:
            return "chop_whipsaw"
        if hurst <= self.hurst_lo and vol_ratio <= self.vol_ratio_hi:
            return "drift_noise"
        return "transition"

    def _mtf_alignment(self, instrument: str) -> Dict[str, Any]:
        if not self.use_mtf_alignment:
            return {"available": False}

        details: Dict[str, Any] = {}
        weighted_vote = 0.0
        weight_sum = 0.0

        for tf in self.mtf_default_set:
            rec = self._get_tf_rec(instrument, tf)
            if not isinstance(rec, dict):
                continue
            closes = rec.get("close")
            highs = rec.get("high")
            lows = rec.get("low")
            vols = rec.get("volume")
            if not isinstance(closes, (list, tuple, np.ndarray)) or len(closes) < 80:
                continue
            closes_l = [float(x) for x in list(closes)[-260:]]
            highs_l = [float(x) for x in list(highs)[-260:]] if isinstance(highs, (list, tuple, np.ndarray)) else closes_l
            lows_l = [float(x) for x in list(lows)[-260:]] if isinstance(lows, (list, tuple, np.ndarray)) else closes_l

            pv = self._parkinson_vol(highs_l, lows_l, self.vol_lookback)
            baseline_n = self.vol_lookback * max(2, self.vol_baseline_mult)
            pv_b = self._parkinson_vol(highs_l, lows_l, min(len(highs_l), baseline_n))
            vol_ratio = pv / max(1e-12, pv_b) if pv_b > 0 else 1.0
            hurst = self._hurst_exponent_rs(closes_l, min(len(closes_l), self.hurst_lookback))
            quad = self._classify_quadrant(hurst, vol_ratio)

            # Reduce quadrant to "trendiness vote": +1 trend-friendly, -1 anti-trend (chop/whipsaw), 0 neutral
            if quad in ("goldilocks", "panic_mania"):
                v = 1.0
            elif quad == "chop_whipsaw":
                v = -1.0
            else:
                v = 0.0

            w = float(self.mtf_weights.get(tf, 0.5))
            weighted_vote += v * w
            weight_sum += w

            details[tf] = {
                "hurst": float(hurst),
                "vol_ratio": float(vol_ratio),
                "quadrant": quad,
                "vote": float(v),
                "weight": float(w),
            }

        if weight_sum <= 1e-12:
            return {"available": False}

        score = float(np.clip(weighted_vote / weight_sum, -1.0, 1.0))
        agreement = float(abs(score))
        dominant = "trend_friendly" if score > 0.25 else "chop_risk" if score < -0.25 else "mixed"

        return {
            "available": True,
            "score": score,
            "agreement": agreement,
            "dominant": dominant,
            "details": details,
        }

    def _apply_persistence(self, inst: str, quadrant: str, action: str, confidence: float) -> Tuple[str, str, float, Dict[str, Any]]:
        """
        Hysteresis to avoid flip-flopping:
        - Quadrant changes require persistence unless it is a "chop_whipsaw" kill switch.
        """
        meta = {"applied": False, "kept_quadrant": quadrant, "kept_action": action}

        if quadrant == "chop_whipsaw":
            # Kill switch should override immediately.
            self._regime_hist[inst].append(quadrant)
            self._action_hist[inst].append("flat")
            return quadrant, "flat", float(_clip(max(confidence, 0.40), 0.10, 0.95)), {"applied": True, "reason": "kill_switch"}

        hist = self._regime_hist.get(inst)
        if hist is None:
            return quadrant, action, confidence, meta

        last = hist[-1] if len(hist) > 0 else "unknown"
        if last == quadrant:
            self._regime_hist[inst].append(quadrant)
            self._action_hist[inst].append(action)
            return quadrant, action, float(_clip(confidence * 1.05, 0.10, 0.95)), {"applied": True, "reason": "same_regime_boost"}

        # Quadrant changed: require persistence to confirm
        recent = list(hist)[-max(1, self.min_regime_persistence - 1):]
        # If recent already oscillating, damp flips hard
        oscillating = (len(set(recent + [quadrant])) > 2)

        if len(hist) >= 1 and self.min_regime_persistence > 1:
            # If flip not yet "earned", damp confidence and keep prior action bias.
            conf = confidence * (0.75 if not oscillating else 0.65)
            kept_action = self._action_hist[inst][-1] if len(self._action_hist[inst]) > 0 else "flat"
            kept_quadrant = last
            hist.append(last)
            self._action_hist[inst].append(kept_action)
            return kept_quadrant, kept_action, float(_clip(conf, 0.10, 0.95)), {
                "applied": True,
                "reason": "persistence_hold",
                "oscillating": bool(oscillating),
                "from": last,
                "to": quadrant,
            }

        self._regime_hist[inst].append(quadrant)
        self._action_hist[inst].append(action)
        return quadrant, action, confidence, meta

    # ──────────────────────────────────────────────────────────────
    # CORE VOTING HOOK
    # ──────────────────────────────────────────────────────────────

    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        name = self.__class__.__name__

        votes: Dict[str, Dict[str, Any]] = {}
        analysis_by_inst: Dict[str, Dict[str, Any]] = {}

        if self.debug_enabled:
            hist = self._get_historical()
            self._debug_write(
                "tick_start",
                {
                    "market_data_keys": list(market_data.keys()) if isinstance(market_data, dict) else str(type(market_data)),
                    "historical_symbols": list(hist.keys())[:30] if isinstance(hist, dict) else None,
                    "primary_tf": self.primary_tf,
                    "use_forming_bar": self.use_forming_bar,
                },
            )

        for inst in self.instruments:
            inst_norm = normalize_instrument(inst)
            try:
                ohlcv = self._get_ohlcv_series(inst_norm, self.primary_tf)
                ohlcv = self._maybe_apply_forming_bar(inst_norm, self.primary_tf, ohlcv)

                closes = (ohlcv.get("close") or [])[-300:]
                highs = (ohlcv.get("high") or closes)[-300:]
                lows = (ohlcv.get("low") or closes)[-300:]
                vols = (ohlcv.get("volume") or [1.0] * len(closes))[-300:]

                if len(closes) < max(80, self.hurst_lookback, self.vol_lookback * self.vol_baseline_mult):
                    votes[inst_norm] = {
                        "instrument": inst_norm,
                        "action": "flat",
                        "confidence": 0.10,
                        "magnitude": 0.0,
                        "signal_strength": 0.0,
                        "rationale": f"{inst_norm}: flat (insufficient data closes={len(closes)})",
                    }
                    analysis_by_inst[inst_norm] = {"status": "insufficient_data", "closes_len": len(closes)}
                    self._regime_hist[inst_norm].append("unknown")
                    self._action_hist[inst_norm].append("flat")
                    continue

                # Cache (base hashes closes only) + validate last high/low/vol to avoid stale packs
                cached = self._get_cached_indicators(inst_norm, closes)
                cache_ok = False
                if isinstance(cached, dict):
                    if (
                        _safe_float(cached.get("high_last"), -999) == _safe_float(highs[-1], -998)
                        and _safe_float(cached.get("low_last"), -999) == _safe_float(lows[-1], -998)
                        and _safe_float(cached.get("vol_last"), -999) == _safe_float(vols[-1], -998)
                    ):
                        cache_ok = True

                if cache_ok:
                    pack = cached
                else:
                    # Core regime math
                    pv = self._parkinson_vol(highs, lows, self.vol_lookback)
                    baseline_n = min(len(highs), self.vol_lookback * max(2, self.vol_baseline_mult))
                    pv_b = self._parkinson_vol(highs, lows, baseline_n)
                    vol_ratio = pv / max(1e-12, pv_b) if pv_b > 0 else 1.0
                    vol_ratio = float(_clip(vol_ratio, 0.3, 3.5))

                    hurst = self._hurst_exponent_rs(closes, self.hurst_lookback)
                    chop = self._choppiness_index(highs, lows, closes, self.chop_period)
                    fast_slope = self._fast_slope(closes)

                    atr = self._atr(highs, lows, closes, self.atr_period)
                    rvol = self._realized_vol(closes, min(60, len(closes) - 2))

                    vratio = self._volume_ratio(vols, fast=5, slow=20) if self.use_volume_confirmation else 1.0
                    vp = self._volume_profile_regime(closes, vols, window=80, bins=12)
                    struct = self._structural_regime(highs, lows, closes)
                    mtf = self._mtf_alignment(inst_norm) if self.use_mtf_alignment else {"available": False}
                    xasset = self._cross_asset_proxy(primary_action_hint="flat") if self.use_cross_asset else {"available": False}

                    quadrant = self._classify_quadrant(hurst, vol_ratio)

                    pack = {
                        "pv": pv,
                        "pv_baseline": pv_b,
                        "vol_ratio": vol_ratio,
                        "hurst": hurst,
                        "chop": chop,
                        "fast_slope": fast_slope,
                        "atr": atr,
                        "realized_vol": rvol,
                        "volume_ratio": vratio,
                        "vp": vp,
                        "structure": struct,
                        "mtf": mtf,
                        "xasset": xasset,
                        "quadrant": quadrant,
                        # cache validation:
                        "high_last": _safe_float(highs[-1], 0.0),
                        "low_last": _safe_float(lows[-1], 0.0),
                        "vol_last": _safe_float(vols[-1], 0.0),
                    }
                    self._set_cached_indicators(inst_norm, closes, pack)

                pack = pack if isinstance(pack, dict) else {}
                quadrant = str(pack.get("quadrant", "unknown"))
                hurst = float(pack.get("hurst", 0.50))
                vol_ratio = float(pack.get("vol_ratio", 1.0))
                chop = float(pack.get("chop", 50.0))
                fast_slope = float(pack.get("fast_slope", 0.0))
                vratio = float(pack.get("volume_ratio", 1.0))
                struct = pack.get("structure", {"available": False})
                mtf = pack.get("mtf", {"available": False})
                xasset = pack.get("xasset", {"available": False})
                vp = pack.get("vp", {"available": False})

                # Base confidence from "distance into quadrant"
                hurst_dist = 0.0
                if hurst >= self.hurst_hi:
                    hurst_dist = (hurst - self.hurst_hi) / max(1e-12, (1.0 - self.hurst_hi))
                elif hurst <= self.hurst_lo:
                    hurst_dist = (self.hurst_lo - hurst) / max(1e-12, self.hurst_lo)
                hurst_dist = float(_clip(hurst_dist, 0.0, 1.0))

                vol_dist = 0.0
                if vol_ratio >= self.vol_ratio_hi:
                    vol_dist = (vol_ratio - self.vol_ratio_hi) / max(1e-12, (3.5 - self.vol_ratio_hi))
                elif vol_ratio <= self.vol_ratio_lo:
                    vol_dist = (self.vol_ratio_lo - vol_ratio) / max(1e-12, self.vol_ratio_lo)
                vol_dist = float(_clip(vol_dist, 0.0, 1.0))

                base_conf = 0.35 + 0.35 * hurst_dist + 0.20 * vol_dist
                base_conf = float(_clip(base_conf, 0.15, 0.90))

                # CHOP filter (theme-level veto / dampener)
                chop_meta = {"available": bool(self.use_chop_filter), "chop": chop}
                if self.use_chop_filter:
                    if chop >= self.chop_chaos_thr:
                        # chaotic -> flat
                        quadrant = "chop_whipsaw"
                        base_conf = float(_clip(max(base_conf, 0.45), 0.10, 0.95))
                        chop_meta["regime"] = "chaotic"
                    elif chop >= self.chop_thr:
                        base_conf = float(_clip(base_conf * self.chop_conf_mult_transition, 0.10, 0.95))
                        chop_meta["regime"] = "transition"
                    else:
                        base_conf = float(_clip(base_conf * self.chop_conf_mult_trending, 0.10, 0.95))
                        chop_meta["regime"] = "trending"

                # Direction bias logic (XAUUSD-tuned):
                # - In trend-friendly quadrants: follow fast_slope sign
                # - In chop/whipsaw: flat
                # - In drift/noise: mostly flat unless strong breakout
                bias = 0.0
                trend_dir = 1.0 if fast_slope > 0.10 else -1.0 if fast_slope < -0.10 else 0.0

                breakout_score = 0.0
                if isinstance(struct, dict):
                    breakout_score = float(struct.get("breakout_score", 0.0))

                if quadrant in ("goldilocks", "panic_mania"):
                    bias = float(_clip(trend_dir * (0.55 + 0.45 * abs(fast_slope)), -1.0, 1.0))
                elif quadrant == "drift_noise":
                    # Only lean if structure suggests breakout; otherwise flat.
                    if abs(breakout_score) > 0.5:
                        bias = float(_clip(breakout_score * 0.55, -1.0, 1.0))
                        base_conf *= 0.85
                    else:
                        bias = 0.0
                        base_conf *= 0.80
                else:
                    # chop_whipsaw/transition => bias toward 0
                    bias = float(_clip(trend_dir * 0.15, -0.25, 0.25))
                    base_conf *= 0.75

                # Volume-confirmed regime shifts: if regime is shifting but volume does not expand, damp confidence
                last_quad = str(self._state.get(inst_norm, {}).get("last_quadrant", "unknown"))
                if self.use_volume_confirmation and quadrant != last_quad and vratio < self.regime_shift_vol_mult:
                    base_conf = float(_clip(base_conf * 0.85, 0.10, 0.95))

                # MTF alignment: confidence only
                mtf_adj = 0.0
                if isinstance(mtf, dict) and mtf.get("available"):
                    score = float(mtf.get("score", 0.0))
                    if quadrant in ("goldilocks", "panic_mania") and score > 0.25:
                        mtf_adj = self.mtf_agreement_bonus * float(mtf.get("agreement", 0.0))
                    elif quadrant == "chop_whipsaw" and score < -0.25:
                        mtf_adj = self.mtf_agreement_bonus * float(mtf.get("agreement", 0.0))
                    elif quadrant in ("goldilocks", "panic_mania") and score < -0.25:
                        mtf_adj = -self.mtf_disagreement_penalty * float(mtf.get("agreement", 0.0))
                    base_conf = float(_clip(base_conf + mtf_adj, 0.10, 0.95))

                # Cross-asset proxy: for XAU, risk-off supports long; risk-on reduces long bias
                xasset_adj = 0.0
                if isinstance(xasset, dict) and xasset.get("available") and inst_norm == "XAUUSD":
                    xasset_adj = float(xasset.get("conf_delta", 0.0))
                    base_conf = float(_clip(base_conf + xasset_adj, 0.10, 0.95))
                    # Bias alignment pushes toward long if risk-off is strong
                    bias = float(_clip(bias + float(xasset.get("xau_alignment", 0.0)) * 0.25, -1.0, 1.0))

                # Action decision from bias band
                if quadrant == "chop_whipsaw":
                    action = "flat"
                else:
                    if bias > self.bias_band:
                        action = "long"
                    elif bias < -self.bias_band:
                        action = "short"
                    else:
                        action = "flat"

                # Persistence / hysteresis
                quadrant, action, base_conf, pers_meta = self._apply_persistence(inst_norm, quadrant, action, base_conf)

                # Strength is confidence-weighted but damped in noisy regimes
                noise_pen = 1.0
                if quadrant in ("drift_noise", "transition"):
                    noise_pen = 0.80
                if quadrant == "panic_mania":
                    noise_pen = 0.85
                if quadrant == "chop_whipsaw":
                    noise_pen = 0.70

                strength = float(_clip(base_conf * noise_pen, 0.0, 1.0))
                if action == "flat":
                    strength = 0.0

                thesis = (
                    f"{inst_norm}: {action} | quad={quadrant} | H={hurst:.3f} | volR={vol_ratio:.2f} | "
                    f"chop={chop:.1f} | slope={fast_slope:+.2f} | bias={bias:+.2f} | conf={base_conf:.2f} | "
                    f"mtf_adj={mtf_adj:+.2f} xasset={xasset_adj:+.2f}"
                )

                votes[inst_norm] = {
                    "instrument": inst_norm,
                    "action": action,
                    "confidence": float(_clip(base_conf, 0.10, 0.95)),
                    "magnitude": strength,
                    "signal_strength": strength,
                    "rationale": thesis,
                }

                volatility_regime = (
                    "high"
                    if vol_ratio >= self.vol_ratio_hi
                    else "low"
                    if vol_ratio <= self.vol_ratio_lo
                    else "medium"
                )

                risk_regime = "neutral"
                try:
                    if isinstance(xasset, dict) and xasset.get("available"):
                        r_on = float(xasset.get("risk_on", 0.0) or 0.0)
                        r_off = float(xasset.get("risk_off", 0.0) or 0.0)
                        if (r_on - r_off) > 0.15:
                            risk_regime = "risk_on"
                        elif (r_off - r_on) > 0.15:
                            risk_regime = "risk_off"
                    else:
                        if quadrant == "goldilocks":
                            risk_regime = "risk_on"
                        elif quadrant == "panic_mania":
                            risk_regime = "risk_off"
                except Exception:
                    risk_regime = "neutral"

                analysis_by_inst[inst_norm] = {
                    "status": "ok",
                    "tf": self.primary_tf,
                    "quadrant": quadrant,
                    "risk_regime": risk_regime,
                    "volatility_regime": volatility_regime,
                    "hurst": hurst,
                    "vol_ratio": vol_ratio,
                    "parkinson_vol": float(pack.get("pv", 0.0)),
                    "parkinson_baseline": float(pack.get("pv_baseline", 0.0)),
                    "chop": chop_meta,
                    "fast_slope": fast_slope,
                    "atr": float(pack.get("atr", 0.0)),
                    "realized_vol": float(pack.get("realized_vol", 0.0)),
                    "volume_ratio": vratio,
                    "vp": vp if isinstance(vp, dict) else {"available": False},
                    "structure": struct if isinstance(struct, dict) else {"available": False},
                    "mtf": mtf if isinstance(mtf, dict) else {"available": False},
                    "xasset": xasset if isinstance(xasset, dict) else {"available": False},
                    "bias": bias,
                    "action": action,
                    "confidence": float(_clip(base_conf, 0.10, 0.95)),
                    "strength": strength,
                    "persistence": pers_meta,
                }

                # Update state
                self._state[inst_norm].update(
                    {
                        "last_quadrant": quadrant,
                        "last_action": action,
                        "last_conf": float(_clip(base_conf, 0.10, 0.95)),
                        "last_vol_ratio": vol_ratio,
                        "last_hurst": hurst,
                        "last_chop": chop,
                        "last_volume_ratio": vratio,
                        "last_ts": _dt.datetime.now().isoformat(),
                    }
                )

                if self.debug_enabled:
                    dump_n = max(20, min(self.debug_dump_bars, 200))
                    self._debug_write(
                        "instrument_forensic",
                        {
                            "instrument": inst_norm,
                            "tf": self.primary_tf,
                            "prices_tail": closes[-dump_n:],
                            "high_tail": highs[-dump_n:],
                            "low_tail": lows[-dump_n:],
                            "vol_tail": vols[-dump_n:],
                            "components": analysis_by_inst[inst_norm],
                            "thesis": thesis,
                        },
                    )

            except Exception as e:
                votes[inst_norm] = {
                    "instrument": inst_norm,
                    "action": "flat",
                    "confidence": 0.10,
                    "magnitude": 0.0,
                    "signal_strength": 0.0,
                    "rationale": f"{inst_norm}: flat (error {type(e).__name__}: {e})",
                }
                analysis_by_inst[inst_norm] = {
                    "status": "error",
                    "quadrant": "unknown",
                    "risk_regime": "neutral",
                    "volatility_regime": "medium",
                    "bias": 0.0,
                    "action": "flat",
                    "confidence": 0.10,
                    "strength": 0.0,
                    "error": f"{type(e).__name__}: {e}",
                }
                self._regime_hist[inst_norm].append("unknown")
                self._action_hist[inst_norm].append("flat")
                self._debug_write("instrument_error", {"instrument": inst_norm, "error": f"{type(e).__name__}: {e}"})

        if not votes:
            return self._neutral_proposal("No instruments configured / no votes produced")

        # Leader selection: confidence * strength (consistent with MomentumExpert)
        def _score(v: Dict[str, Any]) -> float:
            return _safe_float(v.get("confidence"), 0.0) * max(0.1, _safe_float(v.get("signal_strength"), 0.0))

        leader_inst = max(votes.keys(), key=lambda k: _score(votes[k]))
        leader_vote = votes[leader_inst]
        global_action = str(leader_vote.get("action", "flat"))
        global_strength = _safe_float(leader_vote.get("signal_strength"), 0.0)
        thesis = str(leader_vote.get("rationale", f"leader={leader_inst}"))

        # Publish side-channel analysis (per-instrument votes are published by VotingExpertBase)
        try:
            leader_a = analysis_by_inst.get(leader_inst, {}) if isinstance(analysis_by_inst, dict) else {}
            self.smart_bus.set(
                "theme_analysis",
                {
                    "leader_instrument": leader_inst,
                    "regime": str(leader_a.get("quadrant", "unknown")),
                    "quadrant": str(leader_a.get("quadrant", "unknown")),
                    "bias": float(leader_a.get("bias", 0.0)),
                    "confidence": float(leader_vote.get("confidence", 0.10)),
                    "per_instrument": analysis_by_inst,
                    "timestamp": _dt.datetime.now().isoformat(),
                },
                module=name,
                thesis=f"Theme analysis leader={leader_inst}",
            )
        except Exception as e:
            self._debug_write("bus_publish_error", {"error": f"{type(e).__name__}: {e}"})

        elapsed_ms = (time.time() - t0) * 1000.0
        self._debug_write("tick_end", {"leader": leader_inst, "global_action": global_action, "elapsed_ms": elapsed_ms})

        max_strength = float(getattr(self, "max_signal_strength", 1.0))
        proposal: Dict[str, Any] = {
            "action": global_action,
            "signal_strength": float(global_strength),
            "position_size": float(min(max_strength, global_strength * 0.5)),
            "leader_instrument": leader_inst,
            "reason": thesis,
            "proposals": votes,          # votes only
            "per_instrument": votes,     # votes only (alias)
            "analysis": analysis_by_inst,  # rich analysis side-channel (not consumed by base reframing)
        }
        return proposal

    async def _calculate_expert_specific_confidence(self, proposal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        try:
            # Confidence anchored to leader vote
            action = str(proposal.get("action", "flat"))
            sig = _safe_float(proposal.get("signal_strength"), 0.0)
            base = 0.30 + sig * 0.55

            leader = str(proposal.get("leader_instrument", ""))
            analysis = proposal.get("analysis", {})
            leader_a = analysis.get(leader, {}) if isinstance(analysis, dict) else {}

            quad = str(leader_a.get("quadrant", "unknown"))
            chop = leader_a.get("chop", {})
            chop_val = _safe_float(chop.get("chop"), 50.0) if isinstance(chop, dict) else 50.0

            # Boost confidence for clear “kill switch” or clear trend-friendly regimes
            if quad == "chop_whipsaw":
                base *= 1.15
            elif quad in ("goldilocks", "panic_mania"):
                base *= 1.10

            # Penalize if CHOP is very high but action is not flat
            if chop_val >= 65.0 and action in ("long", "short"):
                base *= 0.85

            return float(np.clip(base, 0.15, 0.95))
        except Exception:
            return 0.40

    # ──────────────────────────────────────────────────────────────
    # NEUTRAL HELPERS
    # ──────────────────────────────────────────────────────────────

    def _neutral_proposal(self, reason: str) -> Dict[str, Any]:
        name = self.__class__.__name__
        inst = normalize_instrument(self.primary_instrument)
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
                "status": "neutral",
                "quadrant": "unknown",
                "risk_regime": "neutral",
                "volatility_regime": "medium",
                "bias": 0.0,
                "action": "flat",
                "confidence": 0.10,
                "strength": 0.0,
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
                "theme_analysis",
                {
                    "leader_instrument": inst,
                    "regime": "unknown",
                    "quadrant": "unknown",
                    "bias": 0.0,
                    "confidence": 0.10,
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
            "reason": f"Theme flat: {reason}",
            "proposals": per_inst_votes,
            "per_instrument": per_inst_votes,
            "analysis": per_inst_analysis,
        }

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Thin wrapper:
        - Delegates all real logic to VotingExpertBase.process()
        - Adds legacy alias keys for downstream compatibility (no duplication).
        """
        out = await super().process(**inputs)
        name = self.__class__.__name__

        proposal = out.get("voting_proposal") or out.get(f"{name}_voting_proposal", {})
        confidence = out.get("confidence", out.get(f"{name}_confidence", 0.0))

        # Pull theme_analysis from bus for dashboards; fallback to proposal analysis
        analysis = out.get("theme_analysis")
        if analysis is None:
            try:
                analysis = self.smart_bus.get("theme_analysis", name, default=None)
            except Exception:
                analysis = None
        if analysis is None and isinstance(proposal, dict):
            analysis = {
                "leader_instrument": proposal.get("leader_instrument", normalize_instrument(self.primary_instrument)),
                "regime": "unknown",
                "quadrant": "unknown",
                "bias": 0.0,
                "confidence": _safe_float(confidence, 0.10),
                "per_instrument": proposal.get("analysis", {}),
                "timestamp": _dt.datetime.now().isoformat(),
            }

        # Legacy aliases (mirrors how MomentumExpert preserves compat)
        out.setdefault("ThemeExpert_voting_proposal", proposal)
        out.setdefault("ThemeExpert_confidence", confidence)
        out.setdefault("theme_voting_proposal", proposal)
        out.setdefault("theme_confidence", confidence)
        out.setdefault("theme_analysis", analysis)

        # Required contract keys (ModuleRegistry + Orchestrator enforced)
        # Derived from theme_analysis (leader) with safe defaults.
        leader_inst = "XAUUSD"
        try:
            if isinstance(analysis, dict):
                leader_inst = normalize_instrument(str(analysis.get("leader_instrument", leader_inst)))
        except Exception:
            leader_inst = "XAUUSD"

        leader_a: Dict[str, Any] = {}
        try:
            if isinstance(analysis, dict):
                per_inst = analysis.get("per_instrument")
                if isinstance(per_inst, dict):
                    leader_a = per_inst.get(leader_inst) or {}
                    if not isinstance(leader_a, dict):
                        leader_a = {}
        except Exception:
            leader_a = {}

        quadrant = str((leader_a.get("quadrant") if isinstance(leader_a, dict) else None) or (analysis.get("quadrant") if isinstance(analysis, dict) else None) or "unknown").lower()
        if quadrant in ("goldilocks", "panic_mania"):
            trend_regime = "trending"
        elif quadrant == "chop_whipsaw":
            trend_regime = "choppy"
        else:
            trend_regime = "mixed"

        vol_regime = str((leader_a.get("volatility_regime") if isinstance(leader_a, dict) else None) or "medium").lower()
        if vol_regime not in ("low", "medium", "high"):
            vol_regime = "medium"

        risk_regime = str((leader_a.get("risk_regime") if isinstance(leader_a, dict) else None) or "neutral").lower()
        if risk_regime not in ("risk_on", "risk_off", "neutral"):
            risk_regime = "neutral"

        composite_score = 0.0
        try:
            if isinstance(analysis, dict):
                composite_score = float(analysis.get("bias", 0.0) or 0.0)
        except Exception:
            composite_score = 0.0

        agreement_score = 0.0
        try:
            mtf = leader_a.get("mtf") if isinstance(leader_a, dict) else None
            if isinstance(mtf, dict) and mtf.get("available"):
                agreement_score = float(mtf.get("agreement", 0.0) or 0.0)
        except Exception:
            agreement_score = 0.0

        out.setdefault("theme_volatility_regime", vol_regime)
        out.setdefault("theme_trend_regime", trend_regime)
        out.setdefault("theme_risk_regime", risk_regime)
        out.setdefault("theme_composite_score", composite_score)
        out.setdefault("agreement_score", agreement_score)
        out.setdefault("theme_expert_analysis", analysis if isinstance(analysis, dict) else {})
        out.setdefault("theme_expert_thesis", str(out.get("_thesis") or out.get("thesis") or ""))

        if self.debug_enabled:
            out.setdefault("theme_debug", {"enabled": True, "path": self.debug_path})

        return out

    # ──────────────────────────────────────────────────────────────
    # STATE PERSISTENCE
    # ──────────────────────────────────────────────────────────────

    def _get_custom_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "state": {},
            "regime_hist": {},
            "action_hist": {},
        }
        for inst, st in self._state.items():
            state["state"][inst] = dict(st)
        for inst, h in self._regime_hist.items():
            state["regime_hist"][inst] = list(h)[-self.max_regime_history:]
        for inst, h in self._action_hist.items():
            state["action_hist"][inst] = list(h)[-self.max_regime_history:]
        return state

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        st = state.get("state", {})
        if isinstance(st, dict):
            for inst, saved in st.items():
                inst_n = normalize_instrument(inst)
                if inst_n in self._state and isinstance(saved, dict):
                    self._state[inst_n].update(saved)

        rh = state.get("regime_hist", {})
        if isinstance(rh, dict):
            for inst, arr in rh.items():
                inst_n = normalize_instrument(inst)
                if inst_n in self._regime_hist and isinstance(arr, list):
                    self._regime_hist[inst_n] = deque([str(x) for x in arr][-self.max_regime_history:], maxlen=self.max_regime_history)

        ah = state.get("action_hist", {})
        if isinstance(ah, dict):
            for inst, arr in ah.items():
                inst_n = normalize_instrument(inst)
                if inst_n in self._action_hist and isinstance(arr, list):
                    self._action_hist[inst_n] = deque([str(x) for x in arr][-self.max_regime_history:], maxlen=self.max_regime_history)

        self.log_info(f"[ThemeExpert] state restored | instruments={len(self._state)}")
