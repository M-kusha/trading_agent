# ─────────────────────────────────────────────────────────────
# File: modules/market/components/fractal_regime.py
# Fractal Regime Analysis Component (Production-Ready)
# ─────────────────────────────────────────────────────────────

from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import math
import numpy as np

# Make wavelets optional in runtime environments without pywt
try:
    import pywt  # type: ignore
    _HAS_PYWT = True
except Exception:
    pywt = None  # type: ignore
    _HAS_PYWT = False

from ..shared.base_component import BaseMarketComponent


class FractalRegimeComponent(BaseMarketComponent):
    """
    Robust fractal analysis for market regime detection.

    - Computes Hurst exponent (robust log–log slope).
    - Computes Variance Ratio across multiple lags and maps it to a bounded signal.
    - Computes (optional) Wavelet energy ratio (detail/total) if pywt is available.
    - Applies smoothing and hysteresis to avoid flip-flopping.
    - Emits stable outputs + extra diagnostics for downstream modules.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        # Default config (safe, conservative)
        default_config = {
            'window': 256,                 # more signal, still cheap
            'coeff_h': 0.45,
            'coeff_vr': 0.35,
            'coeff_we': 0.20,
            # Hysteresis thresholds (tune to your asset)
            'noise_to_volatile': 0.32,
            'volatile_to_noise': 0.24,
            'volatile_to_trending': 0.62,
            'trending_to_volatile': 0.50,
            # Smoothing & stability
            'smooth_median': 5,            # median window for score
            'min_points': 16,              # minimal points to attempt metrics
            'max_series': 1024,            # cap to avoid huge arrays
            # Variance ratio settings
            'vr_lags': [2, 4, 8],          # multi-lag VR, auto-cropped by len
            'vr_clip': 5.0,                # clip to reduce outliers before mapping
            # Wavelet
            'wavelet': 'db4',
            'wavelet_max_level': 2,
            # Synthetic fallback
            'synthetic_seed': 1337,
            'synthetic_sigma': 0.001,
        }
        # merge user config
        default_config.update(config or {})

        super().__init__(
            name="FractalRegime",
            config=default_config,
            **kwargs
        )

    # -------------------------------
    # Lifecycle
    # -------------------------------
    def initialize(self):
        """Initialize internal state & validated config."""
        cfg = self.config

        # Scalar params
        self.window = int(max(8, cfg['window']))
        self.coeff_h = float(cfg['coeff_h'])
        self.coeff_vr = float(cfg['coeff_vr'])
        self.coeff_we = float(cfg['coeff_we'])

        # Normalize weights so their sum = 1 (avoids surprise scaling)
        wsum = max(1e-9, (self.coeff_h + self.coeff_vr + self.coeff_we))
        self.coeff_h /= wsum
        self.coeff_vr /= wsum
        self.coeff_we /= wsum

        # Hysteresis thresholds
        self._noise_to_volatile = float(cfg['noise_to_volatile'])
        self._volatile_to_noise = float(cfg['volatile_to_noise'])
        self._volatile_to_trending = float(cfg['volatile_to_trending'])
        self._trending_to_volatile = float(cfg['trending_to_volatile'])

        # Windows & caps
        self._smooth_median = int(max(1, cfg.get('smooth_median', 5)))
        self._min_points = int(max(8, cfg.get('min_points', 16)))
        self._max_series = int(max(self.window, cfg.get('max_series', 1024)))

        # Variance ratio lags
        lags = cfg.get('vr_lags', [2, 4, 8])
        self._vr_lags = sorted({int(l) for l in lags if int(l) >= 2})
        self._vr_clip = float(cfg.get('vr_clip', 5.0))

        # Wavelet
        self._wavelet = str(cfg.get('wavelet', 'db4'))
        self._wavelet_max_level = int(max(1, cfg.get('wavelet_max_level', 2)))
        self._wavelets_enabled = bool(_HAS_PYWT)

        # Synthetic fallback
        self._rng = np.random.default_rng(int(cfg.get('synthetic_seed', 1337)))
        self._synthetic_sigma = float(cfg.get('synthetic_sigma', 0.001))

        # State buffers
        self._buf = deque(maxlen=max(3, int(self.window * 0.75)))
        self._regime_history = deque(maxlen=128)
        self._fractal_metrics_history = deque(maxlen=256)

        # Current state
        self.label = "noise"
        self.regime_strength = 0.0
        self._trend_direction = 0.0
        self._regime_stability_score = 1.0

        # Diagnostics
        self._last_used_synthetic = False
        self._last_series_len = 0
        self._last_volatility = 0.0

        self.trace(
            f"Fractal regime initialized | weights(H/VR/WE)={self.coeff_h:.2f}/{self.coeff_vr:.2f}/{self.coeff_we:.2f} | "
            f"wavelets={'on' if self._wavelets_enabled else 'off'}",
            level="INFO"
        )

    # -------------------------------
    # Public analysis
    # -------------------------------
    async def analyze_impl(self, **inputs) -> Dict[str, Any]:
        """Perform robust fractal regime analysis."""
        self.trace("Starting fractal analysis", level="TRACE")

        market_data = inputs.get('market_data', {})
        # shared_context = inputs.get('shared_context', {})  # not used yet

        # 1) Price series
        price_series = self._extract_price_series(market_data)
        if price_series is None or price_series.size < self._min_points:
            self.trace("Insufficient data for fractal analysis", level="WARNING")
            return self.get_fallback_result("Insufficient data")

        # Cap series size for predictable performance
        if price_series.size > self._max_series:
            price_series = price_series[-self._max_series:]

        # 2) Compute metrics
        self.trace(f"Analyzing series len={price_series.size}", level="TRACE")
        metrics = self._compute_fractal_metrics(price_series)

        # 3) Determine regime & trend
        regime, strength = self._process_regime_signals(metrics)
        trend_direction = self._calculate_trend_direction(price_series)

        # 4) Update internal state
        self._update_regime_state(regime, strength)

        # 5) Build stable output (keep existing keys)
        out = {
            'market_regime': regime,                                        # 'noise' | 'volatile' | 'trending' | 'ranging'
            'regime_strength': float(np.clip(strength, 0.0, 1.0)),         # [0,1]
            'trend_direction': float(np.clip(trend_direction, -1.0, 1.0)), # [-1,1]
            'fractal_metrics': metrics,                                     # dict with details
            'regime_data': {
                'id': self._regime_to_id(regime),
                'market_regime': regime,
                'regime_strength': float(np.clip(strength, 0.0, 1.0)),
                'trend_direction': float(np.clip(trend_direction, -1.0, 1.0)),
            },
            'regime_stability': self._calculate_stability(),               # [0,1] higher => more stable
            'processing_success': True,
            # Bonus diagnostics for downstream modules (non-breaking)
            'current_volatility': float(metrics.get('volatility', 0.0)),
            'data_quality': {
                'used_synthetic': bool(self._last_used_synthetic),
                'series_length': int(self._last_series_len),
                'min_points_required': int(self._min_points)
            },
            'fractal_capabilities': {
                'wavelets': bool(self._wavelets_enabled),
                'vr_lags': list(self._vr_lags),
                'window': int(self.window),
                'version': "2.0"
            }
        }
        return out

    # -------------------------------
    # Data extraction
    # -------------------------------
    def _extract_price_series(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract a close-price series from multiple possible shapes of market_data."""
        self.trace("Extracting price series", level="TRACE")
        self._last_used_synthetic = False
        self._last_series_len = 0

        # 1) Simple flat list of prices
        prices = market_data.get('prices')
        if isinstance(prices, list) and len(prices) >= 2 and np.isscalar(prices[-1]):
            arr = np.asarray(prices[-int(self.window * 2):], dtype=np.float64)
            self._last_series_len = int(arr.size)
            return arr

        # 2) Instrument dicts with OHLC arrays
        for symbol in ('XAUUSD',):
            if symbol in market_data:
                d = market_data[symbol]
                if isinstance(d, dict) and 'close' in d:
                    arr = np.asarray(d['close'], dtype=np.float64)
                    # If multi-timeframe structure exists, prefer the most granular
                    # np.isscalar returns a bool; we just ensure it's an array-like with len >= 2
                    if (not np.isscalar(arr)) and arr.size >= 2:
                        arr = arr[-max(self.window, self._min_points):]
                        self._last_series_len = int(arr.size)
                        return arr

        # 3) Nested shapes: 'multi_timeframe_data' or 'historical_prices' as
        #    { instrument: { timeframe: { close: [...] } } }
        for container_key in ('multi_timeframe_data', 'historical_prices', 'market_data'):
            nested = market_data.get(container_key)
            if isinstance(nested, dict):
                # Prefer common symbols and granular TFs if available
                instruments = list(nested.keys())
                if not instruments:
                    continue
                # Simple preference order
                inst_pref = ['XAUUSD'] + instruments
                tf_pref = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
                chosen_inst = next((s for s in inst_pref if s in nested), instruments[0])
                inst_block = nested.get(chosen_inst)
                if isinstance(inst_block, dict):
                    # If already a flat OHLC block with arrays
                    if 'close' in inst_block and isinstance(inst_block['close'], (list, np.ndarray)):
                        arr = np.asarray(inst_block['close'], dtype=np.float64)
                        if arr.size >= 2:
                            arr = arr[-max(self.window, self._min_points):]
                            self._last_series_len = int(arr.size)
                            self.trace(
                                f"Fractal source selected: {container_key}:{chosen_inst} flat close series len={arr.size}",
                                level="TRACE",
                            )
                            return arr
                    # Otherwise expect per-timeframe dicts
                    # Pick preferred timeframe if present
                    if any(isinstance(v, dict) for v in inst_block.values()):
                        # gather available timeframes
                        tfs = list(inst_block.keys())
                        chosen_tf = next((t for t in tf_pref if t in inst_block), tfs[0])
                        tf_rec = inst_block.get(chosen_tf)
                        if isinstance(tf_rec, dict):
                            # Some shapes have current_bar + arrays; prefer arrays
                            series = tf_rec.get('close') or tf_rec.get('prices')
                            if isinstance(series, (list, np.ndarray)):
                                arr = np.asarray(series, dtype=np.float64)
                                if arr.size >= 2:
                                    arr = arr[-max(self.window, self._min_points):]
                                    self._last_series_len = int(arr.size)
                                    self.trace(
                                        f"Fractal source selected: {container_key}:{chosen_inst}:{chosen_tf} close series len={arr.size}",
                                        level="TRACE",
                                    )
                                    return arr
                            # Fallback: if only current_bar present, cannot build series
                            cb = tf_rec.get('current_bar')
                            if isinstance(cb, dict) and 'close' in cb:
                                # accumulate a minimal synthetic history around the current close
                                base = float(cb.get('close') or 0.0)
                                if base > 0:
                                    noise = self._rng.normal(0.0, self._synthetic_sigma, self.window)
                                    arr = (base * (1.0 + np.cumsum(noise))).astype(np.float64)
                                    self._last_series_len = int(arr.size)
                                    self._last_used_synthetic = True
                                    self.trace(
                                        f"Fractal source had only current_bar for {container_key}:{chosen_inst}:{chosen_tf}; synthesized series around base={base:.5f}",
                                        level="TRACE",
                                    )
                                    return arr

        # 4) Synthetic fallback (keeps pipeline alive)
        self.trace("No price data found, generating synthetic", level="WARNING")
        self._last_used_synthetic = True
        arr = (1.1 + self._rng.normal(0.0, self._synthetic_sigma, self.window)).astype(np.float64)
        self._last_series_len = int(arr.size)
        return arr

    # -------------------------------
    # Metrics
    # -------------------------------
    def _compute_fractal_metrics(self, ts: np.ndarray) -> Dict[str, float]:
        """Compute Hurst, VR(signalized), Wavelet energy, and volatility."""
        metrics: Dict[str, float] = {
            "H": 0.5,
            "VR_raw": 1.0,
            "VR_signal": 0.0,  # mapped to [-1,1] (neg ~ mean-reverting, pos ~ trending)
            "WE": 0.0,
            "volatility": 0.0
        }

        if ts.size < self._min_points or float(np.std(ts)) < 1e-12:
            return metrics

        try:
            # Basic volatility on last N returns
            vol = self._safe_volatility(ts)
            metrics["volatility"] = float(vol)

            # Hurst on up to 500 points
            metrics["H"] = self._hurst_enhanced(ts)

            # Variance ratio (multi-lag) -> raw VR ~ 1 for random walk
            vr_raw = self._variance_ratio_multi(ts, self._vr_lags, clip=self._vr_clip)
            metrics["VR_raw"] = float(vr_raw)
            # Map VR to symmetric signal in [-1, 1]
            # VR > 1 => trending (+), VR < 1 => mean-reverting (-), VR == 1 => neutral (0)
            metrics["VR_signal"] = self._map_vr_to_signal(vr_raw)

            # Wavelet energy ratio (if available)
            if self._wavelets_enabled and ts.size >= 16:
                metrics["WE"] = self._wavelet_energy_enhanced(
                    ts, wavelet=self._wavelet, max_level=self._wavelet_max_level
                )

            # Keep a light history for diagnostics
            self._fractal_metrics_history.append({
                'metrics': dict(metrics),
                'series_length': int(ts.size)
            })

        except Exception as e:
            self.trace(f"Error computing fractal metrics: {e}", level="ERROR")

        return metrics

    @staticmethod
    def _safe_volatility(ts: np.ndarray, lookback: int = 64) -> float:
        """Return std of returns over last N points with guardrails."""
        t = ts[-max(lookback, 8):].astype(float)
        if t.size < 3 or float(np.std(t)) < 1e-16:
            return 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            base = np.where(t[:-1] == 0.0, 1.0, t[:-1])
            rets = (t[1:] - t[:-1]) / base
        rets = rets[np.isfinite(rets)]
        if rets.size < 2:
            return 0.0
        return float(np.std(rets))

    @staticmethod
    def _hurst_enhanced(series: np.ndarray) -> float:
        """Robust Hurst exponent via log–log slope of std of increments over lags."""
        s = series[-500:].astype(float)
        if s.size < 10 or float(np.std(s)) < 1e-12:
            return 0.5

        try:
            max_lag = max(2, min(50, s.size // 3))
            if max_lag < 3:
                return 0.5

            # Choose spaced lags for stability
            lags = np.unique(np.logspace(0.3, np.log10(max_lag), 15).astype(int))
            lags = lags[(lags >= 2) & (lags < s.size)]
            if lags.size < 3:
                return 0.5

            tau = []
            for lag in lags:
                diff = s[lag:] - s[:-lag]
                if diff.size > 0:
                    tau.append(np.std(diff))

            tau = np.asarray(tau, dtype=float)
            valid = (tau > 0) & np.isfinite(tau)
            if np.sum(valid) < 3:
                return 0.5

            log_lags = np.log(lags[valid])
            log_tau = np.log(tau[valid])
            coef = np.polyfit(log_lags, log_tau, 1)
            slope = float(coef[0])
            hurst = float(np.clip(slope * 2.0, 0.0, 1.0))
            return hurst
        except Exception:
            return 0.5

    @staticmethod
    def _variance_ratio_multi(ts: np.ndarray, lags: List[int], clip: float = 5.0) -> float:
        """
        Multi-lag variance ratio averaged across lags, clipped to control outliers.
        VR(k) = Var(k-step returns)/[k * Var(1-step returns)].
        """
        t = ts[-max(256, 4):].astype(float)
        if t.size < 4:
            return 1.0

        try:
            one_ret = t[1:] - t[:-1]
            var_1 = float(np.var(one_ret))
            if not np.isfinite(var_1) or var_1 <= 1e-14:
                return 1.0

            vrs = []
            for k in lags:
                if k >= t.size:
                    continue
                k_ret = t[k:] - t[:-k]
                if k_ret.size == 0:
                    continue
                var_k = float(np.var(k_ret))
                vr = (var_k / max(var_1, 1e-14)) / float(k)
                if np.isfinite(vr):
                    vrs.append(float(np.clip(vr, 1.0 / max(clip, 1e-6), clip)))

            if not vrs:
                return 1.0
            return float(np.mean(vrs))
        except Exception:
            return 1.0

    @staticmethod
    def _map_vr_to_signal(vr: float) -> float:
        """
        Map VR in (0,∞) to a smooth bounded signal in [-1,1]:
          signal = tanh( ln(VR) ), so VR>1 => positive (trend), VR<1 => negative (mean-revert).
        """
        if not np.isfinite(vr) or vr <= 0:
            return 0.0
        val = math.tanh(math.log(vr))
        return float(np.clip(val, -1.0, 1.0))

    @staticmethod
    def _wavelet_energy_enhanced(series: np.ndarray, wavelet: str = "db4", max_level: int = 2) -> float:
        """Detail energy / total energy using DWT, safely bounded."""
        if not _HAS_PYWT:
            return 0.0
        s = series[-256:].astype(float)
        if s.size < 16 or float(np.std(s)) < 1e-12:
            return 0.0
        try:
            lvl_max = min(max_level, pywt.dwt_max_level(len(s), wavelet))  # type: ignore
            if lvl_max < 1:
                return 0.0
            coeffs = pywt.wavedec(s, wavelet, level=lvl_max)  # type: ignore
            detail_energy = 0.0
            for i in range(1, len(coeffs)):
                c = np.asarray(coeffs[i], dtype=float)
                detail_energy += float(np.sum(c * c))
            total = float(np.sum(s * s))
            if total <= 1e-14:
                return 0.0
            er = float(np.clip(detail_energy / total, 0.0, 1.0))
            return er if np.isfinite(er) else 0.0
        except Exception:
            return 0.0

    # -------------------------------
    # Regime logic
    # -------------------------------
    def _process_regime_signals(self, metrics: Dict[str, float]) -> Tuple[str, float]:
        """
        Combine H, VR_signal, WE into a strength in [0,1] and classify with hysteresis.
        - H is centered at 0.5; we map to [-1,1] around 0.5 to represent anti/persistence.
        - VR_signal already in [-1,1].
        - WE in [0,1] but we treat it as "choppiness" proxy contributing to volatility.
        """
        H = float(np.clip(metrics.get("H", 0.5), 0.0, 1.0))
        # Map H around 0.5 -> [-1,1]
        H_sig = float(np.clip((H - 0.5) * 2.0, -1.0, 1.0))

        VR_sig = float(np.clip(metrics.get("VR_signal", 0.0), -1.0, 1.0))
        WE = float(np.clip(metrics.get("WE", 0.0), 0.0, 1.0))

        # Weighted combination. Intuition:
        # - H_sig > 0 supports trending; <0 supports mean-reversion (ranging).
        # - VR_sig > 0 supports trending; <0 supports mean-reversion.
        # - WE increases microstructure noise => leans toward "volatile/noise".
        trend_component = self.coeff_h * H_sig + self.coeff_vr * VR_sig
        noise_component = self.coeff_we * WE

        # Raw combined trend score in [-1, 1]; subtract noise penalty
        raw = float(np.clip(trend_component - 0.5 * noise_component, -1.0, 1.0))

        # Normalize to [0,1] strength: magnitude of conviction
        # 0 => neutral/uncertain, 1 => very strong
        strength = float(np.clip(0.5 * (abs(raw) + (1.0 - 0.5 * noise_component)), 0.0, 1.0))

        # Smooth the score over recent observations
        self._buf.append(strength)
        if len(self._buf) >= self._smooth_median:
            smoothed = float(np.median(list(self._buf)[-self._smooth_median:]))
        else:
            smoothed = float(np.mean(self._buf)) if self._buf else 0.0

        # Classify with hysteresis
        old = self.label
        new = self._determine_regime_with_hysteresis(old, smoothed, raw)

        if new != old:
            self.trace(f"Regime transition: {old} -> {new}", level="INFO")

        return new, float(np.clip(smoothed, 0.0, 1.0))

    def _determine_regime_with_hysteresis(self, old_label: str, strength: float, raw_trend: float) -> str:
        """
        Apply hysteresis on strength with a bias from raw_trend sign:
          - positive raw_trend biases to 'trending'
          - negative raw_trend biases to 'ranging'
          - high WE (already penalized) tends to 'volatile' if strength moderate
        """
        # Base decision from thresholds
        if old_label == "noise":
            base = "volatile" if strength >= self._noise_to_volatile else "noise"
        elif old_label == "volatile":
            if strength >= self._volatile_to_trending:
                base = "trending" if raw_trend >= 0 else "ranging"
            elif strength < self._volatile_to_noise:
                base = "noise"
            else:
                base = "volatile"
        else:  # previously trending/ranging considered "trending" path for hysteresis
            base = "volatile" if strength < self._trending_to_volatile else ("trending" if raw_trend >= 0 else "ranging")

        # Anti-flip stability gate: if recent history is mixed and conviction near mid, hold old label
        if len(self._regime_history) >= 6 and base != old_label:
            recent = [r[0] for r in list(self._regime_history)[-6:]]
            unique_recent = len(set(recent))
            if unique_recent >= 3 and 0.35 <= strength <= 0.65:
                base = old_label

        return base

    def _calculate_trend_direction(self, ts: np.ndarray) -> float:
        """Signed percent change (short vs long mean) with clipping, robust to flats."""
        if ts.size < 3:
            return 0.0
        try:
            short_n, long_n = 8, 24
            short = float(np.mean(ts[-short_n:])) if ts.size >= short_n else float(ts[-1])
            long = float(np.mean(ts[-long_n:])) if ts.size >= long_n else float(np.mean(ts))
            denom = long if abs(long) > 1e-12 else 1.0
            return float(np.clip((short - long) / abs(denom), -1.0, 1.0))
        except Exception:
            return 0.0

    def _update_regime_state(self, regime: str, strength: float):
        """Update internal regime state & history."""
        self.label = regime
        self.regime_strength = float(np.clip(strength, 0.0, 1.0))
        # keep last computed trend direction if available
        self._regime_history.append((regime, self.regime_strength, self._trend_direction))

    def _calculate_stability(self) -> float:
        """Higher when recent labels have lower variety."""
        if len(self._regime_history) < 8:
            return 0.5
        recent = [r[0] for r in list(self._regime_history)[-12:]]
        uniq = len(set(recent))
        # 1.0 -> all same label; degrade 0.15 per additional unique regime
        stability = max(0.0, 1.0 - (uniq - 1) * 0.15)
        self._regime_stability_score = float(np.clip(stability, 0.0, 1.0))
        return self._regime_stability_score

    @staticmethod
    def _regime_to_id(regime: str) -> int:
        """Stable ID mapping."""
        mapping = {"noise": 0, "ranging": 1, "trend": 2, "trending": 2, "volatile": 3}
        return mapping.get(regime, mapping.get(regime.lower(), 0))

    # -------------------------------
    # Caching (optional speedup)
    # -------------------------------
    def _get_cache_key(self, inputs: Dict[str, Any]) -> Optional[str]:
        """
        Cache key from last few price stats to avoid recompute when data unchanged.
        Uses series length, last value, and std signature.
        """
        md = inputs.get('market_data', {})
        ps = self._extract_price_series(md)
        if ps is None or ps.size < self._min_points:
            return None
        tail = ps[-min(32, ps.size):]
        key = (
            int(ps.size),
            float(round(ps[-1], 8)),
            float(round(np.std(tail), 8)),
        )
        return f"{self.name}:{key}"

    # -------------------------------
    # Fallback
    # -------------------------------
    def get_fallback_result(self, error: str) -> Dict[str, Any]:
        """Provide safe, structured fallback on error."""
        return {
            'market_regime': self.label,
            'regime_strength': 0.0,
            'trend_direction': 0.0,
            'fractal_metrics': {"H": 0.5, "VR_raw": 1.0, "VR_signal": 0.0, "WE": 0.0, "volatility": 0.0},
            'regime_data': {
                'id': self._regime_to_id(self.label),
                'market_regime': self.label,
                'regime_strength': 0.0,
                'trend_direction': 0.0,
            },
            'regime_stability': float(getattr(self, "_regime_stability_score", 0.5)),
            'processing_success': False,
            'error': error,
            'data_quality': {
                'used_synthetic': bool(self._last_used_synthetic),
                'series_length': int(self._last_series_len),
                'min_points_required': int(self._min_points)
            },
        }
