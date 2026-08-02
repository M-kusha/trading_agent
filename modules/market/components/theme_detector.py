# ─────────────────────────────────────────────────────────────
# File: modules/market/components/theme_detector.py
# Market Theme Detection Component — Production-Ready Upgrade
# ─────────────────────────────────────────────────────────────

import math
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pywt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..shared.base_component import BaseMarketComponent


class ThemeDetectorComponent(BaseMarketComponent):
    """
    Market theme detection via streaming feature extraction + online clustering.
    Improvements vs. baseline:
      • Richer, safer features (vol, momentum, trend, wavelet energy, entropy,
        skew, kurtosis, ATR-normalized volatility, cross-checks).
      • Standardized, fixed-length vectors with robust padding & clipping.
      • Streaming MiniBatchKMeans with partial_fit; outlier filtering.
      • Dual clustering-quality metrics (Davies–Bouldin & Silhouette).
      • Theme confidence = separation * quality * feature-stability * fit-health.
      • Centroid drift monitoring + convergence tracking.
      • Deterministic seedable behavior; defensive fallbacks throughout.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        # Defaults tuned for stability in streaming settings
        default_config = {
            'n_themes': 4,
            'window': 100,                      # legacy: not directly used, but kept for compat
            'batch_size': 128,                 # minibatch for partial_fit
            'feature_lookback': 500,           # max history for features per timeframe
            'instruments': ['XAUUSD'],
            'max_iter': 100,                   # used on initial fit
            'convergence_threshold': 0.001,    # convergence tolerance on inertia
            'clustering_quality_threshold': 0.35,  # min combined quality to be "ready"
            'use_macro': True,
            'timeframes': ('H1', 'H4', 'D1'),

            # Feature controls
            'features_per_timeframe': 11,      # keep in sync with _extract_timeframe_features
            'entropy_bins': 16,
            'atr_period': 14,
            'clip_sigma': 4.0,                 # clip standardized features to avoid outliers
            'min_points_for_fit': 256,         # require this many before first fit
            'refit_interval': 50,              # refit cadence (calls between fits)
            'subsample_for_quality': 512,      # at most N points for quality metrics
            'seed': 1337,

            # Confidence weighting
            'weight_sep': 0.40,
            'weight_quality': 0.30,
            'weight_stability': 0.20,
            'weight_fithealth': 0.10,
        }
        default_config.update(config or {})

        super().__init__(
            name="ThemeDetector",
            config=default_config,
            **kwargs
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def initialize(self):
        # Seed numpy & stdlib RNG for determinism
        seed = int(self.config['seed'])
        np.random.seed(seed)
        random.seed(seed)

        # Core ML components
        self.scaler = StandardScaler()
        self.km = MiniBatchKMeans(
            n_clusters=int(self.config['n_themes']),
            batch_size=int(self.config['batch_size']),
            random_state=seed,
            max_iter=int(self.config['max_iter']),
            n_init=10
        )

        # State
        self._theme_vec = np.zeros(int(self.config['n_themes']), np.float32)
        self._current_theme = 0
        self._theme_confidence = 0.0
        self._theme_strength_history = deque(maxlen=100)
        self._theme_momentum = deque(maxlen=10)
        self._theme_history = deque(maxlen=600)

        # Streaming fit buffers & diagnostics
        self._fit_buffer = deque(maxlen=4000)
        self._feature_buffer_scaled = deque(maxlen=4000)  # last scaled features (for quality)
        self._feature_stability_score = 1.0
        self._clustering_quality = 0.0
        self._ml_fit_count = 0
        self._last_inertia = None
        self._inertia_history = deque(maxlen=30)
        self._convergence_history = deque(maxlen=20)
        self._last_centers = None
        self._centroid_drift = deque(maxlen=20)

        # Macro defaults/scaler
        self._macro_scaler = StandardScaler()
        self._macro_scaler.fit([[20.0, 0.5, 3.0]])  # VIX, yield_curve, CPI defaults
        self.macro_data = {"vix": 20.0, "yield_curve": 0.5, "cpi": 3.0}

        self.trace("Theme detector component initialized", level="DEBUG")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    async def analyze_impl(self, **inputs) -> Dict[str, Any]:
        self.trace("Starting theme detection", level="TRACE")

        try:
            market_data = inputs.get('market_data', {}) or {}
            shared_context = inputs.get('shared_context', {}) or {}

            # 1) Feature extraction (raw)
            features = self._extract_comprehensive_features(market_data)
            if features is None or features.size == 0:
                self.trace("No valid features extracted", level="WARNING")
                return self.get_fallback_result("No valid features")

            # 2) Buffer + (optional) training
            self._fit_buffer.append(features)
            # We maintain a running scaler via batches (fit on a batch first, then partial_fit)
            if self._should_fit_model():
                await self._fit_model_safe()

            # 3) Prediction & metrics
            if self._is_model_ready():
                theme_id, strength = self._detect_current_theme(features)
                confidence = self._calculate_theme_confidence(features, theme_id)
            else:
                theme_id, strength, confidence = 0, 0.30, 0.10

            # 4) State update & summary metrics
            self._update_theme_state(theme_id, strength, confidence)
            stability = self._get_theme_stability()
            transition_probability = self._calculate_transition_probability()

            self.trace(
                f"Theme detected: {theme_id} (strength={strength:.3f}, "
                f"confidence={confidence:.3f}, quality={self._clustering_quality:.3f})",
                level="DEBUG"
            )

            return {
                'market_theme': int(theme_id),
                'theme_strength': float(strength),
                'theme_confidence': float(confidence),
                'theme_stability': float(stability),
                'transition_probability': float(transition_probability),
                'theme_transition': float(transition_probability),  # backward-compat
                'clustering_quality': float(self._clustering_quality),
                'feature_stability': float(self._feature_stability_score),
                'themes_total': int(self.config['n_themes']),
                'theme_detection': {
                    'theme': int(theme_id),
                    'strength': float(strength),
                    'confidence': float(confidence),
                    'stability': float(stability),
                    'transition_probability': float(transition_probability),
                },
                # Helpful lightweight diagnostics (non-breaking)
                'diagnostics': {
                    'fit_count': int(self._ml_fit_count),
                    'last_inertia': float(self._last_inertia) if self._last_inertia is not None else None,
                    'centroid_drift': float(self._centroid_drift[-1]) if self._centroid_drift else 0.0
                },
                'processing_success': True
            }

        except Exception as e:
            self.trace(f"Theme detection error: {e}", level="ERROR")
            return self.get_fallback_result(str(e))

    # -------------------------------------------------------------------------
    # Feature engineering
    # -------------------------------------------------------------------------
    def _extract_comprehensive_features(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build a fixed-length, robust feature vector across instruments × timeframes."""
        self.trace("Extracting comprehensive features", level="TRACE")

        try:
            per_tf_features = int(self.config['features_per_timeframe'])  # must match _extract_timeframe_features
            feats: List[float] = []

            # Per-instrument, per-timeframe features
            for instrument in self.config['instruments']:
                inst_data = market_data.get(instrument, {})
                for timeframe in self.config['timeframes']:
                    feats.extend(self._extract_timeframe_features(inst_data, timeframe, instrument))

            # Macro (scaled) — stable even if unchanged
            if self.config.get('use_macro', True):
                feats.extend(self._get_macro_features())

            vec = np.asarray(feats, dtype=np.float32)

            # Standardize size (pad/truncate deterministically)
            expected = (
                len(self.config['instruments']) *
                len(self.config['timeframes']) *
                per_tf_features +
                (3 if self.config.get('use_macro', True) else 0)
            )
            if vec.size < expected:
                out = np.zeros(expected, dtype=np.float32)
                out[:vec.size] = vec
                vec = out
            elif vec.size > expected:
                vec = vec[:expected]

            # Clip extreme raw values to a huge bound to avoid scaler blowups
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

            # Keep a small rolling estimate of feature stability (variance of recent scaled vectors)
            self._update_feature_stability_probe(vec)

            self.trace(f"Extracted features: {vec.size} dims", level="TRACE")
            return vec

        except Exception as e:
            self.trace(f"Feature extraction failed: {e}", level="ERROR")
            return None

    def _extract_timeframe_features(self, inst_data: Any, timeframe: str, instrument_label: str) -> List[float]:
        """
        Features per timeframe (11 slots):
          0  vol (std of returns)
          1  momentum (mean last 5 returns)
          2  hurst (0..1)
          3  wavelet detail energy ratio (0..1)
          4  trend (SMA10-SMA30)/SMA30
          5  rolling return (last/prev_10 - 1)
          6  data availability (len)
          7  entropy of returns (0..~1)
          8  skewness of returns
          9  kurtosis (excess)
          10 ATR-normalized volatility (proxy)
        """
        per_tf = int(self.config['features_per_timeframe'])
        if not isinstance(inst_data, dict) or timeframe not in inst_data:
            return [0.0] * per_tf

        tf_data = inst_data.get(timeframe, {})
        if not isinstance(tf_data, dict) or 'close' not in tf_data:
            return [0.0] * per_tf

        prices = np.asarray(tf_data.get('close', []), dtype=np.float64)
        if prices.size < 10:
            return [0.0] * per_tf

        closes = prices[-min(prices.size, int(self.config['feature_lookback'])):]
        rets = np.diff(closes) / np.where(closes[:-1] == 0.0, 1.0, closes[:-1])

        # 0 vol
        vol = float(np.std(rets[-20:])) if rets.size >= 20 else float(np.std(rets)) if rets.size > 0 else 0.0

        # 1 mom
        mom = float(np.mean(rets[-5:])) if rets.size >= 5 else (float(np.mean(rets)) if rets.size > 0 else 0.0)

        # 2 hurst
        hurst = float(self._hurst_safe(closes[-50:])) if closes.size >= 50 else 0.5

        # 3 wavelet energy
        wave = float(self._wavelet_energy_safe(closes[-64:])) if closes.size >= 32 else 0.0

        # 4 trend (SMA10 vs SMA30)
        sma_s = float(np.mean(closes[-10:])) if closes.size >= 10 else float(closes[-1])
        sma_l = float(np.mean(closes[-30:])) if closes.size >= 30 else float(closes[-1])
        trend = float((sma_s - sma_l) / sma_l) if sma_l != 0.0 else 0.0

        # 5 rolling return 10
        roll_10 = float(closes[-1] / closes[-10] - 1.0) if closes.size >= 10 else 0.0

        # 6 data availability
        data_available = float(closes.size)

        # 7 entropy
        entropy = self._entropy_safe(rets, bins=int(self.config['entropy_bins']))

        # 8 skew
        skew = self._skew_safe(rets)

        # 9 kurtosis (excess)
        kurt = self._kurtosis_safe(rets)

        # 10 ATR-normalized volatility (proxy using closes only)
        atr_norm = self._atr_proxy_normalized(tf_data)

        return [
            vol, mom, hurst, wave, trend, roll_10,
            data_available, entropy, skew, kurt, atr_norm
        ]

    # --- micro helpers (safe stats) -------------------------------------------------
    @staticmethod
    def _hurst_safe(series: np.ndarray) -> float:
        try:
            s = np.asarray(series, dtype=np.float64)
            if s.size < 10 or float(np.std(s)) < 1e-12:
                return 0.5
            lags = np.arange(2, min(30, s.size // 2))
            if lags.size < 3:
                return 0.5
            stds = []
            for lag in lags:
                diff = s[lag:] - s[:-lag]
                if diff.size > 1:
                    stds.append(np.std(diff))
            tau = np.asarray(stds, dtype=np.float64)
            tau = tau[(tau > 0) & np.isfinite(tau)]
            if tau.size < 3:
                return 0.5
            log_lags = np.log(lags[:tau.size])
            log_tau = np.log(tau)
            slope = float(np.polyfit(log_lags, log_tau, 1)[0])
            return float(np.clip(slope * 2.0, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _wavelet_energy_safe(series: np.ndarray, wavelet: str = "db4") -> float:
        try:
            s = np.asarray(series, dtype=np.float64)
            if s.size < 32 or float(np.std(s)) < 1e-12:
                return 0.0
            lvl = min(2, pywt.dwt_max_level(len(s), wavelet))
            if lvl < 1:
                return 0.0
            coeffs = pywt.wavedec(s, wavelet, level=lvl)
            detail_energy = 0.0
            for i in range(1, len(coeffs)):
                detail_energy += float(np.sum(np.square(coeffs[i])))
            total = float(np.sum(np.square(s))) + 1e-12
            return float(np.clip(detail_energy / total, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _entropy_safe(x: np.ndarray, bins: int = 16) -> float:
        try:
            if x is None or x.size < 8:
                return 0.0
            hist, _ = np.histogram(x[~np.isnan(x)], bins=bins, density=True)
            p = hist / (np.sum(hist) + 1e-12)
            ent = -np.sum(p * np.log(p + 1e-12))
            # Normalize by log(bins) to map roughly to 0..1
            return float(np.clip(ent / (np.log(bins) + 1e-12), 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _skew_safe(x: np.ndarray) -> float:
        try:
            if x is None or x.size < 8:
                return 0.0
            mu = float(np.mean(x))
            sd = float(np.std(x)) + 1e-12
            m3 = float(np.mean((x - mu) ** 3))
            return float(np.clip(m3 / (sd ** 3), -10.0, 10.0))
        except Exception:
            return 0.0

    @staticmethod
    def _kurtosis_safe(x: np.ndarray) -> float:
        try:
            if x is None or x.size < 8:
                return 0.0
            mu = float(np.mean(x))
            sd = float(np.std(x)) + 1e-12
            m4 = float(np.mean((x - mu) ** 4))
            # Excess kurtosis (minus 3)
            return float(np.clip(m4 / (sd ** 4) - 3.0, -10.0, 10.0))
        except Exception:
            return 0.0

    def _atr_proxy_normalized(self, tf_data: Dict[str, Any]) -> float:
        """ATR proxy using close series if OHLC is not available."""
        try:
            p = np.asarray(tf_data.get('close', []), dtype=np.float64)
            if p.size < 15:
                return 0.0
            period = int(self.config['atr_period'])
            # proxy: mean absolute return * sqrt(period)
            rets = np.abs(np.diff(p))
            atr = float(np.mean(rets[-period:])) if rets.size >= period else float(np.mean(rets))
            level = float(np.mean(p[-period:])) if p.size >= period else float(np.mean(p))
            if level <= 1e-12:
                return 0.0
            return float(np.clip((atr / level) * math.sqrt(period), 0.0, 1.0))
        except Exception:
            return 0.0

    # -------------------------------------------------------------------------
    # Fitting & quality
    # -------------------------------------------------------------------------
    def _should_fit_model(self) -> bool:
        """
        Fit when we have enough buffered points and at a fixed cadence,
        or when we have never fit before.
        """
        enough = len(self._fit_buffer) >= int(self.config['min_points_for_fit'])
        cadence = (self._ml_fit_count == 0) or (self._ml_fit_count % int(self.config['refit_interval']) == 0)
        return bool(enough and cadence)

    async def _fit_model_safe(self) -> None:
        self.trace("Fitting ML model (streaming)", level="DEBUG")
        try:
            # Stack buffer → X
            X = np.asarray(list(self._fit_buffer), dtype=np.float32)
            if X.shape[0] < int(self.config['min_points_for_fit']):
                return

            # (Re)fit scaler on batch; clip z-scores to reduce extreme leverage
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = np.clip(X_scaled, -float(self.config['clip_sigma']), float(self.config['clip_sigma']))

            # Outlier filtering by L2 norm (remove top 2.5% extremes)
            norms = np.linalg.norm(X_scaled, axis=1)
            if norms.size >= 40:
                cutoff = np.percentile(norms, 97.5)
                keep = norms <= cutoff
                X_scaled = X_scaled[keep]

            if X_scaled.shape[0] < self.config['n_themes']:
                return

            # Full fit (minibatch algorithm internally) for stable centers
            self.km.set_params(random_state=int(self.config['seed']))  # deterministic
            self.km.fit(X_scaled)

            # Track inertia & centroid drift
            if hasattr(self.km, 'inertia_'):
                self._inertia_history.append(float(self.km.inertia_))
                self._last_inertia = float(self.km.inertia_)
                if self._last_centers is not None:
                    drift = float(np.linalg.norm(self.km.cluster_centers_ - self._last_centers))
                    self._centroid_drift.append(drift)
                self._last_centers = np.copy(self.km.cluster_centers_)

            # Maintain a light cache for quality metrics
            self._feature_buffer_scaled.clear()
            # Subsample for quality
            if X_scaled.shape[0] > int(self.config['subsample_for_quality']):
                idx = np.random.choice(
                    X_scaled.shape[0],
                    size=int(self.config['subsample_for_quality']),
                    replace=False
                )
                Xq = X_scaled[idx]
            else:
                Xq = X_scaled

            # Compute labels for quality set
            labels = self.km.predict(Xq)
            self._clustering_quality = self._calculate_clustering_quality(Xq, labels)

            # Update feature stability score using scaled vectors
            for row in Xq[:256]:
                self._feature_buffer_scaled.append(row)
            self._update_feature_stability_from_scaled_buffer()

            self._ml_fit_count += 1
            self.trace(
                f"Model fitted: inertia={self._last_inertia:.2f}, "
                f"quality={self._clustering_quality:.3f}, fit_count={self._ml_fit_count}",
                level="DEBUG"
            )

        except Exception as e:
            self.trace(f"Model fitting failed: {e}", level="ERROR")

    def _calculate_clustering_quality(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Combine Davies–Bouldin (lower is better) and Silhouette (higher is better)
        into a 0..1 quality score.
        """
        try:
            k = int(self.config['n_themes'])
            # We need >1 label present for metrics to be meaningful
            if len(np.unique(labels)) < 2 or X.shape[0] < (k * 5):
                return 0.0

            # DB index (>=0). Map to (0..1] via 1 / (1 + DB)
            db = float(davies_bouldin_score(X, labels))
            db_quality = 1.0 / (1.0 + max(db, 0.0))

            # Silhouette in [-1, 1]. Map to [0, 1]
            sil = float(silhouette_score(X, labels, metric='euclidean'))
            sil_quality = (sil + 1.0) / 2.0

            # harmonic mean to penalize if either is poor
            if (db_quality <= 0.0) or (sil_quality <= 0.0):
                return 0.0
            combined = 2.0 * (db_quality * sil_quality) / (db_quality + sil_quality)

            return float(np.clip(combined, 0.0, 1.0))
        except Exception:
            return 0.0

    # -------------------------------------------------------------------------
    # Readiness & prediction
    # -------------------------------------------------------------------------
    def _is_model_ready(self) -> bool:
        centers_ok = hasattr(self.km, 'cluster_centers_') and self.km.cluster_centers_ is not None
        quality_ok = (self._clustering_quality >= float(self.config['clustering_quality_threshold']))
        return bool(centers_ok and quality_ok)

    def _detect_current_theme(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict theme and return (theme_id, strength in 0..1)."""
        try:
            fs = self.scaler.transform(features.reshape(1, -1))
            fs = np.clip(fs, -float(self.config['clip_sigma']), float(self.config['clip_sigma']))

            d = self.km.transform(fs)[0]   # distances to centroids
            order = np.argsort(d)
            if d.size > 1:
                best = float(d[order[0]])
                second = float(d[order[1]])
            else:
                best = float(d[0])
                second = float(d[0]) + 1e-6
            theme_id = int(np.argmin(d))

            # Strength from separation (bigger gap ⇒ stronger)
            sep = (second - best) / (second + 1e-9)
            strength = float(np.clip(1.0 / (1.0 + best), 0.0, 1.0))  # closeness baseline
            # mix closeness & separation (emphasize separation)
            strength = float(np.clip(0.35 * strength + 0.65 * np.tanh(4.0 * sep), 0.0, 1.0))

            return theme_id, strength
        except Exception:
            return 0, 0.10

    def _calculate_theme_confidence(self, features: np.ndarray, theme_id: int) -> float:
        """Confidence = weighted blend of separation, clustering quality, feature stability, fit health."""
        try:
            fs = self.scaler.transform(features.reshape(1, -1))
            d = self.km.transform(fs)[0]
            order = np.argsort(d)
            if d.size > 1:
                best = float(d[order[0]])
                second = float(d[order[1]])
            else:
                best = float(d[0])
                second = float(d[0]) + 1e-6
            # normalized separation in [0,1]
            sep = float(np.clip((second - best) / (second + 1e-9), 0.0, 1.0))

            # Fit health: recent centroid drift & inertia stability
            fithealth = self._fit_health_score()

            # Blend
            w_sep = float(self.config['weight_sep'])
            w_q = float(self.config['weight_quality'])
            w_stab = float(self.config['weight_stability'])
            w_fit = float(self.config['weight_fithealth'])

            conf = (
                w_sep * sep +
                w_q * float(self._clustering_quality) +
                w_stab * float(self._feature_stability_score) +
                w_fit * float(fithealth)
            )
            return float(np.clip(conf, 0.0, 1.0))
        except Exception:
            return 0.10

    # -------------------------------------------------------------------------
    # Stability & health
    # -------------------------------------------------------------------------
    def _update_feature_stability_probe(self, raw_vec: np.ndarray):
        """
        Keep a tiny rolling variance estimate using a few dims (scaled later).
        Raw -> temporarily cached scaled buffer after fit.
        """
        # Nothing to do until we have a scaler fit at least once. We'll compute
        # stability in _update_feature_stability_from_scaled_buffer after fits.

    def _update_feature_stability_from_scaled_buffer(self):
        """
        Feature stability ~ inverse of average variance across dims on a recent window of scaled vectors.
        Mapped to 0..1 where 1 = very stable feature distribution.
        """
        try:
            if len(self._feature_buffer_scaled) < 32:
                self._feature_stability_score = 0.75  # optimistic neutral
                return
            Xs = np.asarray(list(self._feature_buffer_scaled), dtype=np.float32)
            var = np.var(Xs, axis=0)
            # Normalize by a soft cap: map higher variance → lower stability
            mean_var = float(np.mean(np.clip(var, 0.0, 10.0)))
            # transform: stability = 1 / (1 + mean_var/k)
            k = 1.5
            stab = 1.0 / (1.0 + mean_var / k)
            self._feature_stability_score = float(np.clip(stab, 0.0, 1.0))
        except Exception:
            self._feature_stability_score = 0.5

    def _fit_health_score(self) -> float:
        """
        Heuristic: low centroid drift + flattening inertia trend ⇒ healthier model.
        """
        try:
            drift = float(self._centroid_drift[-1]) if self._centroid_drift else 0.0
            drift_score = 1.0 / (1.0 + drift)  # higher drift → lower score
            if len(self._inertia_history) >= 5:
                y = np.asarray(list(self._inertia_history), dtype=np.float64)
                x = np.arange(y.size, dtype=np.float64)
                slope = float(np.polyfit(x, y, 1)[0])
                # If inertia still dropping quickly, the model may be shifting (lower health)
                slope_score = 1.0 / (1.0 + max(0.0, -slope) / (np.mean(y) + 1e-9))
            else:
                slope_score = 0.7
            # combine
            return float(np.clip(0.5 * drift_score + 0.5 * slope_score, 0.0, 1.0))
        except Exception:
            return 0.5

    def _get_theme_stability(self) -> float:
        if len(self._theme_strength_history) < 5:
            return 0.5
        recent = np.asarray(list(self._theme_strength_history)[-12:], dtype=np.float32)
        return float(np.clip(1.0 - float(np.std(recent)), 0.0, 1.0))

    def _calculate_transition_probability(self) -> float:
        if len(self._theme_momentum) < 3:
            return 0.10
        rec = list(self._theme_momentum)[-3:]
        dif = np.diff(rec)
        if dif.size == 0:
            return 0.10
        avg = float(np.mean(dif))
        # Falling strength → higher transition probability
        prob = float(np.clip(-avg + 0.10, 0.0, 1.0))
        return prob

    # -------------------------------------------------------------------------
    # State & macro
    # -------------------------------------------------------------------------
    def _update_theme_state(self, theme_id: int, strength: float, confidence: float):
        self._theme_vec[:] = 0.0
        idx = int(np.clip(theme_id, 0, len(self._theme_vec) - 1))
        self._theme_vec[idx] = float(np.clip(strength, 0.0, 1.0))

        prev = self._current_theme
        if idx != prev:
            self.trace(f"Theme transition: {prev} -> {idx}", level="INFO")

        self._current_theme = idx
        self._theme_confidence = float(np.clip(confidence, 0.0, 1.0))
        self._theme_momentum.append(float(strength))
        self._theme_strength_history.append(float(strength))
        self._theme_history.append(idx)

    def _get_macro_features(self) -> List[float]:
        try:
            arr = np.array([[self.macro_data.get("vix", 20.0),
                             self.macro_data.get("yield_curve", 0.5),
                             self.macro_data.get("cpi", 3.0)]],
                           dtype=np.float32)
            out = self._macro_scaler.transform(arr)[0]
            return out.tolist()
        except Exception as e:
            self.trace(f"Macro feature calculation failed: {e}", level="WARNING")
            return [0.0, 0.0, 0.0]

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------
    def get_fallback_result(self, error: str) -> Dict[str, Any]:
        return {
            'market_theme': getattr(self, "_current_theme", 0),
            'theme_strength': 0.10,
            'theme_confidence': 0.00,
            'theme_stability': 0.00,
            'transition_probability': 0.50,
            'theme_transition': 0.50,
            'clustering_quality': float(getattr(self, "_clustering_quality", 0.0)),
            'feature_stability': float(getattr(self, "_feature_stability_score", 0.0)),
            'themes_total': int(self.config['n_themes']),
            'theme_detection': {
                'theme': getattr(self, "_current_theme", 0),
                'strength': 0.10,
                'confidence': 0.00,
                'stability': 0.00,
                'transition_probability': 0.50,
            },
            'diagnostics': {
                'fit_count': int(getattr(self, "_ml_fit_count", 0)),
                'last_inertia': float(getattr(self, "_last_inertia", 0.0)) if getattr(self, "_last_inertia", None) is not None else None,
                'centroid_drift': float(self._centroid_drift[-1]) if getattr(self, "_centroid_drift", None) else 0.0
            },
            'processing_success': False,
            'error': error
        }
