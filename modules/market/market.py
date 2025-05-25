#modules/market.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress, zscore
from collections import deque
from typing import Any, List, Dict, Tuple
import pywt
from sklearn.cluster import MiniBatchKMeans       # online‑friendly
import tensorflow as tf
from ..core.core import Module

# --------------------------------------------------------------------------- #
# MarketThemeDetector
# --------------------------------------------------------------------------- #
class MarketThemeDetector(Module):
    """
    Online MiniBatch‑KMeans over multi‑time‑frame price features + macro data.
    Produces an n‑dim one‑hot‑ish theme‑strength vector.
    """

    def __init__(
        self,
        instruments: List[str],
        n_themes: int = 4,
        window: int = 100,
        debug: bool = False,
    ):
        self.instruments   = instruments
        self.n_themes      = n_themes
        self.window        = window
        self.debug         = debug

        self.scaler        = StandardScaler()
        self.km            = MiniBatchKMeans(
            n_clusters=n_themes,
            batch_size=max(64, n_themes * 16),
            random_state=0,
        )
        self._fit_buffer   = deque(maxlen=2000)
        self._theme_vec    = np.zeros(n_themes, np.float32)

        # dynamic scaler for macro data
        self._macro_scaler = StandardScaler()
        self._macro_scaler.fit([[20.0, 0.5, 3.0]])
        self.macro_data = {"vix": 20.0, "yield_curve": 0.5, "cpi": 3.0}

    # ------------------------------------------------------------------ #
    def reset(self):
        self._fit_buffer.clear()
        self._theme_vec.fill(0.0)

    # ------------------------------------------------------------------ #
    def update_macro(self, indicator: str, value: float):
        if indicator in self.macro_data:
            self.macro_data[indicator] = value

    # ------------------------------------------------------------------ #
    @staticmethod
    def _hurst(series: np.ndarray) -> float:
        series = series[:500]
        if series.size < 10 or np.all(series == series[0]):
            return 0.5

        lags = np.arange(2, min(100, series.size // 2))
        if lags.size == 0:
            return 0.5

        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_lags = np.log(lags)
                log_tau = np.log(tau)
                slope, *_ = linregress(log_lags, log_tau)
            return float(slope * 2.0) if np.isfinite(slope) else 0.5
        except Exception:
            return 0.5


    @staticmethod
    def _wavelet_energy(series: np.ndarray, wavelet: str = "db4") -> float:
        series = series[:256]
        if series.size < 16:
            return 0.0
        try:
            level = min(1, pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len))
            coeffs = pywt.wavedec(series, wavelet, level=level)
            return float(np.sum(coeffs[-1] ** 2) / (np.sum(series ** 2) + 1e-8))
        except Exception:
            return 0.0



    # ------------------------------------------------------------------ #
    def _mts_features(
        self, data: Dict[str, Dict[str, pd.DataFrame]], t: int
    ) -> np.ndarray:
        feats: List[float] = []
        for tf in ["H1", "H4", "D1"]:
            for inst in self.instruments:
                df = data[inst][tf]
                sl = df.iloc[max(0, t - self.window) : t]["close"]
                ret = sl.pct_change().dropna().values.astype(np.float32)

                feats.extend(
                    [
                        ret.mean(),
                        ret.std(),
                        pd.Series(ret).skew(),
                        pd.Series(ret).kurtosis(),
                        (df["high"] - df["low"])
                        .iloc[max(0, t - self.window) : t]
                        .mean(),
                        self._hurst(ret),
                        self._wavelet_energy(ret),
                    ]
                )

        macro = self._macro_scaler.transform(
            [
                [
                    self.macro_data["vix"],
                    self.macro_data["yield_curve"],
                    self.macro_data["cpi"],
                ]
            ]
        )[0]
        feats.extend(macro.tolist())
        return np.asarray(feats, np.float32)

    # ------------------------------------------------------------------ #
    def fit_if_needed(self, data: Dict, t: int):
        x = self._mts_features(data, t)
        self._fit_buffer.append(x)

        # need enough samples before first partial_fit
        if len(self._fit_buffer) < max(64, self.n_themes * 10):
            return

        X = self.scaler.fit_transform(np.vstack(self._fit_buffer))
        self.km.partial_fit(X)

    # ------------------------------------------------------------------ #
    def detect(self, data: Dict, t: int) -> Tuple[int, float]:
        if not hasattr(self.scaler, "mean_"):       # not fitted yet
            return 0, 0.0

        x_raw = self._mts_features(data, t).reshape(1, -1)
        x = self.scaler.transform(x_raw)

        lab = int(self.km.predict(x)[0])
        dist = self.km.transform(x)[0].min()
        strength = float(1.0 / (1.0 + dist))

        # build one‑hot‑strength vector
        self._theme_vec.fill(0.0)
        self._theme_vec[lab] = strength
        return lab, strength

    # ------------------------------------------------------------------ #
    def get_observation_components(self) -> np.ndarray:
        return self._theme_vec.copy()

    def step(self, **kwargs):
        """
        This method is not used in this module.
        It is here to satisfy the Module interface.
        """
        pass

    def get_state(self):
        return {
            "themes": self.detected_themes,  # Assuming 'detected_themes' exists
        }

    def set_state(self, state):
        self.detected_themes = state.get("themes", [])


# --------------------------------------------------------------------------- #
# FractalRegimeConfirmation
# --------------------------------------------------------------------------- #
class FractalRegimeConfirmation(Module):
    """
    Confirms broad regime (trending / volatile / noise) from fractal metrics
    plus theme confidence.
    """

    def __init__(self, window: int = 100, debug: bool = False):
        self.window = window
        self.debug = debug
        self._buf = deque(maxlen=int(window * 0.75))
        self.regime_strength: float = 0.0
        self.label: str = "noise"

    def reset(self):
        self._buf.clear()
        self.regime_strength, self.label = 0.0, "noise"

    # ------------- internal metrics ---------------- #
    @staticmethod
    def _hurst(series: np.ndarray) -> float:
        series = series[:500]
        if series.size < 10 or np.all(series == series[0]):
            return 0.5

        lags = np.arange(2, min(100, series.size // 2))
        if lags.size == 0:
            return 0.5

        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_lags = np.log(lags)
                log_tau = np.log(tau)
                slope, *_ = linregress(log_lags, log_tau)
            return float(slope * 2.0) if np.isfinite(slope) else 0.5
        except Exception:
            return 0.5


    @staticmethod
    def _var_ratio(ts: np.ndarray) -> float:
        if ts.size < 2:
            return 1.0
        ts = ts[-300:]
        try:
            return float(np.var(ts[1:] - ts[:-1]) / (np.var(ts) + 1e-8))
        except Exception:
            return 1.0

    @staticmethod
    def _wavelet_energy(series: np.ndarray, wavelet: str = "db4") -> float:
        series = series[:256]
        if series.size < 16:
            return 0.0
        try:
            level = min(1, pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len))
            coeffs = pywt.wavedec(series, wavelet, level=level)
            return float(np.sum(coeffs[-1] ** 2) / (np.sum(series ** 2) + 1e-8))
        except Exception:
            return 0.0


    # ------------- main step ----------------------- #
    def step(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        current_step: int,
        theme_detector: MarketThemeDetector,
    ) -> Tuple[str, float]:
        inst = next(iter(data_dict))
        ts = (
            data_dict[inst]["D1"]["close"]
            .values[max(0, current_step - self.window) : current_step]
            .astype(np.float32)
        )

        H = self._hurst(ts)
        VR = self._var_ratio(ts)
        WE = self._wavelet_energy(ts)


        # keep theme detector warmed‑up
        theme_detector.fit_if_needed(data_dict, current_step)
        _, theme_conf = theme_detector.detect(data_dict, current_step)

        score = 0.4 * H + 0.3 * VR + 0.3 * WE
        self._buf.append(score)
        self.regime_strength = float(np.mean(self._buf) * theme_conf)

        if self.regime_strength > 0.6:
            self.label = "trending"
        elif self.regime_strength < 0.3:
            self.label = "volatile"
        else:
            self.label = "noise"

        if self.debug:
            print(
                f"[FRC] label={self.label:<9}  strength={self.regime_strength:.3f}"
            )
        return self.label, self.regime_strength

    # ---------------------------------------------- #
    def get_observation_components(self) -> np.ndarray:
        return np.array([self.regime_strength], np.float32)

# --------------------------------------------------------------------------- #
# TimeAwareRiskScaling
# --------------------------------------------------------------------------- #
class TimeAwareRiskScaling(Module):
    """
    Adjusts portfolio‑level risk by hourly volatility and session seasonality.
    """

    def __init__(self, debug: bool = False):
        self.vol_profile = np.zeros(24, np.float32)
        self.seasonality_factor: float = 1.0
        self.debug = debug

    def reset(self):
        self.vol_profile.fill(0.0)
        self.seasonality_factor = 1.0

    @staticmethod
    def _session(hour: int) -> str:
        if 0 <= hour < 8:
            return "asian"
        if 8 <= hour < 16:
            return "european"
        return "us"

    def step(self, **kwargs):
        if "data_dict" not in kwargs or "current_step" not in kwargs:
            return  # Skip execution if inputs are missing (e.g. during dummy init)

        data_dict = kwargs["data_dict"]
        current_step = kwargs["current_step"]

        ts = pd.Timestamp(data_dict["timestamp"])
        hour = ts.hour % 24

        rets = np.asarray(data_dict.get("returns", []), np.float32)[-100:]
        if rets.size == 0:
            rets = np.zeros(100, np.float32)  # safe-default
        vol = float(np.nanstd(rets))
        self.vol_profile[hour] = vol

        max_vol = self.vol_profile.max() + 1e-8
        base_factor = 1.0 - (vol / max_vol)

        session = self._session(hour)
        sess_map = {
            "asian":    1.0 + 0.3 * base_factor,
            "european": base_factor,
            "us":       1.0 - 0.4 * (1.0 - base_factor),
        }
        self.seasonality_factor = float(sess_map[session])

        if self.debug:
            print(
                f"[TARS] hr={hour:02d} sess={session:<8} "
                f"vol={vol:.5f} factor={self.seasonality_factor:.3f}"
            )


    def get_observation_components(self) -> np.ndarray:
        return np.array([self.seasonality_factor], np.float32)

# --------------------------------------------------------------------------- #
class LiquidityHeatmapLayer(Module):
    """
    Maintains recent (spread, depth) tuples and LSTM‑forecasts short‑term liquidity.
    """

    def __init__(self, action_dim: int, debug: bool = False):
        super().__init__()
        self.action_dim = action_dim
        self.debug = debug
        self.bids: List[Tuple[float, float]] = []
        self.asks: List[Tuple[float, float]] = []
        self.history: deque[Tuple[float, float]] = deque(maxlen=200)

        self._trained = False
        self._model = self._build_lstm()

    # ------------------------------------------------------------------ #
    def _build_lstm(self):
        with tf.device("/CPU:0"):  # <- Force CPU to avoid GPU/XLA bug
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(10, 2)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(2),
            ])
            model.compile(optimizer="adam", loss="mse")
            return model

    # ------------------------------------------------------------------ #
    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self.history.clear()
        self._trained = False

    # ------------------------------------------------------------------ #
    def _make_seqs(self):
        X, y = [], []
        rows = list(self.history)
        for i in range(len(rows) - 10):
            X.append(rows[i : i + 10])
            y.append(rows[i + 10])
        return np.asarray(X, np.float32), np.asarray(y, np.float32)

    def _train_if_ready(self):
        if self._trained or len(self.history) < 100:
            return
        X, y = self._make_seqs()
        if X.size:
            self._model.fit(X, y, epochs=10, verbose=0)
            self._trained = True
            if self.debug:
                print(f"[LHL] trained on {len(X)} sequences")

    # ------------------------------------------------------------------ #
    def step(self, **kwargs):
        if "order_book" not in kwargs:
            return  # skip if not available (e.g. during dummy input or setup)

        order_book = kwargs["order_book"]
        raw_bids = order_book.get("bids", [])
        raw_asks = order_book.get("asks", [])

        bid_sizes = np.array([q for _, q in raw_bids], np.float32)
        ask_sizes = np.array([q for _, q in raw_asks], np.float32)

        bid_mask = (
            np.abs(zscore(bid_sizes)) < 2.5 if bid_sizes.size else np.array([], bool)
        )
        ask_mask = (
            np.abs(zscore(ask_sizes)) < 2.5 if ask_sizes.size else np.array([], bool)
        )

        bids_f = [raw_bids[i] for i in range(len(raw_bids)) if bid_mask.size and bid_mask[i]]
        asks_f = [raw_asks[i] for i in range(len(raw_asks)) if ask_mask.size and ask_mask[i]]

        self.bids = sorted(bids_f, key=lambda x: -x[1])[:5]
        self.asks = sorted(asks_f, key=lambda x: x[1])[:5]

        bid_depth = sum(q for _, q in self.bids)
        ask_depth = sum(q for _, q in self.asks)
        spread = self.asks[0][0] - self.bids[0][0] if (self.bids and self.asks) else 0.0

        self.history.append((spread, bid_depth + ask_depth))
        self._train_if_ready()

        if self.debug:
            print(f"[LHL] spread={spread:.6f} depth={bid_depth+ask_depth:.1f}")


    # ------------------------------------------------------------------ #
    def predict_liquidity(self, steps: int = 4) -> Tuple[float, float]:
        if not self._trained or len(self.history) < 10:
            return 0.0, 0.0
        seq = np.array([list(self.history)[-10:]], np.float32)
        pred = self._model.predict(seq, verbose=0)[0]
        return float(pred[0]), float(pred[1])

    def current_score(self) -> float:
        if not self.history:
            return 1.0
        spread, depth = self.history[-1]
        return float(np.log1p(depth) * np.exp(-spread * 100.0))

    def get_observation_components(self) -> np.ndarray:
        return np.array([self.current_score()], np.float32)
    
    def propose_action(self, obs: Any) -> np.ndarray:
        liq = float(np.clip(self.current_score(), 0.1, 1.0))
        return np.full(self.action_dim, liq, dtype=np.float32)

    def confidence(self, obs: Any) -> float:
        """
        Higher confidence in high liquidity environments.
        """
        score = self.current_score()
        conf = float(np.clip(score, 0.1, 1.0))  # Avoid zero
        if self.debug:
            print(f"[LiquidityHeatmapLayer] Liquidity score={score:.2f}, confidence={conf:.2f}")
        return conf



# --------------------------------------------------------------------------- #
# RegimePerformanceMatrix
# --------------------------------------------------------------------------- #
class RegimePerformanceMatrix(Module):
    """
    Tracks PnL across realised vs predicted volatility regimes.
    """

    def __init__(self, n_regimes: int = 3, decay: float = 0.95, debug: bool = False):
        self.n = n_regimes
        self.decay = decay
        self.debug = debug

        self.vol_history = deque(maxlen=500)
        self.volatility_regimes = np.array([0.1, 0.3, 0.5], np.float32)
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.n, self.n), np.float32)
        self.last_volatility = 0.0
        self.last_liquidity = 1.0
        self.vol_history.clear()

    # ------------------------------------------------------------------ #
    def step(self, **kwargs):
        if not all(k in kwargs for k in ("pnl", "volatility", "predicted_regime")):
            return  # skip if any required input is missing

        pnl = kwargs["pnl"]
        volatility = kwargs["volatility"]
        predicted_regime = kwargs["predicted_regime"]

        self.vol_history.append(volatility)

        if len(self.vol_history) >= 20:
            self.volatility_regimes = np.quantile(
                np.asarray(self.vol_history), [0.25, 0.5, 0.75]
            ).astype(np.float32)

        true_reg = min(
            int(np.digitize(volatility, self.volatility_regimes)), self.n - 1
        )

        self.matrix *= self.decay
        self.matrix[true_reg, predicted_regime] += pnl

        self.last_volatility = volatility
        if self.debug:
            print(
                f"[RPM] true={true_reg} pred={predicted_regime} "
                f"pnl={pnl:+.2f} thr={self.volatility_regimes.round(4)}"
            )


    # ------------------------------------------------------------------ #
    def stress_test(
        self,
        scenario: str,
        volatility: float | None = None,
        liquidity_score: float | None = None,
    ) -> Dict[str, float]:
        crisis = {
            "flash-crash": {"vol_mult": 3.0, "liq_mult": 0.2},
            "rate-spike": {"vol_mult": 2.5, "liq_mult": 0.5},
            "default-wave": {"vol_mult": 2.0, "liq_mult": 0.3},
        }.get(scenario, {"vol_mult": 1.0, "liq_mult": 1.0})

        vol = volatility if volatility is not None else self.last_volatility
        liq = liquidity_score if liquidity_score is not None else self.last_liquidity
        return {"volatility": vol * crisis["vol_mult"], "liquidity": liq * crisis["liq_mult"]}

    def get_observation_components(self) -> np.ndarray:
        flat = self.matrix.flatten()
        acc = [
            float(self.matrix[i, i] / (self.matrix[i].sum() + 1e-8))
            for i in range(self.n)
        ]
        return np.concatenate([flat, np.asarray(acc, np.float32)])
