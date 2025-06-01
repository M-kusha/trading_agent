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
import copy
import random

# --------------------------------------------------------------------------- #
# MarketThemeDetector (now evolutionary)
# --------------------------------------------------------------------------- #
class MarketThemeDetector(Module):
    """
    Online MiniBatch‑KMeans over multi‑time‑frame price features + macro data.
    Produces an n‑dim one‑hot‑ish theme‑strength vector.
    Now supports evolutionary (genome) adaptation.
    """

    def __init__(
        self,
        instruments: List[str],
        n_themes: int = 4,
        window: int = 100,
        debug: bool = False,
        genome: Dict[str, Any] = None,
    ):
        # Evolve these hyperparams!
        if genome:
            n_themes = genome.get("n_themes", n_themes)
            window = genome.get("window", window)

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

        # Evolutionary genome
        self.genome = {
            "n_themes": self.n_themes,
            "window": self.window
        }

    def reset(self):
        self._fit_buffer.clear()
        self._theme_vec.fill(0.0)

    def update_macro(self, indicator: str, value: float):
        if indicator in self.macro_data:
            self.macro_data[indicator] = value

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

    def fit_if_needed(self, data: Dict, t: int):
        x = self._mts_features(data, t)
        self._fit_buffer.append(x)
        if len(self._fit_buffer) < max(64, self.n_themes * 10):
            return
        X = self.scaler.fit_transform(np.vstack(self._fit_buffer))
        self.km.partial_fit(X)

    def detect(self, data: Dict, t: int) -> Tuple[int, float]:
        if not hasattr(self.scaler, "mean_"):
            return 0, 0.0
        x_raw = self._mts_features(data, t).reshape(1, -1)
        x = self.scaler.transform(x_raw)
        lab = int(self.km.predict(x)[0])
        dist = self.km.transform(x)[0].min()
        strength = float(1.0 / (1.0 + dist))
        self._theme_vec.fill(0.0)
        self._theme_vec[lab] = strength
        return lab, strength

    def get_observation_components(self) -> np.ndarray:
        return self._theme_vec.copy()

    def step(self, **kwargs):
        pass

    # --- Evolutionary methods ---
    def get_genome(self):
        return self.genome.copy()
    def set_genome(self, genome):
        # Only allow hyperparam changes
        self.n_themes = int(genome.get("n_themes", self.n_themes))
        self.window = int(genome.get("window", self.window))
        self.genome = {"n_themes": self.n_themes, "window": self.window}
        # Re-init the MiniBatchKMeans to use new n_themes
        self.km = MiniBatchKMeans(
            n_clusters=self.n_themes,
            batch_size=max(64, self.n_themes * 16),
            random_state=0,
        )
        self._theme_vec = np.zeros(self.n_themes, np.float32)
    def mutate(self, mutation_rate=0.2):
        g = self.genome.copy()
        if random.random() < mutation_rate:
            g["n_themes"] = int(np.clip(self.n_themes + np.random.randint(-1, 2), 2, 8))
        if random.random() < mutation_rate:
            g["window"] = int(np.clip(self.window + np.random.randint(-20, 20), 20, 200))
        self.set_genome(g)
    def crossover(self, other):
        g1, g2 = self.genome, other.genome
        new_g = {k: random.choice([g1[k], g2[k]]) for k in g1}
        return MarketThemeDetector(self.instruments, genome=new_g, debug=self.debug)

    # --- State save/load below is unchanged, see your previous version ---

    def get_state(self):
        scaler_state = {
            "mean_": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else None,
            "scale_": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
        }
        km_state = {
            "cluster_centers_": self.km.cluster_centers_.tolist() if hasattr(self.km, "cluster_centers_") else None
        }
        macro_state = dict(self.macro_data)
        fit_buffer = list(self._fit_buffer)
        theme_vec = self._theme_vec.tolist()
        return {
            "scaler": scaler_state,
            "km": km_state,
            "macro_data": macro_state,
            "fit_buffer": fit_buffer,
            "theme_vec": theme_vec,
            "genome": self.genome.copy()
        }
    def set_state(self, state):
        scaler = state.get("scaler", {})
        if scaler.get("mean_") is not None and scaler.get("scale_") is not None:
            self.scaler.mean_ = np.array(scaler["mean_"], dtype=np.float64)
            self.scaler.scale_ = np.array(scaler["scale_"], dtype=np.float64)
        km = state.get("km", {})
        if km.get("cluster_centers_") is not None:
            self.km.cluster_centers_ = np.array(km["cluster_centers_"], dtype=np.float64)
        self.macro_data = dict(state.get("macro_data", self.macro_data))
        self._fit_buffer = deque(state.get("fit_buffer", []), maxlen=2000)
        self._theme_vec = np.array(state.get("theme_vec", [0.0]*self.n_themes), dtype=np.float32)
        self.set_genome(state.get("genome", self.genome))



# --------------------------------------------------------------------------- #
# FractalRegimeConfirmation
# --------------------------------------------------------------------------- #
class FractalRegimeConfirmation(Module):
    """
    Confirms market regime (trending, volatile, noise) using fractal metrics
    plus macro theme confidence. Now supports evolutionary adaptation!
    """
    def __init__(self, window: int = 100, debug: bool = False, genome: Dict[str, Any] = None):
        if genome:
            window = genome.get("window", window)
            self.coeff_h = genome.get("coeff_h", 0.4)
            self.coeff_vr = genome.get("coeff_vr", 0.3)
            self.coeff_we = genome.get("coeff_we", 0.3)
        else:
            self.coeff_h = 0.4
            self.coeff_vr = 0.3
            self.coeff_we = 0.3

        self.window = window
        self.debug = debug
        self._buf = deque(maxlen=int(window * 0.75))
        self.regime_strength: float = 0.0
        self.label: str = "noise"
        self._forced_label: str | None = None  # Used for test monkeypatching
        self._forced_strength: float | None = None

        # Genome for evolution
        self.genome = {
            "window": self.window,
            "coeff_h": self.coeff_h,
            "coeff_vr": self.coeff_vr,
            "coeff_we": self.coeff_we,
        }

    def reset(self):
        self._buf.clear()
        self.regime_strength, self.label = 0.0, "noise"
        self._forced_label = None
        self._forced_strength = None

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

    def step(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        current_step: int,
        theme_detector: Any = None,
    ) -> Tuple[str, float]:
        if self._forced_label is not None and self._forced_strength is not None:
            if self.debug:
                print(f"[FRC] Forced regime: {self._forced_label} ({self._forced_strength:.3f})")
            self.label = self._forced_label
            self.regime_strength = self._forced_strength
            return self.label, self.regime_strength

        inst = next(iter(data_dict))
        ts = (
            data_dict[inst]["D1"]["close"]
            .values[max(0, current_step - self.window) : current_step]
            .astype(np.float32)
        )
        H = self._hurst(ts)
        VR = self._var_ratio(ts)
        WE = self._wavelet_energy(ts)
        theme_conf = 1.0

        if theme_detector is not None:
            theme_detector.fit_if_needed(data_dict, current_step)
            _, theme_conf = theme_detector.detect(data_dict, current_step)

        score = self.coeff_h * H + self.coeff_vr * VR + self.coeff_we * WE
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

    def force_regime(self, label: str, strength: float):
        self._forced_label = label
        self._forced_strength = float(strength)

    def clear_forced_regime(self):
        self._forced_label = None
        self._forced_strength = None

    def get_observation_components(self) -> np.ndarray:
        label_id = {"trending": 1, "volatile": -1, "noise": 0}.get(self.label, 0)
        return np.array([float(label_id), float(self.regime_strength)], dtype=np.float32)

    # --- Evolutionary methods ---
    def get_genome(self):
        return self.genome.copy()
    def set_genome(self, genome):
        self.window = int(genome.get("window", self.window))
        self.coeff_h = float(genome.get("coeff_h", self.coeff_h))
        self.coeff_vr = float(genome.get("coeff_vr", self.coeff_vr))
        self.coeff_we = float(genome.get("coeff_we", self.coeff_we))
        self.genome = {
            "window": self.window,
            "coeff_h": self.coeff_h,
            "coeff_vr": self.coeff_vr,
            "coeff_we": self.coeff_we,
        }
        self._buf = deque(maxlen=int(self.window * 0.75))
    def mutate(self, mutation_rate=0.2):
        g = self.genome.copy()
        if random.random() < mutation_rate:
            g["window"] = int(np.clip(self.window + np.random.randint(-20, 20), 20, 200))
        if random.random() < mutation_rate:
            g["coeff_h"] = float(np.clip(self.coeff_h + np.random.uniform(-0.1, 0.1), 0.1, 1.0))
        if random.random() < mutation_rate:
            g["coeff_vr"] = float(np.clip(self.coeff_vr + np.random.uniform(-0.1, 0.1), 0.1, 1.0))
        if random.random() < mutation_rate:
            g["coeff_we"] = float(np.clip(self.coeff_we + np.random.uniform(-0.1, 0.1), 0.1, 1.0))
        self.set_genome(g)
    def crossover(self, other):
        g1, g2 = self.genome, other.genome
        new_g = {k: random.choice([g1[k], g2[k]]) for k in g1}
        return FractalRegimeConfirmation(genome=new_g, debug=self.debug)

    def get_state(self) -> Dict[str, Any]:
        return {
            "regime_strength": float(self.regime_strength),
            "label": self.label,
            "_buf": list(self._buf),
            "_forced_label": self._forced_label,
            "_forced_strength": self._forced_strength,
            "genome": self.genome.copy(),
        }

    def set_state(self, state: Dict[str, Any]):
        self.regime_strength = float(state.get("regime_strength", 0.0))
        self.label = state.get("label", "noise")
        self._buf = deque(state.get("_buf", []), maxlen=int(self.window * 0.75))
        self._forced_label = state.get("_forced_label", None)
        self._forced_strength = state.get("_forced_strength", None)
        self.set_genome(state.get("genome", self.genome))


# --------------------------------------------------------------------------- #
# TimeAwareRiskScaling
# --------------------------------------------------------------------------- #
class TimeAwareRiskScaling(Module):
    """
    Adjusts portfolio-level risk by hourly volatility and session seasonality.
    Now supports evolutionary adaptation!
    """
    def __init__(self, debug: bool = False, genome: Dict[str, Any] = None):
        # Genome-based parameters
        if genome:
            self.asian_end = int(genome.get("asian_end", 8))
            self.euro_end = int(genome.get("euro_end", 16))
            self.decay = float(genome.get("decay", 0.9))
            self.base_factor = float(genome.get("base_factor", 1.0))
        else:
            self.asian_end = 8
            self.euro_end = 16
            self.decay = 0.9
            self.base_factor = 1.0

        self.vol_profile = np.zeros(24, np.float32)
        self.seasonality_factor: float = 1.0
        self.debug = debug

        self.genome = {
            "asian_end": self.asian_end,
            "euro_end": self.euro_end,
            "decay": self.decay,
            "base_factor": self.base_factor,
        }

    def reset(self):
        self.vol_profile.fill(0.0)
        self.seasonality_factor = 1.0

    def _session(self, hour: int) -> str:
        if 0 <= hour < self.asian_end:
            return "asian"
        if self.asian_end <= hour < self.euro_end:
            return "european"
        return "us"

    def step(self, **kwargs):
        if "data_dict" not in kwargs or "current_step" not in kwargs:
            return  # Skip if inputs missing

        data_dict = kwargs["data_dict"]
        current_step = kwargs["current_step"]

        ts = pd.Timestamp(data_dict.get("timestamp", pd.Timestamp.now()))
        hour = ts.hour % 24

        rets = np.asarray(data_dict.get("returns", []), np.float32)[-100:]
        if rets.size == 0:
            rets = np.zeros(100, np.float32)
        vol = float(np.nanstd(rets))
        # decay old profile to evolve
        self.vol_profile = self.vol_profile * self.decay
        self.vol_profile[hour] = vol

        max_vol = self.vol_profile.max() + 1e-8
        base_factor = self.base_factor - (vol / max_vol)

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

    # --- Evolutionary methods ---
    def get_genome(self):
        return self.genome.copy()
    def set_genome(self, genome):
        self.asian_end = int(genome.get("asian_end", self.asian_end))
        self.euro_end = int(genome.get("euro_end", self.euro_end))
        self.decay = float(genome.get("decay", self.decay))
        self.base_factor = float(genome.get("base_factor", self.base_factor))
        self.genome = {
            "asian_end": self.asian_end,
            "euro_end": self.euro_end,
            "decay": self.decay,
            "base_factor": self.base_factor,
        }
    def mutate(self, mutation_rate=0.2):
        g = self.genome.copy()
        if np.random.rand() < mutation_rate:
            g["asian_end"] = int(np.clip(self.asian_end + np.random.randint(-1, 2), 4, 12))
        if np.random.rand() < mutation_rate:
            g["euro_end"] = int(np.clip(self.euro_end + np.random.randint(-1, 2), 12, 20))
        if np.random.rand() < mutation_rate:
            g["decay"] = float(np.clip(self.decay + np.random.uniform(-0.05, 0.05), 0.8, 1.0))
        if np.random.rand() < mutation_rate:
            g["base_factor"] = float(np.clip(self.base_factor + np.random.uniform(-0.2, 0.2), 0.5, 1.5))
        self.set_genome(g)
    def crossover(self, other):
        g1, g2 = self.genome, other.genome
        new_g = {k: np.random.choice([g1[k], g2[k]]) for k in g1}
        return TimeAwareRiskScaling(genome=new_g, debug=self.debug)

    def get_state(self):
        return {
            "vol_profile": self.vol_profile.tolist(),
            "seasonality_factor": float(self.seasonality_factor),
            "genome": self.genome.copy()
        }
    def set_state(self, state):
        self.vol_profile = np.array(state.get("vol_profile", [0.0]*24), dtype=np.float32)
        self.seasonality_factor = float(state.get("seasonality_factor", 1.0))
        self.set_genome(state.get("genome", self.genome))


# --------------------------------------------------------------------------- #
class LiquidityHeatmapLayer(Module):
    """
    Maintains recent (spread, depth) tuples and LSTM‑forecasts short‑term liquidity.
    Now supports neuroevolution: evolve both architecture (genome) AND weights!
    """

    def __init__(self, action_dim: int, debug: bool = False, genome: dict = None, weights: list = None):
        super().__init__()
        # Evolvable architecture params (genome)
        if genome:
            self.lstm_units = int(genome.get("lstm_units", 32))
            self.seq_len = int(genome.get("seq_len", 10))
            self.dense_units = int(genome.get("dense_units", 2))
            self.train_epochs = int(genome.get("train_epochs", 10))
        else:
            self.lstm_units = 32
            self.seq_len = 10
            self.dense_units = 2
            self.train_epochs = 10

        self.action_dim = action_dim
        self.debug = debug
        self.bids: List[Tuple[float, float]] = []
        self.asks: List[Tuple[float, float]] = []
        self.history: deque[Tuple[float, float]] = deque(maxlen=200)

        self._trained = False
        self._model = self._build_lstm()
        if weights is not None:
            self._set_weights(weights)

        # Genome for reproduction
        self.genome = {
            "lstm_units": self.lstm_units,
            "seq_len": self.seq_len,
            "dense_units": self.dense_units,
            "train_epochs": self.train_epochs,
        }

    def _build_lstm(self):
        with tf.device("/CPU:0"):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.seq_len, 2)),
                tf.keras.layers.LSTM(self.lstm_units),
                tf.keras.layers.Dense(self.dense_units),
            ])
            model.compile(optimizer="adam", loss="mse")
            return model

    # --------- Weight helpers for neuroevolution ----------
    def _get_weights(self):
        # Returns a list of numpy arrays representing all model weights
        return self._model.get_weights()

    def _set_weights(self, weights):
        self._model.set_weights([np.copy(w) for w in weights])

    def _weights_like(self, other):
        # Make sure weights shapes match (important for crossover)
        ws1 = self._get_weights()
        ws2 = other._get_weights()
        return all(w1.shape == w2.shape for w1, w2 in zip(ws1, ws2))

    def clone_with_weights(self):
        # Return a new instance with the same weights & genome
        return LiquidityHeatmapLayer(
            self.action_dim,
            debug=self.debug,
            genome=self.genome.copy(),
            weights=self._get_weights(),
        )

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self.history.clear()
        self._trained = False
        self._model = self._build_lstm()
        self._set_weights(self._get_weights())  # Keep weights as before

    # ------------------------------------------------------

    def _make_seqs(self):
        X, y = [], []
        rows = list(self.history)
        for i in range(len(rows) - self.seq_len):
            X.append(rows[i : i + self.seq_len])
            y.append(rows[i + self.seq_len])
        return np.asarray(X, np.float32), np.asarray(y, np.float32)

    def _train_if_ready(self):
        if self._trained or len(self.history) < 2 * self.seq_len:
            return
        X, y = self._make_seqs()
        if X.size:
            self._model.fit(X, y, epochs=self.train_epochs, verbose=0)
            self._trained = True
            if self.debug:
                print(f"[LHL] trained on {len(X)} sequences (units={self.lstm_units}, seq_len={self.seq_len})")

    def step(self, **kwargs):
        if "order_book" not in kwargs:
            return

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

    def predict_liquidity(self, steps: int = 4) -> Tuple[float, float]:
        if not self._trained or len(self.history) < self.seq_len:
            return 0.0, 0.0
        seq = np.array([list(self.history)[-self.seq_len:]], np.float32)
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
        score = self.current_score()
        conf = float(np.clip(score, 0.1, 1.0))
        if self.debug:
            print(f"[LiquidityHeatmapLayer] Liquidity score={score:.2f}, confidence={conf:.2f}")
        return conf

    # --- Evolutionary methods ---
    def get_genome(self):
        return self.genome.copy()

    def set_genome(self, genome):
        self.lstm_units = int(genome.get("lstm_units", self.lstm_units))
        self.seq_len = int(genome.get("seq_len", self.seq_len))
        self.dense_units = int(genome.get("dense_units", self.dense_units))
        self.train_epochs = int(genome.get("train_epochs", self.train_epochs))
        self.genome = {
            "lstm_units": self.lstm_units,
            "seq_len": self.seq_len,
            "dense_units": self.dense_units,
            "train_epochs": self.train_epochs,
        }
        self._model = self._build_lstm()

    def mutate(self, mutation_rate=0.2, weight_mutate_std=0.05):
        g = self.genome.copy()
        if np.random.rand() < mutation_rate:
            g["lstm_units"] = int(np.clip(self.lstm_units + np.random.randint(-8, 9), 8, 128))
        if np.random.rand() < mutation_rate:
            g["seq_len"] = int(np.clip(self.seq_len + np.random.randint(-2, 3), 5, 20))
        if np.random.rand() < mutation_rate:
            g["dense_units"] = int(np.clip(self.dense_units + np.random.randint(-1, 2), 1, 8))
        if np.random.rand() < mutation_rate:
            g["train_epochs"] = int(np.clip(self.train_epochs + np.random.randint(-2, 3), 2, 20))
        self.set_genome(g)

        # Mutate weights (add small Gaussian noise)
        weights = self._get_weights()
        mutated = []
        for w in weights:
            if np.issubdtype(w.dtype, np.floating):
                noise = np.random.randn(*w.shape) * weight_mutate_std
                mutated.append(w + noise)
            else:
                mutated.append(w)
        self._set_weights(mutated)

    def crossover(self, other, weight_mix_prob=0.5):
        # Mix architecture
        g1, g2 = self.genome, other.genome
        new_g = {k: np.random.choice([g1[k], g2[k]]) for k in g1}

        # Mix weights
        ws1, ws2 = self._get_weights(), other._get_weights()
        new_ws = []
        for w1, w2 in zip(ws1, ws2):
            if np.issubdtype(w1.dtype, np.floating) and w1.shape == w2.shape:
                mask = np.random.rand(*w1.shape) < weight_mix_prob
                mixed = np.where(mask, w1, w2)
                new_ws.append(mixed)
            else:
                new_ws.append(w1)

        return LiquidityHeatmapLayer(
            self.action_dim, debug=self.debug, genome=new_g, weights=new_ws
        )

    def get_state(self):
        return {
            "history": list(self.history),
            "trained": bool(self._trained),
            "genome": self.genome.copy(),
            "weights": [w.copy() for w in self._get_weights()],
        }

    def set_state(self, state):
        self.history = deque(state.get("history", []), maxlen=200)
        self._trained = bool(state.get("trained", False))
        self.set_genome(state.get("genome", self.genome))
        if "weights" in state:
            self._set_weights(state["weights"])





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
    
    def get_state(self):
        return {
            "matrix": self.matrix.tolist(),
            "vol_history": list(self.vol_history),
            "volatility_regimes": self.volatility_regimes.tolist(),
            "last_volatility": float(self.last_volatility),
            "last_liquidity": float(self.last_liquidity),
        }
    def set_state(self, state):
        self.matrix = np.array(state.get("matrix", np.zeros((self.n, self.n))), dtype=np.float32)
        self.vol_history = deque(state.get("vol_history", []), maxlen=500)
        self.volatility_regimes = np.array(state.get("volatility_regimes", [0.1, 0.3, 0.5]), dtype=np.float32)
        self.last_volatility = float(state.get("last_volatility", 0.0))
        self.last_liquidity = float(state.get("last_liquidity", 1.0))

