import logging
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress, zscore
from collections import deque
from typing import Any, List, Dict, Tuple, Optional
import pywt
from sklearn.cluster import MiniBatchKMeans
import tensorflow as tf
from ..core.core import Module
import copy
import random

# ────────────────────────────────────────────────────────────────────────────
def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────

class MarketThemeDetector:
    def __init__(
        self,
        instruments: List[str],
        n_themes: int = 4,
        window: int = 100,
        debug: bool = True,
        genome: Dict[str, Any] | None = None,
    ):
        # ── genome overrides ───────────────────────────────────────────
        if genome:
            n_themes = genome.get("n_themes", n_themes)
            window   = genome.get("window",   window)

        self.instruments = instruments
        self.n_themes    = n_themes
        self.window      = window
        self.debug       = debug

        # ── ML helpers ────────────────────────────────────────────────
        self.scaler = StandardScaler()
        self.km     = MiniBatchKMeans(
            n_clusters=n_themes,
            batch_size=max(64, n_themes * 16),
            random_state=0,
        )
        self._fit_buffer = deque(maxlen=2000)
        self._theme_vec  = np.zeros(n_themes, np.float32)
        
        # NEW: Track theme characteristics for better signal generation
        self._theme_profiles = {}  # Will store mean features per theme
        self._current_theme = 0
        self._theme_momentum = deque(maxlen=10)  # Track theme changes

        # ── macro placeholder (scaled) ────────────────────────────────
        self._macro_scaler = StandardScaler()
        self._macro_scaler.fit([[20.0, 0.5, 3.0]])
        self.macro_data = {"vix": 20.0, "yield_curve": 0.5, "cpi": 3.0}

        # ── evolutionary DNA ──────────────────────────────────────────
        self.genome = {"n_themes": n_themes, "window": window}

        log_file = "logs/regime/market_theme_regime.log"
        
        # 2) Ensure the full directory exists
        _ensure_dir(os.path.dirname(log_file))

        # 3) Configure logger
        self.logger = logging.getLogger("MarketThemeDetector")
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG if debug else logging.INFO)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.propagate = False

        self.logger.info("[MarketThemeDetector] Initialized")

    # ───────────────────────── helpers ────────────────────────────────
    @staticmethod
    def _hurst(series: np.ndarray) -> float:
        series = series[:500]
        if series.size < 10 or np.all(series == series[0]):
            return 0.5
        lags  = np.arange(2, min(100, series.size // 2))
        if lags.size == 0:
            return 0.5
        tau   = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        with np.errstate(divide='ignore', invalid='ignore'):
            slope, *_ = linregress(np.log(lags), np.log(tau))
        return float(slope * 2.0) if np.isfinite(slope) else 0.5

    @staticmethod
    def _wavelet_energy(series: np.ndarray, wavelet: str = "db4") -> float:
        series = series[:256]
        if series.size < 16:
            return 0.0
        try:
            level = min(1, pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len))
            coeffs = pywt.wavedec(series, wavelet, level=level)
            return float(np.sum(coeffs[-1] ** 2) / (np.sum(series ** 2) + 1e-8))
        except:
            return 0.0

    # ───────────────────────── feature builder ────────────────────────
    def _mts_features(
        self, data: Dict[str, Dict[str, pd.DataFrame]], t: int
    ) -> np.ndarray:
        feats: List[float] = []
        for tf in ("H1", "H4", "D1"):
            for inst in self.instruments:
                df  = data[inst][tf]
                sl  = df.iloc[max(0, t - self.window): t]["close"]
                if len(sl) < 2:
                    feats.extend([0.0] * 7)
                    continue
                    
                ret = sl.pct_change().dropna().values.astype(np.float32)
                if len(ret) < 1:
                    feats.extend([0.0] * 7)
                    continue

                feats += [
                    float(ret.mean()), 
                    float(ret.std()),
                    float(pd.Series(ret).skew()) if len(ret) > 2 else 0.0, 
                    float(pd.Series(ret).kurtosis()) if len(ret) > 2 else 0.0,
                    float((df["high"] - df["low"]).iloc[max(0, t - self.window): t].mean()),
                    self._hurst(ret), 
                    self._wavelet_energy(ret),
                ]

        macro = self._macro_scaler.transform([[self.macro_data["vix"],
                                               self.macro_data["yield_curve"],
                                               self.macro_data["cpi"]]])[0]
        feats.extend(macro.tolist())
        return np.asarray(feats, np.float32)

    # ───────────────────────── public API ─────────────────────────────
    def reset(self):
        self._fit_buffer.clear()
        self._theme_vec.fill(0.0)
        self._theme_profiles.clear()
        self._current_theme = 0
        self._theme_momentum.clear()

    def update_macro(self, indicator: str, value: float):
        if indicator in self.macro_data:
            self.macro_data[indicator] = value

    def fit_if_needed(self, data: Dict, t: int):
        x = self._mts_features(data, t)
        self._fit_buffer.append(x)
        if len(self._fit_buffer) < max(64, self.n_themes * 10):
            return
        X = self.scaler.fit_transform(np.vstack(self._fit_buffer))
        self.km.partial_fit(X)
        
        # Update theme profiles
        if hasattr(self.km, 'cluster_centers_'):
            for i in range(self.n_themes):
                self._theme_profiles[i] = self.km.cluster_centers_[i]
        
        if self.debug and len(self._fit_buffer) % 500 == 0:
            self.logger.debug(f"[MTD] partial-fit on {len(self._fit_buffer)} points.")

    def detect(self, data: Dict, t: int) -> Tuple[int, float]:
        if not hasattr(self.scaler, "mean_"):
            return 0, 0.0
        x      = self.scaler.transform(self._mts_features(data, t).reshape(1, -1))
        lab    = int(self.km.predict(x)[0])
        dist   = self.km.transform(x)[0].min()
        strength = float(1.0 / (1.0 + dist))
        
        # Update theme tracking
        self._current_theme = lab
        self._theme_momentum.append(lab)
        
        self._theme_vec.fill(0.0)
        self._theme_vec[lab] = strength
        return lab, strength
    
    # ────────── IMPROVED: hooks required by StrategyArbiter ──────────
    def set_action_dim(self, dim: int):
        """Call once from the env so we know how many numbers to output."""
        self._action_dim = int(dim)

    def propose_action(self, obs: Any = None) -> np.ndarray:
        """
        Improved action generation based on theme characteristics
        """
        if not hasattr(self, "_action_dim"):
            self._action_dim = 2
            
        action = np.zeros(self._action_dim, np.float32)
        
        # Get current theme strength
        theme_strength = float(self._theme_vec.max())
        current_theme = int(np.argmax(self._theme_vec))
        
        # Analyze theme momentum (are we transitioning between themes?)
        if len(self._theme_momentum) >= 3:
            recent_themes = list(self._theme_momentum)[-3:]
            theme_stability = len(set(recent_themes)) == 1
        else:
            theme_stability = False
        
        # Generate signals based on theme characteristics
        if theme_strength > 0.3:  # Only trade when we have confidence
            # Map themes to trading strategies
            if current_theme == 0:  # E.g., "Risk-on" theme
                # Tend to be long risk assets
                for i in range(0, self._action_dim, 2):
                    action[i] = 0.3 * theme_strength
                    action[i+1] = 0.5  # Medium duration
                    
            elif current_theme == 1:  # E.g., "Risk-off" theme
                # Tend to be long safe havens (XAU), short risk (EUR)
                if self._action_dim >= 4:
                    action[0] = -0.3 * theme_strength  # Short EUR/USD
                    action[1] = 0.5
                    action[2] = 0.3 * theme_strength   # Long XAU/USD
                    action[3] = 0.7  # Longer duration for safe haven
                    
            elif current_theme == 2:  # E.g., "High volatility" theme
                # Smaller positions, shorter duration
                for i in range(0, self._action_dim, 2):
                    if theme_stability:
                        action[i] = np.random.choice([-0.1, 0.1]) * theme_strength
                        action[i+1] = 0.3  # Short duration
                        
            elif current_theme == 3:  # E.g., "Trending" theme
                # Follow momentum with higher confidence
                if hasattr(self, '_theme_profiles') and current_theme in self._theme_profiles:
                    profile = self._theme_profiles[current_theme]
                    # Use first component as direction indicator
                    direction = np.sign(profile[0]) if len(profile) > 0 else 1.0
                    for i in range(0, self._action_dim, 2):
                        action[i] = direction * 0.4 * theme_strength
                        action[i+1] = 0.7  # Longer duration for trends
        
        # Apply theme transition penalty
        if not theme_stability and len(self._theme_momentum) > 1:
            action *= 0.5  # Reduce size during transitions
            
        return action

    def confidence(self, obs: Any = None) -> float:
        """Return the detector's current certainty (0…1)."""
        base_conf = float(self._theme_vec.max())
        
        # Boost confidence if theme is stable
        if len(self._theme_momentum) >= 3:
            recent_themes = list(self._theme_momentum)[-3:]
            if len(set(recent_themes)) == 1:
                base_conf = min(base_conf * 1.2, 1.0)
                
        return base_conf

    # —— Gym-style hook ————————————————————————————————
    def get_observation_components(self) -> np.ndarray:
        return self._theme_vec.copy()

    def step(self, **kwargs):        # kept for interface compatibility
        pass

    # ───────────────────────── evolution utils ────────────────────────
    def get_genome(self):  
        return self.genome.copy()

    def set_genome(self, genome: Dict[str, Any]):
        self.n_themes = int(genome.get("n_themes", self.n_themes))
        self.window   = int(genome.get("window",   self.window))
        self.genome   = {"n_themes": self.n_themes, "window": self.window}
        self.km       = MiniBatchKMeans(n_clusters=self.n_themes,
                                        batch_size=max(64, self.n_themes * 16),
                                        random_state=0)
        self._theme_vec = np.zeros(self.n_themes, np.float32)

    def mutate(self, rate: float = 0.2):
        g = self.get_genome()
        if random.random() < rate:
            g["n_themes"] = int(np.clip(self.n_themes + random.choice([-1, 1]), 2, 8))
        if random.random() < rate:
            g["window"]   = int(np.clip(self.window   + random.randint(-20, 20), 20, 200))
        self.set_genome(g)

    def crossover(self, other: "MarketThemeDetector"):
        new_g = {k: random.choice([self.genome[k], other.genome[k]]) for k in self.genome}
        return MarketThemeDetector(self.instruments, genome=new_g, debug=self.debug)

    # ───────────────────────── persistence ────────────────────────────
    def get_state(self):
        return {
            "scaler": {
                "mean_":  self.scaler.mean_.tolist()  if hasattr(self.scaler, "mean_")  else None,
                "scale_": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
            },
            "km": {
                "cluster_centers_":
                    self.km.cluster_centers_.tolist() if hasattr(self.km, "cluster_centers_") else None
            },
            "macro_data":     dict(self.macro_data),
            "fit_buffer":     list(self._fit_buffer),
            "theme_vec":      self._theme_vec.tolist(),
            "genome":         self.genome.copy(),
            "theme_profiles": {k: v.tolist() for k, v in self._theme_profiles.items()},
            "current_theme":  self._current_theme,
            "theme_momentum": list(self._theme_momentum),
        }

    def set_state(self, state):
        sc = state.get("scaler", {})
        if sc.get("mean_") is not None:
            self.scaler.mean_  = np.asarray(sc["mean_"])
            self.scaler.scale_ = np.asarray(sc["scale_"])
        km = state.get("km", {})
        if km.get("cluster_centers_") is not None:
            self.km.cluster_centers_ = np.asarray(km["cluster_centers_"])
        self.macro_data  = dict(state.get("macro_data", self.macro_data))
        self._fit_buffer = deque(state.get("fit_buffer", []), maxlen=2000)
        self._theme_vec  = np.asarray(state.get("theme_vec", [0.0]*self.n_themes), np.float32)
        self.set_genome(state.get("genome", self.genome))
        
        # Restore new state
        self._theme_profiles = {int(k): np.asarray(v) for k, v in state.get("theme_profiles", {}).items()}
        self._current_theme = state.get("current_theme", 0)
        self._theme_momentum = deque(state.get("theme_momentum", []), maxlen=10)


# --------------------------------------------------------------------------- #
# FractalRegimeConfirmation
# --------------------------------------------------------------------------- #
class FractalRegimeConfirmation(Module):
    """
    Confirms market regime (trending, volatile, noise) using fractal metrics
    plus macro theme confidence. Now supports evolutionary adaptation!
    """
    def __init__(self, window: int = 100, debug: bool = True, genome: Dict[str, Any] = None):
        if genome:
            window = genome.get("window", window)
            self.coeff_h = genome.get("coeff_h", 0.4)
            self.coeff_vr = genome.get("coeff_vr", 0.3)
            self.coeff_we = genome.get("coeff_we", 0.3)
        else:
            self.coeff_h = 0.4
            self.coeff_vr = 0.3
            self.coeff_we = 0.3

        self._noise_to_volatile    = 0.30
        self._volatile_to_noise    = 0.20
        self._volatile_to_trending = 0.60
        self._trending_to_volatile = 0.50

        self.window = window
        self.debug  = debug
        self._buf   = deque(maxlen=int(window * 0.75))
        self.regime_strength = 0.0
        self.label          = "noise"
        
        # NEW: Track regime characteristics
        self._regime_history = deque(maxlen=50)
        self._trend_direction = 0.0  # -1 to 1

        self._forced_label: str | None = None  # Used for test monkeypatching
        self._forced_strength: float | None = None

        # Genome for evolution
        self.genome = {
            "window": self.window,
            "coeff_h": self.coeff_h,
            "coeff_vr": self.coeff_vr,
            "coeff_we": self.coeff_we,
        }
                
        # Logger for market regime changes
        self.logger = logging.getLogger("FractalRegimeConfirmation")
        if not self.logger.handlers:
            handler = logging.FileHandler("logs/regime/fractal_regime.log")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)  # Set to DEBUG for more detailed logs

    def reset(self):
        self._buf.clear()
        self.regime_strength, self.label = 0.0, "noise"
        self._forced_label = None
        self._forced_strength = None
        self._regime_history.clear()
        self._trend_direction = 0.0

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
        data_dict: Dict[str, Dict[str, pd.DataFrame]] | None = None,
        current_step: int | None = None,
        theme_detector: Any = None,
    ) -> Tuple[str, float]:
        """
        Compute H, VR, WE; blend with theme_conf; update label with hysteresis.
        Called by the env once per bar **and** once during reset with no args.
        """
        # ── early-exit on dummy call ───────────────────────────────────────
        if data_dict is None or current_step is None:
            return self.label, self.regime_strength

        # 1) Forced override (for testing)
        if self._forced_label is not None:
            self.label = self._forced_label
            self.regime_strength = self._forced_strength  # type: ignore
            return self.label, self.regime_strength

        # 2) Extract price series
        inst = next(iter(data_dict))
        df = data_dict[inst]["D1"]
        ts = df["close"].values[max(0, current_step - self.window) : current_step].astype(np.float32)
        
        # Calculate trend direction
        if len(ts) >= 20:
            recent = ts[-20:]
            old = ts[-40:-20] if len(ts) >= 40 else ts[:20]
            self._trend_direction = np.clip((recent.mean() - old.mean()) / (old.std() + 1e-8), -1, 1)

        # 3) Compute fractal metrics
        H  = self._hurst(ts)
        VR = self._var_ratio(ts)
        WE = self._wavelet_energy(ts)

        # 4) Theme weighting
        theme_conf = 1.0
        if theme_detector is not None:
            theme_detector.fit_if_needed(data_dict, current_step)
            _, theme_conf = theme_detector.detect(data_dict, current_step)

        # 5) Aggregate score & buffer
        score = self.coeff_h * H + self.coeff_vr * VR + self.coeff_we * WE
        self._buf.append(score)
        strength = float(np.mean(self._buf) * theme_conf)
        self.regime_strength = strength

        # 6) State-machine with hysteresis
        old = self.label
        if old == "noise":
            new = "volatile" if strength >= self._noise_to_volatile else "noise"

        elif old == "volatile":
            if strength >= self._volatile_to_trending:
                new = "trending"
            elif strength < self._volatile_to_noise:
                new = "noise"
            else:
                new = "volatile"

        else:  # old == "trending"
            new = "volatile" if strength < self._trending_to_volatile else "trending"

        # 7) Log once on real change
        if new != old:
            self.logger.info(
                f"Market regime changed from {old} to {new}. Strength: {strength:.3f}"
            )
            self.label = new
            
        # Track regime history
        self._regime_history.append((self.label, strength, self._trend_direction))

        # 8) Optional debug print
        if self.debug:
            print(f"[FRC] label={self.label:<8} strength={strength:.3f} trend_dir={self._trend_direction:.3f}")

        return self.label, strength
    
    # ────────── IMPROVED voting-committee hooks ──────────
    def set_action_dim(self, dim: int):
        self._action_dim = int(dim)

    def propose_action(self, obs: Any = None) -> np.ndarray:
        """
        Improved action generation based on regime and trend direction
        """
        if not hasattr(self, "_action_dim"):
            self._action_dim = 2
            
        action = np.zeros(self._action_dim, np.float32)
        
        # Base signal from regime
        if self.label == "trending":
            # Follow the trend direction
            base_signal = self._trend_direction * self.regime_strength
            duration = 0.7  # Longer duration for trends
            
        elif self.label == "volatile":
            # Counter-trend with smaller size
            base_signal = -self._trend_direction * self.regime_strength * 0.5
            duration = 0.3  # Short duration in volatile markets
            
        else:  # noise
            # No strong signal in noise regime
            base_signal = 0.0
            duration = 0.5
            
        # Apply to all instruments
        for i in range(0, self._action_dim, 2):
            action[i] = base_signal
            action[i+1] = duration
            
        # Adjust based on regime stability
        if len(self._regime_history) >= 5:
            recent_regimes = [r[0] for r in list(self._regime_history)[-5:]]
            if len(set(recent_regimes)) > 2:  # Regime is unstable
                action *= 0.5  # Reduce position size
                
        return action

    def confidence(self, obs: Any = None) -> float:
        base_conf = float(self.regime_strength)
        
        # Higher confidence in trending regimes
        if self.label == "trending":
            base_conf = min(base_conf * 1.3, 1.0)
        elif self.label == "noise":
            base_conf *= 0.7
            
        return base_conf

    def force_regime(self, label: str, strength: float):
        self._forced_label = label
        self._forced_strength = float(strength)

    def clear_forced_regime(self):
        self._forced_label = None
        self._forced_strength = None

    def get_observation_components(self) -> np.ndarray:
        label_id = {"trending": 1, "volatile": -1, "noise": 0}.get(self.label, 0)
        return np.array([float(label_id), float(self.regime_strength), self._trend_direction], dtype=np.float32)

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
            "_regime_history": list(self._regime_history),
            "_trend_direction": float(self._trend_direction),
        }

    def set_state(self, state: Dict[str, Any]):
        self.regime_strength = float(state.get("regime_strength", 0.0))
        self.label = state.get("label", "noise")
        self._buf = deque(state.get("_buf", []), maxlen=int(self.window * 0.75))
        self._forced_label = state.get("_forced_label", None)
        self._forced_strength = state.get("_forced_strength", None)
        self.set_genome(state.get("genome", self.genome))
        self._regime_history = deque(state.get("_regime_history", []), maxlen=50)
        self._trend_direction = float(state.get("_trend_direction", 0.0))


# --------------------------------------------------------------------------- #
# TimeAwareRiskScaling
# --------------------------------------------------------------------------- #
class TimeAwareRiskScaling(Module):
    """
    Adjusts portfolio-level risk by hourly volatility and session seasonality.
    Now supports evolutionary adaptation!
    """
    def __init__(self, debug: bool = True, genome: Dict[str, Any] = None):
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
        base_factor = float(np.clip(base_factor, 0.0, 2.0))  # Clamp to [0.0, 2.0]

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

    def __init__(self, action_dim: int, debug: bool = True, genome: dict = None, weights: list = None):
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
        # Don't rebuild model on reset, just clear training flag

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
        # For now, simulate liquidity data since we don't have real order book
        # In production, this would use real order book data
        
        # Simulate spread and depth based on volatility
        if "env" in kwargs:
            env = kwargs["env"]
            vol = env.get_volatility_profile().get(env.instruments[0], 0.01)
            
            # Higher volatility = wider spread, lower depth
            spread = vol * np.random.uniform(0.5, 1.5)
            depth = 1000 / (1 + vol * 10) * np.random.uniform(0.8, 1.2)
            
            self.history.append((spread, depth))
            self._train_if_ready()
            
            if self.debug and len(self.history) % 50 == 0:
                print(f"[LHL] spread={spread:.6f} depth={depth:.1f}")

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
        # Lower spread and higher depth = better liquidity
        score = float(np.log1p(depth) * np.exp(-spread * 100.0))
        return np.clip(score, 0.0, 1.0)

    def get_observation_components(self) -> np.ndarray:
        return np.array([self.current_score()], np.float32)

    def propose_action(self, obs: Any) -> np.ndarray:
        """
        Adjust position size based on liquidity conditions
        """
        liq_score = self.current_score()
        action = np.zeros(self.action_dim, np.float32)
        
        if liq_score < 0.3:
            # Poor liquidity - no trading
            return action
            
        # Scale position size by liquidity
        base_size = 0.3
        for i in range(0, self.action_dim, 2):
            action[i] = base_size * liq_score
            action[i+1] = 0.5  # Standard duration
            
        return action

    def confidence(self, obs: Any) -> float:
        score = self.current_score()
        # High confidence when liquidity is good
        conf = float(np.clip(score, 0.1, 1.0))
        if self.debug:
            print(f"[LiquidityHeatmapLayer] Liquidity score={score:.2f}, confidence={conf:.2f}")
        return conf

    # --- Evolutionary methods ---
    def get_genome(self):
        return self.genome.copy()

    def set_genome(self, genome):
        old_units = self.lstm_units
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
        # Only rebuild model if architecture changed
        if old_units != self.lstm_units:
            self._model = self._build_lstm()
            self._trained = False

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
        if self._trained:
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

        # Mix weights only if architectures match
        new_ws = None
        if self._weights_like(other):
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
            "weights": [w.copy() for w in self._get_weights()] if self._trained else None,
        }

    def set_state(self, state):
        self.history = deque(state.get("history", []), maxlen=200)
        self._trained = bool(state.get("trained", False))
        self.set_genome(state.get("genome", self.genome))
        if state.get("weights") is not None and self._trained:
            self._set_weights(state["weights"])


# --------------------------------------------------------------------------- #
# RegimePerformanceMatrix
# --------------------------------------------------------------------------- #
class RegimePerformanceMatrix(Module):
    """
    Tracks PnL across realised vs predicted volatility regimes.
    """

    def __init__(self, n_regimes: int = 3, decay: float = 0.95, debug: bool = True):
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
        
        return {
            "volatility": vol * crisis["vol_mult"], 
            "liquidity": liq * crisis["liq_mult"]
        }

    def get_observation_components(self) -> np.ndarray:
        flat = self.matrix.flatten()
        acc = []
        for i in range(self.n):
            row_sum = self.matrix[i].sum()
            if row_sum > 1e-4:
                acc.append(float(self.matrix[i, i] / row_sum))
            else:
                acc.append(0.0)
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