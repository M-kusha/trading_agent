# ─────────────────────────────────────────────────────────────
# File: modules/market/market_theme_detector.py
# ─────────────────────────────────────────────────────────────

import logging
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from collections import deque
from typing import Any, List, Dict, Tuple
import pywt
from sklearn.cluster import MiniBatchKMeans
from modules.core.core import Module
from utils.get_dir import _ensure_dir
import random

class MarketThemeDetector(Module):
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