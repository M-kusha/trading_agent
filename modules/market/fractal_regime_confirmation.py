# ─────────────────────────────────────────────────────────────
# File: modules/market/fractal_regime_confirmation.py
# ─────────────────────────────────────────────────────────────

import logging
import numpy as np
import pandas as pd
from scipy.stats import linregress
from collections import deque
from typing import Any,Dict, Tuple
import pywt
from ..core.core import Module
import random

class FractalRegimeConfirmation(Module):
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