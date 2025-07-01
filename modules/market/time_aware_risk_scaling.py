# ─────────────────────────────────────────────────────────────
# File: modules/market/time_aware_risk_scaling.py
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Any, Dict
from ..core.core import Module

class TimeAwareRiskScaling(Module):
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
