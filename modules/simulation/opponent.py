# modules/opponent.py

import numpy as np
import pandas as pd
import random
from typing import Dict
from modules.core.core import Module

class OpponentSimulator(Module):
    """
    Simulates adversarial or randomized market opponents.
    Supports evolutionary mutation/crossover of intensity and simulation mode.
    """
    MODES = ["random", "adversarial", "trend-follow"]

    def __init__(self, mode: str = "random", intensity: float = 1.0, debug=False):
        self.mode      = mode if mode in self.MODES else "random"
        self.intensity = float(intensity)
        self.debug     = debug
        self.simulation_state = {}

    def reset(self):
        self.simulation_state = {}

    def step(self, **kwargs):
        # No-op; stateless by default
        pass

    def apply(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Applies the current simulation mode to the input price data.
        """
        out = {}
        for inst, tfs in data_dict.items():
            out[inst] = {}
            for tf, df in tfs.items():
                arr = df["close"].values.astype(np.float32)
                vol = df["volatility"].values.astype(np.float32) if "volatility" in df else np.ones_like(arr)
                arr_sim = arr.copy()

                if self.mode == "random":
                    noise = np.random.randn(len(arr)) * vol * self.intensity
                    arr_sim = arr + noise

                elif self.mode == "adversarial":
                    # Worst-case: shift prices against typical trend direction
                    direction = -np.sign(np.mean(np.diff(arr)))  # reverse main trend
                    arr_sim = arr + direction * vol * self.intensity * np.linspace(0.5, 1.5, len(arr))

                elif self.mode == "trend-follow":
                    # Boost existing trend and volatility
                    trend = np.cumsum(np.random.randn(len(arr))) * vol * 0.2 * self.intensity
                    arr_sim = arr + trend

                tmp = df.copy()
                tmp["close"] = arr_sim
                out[inst][tf] = tmp

                if self.debug:
                    print(f"[OpponentSimulator] {self.mode=} {self.intensity=} {inst=} {tf=}")

        return out

    def mutate(self, std=0.2):
        """Perturb intensity and possibly mode."""
        self.intensity = float(np.clip(self.intensity + np.random.normal(0, std), 0.05, 10.0))
        if random.random() < 0.2:
            self.mode = random.choice(self.MODES)

    def crossover(self, other: "OpponentSimulator") -> "OpponentSimulator":
        """Cross two OpponentSimulators."""
        mode = self.mode if random.random() < 0.5 else other.mode
        intensity = self.intensity if random.random() < 0.5 else other.intensity
        return OpponentSimulator(mode=mode, intensity=intensity, debug=self.debug or other.debug)

    def get_observation_components(self) -> np.ndarray:
        # Could output intensity and mode encoding if needed
        mode_idx = self.MODES.index(self.mode)
        return np.array([self.intensity, mode_idx], dtype=np.float32)

    def get_state(self):
        return {
            "mode": self.mode,
            "intensity": self.intensity,
            "simulation_state": self.simulation_state,
        }

    def set_state(self, state):
        self.mode = state.get("mode", self.mode)
        self.intensity = float(state.get("intensity", self.intensity))
        self.simulation_state = state.get("simulation_state", {})

