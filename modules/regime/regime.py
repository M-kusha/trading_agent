import numpy as np
from collections import deque
from ..core.core import Module
import copy

class MarketRegimeSwitcher(Module):
    """
    Evolutionary Market Regime Switcher.
    Classifies current regime with tunable/evolving parameters.
    Regimes:
      - trending_up
      - trending_down
      - mean_reverting
      - high_vol
      - low_vol
      - neutral
    """
    REGIMES = ["trending_up", "trending_down", "mean_reverting", "high_vol", "low_vol", "neutral"]

    def __init__(
        self,
        window: int = 50,
        vol_window: int = 20,
        trend_factor: float = 0.5,
        mean_thr_factor: float = 0.1,
        vol_high_pct: float = 80.0,
        vol_low_pct: float = 20.0,
        debug: bool = False,
    ):
        self.window = window
        self.vol_window = vol_window
        self.trend_factor = trend_factor
        self.mean_thr_factor = mean_thr_factor
        self.vol_high_pct = vol_high_pct
        self.vol_low_pct = vol_low_pct
        self.debug = debug
        self.reset()

    # Evolutionary mutation/crossover methods
    def mutate(self, std: float = 0.1):
        # Mutate thresholds and windows
        self.window += int(np.random.choice([-2, -1, 0, 1, 2]))
        self.window = int(np.clip(self.window, 10, 200))
        self.vol_window += int(np.random.choice([-2, -1, 0, 1, 2]))
        self.vol_window = int(np.clip(self.vol_window, 5, self.window))
        self.trend_factor += np.random.normal(0, std)
        self.trend_factor = float(np.clip(self.trend_factor, 0.1, 2.0))
        self.mean_thr_factor += np.random.normal(0, std / 2)
        self.mean_thr_factor = float(np.clip(self.mean_thr_factor, 0.01, 0.5))
        self.vol_high_pct += np.random.normal(0, 2.0)
        self.vol_high_pct = float(np.clip(self.vol_high_pct, 60.0, 99.0))
        self.vol_low_pct += np.random.normal(0, 2.0)
        self.vol_low_pct = float(np.clip(self.vol_low_pct, 1.0, 40.0))
        if self.debug:
            print("[MarketRegimeSwitcher] Mutated parameters.")

    def crossover(self, other: "MarketRegimeSwitcher"):
        child = copy.deepcopy(self)
        for attr in [
            "window", "vol_window", "trend_factor",
            "mean_thr_factor", "vol_high_pct", "vol_low_pct"
        ]:
            if np.random.rand() > 0.5:
                setattr(child, attr, getattr(other, attr))
        if self.debug:
            print("[MarketRegimeSwitcher] Crossover complete.")
        return child

    def reset(self):
        self.prices = deque(maxlen=self.window)
        self.regime = "neutral"
        self.volatility = 0.0

    def step(self, price: float = None, **kwargs):
        if price is not None:
            self.prices.append(price)

        if len(self.prices) < self.window:
            self.regime = "neutral"
            self.volatility = 0.0
            return

        prices = np.array(self.prices)
        returns = np.diff(prices)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        self.volatility = np.std(prices[-self.vol_window:]) if len(prices) >= self.vol_window else 0.0

        # Now, all regime thresholds are *parameters* that can evolve!
        trend_thr = self.trend_factor * std_ret
        vol_high_thr = np.percentile(np.abs(returns), self.vol_high_pct)
        vol_low_thr = np.percentile(np.abs(returns), self.vol_low_pct)
        mean_thr = self.mean_thr_factor * std_ret

        # Regime logic
        if mean_ret > trend_thr:
            self.regime = "trending_up"
        elif mean_ret < -trend_thr:
            self.regime = "trending_down"
        elif self.volatility > vol_high_thr:
            self.regime = "high_vol"
        elif self.volatility < vol_low_thr:
            self.regime = "low_vol"
        elif abs(mean_ret) < mean_thr:
            self.regime = "mean_reverting"
        else:
            self.regime = "neutral"

        if self.debug:
            print(f"[MarketRegimeSwitcher] regime={self.regime} (mean_ret={mean_ret:.4f}, vol={self.volatility:.4f})")

    def get_regime(self) -> str:
        return self.regime

    def get_observation_components(self) -> np.ndarray:
        arr = np.zeros(len(self.REGIMES), dtype=np.float32)
        if self.regime in self.REGIMES:
            arr[self.REGIMES.index(self.regime)] = 1.0
        return arr

    def get_state(self):
        return {
            "params": {
                "window": self.window,
                "vol_window": self.vol_window,
                "trend_factor": self.trend_factor,
                "mean_thr_factor": self.mean_thr_factor,
                "vol_high_pct": self.vol_high_pct,
                "vol_low_pct": self.vol_low_pct,
            },
            "prices": list(self.prices),
            "regime": self.regime,
            "volatility": self.volatility
        }

    def set_state(self, state):
        params = state.get("params", {})
        self.window = int(params.get("window", self.window))
        self.vol_window = int(params.get("vol_window", self.vol_window))
        self.trend_factor = float(params.get("trend_factor", self.trend_factor))
        self.mean_thr_factor = float(params.get("mean_thr_factor", self.mean_thr_factor))
        self.vol_high_pct = float(params.get("vol_high_pct", self.vol_high_pct))
        self.vol_low_pct = float(params.get("vol_low_pct", self.vol_low_pct))
        self.prices = deque(state.get("prices", []), maxlen=self.window)
        self.regime = state.get("regime", "neutral")
        self.volatility = float(state.get("volatility", 0.0))
