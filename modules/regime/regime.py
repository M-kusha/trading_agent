# modules/regime.py

import numpy as np
from collections import deque
from ..core.core import Module

class MarketRegimeSwitcher(Module):
    """
    Analyzes recent price/feature data to classify current regime.
    Can be used to adapt strategy, risk, reward, or logging.
    Outputs regime as one-hot (for obs), and string for logging/explanation.

    Regimes:
      - trending_up
      - trending_down
      - mean_reverting
      - high_vol
      - low_vol
      - neutral
    """
    REGIMES = ["trending_up", "trending_down", "mean_reverting", "high_vol", "low_vol", "neutral"]

    def __init__(self, window: int = 50, vol_window: int = 20, debug: bool = False):
        self.window = window          # Lookback for trend/mean-reversion
        self.vol_window = vol_window  # Lookback for volatility calculation
        self.debug = debug
        self.reset()

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

        # Define thresholds (can be tuned/learned)
        trend_thr = 0.5 * std_ret
        vol_high_thr = np.percentile(np.abs(returns), 80)
        vol_low_thr = np.percentile(np.abs(returns), 20)

        # Regime logic
        if mean_ret > trend_thr:
            self.regime = "trending_up"
        elif mean_ret < -trend_thr:
            self.regime = "trending_down"
        elif self.volatility > vol_high_thr:
            self.regime = "high_vol"
        elif self.volatility < vol_low_thr:
            self.regime = "low_vol"
        elif abs(mean_ret) < 0.1 * std_ret:
            self.regime = "mean_reverting"
        else:
            self.regime = "neutral"

        if self.debug:
            print(f"[MarketRegimeSwitcher] regime={self.regime} (mean_ret={mean_ret:.4f}, vol={self.volatility:.4f})")

    def get_regime(self) -> str:
        return self.regime

    def get_observation_components(self) -> np.ndarray:
        # One-hot encode current regime for obs space
        arr = np.zeros(len(self.REGIMES), dtype=np.float32)
        if self.regime in self.REGIMES:
            arr[self.REGIMES.index(self.regime)] = 1.0
        return arr
