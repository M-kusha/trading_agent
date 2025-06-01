# modules/visualization.py

import numpy as np
from typing import List
from ..core.core import Module

class VisualizationInterface(Module):
    """Collects step records for UI/log visualization. Can be extended to stream to a frontend."""
    def __init__(self, debug=False):
        self.debug   = debug
        self.records = []

    def reset(self):
        self.records.clear()

    def step(self, **kwargs):
        pass

    def record_step(self, **kwargs):
        self.records.append(kwargs)
        if self.debug:
            print(f"[VisualizationInterface] {kwargs}")

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


class TradeMapVisualizer(Module):
    """
    Plots price series with buy/sell trade markers for manual or automated review.
    Trades should be dicts with 'entry_idx', 'exit_idx', and 'pnl'.
    """
    def __init__(self, debug=False):
        self.debug = debug

    def reset(self):
        pass

    def step(self, **kwargs):
        pass

    def plot(self, series: List[float], trades: List[dict]):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        series_np = np.asarray(series)
        ax.plot(series_np, color="black", label="Price")

        for tr in trades:
            ei = int(tr.get("entry_idx", 0))
            xi = int(tr.get("exit_idx", 0))
            p  = tr.get("pnl", 0)
            color = "green" if p >= 0 else "red"

            # Robust index checks
            if 0 <= ei < len(series_np):
                ax.scatter([ei], [series_np[ei]], marker="^", c=color, label="Entry" if color == "green" else "Short")
            if 0 <= xi < len(series_np):
                ax.scatter([xi], [series_np[xi]], marker="v", c=color, label="Exit" if color == "green" else "Cover")
            if 0 <= ei < len(series_np) and 0 <= xi < len(series_np):
                ax.plot([ei, xi], [series_np[ei], series_np[xi]], c=color, alpha=0.6)

        ax.set_title("Trade Map")
        ax.set_xlabel("Step")
        ax.set_ylabel("Price")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)
