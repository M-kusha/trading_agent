# modules/visualization.py

import numpy as np
from typing import List, Dict, Any, Optional
from ..core.core import Module
import copy
import random
import datetime

class VisualizationInterface(Module):
    """
    Collects step records for UI/log visualization.
    Evolvable/debuggable: supports mutation of debug mode, stream sample size, etc.
    Can be extended to stream to a frontend (stream_update).
    """

    def __init__(self, debug: bool = True, max_steps: int = 500):
        self.debug = debug
        self.max_steps = int(max_steps)
        self.records: List[Dict[str, Any]] = []
        self._decision_trace: List[Dict[str, Any]] = []

    def reset(self):
        self.records.clear()
        self._decision_trace.clear()

    def step(self, **kwargs):
        # Optionally call this each agent step to record info for later visualization.
        self.record_step(**kwargs)

    def record_step(self, **kwargs):
        entry = dict(kwargs)
        entry["_time"] = datetime.datetime.utcnow().isoformat()
        self.records.append(entry)
        if len(self.records) > self.max_steps:
            self.records = self.records[-self.max_steps:]
        self._decision_trace.append(entry)
        if len(self._decision_trace) > self.max_steps:
            self._decision_trace = self._decision_trace[-self.max_steps:]
        if self.debug:
            print(f"[VisualizationInterface] {entry}")
        self.stream_update(entry)

    def stream_update(self, entry: Dict[str, Any]):
        # Extension point: send updates to web dashboard, stream, etc.
        pass

    def get_last_steps(self, n: int = 20) -> List[Dict[str, Any]]:
        return self.records[-n:]

    def get_last_decisions(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._decision_trace[-n:]

    # Evolutionary logic for adaptive visualization
    def mutate(self, std: float = 1.0):
        self.max_steps = max(50, int(self.max_steps + np.random.normal(0, std * 50)))
        if random.random() < 0.2:
            self.debug = not self.debug

    def crossover(self, other: "VisualizationInterface") -> "VisualizationInterface":
        debug = self.debug if random.random() < 0.5 else other.debug
        max_steps = self.max_steps if random.random() < 0.5 else other.max_steps
        return VisualizationInterface(debug=debug, max_steps=max_steps)

    def get_state(self):
        return {
            "debug": self.debug,
            "max_steps": self.max_steps,
            "records": copy.deepcopy(self.records),
            "_decision_trace": copy.deepcopy(self._decision_trace),
        }

    def set_state(self, state):
        self.debug = bool(state.get("debug", self.debug))
        self.max_steps = int(state.get("max_steps", self.max_steps))
        self.records = list(state.get("records", []))
        self._decision_trace = list(state.get("_decision_trace", []))

    def get_observation_components(self) -> np.ndarray:
        # Optionally encode recent record count or debug status
        return np.array([float(len(self.records)), float(self.debug)], dtype=np.float32)


class TradeMapVisualizer(Module):
    """
    Plots price series with buy/sell trade markers for manual or automated review.
    Evolvable/debuggable: supports mutation of plot style, symbol size, debug.
    Can be extended for more plot types (volatility, overlays, heatmaps).
    """

    def __init__(self, debug: bool = False, marker_size: int = 60, style: str = "default"):
        self.debug = debug
        self.marker_size = int(marker_size)
        self.style = style
        self._last_fig = None

    def reset(self):
        self._last_fig = None

    def step(self, **kwargs):
        # Typically not needed unless making live plots each step
        pass

    def plot(self, series: List[float], trades: List[dict], show: bool = False, title: str = "Trade Map") -> Any:
        import matplotlib.pyplot as plt

        # Optional: apply style
        if self.style != "default":
            plt.style.use(self.style)

        fig, ax = plt.subplots()
        series_np = np.asarray(series)
        ax.plot(series_np, label="Price")

        for tr in trades:
            ei = int(tr.get("entry_idx", 0))
            xi = int(tr.get("exit_idx", 0))
            p = tr.get("pnl", 0)
            color = "green" if p >= 0 else "red"

            # Robust index checks
            if 0 <= ei < len(series_np):
                ax.scatter([ei], [series_np[ei]], marker="^", c=color, s=self.marker_size, label="Entry" if color == "green" else "Short")
            if 0 <= xi < len(series_np):
                ax.scatter([xi], [series_np[xi]], marker="v", c=color, s=self.marker_size, label="Exit" if color == "green" else "Cover")
            if 0 <= ei < len(series_np) and 0 <= xi < len(series_np):
                ax.plot([ei, xi], [series_np[ei], series_np[xi]], c=color, alpha=0.6, linestyle="--" if p < 0 else "-")

        # Optionally add PnL histogram or trade annotation
        if trades and self.debug:
            pnls = [tr.get("pnl", 0) for tr in trades]
            print(f"[TradeMapVisualizer] Trades: {len(trades)}, Mean PnL: {np.mean(pnls):.2f}")

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Price")
        # Only show unique legend items
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best")
        fig.tight_layout()
        self._last_fig = fig
        if show:
            plt.show()
        return fig

    def get_last_figure(self):
        return self._last_fig

    # Evolutionary logic
    def mutate(self, std: float = 1.0):
        self.marker_size = max(10, int(self.marker_size + np.random.normal(0, std * 10)))
        if random.random() < 0.2:
            self.debug = not self.debug
        if random.random() < 0.1:
            self.style = random.choice(["default", "ggplot", "seaborn", "bmh", "classic"])

    def crossover(self, other: "TradeMapVisualizer") -> "TradeMapVisualizer":
        debug = self.debug if random.random() < 0.5 else other.debug
        marker_size = self.marker_size if random.random() < 0.5 else other.marker_size
        style = self.style if random.random() < 0.5 else other.style
        return TradeMapVisualizer(debug=debug, marker_size=marker_size, style=style)

    def get_state(self):
        return {
            "debug": self.debug,
            "marker_size": self.marker_size,
            "style": self.style,
        }

    def set_state(self, state):
        self.debug = bool(state.get("debug", self.debug))
        self.marker_size = int(state.get("marker_size", self.marker_size))
        self.style = str(state.get("style", self.style))

    def get_observation_components(self) -> np.ndarray:
        # Optionally encode marker size, debug, style idx
        styles = ["default", "ggplot", "seaborn", "bmh", "classic"]
        style_idx = styles.index(self.style) if self.style in styles else 0
        return np.array([float(self.marker_size), float(self.debug), float(style_idx)], dtype=np.float32)
