#modules/visualization.py   

import numpy as np
from typing import List
from ..core.core import Module

class VisualizationInterface(Module):
    def __init__(self, debug=False):
        self.debug   = debug
        self.records = []
    def reset(self): self.records.clear()
    def step(self, **kwargs): pass
    def record_step(self, **kwargs):
        self.records.append(kwargs)
        if self.debug: print(f"[VI] {kwargs}")
    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)
    
class TradeMapVisualizer(Module):
    def __init__(self, debug=False):
        self.debug = debug
    def reset(self): pass
    def step(self, **kwargs): pass
    def plot(self, series: List[float], trades: List[dict]):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(series, color="black")
        for tr in trades:
            ei = tr.get("entry_idx",0)
            xi = tr.get("exit_idx",0)
            p  = tr["pnl"]
            c  = "green" if p>=0 else "red"
            ax.scatter([ei],[series[ei]],marker="^",c=c)
            ax.scatter([xi],[series[xi]],marker="v",c=c)
            ax.plot([ei,xi],[series[ei],series[xi]],c=c,alpha=0.6)
        return fig
    def get_observation_components(self)->np.ndarray:
        return np.zeros(1, dtype=np.float32)
