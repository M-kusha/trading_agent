# modules/opponent.py

import numpy as np
import pandas as pd
from typing import Dict
from modules.core.core import Module

class OpponentSimulator(Module):
    def __init__(self, mode: str="random", intensity: float=1.0, debug=False):
        self.mode      = mode
        self.intensity = intensity
        self.debug     = debug
    def reset(self): pass
    def step(self, **kwargs): pass
    def apply(self, data_dict: Dict[str,Dict[str,pd.DataFrame]]):
        out = {}
        for inst, tfs in data_dict.items():
            out[inst] = {}
            for tf, df in tfs.items():
                arr = df["close"].values.astype(np.float32)
                vol = df["volatility"].values.astype(np.float32)
                if self.mode=="random":
                    noise = np.random.randn(len(arr)).astype(np.float32)*vol*self.intensity
                    arr = arr + noise
                tmp = df.copy()
                tmp["close"] = arr
                out[inst][tf] = tmp
        return out
    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)
