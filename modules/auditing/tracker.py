# modules/tracker.py

import numpy as np
from typing import Any, Tuple, List
from modules.core.core import Module
class TradeThesisTracker(Module):
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        self.records: List[Tuple[Any, float]] = []

    def step(self, **kwargs) -> None:
        pass

    def record(self, thesis: Any, pnl: float) -> None:
        self.records.append((thesis, pnl))

    def get_observation_components(self) -> np.ndarray:
        # outputs [uniq_theses, mean_pnl, sd_pnl] + up to 4 per-thesis means = total length 7
        if not self.records:
            return np.zeros(7, dtype=np.float32)

        theses, pnls = zip(*self.records)
        uniq = len(set(theses))
        mean_p = float(np.mean(pnls))
        sd_p = float(np.std(pnls))

        # per-thesis mean for up to 4 unique theses
        per: List[float] = []
        for t in list(dict.fromkeys(theses))[:4]:
            vs = [p for (th, p) in self.records if th == t]
            per.append(float(np.mean(vs)))
        per += [0.0] * (4 - len(per))

        return np.array([uniq, mean_p, sd_p, *per], dtype=np.float32)
