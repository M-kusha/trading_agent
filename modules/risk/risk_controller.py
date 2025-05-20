#modules/risk_controller.py

import numpy as np
from collections import deque
from typing import Dict
from modules.core.core import Module

class DynamicRiskController(Module):
    def __init__(self, params: Dict[str,float], debug=False):
        self.freeze_counter = params.get("freeze_counter",0)
        self.base_duration  = params.get("freeze_duration",5)
        self.vol_history    = deque(maxlen=params.get("vol_history_len",100))
        self.dd_th          = params.get("dd_threshold",0.2)
        self.vol_th         = params.get("vol_ratio_threshold",1.5)
        self.portfolio      = {"freeze_counter":self.freeze_counter}
        self.debug = debug

    def reset(self):
        self.freeze_counter = 0
        self.vol_history.clear()

    def step(self, **kwargs):
        pass

    def adjust_risk(self, stats: Dict[str,float]) -> None:
        """
        Record drawdown & volatility; adjust freeze_counter.
        Only auto-freeze on very first sample if vol > 5%.
        """
        dd  = stats.get("drawdown", 0.0)
        vol = stats.get("volatility", 0.0)

        # 1) record volatility
        self.vol_history.append(vol)

        # 2) on the *very first* data‐point, only freeze if vol is meaningful
        if len(self.vol_history) == 1:
            if vol > 0.05:
                self.freeze_counter = self.base_duration
            else:
                self.freeze_counter = 0
            self.portfolio["freeze_counter"] = self.freeze_counter
            return

        # 3) compute volatility ratio & duration
        avg_vol = float(np.mean(self.vol_history))
        vr = vol / (avg_vol + 1e-8)
        dur = int(self.base_duration * np.clip(vr, 0.5, 2.0))
        dur = max(1, min(dur, 10))

        # 4) freeze logic
        if dd > self.dd_th or vr > self.vol_th:
            self.freeze_counter = dur
        else:
            self.freeze_counter = max(0, self.freeze_counter - 1)

        # 5) publish
        self.portfolio["freeze_counter"] = self.freeze_counter

    def get_observation_components(self) -> np.ndarray:
        return np.array([self.freeze_counter], dtype=np.float32)
