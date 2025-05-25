#modules/risk_controller.py

import numpy as np
from collections import deque
from typing import Dict, Any
from modules.core.core import Module

class DynamicRiskController(Module):
    def __init__(self, params: Dict[str, float], action_dim, debug=False):
       
        self.freeze_counter = params.get("freeze_counter",0)
        self.base_duration  = params.get("freeze_duration",5)
        self.vol_history    = deque(maxlen=params.get("vol_history_len",100))
        self.dd_th          = params.get("dd_threshold",0.2)
        self.vol_th         = params.get("vol_ratio_threshold",1.5)
        self.portfolio      = {"freeze_counter":self.freeze_counter}
        self.action_dim = action_dim
        self.debug = debug

    def reset(self):
        self.freeze_counter = 0
        self.vol_history.clear()

    def step(self, **kwargs):
        pass

    def adjust_risk(self, stats: Dict[str,float]) -> None:
        dd  = stats.get("drawdown", 0.0)
        vol = stats.get("volatility", 0.0)

        # 1) Record volatility
        self.vol_history.append(vol)

        # 2) On the very first data-point, only freeze if vol is meaningful
        if len(self.vol_history) == 1:
            if vol > 0.05:  # Only freeze if volatility is above 5%
                self.freeze_counter = self.base_duration
            else:
                self.freeze_counter = 0
            self.portfolio["freeze_counter"] = self.freeze_counter
            return

        # 3) Compute volatility ratio and freeze duration
        avg_vol = float(np.mean(self.vol_history))
        vr = vol / (avg_vol + 1e-8)
        dur = int(self.base_duration * np.clip(vr, 0.5, 2.0))
        dur = max(1, min(dur, 10))

        # 4) Freeze logic
        if dd > self.dd_th or vr > self.vol_th:
            self.freeze_counter = dur
        else:
            self.freeze_counter = max(0, self.freeze_counter - 1)

        # 5) Publish updated freeze_counter
        self.portfolio["freeze_counter"] = self.freeze_counter


    def get_observation_components(self) -> np.ndarray:
        return np.array([self.freeze_counter], dtype=np.float32)
    
    def get_state(self):
        return {
            "risk_factors": self.risk_factors,
        }

    def set_state(self, state):
        self.risk_factors = state.get("risk_factors", {})

    def propose_action(self, obs: Any) -> np.ndarray:
        """
        Output ∈ [0,1] per action dimension, acting as a multiplicative
        throttle. 0 ⇒ freeze, 1 ⇒ full size.
        """
        scale = 0.0 if self.freeze_counter > 0 else 1.0
        return np.full(self.action_dim, scale, dtype=np.float32)
    
    def confidence(self, obs: Any) -> float:
        """
        Returns lower confidence if we're in a risk-freeze state.
        """
        if self.freeze_counter > 0:
            conf = 0.3
        else:
            conf = 1.0

        if self.debug:
            print(f"[DynamicRiskController] Freeze={self.freeze_counter}, confidence={conf:.2f}")
        return conf
