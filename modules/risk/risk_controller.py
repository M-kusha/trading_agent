import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List
from modules.core.core import Module
import copy
import random

class DynamicRiskController(Module):
    """
    Evolvable risk controller:
      - Throttles risk dynamically using drawdown & volatility, with freeze logic.
      - Parameters (freeze, thresholds) are evolvable (mutation, crossover).
      - Audit: state hot-swap, robust obs, propose_action, confidence.
    """

    DEFAULTS = {
        "freeze_duration": 5,
        "vol_history_len": 100,
        "dd_threshold": 0.2,
        "vol_ratio_threshold": 1.5,
    }

    def __init__(self, params: Optional[Dict[str, float]] = None, action_dim: int = 1, debug: bool = False):
        # Allow param dict or defaults (for evolutionary ops)
        p = copy.deepcopy(params) if params else dict(self.DEFAULTS)
        self.base_duration  = int(p.get("freeze_duration", self.DEFAULTS["freeze_duration"]))
        self.vol_history    = deque(maxlen=int(p.get("vol_history_len", self.DEFAULTS["vol_history_len"])))
        self.dd_th          = float(p.get("dd_threshold", self.DEFAULTS["dd_threshold"]))
        self.vol_th         = float(p.get("vol_ratio_threshold", self.DEFAULTS["vol_ratio_threshold"]))
        self.action_dim     = int(action_dim)
        self.debug          = debug

        self.freeze_counter = 0
        self._last_vol      = 0.0
        self._last_dd       = 0.0

        # For evolution
        self._params = {
            "freeze_duration": self.base_duration,
            "vol_history_len": self.vol_history.maxlen,
            "dd_threshold": self.dd_th,
            "vol_ratio_threshold": self.vol_th,
        }

    # ───────────────────────────────────────────── #
    # EVOLUTIONARY LOGIC

    def mutate(self, std: float = 0.1):
        """Randomly perturb internal risk thresholds and memory windows."""
        # Mutate each param
        self.base_duration = int(np.clip(
            self.base_duration + int(np.random.normal(0, std * 4)),
            1, 20
        ))
        new_vhl = int(np.clip(self.vol_history.maxlen + int(np.random.normal(0, std * 30)), 10, 1000))
        # reset vol_history with new window
        self.vol_history = deque(list(self.vol_history)[-new_vhl:], maxlen=new_vhl)
        self.dd_th = float(np.clip(self.dd_th + np.random.normal(0, std * 0.1), 0.05, 0.8))
        self.vol_th = float(np.clip(self.vol_th + np.random.normal(0, std * 0.2), 0.3, 5.0))

        # Save mutated params
        self._params = {
            "freeze_duration": self.base_duration,
            "vol_history_len": self.vol_history.maxlen,
            "dd_threshold": self.dd_th,
            "vol_ratio_threshold": self.vol_th,
        }

    def crossover(self, other: "DynamicRiskController") -> "DynamicRiskController":
        """Mix params from two controllers to make a child."""
        # Average or randomly choose for each param
        def blend(a, b):
            return a if np.random.rand() > 0.5 else b

        params = {
            "freeze_duration": blend(self.base_duration, other.base_duration),
            "vol_history_len": blend(self.vol_history.maxlen, other.vol_history.maxlen),
            "dd_threshold": blend(self.dd_th, other.dd_th),
            "vol_ratio_threshold": blend(self.vol_th, other.vol_th),
        }
        # Child uses parent's action_dim, debug = either parent debug
        return DynamicRiskController(params=params, action_dim=self.action_dim, debug=self.debug or other.debug)

    def get_params(self) -> Dict[str, float]:
        return dict(self._params)

    # ───────────────────────────────────────────── #
    # CORE MODULE INTERFACE

    def reset(self):
        self.freeze_counter = 0
        self.vol_history.clear()
        self._last_vol = 0.0
        self._last_dd = 0.0

    def step(self, stats: Optional[Dict[str, float]] = None, **kwargs):
        if stats:
            self.adjust_risk(stats)

    def adjust_risk(self, stats: Dict[str, float]) -> None:
        dd  = float(stats.get("drawdown", 0.0))
        vol = float(stats.get("volatility", 0.0))
        self._last_dd = dd
        self._last_vol = vol

        # 1) Record volatility
        self.vol_history.append(vol)

        # 2) On the first datapoint, only freeze if vol is meaningful
        if len(self.vol_history) == 1:
            self.freeze_counter = self.base_duration if vol > 0.05 else 0
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

        if self.debug:
            print(f"[DynamicRiskController] adj_risk: dd={dd:.3f} vol={vol:.3f} vr={vr:.2f} freeze={self.freeze_counter} (dd_th={self.dd_th:.2f}, vol_th={self.vol_th:.2f})")

    def get_observation_components(self) -> np.ndarray:
        # 0.0 during a freeze, 1.0 otherwise
        scale = 0.0 if self.freeze_counter > 0 else 1.0
        obs = np.array([scale, self._last_dd, self._last_vol], dtype=np.float32)
        if self.debug:
            print(f"[DynamicRiskController] freeze_counter={self.freeze_counter}, obs={obs}")
        return obs

    def get_state(self) -> Dict[str, Any]:
        return {
            "freeze_counter": int(self.freeze_counter),
            "vol_history": list(self.vol_history),
            "_last_dd": float(self._last_dd),
            "_last_vol": float(self._last_vol),
            "params": self.get_params()
        }

    def set_state(self, state: Dict[str, Any]):
        self.freeze_counter = int(state.get("freeze_counter", 0))
        vhl = state.get("params", {}).get("vol_history_len", self.vol_history.maxlen)
        self.vol_history = deque(state.get("vol_history", []), maxlen=int(vhl))
        self._last_dd = float(state.get("_last_dd", 0.0))
        self._last_vol = float(state.get("_last_vol", 0.0))
        # Restore params for further evolution
        params = state.get("params", {})
        self.base_duration = int(params.get("freeze_duration", self.base_duration))
        self.dd_th = float(params.get("dd_threshold", self.dd_th))
        self.vol_th = float(params.get("vol_ratio_threshold", self.vol_th))
        self._params = params if params else self._params

    def propose_action(self, obs: Any = None) -> np.ndarray:
        """
        Output ∈ [0,1] per action dimension, acting as a multiplicative
        throttle. 0 ⇒ freeze, 1 ⇒ full size.
        """
        scale = 0.0 if self.freeze_counter > 0 else 1.0
        return np.full(self.action_dim, scale, dtype=np.float32)

    def confidence(self, obs: Any = None) -> float:
        """
        Returns lower confidence if in a risk-freeze state.
        """
        conf = 0.3 if self.freeze_counter > 0 else 1.0
        if self.debug:
            print(f"[DynamicRiskController] Freeze={self.freeze_counter}, confidence={conf:.2f}")
        return conf
