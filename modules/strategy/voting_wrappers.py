# modules/strategy/voting_wrappers.py

import numpy as np
from typing import Any, List, Dict
import math
import torch
import logging

from modules.risk.risk_monitor import ActiveTradeMonitor
from modules.strategy.strategy import MetaRLController

# Utility for action vector construction
def _dir_size_to_vec(angle: float, magnitude: float, n_instruments: int, action_dim: int) -> np.ndarray:
    vec = np.zeros(action_dim, np.float32)
    if magnitude <= 0:
        return vec
    idx = int(round((angle % 360) / 360 * n_instruments)) % n_instruments
    vec[2 * idx] = magnitude
    vec[2 * idx + 1] = 0.5
    return vec

# ========== 1. MarketThemeDetector → ThemeExpert ==========
class ThemeExpert:
    """
    Converts (label,strength) from MarketThemeDetector into an action.
    Long on 'up-trend', flat otherwise. Supports module state, debug, and reset.
    """
    def __init__(self, detector, env_ref, trend_label: str = "trending", max_size: float = 1.0, debug=True):
        self.det         = detector
        self.env         = env_ref
        self.trend_label = trend_label
        self.max_size    = max_size
        self._last_strength = 0.0
        self._zero = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug
        self.logger = logging.getLogger("ThemeExpert")
        if self.debug and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def reset(self):
        self._last_strength = 0.0

    def propose_action(self, obs: Any, extras: Dict = None) -> np.ndarray:
        # Market-closed check
        if extras and not extras.get("market_open", True):
            if self.debug:
                self.logger.debug("Market closed: abstaining from action.")
            return self._zero.copy()

        try:
            lab, stren = self.det.detect(self.env.data, self.env.current_step)
        except Exception as e:
            lab, stren = "none", 0.0
            if self.debug:
                self.logger.debug(f"Detector error: {e}")

        self._last_strength = float(np.clip(stren, 0.0, 1.0))
        if lab != self.trend_label:
            return self._zero.copy()
        vec = self._zero.copy()
        vec[0] = self._last_strength * self.max_size
        vec[1] = 0.5
        return vec

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        return 0.5 + 0.5 * self._last_strength

    def get_state(self):
        return {"_last_strength": self._last_strength}

    def set_state(self, state):
        self._last_strength = state.get("_last_strength", 0.0)

# ========== 2. TimeAwareRiskScaling → SeasonalityRiskExpert ==========
class SeasonalityRiskExpert:
    """
    Scales risk: outputs a vector to multiply baseline size by seasonality_factor.
    """
    def __init__(self, tars_module, env_ref, debug=True):
        self.tars = tars_module
        self.env  = env_ref
        self._zero = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug

    def reset(self):
        pass

    def propose_action(self, obs: Any, extras: Dict = None) -> np.ndarray:
        f = float(self.tars.seasonality_factor)
        if not math.isfinite(f):
            f = 1.0
        vec = self._zero.copy()
        for i in range(0, self.env.action_dim, 2):
            vec[i] = f - 1.0
        return vec

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        f = float(self.tars.seasonality_factor)
        return float(min(1.0, 0.5 + abs(f - 1.0)))

# ========== 3. MetaRLController → MetaRLExpert ==========
class MetaRLExpert:
    """
    Uses the Meta-RL policy’s own action; confidence ~ (1 − entropy).
    """
    def __init__(self, meta_rl: "MetaRLController", env_ref, debug=True):
        self.mrl  = meta_rl
        self.env  = env_ref
        self.last_action = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug

    def reset(self):
        self.last_action[:] = 0.0

    def _call_policy(self, obs_vec: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.mrl.device).unsqueeze(0)
        raw = self.mrl.act(obs_t)
        if isinstance(raw, dict):
            raw = raw.get("action", raw)
        act = np.asarray(raw, dtype=np.float32).reshape(-1)
        if act.size < self.env.action_dim:
            act = np.pad(act, (0, self.env.action_dim - act.size))
        return np.clip(act[: self.env.action_dim], -1.0, 1.0)

    def propose_action(self, obs: np.ndarray, extras: Dict = None) -> np.ndarray:
        self.last_action = self._call_policy(obs)
        return self.last_action.copy()

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        ent = getattr(self.mrl.agent, "last_entropy", None)
        if ent is None or not math.isfinite(ent):
            return 0.6
        ent = float(np.clip(ent, 0.0, 5.0))
        return 1.0 - ent / 5.0

# ========== 4. ActiveTradeMonitor → TradeMonitorVetoExpert ==========
class TradeMonitorVetoExpert:
    """
    If monitor alerts, it outputs all-zero sizes (i.e., veto);
    otherwise neutral with low confidence.
    """
    def __init__(self, monitor: "ActiveTradeMonitor", env_ref, debug=True):
        self.mon  = monitor
        self.env  = env_ref
        self._zero = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug

    def reset(self):
        pass

    def propose_action(self, obs: Any, extras: Dict = None) -> np.ndarray:
        return self._zero.copy()

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        return 0.2 if self.mon.alerted else 0.6

# ========== 5. FractalRegimeConfirmation → RegimeBiasExpert ==========
class RegimeBiasExpert:
    """
    Long-bias in trending regime, flat in noise, short-bias in volatile.
    """
    def __init__(self, frc_module, env_ref, max_size=0.7, debug=True):
        self.frc  = frc_module
        self.env  = env_ref
        self.max_size = max_size
        self._zero = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug

    def reset(self):
        pass

    def propose_action(self, obs: Any, extras: Dict = None) -> np.ndarray:
        label = self.frc.label
        strength = float(np.clip(self.frc.regime_strength, 0.0, 1.0))
        if label == "trending":
            direction = 1
        elif label == "volatile":
            direction = -1
        else:
            return self._zero.copy()
        vec = self._zero.copy()
        vec[0] = direction * strength * self.max_size
        vec[1] = 0.5
        return vec

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        return 0.5 + 0.5 * float(abs(self.frc.regime_strength))

# ========== Example: Add get_state/set_state for stateful experts ==========
# For any of the above, add:
#   def get_state(self): ...
#   def set_state(self, state): ...

