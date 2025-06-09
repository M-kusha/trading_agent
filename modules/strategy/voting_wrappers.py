# modules/strategy/voting_wrappers.py
import numpy as np
from typing import Any, List, Dict
import math

import torch

from modules.risk.risk_monitor import ActiveTradeMonitor
from modules.strategy.strategy import MetaRLController

# ------------------------------------------------------------
# Helper for angle → position allocation (size channel only)
# ------------------------------------------------------------
def _dir_size_to_vec(angle: float,
                     magnitude: float,
                     n_instruments: int,
                     action_dim: int) -> np.ndarray:
    """
    Map a polar angle (0 = long first instrument) and magnitude [0,1]
    into a full (size,duration)*N action vector.
    Duration channel is left at 0.5 (neutral).
    """
    vec = np.zeros(action_dim, np.float32)
    if magnitude <= 0:
        return vec

    # Simple sector mapping
    idx = int(round((angle % 360) / 360 * n_instruments)) % n_instruments
    vec[2 * idx] = magnitude          # size component
    vec[2 * idx + 1] = 0.5            # hold-time neutral
    return vec


# ============================================================
# 1. MarketThemeDetector → ThemeExpert
# ============================================================
class ThemeExpert:
    """
    Converts (label,strength) from MarketThemeDetector into an action.
    Long on 'up-trend', flat otherwise.
    """
    def __init__(self,
                 detector,                # MarketThemeDetector
                 env_ref,                 # EnhancedTradingEnv
                 trend_label: str = "trending",
                 max_size: float = 1.0):
        self.det     = detector
        self.env     = env_ref
        self.trend_label = trend_label
        self.max_size = max_size
        self._last_strength = 0.0
        self._zero = np.zeros(self.env.action_dim, np.float32)

    # -------- voting interface -------------------------------
    def propose_action(self, obs: Any) -> np.ndarray:
        # Ask the detector for the latest label/strength on *this* step
        try:
            lab, stren = self.det.detect(self.env.data, self.env.current_step)
        except Exception:
            lab, stren = "none", 0.0
        self._last_strength = float(np.clip(stren, 0.0, 1.0))

        if lab != self.trend_label:
            return self._zero.copy()

        vec = self._zero.copy()
        vec[0] = self._last_strength * self.max_size
        vec[1] = 0.5
        return vec

    def confidence(self, obs: Any) -> float:
        # Use strength as 0-1 confidence, but keep ≥0.5 neutral baseline
        return 0.5 + 0.5 * self._last_strength


# ============================================================
# 2. TimeAwareRiskScaling → SeasonalityRiskExpert
# ============================================================
class SeasonalityRiskExpert:
    """
    Scales *everyone’s* risk: outputs a vector that multiplies baseline
    size by seasonality_factor – no opinion on direction.
    """
    def __init__(self, tars_module, env_ref):
        self.tars = tars_module
        self.env  = env_ref
        self._zero = np.zeros(self.env.action_dim, np.float32)

    def propose_action(self, obs: Any) -> np.ndarray:
        f = float(self.tars.seasonality_factor)
        # Encode only a size-scaling suggestion (duration untouched)
        vec = self._zero.copy()
        if not math.isfinite(f):
            f = 1.0
        vec[:] = 0.0
        # put the scaling factor in *every* size slot
        for i in range(0, self.env.action_dim, 2):
            vec[i] = f - 1.0     # >0 increases risk, <0 reduces
        return vec

    def confidence(self, obs: Any) -> float:
        # More extreme scaling  ⇒ higher confidence
        f = float(self.tars.seasonality_factor)
        return float(min(1.0, 0.5 + abs(f - 1.0)))


# ============================================================
# 3. MetaRLController → MetaRLExpert
# ============================================================
class MetaRLExpert:
    """
    Uses the Meta-RL policy’s own action; confidence ~ (1 − entropy).
    """
    def __init__(self, meta_rl: "MetaRLController", env_ref):
        self.mrl  = meta_rl
        self.env  = env_ref
        self.last_action = np.zeros(self.env.action_dim, np.float32)

    def _call_policy(self, obs_vec: np.ndarray) -> np.ndarray:
        """Run policy → flatten to (action_dim,) NumPy float32."""
        obs_t = torch.tensor(obs_vec, dtype=torch.float32,
                             device=self.mrl.device).unsqueeze(0)   # (1, obs_dim)

        raw = self.mrl.act(obs_t)           # could be tensor, list, or dict
        if isinstance(raw, dict):           # e.g. {'action': …}
            raw = raw.get("action", raw)

        act = np.asarray(raw, dtype=np.float32).reshape(-1)          # flatten
        if act.size < self.env.action_dim:                           # pad if short
            act = np.pad(act, (0, self.env.action_dim - act.size))
        return np.clip(act[: self.env.action_dim], -1.0, 1.0)

    # ---- voting interface ------------------------------------------
    def propose_action(self, obs: np.ndarray) -> np.ndarray:
        self.last_action = self._call_policy(obs)
        return self.last_action.copy()        # (action_dim,)

    def confidence(self, obs: Any) -> float:
        ent = getattr(self.mrl.agent, "last_entropy", None)
        if ent is None or not math.isfinite(ent):
            return 0.6                       # neutral if unknown
        ent = float(np.clip(ent, 0.0, 5.0))
        return 1.0 - ent / 5.0               # 0 → low confidence; 1 → high

# ============================================================
# 4. ActiveTradeMonitor → TradeMonitorVetoExpert
# ============================================================
class TradeMonitorVetoExpert:
    """
    If monitor alerts, it outputs all-zero sizes (i.e., veto);
    otherwise neutral with low confidence.
    """
    def __init__(self, monitor: "ActiveTradeMonitor", env_ref):
        self.mon  = monitor
        self.env  = env_ref
        self._zero = np.zeros(self.env.action_dim, np.float32)

    def propose_action(self, obs: Any) -> np.ndarray:
        return self._zero.copy()   # no direct proposal

    def confidence(self, obs: Any) -> float:
        return 0.2 if self.mon.alerted else 0.6


# ============================================================
# 5. FractalRegimeConfirmation → RegimeBiasExpert
# ============================================================
class RegimeBiasExpert:
    """
    Long-bias in trending regime, flat in noise, short-bias in volatile.
    """
    def __init__(self, frc_module, env_ref, max_size=0.7):
        self.frc  = frc_module
        self.env  = env_ref
        self.max_size = max_size
        self._zero = np.zeros(self.env.action_dim, np.float32)

    def propose_action(self, obs: Any) -> np.ndarray:
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

    def confidence(self, obs: Any) -> float:
        return 0.5 + 0.5 * float(abs(self.frc.regime_strength))
