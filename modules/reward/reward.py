#modules/reward.py

import numpy as np
from typing import List
from ..core.core import Module

class RiskAdjustedReward(Module):
    def __init__(self, initial_balance: float, env=None, debug=False):
        self.initial_balance = initial_balance
        self.env   = env
        self.debug = debug
        self.regime_weights = np.array([0.5, 0.3, 0.2], np.float32)

    def reset(self): ...
    def step(self, **kwargs): ...

    def calculate(
        self,
        current_balance: float,
        trades: List[dict],
        current_drawdown: float,
        regime_onehot: np.ndarray,
        actions: np.ndarray
    ) -> float:

        # ---------- 0) no‑trade → small exploration penalty -------------- #
        if not trades:
            return -0.1 * float(np.sqrt((actions**2).mean()))

        # ---------- 1) basic PnL & tail risk ----------------------------- #
        pnl  = sum(t["pnl"] for t in trades)
        rets = np.array([t["pnl"] for t in trades], np.float32)
        var  = np.percentile(rets, 5)
        cvar = rets[rets <= var].mean() if np.any(rets <= var) else var
        tail_pen = -0.5 * var - 0.5 * cvar

        # ---------- 2) draw‑down & risk penalties ------------------------ #
        dd_pen   = current_drawdown * 10.0
        risk_pen = float(np.sqrt((actions**2).mean())) * 0.3

        # ---------- 3) regime bonus -------------------------------------- #
        regime_bonus = float(regime_onehot.dot(self.regime_weights))

        # ---------- 4) MistakeMemory proximity penalty ------------------- #
        mm_pen = 0.0
        if self.env is not None and hasattr(self.env, "mistake_memory"):
            mm_pen = float(
                self.env.mistake_memory.get_observation_components()[0]
            )

        reward = pnl - dd_pen - risk_pen - tail_pen - 0.5 * mm_pen + regime_bonus
        return float(reward)

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, np.float32)