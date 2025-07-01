# ─────────────────────────────────────────────────────────────
# File: modules/market/regime_performance_matrix.py
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
from typing import Dict
from ..core.core import Module

class RegimePerformanceMatrix(Module):

    def __init__(self, n_regimes: int = 3, decay: float = 0.95, debug: bool = True):
        self.n = n_regimes
        self.decay = decay
        self.debug = debug

        self.vol_history = deque(maxlen=500)
        self.volatility_regimes = np.array([0.1, 0.3, 0.5], np.float32)
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.n, self.n), np.float32)
        self.last_volatility = 0.0
        self.last_liquidity = 1.0
        self.vol_history.clear()

    def step(self, **kwargs):
        if not all(k in kwargs for k in ("pnl", "volatility", "predicted_regime")):
            return  # skip if any required input is missing

        pnl = kwargs["pnl"]
        volatility = kwargs["volatility"]
        predicted_regime = kwargs["predicted_regime"]

        self.vol_history.append(volatility)

        if len(self.vol_history) >= 20:
            self.volatility_regimes = np.quantile(
                np.asarray(self.vol_history), [0.25, 0.5, 0.75]
            ).astype(np.float32)

        true_reg = min(
            int(np.digitize(volatility, self.volatility_regimes)), self.n - 1
        )

        self.matrix *= self.decay
        self.matrix[true_reg, predicted_regime] += pnl

        self.last_volatility = volatility
        if self.debug:
            print(
                f"[RPM] true={true_reg} pred={predicted_regime} "
                f"pnl={pnl:+.2f} thr={self.volatility_regimes.round(4)}"
            )

    def stress_test(
        self,
        scenario: str,
        volatility: float | None = None,
        liquidity_score: float | None = None,
    ) -> Dict[str, float]:
        crisis = {
            "flash-crash": {"vol_mult": 3.0, "liq_mult": 0.2},
            "rate-spike": {"vol_mult": 2.5, "liq_mult": 0.5},
            "default-wave": {"vol_mult": 2.0, "liq_mult": 0.3},
        }.get(scenario, {"vol_mult": 1.0, "liq_mult": 1.0})

        vol = volatility if volatility is not None else self.last_volatility
        liq = liquidity_score if liquidity_score is not None else self.last_liquidity
        
        return {
            "volatility": vol * crisis["vol_mult"], 
            "liquidity": liq * crisis["liq_mult"]
        }

    def get_observation_components(self) -> np.ndarray:
        flat = self.matrix.flatten()
        acc = []
        for i in range(self.n):
            row_sum = self.matrix[i].sum()
            if row_sum > 1e-4:
                acc.append(float(self.matrix[i, i] / row_sum))
            else:
                acc.append(0.0)
        return np.concatenate([flat, np.asarray(acc, np.float32)])
    
    def get_state(self):
        return {
            "matrix": self.matrix.tolist(),
            "vol_history": list(self.vol_history),
            "volatility_regimes": self.volatility_regimes.tolist(),
            "last_volatility": float(self.last_volatility),
            "last_liquidity": float(self.last_liquidity),
        }
        
    def set_state(self, state):
        self.matrix = np.array(state.get("matrix", np.zeros((self.n, self.n))), dtype=np.float32)
        self.vol_history = deque(state.get("vol_history", []), maxlen=500)
        self.volatility_regimes = np.array(state.get("volatility_regimes", [0.1, 0.3, 0.5]), dtype=np.float32)
        self.last_volatility = float(state.get("last_volatility", 0.0))
        self.last_liquidity = float(state.get("last_liquidity", 1.0))