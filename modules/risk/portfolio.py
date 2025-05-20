# modules/portfolio.py
import numpy as np
from typing import Dict, List, Union
from collections import deque
from modules.core.core import Module

class PortfolioRiskSystem(Module):
    """
    Per‑instrument max‑position limits from VaR + draw‑down,
    correlation‑adjusted for a multi‑asset basket.
    """

    def __init__(
        self,
        var_window: int = 50,
        dd_limit: float = 0.20,
        instruments: List[str] | None = None,
        risk_mult: float = 3.5,
        debug: bool = False,
    ):
        self.var_window = var_window
        self.dd_limit = dd_limit
        self.instruments = instruments or []
        self.risk_mult = risk_mult
        self.debug = debug
        self.returns_history: deque[Union[float, np.ndarray]] = deque(
            maxlen=var_window
        )

    # ------------------------------------------------------------------ #
    def reset(self):
        self.returns_history.clear()

    def step(
        self,
        pnl: float | None = None,
        returns: Dict[str, float] | None = None,
        **__,
    ):
        if returns is not None and self.instruments:
            vec = np.array([returns.get(i, 0.0) for i in self.instruments], np.float32)
            self.returns_history.append(vec)
        elif pnl is not None:
            self.returns_history.append(float(pnl))

    # ------------------------------------------------------------------ #
    def get_limits(self, balance: float) -> Dict[str, float]:
        if not self.returns_history:
            cap = balance * self.dd_limit
            return (
                {inst: cap for inst in self.instruments}
                if self.instruments
                else {"max_position_size": cap}
            )

        sample = self.returns_history[0]

        # -------- Multi‑asset branch ----------------------------------- #
        if isinstance(sample, np.ndarray):
            R = np.stack(self.returns_history, axis=0)  # (T,N)
            corr = np.nan_to_num(np.corrcoef(R, rowvar=False))  # (N,N)
            N = corr.shape[0]
            w = np.ones(N, np.float32) / N
            port_var = float(w @ corr @ w)
            port_var = np.clip(port_var, 1e-3, None)

            total_risk = balance * self.dd_limit
            limits = total_risk * (w / np.sqrt(port_var))  # use σ not σ²

            if self.debug:
                print(f"[RiskSys] σ={port_var**0.5:.5f} limits={limits}")
            return dict(zip(self.instruments, limits.tolist()))

        # -------- Scalar VaR branch ------------------------------------ #
        sorted_rets = sorted(self.returns_history)
        idx = int(0.05 * len(sorted_rets))
        var95 = -sorted_rets[idx] if sorted_rets[idx] < 0 else 0.0

        cap_var = var95 * self.risk_mult
        cap_dd = balance * self.dd_limit
        max_pos = min(cap_var, cap_dd)

        if self.debug:
            print(f"[RiskSys] VaR={var95:.5f} cap={max_pos:.2f}")
        return {"max_position_size": float(max_pos)}

    # ------------------------------------------------------------------ #
    def get_observation_components(self) -> np.ndarray:
        return np.zeros(len(self.instruments) or 1, np.float32)