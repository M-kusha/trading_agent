import numpy as np
from typing import Dict, List, Union, Optional, Any
from collections import deque
from modules.core.core import Module
import random

class PortfolioRiskSystem(Module):
    """
    Per-instrument max-position limits from VaR + drawdown,
    correlation-adjusted for a multi-asset basket.
    Now with evolutionary logic: all limits/params can be evolved!
    """

    def __init__(
        self,
        var_window: int = 50,
        dd_limit: float = 0.20,
        instruments: Optional[List[str]] = None,
        risk_mult: float = 3.5,
        debug: bool = False,
    ):
        self.var_window = int(var_window)
        self.dd_limit = float(dd_limit)
        self.instruments = instruments or []
        self.risk_mult = float(risk_mult)
        self.debug = debug
        self.returns_history: deque[Union[float, np.ndarray]] = deque(
            maxlen=self.var_window
        )
        # Store risk config for evolutionary ops
        self.risk_config = {
            "var_window": self.var_window,
            "dd_limit": self.dd_limit,
            "risk_mult": self.risk_mult,
            "instruments": self.instruments.copy(),
        }

    # ================= Evolutionary ops ==================== #
    def mutate(self, std=0.1):
        # Mutate risk-related parameters
        self.var_window = int(np.clip(self.var_window + np.random.normal(0, std * 20), 10, 500))
        self.dd_limit = float(np.clip(self.dd_limit + np.random.normal(0, std * 0.05), 0.01, 0.99))
        self.risk_mult = float(np.clip(self.risk_mult + np.random.normal(0, std * 2), 0.1, 10.0))
        # Randomly drop/add an instrument (rare)
        if self.instruments and np.random.rand() < 0.1:
            if np.random.rand() < 0.5 and len(self.instruments) > 1:
                # Remove a random instrument
                to_remove = random.choice(self.instruments)
                self.instruments.remove(to_remove)
            else:
                # Add a synthetic instrument (demo, real would check env)
                to_add = f"SYNTH_{random.randint(0,99)}"
                if to_add not in self.instruments:
                    self.instruments.append(to_add)
        # Save new config
        self.risk_config = {
            "var_window": self.var_window,
            "dd_limit": self.dd_limit,
            "risk_mult": self.risk_mult,
            "instruments": self.instruments.copy(),
        }

    def crossover(self, other: "PortfolioRiskSystem"):
        # Blend parameters and union instrument lists
        new_var_window = int(self.var_window if np.random.rand() > 0.5 else other.var_window)
        new_dd_limit = float(self.dd_limit if np.random.rand() > 0.5 else other.dd_limit)
        new_risk_mult = float(self.risk_mult if np.random.rand() > 0.5 else other.risk_mult)
        new_instruments = list(set(self.instruments) | set(other.instruments))
        child = PortfolioRiskSystem(
            var_window=new_var_window,
            dd_limit=new_dd_limit,
            instruments=new_instruments,
            risk_mult=new_risk_mult,
            debug=self.debug or other.debug,
        )
        return child

    def get_params(self) -> Dict[str, Any]:
        return dict(self.risk_config)

    # ================= Pipeline core ==================== #
    def reset(self):
        self.returns_history.clear()

    def step(
        self,
        pnl: Optional[float] = None,
        returns: Optional[Dict[str, float]] = None,
        **__,
    ):
        if returns is not None and self.instruments:
            vec = np.array([returns.get(i, 0.0) for i in self.instruments], np.float32)
            self.returns_history.append(vec)
        elif pnl is not None:
            self.returns_history.append(float(pnl))

    def get_limits(self, balance: float) -> Dict[str, float]:
        if not self.returns_history:
            cap = balance * self.dd_limit
            return (
                {inst: cap for inst in self.instruments}
                if self.instruments
                else {"max_position_size": cap}
            )

        sample = self.returns_history[0]

        # -------- Multi-asset branch ------------- #
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

        # -------- Scalar VaR branch ------------- #
        sorted_rets = sorted(self.returns_history)
        idx = int(0.05 * len(sorted_rets))
        var95 = -sorted_rets[idx] if sorted_rets[idx] < 0 else 0.0

        cap_var = var95 * self.risk_mult
        cap_dd = balance * self.dd_limit
        max_pos = min(cap_var, cap_dd)

        if self.debug:
            print(f"[RiskSys] VaR={var95:.5f} cap={max_pos:.2f}")
        return {"max_position_size": float(max_pos)}

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(len(self.instruments) or 1, np.float32)

    def get_state(self):
        return dict(self.risk_config)

    def set_state(self, state):
        self.var_window = int(state.get("var_window", self.var_window))
        self.dd_limit = float(state.get("dd_limit", self.dd_limit))
        self.risk_mult = float(state.get("risk_mult", self.risk_mult))
        self.instruments = list(state.get("instruments", self.instruments))
        self.risk_config = dict(state)
