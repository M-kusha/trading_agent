import numpy as np
import logging
import random
import json
import os
import datetime
from typing import Dict, List, Union, Optional, Any
from collections import deque
from modules.core.core import Module

class PortfolioRiskSystem(Module):
    """
    Per-instrument max-position limits from VaR + drawdown,
    correlation-adjusted for a multi-asset basket.
    Adds a structured, explainable audit trail for every limit decision.
    """

    def __init__(
        self,
        var_window: int = 50,
        dd_limit: float = 0.20,
        instruments: Optional[List[str]] = None,
        risk_mult: float = 3.5,
        debug: bool = True,
        audit_log_path: str = "logs/risk/portfolio_risk_audit.jsonl",
    ):
        # ── logger setup ─────────────────────────────────────────────
        self.logger = logging.getLogger("PortfolioRiskSystem")
        if not self.logger.handlers:
            h = logging.FileHandler("logs/risk/portfolio_risk.log")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        # ── core config ──────────────────────────────────────────────
        self.var_window = int(var_window)
        self.dd_limit = float(dd_limit)
        self.instruments = instruments or []
        self.risk_mult = float(risk_mult)
        self.debug = debug

        self.returns_history: deque[Union[float, np.ndarray]] = deque(
            maxlen=self.var_window
        )
        self.risk_config = {
            "var_window": self.var_window,
            "dd_limit": self.dd_limit,
            "risk_mult": self.risk_mult,
            "instruments": self.instruments.copy(),
        }
        self.audit_log_path = audit_log_path
        self.last_audit: Dict[str, Any] = {}

        # ensure audit directory exists
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
        self.logger.info(f"Initialized PortfolioRiskSystem | var_window={self.var_window} "
                         f"dd_limit={self.dd_limit:.2f} risk_mult={self.risk_mult:.2f}")
        





    def prime_returns_with_random(self):
        """
        Fills returns_history with small neutral random values.
        Handles both scalar and vector return modes.
        """
        self.returns_history.clear()
        for _ in range(self.var_window):
            if self.instruments:
                vec = np.random.normal(0, 0.0002, len(self.instruments)).astype(np.float32)
                self.returns_history.append(vec)
            else:
                val = float(np.random.normal(0, 0.0002))
                self.returns_history.append(val)
        self.logger.info(f"[PRIME] returns_history filled with neutral randoms ({self.var_window} samples)")

    def prime_returns_with_history(self, price_dict: dict):
        """
        Fills returns_history with real log returns from prices in price_dict.
        price_dict: {instrument: np.ndarray or list of close prices}
        """
        self.returns_history.clear()
        for i in range(self.var_window):
            returns = []
            for inst in self.instruments:
                prices = np.asarray(price_dict.get(inst, []))
                if prices.size > i + 1:
                    r = np.log(prices[i + 1] / prices[i])
                else:
                    r = 0.0
                returns.append(r)
            self.returns_history.append(np.array(returns, dtype=np.float32))
        self.logger.info(f"[PRIME] returns_history filled with {self.var_window} real log-return vectors")


    # ================= Evolutionary ops ==================== #
    def mutate(self, std=0.1):
        old = (self.var_window, self.dd_limit, self.risk_mult, list(self.instruments))
        # adjust parameters
        self.var_window = int(np.clip(self.var_window + np.random.normal(0, std * 20), 10, 500))
        self.dd_limit = float(np.clip(self.dd_limit + np.random.normal(0, std * 0.05), 0.01, 0.99))
        self.risk_mult = float(np.clip(self.risk_mult + np.random.normal(0, std * 2), 0.1, 10.0))
        # occasionally tweak instrument list
        if self.instruments and np.random.rand() < 0.1:
            if np.random.rand() < 0.5 and len(self.instruments) > 1:
                to_remove = random.choice(self.instruments)
                self.instruments.remove(to_remove)
            else:
                to_add = f"SYNTH_{random.randint(0,99)}"
                if to_add not in self.instruments:
                    self.instruments.append(to_add)
        self.risk_config = {
            "var_window": self.var_window,
            "dd_limit": self.dd_limit,
            "risk_mult": self.risk_mult,
            "instruments": self.instruments.copy(),
        }
        self.logger.info(f"Mutate called | old={old} new="
                         f"{(self.var_window, self.dd_limit, self.risk_mult, self.instruments)}")

    def crossover(self, other: "PortfolioRiskSystem"):
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
            audit_log_path=self.audit_log_path,
        )
        self.logger.info("Crossover produced child with config: "
                         f"{child.get_params()}")
        return child

    def get_params(self) -> Dict[str, Any]:
        return dict(self.risk_config)

    def reset(self):
        self.returns_history.clear()
        self.logger.debug("Reset returns history")

    def step(
        self,
        pnl: Optional[float] = None,
        returns: Optional[Dict[str, float]] = None,
        **__,
    ):
        if returns is not None and self.instruments:
            vec = np.array([returns.get(i, 0.0) for i in self.instruments], np.float32)
            self.returns_history.append(vec)
            self.logger.debug(f"Appended returns vector: {vec.tolist()}")
        elif pnl is not None:
            self.returns_history.append(float(pnl))
            self.logger.debug(f"Appended PnL: {pnl:.2f}")

    def get_limits(self, balance: float) -> Dict[str, float]:
        self.logger.info(f"Computing limits | balance={balance:.2f} "
                         f"| history_length={len(self.returns_history)}")

        rationale = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "params": dict(self.risk_config),
            "balance": balance,
            "returns_window": self.var_window,
            "method": None,
            "rationale": {},
        }

        # ── no history  drawdown‐only cap
        if not self.returns_history:
            cap = balance * self.dd_limit
            rationale["method"] = "dd_only"
            rationale["rationale"] = {
                "reason": "No return history; drawdown cap only.",
                "dd_limit": self.dd_limit,
                "limit_per_instrument": cap,
            }
            self.logger.info(f"  DD-only cap: {cap:.2f} per instrument")
            self._audit_and_return(rationale, self.instruments, cap)
            return {inst: cap for inst in self.instruments} if self.instruments else {"max_position_size": cap}

        sample = self.returns_history[0]

        # ── multi‐asset VaR + correlation
        if isinstance(sample, np.ndarray):
            R = np.stack(self.returns_history, axis=0)  # (T,N)
            corr = np.nan_to_num(np.corrcoef(R, rowvar=False))  # (N,N)
            N = corr.shape[0]
            w = np.ones(N, np.float32) / N
            port_var = float(w @ corr @ w)
            port_var = np.clip(port_var, 1e-3, None)
            total_risk = balance * self.dd_limit
            limits = total_risk * (w / np.sqrt(port_var))  # use σ not σ²

            rationale["method"] = "multi_asset"
            rationale["rationale"] = {
                "reason": "Multi-asset VaR+correlation.",
                "dd_limit": self.dd_limit,
                "weights": w.tolist(),
                "portfolio_variance": port_var,
                "correlation_matrix": corr.tolist(),
                "total_risk_cap": total_risk,
                "limits": dict(zip(self.instruments, limits.tolist())),
            }
            self.logger.info(f"  Multi-asset VaR branch: var_window={self.var_window}, "
                             f"portfolio_variance={port_var:.4f}")
            self._audit_and_return(rationale, self.instruments, limits)
            return dict(zip(self.instruments, limits.tolist()))

        # ── scalar VaR + drawdown
        sorted_rets = sorted(self.returns_history)
        idx = int(0.05 * len(sorted_rets))
        var95 = -sorted_rets[idx] if sorted_rets[idx] < 0 else 0.0
        cap_var = var95 * self.risk_mult
        cap_dd = balance * self.dd_limit
        max_pos = min(cap_var, cap_dd)

        rationale["method"] = "scalar_VaR"
        rationale["rationale"] = {
            "reason": "Univariate VaR+drawdown.",
            "var95": var95,
            "risk_mult": self.risk_mult,
            "var_limit": cap_var,
            "dd_limit": self.dd_limit,
            "dd_limit_abs": cap_dd,
            "max_position": max_pos,
        }
        self.logger.info(f"  Scalar-VaR branch: var95={var95:.4f}, "
                         f"cap_var={cap_var:.4f}, cap_dd={cap_dd:.2f}")
        self._audit_and_return(rationale, ["max_position_size"], max_pos)
        return {"max_position_size": float(max_pos)}

    def _audit_and_return(self, rationale, keys, values):
        if isinstance(values, np.ndarray):
            values = values.tolist()
        rationale["decision"] = dict(zip(keys,
                                        values if isinstance(values, list) else [values]))
        self.last_audit = rationale
        # write JSON‐line
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(rationale) + "\n")
        if self.debug:
            self.logger.debug(f"[AUDIT] {json.dumps(rationale)}")

    def get_last_audit(self) -> Dict[str, Any]:
        return self.last_audit.copy()

    def get_audit_log(self, n: int = 50) -> List[Dict[str, Any]]:
        if not os.path.exists(self.audit_log_path):
            return []
        with open(self.audit_log_path, "r") as f:
            lines = f.readlines()[-n:]
        return [json.loads(l) for l in lines]

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
