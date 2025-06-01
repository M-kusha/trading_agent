from __future__ import annotations
from typing import Any, List, Set
import numpy as np
import random
from modules.utils.info_bus import InfoBus
from modules.core.core import Module

class ComplianceModule(Module):
    """
    Evolutionary, rule-based compliance check.  
    – max_leverage: maximum allowed leverage (default 20×)  
    – max_single_position_risk: the maximum % of equity that a single trade may consume (now default 10%)  
    – prohibited_symbols: any symbols that are never allowed
    """

    def __init__(
        self, 
        max_leverage: float = 20.0, 
        max_single_position_risk: float = 0.10,  # raised from 0.02 up to 0.10
        prohibited_symbols: Set[str] = None
    ):
        super().__init__()

        # 1) maximum allowable leverage (e.g. 20×)  
        self.max_leverage = float(max_leverage)

        # 2) single‐position risk is now 10% of equity (was 2%)  
        self.max_single_position_risk = float(max_single_position_risk)

        # 3) which symbols are outright forbidden  
        self.prohibited_symbols = set(prohibited_symbols) if prohibited_symbols else {"RUB", "TRY"}

        # we’ll accumulate any “flag” messages here  
        self.last_flags: List[str] = []

        # set up a logger so we can debug exactly which rule is failing, if any  
        self.logger = self._get_logger()

    def _get_logger(self):
        import logging
        lg = logging.getLogger("ComplianceModule")
        if not lg.handlers:
            h = logging.FileHandler("compliance.log")
            h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            lg.addHandler(h)
        lg.setLevel(logging.DEBUG)
        return lg

    # ──────────────────────────────────────────────────────────────────────────
    # Evolutionary operators (unchanged)  
    # ──────────────────────────────────────────────────────────────────────────
    def mutate(self, std=0.1):
        # Slightly nudge leverage / risk thresholds
        self.max_leverage += np.random.normal(0, std * 5)
        self.max_leverage = float(np.clip(self.max_leverage, 1.0, 100.0))

        self.max_single_position_risk += np.random.normal(0, std * 0.02)
        # Clamp to [0.001, 0.5] so risk % stays reasonable
        self.max_single_position_risk = float(
            np.clip(self.max_single_position_risk, 0.001, 0.5)
        )

        # Occasionally add or remove one “prohibited” symbol
        if np.random.rand() < 0.1:
            choices = ["RUB", "TRY", "BTC", "XAU", "USD", "CNY", "INR", "ZAR", "BRL"]
            if np.random.rand() < 0.5 and self.prohibited_symbols:
                to_remove = random.choice(list(self.prohibited_symbols))
                self.prohibited_symbols.discard(to_remove)
            else:
                to_add = random.choice(choices)
                self.prohibited_symbols.add(to_add)

    def crossover(self, other: "ComplianceModule"):
        # Blend hyperparameters and merge prohibited sets
        child = ComplianceModule(
            max_leverage=(
                self.max_leverage
                if np.random.rand() > 0.5
                else other.max_leverage
            ),
            max_single_position_risk=(
                self.max_single_position_risk
                if np.random.rand() > 0.5
                else other.max_single_position_risk
            ),
            prohibited_symbols=(
                self.prohibited_symbols | other.prohibited_symbols
                if np.random.rand() > 0.5
                else self.prohibited_symbols & other.prohibited_symbols
            )
        )
        return child

    def get_params(self):
        return {
            "max_leverage": self.max_leverage,
            "max_single_position_risk": self.max_single_position_risk,
            "prohibited_symbols": list(self.prohibited_symbols),
        }

    def reset(self) -> None:
        self.last_flags.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # This is the core logic that checks a single trade’s inputs  
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, **data: Any) -> bool:
        """
        Run through our rule checks.  If any rule “flags,” return False.  
        Otherwise return True.
        """
        info = InfoBus(**data)
        self.last_flags.clear()

        symbol        = info.get("extras", {}).get("symbol")
        current_price = info.get("current_price", 0.0)
        risk          = info.get("risk", {})
        raw_action    = info.get("raw_action")
        extras        = info.get("extras", {})

        # ——————————————————————————————————————————————————————
        # 1) Prohibited‐symbol check
        # ——————————————————————————————————————————————————————
        if symbol in self.prohibited_symbols:
            msg = f"– Compliance: trading {symbol} is prohibited."
            self.last_flags.append(msg)
            self.logger.debug(msg)

        # ——————————————————————————————————————————————————————
        # 2) Single‐position‐risk check (now uses 10% of equity)
        # ——————————————————————————————————————————————————————
        if (raw_action is not None) and ("size" in extras):
            size = extras["size"]  # absolute lots
            position_value = size * current_price
            equity = risk.get("equity", 1e-8)

            # If position_value/equity exceeds 10% → flag it
            fraction_risk = position_value / max(equity, 1e-8)
            if fraction_risk > self.max_single_position_risk:
                msg = (
                    f"– Compliance: single trade risk "
                    f"{fraction_risk:.2%} exceeds "
                    f"{self.max_single_position_risk:.0%} limit."
                )
                self.last_flags.append(msg)
                self.logger.debug(msg)

        # ——————————————————————————————————————————————————————
        # 3) Leverage check
        # ——————————————————————————————————————————————————————
        lev = risk.get("margin_used", 0.0) / max(risk.get("equity", 1e-8), 1e-9)
        if lev > self.max_leverage:
            msg = (
                f"– Compliance: leverage {lev:.1f}× "
                f"exceeded {self.max_leverage:.1f}× limit."
            )
            self.last_flags.append(msg)
            self.logger.debug(msg)

        # If we have collected any flags → reject the trade
        if self.last_flags:
            # push flagged messages back into the InfoBus so caller can see them
            info.setdefault("compliance_flags", []).extend(self.last_flags)
            return False

        # No flags = valid
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # validate_trade is called from the environment.  We reconstruct exactly
    # what `step(...)` wants to see:
    # ──────────────────────────────────────────────────────────────────────────
    def validate_trade(self, trade: dict, env) -> bool:
        """
        Build a small ‘info‐dict’ from `trade` + `env` and pass it to `step(...)`.
        """

        symbol = trade.get("instrument")
        size   = abs(trade.get("size", 0.0))
        df = env.data.get(symbol, {}).get("D1")
        if (df is None) or (env.current_step >= len(df)):
            return False

        current_price = float(df.iloc[env.current_step]["close"])
        equity        = float(env.balance)
        margin_used   = size * current_price  # assume 1:1 margin = notional

        data = {
            "extras": {
                "symbol": symbol,
                "size":   size
            },
            "current_price": current_price,
            "risk": {
                "equity":      equity,
                "margin_used": margin_used
            },
            "raw_action": trade.get("size"),
        }

        return self.step(**data)

    def get_observation_components(self) -> np.ndarray:
        # Compliance does not add to the observation vector
        return np.zeros(0, np.float32)

    def get_state(self):
        return {
            "max_leverage":             self.max_leverage,
            "max_single_position_risk": self.max_single_position_risk,
            "prohibited_symbols":       list(self.prohibited_symbols)
        }

    def set_state(self, state):
        self.max_leverage = float(state.get("max_leverage", self.max_leverage))
        self.max_single_position_risk = float(
            state.get("max_single_position_risk", self.max_single_position_risk)
        )
        self.prohibited_symbols = set(state.get("prohibited_symbols", self.prohibited_symbols))
