# modules/compliance_module.py
from __future__ import annotations
from typing import Any, List
import numpy as np
from modules.utils.info_bus import InfoBus
from modules.core.core import Module

class ComplianceModule(Module):
    """Hard-stop rule-based compliance check.

    Fails the action if leverage or risk exceeds pre-defined limits.
    """

    MAX_LEVERAGE = 20
    MAX_SINGLE_POSITION_RISK = 0.02  # 2 % of equity
    PROHIBITED_SYMBOLS = {"RUB", "TRY"}

    def __init__(self) -> None:
        super().__init__()
        # collect any violation messages here
        self.last_flags: List[str] = []

    def reset(self) -> None:
        """No internal state to reset beyond clearing flags."""
        self.last_flags.clear()

    def step(self, **data: Any) -> bool:
        """
        Accepts whatever the pipeline hands you, wraps it in InfoBus,
        and then applies your old logic.
        """
        info = InfoBus(**data)
        # start each call with a fresh flag list
        self.last_flags.clear()

        symbol        = info.get("extras", {}).get("symbol")
        current_price = info.get("current_price")
        risk          = info.get("risk", {})
        raw_action    = info.get("raw_action")
        extras        = info.get("extras", {})

        # 1) Prohibited symbols
        if symbol in self.PROHIBITED_SYMBOLS:
            self._flag(f"Trading {symbol} is prohibited.")

        # 2) Single-position risk
        if raw_action is not None and "size" in extras:
            size = extras["size"]
            position_value = size * current_price
            if position_value / risk.get("equity", 1e-8) > self.MAX_SINGLE_POSITION_RISK:
                self._flag(
                    f"Single trade risk {position_value / risk['equity']:.2%} "
                    f"exceeds {self.MAX_SINGLE_POSITION_RISK:.0%}"
                )

        # 3) Leverage check
        lev = risk.get("margin_used", 0.0) / max(risk.get("equity", 1e-8), 1e-9)
        if lev > self.MAX_LEVERAGE:
            self._flag(f"Leverage {lev:.1f}× exceeded {self.MAX_LEVERAGE}× limit.")

        # if any flags were raised, write them back into the bus and fail
        if self.last_flags:
            info.setdefault("compliance_flags", []).extend(self.last_flags)
            return False

        return True

    def validate_trade(self, trade: dict, env) -> bool:
        """
        Build the InfoBus inputs from the env + trade dict, then call step().
        Returns True if the trade passes all compliance rules.
        """
        symbol = trade.get("instrument")
        size   = abs(trade.get("size", 0.0))

        # lookup current price
        df = env.data.get(symbol, {}).get("D1")
        if df is None or env.current_step >= len(df):
            # if we can't look up price, deny
            return False
        current_price = float(df.iloc[env.current_step]["close"])

        # equity and margin used
        equity      = float(env.balance)
        margin_used = size * current_price  # simplistic margin calc

        data = {
            "extras": {
                "symbol": symbol,
                "size": size
            },
            "current_price": current_price,
            "risk": {
                "equity": equity,
                "margin_used": margin_used
            },
            # raw_action is the signed intensity or size
            "raw_action": trade.get("size")
        }
        return self.step(**data)

    def get_observation_components(self) -> np.ndarray:
        """
        ComplianceModule doesn’t add any raw features—
        so just return a zero-length vector.
        """
        return np.zeros(0, np.float32)

    def get_state(self):
        """
        No internal persisted state in this module besides last_flags,
        which we don't need to serialize across sessions.
        """
        return {}

    def set_state(self, state):
        """Nothing to reload."""
        return

    def _flag(self, msg: str) -> None:
        """Helper to record a compliance violation message."""
        self.last_flags.append(msg)
