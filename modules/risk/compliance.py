# modules/risk/compliance.py

from __future__ import annotations
from typing import Any, List, Dict
import numpy as np
import json
import os
import logging
from logging.handlers import RotatingFileHandler

from modules.utils.info_bus import InfoBus, now_utc
from modules.core.core import Module

class ComplianceModule(Module):
    """
    Rule-based compliance module with explainability and resilient audit trail.
    Enforces a configurable allow-list and strict risk/leverage limits.
    """

    DEFAULT_ALLOWED = {"XAUUSD", "EURUSD"}

    def __init__(
        self,
        max_leverage: float = 20.0,
        max_single_position_risk: float = 0.10,
        audit_log_path: str = "logs/compliance_audit.jsonl",
        allowed_symbols: List[str] | None = None,
        debug: bool = False,
    ):
        # Module superclass takes no args
        super().__init__()
        self.debug = debug

        # 1) Load or override allow-list
        env_syms = os.getenv("COMPLIANCE_SYMBOLS")
        if allowed_symbols:
            self.allowed_symbols = set(sym.strip().upper() for sym in allowed_symbols)
        elif env_syms:
            self.allowed_symbols = set(s.strip().upper() for s in env_syms.split(","))
        else:
            self.allowed_symbols = set(self.DEFAULT_ALLOWED)

        # 2) Risk parameters
        self.max_leverage = float(max_leverage)
        self.max_single_position_risk = float(max_single_position_risk)

        # 3) Audit trail
        self.audit_log_path = audit_log_path
        # ensure audit log directory exists
        os.makedirs(os.path.dirname(audit_log_path) or ".", exist_ok=True)

        # 4) State
        self.last_flags: List[str] = []
        self.last_audit: Dict[str, Any] = {}

        # 5) Dedicated human-readable logger with rotation
        self.logger = self._get_logger()

    def _get_logger(self) -> logging.Logger:
        lg = logging.getLogger("ComplianceModule")
        if not any(isinstance(h, RotatingFileHandler) for h in lg.handlers):
            # make sure the logs folder exists
            log_path = "logs/compliance.log"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            handler = RotatingFileHandler(
                log_path, maxBytes=10_000_000, backupCount=5
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            lg.addHandler(handler)
        lg.setLevel(logging.DEBUG if self.debug else logging.INFO)
        lg.propagate = False
        return lg

    def mutate(self, std: float = 0.1):
        """Evolutionary tuning of risk limits."""
        self.max_leverage += np.random.normal(0, std * 5)
        self.max_leverage = float(np.clip(self.max_leverage, 1.0, 100.0))

        self.max_single_position_risk += np.random.normal(0, std * 0.02)
        self.max_single_position_risk = float(
            np.clip(self.max_single_position_risk, 0.001, 0.5)
        )

        self.logger.debug(
            f"[mutate] max_leverage={self.max_leverage:.2f}, "
            f"max_single_position_risk={self.max_single_position_risk:.3f}"
        )

    def crossover(self, other: "ComplianceModule") -> "ComplianceModule":
        """Combine parameters from two modules."""
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
            audit_log_path=self.audit_log_path,
            allowed_symbols=list(self.allowed_symbols),
            debug=self.debug,
        )
        child.logger.debug("[crossover] created child with combined parameters")
        return child

    def get_params(self) -> Dict[str, Any]:
        return {
            "max_leverage": self.max_leverage,
            "max_single_position_risk": self.max_single_position_risk,
            "allowed_symbols": sorted(self.allowed_symbols),
        }

    def reset(self) -> None:
        """Clear per-trade state flags and audit record."""
        self.last_flags.clear()
        self.last_audit = {}

    def step(self, **data: Any) -> bool:
        """
        Run rule checks only when validating a real trade.
        Returns True if all checks pass, else False.
        Logs a structured audit trail to self.audit_log_path.
        """
        # 1) Early-exit on non-trade calls
        extras = data.get("extras", {})
        if not extras or extras.get("symbol") is None:
            return True

        # 2) Build context
        info = InfoBus(**data)
        self.last_flags.clear()

        symbol_raw = extras.get("symbol", "").upper()
        symbol = symbol_raw.replace("/", "")

        rationale: Dict[str, Any] = {
            "timestamp":     info.get("timestamp") or now_utc(),
            "symbol_raw":    symbol_raw,
            "symbol":        symbol,
            "current_price": info.get("current_price", 0.0),
            "risk":          info.get("risk", {}),
            "raw_action":    info.get("raw_action"),
            "params":        self.get_params(),
            "results":       [],
            "final_decision": "ACCEPT",
        }

        # 3) Allow-list
        if symbol not in self.allowed_symbols:
            msg = f"Trading {symbol_raw} is not permitted by allow-list."
            rationale["results"].append({
                "rule":   "allowed_symbol",
                "passed": False,
                "symbol": symbol_raw,
                "detail": msg
            })
            self.last_flags.append(msg)
            self.logger.debug(msg)
        else:
            rationale["results"].append({
                "rule":   "allowed_symbol",
                "passed": True,
                "symbol": symbol_raw
            })

        # 4) Single-position-risk
        if isinstance(extras.get("size"), (int, float)):
            size = float(extras["size"])
            price = float(info.get("current_price", 0.0))
            equity = float(info.get("risk", {}).get("equity", 1e-8))
            fraction_risk = (size * price) / max(equity, 1e-8)

            if fraction_risk > self.max_single_position_risk:
                msg = (
                    f"Single trade risk {fraction_risk:.2%} "
                    f"exceeds {self.max_single_position_risk:.0%} limit."
                )
                rationale["results"].append({
                    "rule":   "max_single_position_risk",
                    "passed": False,
                    "value":  fraction_risk,
                    "detail": msg
                })
                self.last_flags.append(msg)
                self.logger.debug(msg)
            else:
                rationale["results"].append({
                    "rule":   "max_single_position_risk",
                    "passed": True,
                    "value":  fraction_risk
                })

        # 5) Leverage check
        risk = info.get("risk", {})
        margin = float(risk.get("margin_used", 0.0))
        equity = float(risk.get("equity", 1e-8))
        lev = margin / max(equity, 1e-9)

        if lev > self.max_leverage:
            msg = f"Leverage {lev:.1f}× exceeds {self.max_leverage:.1f}× limit."
            rationale["results"].append({
                "rule":   "max_leverage",
                "passed": False,
                "value":  lev,
                "detail": msg
            })
            self.last_flags.append(msg)
            self.logger.debug(msg)
        else:
            rationale["results"].append({
                "rule":   "max_leverage",
                "passed": True,
                "value":  lev
            })

        # 6) Final decision
        result = True
        if self.last_flags:
            rationale["final_decision"] = "REJECT"
            data.setdefault("compliance_flags", []).extend(self.last_flags)
            result = False

        # 7) Audit-trail write (resilient)
        self.last_audit = rationale
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(rationale) + "\n")
        except Exception as e:
            self.logger.error("Failed to write audit log: %s", e)

        return result

    def validate_trade(self, trade: dict, env) -> bool:
        """
        Adapter from a trade dict to compliance.step(). Returns False if
        trade is invalid or data missing.
        """
        symbol = trade.get("instrument")
        size   = abs(trade.get("size", 0.0))

        df = env.data.get(symbol, {}).get("D1")
        if df is None or env.current_step >= len(df):
            self.logger.error("validate_trade: missing price series for %s", symbol)
            return False

        current_price = float(df.iloc[env.current_step]["close"])
        equity        = float(env.balance)
        margin_used   = size * current_price

        payload = {
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
        return self.step(**payload)

    def get_last_audit(self) -> Dict[str, Any]:
        """Get the last audit rationale for UI/LLM/analysis."""
        return self.last_audit.copy()

    def get_audit_log(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the last n audit entries."""
        if not os.path.exists(self.audit_log_path):
            return []
        try:
            with open(self.audit_log_path, "r") as f:
                lines = f.readlines()[-n:]
            return [json.loads(l) for l in lines]
        except Exception as e:
            self.logger.error("get_audit_log failed: %s", e)
            return []

    def get_observation_components(self) -> np.ndarray:
        """Compliance produces no numeric features."""
        return np.zeros(0, np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "max_leverage":             self.max_leverage,
            "max_single_position_risk": self.max_single_position_risk,
            "allowed_symbols":          sorted(self.allowed_symbols),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.max_leverage = float(state.get("max_leverage", self.max_leverage))
        self.max_single_position_risk = float(
            state.get("max_single_position_risk", self.max_single_position_risk)
        )
        allowed = state.get("allowed_symbols")
        if isinstance(allowed, list):
            self.allowed_symbols = set(s.upper() for s in allowed)
        self.logger.debug(
            f"set_state: max_leverage={self.max_leverage}, "
            f"max_single_position_risk={self.max_single_position_risk}, "
            f"allowed_symbols={self.allowed_symbols}"
        )
