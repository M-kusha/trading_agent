# modules/risk/compliance.py

from __future__ import annotations
from typing import Any, List, Dict, Optional, Set
import numpy as np
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

from modules.core.core import Module


class ComplianceModule(Module):
    """
    FIXED: Practical compliance module that actually allows trading.
    
    Key improvements:
    - Flexible symbol handling (both EUR/USD and EURUSD formats)
    - More reasonable default limits
    - Simple, clear validation logic
    - Better integration with trading environment
    - Support for live trading requirements
    """

    # More practical defaults
    DEFAULT_ALLOWED = {
        "EUR/USD", "EURUSD",
        "XAU/USD", "XAUUSD", 
        "GBP/USD", "GBPUSD",
        "USD/JPY", "USDJPY",
        "AUD/USD", "AUDUSD",
        "USD/CHF", "USDCHF",
        "NZD/USD", "NZDUSD",
        "EUR/GBP", "EURGBP",
    }

    def __init__(
        self,
        max_leverage: float = 30.0,  # More reasonable for forex
        max_single_position_risk: float = 0.20,  # 20% per position
        max_total_risk: float = 0.50,  # 50% total exposure
        max_daily_trades: int = 100,  # Prevent overtrading
        min_trade_size: float = 0.01,  # Minimum lot size
        max_trade_size: float = 10.0,  # Maximum lot size
        audit_log_path: str = "logs/risk/compliance_audit.jsonl",
        allowed_symbols: Optional[List[str]] = None,
        restricted_hours: Optional[List[int]] = None,  # Hours when trading is restricted
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug

        # 1) Symbol allow-list with flexible handling
        if allowed_symbols:
            self.allowed_symbols = self._normalize_symbols(allowed_symbols)
        else:
            # Check environment variable
            env_syms = os.getenv("COMPLIANCE_SYMBOLS")
            if env_syms:
                self.allowed_symbols = self._normalize_symbols(env_syms.split(","))
            else:
                self.allowed_symbols = self.DEFAULT_ALLOWED.copy()

        # 2) Risk parameters (more reasonable)
        self.max_leverage = float(max_leverage)
        self.max_single_position_risk = float(max_single_position_risk)
        self.max_total_risk = float(max_total_risk)
        self.max_daily_trades = int(max_daily_trades)
        self.min_trade_size = float(min_trade_size)
        self.max_trade_size = float(max_trade_size)
        
        # 3) Time restrictions
        self.restricted_hours = set(restricted_hours) if restricted_hours else set()
        
        # 4) Daily tracking
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.total_exposure = 0.0
        
        # 5) Audit trail
        self.audit_log_path = audit_log_path
        os.makedirs(os.path.dirname(audit_log_path) or ".", exist_ok=True)
        self.audit_logger = self._get_audit_logger()
        
        # 6) State tracking
        self.last_flags: List[str] = []
        self.last_audit: Dict[str, Any] = {}
        self.violations_count = 0
        
        # 7) Logger
        self.logger = self._get_logger()
        
        self.logger.info(
            f"ComplianceModule initialized: leverage={max_leverage}, "
            f"position_risk={max_single_position_risk}, symbols={len(self.allowed_symbols)}"
        )

    def _normalize_symbols(self, symbols: List[str]) -> Set[str]:
        """Normalize symbols to handle both EUR/USD and EURUSD formats"""
        normalized = set()
        for sym in symbols:
            sym = sym.strip().upper()
            normalized.add(sym)
            # Add both formats
            if "/" in sym:
                normalized.add(sym.replace("/", ""))
            elif len(sym) == 6:  # Likely EURUSD format
                normalized.add(f"{sym[:3]}/{sym[3:]}")
        return normalized

    def _get_logger(self) -> logging.Logger:
        """Setup rotating file logger"""
        logger = logging.getLogger(f"ComplianceModule_{id(self)}")
        if not logger.handlers:
            log_path = "logs/risk/compliance.log"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            handler = RotatingFileHandler(
                log_path, maxBytes=10_000_000, backupCount=5
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            logger.addHandler(handler)
        
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        logger.propagate = False
        return logger

    def _get_audit_logger(self) -> logging.Logger:
        """Setup audit trail logger"""
        audit_logger = logging.getLogger(f"ComplianceAudit_{id(self)}")
        if not audit_logger.handlers:
            handler = logging.FileHandler(self.audit_log_path, mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            audit_logger.addHandler(handler)
            audit_logger.setLevel(logging.INFO)
            audit_logger.propagate = False
        return audit_logger

    def mutate(self, std: float = 0.1):
        """Evolutionary tuning of risk limits"""
        # Mutate leverage
        self.max_leverage = float(np.clip(
            self.max_leverage + np.random.normal(0, std * 10),
            5.0, 100.0
        ))
        
        # Mutate position risk
        self.max_single_position_risk = float(np.clip(
            self.max_single_position_risk + np.random.normal(0, std * 0.05),
            0.05, 0.50
        ))
        
        # Mutate total risk
        self.max_total_risk = float(np.clip(
            self.max_total_risk + np.random.normal(0, std * 0.1),
            0.2, 1.0
        ))
        
        self.logger.debug(f"Mutated: leverage={self.max_leverage:.1f}, position_risk={self.max_single_position_risk:.2f}")

    def crossover(self, other: "ComplianceModule") -> "ComplianceModule":
        """Combine parameters from two modules"""
        child = ComplianceModule(
            max_leverage=self.max_leverage if np.random.rand() > 0.5 else other.max_leverage,
            max_single_position_risk=self.max_single_position_risk if np.random.rand() > 0.5 else other.max_single_position_risk,
            max_total_risk=self.max_total_risk if np.random.rand() > 0.5 else other.max_total_risk,
            max_daily_trades=self.max_daily_trades if np.random.rand() > 0.5 else other.max_daily_trades,
            audit_log_path=self.audit_log_path,
            allowed_symbols=list(self.allowed_symbols),
            debug=self.debug,
        )
        return child

    def reset(self) -> None:
        """Reset per-episode state"""
        self.last_flags.clear()
        self.last_audit = {}
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.total_exposure = 0.0
        self.violations_count = 0

    def validate_trade(
        self,
        instrument: str,
        size: float,
        price: float,
        balance: float,
        current_positions: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Validate a proposed trade against compliance rules.
        
        Returns True if trade is allowed, False otherwise.
        Updates last_flags with any violations.
        """
        self.last_flags.clear()
        timestamp = timestamp or datetime.now()
        
        # Build audit record
        audit = {
            "timestamp": timestamp.isoformat(),
            "instrument": instrument,
            "size": size,
            "price": price,
            "balance": balance,
            "checks": [],
            "passed": True,
        }
        
        # 1. Check daily trade limit
        current_date = timestamp.date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
            
        if self.daily_trade_count >= self.max_daily_trades:
            msg = f"Daily trade limit reached ({self.max_daily_trades})"
            self.last_flags.append(msg)
            audit["checks"].append({"rule": "daily_limit", "passed": False, "message": msg})
            audit["passed"] = False
        else:
            audit["checks"].append({"rule": "daily_limit", "passed": True})
        
        # 2. Check trading hours
        current_hour = timestamp.hour
        if current_hour in self.restricted_hours:
            msg = f"Trading restricted at hour {current_hour}"
            self.last_flags.append(msg)
            audit["checks"].append({"rule": "trading_hours", "passed": False, "message": msg})
            audit["passed"] = False
        else:
            audit["checks"].append({"rule": "trading_hours", "passed": True})
        
        # 3. Check allowed symbols
        normalized_inst = instrument.upper()
        if normalized_inst not in self.allowed_symbols:
            # Try without slash
            if "/" in normalized_inst:
                normalized_inst = normalized_inst.replace("/", "")
            
            if normalized_inst not in self.allowed_symbols:
                msg = f"Symbol {instrument} not in allowed list"
                self.last_flags.append(msg)
                audit["checks"].append({"rule": "allowed_symbol", "passed": False, "message": msg})
                audit["passed"] = False
            else:
                audit["checks"].append({"rule": "allowed_symbol", "passed": True})
        else:
            audit["checks"].append({"rule": "allowed_symbol", "passed": True})
        
        # 4. Check trade size limits
        abs_size = abs(size)
        if abs_size < self.min_trade_size:
            msg = f"Trade size {abs_size:.4f} below minimum {self.min_trade_size}"
            self.last_flags.append(msg)
            audit["checks"].append({"rule": "min_size", "passed": False, "message": msg})
            audit["passed"] = False
        elif abs_size > self.max_trade_size:
            msg = f"Trade size {abs_size:.4f} exceeds maximum {self.max_trade_size}"
            self.last_flags.append(msg)
            audit["checks"].append({"rule": "max_size", "passed": False, "message": msg})
            audit["passed"] = False
        else:
            audit["checks"].append({"rule": "trade_size", "passed": True})
        
        # 5. Check position risk
        position_value = abs_size * price * 100_000  # Assuming standard lot
        position_risk = position_value / max(balance, 1.0)
        
        if position_risk > self.max_single_position_risk:
            msg = f"Position risk {position_risk:.1%} exceeds limit {self.max_single_position_risk:.1%}"
            self.last_flags.append(msg)
            audit["checks"].append({"rule": "position_risk", "passed": False, "message": msg})
            audit["passed"] = False
        else:
            audit["checks"].append({"rule": "position_risk", "passed": True, "value": position_risk})
        
        # 6. Check total exposure
        total_exposure = position_value
        if current_positions:
            for pos in current_positions.values():
                if isinstance(pos, dict):
                    pos_size = abs(pos.get("size", 0) or pos.get("lots", 0))
                    pos_price = pos.get("price_open", price)
                    total_exposure += pos_size * pos_price * 100_000
        
        total_risk = total_exposure / max(balance, 1.0)
        if total_risk > self.max_total_risk:
            msg = f"Total risk {total_risk:.1%} exceeds limit {self.max_total_risk:.1%}"
            self.last_flags.append(msg)
            audit["checks"].append({"rule": "total_risk", "passed": False, "message": msg})
            audit["passed"] = False
        else:
            audit["checks"].append({"rule": "total_risk", "passed": True, "value": total_risk})
        
        # 7. Check leverage
        leverage = total_exposure / max(balance, 1.0)
        if leverage > self.max_leverage:
            msg = f"Leverage {leverage:.1f}x exceeds limit {self.max_leverage:.1f}x"
            self.last_flags.append(msg)
            audit["checks"].append({"rule": "leverage", "passed": False, "message": msg})
            audit["passed"] = False
        else:
            audit["checks"].append({"rule": "leverage", "passed": True, "value": leverage})
        
        # Record audit
        self.last_audit = audit
        if audit["passed"]:
            self.daily_trade_count += 1
        else:
            self.violations_count += 1
            
        # Log to audit trail
        try:
            self.audit_logger.info(json.dumps(audit))
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
        
        # Log summary
        if self.debug or not audit["passed"]:
            self.logger.info(
                f"Trade validation: {instrument} size={size:.2f} "
                f"passed={audit['passed']} violations={len(self.last_flags)}"
            )
        
        return audit["passed"]

    def step(self, **kwargs) -> bool:
        """
        Legacy interface - converts kwargs to validate_trade call.
        Returns True if trade passes or no trade data provided.
        """
        # Extract trade data from kwargs
        trade = kwargs.get("trade")
        if not trade:
            return True  # No trade to validate
            
        env = kwargs.get("env")
        if not env:
            self.logger.warning("No environment provided for validation")
            return True
            
        # Extract parameters
        instrument = trade.get("instrument", "")
        size = abs(trade.get("size", 0.0))
        
        # Get price from environment
        try:
            df = env.data.get(instrument, {}).get("D1")
            if df is not None and env.current_step < len(df):
                price = float(df.iloc[env.current_step]["close"])
            else:
                price = 1.0  # Default
        except:
            price = 1.0
            
        balance = getattr(env, "balance", 10000.0)
        positions = getattr(env, "open_positions", {})
        
        return self.validate_trade(
            instrument=instrument,
            size=size,
            price=price,
            balance=balance,
            current_positions=positions
        )

    def get_observation_components(self) -> np.ndarray:
        """Return compliance state as observation"""
        return np.array([
            float(self.daily_trade_count) / max(self.max_daily_trades, 1),
            float(self.violations_count) / 100.0,  # Normalized
            float(len(self.last_flags) > 0),  # Has violations
        ], dtype=np.float32)

    def get_last_audit(self) -> Dict[str, Any]:
        """Get the last audit record"""
        return self.last_audit.copy()

    def get_audit_log(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the last n audit entries"""
        if not os.path.exists(self.audit_log_path):
            return []
        
        try:
            from collections import deque
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                lines = deque(f, maxlen=n)
            return [json.loads(line) for line in lines if line.strip()]
        except Exception as e:
            self.logger.error(f"Failed to read audit log: {e}")
            return []

    def get_state(self) -> Dict[str, Any]:
        """Get state for serialization"""
        return {
            "max_leverage": self.max_leverage,
            "max_single_position_risk": self.max_single_position_risk,
            "max_total_risk": self.max_total_risk,
            "max_daily_trades": self.max_daily_trades,
            "min_trade_size": self.min_trade_size,
            "max_trade_size": self.max_trade_size,
            "allowed_symbols": list(self.allowed_symbols),
            "restricted_hours": list(self.restricted_hours),
            "daily_trade_count": self.daily_trade_count,
            "last_trade_date": self.last_trade_date.isoformat() if self.last_trade_date else None,
            "violations_count": self.violations_count,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialization"""
        self.max_leverage = float(state.get("max_leverage", self.max_leverage))
        self.max_single_position_risk = float(state.get("max_single_position_risk", self.max_single_position_risk))
        self.max_total_risk = float(state.get("max_total_risk", self.max_total_risk))
        self.max_daily_trades = int(state.get("max_daily_trades", self.max_daily_trades))
        self.min_trade_size = float(state.get("min_trade_size", self.min_trade_size))
        self.max_trade_size = float(state.get("max_trade_size", self.max_trade_size))
        
        if "allowed_symbols" in state:
            self.allowed_symbols = self._normalize_symbols(state["allowed_symbols"])
        
        if "restricted_hours" in state:
            self.restricted_hours = set(state["restricted_hours"])
            
        self.daily_trade_count = state.get("daily_trade_count", 0)
        self.violations_count = state.get("violations_count", 0)
        
        if state.get("last_trade_date"):
            from datetime import date
            self.last_trade_date = date.fromisoformat(state["last_trade_date"])

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of compliance status"""
        return {
            "daily_trades": f"{self.daily_trade_count}/{self.max_daily_trades}",
            "violations": self.violations_count,
            "last_flags": self.last_flags,
            "allowed_symbols": len(self.allowed_symbols),
            "leverage_limit": f"{self.max_leverage:.1f}x",
            "position_risk_limit": f"{self.max_single_position_risk:.1%}",
            "total_risk_limit": f"{self.max_total_risk:.1%}",
        }