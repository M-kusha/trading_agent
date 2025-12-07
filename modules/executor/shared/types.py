from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import math
import time
import datetime as _dt


DEFAULT_CONTRACT_SIZE = 100_000.0  # default FX contract


def _sf(v: Any, default: float = 0.0) -> float:
    """Safe finite float."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _iso(ts: Union[str, int, float, None]) -> Optional[str]:
    """Best-effort ISO8601 UTC 'Z' timestamp, pass through strings."""
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts
    try:
        t = float(ts)
        if not math.isfinite(t):
            return None
        return _dt.datetime.utcfromtimestamp(t).isoformat(timespec="seconds") + "Z"
    except Exception:
        return None


def _contract_size_for_symbol(symbol: str) -> float:
    """
    Symbol-specific contract size (units per 1.0 lot).

    This mirrors the logic used in Executor and MT5Adapter:
    - XAU*/GOLD* : 100 oz per lot
    - XAG*/SILVER*: 5000 oz per lot
    - BTC*/ETH*  : 1 unit per lot
    - else       : 100,000 units (standard FX)
    """
    s = (symbol or "").upper().replace("_", "").replace("/", "")
    if not s:
        return DEFAULT_CONTRACT_SIZE

    # Metals
    if "XAU" in s or "GOLD" in s:
        return 100.0
    if "XAG" in s or "SILVER" in s:
        return 5000.0

    # Crypto
    if "BTC" in s or "ETH" in s:
        return 1.0

    # Default FX
    return DEFAULT_CONTRACT_SIZE


@dataclass(slots=True)
class PositionSnap:
    instrument: str
    side: int                # +1 long, -1 short
    units: float
    entry_price: float
    notional_eur: float = 0.0
    open_time: Optional[Union[str, int, float]] = None
    peak_unrealized: float = 0.0          # optional, some UIs use this
    entry_step: Optional[int] = None      # optional step index (sim)
    
    # Decision context at entry (for PPO autonomy tracking)
    ppo_direction: Optional[str] = None       # PPO's direction at entry: "long"/"short"/"flat"
    expert_direction: Optional[str] = None    # Expert consensus at entry
    ppo_confidence: Optional[float] = None    # PPO's confidence at entry
    was_ppo_led: Optional[bool] = None        # Was PPO leading when this trade was opened?

    def as_bus(
        self,
        last_price: Optional[float] = None,
        *,
        include_aliases: bool = True
    ) -> Dict[str, Any]:
        # sanitize
        s = 1 if self.side >= 0 else -1
        u = abs(_sf(self.units))
        ep = _sf(self.entry_price)
        notional = _sf(self.notional_eur) or (u * ep)
        lp = _sf(last_price) if last_price is not None else ep  # default to entry if no current price

        upnl = 0.0
        if lp > 0.0 and ep > 0.0 and u > 0.0:
            upnl = (lp - ep) * s * u

        # Determine position type/action
        if s > 0:
            pos_type = "BUY"
            action = "LONG"
        else:
            pos_type = "SELL"
            action = "SHORT"

        # Contract size & lots (symbol-aware: FX, XAU, XAG, BTC, ETH)
        cs = _contract_size_for_symbol(self.instrument)
        lot_size = u / cs if (u > 0.0 and cs > 0.0) else 0.0

        out: Dict[str, Any] = {
            "instrument": self.instrument,
            "side": int(s),
            "units": float(u),
            "entry_price": float(ep),
            "notional_eur": float(notional),
            "unrealized_pnl": float(upnl),
            "unrealized_pnl_eur": float(upnl),  # alias some modules expect
            "open_time": _iso(self.open_time) if self.open_time else time.time(),
            "peak_unrealized": float(_sf(self.peak_unrealized)),

            # Additional fields for visualizer and monitoring
            "type": pos_type,
            "action": action,
            "current_price": float(lp),
            "price": float(lp),        # alias
            "pnl": float(upnl),        # alias for unrealized_pnl
            "profit": float(upnl),     # alias

            # Contract / lot details
            "contract_size": float(cs),
            "lot_size": float(lot_size),
            "volume": float(lot_size),  # alias (MT5 convention: volume=lots)
            "lots": float(lot_size),    # alias

            # Simple identifiers (sim-mode tickets)
            "id": self.entry_step if self.entry_step is not None else hash(self.instrument) % 10000,
            "ticket": self.entry_step if self.entry_step is not None else hash(self.instrument) % 10000,

            "open_price": float(ep),  # alias
            "entry_time": _iso(self.open_time) if self.open_time else time.time(),
        }

        if self.entry_step is not None:
            out["entry_step"] = int(self.entry_step)

        if include_aliases:
            # a couple of common keys different parts of the stack look for
            out["symbol"] = self.instrument
            out["price_open"] = float(ep)  # some legacy consumers use price_open
        return out


@dataclass(slots=True)
class TradeFill:
    id: str
    ts: float
    step: int
    instrument: str
    action: str
    side: int
    units: float
    price: float
    notional_eur: float
    realized_pnl: float = 0.0
    origin_id: str = ""
    comment: str = ""
    ticket: Optional[Union[int, str]] = None  # live brokers often provide one

    def as_bus(self) -> Dict[str, Any]:
        # sanitize
        s = 1 if self.side >= 0 else -1
        u = abs(_sf(self.units))
        px = _sf(self.price)
        notional = _sf(self.notional_eur) or (u * px)
        rpnl = _sf(self.realized_pnl)

        # Contract-aware lot computation (aligned with positions)
        cs = _contract_size_for_symbol(self.instrument)
        lots = u / cs if (u > 0.0 and cs > 0.0) else 0.0

        # Direction string for dashboard display
        direction = "BUY" if s > 0 else "SELL"

        out: Dict[str, Any] = {
            "id": self.id,
            "ts": float(_sf(self.ts, time.time())),
            "step": int(self.step),
            "instrument": self.instrument,
            "symbol": self.instrument,        # alias
            "action": self.action,
            "direction": direction,           # human-readable direction
            "type": direction,                # alias for direction
            "side": int(s),
            "units": float(u),
            "price": float(px),
            "entry_price": float(px),         # alias for dashboard compatibility
            "exit_price": float(px),          # alias for dashboard (same as price for fills)
            "notional_eur": float(notional),
            "notional": float(notional),      # alias
            "realized_pnl": rpnl,
            "pnl": rpnl,                      # alias
            "pnl_eur": rpnl,                  # alias
            "profit": rpnl,                   # alias for dashboard
            "origin_id": self.origin_id,
            "comment": self.comment,

            # Contract / lot info for fills (matches PositionSnap)
            "contract_size": float(cs),
            "lots": float(lots),
            "volume": float(lots),            # alias
        }
        if self.ticket is not None:
            out["ticket"] = self.ticket
        return out
