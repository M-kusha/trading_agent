# modules/executor/shared/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import math
import time
import datetime as _dt


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

        # Calculate lot size (assuming standard forex contract size)
        lot_size = u / 100000.0 if u > 0 else 0.0

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
            "price": float(lp),  # alias
            "pnl": float(upnl),  # alias for unrealized_pnl
            "profit": float(upnl),  # alias
            "lot_size": float(lot_size),
            "volume": float(lot_size),  # alias
            "lots": float(lot_size),  # alias
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

        out: Dict[str, Any] = {
            "id": self.id,
            "ts": float(_sf(self.ts, time.time())),
            "step": int(self.step),
            "instrument": self.instrument,
            "symbol": self.instrument,        # alias
            "action": self.action,
            "side": int(s),
            "units": float(u),
            "price": float(px),
            "notional_eur": float(notional),
            "notional": float(notional),      # alias
            "realized_pnl": rpnl,
            "pnl": rpnl,                      # alias
            "pnl_eur": rpnl,                  # alias
            "origin_id": self.origin_id,
            "comment": self.comment,
        }
        if self.ticket is not None:
            out["ticket"] = self.ticket
        return out
