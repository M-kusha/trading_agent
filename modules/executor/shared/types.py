from __future__ import annotations

import datetime as _dt
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

DEFAULT_CONTRACT_SIZE = 100_000.0


def _sf(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _iso(ts: Union[str, int, float, None]) -> Optional[str]:
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


def _parse_timestamp(ts: Union[str, int, float, None]) -> Optional[float]:
    if ts is None:
        return None

    if isinstance(ts, (int, float)):
        try:
            f = float(ts)
            return f if math.isfinite(f) else None
        except Exception:
            return None

    if isinstance(ts, str):
        try:

            s = ts.rstrip("Z")
            dt = _dt.datetime.fromisoformat(s.replace("Z", ""))
            return dt.timestamp()
        except Exception:

            try:
                return float(ts)
            except Exception:
                return None
    return None


def _contract_size_for_symbol(symbol: str) -> float:
    s = (symbol or "").upper().replace("_", "").replace("/", "")
    if not s:
        return DEFAULT_CONTRACT_SIZE


    if "XAU" in s or "GOLD" in s:
        return 100.0
    if "XAG" in s or "SILVER" in s:
        return 5000.0


    if "BTC" in s or "ETH" in s:
        return 1.0


    return DEFAULT_CONTRACT_SIZE


@dataclass(slots=True)
class PositionSnap:
    instrument: str
    side: int
    units: float
    entry_price: float
    notional_eur: float = 0.0
    open_time: Optional[Union[str, int, float]] = None
    peak_unrealized: float = 0.0
    entry_step: Optional[int] = None


    ppo_direction: Optional[str] = None
    expert_direction: Optional[str] = None
    ppo_confidence: Optional[float] = None
    was_ppo_led: Optional[bool] = None

    def as_bus(
        self,
        last_price: Optional[float] = None,
        *,
        include_aliases: bool = True
    ) -> Dict[str, Any]:

        s = 1 if self.side >= 0 else -1
        u = abs(_sf(self.units))
        ep = _sf(self.entry_price)
        notional = _sf(self.notional_eur) or (u * ep)
        lp = _sf(last_price) if last_price is not None else ep

        upnl = 0.0
        if lp > 0.0 and ep > 0.0 and u > 0.0:
            upnl = (lp - ep) * s * u


        if s > 0:
            pos_type = "BUY"
            action = "LONG"
        else:
            pos_type = "SELL"
            action = "SHORT"


        cs = _contract_size_for_symbol(self.instrument)
        lot_size = u / cs if (u > 0.0 and cs > 0.0) else 0.0

        open_ts = _parse_timestamp(self.open_time) or 0.0
        age_hours = max(0.0, (time.time() - open_ts) / 3600.0) if open_ts > 0.0 else 0.0

        out: Dict[str, Any] = {
            "instrument": self.instrument,
            "side": int(s),
            "units": float(u),
            "entry_price": float(ep),
            "notional_eur": float(notional),
            "unrealized_pnl": float(upnl),
            "unrealized_pnl_eur": float(upnl),
            "open_time": _iso(self.open_time) if self.open_time else time.time(),
            "age_hours": float(age_hours),
            "peak_unrealized": float(_sf(self.peak_unrealized)),


            "type": pos_type,
            "action": action,
            "current_price": float(lp),
            "price": float(lp),
            "pnl": float(upnl),
            "profit": float(upnl),


            "contract_size": float(cs),
            "lot_size": float(lot_size),
            "volume": float(lot_size),
            "lots": float(lot_size),


            "id": self.entry_step if self.entry_step is not None else hash(self.instrument) % 10000,
            "ticket": self.entry_step if self.entry_step is not None else hash(self.instrument) % 10000,

            "open_price": float(ep),
            "entry_time": _iso(self.open_time) if self.open_time else time.time(),
        }

        if self.entry_step is not None:
            out["entry_step"] = int(self.entry_step)

        if include_aliases:

            out["symbol"] = self.instrument
            out["price_open"] = float(ep)
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
    ticket: Optional[Union[int, str]] = None

    def as_bus(self) -> Dict[str, Any]:

        s = 1 if self.side >= 0 else -1
        u = abs(_sf(self.units))
        px = _sf(self.price)
        notional = _sf(self.notional_eur) or (u * px)
        rpnl = _sf(self.realized_pnl)


        cs = _contract_size_for_symbol(self.instrument)
        lots = u / cs if (u > 0.0 and cs > 0.0) else 0.0


        direction = "BUY" if s > 0 else "SELL"

        out: Dict[str, Any] = {
            "id": self.id,
            "ts": float(_sf(self.ts, time.time())),
            "step": int(self.step),
            "instrument": self.instrument,
            "symbol": self.instrument,
            "action": self.action,
            "direction": direction,
            "type": direction,
            "side": int(s),
            "units": float(u),
            "price": float(px),
            "entry_price": float(px),
            "exit_price": float(px),
            "notional_eur": float(notional),
            "notional": float(notional),
            "realized_pnl": rpnl,
            "pnl": rpnl,
            "pnl_eur": rpnl,
            "profit": rpnl,
            "origin_id": self.origin_id,
            "comment": self.comment,


            "contract_size": float(cs),
            "lots": float(lots),
            "volume": float(lots),
        }
        if self.ticket is not None:
            out["ticket"] = self.ticket
        return out
