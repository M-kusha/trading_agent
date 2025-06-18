# modules/info_bus.py

"""Typed message container passed between modules at every environment step."""

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal, Optional

# ---- Core Per-Trade Info ----
class PositionInfo(TypedDict):
    symbol: str
    side: Literal["long", "short"]
    size: float
    entry_price: float
    unrealised_pnl: float

class RiskSnapshot(TypedDict, total=False):
    balance: float
    equity: float
    margin_used: float
    max_drawdown: float
    open_positions: List[PositionInfo]
    # Optionally add per-instrument risk, VaR, etc.
    var: Optional[float]
    dd_limit: Optional[float]
    max_exposure: Optional[float]

# ---- Voting/Consensus Info ----
class Vote(TypedDict):
    module: str
    action: Literal[0, 1, 2]  # 0 = hold, 1 = buy, 2 = sell
    confidence: float

# ---- Global Market Status ----
class MarketStatus(TypedDict, total=False):
    is_open: bool
    next_open_time: Optional[str]
    next_close_time: Optional[str]
    holiday: Optional[bool]
    reason: Optional[str]  # e.g. "weekend", "market holiday", "maintenance"

# ---- Central InfoBus Payload ----
class InfoBus(TypedDict, total=False):
    # Core trading data
    current_price: float
    atr: float
    timestamp: str  # ISO‑8601
    step_idx: int
    scenario: Optional[str]

    # Model outputs
    raw_action: float
    votes: List[Vote]

    # Risk & compliance
    risk: RiskSnapshot
    compliance_flags: List[str]

    # Market status
    market_status: MarketStatus

    # Alert/audit
    alerts: List[str]
    audit_log: List[str]

    # Arbitrary module extras
    extras: Dict[str, Any]

# ---- Utility Function: UTC ISO8601 Timestamp ----
from datetime import datetime, timezone

def now_utc() -> str:
    """Current UTC ISO8601 timestamp with timezone info (seconds precision)."""
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec='seconds')

