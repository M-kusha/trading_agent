# modules/info_bus.py

"""Typed message container passed between modules at every environment step."""
from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Literal, Optional


class PositionInfo(TypedDict):
    symbol: str
    side: Literal["long", "short"]
    size: float
    entry_price: float
    unrealised_pnl: float


class RiskSnapshot(TypedDict):
    balance: float
    equity: float
    margin_used: float
    max_drawdown: float
    open_positions: List[PositionInfo]


class Vote(TypedDict):
    module: str
    action: Literal[0, 1, 2]  # 0 = hold, 1 = buy, 2 = sell
    confidence: float


class InfoBus(TypedDict, total=False):
    # Core trading data
    current_price: float
    atr: float
    timestamp: str  # ISO‑8601

    # Model outputs
    raw_action: float
    votes: List[Vote]

    # Risk & compliance
    risk: RiskSnapshot
    compliance_flags: List[str]

    # Arbitrary module extras
    extras: Dict[str, Any]
