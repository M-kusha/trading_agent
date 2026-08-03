# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from modules.executor.executor import Executor, ExecutorConfig


class _Bus:
    def __init__(self, state=None):
        self.state = state

    def get(self, *_args, **_kwargs):
        return self.state


class _Adapter:
    def __init__(self):
        self.orders = []

    def is_connected(self):
        return True

    def get_account_info(self):
        return {"balance": 100_000.0, "equity": 100_000.0}

    def market_order(self, symbol, side, lots):
        self.orders.append((symbol, side, lots))
        return {"ok": True}


def _executor() -> Executor:
    executor = object.__new__(Executor)
    executor.cfg = ExecutorConfig()
    executor.initial_balance = 100_000.0
    executor.balance = 100_000.0
    executor.equity = 100_000.0
    executor.day_start_balance = 100_000.0
    executor.peak_balance = 100_000.0
    executor._risk_day = None
    executor._risk_anchor_authoritative = False
    executor.step_idx = 7
    executor.trades = []
    executor.closed_positions = []
    executor.bus = _Bus()
    executor.logger = SimpleNamespace(error=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)
    executor.adapter = _Adapter()
    return executor


def test_executor_publishes_the_complete_live_position_and_risk_contract():
    executor = _executor()
    executor._risk_anchor_authoritative = True
    executor.trades = [{"ts": time.time(), "action": "open_long", "comment": "open"}]
    executor.closed_positions = [{"pnl": 15.0}, {"pnl": -10.0}, {"pnl": -5.0}]
    positions = {
        "XAUUSD": {
            "instrument": "XAUUSD",
            "side": 1,
            "entry_price": 4_000.0,
            "lots": 0.5,
            "sl": 3_990.0,
        }
    }

    state = executor._build_live_account_state(positions)
    assert state["has_position"] is True
    assert state["position_count"] == 1
    assert state["position"] == positions["XAUUSD"]
    assert state["trades_today"] == 1
    assert state["consecutive_losses"] == 2
    assert state["risk_anchor_authoritative"] is True


def test_midday_restart_without_day_anchor_blocks_new_exposure(monkeypatch):
    monkeypatch.delenv("LIVE_DAY_START_BALANCE", raising=False)
    executor = _executor()
    executor._estimate_live_order_risk = lambda *_a, **_k: 100.0

    result = executor._guarded_market_order("XAUUSD", 1, 0.1, route="test")
    assert result["ok"] is False
    assert "authoritative broker-day risk anchor" in result["error"]
    assert executor.adapter.orders == []


def test_explicit_day_anchor_allows_guarded_order_inside_budget(monkeypatch):
    monkeypatch.setenv("LIVE_DAY_START_BALANCE", "100000")
    executor = _executor()
    executor._estimate_live_order_risk = lambda *_a, **_k: 100.0

    result = executor._guarded_market_order("XAUUSD", 1, 0.1, route="test")
    assert result == {"ok": True}
    assert executor._risk_anchor_authoritative is True
    assert executor.adapter.orders == [("XAUUSD", 1, 0.1)]
