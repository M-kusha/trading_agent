"""Viz Probe: SmartInfoBus watcher for simulation training.

PowerShell usage:
    .\\.venv\\Scripts\\python.exe .\\viz_probe.py --once
    .\\.venv\\Scripts\\python.exe .\\viz_probe.py --interval 2
"""

from __future__ import annotations

import argparse
import time
import datetime as dt
from typing import Any, Dict, List

from modules.utils.info_bus import InfoBusManager


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _first_non_empty(*values):
    for v in values:
        if v not in (None, {}, [], ""):
            return v
    return None


def collect_snapshot(bus) -> Dict[str, Any]:
    snap: Dict[str, Any] = {"timestamp": dt.datetime.now().isoformat(timespec="seconds")}

    # Step/episode
    step_data = bus.get("step_data", "viz_probe") or {}
    snap["step"] = step_data.get("step_idx")
    snap["episode"] = step_data.get("episode_idx")

    # Financials with fallbacks
    risk = bus.get("risk_metrics", "viz_probe") or {}
    port = bus.get("portfolio_metrics", "viz_probe") or {}
    market_state = bus.get("market_state", "viz_probe") or {}
    top_balance = bus.get("balance", "viz_probe")
    top_equity = bus.get("equity", "viz_probe")

    balance = _first_non_empty(risk.get("balance"), port.get("balance"), top_balance, 10000.0)
    equity = _first_non_empty(risk.get("equity"), top_equity, balance)
    drawdown = _first_non_empty(risk.get("current_drawdown"), port.get("drawdown"), market_state.get("current_drawdown"), 0.0)

    snap["balance"] = _safe_float(balance, 10000.0)
    snap["equity"] = _safe_float(equity, snap["balance"]) 
    snap["drawdown"] = _safe_float(drawdown, 0.0)

    # Positions (list or dict) and exposure
    positions = bus.get("positions", "viz_probe")
    if not positions:
        positions = bus.get("current_positions", "viz_probe") or []
    if isinstance(positions, dict):
        positions_list = list(positions.values())
    elif isinstance(positions, list):
        positions_list = positions
    # Committee / vote
    committee_decision = bus.get("committee_decision", "viz_probe") or {}
    committee_conf = bus.get("committee_confidence", "viz_probe")
    trade_vote = bus.get("trade_vote", "viz_probe")
    snap["decision"] = committee_decision.get("action") or trade_vote or "?"
    snap["confidence"] = _safe_float(committee_conf, 0.0)

    # Strategy signals (short summary if present)
    instr = bus.get("instrument_signals", "viz_probe") or {}
    if isinstance(instr, dict):
        # Show up to 3 instruments with intensity
        items = []
        for sym, d in list(instr.items())[:3]:
            if isinstance(d, dict):
                items.append(f"{sym}:{_safe_float(d.get('intensity'), 0.0):+.2f}")
        snap["signals"] = ", ".join(items)
    else:
        snap["signals"] = ""

    # VisualizationInterface footprint
    viz_data = bus.get("visualization_data", "viz_probe") or {}
    snap["viz_records"] = int(viz_data.get("total_records", 0))

    return snap


def print_snapshot(snap: Dict[str, Any]) -> None:
    step = snap.get("step")
    episode = snap.get("episode")
    balance = snap.get("balance", 0.0)
    equity = snap.get("equity", 0.0)
    dd = snap.get("drawdown", 0.0)
    pos = snap.get("positions_count", 0)
    exp = snap.get("exposure", 0.0)
    trades = snap.get("recent_trades_count", 0)
    pnl_total = snap.get("pnl_total", 0.0)
    decision = snap.get("decision", "?")
    conf = snap.get("confidence", 0.0)
    signals = snap.get("signals", "")
    viz_records = snap.get("viz_records", 0)

    line = (
        f"[VizProbe] step={step} ep={episode} | Bal={balance:,.2f} Eq={equity:,.2f} "
        f"DD={dd:.1%} | Pos={pos} Exp={exp:.2f} | Trades={trades} PnL={pnl_total:+.2f} "
        f"| Decision={decision} conf={conf:.2f} | Sig[{signals}] | Rec={viz_records}"
    )
    print(line)


def main():
    ap = argparse.ArgumentParser(description="Watch SmartInfoBus trading snapshot")
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between snapshots when watching")
    ap.add_argument("--once", action="store_true", help="Print a single snapshot and exit")
    args = ap.parse_args()

    bus = InfoBusManager.get_instance()

    if args.once:
        snap = collect_snapshot(bus)
        print_snapshot(snap)
        return

    try:
        while True:
            snap = collect_snapshot(bus)
            print_snapshot(snap)
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
