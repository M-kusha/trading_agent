#!/usr/bin/env python3
"""Pull XAUUSD bars from the connected MT5 terminal. Read-only.

This places no orders and changes nothing on the account - it calls
copy_rates_range and writes CSVs.

The processed dataset ends 2025-12-18, so everything after that is genuinely
unseen: not a split of data the model trained near, but bars from the broker
the agent would actually trade through, at that broker's real spreads.

The env reads only open/high/low/close/volume (plus spread when present) - the
58 engineered columns in data/processed are not consulted on the training path -
so raw OHLCV is sufficient to evaluate on.

Usage:
    python evaluate/pull_ftmo_data.py --from 2025-12-01 --out data/ftmo_live
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pull_ftmo")

TIMEFRAMES = {"M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1"}


def pull(symbol: str, start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol {symbol} not found")
        if not info.visible:
            mt5.symbol_select(symbol, True)

        acct = mt5.account_info()
        if acct is not None:
            logger.info("Connected: account %s on %s (READ-ONLY, no orders)", acct.login, acct.server)
        logger.info("Current %s spread: %s points", symbol, info.spread)

        out: Dict[str, pd.DataFrame] = {}
        for name, attr in TIMEFRAMES.items():
            rates = mt5.copy_rates_range(symbol, getattr(mt5, attr), start, end)
            if rates is None or len(rates) == 0:
                logger.warning("%s: no bars returned (%s)", name, mt5.last_error())
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.rename(columns={"tick_volume": "volume", "real_volume": "real_volume"})
            keep = [c for c in ("time", "open", "high", "low", "close", "volume", "spread") if c in df.columns]
            df = df[keep].sort_values("time").reset_index(drop=True)
            out[name] = df
            logger.info(
                "%-4s %6d bars  %s -> %s  median spread %.1f pts",
                name, len(df), df["time"].iloc[0], df["time"].iloc[-1],
                float(df["spread"].median()) if "spread" in df else float("nan"),
            )
        return out
    finally:
        mt5.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--from", dest="start", default="2025-12-01")
    ap.add_argument("--to", dest="end", default=None)
    ap.add_argument("--out", default="data/ftmo_live")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) if args.end else datetime.now() + timedelta(days=1)

    frames = pull(args.symbol, start, end)
    if not frames:
        logger.error("No data pulled")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tf, df in frames.items():
        path = out_dir / f"{args.symbol}_{tf}.csv"
        df.to_csv(path, index=False)
        logger.info("wrote %s (%d rows)", path, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
