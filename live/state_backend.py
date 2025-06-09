# live/state_backend.py
from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from typing import Dict, List

import MetaTrader5 as mt5
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ------------------------------ util --------------------------------------- #
def _lru_cache_with_expiry(seconds: int):
    """Decorator factory: LRU cache with time‑based expiry."""

    def decorator(fn):
        cached_fn = lru_cache(maxsize=None)(fn)

        def wrapper(self, *args, **kwargs):
            key = f"_cache_ts_{fn.__name__}_{args}_{kwargs}"
            ts_now = dt.datetime.utcnow().timestamp()
            if not hasattr(self, key) or ts_now - getattr(self, key) > seconds:
                cached_fn.cache_clear()  # type: ignore[attr-defined]
                setattr(self, key, ts_now)
            return cached_fn(self, *args, **kwargs)

        return wrapper

    return decorator


# ---------------------------- main class ----------------------------------- #
class StateBackend:
    """Pulls snapshots from MT5 and offers derived analytics."""

    _VOL_WINDOW_DAYS = 30
    _TF_MAP = {
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    def __init__(self) -> None:
        if not mt5.initialize():
            raise RuntimeError(f"Could not initialise MT5: {mt5.last_error()}")
        LOGGER.info("[StateBackend] MetaTrader5 initialised")

    # ---------------------------------------------------------------- account
    def get_account_snapshot(self) -> Dict:
        info = mt5.account_info()
        if info is None:
            LOGGER.error("account_info() failed: %s", mt5.last_error())
            return {}

        positions = mt5.positions_get()
        pos_list: List[Dict] = []
        if positions:
            for p in positions:
                pos_list.append(
                    dict(
                        ticket=p.ticket,
                        symbol=p.symbol,
                        side="buy" if p.type == 0 else "sell",
                        volume=float(p.volume),
                        price_open=float(p.price_open),
                        profit=float(p.profit),
                    )
                )

        equ = float(getattr(info, "equity", 0.0))
        bal = float(getattr(info, "balance", equ))
        dd = (bal - equ) / max(bal, 1e-8)

        return dict(
            balance=bal,
            equity=equ,
            margin=float(getattr(info, "margin", 0.0)),
            drawdown=round(dd, 4),
            timestamp=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            positions=pos_list,
        )

    # -------------------------------------------------------------- volatility
    @_lru_cache_with_expiry(seconds=12 * 60 * 60)  # refresh twice a day
    def compute_vol_profile(
        self,
        symbol: str = "XAUUSD",
        tf: str = "H1",
    ) -> Dict:
        """Return ATR‑style volatility profile for *symbol* on timeframe *tf*."""
        tf = tf.upper()
        if tf not in self._TF_MAP:
            return {}

        symbol_mt = symbol.replace("/", "").upper()

        utc_now = dt.datetime.utcnow()
        from_dt = utc_now - dt.timedelta(days=self._VOL_WINDOW_DAYS)

        rates = mt5.copy_rates_range(
            symbol_mt,
            self._TF_MAP[tf],
            from_dt,
            utc_now,
        )
        if rates is None or len(rates) == 0:
            LOGGER.warning("[StateBackend] No history for %s on %s", symbol_mt, tf)
            return {}

        df = pd.DataFrame(rates)
        df["atr"] = df["high"] - df["low"]
        df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        return dict(
            symbol=symbol_mt,
            timeframe=tf,
            window_days=self._VOL_WINDOW_DAYS,
            atr=list(df[["ts", "atr"]].itertuples(index=False, name=None)),
        )
