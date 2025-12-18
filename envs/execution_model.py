# envs/execution_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ExecutionConfig:
    """
    Execution anti-cheat model:
    - Spread (bid/ask) + per-lot commission
    - Slippage (worse fills) scaled by volatility proxy
    - Latency (fills occur N bars after decision)
    """
    # Gold-friendly defaults (tune per broker)
    base_spread_points: float = 0.20          # e.g., 0.20 = $0.20 on XAUUSD
    spread_mult_range: Tuple[float, float] = (0.85, 1.40)

    slippage_points_sigma: float = 0.05       # stdev in points
    slippage_mult_range: Tuple[float, float] = (0.60, 1.80)

    commission_per_lot: float = 0.0           # EUR (set if your broker charges)
    latency_bars: int = 1                     # action at t -> fill at t+1

    # Safety caps
    max_spread_points: float = 1.50
    max_slippage_points: float = 0.50


class ExecutionModel:
    def __init__(self, cfg: ExecutionConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

    def _spread_points(self, vol_proxy: float) -> float:
        # vol_proxy in [0..1] typically. Higher vol -> wider spread.
        mult = self.rng.uniform(self.cfg.spread_mult_range[0], self.cfg.spread_mult_range[1])
        spread = self.cfg.base_spread_points * mult * (1.0 + 1.5 * float(np.clip(vol_proxy, 0.0, 1.0)))
        return float(np.clip(spread, 0.0, self.cfg.max_spread_points))

    def quote(self, mid: float, vol_proxy: float) -> Tuple[float, float, float]:
        """
        Returns (bid, ask, spread_points).
        """
        spread = self._spread_points(vol_proxy)
        bid = mid - spread / 2.0
        ask = mid + spread / 2.0
        return float(bid), float(ask), float(spread)

    def _slippage_points(self, vol_proxy: float) -> float:
        mult = self.rng.uniform(self.cfg.slippage_mult_range[0], self.cfg.slippage_mult_range[1])
        slip = abs(self.rng.normal(0.0, self.cfg.slippage_points_sigma)) * mult * (1.0 + 2.0 * vol_proxy)
        return float(np.clip(slip, 0.0, self.cfg.max_slippage_points))

    def fill_entry(self, mid: float, direction: str, lot: float, vol_proxy: float) -> Tuple[float, float, Dict[str, Any]]:
        bid, ask, spread = self.quote(mid, vol_proxy)
        slip = self._slippage_points(vol_proxy)

        if direction == "long":
            fill = ask + slip  # worse for buyer
        else:
            fill = bid - slip  # worse for seller

        fee = self.cfg.commission_per_lot * float(lot)
        meta = {"bid": bid, "ask": ask, "spread_points": spread, "slippage_points": slip, "commission": fee}
        return float(fill), float(fee), meta

    def fill_exit(self, mid: float, direction: str, lot: float, vol_proxy: float) -> Tuple[float, float, Dict[str, Any]]:
        bid, ask, spread = self.quote(mid, vol_proxy)
        slip = self._slippage_points(vol_proxy)

        # Exit is opposite side of entry
        if direction == "long":
            fill = bid - slip  # worse for seller
        else:
            fill = ask + slip  # worse for buyer

        fee = self.cfg.commission_per_lot * float(lot)
        meta = {"bid": bid, "ask": ask, "spread_points": spread, "slippage_points": slip, "commission": fee}
        return float(fill), float(fee), meta
