"""
Execution Model for Trading Environment (FTMO-aware)
====================================================

Adds:
- FTMO-style commission presets (Forex/Exotics, Indices=0, Metals/Commodities CFDs)
- Optional "Volume Bands" execution (price tier depends on lot size), per FTMO execution update.
- Commission modes:
  * PER_LOT_PER_SIDE (ideal for FX)
  * PER_VOLUME_PER_SIDE (for CFDs where broker defines "volume" differently)

Notes:
- Spreads on FTMO are not fixed constants; your existing stochastic spread model remains valid.
- For PER_VOLUME_PER_SIDE you MUST define what "volume" means for the symbol on your platform
  (MT4/MT5 contract specs vary by broker/symbol).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, List

import numpy as np

if TYPE_CHECKING:
    from envs.curriculum.curriculum_config import ExecutionDifficulty


# =============================================================================
# Commission modeling
# =============================================================================

class CommissionMode(str, Enum):
    NONE = "none"
    PER_LOT_PER_SIDE = "per_lot_per_side"
    PER_VOLUME_PER_SIDE = "per_volume_per_side"


@dataclass
class CommissionSpec:
    mode: CommissionMode = CommissionMode.NONE

    # For PER_LOT_PER_SIDE (typical FX): commission_rate is currency per lot per side.
    commission_rate: float = 0.0

    # For PER_VOLUME_PER_SIDE (CFDs): commission_rate is currency per "volume unit" per side.
    # What "volume unit" means is broker/symbol specific.
    # Provide volume_to_units to convert (mid, lot_size) -> units.
    volume_to_units: Optional[Callable[[float, float], float]] = None


# =============================================================================
# FTMO-style Volume Bands (optional)
# =============================================================================

@dataclass(frozen=True)
class VolumeBand:
    """
    A band defines execution adjustments for a lot-size tier.

    Example logic:
    - Find first band where lot_size < max_lots
    - Apply band_spread_mult (tight/loose) and band_slippage_mult to costs
    """
    max_lots_exclusive: float
    band_spread_mult: float = 1.0
    band_slippage_mult: float = 1.0


@dataclass
class ExecutionConfig:
    """
    Execution anti-cheat model configuration.

    Controls:
    - Spread (bid/ask) + commission
    - Slippage (worse fills) scaled by volatility proxy and order size
    - Latency (fills occur N bars after decision)

    FTMO-aware additions:
    - commission_spec: CommissionSpec (mode + rate + optional converter)
    - volume_bands_enabled + volume_bands: optional banded execution by lot size
    
    DATA SPREAD CONTROL (Jan 2026):
    - use_data_spread: If True, use actual spreads from data file (real FTMO conditions)
    - data_spread_scale: Discount factor applied to data spread (1.0 = full, 0.5 = half)
    """
    # Base spread configuration (fallback when no data spread)
    base_spread_points: float = 0.20
    spread_mult_range: Tuple[float, float] = (0.85, 1.40)

    # Slippage configuration
    slippage_points_sigma: float = 0.05
    slippage_mult_range: Tuple[float, float] = (0.60, 1.80)

    # Latency
    latency_bars: int = 1

    # Safety caps
    max_spread_points: float = 1.50
    max_slippage_points: float = 0.50
    
    # DATA SPREAD CONTROL
    # If True, prefer actual spreads from data file over synthetic calculation
    use_data_spread: bool = True   # Default True: use real FTMO spreads when available
    data_spread_scale: float = 1.0  # 1.0 = full spread, 0.5 = half (curriculum discount)

    # Volatility scaling factors
    spread_vol_factor: float = 1.5
    slippage_vol_factor: float = 2.0

    # Order size impact (larger orders get worse fills)
    size_impact_enabled: bool = True
    size_impact_factor: float = 0.08

    # Spread widening during volatility spikes
    volatility_spike_enabled: bool = True
    volatility_spike_threshold: float = 0.8
    volatility_spike_multiplier: float = 1.5

    # Rejection simulation
    rejection_enabled: bool = False
    rejection_probability: float = 0.0
    rejection_vol_factor: float = 0.0

    # Spread shock simulation (news events, liquidity gaps)
    # Used for adversarial evaluation to test robustness
    spread_shock_enabled: bool = False
    spread_shock_probability: float = 0.02  # 2% of quotes have widened spread
    spread_shock_multiplier: float = 3.0    # 3x normal spread during shock

    # Commission modeling
    commission_spec: CommissionSpec = field(default_factory=CommissionSpec)

    # Optional FTMO-like Volume Bands execution
    volume_bands_enabled: bool = False
    volume_bands: Tuple[VolumeBand, ...] = (
        VolumeBand(1.0, 1.00, 1.00),   # [0,1)
        VolumeBand(15.0, 1.05, 1.10),  # [1,15)
        VolumeBand(30.0, 1.10, 1.20),  # [15,30)
        VolumeBand(50.0, 1.15, 1.30),  # [30,50)
        VolumeBand(float("inf"), 1.20, 1.40),  # >=50
    )


class ExecutionModel:
    """
    Execution model that simulates broker conditions.

    Guarantees:
    - Entries fill worse (ask+slip for long, bid-slip for short)
    - Exits fill worse (bid-slip for long exit, ask+slip for short exit)
    - Slippage always against trader
    """

    def __init__(self, cfg: ExecutionConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

        # Per-episode randomization multipliers (domain randomization)
        self._spread_mult: float = 1.0
        self._slippage_mult: float = 1.0

        # Stats
        self._total_spread_cost: float = 0.0
        self._total_slippage_cost: float = 0.0
        self._total_commission: float = 0.0
        self._fill_count: int = 0

    def set_episode_randomization(self, spread_mult: float = 1.0, slippage_mult: float = 1.0) -> None:
        self._spread_mult = float(spread_mult)
        self._slippage_mult = float(slippage_mult)

    def reset_stats(self) -> None:
        self._total_spread_cost = 0.0
        self._total_slippage_cost = 0.0
        self._total_commission = 0.0
        self._fill_count = 0

    # ----------------------------
    # Helpers: volume band selection
    # ----------------------------
    def _band_multipliers(self, lot_size: float) -> Tuple[float, float]:
        if not self.cfg.volume_bands_enabled:
            return 1.0, 1.0

        ls = float(max(lot_size, 0.0))
        for band in self.cfg.volume_bands:
            if ls < band.max_lots_exclusive:
                return float(band.band_spread_mult), float(band.band_slippage_mult)
        return 1.0, 1.0

    def _compute_spread_points(self, vol_proxy: float, lot_size: float) -> float:
        mult = self.rng.uniform(self.cfg.spread_mult_range[0], self.cfg.spread_mult_range[1])
        vol_factor = 1.0 + self.cfg.spread_vol_factor * float(np.clip(vol_proxy, 0.0, 1.0))

        spike_mult = 1.0
        if self.cfg.volatility_spike_enabled and vol_proxy >= self.cfg.volatility_spike_threshold:
            spike_mult = self.cfg.volatility_spike_multiplier

        band_spread_mult, _ = self._band_multipliers(lot_size)

        spread = (
            self.cfg.base_spread_points
            * mult
            * vol_factor
            * spike_mult
            * self._spread_mult
            * band_spread_mult
        )
        return float(np.clip(spread, 0.0, self.cfg.max_spread_points))

    def _compute_slippage_points(self, vol_proxy: float, lot_size: float) -> float:
        base_slip = abs(self.rng.normal(0.0, self.cfg.slippage_points_sigma))
        mult = self.rng.uniform(self.cfg.slippage_mult_range[0], self.cfg.slippage_mult_range[1])
        vol_factor = 1.0 + self.cfg.slippage_vol_factor * float(np.clip(vol_proxy, 0.0, 1.0))

        size_factor = 1.0
        if self.cfg.size_impact_enabled:
            size_factor = 1.0 + self.cfg.size_impact_factor * (float(lot_size) - 1.0)
            size_factor = max(1.0, size_factor)

        _, band_slip_mult = self._band_multipliers(lot_size)

        slippage = (
            base_slip
            * mult
            * vol_factor
            * size_factor
            * self._slippage_mult
            * band_slip_mult
        )
        return float(np.clip(slippage, 0.0, self.cfg.max_slippage_points))

    # ----------------------------
    # Commission
    # ----------------------------
    def _commission_per_side(self, mid: float, lot_size: float) -> float:
        spec = self.cfg.commission_spec
        if spec.mode == CommissionMode.NONE:
            return 0.0

        if spec.mode == CommissionMode.PER_LOT_PER_SIDE:
            return float(spec.commission_rate) * float(lot_size)

        if spec.mode == CommissionMode.PER_VOLUME_PER_SIDE:
            if spec.volume_to_units is None:
                # Safe default: treat lot_size as "volume units" (NOT accurate for many CFDs).
                # You should override volume_to_units for symbol-accurate behavior.
                units = float(lot_size)
            else:
                units = float(spec.volume_to_units(float(mid), float(lot_size)))
            return float(spec.commission_rate) * units

        return 0.0

    def quote(
        self, 
        mid: float, 
        vol_proxy: float, 
        lot_size: float = 1.0,
        data_spread: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Get bid/ask quote.
        
        Args:
            mid: Mid price
            vol_proxy: Volatility proxy [0,1]
            lot_size: Order size
            data_spread: OPTIONAL actual spread from data file (in price points).
                         If provided, uses this instead of synthetic spread.
                         Critical for realistic training with actual broker spreads.
        """
        if data_spread is not None and data_spread > 0:
            # Use actual data spread with optional randomization
            spread = float(data_spread) * self._spread_mult
        else:
            # Fallback to synthetic spread
            spread = self._compute_spread_points(vol_proxy, lot_size)
        
        # Apply spread shock if enabled (simulates news events, liquidity gaps)
        if self.cfg.spread_shock_enabled:
            if self.rng.random() < self.cfg.spread_shock_probability:
                spread = spread * self.cfg.spread_shock_multiplier
        
        half = spread / 2.0
        bid = mid - half
        ask = mid + half
        return float(bid), float(ask), float(spread)

    def check_rejection(self, vol_proxy: float) -> bool:
        if not self.cfg.rejection_enabled:
            return False
        rejection_prob = self.cfg.rejection_probability + self.cfg.rejection_vol_factor * float(vol_proxy)
        return self.rng.random() < rejection_prob

    def fill_entry(self, mid: float, direction: str, lot_size: float, vol_proxy: float, data_spread: Optional[float] = None) -> Tuple[float, float, Dict[str, Any]]:
        bid, ask, spread = self.quote(mid, vol_proxy, lot_size, data_spread=data_spread)
        slippage = self._compute_slippage_points(vol_proxy, lot_size)

        if direction == "long":
            fill = ask + slippage
        else:
            fill = bid - slippage

        commission_side = self._commission_per_side(mid, lot_size)
        # Entry commission is "per side"
        commission = commission_side

        self._total_spread_cost += spread
        self._total_slippage_cost += slippage
        self._total_commission += commission
        self._fill_count += 1

        meta = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_points": spread,
            "slippage_points": slippage,
            "commission": commission,
            "commission_mode": self.cfg.commission_spec.mode.value,
            "direction": direction,
            "lot_size": float(lot_size),
            "vol_proxy": float(vol_proxy),
        }
        return float(fill), float(commission), meta

    def fill_exit(self, mid: float, direction: str, lot_size: float, vol_proxy: float, data_spread: Optional[float] = None) -> Tuple[float, float, Dict[str, Any]]:
        bid, ask, spread = self.quote(mid, vol_proxy, lot_size, data_spread=data_spread)
        slippage = self._compute_slippage_points(vol_proxy, lot_size)

        if direction == "long":
            fill = bid - slippage
        else:
            fill = ask + slippage

        commission_side = self._commission_per_side(mid, lot_size)
        # Exit commission is "per side"
        commission = commission_side

        self._total_spread_cost += spread
        self._total_slippage_cost += slippage
        self._total_commission += commission
        self._fill_count += 1

        meta = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_points": spread,
            "slippage_points": slippage,
            "commission": commission,
            "commission_mode": self.cfg.commission_spec.mode.value,
            "direction": direction,
            "lot_size": float(lot_size),
            "vol_proxy": float(vol_proxy),
            "is_exit": True,
        }
        return float(fill), float(commission), meta

    def get_execution_stats(self) -> Dict[str, float]:
        return {
            "total_spread_cost": float(self._total_spread_cost),
            "total_slippage_cost": float(self._total_slippage_cost),
            "total_commission": float(self._total_commission),
            "fill_count": float(self._fill_count),
            "avg_spread": float(self._total_spread_cost / max(self._fill_count, 1)),
            "avg_slippage": float(self._total_slippage_cost / max(self._fill_count, 1)),
            "avg_commission": float(self._total_commission / max(self._fill_count, 1)),
        }


# =============================================================================
# FTMO presets
# =============================================================================

def ftmo_commission_spec(symbol_group: str) -> CommissionSpec:
    """
    FTMO commission presets based on published updates.

    - Indices: zero commission.
    - Forex/Exotics: $2.50 per lot per side.
    - Metals/Commodities/Cash III CFDs: 0.0007 per volume per side (needs volume definition).
    """
    g = symbol_group.lower().strip()

    if g in ("indices", "index", "cash_indices"):
        # FTMO highlights indices are commission-free.
        return CommissionSpec(mode=CommissionMode.NONE, commission_rate=0.0)

    if g in ("forex", "fx", "exotics"):
        # FTMO update: $2.50 per lot per side.
        return CommissionSpec(mode=CommissionMode.PER_LOT_PER_SIDE, commission_rate=2.50)

    if g in ("metals_cfd", "metals", "commodities_cfd", "commodities", "cash_iii_cfd"):
        # FTMO update: 0.0007 per volume per side.
        # IMPORTANT: define "volume" via volume_to_units for your platform/symbol.
        return CommissionSpec(mode=CommissionMode.PER_VOLUME_PER_SIDE, commission_rate=0.0007)

    # Default: no commission (explicit is better than implicit)
    return CommissionSpec(mode=CommissionMode.NONE, commission_rate=0.0)


def create_execution_model_for_stage(
    stage_execution: "ExecutionDifficulty",
    rng: np.random.Generator,
    *,
    symbol_group: Optional[str] = None,
    enable_volume_bands: Optional[bool] = None,
    volume_to_units: Optional[Callable[[float, float], float]] = None,
) -> ExecutionModel:
    """
    Create an ExecutionModel configured for a curriculum stage, with optional FTMO commission presets.

    Args:
        stage_execution: ExecutionDifficulty from curriculum stage config
        rng: NumPy random generator
        symbol_group: optional, e.g. "forex", "indices", "metals"
        enable_volume_bands: optional override for volume-bands execution
        volume_to_units: optional converter for PER_VOLUME_PER_SIDE commissions

    Returns:
        Configured ExecutionModel
    """
    # Import here to avoid circular dependency
    from envs.curriculum.curriculum_config import ExecutionDifficulty

    # Base config derived from curriculum stage difficulty
    cfg = ExecutionConfig(
        base_spread_points=stage_execution.base_spread_points,
        spread_mult_range=stage_execution.spread_mult_range,
        max_spread_points=stage_execution.max_spread_points,
        slippage_points_sigma=stage_execution.slippage_points_sigma,
        slippage_mult_range=stage_execution.slippage_mult_range,
        max_slippage_points=stage_execution.max_slippage_points,
        latency_bars=stage_execution.latency_bars,
    )

    # FTMO commission presets (optional)
    if symbol_group is not None:
        cs = ftmo_commission_spec(symbol_group)
        if cs.mode == CommissionMode.PER_VOLUME_PER_SIDE and volume_to_units is not None:
            cs.volume_to_units = volume_to_units
        cfg.commission_spec = cs
    else:
        # Backward-compatibility: map stage_execution.commission_per_lot to PER_LOT_PER_SIDE
        if getattr(stage_execution, "commission_per_lot", 0.0) > 0.0:
            cfg.commission_spec = CommissionSpec(
                mode=CommissionMode.PER_LOT_PER_SIDE,
                commission_rate=float(stage_execution.commission_per_lot),
            )

    # Optional volume-bands toggle (FTMO-style)
    if enable_volume_bands is not None:
        cfg.volume_bands_enabled = bool(enable_volume_bands)

    model = ExecutionModel(cfg, rng)

    # Per-episode randomization (domain randomization)
    if stage_execution.enable_randomization:
        spread_mult = rng.uniform(
            stage_execution.spread_randomization_range[0],
            stage_execution.spread_randomization_range[1],
        )
        slippage_mult = rng.uniform(
            stage_execution.slippage_randomization_range[0],
            stage_execution.slippage_randomization_range[1],
        )
        model.set_episode_randomization(spread_mult, slippage_mult)

    return model
