# envs/execution_model.py
"""
Execution Model for Trading Environment
========================================

Anti-cheat execution model that simulates realistic broker conditions:
- Spread (bid/ask) with volatility scaling
- Slippage (worse fills) scaled by volatility and order size
- Latency (fills occur N bars after decision)
- Commission per lot

Enhanced for Curriculum Learning:
- Execution difficulty scales with curriculum stage
- Domain randomization for robustness
- Configurable via ExecutionDifficulty from curriculum_config

This model ensures the agent cannot exploit unrealistic execution assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from envs.curriculum_config import ExecutionDifficulty


@dataclass
class ExecutionConfig:
    """
    Execution anti-cheat model configuration.
    
    Controls:
    - Spread (bid/ask) + per-lot commission
    - Slippage (worse fills) scaled by volatility proxy
    - Latency (fills occur N bars after decision)
    
    Gold-friendly defaults (tune per broker):
    - XAUUSD typical spread: 0.15-0.30 points
    - XAUUSD typical slippage: 0.01-0.10 points
    """
    # Base spread configuration
    base_spread_points: float = 0.20          # Base spread in price points
    spread_mult_range: Tuple[float, float] = (0.85, 1.40)  # Random multiplier range
    spread_mult: float = 1.0                  # Current episode's spread multiplier (sampled from range)
    
    # Slippage configuration  
    slippage_points_sigma: float = 0.05       # Stdev of slippage (points)
    slippage_mult_range: Tuple[float, float] = (0.60, 1.80)  # Random multiplier range
    slippage_mult: float = 1.0                # Current episode's slippage multiplier (sampled from range)
    
    # Commission and latency
    commission_per_lot: float = 0.0           # EUR per lot (set if broker charges)
    latency_bars: int = 1                     # Action at t -> fill at t+latency_bars
    
    # Safety caps (prevent extreme execution)
    max_spread_points: float = 1.50           # Maximum spread cap
    max_slippage_points: float = 0.50         # Maximum slippage cap
    
    # Volatility scaling factors
    spread_vol_factor: float = 1.5            # How much volatility affects spread
    slippage_vol_factor: float = 2.0          # How much volatility affects slippage
    
    # Order size impact (larger orders get worse fills)
    size_impact_enabled: bool = False         # Enable size-based slippage
    size_impact_factor: float = 0.1           # Slippage multiplier per lot
    
    # Quote staleness (simulate delayed quotes)
    quote_staleness_enabled: bool = False     # Enable stale quotes
    quote_staleness_bars: int = 0             # How many bars old the quote can be
    
    # Spread widening during volatility spikes
    volatility_spike_enabled: bool = True     # Enable spread widening
    volatility_spike_threshold: float = 0.8   # Vol proxy threshold for spike
    volatility_spike_multiplier: float = 1.5  # Spread multiplier during spike
    
    # Rejection simulation (for live-ready stage)
    rejection_enabled: bool = False           # Enable order rejections
    rejection_probability: float = 0.0        # Base rejection probability
    rejection_vol_factor: float = 0.0         # Additional rejection prob per vol unit


class ExecutionModel:
    """
    Execution model that simulates realistic broker conditions.
    
    Features:
    - Bid/ask spread with volatility and time-of-day adjustments
    - Slippage that worsens with volatility and order size
    - Configurable latency between decision and fill
    - Commission tracking
    - Domain randomization via per-episode multipliers
    
    The model ensures that:
    1. Entries always fill at worse prices (ask for long, bid for short)
    2. Exits always fill at worse prices (bid for long, ask for short)
    3. Slippage always works against the trader
    4. High volatility = worse execution
    """
    
    def __init__(
        self,
        cfg: ExecutionConfig,
        rng: np.random.Generator,
    ) -> None:
        """
        Initialize the execution model.
        
        Args:
            cfg: Execution configuration
            rng: NumPy random generator for reproducibility
        """
        self.cfg = cfg
        self.rng = rng
        
        # Per-episode randomization state (set externally if using domain randomization)
        self._spread_mult: float = 1.0
        self._slippage_mult: float = 1.0
        
        # Statistics tracking
        self._total_spread_cost: float = 0.0
        self._total_slippage_cost: float = 0.0
        self._total_commission: float = 0.0
        self._fill_count: int = 0
    
    def set_episode_randomization(
        self,
        spread_mult: float = 1.0,
        slippage_mult: float = 1.0,
    ) -> None:
        """
        Set per-episode randomization multipliers.
        
        Call this at the start of each episode when using domain randomization.
        
        Args:
            spread_mult: Multiplier for spread
            slippage_mult: Multiplier for slippage
        """
        self._spread_mult = float(spread_mult)
        self._slippage_mult = float(slippage_mult)
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._total_spread_cost = 0.0
        self._total_slippage_cost = 0.0
        self._total_commission = 0.0
        self._fill_count = 0
    
    def _compute_spread_points(self, vol_proxy: float) -> float:
        """
        Compute spread in price points.
        
        Spread increases with:
        - Volatility (via vol_proxy)
        - Random variation (spread_mult_range)
        - Volatility spikes (if enabled)
        - Episode randomization multiplier
        
        Args:
            vol_proxy: Volatility proxy in [0, 1] range (higher = more volatile)
            
        Returns:
            Spread in price points
        """
        # Base spread with random variation
        mult = self.rng.uniform(
            self.cfg.spread_mult_range[0],
            self.cfg.spread_mult_range[1],
        )
        
        # Volatility scaling: higher vol = wider spread
        vol_factor = 1.0 + self.cfg.spread_vol_factor * float(np.clip(vol_proxy, 0.0, 1.0))
        
        # Volatility spike multiplier
        spike_mult = 1.0
        if self.cfg.volatility_spike_enabled and vol_proxy >= self.cfg.volatility_spike_threshold:
            spike_mult = self.cfg.volatility_spike_multiplier
        
        # Combine all factors
        spread = (
            self.cfg.base_spread_points
            * mult
            * vol_factor
            * spike_mult
            * self._spread_mult  # Episode randomization
        )
        
        return float(np.clip(spread, 0.0, self.cfg.max_spread_points))
    
    def _compute_slippage_points(
        self,
        vol_proxy: float,
        lot_size: float = 1.0,
    ) -> float:
        """
        Compute slippage in price points.
        
        Slippage increases with:
        - Volatility (via vol_proxy)
        - Order size (if size_impact_enabled)
        - Random variation (slippage_mult_range)
        - Episode randomization multiplier
        
        Args:
            vol_proxy: Volatility proxy in [0, 1] range
            lot_size: Order size in lots
            
        Returns:
            Slippage in price points (always positive - direction handled by caller)
        """
        # Random base slippage (half-normal distribution - always positive)
        base_slip = abs(
            self.rng.normal(0.0, self.cfg.slippage_points_sigma)
        )
        
        # Random multiplier
        mult = self.rng.uniform(
            self.cfg.slippage_mult_range[0],
            self.cfg.slippage_mult_range[1],
        )
        
        # Volatility scaling
        vol_factor = 1.0 + self.cfg.slippage_vol_factor * float(np.clip(vol_proxy, 0.0, 1.0))
        
        # Size impact (larger orders = more slippage)
        size_factor = 1.0
        if self.cfg.size_impact_enabled:
            size_factor = 1.0 + self.cfg.size_impact_factor * (lot_size - 1.0)
            size_factor = max(1.0, size_factor)  # Never reduce slippage
        
        # Combine all factors
        slippage = (
            base_slip
            * mult
            * vol_factor
            * size_factor
            * self._slippage_mult  # Episode randomization
        )
        
        return float(np.clip(slippage, 0.0, self.cfg.max_slippage_points))
    
    def quote(
        self,
        mid: float,
        vol_proxy: float,
    ) -> Tuple[float, float, float]:
        """
        Get current bid/ask quote.
        
        Args:
            mid: Mid-market price
            vol_proxy: Volatility proxy in [0, 1] range
            
        Returns:
            Tuple of (bid, ask, spread_points)
        """
        spread = self._compute_spread_points(vol_proxy)
        half_spread = spread / 2.0
        
        bid = mid - half_spread
        ask = mid + half_spread
        
        return float(bid), float(ask), float(spread)
    
    def check_rejection(self, vol_proxy: float) -> bool:
        """
        Check if an order should be rejected.
        
        Args:
            vol_proxy: Volatility proxy
            
        Returns:
            True if order should be rejected
        """
        if not self.cfg.rejection_enabled:
            return False
        
        rejection_prob = (
            self.cfg.rejection_probability
            + self.cfg.rejection_vol_factor * vol_proxy
        )
        
        return self.rng.random() < rejection_prob
    
    def fill_entry(
        self,
        mid: float,
        direction: str,
        lot_size: float,
        vol_proxy: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Compute entry fill price and costs.
        
        For entries:
        - Long: fill at ask + slippage (worse for buyer)
        - Short: fill at bid - slippage (worse for seller)
        
        Args:
            mid: Mid-market price
            direction: "long" or "short"
            lot_size: Order size in lots
            vol_proxy: Volatility proxy
            
        Returns:
            Tuple of (fill_price, commission, metadata)
        """
        bid, ask, spread = self.quote(mid, vol_proxy)
        slippage = self._compute_slippage_points(vol_proxy, lot_size)
        
        if direction == "long":
            # Buy at ask, slippage makes it worse (higher)
            fill = ask + slippage
        else:
            # Sell at bid, slippage makes it worse (lower)
            fill = bid - slippage
        
        commission = self.cfg.commission_per_lot * float(lot_size)
        
        # Track statistics
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
            "direction": direction,
            "lot_size": lot_size,
            "vol_proxy": vol_proxy,
        }
        
        return float(fill), float(commission), meta
    
    def fill_exit(
        self,
        mid: float,
        direction: str,
        lot_size: float,
        vol_proxy: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Compute exit fill price and costs.
        
        For exits (opposite of entry):
        - Long position exit: fill at bid - slippage (selling, worse for seller)
        - Short position exit: fill at ask + slippage (buying, worse for buyer)
        
        Args:
            mid: Mid-market price
            direction: Original position direction ("long" or "short")
            lot_size: Order size in lots
            vol_proxy: Volatility proxy
            
        Returns:
            Tuple of (fill_price, commission, metadata)
        """
        bid, ask, spread = self.quote(mid, vol_proxy)
        slippage = self._compute_slippage_points(vol_proxy, lot_size)
        
        if direction == "long":
            # Closing long = selling at bid, slippage makes it worse (lower)
            fill = bid - slippage
        else:
            # Closing short = buying at ask, slippage makes it worse (higher)
            fill = ask + slippage
        
        commission = self.cfg.commission_per_lot * float(lot_size)
        
        # Track statistics
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
            "direction": direction,
            "lot_size": lot_size,
            "vol_proxy": vol_proxy,
            "is_exit": True,
        }
        
        return float(fill), float(commission), meta
    
    def get_execution_stats(self) -> Dict[str, float]:
        """
        Get execution statistics.
        
        Returns:
            Dict with execution cost breakdowns
        """
        return {
            "total_spread_cost": self._total_spread_cost,
            "total_slippage_cost": self._total_slippage_cost,
            "total_commission": self._total_commission,
            "fill_count": self._fill_count,
            "avg_spread": self._total_spread_cost / max(self._fill_count, 1),
            "avg_slippage": self._total_slippage_cost / max(self._fill_count, 1),
            "avg_commission": self._total_commission / max(self._fill_count, 1),
        }


def create_execution_model_for_stage(
    stage_execution: "ExecutionDifficulty",  # From curriculum_config
    rng: np.random.Generator,
) -> ExecutionModel:
    """
    Create an ExecutionModel configured for a specific curriculum stage.
    
    This is a convenience function for curriculum integration.
    
    Args:
        stage_execution: ExecutionDifficulty from curriculum stage config
        rng: NumPy random generator
        
    Returns:
        Configured ExecutionModel
    """
    # Import here to avoid circular dependency
    from envs.curriculum_config import ExecutionDifficulty
    
    config = ExecutionConfig(
        base_spread_points=stage_execution.base_spread_points,
        spread_mult_range=stage_execution.spread_mult_range,
        max_spread_points=stage_execution.max_spread_points,
        slippage_points_sigma=stage_execution.slippage_points_sigma,
        slippage_mult_range=stage_execution.slippage_mult_range,
        max_slippage_points=stage_execution.max_slippage_points,
        commission_per_lot=stage_execution.commission_per_lot,
        latency_bars=stage_execution.latency_bars,
    )
    
    model = ExecutionModel(config, rng)
    
    # Set episode randomization if enabled
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
