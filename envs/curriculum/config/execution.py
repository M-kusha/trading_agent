# envs/curriculum/config/execution.py
"""
Execution difficulty and data difficulty settings.

Contains:
- ExecutionDifficulty: Spread, slippage, commission settings
- DataDifficulty: Data filtering for curriculum-based training
- TransitionSettings: Settings for smooth stage transitions

Upgrades (Jan 2026):
- Added small compatibility bridge: if a user sets *randomization_range fields
  but leaves *mult_range at defaults, we copy over to prevent silent mis-wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ExecutionDifficulty:
    """Execution difficulty settings for a curriculum stage."""
    # Execution model (fills / quote model)
    base_spread_points: float = 0.05
    max_spread_points: float = 0.5

    slippage_points_sigma: float = 0.0
    max_slippage_points: float = 0.1

    commission_per_lot: float = 0.0
    latency_bars: int = 0

    # Domain randomization controls (wired to PropFirmConfig ranges by env.set_execution_params)
    enable_randomization: bool = False

    # Preferred names (these are what your env wiring reads)
    spread_mult_range: Tuple[float, float] = (0.9, 1.1)
    slippage_mult_range: Tuple[float, float] = (0.5, 1.5)
    latency_randomization_range: Tuple[int, int] = (0, 0)
    volatility_scale_range: Tuple[float, float] = (1.0, 1.0)

    # Backward/alternate names (kept for compatibility; if used, we bridge them)
    spread_randomization_range: Tuple[float, float] = (0.95, 1.05)
    slippage_randomization_range: Tuple[float, float] = (0.95, 1.05)

    # Spread shock events (for live-robustness testing; used only if ExecutionModel/ExecutionConfig supports it)
    spread_shock_enabled: bool = False
    spread_shock_probability: float = 0.02  # 2% of steps
    spread_shock_multiplier: float = 3.0    # 3x normal spread during shock

    def __post_init__(self) -> None:
        # If a stage config only sets the *randomization_range fields (legacy habit),
        # but leaves *mult_range at default values, copy across to avoid silent “no effect”.
        if self.spread_mult_range == (0.9, 1.1) and self.spread_randomization_range != (0.95, 1.05):
            self.spread_mult_range = self.spread_randomization_range
        if self.slippage_mult_range == (0.5, 1.5) and self.slippage_randomization_range != (0.95, 1.05):
            self.slippage_mult_range = self.slippage_randomization_range


@dataclass
class DataDifficulty:
    """
    Data difficulty settings for curriculum-based data filtering.

    Allows early stages to train on "easier" market conditions
    (clear trends, lower volatility) before introducing complex regimes.
    """
    volatility_percentile_range: Tuple[float, float] = (0.0, 1.0)
    min_trend_clarity: float = 0.0

    include_asian_session: bool = True
    include_london_session: bool = True
    include_ny_session: bool = True
    include_overlap_sessions: bool = True

    exclude_high_impact_news: bool = False
    exclude_market_open_close: bool = False

    prefer_recent_data: bool = False
    recent_data_weight: float = 1.0


@dataclass
class TransitionSettings:
    """
    Settings for smooth stage transitions.

    Prevents sudden destabilization when moving to harder stages.
    """
    lr_warmup_enabled: bool = True
    lr_warmup_factor: float = 0.3
    lr_warmup_steps: int = 10_000

    reward_blend_enabled: bool = True
    reward_blend_episodes: int = 20

    checkpoint_on_transition: bool = True
    transition_cooldown_episodes: int = 50


__all__ = [
    "ExecutionDifficulty",
    "DataDifficulty",
    "TransitionSettings",
]
