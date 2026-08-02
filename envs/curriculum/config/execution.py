

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ExecutionDifficulty:

    base_spread_points: float = 0.05
    max_spread_points: float = 0.5

    slippage_points_sigma: float = 0.0
    max_slippage_points: float = 0.1

    commission_per_lot: float = 0.0
    latency_bars: int = 0


    use_data_spread: bool = False


    data_spread_scale: float = 1.0


    enable_randomization: bool = False


    spread_mult_range: Tuple[float, float] = (0.9, 1.1)
    slippage_mult_range: Tuple[float, float] = (0.5, 1.5)
    latency_randomization_range: Tuple[int, int] = (0, 0)
    volatility_scale_range: Tuple[float, float] = (1.0, 1.0)


    spread_randomization_range: Tuple[float, float] = (0.95, 1.05)
    slippage_randomization_range: Tuple[float, float] = (0.95, 1.05)


    spread_shock_enabled: bool = False
    spread_shock_probability: float = 0.02
    spread_shock_multiplier: float = 3.0

    def __post_init__(self) -> None:


        if self.spread_mult_range == (0.9, 1.1) and self.spread_randomization_range != (0.95, 1.05):
            self.spread_mult_range = self.spread_randomization_range
        if self.slippage_mult_range == (0.5, 1.5) and self.slippage_randomization_range != (0.95, 1.05):
            self.slippage_mult_range = self.slippage_randomization_range


@dataclass
class DataDifficulty:
    volatility_percentile_range: Tuple[float, float] = (0.0, 1.0)
    min_trend_clarity: float = 0.0
    max_trend_clarity: float = 1.0

    include_asian_session: bool = True
    include_london_session: bool = True
    include_ny_session: bool = True
    include_overlap_sessions: bool = True

    exclude_high_impact_news: bool = False
    exclude_market_open_close: bool = False

    prefer_recent_data: bool = False
    recent_data_weight: float = 1.0


    allowed_regimes: Optional[List[str]] = None
    regime_sampling_weights: Dict[str, float] = field(default_factory=dict)
    trend_clarity_threshold: float = 0.40
    high_volatility_threshold: float = 0.70
    low_volatility_threshold: float = 0.30
    news_volatility_threshold: float = 0.90


    include_setup_maturity_metrics: bool = False


@dataclass
class TransitionSettings:
    lr_warmup_enabled: bool = True
    lr_warmup_factor: float = 0.3
    lr_warmup_steps: int = 10_000

    reward_blend_enabled: bool = True
    reward_blend_episodes: int = 20

    checkpoint_on_transition: bool = True
    transition_cooldown_episodes: int = 50


__all__ = [
    "DataDifficulty",
    "ExecutionDifficulty",
    "TransitionSettings",
]
