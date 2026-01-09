# envs/core/__init__.py
"""
Core environment utilities and type definitions.

This package contains:
- env_types: CloseReason, RewardConfig, PropFirmConfig, PropPosition, TradeResult
- execution_model: ExecutionConfig, ExecutionModel
- shared_utils: safe_float, safe_int, clamp, timeframe_to_minutes, etc.
- config: TradingConfig compatibility layer
"""

from envs.core.env_types import (
    CloseReason,
    RewardConfig,
    PropFirmConfig,
    PropPosition,
    TradeResult,
    load_risk_policy,
)

from envs.core.execution_model import (
    ExecutionConfig,
    ExecutionModel,
)

from envs.core.shared_utils import (
    safe_float,
    safe_int,
    clamp,
    direction_sign,
    timeframe_to_minutes,
    bars_per_day_for_timeframe,
    TIMEFRAME_MINUTES,
    DEFAULT_PRIMARY_TIMEFRAME,
)

from envs.core.config import (
    TradingConfig,
    MarketState,
    EpisodeMetrics,
)

__all__ = [
    # Types
    "CloseReason",
    "RewardConfig", 
    "PropFirmConfig",
    "PropPosition",
    "TradeResult",
    "load_risk_policy",
    # Execution
    "ExecutionConfig",
    "ExecutionModel",
    # Utils
    "safe_float",
    "safe_int",
    "clamp",
    "direction_sign",
    "timeframe_to_minutes",
    "bars_per_day_for_timeframe",
    "TIMEFRAME_MINUTES",
    "DEFAULT_PRIMARY_TIMEFRAME",
    # Compat
    "TradingConfig",
    "MarketState",
    "EpisodeMetrics",
]
