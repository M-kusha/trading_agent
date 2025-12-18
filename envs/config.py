# envs/config.py
"""
Compatibility layer - re-exports PropFirmConfig as TradingConfig.
All modules importing TradingConfig from here get the actual PropFirmConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Re-export PropFirmConfig as TradingConfig for backwards compatibility
# This way modules that `from envs.config import TradingConfig` will get PropFirmConfig
from envs.prop_firm_env import PropFirmConfig as TradingConfig

# Also export common names that modules might expect
__all__ = ["TradingConfig", "MarketState", "EpisodeMetrics"]


@dataclass  
class MarketState:
    """Market state snapshot - minimal version for compatibility."""
    
    timestamp: Any = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    spread: float = 0.0
    
    # Additional context
    atr: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0


@dataclass
class EpisodeMetrics:
    """Episode tracking metrics - minimal version."""
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
