# envs/__init__.py
"""
Enhanced Trading Environment Package

A state-of-the-art PPO trading environment with modular architecture.
Clean separation of concerns for better maintainability and testing.
"""

from .config import TradingConfig, MarketState, EpisodeMetrics
from .env import EnhancedTradingEnv
from .shared_utils import TradingPipeline, UnifiedRiskManager

__version__ = "1.0.0"
__author__ = "AI Trading Team"

__all__ = [
    "EnhancedTradingEnv",
    "TradingConfig", 
    "MarketState",
    "EpisodeMetrics",
    "TradingPipeline",
    "UnifiedRiskManager",
    
]

# Package-level constants
DEFAULT_INSTRUMENTS = ["EUR/USD", "XAU/USD"]
DEFAULT_TIMEFRAMES = ["H1", "H4", "D1"]