"""
Modern Trading Environment Package
Clean, zero-legacy architecture with SmartInfoBus v4.0
"""

from .modern_env import ModernTradingEnv
from .config import TradingConfig, MarketState, EpisodeMetrics

# Main exports - no legacy aliases
__all__ = [
    'ModernTradingEnv',
    'TradingConfig', 
    'MarketState',
    'EpisodeMetrics'
]

# Version info
__version__ = "4.0.0"