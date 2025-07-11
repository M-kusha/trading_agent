"""
Enhanced Trading Environment - Modern SmartInfoBus Version
Simplified wrapper around ModernTradingEnv
"""
from .modern_env import ModernTradingEnv
from .config import TradingConfig

# For backward compatibility
EnhancedTradingEnv = ModernTradingEnv

# Export the modern environment as the default
__all__ = ['ModernTradingEnv', 'EnhancedTradingEnv', 'TradingConfig']