"""
Enhanced Trading Environment - Modern SmartInfoBus Version
Simplified wrapper around ModernTradingEnv (backward compatible)
"""
from __future__ import annotations

from typing import List

from .modern_env import ModernTradingEnv
from .config import TradingConfig

# Backward compatibility alias
EnhancedTradingEnv = ModernTradingEnv

__all__: List[str] = ["ModernTradingEnv", "EnhancedTradingEnv", "TradingConfig"]
