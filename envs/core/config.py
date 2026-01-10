# envs/config.py
"""
Compatibility layer - re-exports PropFirmConfig as TradingConfig.
All modules importing TradingConfig from here get the actual PropFirmConfig.

NOTE: EpisodeMetrics is defined in envs/curriculum/metrics.py (full 30+ field version).
      Do not add a stub here - import from envs.curriculum.metrics instead.

CRITICAL: Import from envs.core.env_types (NOT prop_firm_env) to avoid circular imports.
          prop_firm_env -> curriculum -> core.shared_utils -> core.config -> prop_firm_env (CIRCULAR!)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

# FIXED: Import from env_types directly to avoid circular import
# (prop_firm_env tries to import curriculum which eventually imports this file)
from envs.core.env_types import PropFirmConfig as TradingConfig

# Lazy import for EpisodeMetrics to avoid curriculum circular import
if TYPE_CHECKING:
    from envs.curriculum.metrics import EpisodeMetrics as _EpisodeMetrics

def get_episode_metrics_class():
    """Lazy getter for EpisodeMetrics to avoid circular imports."""
    from envs.curriculum.metrics import EpisodeMetrics
    return EpisodeMetrics

# Re-export for backwards compatibility - use lazy import for runtime access
# Note: Direct import would cause: curriculum -> shared_utils -> config -> curriculum (CIRCULAR)
__all__ = ["TradingConfig", "MarketState", "EpisodeMetrics"]

# Create a lazy accessor for EpisodeMetrics
class _EpisodeMetricsLazyLoader:
    """Lazy loader to avoid circular import while maintaining import compatibility."""
    _class = None
    
    def __call__(self, *args, **kwargs):
        if self._class is None:
            from envs.curriculum.metrics import EpisodeMetrics
            self._class = EpisodeMetrics
        return self._class(*args, **kwargs)
    
    def __getattr__(self, name):
        if self._class is None:
            from envs.curriculum.metrics import EpisodeMetrics
            self._class = EpisodeMetrics
        return getattr(self._class, name)

EpisodeMetrics = _EpisodeMetricsLazyLoader()


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
