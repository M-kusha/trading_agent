

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from envs.core.env_types import PropFirmConfig as TradingConfig

if TYPE_CHECKING:
    pass

def get_episode_metrics_class():
    from envs.curriculum.metrics import EpisodeMetrics
    return EpisodeMetrics


__all__ = ["EpisodeMetrics", "MarketState", "TradingConfig"]


class _EpisodeMetricsLazyLoader:
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

    timestamp: Any = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    spread: float = 0.0


    atr: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0
