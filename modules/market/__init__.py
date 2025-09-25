"""
modules.market package exports (modern only)
"""

# Primary orchestrator exports
from .market_module import (
    UnifiedMarketModule,
    MarketConfig,
)

__all__ = [
    'UnifiedMarketModule',
    'MarketConfig',
]

# Version info
__version__ = '1.0.0'

# Re-export component classes for convenience
from .components import (
    FractalRegimeComponent,
    LiquidityHeatmapComponent,
    ThemeDetectorComponent,
    RegimeMatrixComponent,
    TimeRiskComponent,
)

__all__ += [
    'FractalRegimeComponent',
    'LiquidityHeatmapComponent',
    'ThemeDetectorComponent',
    'RegimeMatrixComponent',
    'TimeRiskComponent',
]

