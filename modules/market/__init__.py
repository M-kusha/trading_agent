"""
modules.market package exports (modern only)
"""

# Primary orchestrator exports
from .market_module import (
    MarketConfig,
    UnifiedMarketModule,
)

__all__ = [
    'MarketConfig',
    'UnifiedMarketModule',
]

# Version info
__version__ = '1.0.0'

# Re-export component classes for convenience
from .components import (
    FractalRegimeComponent,
    LiquidityHeatmapComponent,
    RegimeMatrixComponent,
    ThemeDetectorComponent,
    TimeRiskComponent,
)

__all__ += [
    'FractalRegimeComponent',
    'LiquidityHeatmapComponent',
    'RegimeMatrixComponent',
    'ThemeDetectorComponent',
    'TimeRiskComponent',
]

