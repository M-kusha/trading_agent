"""modules.market - shared utilities only.

UnifiedMarketModule and its components (fractal regime, liquidity heatmap,
theme detector, regime matrix, time risk) were removed on 2026-08-02. A 60-step
instrumented training run recorded zero invocations of any of them, and they
recomputed regime and volatility features that already exist in the tracked
feature CSVs the agent reads directly.

What remains is genuinely shared infrastructure that live code depends on:
CircuitBreaker (used by modules/utils/circuit_breaker_utils.py) plus the
state and metrics helpers.

Recover the removed code with:
    git log --diff-filter=D -- modules/market/market_module.py
"""

from .shared.circuit_breaker import CircuitBreaker, CircuitState

__all__ = [
    "CircuitBreaker",
    "CircuitState",
]

__version__ = "2.0.0"
