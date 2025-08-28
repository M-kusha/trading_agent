"""
Modern Trading Environment Package
Clean, zero-legacy architecture with SmartInfoBus v4.0
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

__docformat__ = "google"

# ---------------------------------------------------------------------
# Version (best-effort from package metadata; falls back to constant)
# ---------------------------------------------------------------------
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version  # py>=3.8
except Exception:  # pragma: no cover
    PackageNotFoundError = Exception  # type: ignore
    _pkg_version = None  # type: ignore

# Default version if metadata is unavailable (e.g., editable install)
_DEFAULT_VERSION = "4.0.0"
try:
    __version__ = _pkg_version(__package__ or "envs") if _pkg_version else _DEFAULT_VERSION
except Exception:
    __version__ = _DEFAULT_VERSION

# ---------------------------------------------------------------------
# Public API re-exports
#   - Use TYPE_CHECKING to keep static analyzers happy without side effects
# ---------------------------------------------------------------------
if TYPE_CHECKING:
    from .modern_env import ModernTradingEnv  # noqa: F401
    from .config import (  # noqa: F401
        TradingConfig,
        MarketState,
        EpisodeMetrics,
        ConfigPresets,
        ConfigFactory,
    )
else:
    from .modern_env import ModernTradingEnv
    from .config import (
        TradingConfig,
        MarketState,
        EpisodeMetrics,
        ConfigPresets,
        ConfigFactory,
    )

__all__: List[str] = [
    "ModernTradingEnv",
    "TradingConfig",
    "MarketState",
    "EpisodeMetrics",
    "ConfigPresets",
    "ConfigFactory",
    "__version__",
]
