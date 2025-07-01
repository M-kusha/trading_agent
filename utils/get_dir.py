import os
import numpy as np

try:
    from tomlkit import datetime
except ImportError:
    import datetime

# ── Universal Module import/stub ────────────────────────────
try:
    from modules.core.core import Module
except ImportError:
    class Module:
        def reset(self): pass
        def step(self, *a, **kw): pass
        def get_observation_components(self): pass

# ── Directory and UTC helpers ───────────────────────────────
def _ensure_dir(path: str):
    """
    Ensure the specified directory exists, creating it if necessary.
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def utcnow() -> str:
    """
    Get the current UTC time as an ISO8601 string.
    """
    return datetime.datetime.utcnow().isoformat()

# ═══════════════════════════════════════════════════════════════════════════
# Voting system constants and helper functions
# ═══════════════════════════════════════════════════════════════════════════
_LAYER_W = dict(
    liquidityheatmaplayer=2.0,      # LiquidityHeatmapLayer
    lhl=2.0,                        # Short form
    fractalregimeconfirmation=1.5,  # FractalRegimeConfirmation  
    frc=1.5,                        # Short form
    marketthemedetector=1.0,        # MarketThemeDetector
    mtd=1.0,                        # Short form
    markerregimeswitcher=1.0,       # MarketRegimeSwitcher
    switcher=1.0,                   # Short form
    # New additions for better voting
    positionmanager=1.5,            # Position manager has good judgment
    themeexpert=1.2,                # Theme expert
    regimebiasexpert=1.3,           # Regime expert
    seasonalityriskexpert=1.1,      # Seasonality expert
    metarlexpert=1.4,               # Meta-RL expert
    trademonitorvetoexpert=0.8,     # Veto expert (lower weight)
    dynamicriskcontroller=1.0,      # Risk controller
)

_SIG_K     = 4.0        # REDUCED from 8.0 - gentler slope
_SIG_KNEE  = 0.15       # REDUCED from 0.20 - lower threshold
_BASE_GATE = 0.15       # REDUCED from 0.25 - easier base gate
_VOL_REF   = 0.02       # INCREASED from 0.01 - less sensitive to volatility

def _squash(c: float) -> float:
    """
    Gentler squashing function for confidence values.
    """
    return 1.0 / (1.0 + np.exp(-_SIG_K * (c - _SIG_KNEE)))

def _smart_gate(volatility: float, maj: int) -> float:
    """
    Less restrictive gate that allows more trades through.
    Args:
        volatility: Current market volatility
        maj: Majority direction (+1 or -1)
    Returns:
        Gate threshold (lower = easier to pass)
    """
    gate = _BASE_GATE
    if volatility > _VOL_REF * 2:
        gate *= 1.2  # Only 20% increase instead of doubling
    if abs(maj) > 0:
        gate *= 0.8
    return gate
