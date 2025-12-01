import os
from datetime import datetime, timezone
import numpy as np

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
    return datetime.now(timezone.utc).isoformat()

# ═══════════════════════════════════════════════════════════════════════════
# Voting system constants and helper functions
# ═══════════════════════════════════════════════════════════════════════════
_LAYER_W = dict(
    liquidityheatmaplayer=2.0,      # LiquidityHeatmapLayer
    lhl=2.0,                        # Short form
    fractalregimeconfirmation=1.5,  # FractalRegimeConfirmation  
    frc=1.5,                        # Short form
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

# ═══════════════════════════════════════════════════════════════════════════
# MODE-AWARE GATE PARAMETERS
# Automatically switch between LIVE (conservative) and TRAINING (exploratory)
# ═══════════════════════════════════════════════════════════════════════════

# Global mode flag - set by TradingModeManager or config
_TRADING_MODE: str = "TRAINING"  # "LIVE" or "TRAINING"

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE PARAMETERS (Conservative - protect capital)
# ─────────────────────────────────────────────────────────────────────────────
_LIVE_PARAMS = {
    "SIG_K": 6.0,           # Steeper slope for sharper confidence cutoff
    "SIG_KNEE": 0.25,       # Higher knee = need more confidence to pass
    "BASE_GATE": 0.30,      # Higher base gate = harder to trigger trades
    "VOL_REF": 0.015,       # More sensitive to volatility
    "VOL_MULT_EXTREME": 1.8,  # Multiplier for extreme volatility
    "VOL_MULT_HIGH": 1.5,     # Multiplier for high volatility
    "VOL_MULT_ELEVATED": 1.2, # Multiplier for elevated volatility
    "CONSENSUS_DISCOUNT": 0.90,  # Only 10% reduction for strong consensus
}

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING MODE PARAMETERS (Exploratory - allow learning)
# ─────────────────────────────────────────────────────────────────────────────
_TRAINING_PARAMS = {
    "SIG_K": 4.0,           # Gentler slope for exploration
    "SIG_KNEE": 0.15,       # Lower knee = easier to pass
    "BASE_GATE": 0.15,      # Lower base gate = more trades for learning
    "VOL_REF": 0.02,        # Less sensitive to volatility
    "VOL_MULT_EXTREME": 1.4,  # Smaller multiplier
    "VOL_MULT_HIGH": 1.2,     # Smaller multiplier
    "VOL_MULT_ELEVATED": 1.1, # Smaller multiplier
    "CONSENSUS_DISCOUNT": 0.80,  # 20% reduction for strong consensus
}


def set_trading_mode(mode: str) -> None:
    """
    Set the global trading mode. Call this at startup based on config.
    
    Args:
        mode: "LIVE" for conservative real-money trading,
              "TRAINING" for exploratory learning mode
    """
    global _TRADING_MODE
    mode = mode.upper().strip()
    if mode not in ("LIVE", "TRAINING"):
        mode = "TRAINING"  # Default to safer exploratory mode
    _TRADING_MODE = mode


def get_trading_mode() -> str:
    """Get the current trading mode."""
    return _TRADING_MODE


def get_gate_params() -> dict:
    """Get the current gate parameters based on trading mode."""
    if _TRADING_MODE == "LIVE":
        return _LIVE_PARAMS.copy()
    return _TRAINING_PARAMS.copy()


# Legacy accessors (for backward compatibility) - now mode-aware
def _get_sig_k() -> float:
    return get_gate_params()["SIG_K"]

def _get_sig_knee() -> float:
    return get_gate_params()["SIG_KNEE"]

def _get_base_gate() -> float:
    return get_gate_params()["BASE_GATE"]

def _get_vol_ref() -> float:
    return get_gate_params()["VOL_REF"]

def _squash(c: float) -> float:
    """
    Mode-aware squashing function for confidence values.
    Uses steeper curve in LIVE mode, gentler in TRAINING.
    """
    params = get_gate_params()
    return 1.0 / (1.0 + np.exp(-params["SIG_K"] * (c - params["SIG_KNEE"])))


def _smart_gate(volatility: float, maj: int) -> float:
    """
    Mode-aware gate for trading decisions.
    
    LIVE mode: Conservative - protects capital with high thresholds
    TRAINING mode: Exploratory - allows more trades for learning
    
    Args:
        volatility: Current market volatility
        maj: Majority direction (+1 or -1)
    Returns:
        Gate threshold (higher = harder to pass)
    """
    params = get_gate_params()
    gate = params["BASE_GATE"]
    vol_ref = params["VOL_REF"]
    
    # Volatility-based scaling (more aggressive in LIVE mode)
    if volatility > vol_ref * 3:
        gate *= params["VOL_MULT_EXTREME"]
    elif volatility > vol_ref * 2:
        gate *= params["VOL_MULT_HIGH"]
    elif volatility > vol_ref * 1.5:
        gate *= params["VOL_MULT_ELEVATED"]
    
    # Reduce gate when majority agrees (smaller reduction in LIVE mode)
    if abs(maj) > 0:
        gate *= params["CONSENSUS_DISCOUNT"]
    
    return gate


def get_gate_info() -> dict:
    """
    Get current gate configuration for logging/debugging.
    Returns dict with mode and all active parameters.
    """
    return {
        "mode": _TRADING_MODE,
        "params": get_gate_params(),
        "description": "CONSERVATIVE" if _TRADING_MODE == "LIVE" else "EXPLORATORY",
    }