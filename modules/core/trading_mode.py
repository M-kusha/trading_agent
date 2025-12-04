# modules/core/trading_mode.py
"""
Central Trading Mode Manager
═══════════════════════════════════════════════════════════════════════════════

Single point of control for switching between LIVE and TRAINING modes.
This module coordinates mode changes across all mode-aware subsystems:
  - Gate parameters (utils/get_dir.py)
  - Voting thresholds (modules/voting/core/constants.py)
  - Reward penalties (modules/reward/shared/reward_config.py)

Usage:
    from modules.core.trading_mode import TradingModeManager
    
    # At startup:
    TradingModeManager.set_mode("LIVE")  # For live trading
    TradingModeManager.set_mode("TRAINING")  # For training/backtesting
    
    # Check current mode:
    mode = TradingModeManager.get_mode()
    
    # Get mode info for logging:
    info = TradingModeManager.get_mode_info()
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    """Trading mode enumeration."""
    LIVE = "LIVE"
    TRAINING = "TRAINING"


@dataclass
class ModeInfo:
    """Information about the current trading mode."""
    mode: TradingMode
    description: str
    gate_style: str
    risk_tolerance: str
    exploration: str


class TradingModeManager:
    """
    Central manager for trading mode configuration.
    
    Coordinates mode changes across all mode-aware subsystems.
    Thread-safe singleton pattern.
    """
    
    _current_mode: TradingMode = TradingMode.TRAINING
    _initialized: bool = False
    
    # Mode descriptions for logging
    _MODE_INFO = {
        TradingMode.LIVE: ModeInfo(
            mode=TradingMode.LIVE,
            description="LIVE TRADING - Real money, conservative settings",
            gate_style="Conservative (high thresholds, strict filtering)",
            risk_tolerance="Low (protect capital at all costs)",
            exploration="Minimal (only high-confidence trades)",
        ),
        TradingMode.TRAINING: ModeInfo(
            mode=TradingMode.TRAINING,
            description="TRAINING MODE - Simulation, exploratory settings",
            gate_style="Permissive (lower thresholds, allow exploration)",
            risk_tolerance="Moderate (balance learning vs. losses)",
            exploration="High (encourage trade diversity)",
        ),
    }
    
    @classmethod
    def set_mode(cls, mode: str, silent: bool = False) -> None:
        """
        Set the global trading mode and propagate to all subsystems.
        
        Args:
            mode: "LIVE" or "TRAINING"
            silent: If True, suppress logging output
        """
        # Normalize mode string
        mode_str = mode.upper().strip()
        if mode_str not in ("LIVE", "TRAINING"):
            logger.warning(f"Invalid mode '{mode}', defaulting to TRAINING")
            mode_str = "TRAINING"
        
        new_mode = TradingMode(mode_str)
        old_mode = cls._current_mode
        cls._current_mode = new_mode
        
        # Propagate to all subsystems
        cls._propagate_mode(mode_str)
        
        if not silent:
            if old_mode != new_mode or not cls._initialized:
                info = cls._MODE_INFO[new_mode]
                logger.info("═" * 70)
                logger.info(f"🎯 TRADING MODE: {info.description}")
                logger.info(f"   Gate Style: {info.gate_style}")
                logger.info(f"   Risk Tolerance: {info.risk_tolerance}")
                logger.info(f"   Exploration: {info.exploration}")
                logger.info("═" * 70)
        
        cls._initialized = True
    
    @classmethod
    def _propagate_mode(cls, mode: str) -> None:
        """Propagate mode change to all subsystems."""
        
        # 1. Gate parameters (utils/get_dir.py)
        try:
            from utils.get_dir import set_trading_mode
            set_trading_mode(mode)
        except ImportError:
            logger.debug("Could not import set_trading_mode from utils.get_dir")
        except Exception as e:
            logger.warning(f"Failed to set gate trading mode: {e}")
        
        # 2. Voting thresholds (modules/voting/core/constants.py)
        try:
            from modules.voting.core.constants import set_voting_mode
            set_voting_mode(mode)
        except ImportError:
            logger.debug("Could not import set_voting_mode from voting constants")
        except Exception as e:
            logger.warning(f"Failed to set voting mode: {e}")
        
        # 3. Reward penalties (modules/reward/shared/reward_config.py)
        try:
            from modules.reward.shared.reward_config import set_reward_mode
            set_reward_mode(mode)
        except ImportError:
            logger.debug("Could not import set_reward_mode from reward config")
        except Exception as e:
            logger.warning(f"Failed to set reward mode: {e}")
    
    @classmethod
    def get_mode(cls) -> TradingMode:
        """Get the current trading mode."""
        return cls._current_mode
    
    @classmethod
    def get_mode_str(cls) -> str:
        """Get the current trading mode as string."""
        return cls._current_mode.value
    
    @classmethod
    def is_live(cls) -> bool:
        """Check if currently in LIVE mode."""
        return cls._current_mode == TradingMode.LIVE
    
    @classmethod
    def is_training(cls) -> bool:
        """Check if currently in TRAINING mode."""
        return cls._current_mode == TradingMode.TRAINING
    
    @classmethod
    def get_mode_info(cls) -> ModeInfo:
        """Get detailed information about the current mode."""
        return cls._MODE_INFO[cls._current_mode]
    
    @classmethod
    def get_all_parameters(cls) -> Dict[str, Any]:
        """
        Get all mode-aware parameters from all subsystems.
        Useful for logging/debugging.
        """
        result = {
            "mode": cls._current_mode.value,
            "is_live": cls.is_live(),
        }
        
        # Collect gate parameters
        try:
            from utils.get_dir import get_gate_params, get_trading_mode
            result["gate"] = {
                "mode": get_trading_mode(),
                "params": get_gate_params(),
            }
        except Exception:
            result["gate"] = {"error": "Could not retrieve"}
        
        # Collect voting thresholds
        try:
            from modules.voting.core.constants import get_thresholds, get_voting_mode
            result["voting"] = {
                "mode": get_voting_mode(),
                "thresholds": get_thresholds(),
            }
        except Exception:
            result["voting"] = {"error": "Could not retrieve"}
        
        # Collect reward parameters
        try:
            from modules.reward.shared.reward_config import get_reward_params, get_reward_mode
            result["reward"] = {
                "mode": get_reward_mode(),
                "params": get_reward_params(),
            }
        except Exception:
            result["reward"] = {"error": "Could not retrieve"}
        
        return result
    
    @classmethod
    def from_config(cls, config: Any, silent: bool = False) -> None:
        """
        Set mode based on TradingConfig object.
        
        IMPORTANT: Only upgrades to LIVE mode, never downgrades from LIVE.
        This prevents module initialization from resetting an already-set LIVE mode.
        
        Args:
            config: TradingConfig instance with live_mode attribute
            silent: If True, suppress logging
        """
        if hasattr(config, 'live_mode') and config.live_mode:
            cls.set_mode("LIVE", silent=silent)
        elif not cls.is_live():
            # Only set TRAINING if not already in LIVE mode
            cls.set_mode("TRAINING", silent=silent)
        # else: Already in LIVE mode, don't downgrade


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions for quick access
# ═══════════════════════════════════════════════════════════════════════════════

def set_live_mode(silent: bool = False) -> None:
    """Quick function to switch to LIVE mode."""
    TradingModeManager.set_mode("LIVE", silent=silent)


def set_training_mode(silent: bool = False) -> None:
    """Quick function to switch to TRAINING mode."""
    TradingModeManager.set_mode("TRAINING", silent=silent)


def is_live() -> bool:
    """Quick check if in LIVE mode."""
    return TradingModeManager.is_live()


def is_training() -> bool:
    """Quick check if in TRAINING mode."""
    return TradingModeManager.is_training()


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-initialize from environment variable if present
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_init() -> None:
    """Auto-initialize from TRADING_MODE environment variable."""
    import os
    mode = os.environ.get("TRADING_MODE", "").upper()
    if mode in ("LIVE", "TRAINING"):
        TradingModeManager.set_mode(mode, silent=True)


_auto_init()
