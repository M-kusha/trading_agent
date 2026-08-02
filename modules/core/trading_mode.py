

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    LIVE = "LIVE"
    TRAINING = "TRAINING"


@dataclass
class ModeInfo:
    mode: TradingMode
    description: str
    gate_style: str
    risk_tolerance: str
    exploration: str


class TradingModeManager:

    _current_mode: TradingMode = TradingMode.TRAINING
    _initialized: bool = False


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

        mode_str = mode.upper().strip()
        if mode_str not in ("LIVE", "TRAINING"):
            logger.warning(f"Invalid mode '{mode}', defaulting to TRAINING")
            mode_str = "TRAINING"

        new_mode = TradingMode(mode_str)
        old_mode = cls._current_mode
        cls._current_mode = new_mode


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


        try:
            from utils.get_dir import set_trading_mode
            set_trading_mode(mode)
        except ImportError:
            logger.debug("Could not import set_trading_mode from utils.get_dir")
        except Exception as e:
            logger.warning(f"Failed to set gate trading mode: {e}")


        try:
            from modules.voting.core.constants import set_voting_mode
            set_voting_mode(mode)
        except ImportError:
            logger.debug("Could not import set_voting_mode from voting constants")
        except Exception as e:
            logger.warning(f"Failed to set voting mode: {e}")


        try:
            from modules.reward.shared.reward_config import set_reward_mode
            set_reward_mode(mode)
        except ImportError:
            logger.debug("Could not import set_reward_mode from reward config")
        except Exception as e:
            logger.warning(f"Failed to set reward mode: {e}")


        try:
            from modules.utils.info_bus import InfoBusManager
            bus = InfoBusManager.get_instance()
            if hasattr(bus, 'set_live_mode'):
                bus.set_live_mode(mode == "LIVE")
        except ImportError:
            logger.debug("Could not import InfoBusManager")
        except Exception as e:
            logger.warning(f"Failed to set InfoBus live mode: {e}")

    @classmethod
    def get_mode(cls) -> TradingMode:
        return cls._current_mode

    @classmethod
    def get_mode_str(cls) -> str:
        return cls._current_mode.value

    @classmethod
    def is_live(cls) -> bool:
        return cls._current_mode == TradingMode.LIVE

    @classmethod
    def is_training(cls) -> bool:
        return cls._current_mode == TradingMode.TRAINING

    @classmethod
    def get_mode_info(cls) -> ModeInfo:
        return cls._MODE_INFO[cls._current_mode]

    @classmethod
    def get_all_parameters(cls) -> Dict[str, Any]:
        result = {
            "mode": cls._current_mode.value,
            "is_live": cls.is_live(),
        }


        try:
            from utils.get_dir import get_gate_params, get_trading_mode
            result["gate"] = {
                "mode": get_trading_mode(),
                "params": get_gate_params(),
            }
        except Exception:
            result["gate"] = {"error": "Could not retrieve"}


        try:
            from modules.voting.core.constants import get_thresholds, get_voting_mode
            result["voting"] = {
                "mode": get_voting_mode(),
                "thresholds": get_thresholds(),
            }
        except Exception:
            result["voting"] = {"error": "Could not retrieve"}


        try:
            from modules.reward.shared.reward_config import get_reward_mode, get_reward_params
            result["reward"] = {
                "mode": get_reward_mode(),
                "params": get_reward_params(),
            }
        except Exception:
            result["reward"] = {"error": "Could not retrieve"}

        return result

    @classmethod
    def from_config(cls, config: Any, silent: bool = False) -> None:
        if hasattr(config, 'live_mode') and config.live_mode:
            cls.set_mode("LIVE", silent=silent)
        elif not cls.is_live():

            cls.set_mode("TRAINING", silent=silent)


def set_live_mode(silent: bool = False) -> None:
    TradingModeManager.set_mode("LIVE", silent=silent)


def set_training_mode(silent: bool = False) -> None:
    TradingModeManager.set_mode("TRAINING", silent=silent)


def is_live() -> bool:
    return TradingModeManager.is_live()


def is_training() -> bool:
    return TradingModeManager.is_training()


def _auto_init() -> None:
    import os
    mode = os.environ.get("TRADING_MODE", "").upper()
    if mode in ("LIVE", "TRAINING"):
        TradingModeManager.set_mode(mode, silent=True)


_auto_init()
