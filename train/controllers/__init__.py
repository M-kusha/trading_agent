"""
train/controllers/__init__.py

Re-exports for backward compatibility.
Import controllers from this package:
    from train.controllers import PIDController, SmartEntropyController, TrainingHealthWatchdog
    from train.controllers import SmartLRController, SmartClipController
"""

from .pid_controller import PIDController
from .entropy_controller import SmartEntropyController
from .lr_controller import SmartLRController
from .clip_controller import SmartClipController
from .health_watchdog import TrainingHealthWatchdog

__all__ = [
    "PIDController",
    "SmartEntropyController",
    "SmartLRController",
    "SmartClipController",
    "TrainingHealthWatchdog",
]
