"""
train/controllers/__init__.py

Re-exports for backward compatibility.
Import controllers from this package:
    from train.controllers import PIDController, SmartEntropyController, TrainingHealthWatchdog
"""

from .pid_controller import PIDController
from .entropy_controller import SmartEntropyController
from .health_watchdog import TrainingHealthWatchdog

__all__ = [
    "PIDController",
    "SmartEntropyController",
    "TrainingHealthWatchdog",
]
