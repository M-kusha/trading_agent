"""
train/__init__.py

Training package for PPO-based prop firm trading agent.

Subpackages:
- controllers: PID, entropy, and training health controllers
- callbacks: SB3 training callbacks for episode tracking and curriculum
"""

# Re-export commonly used items for convenience
from .controllers import PIDController, SmartEntropyController, TrainingHealthWatchdog
from .callbacks import VecEpisodeTradingCallback, CurriculumCheckpointCallback, CurriculumTrainingCallback

__all__ = [
    # Controllers
    "PIDController",
    "SmartEntropyController",
    "TrainingHealthWatchdog",
    # Callbacks
    "VecEpisodeTradingCallback",
    "CurriculumCheckpointCallback",
    "CurriculumTrainingCallback",
]
