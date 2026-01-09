"""
train/callbacks/checkpoint_callback.py

CurriculumCheckpointCallback - Saves curriculum state alongside model checkpoints.
Extracted from train_prop_firm.py for modularity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class CurriculumCheckpointCallback(BaseCallback):
    """
    Saves curriculum state alongside regular model checkpoints.
    
    This ensures we can fully resume training with:
    - Model weights
    - Curriculum stage
    - Rolling stats window
    - Episode counts
    """
    
    def __init__(
        self,
        curriculum_manager: Any,
        save_freq: int,
        save_path: str,
        name_prefix: str = "curriculum_state",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self._last_save = 0
    
    def _on_step(self) -> bool:
        if self.curriculum_manager is None:
            return True
        
        # Save at same frequency as CheckpointCallback
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.save_path.mkdir(parents=True, exist_ok=True)
            
            state_path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps.json"
            try:
                self.curriculum_manager.save(state_path)
                if self.verbose >= 1:
                    logger.info(f"  📁 Curriculum state saved: {state_path.name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Curriculum state save failed: {e}")
            
            self._last_save = self.num_timesteps
        
        return True
