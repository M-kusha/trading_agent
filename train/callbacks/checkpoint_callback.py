
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class CurriculumCheckpointCallback(BaseCallback):

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


        self.save_freq = int(save_freq)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.curriculum_manager is None:
            return True


        if self.save_freq > 0 and (self.n_calls % self.save_freq == 0):
            self.save_path.mkdir(parents=True, exist_ok=True)

            step = int(self.num_timesteps)
            state_path = self.save_path / f"{self.name_prefix}_{step}_steps.json"
            try:
                self.curriculum_manager.save(state_path)
                if self.verbose >= 1:
                    logger.info(f"  📁 Curriculum state saved: {state_path.name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Curriculum state save failed: {e}")

        return True
