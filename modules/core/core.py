# modules/core/core.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict

class Module(ABC):
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def step(self, **kwargs) -> None: ...
    @abstractmethod
    def get_observation_components(self) -> np.ndarray: ...

    def propose_action(self, obs: Any) -> np.ndarray:
        return np.zeros(getattr(obs, "shape", [2])[0] // 2, np.float32)

    def confidence(self, obs: Any) -> float:
        return 0.5

    def get_health_status(self) -> dict:
        return {
            "status": "OK",
            "module": self.__class__.__name__,
            "details": None
        }

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]):
        pass

    # NEUROEVOLUTION EXTENSION
    def mutate(self, noise_std=0.01):
        pass  # Override if module is evolvable

    def crossover(self, other):
        pass  # Override if module is evolvable
