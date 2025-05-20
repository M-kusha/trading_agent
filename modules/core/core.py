# modules/core.py

from abc import ABC, abstractmethod
import numpy as np

class Module(ABC):
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def step(self, **kwargs) -> None: ...
    @abstractmethod
    def get_observation_components(self) -> np.ndarray: ...

    # fallbacks for StrategyArbiter
    def propose_action(self, obs, /) -> np.ndarray:
        return np.zeros(obs.shape[0]//2, np.float32)
    def confidence(self, obs, /) -> float:
        return 0.5

    # NEW: Health status method (default implementation)
    def get_health_status(self) -> dict:
        return {
            "status": "OK",
            "module": self.__class__.__name__,
            "details": None
        }
