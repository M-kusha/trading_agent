# ─────────────────────────────────────────────────────────────
#  # modules/core/core.py
# ─────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.utils.info_bus import InfoBus

class Module(ABC):
    """Enhanced base module with InfoBus integration"""
    
    def __init__(self):
        self._initialized = True
        self._health_status = "OK"
        self._last_error = None
        
    @abstractmethod
    def reset(self) -> None:
        """Reset module to initial state"""
        ...
        
    @abstractmethod
    def step(self, info_bus: Optional['InfoBus'] = None, **kwargs) -> None:
        """Process one step with optional InfoBus"""
        ...
        
    @abstractmethod
    def get_observation_components(self) -> np.ndarray:
        """Return observation vector components"""
        ...
        
    def propose_action(self, obs: Any, info_bus: Optional['InfoBus'] = None) -> np.ndarray:
        """Propose trading action based on observation and InfoBus data"""
        # Default implementation
        if hasattr(obs, 'shape'):
            action_dim = obs.shape[0] // 2 if len(obs.shape) > 0 else 2
        else:
            action_dim = 2
        return np.zeros(action_dim, dtype=np.float32)
        
    def confidence(self, obs: Any, info_bus: Optional['InfoBus'] = None) -> float:
        """Return confidence in proposed action"""
        return 0.5
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get module health for monitoring"""
        return {
            "status": self._health_status,
            "module": self.__class__.__name__,
            "initialized": self._initialized,
            "last_error": str(self._last_error) if self._last_error else None,
            "details": self._get_health_details()
        }
        
    def _get_health_details(self) -> Optional[Dict[str, Any]]:
        """Override to provide module-specific health details"""
        return None
        
    def get_state(self) -> Dict[str, Any]:
        """Get module state for checkpointing"""
        return {}
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore module state from checkpoint"""
        pass
        
    # NEUROEVOLUTION EXTENSION
    def mutate(self, noise_std: float = 0.01) -> None:
        """Mutate module parameters for evolution"""
        pass
        
    def crossover(self, other: 'Module') -> 'Module':
        """Create offspring through crossover"""
        return self  # Default: no crossover
        
    def fitness(self) -> float:
        """Return fitness score for evolution"""
        return 0.0