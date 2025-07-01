# modules/voting/time_horizon_aligner.py

from typing import List
import numpy as np
from modules.core.core import Module


class TimeHorizonAligner(Module):
    def __init__(self, horizons: List[int]):
        self.horizons = np.asarray(horizons, np.float32)
        self.clock = 0
        
    def reset(self):
        self.clock = 0
        
    def step(self, **kwargs):
        self.clock += 1
        
    def get_observation_components(self) -> np.ndarray:
        return np.asarray([self.clock], np.float32)
        
    def apply(self, weights: np.ndarray) -> np.ndarray:
        """Apply time-based scaling to weights"""
        # Distance from each horizon
        distances = 1.0 / (1.0 + np.abs(self.clock - self.horizons))
        
        # Normalize
        distances = distances / distances.sum()
        
        # Apply to weights
        if len(weights) == len(distances):
            return weights * distances
        else:
            # Fallback if size mismatch
            return weights
