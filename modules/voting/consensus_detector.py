
# modules/voting/consensus_detector.py

from typing import List
import numpy as np
from modules.core.core import Module


class ConsensusDetector(Module):
    
    def __init__(self, n_members: int, threshold: float = 0.6):
        self.n_members = n_members
        self.threshold = threshold
        self.last_consensus = 0.0
        
    def reset(self):
        self.last_consensus = 0.0
        
    def step(self, **kwargs):
        pass
    
    def resize(self, n_members: int):
        self.n_members = n_members
        
    def get_observation_components(self) -> np.ndarray:
        return np.array([self.last_consensus], dtype=np.float32)
        
    def compute_consensus(self, actions: List[np.ndarray], confidences: List[float]) -> float:
        """
        Compute consensus level from member actions and confidences.
        
        Returns value between 0 (no consensus) and 1 (perfect consensus).
        """
        if actions is None or len(actions) < 2:
            return 0.5
            
        # Compute pairwise agreement
        agreements = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                # Cosine similarity between action vectors
                a1, a2 = actions[i], actions[j]
                norm1, norm2 = np.linalg.norm(a1), np.linalg.norm(a2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(a1, a2) / (norm1 * norm2)
                    # Weight by confidence
                    weight = confidences[i] * confidences[j]
                    agreements.append(similarity * weight)
                    
        if agreements:
            self.last_consensus = float(np.mean(agreements))
        else:
            self.last_consensus = 0.0
            
        return self.last_consensus

