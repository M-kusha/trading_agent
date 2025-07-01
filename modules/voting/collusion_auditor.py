
# modules/voting/collusion_auditor.py
from typing import List
import numpy as np

from modules.core.core import Module


class CollusionAuditor(Module):
    
    def __init__(self, n_members: int, window: int = 10, threshold: float = 0.9, debug: bool = False):
        self.n_members = n_members
        self.window = window
        self.threshold = threshold  # High threshold to avoid false positives
        self.debug = debug
        
        self.vote_history = []
        self.collusion_score = 0.0
        self.suspicious_pairs = set()
        
    def reset(self):
        self.vote_history.clear()
        self.collusion_score = 0.0
        self.suspicious_pairs.clear()
        
    def step(self, **kwargs):
        pass
        
    def get_observation_components(self) -> np.ndarray:
        return np.array([self.collusion_score], dtype=np.float32)
        
    def check_collusion(self, actions: List[np.ndarray]) -> float:
        """
        Check for suspicious voting patterns.
        Returns score between 0 (no collusion) and 1 (definite collusion).
        """
        if len(actions) < 2:
            return 0.0
            
        # Add to history
        self.vote_history.append(actions)
        if len(self.vote_history) > self.window:
            self.vote_history.pop(0)
            
        # Need enough history
        if len(self.vote_history) < 5:
            return 0.0
            
        # Check for pairs that always vote together
        pair_agreements = {}
        
        for votes in self.vote_history:
            for i in range(len(votes)):
                for j in range(i + 1, len(votes)):
                    pair = (i, j)
                    
                    # Compute agreement
                    v1, v2 = votes[i], votes[j]
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        
                        if pair not in pair_agreements:
                            pair_agreements[pair] = []
                        pair_agreements[pair].append(similarity)
                        
        # Find suspicious pairs
        self.suspicious_pairs.clear()
        suspicious_count = 0
        
        for pair, agreements in pair_agreements.items():
            avg_agreement = np.mean(agreements)
            if avg_agreement > self.threshold:
                self.suspicious_pairs.add(pair)
                suspicious_count += 1
                
        # Calculate overall score
        max_pairs = self.n_members * (self.n_members - 1) / 2
        self.collusion_score = suspicious_count / max_pairs if max_pairs > 0 else 0.0
        
        return self.collusion_score
