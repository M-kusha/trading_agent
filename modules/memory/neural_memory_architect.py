# ─────────────────────────────────────────────────────────────
# File: modules/memory/neural_memory_architect.py
# ─────────────────────────────────────────────────────────────

from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from modules.core.core import Module

class NeuralMemoryArchitect(Module):
    def __init__(
        self, 
        embed_dim: int = 32, 
        num_heads: int = 4, 
        max_len: int = 500,
        memory_decay: float = 0.95,  # New: memory importance decay
        debug: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.memory_decay = memory_decay
        self.debug = debug
        
        # Neural components
        self.encoder = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.value_head = nn.Linear(embed_dim, 1)  # Estimate memory value
        
        # Memory buffer with importance scores
        self.buffer = torch.zeros((0, embed_dim), dtype=torch.float32)
        self.importance_scores = torch.zeros(0, dtype=torch.float32)
        self.memory_metadata = []  # Store context for each memory
        
        # Move to CPU
        self.encoder = self.encoder.to("cpu")
        self.attn = self.attn.to("cpu")
        self.value_head = self.value_head.to("cpu")

    def reset(self):
        """Clear all memories"""
        self.buffer = torch.zeros((0, self.embed_dim), dtype=torch.float32)
        self.importance_scores = torch.zeros(0, dtype=torch.float32)
        self.memory_metadata = []

    def step(self, experience: Any = None, **kwargs) -> None:
        """Store new experience with enhanced context"""
        # Extract observation
        if experience is None:
            obs = np.zeros(self.embed_dim, dtype=np.float32)
            metadata = {"empty": True}
        elif isinstance(experience, dict):
            obs = experience.get('obs', np.zeros(self.embed_dim))
            metadata = {
                'reward': experience.get('reward', 0.0),
                'action': experience.get('action', None),
                'done': experience.get('done', False),
                'info': experience.get('info', {})
            }
            
            # Handle observation sizing
            if isinstance(obs, np.ndarray):
                if obs.size < self.embed_dim:
                    obs = np.pad(obs, (0, self.embed_dim - obs.size), mode='constant')
                elif obs.size > self.embed_dim:
                    obs = obs[:self.embed_dim]
            else:
                obs = np.zeros(self.embed_dim, dtype=np.float32)
        else:
            # Handle array input
            obs = np.asarray(experience, dtype=np.float32).flatten()
            if obs.size < self.embed_dim:
                obs = np.pad(obs, (0, self.embed_dim - obs.size), mode='constant')
            elif obs.size > self.embed_dim:
                obs = obs[:self.embed_dim]
            metadata = {"raw": True}

        # Encode observation
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = self.encoder(x)
            importance = torch.sigmoid(self.value_head(z)).squeeze()

        # Update buffer with importance-based replacement
        if len(self.buffer) < self.max_len:
            self.buffer = torch.cat([self.buffer, z], dim=0)
            self.importance_scores = torch.cat([
                self.importance_scores * self.memory_decay,  # Decay old memories
                importance.unsqueeze(0)
            ])
            self.memory_metadata.append(metadata)
        else:
            # Replace least important memory
            min_idx = torch.argmin(self.importance_scores).item()
            self.buffer[min_idx] = z.squeeze(0)
            self.importance_scores[min_idx] = importance
            self.memory_metadata[min_idx] = metadata

        if self.debug:
            print(f"[NMA] Stored memory {len(self.buffer)}/{self.max_len}, "
                  f"importance={importance.item():.3f}")

    def retrieve(self, query: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve most relevant memories with metadata"""
        if self.buffer.size(0) == 0:
            return {
                "embeddings": np.zeros((1, self.embed_dim), dtype=np.float32),
                "metadata": [{}],
                "scores": np.zeros(1, dtype=np.float32)
            }
        
        # Prepare query
        q = torch.from_numpy(query.astype(np.float32)).unsqueeze(0).unsqueeze(1)
        mem = self.buffer.unsqueeze(1)
        
        # Attention-based retrieval
        with torch.no_grad():
            out, attn_weights = self.attn(q, mem, mem)
            
        # Get top-k memories
        k = min(top_k, len(self.buffer))
        if attn_weights is not None:
            scores, indices = torch.topk(attn_weights.squeeze(), k)
            
            retrieved_embeddings = self.buffer[indices].cpu().numpy()
            retrieved_metadata = [self.memory_metadata[i] for i in indices]
            retrieved_scores = scores.cpu().numpy()
        else:
            # Fallback: return most recent
            retrieved_embeddings = self.buffer[-k:].cpu().numpy()
            retrieved_metadata = self.memory_metadata[-k:]
            retrieved_scores = self.importance_scores[-k:].cpu().numpy()
        
        if self.debug:
            print(f"[NMA] Retrieved {k} memories, top score={retrieved_scores[0]:.3f}")
        
        return {
            "embeddings": retrieved_embeddings,
            "metadata": retrieved_metadata,
            "scores": retrieved_scores
        }

    def get_observation_components(self) -> np.ndarray:
        """Return memory statistics"""
        if len(self.buffer) > 0:
            # Calculate memory statistics
            avg_importance = self.importance_scores.mean().item()
            memory_usage = len(self.buffer) / self.max_len
            
            # Get reward statistics from metadata
            rewards = [m.get('reward', 0.0) for m in self.memory_metadata 
                      if isinstance(m, dict) and 'reward' in m]
            avg_reward = np.mean(rewards) if rewards else 0.0
            
            return np.array([
                memory_usage,
                avg_importance,
                avg_reward,
                float(len(self.buffer))
            ], dtype=np.float32)
        
        return np.zeros(4, dtype=np.float32)

    def prune_memories(self, threshold: float = 0.1):
        """Remove low-importance memories"""
        if len(self.buffer) == 0:
            return
            
        # Keep memories above threshold
        mask = self.importance_scores > threshold
        self.buffer = self.buffer[mask]
        self.importance_scores = self.importance_scores[mask]
        self.memory_metadata = [m for i, m in enumerate(self.memory_metadata) if mask[i]]
        
        if self.debug:
            print(f"[NMA] Pruned to {len(self.buffer)} memories")

    def get_state(self) -> Dict[str, Any]:
        """Save complete state including metadata"""
        return {
            "encoder": self.encoder.state_dict(),
            "attn": self.attn.state_dict(),
            "value_head": self.value_head.state_dict(),
            "buffer": self.buffer.cpu().numpy().tolist(),
            "importance_scores": self.importance_scores.cpu().numpy().tolist(),
            "memory_metadata": self.memory_metadata,
            "memory_decay": self.memory_decay
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore complete state"""
        if "encoder" in state:
            self.encoder.load_state_dict(state["encoder"])
        if "attn" in state:
            self.attn.load_state_dict(state["attn"])
        if "value_head" in state:
            self.value_head.load_state_dict(state["value_head"])
        if "buffer" in state:
            self.buffer = torch.tensor(state["buffer"], dtype=torch.float32)
        if "importance_scores" in state:
            self.importance_scores = torch.tensor(state["importance_scores"], dtype=torch.float32)
        if "memory_metadata" in state:
            self.memory_metadata = state["memory_metadata"]
        self.memory_decay = state.get("memory_decay", 0.95)

    # ---- Neuroevolution Methods ----
    def mutate(self, noise_std: float = 0.05):
        """Mutate neural weights"""
        with torch.no_grad():
            for module in [self.encoder, self.attn, self.value_head]:
                for param in module.parameters():
                    param.data += noise_std * torch.randn_like(param.data)
        if self.debug:
            print("[NMA] Mutated weights")

    def crossover(self, other: "NeuralMemoryArchitect") -> "NeuralMemoryArchitect":
        """Create offspring through crossover"""
        child = NeuralMemoryArchitect(
            self.embed_dim, 
            self.attn.num_heads, 
            self.max_len,
            self.memory_decay,
            self.debug
        )
        
        # Crossover neural weights
        with torch.no_grad():
            for module_name in ['encoder', 'attn', 'value_head']:
                child_module = getattr(child, module_name)
                self_module = getattr(self, module_name)
                other_module = getattr(other, module_name)
                
                for (p_child, p_self, p_other) in zip(
                    child_module.parameters(),
                    self_module.parameters(),
                    other_module.parameters()
                ):
                    mask = torch.rand_like(p_self) > 0.5
                    p_child.data = torch.where(mask, p_self.data, p_other.data)
        
        # Inherit buffer from better parent (by avg importance)
        if self.importance_scores.mean() > other.importance_scores.mean():
            child.buffer = self.buffer.clone()
            child.importance_scores = self.importance_scores.clone()
            child.memory_metadata = self.memory_metadata.copy()
        else:
            child.buffer = other.buffer.clone()
            child.importance_scores = other.importance_scores.clone()
            child.memory_metadata = other.memory_metadata.copy()
            
        if self.debug:
            print("[NMA] Created child via crossover")
        return child