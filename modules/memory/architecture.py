# ╔═════════════════════════════════════════════════════════════════════╗
# modules/architecture.py

import numpy as np
import torch
import torch.nn as nn
from modules.core.core import Module


class NeuralMemoryArchitect(Module):
    """
    Encodes experiences via a linear layer and retrieves via multihead attention.
    """
    def __init__(self, embed_dim: int=32, num_heads: int=4, max_len: int=500, debug=False):
        self.embed_dim = embed_dim
        self.max_len   = max_len
        self.debug     = debug
        self.encoder   = nn.Linear(embed_dim, embed_dim).to("cpu")
        self.attn      = nn.MultiheadAttention(embed_dim, num_heads).to("cpu")
        self.buffer    = torch.zeros((0, embed_dim), dtype=torch.float32)
    def reset(self):
        self.buffer = torch.zeros((0, self.embed_dim), dtype=torch.float32)
    def step(self, experience: np.ndarray):
        x = torch.from_numpy(experience.astype(np.float32)).unsqueeze(0)
        z = self.encoder(x)
        if len(self.buffer) < self.max_len:
            self.buffer = torch.cat([self.buffer, z], dim=0)
        else:
            self.buffer = torch.cat([self.buffer[1:], z], dim=0)
        if self.debug:
            print(f"[NMA] size={self.buffer.size(0)}")
    def retrieve(self, query: np.ndarray) -> np.ndarray:
        if self.buffer.size(0)==0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        q = torch.from_numpy(query.astype(np.float32)).unsqueeze(0).unsqueeze(1)  # (1,1,e)
        mem = self.buffer.unsqueeze(1)  # (L,1,e)
        out,_ = self.attn(q, mem, mem)  # (1,1,e)
        vec = out.squeeze(1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        if self.debug:
            print(f"[NMA] retrieved={vec}")
        return vec
    def get_observation_components(self)->np.ndarray:
        return np.zeros(self.embed_dim, dtype=np.float32)
    

    def get_state(self):
        return {
            "weights": self.model.state_dict(),
            "architecture": self.model_architecture,
        }

    def set_state(self, state):
        self.model.load_state_dict(state.get("weights", {}))
        self.model_architecture = state.get("architecture", None)   
 
