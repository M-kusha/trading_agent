import numpy as np
import torch
import torch.nn as nn
from modules.core.core import Module

class NeuralMemoryArchitect(Module):
    """
    Encodes experiences via a linear layer and retrieves via multihead attention.
    Evolvable: supports mutation/crossover of weights.
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
        # Could expose most recent or aggregated memory
        return np.zeros(self.embed_dim, dtype=np.float32)

    def get_state(self):
        # Save weights, buffer
        return {
            "encoder": self.encoder.state_dict(),
            "attn": self.attn.state_dict(),
            "buffer": self.buffer.detach().cpu().numpy().tolist(),
        }

    def set_state(self, state):
        if "encoder" in state:
            self.encoder.load_state_dict(state["encoder"])
        if "attn" in state:
            self.attn.load_state_dict(state["attn"])
        if "buffer" in state:
            self.buffer = torch.tensor(state["buffer"], dtype=torch.float32)

    # ---- Neuroevolution Methods ----

    def mutate(self, noise_std=0.05):
        # Add Gaussian noise to all weights
        for param in self.encoder.parameters():
            param.data += noise_std * torch.randn_like(param.data)
        for param in self.attn.parameters():
            param.data += noise_std * torch.randn_like(param.data)
        if self.debug:
            print("[NMA] Mutated weights")

    def crossover(self, other: "NeuralMemoryArchitect"):
        # Simple uniform crossover
        child = NeuralMemoryArchitect(self.embed_dim, self.attn.num_heads, self.max_len, self.debug)
        # Crossover encoder weights
        for (p_self, p_other, p_child) in zip(
            self.encoder.parameters(), other.encoder.parameters(), child.encoder.parameters()
        ):
            mask = torch.rand_like(p_self) > 0.5
            p_child.data = torch.where(mask, p_self.data, p_other.data)
        # Crossover attn weights
        for (p_self, p_other, p_child) in zip(
            self.attn.parameters(), other.attn.parameters(), child.attn.parameters()
        ):
            mask = torch.rand_like(p_self) > 0.5
            p_child.data = torch.where(mask, p_self.data, p_other.data)
        # Buffer: just take from self for simplicity (can be made fancier)
        child.buffer = self.buffer.clone()
        if self.debug:
            print("[NMA] Created child via crossover")
        return child
