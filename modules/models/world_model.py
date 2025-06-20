# modules/world_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modules.core.core import Module
import copy

class RNNWorldModel(nn.Module, Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        device: str = "cpu",
        debug=True
    ):
        super().__init__()
        self.device    = torch.device(device)
        self.debug     = debug
        self.lstm      = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head      = nn.Linear(hidden_size, input_size)
        self.opt       = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.to(self.device)
        # For evolutionary use: track initial hyperparams
        self._hidden_size = hidden_size
        self._num_layers  = num_layers
        self._dropout     = dropout

    def reset(self) -> None:
        pass  # No RNN state to clear for stateless batch ops

    def step(self, **kwargs) -> None:
        pass  # Placeholder implementation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RNNWorldModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_size).
        """
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :])
        


    def fit(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
        seq_len: int = 50,
        batch_size: int = 64,
        epochs: int = 5
    ):
        X, Y = [], []
        T = len(feature1)
        for i in range(seq_len, T-1):
            seq = np.stack([feature1[i-seq_len:i], feature2[i-seq_len:i]], axis=1)
            tgt = np.array([feature1[i], feature2[i]], dtype=np.float32)
            X.append(seq)
            Y.append(tgt)

        ds = TensorDataset(X, Y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        ds = TensorDataset(X, Y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.train()
        for ep in range(epochs):
            tot = 0.0
            for xb, yb in loader:
                self.opt.zero_grad()
                pred = self(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.opt.step()
                tot += loss.item() * xb.size(0)
            if self.debug:
                print(f"[RNNWorldModel] Epoch {ep+1}/{epochs} loss={tot/len(ds):.6f}")

    def simulate(
        self,
        init_returns: np.ndarray,
        init_vol: np.ndarray,
        steps: int = 10
    ) -> np.ndarray:
        seq_len = init_returns.shape[0]
        hist = np.stack([init_returns, init_vol], axis=1).astype(np.float32)
        out = []

        self.eval()
        with torch.no_grad():
            for _ in range(steps):
                x = torch.tensor(hist[-seq_len:], dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
                pred = self(x)[0].cpu().numpy()
                out.append(pred)
                hist = np.vstack([hist, pred])

        return np.array(out, dtype=np.float32)

    def get_observation_components(self) -> np.ndarray:
        """
        Returns an empty array because the world model does not directly expose observation components.
        This method is provided for interface compatibility with other modules.
        """
        return np.zeros(0, dtype=np.float32)

    # ===================== NEUROEVOLUTION ============================
    def mutate(self, std: float = 0.05):
        """
        Mutates weights in-place using Gaussian noise. Use carefully—best with copy().
        """
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * std
                param.add_(noise.to(param.device))
    def crossover(self, other: "RNNWorldModel"):
        """
        Crossover weights with another world model to create a new offspring.

        Note:
            Both models must have identical architectures (same layer sizes, number of layers, etc.)
            for the crossover to work correctly.
        """
        child = copy.deepcopy(self)
        with torch.no_grad():
            for p_child, p_self, p_other in zip(child.parameters(), self.parameters(), other.parameters()):
                mask = torch.rand_like(p_child) > 0.5
                p_child.copy_(torch.where(mask, p_self, p_other))
        if self.debug:
            print("[RNNWorldModel] Crossover complete")
        return child

    def get_state(self):
        """
        Save full weights and optimizer for checkpointing
        """
        return {
            "model_state": self.state_dict(),
            "optimizer_state": self.opt.state_dict(),
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "dropout": self._dropout
        }

    def set_state(self, state):
        """
        Loads the model and optimizer state from the provided dictionary.
        """
        self.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            self.opt.load_state_dict(state["optimizer_state"])
        self._hidden_size = state.get("hidden_size", self._hidden_size)
        self._num_layers  = state.get("num_layers", self._num_layers)
        self._dropout     = state.get("dropout", self._dropout)
