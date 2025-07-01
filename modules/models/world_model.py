# ─────────────────────────────────────────────────────────────
# modules/models/world_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modules.core.core import Module
from typing import Dict, Any, Optional, Tuple

class RNNWorldModel(nn.Module, Module):
    def __init__(
        self,
        input_size: int = 2,  # Default for returns + volatility
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        device: str = "cpu",
        debug: bool = True
    ):
        nn.Module.__init__(self)  # Explicit parent init
        Module.__init__(self)
        
        self.input_size = input_size
        self.device = torch.device(device)
        self.debug = debug
        
        # Model architecture
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size, input_size)
        
        # Training components
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Move to device
        self.to(self.device)
        
        # For evolutionary use
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        
        # Tracking
        self.last_prediction = None
        self.prediction_error = 0.0
        self.trained = False

    def reset(self) -> None:
        """Reset tracking variables"""
        self.last_prediction = None
        self.prediction_error = 0.0

    def step(self, market_data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Process market data and update predictions"""
        if market_data is None:
            return
            
        # Extract price data
        prices = market_data.get('prices', {})
        if not prices:
            return
            
        # Calculate returns and volatility for prediction
        for inst, price_series in prices.items():
            if len(price_series) > 2:
                returns = np.diff(np.log(price_series[-20:]))  # Last 20 bars
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
                # Make prediction if trained
                if self.trained:
                    self.last_prediction = self.predict_next(returns, volatility)
                    
                break  # Just use first instrument for now
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict next values"""
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :])

    def fit(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
        seq_len: int = 50,
        batch_size: int = 64,
        epochs: int = 5
    ) -> float:
        """Train model on historical data - FIXED"""
        # Prepare sequences
        X, Y = [], []
        T = len(feature1)
        
        if T < seq_len + 1:
            if self.debug:
                print(f"[RNNWorldModel] Insufficient data: {T} < {seq_len + 1}")
            return float('inf')
        
        for i in range(seq_len, T - 1):
            seq = np.stack([feature1[i-seq_len:i], feature2[i-seq_len:i]], axis=1)
            tgt = np.array([feature1[i], feature2[i]], dtype=np.float32)
            X.append(torch.FloatTensor(seq))
            Y.append(torch.FloatTensor(tgt))
        
        if not X:
            return float('inf')

        # Create dataset - FIXED: No duplicate
        dataset = TensorDataset(torch.stack(X), torch.stack(Y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.train()
        final_loss = 0.0
        
        for ep in range(epochs):
            tot_loss = 0.0
            num_batches = 0
            
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                self.opt.zero_grad()
                pred = self(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.opt.step()
                tot_loss += loss.item() * xb.size(0)
                num_batches += xb.size(0)
            
            avg_loss = tot_loss / num_batches
            final_loss = avg_loss
            
            if self.debug:
                print(f"[RNNWorldModel] Epoch {ep+1}/{epochs} loss={avg_loss:.6f}")
        
        self.trained = True
        self.prediction_error = final_loss
        return final_loss

    def simulate(
        self,
        init_returns: np.ndarray,
        init_vol: np.ndarray,
        steps: int = 10
    ) -> np.ndarray:
        """Simulate future market scenarios"""
        seq_len = len(init_returns)
        
        if seq_len != len(init_vol):
            raise ValueError("Returns and volatility arrays must have same length")
        
        # Prepare history
        hist = np.stack([init_returns, init_vol], axis=1).astype(np.float32)
        predictions = []

        self.eval()
        with torch.no_grad():
            for _ in range(steps):
                # Use last seq_len points
                x = torch.tensor(hist[-seq_len:], dtype=torch.float32, device=self.device)
                x = x.unsqueeze(0)  # Add batch dimension
                
                # Predict next point
                pred = self(x)[0].cpu().numpy()
                predictions.append(pred)
                
                # Update history
                hist = np.vstack([hist, pred])

        return np.array(predictions, dtype=np.float32)
    
    def predict_next(self, returns: np.ndarray, volatility: float) -> Dict[str, float]:
        """Predict next return and volatility"""
        if len(returns) < 2:
            return {"return": 0.0, "volatility": volatility}
            
        # Prepare volatility array
        vol_array = np.full_like(returns, volatility)
        
        # Simulate one step
        try:
            pred = self.simulate(returns, vol_array, steps=1)
            return {
                "return": float(pred[0, 0]),
                "volatility": float(pred[0, 1]) if pred.shape[1] > 1 else volatility
            }
        except Exception as e:
            if self.debug:
                print(f"[RNNWorldModel] Prediction error: {e}")
            return {"return": 0.0, "volatility": volatility}

    def get_observation_components(self) -> np.ndarray:
        """Return prediction confidence and error metrics"""
        if self.last_prediction is not None:
            return np.array([
                self.last_prediction.get("return", 0.0),
                self.last_prediction.get("volatility", 0.02),
                self.prediction_error,
                float(self.trained)
            ], dtype=np.float32)
        return np.array([0.0, 0.02, 1.0, 0.0], dtype=np.float32)

    # ===================== NEUROEVOLUTION ============================
    def mutate(self, std: float = 0.05):
        """Mutate weights for evolution"""
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * std
                param.add_(noise.to(param.device))
        if self.debug:
            print(f"[RNNWorldModel] Mutated with std={std}")
            
    def crossover(self, other: "RNNWorldModel") -> "RNNWorldModel":
        """Create offspring through crossover"""
        if (self._hidden_size != other._hidden_size or 
            self._num_layers != other._num_layers):
            raise ValueError("Cannot crossover models with different architectures")
            
        child = RNNWorldModel(
            input_size=self.input_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
            device=str(self.device),
            debug=self.debug
        )
        
        with torch.no_grad():
            for p_child, p_self, p_other in zip(
                child.parameters(), self.parameters(), other.parameters()
            ):
                mask = torch.rand_like(p_child) > 0.5
                p_child.copy_(torch.where(mask, p_self, p_other))
                
        if self.debug:
            print("[RNNWorldModel] Crossover complete")
        return child

    def get_state(self) -> Dict[str, Any]:
        """Save complete state"""
        return {
            "model_state": self.state_dict(),
            "optimizer_state": self.opt.state_dict(),
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
            "trained": self.trained,
            "prediction_error": self.prediction_error,
            "last_prediction": self.last_prediction
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore complete state"""
        self.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            self.opt.load_state_dict(state["optimizer_state"])
        self._hidden_size = state.get("hidden_size", self._hidden_size)
        self._num_layers = state.get("num_layers", self._num_layers)
        self._dropout = state.get("dropout", self._dropout)
        self.trained = state.get("trained", False)
        self.prediction_error = state.get("prediction_error", 0.0)
        self.last_prediction = state.get("last_prediction")