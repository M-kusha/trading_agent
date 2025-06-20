import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.core import Module
from typing import Optional, Union, List
import logging

# ─────────────────────────────────────────────────────────────
class AdvancedFeatureEngine(Module):
    """
    FIXED: Now properly handles price data and generates meaningful features
    even with limited history.
    """
    def __init__(self, window_sizes=[7, 14, 28], debug=True):
        self.windows = window_sizes
        self.debug = debug
        self.last_feats = np.zeros(len(self.windows) * 4 + 4, np.float32)  # More features
        self.price_buffer = []  # Store recent prices for bootstrapping
        self.max_buffer_size = max(window_sizes) + 10
        
        # Setup logging
        self.logger = logging.getLogger("AdvancedFeatureEngine")
        if self.debug and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def reset(self):
        self.last_feats[:] = 0.0
        self.price_buffer.clear()

    def step(self, **kwargs):
        """Update price buffer if price data provided"""
        # Try to extract price from various possible keys
        for key in ['price', 'prices', 'close', 'price_series']:
            if key in kwargs:
                prices = kwargs[key]
                if isinstance(prices, (list, np.ndarray)):
                    self._update_buffer(prices)
                elif isinstance(prices, (int, float)):
                    self._update_buffer([prices])

    def _update_buffer(self, prices: Union[List, np.ndarray]):
        """Update internal price buffer"""
        if isinstance(prices, np.ndarray):
            prices = prices.tolist()
        
        # Add new prices to buffer
        for p in prices:
            if np.isfinite(p) and p > 0:
                self.price_buffer.append(float(p))
                
        # Trim buffer to max size
        if len(self.price_buffer) > self.max_buffer_size:
            self.price_buffer = self.price_buffer[-self.max_buffer_size:]

    def transform(self, price_series: np.ndarray) -> np.ndarray:
        """
        Extract features from price series with better handling of edge cases
        """
        # Ensure valid input
        price_series = np.asarray(price_series, dtype=np.float32)
        
        # Remove NaN/Inf values
        valid_mask = np.isfinite(price_series) & (price_series > 0)
        valid_prices = price_series[valid_mask]
        
        # If no valid prices in input, try to use buffer
        if len(valid_prices) == 0:
            if self.price_buffer:
                valid_prices = np.array(self.price_buffer[-28:], dtype=np.float32)
                if self.debug:
                    self.logger.debug(f"Using buffer prices: {len(valid_prices)} values")
            else:
                # Generate synthetic prices for bootstrapping
                valid_prices = self._generate_synthetic_prices()
                if self.debug:
                    self.logger.debug("Using synthetic prices for bootstrapping")
        
        # Update buffer with valid prices
        if len(valid_prices) > 0:
            self._update_buffer(valid_prices)
        
        # Extract features
        feats = []
        
        for w in self.windows:
            if len(valid_prices) >= w:
                window_prices = valid_prices[-w:]
            else:
                # Pad with mean if not enough data
                if len(valid_prices) > 0:
                    mean_price = np.mean(valid_prices)
                    padding = [mean_price] * (w - len(valid_prices))
                    window_prices = np.concatenate([padding, valid_prices])
                else:
                    window_prices = np.ones(w) * 1.0
            
            # Calculate various statistics
            # 1. Volatility (std)
            vol = np.std(window_prices) if len(window_prices) > 1 else 0.01
            feats.append(float(vol))
            
            # 2. Return
            ret = (window_prices[-1] - window_prices[0]) / window_prices[0] if window_prices[0] > 0 else 0.0
            feats.append(float(ret))
            
            # 3. Mean reversion indicator
            mean_price = np.mean(window_prices)
            if mean_price > 0:
                mean_rev = (window_prices[-1] - mean_price) / mean_price
            else:
                mean_rev = 0.0
            feats.append(float(mean_rev))
            
            # 4. Trend strength (linear regression slope)
            if len(window_prices) > 1:
                x = np.arange(len(window_prices))
                # Normalize prices to prevent numerical issues
                norm_prices = window_prices / window_prices[0] if window_prices[0] > 0 else window_prices
                if np.std(x) > 0 and np.std(norm_prices) > 0:
                    slope = np.polyfit(x, norm_prices, 1)[0]
                else:
                    slope = 0.0
            else:
                slope = 0.0
            feats.append(float(slope))
        
        # Global features
        if len(valid_prices) > 1:
            # 1. Spread (recent volatility)
            recent_diffs = np.diff(valid_prices[-10:]) if len(valid_prices) > 10 else np.diff(valid_prices)
            spread = np.mean(np.abs(recent_diffs)) if len(recent_diffs) > 0 else 0.0
            feats.append(float(spread))
            
            # 2. Momentum
            if len(valid_prices) >= 5:
                momentum = (valid_prices[-1] - valid_prices[-5]) / valid_prices[-5] if valid_prices[-5] > 0 else 0.0
            else:
                momentum = 0.0
            feats.append(float(momentum))
            
            # 3. Volatility ratio (short/long)
            short_vol = np.std(valid_prices[-7:]) if len(valid_prices) >= 7 else 0.01
            long_vol = np.std(valid_prices) if len(valid_prices) > 1 else 0.01
            vol_ratio = short_vol / (long_vol + 1e-8)
            feats.append(float(vol_ratio))
            
            # 4. Price position (where current price is relative to recent range)
            high = np.max(valid_prices[-20:]) if len(valid_prices) >= 20 else np.max(valid_prices)
            low = np.min(valid_prices[-20:]) if len(valid_prices) >= 20 else np.min(valid_prices)
            if high > low:
                price_pos = (valid_prices[-1] - low) / (high - low)
            else:
                price_pos = 0.5
            feats.append(float(price_pos))
        else:
            # Default values for global features
            feats.extend([0.0, 0.0, 1.0, 0.5])
        
        # Ensure all features are finite
        feats = [np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=-1.0) for f in feats]
        
        self.last_feats = np.array(feats, np.float32)
        
        if self.debug:
            self.logger.debug(f"Extracted {len(feats)} features from {len(valid_prices)} prices")
            self.logger.debug(f"Features: vol={feats[0]:.4f}, ret={feats[1]:.4f}, momentum={feats[-4]:.4f}")
        
        return self.last_feats

    def _generate_synthetic_prices(self, n=30):
        """Generate synthetic prices for bootstrapping"""
        # Start with a base price
        base_price = 1.0
        
        # Generate random walk
        returns = np.random.normal(0, 0.01, n)  # 1% daily volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        return prices.astype(np.float32)

    def get_observation_components(self) -> np.ndarray:
        """Always returns valid shape, NaN/Inf-safe"""
        arr = np.nan_to_num(self.last_feats, nan=0.0, posinf=1.0, neginf=-1.0)
        return arr.copy()

# ─────────────────────────────────────────────────────────────
class MultiScaleFeatureEngine(Module):
    """
    FIXED: Better handling of empty inputs and more robust feature fusion
    """
    def __init__(
        self,
        afe: "AdvancedFeatureEngine",
        embed_dim: int = 32,
        debug: bool = False,
    ):
        self.afe = afe
        self.debug = debug
        self.embed_dim = embed_dim
        
        # Calculate actual input dimension based on AFE
        self.in_dim = len(afe.windows) * 4 + 4  # 4 features per window + 4 global
        
        # Update neural network to match correct input size
        self.proj = nn.Sequential(
            nn.Linear(self.in_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),  # Add dropout for regularization
        )
        
        # Attention mechanism
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection with residual
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Initialize weights
        self._init_weights()
        
        # Device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()
        
        self.last_embedding = np.zeros(embed_dim, dtype=np.float32)
        
        # Setup logging
        self.logger = logging.getLogger("MultiScaleFeatureEngine")
        if self.debug and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in [self.proj, self.to_q, self.to_k, self.to_v, self.out]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _move_to_device(self):
        for m in [self.proj, self.to_q, self.to_k, self.to_v, self.out]:
            m.to(self.device)

    def reset(self):
        self.last_embedding[:] = 0.0
        self.afe.reset()

    def step(
        self,
        price_h1: Optional[np.ndarray] = None,
        price_h4: Optional[np.ndarray] = None,
        price_d1: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Process multi-scale price data with better error handling"""
        
        def _to_series(arr: Optional[np.ndarray], name: str) -> np.ndarray:
            """Convert input to valid price series"""
            if arr is None:
                if self.debug:
                    self.logger.debug(f"{name} is None")
                return np.array([], dtype=np.float32)
                
            arr = np.asarray(arr, dtype=np.float32)
            
            # Remove NaN/Inf
            valid_mask = np.isfinite(arr) & (arr > 0)
            valid_arr = arr[valid_mask]
            
            if self.debug and len(valid_arr) < len(arr):
                self.logger.debug(f"{name}: {len(valid_arr)}/{len(arr)} valid prices")
                
            return valid_arr

        # Convert inputs to valid series
        h1 = _to_series(price_h1, "H1")
        h4 = _to_series(price_h4, "H4")
        d1 = _to_series(price_d1, "D1")
        
        # If all inputs are empty, try to use AFE's buffer
        if len(h1) == 0 and len(h4) == 0 and len(d1) == 0:
            if self.afe.price_buffer:
                # Use buffer as H1 data
                h1 = np.array(self.afe.price_buffer[-28:], dtype=np.float32)
                if self.debug:
                    self.logger.debug(f"Using AFE buffer: {len(h1)} prices")
            else:
                # Generate synthetic data for bootstrapping
                h1 = self.afe._generate_synthetic_prices(30)
                if self.debug:
                    self.logger.debug("Using synthetic prices for all timeframes")
        
        # Use cascading fallback: D1 -> H4 -> H1
        if len(d1) == 0:
            d1 = h4 if len(h4) > 0 else h1
        if len(h4) == 0:
            h4 = h1
        
        # Extract features at each scale
        with torch.no_grad():
            f1 = self.afe.transform(h1)
            f4 = self.afe.transform(h4)
            fD = self.afe.transform(d1)
            
            # Ensure correct dimensions
            if f1.shape[0] != self.in_dim:
                self.logger.warning(f"Feature dimension mismatch: {f1.shape[0]} vs {self.in_dim}")
                # Pad or truncate
                if f1.shape[0] < self.in_dim:
                    f1 = np.pad(f1, (0, self.in_dim - f1.shape[0]))
                    f4 = np.pad(f4, (0, self.in_dim - f4.shape[0]))
                    fD = np.pad(fD, (0, self.in_dim - fD.shape[0]))
                else:
                    f1 = f1[:self.in_dim]
                    f4 = f4[:self.in_dim]
                    fD = fD[:self.in_dim]
            
            # Convert to tensors
            f1_t = torch.from_numpy(f1).float().to(self.device)
            f4_t = torch.from_numpy(f4).float().to(self.device)
            fD_t = torch.from_numpy(fD).float().to(self.device)
            
            # Project features
            x1 = self.proj(f1_t)
            x4 = self.proj(f4_t)
            xD = self.proj(fD_t)
            
            # Stack for attention
            X = torch.stack([x1, x4, xD], dim=0).unsqueeze(0)  # (1, 3, embed_dim)
            
            # Self-attention
            Q = self.to_q(X)
            K = self.to_k(X)
            V = self.to_v(X)
            
            # Scaled dot-product attention
            d_k = Q.shape[-1]
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
            weights = F.softmax(scores, dim=-1)
            attended = torch.matmul(weights, V)
            
            # Average pooling across timeframes
            pooled = attended.mean(dim=1).squeeze(0)  # (embed_dim,)
            
            # Output projection with residual connection
            out = self.out(pooled) + pooled
            
            # Convert to numpy
            out_np = out.detach().cpu().numpy().astype(np.float32)
            
            # Ensure finite values
            out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=-1.0)
            
            self.last_embedding = out_np
            
            if self.debug:
                self.logger.debug(f"Processed features: H1_vol={f1[0]:.4f}, H4_vol={f4[0]:.4f}, D1_vol={fD[0]:.4f}")
                self.logger.debug(f"Embedding norm: {np.linalg.norm(out_np):.4f}")

    def get_observation_components(self) -> np.ndarray:
        """Always returns valid shape, NaN/Inf safe"""
        arr = np.nan_to_num(self.last_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        return arr.copy()
        
    def get_state(self):
        """Get state for serialization"""
        return {
            "last_embedding": self.last_embedding.tolist(),
            "afe_buffer": self.afe.price_buffer,
            "afe_last_feats": self.afe.last_feats.tolist(),
        }
        
    def set_state(self, state):
        """Restore state from serialization"""
        if "last_embedding" in state:
            self.last_embedding = np.array(state["last_embedding"], dtype=np.float32)
        if "afe_buffer" in state:
            self.afe.price_buffer = state["afe_buffer"]
        if "afe_last_feats" in state:
            self.afe.last_feats = np.array(state["afe_last_feats"], dtype=np.float32)