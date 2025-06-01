import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.core import Module
from typing import Optional

# ─────────────────────────────────────────────────────────────
class AdvancedFeatureEngine(Module):
    def __init__(self, window_sizes=[7, 14, 28], debug=False):
        self.windows = window_sizes
        self.debug = debug
        self.last_feats = np.zeros(len(self.windows) + 1, np.float32)

    def reset(self):
        self.last_feats[:] = 0.0

    def step(self, **kwargs):
        pass  # Not used directly, present for interface compatibility

    def transform(self, price_series: np.ndarray) -> np.ndarray:
        # Ensure valid input
        price_series = np.asarray(price_series, dtype=np.float32)
        feats = []
        for w in self.windows:
            block = price_series[-w:] if len(price_series) >= w else price_series
            if len(block) == 0 or np.any(np.isnan(block)) or np.any(np.isinf(block)):
                val = 0.0
            else:
                val = np.std(block).astype(np.float32)
            feats.append(np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6))
        if len(price_series) > 1:
            spread = np.mean(np.diff(price_series)).astype(np.float32)
        else:
            spread = 0.0
        feats.append(np.nan_to_num(spread, nan=0.0, posinf=1e6, neginf=-1e6))
        self.last_feats = np.array(feats, np.float32)
        if self.debug:
            print(f"[AFE] windows={self.windows}, feats={self.last_feats}")
        return self.last_feats

    def get_observation_components(self) -> np.ndarray:
        # Always returns valid shape, NaN/Inf-safe
        arr = np.nan_to_num(self.last_feats, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr.copy()

# ─────────────────────────────────────────────────────────────
class MultiScaleFeatureEngine(Module):
    def __init__(
        self,
        afe: "AdvancedFeatureEngine",
        embed_dim: int = 32,
        debug: bool = False,
    ):
        self.afe = afe
        self.debug = debug
        in_dim = len(afe.windows) + 1

        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        # Proper device handling (can add CUDA support if needed)
        self.device = torch.device("cpu")
        self._move_to_device()

        self.last_embedding = np.zeros(embed_dim, dtype=np.float32)

    def _move_to_device(self):
        for m in (self.proj, self.to_q, self.to_k, self.to_v, self.out):
            m.to(self.device)

    def reset(self):
        self.last_embedding[:] = 0.0

    def step(
        self,
        price_h1: Optional[np.ndarray] = None,
        price_h4: Optional[np.ndarray] = None,
        price_d1: Optional[np.ndarray] = None,
    ):
        def _to_ser(arr: Optional[np.ndarray]) -> np.ndarray:
            if arr is None or len(arr) == 0:
                return np.zeros(1, dtype=np.float32)
            arr = np.asarray(arr, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            return arr

        # Timeframe fallback logic
        h1 = _to_ser(price_h1)
        h4 = _to_ser(price_h4) if price_h4 is not None and len(price_h4) > 0 else h1
        d1 = _to_ser(price_d1) if price_d1 is not None and len(price_d1) > 0 else h4

        # Extract AFE features at each scale (robust to NaN/Inf)
        f1 = np.nan_to_num(self.afe.transform(h1), nan=0.0, posinf=1e6, neginf=-1e6)
        f4 = np.nan_to_num(self.afe.transform(h4), nan=0.0, posinf=1e6, neginf=-1e6)
        fD = np.nan_to_num(self.afe.transform(d1), nan=0.0, posinf=1e6, neginf=-1e6)

        # Project & self-attend (always safe for batch dim)
        X = torch.stack([
            self.proj(torch.from_numpy(f1).to(self.device)),
            self.proj(torch.from_numpy(f4).to(self.device)),
            self.proj(torch.from_numpy(fD).to(self.device)),
        ], dim=0)  # (3, embed_dim)

        Q, K, V = self.to_q(X), self.to_k(X), self.to_v(X)
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(Q.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        fused = weights @ V

        agg = fused.mean(dim=0)
        out = self.out(agg).detach().cpu().numpy().astype(np.float32)

        out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        self.last_embedding = out

        if self.debug:
            print(f"[MSFE] H1={f1}, H4={f4}, D1={fD}, embed={out}")

    def get_observation_components(self) -> np.ndarray:
        # Always returns valid shape, NaN/Inf safe
        arr = np.nan_to_num(self.last_embedding, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr.copy()
