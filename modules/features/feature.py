# modules/feature.py

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from ..core.core import Module

class AdvancedFeatureEngine(Module):
    def __init__(self, window_sizes=[7, 14, 28], debug=False):
        self.windows = window_sizes
        self.debug = debug
        self.last_feats = np.zeros(len(self.windows) + 1, np.float32)

    def reset(self):
        self.last_feats[:] = 0.0          # ← ensures fresh episode starts clean

    def step(self, **kwargs): pass       # not used directly by pipeline

    def transform(self, price_series: np.ndarray) -> np.ndarray:
        feats = []
        for w in self.windows:
            block = price_series[-w:] if len(price_series) >= w else price_series
            feats.append(np.std(block).astype(np.float32))
        spread = np.mean(np.diff(price_series)).astype(np.float32) if len(price_series) > 1 else 0.0
        feats.append(spread)
        self.last_feats = np.array(feats, np.float32)
        return self.last_feats

    def get_observation_components(self) -> np.ndarray:
        return self.last_feats.copy()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.core import Module
from typing import Optional

class MultiScaleFeatureEngine(Module):
    def __init__(
        self,
        afe: "AdvancedFeatureEngine",
        embed_dim: int = 32,
        debug: bool = False,
    ):
        self.afe   = afe
        self.debug = debug

        # input dim = (# windows in AFE) + 1
        in_dim = len(afe.windows) + 1

        # ─── Replace simple Linear with Linear→ReLU→LayerNorm ───────────────
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.out  = nn.Linear(embed_dim, embed_dim)

        # device handling
        self.device = torch.device("cpu")
        for m in (self.proj, self.to_q, self.to_k, self.to_v, self.out):
            m.to(self.device)

        # last embedding placeholder
        self.last_embedding = np.zeros(embed_dim, dtype=np.float32)

    def reset(self):
        # nothing persistent to flush
        pass

    def step(
        self,
        price_h1: Optional[np.ndarray] = None,
        price_h4: Optional[np.ndarray] = None,
        price_d1: Optional[np.ndarray] = None,
    ):
        """
        3-scale fusion:
         - if a timeframe is missing, fall back sensibly:
           H4→H1, D1→H4→H1
        """
        def _to_ser(arr: Optional[np.ndarray]) -> np.ndarray:
            if arr is None:
                return np.zeros(1, dtype=np.float32)
            return np.asarray(arr, np.float32)

        # ensure numeric arrays
        h1 = _to_ser(price_h1)
        h4 = _to_ser(price_h4) if price_h4 is not None else h1
        d1 = _to_ser(price_d1) if price_d1 is not None else h4

        # extract AFE features at each scale
        f1 = self.afe.transform(h1)
        f4 = self.afe.transform(h4)
        fD = self.afe.transform(d1)

        # project & self-attend
        X = torch.stack([
            self.proj(torch.from_numpy(f1).to(self.device)),
            self.proj(torch.from_numpy(f4).to(self.device)),
            self.proj(torch.from_numpy(fD).to(self.device)),
        ], dim=0)  # shape (3, embed_dim)

        Q, K, V = self.to_q(X), self.to_k(X), self.to_v(X)
        scores  = (Q @ K.transpose(-2, -1)) / np.sqrt(Q.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        fused   = weights @ V

        # aggregate and output
        agg = fused.mean(dim=0)
        out = self.out(agg).detach().cpu().numpy().astype(np.float32)

        self.last_embedding = out
        if self.debug:
            print(f"[MSFE] embed={out}")

    def get_observation_components(self) -> np.ndarray:
        return self.last_embedding.copy()
