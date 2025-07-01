# ─────────────────────────────────────────────────────────────
# File: modules/features/multiscale_feature_engine.py
# ─────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
from modules.features.advanced_feature_engine import AdvancedFeatureEngine
from ..core.core import Module
from typing import Optional
import logging


class MultiScaleFeatureEngine(Module):

    # multiscale_feature_engine.py
    """
    Fuses three time-scales (H1, H4, D1) into a fixed 32-dim embedding.
    Works with any observation size thanks to the *constant* in_dim and
    uses PyTorch on CPU/GPU transparently.
    """

    def __init__(self, afe: AdvancedFeatureEngine, embed_dim: int = 32, debug: bool = False):
        self.afe = afe
        self.debug = debug

        # feature dimensions
        self.in_dim = afe.out_dim
        self.out_dim = embed_dim

        # neural layers
        self.proj = nn.Sequential(
            nn.Linear(self.in_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
        )
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self._init_weights()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()

        self.last_embedding = np.zeros(self.out_dim, dtype=np.float32)

        # logging
        self.logger = logging.getLogger("MultiScaleFeatureEngine")
        if self.debug and not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.DEBUG)

    # ── lifecycle ────────────────────────────────────────────
    def reset(self):
        self.last_embedding[:] = 0.0
        self.afe.reset()

    # ── main step ────────────────────────────────────────────
    def step(
        self,
        price_h1: Optional[np.ndarray] = None,
        price_h4: Optional[np.ndarray] = None,
        price_d1: Optional[np.ndarray] = None,
        **kwargs,
    ):
        def _valid_series(arr: Optional[np.ndarray]) -> np.ndarray:
            if arr is None:
                return np.empty(0, dtype=np.float32)
            arr = np.asarray(arr, dtype=np.float32)
            return arr[np.isfinite(arr) & (arr > 0)]

        h1 = _valid_series(price_h1)
        h4 = _valid_series(price_h4)
        d1 = _valid_series(price_d1)

        if h1.size == h4.size == d1.size == 0:
            if self.afe.price_buffer:
                h1 = np.array(self.afe.price_buffer[-28:], dtype=np.float32)
                if self.debug:
                    self.logger.debug(f"Using AFE buffer: {h1.size} prices")
            else:
                h1 = self.afe._generate_synthetic_prices(30)
                if self.debug:
                    self.logger.debug("Using synthetic prices for all timeframes")

        if d1.size == 0:
            d1 = h4 if h4.size else h1
        if h4.size == 0:
            h4 = h1

        # feature extraction
        with torch.no_grad():
            f1 = self.afe.transform(h1)
            f4 = self.afe.transform(h4)
            fD = self.afe.transform(d1)

            # strict check
            assert f1.size == self.in_dim, (
                f"AFE output {f1.size} dims, expected {self.in_dim}"
            )

            # tensors
            f1_t = torch.from_numpy(f1).to(self.device)
            f4_t = torch.from_numpy(f4).to(self.device)
            fD_t = torch.from_numpy(fD).to(self.device)

            x1, x4, xD = self.proj(f1_t), self.proj(f4_t), self.proj(fD_t)
            X = torch.stack((x1, x4, xD), dim=0).unsqueeze(0)  # (1, 3, d)

            # scaled-dot attention
            Q, K, V = self.to_q(X), self.to_k(X), self.to_v(X)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.out_dim)
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, V)

            pooled = attended.mean(dim=1).squeeze(0)
            out_vec = self.out(pooled) + pooled

            out_np = out_vec.cpu().numpy().astype(np.float32)
            self.last_embedding = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=-1.0)

            if self.debug:
                self.logger.debug(
                    f"Embedding norm: {np.linalg.norm(self.last_embedding):.4f}"
                )

    # ── helpers ─────────────────────────────────────────────
    def _init_weights(self):
        for mod in (self.proj, self.to_q, self.to_k, self.to_v, self.out):
            for layer in mod.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _move_to_device(self):
        for m in (self.proj, self.to_q, self.to_k, self.to_v, self.out):
            m.to(self.device)

    def get_observation_components(self) -> np.ndarray:
        return np.nan_to_num(self.last_embedding, nan=0.0, posinf=1.0, neginf=-1.0).copy()

    # serialization
    def get_state(self):
        return {
            "last_embedding": self.last_embedding.tolist(),
            "afe_buffer": self.afe.price_buffer,
            "afe_last_feats": self.afe.last_feats.tolist(),
        }

    def set_state(self, state):
        self.last_embedding = np.array(state.get("last_embedding", self.last_embedding), dtype=np.float32)
        self.afe.price_buffer = state.get("afe_buffer", self.afe.price_buffer)
        self.afe.last_feats = np.array(state.get("afe_last_feats", self.afe.last_feats), dtype=np.float32)
