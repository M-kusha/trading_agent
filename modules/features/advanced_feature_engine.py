# ─────────────────────────────────────────────────────────────
# File: modules/features/advanced_feature_engine.py
# ─────────────────────────────────────────────────────────────

import numpy as np
from ..core.core import Module
from typing import  Union, List
import logging

# ─────────────────────────────────────────────────────────────
class AdvancedFeatureEngine(Module):
    def __init__(self, window_sizes: List[int] = [7, 14, 28], debug: bool = True):
        self.windows = sorted(window_sizes)
        self.debug = debug

        # 4 statistics per window + 4 global features
        self.out_dim = len(self.windows) * 4 + 4
        self.last_feats = np.zeros(self.out_dim, np.float32)

        # Buffer for boot-strapping
        self.price_buffer: List[float] = []
        self.max_buffer_size = max(self.windows) + 10

        # logging
        self.logger = logging.getLogger("AdvancedFeatureEngine")
        if self.debug and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    # ── lifecycle ────────────────────────────────────────────
    def reset(self):
        self.last_feats[:] = 0.0
        self.price_buffer.clear()

    def step(self, **kwargs):
        """Accept price(s) from env.step() info dicts etc."""
        for key in ("price", "prices", "close", "price_series"):
            if key in kwargs:
                src = kwargs[key]
                if isinstance(src, (float, int)):
                    self._update_buffer([src])
                elif isinstance(src, (list, np.ndarray)):
                    self._update_buffer(src)

    # ── transform ───────────────────────────────────────────
    def transform(self, price_series: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Return feature vector of length self.out_dim."""
        prices = np.asarray(price_series, dtype=np.float32)
        mask = np.isfinite(prices) & (prices > 0)
        valid_prices = prices[mask]

        if valid_prices.size == 0:
            if self.price_buffer:
                valid_prices = np.array(self.price_buffer[-28:], dtype=np.float32)
                if self.debug:
                    self.logger.debug(f"Using buffer prices: {valid_prices.size}")
            else:
                valid_prices = self._generate_synthetic_prices(30)
                if self.debug:
                    self.logger.debug("Using synthetic prices for boot-strapping")

        if valid_prices.size:
            self._update_buffer(valid_prices)

        feats: List[float] = []

        # per-window stats
        for w in self.windows:
            window = (
                valid_prices[-w:]
                if valid_prices.size >= w
                else np.concatenate(
                    [np.full(w - valid_prices.size, valid_prices.mean()), valid_prices]
                )
            )

            # 1. volatility
            vol = np.std(window) if window.size > 4 else 0.01
            feats.append(float(vol))

            # 2. return
            ret = (window[-1] - window[0]) / window[0] if window[0] > 0 else 0.0
            feats.append(float(ret))

            # 3. mean-reversion
            mean_p = window.mean()
            mean_rev = (window[-1] - mean_p) / mean_p if mean_p > 0 else 0.0
            feats.append(float(mean_rev))

            # 4. slope (trend strength)
            if window.size > 1:
                x = np.arange(window.size, dtype=np.float32)
                norm = window / window[0] if window[0] > 0 else window
                slope = np.polyfit(x, norm, 1)[0] if np.std(norm) else 0.0
            else:
                slope = 0.0
            feats.append(float(slope))

        # global features
        if valid_prices.size > 1:
            diffs = np.diff(valid_prices[-10:]) if valid_prices.size > 10 else np.diff(valid_prices)
            spread = np.mean(np.abs(diffs)) if diffs.size else 0.0
            feats.append(float(spread))

            momentum = (
                (valid_prices[-1] - valid_prices[-5]) / valid_prices[-5]
                if valid_prices.size >= 5 and valid_prices[-5] > 0
                else 0.0
            )
            feats.append(float(momentum))

            short_vol = np.std(valid_prices[-7:]) if valid_prices.size >= 7 else 0.01
            long_vol = np.std(valid_prices) if valid_prices.size > 4 else 0.01
            feats.append(float(short_vol / (long_vol + 1e-8)))

            high = valid_prices[-20:].max() if valid_prices.size >= 20 else valid_prices.max()
            low = valid_prices[-20:].min() if valid_prices.size >= 20 else valid_prices.min()
            price_pos = (valid_prices[-1] - low) / (high - low) if high > low else 0.5
            feats.append(float(price_pos))
        else:
            feats.extend([0.0, 0.0, 1.0, 0.5])

        # length-enforce
        feats = np.asarray(feats, dtype=np.float32)
        if feats.size < self.out_dim:
            feats = np.pad(feats, (0, self.out_dim - feats.size))
        else:
            feats = feats[:self.out_dim]

        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        self.last_feats = feats

        if self.debug:
            self.logger.debug(
                f"Extracted {feats.size} features | vol={feats[0]:.4f}, ret={feats[1]:.4f}"
            )

        return self.last_feats

    # ── helpers ─────────────────────────────────────────────
    def _update_buffer(self, prices: Union[List[float], np.ndarray]):
        for p in prices:
            if np.isfinite(p) and p > 0:
                self.price_buffer.append(float(p))
        if len(self.price_buffer) > self.max_buffer_size:
            self.price_buffer = self.price_buffer[-self.max_buffer_size :]

    @staticmethod
    def _generate_synthetic_prices(n: int = 30) -> np.ndarray:
        base = 1.0
        returns = np.random.normal(0.0, 0.01, n).astype(np.float32)
        return base * np.exp(np.cumsum(returns))

    # serialization helpers
    def get_observation_components(self) -> np.ndarray:
        return np.nan_to_num(self.last_feats, nan=0.0, posinf=1.0, neginf=-1.0).copy()

