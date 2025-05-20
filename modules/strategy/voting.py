# modules/voting_modules.py

from __future__ import annotations
from statistics import mean
from typing import List
import numpy as np
from collections import deque
from typing import List, Any, Dict
from modules.utils.info_bus import Vote

# -----------------------------------------------------------------------------#
# Base interface imported from all_modules.py
# -----------------------------------------------------------------------------#
try:
    from modules.core.core import Module
except ImportError:
    # minimal stub for static analysis / unit tests
    class Module:  # type: ignore
        def reset(self): ...
        def step(self, *a, **kw): ...
        def get_observation_components(self): ...


# -----------------------------------------------------------------------------#
# Helper filters
# -----------------------------------------------------------------------------#
class ConsensusDetector(Module):
    """
    If ensemble entropy drops below threshold we inject Dirichlet noise to
    re‑diversify.  (Unchanged logic, renamed `step()` -> `apply()` for clarity.)
    """
    def __init__(self, n_members: int, entropy_th: float = 0.5, dir_alpha: float = 0.3):
        self.n = n_members
        self.th = entropy_th
        self.alpha = dir_alpha

    # `Module` compliance ------------------------------------------------------#
    def reset(self): ...
    def step(self, **kwargs): ...
    def get_observation_components(self):
        return np.zeros(self.n, dtype=np.float32)

    # new public API -----------------------------------------------------------#
    def apply(self, w: np.ndarray) -> np.ndarray:
        p = w / (w.sum() + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-8))
        if entropy < self.th:
            noise = np.random.dirichlet([self.alpha] * self.n)
            return 0.8 * p + 0.2 * noise
        return w





class CollusionAuditor:
    """Detects systematic collusion in module votes.

    Principle: if the Pearson correlation between any two modules' vote
    series > threshold for a minimum window, raise alert.
    """

    def __init__(self, window: int = 250, corr_threshold: float = 0.95) -> None:
        self.window = window
        self.corr_threshold = corr_threshold
        self._history: dict[str, List[int]] = {}

    # --------------------------------------------------------------------- #
    def add_votes(self, votes: List[Vote]) -> None:
        for v in votes:
            self._history.setdefault(v["module"], []).append(v["action"])
            # Maintain fixed‑length window
            h = self._history[v["module"]]
            if len(h) > self.window:
                del h[0]

    def check(self) -> List[str]:
        """Returns list of module names involved in suspected collusion."""
        suspects: List[str] = []
        modules = list(self._history)
        for i, m1 in enumerate(modules):
            for m2 in modules[i + 1 :]:
                corr = self._pearson(self._history[m1], self._history[m2])
                if corr is not None and corr > self.corr_threshold:
                    suspects.extend([m1, m2])
        return sorted(set(suspects))

    # ------------------------------------------------------------------ utils
    def _pearson(self, xs: List[int], ys: List[int]) -> float | None:
        if len(xs) < 2 or len(xs) != len(ys):
            return None
        mean_x, mean_y = mean(xs), mean(ys)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
        den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
        if den_x == 0 or den_y == 0:
            return None
        return num / (den_x * den_y)

class TimeHorizonAligner(Module):
    """
    Discount each member’s vote by distance between its decision horizon and
    the common step clock.  Implemented as `apply()`.
    """
    def __init__(self, horizons: List[int]):
        self.horizons = np.array(horizons, dtype=np.float32)
        self.clock = 0

    def reset(self):
        self.clock = 0

    def step(self, **kwargs): ...
    def get_observation_components(self):
        return np.array([self.clock], dtype=np.float32)

    def apply(self, w: np.ndarray) -> np.ndarray:
        disc = 1.0 / (1.0 + np.abs(self.clock - self.horizons))
        self.clock += 1
        return w * disc


class AlternativeRealitySampler:
    """
    Generates perturbed replicas of the current weight vector.  Pure
    functional helper – *does not* mutate the original weights.
    """
    def __init__(self, dim: int, n_samples: int = 5, sigma: float = 0.05):
        self.dim = dim
        self.n = n_samples
        self.sigma = sigma

    def sample(self, w: np.ndarray) -> np.ndarray:
        return w[None, :] + np.random.randn(self.n, self.dim) * self.sigma


# -----------------------------------------------------------------------------#
# StrategyArbiter – new two‑phase implementation
# -----------------------------------------------------------------------------#
# voting_modules.py  (replace whole StrategyArbiter class)




# voting_modules.py
# voting_modules.py
class StrategyArbiter(Module):
    """
    Two-phase voting controller with diversity filters.
    """

    def __init__(
        self,
        members: List[Module],
        init_weights: List[float] | np.ndarray,
        action_dim: int,
        adapt_rate: float = 0.01,
        consensus: ConsensusDetector | None = None,
        collusion: CollusionAuditor | None = None,
        horizon_aligner: TimeHorizonAligner | None = None,
    ):
        self.members   = members
        self.weights   = np.asarray(init_weights, np.float32)
        assert self.weights.shape == (len(members),)

        self.action_dim = action_dim
        self.adapt_rate = adapt_rate
        self.consensus  = consensus
        self.collusion  = collusion
        self.haligner   = horizon_aligner

        # per-step caches
        self.last_alpha: np.ndarray | None = None
        self._scores:    np.ndarray | None = None

        self._b       = 0.0      # EW-baseline
        self._b_beta  = 0.98

    # ───────────── phase 1 – propose ─────────────────────────────────── #
    def propose(self, obs: Any) -> np.ndarray:
        proposals, confidences = [], []

        for m in self.members:
            try:
                prop = m.propose_action(obs)
            except Exception:
                prop = None
            if prop is None or prop.size != self.action_dim:
                prop = np.zeros(self.action_dim, np.float32)
            proposals.append(prop)

            try:
                conf = float(m.confidence(obs))
            except Exception:
                conf = 0.5
            confidences.append(conf)

        proposals   = np.stack(proposals, axis=0)           # (M,A)
        confidences = np.asarray(confidences, np.float32)   # (M,)

        # diversity filters (order doesn’t matter)
        w = self.weights.copy()
        for filt in (self.consensus, self.collusion, self.haligner):
            if filt is not None and hasattr(filt, "apply"):
                w = filt.apply(w)

        scores = w * confidences
        exp    = np.exp(scores - scores.max())
        alpha  = exp / exp.sum()                            # (M,)

        blend  = (alpha[:, None] * proposals).sum(axis=0)

        # cache for credit assignment
        self.last_alpha = alpha
        self._scores    = scores
        return blend.astype(np.float32)

    # ───────────── phase 2 – reward update ───────────────────────────── #
    def update_reward(self, reward: float) -> None:
        if self.last_alpha is None:
            return

        # baseline
        self._b  = self._b_beta * self._b + (1 - self._b_beta) * reward
        adv      = reward - self._b

        grad = adv * (self.last_alpha - self.last_alpha.mean())
        self.weights += self.adapt_rate * grad

        # keep weights stable and normalised
        self.weights = np.clip(self.weights, 1e-4, None)
        self.weights /= self.weights.sum()

        # slow learning-rate decay – prevents blow-ups
        self.adapt_rate = max(1e-4, self.adapt_rate * 0.999)

        self.last_alpha = None
        self._scores    = None

    # ───────────── Module boiler-plate ───────────────────────────────── #
    def reset(self):
        self.last_alpha = None
        self._scores    = None

    def step(self, **kwargs):
        raise RuntimeError("Use propose()/update_reward()")

    def get_observation_components(self) -> np.ndarray:
        return self.weights.copy()



