"""
strategy_arbiter.py
"""
from __future__ import annotations

# --- Standard Library ----------------------------------------------------- #
from statistics import mean
from typing import List, Any

# --- Third‑Party ---------------------------------------------------------- #
import numpy as np

# --- Internal Modules ----------------------------------------------------- #
from modules.utils.info_bus import Vote

try:
    # Base interface imported from all_modules.py
    from modules.core.core import Module
except ImportError:
    # Fallback stub so static analysis / unit tests do not fail when the
    # full environment is not available.
    class Module:  # type: ignore
        """Minimal stub of the `Module` interface used only when the real package
        cannot be imported (e.g. during static type‑checking or documentation
        builds).  It mimics the public API expected by the helper classes
        defined below.
        """

        def reset(self) -> None: ...
        def step(self, *a, **kw): ...
        def get_observation_components(self): ...

# --------------------------------------------------------------------------- #
# Helper Filters
# --------------------------------------------------------------------------- #
class ConsensusDetector(Module):
    """Entropy‑based confidence smoother.

    When the entropy of the ensemble confidences falls below the configured
    threshold, Dirichlet noise is injected to re‑diversify the weight vector.
    The logic is identical to the original implementation; only the public
    method was renamed from `step()` to `apply()` for clarity.
    """

    def __init__(
        self,
        n_members: int,
        entropy_th: float = 0.5,
        dir_alpha: float = 0.3,
    ) -> None:
        self.n = n_members
        self.th = entropy_th
        self.alpha = dir_alpha

    # -------------------- `Module` compliance ------------------------------ #
    def reset(self) -> None: ...

    def step(self, **kwargs) -> None: ...

    def get_observation_components(self):
        return np.zeros(self.n, dtype=np.float32)

    # -------------------- public API --------------------------------------- #
    def apply(self, w: np.ndarray) -> np.ndarray:
        """Return a possibly smoothed copy of *w*."""
        p = w / (w.sum() + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-8))

        if entropy < self.th:
            noise = np.random.dirichlet([self.alpha] * self.n)
            return 0.8 * p + 0.2 * noise

        return w


class CollusionAuditor:
    """Detect systematic collusion in committee votes.

    If the Pearson correlation between any two members' vote histories
    exceeds *corr_threshold* for the sliding *window*, the corresponding
    module names are returned by :py:meth:`check`.
    """

    def __init__(self, window: int = 250, corr_threshold: float = 0.95) -> None:
        self.window = window
        self.corr_threshold = corr_threshold
        self._history: dict[str, List[int]] = {}

    # -------------------- public API --------------------------------------- #
    def add_votes(self, votes: List[Vote]) -> None:
        """Append the latest *votes* to the sliding history window."""
        for v in votes:
            self._history.setdefault(v["module"], []).append(v["action"])

            # Maintain fixed‑length window per module
            h = self._history[v["module"]]
            if len(h) > self.window:
                del h[0]

    def check(self) -> List[str]:
        """Return names of modules involved in suspected collusion."""
        suspects: List[str] = []
        modules = list(self._history)

        for i, m1 in enumerate(modules):
            for m2 in modules[i + 1 :]:
                corr = self._pearson(self._history[m1], self._history[m2])
                if corr is not None and corr > self.corr_threshold:
                    suspects.extend([m1, m2])

        return sorted(set(suspects))

    # -------------------- internal helpers --------------------------------- #
    def _pearson(self, xs: List[int], ys: List[int]) -> float | None:
        """Compute Pearson correlation or *None* on degenerate input."""
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
    """Discount confidences by horizon mismatch between members and master clock."""

    def __init__(self, horizons: List[int]):
        self.horizons = np.asarray(horizons, np.float32)
        self.clock = 0

    # -------------------- `Module` compliance ------------------------------ #
    def reset(self) -> None:
        self.clock = 0

    def step(self, **kwargs) -> None: ...

    def get_observation_components(self):
        return np.asarray([self.clock], np.float32)

    # -------------------- public API --------------------------------------- #
    def apply(self, w: np.ndarray) -> np.ndarray:
        disc = 1.0 / (1.0 + np.abs(self.clock - self.horizons))
        self.clock += 1
        return w * disc


class AlternativeRealitySampler:
    """Generate perturbed replicas of a weight vector *without* mutation."""

    def __init__(self, dim: int, n_samples: int = 5, sigma: float = 0.05):
        self.dim = dim
        self.n = n_samples
        self.sigma = sigma

    def sample(self, w: np.ndarray) -> np.ndarray:
        return w[None, :] + np.random.randn(self.n, self.dim) * self.sigma


# --------------------------------------------------------------------------- #
# StrategyArbiter – two‑phase reinforcement‑style weight adaptation
# --------------------------------------------------------------------------- #
class StrategyArbiter(Module):
    """Committee‑of‑experts action blender with REINFORCE weight updates.

    **New in this patch**
    ---------------------
    * `PRIOR_BLEND` – how much the *learned* weight vector influences the final
      per‑step weights. Set to ~0.3 by default (tweakable).
    * `propose()` now combines *both* the committee’s *current* confidences and
      the *persistent* `self.weights` learned via policy‑gradient updates.
    * Everything else (gradient update logic, helper hooks) is unchanged.
    """

    PRIOR_BLEND: float = 0.3  # weight on learned prior vs fresh confidences

    # ------------------------------ INIT ----------------------------------- #
    def __init__(
        self,
        members: List[Module],
        init_weights: List[float] | np.ndarray,
        action_dim: int,
        adapt_rate: float = 0.01,
        consensus: Module | None = None,
        collusion: Module | None = None,
        horizon_aligner: Module | None = None,
        debug: bool = True,
    ) -> None:
        self.members = members
        self.weights = np.asarray(init_weights, np.float32)
        assert self.weights.shape == (len(members),)

        self.action_dim = action_dim
        self.adapt_rate = adapt_rate

        self.consensus = consensus
        self.collusion = collusion
        self.haligner  = horizon_aligner
        self.debug     = debug

        self.last_alpha: np.ndarray | None = None
        self._b       = 0.0   # moving baseline for REINFORCE advantage
        self._b_beta  = 0.98

    # -------------------------------------------------------------------- #
    # Main interaction surface
    # -------------------------------------------------------------------- #
    def propose(self, obs: Any) -> np.ndarray:
        """Return blended action vector for *obs* (see docstring for flow)."""
        proposals, confidences = [], []

        for m in self.members:
            # ------- proposal ---------------------------------------------
            try:
                prop = m.propose_action(obs)
                if prop.shape != (self.action_dim,):
                    raise ValueError("shape mismatch")
            except Exception as e:  # noqa: BLE001
                if self.debug:
                    print(f"[Arbiter] {m.__class__.__name__} propose_error: {e}")
                prop = np.zeros(self.action_dim, np.float32)
            proposals.append(prop)

            # ------- confidence -------------------------------------------
            try:
                conf = float(m.confidence(obs))
                if not np.isfinite(conf):
                    raise ValueError("non‑finite confidence")
            except Exception as e:  # noqa: BLE001
                if self.debug:
                    print(f"[Arbiter] {m.__class__.__name__} conf_error: {e}")
                conf = 0.5
            confidences.append(conf)

        proposals_arr   = np.stack(proposals, axis=0)         # (M, A)
        confidences_arr = np.asarray(confidences, np.float32) # (M,)

        # ---- optional helpers -------------------------------------------
        if self.consensus is not None:
            confidences_arr = self.consensus.apply(confidences_arr)
        if self.haligner is not None:
            confidences_arr = self.haligner.apply(confidences_arr)

        # ---- blend prior (learned) and likelihood (current confidence) ---
        prior = self.weights / (self.weights.sum() + 1e-8)
        like  = confidences_arr / (confidences_arr.sum() + 1e-8)
        w_raw = (1.0 - self.PRIOR_BLEND) * like + self.PRIOR_BLEND * prior
        w     = w_raw / (w_raw.sum() + 1e-8)

        action = np.dot(w, proposals_arr)  # (A,)
        if not np.all(np.isfinite(action)):
            if self.debug:
                print("[Arbiter] NaN in blended action → zeroing")
            action = np.nan_to_num(action)

        self.last_alpha = w.copy()
        return action.astype(np.float32, copy=False)

    # -------------------------------------------------------------------- #
    def update_reward(self, reward: float) -> None:
        if self.last_alpha is None:
            return
        self._b = self._b_beta * self._b + (1 - self._b_beta) * reward
        adv     = reward - self._b
        grad    = adv * (self.last_alpha - self.last_alpha.mean())
        self.weights += self.adapt_rate * grad
        self.weights  = np.clip(self.weights, 1e-4, None)
        self.weights /= self.weights.sum()
        self.adapt_rate = max(1e-4, self.adapt_rate * 0.999)
        self.last_alpha = None

    # ------------ misc helpers (unchanged) ------------------------------ #
    @property
    def genome_dim(self) -> int: return len(self.weights)
    def get_genome(self) -> np.ndarray: return self.weights.copy()
    def set_genome(self, g: np.ndarray):
        assert g.shape == (len(self.weights),)
        self.weights = np.clip(g, 1e-4, None)
        self.weights /= self.weights.sum()

    # Module boiler‑plate --------------------------------------------------
    def reset(self): self.last_alpha = None
    def step(self, **kw): raise RuntimeError("use propose()/update_reward()")
    def get_observation_components(self): return self.weights.copy()
    def get_state(self): return {"weights": self.weights}
    def set_state(self, st): self.set_genome(np.asarray(st.get("weights", self.weights)))