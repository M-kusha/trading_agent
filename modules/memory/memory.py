# modules/memory.py

from typing import List, Any, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from ..core.core import Module
import random
import copy

# ─────────────────────────────────────────────────────────────
class MistakeMemory(Module):
    """Clusters losing trades and provides stats to the agent.

    **Fixes introduced**
    --------------------
    * `interval` is accepted as an alias for `max_mistakes` to stay compatible
      with the existing env constructor.
    * `step()` now understands the *env*’s current call‑signature:
        – If called with `trades=[…]` it extracts `(features, pnl)` pairs.
        – The classic direct `(features, pnl)` call still works.
    """

    def __init__(
        self,
        max_mistakes: int = 100,
        n_clusters: int = 5,
        *,
        interval: int | None = None,   # <- backward‑compat alias
        debug: bool = False,
    ) -> None:
        if interval is not None:  # prefer explicit param, otherwise fallback
            max_mistakes = interval
        self.max_mistakes = int(max_mistakes)
        self.n_clusters   = int(n_clusters)
        self.debug        = debug
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self._buf: List[tuple[np.ndarray, float, dict]] = []  # [(features, pnl, info)]
        self._km: KMeans | None = None
        self._mean_dist = 0.0
        self._last_dist = 0.0

    # ------------------------------------------------------------------ #
    def step(self, *, trades: List[dict] | None = None,
                   features: np.ndarray | None = None,
                   pnl: float | None = None,
                   info: dict | None = None, **kw):
        """Register new mistakes.

        Parameters
        ----------
        trades  : list of trade‑dicts as passed by the env (optional)
        features: single feature vector (fallback path)
        pnl     : scalar PnL associated with *features*
        info    : arbitrary extra dict
        """
        # ------- env pathway: a list of trade dicts ---------------------
        if trades is not None:
            for tr in trades:
                self.step(features=tr.get("features"), pnl=tr.get("pnl"), info=tr)
            return

        # ------- direct pathway ----------------------------------------
        if features is None or pnl is None or pnl >= 0:
            return  # only record *losing* trades with valid data
        entry = (np.asarray(features, np.float32), float(pnl), info or {})
        self._buf.append(entry)
        if len(self._buf) > self.max_mistakes:
            self._buf = self._buf[-self.max_mistakes:]
        self._fit_clusters()

    # ------------------------------------------------------------------ #
    def _fit_clusters(self):
        if len(self._buf) < self.n_clusters:
            self._km = None; self._mean_dist = self._last_dist = 0.0; return
        X = np.stack([f for f, _, _ in self._buf])
        self._km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self._km.fit(X)
        d = self._km.transform(X)
        mins = d.min(axis=1)
        self._mean_dist = float(mins.mean())
        self._last_dist = float(d[-1].min())
        if self.debug:
            print(f"[MistakeMemory] fitted {len(X)} pts – mean={self._mean_dist:.4f}, last={self._last_dist:.4f}")

    # ------------------------------------------------------------------ #
    def get_observation_components(self) -> np.ndarray:
        k = float(self.n_clusters if self._km is not None else 0)
        return np.array([k, self._mean_dist, self._last_dist], np.float32)

    # State helpers unchanged ------------------------------------------ #
    def get_state(self):
        st = {
            "buf": [(f.tolist(), pnl, info) for f, pnl, info in self._buf],
            "mean": self._mean_dist,
            "last": self._last_dist,
        }
        if self._km is not None:
            st["km_centers"] = self._km.cluster_centers_.tolist()
        return st

    def set_state(self, st):
        self._buf       = [(np.asarray(f, np.float32), pnl, info) for f, pnl, info in st.get("buf", [])]
        self._mean_dist = float(st.get("mean", 0.0))
        self._last_dist = float(st.get("last", 0.0))
        if "km_centers" in st:
            self._km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            self._km.cluster_centers_ = np.asarray(st["km_centers"], np.float32)

    # Neuro‑evolution hooks unchanged ---------------------------------- #
    def mutate(self, noise_std=0.05):
        if self._km is not None:
            self._km.cluster_centers_ += np.random.randn(*self._km.cluster_centers_.shape).astype(np.float32) * noise_std
    def crossover(self, other: "MistakeMemory"):
        child = MistakeMemory(self.max_mistakes, self.n_clusters, debug=self.debug)
        if self._km is not None and other._km is not None:
            c1, c2 = self._km.cluster_centers_, other._km.cluster_centers_
            mix = np.where(np.random.rand(*c1.shape) > 0.5, c1, c2)
            child._km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            child._km.cluster_centers_ = mix
        return child

# ─────────────────────────────────────────────────────────────
class MemoryCompressor(Module):
    """
    Compresses feature memory using PCA, outputs an "intuition vector" representing the main axes of variation.
    Now includes evolutionary mutation/crossover for the intuition vector.
    """
    def __init__(self, compress_interval: int, n_components: int, debug=False):
        self.compress_interval = compress_interval
        self.n_components      = n_components
        self.debug             = debug
        self.reset()

    def reset(self):
        self.memory: List[np.ndarray] = []
        self.intuition_vector         = np.zeros(self.n_components, np.float32)

    def step(self, *_, **__):
        return self.intuition_vector.copy()

    def compress(self, episode: int, trades: List[dict]):
        for tr in trades:
            if "features" in tr:
                vec = np.asarray(tr["features"], np.float32)
                if vec.size != self.n_components:
                    vec = np.pad(vec, (0, max(0, self.n_components - vec.size)))[:self.n_components]
                self.memory.append(vec)
        if episode % self.compress_interval != 0 or len(self.memory) < self.n_components:
            return

        X = np.vstack(self.memory)
        if X.shape[0] > 1 and np.allclose(X, X[0], atol=1e-8):
            self.intuition_vector = X.mean(axis=0).astype(np.float32)
        else:
            stds = X.std(axis=0)
            keep = stds > 0.0
            if keep.sum() == 0:
                self.intuition_vector = X.mean(axis=0).astype(np.float32)
            else:
                X2 = X[:, keep]
                n_comp = min(self.n_components, X2.shape[1])
                pca = PCA(n_components=n_comp)
                Z   = pca.fit_transform(X2)
                X_mean = X.mean(axis=0)
                iv = X_mean.astype(np.float32).copy()
                iv[keep] = Z.mean(axis=0).astype(np.float32)
                self.intuition_vector = iv
        self.memory.clear()

    def get_observation_components(self) -> np.ndarray:
        return self.intuition_vector.copy()

    def get_state(self):
        return {
            "memory": [m.tolist() for m in self.memory],
            "intuition_vector": self.intuition_vector.tolist(),
        }

    def set_state(self, state):
        self.memory = [np.asarray(m, np.float32) for m in state.get("memory", [])]
        self.intuition_vector = np.asarray(state.get("intuition_vector", np.zeros(self.n_components, np.float32)), np.float32)

    # --- NEUROEVOLUTION INTERFACE ---
    def mutate(self, noise_std=0.05):
        noise = np.random.normal(0, noise_std, self.intuition_vector.shape).astype(np.float32)
        self.intuition_vector += noise
        if self.debug:
            print("[MemoryCompressor] Mutated intuition vector")

    def crossover(self, other: "MemoryCompressor"):
        child = MemoryCompressor(self.compress_interval, self.n_components, self.debug)
        mask = np.random.rand(*self.intuition_vector.shape) > 0.5
        child.intuition_vector = np.where(mask, self.intuition_vector, other.intuition_vector)
        return child

# ─────────────────────────────────────────────────────────────
class HistoricalReplayAnalyzer(Module):
    """
    Placeholder for episodic replay analysis (now evolves its 'bonus' param).
    """
    def __init__(self, interval: int=10, bonus: float=0.1, debug=False):
        self.interval = interval
        self.bonus    = bonus
        self.debug    = debug
    def reset(self): pass
    def step(self, **kwargs): pass
    def record_episode(self, data, actions, pnl: float):
        if self.debug:
            print(f"[HRA] Episode PnL={pnl}")
    def maybe_replay(self, episode: int) -> float:
        return self.bonus if episode % self.interval == 0 else 0.0
    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    # --- NEUROEVOLUTION INTERFACE ---
    def mutate(self, noise_std=0.05):
        self.bonus = float(np.clip(self.bonus + np.random.normal(0, noise_std), 0.0, 1.0))
        if self.debug:
            print("[HRA] Mutated bonus parameter")
    def crossover(self, other: "HistoricalReplayAnalyzer"):
        child = HistoricalReplayAnalyzer(self.interval, self.bonus, self.debug)
        child.bonus = random.choice([self.bonus, other.bonus])
        return child

# ─────────────────────────────────────────────────────────────
class PlaybookMemory(Module):
    """
    Stores feature-action-PnL tuples, recalls past profit by similarity, evolves by k-nearest config.
    """
    def __init__(self, max_entries: int=500, k: int=5, debug=False):
        self.max_entries = max_entries
        self.k           = k
        self.debug       = debug
        self._features   = []
        self._pnls       = []
        self._nbrs       = None

    def reset(self):
        self._features.clear()
        self._pnls.clear()
        self._nbrs = None

    def step(self, **kwargs): pass

    def record(self, features: np.ndarray, actions: np.ndarray, pnl: float):
        if len(self._features) >= self.max_entries:
            self._features.pop(0)
            self._pnls.pop(0)
        self._features.append(features.copy())
        self._pnls.append(pnl)
        if len(self._features) >= self.k:
            X = np.vstack(self._features)
            self._nbrs = NearestNeighbors(n_neighbors=min(self.k, len(X))).fit(X)

    def recall(self, features: np.ndarray) -> float:
        if self._nbrs is None: return 0.0
        _, idx = self._nbrs.kneighbors(features.reshape(1,-1))
        vals = [self._pnls[i] for i in idx[0]]
        return float(np.mean(vals))

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    # --- NEUROEVOLUTION INTERFACE ---
    def mutate(self):
        # Randomly mutate k within allowed bounds
        old_k = self.k
        self.k = int(np.clip(self.k + np.random.choice([-1, 1]), 1, self.max_entries))
        if self.debug:
            print(f"[PlaybookMemory] Mutated k: {old_k} -> {self.k}")
    def crossover(self, other: "PlaybookMemory"):
        child = PlaybookMemory(self.max_entries, random.choice([self.k, other.k]), self.debug)
        return child

# ─────────────────────────────────────────────────────────────
class MemoryBudgetOptimizer(Module):
    """
    Evolves memory size allocation limits for trades, mistakes, and plays.
    """
    def __init__(self, max_trades: int, max_mistakes: int, max_plays: int, debug=False):
        self.max_trades, self.max_mistakes, self.max_plays = max_trades, max_mistakes, max_plays
        self.debug = debug

    def reset(self):
        pass

    def step(self, env=None, **kwargs):
        return

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    # --- NEUROEVOLUTION INTERFACE ---
    def mutate(self):
        # Mutate one of the memory limits randomly
        param = random.choice(['max_trades', 'max_mistakes', 'max_plays'])
        old_val = getattr(self, param)
        setattr(self, param, max(1, old_val + random.choice([-1, 1])))
        if self.debug:
            print(f"[MemoryBudgetOptimizer] Mutated {param}: {old_val} -> {getattr(self, param)}")
    def crossover(self, other: "MemoryBudgetOptimizer"):
        child = MemoryBudgetOptimizer(
            max_trades=random.choice([self.max_trades, other.max_trades]),
            max_mistakes=random.choice([self.max_mistakes, other.max_mistakes]),
            max_plays=random.choice([self.max_plays, other.max_plays]),
            debug=self.debug
        )
        return child
