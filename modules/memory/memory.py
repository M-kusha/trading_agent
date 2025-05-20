# modules/memory.py

from typing import List
import numpy as np
from collections import deque
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from ..core.core import Module

class MistakeMemory(Module):
    """
    Stores losing‑trade feature vectors & clusters them every `interval` episodes.
    """
    def __init__(self, interval: int, n_clusters: int = 3, debug=False):
        self.interval = interval
        self.n_clusters = n_clusters
        self.debug = debug
        self.reset()

    def reset(self):
            # start fresh each episode
            self._records: List[np.ndarray] = []
            # use k-means++ for more stable clustering
            self._kmeans = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                random_state=0
            )
            self._episode_counter = 0
            self._last_features: np.ndarray | None = None

    def cluster_mistakes(self, episode: int) -> None:
        """Explicit trigger used by older tests."""
        self._episode_counter = episode - 1
        self.step(episode_done=True)

    def cluster_match_penalty(self, features: np.ndarray) -> float:
        if not hasattr(self._kmeans, 'cluster_centers_'):
            return 0.0
            
        f = np.asarray(features, np.float32).ravel()[:3]
        if f.size < 3:
            f = np.pad(f, (0, 3-f.size))
        d = np.linalg.norm(self._kmeans.cluster_centers_ - f, axis=1)
        return float(np.min(d) / (np.max(d) + 1e-8))
    
    def step(self, trades: List[dict] | None = None, episode_done: bool = False):
        if trades:
            losers = [tr for tr in trades if tr.get("pnl", 0.0) < 0 and "features" in tr]
            for tr in losers:
                vec = np.asarray(tr["features"], np.float32)
                if vec.size != 3:
                    vec = np.pad(vec, (0, 3 - vec.size))[:3]
                self._records.append(vec)
                self._last_features = vec

        if episode_done:
            self._episode_counter += 1
            if self._episode_counter % self.interval == 0 and len(self._records) >= self.n_clusters:
                X = np.vstack(self._records)
                try:
                    # Fit on copy to preserve original kmeans object if fit fails
                    new_kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
                    self._kmeans = new_kmeans
                    if self.debug:
                        print(f"[MistakeMemory] clustered on {len(X)} samples")
                except Exception as e:
                    if self.debug:
                        print(f"[MistakeMemory] clustering failed: {str(e)}")

    def get_observation_components(self) -> np.ndarray:
        if (self._last_features is None or 
            not hasattr(self._kmeans, 'cluster_centers_') or 
            self._kmeans.cluster_centers_.size == 0):
            return np.zeros(1, np.float32)
            
        try:
            d = np.linalg.norm(self._kmeans.cluster_centers_ - self._last_features, axis=1)
            pen = float(np.min(d) / (np.max(d) + 1e-8))
            return np.array([pen], np.float32)
        except Exception:
            return np.zeros(1, np.float32)

    def record(self, trade: dict) -> None:
        """Wrapper expected by unit‑tests."""
        self.step(trades=[trade])


# ─── all_modules.py ─ MemoryCompressor ─────────────────────────────────────────



class MemoryCompressor(Module):
    def __init__(self, compress_interval: int, n_components: int, debug=False):
        self.compress_interval = compress_interval
        self.n_components      = n_components
        self.debug             = debug
        self.reset()

    def reset(self):
        self.memory: List[np.ndarray] = []
        self.intuition_vector         = np.zeros(self.n_components, np.float32)

    def step(self, *_, **__):
        # no‐op for pipeline
        return self.intuition_vector.copy()

    def compress(self, episode: int, trades: List[dict]):
            # accumulate feature vectors
            for tr in trades:
                if "features" in tr:
                    vec = np.asarray(tr["features"], np.float32)
                    # pad/trim to n_components
                    if vec.size != self.n_components:
                        vec = np.pad(
                            vec,
                            (0, max(0, self.n_components - vec.size))
                        )[: self.n_components]
                    self.memory.append(vec)

            # only run PCA every compress_interval steps
            if episode % self.compress_interval != 0 or len(self.memory) < self.n_components:
                return

            X = np.vstack(self.memory)  # shape (T, n_components)

            # 1) trivial case: all rows identical → mean
            if X.shape[0] > 1 and np.allclose(X, X[0], atol=1e-8):
                self.intuition_vector = X.mean(axis=0).astype(np.float32)
                if self.debug:
                    print(f"[MemoryCompressor] skip PCA (all rows identical)")
            else:
                stds = X.std(axis=0)
                keep = stds > 0.0
                if keep.sum() == 0:
                    # no variance anywhere
                    self.intuition_vector = X.mean(axis=0).astype(np.float32)
                    if self.debug:
                        print(f"[MemoryCompressor] skip PCA (all cols constant)")
                else:
                    # run PCA on varying dims
                    X2 = X[:, keep]
                    n_comp = min(self.n_components, X2.shape[1])
                    pca = PCA(n_components=n_comp)
                    Z   = pca.fit_transform(X2)

                    # **Preserve** original means on dropped dims
                    X_mean = X.mean(axis=0)
                    iv = X_mean.astype(np.float32).copy()
                    iv[keep] = Z.mean(axis=0).astype(np.float32)

                    self.intuition_vector = iv
                    if self.debug:
                        print(f"[MemoryCompressor] compressed {X.shape[0]}→{n_comp} dims")

            # clear for next cycle
            self.memory.clear()

    def get_observation_components(self) -> np.ndarray:
        return self.intuition_vector.copy()


class HistoricalReplayAnalyzer(Module):
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


class PlaybookMemory(Module):
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
            self._nbrs = NearestNeighbors(n_neighbors=min(self.k,len(X))).fit(X)
    def recall(self, features: np.ndarray) -> float:
        if self._nbrs is None: return 0.0
        _, idx = self._nbrs.kneighbors(features.reshape(1,-1))
        vals = [self._pnls[i] for i in idx[0]]
        return float(np.mean(vals))
    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)



class MemoryBudgetOptimizer(Module):
    def __init__(self, max_trades: int, max_mistakes: int, max_plays: int, debug=False):
        self.max_trades, self.max_mistakes, self.max_plays = max_trades, max_mistakes, max_plays
        self.debug = debug

    def reset(self):
        pass

    def step(self, env=None, **kwargs):
        # pipeline may pass env=current_env here; no-op for now
        return

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)
