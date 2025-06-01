# modules/playbook.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from modules.memory.memory import PlaybookMemory
from modules.core.core import Module
import random

class PlaybookClusterer(Module):
    """
    Clusters past trade feature vectors, providing a cluster-weighted recall vector
    for new trade situations. Fully evolutionary: n_clusters and PCA dim mutate/crossover.
    """
    def __init__(self, n_clusters: int = 5, pca_dim: int = 2, debug: bool = False):
        self.n_clusters = max(2, int(n_clusters))
        self.pca_dim = max(1, int(pca_dim))
        self.debug = debug
        self._reset_models()
        self._ready = False

    def _reset_models(self):
        self._pca = PCA(n_components=self.pca_dim)
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self._ready = False

    def reset(self):
        self._reset_models()

    def step(self, **kwargs):
        # Not required for clustering logic
        pass

    def fit(self, memory: PlaybookMemory):
        feats = memory._features if hasattr(memory, "_features") else []
        if len(feats) < self.n_clusters:
            self._ready = False
            return
        X = np.vstack(feats)
        pca_n = min(self.pca_dim, min(X.shape) - 1)
        pca_n = max(1, pca_n)
        self._pca = PCA(n_components=pca_n)
        Z = self._pca.fit_transform(X)
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self._kmeans.fit(Z)
        self._ready = True
        if self.debug:
            print(f"[PlaybookClusterer] fitted on {len(X)} samples, pca_dim={pca_n}, n_clusters={self.n_clusters}")

    def recall(self, features: np.ndarray) -> np.ndarray:
        if not self._ready:
            # If not ready, return uniform probabilities
            return np.ones(self.n_clusters, np.float32) / self.n_clusters
        z = self._pca.transform(features.reshape(1, -1))
        d = self._kmeans.transform(z).ravel()
        inv = 1.0 / (d + 1e-8)
        weights = (inv / inv.sum()).astype(np.float32)
        if self.debug:
            print(f"[PlaybookClusterer] recall weights: {weights}")
        return weights

    # Evolutionary logic
    def mutate(self, std: float = 1.0):
        """Randomly change n_clusters and/or PCA dim (in-place)."""
        if random.random() < 0.5:
            self.n_clusters = max(2, int(self.n_clusters + np.random.randint(-1, 2)))
        if random.random() < 0.5:
            self.pca_dim = max(1, int(self.pca_dim + np.random.randint(-1, 2)))
        self._reset_models()
        if self.debug:
            print(f"[PlaybookClusterer] mutated: n_clusters={self.n_clusters}, pca_dim={self.pca_dim}")

    def crossover(self, other: "PlaybookClusterer") -> "PlaybookClusterer":
        n_clusters = self.n_clusters if random.random() < 0.5 else other.n_clusters
        pca_dim = self.pca_dim if random.random() < 0.5 else other.pca_dim
        debug = self.debug or other.debug
        return PlaybookClusterer(n_clusters=n_clusters, pca_dim=pca_dim, debug=debug)

    def get_observation_components(self) -> np.ndarray:
        # Optionally, encode model config for downstream modules
        return np.array([self.n_clusters, self.pca_dim, float(self._ready)], dtype=np.float32)

    # --- State management (save/load) ---
    def get_state(self):
        state = {
            "n_clusters": self.n_clusters,
            "pca_dim": self.pca_dim,
            "_ready": self._ready,
        }
        # Save PCA and KMeans parameters only if ready
        if self._ready:
            state["pca_mean_"] = self._pca.mean_.tolist()
            state["pca_components_"] = self._pca.components_.tolist()
            state["kmeans_centers_"] = self._kmeans.cluster_centers_.tolist()
        return state

    def set_state(self, state):
        self.n_clusters = int(state.get("n_clusters", self.n_clusters))
        self.pca_dim = int(state.get("pca_dim", self.pca_dim))
        self._reset_models()
        self._ready = bool(state.get("_ready", False))
        if self._ready and "pca_mean_" in state and "pca_components_" in state:
            self._pca.mean_ = np.array(state["pca_mean_"], dtype=np.float32)
            self._pca.components_ = np.array(state["pca_components_"], dtype=np.float32)
        if self._ready and "kmeans_centers_" in state:
            self._kmeans.cluster_centers_ = np.array(state["kmeans_centers_"], dtype=np.float32)
