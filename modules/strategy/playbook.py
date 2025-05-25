#modules/playbook.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from modules.memory.memory import PlaybookMemory
from modules.core.core import Module

class PlaybookClusterer(Module):
    def __init__(self, n_clusters: int = 5, debug=False):
        self.n_clusters = n_clusters
        self.debug = debug
        self._pca = PCA(n_components=max(1, n_clusters // 2))
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self._ready = False

    def reset(self):
        self._ready = False

    def step(self, **kwargs): ...

    def fit(self, memory: PlaybookMemory):
        feats = memory._features
        if len(feats) < self.n_clusters:
            return
        X = np.vstack(feats)
        # clip PCA components so they don't exceed rank of X
        pca_n = min(self._pca.n_components, min(X.shape) - 1)
        if pca_n < 1: pca_n = 1
        self._pca = PCA(n_components=pca_n)
        Z = self._pca.fit_transform(X)
        self._kmeans.fit(Z)
        self._ready = True
        if self.debug:
            print(f"[PlaybookClusterer] fitted on {len(X)} samples, dim={pca_n}")

    def recall(self, features: np.ndarray) -> np.ndarray:
        if not self._ready:
            return np.ones(self.n_clusters, np.float32) / self.n_clusters
        z = self._pca.transform(features.reshape(1, -1))
        d = self._kmeans.transform(z).ravel()
        inv = 1.0 / (d + 1e-8)
        return (inv / inv.sum()).astype(np.float32)

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(self.n_clusters, np.float32)


    def get_state(self):
        return {
            "clusters": self.clusters,
        }

    def set_state(self, state):
        self.clusters = state.get("clusters", [])


    def get_state(self):
        return {
            "features": self._features,
            "pnls": self._pnls,
        }

    def set_state(self, state):
        self._features = state.get("features", [])
        self._pnls = state.get("pnls", [])