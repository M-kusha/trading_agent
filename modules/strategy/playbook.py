import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from modules.memory.memory import PlaybookMemory
from modules.core.core import Module
import random
import copy
import datetime

def utcnow():
    return datetime.datetime.utcnow().isoformat()

class PlaybookClusterer(Module):
    """
    Clusters past trade feature vectors, providing a cluster-weighted recall vector
    for new trade situations. Fully evolutionary: n_clusters and PCA dim mutate/crossover.
    Now with full audit trail and explainable recall diagnostics.
    """

    def __init__(self, n_clusters: int = 5, pca_dim: int = 2, debug: bool = True, audit_log_size: int = 50):
        self.n_clusters = max(2, int(n_clusters))
        self.pca_dim = max(1, int(pca_dim))
        self.debug = debug
        self._reset_models()
        self._ready = False
        self.audit_trail = []
        self._audit_log_size = audit_log_size
        self._last_fit_meta = {}
        self._last_recall_meta = {}

    def _reset_models(self):
        self._pca = PCA(n_components=self.pca_dim)
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self._ready = False

    def reset(self):
        self._reset_models()
        self.audit_trail.clear()
        self._last_fit_meta = {}
        self._last_recall_meta = {}

    def step(self, **kwargs):
        # Not required for clustering logic
        pass

    def fit(self, memory: PlaybookMemory):
        feats = memory._features if hasattr(memory, "_features") else []
        fit_meta = {
            "timestamp": utcnow(),
            "fit_attempted": True,
            "features_len": len(feats),
            "n_clusters": self.n_clusters,
            "pca_dim": self.pca_dim,
            "status": "",
        }
        if len(feats) < self.n_clusters:
            self._ready = False
            fit_meta["status"] = f"Not enough data to fit: {len(feats)} < {self.n_clusters}"
            self._record_audit(fit_meta)
            if self.debug:
                print(f"[PlaybookClusterer] {fit_meta['status']}")
            return
        X = np.vstack(feats)
        pca_n = min(self.pca_dim, min(X.shape) - 1)
        pca_n = max(1, pca_n)
        self._pca = PCA(n_components=pca_n)
        Z = self._pca.fit_transform(X)
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self._kmeans.fit(Z)
        self._ready = True
        fit_meta["status"] = f"Fitted OK: X={X.shape}, Z={Z.shape}"
        fit_meta["pca_dim_actual"] = pca_n
        fit_meta["kmeans_centers"] = self._kmeans.cluster_centers_.tolist()
        fit_meta["pca_mean"] = self._pca.mean_.tolist()
        fit_meta["pca_components"] = self._pca.components_.tolist()
        self._last_fit_meta = fit_meta
        self._record_audit(fit_meta)
        if self.debug:
            print(f"[PlaybookClusterer] fitted on {len(X)} samples, pca_dim={pca_n}, n_clusters={self.n_clusters}")

    def recall(self, features: np.ndarray) -> np.ndarray:
        recall_meta = {
            "timestamp": utcnow(),
            "recall_attempted": True,
            "features_shape": features.shape,
            "ready": self._ready,
        }
        if not self._ready:
            recall_meta["weights"] = np.ones(self.n_clusters, np.float32) / self.n_clusters
            recall_meta["status"] = "Clusterer not ready, returning uniform."
            self._record_audit(recall_meta)
            if self.debug:
                print(f"[PlaybookClusterer] Not ready. Returning uniform weights.")
            self._last_recall_meta = recall_meta
            return recall_meta["weights"]
        z = self._pca.transform(features.reshape(1, -1))
        d = self._kmeans.transform(z).ravel()
        inv = 1.0 / (d + 1e-8)
        weights = (inv / inv.sum()).astype(np.float32)
        recall_meta["weights"] = weights.tolist()
        recall_meta["distances"] = d.tolist()
        recall_meta["z"] = z.tolist()
        recall_meta["status"] = "Recall successful."
        self._record_audit(recall_meta)
        if self.debug:
            print(f"[PlaybookClusterer] recall weights: {weights}")
        self._last_recall_meta = recall_meta
        return weights

    def _record_audit(self, entry):
        self.audit_trail.append(entry)
        if len(self.audit_trail) > self._audit_log_size:
            self.audit_trail = self.audit_trail[-self._audit_log_size:]

    def get_last_fit_meta(self):
        return self._last_fit_meta

    def get_last_recall_meta(self):
        return self._last_recall_meta

    def get_audit_trail(self, n=10):
        return self.audit_trail[-n:]

    # Evolutionary logic
    def mutate(self, std: float = 1.0):
        pre = (self.n_clusters, self.pca_dim)
        if random.random() < 0.5:
            self.n_clusters = max(2, int(self.n_clusters + np.random.randint(-1, 2)))
        if random.random() < 0.5:
            self.pca_dim = max(1, int(self.pca_dim + np.random.randint(-1, 2)))
        self._reset_models()
        post = (self.n_clusters, self.pca_dim)
        entry = {
            "timestamp": utcnow(),
            "mutation": True,
            "before": pre,
            "after": post,
        }
        self._record_audit(entry)
        if self.debug:
            print(f"[PlaybookClusterer] mutated: n_clusters={self.n_clusters}, pca_dim={self.pca_dim}")

    def crossover(self, other: "PlaybookClusterer") -> "PlaybookClusterer":
        n_clusters = self.n_clusters if random.random() < 0.5 else other.n_clusters
        pca_dim = self.pca_dim if random.random() < 0.5 else other.pca_dim
        debug = self.debug or other.debug
        entry = {
            "timestamp": utcnow(),
            "crossover": True,
            "from_self": (self.n_clusters, self.pca_dim),
            "from_other": (other.n_clusters, other.pca_dim),
            "result": (n_clusters, pca_dim)
        }
        # You can log this if needed
        return PlaybookClusterer(n_clusters=n_clusters, pca_dim=pca_dim, debug=debug)

    def get_observation_components(self) -> np.ndarray:
        return np.array([self.n_clusters, self.pca_dim, float(self._ready)], dtype=np.float32)

    # --- State management (save/load) ---
    def get_state(self):
        state = {
            "n_clusters": self.n_clusters,
            "pca_dim": self.pca_dim,
            "_ready": self._ready,
            "audit_trail": copy.deepcopy(self.audit_trail),
            "last_fit_meta": copy.deepcopy(self._last_fit_meta),
            "last_recall_meta": copy.deepcopy(self._last_recall_meta),
        }
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
        self.audit_trail = copy.deepcopy(state.get("audit_trail", []))
        self._last_fit_meta = copy.deepcopy(state.get("last_fit_meta", {}))
        self._last_recall_meta = copy.deepcopy(state.get("last_recall_meta", {}))
        if self._ready and "pca_mean_" in state and "pca_components_" in state:
            self._pca.mean_ = np.array(state["pca_mean_"], dtype=np.float32)
            self._pca.components_ = np.array(state["pca_components_"], dtype=np.float32)
        if self._ready and "kmeans_centers_" in state:
            self._kmeans.cluster_centers_ = np.array(state["kmeans_centers_"], dtype=np.float32)
