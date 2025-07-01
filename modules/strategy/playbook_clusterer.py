
# File: modules/strategy/playbook_clusterer.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from modules.memory.playbook_memory import PlaybookMemory
from modules.core.core import Module
import random
import copy
from utils.get_dir import utcnow


class PlaybookClusterer(Module):
    def __init__(self, n_clusters: int = 5, pca_dim: int = 2, debug: bool = True, 
                 audit_log_size: int = 50, bootstrap_trades: int = 10, 
                 exploration_bonus: float = 0.1):
        self.n_clusters = max(2, int(n_clusters))
        self.pca_dim = max(1, int(pca_dim))
        self.debug = debug
        self._reset_models()
        self._ready = False
        self.audit_trail = []
        self._audit_log_size = audit_log_size
        self._last_fit_meta = {}
        self._last_recall_meta = {}
        
        # FIX: Bootstrap parameters to enable initial trading
        self.bootstrap_trades = bootstrap_trades  # Number of trades before requiring clustering
        self.exploration_bonus = exploration_bonus  # Bonus weight for exploration
        self.trades_seen = 0
        self.bootstrap_mode = True
        
        # FIX: Track clustering readiness separately from data availability
        self._min_data_for_clustering = max(self.n_clusters, 3)
        self._soft_ready = False  # Can provide useful signals even if not fully clustered

    def _reset_models(self):
        self._pca = PCA(n_components=self.pca_dim)
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self._ready = False
        # FIX: Initialize with random centers for better initial behavior
        self._pseudo_centers = np.random.randn(self.n_clusters, self.pca_dim) * 0.5

    def reset(self):
        self._reset_models()
        self.audit_trail.clear()
        self._last_fit_meta = {}
        self._last_recall_meta = {}
        self.trades_seen = 0
        self.bootstrap_mode = True
        self._soft_ready = False

    def step(self, **kwargs):
        """FIX: Track trading activity and adjust bootstrap mode"""
        # Check if we're getting trade signals
        if 'trade_executed' in kwargs and kwargs['trade_executed']:
            self.trades_seen += 1
            
        # Exit bootstrap mode after enough trades
        if self.trades_seen >= self.bootstrap_trades:
            self.bootstrap_mode = False
            
        # Update soft readiness based on data availability
        if hasattr(self, '_last_fit_meta') and self._last_fit_meta.get('features_len', 0) >= 2:
            self._soft_ready = True

    def fit(self, memory: PlaybookMemory):
        feats = memory._features if hasattr(memory, "_features") else []
        fit_meta = {
            "timestamp": utcnow(),
            "fit_attempted": True,
            "features_len": len(feats),
            "n_clusters": self.n_clusters,
            "pca_dim": self.pca_dim,
            "bootstrap_mode": self.bootstrap_mode,
            "trades_seen": self.trades_seen,
            "status": "",
        }
        
        # FIX: Allow partial fitting with fewer samples in bootstrap mode
        min_required = self._min_data_for_clustering if not self.bootstrap_mode else 2
        
        if len(feats) < min_required:
            self._ready = False
            fit_meta["status"] = f"Not enough data to fit: {len(feats)} < {min_required} (bootstrap={self.bootstrap_mode})"
            self._record_audit(fit_meta)
            if self.debug:
                print(f"[PlaybookClusterer] {fit_meta['status']}")
            return
            
        # FIX: Handle edge case where we have data but less than desired clusters
        X = np.vstack(feats)
        actual_clusters = min(self.n_clusters, len(X))
        
        # Adjust PCA dimensions
        max_pca_dim = min(X.shape)
        pca_n = max(1, min(self.pca_dim, max_pca_dim))
        
        # Fit PCA
        self._pca = PCA(n_components=pca_n)
        Z = self._pca.fit_transform(X)
        
        # Fit KMeans with adjusted clusters
        self._kmeans = KMeans(n_clusters=actual_clusters, random_state=0)
        self._kmeans.fit(Z)
        
        # FIX: If we have fewer clusters than desired, pad with synthetic centers
        if actual_clusters < self.n_clusters:
            # Generate synthetic centers by adding noise to existing ones
            existing_centers = self._kmeans.cluster_centers_
            synthetic_centers = []
            for i in range(self.n_clusters - actual_clusters):
                base_idx = i % actual_clusters
                synthetic = existing_centers[base_idx] + np.random.randn(pca_n) * 0.1
                synthetic_centers.append(synthetic)
            self._kmeans.cluster_centers_ = np.vstack([existing_centers] + synthetic_centers)
            self._kmeans.n_clusters = self.n_clusters
        
        self._ready = True
        self._soft_ready = True
        fit_meta["status"] = f"Fitted OK: X={X.shape}, Z={Z.shape}, actual_clusters={actual_clusters}"
        fit_meta["pca_dim_actual"] = pca_n
        fit_meta["actual_clusters"] = actual_clusters
        fit_meta["kmeans_centers"] = self._kmeans.cluster_centers_.tolist()
        fit_meta["pca_mean"] = self._pca.mean_.tolist()
        fit_meta["pca_components"] = self._pca.components_.tolist()
        self._last_fit_meta = fit_meta
        self._record_audit(fit_meta)
        if self.debug:
            print(f"[PlaybookClusterer] fitted on {len(X)} samples, pca_dim={pca_n}, clusters={actual_clusters}/{self.n_clusters}")

    def recall(self, features: np.ndarray) -> np.ndarray:
        recall_meta = {
            "timestamp": utcnow(),
            "recall_attempted": True,
            "features_shape": features.shape,
            "ready": self._ready,
            "soft_ready": self._soft_ready,
            "bootstrap_mode": self.bootstrap_mode,
        }
        
        # FIX: Provide more intelligent weights when not ready
        if not self._ready:
            # In bootstrap mode, encourage exploration
            if self.bootstrap_mode:
                # Create non-uniform weights to encourage trading
                weights = np.random.dirichlet(np.ones(self.n_clusters) * 2.0)
                # Add exploration bonus to highest weight
                max_idx = np.argmax(weights)
                weights[max_idx] += self.exploration_bonus
                weights = weights / weights.sum()
                recall_meta["status"] = "Bootstrap mode: returning exploratory weights"
            else:
                # Even when not ready, provide slightly non-uniform weights
                weights = np.ones(self.n_clusters, np.float32) / self.n_clusters
                # Add small random perturbation
                weights += np.random.randn(self.n_clusters) * 0.01
                weights = np.maximum(weights, 0.01)  # Ensure all positive
                weights = weights / weights.sum()
                recall_meta["status"] = "Not ready: returning perturbed uniform weights"
                
            recall_meta["weights"] = weights.tolist()
            self._record_audit(recall_meta)
            if self.debug:
                print(f"[PlaybookClusterer] {recall_meta['status']}")
            self._last_recall_meta = recall_meta
            return weights.astype(np.float32)
        
        # Normal clustering-based recall
        try:
            z = self._pca.transform(features.reshape(1, -1))
            d = self._kmeans.transform(z).ravel()
            
            # FIX: Ensure we have the right number of distances
            if len(d) < self.n_clusters:
                # Pad with large distances
                d = np.pad(d, (0, self.n_clusters - len(d)), constant_values=np.max(d) * 2)
            
            # Calculate inverse distance weights
            inv = 1.0 / (d + 1e-8)
            weights = (inv / inv.sum()).astype(np.float32)
            
            # FIX: Add small exploration component in early stages
            if self.trades_seen < self.bootstrap_trades * 2:
                exploration = np.random.dirichlet(np.ones(self.n_clusters) * 5.0) * 0.1
                weights = 0.9 * weights + 0.1 * exploration
                weights = weights / weights.sum()
            
            recall_meta["weights"] = weights.tolist()
            recall_meta["distances"] = d.tolist()
            recall_meta["z"] = z.tolist()
            recall_meta["status"] = "Recall successful"
            
        except Exception as e:
            # FIX: Fallback to safe weights on any error
            weights = np.ones(self.n_clusters, np.float32) / self.n_clusters
            weights[0] += 0.1  # Slight bias to encourage action
            weights = weights / weights.sum()
            recall_meta["weights"] = weights.tolist()
            recall_meta["status"] = f"Recall error, returning safe weights: {str(e)}"
            recall_meta["error"] = str(e)
            
        self._record_audit(recall_meta)
        if self.debug:
            print(f"[PlaybookClusterer] recall weights: {weights}")
        self._last_recall_meta = recall_meta
        return weights.astype(np.float32)

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
        # FIX: Update min data requirement after mutation
        self._min_data_for_clustering = max(self.n_clusters, 3)
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
        # FIX: Inherit bootstrap parameters
        bootstrap_trades = (self.bootstrap_trades + other.bootstrap_trades) // 2
        exploration_bonus = (self.exploration_bonus + other.exploration_bonus) / 2
        entry = {
            "timestamp": utcnow(),
            "crossover": True,
            "from_self": (self.n_clusters, self.pca_dim),
            "from_other": (other.n_clusters, other.pca_dim),
            "result": (n_clusters, pca_dim)
        }
        return PlaybookClusterer(
            n_clusters=n_clusters, 
            pca_dim=pca_dim, 
            debug=debug,
            bootstrap_trades=bootstrap_trades,
            exploration_bonus=exploration_bonus
        )

    def get_observation_components(self) -> np.ndarray:
        """FIX: Provide more informative observation components"""
        # Basic readiness indicators
        ready_float = float(self._ready)
        soft_ready_float = float(self._soft_ready)
        bootstrap_float = float(self.bootstrap_mode)
        
        # Progress indicators
        data_ratio = min(1.0, self._last_fit_meta.get('features_len', 0) / max(self.n_clusters, 10))
        trade_ratio = min(1.0, self.trades_seen / max(self.bootstrap_trades, 10))
        
        # Clustering quality indicator (if available)
        clustering_quality = 0.5  # Default neutral
        if self._ready and hasattr(self._kmeans, 'inertia_'):
            # Normalize inertia to 0-1 range (lower is better)
            clustering_quality = 1.0 / (1.0 + self._kmeans.inertia_)
        
        # FIX: Return expanded observation vector
        return np.array([
            self.n_clusters,
            self.pca_dim,
            ready_float,
            soft_ready_float,
            bootstrap_float,
            data_ratio,
            trade_ratio,
            clustering_quality,
            self.exploration_bonus,
            float(self.trades_seen)
        ], dtype=np.float32)

    # --- State management (save/load) ---
    def get_state(self):
        state = {
            "n_clusters": self.n_clusters,
            "pca_dim": self.pca_dim,
            "_ready": self._ready,
            "_soft_ready": self._soft_ready,
            "bootstrap_mode": self.bootstrap_mode,
            "trades_seen": self.trades_seen,
            "bootstrap_trades": self.bootstrap_trades,
            "exploration_bonus": self.exploration_bonus,
            "audit_trail": copy.deepcopy(self.audit_trail),
            "last_fit_meta": copy.deepcopy(self._last_fit_meta),
            "last_recall_meta": copy.deepcopy(self._last_recall_meta),
        }
        if self._ready:
            state["pca_mean_"] = self._pca.mean_.tolist()
            state["pca_components_"] = self._pca.components_.tolist()
            state["kmeans_centers_"] = self._kmeans.cluster_centers_.tolist()
            state["kmeans_n_clusters"] = self._kmeans.n_clusters
        return state

    def set_state(self, state):
        self.n_clusters = int(state.get("n_clusters", self.n_clusters))
        self.pca_dim = int(state.get("pca_dim", self.pca_dim))
        self._reset_models()
        self._ready = bool(state.get("_ready", False))
        self._soft_ready = bool(state.get("_soft_ready", False))
        self.bootstrap_mode = bool(state.get("bootstrap_mode", True))
        self.trades_seen = int(state.get("trades_seen", 0))
        self.bootstrap_trades = int(state.get("bootstrap_trades", 10))
        self.exploration_bonus = float(state.get("exploration_bonus", 0.1))
        self.audit_trail = copy.deepcopy(state.get("audit_trail", []))
        self._last_fit_meta = copy.deepcopy(state.get("last_fit_meta", {}))
        self._last_recall_meta = copy.deepcopy(state.get("last_recall_meta", {}))
        
        if self._ready and "pca_mean_" in state and "pca_components_" in state:
            self._pca.mean_ = np.array(state["pca_mean_"], dtype=np.float32)
            self._pca.components_ = np.array(state["pca_components_"], dtype=np.float32)
            self._pca.n_components_ = self._pca.components_.shape[0]
            # Set explained_variance_ and explained_variance_ratio_ to dummy values if not present
            n_components = self._pca.n_components_
            n_features = self._pca.components_.shape[1]
            self._pca.explained_variance_ = np.ones(n_components, dtype=np.float32)
            self._pca.explained_variance_ratio_ = np.ones(n_components, dtype=np.float32) / n_components
            self._pca.singular_values_ = np.ones(n_components, dtype=np.float32)
            
        if self._ready and "kmeans_centers_" in state:
            self._kmeans.cluster_centers_ = np.array(state["kmeans_centers_"], dtype=np.float32)
            self._kmeans.n_clusters = state.get("kmeans_n_clusters", self.n_clusters)