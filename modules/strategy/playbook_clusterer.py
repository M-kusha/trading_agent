# ─────────────────────────────────────────────────────────────
# File: modules/strategy/playbook_clusterer.py
# Enhanced with InfoBus integration & intelligent clustering
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import random
import copy
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None
    KMeans = None

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit
from modules.memory.playbook_memory import PlaybookMemory


class PlaybookClusterer(Module, AnalysisMixin, StateManagementMixin, TradingMixin):
    """
    Enhanced playbook clusterer with InfoBus integration.
    Clusters trading experiences and strategies for pattern recognition and learning.
    Provides intelligent strategy weighting based on historical performance patterns.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        pca_dim: int = 2,
        debug: bool = False,
        audit_log_size: int = 50,
        bootstrap_trades: int = 10,
        exploration_bonus: float = 0.1,
        adaptive_clustering: bool = True,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 200),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.n_clusters = max(2, int(n_clusters))
        self.pca_dim = max(1, int(pca_dim))
        self.debug = bool(debug)
        self.audit_log_size = int(audit_log_size)
        self.bootstrap_trades = int(bootstrap_trades)
        self.exploration_bonus = float(exploration_bonus)
        self.adaptive_clustering = bool(adaptive_clustering)
        
        # Clustering state
        self._reset_models()
        self._ready = False
        self._soft_ready = False
        
        # Enhanced tracking
        self.audit_trail = deque(maxlen=self.audit_log_size)
        self.clustering_history = deque(maxlen=50)
        self.performance_tracking = defaultdict(list)
        
        # Bootstrap and learning parameters
        self.trades_seen = 0
        self.bootstrap_mode = True
        self._min_data_for_clustering = max(self.n_clusters, 3)
        self._last_fit_meta = {}
        self._last_recall_meta = {}
        
        # Clustering quality metrics
        self.clustering_metrics = {
            'silhouette_score': 0.0,
            'inertia': 0.0,
            'cluster_stability': 0.0,
            'last_quality_check': None,
            'quality_trend': 'stable'
        }
        
        # Strategy effectiveness tracking
        self.cluster_effectiveness = defaultdict(lambda: {
            'total_trades': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'avg_confidence': 0.0,
            'last_used': None
        })
        
        # Check sklearn availability
        if not SKLEARN_AVAILABLE:
            self.log_operator_warning("scikit-learn not available - using simplified clustering")
            self._use_simple_clustering = True
        else:
            self._use_simple_clustering = False
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "PlaybookClusterer",
            "logs/strategy/playbook_clusterer.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("PlaybookClusterer")
        
        self.log_operator_info(
            "🧠 Playbook Clusterer initialized",
            n_clusters=self.n_clusters,
            pca_dim=self.pca_dim,
            bootstrap_trades=self.bootstrap_trades,
            sklearn_available=SKLEARN_AVAILABLE
        )

    def _reset_models(self) -> None:
        """Reset clustering models and state"""
        
        if not self._use_simple_clustering:
            self._pca = PCA(n_components=self.pca_dim) if PCA else None
            self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10) if KMeans else None
        else:
            self._pca = None
            self._kmeans = None
        
        self._ready = False
        self._soft_ready = False
        
        # Initialize pseudo centers for bootstrap mode
        self._pseudo_centers = np.random.randn(self.n_clusters, self.pca_dim) * 0.5

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset clustering state
        self._reset_models()
        self.audit_trail.clear()
        self.clustering_history.clear()
        self.performance_tracking.clear()
        
        # Reset learning state
        self.trades_seen = 0
        self.bootstrap_mode = True
        self._soft_ready = False
        
        # Reset metadata
        self._last_fit_meta = {}
        self._last_recall_meta = {}
        
        # Reset metrics
        self.clustering_metrics = {
            'silhouette_score': 0.0,
            'inertia': 0.0,
            'cluster_stability': 0.0,
            'last_quality_check': None,
            'quality_trend': 'stable'
        }
        
        # Reset effectiveness tracking
        self.cluster_effectiveness.clear()
        
        self.log_operator_info("🔄 Playbook Clusterer reset - all clustering data cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and adaptive learning"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - limited clustering functionality")
            return
        
        # Extract context and trading data
        context = extract_standard_context(info_bus)
        clustering_context = self._extract_clustering_context_from_info_bus(info_bus, context)
        
        # Update clustering state based on activity
        self._update_clustering_state(clustering_context)
        
        # Perform adaptive clustering if needed
        if self.adaptive_clustering and self._should_recluster(clustering_context):
            self._perform_adaptive_clustering(info_bus)
        
        # Update InfoBus with clustering status
        self._update_info_bus_with_clustering_data(info_bus)

    def _extract_clustering_context_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract clustering context from InfoBus"""
        
        try:
            # Get trading activity
            recent_trades = info_bus.get('recent_trades', [])
            
            # Check if we have new trade executions
            trade_executed = len(recent_trades) > self.trades_seen
            
            clustering_context = {
                'timestamp': datetime.datetime.now().isoformat(),
                'trade_executed': trade_executed,
                'recent_trades_count': len(recent_trades),
                'market_regime': context.get('regime', 'unknown'),
                'volatility_level': context.get('volatility_level', 'medium'),
                'session_pnl': context.get('session_pnl', 0),
                'new_patterns_detected': self._detect_new_patterns(info_bus),
                'clustering_trigger_score': self._calculate_clustering_trigger_score(info_bus, context)
            }
            
            return clustering_context
            
        except Exception as e:
            self.log_operator_warning(f"Clustering context extraction failed: {e}")
            return {'timestamp': datetime.datetime.now().isoformat()}

    def _detect_new_patterns(self, info_bus: InfoBus) -> bool:
        """Detect if new trading patterns have emerged"""
        
        try:
            # Get recent performance data
            recent_trades = info_bus.get('recent_trades', [])
            
            if len(recent_trades) < 5:
                return False
            
            # Look for significant performance changes
            last_5_pnls = [t.get('pnl', 0) for t in recent_trades[-5:]]
            recent_performance = sum(last_5_pnls)
            
            # Compare to historical patterns
            if hasattr(self, '_last_performance_window'):
                performance_change = abs(recent_performance - self._last_performance_window)
                if performance_change > 100:  # Significant change threshold
                    return True
            
            self._last_performance_window = recent_performance
            return False
            
        except Exception:
            return False

    def _calculate_clustering_trigger_score(self, info_bus: InfoBus, context: Dict[str, Any]) -> float:
        """Calculate score that determines if clustering should be triggered"""
        
        score = 0.0
        
        try:
            # Recent activity score
            recent_trades = info_bus.get('recent_trades', [])
            if len(recent_trades) > self.trades_seen:
                score += 0.3
            
            # Performance volatility score
            if len(recent_trades) >= 10:
                pnls = [t.get('pnl', 0) for t in recent_trades[-10:]]
                pnl_volatility = np.std(pnls) / (abs(np.mean(pnls)) + 1e-6)
                score += min(0.4, pnl_volatility / 5.0)
            
            # Market regime change score
            if context.get('regime') != getattr(self, '_last_regime', 'unknown'):
                score += 0.3
                self._last_regime = context.get('regime')
            
            return min(1.0, score)
            
        except Exception:
            return 0.0

    def _update_clustering_state(self, clustering_context: Dict[str, Any]) -> None:
        """Update clustering state based on new context"""
        
        # Track trading activity
        if clustering_context.get('trade_executed', False):
            self.trades_seen = clustering_context.get('recent_trades_count', self.trades_seen)
        
        # Update bootstrap mode
        if self.trades_seen >= self.bootstrap_trades:
            if self.bootstrap_mode:
                self.log_operator_info(f"🎓 Exiting bootstrap mode after {self.trades_seen} trades")
            self.bootstrap_mode = False
        
        # Update soft readiness
        if self._last_fit_meta.get('features_len', 0) >= 2:
            self._soft_ready = True

    def _should_recluster(self, clustering_context: Dict[str, Any]) -> bool:
        """Determine if clustering should be performed"""
        
        try:
            # Don't recluster too frequently
            if self.clustering_history:
                last_cluster_time = datetime.datetime.fromisoformat(self.clustering_history[-1]['timestamp'])
                time_since_last = datetime.datetime.now() - last_cluster_time
                if time_since_last.total_seconds() < 300:  # 5 minutes minimum
                    return False
            
            # Cluster based on trigger score
            trigger_score = clustering_context.get('clustering_trigger_score', 0)
            if trigger_score > 0.6:
                return True
            
            # Cluster if new patterns detected
            if clustering_context.get('new_patterns_detected', False):
                return True
            
            # Cluster every N trades in bootstrap mode
            if self.bootstrap_mode and self.trades_seen > 0 and self.trades_seen % 5 == 0:
                return True
            
            return False
            
        except Exception:
            return False

    def _perform_adaptive_clustering(self, info_bus: InfoBus) -> None:
        """Perform adaptive clustering with InfoBus integration"""
        
        try:
            # Get playbook memory from InfoBus if available
            playbook_memory = info_bus.get('playbook_memory')
            
            if playbook_memory and hasattr(playbook_memory, '_features'):
                self.fit(playbook_memory)
                
                # Record clustering event
                clustering_record = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'trigger': 'adaptive',
                    'features_count': len(playbook_memory._features),
                    'ready_after': self._ready,
                    'quality_metrics': self.clustering_metrics.copy()
                }
                
                self.clustering_history.append(clustering_record)
                
                self.log_operator_info(
                    "🔄 Adaptive clustering performed",
                    features=len(playbook_memory._features),
                    ready=self._ready
                )
            
        except Exception as e:
            self.log_operator_warning(f"Adaptive clustering failed: {e}")

    def fit(self, memory: PlaybookMemory) -> None:
        """Enhanced fit method with comprehensive error handling and analytics"""
        
        feats = memory._features if hasattr(memory, "_features") else []
        fit_meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "fit_attempted": True,
            "features_len": len(feats),
            "n_clusters": self.n_clusters,
            "pca_dim": self.pca_dim,
            "bootstrap_mode": self.bootstrap_mode,
            "trades_seen": self.trades_seen,
            "use_simple_clustering": self._use_simple_clustering,
            "status": "",
        }
        
        try:
            # Determine minimum required samples
            min_required = self._min_data_for_clustering if not self.bootstrap_mode else 2
            
            if len(feats) < min_required:
                self._ready = False
                fit_meta["status"] = f"Insufficient data: {len(feats)} < {min_required} (bootstrap={self.bootstrap_mode})"
                self._record_audit(fit_meta)
                return
            
            # Prepare feature matrix
            X = np.vstack(feats)
            actual_clusters = min(self.n_clusters, len(X))
            
            if self._use_simple_clustering:
                # Use simple clustering when sklearn is not available
                self._fit_simple_clustering(X, actual_clusters, fit_meta)
            else:
                # Use advanced sklearn-based clustering
                self._fit_sklearn_clustering(X, actual_clusters, fit_meta)
            
            self._ready = True
            self._soft_ready = True
            
            # Update clustering metrics
            self._update_clustering_quality_metrics(X)
            
            fit_meta["status"] = f"Clustering successful: {actual_clusters} clusters from {len(X)} samples"
            fit_meta["clustering_quality"] = self.clustering_metrics.copy()
            
        except Exception as e:
            fit_meta["status"] = f"Clustering failed: {str(e)}"
            fit_meta["error"] = str(e)
            self.log_operator_error(f"Clustering fit failed: {e}")
            
        finally:
            self._last_fit_meta = fit_meta
            self._record_audit(fit_meta)
            
            if self.debug and fit_meta.get("status"):
                print(f"[PlaybookClusterer] {fit_meta['status']}")

    def _fit_simple_clustering(self, X: np.ndarray, actual_clusters: int, fit_meta: Dict) -> None:
        """Simple clustering implementation when sklearn is not available"""
        
        # Simple k-means-like clustering
        n_samples, n_features = X.shape
        
        # Initialize centers randomly
        self._simple_centers = X[np.random.choice(n_samples, actual_clusters, replace=False)]
        
        # Simple iterative clustering (simplified k-means)
        for iteration in range(10):  # Max 10 iterations
            # Assign points to closest centers
            distances = np.linalg.norm(X[:, np.newaxis] - self._simple_centers, axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Update centers
            new_centers = np.array([X[assignments == i].mean(axis=0) 
                                  if np.any(assignments == i) else self._simple_centers[i]
                                  for i in range(actual_clusters)])
            
            # Check convergence
            if np.allclose(self._simple_centers, new_centers, rtol=1e-4):
                break
                
            self._simple_centers = new_centers
        
        # Pad centers if needed
        if actual_clusters < self.n_clusters:
            additional_centers = np.random.randn(self.n_clusters - actual_clusters, n_features) * 0.1
            self._simple_centers = np.vstack([self._simple_centers, additional_centers])
        
        fit_meta["clustering_method"] = "simple"
        fit_meta["iterations"] = iteration + 1

    def _fit_sklearn_clustering(self, X: np.ndarray, actual_clusters: int, fit_meta: Dict) -> None:
        """Advanced sklearn-based clustering"""
        
        # Fit PCA for dimensionality reduction
        max_pca_dim = min(X.shape)
        pca_n = max(1, min(self.pca_dim, max_pca_dim))
        
        self._pca = PCA(n_components=pca_n)
        Z = self._pca.fit_transform(X)
        
        # Fit KMeans clustering
        self._kmeans = KMeans(n_clusters=actual_clusters, random_state=0, n_init=10)
        self._kmeans.fit(Z)
        
        # Handle cluster count discrepancy
        if actual_clusters < self.n_clusters:
            existing_centers = self._kmeans.cluster_centers_
            synthetic_centers = []
            
            for i in range(self.n_clusters - actual_clusters):
                base_idx = i % actual_clusters
                noise = np.random.randn(pca_n) * 0.1
                synthetic = existing_centers[base_idx] + noise
                synthetic_centers.append(synthetic)
            
            if synthetic_centers:
                self._kmeans.cluster_centers_ = np.vstack([existing_centers] + synthetic_centers)
            
            self._kmeans.n_clusters = self.n_clusters
        
        fit_meta.update({
            "clustering_method": "sklearn",
            "pca_dim_actual": pca_n,
            "actual_clusters": actual_clusters,
            "kmeans_inertia": float(self._kmeans.inertia_),
            "pca_explained_variance": float(np.sum(self._pca.explained_variance_ratio_))
        })

    def _update_clustering_quality_metrics(self, X: np.ndarray) -> None:
        """Update clustering quality metrics"""
        
        try:
            if self._use_simple_clustering:
                # Simple quality metrics
                distances = np.linalg.norm(X[:, np.newaxis] - self._simple_centers, axis=2)
                min_distances = np.min(distances, axis=1)
                self.clustering_metrics['inertia'] = float(np.sum(min_distances ** 2))
                self.clustering_metrics['silhouette_score'] = 0.5  # Placeholder
            else:
                # Advanced quality metrics using sklearn
                if hasattr(self._kmeans, 'inertia_'):
                    self.clustering_metrics['inertia'] = float(self._kmeans.inertia_)
                
                # Calculate silhouette score if possible
                try:
                    from sklearn.metrics import silhouette_score
                    Z = self._pca.transform(X)
                    labels = self._kmeans.predict(Z)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(Z, labels)
                        self.clustering_metrics['silhouette_score'] = float(score)
                except ImportError:
                    self.clustering_metrics['silhouette_score'] = 0.5
            
            self.clustering_metrics['last_quality_check'] = datetime.datetime.now().isoformat()
            
            # Determine quality trend
            if len(self.clustering_history) >= 2:
                prev_quality = self.clustering_history[-1].get('quality_metrics', {}).get('silhouette_score', 0)
                current_quality = self.clustering_metrics['silhouette_score']
                
                if current_quality > prev_quality + 0.05:
                    self.clustering_metrics['quality_trend'] = 'improving'
                elif current_quality < prev_quality - 0.05:
                    self.clustering_metrics['quality_trend'] = 'declining'
                else:
                    self.clustering_metrics['quality_trend'] = 'stable'
            
        except Exception as e:
            self.log_operator_warning(f"Quality metrics update failed: {e}")

    def recall(self, features: np.ndarray) -> np.ndarray:
        """Enhanced recall with improved fallback strategies and performance tracking"""
        
        recall_meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "recall_attempted": True,
            "features_shape": features.shape,
            "ready": self._ready,
            "soft_ready": self._soft_ready,
            "bootstrap_mode": self.bootstrap_mode,
        }
        
        try:
            # Enhanced fallback strategy when not ready
            if not self._ready:
                weights = self._generate_fallback_weights(features, recall_meta)
                recall_meta["weights"] = weights.tolist()
                recall_meta["method"] = "fallback"
                self._record_audit(recall_meta)
                self._last_recall_meta = recall_meta
                return weights.astype(np.float32)
            
            # Perform clustering-based recall
            if self._use_simple_clustering:
                weights = self._recall_simple_clustering(features, recall_meta)
            else:
                weights = self._recall_sklearn_clustering(features, recall_meta)
            
            # Apply exploration bonus and post-processing
            weights = self._apply_exploration_and_postprocess(weights, recall_meta)
            
            # Track cluster usage
            dominant_cluster = np.argmax(weights)
            self._track_cluster_usage(dominant_cluster, weights[dominant_cluster])
            
            recall_meta["weights"] = weights.tolist()
            recall_meta["dominant_cluster"] = int(dominant_cluster)
            recall_meta["method"] = "clustering"
            recall_meta["status"] = "Recall successful"
            
        except Exception as e:
            # Ultimate fallback
            weights = self._generate_safe_fallback_weights()
            recall_meta["weights"] = weights.tolist()
            recall_meta["method"] = "error_fallback"
            recall_meta["status"] = f"Recall error: {str(e)}"
            recall_meta["error"] = str(e)
            
            self.log_operator_warning(f"Clustering recall failed: {e}")
        
        finally:
            self._record_audit(recall_meta)
            self._last_recall_meta = recall_meta
            
            if self.debug:
                method = recall_meta.get("method", "unknown")
                dominant = recall_meta.get("dominant_cluster", "N/A")
                print(f"[PlaybookClusterer] Recall: {method}, dominant: {dominant}, weights: {weights}")
        
        return weights.astype(np.float32)

    def _generate_fallback_weights(self, features: np.ndarray, recall_meta: Dict) -> np.ndarray:
        """Generate intelligent fallback weights when clustering is not ready"""
        
        if self.bootstrap_mode:
            # Encourage exploration in bootstrap mode
            weights = np.random.dirichlet(np.ones(self.n_clusters) * 2.0)
            max_idx = np.argmax(weights)
            weights[max_idx] += self.exploration_bonus
            weights = weights / weights.sum()
            recall_meta["status"] = "Bootstrap exploration weights"
        else:
            # Conservative approach when not ready
            weights = np.ones(self.n_clusters, dtype=np.float32) / self.n_clusters
            perturbation = np.random.randn(self.n_clusters) * 0.01
            weights += perturbation
            weights = np.maximum(weights, 0.01)
            weights = weights / weights.sum()
            recall_meta["status"] = "Conservative uniform weights"
        
        return weights

    def _recall_simple_clustering(self, features: np.ndarray, recall_meta: Dict) -> np.ndarray:
        """Recall using simple clustering"""
        
        # Calculate distances to cluster centers
        distances = np.linalg.norm(features.reshape(1, -1) - self._simple_centers, axis=1)
        
        # Ensure we have the right number of distances
        if len(distances) < self.n_clusters:
            distances = np.pad(distances, (0, self.n_clusters - len(distances)), 
                             constant_values=np.max(distances) * 2)
        
        # Convert to weights (inverse distance)
        inv_distances = 1.0 / (distances + 1e-8)
        weights = inv_distances / inv_distances.sum()
        
        recall_meta["distances"] = distances.tolist()
        return weights

    def _recall_sklearn_clustering(self, features: np.ndarray, recall_meta: Dict) -> np.ndarray:
        """Recall using sklearn clustering"""
        
        # Transform features and calculate distances
        z = self._pca.transform(features.reshape(1, -1))
        distances = self._kmeans.transform(z).ravel()
        
        # Ensure correct number of distances
        if len(distances) < self.n_clusters:
            distances = np.pad(distances, (0, self.n_clusters - len(distances)), 
                             constant_values=np.max(distances) * 2)
        
        # Convert to weights
        inv_distances = 1.0 / (distances + 1e-8)
        weights = inv_distances / inv_distances.sum()
        
        recall_meta.update({
            "distances": distances.tolist(),
            "z_features": z.tolist(),
            "pca_components": len(z[0])
        })
        
        return weights

    def _apply_exploration_and_postprocess(self, weights: np.ndarray, recall_meta: Dict) -> np.ndarray:
        """Apply exploration bonus and post-processing to weights"""
        
        # Add exploration component during early learning
        if self.trades_seen < self.bootstrap_trades * 2:
            exploration = np.random.dirichlet(np.ones(self.n_clusters) * 5.0) * 0.1
            weights = 0.9 * weights + 0.1 * exploration
            weights = weights / weights.sum()
            recall_meta["exploration_applied"] = True
        
        # Apply cluster effectiveness adjustments
        effectiveness_adj = self._get_cluster_effectiveness_adjustments()
        if effectiveness_adj is not None:
            weights = weights * effectiveness_adj
            weights = weights / weights.sum()
            recall_meta["effectiveness_adjustment_applied"] = True
        
        return weights

    def _get_cluster_effectiveness_adjustments(self) -> Optional[np.ndarray]:
        """Get effectiveness-based adjustments for cluster weights"""
        
        try:
            if not self.cluster_effectiveness:
                return None
            
            adjustments = np.ones(self.n_clusters)
            
            for i in range(self.n_clusters):
                if i in self.cluster_effectiveness:
                    effectiveness = self.cluster_effectiveness[i]
                    if effectiveness['total_trades'] >= 3:
                        success_rate = effectiveness['successful_trades'] / effectiveness['total_trades']
                        avg_pnl = effectiveness['total_pnl'] / effectiveness['total_trades']
                        
                        # Positive adjustment for successful clusters
                        if success_rate > 0.6 and avg_pnl > 0:
                            adjustments[i] = 1.2
                        # Negative adjustment for poor clusters
                        elif success_rate < 0.3 or avg_pnl < -20:
                            adjustments[i] = 0.8
            
            return adjustments
            
        except Exception:
            return None

    def _generate_safe_fallback_weights(self) -> np.ndarray:
        """Generate safe fallback weights for error cases"""
        
        weights = np.ones(self.n_clusters, dtype=np.float32) / self.n_clusters
        weights[0] += 0.1  # Slight bias to encourage action
        return weights / weights.sum()

    def _track_cluster_usage(self, cluster_id: int, weight: float) -> None:
        """Track cluster usage for effectiveness analysis"""
        
        try:
            self.cluster_effectiveness[cluster_id]['last_used'] = datetime.datetime.now().isoformat()
            
            # Update usage statistics would be done when trade results are available
            # This is a placeholder for tracking
            
        except Exception as e:
            self.log_operator_warning(f"Cluster usage tracking failed: {e}")

    def update_cluster_effectiveness(self, cluster_id: int, trade_successful: bool, pnl: float, confidence: float) -> None:
        """Update cluster effectiveness based on trade results"""
        
        try:
            if 0 <= cluster_id < self.n_clusters:
                effectiveness = self.cluster_effectiveness[cluster_id]
                effectiveness['total_trades'] += 1
                
                if trade_successful:
                    effectiveness['successful_trades'] += 1
                
                effectiveness['total_pnl'] += pnl
                effectiveness['avg_confidence'] = (
                    (effectiveness['avg_confidence'] * (effectiveness['total_trades'] - 1) + confidence) /
                    effectiveness['total_trades']
                )
                
                # Log significant updates
                if effectiveness['total_trades'] % 10 == 0:
                    success_rate = effectiveness['successful_trades'] / effectiveness['total_trades']
                    avg_pnl = effectiveness['total_pnl'] / effectiveness['total_trades']
                    
                    self.log_operator_info(
                        f"📊 Cluster {cluster_id} effectiveness update",
                        trades=effectiveness['total_trades'],
                        success_rate=f"{success_rate:.1%}",
                        avg_pnl=f"€{avg_pnl:.2f}"
                    )
            
        except Exception as e:
            self.log_operator_warning(f"Cluster effectiveness update failed: {e}")

    def _record_audit(self, entry: Dict[str, Any]) -> None:
        """Record audit entry with enhanced information"""
        
        try:
            # Add contextual information
            entry.update({
                'trades_seen': self.trades_seen,
                'bootstrap_mode': self.bootstrap_mode,
                'clustering_ready': self._ready,
                'soft_ready': self._soft_ready
            })
            
            self.audit_trail.append(entry)
            
            # Log significant events
            status = entry.get('status', '')
            if 'successful' in status.lower() or 'error' in status.lower():
                level = 'info' if 'successful' in status.lower() else 'warning'
                getattr(self.logger, f'log_operator_{level}')(
                    f"🧠 Clustering event: {status[:50]}",
                    method=entry.get('method', 'unknown'),
                    ready=self._ready
                )
            
        except Exception as e:
            print(f"[PlaybookClusterer] Audit recording failed: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with clustering quality metrics"""
        
        try:
            # Basic readiness indicators
            ready_float = float(self._ready)
            soft_ready_float = float(self._soft_ready)
            bootstrap_float = float(self.bootstrap_mode)
            
            # Progress indicators
            data_ratio = min(1.0, self._last_fit_meta.get('features_len', 0) / max(self.n_clusters, 10))
            trade_ratio = min(1.0, self.trades_seen / max(self.bootstrap_trades, 10))
            
            # Clustering quality metrics
            clustering_quality = self.clustering_metrics.get('silhouette_score', 0.5)
            inertia_normalized = min(1.0, self.clustering_metrics.get('inertia', 0) / 1000.0)
            
            # Cluster effectiveness metrics
            if self.cluster_effectiveness:
                avg_success_rate = np.mean([
                    eff['successful_trades'] / max(1, eff['total_trades'])
                    for eff in self.cluster_effectiveness.values()
                    if eff['total_trades'] > 0
                ])
                avg_pnl_per_cluster = np.mean([
                    eff['total_pnl'] / max(1, eff['total_trades'])
                    for eff in self.cluster_effectiveness.values()
                    if eff['total_trades'] > 0
                ])
            else:
                avg_success_rate = 0.5
                avg_pnl_per_cluster = 0.0
            
            observation = np.array([
                self.n_clusters / 10.0,  # Normalized cluster count
                self.pca_dim / 10.0,     # Normalized PCA dimensions
                ready_float,              # Clustering ready
                soft_ready_float,         # Soft ready
                bootstrap_float,          # Bootstrap mode
                data_ratio,              # Data availability ratio
                trade_ratio,             # Trading progress ratio
                clustering_quality,       # Clustering quality score
                self.exploration_bonus,   # Exploration bonus
                float(self.trades_seen) / 100.0,  # Normalized trades seen
                avg_success_rate,        # Average cluster success rate
                np.clip(avg_pnl_per_cluster / 50.0, -1.0, 1.0)  # Normalized avg PnL
            ], dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.log_operator_error(f"Invalid clustering observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Clustering observation generation failed: {e}")
            return np.full(12, 0.5, dtype=np.float32)

    def _update_info_bus_with_clustering_data(self, info_bus: InfoBus) -> None:
        """Update InfoBus with clustering analysis results"""
        
        try:
            # Prepare clustering data
            clustering_data = {
                'ready': self._ready,
                'soft_ready': self._soft_ready,
                'bootstrap_mode': self.bootstrap_mode,
                'trades_seen': self.trades_seen,
                'n_clusters': self.n_clusters,
                'clustering_quality': self.clustering_metrics.copy(),
                'cluster_effectiveness': {k: v.copy() for k, v in self.cluster_effectiveness.items()},
                'last_fit_meta': self._last_fit_meta.copy(),
                'last_recall_meta': self._last_recall_meta.copy(),
                'sklearn_available': SKLEARN_AVAILABLE,
                'adaptive_clustering': self.adaptive_clustering
            }
            
            # Add to InfoBus
            InfoBusUpdater.add_module_data(info_bus, 'playbook_clusterer', clustering_data)
            
            # Add clustering readiness alerts
            if not self._ready and self.trades_seen >= self.bootstrap_trades:
                InfoBusUpdater.add_alert(
                    info_bus,
                    "Clustering not ready despite sufficient trades",
                    'playbook_clusterer',
                    'warning',
                    {'trades_seen': self.trades_seen, 'min_required': self._min_data_for_clustering}
                )
            
            # Add quality alerts
            quality_score = self.clustering_metrics.get('silhouette_score', 0.5)
            if quality_score < 0.3 and self._ready:
                InfoBusUpdater.add_alert(
                    info_bus,
                    "Low clustering quality detected",
                    'playbook_clusterer',
                    'warning',
                    {'quality_score': quality_score}
                )
            
        except Exception as e:
            self.log_operator_warning(f"InfoBus clustering update failed: {e}")

    def get_clustering_report(self) -> str:
        """Generate comprehensive clustering analysis report"""
        
        # Cluster effectiveness summary
        effectiveness_summary = ""
        if self.cluster_effectiveness:
            for cluster_id, effectiveness in self.cluster_effectiveness.items():
                if effectiveness['total_trades'] > 0:
                    success_rate = effectiveness['successful_trades'] / effectiveness['total_trades']
                    avg_pnl = effectiveness['total_pnl'] / effectiveness['total_trades']
                    status = "🟢" if success_rate > 0.6 else "🔴" if success_rate < 0.4 else "🟡"
                    effectiveness_summary += f"  • Cluster {cluster_id}: {effectiveness['total_trades']} trades, {success_rate:.1%} success, €{avg_pnl:+.1f} avg {status}\n"
        
        # Recent clustering activity
        recent_activity = ""
        if self.clustering_history:
            for event in list(self.clustering_history)[-3:]:
                timestamp = event['timestamp'][:19].replace('T', ' ')
                trigger = event.get('trigger', 'unknown')
                features = event.get('features_count', 0)
                recent_activity += f"  • {timestamp}: {trigger} clustering, {features} features\n"
        
        return f"""
🧠 PLAYBOOK CLUSTERER REPORT
═══════════════════════════════════════
🎯 Clustering Status:
• Ready: {'✅ Yes' if self._ready else '❌ No'}
• Soft Ready: {'✅ Yes' if self._soft_ready else '❌ No'}
• Bootstrap Mode: {'🎓 Active' if self.bootstrap_mode else '✅ Complete'}
• Trades Seen: {self.trades_seen} / {self.bootstrap_trades} (bootstrap threshold)

⚙️ Configuration:
• Clusters: {self.n_clusters}
• PCA Dimensions: {self.pca_dim}
• Scikit-learn: {'✅ Available' if SKLEARN_AVAILABLE else '❌ Using simple clustering'}
• Adaptive Clustering: {'✅ Enabled' if self.adaptive_clustering else '❌ Disabled'}
• Exploration Bonus: {self.exploration_bonus:.1%}

📊 Clustering Quality:
• Silhouette Score: {self.clustering_metrics.get('silhouette_score', 0):.3f}
• Inertia: {self.clustering_metrics.get('inertia', 0):.1f}
• Quality Trend: {self.clustering_metrics.get('quality_trend', 'unknown').title()}
• Last Check: {self.clustering_metrics.get('last_quality_check', 'Never')[:19]}

🎯 Cluster Effectiveness:
{effectiveness_summary if effectiveness_summary else '  📭 No cluster effectiveness data yet'}

🔄 Recent Activity:
{recent_activity if recent_activity else '  📭 No recent clustering activity'}

📈 Performance Metrics:
• Audit Trail: {len(self.audit_trail)} entries
• Clustering History: {len(self.clustering_history)} events
• Last Fit: {'✅ Success' if self._last_fit_meta.get('status', '').startswith('Clustering successful') else '❌ Failed/None'}
• Last Recall: {'✅ Success' if self._last_recall_meta.get('status') == 'Recall successful' else '❌ Failed/None'}
        """

    # ================== EVOLUTIONARY METHODS ==================

    def mutate(self, std: float = 1.0) -> None:
        """Enhanced mutation with adaptive parameters"""
        
        pre_config = (self.n_clusters, self.pca_dim, self.exploration_bonus)
        
        # Mutate core parameters
        if random.random() < 0.4:
            self.n_clusters = max(2, min(10, int(self.n_clusters + np.random.randint(-1, 2))))
        
        if random.random() < 0.4:
            self.pca_dim = max(1, min(8, int(self.pca_dim + np.random.randint(-1, 2))))
        
        if random.random() < 0.3:
            self.exploration_bonus = max(0.0, min(0.5, self.exploration_bonus + np.random.normal(0, 0.05)))
        
        # Update dependent parameters
        self._min_data_for_clustering = max(self.n_clusters, 3)
        self._reset_models()
        
        post_config = (self.n_clusters, self.pca_dim, self.exploration_bonus)
        
        # Record mutation
        mutation_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "mutation": True,
            "before": pre_config,
            "after": post_config,
            "exploration_bonus": self.exploration_bonus
        }
        self._record_audit(mutation_record)
        
        self.log_operator_info(
            f"🧬 Clustering mutation applied",
            clusters=f"{pre_config[0]} → {post_config[0]}",
            pca_dim=f"{pre_config[1]} → {post_config[1]}",
            exploration=f"{pre_config[2]:.2f} → {post_config[2]:.2f}"
        )

    def crossover(self, other: "PlaybookClusterer") -> "PlaybookClusterer":
        """Enhanced crossover with parameter inheritance"""
        
        # Select parameters from parents
        n_clusters = self.n_clusters if random.random() < 0.5 else other.n_clusters
        pca_dim = self.pca_dim if random.random() < 0.5 else other.pca_dim
        debug = self.debug or other.debug
        
        # Average numerical parameters
        exploration_bonus = (self.exploration_bonus + other.exploration_bonus) / 2
        bootstrap_trades = (self.bootstrap_trades + other.bootstrap_trades) // 2
        
        # Select boolean parameters
        adaptive_clustering = self.adaptive_clustering if random.random() < 0.5 else other.adaptive_clustering
        
        # Create offspring
        offspring = PlaybookClusterer(
            n_clusters=n_clusters,
            pca_dim=pca_dim,
            debug=debug,
            bootstrap_trades=bootstrap_trades,
            exploration_bonus=exploration_bonus,
            adaptive_clustering=adaptive_clustering
        )
        
        # Record crossover
        crossover_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "crossover": True,
            "from_self": (self.n_clusters, self.pca_dim, self.exploration_bonus),
            "from_other": (other.n_clusters, other.pca_dim, other.exploration_bonus),
            "result": (n_clusters, pca_dim, exploration_bonus)
        }
        offspring._record_audit(crossover_record)
        
        return offspring

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        
        state = {
            "config": {
                "n_clusters": self.n_clusters,
                "pca_dim": self.pca_dim,
                "debug": self.debug,
                "bootstrap_trades": self.bootstrap_trades,
                "exploration_bonus": self.exploration_bonus,
                "adaptive_clustering": self.adaptive_clustering
            },
            "clustering_state": {
                "_ready": self._ready,
                "_soft_ready": self._soft_ready,
                "bootstrap_mode": self.bootstrap_mode,
                "trades_seen": self.trades_seen,
                "use_simple_clustering": self._use_simple_clustering
            },
            "metrics": {
                "clustering_metrics": self.clustering_metrics.copy(),
                "cluster_effectiveness": {k: v.copy() for k, v in self.cluster_effectiveness.items()},
                "performance_tracking": {k: list(v) for k, v in self.performance_tracking.items()}
            },
            "history": {
                "audit_trail": list(self.audit_trail),
                "clustering_history": list(self.clustering_history),
                "last_fit_meta": self._last_fit_meta.copy(),
                "last_recall_meta": self._last_recall_meta.copy()
            }
        }
        
        # Add model states if available
        if self._ready:
            if self._use_simple_clustering and hasattr(self, '_simple_centers'):
                state["model_state"] = {
                    "simple_centers": self._simple_centers.tolist()
                }
            elif not self._use_simple_clustering and self._pca and self._kmeans:
                state["model_state"] = {
                    "pca_mean_": self._pca.mean_.tolist(),
                    "pca_components_": self._pca.components_.tolist(),
                    "kmeans_centers_": self._kmeans.cluster_centers_.tolist(),
                    "kmeans_n_clusters": self._kmeans.n_clusters
                }
        
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.n_clusters = int(config.get("n_clusters", self.n_clusters))
        self.pca_dim = int(config.get("pca_dim", self.pca_dim))
        self.debug = bool(config.get("debug", self.debug))
        self.bootstrap_trades = int(config.get("bootstrap_trades", self.bootstrap_trades))
        self.exploration_bonus = float(config.get("exploration_bonus", self.exploration_bonus))
        self.adaptive_clustering = bool(config.get("adaptive_clustering", self.adaptive_clustering))
        
        # Load clustering state
        clustering_state = state.get("clustering_state", {})
        self._ready = bool(clustering_state.get("_ready", False))
        self._soft_ready = bool(clustering_state.get("_soft_ready", False))
        self.bootstrap_mode = bool(clustering_state.get("bootstrap_mode", True))
        self.trades_seen = int(clustering_state.get("trades_seen", 0))
        self._use_simple_clustering = bool(clustering_state.get("use_simple_clustering", not SKLEARN_AVAILABLE))
        
        # Load metrics
        metrics = state.get("metrics", {})
        self.clustering_metrics = metrics.get("clustering_metrics", self.clustering_metrics)
        
        effectiveness_data = metrics.get("cluster_effectiveness", {})
        self.cluster_effectiveness = defaultdict(lambda: {
            'total_trades': 0, 'successful_trades': 0, 'total_pnl': 0.0,
            'avg_confidence': 0.0, 'last_used': None
        })
        for k, v in effectiveness_data.items():
            self.cluster_effectiveness[int(k)] = v
        
        performance_data = metrics.get("performance_tracking", {})
        self.performance_tracking = defaultdict(list)
        for k, v in performance_data.items():
            self.performance_tracking[k] = list(v)
        
        # Load history
        history = state.get("history", {})
        self.audit_trail = deque(history.get("audit_trail", []), maxlen=self.audit_log_size)
        self.clustering_history = deque(history.get("clustering_history", []), maxlen=50)
        self._last_fit_meta = history.get("last_fit_meta", {})
        self._last_recall_meta = history.get("last_recall_meta", {})
        
        # Load model state if available
        model_state = state.get("model_state", {})
        if model_state and self._ready:
            if self._use_simple_clustering and "simple_centers" in model_state:
                self._simple_centers = np.array(model_state["simple_centers"], dtype=np.float32)
            elif not self._use_simple_clustering and all(k in model_state for k in ["pca_mean_", "pca_components_", "kmeans_centers_"]):
                self._restore_sklearn_models(model_state)
        
        # Update dependent parameters
        self._min_data_for_clustering = max(self.n_clusters, 3)

    def _restore_sklearn_models(self, model_state: Dict[str, Any]) -> None:
        """Restore sklearn models from state"""
        
        try:
            if SKLEARN_AVAILABLE:
                # Restore PCA
                self._pca = PCA(n_components=self.pca_dim)
                self._pca.mean_ = np.array(model_state["pca_mean_"], dtype=np.float32)
                self._pca.components_ = np.array(model_state["pca_components_"], dtype=np.float32)
                self._pca.n_components_ = self._pca.components_.shape[0]
                
                # Set dummy explained variance values
                n_components = self._pca.n_components_
                self._pca.explained_variance_ = np.ones(n_components, dtype=np.float32)
                self._pca.explained_variance_ratio_ = np.ones(n_components, dtype=np.float32) / n_components
                self._pca.singular_values_ = np.ones(n_components, dtype=np.float32)
                
                # Restore KMeans
                self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
                self._kmeans.cluster_centers_ = np.array(model_state["kmeans_centers_"], dtype=np.float32)
                self._kmeans.n_clusters = model_state.get("kmeans_n_clusters", self.n_clusters)
                
                self.log_operator_info("🔄 Sklearn models restored from state")
            
        except Exception as e:
            self.log_operator_warning(f"Sklearn model restoration failed: {e}")
            self._reset_models()