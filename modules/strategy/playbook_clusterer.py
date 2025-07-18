"""
🧠 Enhanced Playbook Clusterer with SmartInfoBus Integration v3.0
Advanced clustering system for trading pattern recognition and strategy optimization
"""

import asyncio
import time
import numpy as np
import datetime
import random
import copy
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None
    KMeans = None
    silhouette_score = None

# Type aliases for better type safety
from typing import Union, TYPE_CHECKING

# More specific type for cluster effectiveness to avoid Union issues
class ClusterEffectivenessDict(dict):
    """Typed dictionary for cluster effectiveness data"""
    def __init__(self):
        super().__init__()
        self.update({
            'total_trades': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'avg_confidence': 0.0,
            'avg_trade_duration': 0.0,
            'best_conditions': [],
            'worst_conditions': [],
            'last_used': None,
            'effectiveness_score': 0.5
        })

# Conditional import for type checking
if TYPE_CHECKING:
    from modules.memory.playbook_memory import PlaybookMemory
else:
    # Runtime fallback - PlaybookMemory will be Any type
    PlaybookMemory = Any

# ═══════════════════════════════════════════════════════════════════
# MODERN SMARTINFOBUS IMPORTS
# ═══════════════════════════════════════════════════════════════════
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker

# Conditional import to handle potential circular dependencies
try:
    from modules.memory.playbook_memory import PlaybookMemory
except ImportError:
    # Fallback if PlaybookMemory is not available
    PlaybookMemory = Any


@module(
    name="PlaybookClusterer",
    version="3.0.0",
    category="strategy",
    provides=[
        "cluster_weights", "cluster_analysis", "clustering_health", "cluster_recommendations",
        "cluster_effectiveness", "pattern_analysis", "clustering_thesis"
    ],
    requires=[
        "playbook_memory", "recent_trades", "market_data", "trading_performance", 
        "market_regime", "session_metrics"
    ],
    description="Advanced playbook clustering with intelligent pattern recognition and strategy optimization",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=200,
    priority=7,
    explainable=True,
    hot_reload=True
)
class PlaybookClusterer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🧠 PRODUCTION-GRADE Playbook Clusterer v3.0
    
    Advanced clustering system for trading pattern recognition with:
    - Intelligent clustering using sklearn with robust fallbacks
    - Dynamic cluster effectiveness tracking and optimization
    - Adaptive learning with performance-based weight adjustment
    - SmartInfoBus zero-wiring architecture
    - Comprehensive thesis generation for all clustering decisions
    - Circuit breaker protection and error recovery
    """

    def _initialize(self):
        """Initialize advanced clustering and pattern recognition systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced clustering configuration
        self.n_clusters = self.config.get('n_clusters', 5)
        self.pca_dim = self.config.get('pca_dim', 2)
        self.bootstrap_trades = self.config.get('bootstrap_trades', 10)
        self.exploration_bonus = self.config.get('exploration_bonus', 0.1)
        self.adaptive_clustering = self.config.get('adaptive_clustering', True)
        self.debug = self.config.get('debug', False)
        
        # Check sklearn availability and setup clustering approach
        self._use_simple_clustering = not SKLEARN_AVAILABLE
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available - using advanced simple clustering")
        
        # Initialize clustering models and state
        self._reset_clustering_models()
        
        # Core clustering state
        self._ready = False
        self._soft_ready = False
        self.bootstrap_mode = True
        self.trades_seen = 0
        self._min_data_for_clustering = max(self.n_clusters, 3)
        
        # Enhanced tracking systems
        self.clustering_history = deque(maxlen=100)
        self.performance_tracking = defaultdict(list)
        self._last_fit_meta = {}
        self._last_recall_meta = {}
        
        # Clustering quality metrics
        self.clustering_metrics = {
            'silhouette_score': 0.0,
            'inertia': 0.0,
            'cluster_stability': 0.0,
            'quality_trend': 'stable',
            'last_quality_check': None,
            'data_coverage': 0.0,
            'feature_diversity': 0.0
        }
        
        # Advanced cluster effectiveness tracking
        self.cluster_effectiveness: Dict[int, ClusterEffectivenessDict] = {}
        for i in range(self.n_clusters):
            self.cluster_effectiveness[i] = ClusterEffectivenessDict()
        
        # Pattern recognition state
        self.pattern_cache = {}
        self.pattern_performance = defaultdict(float)
        self._last_performance_window = 0.0
        self._last_regime = 'unknown'
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Advanced clustering intelligence
        self.clustering_intelligence = {
            'reclustering_threshold': 0.7,
            'quality_improvement_threshold': 0.05,
            'effectiveness_decay': 0.95,
            'pattern_memory_factor': 0.9
        }
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🧠",
            message=f"Playbook Clusterer v{version} initialized",
            clusters=self.n_clusters,
            sklearn_available=SKLEARN_AVAILABLE,
            adaptive=self.adaptive_clustering
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="PlaybookClusterer",
            log_path="logs/strategy/playbook_clusterer.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PlaybookClusterer", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _reset_clustering_models(self) -> None:
        """Reset clustering models and state with enhanced initialization"""
        try:
            if not self._use_simple_clustering and SKLEARN_AVAILABLE and PCA is not None and KMeans is not None:
                self._pca = PCA(n_components=self.pca_dim)
                self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
            else:
                self._pca = None
                self._kmeans = None
            
            # Initialize advanced pseudo centers for bootstrap
            self._pseudo_centers = self._generate_intelligent_pseudo_centers()
            self._simple_centers = None
            
            self._ready = False
            self._soft_ready = False
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "model_reset")
            self.logger.error(f"Model reset failed: {error_context}")

    def _generate_intelligent_pseudo_centers(self) -> np.ndarray:
        """Generate intelligent pseudo centers for bootstrap mode"""
        try:
            # Create diverse initial centers based on common trading patterns
            centers = []
            
            # Generate centers representing different trading scenarios
            for i in range(self.n_clusters):
                # Trend-following patterns
                if i % 5 == 0:
                    center = np.array([0.8, 0.6] + [0.0] * (self.pca_dim - 2))
                # Mean reversion patterns  
                elif i % 5 == 1:
                    center = np.array([-0.3, 0.9] + [0.0] * (self.pca_dim - 2))
                # Breakout patterns
                elif i % 5 == 2:
                    center = np.array([0.9, -0.4] + [0.0] * (self.pca_dim - 2))
                # Consolidation patterns
                elif i % 5 == 3:
                    center = np.array([0.1, 0.1] + [0.0] * (self.pca_dim - 2))
                # Volatile patterns
                else:
                    center = np.array([-0.7, -0.8] + [0.0] * (self.pca_dim - 2))
                
                # Add noise for diversity
                noise = np.random.randn(self.pca_dim) * 0.2
                centers.append(center[:self.pca_dim] + noise)
            
            return np.array(centers, dtype=np.float32)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "pseudo_centers")
            # Fallback to random centers
            return np.random.randn(self.n_clusters, self.pca_dim) * 0.5

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Playbook Clusterer v3.0 Initialization Complete:
        
        Advanced Clustering System:
        - Cluster configuration: {self.n_clusters} clusters with {self.pca_dim}-dimensional PCA reduction
        - Machine learning backend: {'Scikit-learn' if SKLEARN_AVAILABLE else 'Advanced simple clustering'}
        - Pattern recognition: Intelligent pseudo-centers with trading pattern templates
        - Adaptive learning: Dynamic reclustering based on performance and market evolution
        
        Current Configuration:
        - Clusters: {self.n_clusters} distinct pattern groups
        - PCA dimensions: {self.pca_dim} for efficient feature representation
        - Bootstrap trades: {self.bootstrap_trades} required for initial learning
        - Exploration bonus: {self.exploration_bonus:.1%} for pattern discovery
        - Adaptive clustering: {'Enabled' if self.adaptive_clustering else 'Disabled'}
        
        Intelligence Features:
        - Cluster effectiveness tracking with detailed performance metrics
        - Pattern recognition and caching for rapid strategy recall
        - Market regime integration for context-aware clustering
        - Quality metrics monitoring with automated improvement detection
        
        Advanced Capabilities:
        - Real-time cluster weight adaptation based on effectiveness
        - Comprehensive pattern analysis with historical optimization
        - Circuit breaker protection for robust error handling
        - State persistence for hot-reload and system continuity
        
        Expected Outcomes:
        - Optimal strategy selection through intelligent pattern clustering
        - Enhanced performance via dynamic cluster effectiveness weighting
        - Adaptive learning that improves pattern recognition over time
        - Transparent clustering decisions with comprehensive explanations
        """
        
        self.smart_bus.set('playbook_clusterer_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'n_clusters': self.n_clusters,
                'pca_dim': self.pca_dim,
                'sklearn_available': SKLEARN_AVAILABLE,
                'adaptive_clustering': self.adaptive_clustering
            }
        }, module='PlaybookClusterer', thesis=thesis)

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive clustering analysis
        
        Args:
            **inputs: Input parameters from the base module system
        
        Returns:
            Dict containing cluster weights, analysis, and recommendations
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive data from SmartInfoBus
            clustering_data = await self._get_comprehensive_clustering_data()
            
            # Update clustering state based on new data
            state_update = await self._update_clustering_state_comprehensive(clustering_data)
            
            # Perform adaptive clustering if needed
            clustering_update = await self._perform_adaptive_clustering_comprehensive(clustering_data)
            
            # Generate cluster analysis and weights
            cluster_analysis = await self._analyze_clusters_comprehensive(clustering_data)
            
            # Update cluster effectiveness based on recent performance
            effectiveness_update = await self._update_cluster_effectiveness_comprehensive(clustering_data)
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_clustering_thesis(
                cluster_analysis, effectiveness_update, clustering_update
            )
            
            # Create comprehensive results
            results = {
                'cluster_weights': cluster_analysis.get('weights', self._generate_safe_fallback_weights()),
                'cluster_analysis': cluster_analysis,
                'clustering_health': self._get_clustering_health_metrics(),
                'cluster_recommendations': self._generate_intelligent_cluster_recommendations(cluster_analysis),
                'cluster_effectiveness': self._get_cluster_effectiveness_summary(),
                'pattern_analysis': self._get_pattern_analysis_summary(),
                'clustering_thesis': thesis
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('PlaybookClusterer', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_clustering_data(self) -> Dict[str, Any]:
        """Get comprehensive data using modern SmartInfoBus patterns"""
        try:
            return {
                'playbook_memory': self.smart_bus.get('playbook_memory', 'PlaybookClusterer'),
                'recent_trades': self.smart_bus.get('recent_trades', 'PlaybookClusterer') or [],
                'market_data': self.smart_bus.get('market_data', 'PlaybookClusterer') or {},
                'trading_performance': self.smart_bus.get('trading_performance', 'PlaybookClusterer') or {},
                'market_regime': self.smart_bus.get('market_regime', 'PlaybookClusterer') or 'unknown',
                'session_metrics': self.smart_bus.get('session_metrics', 'PlaybookClusterer') or {},
                'volatility_data': self.smart_bus.get('volatility_data', 'PlaybookClusterer') or {},
                'market_context': self.smart_bus.get('market_context', 'PlaybookClusterer') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "PlaybookClusterer")
            self.logger.warning(f"Clustering data retrieval incomplete: {error_context}")
            return self._get_safe_clustering_defaults()

    async def _update_clustering_state_comprehensive(self, clustering_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update clustering state with comprehensive analysis"""
        try:
            state_update = {
                'timestamp': datetime.datetime.now().isoformat(),
                'trades_before': self.trades_seen,
                'bootstrap_before': self.bootstrap_mode,
                'ready_before': self._ready
            }
            
            # Update trade count
            recent_trades = clustering_data.get('recent_trades', [])
            new_trade_count = len(recent_trades)
            
            if new_trade_count > self.trades_seen:
                state_update['new_trades'] = new_trade_count - self.trades_seen
                self.trades_seen = new_trade_count
                
                # Check bootstrap completion
                if self.trades_seen >= self.bootstrap_trades and self.bootstrap_mode:
                    self.bootstrap_mode = False
                    state_update['bootstrap_completed'] = True
                    self.logger.info(format_operator_message(
                        icon="🎓",
                        message="Bootstrap mode completed",
                        trades_seen=self.trades_seen,
                        threshold=self.bootstrap_trades
                    ))
            
            # Update pattern detection
            new_patterns = await self._detect_new_patterns_comprehensive(clustering_data)
            state_update['new_patterns_detected'] = new_patterns
            
            # Update readiness based on data availability
            playbook_memory = clustering_data.get('playbook_memory')
            if playbook_memory and hasattr(playbook_memory, '_features'):
                feature_count = len(playbook_memory._features)
                if feature_count >= self._min_data_for_clustering:
                    self._soft_ready = True
                    state_update['soft_ready_achieved'] = True
            
            state_update.update({
                'trades_after': self.trades_seen,
                'bootstrap_after': self.bootstrap_mode,
                'ready_after': self._ready,
                'soft_ready_after': self._soft_ready
            })
            
            return state_update
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_update")
            self.logger.warning(f"Clustering state update failed: {error_context}")
            return {'error': str(error_context)}

    async def _detect_new_patterns_comprehensive(self, clustering_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect new trading patterns with comprehensive analysis"""
        try:
            pattern_detection: Dict[str, Any] = {
                'performance_shift_detected': False,
                'regime_change_detected': False,
                'volatility_shift_detected': False,
                'pattern_diversity_change': False
            }
            
            # Detect performance shifts
            recent_trades = clustering_data.get('recent_trades', [])
            if len(recent_trades) >= 10:
                recent_pnls = [t.get('pnl', 0) for t in recent_trades[-10:]]
                recent_performance = sum(recent_pnls)
                
                performance_change = abs(recent_performance - self._last_performance_window)
                if performance_change > 100:  # Significant change threshold
                    pattern_detection['performance_shift_detected'] = True
                    pattern_detection['performance_change_value'] = performance_change
                
                self._last_performance_window = recent_performance
            
            # Detect regime changes
            current_regime = clustering_data.get('market_regime', 'unknown')
            if current_regime != self._last_regime:
                pattern_detection['regime_change_detected'] = True
                pattern_detection['regime_change_details'] = f"{self._last_regime} → {current_regime}"
                self._last_regime = current_regime
            
            # Detect volatility shifts
            market_context = clustering_data.get('market_context', {})
            volatility_level = market_context.get('volatility_level', 'medium')
            if hasattr(self, '_last_volatility') and volatility_level != self._last_volatility:
                pattern_detection['volatility_shift_detected'] = True
                pattern_detection['volatility_shift_details'] = f"{getattr(self, '_last_volatility', 'unknown')} → {volatility_level}"
            self._last_volatility = volatility_level
            
            return pattern_detection
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "pattern_detection")
            return {'error': str(error_context)}

    async def _perform_adaptive_clustering_comprehensive(self, clustering_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive adaptive clustering with detailed analysis"""
        try:
            clustering_update = {
                'clustering_performed': False,
                'clustering_reason': None,
                'clustering_results': {},
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Check if clustering should be performed
            should_cluster, reason = await self._should_perform_clustering(clustering_data)
            
            if should_cluster:
                playbook_memory = clustering_data.get('playbook_memory')
                
                if playbook_memory and hasattr(playbook_memory, '_features'):
                    # Perform clustering
                    clustering_results = await self._fit_comprehensive(playbook_memory)
                    
                    clustering_update.update({
                        'clustering_performed': True,
                        'clustering_reason': reason,
                        'clustering_results': clustering_results,
                        'ready_after_clustering': self._ready,
                        'quality_metrics': self.clustering_metrics.copy()
                    })
                    
                    # Record clustering event
                    clustering_record = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'trigger': reason,
                        'features_count': len(playbook_memory._features),
                        'ready_after': self._ready,
                        'quality_metrics': self.clustering_metrics.copy(),
                        'effectiveness_scores': {k: v['effectiveness_score'] for k, v in self.cluster_effectiveness.items()}
                    }
                    
                    self.clustering_history.append(clustering_record)
                    
                    self.logger.info(format_operator_message(
                        icon="🔄",
                        message=f"Adaptive clustering: {reason}",
                        features=len(playbook_memory._features),
                        ready=self._ready,
                        quality=f"{self.clustering_metrics.get('silhouette_score', 0):.3f}"
                    ))
            
            return clustering_update
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adaptive_clustering")
            return {'error': str(error_context), 'clustering_performed': False}

    async def _should_perform_clustering(self, clustering_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Determine if clustering should be performed with detailed reasoning"""
        try:
            # Don't cluster too frequently
            if self.clustering_history:
                last_cluster_time = datetime.datetime.fromisoformat(self.clustering_history[-1]['timestamp'])
                time_since_last = datetime.datetime.now() - last_cluster_time
                if time_since_last.total_seconds() < 300:  # 5 minutes minimum
                    return False, "clustering_cooldown"
            
            # Check for sufficient data
            playbook_memory = clustering_data.get('playbook_memory')
            if not playbook_memory:
                return False, "no_playbook_memory"
            
            # Check if playbook memory has features attribute
            if not hasattr(playbook_memory, '_features'):
                return False, "no_features_attribute"
                
            feature_count = len(playbook_memory._features)
            if feature_count < self._min_data_for_clustering:
                return False, f"insufficient_data_{feature_count}"
            
            # Bootstrap mode clustering
            if self.bootstrap_mode and self.trades_seen > 0 and self.trades_seen % 5 == 0:
                return True, "bootstrap_interval"
            
            # Performance shift detection
            if await self._calculate_clustering_trigger_score(clustering_data) > self.clustering_intelligence['reclustering_threshold']:
                return True, "performance_shift"
            
            # Quality improvement opportunity
            if not self._ready or self.clustering_metrics.get('silhouette_score', 0) < 0.4:
                return True, "quality_improvement"
            
            # Regular maintenance clustering
            if self.adaptive_clustering and self.trades_seen > 0 and self.trades_seen % 20 == 0:
                return True, "maintenance_clustering"
            
            return False, "no_trigger"
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "clustering_decision")
            return False, f"decision_error_{error_context}"

    async def _calculate_clustering_trigger_score(self, clustering_data: Dict[str, Any]) -> float:
        """Calculate comprehensive trigger score for clustering decisions"""
        try:
            score = 0.0
            
            # Recent activity score (30% weight)
            recent_trades = clustering_data.get('recent_trades', [])
            if len(recent_trades) > self.trades_seen:
                activity_factor = min(1.0, (len(recent_trades) - self.trades_seen) / 5.0)
                score += activity_factor * 0.3
            
            # Performance volatility score (25% weight)
            if len(recent_trades) >= 10:
                pnls = [t.get('pnl', 0) for t in recent_trades[-10:]]
                if len(pnls) > 1:
                    pnl_volatility = np.std(pnls) / (abs(np.mean(pnls)) + 1e-6)
                    volatility_factor = min(1.0, pnl_volatility / 5.0)
                    score += volatility_factor * 0.25
            
            # Market regime change score (20% weight)
            current_regime = clustering_data.get('market_regime', 'unknown')
            if current_regime != self._last_regime:
                score += 0.2
            
            # Cluster effectiveness variance score (15% weight)
            if self.cluster_effectiveness:
                effectiveness_scores = [v['effectiveness_score'] for v in self.cluster_effectiveness.values()]
                if len(effectiveness_scores) > 1:
                    effectiveness_variance = np.var(effectiveness_scores)
                    if effectiveness_variance > 0.1:
                        score += min(0.15, effectiveness_variance * 1.5)
            
            # Quality degradation score (10% weight)
            current_quality = self.clustering_metrics.get('silhouette_score', 0.5)
            if current_quality < 0.4:
                score += (0.4 - current_quality) * 0.1 / 0.4
            
            return float(min(1.0, score))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "trigger_score")
            return 0.0

    async def _fit_comprehensive(self, memory: Any) -> Dict[str, Any]:
        """Comprehensive fit method with detailed analysis and error handling"""
        start_time = time.time()
        
        fit_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "fit_attempted": True,
            "features_len": 0,
            "status": "",
            "method": "",
            "quality_metrics": {},
            "processing_time_ms": 0
        }
        
        try:
            # Extract and validate features
            features = memory._features if hasattr(memory, "_features") else []
            fit_results["features_len"] = len(features)
            
            if len(features) < self._min_data_for_clustering:
                self._ready = False
                fit_results["status"] = f"Insufficient data: {len(features)} < {self._min_data_for_clustering}"
                return fit_results
            
            # Prepare feature matrix
            X = np.vstack(features)
            actual_clusters = min(self.n_clusters, len(X))
            
            if self._use_simple_clustering:
                clustering_results = await self._fit_advanced_simple_clustering(X, actual_clusters)
                fit_results["method"] = "advanced_simple"
            else:
                clustering_results = await self._fit_sklearn_clustering_comprehensive(X, actual_clusters)
                fit_results["method"] = "sklearn_comprehensive"
            
            fit_results.update(clustering_results)
            
            # Update clustering quality metrics
            quality_metrics = await self._update_clustering_quality_comprehensive(X)
            fit_results["quality_metrics"] = quality_metrics
            
            # Mark as ready
            self._ready = True
            self._soft_ready = True
            
            fit_results["status"] = f"Clustering successful: {actual_clusters} clusters from {len(X)} samples"
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "clustering_fit")
            fit_results["status"] = f"Clustering failed: {error_context}"
            fit_results["error"] = str(error_context)
            self.logger.error(f"Clustering fit failed: {error_context}")
            
        finally:
            fit_results["processing_time_ms"] = (time.time() - start_time) * 1000
            self._last_fit_meta = fit_results
            
            if self.debug:
                self.logger.debug(f"Clustering fit result: {fit_results['status']}")
        
        return fit_results

    async def _fit_advanced_simple_clustering(self, X: np.ndarray, actual_clusters: int) -> Dict[str, Any]:
        """Advanced simple clustering with improved convergence and stability"""
        try:
            n_samples, n_features = X.shape
            results = {
                'actual_clusters': actual_clusters,
                'iterations': 0,
                'convergence_achieved': False
            }
            
            # Intelligent initialization using k-means++ like approach
            centers = []
            if len(X) >= actual_clusters:
                # First center: random point
                centers.append(X[np.random.randint(len(X))])
                
                # Subsequent centers: farthest from existing centers
                for _ in range(1, actual_clusters):
                    distances = []
                    for point in X:
                        min_dist = min(np.linalg.norm(point - center) for center in centers)
                        distances.append(min_dist)
                    
                    # Choose point with maximum distance (weighted by squared distance)
                    distances = np.array(distances) ** 2
                    probs = distances / distances.sum()
                    next_center_idx = np.random.choice(len(X), p=probs)
                    centers.append(X[next_center_idx])
            else:
                centers = [X[i] for i in range(len(X))]
                # Add synthetic centers if needed
                for i in range(len(X), actual_clusters):
                    synthetic_center = X[i % len(X)] + np.random.randn(n_features) * 0.1
                    centers.append(synthetic_center)
            
            self._simple_centers = np.array(centers, dtype=np.float32)
            
            # Improved iterative clustering with momentum
            prev_inertia = float('inf')
            inertia_history = []
            current_inertia = 0.0  # Initialize current_inertia
            center_movement = 0.0  # Initialize center_movement
            assignments = np.zeros(len(X), dtype=int)  # Initialize assignments
            
            for iteration in range(50):  # Increased max iterations
                # Assign points to closest centers
                distances = np.linalg.norm(X[:, np.newaxis] - self._simple_centers, axis=2)
                assignments = np.argmin(distances, axis=1)
                
                # Calculate current inertia
                current_inertia = np.sum(np.min(distances, axis=1) ** 2)
                inertia_history.append(current_inertia)
                
                # Update centers with momentum-based approach
                new_centers = []
                for i in range(actual_clusters):
                    cluster_points = X[assignments == i]
                    if len(cluster_points) > 0:
                        new_center = cluster_points.mean(axis=0)
                        # Apply momentum if we have previous centers
                        if iteration > 0:
                            momentum_factor = 0.1
                            new_center = (1 - momentum_factor) * new_center + momentum_factor * self._simple_centers[i]
                        new_centers.append(new_center)
                    else:
                        # Handle empty clusters by moving center towards data
                        new_center = self._simple_centers[i] + np.random.randn(n_features) * 0.05
                        new_centers.append(new_center)
                
                new_centers = np.array(new_centers, dtype=np.float32)
                
                # Check convergence
                center_movement = np.linalg.norm(new_centers - self._simple_centers)
                improvement = prev_inertia - current_inertia
                
                if center_movement < 1e-4 or (improvement < 1e-6 and iteration > 10):
                    results['convergence_achieved'] = True
                    break
                
                self._simple_centers = new_centers
                prev_inertia = current_inertia
                results['iterations'] = iteration + 1
            
            # Pad centers if needed for consistency
            if actual_clusters < self.n_clusters:
                additional_centers = []
                for i in range(self.n_clusters - actual_clusters):
                    base_idx = i % actual_clusters
                    noise = np.random.randn(n_features) * 0.1
                    synthetic = self._simple_centers[base_idx] + noise
                    additional_centers.append(synthetic)
                
                self._simple_centers = np.vstack([self._simple_centers, additional_centers])
            
            results.update({
                'final_inertia': current_inertia,
                'inertia_improvement': inertia_history[0] - current_inertia if inertia_history else 0,
                'center_stability': center_movement,
                'empty_clusters': sum(1 for i in range(actual_clusters) if np.sum(assignments == i) == 0)
            })
            
            return results
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "simple_clustering")
            return {'error': str(error_context), 'method': 'simple_clustering_failed'}

    async def _fit_sklearn_clustering_comprehensive(self, X: np.ndarray, actual_clusters: int) -> Dict[str, Any]:
        """Comprehensive sklearn-based clustering with advanced analysis"""
        try:
            if not SKLEARN_AVAILABLE or PCA is None or KMeans is None:
                raise ValueError("Scikit-learn not available")
                
            results = {
                'actual_clusters': actual_clusters,
                'method': 'sklearn',
                'pca_components': 0,
                'variance_explained': 0.0
            }
            
            # Advanced PCA with optimal component selection
            max_pca_dim = min(X.shape) - 1
            optimal_pca_dim = min(self.pca_dim, max_pca_dim, actual_clusters + 2)
            
            self._pca = PCA(n_components=optimal_pca_dim)
            Z = self._pca.fit_transform(X)
            
            results['pca_components'] = optimal_pca_dim
            results['variance_explained'] = float(np.sum(self._pca.explained_variance_ratio_))
            
            # Enhanced KMeans with multiple initializations
            best_kmeans = None
            best_inertia = float('inf')
            
            # Try multiple random seeds for robustness
            for seed in [42, 123, 456]:
                kmeans = KMeans(
                    n_clusters=actual_clusters, 
                    random_state=seed, 
                    n_init='auto',  # Updated for newer sklearn versions
                    max_iter=500,  # Increased max_iter
                    algorithm='lloyd'  # Explicit algorithm
                )
                kmeans.fit(Z)
                
                if kmeans.inertia_ is not None and kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_kmeans = kmeans
            
            self._kmeans = best_kmeans
            
            # Handle cluster count discrepancy with intelligent center generation
            if actual_clusters < self.n_clusters and self._kmeans is not None:
                existing_centers = self._kmeans.cluster_centers_
                synthetic_centers = []
                
                # Generate synthetic centers using cluster analysis
                if self._kmeans.labels_ is not None:
                    cluster_stats = self._analyze_cluster_characteristics(Z, self._kmeans.labels_)
                else:
                    cluster_stats = {}
                
                for i in range(self.n_clusters - actual_clusters):
                    # Create synthetic center based on cluster diversity
                    base_idx = i % actual_clusters
                    base_center = existing_centers[base_idx]
                    
                    # Add intelligent perturbation based on cluster spread
                    cluster_spread = cluster_stats.get(base_idx, {}).get('spread', 0.1)
                    perturbation_scale = max(0.1, cluster_spread * 0.5)
                    noise = np.random.randn(optimal_pca_dim) * perturbation_scale
                    
                    synthetic_center = base_center + noise
                    synthetic_centers.append(synthetic_center)
                
                if synthetic_centers:
                    self._kmeans.cluster_centers_ = np.vstack([existing_centers] + synthetic_centers)
                
                self._kmeans.n_clusters = self.n_clusters
            
            if self._kmeans is not None:
                inertia_value = self._kmeans.inertia_ if self._kmeans.inertia_ is not None else 0.0
                results.update({
                    'kmeans_inertia': float(inertia_value),
                    'n_iterations': int(getattr(self._kmeans, 'n_iter_', 0)),
                    'cluster_sizes': [int(np.sum(self._kmeans.labels_ == i)) for i in range(actual_clusters)] if self._kmeans.labels_ is not None else [],
                    'synthetic_centers_added': max(0, self.n_clusters - actual_clusters)
                })
            
            return results
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "sklearn_clustering")
            return {'error': str(error_context), 'method': 'sklearn_clustering_failed'}

    def _analyze_cluster_characteristics(self, Z: np.ndarray, labels: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Analyze characteristics of existing clusters"""
        try:
            cluster_stats = {}
            
            for cluster_id in np.unique(labels):
                cluster_points = Z[labels == cluster_id]
                if len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    
                    cluster_stats[cluster_id] = {
                        'size': len(cluster_points),
                        'spread': float(np.std(distances)),
                        'density': len(cluster_points) / (np.std(distances) + 1e-6),
                        'compactness': 1.0 / (np.mean(distances) + 1e-6)
                    }
            
            return cluster_stats
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "cluster_analysis")
            return {}

    async def _update_clustering_quality_comprehensive(self, X: np.ndarray) -> Dict[str, float]:
        """Update comprehensive clustering quality metrics"""
        try:
            quality_metrics = {}
            
            if self._use_simple_clustering:
                # Advanced quality metrics for simple clustering
                if self._simple_centers is not None:
                    distances = np.linalg.norm(X[:, np.newaxis] - self._simple_centers, axis=2)
                    assignments = np.argmin(distances, axis=1)
                    min_distances = np.min(distances, axis=1)
                    
                    quality_metrics.update({
                        'inertia': float(np.sum(min_distances ** 2)),
                        'avg_distance_to_center': float(np.mean(min_distances)),
                        'max_distance_to_center': float(np.max(min_distances)),
                        'silhouette_score': self._calculate_simple_silhouette_score(X, assignments),
                        'cluster_balance': self._calculate_cluster_balance(assignments),
                        'inter_cluster_distance': self._calculate_inter_cluster_distance(self._simple_centers)
                    })
            else:
                # Advanced quality metrics using sklearn
                if self._pca is not None and self._kmeans is not None:
                    Z = self._pca.transform(X)
                    labels = self._kmeans.predict(Z)
                    
                    inertia_val = self._kmeans.inertia_ if self._kmeans.inertia_ is not None else 0.0
                    # Ensure labels is a numpy array
                    labels_array = labels if isinstance(labels, np.ndarray) else np.array(labels)
                    quality_metrics.update({
                        'inertia': float(inertia_val),
                        'cluster_balance': self._calculate_cluster_balance(labels_array),
                        'inter_cluster_distance': self._calculate_inter_cluster_distance(self._kmeans.cluster_centers_)
                    })
                    
                    # Calculate silhouette score if possible
                    if silhouette_score is not None and len(np.unique(labels)) > 1:
                        try:
                            score = silhouette_score(Z, labels)
                            quality_metrics['silhouette_score'] = float(score)
                        except Exception:
                            quality_metrics['silhouette_score'] = 0.5
                    else:
                        quality_metrics['silhouette_score'] = 0.5
            
            # Calculate additional quality metrics
            quality_metrics.update({
                'data_coverage': min(1.0, len(X) / (self.n_clusters * 5)),
                'feature_diversity': self._calculate_feature_diversity(X),
                'clustering_stability': self._calculate_clustering_stability()
            })
            
            # Determine quality trend
            prev_quality = self.clustering_metrics.get('silhouette_score', 0.5)
            current_quality = quality_metrics.get('silhouette_score', 0.5)
            
            improvement_threshold = self.clustering_intelligence['quality_improvement_threshold']
            if current_quality > prev_quality + improvement_threshold:
                quality_metrics['quality_trend'] = 'improving'
            elif current_quality < prev_quality - improvement_threshold:
                quality_metrics['quality_trend'] = 'declining'
            else:
                quality_metrics['quality_trend'] = 'stable'
            
            # Update stored metrics
            quality_metrics['last_quality_check'] = datetime.datetime.now().isoformat()
            self.clustering_metrics.update(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "quality_metrics")
            return {'silhouette_score': 0.0, 'inertia': 0.0, 'cluster_balance': 0.5}

    def _calculate_simple_silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate a simplified silhouette score for simple clustering"""
        try:
            if len(np.unique(labels)) <= 1:
                return 0.0
            
            silhouette_scores = []
            
            for i, point in enumerate(X):
                # Calculate average distance to points in same cluster
                same_cluster_points = X[labels == labels[i]]
                if len(same_cluster_points) > 1:
                    a = np.mean([np.linalg.norm(point - other) for other in same_cluster_points if not np.array_equal(point, other)])
                else:
                    a = 0
                
                # Calculate minimum average distance to points in other clusters
                b = float('inf')
                for cluster_id in np.unique(labels):
                    if cluster_id != labels[i]:
                        other_cluster_points = X[labels == cluster_id]
                        if len(other_cluster_points) > 0:
                            avg_dist = np.mean([np.linalg.norm(point - other) for other in other_cluster_points])
                            b = min(b, avg_dist)
                
                if b == float('inf'):
                    b = a
                
                # Calculate silhouette score for this point
                if max(a, b) > 0:
                    silhouette_scores.append((b - a) / max(a, b))
                else:
                    silhouette_scores.append(0)
            
            return float(np.mean(silhouette_scores))
            
        except Exception:
            return 0.5

    def _calculate_cluster_balance(self, labels: np.ndarray) -> float:
        """Calculate how balanced the clusters are"""
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(counts) <= 1:
                return 1.0
            
            # Calculate coefficient of variation (lower is better)
            mean_size = np.mean(counts)
            std_size = np.std(counts)
            cv = std_size / (mean_size + 1e-6)
            
            # Convert to balance score (higher is better, 0-1 range)
            balance_score = 1.0 / (1.0 + cv)
            return float(balance_score)
            
        except Exception:
            return 0.5

    def _calculate_inter_cluster_distance(self, centers: np.ndarray) -> float:
        """Calculate average distance between cluster centers"""
        try:
            if len(centers) <= 1:
                return 0.0
            
            distances = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    distances.append(dist)
            
            return float(np.mean(distances))
            
        except Exception:
            return 0.0

    def _calculate_feature_diversity(self, X: np.ndarray) -> float:
        """Calculate diversity of features in the dataset"""
        try:
            if len(X) <= 1:
                return 0.0
            
            # Calculate coefficient of variation for each feature
            feature_cvs = []
            for feature_idx in range(X.shape[1]):
                feature_values = X[:, feature_idx]
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                cv = std_val / (abs(mean_val) + 1e-6)
                feature_cvs.append(cv)
            
            # Average coefficient of variation across features
            diversity = float(np.mean(feature_cvs))
            return min(1.0, diversity)
            
        except Exception:
            return 0.5

    def _calculate_clustering_stability(self) -> float:
        """Calculate clustering stability based on historical performance"""
        try:
            if len(self.clustering_history) < 2:
                return 0.5
            
            # Look at quality scores over time
            recent_scores = []
            for record in list(self.clustering_history)[-5:]:
                quality_metrics = record.get('quality_metrics', {})
                silhouette = quality_metrics.get('silhouette_score', 0.5)
                recent_scores.append(silhouette)
            
            if len(recent_scores) <= 1:
                return 0.5
            
            # Calculate stability as inverse of variance
            variance = np.var(recent_scores)
            stability = 1.0 / (1.0 + variance * 10)  # Scale variance
            
            return float(stability)
            
        except Exception:
            return 0.5

    async def _analyze_clusters_comprehensive(self, clustering_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive cluster analysis with weight generation"""
        try:
            analysis = {
                'timestamp': datetime.datetime.now().isoformat(),
                'weights': None,
                'method': 'unknown',
                'confidence': 0.0,
                'cluster_details': {},
                'effectiveness_applied': False
            }
            
            # Get playbook memory for recall
            playbook_memory = clustering_data.get('playbook_memory')
            
            if playbook_memory and hasattr(playbook_memory, '_features') and len(playbook_memory._features) > 0:
                # Use the latest features for weight calculation
                latest_features = playbook_memory._features[-1]
                weights = await self._recall_comprehensive(latest_features)
                
                analysis.update({
                    'weights': weights.tolist(),
                    'method': 'clustering_recall',
                    'confidence': float(np.max(weights)),
                    'dominant_cluster': int(np.argmax(weights)),
                    'cluster_distribution': self._analyze_weight_distribution(weights)
                })
                
                # Add cluster details
                for i, weight in enumerate(weights):
                    effectiveness = self.cluster_effectiveness[i]
                    analysis['cluster_details'][f'cluster_{i}'] = {
                        'weight': float(weight),
                        'total_trades': effectiveness['total_trades'],
                        'success_rate': effectiveness['successful_trades'] / max(1, effectiveness['total_trades']),
                        'avg_pnl': effectiveness['total_pnl'] / max(1, effectiveness['total_trades']),
                        'effectiveness_score': effectiveness['effectiveness_score']
                    }
            else:
                # Generate fallback weights
                weights = await self._generate_intelligent_fallback_weights(clustering_data)
                analysis.update({
                    'weights': weights.tolist(),
                    'method': 'intelligent_fallback',
                    'confidence': 0.5,
                    'reason': 'no_features_available'
                })
            
            return analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "cluster_analysis")
            # Ultimate fallback
            fallback_weights = self._generate_safe_fallback_weights()
            return {
                'weights': fallback_weights.tolist(),
                'method': 'error_fallback',
                'confidence': 0.0,
                'error': str(error_context)
            }

    def _analyze_weight_distribution(self, weights: np.ndarray) -> Dict[str, float]:
        """Analyze the distribution characteristics of cluster weights"""
        try:
            try:
                log_weights = np.log(weights + 1e-8)
                entropy_val = -float(np.sum(weights * log_weights))
                return {
                    'entropy': entropy_val,
                    'max_weight': float(np.max(weights)),
                    'min_weight': float(np.min(weights)),
                    'weight_variance': float(np.var(weights)),
                    'effective_clusters': int(np.sum(weights > 0.1)),
                    'concentration_ratio': float(np.sum(weights[:2]) if len(weights) >= 2 else weights[0])
                }
            except Exception:
                return {'entropy': 0.0, 'max_weight': 1.0, 'min_weight': 0.0, 'weight_variance': 0.0}
        except Exception:
            return {}

    async def _recall_comprehensive(self, features: np.ndarray) -> np.ndarray:
        """Enhanced recall with comprehensive error handling and optimization"""
        start_time = time.time()
        
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
                weights = await self._generate_intelligent_fallback_weights({'reason': 'not_ready'})
                recall_meta.update({
                    "weights": weights.tolist(),
                    "method": "intelligent_fallback",
                    "status": "Not ready - using intelligent fallback"
                })
                self._last_recall_meta = recall_meta
                return weights.astype(np.float32)
            
            # Perform clustering-based recall
            if self._use_simple_clustering:
                weights = await self._recall_advanced_simple_clustering(features, recall_meta)
            else:
                weights = await self._recall_sklearn_comprehensive(features, recall_meta)
            
            # Apply cluster effectiveness and post-processing
            weights = await self._apply_comprehensive_post_processing(weights, recall_meta)
            
            # Track cluster usage and update effectiveness
            dominant_cluster = int(np.argmax(weights))
            await self._track_cluster_usage_comprehensive(dominant_cluster, weights)
            
            recall_meta.update({
                "weights": weights.tolist(),
                "dominant_cluster": int(dominant_cluster),
                "method": "clustering_comprehensive",
                "status": "Recall successful",
                "processing_time_ms": (time.time() - start_time) * 1000
            })
            
        except Exception as e:
            # Ultimate fallback with error analysis
            error_context = self.error_pinpointer.analyze_error(e, "clustering_recall")
            weights = self._generate_safe_fallback_weights()
            recall_meta.update({
                "weights": weights.tolist(),
                "method": "error_fallback",
                "status": f"Recall error: {error_context}",
                "error": str(error_context)
            })
            
            self.logger.warning(f"Clustering recall failed: {error_context}")
        
        finally:
            self._last_recall_meta = recall_meta
            
            if self.debug:
                method = recall_meta.get("method", "unknown")
                dominant = recall_meta.get("dominant_cluster", "N/A")
                self.logger.debug(f"Recall: {method}, dominant: {dominant}")
        
        return weights.astype(np.float32)

    async def _recall_advanced_simple_clustering(self, features: np.ndarray, recall_meta: Dict) -> np.ndarray:
        """Advanced recall using simple clustering with intelligent distance calculations"""
        try:
            # Calculate sophisticated distances to cluster centers
            distances = np.linalg.norm(features.reshape(1, -1) - self._simple_centers, axis=1)
            
            # Ensure correct number of distances
            if len(distances) < self.n_clusters:
                # Pad with larger distances for missing clusters
                padding = np.full(self.n_clusters - len(distances), np.max(distances) * 1.5)
                distances = np.concatenate([distances, padding])
            
            # Convert to weights using advanced transformation
            # Use inverse distance with temperature scaling
            temperature = 0.5 if self.bootstrap_mode else 1.0
            inv_distances = 1.0 / (distances + 1e-8)
            scaled_weights = np.exp(inv_distances / temperature)
            weights = scaled_weights / scaled_weights.sum()
            
            recall_meta.update({
                "distances": distances.tolist(),
                "temperature": temperature,
                "distance_stats": {
                    "min_distance": float(np.min(distances)),
                    "max_distance": float(np.max(distances)),
                    "avg_distance": float(np.mean(distances))
                }
            })
            
            return weights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "simple_recall")
            recall_meta["simple_recall_error"] = str(error_context)
            return self._generate_safe_fallback_weights()

    async def _recall_sklearn_comprehensive(self, features: np.ndarray, recall_meta: Dict) -> np.ndarray:
        """Comprehensive recall using sklearn with advanced analysis"""
        try:
            if self._pca is None or self._kmeans is None:
                raise ValueError("Sklearn models not initialized")
                
            # Transform features through PCA
            z = self._pca.transform(features.reshape(1, -1))
            
            # Calculate distances to all cluster centers
            distances = self._kmeans.transform(z).ravel()
            
            # Ensure correct number of distances
            if len(distances) < self.n_clusters:
                padding = np.full(self.n_clusters - len(distances), np.max(distances) * 1.5)
                distances = np.concatenate([distances, padding])
            
            # Advanced weight calculation with adaptive temperature
            cluster_qualities = self._get_cluster_quality_scores()
            
            # Use quality-weighted inverse distances
            inv_distances = 1.0 / (distances + 1e-8)
            quality_weights = np.array([cluster_qualities.get(i, 0.5) for i in range(self.n_clusters)])
            
            # Combine distance and quality information
            combined_scores = inv_distances * (1.0 + quality_weights)
            weights = combined_scores / combined_scores.sum()
            
            recall_meta.update({
                "distances": distances.tolist(),
                "z_features": z.tolist(),
                "pca_components": len(z[0]),
                "cluster_qualities": cluster_qualities,
                "quality_adjustment_applied": True
            })
            
            return weights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "sklearn_recall")
            recall_meta["sklearn_recall_error"] = str(error_context)
            return self._generate_safe_fallback_weights()

    def _get_cluster_quality_scores(self) -> Dict[int, float]:
        """Get quality scores for each cluster based on effectiveness"""
        try:
            quality_scores = {}
            
            for cluster_id in range(self.n_clusters):
                if cluster_id in self.cluster_effectiveness:
                    effectiveness = self.cluster_effectiveness[cluster_id]
                    
                    total_trades = float(effectiveness['total_trades'])
                    if total_trades > 0:
                        success_rate = float(effectiveness['successful_trades']) / total_trades
                        avg_pnl = float(effectiveness['total_pnl']) / total_trades
                        
                        # Combine success rate and PnL for quality score
                        pnl_factor = np.tanh(avg_pnl / 50.0)  # Normalize PnL
                        quality = (success_rate + pnl_factor + 1.0) / 3.0  # Average and normalize
                        quality_scores[cluster_id] = max(0.1, min(1.0, quality))
                    else:
                        quality_scores[cluster_id] = 0.5  # Neutral for no data
                else:
                    quality_scores[cluster_id] = 0.5
            
            return quality_scores
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "cluster_quality")
            return {i: 0.5 for i in range(self.n_clusters)}

    async def _apply_comprehensive_post_processing(self, weights: np.ndarray, recall_meta: Dict) -> np.ndarray:
        """Apply comprehensive post-processing to cluster weights"""
        try:
            original_weights = weights.copy()
            
            # Apply exploration bonus during early learning
            if self.trades_seen < self.bootstrap_trades * 2:
                exploration_factor = 1.0 - (self.trades_seen / (self.bootstrap_trades * 2))
                exploration_weights = np.random.dirichlet(np.ones(self.n_clusters) * 5.0)
                weights = (1.0 - self.exploration_bonus * exploration_factor) * weights + \
                         (self.exploration_bonus * exploration_factor) * exploration_weights
                recall_meta["exploration_applied"] = True
                recall_meta["exploration_factor"] = exploration_factor
            
            # Apply cluster effectiveness adjustments
            effectiveness_adj = await self._get_comprehensive_effectiveness_adjustments()
            if effectiveness_adj is not None:
                weights = weights * effectiveness_adj
                weights = weights / weights.sum()
                recall_meta["effectiveness_adjustment_applied"] = True
                recall_meta["effectiveness_adjustments"] = effectiveness_adj.tolist()
            
            # Apply regime-based adjustments
            regime_adj = await self._get_regime_based_adjustments()
            if regime_adj is not None:
                weights = weights * regime_adj
                weights = weights / weights.sum()
                recall_meta["regime_adjustment_applied"] = True
            
            # Ensure minimum weights for stability
            min_weight = 0.01 / self.n_clusters
            weights = np.maximum(weights, min_weight)
            weights = weights / weights.sum()
            
            # Apply momentum to smooth weight changes
            if hasattr(self, '_last_weights') and self._last_weights is not None:
                momentum_factor = 0.1
                weights = (1 - momentum_factor) * weights + momentum_factor * self._last_weights
                weights = weights / weights.sum()
                recall_meta["momentum_applied"] = True
            
            self._last_weights = weights.copy()
            
            try:
                orig_log = np.log(original_weights + 1e-8)
                final_log = np.log(weights + 1e-8)
                recall_meta["weight_changes"] = {
                    "original_entropy": float(-float(np.sum(original_weights * orig_log))),
                    "final_entropy": float(-float(np.sum(weights * final_log))),
                    "max_change": float(np.max(np.abs(weights - original_weights)))
                }
            except Exception:
                recall_meta["weight_changes"] = {"original_entropy": 0.0, "final_entropy": 0.0, "max_change": 0.0}
            
            return weights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "post_processing")
            recall_meta["post_processing_error"] = str(error_context)
            return weights  # Return original weights if post-processing fails

    async def _get_comprehensive_effectiveness_adjustments(self) -> Optional[np.ndarray]:
        """Get comprehensive effectiveness-based adjustments for cluster weights"""
        try:
            if not self.cluster_effectiveness:
                return None
            
            adjustments = np.ones(self.n_clusters, dtype=np.float32)
            
            for cluster_id in range(self.n_clusters):
                if cluster_id in self.cluster_effectiveness:
                    effectiveness = self.cluster_effectiveness[cluster_id]
                    
                    total_trades = int(effectiveness['total_trades'])
                    if total_trades >= 3:
                        success_rate = int(effectiveness['successful_trades']) / total_trades
                        avg_pnl = float(effectiveness['total_pnl']) / total_trades
                        effectiveness_score = effectiveness['effectiveness_score']
                        
                        # Multi-factor adjustment
                        pnl_factor = 1.0 + np.tanh(avg_pnl / 25.0) * 0.2
                        success_factor = 0.8 + success_rate * 0.4
                        effectiveness_factor = 0.9 + effectiveness_score * 0.2
                        
                        # Combine factors
                        combined_adjustment = pnl_factor * success_factor * effectiveness_factor
                        adjustments[cluster_id] = np.clip(combined_adjustment, 0.5, 1.5)
                    
                    # Decay effectiveness over time to prevent over-reliance on old data
                    decay_factor = self.clustering_intelligence['effectiveness_decay']
                    effectiveness['effectiveness_score'] *= decay_factor
            
            # Normalize adjustments to prevent extreme bias
            adjustments = adjustments / np.mean(adjustments)
            return adjustments
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "effectiveness_adjustments")
            return None

    async def _get_regime_based_adjustments(self) -> Optional[np.ndarray]:
        """Get market regime-based adjustments for cluster weights"""
        try:
            current_regime = self.smart_bus.get('market_regime', 'PlaybookClusterer') or 'unknown'
            
            if current_regime == 'unknown':
                return None
            
            # Define regime preferences for different clusters
            regime_preferences = {
                'trending': [1.2, 0.8, 1.1, 0.9, 1.0],  # Favor trend-following clusters
                'ranging': [0.8, 1.2, 0.9, 1.1, 1.0],   # Favor mean-reversion clusters
                'volatile': [1.1, 0.9, 1.2, 0.8, 1.1],  # Favor adaptive clusters
                'breakout': [1.3, 0.7, 1.2, 0.8, 1.0],  # Favor momentum clusters
                'reversal': [0.7, 1.3, 0.8, 1.2, 1.0]   # Favor counter-trend clusters
            }
            
            if current_regime in regime_preferences:
                preferences = regime_preferences[current_regime]
                # Extend or truncate to match cluster count
                if len(preferences) < self.n_clusters:
                    preferences.extend([1.0] * (self.n_clusters - len(preferences)))
                elif len(preferences) > self.n_clusters:
                    preferences = preferences[:self.n_clusters]
                
                return np.array(preferences, dtype=np.float32)
            
            return None
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_adjustments")
            return None

    async def _track_cluster_usage_comprehensive(self, cluster_id: int, weights: np.ndarray):
        """Comprehensive cluster usage tracking with detailed analytics"""
        try:
            if 0 <= cluster_id < self.n_clusters:
                effectiveness = self.cluster_effectiveness[cluster_id]
                effectiveness['last_used'] = datetime.datetime.now().isoformat()
                
                # Track usage patterns
                if not hasattr(self, '_cluster_usage_history'):
                    self._cluster_usage_history = defaultdict(list)
                
                try:
                    log_w = np.log(weights + 1e-8)
                    entropy_val = float(-float(np.sum(weights * log_w)))
                except Exception:
                    entropy_val = 0.0
                
                self._cluster_usage_history[cluster_id].append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'weight': float(weights[cluster_id]),
                    'total_weight_entropy': entropy_val,
                    'market_regime': self.smart_bus.get('market_regime', 'PlaybookClusterer') or 'unknown'
                })
                
                # Trim history to prevent memory growth
                if len(self._cluster_usage_history[cluster_id]) > 100:
                    self._cluster_usage_history[cluster_id] = self._cluster_usage_history[cluster_id][-100:]
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "cluster_usage_tracking")

    async def _generate_intelligent_fallback_weights(self, context: Dict[str, Any]) -> np.ndarray:
        """Generate intelligent fallback weights with context awareness"""
        try:
            reason = context.get('reason', 'unknown')
            
            if reason == 'not_ready' or self.bootstrap_mode:
                # Enhanced exploration for bootstrap/not ready
                if self.trades_seen < 5:
                    # Very early stage - pure exploration
                    weights = np.random.dirichlet(np.ones(self.n_clusters) * 3.0)
                else:
                    # Some experience - guided exploration
                    base_weights = np.ones(self.n_clusters, dtype=np.float32) / self.n_clusters
                    exploration = np.random.dirichlet(np.ones(self.n_clusters) * 2.0) * 0.3
                    weights = 0.7 * base_weights + 0.3 * exploration
                    weights = weights / weights.sum()
                
                # Add slight bias toward first cluster to encourage initial action
                weights[0] += 0.1
                weights = weights / weights.sum()
                
            else:
                # Conservative uniform distribution with small perturbations
                weights = np.ones(self.n_clusters, dtype=np.float32) / self.n_clusters
                perturbation = np.random.randn(self.n_clusters) * 0.02
                weights += perturbation
                weights = np.maximum(weights, 0.01)
                weights = weights / weights.sum()
            
            return weights.astype(np.float32)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "intelligent_fallback")
            return self._generate_safe_fallback_weights()

    def _generate_safe_fallback_weights(self) -> np.ndarray:
        """Generate safe fallback weights for error cases"""
        try:
            weights = np.ones(self.n_clusters, dtype=np.float32) / self.n_clusters
            weights[0] += 0.05  # Slight bias to encourage action
            return weights / weights.sum()
        except Exception:
            # Ultimate fallback
            return np.array([1.0] + [0.0] * (self.n_clusters - 1), dtype=np.float32)

    async def _update_cluster_effectiveness_comprehensive(self, clustering_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update cluster effectiveness with comprehensive analysis"""
        try:
            effectiveness_update = {
                'timestamp': datetime.datetime.now().isoformat(),
                'updates_applied': 0,
                'clusters_analyzed': 0
            }
            
            recent_trades = clustering_data.get('recent_trades', [])
            
            if not recent_trades:
                return effectiveness_update
            
            # Analyze recent trades for cluster effectiveness updates
            # This would typically be called when we know which cluster was used for each trade
            # For now, we'll update based on overall performance trends
            
            if len(recent_trades) >= 5:
                recent_pnls = [t.get('pnl', 0) for t in recent_trades[-5:]]
                avg_recent_pnl = np.mean(recent_pnls)
                
                # Update effectiveness scores based on recent performance
                for cluster_id in range(self.n_clusters):
                    if cluster_id in self.cluster_effectiveness:
                        effectiveness = self.cluster_effectiveness[cluster_id]
                        
                        # Gradual adjustment based on recent performance
                        if avg_recent_pnl > 10:
                            effectiveness['effectiveness_score'] = min(1.0, effectiveness['effectiveness_score'] + 0.01)
                        elif avg_recent_pnl < -10:
                            effectiveness['effectiveness_score'] = max(0.0, effectiveness['effectiveness_score'] - 0.01)
                        
                        effectiveness_update['clusters_analyzed'] += 1
                
                effectiveness_update['avg_recent_pnl'] = float(avg_recent_pnl)
            
            return effectiveness_update
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "effectiveness_update")
            return {'error': str(error_context)}

    def update_cluster_effectiveness_from_trade(self, cluster_id: int, trade_successful: bool, 
                                              pnl: float, confidence: float, trade_duration: float = 0.0) -> None:
        """Update cluster effectiveness based on specific trade results"""
        try:
            if 0 <= cluster_id < self.n_clusters:
                # Ensure cluster exists in effectiveness tracking
                if cluster_id not in self.cluster_effectiveness:
                    self.cluster_effectiveness[cluster_id] = ClusterEffectivenessDict()
                
                effectiveness = self.cluster_effectiveness[cluster_id]
                effectiveness['total_trades'] = int(effectiveness['total_trades']) + 1
                
                if trade_successful:
                    effectiveness['successful_trades'] = int(effectiveness['successful_trades']) + 1
                
                effectiveness['total_pnl'] = float(effectiveness['total_pnl']) + pnl
                
                # Update rolling averages
                total_trades = int(effectiveness['total_trades'])
                current_avg_confidence = float(effectiveness['avg_confidence'])
                effectiveness['avg_confidence'] = (
                    (current_avg_confidence * (total_trades - 1) + confidence) / total_trades
                )
                
                if trade_duration > 0:
                    current_avg_duration = float(effectiveness['avg_trade_duration'])
                    effectiveness['avg_trade_duration'] = (
                        (current_avg_duration * (total_trades - 1) + trade_duration) / total_trades
                    )
                
                # Update effectiveness score
                success_rate = int(effectiveness['successful_trades']) / total_trades
                avg_pnl = float(effectiveness['total_pnl']) / total_trades
                
                # Combined effectiveness score
                pnl_component = np.tanh(avg_pnl / 25.0)  # Normalize PnL component
                success_component = (success_rate - 0.5) * 2  # Center around 0.5, scale to [-1, 1]
                confidence_component = (float(effectiveness['avg_confidence']) - 0.5) * 2
                
                effectiveness['effectiveness_score'] = float(np.clip(
                    0.5 + 0.2 * (pnl_component + success_component + confidence_component) / 3,
                    0.0, 1.0
                ))
                
                # Track conditions for pattern analysis
                market_regime = self.smart_bus.get('market_regime', 'PlaybookClusterer') or 'unknown'
                condition = {
                    'regime': market_regime,
                    'pnl': pnl,
                    'successful': trade_successful,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Ensure conditions lists exist
                if not isinstance(effectiveness['best_conditions'], list):
                    effectiveness['best_conditions'] = []
                if not isinstance(effectiveness['worst_conditions'], list):
                    effectiveness['worst_conditions'] = []
                
                if trade_successful and pnl > 10:
                    effectiveness['best_conditions'].append(condition)
                    # Keep only recent best conditions
                    effectiveness['best_conditions'] = effectiveness['best_conditions'][-10:]
                elif not trade_successful or pnl < -10:
                    effectiveness['worst_conditions'].append(condition)
                    # Keep only recent worst conditions
                    effectiveness['worst_conditions'] = effectiveness['worst_conditions'][-10:]
                
                # Log significant updates
                if total_trades % 10 == 0:
                    self.logger.info(format_operator_message(
                        icon="📊",
                        message=f"Cluster {cluster_id} effectiveness milestone",
                        trades=total_trades,
                        success_rate=f"{success_rate:.1%}",
                        avg_pnl=f"€{avg_pnl:.2f}",
                        effectiveness=f"{effectiveness['effectiveness_score']:.3f}"
                    ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "trade_effectiveness_update")
            self.logger.warning(f"Cluster effectiveness update failed: {error_context}")

    async def _generate_comprehensive_clustering_thesis(self, cluster_analysis: Dict[str, Any],
                                                      effectiveness_update: Dict[str, Any],
                                                      clustering_update: Dict[str, Any]) -> str:
        """Generate comprehensive thesis explaining clustering decisions"""
        try:
            thesis_parts = []
            
            # Executive Summary
            weights = cluster_analysis.get('weights', [])
            if weights:
                dominant_cluster = np.argmax(weights)
                dominant_weight = weights[dominant_cluster]
                thesis_parts.append(
                    f"CLUSTERING: Cluster {dominant_cluster} dominant with {dominant_weight:.1%} weight"
                )
            
            # Method and Readiness
            method = cluster_analysis.get('method', 'unknown')
            confidence = cluster_analysis.get('confidence', 0.0)
            thesis_parts.append(
                f"METHOD: {method.replace('_', ' ').title()} with {confidence:.1%} confidence"
            )
            
            # Clustering Quality
            quality_score = self.clustering_metrics.get('silhouette_score', 0.0)
            quality_trend = self.clustering_metrics.get('quality_trend', 'stable')
            thesis_parts.append(
                f"QUALITY: {quality_score:.3f} silhouette score, trend {quality_trend}"
            )
            
            # Clustering Updates
            if clustering_update.get('clustering_performed', False):
                reason = clustering_update.get('clustering_reason', 'unknown')
                thesis_parts.append(
                    f"UPDATE: Reclustering performed due to {reason.replace('_', ' ')}"
                )
            
            # Effectiveness Insights
            if self.cluster_effectiveness:
                best_cluster = max(self.cluster_effectiveness.items(), 
                                 key=lambda x: x[1]['effectiveness_score'])[0]
                best_score = self.cluster_effectiveness[best_cluster]['effectiveness_score']
                thesis_parts.append(
                    f"EFFECTIVENESS: Cluster {best_cluster} most effective ({best_score:.3f} score)"
                )
            
            # Learning Status
            learning_status = "Bootstrap" if self.bootstrap_mode else "Operational"
            data_coverage = self.clustering_metrics.get('data_coverage', 0.0)
            thesis_parts.append(
                f"STATUS: {learning_status} mode, {data_coverage:.1%} data coverage"
            )
            
            # System Health
            system_health = "Healthy" if not self.is_disabled else "Disabled"
            error_rate = self.error_count / max(1, self.trades_seen)
            thesis_parts.append(
                f"HEALTH: {system_health}, {error_rate:.1%} error rate"
            )
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Clustering thesis generation failed: {error_context}"

    def _generate_intelligent_cluster_recommendations(self, cluster_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on cluster analysis"""
        try:
            recommendations = []
            
            # Primary cluster recommendation
            weights = cluster_analysis.get('weights', [])
            if weights:
                dominant_cluster = np.argmax(weights)
                dominant_weight = weights[dominant_cluster]
                
                if dominant_weight > 0.6:
                    recommendations.append(
                        f"Strong Signal: Focus on Cluster {dominant_cluster} strategies ({dominant_weight:.1%} confidence)"
                    )
                elif dominant_weight < 0.3:
                    recommendations.append(
                        f"Mixed Signals: Diversified approach recommended across multiple clusters"
                    )
                else:
                    recommendations.append(
                        f"Moderate Signal: Cluster {dominant_cluster} preferred with cautious diversification"
                    )
            
            # Effectiveness-based recommendations
            if self.cluster_effectiveness:
                # Find best and worst performing clusters
                effectiveness_scores = {k: v['effectiveness_score'] for k, v in self.cluster_effectiveness.items()
                                      if v['total_trades'] >= 3}
                
                if effectiveness_scores:
                    best_cluster = max(effectiveness_scores.items(), key=lambda x: x[1])[0]
                    worst_cluster = min(effectiveness_scores.items(), key=lambda x: x[1])[0]
                    
                    best_score = effectiveness_scores[best_cluster]
                    worst_score = effectiveness_scores[worst_cluster]
                    
                    if best_score > 0.7:
                        recommendations.append(
                            f"Performance Insight: Cluster {best_cluster} showing excellent results - consider increased allocation"
                        )
                    
                    if worst_score < 0.3 and worst_cluster != best_cluster:
                        recommendations.append(
                            f"Performance Warning: Cluster {worst_cluster} underperforming - reduce exposure"
                        )
            
            # Quality-based recommendations
            quality_score = self.clustering_metrics.get('silhouette_score', 0.0)
            if quality_score < 0.3:
                recommendations.append(
                    "Quality Alert: Low clustering quality detected - consider strategy review"
                )
            elif quality_score > 0.7:
                recommendations.append(
                    "Quality Excellent: High clustering quality - strategies well-separated"
                )
            
            # Bootstrap recommendations
            if self.bootstrap_mode:
                progress = min(1.0, self.trades_seen / self.bootstrap_trades)
                recommendations.append(
                    f"Learning Progress: {progress:.1%} through bootstrap phase - {self.bootstrap_trades - self.trades_seen} trades remaining"
                )
            
            # System health recommendations
            if self.error_count > 2:
                recommendations.append(
                    f"System Health: {self.error_count} errors detected - monitor clustering stability"
                )
            
            # Default recommendation
            if not recommendations:
                recommendations.append(
                    "Continue current clustering approach - system operating optimally"
                )
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendations")
            return [f"Recommendation generation failed: {error_context}"]

    def _get_clustering_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive clustering health metrics"""
        try:
            return {
                'module_name': 'PlaybookClusterer',
                'status': 'disabled' if self.is_disabled else 'healthy',
                'ready': self._ready,
                'soft_ready': self._soft_ready,
                'bootstrap_mode': self.bootstrap_mode,
                'error_count': self.error_count,
                'trades_seen': self.trades_seen,
                'clustering_quality': self.clustering_metrics.get('silhouette_score', 0.0),
                'data_coverage': self.clustering_metrics.get('data_coverage', 0.0),
                'cluster_balance': self.clustering_metrics.get('cluster_balance', 0.0),
                'effectiveness_variance': self._calculate_effectiveness_variance(),
                'last_clustering': self.clustering_history[-1]['timestamp'] if self.clustering_history else None,
                'sklearn_available': SKLEARN_AVAILABLE
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "health_metrics")
            return {'error': str(error_context)}

    def _calculate_effectiveness_variance(self) -> float:
        """Calculate variance in cluster effectiveness scores"""
        try:
            if not self.cluster_effectiveness:
                return 0.0
            
            scores = [float(v['effectiveness_score']) for v in self.cluster_effectiveness.values() 
                     if isinstance(v, dict) and 'effectiveness_score' in v]
            return float(np.var(scores)) if len(scores) > 1 else 0.0
        except Exception:
            return 0.0

    def _get_cluster_effectiveness_summary(self) -> Dict[str, Any]:
        """Get comprehensive cluster effectiveness summary"""
        try:
            summary = {
                'total_clusters': self.n_clusters,
                'clusters_with_data': len([k for k, v in self.cluster_effectiveness.items() 
                                         if isinstance(v, dict) and int(v.get('total_trades', 0)) > 0]),
                'best_cluster': None,
                'worst_cluster': None,
                'avg_effectiveness': 0.0,
                'cluster_details': {}
            }
            
            effectiveness_scores = {}
            for cluster_id, effectiveness in self.cluster_effectiveness.items():
                if isinstance(effectiveness, dict) and int(effectiveness.get('total_trades', 0)) > 0:
                    total_trades = int(effectiveness['total_trades'])
                    successful_trades = int(effectiveness['successful_trades'])
                    total_pnl = float(effectiveness['total_pnl'])
                    effectiveness_score = float(effectiveness['effectiveness_score'])
                    
                    effectiveness_scores[cluster_id] = effectiveness_score
                    
                    summary['cluster_details'][f'cluster_{cluster_id}'] = {
                        'total_trades': total_trades,
                        'success_rate': successful_trades / total_trades,
                        'total_pnl': total_pnl,
                        'avg_pnl': total_pnl / total_trades,
                        'effectiveness_score': effectiveness_score,
                        'last_used': effectiveness.get('last_used')
                    }
            
            if effectiveness_scores:
                best_item = max(effectiveness_scores.items(), key=lambda x: x[1])
                worst_item = min(effectiveness_scores.items(), key=lambda x: x[1])
                summary['best_cluster'] = best_item[0]
                summary['worst_cluster'] = worst_item[0]
                summary['avg_effectiveness'] = np.mean(list(effectiveness_scores.values()))
            
            return summary
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "effectiveness_summary")
            return {'error': str(error_context)}

    def _get_pattern_analysis_summary(self) -> Dict[str, Any]:
        """Get pattern analysis summary"""
        try:
            return {
                'pattern_cache_size': len(self.pattern_cache),
                'clustering_history_length': len(self.clustering_history),
                'last_regime': self._last_regime,
                'performance_window': self._last_performance_window,
                'quality_trend': self.clustering_metrics.get('quality_trend', 'stable'),
                'bootstrap_progress': min(1.0, self.trades_seen / self.bootstrap_trades) if self.bootstrap_mode else 1.0
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "pattern_summary")
            return {'error': str(error_context)}

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive clustering results"""
        try:
            # Core cluster weights
            self.smart_bus.set('cluster_weights', results['cluster_weights'],
                             module='PlaybookClusterer', thesis=thesis)
            
            # Cluster analysis
            analysis_thesis = f"Cluster analysis: {len(results['cluster_analysis'].get('cluster_details', {}))} clusters analyzed"
            self.smart_bus.set('cluster_analysis', results['cluster_analysis'],
                             module='PlaybookClusterer', thesis=analysis_thesis)
            
            # Clustering health
            health_thesis = f"Clustering health: {'Ready' if results['clustering_health'].get('ready', False) else 'Not ready'}"
            self.smart_bus.set('clustering_health', results['clustering_health'],
                             module='PlaybookClusterer', thesis=health_thesis)
            
            # Cluster recommendations
            rec_thesis = f"Generated {len(results['cluster_recommendations'])} clustering recommendations"
            self.smart_bus.set('cluster_recommendations', results['cluster_recommendations'],
                             module='PlaybookClusterer', thesis=rec_thesis)
            
            # Cluster effectiveness
            eff_thesis = f"Cluster effectiveness: {results['cluster_effectiveness'].get('clusters_with_data', 0)} active clusters"
            self.smart_bus.set('cluster_effectiveness', results['cluster_effectiveness'],
                             module='PlaybookClusterer', thesis=eff_thesis)
            
            # Pattern analysis
            pattern_thesis = f"Pattern analysis: {results['pattern_analysis'].get('clustering_history_length', 0)} historical events"
            self.smart_bus.set('pattern_analysis', results['pattern_analysis'],
                             module='PlaybookClusterer', thesis=pattern_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # ERROR HANDLING AND RECOVERY
    # ═══════════════════════════════════════════════════════════════════

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "PlaybookClusterer")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Playbook Clusterer disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('PlaybookClusterer', 'process_time', processing_time, False)
        
        return {
            'cluster_weights': self._generate_safe_fallback_weights().tolist(),
            'cluster_analysis': {'error': str(error_context)},
            'clustering_health': {'status': 'error', 'error_context': str(error_context)},
            'cluster_recommendations': ["Investigate clustering system errors"],
            'cluster_effectiveness': {'error': str(error_context)},
            'pattern_analysis': {'error': str(error_context)},
            'clustering_thesis': f"Clustering error: {error_context}"
        }

    def _get_safe_clustering_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when clustering data retrieval fails"""
        return {
            'playbook_memory': None,
            'recent_trades': [],
            'market_data': {},
            'trading_performance': {},
            'market_regime': 'unknown',
            'session_metrics': {},
            'volatility_data': {},
            'market_context': {}
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'cluster_weights': self._generate_safe_fallback_weights().tolist(),
            'cluster_analysis': {'status': 'disabled'},
            'clustering_health': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'},
            'cluster_recommendations': ["Restart playbook clustering system"],
            'cluster_effectiveness': {'status': 'disabled'},
            'pattern_analysis': {'status': 'disabled'},
            'clustering_thesis': 'Clustering system disabled due to errors'
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT FOR HOT-RELOAD
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        try:
            state = {
                'module_info': {
                    'name': 'PlaybookClusterer',
                    'version': '3.0.0',
                    'last_updated': datetime.datetime.now().isoformat()
                },
                'configuration': {
                    'n_clusters': self.n_clusters,
                    'pca_dim': self.pca_dim,
                    'bootstrap_trades': self.bootstrap_trades,
                    'exploration_bonus': self.exploration_bonus,
                    'adaptive_clustering': self.adaptive_clustering,
                    'debug': self.debug,
                    'use_simple_clustering': self._use_simple_clustering
                },
                'clustering_state': {
                    '_ready': self._ready,
                    '_soft_ready': self._soft_ready,
                    'bootstrap_mode': self.bootstrap_mode,
                    'trades_seen': self.trades_seen,
                    '_min_data_for_clustering': self._min_data_for_clustering
                },
                'metrics': {
                    'clustering_metrics': self.clustering_metrics.copy(),
                    'cluster_effectiveness': {k: v.copy() for k, v in self.cluster_effectiveness.items()},
                    'performance_tracking': {k: list(v) for k, v in self.performance_tracking.items()},
                    'clustering_intelligence': self.clustering_intelligence.copy()
                },
                'history': {
                    'clustering_history': list(self.clustering_history),
                    '_last_fit_meta': self._last_fit_meta.copy(),
                    '_last_recall_meta': self._last_recall_meta.copy(),
                    'pattern_cache': self.pattern_cache.copy(),
                    'pattern_performance': dict(self.pattern_performance)
                },
                'error_state': {
                    'error_count': self.error_count,
                    'is_disabled': self.is_disabled
                },
                'runtime_state': {
                    '_last_performance_window': self._last_performance_window,
                    '_last_regime': self._last_regime,
                    '_last_weights': getattr(self, '_last_weights', None)
                }
            }
            
            # Add model states if available
            if self._ready:
                if self._use_simple_clustering and hasattr(self, '_simple_centers') and self._simple_centers is not None:
                    state["model_state"] = {
                        "simple_centers": self._simple_centers.tolist(),
                        "pseudo_centers": self._pseudo_centers.tolist()
                    }
                elif not self._use_simple_clustering and self._pca and self._kmeans:
                    state["model_state"] = {
                        "pca_mean_": self._pca.mean_.tolist(),
                        "pca_components_": self._pca.components_.tolist(),
                        "pca_explained_variance_ratio_": self._pca.explained_variance_ratio_.tolist(),
                        "kmeans_centers_": self._kmeans.cluster_centers_.tolist(),
                        "kmeans_n_clusters": self._kmeans.n_clusters,
                        "kmeans_inertia_": float(self._kmeans.inertia_ if self._kmeans.inertia_ is not None else 0.0)
                    }
            
            return state
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_serialization")
            self.logger.error(f"State serialization failed: {error_context}")
            return {'error': str(error_context)}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization with comprehensive error handling"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.n_clusters = int(config.get("n_clusters", self.n_clusters))
            self.pca_dim = int(config.get("pca_dim", self.pca_dim))
            self.bootstrap_trades = int(config.get("bootstrap_trades", self.bootstrap_trades))
            self.exploration_bonus = float(config.get("exploration_bonus", self.exploration_bonus))
            self.adaptive_clustering = bool(config.get("adaptive_clustering", self.adaptive_clustering))
            self.debug = bool(config.get("debug", self.debug))
            self._use_simple_clustering = bool(config.get("use_simple_clustering", not SKLEARN_AVAILABLE))
            
            # Load clustering state
            clustering_state = state.get("clustering_state", {})
            self._ready = bool(clustering_state.get("_ready", False))
            self._soft_ready = bool(clustering_state.get("_soft_ready", False))
            self.bootstrap_mode = bool(clustering_state.get("bootstrap_mode", True))
            self.trades_seen = int(clustering_state.get("trades_seen", 0))
            self._min_data_for_clustering = int(clustering_state.get("_min_data_for_clustering", max(self.n_clusters, 3)))
            
            # Load metrics
            metrics = state.get("metrics", {})
            self.clustering_metrics = metrics.get("clustering_metrics", self.clustering_metrics)
            self.clustering_intelligence = metrics.get("clustering_intelligence", self.clustering_intelligence)
            
            # Load cluster effectiveness
            effectiveness_data = metrics.get("cluster_effectiveness", {})
            self.cluster_effectiveness = {}
            for i in range(self.n_clusters):
                self.cluster_effectiveness[i] = ClusterEffectivenessDict()
            
            for k, v in effectiveness_data.items():
                cluster_id = int(k)
                if 0 <= cluster_id < self.n_clusters and isinstance(v, dict):
                    eff_dict = ClusterEffectivenessDict()
                    eff_dict.update({
                        'total_trades': int(v.get('total_trades', 0)),
                        'successful_trades': int(v.get('successful_trades', 0)),
                        'total_pnl': float(v.get('total_pnl', 0.0)),
                        'avg_confidence': float(v.get('avg_confidence', 0.0)),
                        'avg_trade_duration': float(v.get('avg_trade_duration', 0.0)),
                        'best_conditions': list(v.get('best_conditions', [])),
                        'worst_conditions': list(v.get('worst_conditions', [])),
                        'last_used': v.get('last_used'),
                        'effectiveness_score': float(v.get('effectiveness_score', 0.5))
                    })
                    self.cluster_effectiveness[cluster_id] = eff_dict
            
            # Load performance tracking
            performance_data = metrics.get("performance_tracking", {})
            self.performance_tracking = defaultdict(list)
            for k, v in performance_data.items():
                self.performance_tracking[k] = list(v)
            
            # Load history
            history = state.get("history", {})
            self.clustering_history = deque(history.get("clustering_history", []), maxlen=100)
            self._last_fit_meta = history.get("_last_fit_meta", {})
            self._last_recall_meta = history.get("_last_recall_meta", {})
            self.pattern_cache = history.get("pattern_cache", {})
            self.pattern_performance = defaultdict(float, history.get("pattern_performance", {}))
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            # Load runtime state
            runtime_state = state.get("runtime_state", {})
            self._last_performance_window = float(runtime_state.get("_last_performance_window", 0.0))
            self._last_regime = runtime_state.get("_last_regime", 'unknown')
            last_weights = runtime_state.get("_last_weights")
            if last_weights:
                self._last_weights = np.array(last_weights, dtype=np.float32)
            
            # Load model state if available
            model_state = state.get("model_state", {})
            if model_state and self._ready:
                self._restore_clustering_models(model_state)
            else:
                # Reset models if no state or not ready
                self._reset_clustering_models()
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Playbook Clusterer state restored",
                clusters=self.n_clusters,
                ready=self._ready,
                trades_seen=self.trades_seen,
                history_length=len(self.clustering_history)
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")
            # Reset to safe defaults on state restoration failure
            self._reset_clustering_models()

    def _restore_clustering_models(self, model_state: Dict[str, Any]) -> None:
        """Restore clustering models from state with comprehensive error handling"""
        try:
            if self._use_simple_clustering:
                # Restore simple clustering models
                if "simple_centers" in model_state:
                    self._simple_centers = np.array(model_state["simple_centers"], dtype=np.float32)
                if "pseudo_centers" in model_state:
                    self._pseudo_centers = np.array(model_state["pseudo_centers"], dtype=np.float32)
                
                self.logger.info("🔄 Simple clustering models restored from state")
                
            elif SKLEARN_AVAILABLE and not self._use_simple_clustering and PCA is not None and KMeans is not None:
                # Restore sklearn models
                required_keys = ["pca_mean_", "pca_components_", "kmeans_centers_"]
                if all(k in model_state for k in required_keys):
                    
                    # Restore PCA
                    self._pca = PCA(n_components=self.pca_dim)
                    self._pca.mean_ = np.array(model_state["pca_mean_"], dtype=np.float32)
                    self._pca.components_ = np.array(model_state["pca_components_"], dtype=np.float32)
                    self._pca.n_components_ = self._pca.components_.shape[0]
                    
                    # Restore explained variance if available
                    if "pca_explained_variance_ratio_" in model_state:
                        self._pca.explained_variance_ratio_ = np.array(
                            model_state["pca_explained_variance_ratio_"], dtype=np.float32)
                    else:
                        # Set dummy values
                        n_components = self._pca.n_components_
                        self._pca.explained_variance_ratio_ = np.ones(n_components, dtype=np.float32) / n_components
                    
                    # Set other required PCA attributes
                    self._pca.explained_variance_ = self._pca.explained_variance_ratio_ * 100
                    self._pca.singular_values_ = np.sqrt(self._pca.explained_variance_)
                    
                    # Restore KMeans
                    self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                    self._kmeans.cluster_centers_ = np.array(model_state["kmeans_centers_"], dtype=np.float32)
                    self._kmeans.n_clusters = model_state.get("kmeans_n_clusters", self.n_clusters)
                    
                    if "kmeans_inertia_" in model_state:
                        self._kmeans.inertia_ = float(model_state["kmeans_inertia_"])
                    
                    self.logger.info("🔄 Sklearn clustering models restored from state")
                else:
                    self.logger.warning("Incomplete sklearn model state - resetting models")
                    self._reset_clustering_models()
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "model_restoration")
            self.logger.error(f"Model restoration failed: {error_context}")
            self._reset_clustering_models()

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API METHODS (for external use)
    # ═══════════════════════════════════════════════════════════════════

    def fit(self, memory: Any) -> None:  # Using Any for PlaybookMemory to avoid type issues
        """Public method to fit clustering model (async wrapper)"""
        try:
            # Run the async method
            import asyncio
            if asyncio.get_event_loop().is_running():
                # If we're already in an async context, schedule it
                asyncio.create_task(self._fit_comprehensive(memory))
            else:
                # Run it directly
                asyncio.run(self._fit_comprehensive(memory))
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "fit_wrapper")
            self.logger.error(f"Clustering fit wrapper failed: {error_context}")

    def recall(self, features: np.ndarray) -> np.ndarray:
        """Public method to recall cluster weights (async wrapper)"""
        try:
            # Run the async method
            import asyncio
            if asyncio.get_event_loop().is_running():
                # Create a task but return synchronously for compatibility
                loop = asyncio.get_event_loop()
                task = loop.create_task(self._recall_comprehensive(features))
                # For sync compatibility, we need to handle this differently
                return self._recall_sync_fallback(features)
            else:
                # Run it directly
                return asyncio.run(self._recall_comprehensive(features))
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recall_wrapper")
            self.logger.error(f"Clustering recall wrapper failed: {error_context}")
            return self._generate_safe_fallback_weights()

    def _recall_sync_fallback(self, features: np.ndarray) -> np.ndarray:
        """Synchronous fallback for recall when in async context"""
        try:
            if not self._ready:
                return self._generate_safe_fallback_weights()
            
            if self._use_simple_clustering:
                if hasattr(self, '_simple_centers') and self._simple_centers is not None:
                    distances = np.linalg.norm(features.reshape(1, -1) - self._simple_centers, axis=1)
                    if len(distances) < self.n_clusters:
                        distances = np.pad(distances, (0, self.n_clusters - len(distances)), 
                                         constant_values=np.max(distances) * 2)
                    inv_distances = 1.0 / (distances + 1e-8)
                    weights = inv_distances / inv_distances.sum()
                    return weights.astype(np.float32)
            else:
                if self._pca is not None and self._kmeans is not None:
                    z = self._pca.transform(features.reshape(1, -1))
                    distances = self._kmeans.transform(z).ravel()
                    if len(distances) < self.n_clusters:
                        distances = np.pad(distances, (0, self.n_clusters - len(distances)), 
                                         constant_values=np.max(distances) * 2)
                    inv_distances = 1.0 / (distances + 1e-8)
                    weights = inv_distances / inv_distances.sum()
                    return weights.astype(np.float32)
            
            return self._generate_safe_fallback_weights()
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "sync_fallback")
            return self._generate_safe_fallback_weights()

    def get_observation_components(self) -> np.ndarray:
        """Get clustering observation components for environment"""
        try:
            # Basic readiness indicators
            ready_float = float(self._ready)
            soft_ready_float = float(self._soft_ready)
            bootstrap_float = float(self.bootstrap_mode)
            
            # Progress and data indicators
            data_ratio = min(1.0, self._last_fit_meta.get('features_len', 0) / max(self.n_clusters, 10))
            trade_ratio = min(1.0, self.trades_seen / max(self.bootstrap_trades, 10))
            
            # Clustering quality metrics
            clustering_quality = self.clustering_metrics.get('silhouette_score', 0.5)
            data_coverage = self.clustering_metrics.get('data_coverage', 0.0)
            cluster_balance = self.clustering_metrics.get('cluster_balance', 0.5)
            
            # Effectiveness metrics
            if self.cluster_effectiveness:
                effectiveness_scores = [v['effectiveness_score'] for v in self.cluster_effectiveness.values()]
                avg_effectiveness = np.mean(effectiveness_scores)
                effectiveness_variance = np.var(effectiveness_scores)
                
                # Success rate across clusters
                success_rates = []
                for v in self.cluster_effectiveness.values():
                    if v['total_trades'] > 0:
                        success_rates.append(v['successful_trades'] / v['total_trades'])
                avg_success_rate = np.mean(success_rates) if success_rates else 0.5
            else:
                avg_effectiveness = 0.5
                effectiveness_variance = 0.0
                avg_success_rate = 0.5
            
            # Error and stability metrics
            error_rate = min(1.0, self.error_count / max(1, self.trades_seen))
            system_health = 0.0 if self.is_disabled else 1.0
            
            # Historical trend indicators
            quality_trend_score = {
                'improving': 0.8,
                'stable': 0.5,
                'declining': 0.2
            }.get(self.clustering_metrics.get('quality_trend', 'stable'), 0.5)
            
            observation = np.array([
                self.n_clusters / 10.0,         # Normalized cluster count
                self.pca_dim / 10.0,            # Normalized PCA dimensions  
                ready_float,                    # Clustering ready
                soft_ready_float,               # Soft ready
                bootstrap_float,                # Bootstrap mode
                data_ratio,                     # Data availability ratio
                trade_ratio,                    # Trading progress ratio
                clustering_quality,             # Clustering quality score
                data_coverage,                  # Data coverage score
                cluster_balance,                # Cluster balance score
                avg_effectiveness,              # Average cluster effectiveness
                effectiveness_variance,         # Effectiveness variance
                avg_success_rate,               # Average success rate
                error_rate,                     # Error rate
                system_health,                  # System health
                quality_trend_score,            # Quality trend indicator
                self.exploration_bonus,         # Exploration bonus
                float(len(self.clustering_history)) / 50.0  # Normalized history length
            ], dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid clustering observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Clustering observation generation failed: {error_context}")
            return np.full(18, 0.5, dtype=np.float32)

    def get_cluster_recommendations(self) -> Dict[str, Any]:
        """Get current cluster recommendations and analysis"""
        try:
            # Find most effective cluster
            best_cluster = None
            best_effectiveness = 0.0
            
            for cluster_id, effectiveness in self.cluster_effectiveness.items():
                if isinstance(effectiveness, dict):
                    total_trades = int(effectiveness.get('total_trades', 0))
                    if total_trades >= 3:
                        effectiveness_score = float(effectiveness.get('effectiveness_score', 0.0))
                        if effectiveness_score > best_effectiveness:
                            best_effectiveness = effectiveness_score
                            best_cluster = cluster_id
            
            # Get latest weights if available
            latest_weights = getattr(self, '_last_weights', None)
            if latest_weights is not None:
                primary_cluster = int(np.argmax(latest_weights))
                primary_weight = float(latest_weights[primary_cluster])
            else:
                primary_cluster = 0
                primary_weight = 1.0 / self.n_clusters
            
            # Calculate total trades and average effectiveness
            total_trades_across_clusters = sum(
                int(v.get('total_trades', 0)) for v in self.cluster_effectiveness.values()
                if isinstance(v, dict)
            )
            
            clusters_with_data = len([
                k for k, v in self.cluster_effectiveness.items()
                if isinstance(v, dict) and int(v.get('total_trades', 0)) > 0
            ])
            
            avg_effectiveness = 0.0
            if self.cluster_effectiveness:
                effectiveness_scores = [
                    float(v.get('effectiveness_score', 0.5)) for v in self.cluster_effectiveness.values()
                    if isinstance(v, dict)
                ]
                if effectiveness_scores:
                    avg_effectiveness = np.mean(effectiveness_scores)
            
            recommendations = {
                'primary_cluster': primary_cluster,
                'primary_weight': primary_weight,
                'best_effective_cluster': best_cluster,
                'best_effectiveness_score': best_effectiveness,
                'clustering_ready': self._ready,
                'bootstrap_mode': self.bootstrap_mode,
                'clustering_quality': self.clustering_metrics.get('silhouette_score', 0.0),
                'cluster_analytics': {
                    'total_clusters': self.n_clusters,
                    'clusters_with_data': clusters_with_data,
                    'total_trades_across_clusters': total_trades_across_clusters,
                    'avg_cluster_effectiveness': avg_effectiveness,
                    'system_health': 'healthy' if not self.is_disabled else 'disabled'
                }
            }
            
            return recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendations_generation")
            self.logger.warning(f"Cluster recommendations generation failed: {error_context}")
            return {
                'primary_cluster': 0, 
                'primary_weight': 1.0 / self.n_clusters,
                'error': str(error_context)
            }

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def mutate(self, std: float = 1.0) -> None:
        """Enhanced mutation with adaptive parameters and comprehensive tracking"""
        try:
            pre_config = (self.n_clusters, self.pca_dim, self.exploration_bonus)
            
            # Mutate core parameters with intelligence
            if random.random() < 0.4:
                # Intelligent cluster count mutation based on current performance
                if self.clustering_metrics.get('silhouette_score', 0.5) < 0.4:
                    # Poor quality - try different cluster count
                    change = np.random.choice([-1, 1])
                else:
                    # Good quality - smaller changes
                    change = np.random.choice([-1, 0, 1])
                
                self.n_clusters = max(2, min(10, self.n_clusters + change))
            
            if random.random() < 0.4:
                # PCA dimension mutation
                change = np.random.choice([-1, 0, 1])
                self.pca_dim = max(1, min(8, self.pca_dim + change))
            
            if random.random() < 0.3:
                # Exploration bonus mutation
                noise = np.random.normal(0, 0.03 * std)
                self.exploration_bonus = max(0.0, min(0.5, self.exploration_bonus + noise))
            
            if random.random() < 0.2:
                # Bootstrap trades mutation
                change = np.random.choice([-2, -1, 0, 1, 2])
                self.bootstrap_trades = max(5, min(30, self.bootstrap_trades + change))
            
            # Update dependent parameters
            self._min_data_for_clustering = max(self.n_clusters, 3)
            
            # Reset models to apply changes
            self._reset_clustering_models()
            
            post_config = (self.n_clusters, self.pca_dim, self.exploration_bonus)
            
            # Record comprehensive mutation
            mutation_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "mutation": True,
                "before": {
                    "n_clusters": pre_config[0],
                    "pca_dim": pre_config[1], 
                    "exploration_bonus": pre_config[2]
                },
                "after": {
                    "n_clusters": post_config[0],
                    "pca_dim": post_config[1],
                    "exploration_bonus": post_config[2]
                },
                "quality_before": self.clustering_metrics.get('silhouette_score', 0.0),
                "std_factor": std
            }
            
            # Add to clustering history
            self.clustering_history.append(mutation_record)
            
            self.logger.info(format_operator_message(
                icon="🧬",
                message="Clustering mutation applied",
                clusters=f"{pre_config[0]} → {post_config[0]}",
                pca_dim=f"{pre_config[1]} → {post_config[1]}",
                exploration=f"{pre_config[2]:.3f} → {post_config[2]:.3f}"
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "mutation")
            self.logger.error(f"Clustering mutation failed: {error_context}")

    def crossover(self, other: Any) -> "PlaybookClusterer":
        """Enhanced crossover with intelligent parameter inheritance"""
        try:
            # Select parameters from parents with intelligence
            n_clusters = self.n_clusters if random.random() < 0.5 else other.n_clusters
            pca_dim = self.pca_dim if random.random() < 0.5 else other.pca_dim
            debug = self.debug or other.debug
            
            # Weighted average for numerical parameters
            exploration_bonus = (self.exploration_bonus + other.exploration_bonus) / 2
            bootstrap_trades = (self.bootstrap_trades + other.bootstrap_trades) // 2
            
            # Select boolean parameters based on performance
            self_quality = self.clustering_metrics.get('silhouette_score', 0.5)
            other_quality = other.clustering_metrics.get('silhouette_score', 0.5)
            
            if self_quality > other_quality:
                adaptive_clustering = self.adaptive_clustering
            else:
                adaptive_clustering = other.adaptive_clustering
            
                            # Create offspring with inherited configuration
            try:
                # Create new instance using the module's configuration pattern
                offspring_config = {
                    'n_clusters': n_clusters,
                    'pca_dim': pca_dim,
                    'debug': debug,
                    'bootstrap_trades': bootstrap_trades,
                    'exploration_bonus': exploration_bonus,
                    'adaptive_clustering': adaptive_clustering
                }
                
                offspring = PlaybookClusterer()
                offspring._initialize()
                
                # Set inherited parameters using getattr/setattr for safety
                setattr(offspring, 'n_clusters', n_clusters)
                setattr(offspring, 'pca_dim', pca_dim)
                setattr(offspring, 'debug', debug)
                setattr(offspring, 'bootstrap_trades', bootstrap_trades)
                setattr(offspring, 'exploration_bonus', exploration_bonus)
                setattr(offspring, 'adaptive_clustering', adaptive_clustering)
                setattr(offspring, '_min_data_for_clustering', max(n_clusters, 3))
                
                # Reset models with new parameters - use hasattr for safety
                if hasattr(offspring, '_reset_clustering_models') and callable(getattr(offspring, '_reset_clustering_models')):
                    getattr(offspring, '_reset_clustering_models')()
                
            except Exception as e:
                # Fallback: return a copy of self if offspring creation fails
                self.logger.warning(f"Offspring creation failed, using deepcopy fallback: {e}")
                offspring = copy.deepcopy(self)
            
            # Record crossover event
            crossover_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "crossover": True,
                "from_self": {
                    "n_clusters": self.n_clusters,
                    "pca_dim": self.pca_dim,
                    "exploration_bonus": self.exploration_bonus,
                    "quality": self_quality
                },
                "from_other": {
                    "n_clusters": other.n_clusters,
                    "pca_dim": other.pca_dim,
                    "exploration_bonus": other.exploration_bonus,
                    "quality": other_quality
                },
                "result": {
                    "n_clusters": n_clusters,
                    "pca_dim": pca_dim,
                    "exploration_bonus": exploration_bonus
                }
            }
            
            # Safely add crossover record if offspring has clustering_history
            if hasattr(offspring, 'clustering_history'):
                clustering_history = getattr(offspring, 'clustering_history')
                if hasattr(clustering_history, 'append'):
                    clustering_history.append(crossover_record)
            
            return offspring
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "crossover")
            self.logger.error(f"Clustering crossover failed: {error_context}")
            # Return a copy of self as fallback
            return copy.deepcopy(self)

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY AND REPORTING METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_clustering_report(self) -> str:
        """Generate comprehensive clustering analysis report"""
        try:
            # Cluster effectiveness summary
            effectiveness_summary = ""
            if self.cluster_effectiveness:
                for cluster_id, effectiveness in self.cluster_effectiveness.items():
                    if isinstance(effectiveness, dict) and int(effectiveness.get('total_trades', 0)) > 0:
                        total_trades = int(effectiveness['total_trades'])
                        successful_trades = int(effectiveness['successful_trades'])
                        total_pnl = float(effectiveness['total_pnl'])
                        effectiveness_score = float(effectiveness['effectiveness_score'])
                        
                        success_rate = successful_trades / total_trades
                        avg_pnl = total_pnl / total_trades
                        
                        status = "🟢" if effectiveness_score > 0.7 else "🔴" if effectiveness_score < 0.3 else "🟡"
                        effectiveness_summary += f"  • Cluster {cluster_id}: {total_trades} trades, {success_rate:.1%} success, €{avg_pnl:+.1f} avg, {effectiveness_score:.3f} score {status}\n"
            
            # Recent clustering activity
            recent_activity = ""
            if self.clustering_history:
                for event in list(self.clustering_history)[-3:]:
                    timestamp = event['timestamp'][:19].replace('T', ' ')
                    event_type = 'Clustering' if event.get('clustering_performed') else 'Mutation' if event.get('mutation') else 'Crossover' if event.get('crossover') else 'Unknown'
                    details = ""
                    if 'features_count' in event:
                        details = f", {event['features_count']} features"
                    elif 'trigger' in event:
                        details = f", {event['trigger']}"
                    recent_activity += f"  • {timestamp}: {event_type}{details}\n"
            
            # Quality trend analysis
            quality_analysis = ""
            if len(self.clustering_history) >= 3:
                recent_qualities = []
                for record in list(self.clustering_history)[-3:]:
                    quality_metrics = record.get('quality_metrics', {})
                    if 'silhouette_score' in quality_metrics:
                        recent_qualities.append(quality_metrics['silhouette_score'])
                
                if len(recent_qualities) >= 2:
                    trend = "improving" if recent_qualities[-1] > recent_qualities[0] else "declining" if recent_qualities[-1] < recent_qualities[0] else "stable"
                    quality_analysis = f"Recent trend: {trend} (Δ{recent_qualities[-1] - recent_qualities[0]:+.3f})"
            
            return f"""
🧠 PLAYBOOK CLUSTERER COMPREHENSIVE REPORT
═══════════════════════════════════════════════════════════════
🎯 Clustering Status:
• Ready: {'✅ Yes' if self._ready else '❌ No'}
• Soft Ready: {'✅ Yes' if self._soft_ready else '❌ No'}
• Bootstrap Mode: {'🎓 Active' if self.bootstrap_mode else '✅ Complete'}
• Trades Seen: {self.trades_seen} / {self.bootstrap_trades} (bootstrap threshold)
• System Health: {'✅ Healthy' if not self.is_disabled else '🚨 Disabled'}

⚙️ Configuration:
• Clusters: {self.n_clusters}
• PCA Dimensions: {self.pca_dim}
• ML Backend: {'✅ Scikit-learn' if SKLEARN_AVAILABLE and not self._use_simple_clustering else '⚡ Advanced Simple Clustering'}
• Adaptive Clustering: {'✅ Enabled' if self.adaptive_clustering else '❌ Disabled'}
• Exploration Bonus: {self.exploration_bonus:.1%}
• Debug Mode: {'✅ On' if self.debug else '❌ Off'}

📊 Clustering Quality:
• Silhouette Score: {self.clustering_metrics.get('silhouette_score', 0):.3f}
• Data Coverage: {self.clustering_metrics.get('data_coverage', 0):.1%}
• Cluster Balance: {self.clustering_metrics.get('cluster_balance', 0):.3f}
• Quality Trend: {self.clustering_metrics.get('quality_trend', 'unknown').title()}
• {quality_analysis}

🎯 Cluster Effectiveness:
{effectiveness_summary if effectiveness_summary else '  📭 No cluster effectiveness data yet'}

🔄 Recent Activity:
{recent_activity if recent_activity else '  📭 No recent clustering activity'}

📈 Performance Metrics:
• Clustering History: {len(self.clustering_history)} events
• Error Count: {self.error_count} / {self.circuit_breaker_threshold} (threshold)
• Last Fit: {'✅ Success' if self._last_fit_meta.get('status', '').startswith('Clustering successful') else '❌ Failed/None'}
• Last Recall: {'✅ Success' if self._last_recall_meta.get('status') == 'Recall successful' else '❌ Failed/None'}

🧬 Intelligence Settings:
• Reclustering Threshold: {self.clustering_intelligence['reclustering_threshold']:.1%}
• Quality Improvement Threshold: {self.clustering_intelligence['quality_improvement_threshold']:.1%}
• Effectiveness Decay: {self.clustering_intelligence['effectiveness_decay']:.1%}
• Pattern Memory Factor: {self.clustering_intelligence['pattern_memory_factor']:.1%}

🔍 Advanced Analytics:
• Pattern Cache Size: {len(getattr(self, 'pattern_cache', {}))}
• Effectiveness Variance: {self._calculate_effectiveness_variance():.3f}
• Bootstrap Progress: {min(1.0, self.trades_seen / self.bootstrap_trades):.1%}
            """
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "report_generation")
            return f"Report generation failed: {error_context}"

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for system monitoring"""
        try:
            return {
                'module_name': 'PlaybookClusterer',
                'status': 'disabled' if self.is_disabled else 'healthy',
                'metrics': self._get_clustering_health_metrics(),
                'alerts': self._generate_health_alerts(),
                'recommendations': self._generate_health_recommendations()
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "health_status")
            return {'error': str(error_context)}

    def _generate_health_alerts(self) -> List[Dict[str, Any]]:
        """Generate health-related alerts"""
        alerts = []
        
        try:
            if self.is_disabled:
                alerts.append({
                    'severity': 'critical',
                    'message': 'PlaybookClusterer disabled due to errors',
                    'action': 'Investigate error logs and restart module'
                })
            
            if self.error_count > 2:
                alerts.append({
                    'severity': 'warning',
                    'message': f'High error count: {self.error_count}',
                    'action': 'Monitor for recurring clustering issues'
                })
            
            quality_score = self.clustering_metrics.get('silhouette_score', 0.5)
            if quality_score < 0.3 and self._ready:
                alerts.append({
                    'severity': 'warning',
                    'message': f'Low clustering quality: {quality_score:.3f}',
                    'action': 'Review clustering parameters or data quality'
                })
            
            clusters_with_data = len([k for k, v in self.cluster_effectiveness.items() if v['total_trades'] > 0])
            if clusters_with_data < self.n_clusters / 2:
                alerts.append({
                    'severity': 'info',
                    'message': f'Only {clusters_with_data}/{self.n_clusters} clusters have data',
                    'action': 'Continue trading to build cluster effectiveness baselines'
                })
            
            if not self._ready and self.trades_seen >= self.bootstrap_trades:
                alerts.append({
                    'severity': 'warning',
                    'message': 'Clustering not ready despite sufficient trades',
                    'action': 'Check playbook memory availability and data quality'
                })
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "health_alerts")
            alerts.append({
                'severity': 'error',
                'message': f'Health alert generation failed: {error_context}',
                'action': 'Investigate health monitoring system'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        try:
            if self.is_disabled:
                recommendations.append("Restart PlaybookClusterer module after investigating errors")
            
            if len(self.clustering_history) < 10:
                recommendations.append("Insufficient clustering history - continue operations to build performance baseline")
            
            if self.bootstrap_mode and self.trades_seen >= self.bootstrap_trades * 0.8:
                recommendations.append("Bootstrap phase nearly complete - prepare for operational clustering")
            
            quality_trend = self.clustering_metrics.get('quality_trend', 'stable')
            if quality_trend == 'declining':
                recommendations.append("Clustering quality declining - consider parameter adjustment or data review")
            elif quality_trend == 'improving':
                recommendations.append("Clustering quality improving - current configuration performing well")
            
            if not SKLEARN_AVAILABLE:
                recommendations.append("Consider installing scikit-learn for advanced clustering capabilities")
            
            effectiveness_variance = self._calculate_effectiveness_variance()
            if effectiveness_variance > 0.3:
                recommendations.append("High variance in cluster effectiveness - review cluster configuration")
            
            if not recommendations:
                recommendations.append("PlaybookClusterer operating within normal parameters")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "health_recommendations")
            recommendations.append(f"Health recommendation generation failed: {error_context}")
        
        return recommendations

    # ═══════════════════════════════════════════════════════════════════
    # RESET AND CLEANUP METHODS
    # ═══════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        try:
            # Call parent reset
            super().reset()
            
            # Reset clustering state
            self._reset_clustering_models()
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
                'quality_trend': 'stable',
                'last_quality_check': None,
                'data_coverage': 0.0,
                'feature_diversity': 0.0
            }
            
            # Reset effectiveness tracking
            self.cluster_effectiveness.clear()
            
            # Reset pattern analysis
            self.pattern_cache.clear()
            self.pattern_performance.clear()
            self._last_performance_window = 0.0
            self._last_regime = 'unknown'
            
            # Reset error state
            self.error_count = 0
            self.is_disabled = False
            
            # Reset runtime state
            if hasattr(self, '_last_weights'):
                delattr(self, '_last_weights')
            if hasattr(self, '_cluster_usage_history'):
                delattr(self, '_cluster_usage_history')
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Playbook Clusterer reset complete",
                clusters=self.n_clusters,
                bootstrap_threshold=self.bootstrap_trades
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "reset")
            self.logger.error(f"Clustering reset failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # REQUIRED ABSTRACT METHODS FROM BASE MODULE
    # ═══════════════════════════════════════════════════════════════════

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """
        Calculate confidence in clustering decisions and pattern recognition
        
        Args:
            action: The action being evaluated
            **inputs: Additional inputs for confidence calculation
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            # Base confidence from clustering quality
            base_confidence = 0.5
            
            if self._ready:
                # Clustering quality factor
                silhouette = self.clustering_metrics.get('silhouette_score', 0.0)
                quality_factor = max(0.0, min(1.0, (silhouette + 1.0) / 2.0))  # Convert from [-1,1] to [0,1]
                
                # Cluster stability factor
                stability_factor = self.clustering_metrics.get('cluster_stability', 0.5)
                
                # Data coverage factor
                coverage_factor = self.clustering_metrics.get('data_coverage', 0.5)
                
                # Combine quality factors
                base_confidence = (quality_factor * 0.4 + stability_factor * 0.3 + coverage_factor * 0.3)
            
            # Effectiveness adjustment based on recent cluster performance
            effectiveness_adjustment = 1.0
            if self.cluster_effectiveness:
                recent_effectiveness = []
                for cluster_data in self.cluster_effectiveness.values():
                    if cluster_data['total_trades'] > 0:
                        recent_effectiveness.append(cluster_data['effectiveness_score'])
                
                if recent_effectiveness:
                    avg_effectiveness = float(np.mean(recent_effectiveness))
                    effectiveness_adjustment = max(0.5, min(1.5, avg_effectiveness))
            
            # Bootstrap mode penalty
            bootstrap_penalty = 0.8 if self.bootstrap_mode else 1.0
            
            # Error state penalty
            error_penalty = max(0.3, 1.0 - (self.error_count / max(1, self.circuit_breaker_threshold)))
            
            # Calculate final confidence
            final_confidence = base_confidence * effectiveness_adjustment * bootstrap_penalty * error_penalty
            
            return float(max(0.1, min(1.0, final_confidence)))
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """
        Propose optimal actions based on current clustering analysis
        
        Args:
            **inputs: Context inputs for action proposal
            
        Returns:
            Dictionary containing proposed actions and clustering recommendations
        """
        try:
            # Get current clustering data
            clustering_data = await self._get_comprehensive_clustering_data()
            
            # Perform cluster analysis
            cluster_analysis = await self._analyze_clusters_comprehensive(clustering_data)
            
            # Extract key information
            cluster_weights = cluster_analysis.get('weights', self._generate_safe_fallback_weights())
            dominant_cluster = int(np.argmax(cluster_weights)) if cluster_weights is not None else 0
            confidence = cluster_analysis.get('confidence', 0.5)
            
            # Get cluster effectiveness for dominant cluster
            dominant_effectiveness = self.cluster_effectiveness.get(dominant_cluster, ClusterEffectivenessDict())
            
            # Determine action type based on clustering state
            if self.bootstrap_mode:
                action_type = 'exploration'
                strategy_recommendation = 'conservative_exploration'
            elif confidence > 0.7:
                action_type = 'exploitation'
                strategy_recommendation = 'cluster_focused'
            else:
                action_type = 'balanced'
                strategy_recommendation = 'diversified'
            
            # Generate cluster-specific recommendations
            cluster_recommendations = []
            if cluster_weights is not None:
                # Top performing clusters
                top_clusters = sorted(enumerate(cluster_weights), key=lambda x: x[1], reverse=True)[:3]
                for idx, (cluster_id, weight) in enumerate(top_clusters):
                    cluster_data = self.cluster_effectiveness.get(cluster_id, ClusterEffectivenessDict())
                    
                    if idx == 0:  # Dominant cluster
                        cluster_recommendations.append({
                            'cluster_id': cluster_id,
                            'weight': float(weight),
                            'role': 'primary',
                            'effectiveness': cluster_data['effectiveness_score'],
                            'total_trades': cluster_data['total_trades'],
                            'recommendation': 'Primary strategy allocation'
                        })
                    else:  # Supporting clusters
                        cluster_recommendations.append({
                            'cluster_id': cluster_id,
                            'weight': float(weight),
                            'role': 'supporting',
                            'effectiveness': cluster_data['effectiveness_score'],
                            'total_trades': cluster_data['total_trades'],
                            'recommendation': f'Secondary allocation for diversification'
                        })
            
            # Calculate risk management parameters
            risk_adjustment = 1.0
            if confidence < 0.5:
                risk_adjustment = 0.7  # Reduce risk when confidence is low
            elif confidence > 0.8:
                risk_adjustment = 1.2  # Increase risk when confidence is high
            
            # Position sizing recommendations
            base_size = 1.0
            confidence_multiplier = confidence
            effectiveness_multiplier = dominant_effectiveness['effectiveness_score']
            
            suggested_size = base_size * confidence_multiplier * effectiveness_multiplier * risk_adjustment
            
            # Create comprehensive action proposal
            proposed_action = {
                'action_type': action_type,
                'strategy_recommendation': strategy_recommendation,
                'dominant_cluster': dominant_cluster,
                'cluster_confidence': confidence,
                'cluster_weights': cluster_weights.tolist() if cluster_weights is not None else [],
                'cluster_recommendations': cluster_recommendations,
                'position_sizing': {
                    'base_size': base_size,
                    'confidence_multiplier': confidence_multiplier,
                    'effectiveness_multiplier': effectiveness_multiplier,
                    'risk_adjustment': risk_adjustment,
                    'final_size': suggested_size
                },
                'risk_management': {
                    'clustering_confidence': confidence,
                    'pattern_stability': self.clustering_metrics.get('cluster_stability', 0.5),
                    'data_quality': self.clustering_metrics.get('silhouette_score', 0.0),
                    'recommended_approach': 'conservative' if confidence < 0.6 else 'aggressive'
                },
                'pattern_insights': {
                    'bootstrap_mode': self.bootstrap_mode,
                    'clustering_ready': self._ready,
                    'total_patterns': len(self.cluster_effectiveness),
                    'best_pattern_effectiveness': max([v['effectiveness_score'] for v in self.cluster_effectiveness.values()], default=0.5),
                    'pattern_diversity': self.clustering_metrics.get('feature_diversity', 0.5)
                },
                'recommendations': self._generate_intelligent_cluster_recommendations(cluster_analysis),
                'thesis': f"Cluster {dominant_cluster} identified as dominant pattern with {confidence:.1%} confidence. Strategy: {strategy_recommendation} with {len(cluster_recommendations)} supporting patterns.",
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            return proposed_action
            
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                'action_type': 'conservative',
                'strategy_recommendation': 'safe_fallback',
                'dominant_cluster': 0,
                'cluster_confidence': 0.5,
                'cluster_weights': [1.0/self.n_clusters] * self.n_clusters,
                'cluster_recommendations': [],
                'position_sizing': {'final_size': 0.5},
                'risk_management': {'recommended_approach': 'conservative'},
                'pattern_insights': {'bootstrap_mode': True, 'clustering_ready': False},
                'recommendations': ['Use conservative approach until clustering improves'],
                'thesis': 'Clustering system error, using safe fallback approach',
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }