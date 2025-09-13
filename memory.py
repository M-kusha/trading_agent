# modules/memory/components/base.py
"""
Base Component Class for Unified Memory System
Provides common interface and utilities for all memory components
"""

from typing import Dict, Any, Optional, List
import numpy as np
import time
from abc import ABC, abstractmethod


class MemoryComponent(ABC):
    """Base class for all memory components"""
    
    def __init__(self, config: Any, shared_resources: Dict[str, Any]):
        """
        Initialize memory component
        
        Args:
            config: UnifiedMemoryConfig instance
            shared_resources: Dictionary containing shared resources
        """
        self.config = config
        self.shared = shared_resources
        
        # Extract commonly used resources
        self.store = shared_resources['store']
        self.extractor = shared_resources['extractor']
        self.pattern_detector = shared_resources.get('pattern_detector')
        self.scaler = shared_resources['scaler']
        self.encoder = shared_resources.get('encoder')
        self.cache = shared_resources['cache']
        self.utils = shared_resources['utils']
        self.logger = shared_resources['logger']
        
        # Component state
        self._initialized = False
        self._last_process = 0
        self._process_count = 0
        self._error_count = 0
        
        # Initialize component-specific resources
        self._initialize_component()
        self._initialized = True
    
    @abstractmethod
    def _initialize_component(self) -> None:
        """Initialize component-specific resources"""
        pass
    
    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory operations for this component
        
        Args:
            context: Processing context with market data, trades, etc.
            
        Returns:
            Component-specific results dictionary
        """
        pass
    
    def get_relevant_memories(
        self, 
        query: np.ndarray, 
        k: int = 5,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relevant memories from shared store
        
        Args:
            query: Query vector
            k: Number of memories to retrieve
            filter_fn: Optional filter function
            
        Returns:
            List of relevant memory entries
        """
        return self.store.query(query, k, filter_fn)
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(f"{self.__class__.__name__}:{key}")
    
    def cache_put(self, key: str, value: Any, ttl: int = 60) -> None:
        """Put value in cache"""
        self.cache.put(f"{self.__class__.__name__}:{key}", value, ttl)
    
    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        if self.config.debug:
            self.logger.debug(f"[{self.__class__.__name__}] {message}", **kwargs)
    
    def log_error(self, message: str, error: Exception) -> None:
        """Log error message"""
        self._error_count += 1
        self.logger.error(f"[{self.__class__.__name__}] {message}: {error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics"""
        return {
            'initialized': self._initialized,
            'last_process': self._last_process,
            'process_count': self._process_count,
            'error_count': self._error_count
        }# modules/memory/components/base.py
"""
Base Component Class for Unified Memory System
Provides common interface and utilities for all memory components
"""

from typing import Dict, Any, Optional, List
import numpy as np
import time
from abc import ABC, abstractmethod


class MemoryComponent(ABC):
    """Base class for all memory components"""
    
    def __init__(self, config: Any, shared_resources: Dict[str, Any]):
        """
        Initialize memory component
        
        Args:
            config: UnifiedMemoryConfig instance
            shared_resources: Dictionary containing shared resources
        """
        self.config = config
        self.shared = shared_resources
        
        # Extract commonly used resources
        self.store = shared_resources['store']
        self.extractor = shared_resources['extractor']
        self.pattern_detector = shared_resources.get('pattern_detector')
        self.scaler = shared_resources['scaler']
        self.encoder = shared_resources.get('encoder')
        self.cache = shared_resources['cache']
        self.utils = shared_resources['utils']
        self.logger = shared_resources['logger']
        
        # Component state
        self._initialized = False
        self._last_process = 0
        self._process_count = 0
        self._error_count = 0
        
        # Initialize component-specific resources
        self._initialize_component()
        self._initialized = True
    
    @abstractmethod
    def _initialize_component(self) -> None:
        """Initialize component-specific resources"""
        pass
    
    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory operations for this component
        
        Args:
            context: Processing context with market data, trades, etc.
            
        Returns:
            Component-specific results dictionary
        """
        pass
    
    def get_relevant_memories(
        self, 
        query: np.ndarray, 
        k: int = 5,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relevant memories from shared store
        
        Args:
            query: Query vector
            k: Number of memories to retrieve
            filter_fn: Optional filter function
            
        Returns:
            List of relevant memory entries
        """
        return self.store.query(query, k, filter_fn)
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(f"{self.__class__.__name__}:{key}")
    
    def cache_put(self, key: str, value: Any, ttl: int = 60) -> None:
        """Put value in cache"""
        self.cache.put(f"{self.__class__.__name__}:{key}", value, ttl)
    
    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        if self.config.debug:
            self.logger.debug(f"[{self.__class__.__name__}] {message}", **kwargs)
    
    def log_error(self, message: str, error: Exception) -> None:
        """Log error message"""
        self._error_count += 1
        self.logger.error(f"[{self.__class__.__name__}] {message}: {error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics"""
        return {
            'initialized': self._initialized,
            'last_process': self._last_process,
            'process_count': self._process_count,
            'error_count': self._error_count
        }# modules/memory/components/compression.py
"""
Memory Compression Component
Compresses experiences using PCA and neural encoding
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from .base import MemoryComponent


class CompressionComponent(MemoryComponent):
    """Memory compression using PCA and neural encoding"""
    
    def _initialize_component(self) -> None:
        """Initialize compression-specific resources"""
        # Configuration
        self.n_components = self.config.n_components
        self.compression_ratio = self.config.compression_ratio
        self.compress_interval = self.config.compress_interval
        
        # Memory buffers
        self.profit_memory = []
        self.loss_memory = []
        
        # Compressed representations
        self.intuition_vector = np.zeros(self.n_components, dtype=np.float32)
        self.profit_direction = np.zeros(self.n_components, dtype=np.float32)
        self.loss_direction = np.zeros(self.n_components, dtype=np.float32)
        
        # PCA models
        self.profit_pca = PCA(n_components=self.n_components)
        self.loss_pca = PCA(n_components=self.n_components)
        self._profit_pca_fitted = False
        self._loss_pca_fitted = False
        
        # Tracking
        self.compression_count = 0
        self.compression_quality_scores = deque(maxlen=50)
        self.explained_variance_history = deque(maxlen=50)
        self.compression_efficiency = 0.0
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory compression"""
        try:
            # Get memory data
            memory_data = context.get('memory_data', [])
            trades = context.get('trades', [])
            
            # Update memory buffers
            self._update_memory_buffers(memory_data, trades)
            
            # Check if compression needed
            episode = context.get('episode', 0)
            if self._should_compress(episode):
                compression_result = self._perform_compression()
            else:
                compression_result = {'compression_performed': False}
            
            # Update intuition vector
            intuition_result = self._update_intuition_vector()
            compression_result.update(intuition_result)
            
            return self._format_output(compression_result)
            
        except Exception as e:
            self.log_error("Compression processing failed", e)
            return self._get_fallback_output()
    
    def _update_memory_buffers(self, memory_data: List[Dict], trades: List[Dict]) -> None:
        """Update profit and loss memory buffers"""
        # Process memory data
        for entry in memory_data:
            if 'pnl' in entry and 'features' in entry:
                features = np.array(entry['features'], dtype=np.float32)
                pnl = entry['pnl']
                
                if pnl > self.config.replay_profit_threshold:
                    self.profit_memory.append((features, pnl))
                elif pnl < -self.config.replay_profit_threshold / 2:
                    self.loss_memory.append((features, abs(pnl)))
        
        # Process recent trades
        for trade in trades[-10:]:
            if isinstance(trade, dict) and 'pnl' in trade:
                features = self.extractor.extract_trade_features(trade, {})
                pnl = trade['pnl']
                
                if pnl > self.config.replay_profit_threshold:
                    self.profit_memory.append((features, pnl))
                elif pnl < -self.config.replay_profit_threshold / 2:
                    self.loss_memory.append((features, abs(pnl)))
        
        # Limit memory size
        max_size = int(self.config.max_memory_size * 0.1)  # 10% for compression
        if len(self.profit_memory) > max_size:
            self.profit_memory = self.profit_memory[-max_size:]
        if len(self.loss_memory) > max_size:
            self.loss_memory = self.loss_memory[-max_size:]
    
    def _should_compress(self, episode: int) -> bool:
        """Check if compression should be performed"""
        return (
            episode > 0 and
            episode % self.compress_interval == 0 and
            len(self.profit_memory) >= self.n_components
        )
    
    def _perform_compression(self) -> Dict[str, Any]:
        """Perform memory compression using PCA"""
        results = {'compression_performed': True}
        
        # Compress profit patterns
        if len(self.profit_memory) >= self.n_components:
            profit_result = self._compress_profit_patterns()
            results.update(profit_result)
        
        # Compress loss patterns
        if len(self.loss_memory) >= self.n_components:
            loss_result = self._compress_loss_patterns()
            results.update(loss_result)
        
        self.compression_count += 1
        
        return results
    
    def _compress_profit_patterns(self) -> Dict[str, Any]:
        """Compress profitable patterns"""
        try:
            # Extract features and weights
            features = np.array([m[0] for m in self.profit_memory])
            profits = np.array([m[1] for m in self.profit_memory])
            
            # Weight by profit
            weights = profits / np.max(profits) if np.max(profits) > 0 else np.ones_like(profits)
            weighted_features = features * weights.reshape(-1, 1)
            
            # Standardize
            if not self._profit_pca_fitted:
                standardized = self.scaler.fit_transform(weighted_features)
                self._profit_pca_fitted = True
            else:
                standardized = self.scaler.transform(weighted_features)
            
            # Apply PCA
            self.profit_pca.fit(standardized)
            compressed = self.profit_pca.transform(standardized)
            
            # Update profit direction
            profit_weights = weights / np.sum(weights)
            self.profit_direction = np.average(compressed, axis=0, weights=profit_weights).astype(np.float32)
            
            # Calculate quality
            explained_variance = np.sum(self.profit_pca.explained_variance_ratio_)
            self.explained_variance_history.append(explained_variance)
            
            return {
                'profit_compression': {
                    'samples_compressed': len(features),
                    'explained_variance': explained_variance,
                    'profit_direction_strength': float(np.linalg.norm(self.profit_direction)),
                    'avg_profit': float(np.mean(profits))
                }
            }
            
        except Exception as e:
            self.log_error("Profit compression failed", e)
            return {'profit_compression': {'error': str(e)}}
    
    def _compress_loss_patterns(self) -> Dict[str, Any]:
        """Compress loss patterns"""
        try:
            # Extract features and weights
            features = np.array([m[0] for m in self.loss_memory])
            losses = np.array([m[1] for m in self.loss_memory])
            
            # Weight by loss magnitude
            weights = losses / np.max(losses) if np.max(losses) > 0 else np.ones_like(losses)
            weighted_features = features * weights.reshape(-1, 1)
            
            # Standardize and compress
            standardized = self.scaler.transform(weighted_features)
            self.loss_pca.fit(standardized)
            compressed = self.loss_pca.transform(standardized)
            
            # Update loss direction
            loss_weights = weights / np.sum(weights)
            self.loss_direction = np.average(compressed, axis=0, weights=loss_weights).astype(np.float32)
            
            # Calculate quality
            explained_variance = np.sum(self.loss_pca.explained_variance_ratio_)
            
            return {
                'loss_compression': {
                    'samples_compressed': len(features),
                    'explained_variance': explained_variance,
                    'loss_direction_strength': float(np.linalg.norm(self.loss_direction)),
                    'avg_loss': float(np.mean(losses))
                }
            }
            
        except Exception as e:
            self.log_error("Loss compression failed", e)
            return {'loss_compression': {'error': str(e)}}
    
    def _update_intuition_vector(self) -> Dict[str, Any]:
        """Update intuition vector based on compressed patterns"""
        profit_strength = np.linalg.norm(self.profit_direction)
        loss_strength = np.linalg.norm(self.loss_direction)
        
        if profit_strength > 0 and loss_strength > 0:
            # Blend profit and loss directions
            profit_component = self.profit_direction * 2.0  # Profit weight
            loss_component = -self.loss_direction * 1.5     # Loss avoidance weight
            combined = profit_component + loss_component
            
            # Apply learning rate
            learning_rate = 0.1
            self.intuition_vector = (
                (1 - learning_rate) * self.intuition_vector +
                learning_rate * combined
            ).astype(np.float32)
            
            # Normalize
            norm = np.linalg.norm(self.intuition_vector)
            if norm > 1e-8:
                self.intuition_vector = self.intuition_vector / norm
        
        elif profit_strength > 0:
            # Only profit direction
            self.intuition_vector = self.profit_direction.copy()
        
        return {
            'intuition_update': {
                'intuition_strength': float(np.linalg.norm(self.intuition_vector)),
                'profit_strength': float(profit_strength),
                'loss_strength': float(loss_strength)
            }
        }
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements"""
        # Calculate compression efficiency
        if self.explained_variance_history:
            self.compression_efficiency = np.mean(list(self.explained_variance_history))
        
        # Get feature importance from PCA
        feature_importance = {}
        if self._profit_pca_fitted and hasattr(self.profit_pca, 'components_'):
            feature_importance = {
                'profit_components': self.profit_pca.components_.tolist(),
                'explained_variance_ratio': self.profit_pca.explained_variance_ratio_.tolist(),
                'n_features': self.profit_pca.n_features_in_
            }
        
        return {
            'compressed_patterns': {
                'profit_direction': self.profit_direction.tolist(),
                'loss_direction': self.loss_direction.tolist(),
                'profit_strength': float(np.linalg.norm(self.profit_direction)),
                'loss_strength': float(np.linalg.norm(self.loss_direction)),
                'compression_count': self.compression_count
            },
            'feature_importance': feature_importance,
            'intuition_vector': {
                'vector': self.intuition_vector.tolist(),
                'strength': float(np.linalg.norm(self.intuition_vector)),
                'components': self.n_components,
                'last_updated': time.time()
            },
            'memory_compression': {
                'total_memories': len(self.profit_memory) + len(self.loss_memory),
                'profit_memories': len(self.profit_memory),
                'loss_memories': len(self.loss_memory),
                'compression_efficiency': self.compression_efficiency,
                'last_compression': self.compression_count
            }
        }
    
    def _get_fallback_output(self) -> Dict[str, Any]:
        """Get fallback output for errors"""
        return {
            'compressed_patterns': {
                'profit_direction': [],
                'loss_direction': [],
                'profit_strength': 0.0,
                'loss_strength': 0.0,
                'compression_count': 0
            },
            'feature_importance': {},
            'intuition_vector': {
                'vector': [],
                'strength': 0.0,
                'components': self.n_components,
                'last_updated': 0
            },
            'memory_compression': {
                'total_memories': 0,
                'compression_efficiency': 0.0
            }
        }# modules/memory/components/mistakes.py
"""
Mistake Memory Component
Identifies danger zones and provides loss avoidance signals
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import time

from .base import MemoryComponent


class MistakeComponent(MemoryComponent):
    """Mistake detection and avoidance component"""
    
    def _initialize_component(self) -> None:
        """Initialize mistake-specific resources"""
        # Configuration
        self.n_clusters = self.config.n_clusters
        self.danger_threshold = self.config.danger_threshold
        self.avoidance_sensitivity = self.config.avoidance_sensitivity
        self.profit_threshold = self.config.mistake_profit_threshold
        
        # Memory buffers
        self.loss_buffer = []
        self.win_buffer = []
        
        # Clustering
        self.loss_clusterer = None
        self.win_clusterer = None
        self.danger_zones = []
        self.profit_zones = []
        
        # State
        self.consecutive_losses = 0
        self.avoidance_signal = 0.0
        
        # Pattern tracking
        self.loss_patterns = defaultdict(lambda: {
            'count': 0, 'severity': 0.0, 'last_seen': 0
        })
        self.win_patterns = defaultdict(lambda: {
            'count': 0, 'profitability': 0.0, 'last_seen': 0
        })
        
        # Metrics
        self.cluster_quality_scores = deque(maxlen=20)
        self.avoidance_effectiveness = 0.0
        self.false_positive_rate = 0.0
        self.true_positive_rate = 0.0
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process mistake memory operations"""
        try:
            # Process new trades
            learning_result = self._process_learning_data(context)
            
            # Update clustering if needed
            if self._should_update_clustering():
                clustering_result = self._update_clustering()
                learning_result.update(clustering_result)
            
            # Calculate avoidance signals
            avoidance_result = self._calculate_avoidance_signals(context)
            learning_result.update(avoidance_result)
            
            return self._format_output(learning_result)
            
        except Exception as e:
            self.log_error("Mistake processing failed", e)
            return self._get_fallback_output()
    
    def _process_learning_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process trades for learning"""
        trades = context.get('trades', [])
        market_context = context.get('market_context', {})
        
        losses_learned = 0
        wins_learned = 0
        
        for trade in trades[-20:]:
            if not isinstance(trade, dict) or 'pnl' not in trade:
                continue
            
            # Extract features
            features = self._extract_trade_features(trade, market_context)
            if features is None:
                continue
            
            pnl = trade['pnl']
            
            # Store based on outcome
            if pnl < -self.profit_threshold / 2:
                self._process_loss_trade(features, abs(pnl), trade)
                losses_learned += 1
                self.consecutive_losses += 1
            elif pnl > self.profit_threshold:
                self._process_win_trade(features, pnl, trade)
                wins_learned += 1
                self.consecutive_losses = 0
        
        return {
            'losses_learned': losses_learned,
            'wins_learned': wins_learned,
            'total_loss_memories': len(self.loss_buffer),
            'total_win_memories': len(self.win_buffer)
        }
    
    def _extract_trade_features(self, trade: Dict, market_context: Dict) -> Optional[np.ndarray]:
        """Extract features from trade"""
        try:
            features = []
            
            # Trade features
            features.append(trade.get('confidence', 0.5))
            features.append(trade.get('volume', 1.0))
            features.append(trade.get('duration', 1.0))
            
            # Market context
            volatility = market_context.get('volatility', 0.5)
            if isinstance(volatility, dict):
                volatility = list(volatility.values())[0] if volatility else 0.5
            features.append(float(volatility))
            
            # Session encoding
            session_map = {'asian': 0.0, 'european': 0.5, 'us': 1.0}
            session = market_context.get('session', 'unknown')
            features.append(session_map.get(session, 0.25))
            
            # Regime encoding
            regime_map = {'trending': 1.0, 'ranging': 0.0, 'volatile': 0.5}
            regime = market_context.get('regime', 'unknown')
            features.append(regime_map.get(regime, 0.25))
            
            # Pad to minimum size
            while len(features) < 10:
                features.append(0.0)
            
            return np.array(features[:20], dtype=np.float32)
            
        except Exception:
            return None
    
    def _process_loss_trade(self, features: np.ndarray, loss: float, trade: Dict) -> None:
        """Process loss trade"""
        self.loss_buffer.append((features, loss, trade))
        
        # Limit buffer size
        if len(self.loss_buffer) > self.config.max_memory_size // 10:
            self.loss_buffer.pop(0)
        
        # Extract and record pattern
        pattern = self._extract_pattern(features, trade)
        if pattern:
            self._record_loss_pattern(pattern, loss)
    
    def _process_win_trade(self, features: np.ndarray, profit: float, trade: Dict) -> None:
        """Process win trade"""
        self.win_buffer.append((features, profit, trade))
        
        # Limit buffer size
        if len(self.win_buffer) > self.config.max_memory_size // 10:
            self.win_buffer.pop(0)
        
        # Extract and record pattern
        pattern = self._extract_pattern(features, trade)
        if pattern:
            self._record_win_pattern(pattern, profit)
    
    def _extract_pattern(self, features: np.ndarray, trade: Dict) -> Optional[str]:
        """Extract pattern from features and trade"""
        try:
            pattern_elements = []
            
            # Feature-based pattern
            for i, feat in enumerate(features[:5]):
                if feat > 0.7:
                    pattern_elements.append(f'H{i}')
                elif feat < 0.3:
                    pattern_elements.append(f'L{i}')
                else:
                    pattern_elements.append(f'M{i}')
            
            # Add action if available
            action = trade.get('action')
            if action and len(action) > 0:
                if action[0] > 0.5:
                    pattern_elements.append('BUY')
                elif action[0] < -0.5:
                    pattern_elements.append('SELL')
                else:
                    pattern_elements.append('HOLD')
            
            return '_'.join(pattern_elements) if pattern_elements else None
            
        except Exception:
            return None
    
    def _record_loss_pattern(self, pattern: str, loss: float) -> None:
        """Record loss pattern"""
        data = self.loss_patterns[pattern]
        data['count'] += 1
        data['severity'] += loss
        data['last_seen'] = time.time()
    
    def _record_win_pattern(self, pattern: str, profit: float) -> None:
        """Record win pattern"""
        data = self.win_patterns[pattern]
        data['count'] += 1
        data['profitability'] += profit
        data['last_seen'] = time.time()
    
    def _should_update_clustering(self) -> bool:
        """Check if clustering should be updated"""
        return (
            len(self.loss_buffer) >= self.n_clusters and
            len(self.loss_buffer) % 10 == 0
        )
    
    def _update_clustering(self) -> Dict[str, Any]:
        """Update clustering for danger zones"""
        results = {}
        
        # Cluster loss data
        if len(self.loss_buffer) >= self.n_clusters:
            loss_result = self._cluster_loss_data()
            results.update(loss_result)
        
        # Cluster win data
        if len(self.win_buffer) >= self.n_clusters:
            win_result = self._cluster_win_data()
            results.update(win_result)
        
        return {'clustering_updated': True, **results}
    
    def _cluster_loss_data(self) -> Dict[str, Any]:
        """Cluster loss data to identify danger zones"""
        try:
            # Extract features
            features = np.array([entry[0] for entry in self.loss_buffer])
            losses = np.array([entry[1] for entry in self.loss_buffer])
            
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Apply DBSCAN clustering
            self.loss_clusterer = DBSCAN(eps=0.3, min_samples=5)
            clusters = self.loss_clusterer.fit_predict(features_scaled)
            
            # Calculate quality score
            if len(np.unique(clusters)) > 1:
                quality_score = silhouette_score(features_scaled, clusters)
                self.cluster_quality_scores.append(quality_score)
            else:
                quality_score = 0.0
            
            # Identify danger zones (cluster centers)
            self.danger_zones = []
            for cluster_id in np.unique(clusters):
                if cluster_id != -1:  # Ignore noise
                    cluster_mask = clusters == cluster_id
                    cluster_center = np.mean(features_scaled[cluster_mask], axis=0)
                    cluster_severity = np.mean(losses[cluster_mask])
                    
                    self.danger_zones.append({
                        'center': cluster_center.tolist(),
                        'severity': float(cluster_severity),
                        'size': int(np.sum(cluster_mask))
                    })
            
            return {
                'loss_clustering': {
                    'clusters_found': len(self.danger_zones),
                    'quality_score': float(quality_score),
                    'total_losses': len(features)
                }
            }
            
        except Exception as e:
            self.log_error("Loss clustering failed", e)
            return {'loss_clustering': {'error': str(e)}}
    
    def _cluster_win_data(self) -> Dict[str, Any]:
        """Cluster win data to identify profit zones"""
        try:
            # Extract features
            features = np.array([entry[0] for entry in self.win_buffer])
            profits = np.array([entry[1] for entry in self.win_buffer])
            
            # Standardize features
            features_scaled = self.scaler.transform(features)
            
            # Apply DBSCAN clustering
            self.win_clusterer = DBSCAN(eps=0.3, min_samples=5)
            clusters = self.win_clusterer.fit_predict(features_scaled)
            
            # Identify profit zones
            self.profit_zones = []
            for cluster_id in np.unique(clusters):
                if cluster_id != -1:
                    cluster_mask = clusters == cluster_id
                    cluster_center = np.mean(features_scaled[cluster_mask], axis=0)
                    cluster_profit = np.mean(profits[cluster_mask])
                    
                    self.profit_zones.append({
                        'center': cluster_center.tolist(),
                        'profitability': float(cluster_profit),
                        'size': int(np.sum(cluster_mask))
                    })
            
            return {
                'win_clustering': {
                    'clusters_found': len(self.profit_zones),
                    'total_wins': len(features)
                }
            }
            
        except Exception as e:
            self.log_error("Win clustering failed", e)
            return {'win_clustering': {'error': str(e)}}
    
    def _calculate_avoidance_signals(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate avoidance signals"""
        current_features = context.get('features')
        if current_features is None:
            return {
                'avoidance_signal': 0.0,
                'danger_similarity': 0.0,
                'profit_similarity': 0.0
            }
        
        # Calculate similarities
        danger_similarity = self._calculate_danger_similarity(current_features)
        profit_similarity = self._calculate_profit_similarity(current_features)
        
        # Calculate avoidance signal
        avoidance_signal = danger_similarity * self.avoidance_sensitivity
        
        # Reduce avoidance if near profit zone
        if profit_similarity > danger_similarity:
            avoidance_signal *= 0.5
        
        # Amplify based on consecutive losses
        if self.consecutive_losses > 2:
            avoidance_signal *= (1.0 + self.consecutive_losses * 0.1)
        
        self.avoidance_signal = np.clip(avoidance_signal, 0.0, 1.0)
        
        return {
            'avoidance_signal': float(self.avoidance_signal),
            'danger_similarity': float(danger_similarity),
            'profit_similarity': float(profit_similarity),
            'consecutive_losses': self.consecutive_losses
        }
    
    def _calculate_danger_similarity(self, features: np.ndarray) -> float:
        """Calculate similarity to danger zones"""
        if not self.danger_zones:
            return 0.0
        
        try:
            # Ensure features is numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Reshape if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Standardize features
            features_scaled = self.scaler.transform(features)
            
            # Calculate distances to danger zones
            min_distance = float('inf')
            for zone in self.danger_zones:
                center = np.array(zone['center'])
                distance = np.linalg.norm(features_scaled[0] - center)
                weighted_distance = distance / (zone['severity'] + 1.0)
                min_distance = min(min_distance, weighted_distance)
            
            # Convert distance to similarity
            similarity = 1.0 / (1.0 + min_distance)
            return float(similarity)
            
        except Exception:
            return 0.0
    
    def _calculate_profit_similarity(self, features: np.ndarray) -> float:
        """Calculate similarity to profit zones"""
        if not self.profit_zones:
            return 0.0
        
        try:
            # Ensure features is numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Reshape if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Standardize features
            features_scaled = self.scaler.transform(features)
            
            # Calculate distances to profit zones
            min_distance = float('inf')
            for zone in self.profit_zones:
                center = np.array(zone['center'])
                distance = np.linalg.norm(features_scaled[0] - center)
                weighted_distance = distance / (zone['profitability'] + 1.0)
                min_distance = min(min_distance, weighted_distance)
            
            # Convert distance to similarity
            similarity = 1.0 / (1.0 + min_distance)
            return float(similarity)
            
        except Exception:
            return 0.0
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements"""
        # Calculate effectiveness metrics
        if self.cluster_quality_scores:
            avg_quality = np.mean(list(self.cluster_quality_scores))
        else:
            avg_quality = 0.0
        
        return {
            'danger_zones': {
                'zones': self.danger_zones,
                'zone_count': len(self.danger_zones),
                'avoidance_sensitivity': self.avoidance_sensitivity,
                'last_updated': time.time()
            },
            'loss_prevention': {
                'avoidance_effectiveness': self.avoidance_effectiveness,
                'false_positive_rate': self.false_positive_rate,
                'true_positive_rate': self.true_positive_rate,
                'cluster_quality': avg_quality,
                'learning_samples': len(self.loss_buffer) + len(self.win_buffer)
            },
            'mistake_avoidance': {
                'avoidance_signal': self.avoidance_signal,
                'consecutive_losses': self.consecutive_losses,
                'danger_zones_count': len(self.danger_zones),
                'profit_zones_count': len(self.profit_zones),
                'total_loss_memories': len(self.loss_buffer),
                'total_win_memories': len(self.win_buffer)
            },
            'mistake_memory': {
                'current_score': float(np.clip(self.avoidance_signal, 0.0, 1.0)),
                'consecutive_losses': self.consecutive_losses,
                'avoidance_signal': self.avoidance_signal,
                'last_updated': time.time()
            },
            'pattern_recognition': {
                'loss_patterns': dict(list(self.loss_patterns.items())[:10]),
                'win_patterns': dict(list(self.win_patterns.items())[:10]),
                'total_loss_patterns': len(self.loss_patterns),
                'total_win_patterns': len(self.win_patterns)
            }
        }
    
    def _get_fallback_output(self) -> Dict[str, Any]:
        """Get fallback output for errors"""
        return {
            'danger_zones': {'zones': [], 'zone_count': 0},
            'loss_prevention': {'avoidance_effectiveness': 0.0, 'learning_samples': 0},
            'mistake_avoidance': {'avoidance_signal': 0.0, 'consecutive_losses': 0},
            'mistake_memory': {'current_score': 0.0, 'avoidance_signal': 0.0},
            'pattern_recognition': {'loss_patterns': {}, 'win_patterns': {}}
        }# modules/memory/components/neural.py
"""
Neural Memory Component
Implements attention-based memory with importance scoring
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .base import MemoryComponent


class NeuralComponent(MemoryComponent):
    """Neural memory with attention mechanisms"""
    
    def _initialize_component(self) -> None:
        """Initialize neural-specific resources"""
        # Configuration
        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.memory_decay = self.config.memory_decay
        self.importance_threshold = self.config.importance_threshold
        self.max_buffer_size = int(self.config.max_memory_size * 0.2)  # 20% for neural
        
        # Memory buffer
        self.buffer = torch.zeros((0, self.embed_dim), dtype=torch.float32)
        self.importance_scores = torch.zeros(0, dtype=torch.float32)
        self.memory_metadata = []
        
        # Neural components
        self._init_neural_networks()
        
        # Tracking
        self.memories_stored = 0
        self.memories_retrieved = 0
        self.avg_importance = 0.0
        self.attention_efficiency = 0.0
        
        # Performance
        self.retrieval_history = deque(maxlen=100)
        self.importance_evolution = deque(maxlen=500)
        self.attention_patterns = deque(maxlen=50)
    
    def _init_neural_networks(self) -> None:
        """Initialize neural network components"""
        try:
            # Encoder network (use shared if available)
            if self.encoder is not None:
                self.memory_encoder = self.encoder
            else:
                self.memory_encoder = self._create_encoder()
            
            # Multi-head attention
            self.attention = nn.MultiheadAttention(
                self.embed_dim,
                self.num_heads,
                dropout=0.1,
                batch_first=True
            )
            
            # Importance prediction head
            self.value_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim // 2, 1),
                nn.Sigmoid()
            )
            
            # Context integration
            self.context_net = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            )
            
            # Initialize weights
            self._init_weights()
            
        except Exception as e:
            self.log_error("Neural network initialization failed", e)
    
    def _create_encoder(self) -> nn.Module:
        """Create encoder network"""
        class Encoder(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim * 2, dim),
                    nn.LayerNorm(dim)
                )
            
            def forward(self, x):
                return self.net(x)
        
        return Encoder(self.embed_dim)
    
    def _init_weights(self) -> None:
        """Initialize network weights"""
        for module in [self.value_head, self.context_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process neural memory operations"""
        try:
            # Store new experiences
            storage_result = await self._store_experiences(context)
            
            # Perform retrieval if query provided
            query = context.get('query')
            if query is not None:
                retrieval_result = await self._perform_retrieval(query)
                storage_result.update(retrieval_result)
            
            # Update metrics
            self._update_neural_metrics()
            
            return self._format_output(storage_result)
            
        except Exception as e:
            self.log_error("Neural processing failed", e)
            return self._get_fallback_output()
    
    async def _store_experiences(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store new experiences in neural memory"""
        experiences = context.get('experiences', [])
        stored_count = 0
        
        for exp in experiences[-10:]:  # Process recent experiences
            if not isinstance(exp, dict):
                continue
            
            # Extract features
            features = self._extract_experience_features(exp, context)
            if features is None:
                continue
            
            # Encode experience
            encoded = await self._encode_experience(features)
            
            # Calculate importance
            importance = await self._calculate_importance(encoded, exp)
            
            # Store if important enough
            if importance > self.importance_threshold:
                await self._add_to_buffer(encoded, importance, exp)
                stored_count += 1
        
        return {
            'storage_performed': stored_count > 0,
            'memories_stored': stored_count,
            'buffer_size': len(self.buffer)
        }
    
    def _extract_experience_features(self, exp: Dict, context: Dict) -> Optional[np.ndarray]:
        """Extract features from experience"""
        try:
            features = []
            
            # Experience features
            if 'observation' in exp:
                obs = exp['observation']
                if isinstance(obs, np.ndarray):
                    features.extend(obs.flatten()[:self.embed_dim // 2].tolist())
                elif isinstance(obs, (list, tuple)):
                    features.extend(obs[:self.embed_dim // 2])
                else:
                    features.append(float(obs))
            
            # Reward signal
            if 'reward' in exp:
                features.append(float(exp['reward']))
            
            # Action taken
            if 'action' in exp:
                action = exp['action']
                if isinstance(action, (list, tuple, np.ndarray)):
                    features.extend(np.array(action).flatten()[:5].tolist())
                else:
                    features.append(float(action))
            
            # Context features
            market_context = context.get('market_context', {})
            features.append(float(market_context.get('volatility', 0.5)))
            
            # Pad or truncate to embed_dim
            if len(features) < self.embed_dim:
                features.extend([0.0] * (self.embed_dim - len(features)))
            else:
                features = features[:self.embed_dim]
            
            return np.array(features, dtype=np.float32)
            
        except Exception:
            return None
    
    async def _encode_experience(self, features: np.ndarray) -> torch.Tensor:
        """Encode experience using neural encoder"""
        try:
            with torch.no_grad():
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                encoded = self.memory_encoder(features_tensor)
                return encoded.squeeze(0)
        except Exception:
            return torch.zeros(self.embed_dim, dtype=torch.float32)
    
    async def _calculate_importance(self, encoded: torch.Tensor, exp: Dict) -> float:
        """Calculate importance score for experience"""
        try:
            with torch.no_grad():
                # Base importance from value network
                importance = self.value_head(encoded.unsqueeze(0))
                importance_score = float(importance.squeeze())
                
                # Adjust based on reward
                reward = exp.get('reward', 0.0) if isinstance(exp, dict) else 0.0
                if reward > 0:
                    importance_score *= 1.2
                elif reward < 0:
                    importance_score *= 0.8
                
                # Adjust based on novelty
                if len(self.buffer) > 0:
                    similarities = torch.cosine_similarity(
                        encoded.unsqueeze(0),
                        self.buffer,
                        dim=1
                    )
                    max_similarity = float(torch.max(similarities))
                    novelty = 1.0 - max_similarity
                    importance_score *= (0.5 + 0.5 * novelty)
                
                return float(np.clip(importance_score, 0.0, 1.0))
                
        except Exception:
            return 0.0
    
    async def _add_to_buffer(self, encoded: torch.Tensor, importance: float, exp: Dict) -> None:
        """Add memory to buffer"""
        try:
            # Add to buffer
            self.buffer = torch.cat([self.buffer, encoded.unsqueeze(0)], dim=0)
            self.importance_scores = torch.cat([
                self.importance_scores,
                torch.tensor([importance], dtype=torch.float32)
            ])
            
            # Add metadata
            self.memory_metadata.append({
                'timestamp': time.time(),
                'importance': importance,
                'type': exp.get('type', 'unknown')
            })
            
            # Prune if needed
            if len(self.buffer) > self.max_buffer_size:
                await self._prune_buffer()
            
            # Update metrics
            self.memories_stored += 1
            self._update_importance_metrics(importance)
            
        except Exception as e:
            self.log_error("Buffer addition failed", e)
    
    async def _prune_buffer(self) -> None:
        """Prune buffer to maintain size limits"""
        try:
            # Keep most important and most recent
            n_keep = int(self.max_buffer_size * 0.8)
            
            # Sort by importance
            sorted_indices = torch.argsort(self.importance_scores, descending=True)
            
            # Keep top half by importance
            important_indices = sorted_indices[:n_keep // 2]
            
            # Keep most recent
            recent_indices = torch.arange(
                max(0, len(self.buffer) - n_keep // 2),
                len(self.buffer)
            )
            
            # Combine and keep unique
            keep_indices = torch.unique(torch.cat([important_indices, recent_indices]))
            
            # Update buffer
            self.buffer = self.buffer[keep_indices]
            self.importance_scores = self.importance_scores[keep_indices]
            
            # Update metadata
            keep_set = set(keep_indices.tolist())
            self.memory_metadata = [
                m for i, m in enumerate(self.memory_metadata)
                if i in keep_set
            ]
            
        except Exception as e:
            self.log_error("Buffer pruning failed", e)
    
    async def _perform_retrieval(self, query: Any) -> Dict[str, Any]:
        """Perform memory retrieval using attention"""
        try:
            if len(self.buffer) == 0:
                return {'retrieval_performed': False, 'reason': 'empty_buffer'}
            
            # Process query
            query_tensor = self._process_query(query)
            if query_tensor is None:
                return {'retrieval_performed': False, 'reason': 'invalid_query'}
            
            # Perform attention-based retrieval
            with torch.no_grad():
                # Add batch dimension
                query_batch = query_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
                memory_batch = self.buffer.unsqueeze(0)  # [1, n_memories, embed_dim]
                
                # Apply attention
                attended, attention_weights = self.attention(
                    query_batch,
                    memory_batch,
                    memory_batch
                )
                
                # Get attention weights
                weights = attention_weights.squeeze().cpu().numpy()
            
            # Get top-k memories
            k = min(5, len(self.buffer))
            top_indices = np.argsort(weights)[-k:][::-1]
            
            # Prepare retrieved memories
            retrieved = []
            similarity_scores = []
            
            for idx in top_indices:
                memory = {
                    'embedding': self.buffer[idx].cpu().numpy().tolist(),
                    'importance': float(self.importance_scores[idx]),
                    'metadata': self.memory_metadata[idx] if idx < len(self.memory_metadata) else {},
                    'attention_weight': float(weights[idx])
                }
                retrieved.append(memory)
                similarity_scores.append(float(weights[idx]))
            
            # Update metrics
            self.memories_retrieved += 1
            self.retrieval_history.append({
                'timestamp': time.time(),
                'retrieved_count': len(retrieved),
                'avg_similarity': np.mean(similarity_scores)
            })
            
            return {
                'retrieval_performed': True,
                'retrieved_memories': retrieved,
                'similarity_scores': similarity_scores,
                'attention_weights': weights.tolist()
            }
            
        except Exception as e:
            self.log_error("Retrieval failed", e)
            return {'retrieval_performed': False, 'error': str(e)}
    
    def _process_query(self, query: Any) -> Optional[torch.Tensor]:
        """Process query into tensor"""
        try:
            if isinstance(query, torch.Tensor):
                query_tensor = query
            elif isinstance(query, np.ndarray):
                query_tensor = torch.tensor(query, dtype=torch.float32)
            elif isinstance(query, (list, tuple)):
                query_tensor = torch.tensor(query, dtype=torch.float32)
            else:
                return None
            
            # Ensure correct dimensions
            if query_tensor.dim() == 1:
                if len(query_tensor) < self.embed_dim:
                    # Pad with zeros
                    padding = torch.zeros(self.embed_dim - len(query_tensor))
                    query_tensor = torch.cat([query_tensor, padding])
                elif len(query_tensor) > self.embed_dim:
                    # Truncate
                    query_tensor = query_tensor[:self.embed_dim]
            
            return query_tensor
            
        except Exception:
            return None
    
    def _update_importance_metrics(self, importance: float) -> None:
        """Update importance-related metrics"""
        # Update running average
        if self.memories_stored == 1:
            self.avg_importance = importance
        else:
            self.avg_importance = (
                self.avg_importance * (self.memories_stored - 1) + importance
            ) / self.memories_stored
        
        # Track evolution
        self.importance_evolution.append({
            'timestamp': time.time(),
            'importance': importance,
            'avg': self.avg_importance
        })
    
    def _update_neural_metrics(self) -> None:
        """Update neural performance metrics"""
        # Calculate attention efficiency
        if len(self.buffer) > 0:
            high_importance = torch.sum(self.importance_scores > 0.7).item()
            self.attention_efficiency = high_importance / len(self.buffer)
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements"""
        # Calculate statistics
        if len(self.importance_scores) > 0:
            importance_stats = {
                'average_importance': float(torch.mean(self.importance_scores)),
                'max_importance': float(torch.max(self.importance_scores)),
                'min_importance': float(torch.min(self.importance_scores)),
                'std_importance': float(torch.std(self.importance_scores)),
                'total_scored': len(self.importance_scores)
            }
        else:
            importance_stats = {
                'average_importance': 0.0,
                'max_importance': 0.0,
                'min_importance': 0.0,
                'std_importance': 0.0,
                'total_scored': 0
            }
        
        return {
            'attention_retrieval': {
                'retrieved_count': len(result.get('retrieved_memories', [])),
                'similarity_scores': result.get('similarity_scores', []),
                'top_k': 5,
                'attention_heads': self.num_heads
            },
            'importance_scoring': importance_stats,
            'memory_embedding': {
                'embedding_dim': self.embed_dim,
                'total_embeddings': len(self.buffer),
                'importance_threshold': self.importance_threshold,
                'decay_rate': self.memory_decay
            },
            'neural_memory': {
                'buffer_size': len(self.buffer),
                'memory_utilization': len(self.buffer) / max(1, self.max_buffer_size),
                'average_importance': self.avg_importance,
                'neural_performance_score': self._calculate_performance_score(),
                'last_updated': time.time()
            }
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate neural performance score"""
        # Combine multiple factors
        utilization = len(self.buffer) / max(1, self.max_buffer_size)
        
        # Optimal utilization is around 0.7
        utilization_score = 1.0 - abs(utilization - 0.7) * 2
        
        # Importance quality
        importance_score = self.avg_importance
        
        # Attention efficiency
        efficiency_score = self.attention_efficiency
        
        # Combined score
        score = (
            utilization_score * 0.3 +
            importance_score * 0.4 +
            efficiency_score * 0.3
        ) * 100
        
        return float(np.clip(score, 0, 100))
    
    def _get_fallback_output(self) -> Dict[str, Any]:
        """Get fallback output for errors"""
        return {
            'attention_retrieval': {
                'retrieved_count': 0,
                'similarity_scores': [],
                'top_k': 5,
                'attention_heads': self.num_heads
            },
            'importance_scoring': {
                'average_importance': 0.0,
                'total_scored': 0
            },
            'memory_embedding': {
                'embedding_dim': self.embed_dim,
                'total_embeddings': 0,
                'importance_threshold': self.importance_threshold,
                'decay_rate': self.memory_decay
            },
            'neural_memory': {
                'buffer_size': 0,
                'memory_utilization': 0.0,
                'average_importance': 0.0,
                'neural_performance_score': 0.0,
                'last_updated': 0
            }
        }# modules/memory/components/playbook.py
"""
Playbook Memory Component
Context-aware pattern recognition and recall
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

from .base import MemoryComponent


class PlaybookComponent(MemoryComponent):
    """Playbook memory for pattern recall"""
    
    def _initialize_component(self) -> None:
        """Initialize playbook-specific resources"""
        # Configuration
        self.k_neighbors = self.config.k_neighbors
        self.similarity_threshold = self.config.similarity_threshold
        self.pattern_memory_size = self.config.pattern_memory_size
        self.max_entries = int(self.config.max_memory_size * 0.3)  # 30% for playbook
        
        # Memory storage
        self.features = []
        self.actions = []
        self.pnls = []
        self.contexts = []
        self.timestamps = []
        self.trade_metadata = []
        
        # Pattern tracking
        self.pattern_effectiveness = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_pnl': 0.0
        })
        self.context_patterns = defaultdict(int)
        
        # ML models
        self.knn_model = None
        self.knn_fitted = False
        
        # Metrics
        self.recall_history = deque(maxlen=100)
        self.memory_quality_score = 0.0
        self.prediction_accuracy = 0.0
        self.pattern_diversity = 0.0
        self.recall_efficiency = 0.0
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process playbook memory operations"""
        try:
            # Process new trades
            storage_result = await self._process_trades(context)
            
            # Perform recall if requested
            if context.get('recall_requested', False):
                recall_result = await self._perform_recall(context)
                storage_result.update(recall_result)
            
            # Update models if needed
            if len(self.features) >= self.k_neighbors and not self.knn_fitted:
                await self._fit_models()
            
            # Update analytics
            self._update_analytics()
            
            return self._format_output(storage_result)
            
        except Exception as e:
            self.log_error("Playbook processing failed", e)
            return self._get_fallback_output()
    
    async def _process_trades(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process and store trades"""
        trades = context.get('trades', [])
        market_context = context.get('market_context', {})
        prices = context.get('prices', {})
        
        trades_processed = 0
        
        for trade in trades[-10:]:  # Process recent trades
            if not isinstance(trade, dict) or 'pnl' not in trade:
                continue
            
            # Extract features
            features = self._extract_trade_features(trade, market_context, prices)
            action = self._extract_trade_action(trade)
            pnl = float(trade.get('pnl', 0.0))
            
            # Store trade
            await self._store_trade(features, action, pnl, market_context, trade)
            trades_processed += 1
        
        return {
            'trades_processed': trades_processed,
            'memory_size': len(self.features)
        }
    
    def _extract_trade_features(
        self, 
        trade: Dict, 
        market_context: Dict,
        prices: Dict
    ) -> np.ndarray:
        """Extract features from trade and context"""
        features = []
        
        # Market regime
        regime_map = {
            'trending': [1, 0, 0],
            'volatile': [0, 1, 0],
            'ranging': [0, 0, 1]
        }
        regime = market_context.get('regime', 'unknown')
        features.extend(regime_map.get(regime, [0.33, 0.33, 0.33]))
        
        # Volatility
        vol_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0}
        vol = market_context.get('volatility_level', 'medium')
        features.append(vol_map.get(vol, 0.5))
        
        # Risk context
        features.extend([
            float(market_context.get('drawdown_pct', 0.0)) / 100.0,
            float(market_context.get('exposure_pct', 0.0)) / 100.0,
            float(market_context.get('position_count', 0)) / 10.0
        ])
        
        # Session
        session_map = {
            'asian': [1, 0, 0],
            'european': [0, 1, 0],
            'american': [0, 0, 1]
        }
        session = market_context.get('session', 'unknown')
        features.extend(session_map.get(session, [0.25, 0.25, 0.25]))
        
        # Trade features
        features.extend([
            float(trade.get('size', 0.0)),
            float(trade.get('confidence', 0.5)),
            1.0 if trade.get('side') == 'buy' else -1.0 if trade.get('side') == 'sell' else 0.0
        ])
        
        # Price context
        symbol = trade.get('symbol', 'EUR/USD')
        if symbol in prices:
            current_price = float(prices[symbol])
            entry_price = float(trade.get('price', current_price))
            price_change = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            features.extend([current_price / 2.0, price_change])
        else:
            features.extend([0.5, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_trade_action(self, trade: Dict) -> np.ndarray:
        """Extract action from trade"""
        size = float(trade.get('size', 0.0))
        side = trade.get('side', 'hold')
        
        if side == 'buy':
            action = [size, 0.0]
        elif side == 'sell':
            action = [-size, 0.0]
        else:
            action = [0.0, 0.0]
        
        return np.array(action, dtype=np.float32)
    
    async def _store_trade(
        self,
        features: np.ndarray,
        action: np.ndarray,
        pnl: float,
        market_context: Dict,
        trade: Dict
    ) -> None:
        """Store trade in memory"""
        # Apply memory decay
        if self.pnls:
            self.pnls = [p * 0.98 for p in self.pnls]  # Decay factor
        
        # Manage memory size
        if len(self.features) >= self.max_entries:
            # Remove oldest
            self.features.pop(0)
            self.actions.pop(0)
            self.pnls.pop(0)
            self.contexts.pop(0)
            self.timestamps.pop(0)
            self.trade_metadata.pop(0)
        
        # Store new trade
        self.features.append(features)
        self.actions.append(action)
        self.pnls.append(pnl)
        self.contexts.append(market_context.copy())
        self.timestamps.append(time.time())
        
        metadata = {
            'timestamp': time.time(),
            'pnl': pnl,
            'regime': market_context.get('regime'),
            'volatility': market_context.get('volatility_level'),
            'session': market_context.get('session')
        }
        self.trade_metadata.append(metadata)
        
        # Update pattern tracking
        self._update_pattern_tracking(market_context, pnl)
    
    def _update_pattern_tracking(self, context: Dict, pnl: float) -> None:
        """Update pattern effectiveness tracking"""
        # Create pattern key
        regime = context.get('regime', 'unknown')
        vol = context.get('volatility_level', 'medium')
        session = context.get('session', 'unknown')
        pattern_key = f"{regime}_{vol}_{session}"
        
        # Update context patterns
        self.context_patterns[pattern_key] += 1
        
        # Update effectiveness
        pattern_data = self.pattern_effectiveness[pattern_key]
        if pnl > 0:
            pattern_data['wins'] += 1
        else:
            pattern_data['losses'] += 1
        pattern_data['total_pnl'] += pnl
    
    async def _fit_models(self) -> None:
        """Fit KNN model for recall"""
        try:
            if len(self.features) < self.k_neighbors:
                return
            
            # Prepare feature matrix
            X = np.vstack(self.features)
            
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit KNN
            self.knn_model = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, len(X)),
                metric='euclidean'
            )
            self.knn_model.fit(X_scaled)
            self.knn_fitted = True
            
            # Update quality score
            profitable = sum(1 for p in self.pnls if p > 0)
            self.memory_quality_score = (profitable / max(1, len(self.pnls))) * 100
            
        except Exception as e:
            self.log_error("Model fitting failed", e)
    
    async def _perform_recall(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform memory recall"""
        try:
            if not self.knn_fitted or len(self.features) == 0:
                return {
                    'recall_performed': False,
                    'reason': 'model_not_fitted'
                }
            
            # Get query features
            query_features = context.get('query_features')
            if query_features is None:
                # Use current context as query
                market_context = context.get('market_context', {})
                prices = context.get('prices', {})
                query_features = self._create_query_features(market_context, prices)
            
            # Normalize query
            query = np.array(query_features).reshape(1, -1)
            query_scaled = self.scaler.transform(query)
            
            # Find similar trades
            distances, indices = self.knn_model.kneighbors(query_scaled)
            
            indices = indices[0]
            distances = distances[0]
            
            # Extract similar trade data
            similar_pnls = [self.pnls[i] for i in indices]
            similar_actions = [self.actions[i] for i in indices]
            
            # Calculate predictions
            expected_pnl = np.mean(similar_pnls)
            confidence = np.exp(-np.mean(distances))
            
            # Weighted action recommendation
            weights = np.exp(-distances)
            weights = weights / np.sum(weights)
            recommended_action = np.average(similar_actions, axis=0, weights=weights)
            
            # Record recall
            self.recall_history.append({
                'timestamp': time.time(),
                'expected_pnl': expected_pnl,
                'confidence': confidence,
                'similar_trades': len(indices)
            })
            
            return {
                'recall_performed': True,
                'expected_pnl': float(expected_pnl),
                'confidence': float(confidence),
                'recommended_action': recommended_action.tolist(),
                'similar_trades': len(indices),
                'profitable_matches': sum(1 for pnl in similar_pnls if pnl > 0)
            }
            
        except Exception as e:
            self.log_error("Recall failed", e)
            return {'recall_performed': False, 'error': str(e)}
    
    def _create_query_features(self, market_context: Dict, prices: Dict) -> np.ndarray:
        """Create query features from current context"""
        # Create a dummy trade for feature extraction
        dummy_trade = {
            'size': 1.0,
            'confidence': 0.5,
            'side': 'hold',
            'symbol': 'EUR/USD',
            'price': 1.0
        }
        
        return self._extract_trade_features(dummy_trade, market_context, prices)
    
    def _update_analytics(self) -> None:
        """Update analytics metrics"""
        # Pattern diversity
        unique_patterns = len(self.pattern_effectiveness)
        self.pattern_diversity = min(1.0, unique_patterns / 20.0)
        
        # Recall efficiency
        if self.recall_history:
            recent_recalls = list(self.recall_history)[-10:]
            self.recall_efficiency = np.mean([r['confidence'] for r in recent_recalls])
        
        # Prediction accuracy
        if self.pattern_effectiveness:
            total_profitable = sum(p['wins'] for p in self.pattern_effectiveness.values())
            total_trades = sum(p['wins'] + p['losses'] for p in self.pattern_effectiveness.values())
            self.prediction_accuracy = total_profitable / max(total_trades, 1)
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements"""
        # Get top pattern
        top_pattern = None
        if self.pattern_effectiveness:
            top_pattern = max(
                self.pattern_effectiveness.items(),
                key=lambda x: x[1]['total_pnl']
            )[0]
        
        # Get last recall
        last_recall = self.recall_history[-1] if self.recall_history else None
        
        return {
            'memory_analytics': {
                'total_recalls': len(self.recall_history),
                'recent_performance': self.recall_efficiency,
                'memory_health': 'healthy' if self.memory_quality_score > 50 else 'warning',
                'circuit_breaker_state': 'CLOSED'
            },
            'pattern_memory': {
                'total_patterns': len(self.pattern_effectiveness),
                'pattern_effectiveness': dict(list(self.pattern_effectiveness.items())[:50]),
                'pattern_diversity': self.pattern_diversity,
                'top_pattern': top_pattern
            },
            'playbook_quality': {
                'memory_utilization': len(self.features) / max(1, self.max_entries),
                'quality_score': self.memory_quality_score,
                'models_fitted': self.knn_fitted,
                'adaptive_k': min(self.k_neighbors, len(self.features))
            },
            'playbook_recall': {
                'memory_entries': len(self.features),
                'patterns_identified': len(self.pattern_effectiveness),
                'recall_efficiency': self.recall_efficiency,
                'prediction_accuracy': self.prediction_accuracy,
                'last_recall': last_recall
            }
        }
    
    def _get_fallback_output(self) -> Dict[str, Any]:
        """Get fallback output for errors"""
        return {
            'memory_analytics': {
                'total_recalls': 0,
                'recent_performance': 0.0,
                'memory_health': 'unknown',
                'circuit_breaker_state': 'CLOSED'
            },
            'pattern_memory': {
                'total_patterns': 0,
                'pattern_effectiveness': {},
                'pattern_diversity': 0.0,
                'top_pattern': None
            },
            'playbook_quality': {
                'memory_utilization': 0.0,
                'quality_score': 0.0,
                'models_fitted': False,
                'adaptive_k': self.k_neighbors
            },
            'playbook_recall': {
                'memory_entries': 0,
                'patterns_identified': 0,
                'recall_efficiency': 0.0,
                'prediction_accuracy': 0.0,
                'last_recall': None
            }
        }# modules/memory/components/replay.py
"""
Historical Replay Component
Analyzes sequences and patterns for learning optimization
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
import numpy as np
import time

from .base import MemoryComponent


class ReplayComponent(MemoryComponent):
    """Historical replay analysis component"""
    
    def _initialize_component(self) -> None:
        """Initialize replay-specific resources"""
        # Replay configuration
        self.replay_interval = self.config.replay_interval
        self.replay_decay = self.config.replay_decay
        self.sequence_len = self.config.sequence_len
        self.profit_threshold = self.config.replay_profit_threshold
        
        # State
        self.episode_buffer = deque(maxlen=50)
        self.profitable_sequences = []
        self.sequence_patterns = defaultdict(lambda: {
            'count': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0, 'last_seen': 0
        })
        self.current_sequence = []
        
        # Metrics
        self.replay_bonus = 0.0
        self.best_sequence_pnl = 0.0
        self.sequences_analyzed = 0
        self.patterns_identified = 0
        
        # Quality tracking
        self.sequence_quality_scores = deque(maxlen=100)
        self.pattern_evolution = deque(maxlen=100)
        self.learning_effectiveness = deque(maxlen=200)
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process historical replay analysis"""
        try:
            # Extract sequence data
            sequence_data = self._extract_sequence_data(context)
            
            # Process current sequence
            sequence_result = self._process_trading_sequence(sequence_data)
            
            # Check for episode completion
            if context.get('episode_data', {}).get('completed', False):
                episode_result = self._analyze_episode_patterns(sequence_data)
                sequence_result.update(episode_result)
            
            # Generate replay recommendations
            episode = context.get('episode', 0)
            if self._should_replay(episode):
                replay_result = self._generate_replay_recommendations()
                sequence_result.update(replay_result)
            
            # Update metrics
            self._update_metrics(sequence_result)
            
            return self._format_output(sequence_result)
            
        except Exception as e:
            self.log_error("Replay processing failed", e)
            return self._get_fallback_output()
    
    def _extract_sequence_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sequence data from context"""
        trades = context.get('trades', [])
        actions = context.get('actions', [])
        features = context.get('features')
        episode_data = context.get('episode_data', {})
        
        # Get current action
        current_action = None
        if actions and len(actions) > 0:
            current_action = actions[-1] if isinstance(actions[-1], (list, np.ndarray)) else [0.0, 0.0]
        
        return {
            'trades': trades,
            'features': features,
            'current_action': current_action,
            'episode_data': episode_data,
            'timestamp': time.time()
        }
    
    def _process_trading_sequence(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process current trading sequence"""
        # Add current step to sequence
        if sequence_data['current_action'] is not None:
            step = {
                'action': sequence_data['current_action'],
                'timestamp': sequence_data['timestamp'],
                'features': sequence_data['features']
            }
            self.current_sequence.append(step)
        
        # Trim sequence to max length
        if len(self.current_sequence) > self.sequence_len:
            self.current_sequence = self.current_sequence[-self.sequence_len:]
        
        # Calculate sequence quality
        quality = self._calculate_sequence_quality(self.current_sequence)
        self.sequence_quality_scores.append(quality)
        
        self.sequences_analyzed += 1
        
        return {
            'current_sequence_length': len(self.current_sequence),
            'sequence_quality': quality,
            'sequences_processed': self.sequences_analyzed
        }
    
    def _calculate_sequence_quality(self, sequence: List[Dict]) -> float:
        """Calculate quality score of a trading sequence"""
        if len(sequence) < 2:
            return 0.0
        
        try:
            # Action consistency
            actions = [step.get('action', [0, 0]) for step in sequence]
            magnitudes = [np.linalg.norm(a) for a in actions]
            action_variance = np.var(magnitudes) if magnitudes else 0.0
            consistency_score = max(0.0, 1.0 - action_variance)
            
            # Temporal regularity
            timestamps = [step.get('timestamp', 0) for step in sequence]
            if len(timestamps) >= 2:
                time_diffs = np.diff(timestamps)
                temporal_score = max(0.0, 1.0 - np.std(time_diffs) / 100.0)
            else:
                temporal_score = 1.0
            
            # Combined quality
            quality = consistency_score * 0.6 + temporal_score * 0.4
            return float(np.clip(quality, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _analyze_episode_patterns(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in completed episode"""
        episode_data = sequence_data['episode_data']
        episode_pnl = episode_data.get('pnl', 0.0)
        
        # Store episode
        episode_record = {
            'sequence': list(self.current_sequence),
            'pnl': episode_pnl,
            'timestamp': time.time()
        }
        self.episode_buffer.append(episode_record)
        
        # Process profitable sequence
        if episode_pnl > self.profit_threshold:
            self._process_profitable_sequence(episode_record)
        
        # Extract and record pattern
        pattern = self._extract_sequence_pattern(self.current_sequence)
        if pattern:
            self._update_pattern_data(pattern, episode_pnl)
        
        # Reset sequence
        self.current_sequence = []
        
        return {
            'episode_analyzed': True,
            'episode_pnl': episode_pnl,
            'pattern_extracted': pattern is not None,
            'profitable_sequence': episode_pnl > self.profit_threshold
        }
    
    def _extract_sequence_pattern(self, sequence: List[Dict]) -> Optional[str]:
        """Extract pattern from sequence"""
        if len(sequence) < 3:
            return None
        
        try:
            pattern_elements = []
            
            for step in sequence:
                action = step.get('action', [0, 0])
                if isinstance(action, (list, np.ndarray)) and len(action) > 0:
                    # Discretize action
                    if action[0] > 0.5:
                        pattern_elements.append('L')  # Long
                    elif action[0] < -0.5:
                        pattern_elements.append('S')  # Short
                    else:
                        pattern_elements.append('H')  # Hold
            
            return ''.join(pattern_elements) if pattern_elements else None
            
        except Exception:
            return None
    
    def _update_pattern_data(self, pattern: str, pnl: float) -> None:
        """Update pattern tracking data"""
        data = self.sequence_patterns[pattern]
        data['count'] += 1
        data['total_pnl'] += pnl
        data['avg_pnl'] = data['total_pnl'] / data['count']
        data['last_seen'] = time.time()
        
        self.patterns_identified = len(self.sequence_patterns)
    
    def _process_profitable_sequence(self, episode: Dict[str, Any]) -> None:
        """Process and store profitable sequence"""
        self.profitable_sequences.append(episode)
        
        # Keep only best sequences
        if len(self.profitable_sequences) > 100:
            self.profitable_sequences.sort(key=lambda x: x['pnl'], reverse=True)
            self.profitable_sequences = self.profitable_sequences[:100]
        
        # Update best PnL
        if episode['pnl'] > self.best_sequence_pnl:
            self.best_sequence_pnl = episode['pnl']
    
    def _should_replay(self, episode: int) -> bool:
        """Check if replay should be triggered"""
        return (
            self.replay_interval > 0 and
            episode > 0 and
            episode % self.replay_interval == 0
        )
    
    def _generate_replay_recommendations(self) -> Dict[str, Any]:
        """Generate replay recommendations"""
        if not self.profitable_sequences:
            return {
                'replay_recommended': False,
                'replay_bonus': 0.0
            }
        
        # Find best sequence
        best_sequence = max(self.profitable_sequences, key=lambda x: x['pnl'])
        
        # Calculate time-decayed bonus
        hours_since = (time.time() - best_sequence['timestamp']) / 3600
        decay_factor = self.replay_decay ** hours_since
        replay_bonus = (best_sequence['pnl'] / 100.0) * decay_factor
        
        self.replay_bonus = replay_bonus
        
        # Find best pattern
        best_pattern = None
        if self.sequence_patterns:
            best_pattern = max(
                self.sequence_patterns.items(),
                key=lambda x: x[1]['avg_pnl'] * x[1]['count']
            )[0]
        
        return {
            'replay_recommended': True,
            'replay_bonus': replay_bonus,
            'best_sequence_pnl': best_sequence['pnl'],
            'best_pattern': best_pattern,
            'total_patterns': len(self.sequence_patterns)
        }
    
    def _update_metrics(self, result: Dict[str, Any]) -> None:
        """Update component metrics"""
        self._last_process = time.time()
        self._process_count += 1
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements"""
        # Calculate averages
        avg_quality = np.mean(list(self.sequence_quality_scores)) if self.sequence_quality_scores else 0.0
        
        return {
            'learning_progress': {
                'episodes_processed': len(self.episode_buffer),
                'profitable_sequences': len(self.profitable_sequences),
                'patterns_learned': self.patterns_identified,
                'learning_rate': len(self.profitable_sequences) / max(len(self.episode_buffer), 1)
            },
            'pattern_analysis': {
                'total_patterns': len(self.sequence_patterns),
                'profitable_patterns': sum(1 for p in self.sequence_patterns.values() if p['avg_pnl'] > 0),
                'best_pattern': result.get('best_pattern'),
                'pattern_confidence': self._calculate_pattern_confidence()
            },
            'replay_sequences': {
                'total_sequences': len(self.profitable_sequences),
                'best_sequence_pnl': self.best_sequence_pnl,
                'replay_bonus': self.replay_bonus,
                'sequences_analyzed': self.sequences_analyzed
            },
            'sequence_quality': {
                'current_quality': result.get('sequence_quality', 0.0),
                'average_quality': avg_quality,
                'quality_trend': self._calculate_quality_trend(),
                'episodes_processed': len(self.episode_buffer)
            }
        }
    
    def _calculate_pattern_confidence(self) -> float:
        """Calculate overall pattern confidence"""
        if not self.sequence_patterns:
            return 0.0
        
        confidences = []
        for pattern_data in self.sequence_patterns.values():
            if pattern_data['count'] > 0:
                # Confidence based on consistency and frequency
                consistency = 1.0 - abs(pattern_data['avg_pnl']) / max(abs(pattern_data['total_pnl']), 1.0)
                frequency = min(1.0, pattern_data['count'] / 10.0)
                confidences.append(consistency * frequency)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _calculate_quality_trend(self) -> str:
        """Calculate sequence quality trend"""
        if len(self.sequence_quality_scores) < 10:
            return 'insufficient_data'
        
        recent = np.mean(list(self.sequence_quality_scores)[-5:])
        older = np.mean(list(self.sequence_quality_scores)[-10:-5])
        
        if recent > older * 1.1:
            return 'improving'
        elif recent < older * 0.9:
            return 'declining'
        else:
            return 'stable'
    
    def _get_fallback_output(self) -> Dict[str, Any]:
        """Get fallback output for errors"""
        return {
            'learning_progress': {'episodes_processed': 0, 'patterns_learned': 0},
            'pattern_analysis': {'total_patterns': 0, 'profitable_patterns': 0},
            'replay_sequences': {'total_sequences': 0, 'best_sequence_pnl': 0.0},
            'sequence_quality': {'current_quality': 0.0, 'average_quality': 0.0}
        }# modules/memory/shared/feature_extractor.py
"""
Unified Feature Extractor
Standardized feature extraction for all memory components
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from collections import OrderedDict


class UnifiedFeatureExtractor:
    """
    Unified feature extraction for consistent representations
    
    Features:
    - Market context encoding
    - Trade feature extraction
    - Observation processing
    - Feature normalization
    - Caching for efficiency
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        # Feature dimensions
        self.market_features_dim = 10
        self.trade_features_dim = 10
        self.observation_features_dim = 20
        self.total_dim = 40
        
        # Encoding maps
        self.regime_map = {
            'trending': np.array([1.0, 0.0, 0.0]),
            'volatile': np.array([0.0, 1.0, 0.0]),
            'ranging': np.array([0.0, 0.0, 1.0]),
            'unknown': np.array([0.33, 0.33, 0.33])
        }
        
        self.session_map = {
            'asian': np.array([1.0, 0.0, 0.0, 0.0]),
            'european': np.array([0.0, 1.0, 0.0, 0.0]),
            'us': np.array([0.0, 0.0, 1.0, 0.0]),
            'american': np.array([0.0, 0.0, 1.0, 0.0]),
            'closed': np.array([0.0, 0.0, 0.0, 1.0]),
            'unknown': np.array([0.25, 0.25, 0.25, 0.25])
        }
        
        self.volatility_map = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'extreme': 1.0
        }
        
        # Feature cache
        self._cache = {}
        self._cache_size = 1000
    
    def extract(
        self,
        observations: Optional[Union[np.ndarray, List, Dict]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        trade: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract features from various inputs
        
        Args:
            observations: Raw observations
            market_context: Market context dictionary
            trade: Trade information
            
        Returns:
            Feature vector
        """
        features = []
        
        # Extract market features
        if market_context is not None:
            market_features = self.extract_market_features(market_context)
            features.append(market_features)
        
        # Extract trade features
        if trade is not None:
            trade_features = self.extract_trade_features(trade, market_context or {})
            features.append(trade_features)
        
        # Extract observation features
        if observations is not None:
            obs_features = self.extract_observation_features(observations)
            features.append(obs_features)
        
        if not features:
            # Return default features
            return np.zeros(self.total_dim, dtype=np.float32)
        
        # Concatenate all features
        combined = np.concatenate(features)
        
        # Ensure consistent dimension
        if len(combined) < self.total_dim:
            combined = np.pad(combined, (0, self.total_dim - len(combined)), 'constant')
        elif len(combined) > self.total_dim:
            combined = combined[:self.total_dim]
        
        return combined.astype(np.float32)
    
    def extract_market_features(self, market_context: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from market context
        
        Args:
            market_context: Market context dictionary
            
        Returns:
            Market feature vector
        """
        features = []
        
        # Market regime (3 features)
        regime = market_context.get('regime', 'unknown')
        features.extend(self.regime_map.get(str(regime).lower(), self.regime_map['unknown']))
        
        # Volatility (1 feature)
        volatility = market_context.get('volatility', 0.5)
        if isinstance(volatility, str):
            volatility = self.volatility_map.get(volatility.lower(), 0.5)
        elif isinstance(volatility, dict):
            # Take first numeric value
            volatility = next((v for v in volatility.values() if isinstance(v, (int, float))), 0.5)
        features.append(float(volatility))
        
        # Session (4 features)
        session = market_context.get('session', 'unknown')
        features.extend(self.session_map.get(str(session).lower(), self.session_map['unknown']))
        
        # Risk metrics (2 features)
        drawdown = float(market_context.get('drawdown_pct', 0.0)) / 100.0
        exposure = float(market_context.get('exposure_pct', 0.0)) / 100.0
        features.extend([drawdown, exposure])
        
        # Pad to fixed dimension
        while len(features) < self.market_features_dim:
            features.append(0.0)
        
        return np.array(features[:self.market_features_dim], dtype=np.float32)
    
    def extract_trade_features(
        self,
        trade: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract features from trade
        
        Args:
            trade: Trade dictionary
            market_context: Optional market context
            
        Returns:
            Trade feature vector
        """
        features = []
        
        # Trade characteristics (5 features)
        features.append(float(trade.get('size', 0.0)))
        features.append(float(trade.get('confidence', 0.5)))
        features.append(float(trade.get('volume', 1.0)))
        features.append(float(trade.get('duration', 1.0)))
        
        # Side encoding
        side = str(trade.get('side', 'hold')).lower()
        if side == 'buy':
            features.append(1.0)
        elif side == 'sell':
            features.append(-1.0)
        else:
            features.append(0.0)
        
        # PnL and risk (2 features)
        features.append(float(trade.get('pnl', 0.0)) / 100.0)  # Normalize
        features.append(float(trade.get('risk', 0.0)))
        
        # Price movement (2 features)
        entry_price = float(trade.get('entry_price', 1.0))
        exit_price = float(trade.get('exit_price', entry_price))
        price_change = (exit_price - entry_price) / entry_price if entry_price != 0 else 0.0
        features.extend([entry_price, price_change])
        
        # Time of day (1 feature) - normalized hour
        timestamp = trade.get('timestamp', 0)
        if timestamp:
            hour = datetime.fromtimestamp(timestamp).hour
            features.append(hour / 24.0)
        else:
            features.append(0.5)
        
        # Pad to fixed dimension
        while len(features) < self.trade_features_dim:
            features.append(0.0)
        
        return np.array(features[:self.trade_features_dim], dtype=np.float32)
    
    def extract_observation_features(
        self,
        observations: Union[np.ndarray, List, Dict]
    ) -> np.ndarray:
        """
        Extract features from observations
        
        Args:
            observations: Raw observations (various formats)
            
        Returns:
            Observation feature vector
        """
        # Handle different observation formats
        if isinstance(observations, dict):
            # Extract values from dict
            features = []
            for key in sorted(observations.keys()):
                value = observations[key]
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, (list, tuple, np.ndarray)):
                    features.extend(np.array(value).flatten()[:5].tolist())
            
            obs_array = np.array(features, dtype=np.float32)
        
        elif isinstance(observations, (list, tuple)):
            obs_array = np.array(observations, dtype=np.float32).flatten()
        
        elif isinstance(observations, np.ndarray):
            obs_array = observations.flatten().astype(np.float32)
        
        else:
            # Single value
            obs_array = np.array([float(observations)], dtype=np.float32)
        
        # Ensure fixed dimension
        if len(obs_array) < self.observation_features_dim:
            obs_array = np.pad(
                obs_array,
                (0, self.observation_features_dim - len(obs_array)),
                'constant'
            )
        elif len(obs_array) > self.observation_features_dim:
            obs_array = obs_array[:self.observation_features_dim]
        
        return obs_array
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature vector
        
        Args:
            features: Raw feature vector
            
        Returns:
            Normalized features
        """
        # Z-score normalization with clipping
        mean = np.mean(features)
        std = np.std(features)
        
        if std > 0:
            normalized = (features - mean) / std
            # Clip to prevent extreme values
            normalized = np.clip(normalized, -3, 3)
        else:
            normalized = features - mean
        
        return normalized.astype(np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        names = []
        
        # Market features
        names.extend(['regime_trending', 'regime_volatile', 'regime_ranging'])
        names.append('volatility')
        names.extend(['session_asian', 'session_european', 'session_us', 'session_closed'])
        names.extend(['drawdown_pct', 'exposure_pct'])
        
        # Trade features
        names.extend(['trade_size', 'confidence', 'volume', 'duration', 'side'])
        names.extend(['pnl_normalized', 'risk', 'entry_price', 'price_change', 'hour_normalized'])
        
        # Observation features
        for i in range(self.observation_features_dim):
            names.append(f'obs_{i}')
        
        return names[:self.total_dim]
    
    def get_feature_importance(
        self,
        features: np.ndarray,
        target: float
    ) -> Dict[str, float]:
        """
        Calculate feature importance (simplified)
        
        Args:
            features: Feature vector
            target: Target value (e.g., PnL)
            
        Returns:
            Feature importance dictionary
        """
        names = self.get_feature_names()
        
        # Simple correlation-based importance
        importance = {}
        for i, name in enumerate(names):
            if i < len(features):
                # Importance = feature value * sign(target)
                importance[name] = float(features[i] * np.sign(target))
        
        return importance# modules/memory/shared/memory_store.py
"""
Unified Memory Store
Central storage backend for all memory components with efficient indexing and retrieval
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from collections import deque
import numpy as np
import heapq
import time
import pickle
import threading
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """Single memory entry with metadata"""
    id: str
    timestamp: float
    features: np.ndarray
    action: np.ndarray
    pnl: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For heap operations"""
        return self.importance < other.importance


class UnifiedMemoryStore:
    """
    Central memory storage with efficient operations
    
    Features:
    - Fast insertion and retrieval
    - Similarity search using multiple metrics
    - Automatic memory management
    - Thread-safe operations
    - Persistence support
    """
    
    def __init__(self, max_size: int = 10000, batch_size: int = 32):
        """
        Initialize memory store
        
        Args:
            max_size: Maximum number of memories to store
            batch_size: Default batch size for operations
        """
        self.max_size = max_size
        self.batch_size = batch_size
        
        # Primary storage
        self.memories: Dict[str, MemoryEntry] = {}
        self.memory_list: List[MemoryEntry] = []
        
        # Indices for fast lookup
        self.timestamp_index: List[Tuple[float, str]] = []
        self.pnl_index: List[Tuple[float, str]] = []
        self.importance_heap: List[MemoryEntry] = []
        
        # Feature matrix for similarity search
        self.feature_matrix: Optional[np.ndarray] = None
        self.feature_ids: List[str] = []
        self._feature_matrix_dirty = False
        
        # Statistics
        self.total_stored = 0
        self.total_retrieved = 0
        self.total_pruned = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Memory pressure management
        self.pressure_threshold = 0.9
        self.cleanup_ratio = 0.8
    
    def add(self, entry: Dict[str, Any]) -> str:
        """
        Add single memory entry
        
        Args:
            entry: Memory entry dictionary
            
        Returns:
            Entry ID
        """
        with self._lock:
            # Generate ID
            entry_id = f"mem_{self.total_stored}_{int(time.time() * 1000000)}"
            
            # Create memory entry
            memory = MemoryEntry(
                id=entry_id,
                timestamp=entry.get('timestamp', time.time()),
                features=np.array(entry.get('features', []), dtype=np.float32),
                action=np.array(entry.get('action', [0, 0]), dtype=np.float32),
                pnl=float(entry.get('pnl', 0.0)),
                context=entry.get('context', {}),
                metadata=entry.get('metadata', {}),
                importance=float(entry.get('importance', 0.5))
            )
            
            # Store in primary storage
            self.memories[entry_id] = memory
            self.memory_list.append(memory)
            
            # Update indices
            self.timestamp_index.append((memory.timestamp, entry_id))
            self.pnl_index.append((memory.pnl, entry_id))
            heapq.heappush(self.importance_heap, memory)
            
            # Mark feature matrix as dirty
            self._feature_matrix_dirty = True
            
            # Update statistics
            self.total_stored += 1
            
            # Check memory pressure
            if self.size() > self.max_size:
                self._auto_prune()
            
            return entry_id
    
    async def add_batch(self, entries: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple memory entries
        
        Args:
            entries: List of memory entries
            
        Returns:
            List of entry IDs
        """
        ids = []
        
        with self._lock:
            for entry in entries:
                entry_id = self.add(entry)
                ids.append(entry_id)
        
        return ids
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get memory by ID
        
        Args:
            entry_id: Memory entry ID
            
        Returns:
            Memory entry or None
        """
        with self._lock:
            memory = self.memories.get(entry_id)
            if memory:
                memory.access_count += 1
                memory.last_accessed = time.time()
                self.total_retrieved += 1
            return memory
    
    def query(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_fn: Optional[Callable[[MemoryEntry], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query memories by similarity
        
        Args:
            query_vector: Query vector for similarity search
            k: Number of results to return
            filter_fn: Optional filter function
            
        Returns:
            List of similar memories
        """
        with self._lock:
            # Update feature matrix if needed
            if self._feature_matrix_dirty:
                self._rebuild_feature_matrix()
            
            if self.feature_matrix is None or len(self.feature_matrix) == 0:
                return []
            
            # Normalize query vector
            query = np.array(query_vector, dtype=np.float32)
            if query.ndim == 1:
                query = query.reshape(1, -1)
            
            # Ensure dimensions match
            if query.shape[1] != self.feature_matrix.shape[1]:
                # Pad or truncate
                target_dim = self.feature_matrix.shape[1]
                if query.shape[1] < target_dim:
                    padding = np.zeros((1, target_dim - query.shape[1]))
                    query = np.concatenate([query, padding], axis=1)
                else:
                    query = query[:, :target_dim]
            
            # Calculate similarities
            similarities = self._calculate_similarities(query, self.feature_matrix)
            
            # Get top-k indices
            top_indices = np.argsort(similarities[0])[-k:][::-1]
            
            # Filter and prepare results
            results = []
            for idx in top_indices:
                if idx < len(self.feature_ids):
                    memory = self.memories.get(self.feature_ids[idx])
                    if memory:
                        # Apply filter if provided
                        if filter_fn and not filter_fn(memory):
                            continue
                        
                        # Update access info
                        memory.access_count += 1
                        memory.last_accessed = time.time()
                        
                        # Convert to dict
                        results.append(self._memory_to_dict(memory))
            
            self.total_retrieved += len(results)
            return results
    
    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """
        Get n most recent memories
        
        Args:
            n: Number of memories to retrieve
            
        Returns:
            List of recent memories
        """
        with self._lock:
            # Sort by timestamp
            sorted_timestamps = sorted(self.timestamp_index, reverse=True)[:n]
            
            results = []
            for timestamp, entry_id in sorted_timestamps:
                memory = self.memories.get(entry_id)
                if memory:
                    results.append(self._memory_to_dict(memory))
            
            return results
    
    def get_by_filter(
        self,
        filter_fn: Callable[[Dict[str, Any]], bool],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get memories matching filter
        
        Args:
            filter_fn: Filter function
            limit: Optional limit on results
            
        Returns:
            List of matching memories
        """
        with self._lock:
            results = []
            count = 0
            
            for memory in self.memory_list:
                memory_dict = self._memory_to_dict(memory)
                if filter_fn(memory_dict):
                    results.append(memory_dict)
                    count += 1
                    
                    if limit and count >= limit:
                        break
            
            return results
    
    def update_importance(self, entry_id: str, importance: float) -> bool:
        """
        Update memory importance
        
        Args:
            entry_id: Memory entry ID
            importance: New importance value
            
        Returns:
            Success flag
        """
        with self._lock:
            memory = self.memories.get(entry_id)
            if memory:
                memory.importance = float(np.clip(importance, 0.0, 1.0))
                # Re-heapify importance heap
                heapq.heapify(self.importance_heap)
                return True
            return False
    
    def cleanup(self, keep_ratio: float = 0.8) -> int:
        """
        Clean up memory store
        
        Args:
            keep_ratio: Ratio of memories to keep
            
        Returns:
            Number of memories removed
        """
        with self._lock:
            target_size = int(self.max_size * keep_ratio)
            if self.size() <= target_size:
                return 0
            
            to_remove = self.size() - target_size
            
            # Score memories for removal
            scores = []
            for memory in self.memory_list:
                # Lower score = more likely to remove
                score = self._calculate_retention_score(memory)
                scores.append((score, memory.id))
            
            # Sort and remove lowest scoring
            scores.sort()
            removed = 0
            
            for score, entry_id in scores[:to_remove]:
                if self._remove_memory(entry_id):
                    removed += 1
            
            self.total_pruned += removed
            return removed
    
    def size(self) -> int:
        """Get current number of memories"""
        return len(self.memories)
    
    def utilization(self) -> float:
        """Get memory utilization ratio"""
        return self.size() / max(1, self.max_size)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics"""
        with self._lock:
            return {
                'size': self.size(),
                'max_size': self.max_size,
                'utilization': self.utilization(),
                'total_stored': self.total_stored,
                'total_retrieved': self.total_retrieved,
                'total_pruned': self.total_pruned,
                'avg_importance': np.mean([m.importance for m in self.memory_list]) if self.memory_list else 0.0,
                'avg_access_count': np.mean([m.access_count for m in self.memory_list]) if self.memory_list else 0.0
            }
    
    def save(self, filepath: str) -> bool:
        """
        Save memory store to disk
        
        Args:
            filepath: Path to save file
            
        Returns:
            Success flag
        """
        try:
            with self._lock:
                state = {
                    'memories': self.memories,
                    'memory_list': self.memory_list,
                    'timestamp_index': self.timestamp_index,
                    'pnl_index': self.pnl_index,
                    'total_stored': self.total_stored,
                    'total_retrieved': self.total_retrieved,
                    'total_pruned': self.total_pruned
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(state, f)
                
                return True
        except Exception:
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Load memory store from disk
        
        Args:
            filepath: Path to save file
            
        Returns:
            Success flag
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            with self._lock:
                self.memories = state['memories']
                self.memory_list = state['memory_list']
                self.timestamp_index = state['timestamp_index']
                self.pnl_index = state['pnl_index']
                self.total_stored = state['total_stored']
                self.total_retrieved = state['total_retrieved']
                self.total_pruned = state['total_pruned']
                
                # Rebuild heap
                self.importance_heap = list(self.memory_list)
                heapq.heapify(self.importance_heap)
                
                # Mark feature matrix as dirty
                self._feature_matrix_dirty = True
            
            return True
        except Exception:
            return False
    
    # Private methods
    
    def _rebuild_feature_matrix(self) -> None:
        """Rebuild feature matrix for similarity search"""
        if not self.memory_list:
            self.feature_matrix = None
            self.feature_ids = []
            return
        
        # Extract features and IDs
        features = []
        ids = []
        
        for memory in self.memory_list:
            if memory.features is not None and len(memory.features) > 0:
                features.append(memory.features)
                ids.append(memory.id)
        
        if features:
            # Ensure all features have same dimension
            max_dim = max(len(f) for f in features)
            padded_features = []
            
            for f in features:
                if len(f) < max_dim:
                    padded = np.pad(f, (0, max_dim - len(f)), 'constant')
                    padded_features.append(padded)
                else:
                    padded_features.append(f)
            
            self.feature_matrix = np.vstack(padded_features)
            self.feature_ids = ids
        else:
            self.feature_matrix = None
            self.feature_ids = []
        
        self._feature_matrix_dirty = False
    
    def _calculate_similarities(
        self,
        query: np.ndarray,
        features: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarities"""
        # Normalize vectors
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarity
        similarities = np.dot(query_norm, features_norm.T)
        return similarities
    
    def _calculate_retention_score(self, memory: MemoryEntry) -> float:
        """Calculate retention score for memory"""
        # Higher score = more likely to keep
        
        # Importance factor
        importance_score = memory.importance
        
        # Recency factor
        age = time.time() - memory.timestamp
        recency_score = np.exp(-age / 3600)  # Decay over hours
        
        # Access frequency factor
        access_score = np.log1p(memory.access_count) / 10.0
        
        # PnL factor (keep extreme values)
        pnl_score = abs(memory.pnl) / 100.0
        
        # Combined score
        score = (
            importance_score * 0.4 +
            recency_score * 0.3 +
            access_score * 0.2 +
            pnl_score * 0.1
        )
        
        return float(score)
    
    def _remove_memory(self, entry_id: str) -> bool:
        """Remove memory by ID"""
        memory = self.memories.pop(entry_id, None)
        if memory:
            # Remove from list
            self.memory_list = [m for m in self.memory_list if m.id != entry_id]
            
            # Remove from indices
            self.timestamp_index = [(t, id) for t, id in self.timestamp_index if id != entry_id]
            self.pnl_index = [(p, id) for p, id in self.pnl_index if id != entry_id]
            self.importance_heap = [m for m in self.importance_heap if m.id != entry_id]
            heapq.heapify(self.importance_heap)
            
            # Mark feature matrix as dirty
            self._feature_matrix_dirty = True
            
            return True
        return False
    
    def _auto_prune(self) -> None:
        """Automatically prune when over capacity"""
        if self.utilization() > self.pressure_threshold:
            self.cleanup(self.cleanup_ratio)
    
    def _memory_to_dict(self, memory: MemoryEntry) -> Dict[str, Any]:
        """Convert memory entry to dictionary"""
        return {
            'id': memory.id,
            'timestamp': memory.timestamp,
            'features': memory.features.tolist() if memory.features is not None else [],
            'action': memory.action.tolist() if memory.action is not None else [],
            'pnl': memory.pnl,
            'context': memory.context,
            'metadata': memory.metadata,
            'importance': memory.importance,
            'access_count': memory.access_count,
            'last_accessed': memory.last_accessed
        }# modules/memory/shared/pattern_detector.py
"""
Unified Pattern Detector
Advanced pattern detection and mining for memory components
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
import hashlib


@dataclass
class Pattern:
    """Pattern representation"""
    id: str
    sequence: List[str]
    frequency: int
    confidence: float
    avg_outcome: float
    metadata: Dict[str, Any]
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id


class UnifiedPatternDetector:
    """
    Unified pattern detection using multiple algorithms
    
    Features:
    - Sequential pattern mining
    - Motif discovery
    - Anomaly detection
    - Pattern clustering
    - Confidence scoring
    """
    
    def __init__(self):
        """Initialize pattern detector"""
        # Pattern storage
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Configuration
        self.min_support = 2
        self.min_confidence = 0.6
        self.max_pattern_length = 10
        
        # Statistics
        self.total_sequences_processed = 0
        self.total_patterns_found = 0
    
    def detect_patterns(
        self,
        sequences: List[List[Any]],
        outcomes: Optional[List[float]] = None
    ) -> List[Pattern]:
        """
        Detect patterns in sequences
        
        Args:
            sequences: List of sequences
            outcomes: Optional outcomes for each sequence
            
        Returns:
            List of detected patterns
        """
        if not sequences:
            return []
        
        # Convert sequences to string representation
        string_sequences = self._convert_to_strings(sequences)
        
        # Mine frequent patterns
        frequent_patterns = self._mine_frequent_patterns(string_sequences)
        
        # Calculate pattern metrics
        patterns = []
        for pattern_seq, support in frequent_patterns.items():
            if support >= self.min_support:
                pattern = self._create_pattern(
                    pattern_seq,
                    support,
                    string_sequences,
                    outcomes
                )
                if pattern.confidence >= self.min_confidence:
                    patterns.append(pattern)
                    self._store_pattern(pattern)
        
        self.total_sequences_processed += len(sequences)
        self.total_patterns_found += len(patterns)
        
        return patterns
    
    def find_motifs(
        self,
        sequence: List[Any],
        window_size: int = 3
    ) -> List[Tuple[int, List[Any]]]:
        """
        Find recurring motifs in a sequence
        
        Args:
            sequence: Input sequence
            window_size: Motif window size
            
        Returns:
            List of (position, motif) tuples
        """
        if len(sequence) < window_size * 2:
            return []
        
        motifs = []
        seen_motifs = defaultdict(list)
        
        # Slide window through sequence
        for i in range(len(sequence) - window_size + 1):
            window = tuple(sequence[i:i + window_size])
            seen_motifs[window].append(i)
        
        # Find repeated motifs
        for motif, positions in seen_motifs.items():
            if len(positions) >= 2:
                for pos in positions:
                    motifs.append((pos, list(motif)))
        
        return sorted(motifs, key=lambda x: x[0])
    
    def detect_anomalies(
        self,
        sequence: List[Any],
        known_patterns: Optional[List[Pattern]] = None
    ) -> List[Tuple[int, Any, float]]:
        """
        Detect anomalies in sequence
        
        Args:
            sequence: Input sequence
            known_patterns: Optional known patterns
            
        Returns:
            List of (position, element, anomaly_score) tuples
        """
        if not sequence:
            return []
        
        anomalies = []
        
        # Use known patterns or stored patterns
        patterns = known_patterns or list(self.patterns.values())
        if not patterns:
            return []
        
        # Convert sequence to string
        string_seq = self._convert_to_strings([sequence])[0]
        
        # Check each position
        for i, element in enumerate(string_seq):
            # Calculate how well this element fits known patterns
            fit_score = self._calculate_pattern_fit(
                string_seq,
                i,
                patterns
            )
            
            # High anomaly score = doesn't fit patterns
            anomaly_score = 1.0 - fit_score
            
            if anomaly_score > 0.7:  # Threshold
                anomalies.append((i, sequence[i], anomaly_score))
        
        return anomalies
    
    def match_pattern(
        self,
        sequence: List[Any],
        pattern: Pattern
    ) -> List[int]:
        """
        Find pattern matches in sequence
        
        Args:
            sequence: Input sequence
            pattern: Pattern to match
            
        Returns:
            List of match positions
        """
        string_seq = self._convert_to_strings([sequence])[0]
        pattern_seq = pattern.sequence
        
        matches = []
        pattern_len = len(pattern_seq)
        
        for i in range(len(string_seq) - pattern_len + 1):
            if string_seq[i:i + pattern_len] == pattern_seq:
                matches.append(i)
        
        return matches
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID"""
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_outcome(
        self,
        min_outcome: float
    ) -> List[Pattern]:
        """Get patterns with minimum average outcome"""
        return [
            p for p in self.patterns.values()
            if p.avg_outcome >= min_outcome
        ]
    
    def get_similar_patterns(
        self,
        pattern: Pattern,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[Pattern, float]]:
        """
        Find similar patterns
        
        Args:
            pattern: Reference pattern
            similarity_threshold: Minimum similarity
            
        Returns:
            List of (pattern, similarity) tuples
        """
        similar = []
        
        for other in self.patterns.values():
            if other.id != pattern.id:
                similarity = self._calculate_pattern_similarity(pattern, other)
                if similarity >= similarity_threshold:
                    similar.append((other, similarity))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def cluster_patterns(
        self,
        n_clusters: int = 5
    ) -> Dict[int, List[Pattern]]:
        """
        Cluster patterns into groups
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dictionary of cluster_id -> patterns
        """
        if not self.patterns:
            return {}
        
        # Simple clustering based on pattern similarity
        patterns_list = list(self.patterns.values())
        clusters = defaultdict(list)
        
        # Initialize clusters with diverse patterns
        for i in range(min(n_clusters, len(patterns_list))):
            clusters[i].append(patterns_list[i])
        
        # Assign remaining patterns to nearest cluster
        for pattern in patterns_list[n_clusters:]:
            best_cluster = 0
            best_similarity = 0.0
            
            for cluster_id, cluster_patterns in clusters.items():
                # Average similarity to cluster
                similarities = [
                    self._calculate_pattern_similarity(pattern, p)
                    for p in cluster_patterns
                ]
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_cluster = cluster_id
            
            clusters[best_cluster].append(pattern)
        
        return dict(clusters)
    
    # Private methods
    
    def _convert_to_strings(self, sequences: List[List[Any]]) -> List[List[str]]:
        """Convert sequences to string representation"""
        string_sequences = []
        
        for sequence in sequences:
            string_seq = []
            for element in sequence:
                if isinstance(element, (list, tuple, np.ndarray)):
                    # Discretize continuous values
                    string_elem = self._discretize_vector(element)
                elif isinstance(element, (int, float)):
                    # Discretize scalar
                    string_elem = self._discretize_scalar(element)
                else:
                    string_elem = str(element)
                
                string_seq.append(string_elem)
            
            string_sequences.append(string_seq)
        
        return string_sequences
    
    def _discretize_vector(self, vector: Union[List, np.ndarray]) -> str:
        """Discretize vector into string"""
        vector = np.array(vector)
        
        # Simple discretization: high/low for each dimension
        discrete = []
        for val in vector[:5]:  # Limit dimensions
            if val > 0.5:
                discrete.append('H')
            elif val < -0.5:
                discrete.append('L')
            else:
                discrete.append('M')
        
        return ''.join(discrete)
    
    def _discretize_scalar(self, value: float) -> str:
        """Discretize scalar into string"""
        if value > 0.5:
            return 'POS_HIGH'
        elif value > 0:
            return 'POS_LOW'
        elif value < -0.5:
            return 'NEG_HIGH'
        elif value < 0:
            return 'NEG_LOW'
        else:
            return 'ZERO'
    
    def _mine_frequent_patterns(
        self,
        sequences: List[List[str]]
    ) -> Dict[Tuple[str, ...], int]:
        """Mine frequent patterns using Apriori-like algorithm"""
        patterns = defaultdict(int)
        
        # Count all subsequences
        for sequence in sequences:
            seen_patterns = set()
            
            # Generate all subsequences
            for length in range(1, min(len(sequence) + 1, self.max_pattern_length)):
                for start in range(len(sequence) - length + 1):
                    pattern = tuple(sequence[start:start + length])
                    
                    # Count each pattern once per sequence
                    if pattern not in seen_patterns:
                        patterns[pattern] += 1
                        seen_patterns.add(pattern)
        
        return dict(patterns)
    
    def _create_pattern(
        self,
        pattern_seq: Tuple[str, ...],
        support: int,
        sequences: List[List[str]],
        outcomes: Optional[List[float]]
    ) -> Pattern:
        """Create pattern object with metrics"""
        # Generate pattern ID
        pattern_str = '_'.join(pattern_seq)
        pattern_id = hashlib.md5(pattern_str.encode()).hexdigest()[:12]
        
        # Calculate confidence and outcome
        confidence = support / max(len(sequences), 1)
        
        avg_outcome = 0.0
        if outcomes:
            # Find sequences containing this pattern
            pattern_outcomes = []
            for i, sequence in enumerate(sequences):
                if self._contains_pattern(sequence, pattern_seq):
                    if i < len(outcomes):
                        pattern_outcomes.append(outcomes[i])
            
            if pattern_outcomes:
                avg_outcome = np.mean(pattern_outcomes)
        
        return Pattern(
            id=pattern_id,
            sequence=list(pattern_seq),
            frequency=support,
            confidence=confidence,
            avg_outcome=avg_outcome,
            metadata={
                'length': len(pattern_seq),
                'first_element': pattern_seq[0] if pattern_seq else None,
                'last_element': pattern_seq[-1] if pattern_seq else None
            }
        )
    
    def _contains_pattern(
        self,
        sequence: List[str],
        pattern: Tuple[str, ...]
    ) -> bool:
        """Check if sequence contains pattern"""
        pattern_len = len(pattern)
        for i in range(len(sequence) - pattern_len + 1):
            if tuple(sequence[i:i + pattern_len]) == pattern:
                return True
        return False
    
    def _store_pattern(self, pattern: Pattern) -> None:
        """Store pattern in index"""
        self.patterns[pattern.id] = pattern
        
        # Index by elements
        for element in pattern.sequence:
            self.pattern_index[element].add(pattern.id)
    
    def _calculate_pattern_fit(
        self,
        sequence: List[str],
        position: int,
        patterns: List[Pattern]
    ) -> float:
        """Calculate how well position fits known patterns"""
        if not patterns:
            return 0.5
        
        fit_scores = []
        
        for pattern in patterns:
            # Check if this position could be part of this pattern
            for i, element in enumerate(pattern.sequence):
                # Check positions where this element could appear
                potential_start = position - i
                
                if potential_start >= 0 and potential_start + len(pattern.sequence) <= len(sequence):
                    # Check if pattern would match at this position
                    matches = True
                    for j, pattern_elem in enumerate(pattern.sequence):
                        if potential_start + j < len(sequence):
                            if sequence[potential_start + j] != pattern_elem:
                                matches = False
                                break
                    
                    if matches:
                        # Weight by pattern confidence
                        fit_scores.append(pattern.confidence)
        
        return max(fit_scores) if fit_scores else 0.0
    
    def _calculate_pattern_similarity(
        self,
        pattern1: Pattern,
        pattern2: Pattern
    ) -> float:
        """Calculate similarity between patterns"""
        seq1 = pattern1.sequence
        seq2 = pattern2.sequence
        
        # Jaccard similarity of elements
        set1 = set(seq1)
        set2 = set(seq2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Length similarity
        max_len = max(len(seq1), len(seq2))
        min_len = min(len(seq1), len(seq2))
        length_sim = min_len / max_len if max_len > 0 else 1.0
        
        # Outcome similarity
        outcome_diff = abs(pattern1.avg_outcome - pattern2.avg_outcome)
        outcome_sim = 1.0 / (1.0 + outcome_diff / 100.0)
        
        # Combined similarity
        similarity = (jaccard * 0.5 + length_sim * 0.2 + outcome_sim * 0.3)
        
        return float(similarity)# modules/memory/shared/utils.py
"""
Shared Utilities for Unified Memory System
Common functions and helper classes
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from collections import OrderedDict, deque
from datetime import datetime
import numpy as np
import hashlib
import time
import json
import threading


class LRUCache:
    """
    Thread-safe LRU cache implementation
    """
    
    def __init__(self, maxsize: int = 1000):
        """
        Initialize LRU cache
        
        Args:
            maxsize: Maximum cache size
        """
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Put value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        with self.lock:
            # Store with expiry time if TTL provided
            if ttl:
                expiry = time.time() + ttl
                self.cache[key] = (value, expiry)
            else:
                self.cache[key] = value
            
            # Move to end
            self.cache.move_to_end(key)
            
            # Remove oldest if over capacity
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self.lock:
            current_time = time.time()
            expired = []
            
            for key, value in self.cache.items():
                if isinstance(value, tuple) and len(value) == 2:
                    _, expiry = value
                    if expiry < current_time:
                        expired.append(key)
            
            for key in expired:
                del self.cache[key]
            
            return len(expired)


class MemoryUtils:
    """
    Utility functions for memory operations
    """
    
    @staticmethod
    def extract_action(trade: Dict[str, Any]) -> np.ndarray:
        """
        Extract action from trade
        
        Args:
            trade: Trade dictionary
            
        Returns:
            Action vector [position_delta, confidence]
        """
        action = trade.get('action')
        
        if action is not None:
            if isinstance(action, (list, tuple, np.ndarray)):
                action_array = np.array(action, dtype=np.float32)
                if len(action_array) >= 2:
                    return action_array[:2]
                elif len(action_array) == 1:
                    return np.array([action_array[0], 0.5], dtype=np.float32)
            else:
                return np.array([float(action), 0.5], dtype=np.float32)
        
        # Fallback: derive from trade side and size
        side = str(trade.get('side', 'hold')).lower()
        size = float(trade.get('size', 0.0))
        confidence = float(trade.get('confidence', 0.5))
        
        if side == 'buy':
            position_delta = size
        elif side == 'sell':
            position_delta = -size
        else:
            position_delta = 0.0
        
        return np.array([position_delta, confidence], dtype=np.float32)
    
    @staticmethod
    def calculate_pnl(
        entry_price: float,
        exit_price: float,
        size: float,
        side: str
    ) -> float:
        """
        Calculate PnL from trade parameters
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            side: Trade side ('buy' or 'sell')
            
        Returns:
            PnL value
        """
        if side == 'buy':
            pnl = (exit_price - entry_price) * size
        elif side == 'sell':
            pnl = (entry_price - exit_price) * size
        else:
            pnl = 0.0
        
        return float(pnl)
    
    @staticmethod
    def encode_context(context: Dict[str, Any]) -> str:
        """
        Encode context to string for hashing/comparison
        
        Args:
            context: Context dictionary
            
        Returns:
            Encoded string
        """
        # Extract key context elements
        key_elements = []
        
        for key in ['regime', 'volatility', 'session', 'trend']:
            if key in context:
                value = context[key]
                if isinstance(value, dict):
                    # Take first value if dict
                    value = list(value.values())[0] if value else 'unknown'
                key_elements.append(f"{key}:{value}")
        
        return '_'.join(key_elements)
    
    @staticmethod
    def hash_features(features: np.ndarray, precision: int = 2) -> str:
        """
        Create hash of feature vector
        
        Args:
            features: Feature vector
            precision: Decimal precision for hashing
            
        Returns:
            Hash string
        """
        # Round features to reduce noise
        rounded = np.round(features, precision)
        
        # Convert to bytes and hash
        feature_bytes = rounded.tobytes()
        return hashlib.md5(feature_bytes).hexdigest()[:16]
    
    @staticmethod
    def calculate_similarity(
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """
        Calculate similarity between vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Similarity score
        """
        # Ensure same dimensions
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
        
        if metric == 'cosine':
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(np.clip(similarity, -1, 1))
        
        elif metric == 'euclidean':
            # Euclidean distance converted to similarity
            distance = np.linalg.norm(vec1 - vec2)
            similarity = 1.0 / (1.0 + distance)
            return float(similarity)
        
        elif metric == 'manhattan':
            # Manhattan distance converted to similarity
            distance = np.sum(np.abs(vec1 - vec2))
            similarity = 1.0 / (1.0 + distance)
            return float(similarity)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    def smooth_signal(
        signal: Union[float, np.ndarray],
        history: deque,
        alpha: float = 0.1
    ) -> float:
        """
        Exponential smoothing of signal
        
        Args:
            signal: Current signal value
            history: Historical values
            alpha: Smoothing factor (0-1)
            
        Returns:
            Smoothed signal
        """
        if not history:
            return float(signal)
        
        # Add current signal to history
        history.append(float(signal))
        
        # Apply exponential smoothing
        smoothed = history[0]
        for value in list(history)[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        return float(smoothed)
    
    @staticmethod
    def calculate_entropy(distribution: np.ndarray) -> float:
        """
        Calculate Shannon entropy of distribution
        
        Args:
            distribution: Probability distribution
            
        Returns:
            Entropy value
        """
        # Normalize to probabilities
        probs = distribution / np.sum(distribution)
        
        # Remove zeros
        probs = probs[probs > 0]
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        return float(entropy)
    
    @staticmethod
    def detect_outliers(
        values: np.ndarray,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> np.ndarray:
        """
        Detect outliers in values
        
        Args:
            values: Array of values
            method: Detection method ('iqr', 'zscore', 'isolation')
            threshold: Outlier threshold
            
        Returns:
            Boolean array of outlier flags
        """
        if len(values) < 3:
            return np.zeros(len(values), dtype=bool)
        
        if method == 'iqr':
            # Interquartile range method
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            
            outliers = (values < lower) | (values > upper)
        
        elif method == 'zscore':
            # Z-score method
            mean = np.mean(values)
            std = np.std(values)
            
            if std > 0:
                z_scores = np.abs((values - mean) / std)
                outliers = z_scores > threshold
            else:
                outliers = np.zeros(len(values), dtype=bool)
        
        else:
            # Simple percentile method
            lower = np.percentile(values, 5)
            upper = np.percentile(values, 95)
            outliers = (values < lower) | (values > upper)
        
        return outliers
    
    @staticmethod
    def normalize_array(
        array: np.ndarray,
        method: str = 'minmax'
    ) -> np.ndarray:
        """
        Normalize array values
        
        Args:
            array: Input array
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Normalized array
        """
        if len(array) == 0:
            return array
        
        if method == 'minmax':
            # Min-max normalization
            min_val = np.min(array)
            max_val = np.max(array)
            
            if max_val - min_val > 0:
                normalized = (array - min_val) / (max_val - min_val)
            else:
                normalized = array - min_val
        
        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(array)
            std = np.std(array)
            
            if std > 0:
                normalized = (array - mean) / std
            else:
                normalized = array - mean
        
        elif method == 'robust':
            # Robust normalization using median and MAD
            median = np.median(array)
            mad = np.median(np.abs(array - median))
            
            if mad > 0:
                normalized = (array - median) / mad
            else:
                normalized = array - median
        
        else:
            normalized = array
        
        return normalized.astype(np.float32)
    
    @staticmethod
    def create_time_features(timestamp: float) -> Dict[str, float]:
        """
        Create time-based features
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Dictionary of time features
        """
        dt = datetime.fromtimestamp(timestamp)
        
        return {
            'hour': dt.hour / 24.0,
            'day_of_week': dt.weekday() / 7.0,
            'day_of_month': dt.day / 31.0,
            'month': dt.month / 12.0,
            'quarter': (dt.month - 1) // 3 / 4.0,
            'is_weekend': 1.0 if dt.weekday() >= 5 else 0.0,
            'is_month_end': 1.0 if dt.day >= 28 else 0.0
        }
    
    @staticmethod
    def format_memory_size(bytes_size: int) -> str:
        """
        Format memory size for display
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            Formatted string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        
        return f"{bytes_size:.2f} TB"
    
    @staticmethod
    def estimate_memory_usage(obj: Any) -> int:
        """
        Estimate memory usage of object
        
        Args:
            obj: Object to measure
            
        Returns:
            Estimated size in bytes
        """
        import sys
        
        size = sys.getsizeof(obj)
        
        # Handle containers
        if isinstance(obj, dict):
            size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            size += sum(sys.getsizeof(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            size = obj.nbytes
        
        return size


class RingBuffer:
    """
    Efficient ring buffer for streaming data
    """
    
    def __init__(self, capacity: int, dtype: type = float):
        """
        Initialize ring buffer
        
        Args:
            capacity: Buffer capacity
            dtype: Data type for numpy array
        """
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=dtype)
        self.position = 0
        self.size = 0
    
    def append(self, value: float) -> None:
        """Add value to buffer"""
        self.buffer[self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get(self) -> np.ndarray:
        """Get buffer contents in order"""
        if self.size < self.capacity:
            return self.buffer[:self.size]
        else:
            # Reorder circular buffer
            return np.concatenate([
                self.buffer[self.position:],
                self.buffer[:self.position]
            ])
    
    def mean(self) -> float:
        """Get mean of buffer"""
        if self.size == 0:
            return 0.0
        return float(np.mean(self.buffer[:self.size]))
    
    def std(self) -> float:
        """Get standard deviation of buffer"""
        if self.size < 2:
            return 0.0
        return float(np.std(self.buffer[:self.size]))
    
    def clear(self) -> None:
        """Clear buffer"""
        self.buffer.fill(0)
        self.position = 0
        self.size = 0


class MemoryStatistics:
    """
    Statistics tracking for memory system
    """
    
    def __init__(self):
        """Initialize statistics"""
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.values = defaultdict(list)
        self.start_times = {}
    
    def increment(self, name: str, amount: int = 1) -> None:
        """Increment counter"""
        self.counters[name] += amount
    
    def record_time(self, name: str, duration: float) -> None:
        """Record time duration"""
        self.timers[name].append(duration)
        # Keep only recent times
        if len(self.timers[name]) > 1000:
            self.timers[name] = self.timers[name][-1000:]
    
    def record_value(self, name: str, value: float) -> None:
        """Record value"""
        self.values[name].append(value)
        # Keep only recent values
        if len(self.values[name]) > 1000:
            self.values[name] = self.values[name][-1000:]
    
    def start_timer(self, name: str) -> None:
        """Start timer"""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop timer and record duration"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.record_time(name, duration)
            del self.start_times[name]
            return duration
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        summary = {
            'counters': dict(self.counters),
            'timers': {},
            'values': {}
        }
        
        # Timer statistics
        for name, times in self.timers.items():
            if times:
                summary['timers'][name] = {
                    'count': len(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times)
                }
        
        # Value statistics
        for name, values in self.values.items():
            if values:
                summary['values'][name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all statistics"""
        self.counters.clear()
        self.timers.clear()
        self.values.clear()
        self.start_times.clear() alright  # modules/memory/debug/memory_logger.py
"""
Unified Debug Logger for Memory System
Comprehensive logging with multiple levels and formats
"""

from ast import Tuple
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from collections import deque
import time
import json
import threading
import traceback
import os
from pathlib import Path


class MemoryDebugLogger:
    """
    Comprehensive debug logging for unified memory system
    
    Features:
    - Multiple log levels (TRACE, DEBUG, INFO, WARNING, ERROR)
    - Component-specific coloring
    - Performance profiling
    - Memory tracking
    - Structured and unstructured logging
    - Log rotation and compression
    """
    
    def __init__(
        self,
        enabled: bool = False,
        level: str = "INFO",
        log_path: str = "logs/memory/unified_debug.log",
        max_size_mb: int = 100,
        rotation_count: int = 5
    ):
        """
        Initialize debug logger
        
        Args:
            enabled: Enable logging
            level: Log level
            log_path: Path to log file
            max_size_mb: Maximum log file size in MB
            rotation_count: Number of rotated files to keep
        """
        self.enabled = enabled
        self.level = level
        self.log_path = log_path
        self.max_size_mb = max_size_mb
        self.rotation_count = rotation_count
        
        # Log levels
        self.levels = {
            'TRACE': 0,
            'DEBUG': 1,
            'INFO': 2,
            'WARNING': 3,
            'ERROR': 4
        }
        self.current_level = self.levels.get(level.upper(), 2)
        
        # Component colors for terminal output
        self.colors = {
            'replay': '\033[94m',      # Blue
            'compression': '\033[92m',  # Green
            'mistakes': '\033[91m',     # Red
            'neural': '\033[95m',       # Magenta
            'playbook': '\033[93m',     # Yellow
            'budget': '\033[96m',       # Cyan
            'unified': '\033[97m',      # White
            'reset': '\033[0m',
            'bold': '\033[1m',
            'underline': '\033[4m'
        }
        
        # Create log directory
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Log file handle
        self.file_handle = None
        self._open_log_file()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance tracking
        self.operation_times = deque(maxlen=1000)
        self.component_times = {
            'replay': deque(maxlen=100),
            'compression': deque(maxlen=100),
            'mistakes': deque(maxlen=100),
            'neural': deque(maxlen=100),
            'playbook': deque(maxlen=100),
            'budget': deque(maxlen=100)
        }
        
        # Memory tracking
        self.memory_snapshots = deque(maxlen=100)
        
        # Statistics
        self.log_counts = {
            'TRACE': 0,
            'DEBUG': 0,
            'INFO': 0,
            'WARNING': 0,
            'ERROR': 0
        }
    
    def _open_log_file(self) -> None:
        """Open log file handle"""
        try:
            self.file_handle = open(self.log_path, 'a', encoding='utf-8')
        except Exception as e:
            print(f"Failed to open log file: {e}")
            self.file_handle = None
    
    def _check_rotation(self) -> None:
        """Check if log rotation is needed"""
        try:
            if os.path.exists(self.log_path):
                size_mb = os.path.getsize(self.log_path) / (1024 * 1024)
                if size_mb > self.max_size_mb:
                    self._rotate_logs()
        except Exception:
            pass
    
    def _rotate_logs(self) -> None:
        """Rotate log files"""
        try:
            if self.file_handle:
                self.file_handle.close()
            
            # Rotate existing files
            for i in range(self.rotation_count - 1, 0, -1):
                old_path = f"{self.log_path}.{i}"
                new_path = f"{self.log_path}.{i + 1}"
                
                if os.path.exists(old_path):
                    if i == self.rotation_count - 1:
                        os.remove(old_path)
                    else:
                        os.rename(old_path, new_path)
            
            # Move current to .1
            if os.path.exists(self.log_path):
                os.rename(self.log_path, f"{self.log_path}.1")
            
            # Open new file
            self._open_log_file()
            
        except Exception:
            pass
    
    def log(
        self,
        level: str,
        message: str,
        component: Optional[str] = None,
        data: Optional[Any] = None
    ) -> None:
        """
        Generic log method
        
        Args:
            level: Log level
            message: Log message
            component: Component name
            data: Additional data
        """
        if not self.enabled:
            return
        
        level_value = self.levels.get(level.upper(), 2)
        if level_value < self.current_level:
            return
        
        with self.lock:
            timestamp = datetime.now()
            
            # Format log entry
            entry = self._format_entry(timestamp, level, message, component, data)
            
            # Write to file
            if self.file_handle:
                self.file_handle.write(entry + '\n')
                self.file_handle.flush()
            
            # Print to console if DEBUG mode
            if self.current_level <= 1:  # DEBUG or TRACE
                self._print_colored(timestamp, level, message, component, data)
            
            # Update statistics
            self.log_counts[level.upper()] += 1
            
            # Check rotation
            self._check_rotation()
    
    def trace(self, message: str, **kwargs) -> None:
        """Log trace message"""
        self.log('TRACE', message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self.log('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self.log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self.log('WARNING', message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log error message"""
        if exception:
            tb = traceback.format_exc()
            kwargs['traceback'] = tb
        self.log('ERROR', message, **kwargs)
    
    def log_input(self, operation: str, data: Any) -> None:
        """Log operation input"""
        if not self.enabled:
            return
        
        self.debug(
            f"Operation Input: {operation}",
            component='unified',
            data={
                'operation': operation,
                'input_keys': list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                'timestamp': time.time()
            }
        )
    
    def log_output(self, operation: str, data: Any) -> None:
        """Log operation output"""
        if not self.enabled:
            return
        
        self.debug(
            f"Operation Output: {operation}",
            component='unified',
            data={
                'operation': operation,
                'output_keys': list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                'timestamp': time.time()
            }
        )
    
    def log_component_operation(
        self,
        component: str,
        operation: str,
        data: Dict[str, Any]
    ) -> None:
        """Log component-specific operation"""
        if not self.enabled:
            return
        
        # Record timing
        if 'time_ms' in data:
            self.component_times[component].append(data['time_ms'])
        
        self.debug(
            f"[{component.upper()}] {operation}",
            component=component,
            data=data
        )
    
    def log_memory_operation(self, operation: str, entry: Dict[str, Any]) -> None:
        """Log memory store operation"""
        if not self.enabled or self.current_level > 1:  # Only in DEBUG/TRACE
            return
        
        self.debug(
            f"Memory Operation: {operation}",
            component='store',
            data={
                'operation': operation,
                'timestamp': entry.get('timestamp'),
                'pnl': entry.get('pnl'),
                'importance': entry.get('importance'),
                'metadata': entry.get('metadata')
            }
        )
    
    def log_bus_update(self, key: str, value: Any) -> None:
        """Log SmartInfoBus update"""
        if not self.enabled or self.current_level > 0:  # Only in TRACE
            return
        
        self.trace(
            f"Bus Update: {key}",
            component='bus',
            data={
                'key': key,
                'value_type': type(value).__name__,
                'value_sample': self._truncate(value, 100)
            }
        )
    
    def log_bus_updates(self, updates: List[Tuple[str, Any]]) -> None:
        """Log multiple bus updates"""
        if not self.enabled or self.current_level > 0:  # Only in TRACE
            return
        
        update_summary = {
            key: type(value).__name__ 
            for key, value in updates[:10]  # Limit to first 10
        }
        
        self.trace(
            f"Batch Bus Update: {len(updates)} keys",
            component='bus',
            data={'updates': update_summary}
        )
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics"""
        if not self.enabled:
            return
        
        # Record in memory
        self.operation_times.append(metrics.get('processing_time_ms', 0))
        
        # Add memory snapshot
        if 'total_memories' in metrics:
            self.memory_snapshots.append({
                'timestamp': time.time(),
                'total_memories': metrics['total_memories'],
                'utilization': metrics.get('utilization', 0)
            })
        
        self.info(
            "Performance Metrics",
            component='performance',
            data=metrics
        )
    
    def log_error(self, context: str, error: Exception) -> None:
        """Log error with context"""
        self.error(
            f"Error in {context}",
            exception=error,
            component='error',
            data={
                'context': context,
                'error_type': type(error).__name__,
                'error_message': str(error)
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logger statistics"""
        with self.lock:
            # Calculate average times
            avg_times = {}
            for component, times in self.component_times.items():
                if times:
                    avg_times[component] = {
                        'avg_ms': sum(times) / len(times),
                        'max_ms': max(times),
                        'min_ms': min(times)
                    }
            
            return {
                'enabled': self.enabled,
                'level': self.level,
                'log_counts': dict(self.log_counts),
                'total_logs': sum(self.log_counts.values()),
                'component_performance': avg_times,
                'memory_snapshots': len(self.memory_snapshots),
                'file_size_mb': self._get_file_size_mb()
            }
    
    def _format_entry(
        self,
        timestamp: datetime,
        level: str,
        message: str,
        component: Optional[str],
        data: Optional[Any]
    ) -> str:
        """Format log entry"""
        # Basic format
        entry_parts = [
            timestamp.isoformat(),
            f"[{level.upper():8}]"
        ]
        
        if component:
            entry_parts.append(f"[{component:12}]")
        
        entry_parts.append(message)
        
        # Add data if present
        if data is not None:
            if isinstance(data, dict):
                data_str = json.dumps(data, indent=2, default=str)
            else:
                data_str = str(data)
            
            entry_parts.append(f"\n  DATA: {data_str}")
        
        return " ".join(entry_parts)
    
    def _print_colored(
        self,
        timestamp: datetime,
        level: str,
        message: str,
        component: Optional[str],
        data: Optional[Any]
    ) -> None:
        """Print colored output to console"""
        # Get component color
        color = self.colors.get(component, '') if component else ''
        reset = self.colors['reset']
        
        # Level colors
        level_colors = {
            'TRACE': '\033[90m',    # Gray
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m'     # Red
        }
        level_color = level_colors.get(level.upper(), '')
        
        # Format output
        time_str = timestamp.strftime('%H:%M:%S.%f')[:-3]
        
        if component:
            print(f"{level_color}[{level:5}] {time_str} {color}[{component}] {message}{reset}")
        else:
            print(f"{level_color}[{level:5}] {time_str} {message}{reset}")
        
        # Print data if in TRACE mode
        if data and self.current_level == 0:
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {self._truncate(value, 200)}")
            else:
                print(f"  {self._truncate(data, 200)}")
    
    def _truncate(self, data: Any, max_length: int = 200) -> str:
        """Truncate data for display"""
        str_data = str(data)
        if len(str_data) > max_length:
            return str_data[:max_length] + "..."
        return str_data
    
    def _get_file_size_mb(self) -> float:
        """Get current log file size in MB"""
        try:
            if os.path.exists(self.log_path):
                return os.path.getsize(self.log_path) / (1024 * 1024)
        except Exception:
            pass
        return 0.0
    
    def close(self) -> None:
        """Close logger and clean up resources"""
        with self.lock:
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class PerformanceProfiler:
    """
    Performance profiler for memory operations
    """
    
    def __init__(self, logger: Optional[MemoryDebugLogger] = None):
        """
        Initialize profiler
        
        Args:
            logger: Optional debug logger
        """
        self.logger = logger
        self.timers = {}
        self.profiles = defaultdict(list)
    
    def start(self, name: str) -> None:
        """Start timing an operation"""
        self.timers[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing and record duration"""
        if name in self.timers:
            duration = (time.perf_counter() - self.timers[name]) * 1000  # Convert to ms
            del self.timers[name]
            
            # Record profile
            self.profiles[name].append(duration)
            
            # Log if logger available
            if self.logger:
                self.logger.trace(
                    f"Operation '{name}' took {duration:.2f}ms",
                    component='profiler'
                )
            
            return duration
        return 0.0
    
    def get_profile(self, name: str) -> Dict[str, float]:
        """Get profile statistics for operation"""
        times = self.profiles.get(name, [])
        
        if not times:
            return {'count': 0}
        
        import numpy as np
        
        return {
            'count': len(times),
            'total_ms': sum(times),
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99)
        }
    
    def get_all_profiles(self) -> Dict[str, Dict[str, float]]:
        """Get all operation profiles"""
        return {
            name: self.get_profile(name)
            for name in self.profiles.keys()
        }
    
    def reset(self) -> None:
        """Reset profiler"""
        self.timers.clear()
        self.profiles.clear()