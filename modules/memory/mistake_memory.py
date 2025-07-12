# ─────────────────────────────────────────────────────────────
# File: modules/memory/mistake_memory.py
# 🚀 PRODUCTION-READY Mistake Memory System
# Advanced loss avoidance with clustering and SmartInfoBus integration
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@dataclass
class MistakeConfig:
    """Configuration for Mistake Memory"""
    max_mistakes: int = 100
    n_clusters: int = 5
    profit_threshold: float = 10.0
    cluster_update_threshold: int = 10
    avoidance_sensitivity: float = 1.0
    pattern_memory_size: int = 50
    danger_zone_weight: float = 2.0
    
    # Performance thresholds
    max_processing_time_ms: float = 250
    circuit_breaker_threshold: int = 3
    min_cluster_quality: float = 0.3
    
    # Learning parameters
    learning_rate: float = 0.1
    false_positive_threshold: float = 0.2
    min_samples_for_clustering: int = 10


@module(
    name="MistakeMemory",
    version="3.0.0",
    category="memory",
    provides=["mistake_avoidance", "danger_zones", "pattern_recognition", "loss_prevention"],
    requires=["trades", "features", "market_context", "risk_data"],
    description="Advanced mistake memory with clustering for loss avoidance and pattern recognition",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class MistakeMemory(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced mistake memory with SmartInfoBus integration.
    Learns from both losses and wins using clustering to identify danger and profit zones.
    """

    def __init__(self, 
                 config: Optional[MistakeConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        self.config = config or MistakeConfig()
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize mistake memory state
        self._initialize_mistake_state()
        
        self.logger.info(
            format_operator_message(
                "🧠", "MISTAKE_MEMORY_INITIALIZED",
                details=f"Max mistakes: {self.config.max_mistakes}, Clusters: {self.config.n_clusters}",
                result="Loss avoidance system ready",
                context="mistake_learning"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for mistake memory"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="MistakeMemory", 
            log_path="logs/mistake_memory.log", 
            max_lines=3000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("MistakeMemory", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for clustering operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        self._start_monitoring()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        """Initialize genome-based parameters"""
        if genome:
            self.genome = {
                "max_mistakes": int(genome.get("max_mistakes", self.config.max_mistakes)),
                "n_clusters": int(genome.get("n_clusters", self.config.n_clusters)),
                "profit_threshold": float(genome.get("profit_threshold", self.config.profit_threshold)),
                "cluster_update_threshold": int(genome.get("cluster_update_threshold", self.config.cluster_update_threshold)),
                "avoidance_sensitivity": float(genome.get("avoidance_sensitivity", self.config.avoidance_sensitivity)),
                "pattern_memory_size": int(genome.get("pattern_memory_size", self.config.pattern_memory_size)),
                "danger_zone_weight": float(genome.get("danger_zone_weight", self.config.danger_zone_weight))
            }
        else:
            self.genome = {
                "max_mistakes": self.config.max_mistakes,
                "n_clusters": self.config.n_clusters,
                "profit_threshold": self.config.profit_threshold,
                "cluster_update_threshold": self.config.cluster_update_threshold,
                "avoidance_sensitivity": self.config.avoidance_sensitivity,
                "pattern_memory_size": self.config.pattern_memory_size,
                "danger_zone_weight": self.config.danger_zone_weight
            }

    def _initialize_mistake_state(self):
        """Initialize mistake memory state"""
        # Memory buffers for wins and losses
        self._loss_buf: List[Tuple[np.ndarray, float, Dict]] = []
        self._win_buf: List[Tuple[np.ndarray, float, Dict]] = []
        
        # Clustering models
        self._km_loss: Optional[KMeans] = None
        self._km_win: Optional[KMeans] = None
        self._scaler = StandardScaler()
        
        # Cluster analysis
        self._mean_dist = 0.0
        self._last_dist = 0.0
        self._danger_zones: List[np.ndarray] = []
        self._profit_zones: List[np.ndarray] = []
        
        # Enhanced pattern tracking
        self._consecutive_losses = 0
        self._loss_patterns = {}  # pattern -> (count, severity, last_seen)
        self._win_patterns = {}   # pattern -> (count, profitability, last_seen)
        self._avoidance_signal = 0.0
        
        # Advanced analytics
        self._pattern_evolution = deque(maxlen=100)
        self._danger_zone_violations = deque(maxlen=50)
        self._learning_effectiveness = deque(maxlen=200)
        self._market_context_correlations = {}
        
        # Adaptive learning
        self._cluster_quality_scores = deque(maxlen=20)
        self._prediction_accuracy = 0.0
        self._false_positive_rate = 0.0
        self._true_positive_rate = 0.0
        
        # Performance metrics
        self._mistake_performance = {
            'total_losses_learned': 0,
            'total_wins_learned': 0,
            'avoidance_effectiveness': 0.0,
            'cluster_updates': 0,
            'pattern_discoveries': 0
        }

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_mistake_health()
                    self._analyze_avoidance_effectiveness()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module"""
        try:
            # Set initial mistake memory status in SmartInfoBus
            initial_status = {
                "danger_zones": [],
                "profit_zones": [],
                "avoidance_signal": 0.0,
                "consecutive_losses": 0,
                "pattern_count": 0
            }
            
            self.smart_bus.set(
                'mistake_avoidance',
                initial_status,
                module='MistakeMemory',
                thesis="Initial mistake memory and avoidance status"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process mistake memory and learning"""
        start_time = time.time()
        
        try:
            # Extract mistake learning data
            learning_data = await self._extract_learning_data(**inputs)
            
            if not learning_data:
                return await self._handle_no_data_fallback()
            
            # Process learning from trades
            learning_result = await self._process_learning_data(learning_data)
            
            # Update clustering if needed
            clustering_result = await self._update_clustering()
            learning_result.update(clustering_result)
            
            # Calculate avoidance signals
            avoidance_result = await self._calculate_avoidance_signals(learning_data)
            learning_result.update(avoidance_result)
            
            # Generate thesis
            thesis = await self._generate_mistake_thesis(learning_data, learning_result)
            
            # Update SmartInfoBus
            await self._update_mistake_smart_bus(learning_result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return learning_result
            
        except Exception as e:
            return await self._handle_mistake_error(e, start_time)

    async def _extract_learning_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract learning data from SmartInfoBus"""
        try:
            # Get trades data
            trades = self.smart_bus.get('trades', 'MistakeMemory') or []
            
            # Get features
            features = self.smart_bus.get('features', 'MistakeMemory')
            
            # Get market context
            market_context = self.smart_bus.get('market_context', 'MistakeMemory') or {}
            
            # Get risk data
            risk_data = self.smart_bus.get('risk_data', 'MistakeMemory') or {}
            
            # Get current features if provided
            current_features = inputs.get('features', features)
            
            return {
                'trades': trades,
                'features': features,
                'current_features': current_features,
                'market_context': market_context,
                'risk_data': risk_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract learning data: {e}")
            return None

    async def _process_learning_data(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process learning data from trades"""
        try:
            trades = learning_data.get('trades', [])
            market_context = learning_data.get('market_context', {})
            
            if not trades:
                return {'learning_processed': False, 'reason': 'no_trades'}
            
            # Process recent trades
            losses_learned = 0
            wins_learned = 0
            
            for trade in trades[-20:]:  # Process last 20 trades
                if not isinstance(trade, dict) or 'pnl' not in trade:
                    continue
                
                # Extract features for this trade
                trade_features = self._extract_trade_features(trade, market_context)
                if trade_features is None:
                    continue
                
                # Process based on outcome
                if trade['pnl'] < -self.genome["profit_threshold"] / 2:
                    # Significant loss
                    self._process_loss_trade(trade_features, abs(trade['pnl']), trade)
                    losses_learned += 1
                elif trade['pnl'] > self.genome["profit_threshold"]:
                    # Significant win
                    self._process_win_trade(trade_features, trade['pnl'], trade)
                    wins_learned += 1
            
            # Update performance metrics
            self._mistake_performance['total_losses_learned'] += losses_learned
            self._mistake_performance['total_wins_learned'] += wins_learned
            
            return {
                'learning_processed': True,
                'losses_learned': losses_learned,
                'wins_learned': wins_learned,
                'total_loss_memories': len(self._loss_buf),
                'total_win_memories': len(self._win_buf)
            }
            
        except Exception as e:
            self.logger.error(f"Learning data processing failed: {e}")
            return {'learning_processed': False, 'error': str(e)}

    def _extract_trade_features(self, trade: Dict[str, Any], market_context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from trade data"""
        try:
            # Start with basic trade features
            features = []
            
            # Trade characteristics
            if 'confidence' in trade:
                features.append(trade['confidence'])
            if 'volume' in trade:
                features.append(trade.get('volume', 1.0))
            if 'duration' in trade:
                features.append(trade.get('duration', 1.0))
            
            # Market context features
            if 'volatility' in market_context:
                vol = market_context['volatility']
                if isinstance(vol, dict):
                    features.extend(list(vol.values())[:3])
                else:
                    features.append(float(vol))
            
            # Session and regime features
            if 'session' in market_context:
                session_map = {'asian': 0.0, 'european': 0.5, 'us': 1.0}
                features.append(session_map.get(market_context['session'], 0.25))
            
            if 'regime' in market_context:
                regime_map = {'trending': 1.0, 'ranging': 0.0, 'volatile': 0.5}
                features.append(regime_map.get(market_context['regime'], 0.25))
            
            # Ensure minimum feature length
            while len(features) < 8:
                features.append(0.0)
            
            return np.array(features[:20])  # Limit to 20 features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def _process_loss_trade(self, features: np.ndarray, loss_amount: float, trade_info: Dict[str, Any]):
        """Process a loss trade for learning"""
        try:
            # Create loss entry
            loss_entry = (features, loss_amount, trade_info)
            
            # Add to loss buffer
            self._loss_buf.append(loss_entry)
            
            # Limit buffer size
            if len(self._loss_buf) > self.genome["max_mistakes"]:
                self._loss_buf.pop(0)
            
            # Update consecutive losses
            self._consecutive_losses += 1
            
            # Extract and record pattern
            pattern = self._extract_pattern(features, trade_info)
            if pattern:
                self._record_loss_pattern(pattern, loss_amount)
            
            # Log significant loss
            if loss_amount > self.genome["profit_threshold"]:
                self.logger.warning(
                    format_operator_message(
                        "💸", "SIGNIFICANT_LOSS_LEARNED",
                        loss_amount=f"{loss_amount:.2f}",
                        consecutive_losses=self._consecutive_losses,
                        pattern=pattern[:10] if pattern else "unknown",
                        context="loss_learning"
                    )
                )
            
        except Exception as e:
            self.logger.error(f"Loss trade processing failed: {e}")

    def _process_win_trade(self, features: np.ndarray, profit_amount: float, trade_info: Dict[str, Any]):
        """Process a win trade for learning"""
        try:
            # Create win entry
            win_entry = (features, profit_amount, trade_info)
            
            # Add to win buffer
            self._win_buf.append(win_entry)
            
            # Limit buffer size
            if len(self._win_buf) > self.genome["max_mistakes"]:
                self._win_buf.pop(0)
            
            # Reset consecutive losses
            self._consecutive_losses = 0
            
            # Extract and record pattern
            pattern = self._extract_pattern(features, trade_info)
            if pattern:
                self._record_win_pattern(pattern, profit_amount)
            
        except Exception as e:
            self.logger.error(f"Win trade processing failed: {e}")

    def _extract_pattern(self, features: np.ndarray, trade_info: Dict[str, Any]) -> Optional[str]:
        """Extract pattern signature from trade"""
        try:
            # Simple pattern extraction based on feature discretization
            pattern_elements = []
            
            # Discretize first few features
            for i, feature in enumerate(features[:5]):
                if feature > 0.7:
                    pattern_elements.append(f"H{i}")  # High
                elif feature < 0.3:
                    pattern_elements.append(f"L{i}")  # Low
                else:
                    pattern_elements.append(f"M{i}")  # Medium
            
            # Add trade info if available
            if 'action' in trade_info:
                action = trade_info['action']
                if isinstance(action, (list, np.ndarray)) and len(action) > 0:
                    if action[0] > 0.5:
                        pattern_elements.append("BUY")
                    elif action[0] < -0.5:
                        pattern_elements.append("SELL")
                    else:
                        pattern_elements.append("HOLD")
            
            return "_".join(pattern_elements) if pattern_elements else None
            
        except Exception as e:
            self.logger.error(f"Pattern extraction failed: {e}")
            return None

    def _record_loss_pattern(self, pattern: str, loss_amount: float):
        """Record loss pattern for analysis"""
        if pattern not in self._loss_patterns:
            self._loss_patterns[pattern] = {'count': 0, 'total_severity': 0.0, 'last_seen': time.time()}
        
        self._loss_patterns[pattern]['count'] += 1
        self._loss_patterns[pattern]['total_severity'] += loss_amount
        self._loss_patterns[pattern]['last_seen'] = time.time()

    def _record_win_pattern(self, pattern: str, profit_amount: float):
        """Record win pattern for analysis"""
        if pattern not in self._win_patterns:
            self._win_patterns[pattern] = {'count': 0, 'total_profitability': 0.0, 'last_seen': time.time()}
        
        self._win_patterns[pattern]['count'] += 1
        self._win_patterns[pattern]['total_profitability'] += profit_amount
        self._win_patterns[pattern]['last_seen'] = time.time()

    async def _update_clustering(self) -> Dict[str, Any]:
        """Update clustering models if needed"""
        try:
            clustering_updated = False
            
            # Check if we need to update clustering
            total_samples = len(self._loss_buf) + len(self._win_buf)
            if total_samples % self.genome["cluster_update_threshold"] == 0 and total_samples > 0:
                clustering_result = await self._perform_clustering()
                clustering_updated = True
                self._mistake_performance['cluster_updates'] += 1
                return {
                    'clustering_updated': clustering_updated,
                    'clustering_result': clustering_result
                }
            
            return {'clustering_updated': False}
            
        except Exception as e:
            self.logger.error(f"Clustering update failed: {e}")
            return {'clustering_updated': False, 'error': str(e)}

    async def _perform_clustering(self) -> Dict[str, Any]:
        """Perform clustering on loss and win data"""
        try:
            results = {}
            
            # Cluster loss data
            if len(self._loss_buf) >= self.config.min_samples_for_clustering:
                loss_result = await self._cluster_loss_data()
                results['loss_clustering'] = loss_result
            
            # Cluster win data
            if len(self._win_buf) >= self.config.min_samples_for_clustering:
                win_result = await self._cluster_win_data()
                results['win_clustering'] = win_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {'error': str(e)}

    async def _cluster_loss_data(self) -> Dict[str, Any]:
        """Cluster loss data to identify danger zones"""
        try:
            # Extract features and weights
            features = np.array([entry[0] for entry in self._loss_buf])
            losses = np.array([entry[1] for entry in self._loss_buf])
            
            # Standardize features
            features_scaled = self._scaler.fit_transform(features)
            
            # Perform clustering
            n_clusters = min(self.genome["n_clusters"], len(features))
            self._km_loss = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self._km_loss.fit_predict(features_scaled)
            
            # Calculate cluster quality
            if len(np.unique(cluster_labels)) > 1:
                quality_score = silhouette_score(features_scaled, cluster_labels)
                self._cluster_quality_scores.append(quality_score)
            else:
                quality_score = 0.0
            
            # Identify danger zones (cluster centers)
            self._danger_zones = self._km_loss.cluster_centers_.tolist()
            
            # Calculate cluster severities
            cluster_severities = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_severities[i] = np.mean(losses[cluster_mask])
            
            return {
                'n_clusters': n_clusters,
                'quality_score': quality_score,
                'cluster_severities': cluster_severities,
                'danger_zones_count': len(self._danger_zones)
            }
            
        except Exception as e:
            self.logger.error(f"Loss clustering failed: {e}")
            return {'error': str(e)}

    async def _cluster_win_data(self) -> Dict[str, Any]:
        """Cluster win data to identify profit zones"""
        try:
            # Extract features and weights
            features = np.array([entry[0] for entry in self._win_buf])
            profits = np.array([entry[1] for entry in self._win_buf])
            
            # Standardize features
            features_scaled = self._scaler.transform(features)
            
            # Perform clustering
            n_clusters = min(self.genome["n_clusters"], len(features))
            self._km_win = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self._km_win.fit_predict(features_scaled)
            
            # Calculate cluster quality
            if len(np.unique(cluster_labels)) > 1:
                quality_score = silhouette_score(features_scaled, cluster_labels)
            else:
                quality_score = 0.0
            
            # Identify profit zones (cluster centers)
            self._profit_zones = self._km_win.cluster_centers_.tolist()
            
            # Calculate cluster profitabilities
            cluster_profitabilities = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_profitabilities[i] = np.mean(profits[cluster_mask])
            
            return {
                'n_clusters': n_clusters,
                'quality_score': quality_score,
                'cluster_profitabilities': cluster_profitabilities,
                'profit_zones_count': len(self._profit_zones)
            }
            
        except Exception as e:
            self.logger.error(f"Win clustering failed: {e}")
            return {'error': str(e)}

    async def _calculate_avoidance_signals(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate avoidance signals based on current features"""
        try:
            current_features = learning_data.get('current_features')
            if current_features is None:
                return {'avoidance_signal': 0.0, 'danger_similarity': 0.0}
            
            # Calculate similarity to danger zones
            danger_similarity = self._calculate_danger_similarity(current_features)
            
            # Calculate similarity to profit zones
            profit_similarity = self._calculate_profit_similarity(current_features)
            
            # Calculate avoidance signal
            avoidance_signal = danger_similarity * self.genome["avoidance_sensitivity"]
            
            # Adjust for profit zones
            if profit_similarity > danger_similarity:
                avoidance_signal *= 0.5  # Reduce avoidance if in profit zone
            
            # Apply consecutive loss penalty
            if self._consecutive_losses > 2:
                avoidance_signal *= (1.0 + self._consecutive_losses * 0.1)
            
            self._avoidance_signal = avoidance_signal
            
            return {
                'avoidance_signal': avoidance_signal,
                'danger_similarity': danger_similarity,
                'profit_similarity': profit_similarity,
                'consecutive_losses': self._consecutive_losses
            }
            
        except Exception as e:
            self.logger.error(f"Avoidance signal calculation failed: {e}")
            return {'avoidance_signal': 0.0, 'error': str(e)}

    def _calculate_danger_similarity(self, features: np.ndarray) -> float:
        """Calculate similarity to danger zones"""
        if not self._danger_zones or self._km_loss is None:
            return 0.0
        
        try:
            # Standardize features
            features_scaled = self._scaler.transform(features.reshape(1, -1))
            
            # Calculate distances to danger zone centers
            distances = []
            for center in self._danger_zones:
                distance = np.linalg.norm(features_scaled[0] - center)
                distances.append(distance)
            
            # Return inverse of minimum distance (higher = more similar)
            min_distance = min(distances)
            return 1.0 / (1.0 + min_distance)
            
        except Exception as e:
            self.logger.error(f"Danger similarity calculation failed: {e}")
            return 0.0

    def _calculate_profit_similarity(self, features: np.ndarray) -> float:
        """Calculate similarity to profit zones"""
        if not self._profit_zones or self._km_win is None:
            return 0.0
        
        try:
            # Standardize features
            features_scaled = self._scaler.transform(features.reshape(1, -1))
            
            # Calculate distances to profit zone centers
            distances = []
            for center in self._profit_zones:
                distance = np.linalg.norm(features_scaled[0] - center)
                distances.append(distance)
            
            # Return inverse of minimum distance (higher = more similar)
            min_distance = min(distances)
            return 1.0 / (1.0 + min_distance)
            
        except Exception as e:
            self.logger.error(f"Profit similarity calculation failed: {e}")
            return 0.0

    async def _generate_mistake_thesis(self, learning_data: Dict[str, Any], 
                                     learning_result: Dict[str, Any]) -> str:
        """Generate comprehensive mistake memory thesis"""
        try:
            # Memory statistics
            total_losses = len(self._loss_buf)
            total_wins = len(self._win_buf)
            consecutive_losses = self._consecutive_losses
            
            # Avoidance metrics
            avoidance_signal = learning_result.get('avoidance_signal', 0.0)
            danger_similarity = learning_result.get('danger_similarity', 0.0)
            
            # Pattern analysis
            loss_patterns = len(self._loss_patterns)
            win_patterns = len(self._win_patterns)
            
            thesis_parts = [
                f"Mistake Memory Analysis: {total_losses} losses and {total_wins} wins stored for pattern recognition",
                f"Avoidance system: {avoidance_signal:.3f} signal strength with {consecutive_losses} consecutive losses",
                f"Pattern recognition: {loss_patterns} loss patterns and {win_patterns} win patterns identified"
            ]
            
            # Clustering status
            clustering_updated = learning_result.get('clustering_updated', False)
            if clustering_updated:
                clustering_result = learning_result.get('clustering_result', {})
                danger_zones = clustering_result.get('loss_clustering', {}).get('danger_zones_count', 0)
                profit_zones = clustering_result.get('win_clustering', {}).get('profit_zones_count', 0)
                thesis_parts.append(f"Clustering updated: {danger_zones} danger zones and {profit_zones} profit zones identified")
            
            # Danger assessment
            if danger_similarity > 0.5:
                thesis_parts.append(f"HIGH DANGER: Current situation similar to loss patterns (similarity: {danger_similarity:.2f})")
            elif danger_similarity > 0.3:
                thesis_parts.append(f"MODERATE RISK: Some similarity to loss patterns detected")
            else:
                thesis_parts.append("LOW RISK: Current situation differs from known loss patterns")
            
            # Learning effectiveness
            if self._cluster_quality_scores:
                avg_quality = np.mean(list(self._cluster_quality_scores)[-3:])
                thesis_parts.append(f"Learning quality: {avg_quality:.2f} clustering effectiveness")
            
            # Consecutive loss warning
            if consecutive_losses > 3:
                thesis_parts.append(f"WARNING: {consecutive_losses} consecutive losses - elevated avoidance measures active")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Mistake thesis generation failed: {str(e)} - Loss avoidance system maintaining basic functionality"

    async def _update_mistake_smart_bus(self, learning_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with mistake memory results"""
        try:
            # Mistake avoidance data
            avoidance_data = {
                'avoidance_signal': self._avoidance_signal,
                'consecutive_losses': self._consecutive_losses,
                'danger_zones_count': len(self._danger_zones),
                'profit_zones_count': len(self._profit_zones),
                'total_loss_memories': len(self._loss_buf),
                'total_win_memories': len(self._win_buf)
            }
            
            self.smart_bus.set(
                'mistake_avoidance',
                avoidance_data,
                module='MistakeMemory',
                thesis=thesis
            )
            
            # Danger zones
            danger_zones_data = {
                'zones': self._danger_zones,
                'zone_count': len(self._danger_zones),
                'avoidance_sensitivity': self.genome["avoidance_sensitivity"],
                'last_updated': time.time()
            }
            
            self.smart_bus.set(
                'danger_zones',
                danger_zones_data,
                module='MistakeMemory',
                thesis=f"Identified {len(self._danger_zones)} danger zones from clustering analysis"
            )
            
            # Pattern recognition
            pattern_data = {
                'loss_patterns': dict(list(self._loss_patterns.items())[:10]),  # Top 10 patterns
                'win_patterns': dict(list(self._win_patterns.items())[:10]),
                'total_loss_patterns': len(self._loss_patterns),
                'total_win_patterns': len(self._win_patterns),
                'pattern_memory_size': self.genome["pattern_memory_size"]
            }
            
            self.smart_bus.set(
                'pattern_recognition',
                pattern_data,
                module='MistakeMemory',
                thesis="Pattern recognition from trading outcomes for loss avoidance"
            )
            
            # Loss prevention metrics
            prevention_data = {
                'avoidance_effectiveness': self._mistake_performance['avoidance_effectiveness'],
                'false_positive_rate': self._false_positive_rate,
                'true_positive_rate': self._true_positive_rate,
                'cluster_quality': np.mean(list(self._cluster_quality_scores)) if self._cluster_quality_scores else 0.0,
                'learning_samples': len(self._loss_buf) + len(self._win_buf)
            }
            
            self.smart_bus.set(
                'loss_prevention',
                prevention_data,
                module='MistakeMemory',
                thesis="Loss prevention effectiveness and learning metrics"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no learning data is available"""
        self.logger.warning("No learning data available - using cached mistake memory")
        
        return {
            'avoidance_signal': self._avoidance_signal,
            'total_loss_memories': len(self._loss_buf),
            'total_win_memories': len(self._win_buf),
            'consecutive_losses': self._consecutive_losses,
            'fallback_reason': 'no_learning_data'
        }

    async def _handle_mistake_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle mistake memory errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "MistakeMemory")
        explanation = self.english_explainer.explain_error(
            "MistakeMemory", str(error), "mistake learning"
        )
        
        self.logger.error(
            format_operator_message(
                "💥", "MISTAKE_MEMORY_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="mistake_learning"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'avoidance_signal': self._avoidance_signal,
            'total_loss_memories': len(self._loss_buf),
            'total_win_memories': len(self._win_buf),
            'consecutive_losses': self._consecutive_losses,
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    def _update_mistake_health(self):
        """Update mistake memory health metrics"""
        try:
            # Check clustering quality
            if self._cluster_quality_scores:
                avg_quality = np.mean(list(self._cluster_quality_scores)[-3:])
                if avg_quality < self.config.min_cluster_quality:
                    self._health_status = 'warning'
                else:
                    self._health_status = 'healthy'
            
            # Check consecutive losses
            if self._consecutive_losses > 5:
                self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_avoidance_effectiveness(self):
        """Analyze avoidance effectiveness"""
        try:
            if len(self._learning_effectiveness) >= 10:
                # Calculate effectiveness metrics
                recent_effectiveness = list(self._learning_effectiveness)[-10:]
                avg_effectiveness = np.mean(recent_effectiveness)
                
                self._mistake_performance['avoidance_effectiveness'] = avg_effectiveness
                
                if avg_effectiveness > 0.7:
                    self.logger.info(
                        format_operator_message(
                            "🛡️", "HIGH_AVOIDANCE_EFFECTIVENESS",
                            effectiveness=f"{avg_effectiveness:.2f}",
                            danger_zones=len(self._danger_zones),
                            context="avoidance_analysis"
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Avoidance effectiveness analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'MistakeMemory', 'learning_cycle', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'MistakeMemory', 'learning_cycle', 0, False
        )

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'loss_buffer': [(entry[0].tolist(), entry[1], entry[2]) for entry in self._loss_buf[-50:]],
            'win_buffer': [(entry[0].tolist(), entry[1], entry[2]) for entry in self._win_buf[-50:]],
            'loss_patterns': self._loss_patterns.copy(),
            'win_patterns': self._win_patterns.copy(),
            'danger_zones': self._danger_zones,
            'profit_zones': self._profit_zones,
            'genome': self.genome.copy(),
            'consecutive_losses': self._consecutive_losses,
            'avoidance_signal': self._avoidance_signal,
            'mistake_performance': self._mistake_performance.copy(),
            'circuit_breaker': self.circuit_breaker.copy(),
            'health_status': self._health_status
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'loss_buffer' in state:
            self._loss_buf = [(np.array(entry[0]), entry[1], entry[2]) for entry in state['loss_buffer']]
        
        if 'win_buffer' in state:
            self._win_buf = [(np.array(entry[0]), entry[1], entry[2]) for entry in state['win_buffer']]
        
        if 'loss_patterns' in state:
            self._loss_patterns = state['loss_patterns']
        
        if 'win_patterns' in state:
            self._win_patterns = state['win_patterns']
        
        if 'danger_zones' in state:
            self._danger_zones = state['danger_zones']
        
        if 'profit_zones' in state:
            self._profit_zones = state['profit_zones']
        
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'consecutive_losses' in state:
            self._consecutive_losses = state['consecutive_losses']
        
        if 'avoidance_signal' in state:
            self._avoidance_signal = state['avoidance_signal']
        
        if 'mistake_performance' in state:
            self._mistake_performance.update(state['mistake_performance'])
        
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])
        
        if 'health_status' in state:
            self._health_status = state['health_status']

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'total_memories': len(self._loss_buf) + len(self._win_buf),
            'consecutive_losses': self._consecutive_losses,
            'avoidance_signal': self._avoidance_signal
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def check_similarity_to_mistakes(self, features: np.ndarray) -> float:
        """Legacy compatibility for mistake similarity check"""
        return self._calculate_danger_similarity(features)
    
    def propose_action(self, obs: Any = None, **kwargs) -> np.ndarray:
        """Legacy compatibility for action proposal"""
        # Return avoidance signal as action modification
        return np.array([-self._avoidance_signal, 0.0])
    
    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        # Return inverse of avoidance signal as confidence
        return max(0.0, 1.0 - self._avoidance_signal)