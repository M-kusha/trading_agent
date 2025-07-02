# ─────────────────────────────────────────────────────────────
# File: modules/memory/mistake_memory.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
from sklearn.cluster import KMeans
import datetime
import random

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import TradingMixin, RiskMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class MistakeMemory(Module, TradingMixin, RiskMixin):
    """
    Enhanced mistake memory with infrastructure integration.
    Learns from both losses and wins using clustering to identify danger and profit zones.
    """
    
    def __init__(self, max_mistakes: int = 100, n_clusters: int = 5, 
                 profit_threshold: float = 10.0, debug: bool = True,
                 genome: Optional[Dict[str, Any]] = None, **kwargs):
        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=300,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome, max_mistakes, n_clusters, profit_threshold)
        
        # Enhanced state initialization
        self._initialize_module_state()
        
        self.log_operator_info(
            "Mistake memory initialized",
            max_mistakes=self.max_mistakes,
            n_clusters=self.n_clusters,
            profit_threshold=f"€{self.profit_threshold:.2f}",
            avoidance_learning="enabled"
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict], max_mistakes: int, 
                                    n_clusters: int, profit_threshold: float):
        """Initialize genome-based parameters"""
        if genome:
            self.max_mistakes = int(genome.get("max_mistakes", max_mistakes))
            self.n_clusters = int(genome.get("n_clusters", n_clusters))
            self.profit_threshold = float(genome.get("profit_threshold", profit_threshold))
            self.cluster_update_threshold = int(genome.get("cluster_update_threshold", 10))
            self.avoidance_sensitivity = float(genome.get("avoidance_sensitivity", 1.0))
            self.pattern_memory_size = int(genome.get("pattern_memory_size", 50))
            self.danger_zone_weight = float(genome.get("danger_zone_weight", 2.0))
        else:
            self.max_mistakes = max_mistakes
            self.n_clusters = n_clusters
            self.profit_threshold = profit_threshold
            self.cluster_update_threshold = 10
            self.avoidance_sensitivity = 1.0
            self.pattern_memory_size = 50
            self.danger_zone_weight = 2.0

        # Store genome for evolution
        self.genome = {
            "max_mistakes": self.max_mistakes,
            "n_clusters": self.n_clusters,
            "profit_threshold": self.profit_threshold,
            "cluster_update_threshold": self.cluster_update_threshold,
            "avoidance_sensitivity": self.avoidance_sensitivity,
            "pattern_memory_size": self.pattern_memory_size,
            "danger_zone_weight": self.danger_zone_weight
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_trading_state()
        self._initialize_risk_state()
        
        # Memory buffers for wins and losses
        self._loss_buf: List[Tuple[np.ndarray, float, Dict]] = []
        self._win_buf: List[Tuple[np.ndarray, float, Dict]] = []
        
        # Clustering models
        self._km_loss: Optional[KMeans] = None
        self._km_win: Optional[KMeans] = None
        
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

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_trading_state()
        self._reset_risk_state()
        
        # Clear memory buffers
        self._loss_buf.clear()
        self._win_buf.clear()
        
        # Reset clustering
        self._km_loss = None
        self._km_win = None
        self._mean_dist = 0.0
        self._last_dist = 0.0
        self._danger_zones.clear()
        self._profit_zones.clear()
        
        # Reset patterns and signals
        self._consecutive_losses = 0
        self._loss_patterns.clear()
        self._win_patterns.clear()
        self._avoidance_signal = 0.0
        
        # Reset analytics
        self._pattern_evolution.clear()
        self._danger_zone_violations.clear()
        self._learning_effectiveness.clear()
        self._market_context_correlations.clear()
        self._cluster_quality_scores.clear()
        self._prediction_accuracy = 0.0
        self._false_positive_rate = 0.0
        self._true_positive_rate = 0.0

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract trading data for learning
        trading_data = self._extract_trading_data(info_bus, kwargs)
        
        # Process learning from trades
        if trading_data.get('source') != 'insufficient_data':
            self._process_learning_data(trading_data)
        
        # Update avoidance signals
        self._update_avoidance_signals()

    def _extract_trading_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trading data from InfoBus or kwargs"""
        
        # Try InfoBus first
        if info_bus:
            # Extract recent trades for learning
            recent_trades = info_bus.get('recent_trades', [])
            market_context = info_bus.get('market_context', {})
            
            # Extract features from current market state
            current_features = self._extract_features_from_info_bus(info_bus)
            
            return {
                'recent_trades': recent_trades,
                'current_features': current_features,
                'market_context': market_context,
                'step_idx': info_bus.get('step_idx', 0),
                'regime': InfoBusExtractor.get_market_regime(info_bus),
                'volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
                'session': InfoBusExtractor.get_session(info_bus),
                'source': 'info_bus'
            }
        
        # Try kwargs (backward compatibility)
        extracted_data = {}
        
        if "trades" in kwargs:
            extracted_data['trades'] = kwargs["trades"]
            
        if "features" in kwargs and "pnl" in kwargs:
            extracted_data.update({
                'features': kwargs["features"],
                'pnl': kwargs["pnl"],
                'info': kwargs.get("info", {})
            })
            
        if extracted_data:
            extracted_data['source'] = 'kwargs'
            return extracted_data
        
        # Return insufficient data marker
        return {'source': 'insufficient_data'}

    def _extract_features_from_info_bus(self, info_bus: InfoBus) -> np.ndarray:
        """Extract features from InfoBus for mistake learning"""
        
        features = []
        
        # Market regime features
        regime = InfoBusExtractor.get_market_regime(info_bus)
        regime_encoding = {'trending': 1.0, 'volatile': 0.5, 'ranging': 0.0, 'unknown': 0.25}.get(regime, 0.25)
        features.append(regime_encoding)
        
        # Volatility features
        vol_level = InfoBusExtractor.get_volatility_level(info_bus)
        vol_encoding = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0}.get(vol_level, 0.5)
        features.append(vol_encoding)
        
        # Risk features
        drawdown = InfoBusExtractor.get_drawdown_pct(info_bus) / 100.0
        exposure = InfoBusExtractor.get_exposure_pct(info_bus) / 100.0
        features.extend([drawdown, exposure])
        
        # Session features
        session = InfoBusExtractor.get_session(info_bus)
        session_encoding = {'asian': 0.2, 'european': 0.5, 'american': 0.8, 'closed': 0.0}.get(session, 0.5)
        features.append(session_encoding)
        
        # Position and market stress features
        position_count = InfoBusExtractor.get_position_count(info_bus)
        features.append(min(1.0, position_count / 5.0))  # Normalize to 0-1
        
        # Add price momentum if available
        market_context = info_bus.get('market_context', {})
        if 'volatility' in market_context:
            vol_data = market_context['volatility']
            if isinstance(vol_data, dict):
                avg_vol = np.mean(list(vol_data.values()))
                features.append(min(1.0, avg_vol * 50))  # Normalize
            else:
                features.append(min(1.0, float(vol_data) * 50))
        else:
            features.append(0.5)
        
        # Extend to minimum feature size
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)  # Cap at 20 features

    def _process_learning_data(self, trading_data: Dict[str, Any]):
        """Process trading data for mistake learning"""
        
        try:
            # Process batch trades if available
            if 'trades' in trading_data:
                trades = trading_data['trades']
                self.log_operator_info(f"Processing batch of {len(trades)} trades for learning")
                
                processed_count = 0
                for trade in trades:
                    if self._process_individual_trade(trade, trading_data.get('market_context', {})):
                        processed_count += 1
                
                self.log_operator_info(f"Batch learning completed: {processed_count}/{len(trades)} trades processed")
                
            # Process individual trade data
            elif 'features' in trading_data and 'pnl' in trading_data:
                features = trading_data['features']
                pnl = trading_data['pnl']
                info = trading_data.get('info', {})
                
                # Enhance info with market context
                enhanced_info = info.copy()
                enhanced_info.update(trading_data.get('market_context', {}))
                
                self._process_individual_trade({
                    'features': features,
                    'pnl': pnl,
                    'info': enhanced_info
                }, trading_data.get('market_context', {}))
                
            # Process recent trades from InfoBus
            elif 'recent_trades' in trading_data:
                recent_trades = trading_data['recent_trades']
                market_context = trading_data.get('market_context', {})
                
                for trade in recent_trades:
                    # Use InfoBus features for trade analysis
                    current_features = trading_data.get('current_features', np.zeros(10))
                    
                    enhanced_trade = {
                        'features': current_features,
                        'pnl': trade.get('pnl', 0),
                        'info': {**trade, **market_context}
                    }
                    
                    self._process_individual_trade(enhanced_trade, market_context)
            
        except Exception as e:
            self.log_operator_error(f"Learning data processing failed: {e}")

    def _process_individual_trade(self, trade_data: Dict, market_context: Dict) -> bool:
        """Process individual trade for mistake learning"""
        
        try:
            features = np.asarray(trade_data.get('features', []), dtype=np.float32)
            pnl = float(trade_data.get('pnl', 0))
            info = trade_data.get('info', {})
            
            if features.size == 0:
                return False
            
            # Ensure minimum feature size
            if features.size < 10:
                padding = np.zeros(10 - features.size, dtype=np.float32)
                features = np.concatenate([features, padding])
            
            # Create enhanced info with market context
            enhanced_info = {**info, **market_context}
            enhanced_info['timestamp'] = datetime.datetime.now().isoformat()
            
            entry = (features, pnl, enhanced_info)
            
            # Categorize and store trade
            if pnl < 0:
                self._process_loss_trade(entry)
            elif pnl > self.profit_threshold:
                self._process_win_trade(entry)
            
            # Update trading metrics
            self._update_trading_metrics({'pnl': pnl})
            
            # Refit clusters if enough data
            self._refit_clusters_if_needed()
            
            return True
            
        except Exception as e:
            self.log_operator_warning(f"Individual trade processing failed: {e}")
            return False

    def _process_loss_trade(self, entry: Tuple[np.ndarray, float, Dict]):
        """Process a losing trade for learning"""
        
        features, pnl, info = entry
        
        # Add to loss buffer
        self._loss_buf.append(entry)
        if len(self._loss_buf) > self.max_mistakes:
            removed = self._loss_buf.pop(0)
            
        self._consecutive_losses += 1
        
        # Extract and track loss pattern
        pattern = self._extract_pattern(features, info)
        current_time = datetime.datetime.now().isoformat()
        
        if pattern in self._loss_patterns:
            count, severity, _ = self._loss_patterns[pattern]
            new_severity = (severity * count + abs(pnl)) / (count + 1)
            self._loss_patterns[pattern] = (count + 1, new_severity, current_time)
        else:
            self._loss_patterns[pattern] = (1, abs(pnl), current_time)
        
        # Log significant losses
        if abs(pnl) > 20:  # Significant loss
            self.log_operator_warning(
                f"Significant loss recorded",
                pnl=f"€{pnl:.2f}",
                pattern=pattern,
                consecutive_losses=self._consecutive_losses,
                total_losses=len(self._loss_buf)
            )
        else:
            self.log_operator_info(
                f"Loss recorded",
                pnl=f"€{pnl:.2f}",
                pattern=pattern,
                consecutive_losses=self._consecutive_losses
            )
        
        # Update risk alerts
        if self._consecutive_losses >= 5:
            self._risk_alerts.append({
                'type': 'consecutive_losses',
                'count': self._consecutive_losses,
                'pattern': pattern,
                'timestamp': current_time
            })

    def _process_win_trade(self, entry: Tuple[np.ndarray, float, Dict]):
        """Process a winning trade for learning"""
        
        features, pnl, info = entry
        
        # Add to win buffer
        self._win_buf.append(entry)
        if len(self._win_buf) > self.max_mistakes // 2:
            removed = self._win_buf.pop(0)
        
        # Reset consecutive losses
        self._consecutive_losses = 0
        
        # Extract and track win pattern
        pattern = self._extract_pattern(features, info)
        current_time = datetime.datetime.now().isoformat()
        
        if pattern in self._win_patterns:
            count, profitability, _ = self._win_patterns[pattern]
            new_profitability = (profitability * count + pnl) / (count + 1)
            self._win_patterns[pattern] = (count + 1, new_profitability, current_time)
        else:
            self._win_patterns[pattern] = (1, pnl, current_time)
        
        self.log_operator_info(
            f"Profitable trade recorded",
            pnl=f"€{pnl:.2f}",
            pattern=pattern,
            total_wins=len(self._win_buf)
        )

    def _extract_pattern(self, features: np.ndarray, info: Dict) -> str:
        """Enhanced pattern extraction from features and context"""
        
        try:
            # Extract context information
            volatility = info.get("volatility", 0)
            regime = info.get("regime", "unknown")
            hour = info.get("hour", -1)
            session = info.get("session", "unknown")
            drawdown = info.get("drawdown_pct", 0)
            
            # Categorize volatility
            if isinstance(volatility, (int, float)):
                vol_level = "high" if volatility > 0.02 else "medium" if volatility > 0.01 else "low"
            else:
                vol_level = "unknown"
            
            # Categorize time session
            if hour >= 0:
                if 0 <= hour < 8:
                    time_session = "asian"
                elif 8 <= hour < 16:
                    time_session = "european"
                else:
                    time_session = "us"
            else:
                time_session = session if session != "unknown" else "unknown"
            
            # Risk level
            risk_level = "high" if drawdown > 5 else "medium" if drawdown > 2 else "low"
            
            # Feature-based pattern
            feature_pattern = ""
            if len(features) >= 3:
                # Simple feature binning
                f1_bin = "H" if features[0] > 0.6 else "M" if features[0] > 0.3 else "L"
                f2_bin = "H" if features[1] > 0.6 else "M" if features[1] > 0.3 else "L"
                f3_bin = "H" if features[2] > 0.6 else "M" if features[2] > 0.3 else "L"
                feature_pattern = f"{f1_bin}{f2_bin}{f3_bin}"
            
            # Combined pattern
            pattern = f"{regime}_{vol_level}_{time_session}_{risk_level}_{feature_pattern}"
            
            return pattern[:50]  # Limit length
            
        except Exception as e:
            self.log_operator_warning(f"Pattern extraction failed: {e}")
            return "unknown"

    def _refit_clusters_if_needed(self):
        """Refit clusters when sufficient new data is available"""
        
        if len(self._loss_buf) >= self.cluster_update_threshold:
            if len(self._loss_buf) % self.cluster_update_threshold == 0:
                self._fit_clusters()

    def _fit_clusters(self):
        """Enhanced cluster fitting with quality assessment"""
        
        try:
            cluster_results = {}
            
            # Fit loss clusters
            if len(self._loss_buf) >= self.n_clusters:
                loss_results = self._fit_loss_clusters()
                cluster_results['loss'] = loss_results
            
            # Fit win clusters
            if len(self._win_buf) >= 3:
                win_results = self._fit_win_clusters()
                cluster_results['win'] = win_results
            
            # Update avoidance effectiveness
            self._update_avoidance_effectiveness(cluster_results)
            
        except Exception as e:
            self.log_operator_error(f"Cluster fitting failed: {e}")

    def _fit_loss_clusters(self) -> Dict[str, Any]:
        """Fit clusters for loss patterns"""
        
        try:
            X_loss = np.stack([f for f, _, _ in self._loss_buf])
            
            # Fit clustering model
            self._km_loss = KMeans(
                n_clusters=min(self.n_clusters, len(X_loss)),
                n_init=10,
                random_state=42,
                max_iter=300
            )
            self._km_loss.fit(X_loss)
            
            # Store danger zones
            self._danger_zones = self._km_loss.cluster_centers_.copy()
            
            # Calculate distance metrics
            distances = self._km_loss.transform(X_loss)
            min_distances = distances.min(axis=1)
            
            self._mean_dist = float(min_distances.mean())
            self._last_dist = float(distances[-1].min()) if len(distances) > 0 else 0.0
            
            # Calculate clustering quality
            inertia = self._km_loss.inertia_
            silhouette_approx = 1.0 / (1.0 + inertia / len(X_loss))  # Approximation
            self._cluster_quality_scores.append(silhouette_approx)
            
            self.log_operator_info(
                f"Loss clusters fitted",
                n_clusters=len(self._danger_zones),
                samples=len(X_loss),
                mean_distance=f"{self._mean_dist:.3f}",
                quality_score=f"{silhouette_approx:.3f}"
            )
            
            return {
                'clusters': len(self._danger_zones),
                'samples': len(X_loss),
                'mean_distance': self._mean_dist,
                'quality': silhouette_approx
            }
            
        except Exception as e:
            self.log_operator_error(f"Loss cluster fitting failed: {e}")
            return {}

    def _fit_win_clusters(self) -> Dict[str, Any]:
        """Fit clusters for win patterns"""
        
        try:
            X_win = np.stack([f for f, _, _ in self._win_buf])
            
            # Determine number of clusters
            n_win_clusters = min(3, len(X_win))
            
            # Fit clustering model
            self._km_win = KMeans(
                n_clusters=n_win_clusters,
                n_init=10,
                random_state=42,
                max_iter=300
            )
            self._km_win.fit(X_win)
            
            # Store profit zones
            self._profit_zones = self._km_win.cluster_centers_.copy()
            
            self.log_operator_info(
                f"Win clusters fitted",
                n_clusters=n_win_clusters,
                samples=len(X_win),
                avg_profit=f"€{np.mean([pnl for _, pnl, _ in self._win_buf]):.2f}"
            )
            
            return {
                'clusters': n_win_clusters,
                'samples': len(X_win),
                'avg_profit': np.mean([pnl for _, pnl, _ in self._win_buf])
            }
            
        except Exception as e:
            self.log_operator_error(f"Win cluster fitting failed: {e}")
            return {}

    def _update_avoidance_signals(self):
        """Update comprehensive avoidance signals"""
        
        try:
            # Base signal from consecutive losses
            base_signal = min(self._consecutive_losses * 0.1, 0.5)
            
            # Pattern frequency signal
            pattern_signal = 0.0
            if self._loss_patterns:
                max_pattern_count = max(count for count, _, _ in self._loss_patterns.values())
                pattern_signal = min(max_pattern_count * 0.05, 0.3)
            
            # Risk alert signal
            risk_signal = 0.0
            if len(self._risk_alerts) > 0:
                recent_alerts = [alert for alert in self._risk_alerts 
                               if 'consecutive_losses' in alert.get('type', '')]
                risk_signal = min(len(recent_alerts) * 0.1, 0.2)
            
            # Combined avoidance signal
            old_signal = self._avoidance_signal
            self._avoidance_signal = min(base_signal + pattern_signal + risk_signal, 0.8)
            
            # Log significant changes
            if abs(self._avoidance_signal - old_signal) > 0.1:
                self.log_operator_info(
                    f"Avoidance signal updated",
                    old_signal=f"{old_signal:.3f}",
                    new_signal=f"{self._avoidance_signal:.3f}",
                    consecutive_losses=self._consecutive_losses,
                    pattern_count=len(self._loss_patterns)
                )
            
            # Update performance metrics
            self._update_performance_metric('avoidance_signal', self._avoidance_signal)
            self._update_performance_metric('consecutive_losses', self._consecutive_losses)
            
        except Exception as e:
            self.log_operator_warning(f"Avoidance signal update failed: {e}")

    def _update_avoidance_effectiveness(self, cluster_results: Dict[str, Any]):
        """Update learning effectiveness tracking"""
        
        try:
            effectiveness_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'step': self._step_count,
                'cluster_results': cluster_results,
                'loss_count': len(self._loss_buf),
                'win_count': len(self._win_buf),
                'avoidance_signal': self._avoidance_signal,
                'consecutive_losses': self._consecutive_losses
            }
            
            self._learning_effectiveness.append(effectiveness_data)
            
        except Exception as e:
            self.log_operator_warning(f"Effectiveness tracking update failed: {e}")

    def check_similarity_to_mistakes(self, features: np.ndarray, info_bus: Optional[InfoBus] = None) -> float:
        """Enhanced similarity check with market context"""
        
        try:
            if self._km_loss is None or len(self._danger_zones) == 0:
                return 0.0
            
            # Prepare features
            if features.size < 10:
                padding = np.zeros(10 - features.size, dtype=np.float32)
                features = np.concatenate([features, padding])
            
            features_2d = features.reshape(1, -1)
            
            # Distance to nearest danger zone
            if len(self._danger_zones) > 0:
                distances = np.linalg.norm(self._danger_zones - features_2d, axis=1)
                min_danger_dist = distances.min()
            else:
                min_danger_dist = float('inf')
            
            # Distance to nearest profit zone
            min_profit_dist = float('inf')
            if len(self._profit_zones) > 0:
                profit_distances = np.linalg.norm(self._profit_zones - features_2d, axis=1)
                min_profit_dist = profit_distances.min()
            
            # Calculate danger score
            danger_score = np.exp(-min_danger_dist * self.danger_zone_weight)
            
            # Increase danger if closer to losses than profits
            if min_profit_dist != float('inf') and min_danger_dist < min_profit_dist:
                danger_score *= 1.3
            
            # Apply avoidance sensitivity
            danger_score *= self.avoidance_sensitivity
            
            # Market context adjustment
            if info_bus:
                context_multiplier = self._get_context_danger_multiplier(info_bus)
                danger_score *= context_multiplier
            
            # Clip and track
            danger_score = float(np.clip(danger_score, 0, 1))
            
            # Track danger zone violations
            if danger_score > 0.6:
                self._danger_zone_violations.append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'danger_score': danger_score,
                    'min_danger_dist': min_danger_dist,
                    'consecutive_losses': self._consecutive_losses
                })
                
                self.log_operator_warning(
                    f"High danger zone proximity",
                    danger_score=f"{danger_score:.3f}",
                    min_distance=f"{min_danger_dist:.3f}",
                    consecutive_losses=self._consecutive_losses
                )
            
            return danger_score
            
        except Exception as e:
            self.log_operator_error(f"Similarity check failed: {e}")
            return 0.0

    def _get_context_danger_multiplier(self, info_bus: InfoBus) -> float:
        """Get danger multiplier based on market context"""
        
        multiplier = 1.0
        
        # High volatility increases danger
        vol_level = InfoBusExtractor.get_volatility_level(info_bus)
        if vol_level in ['high', 'extreme']:
            multiplier *= 1.2
        
        # High drawdown increases danger
        drawdown = InfoBusExtractor.get_drawdown_pct(info_bus)
        if drawdown > 10:
            multiplier *= 1.3
        elif drawdown > 5:
            multiplier *= 1.1
        
        # High exposure increases danger
        exposure = InfoBusExtractor.get_exposure_pct(info_bus)
        if exposure > 80:
            multiplier *= 1.2
        
        return multiplier

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with comprehensive metrics"""
        
        try:
            # Cluster information
            n_loss_clusters = float(len(self._danger_zones))
            n_win_clusters = float(len(self._profit_zones))
            
            # Distance metrics
            mean_dist = self._mean_dist
            last_dist = self._last_dist
            
            # Pattern analysis
            loss_pattern_diversity = float(len(self._loss_patterns))
            win_pattern_diversity = float(len(self._win_patterns))
            
            # Performance ratios
            total_trades = len(self._loss_buf) + len(self._win_buf)
            win_loss_ratio = len(self._win_buf) / max(1, len(self._loss_buf))
            memory_utilization = total_trades / max(1, self.max_mistakes)
            
            # Avoidance effectiveness
            avg_cluster_quality = (np.mean(list(self._cluster_quality_scores)) 
                                 if self._cluster_quality_scores else 0.0)
            
            # Risk indicators
            danger_violations = len(self._danger_zone_violations)
            
            # Combine all components
            observation = np.array([
                n_loss_clusters,
                mean_dist,
                last_dist,
                self._avoidance_signal,
                n_win_clusters,
                win_loss_ratio,
                loss_pattern_diversity,
                win_pattern_diversity,
                memory_utilization,
                avg_cluster_quality,
                float(self._consecutive_losses),
                float(danger_violations)
            ], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(12, dtype=np.float32)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Propose actions based on mistake avoidance"""
        
        # Determine action dimension
        action_dim = 2
        if hasattr(obs, 'shape') and len(obs.shape) > 0:
            action_dim = obs.shape[0]
        
        # Base action on avoidance signal
        avoidance_strength = self._avoidance_signal
        
        # Check current market similarity to mistakes
        current_features = np.zeros(10)
        if info_bus:
            current_features = self._extract_features_from_info_bus(info_bus)
        
        danger_score = self.check_similarity_to_mistakes(current_features, info_bus)
        
        # Combined avoidance signal
        total_avoidance = min(1.0, avoidance_strength + danger_score)
        
        # Generate conservative action when high avoidance
        if total_avoidance > 0.5:
            # Strong avoidance - recommend position reduction
            action = np.full(action_dim, -total_avoidance * 0.5, dtype=np.float32)
        elif total_avoidance > 0.3:
            # Moderate avoidance - reduce position sizes
            action = np.full(action_dim, -total_avoidance * 0.3, dtype=np.float32)
        else:
            # Low risk - neutral recommendation
            action = np.zeros(action_dim, dtype=np.float32)
        
        return action

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return confidence in mistake avoidance recommendations"""
        
        base_confidence = 0.5
        
        # Confidence from cluster quality
        if self._cluster_quality_scores:
            avg_quality = np.mean(list(self._cluster_quality_scores))
            base_confidence += avg_quality * 0.3
        
        # Confidence from data volume
        total_data = len(self._loss_buf) + len(self._win_buf)
        data_confidence = min(0.2, total_data / 100.0)
        base_confidence += data_confidence
        
        # Confidence from pattern diversity
        pattern_confidence = min(0.2, len(self._loss_patterns) / 20.0)
        base_confidence += pattern_confidence
        
        # Reduce confidence during high consecutive losses
        if self._consecutive_losses > 3:
            base_confidence *= 0.8
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with validation"""
        self.max_mistakes = int(np.clip(genome.get("max_mistakes", self.max_mistakes), 50, 500))
        self.n_clusters = int(np.clip(genome.get("n_clusters", self.n_clusters), 3, 15))
        self.profit_threshold = float(np.clip(genome.get("profit_threshold", self.profit_threshold), 1.0, 50.0))
        self.cluster_update_threshold = int(np.clip(genome.get("cluster_update_threshold", self.cluster_update_threshold), 5, 50))
        self.avoidance_sensitivity = float(np.clip(genome.get("avoidance_sensitivity", self.avoidance_sensitivity), 0.1, 3.0))
        self.pattern_memory_size = int(np.clip(genome.get("pattern_memory_size", self.pattern_memory_size), 20, 200))
        self.danger_zone_weight = float(np.clip(genome.get("danger_zone_weight", self.danger_zone_weight), 0.5, 5.0))
        
        self.genome = {
            "max_mistakes": self.max_mistakes,
            "n_clusters": self.n_clusters,
            "profit_threshold": self.profit_threshold,
            "cluster_update_threshold": self.cluster_update_threshold,
            "avoidance_sensitivity": self.avoidance_sensitivity,
            "pattern_memory_size": self.pattern_memory_size,
            "danger_zone_weight": self.danger_zone_weight
        }
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with performance-based adaptation"""
        g = self.genome.copy()
        mutations = []
        
        if np.random.rand() < mutation_rate:
            old_val = g["n_clusters"]
            g["n_clusters"] = int(np.clip(old_val + np.random.randint(-1, 2), 3, 15))
            mutations.append(f"n_clusters: {old_val} → {g['n_clusters']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["avoidance_sensitivity"]
            g["avoidance_sensitivity"] = float(np.clip(old_val + np.random.uniform(-0.2, 0.2), 0.1, 3.0))
            mutations.append(f"avoidance: {old_val:.2f} → {g['avoidance_sensitivity']:.2f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["danger_zone_weight"]
            g["danger_zone_weight"] = float(np.clip(old_val + np.random.uniform(-0.3, 0.3), 0.5, 5.0))
            mutations.append(f"danger_weight: {old_val:.2f} → {g['danger_zone_weight']:.2f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["cluster_update_threshold"]
            g["cluster_update_threshold"] = int(np.clip(old_val + np.random.randint(-3, 4), 5, 50))
            mutations.append(f"update_threshold: {old_val} → {g['cluster_update_threshold']}")
        
        if mutations:
            self.log_operator_info(f"Mistake memory mutation applied", changes=", ".join(mutations))
            
        # Also mutate cluster centers if they exist
        if np.random.rand() < mutation_rate * 0.3:
            noise_std = 0.05
            if self._km_loss is not None:
                noise = np.random.normal(0, noise_std, self._km_loss.cluster_centers_.shape)
                self._km_loss.cluster_centers_ += noise.astype(np.float32)
                self._danger_zones = self._km_loss.cluster_centers_.copy()
                
            if self._km_win is not None:
                noise = np.random.normal(0, noise_std, self._km_win.cluster_centers_.shape)
                self._km_win.cluster_centers_ += noise.astype(np.float32)
                self._profit_zones = self._km_win.cluster_centers_.copy()
        
        self.set_genome(g)
        
    def crossover(self, other: "MistakeMemory") -> "MistakeMemory":
        """Enhanced crossover with effectiveness-based selection"""
        if not isinstance(other, MistakeMemory):
            self.log_operator_warning("Crossover with incompatible type")
            return self
        
        # Performance-based crossover
        self_effectiveness = len(self._win_buf) / max(len(self._loss_buf), 1)
        other_effectiveness = len(other._win_buf) / max(len(other._loss_buf), 1)
        
        # Favor more effective parent
        if self_effectiveness > other_effectiveness:
            bias = 0.7  # Favor self
        else:
            bias = 0.3  # Favor other
        
        new_g = {k: (self.genome[k] if np.random.rand() < bias else other.genome[k]) for k in self.genome}
        
        child = MistakeMemory(genome=new_g, debug=self.config.debug)
        
        # Inherit cluster centers from better parent
        if self_effectiveness > other_effectiveness:
            if self._km_loss is not None:
                child._km_loss = self._km_loss
                child._danger_zones = self._danger_zones.copy()
            if self._km_win is not None:
                child._km_win = self._km_win
                child._profit_zones = self._profit_zones.copy()
        else:
            if other._km_loss is not None:
                child._km_loss = other._km_loss
                child._danger_zones = other._danger_zones.copy()
            if other._km_win is not None:
                child._km_win = other._km_win
                child._profit_zones = other._profit_zones.copy()
        
        return child

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check buffer sizes
            if len(self._loss_buf) > self.max_mistakes * 1.1:
                return False
            if len(self._win_buf) > (self.max_mistakes // 2) * 1.1:
                return False
                
            # Check clustering consistency
            if self._km_loss is not None and len(self._danger_zones) != self._km_loss.n_clusters:
                return False
            if self._km_win is not None and len(self._profit_zones) != self._km_win.n_clusters:
                return False
            
            # Check pattern tracking
            if self._consecutive_losses < 0:
                return False
            if not (0.0 <= self._avoidance_signal <= 1.0):
                return False
                
            # Check distance metrics
            if not np.isfinite(self._mean_dist) or not np.isfinite(self._last_dist):
                return False
                
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        mistake_details = {
            'memory_info': {
                'loss_memories': len(self._loss_buf),
                'win_memories': len(self._win_buf),
                'max_capacity': self.max_mistakes,
                'consecutive_losses': self._consecutive_losses,
                'avoidance_signal': self._avoidance_signal
            },
            'clustering_info': {
                'danger_zones': len(self._danger_zones),
                'profit_zones': len(self._profit_zones),
                'mean_distance': self._mean_dist,
                'last_distance': self._last_dist,
                'cluster_quality': (np.mean(list(self._cluster_quality_scores)) 
                                  if self._cluster_quality_scores else 0.0)
            },
            'pattern_info': {
                'loss_patterns': len(self._loss_patterns),
                'win_patterns': len(self._win_patterns),
                'danger_violations': len(self._danger_zone_violations),
                'learning_records': len(self._learning_effectiveness)
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(mistake_details)
            return base_details
        
        return mistake_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        
        # Convert buffers to serializable format
        loss_buf_serializable = [(f.tolist(), pnl, info) for f, pnl, info in self._loss_buf[-50:]]  # Keep recent
        win_buf_serializable = [(f.tolist(), pnl, info) for f, pnl, info in self._win_buf[-25:]]
        
        # Prepare clustering state
        km_loss_state = {}
        if self._km_loss is not None:
            km_loss_state = {
                "cluster_centers": self._km_loss.cluster_centers_.tolist(),
                "n_clusters": self._km_loss.n_clusters
            }
        
        km_win_state = {}
        if self._km_win is not None:
            km_win_state = {
                "cluster_centers": self._km_win.cluster_centers_.tolist(),
                "n_clusters": self._km_win.n_clusters
            }
        
        return {
            "loss_buf": loss_buf_serializable,
            "win_buf": win_buf_serializable,
            "km_loss": km_loss_state,
            "km_win": km_win_state,
            "mean_dist": self._mean_dist,
            "last_dist": self._last_dist,
            "consecutive_losses": self._consecutive_losses,
            "loss_patterns": dict(self._loss_patterns),
            "win_patterns": dict(self._win_patterns),
            "avoidance_signal": self._avoidance_signal,
            "genome": self.genome.copy(),
            "cluster_quality_scores": list(self._cluster_quality_scores)[-10:],  # Keep recent
            "danger_zone_violations": list(self._danger_zone_violations)[-20:],
            "learning_effectiveness": list(self._learning_effectiveness)[-30:]
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        # Restore memory buffers
        loss_buf_data = module_state.get("loss_buf", [])
        self._loss_buf = [(np.asarray(f, np.float32), pnl, info) for f, pnl, info in loss_buf_data]
        
        win_buf_data = module_state.get("win_buf", [])
        self._win_buf = [(np.asarray(f, np.float32), pnl, info) for f, pnl, info in win_buf_data]
        
        # Restore clustering models
        km_loss_data = module_state.get("km_loss", {})
        if "cluster_centers" in km_loss_data:
            self._km_loss = KMeans(
                n_clusters=km_loss_data["n_clusters"],
                n_init=10,
                random_state=42
            )
            self._km_loss.cluster_centers_ = np.asarray(km_loss_data["cluster_centers"], np.float32)
            self._danger_zones = self._km_loss.cluster_centers_.copy()
        
        km_win_data = module_state.get("km_win", {})
        if "cluster_centers" in km_win_data:
            self._km_win = KMeans(
                n_clusters=km_win_data["n_clusters"],
                n_init=10,
                random_state=42
            )
            self._km_win.cluster_centers_ = np.asarray(km_win_data["cluster_centers"], np.float32)
            self._profit_zones = self._km_win.cluster_centers_.copy()
        
        # Restore other state
        self._mean_dist = module_state.get("mean_dist", 0.0)
        self._last_dist = module_state.get("last_dist", 0.0)
        self._consecutive_losses = module_state.get("consecutive_losses", 0)
        self._loss_patterns = module_state.get("loss_patterns", {})
        self._win_patterns = module_state.get("win_patterns", {})
        self._avoidance_signal = module_state.get("avoidance_signal", 0.0)
        self.set_genome(module_state.get("genome", self.genome))
        self._cluster_quality_scores = deque(module_state.get("cluster_quality_scores", []), maxlen=20)
        self._danger_zone_violations = deque(module_state.get("danger_zone_violations", []), maxlen=50)
        self._learning_effectiveness = deque(module_state.get("learning_effectiveness", []), maxlen=200)

    def get_mistake_analysis_report(self) -> str:
        """Generate operator-friendly mistake analysis report"""
        
        # Current status
        total_memories = len(self._loss_buf) + len(self._win_buf)
        win_rate = len(self._win_buf) / max(total_memories, 1)
        
        # Pattern analysis
        top_loss_pattern = "None"
        if self._loss_patterns:
            top_pattern = max(self._loss_patterns.items(), key=lambda x: x[1][0])
            pattern_name, (count, severity, _) = top_pattern
            top_loss_pattern = f"{pattern_name} ({count}x, €{severity:.2f})"
        
        # Avoidance effectiveness
        if self._avoidance_signal > 0.6:
            avoidance_status = "🚨 High Alert"
        elif self._avoidance_signal > 0.3:
            avoidance_status = "⚠️ Moderate Caution"
        else:
            avoidance_status = "✅ Low Risk"
        
        # Clustering quality
        cluster_quality = (np.mean(list(self._cluster_quality_scores)) 
                         if self._cluster_quality_scores else 0.0)
        
        return f"""
🧠 MISTAKE MEMORY ANALYSIS
═══════════════════════════════════════
📊 Memory Status: {total_memories}/{self.max_mistakes} used
🎯 Win Rate: {win_rate:.1%}
🚨 Avoidance Status: {avoidance_status} ({self._avoidance_signal:.3f})
🔥 Consecutive Losses: {self._consecutive_losses}

🎯 CLUSTERING ANALYSIS
• Danger Zones: {len(self._danger_zones)} clusters
• Profit Zones: {len(self._profit_zones)} clusters  
• Cluster Quality: {cluster_quality:.3f}
• Mean Distance: {self._mean_dist:.3f}

📈 PATTERN RECOGNITION
• Loss Patterns: {len(self._loss_patterns)}
• Win Patterns: {len(self._win_patterns)}
• Top Loss Pattern: {top_loss_pattern}
• Danger Violations: {len(self._danger_zone_violations)}

🔧 CONFIGURATION
• Max Mistakes: {self.max_mistakes}
• Clusters: {self.n_clusters}
• Profit Threshold: €{self.profit_threshold:.2f}
• Avoidance Sensitivity: {self.avoidance_sensitivity:.2f}
• Danger Zone Weight: {self.danger_zone_weight:.2f}

💡 LEARNING EFFECTIVENESS
• Memory Records: {len(self._loss_buf)} losses, {len(self._win_buf)} wins
• Pattern Diversity: {len(self._loss_patterns) + len(self._win_patterns)} unique patterns
• Learning History: {len(self._learning_effectiveness)} effectiveness records
        """

    # Maintain backward compatibility
    def step(self, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, **kwargs)

    def get_state(self):
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state):
        """Backward compatibility state method"""
        super().set_state(state)