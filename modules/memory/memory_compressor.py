# ─────────────────────────────────────────────────────────────
# File: modules/memory/memory_compressor.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
from sklearn.decomposition import PCA
import datetime
import random

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import TradingMixin, AnalysisMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class MemoryCompressor(Module, TradingMixin, AnalysisMixin):
    """
    Enhanced memory compressor with infrastructure integration.
    Compresses trading experiences into actionable intuition vectors using PCA and pattern analysis.
    """
    
    def __init__(self, compress_interval: int = 10, n_components: int = 8, 
                 profit_threshold: float = 10.0, debug: bool = True,
                 genome: Optional[Dict[str, Any]] = None, **kwargs):
        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=500,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome, compress_interval, n_components, profit_threshold)
        
        # Enhanced state initialization
        self._initialize_module_state()
        
        self.log_operator_info(
            "Memory compressor initialized",
            compression_interval=self.compress_interval,
            n_components=self.n_components,
            profit_threshold=f"€{self.profit_threshold:.2f}",
            max_memory_size=self.max_memory_size
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict], compress_interval: int, 
                                    n_components: int, profit_threshold: float):
        """Initialize genome-based parameters"""
        if genome:
            self.compress_interval = int(genome.get("compress_interval", compress_interval))
            self.n_components = int(genome.get("n_components", n_components))
            self.profit_threshold = float(genome.get("profit_threshold", profit_threshold))
            self.max_memory_size = int(genome.get("max_memory_size", 1000))
            self.compression_ratio = float(genome.get("compression_ratio", 0.7))
            self.learning_rate = float(genome.get("learning_rate", 0.1))
            self.profit_weight = float(genome.get("profit_weight", 2.0))
            self.loss_avoidance_weight = float(genome.get("loss_avoidance_weight", 1.5))
        else:
            self.compress_interval = compress_interval
            self.n_components = n_components
            self.profit_threshold = profit_threshold
            self.max_memory_size = 1000
            self.compression_ratio = 0.7
            self.learning_rate = 0.1
            self.profit_weight = 2.0
            self.loss_avoidance_weight = 1.5

        # Store genome for evolution
        self.genome = {
            "compress_interval": self.compress_interval,
            "n_components": self.n_components,
            "profit_threshold": self.profit_threshold,
            "max_memory_size": self.max_memory_size,
            "compression_ratio": self.compression_ratio,
            "learning_rate": self.learning_rate,
            "profit_weight": self.profit_weight,
            "loss_avoidance_weight": self.loss_avoidance_weight
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_trading_state()
        self._initialize_analysis_state()
        
        # Memory storage
        self.profit_memory: List[Tuple[np.ndarray, float]] = []
        self.loss_memory: List[Tuple[np.ndarray, float]] = []
        
        # Compressed representations
        self.intuition_vector = np.zeros(self.n_components, np.float32)
        self.profit_direction = np.zeros(self.n_components, np.float32)
        self.loss_direction = np.zeros(self.n_components, np.float32)
        
        # Enhanced tracking
        self._compression_count = 0
        self._compression_history = deque(maxlen=50)
        self._intuition_evolution = deque(maxlen=100)
        self._pattern_strength_history = deque(maxlen=200)
        
        # Compression analytics
        self._compression_quality_scores = deque(maxlen=50)
        self._explained_variance_history = deque(maxlen=50)
        self._feature_importance_tracking = {}
        self._compression_efficiency = 0.0
        
        # Market context integration
        self._market_context_memory = deque(maxlen=200)
        self._context_profit_correlations = {}
        
        # Adaptive parameters
        self._adaptive_thresholds = {
            'min_compression_quality': 0.3,
            'feature_stability_threshold': 0.8,
            'pattern_confidence_threshold': 0.6
        }

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_trading_state()
        self._reset_analysis_state()
        
        # Clear memory storage
        self.profit_memory.clear()
        self.loss_memory.clear()
        
        # Reset compressed representations
        self.intuition_vector.fill(0.0)
        self.profit_direction.fill(0.0)
        self.loss_direction.fill(0.0)
        
        # Reset tracking
        self._compression_count = 0
        self._compression_history.clear()
        self._intuition_evolution.clear()
        self._pattern_strength_history.clear()
        self._compression_quality_scores.clear()
        self._explained_variance_history.clear()
        self._feature_importance_tracking.clear()
        self._compression_efficiency = 0.0
        self._market_context_memory.clear()
        self._context_profit_correlations.clear()

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Return current intuition vector (main functionality)
        current_intuition = self._get_current_intuition()
        
        # Track intuition evolution
        self._track_intuition_evolution(current_intuition)
        
        # Update pattern strength
        self._update_pattern_strength()

    def _get_current_intuition(self) -> np.ndarray:
        """Get current intuition vector with enhanced blending"""
        
        try:
            # Base intuition from profit patterns
            base_intuition = self.intuition_vector.copy()
            
            # Blend with profit direction if available
            if np.linalg.norm(self.profit_direction) > 0:
                profit_strength = np.linalg.norm(self.profit_direction)
                alpha = min(0.8, profit_strength)  # Weight towards profit
                
                combined = alpha * self.profit_direction + (1 - alpha) * base_intuition
                
                # Normalize to prevent explosion
                norm = np.linalg.norm(combined)
                if norm > 1e-8:
                    combined = combined / norm
                else:
                    combined = base_intuition
                
                return combined.astype(np.float32)
            
            return base_intuition
            
        except Exception as e:
            self.log_operator_warning(f"Intuition calculation failed: {e}")
            return self.intuition_vector.copy()

    def _track_intuition_evolution(self, current_intuition: np.ndarray):
        """Track how intuition evolves over time"""
        
        # Store evolution
        self._intuition_evolution.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'step': self._step_count,
            'intuition': current_intuition.copy(),
            'profit_strength': np.linalg.norm(self.profit_direction),
            'loss_strength': np.linalg.norm(self.loss_direction)
        })
        
        # Calculate evolution metrics
        if len(self._intuition_evolution) >= 2:
            prev_intuition = self._intuition_evolution[-2]['intuition']
            change = np.linalg.norm(current_intuition - prev_intuition)
            
            # Log significant changes
            if change > 0.1:
                self.log_operator_info(
                    f"Significant intuition evolution",
                    change=f"{change:.3f}",
                    profit_strength=f"{np.linalg.norm(self.profit_direction):.3f}",
                    loss_strength=f"{np.linalg.norm(self.loss_direction):.3f}"
                )

    def _update_pattern_strength(self):
        """Update pattern strength metrics"""
        
        try:
            # Calculate current pattern strength
            profit_strength = np.linalg.norm(self.profit_direction)
            loss_strength = np.linalg.norm(self.loss_direction)
            
            # Pattern clarity (difference between profit and loss directions)
            if profit_strength > 0 and loss_strength > 0:
                pattern_clarity = profit_strength / (loss_strength + 1e-8)
            else:
                pattern_clarity = 1.0
            
            # Combined pattern strength
            pattern_strength = profit_strength * pattern_clarity
            
            self._pattern_strength_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'profit_strength': profit_strength,
                'loss_strength': loss_strength,
                'pattern_clarity': pattern_clarity,
                'combined_strength': pattern_strength
            })
            
            # Update performance metrics
            self._update_performance_metric('pattern_strength', pattern_strength)
            self._update_performance_metric('pattern_clarity', pattern_clarity)
            
        except Exception as e:
            self.log_operator_warning(f"Pattern strength update failed: {e}")

    def compress(self, episode: int, trades: List[Dict], info_bus: Optional[InfoBus] = None):
        """Enhanced compression with InfoBus context integration"""
        
        try:
            self.log_operator_info(
                f"Starting compression for episode {episode}",
                trades_count=len(trades),
                compression_number=self._compression_count + 1
            )
            
            # Extract market context if available
            market_context = self._extract_market_context(info_bus)
            
            # Process trades into memory
            profit_added, loss_added = self._process_trades_into_memory(trades, market_context)
            
            # Perform compression if at interval
            if episode % self.compress_interval == 0:
                self._perform_compression(episode, market_context)
            
            self.log_operator_info(
                f"Episode {episode} compression completed",
                profit_trades_added=profit_added,
                loss_trades_added=loss_added,
                total_profit_memories=len(self.profit_memory),
                total_loss_memories=len(self.loss_memory)
            )
            
        except Exception as e:
            self.log_operator_error(f"Compression failed: {e}")
            self._update_health_status("DEGRADED", f"Compression failed: {e}")

    def _extract_market_context(self, info_bus: Optional[InfoBus]) -> Dict[str, Any]:
        """Extract market context for compression"""
        
        if not info_bus:
            return {}
        
        return {
            'regime': InfoBusExtractor.get_market_regime(info_bus),
            'volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
            'session': InfoBusExtractor.get_session(info_bus),
            'drawdown_pct': InfoBusExtractor.get_drawdown_pct(info_bus),
            'exposure_pct': InfoBusExtractor.get_exposure_pct(info_bus),
            'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat())
        }

    def _process_trades_into_memory(self, trades: List[Dict], market_context: Dict[str, Any]) -> Tuple[int, int]:
        """Process trades into profit and loss memories"""
        
        profit_added = 0
        loss_added = 0
        
        for trade in trades:
            if "features" not in trade or "pnl" not in trade:
                continue
            
            try:
                # Extract and validate features
                features = np.asarray(trade["features"], dtype=np.float32)
                if features.size == 0:
                    continue
                
                # Resize features to match n_components
                if features.size != self.n_components:
                    if features.size > self.n_components:
                        features = features[:self.n_components]
                    else:
                        padding = np.zeros(self.n_components - features.size, dtype=np.float32)
                        features = np.concatenate([features, padding])
                
                pnl = float(trade["pnl"])
                
                # Add market context to features if available
                enhanced_features = self._enhance_features_with_context(features, market_context)
                
                # Categorize and store
                if pnl > self.profit_threshold:
                    self.profit_memory.append((enhanced_features, pnl))
                    profit_added += 1
                    
                    # Update trading metrics
                    self._update_trading_metrics({'pnl': pnl})
                    
                elif pnl < -1.0:  # Significant loss
                    self.loss_memory.append((enhanced_features, abs(pnl)))
                    loss_added += 1
                    
                    # Update trading metrics
                    self._update_trading_metrics({'pnl': pnl})
                
            except Exception as e:
                self.log_operator_warning(f"Trade processing failed: {e}")
                continue
        
        # Trim memory if needed
        self._trim_memory_buffers()
        
        # Store market context
        if market_context:
            self._market_context_memory.append({
                'context': market_context,
                'profit_trades': profit_added,
                'loss_trades': loss_added
            })
        
        return profit_added, loss_added

    def _enhance_features_with_context(self, features: np.ndarray, market_context: Dict[str, Any]) -> np.ndarray:
        """Enhance features with market context information"""
        
        if not market_context:
            return features
        
        try:
            # Create context encoding
            context_features = []
            
            # Regime encoding
            regime = market_context.get('regime', 'unknown')
            regime_encoding = {'trending': 1.0, 'volatile': 0.5, 'ranging': 0.0, 'unknown': 0.25}.get(regime, 0.25)
            context_features.append(regime_encoding)
            
            # Volatility encoding
            vol_level = market_context.get('volatility_level', 'medium')
            vol_encoding = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0}.get(vol_level, 0.5)
            context_features.append(vol_encoding)
            
            # Risk encoding
            drawdown = market_context.get('drawdown_pct', 0) / 100.0  # Normalize
            exposure = market_context.get('exposure_pct', 0) / 100.0   # Normalize
            context_features.extend([drawdown, exposure])
            
            # Blend context with features (small influence)
            context_array = np.array(context_features, dtype=np.float32)
            context_influence = 0.1  # 10% influence
            
            # Apply context influence to first few components
            enhanced_features = features.copy()
            for i, ctx_val in enumerate(context_array):
                if i < len(enhanced_features):
                    enhanced_features[i] = ((1 - context_influence) * enhanced_features[i] + 
                                          context_influence * ctx_val)
            
            return enhanced_features
            
        except Exception as e:
            self.log_operator_warning(f"Feature enhancement failed: {e}")
            return features

    def _trim_memory_buffers(self):
        """Trim memory buffers to prevent overflow"""
        
        # Trim profit memory
        if len(self.profit_memory) > self.max_memory_size:
            removed = len(self.profit_memory) - self.max_memory_size
            self.profit_memory = self.profit_memory[-self.max_memory_size:]
            
        # Trim loss memory
        if len(self.loss_memory) > self.max_memory_size:
            removed = len(self.loss_memory) - self.max_memory_size
            self.loss_memory = self.loss_memory[-self.max_memory_size:]

    def _perform_compression(self, episode: int, market_context: Dict[str, Any]):
        """Perform PCA compression with enhanced analytics"""
        
        self._compression_count += 1
        
        try:
            self.log_operator_info(
                f"Performing compression #{self._compression_count}",
                episode=episode,
                profit_memories=len(self.profit_memory),
                loss_memories=len(self.loss_memory)
            )
            
            compression_results = {}
            
            # Compress profit patterns
            if len(self.profit_memory) >= 5:
                compression_results['profit'] = self._compress_profit_patterns()
            
            # Compress loss patterns
            if len(self.loss_memory) >= 5:
                compression_results['loss'] = self._compress_loss_patterns()
            
            # Update main intuition vector
            if compression_results:
                self._update_intuition_vector(compression_results)
            
            # Record compression
            self._record_compression(episode, compression_results, market_context)
            
        except Exception as e:
            self.log_operator_error(f"Compression execution failed: {e}")

    def _compress_profit_patterns(self) -> Dict[str, Any]:
        """Compress profit patterns using weighted PCA"""
        
        try:
            # Prepare data
            profit_vectors = []
            weights = []
            
            # Use recent profitable memories
            recent_memories = self.profit_memory[-min(200, len(self.profit_memory)):]
            
            for features, pnl in recent_memories:
                profit_vectors.append(features)
                # Weight by profit amount and recency
                weight = pnl * self.profit_weight
                weights.append(weight)
            
            if not profit_vectors:
                return {}
            
            X_profit = np.vstack(profit_vectors)
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            # Weighted average for profit direction
            old_direction = self.profit_direction.copy()
            self.profit_direction = np.average(X_profit, axis=0, weights=weights)
            
            # PCA for main components
            compression_quality = 0.0
            explained_variance = 0.0
            
            if X_profit.shape[0] > max(3, self.n_components):
                try:
                    n_comp = min(self.n_components, X_profit.shape[0] - 1)
                    pca = PCA(n_components=n_comp)
                    pca.fit(X_profit)
                    
                    # Update intuition with first principal component
                    if hasattr(pca, 'components_') and len(pca.components_) > 0:
                        # Blend old intuition with new
                        new_component = pca.components_[0]
                        if len(new_component) == len(self.intuition_vector):
                            self.intuition_vector = (
                                (1 - self.learning_rate) * self.intuition_vector + 
                                self.learning_rate * new_component
                            )
                        
                        explained_variance = pca.explained_variance_ratio_[0]
                        compression_quality = explained_variance
                        
                        self.log_operator_info(
                            f"Profit PCA completed",
                            n_components=n_comp,
                            explained_variance=f"{explained_variance:.3f}",
                            samples=len(X_profit)
                        )
                
                except Exception as e:
                    self.log_operator_warning(f"Profit PCA failed: {e}")
            
            # Calculate direction change
            direction_change = np.linalg.norm(self.profit_direction - old_direction)
            
            return {
                'direction_change': direction_change,
                'compression_quality': compression_quality,
                'explained_variance': explained_variance,
                'samples_processed': len(X_profit),
                'avg_profit': np.mean([pnl for _, pnl in recent_memories])
            }
            
        except Exception as e:
            self.log_operator_error(f"Profit pattern compression failed: {e}")
            return {}

    def _compress_loss_patterns(self) -> Dict[str, Any]:
        """Compress loss patterns for avoidance"""
        
        try:
            # Prepare loss data
            loss_vectors = [features for features, _ in self.loss_memory[-min(200, len(self.loss_memory)):]]
            
            if not loss_vectors:
                return {}
            
            X_loss = np.vstack(loss_vectors)
            
            # Update loss direction (patterns to avoid)
            old_loss_direction = self.loss_direction.copy()
            self.loss_direction = np.mean(X_loss, axis=0)
            
            # Calculate avoidance strength
            loss_change = np.linalg.norm(self.loss_direction - old_loss_direction)
            
            self.log_operator_info(
                f"Loss pattern compression completed",
                direction_change=f"{loss_change:.3f}",
                samples=len(X_loss),
                avg_loss=f"{np.mean([loss for _, loss in self.loss_memory[-min(50, len(self.loss_memory)):]]):.2f}"
            )
            
            return {
                'direction_change': loss_change,
                'samples_processed': len(X_loss),
                'avg_loss': np.mean([loss for _, loss in self.loss_memory[-min(50, len(self.loss_memory)):]])
            }
            
        except Exception as e:
            self.log_operator_error(f"Loss pattern compression failed: {e}")
            return {}

    def _update_intuition_vector(self, compression_results: Dict[str, Any]):
        """Update main intuition vector based on compression results"""
        
        try:
            # Calculate intuition quality
            intuition_quality = 0.0
            
            if 'profit' in compression_results:
                profit_quality = compression_results['profit'].get('compression_quality', 0.0)
                intuition_quality += profit_quality * 0.8
            
            if 'loss' in compression_results:
                # Loss patterns add to quality through avoidance
                loss_samples = compression_results['loss'].get('samples_processed', 0)
                loss_quality = min(0.2, loss_samples / 100.0)
                intuition_quality += loss_quality * 0.2
            
            # Store quality score
            self._compression_quality_scores.append(intuition_quality)
            
            # Update compression efficiency
            if len(self.profit_memory) + len(self.loss_memory) > 0:
                compression_ratio = self.n_components / (len(self.profit_memory) + len(self.loss_memory))
                self._compression_efficiency = min(1.0, compression_ratio * intuition_quality)
            
            # Update performance metrics
            self._update_performance_metric('intuition_quality', intuition_quality)
            self._update_performance_metric('compression_efficiency', self._compression_efficiency)
            
        except Exception as e:
            self.log_operator_warning(f"Intuition vector update failed: {e}")

    def _record_compression(self, episode: int, compression_results: Dict[str, Any], 
                          market_context: Dict[str, Any]):
        """Record compression for analysis"""
        
        compression_record = {
            'episode': episode,
            'compression_count': self._compression_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'results': compression_results.copy(),
            'market_context': market_context.copy(),
            'memory_sizes': {
                'profit': len(self.profit_memory),
                'loss': len(self.loss_memory)
            },
            'intuition_norm': np.linalg.norm(self.intuition_vector),
            'profit_direction_norm': np.linalg.norm(self.profit_direction),
            'loss_direction_norm': np.linalg.norm(self.loss_direction)
        }
        
        self._compression_history.append(compression_record)

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with comprehensive metrics"""
        
        try:
            # Base intuition vector
            base_observation = self.intuition_vector.copy()
            
            # Compression metrics
            profit_strength = np.linalg.norm(self.profit_direction)
            loss_strength = np.linalg.norm(self.loss_direction)
            
            # Pattern clarity
            if profit_strength > 0 and loss_strength > 0:
                pattern_clarity = profit_strength / (loss_strength + 1e-8)
            else:
                pattern_clarity = 1.0
            
            # Memory utilization
            memory_utilization = (len(self.profit_memory) + len(self.loss_memory)) / (2 * self.max_memory_size)
            
            # Compression quality
            avg_compression_quality = (np.mean(list(self._compression_quality_scores)) 
                                     if self._compression_quality_scores else 0.0)
            
            # Learning progress
            learning_progress = min(1.0, self._compression_count / 20.0)  # Normalize
            
            # Enhanced metrics
            meta_metrics = np.array([
                profit_strength,
                loss_strength,
                pattern_clarity,
                memory_utilization,
                avg_compression_quality,
                learning_progress,
                self._compression_efficiency
            ], dtype=np.float32)
            
            # Combine all components
            observation = np.concatenate([base_observation, meta_metrics])
            
            return observation.astype(np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(self.n_components + 7, dtype=np.float32)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Propose actions based on compressed intuition"""
        
        # Determine action dimension
        action_dim = 2
        if hasattr(obs, 'shape') and len(obs.shape) > 0:
            action_dim = obs.shape[0]
        
        # Get current intuition
        current_intuition = self._get_current_intuition()
        
        # Scale intuition to action space
        if len(current_intuition) > 0:
            # Use first components for action direction
            action_influence = current_intuition[:min(action_dim, len(current_intuition))]
            
            # Extend or truncate to match action_dim
            if len(action_influence) < action_dim:
                padding = np.zeros(action_dim - len(action_influence), dtype=np.float32)
                action_influence = np.concatenate([action_influence, padding])
            else:
                action_influence = action_influence[:action_dim]
            
            # Scale to reasonable action range
            action_influence = np.tanh(action_influence) * 0.3  # Limit to ±0.3
            
            return action_influence.astype(np.float32)
        
        return np.zeros(action_dim, dtype=np.float32)

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return confidence in compressed patterns"""
        
        base_confidence = 0.5
        
        # Confidence from compression quality
        if self._compression_quality_scores:
            avg_quality = np.mean(list(self._compression_quality_scores))
            base_confidence += avg_quality * 0.3
        
        # Confidence from pattern strength
        profit_strength = np.linalg.norm(self.profit_direction)
        if profit_strength > 0.1:
            base_confidence += min(0.2, profit_strength)
        
        # Confidence from compression count (experience)
        experience_bonus = min(0.2, self._compression_count / 50.0)
        base_confidence += experience_bonus
        
        # Confidence from memory size
        if len(self.profit_memory) > 20:
            base_confidence += 0.1
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with validation"""
        self.compress_interval = int(np.clip(genome.get("compress_interval", self.compress_interval), 5, 50))
        self.n_components = int(np.clip(genome.get("n_components", self.n_components), 3, 20))
        self.profit_threshold = float(np.clip(genome.get("profit_threshold", self.profit_threshold), 1.0, 50.0))
        self.max_memory_size = int(np.clip(genome.get("max_memory_size", self.max_memory_size), 100, 2000))
        self.compression_ratio = float(np.clip(genome.get("compression_ratio", self.compression_ratio), 0.1, 1.0))
        self.learning_rate = float(np.clip(genome.get("learning_rate", self.learning_rate), 0.01, 0.5))
        self.profit_weight = float(np.clip(genome.get("profit_weight", self.profit_weight), 0.5, 5.0))
        self.loss_avoidance_weight = float(np.clip(genome.get("loss_avoidance_weight", self.loss_avoidance_weight), 0.5, 3.0))
        
        # Update genome
        self.genome = {
            "compress_interval": self.compress_interval,
            "n_components": self.n_components,
            "profit_threshold": self.profit_threshold,
            "max_memory_size": self.max_memory_size,
            "compression_ratio": self.compression_ratio,
            "learning_rate": self.learning_rate,
            "profit_weight": self.profit_weight,
            "loss_avoidance_weight": self.loss_avoidance_weight
        }
        
        # Resize vectors if n_components changed
        if len(self.intuition_vector) != self.n_components:
            old_intuition = self.intuition_vector.copy()
            old_profit = self.profit_direction.copy()
            old_loss = self.loss_direction.copy()
            
            self.intuition_vector = np.zeros(self.n_components, np.float32)
            self.profit_direction = np.zeros(self.n_components, np.float32)
            self.loss_direction = np.zeros(self.n_components, np.float32)
            
            # Copy over compatible components
            min_size = min(len(old_intuition), self.n_components)
            self.intuition_vector[:min_size] = old_intuition[:min_size]
            self.profit_direction[:min_size] = old_profit[:min_size]
            self.loss_direction[:min_size] = old_loss[:min_size]
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with performance-based adaptation"""
        g = self.genome.copy()
        mutations = []
        
        if np.random.rand() < mutation_rate:
            old_val = g["compress_interval"]
            g["compress_interval"] = int(np.clip(old_val + np.random.randint(-3, 4), 5, 50))
            mutations.append(f"interval: {old_val} → {g['compress_interval']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["n_components"]
            g["n_components"] = int(np.clip(old_val + np.random.randint(-1, 2), 3, 20))
            mutations.append(f"components: {old_val} → {g['n_components']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["profit_threshold"]
            g["profit_threshold"] = float(np.clip(old_val + np.random.uniform(-2, 2), 1.0, 50.0))
            mutations.append(f"threshold: {old_val:.1f} → {g['profit_threshold']:.1f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["learning_rate"]
            g["learning_rate"] = float(np.clip(old_val + np.random.uniform(-0.02, 0.02), 0.01, 0.5))
            mutations.append(f"lr: {old_val:.3f} → {g['learning_rate']:.3f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["profit_weight"]
            g["profit_weight"] = float(np.clip(old_val + np.random.uniform(-0.3, 0.3), 0.5, 5.0))
            mutations.append(f"profit_weight: {old_val:.2f} → {g['profit_weight']:.2f}")
        
        if mutations:
            self.log_operator_info(f"Memory compressor mutation applied", changes=", ".join(mutations))
            
        # Also mutate the intuition vector slightly
        if np.random.rand() < mutation_rate * 0.5:
            noise = np.random.normal(0, 0.05, self.intuition_vector.shape).astype(np.float32)
            self.intuition_vector += noise
            
        self.set_genome(g)
        
    def crossover(self, other: "MemoryCompressor") -> "MemoryCompressor":
        """Enhanced crossover with performance-based selection"""
        if not isinstance(other, MemoryCompressor):
            self.log_operator_warning("Crossover with incompatible type")
            return self
        
        # Performance-based crossover
        self_quality = np.mean(list(self._compression_quality_scores)) if self._compression_quality_scores else 0.0
        other_quality = np.mean(list(other._compression_quality_scores)) if other._compression_quality_scores else 0.0
        
        # Favor higher quality parent
        if self_quality > other_quality:
            bias = 0.7  # Favor self
        else:
            bias = 0.3  # Favor other
        
        new_g = {k: (self.genome[k] if np.random.rand() < bias else other.genome[k]) for k in self.genome}
        
        child = MemoryCompressor(genome=new_g, debug=self.config.debug)
        
        # Cross intuition vectors
        if len(self.intuition_vector) == len(other.intuition_vector):
            mask = np.random.rand(*self.intuition_vector.shape) > 0.5
            child.intuition_vector = np.where(mask, self.intuition_vector, other.intuition_vector)
            child.profit_direction = np.where(mask, self.profit_direction, other.profit_direction)
        
        return child

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check vector dimensions
            if len(self.intuition_vector) != self.n_components:
                return False
            if len(self.profit_direction) != self.n_components:
                return False
            if len(self.loss_direction) != self.n_components:
                return False
                
            # Check vector validity
            if not np.all(np.isfinite(self.intuition_vector)):
                return False
            if not np.all(np.isfinite(self.profit_direction)):
                return False
            if not np.all(np.isfinite(self.loss_direction)):
                return False
            
            # Check memory sizes
            if len(self.profit_memory) > self.max_memory_size * 1.1:  # Allow small overflow
                return False
            if len(self.loss_memory) > self.max_memory_size * 1.1:
                return False
            
            # Check compression count
            if self._compression_count < 0:
                return False
                
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        compressor_details = {
            'compression_info': {
                'total_compressions': self._compression_count,
                'compression_interval': self.compress_interval,
                'n_components': self.n_components,
                'compression_efficiency': self._compression_efficiency
            },
            'memory_info': {
                'profit_memories': len(self.profit_memory),
                'loss_memories': len(self.loss_memory),
                'max_memory_size': self.max_memory_size,
                'memory_utilization': (len(self.profit_memory) + len(self.loss_memory)) / (2 * self.max_memory_size)
            },
            'pattern_info': {
                'intuition_norm': np.linalg.norm(self.intuition_vector),
                'profit_direction_norm': np.linalg.norm(self.profit_direction),
                'loss_direction_norm': np.linalg.norm(self.loss_direction),
                'avg_compression_quality': (np.mean(list(self._compression_quality_scores)) 
                                           if self._compression_quality_scores else 0.0)
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(compressor_details)
            return base_details
        
        return compressor_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        
        # Convert memory to serializable format
        profit_memory_serializable = [(vec.tolist(), pnl) for vec, pnl in self.profit_memory[-100:]]  # Keep recent
        loss_memory_serializable = [(vec.tolist(), loss) for vec, loss in self.loss_memory[-100:]]
        
        return {
            "profit_memory": profit_memory_serializable,
            "loss_memory": loss_memory_serializable,
            "intuition_vector": self.intuition_vector.tolist(),
            "profit_direction": self.profit_direction.tolist(),
            "loss_direction": self.loss_direction.tolist(),
            "compression_count": self._compression_count,
            "genome": self.genome.copy(),
            "compression_history": list(self._compression_history)[-20:],  # Keep recent only
            "compression_quality_scores": list(self._compression_quality_scores)[-30:],
            "compression_efficiency": self._compression_efficiency,
            "pattern_strength_history": list(self._pattern_strength_history)[-50:],
            "adaptive_thresholds": self._adaptive_thresholds.copy()
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        # Restore memory
        profit_memory_data = module_state.get("profit_memory", [])
        self.profit_memory = [(np.asarray(vec, np.float32), pnl) for vec, pnl in profit_memory_data]
        
        loss_memory_data = module_state.get("loss_memory", [])
        self.loss_memory = [(np.asarray(vec, np.float32), loss) for vec, loss in loss_memory_data]
        
        # Restore vectors
        self.intuition_vector = np.asarray(module_state.get("intuition_vector", 
            np.zeros(self.n_components)), np.float32)
        self.profit_direction = np.asarray(module_state.get("profit_direction", 
            np.zeros(self.n_components)), np.float32)
        self.loss_direction = np.asarray(module_state.get("loss_direction", 
            np.zeros(self.n_components)), np.float32)
        
        # Restore other state
        self._compression_count = module_state.get("compression_count", 0)
        self.set_genome(module_state.get("genome", self.genome))
        self._compression_history = deque(module_state.get("compression_history", []), maxlen=50)
        self._compression_quality_scores = deque(module_state.get("compression_quality_scores", []), maxlen=50)
        self._compression_efficiency = module_state.get("compression_efficiency", 0.0)
        self._pattern_strength_history = deque(module_state.get("pattern_strength_history", []), maxlen=200)
        self._adaptive_thresholds = module_state.get("adaptive_thresholds", self._adaptive_thresholds)

    def get_compression_analysis_report(self) -> str:
        """Generate operator-friendly compression analysis report"""
        
        # Current pattern strengths
        profit_strength = np.linalg.norm(self.profit_direction)
        loss_strength = np.linalg.norm(self.loss_direction)
        intuition_strength = np.linalg.norm(self.intuition_vector)
        
        # Pattern clarity
        if profit_strength > 0 and loss_strength > 0:
            pattern_clarity = profit_strength / (loss_strength + 1e-8)
        else:
            pattern_clarity = 1.0
        
        # Memory utilization
        memory_util = (len(self.profit_memory) + len(self.loss_memory)) / (2 * self.max_memory_size)
        
        # Compression quality
        avg_quality = (np.mean(list(self._compression_quality_scores)) 
                      if self._compression_quality_scores else 0.0)
        
        return f"""
🧠 MEMORY COMPRESSOR ANALYSIS
═══════════════════════════════════════
🔄 Compressions Performed: {self._compression_count}
💡 Intuition Strength: {intuition_strength:.3f}
📊 Compression Quality: {avg_quality:.3f}
⚡ Efficiency: {self._compression_efficiency:.3f}

🎯 PATTERN ANALYSIS
• Profit Direction Strength: {profit_strength:.3f}
• Loss Direction Strength: {loss_strength:.3f}
• Pattern Clarity: {pattern_clarity:.3f}
• Components: {self.n_components}

💾 MEMORY STATUS
• Profit Memories: {len(self.profit_memory):,}/{self.max_memory_size:,}
• Loss Memories: {len(self.loss_memory):,}/{self.max_memory_size:,}
• Memory Utilization: {memory_util:.1%}
• Profit Threshold: €{self.profit_threshold:.2f}

🔧 COMPRESSION SETTINGS
• Compression Interval: {self.compress_interval} episodes
• Learning Rate: {self.learning_rate:.3f}
• Profit Weight: {self.profit_weight:.2f}
• Loss Avoidance Weight: {self.loss_avoidance_weight:.2f}

📈 PERFORMANCE METRICS
• Trading Records: {self._trades_processed}
• Pattern Evolution Points: {len(self._intuition_evolution)}
• Quality Score History: {len(self._compression_quality_scores)}
        """

    # Maintain backward compatibility
    def step(self, *args, **kwargs):
        """Backward compatibility step method"""
        return self._step_impl(None, **kwargs)

    def get_state(self):
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state):
        """Backward compatibility state method"""
        super().set_state(state)