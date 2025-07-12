# ─────────────────────────────────────────────────────────────
# File: modules/memory/memory_compressor.py
# 🚀 PRODUCTION-READY Memory Compression System
# Advanced experience compression with SmartInfoBus integration
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@dataclass
class CompressorConfig:
    """Configuration for Memory Compressor"""
    compress_interval: int = 10
    n_components: int = 8
    profit_threshold: float = 10.0
    max_memory_size: int = 1000
    compression_ratio: float = 0.7
    learning_rate: float = 0.1
    
    # Weighting parameters
    profit_weight: float = 2.0
    loss_avoidance_weight: float = 1.5
    
    # Performance thresholds
    max_processing_time_ms: float = 300
    circuit_breaker_threshold: int = 3
    min_compression_quality: float = 0.3
    
    # Feature parameters
    feature_stability_threshold: float = 0.8
    pattern_confidence_threshold: float = 0.6


@module(
    name="MemoryCompressor",
    version="3.0.0",
    category="memory",
    provides=["intuition_vector", "compressed_patterns", "memory_compression", "feature_importance"],
    requires=["trades", "features", "market_context", "episode_data"],
    description="Advanced memory compression with PCA and pattern analysis for trading intuition",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class MemoryCompressor(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced memory compressor with SmartInfoBus integration.
    Compresses trading experiences into actionable intuition vectors using PCA and pattern analysis.
    """

    def __init__(self, 
                 config: Optional[CompressorConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        self.config = config or CompressorConfig()
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize compression state
        self._initialize_compression_state()
        
        self.logger.info(
            format_operator_message(
                "🗜️", "MEMORY_COMPRESSOR_INITIALIZED",
                details=f"Components: {self.config.n_components}, Interval: {self.config.compress_interval}",
                result="Memory compression system ready",
                context="memory_compression"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for memory compression"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="MemoryCompressor", 
            log_path="logs/memory_compression.log", 
            max_lines=3000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("MemoryCompressor", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for compression operations
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
                "compress_interval": int(genome.get("compress_interval", self.config.compress_interval)),
                "n_components": int(genome.get("n_components", self.config.n_components)),
                "profit_threshold": float(genome.get("profit_threshold", self.config.profit_threshold)),
                "max_memory_size": int(genome.get("max_memory_size", self.config.max_memory_size)),
                "compression_ratio": float(genome.get("compression_ratio", self.config.compression_ratio)),
                "learning_rate": float(genome.get("learning_rate", self.config.learning_rate)),
                "profit_weight": float(genome.get("profit_weight", self.config.profit_weight)),
                "loss_avoidance_weight": float(genome.get("loss_avoidance_weight", self.config.loss_avoidance_weight))
            }
        else:
            self.genome = {
                "compress_interval": self.config.compress_interval,
                "n_components": self.config.n_components,
                "profit_threshold": self.config.profit_threshold,
                "max_memory_size": self.config.max_memory_size,
                "compression_ratio": self.config.compression_ratio,
                "learning_rate": self.config.learning_rate,
                "profit_weight": self.config.profit_weight,
                "loss_avoidance_weight": self.config.loss_avoidance_weight
            }

    def _initialize_compression_state(self):
        """Initialize memory compression state"""
        # Memory storage
        self.profit_memory: List[Tuple[np.ndarray, float]] = []
        self.loss_memory: List[Tuple[np.ndarray, float]] = []
        
        # Compressed representations
        self.intuition_vector = np.zeros(self.genome["n_components"], np.float32)
        self.profit_direction = np.zeros(self.genome["n_components"], np.float32)
        self.loss_direction = np.zeros(self.genome["n_components"], np.float32)
        
        # PCA components
        self.profit_pca = PCA(n_components=self.genome["n_components"])
        self.loss_pca = PCA(n_components=self.genome["n_components"])
        self.scaler = StandardScaler()
        
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
        
        # Performance metrics
        self._compression_performance = {
            'episodes_processed': 0,
            'total_compressions': 0,
            'avg_compression_time': 0.0,
            'memory_utilization': 0.0
        }

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_compression_health()
                    self._analyze_compression_efficiency()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module"""
        try:
            # Set initial compression status in SmartInfoBus
            initial_status = {
                "compression_count": 0,
                "intuition_vector": self.intuition_vector.tolist(),
                "memory_size": {"profit": 0, "loss": 0},
                "compression_efficiency": 0.0
            }
            
            self.smart_bus.set(
                'memory_compression',
                initial_status,
                module='MemoryCompressor',
                thesis="Initial memory compression status"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process memory compression"""
        start_time = time.time()
        
        try:
            # Extract compression data
            compression_data = await self._extract_compression_data(**inputs)
            
            if not compression_data:
                return await self._handle_no_data_fallback()
            
            # Process memory updates
            memory_result = await self._process_memory_updates(compression_data)
            
            # Check if compression should be performed
            episode = compression_data.get('episode', 0)
            if self._should_compress(episode):
                compression_result = await self._perform_compression(compression_data)
                memory_result.update(compression_result)
            
            # Update intuition vector
            intuition_result = await self._update_intuition_vector()
            memory_result.update(intuition_result)
            
            # Generate thesis
            thesis = await self._generate_compression_thesis(compression_data, memory_result)
            
            # Update SmartInfoBus
            await self._update_compression_smart_bus(memory_result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return memory_result
            
        except Exception as e:
            return await self._handle_compression_error(e, start_time)

    async def _extract_compression_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract compression data from SmartInfoBus"""
        try:
            # Get trades data
            trades = self.smart_bus.get('trades', 'MemoryCompressor') or []
            
            # Get features
            features = self.smart_bus.get('features', 'MemoryCompressor')
            
            # Get market context
            market_context = self.smart_bus.get('market_context', 'MemoryCompressor') or {}
            
            # Get episode data
            episode_data = self.smart_bus.get('episode_data', 'MemoryCompressor') or {}
            
            # Get current episode number
            episode = inputs.get('episode', episode_data.get('episode', 0))
            
            return {
                'trades': trades,
                'features': features,
                'market_context': market_context,
                'episode_data': episode_data,
                'episode': episode,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract compression data: {e}")
            return None

    async def _process_memory_updates(self, compression_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory updates with new trading data"""
        try:
            trades = compression_data.get('trades', [])
            features = compression_data.get('features')
            market_context = compression_data.get('market_context', {})
            
            # Process trades into memory
            if trades and features is not None:
                profit_count, loss_count = self._process_trades_into_memory(trades, features, market_context)
            else:
                profit_count, loss_count = 0, 0
            
            # Update memory utilization
            total_memory = len(self.profit_memory) + len(self.loss_memory)
            memory_utilization = total_memory / self.genome["max_memory_size"]
            self._compression_performance['memory_utilization'] = memory_utilization
            
            # Trim memory if needed
            if total_memory > self.genome["max_memory_size"]:
                self._trim_memory_buffers()
            
            return {
                'memory_updates': {
                    'profit_added': profit_count,
                    'loss_added': loss_count,
                    'total_profit_memories': len(self.profit_memory),
                    'total_loss_memories': len(self.loss_memory),
                    'memory_utilization': memory_utilization
                }
            }
            
        except Exception as e:
            self.logger.error(f"Memory update processing failed: {e}")
            return {'memory_updates': {'error': str(e)}}

    def _process_trades_into_memory(self, trades: List[Dict[str, Any]], 
                                  features: np.ndarray, market_context: Dict[str, Any]) -> Tuple[int, int]:
        """Process trades into profit/loss memory with feature enhancement"""
        profit_count = 0
        loss_count = 0
        
        try:
            # Enhance features with market context
            enhanced_features = self._enhance_features_with_context(features, market_context)
            
            for trade in trades[-20:]:  # Process last 20 trades
                if not isinstance(trade, dict) or 'pnl' not in trade:
                    continue
                
                pnl = trade['pnl']
                
                # Create feature vector for this trade
                trade_features = enhanced_features.copy()
                
                # Add trade-specific features
                if 'confidence' in trade:
                    trade_features = np.append(trade_features, trade['confidence'])
                if 'volume' in trade:
                    trade_features = np.append(trade_features, trade.get('volume', 1.0))
                
                # Store in appropriate memory
                if pnl > self.genome["profit_threshold"]:
                    self.profit_memory.append((trade_features, pnl))
                    profit_count += 1
                elif pnl < -self.genome["profit_threshold"] / 2:  # Store significant losses
                    self.loss_memory.append((trade_features, abs(pnl)))
                    loss_count += 1
            
            return profit_count, loss_count
            
        except Exception as e:
            self.logger.error(f"Trade processing failed: {e}")
            return 0, 0

    def _enhance_features_with_context(self, features: np.ndarray, market_context: Dict[str, Any]) -> np.ndarray:
        """Enhance features with market context"""
        try:
            enhanced = features.copy() if features is not None else np.zeros(10)
            
            # Add market context features
            context_features = []
            
            # Volatility context
            if 'volatility' in market_context:
                vol = market_context['volatility']
                if isinstance(vol, dict):
                    context_features.extend(list(vol.values())[:3])
                else:
                    context_features.append(float(vol))
            
            # Session context
            if 'session' in market_context:
                session_map = {'asian': 0.0, 'european': 0.5, 'us': 1.0}
                context_features.append(session_map.get(market_context['session'], 0.25))
            
            # Trend context
            if 'trend' in market_context:
                trend = market_context['trend']
                if isinstance(trend, (int, float)):
                    context_features.append(float(trend))
                elif isinstance(trend, str):
                    trend_map = {'up': 1.0, 'down': -1.0, 'sideways': 0.0}
                    context_features.append(trend_map.get(trend, 0.0))
            
            # Combine features
            if context_features:
                enhanced = np.concatenate([enhanced, np.array(context_features)])
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Feature enhancement failed: {e}")
            return features if features is not None else np.zeros(10)

    def _should_compress(self, episode: int) -> bool:
        """Check if compression should be performed"""
        return (episode % self.genome["compress_interval"]) == 0 and episode > 0

    async def _perform_compression(self, compression_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform memory compression using PCA"""
        try:
            compression_results = {}
            
            # Compress profit patterns
            if len(self.profit_memory) >= self.genome["n_components"]:
                profit_result = self._compress_profit_patterns()
                compression_results.update(profit_result)
            
            # Compress loss patterns
            if len(self.loss_memory) >= self.genome["n_components"]:
                loss_result = self._compress_loss_patterns()
                compression_results.update(loss_result)
            
            # Record compression
            self._record_compression(compression_data, compression_results)
            
            self._compression_count += 1
            self._compression_performance['total_compressions'] += 1
            
            return {
                'compression_performed': True,
                'compression_results': compression_results,
                'compression_count': self._compression_count
            }
            
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return {'compression_performed': False, 'error': str(e)}

    def _compress_profit_patterns(self) -> Dict[str, Any]:
        """Compress profitable patterns using PCA"""
        try:
            if len(self.profit_memory) < self.genome["n_components"]:
                return {'profit_compression': 'insufficient_data'}
            
            # Extract features and weights
            features = np.array([mem[0] for mem in self.profit_memory])
            profits = np.array([mem[1] for mem in self.profit_memory])
            
            # Weight by profit magnitude
            weights = profits / np.max(profits) if np.max(profits) > 0 else np.ones_like(profits)
            weighted_features = features * weights.reshape(-1, 1)
            
            # Standardize features
            if not hasattr(self, '_profit_scaler_fitted'):
                self.scaler.fit(weighted_features)
                self._profit_scaler_fitted = True
            
            standardized_features = self.scaler.transform(weighted_features)
            
            # Apply PCA
            self.profit_pca.fit(standardized_features)
            compressed = self.profit_pca.transform(standardized_features)
            
            # Update profit direction with weighted average
            profit_weights = weights / np.sum(weights)
            self.profit_direction = np.average(compressed, axis=0, weights=profit_weights).astype(np.float32)
            
            # Calculate compression quality
            explained_variance = np.sum(self.profit_pca.explained_variance_ratio_)
            self._explained_variance_history.append(explained_variance)
            
            return {
                'profit_compression': {
                    'samples_compressed': len(features),
                    'explained_variance': explained_variance,
                    'profit_direction_strength': np.linalg.norm(self.profit_direction),
                    'avg_profit': np.mean(profits)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Profit pattern compression failed: {e}")
            return {'profit_compression': f'error: {str(e)}'}

    def _compress_loss_patterns(self) -> Dict[str, Any]:
        """Compress loss patterns for avoidance"""
        try:
            if len(self.loss_memory) < self.genome["n_components"]:
                return {'loss_compression': 'insufficient_data'}
            
            # Extract features and weights
            features = np.array([mem[0] for mem in self.loss_memory])
            losses = np.array([mem[1] for mem in self.loss_memory])
            
            # Weight by loss magnitude
            weights = losses / np.max(losses) if np.max(losses) > 0 else np.ones_like(losses)
            weighted_features = features * weights.reshape(-1, 1)
            
            # Standardize and compress
            standardized_features = self.scaler.transform(weighted_features)
            self.loss_pca.fit(standardized_features)
            compressed = self.loss_pca.transform(standardized_features)
            
            # Update loss direction with weighted average
            loss_weights = weights / np.sum(weights)
            self.loss_direction = np.average(compressed, axis=0, weights=loss_weights).astype(np.float32)
            
            # Calculate compression quality
            explained_variance = np.sum(self.loss_pca.explained_variance_ratio_)
            
            return {
                'loss_compression': {
                    'samples_compressed': len(features),
                    'explained_variance': explained_variance,
                    'loss_direction_strength': np.linalg.norm(self.loss_direction),
                    'avg_loss': np.mean(losses)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Loss pattern compression failed: {e}")
            return {'loss_compression': f'error: {str(e)}'}

    async def _update_intuition_vector(self) -> Dict[str, Any]:
        """Update intuition vector based on compressed patterns"""
        try:
            # Blend profit and loss directions
            profit_strength = np.linalg.norm(self.profit_direction)
            loss_strength = np.linalg.norm(self.loss_direction)
            
            if profit_strength > 0 and loss_strength > 0:
                # Create intuition by moving towards profit and away from loss
                profit_component = self.profit_direction * self.genome["profit_weight"]
                loss_component = -self.loss_direction * self.genome["loss_avoidance_weight"]
                
                combined = profit_component + loss_component
                
                # Apply learning rate
                self.intuition_vector = (
                    (1 - self.genome["learning_rate"]) * self.intuition_vector +
                    self.genome["learning_rate"] * combined
                ).astype(np.float32)
                
                # Normalize to prevent explosion
                norm = np.linalg.norm(self.intuition_vector)
                if norm > 1e-8:
                    self.intuition_vector = self.intuition_vector / norm
            
            elif profit_strength > 0:
                # Only profit direction available
                self.intuition_vector = (
                    (1 - self.genome["learning_rate"]) * self.intuition_vector +
                    self.genome["learning_rate"] * self.profit_direction
                ).astype(np.float32)
            
            # Track intuition evolution
            self._intuition_evolution.append({
                'timestamp': time.time(),
                'intuition': self.intuition_vector.copy(),
                'profit_strength': profit_strength,
                'loss_strength': loss_strength
            })
            
            return {
                'intuition_update': {
                    'intuition_strength': np.linalg.norm(self.intuition_vector),
                    'profit_strength': profit_strength,
                    'loss_strength': loss_strength,
                    'learning_rate': self.genome["learning_rate"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Intuition vector update failed: {e}")
            return {'intuition_update': f'error: {str(e)}'}

    def _trim_memory_buffers(self):
        """Trim memory buffers to maintain size limits"""
        try:
            max_size = self.genome["max_memory_size"]
            
            # Keep most recent and most profitable
            if len(self.profit_memory) > max_size // 2:
                # Sort by profit and keep top performers + recent
                sorted_profits = sorted(self.profit_memory, key=lambda x: x[1], reverse=True)
                top_half = max_size // 4
                recent_half = max_size // 4
                
                self.profit_memory = sorted_profits[:top_half] + sorted_profits[-recent_half:]
            
            # Keep most recent and most costly losses
            if len(self.loss_memory) > max_size // 2:
                sorted_losses = sorted(self.loss_memory, key=lambda x: x[1], reverse=True)
                top_half = max_size // 4
                recent_half = max_size // 4
                
                self.loss_memory = sorted_losses[:top_half] + sorted_losses[-recent_half:]
            
        except Exception as e:
            self.logger.error(f"Memory trimming failed: {e}")

    def _record_compression(self, compression_data: Dict[str, Any], compression_results: Dict[str, Any]):
        """Record compression results for analysis"""
        try:
            compression_record = {
                'timestamp': time.time(),
                'episode': compression_data.get('episode', 0),
                'profit_memories': len(self.profit_memory),
                'loss_memories': len(self.loss_memory),
                'compression_results': compression_results,
                'intuition_strength': np.linalg.norm(self.intuition_vector)
            }
            
            self._compression_history.append(compression_record)
            
            # Calculate compression quality
            if 'profit_compression' in compression_results:
                profit_var = compression_results['profit_compression'].get('explained_variance', 0)
                loss_var = compression_results.get('loss_compression', {}).get('explained_variance', 0)
                
                quality = (profit_var + loss_var) / 2
                self._compression_quality_scores.append(quality)
            
        except Exception as e:
            self.logger.error(f"Compression recording failed: {e}")

    async def _generate_compression_thesis(self, compression_data: Dict[str, Any], 
                                         memory_result: Dict[str, Any]) -> str:
        """Generate comprehensive compression thesis"""
        try:
            # Memory status
            total_memories = len(self.profit_memory) + len(self.loss_memory)
            profit_memories = len(self.profit_memory)
            loss_memories = len(self.loss_memory)
            
            # Compression status
            compression_performed = memory_result.get('compression_performed', False)
            intuition_strength = np.linalg.norm(self.intuition_vector)
            
            thesis_parts = [
                f"Memory Compression Analysis: {total_memories} total memories ({profit_memories} profitable, {loss_memories} loss patterns)",
                f"Intuition vector strength: {intuition_strength:.4f} representing compressed trading experience",
                f"Memory utilization: {self._compression_performance['memory_utilization']:.1%} of maximum capacity"
            ]
            
            if compression_performed:
                compression_results = memory_result.get('compression_results', {})
                thesis_parts.append(f"Compression cycle {self._compression_count} completed with pattern analysis")
                
                if 'profit_compression' in compression_results:
                    profit_info = compression_results['profit_compression']
                    if isinstance(profit_info, dict):
                        explained_var = profit_info.get('explained_variance', 0)
                        thesis_parts.append(f"Profit patterns: {explained_var:.1%} variance explained by {self.genome['n_components']} components")
            
            # Pattern evolution
            if self._intuition_evolution:
                recent_changes = len([e for e in self._intuition_evolution if time.time() - e['timestamp'] < 3600])
                thesis_parts.append(f"Intuition evolution: {recent_changes} updates in last hour showing learning adaptation")
            
            # Compression efficiency
            if self._compression_quality_scores:
                avg_quality = np.mean(list(self._compression_quality_scores)[-5:])
                thesis_parts.append(f"Compression quality: {avg_quality:.1%} (target: {self.config.min_compression_quality:.1%}+)")
            
            # Learning assessment
            profit_ratio = profit_memories / max(total_memories, 1)
            if profit_ratio > 0.6:
                thesis_parts.append("High profit pattern density indicates effective learning and memory retention")
            elif profit_ratio < 0.3:
                thesis_parts.append("Low profit pattern density suggests need for strategy adjustment")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Compression thesis generation failed: {str(e)} - Memory compression continuing with basic analysis"

    async def _update_compression_smart_bus(self, memory_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with compression results"""
        try:
            # Intuition vector
            self.smart_bus.set(
                'intuition_vector',
                {
                    'vector': self.intuition_vector.tolist(),
                    'strength': float(np.linalg.norm(self.intuition_vector)),
                    'components': self.genome["n_components"],
                    'last_updated': time.time()
                },
                module='MemoryCompressor',
                thesis=thesis
            )
            
            # Compressed patterns
            pattern_data = {
                'profit_direction': self.profit_direction.tolist(),
                'loss_direction': self.loss_direction.tolist(),
                'profit_strength': float(np.linalg.norm(self.profit_direction)),
                'loss_strength': float(np.linalg.norm(self.loss_direction)),
                'compression_count': self._compression_count
            }
            
            self.smart_bus.set(
                'compressed_patterns',
                pattern_data,
                module='MemoryCompressor',
                thesis="Compressed profit and loss patterns for intuitive trading decisions"
            )
            
            # Memory compression status
            compression_status = {
                'total_memories': len(self.profit_memory) + len(self.loss_memory),
                'profit_memories': len(self.profit_memory),
                'loss_memories': len(self.loss_memory),
                'memory_utilization': self._compression_performance['memory_utilization'],
                'compression_efficiency': self._compression_efficiency,
                'last_compression': self._compression_count
            }
            
            self.smart_bus.set(
                'memory_compression',
                compression_status,
                module='MemoryCompressor',
                thesis="Memory compression status and utilization metrics"
            )
            
            # Feature importance
            if hasattr(self.profit_pca, 'components_'):
                feature_importance = {
                    'profit_components': self.profit_pca.components_.tolist(),
                    'explained_variance_ratio': self.profit_pca.explained_variance_ratio_.tolist(),
                    'n_features': getattr(self.profit_pca, 'n_features_in_', 0)
                }
                
                self.smart_bus.set(
                    'feature_importance',
                    feature_importance,
                    module='MemoryCompressor',
                    thesis="Feature importance analysis from PCA compression"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no compression data is available"""
        self.logger.warning("No compression data available - returning current intuition")
        
        return {
            'intuition_strength': float(np.linalg.norm(self.intuition_vector)),
            'total_memories': len(self.profit_memory) + len(self.loss_memory),
            'compression_count': self._compression_count,
            'fallback_reason': 'no_compression_data'
        }

    async def _handle_compression_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle compression errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "MemoryCompressor")
        explanation = self.english_explainer.explain_error(
            "MemoryCompressor", str(error), "memory compression"
        )
        
        self.logger.error(
            format_operator_message(
                "💥", "MEMORY_COMPRESSION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="memory_compression"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'intuition_strength': float(np.linalg.norm(self.intuition_vector)),
            'total_memories': len(self.profit_memory) + len(self.loss_memory),
            'compression_count': self._compression_count,
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    def _update_compression_health(self):
        """Update compression health metrics"""
        try:
            # Check compression quality
            if self._compression_quality_scores:
                avg_quality = np.mean(list(self._compression_quality_scores)[-5:])
                if avg_quality < self.config.min_compression_quality:
                    self._health_status = 'warning'
                else:
                    self._health_status = 'healthy'
            
            # Check memory utilization
            if self._compression_performance['memory_utilization'] > 0.95:
                self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_compression_efficiency(self):
        """Analyze compression efficiency"""
        try:
            if len(self._compression_history) >= 3:
                recent_compressions = list(self._compression_history)[-3:]
                
                # Calculate efficiency trend
                intuition_strengths = [c['intuition_strength'] for c in recent_compressions]
                if len(intuition_strengths) >= 2:
                    trend = intuition_strengths[-1] - intuition_strengths[0]
                    
                    if trend > 0.1:  # Improving intuition
                        self.logger.info(
                            format_operator_message(
                                "📈", "COMPRESSION_EFFICIENCY_IMPROVING",
                                trend=f"{trend:.4f}",
                                recent_strength=f"{intuition_strengths[-1]:.4f}",
                                context="efficiency_analysis"
                            )
                        )
            
        except Exception as e:
            self.logger.error(f"Efficiency analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'MemoryCompressor', 'compression_cycle', processing_time, True
        )
        
        # Update performance metrics
        self._compression_performance['episodes_processed'] += 1
        
        # Update average processing time
        current_avg = self._compression_performance['avg_compression_time']
        episodes = self._compression_performance['episodes_processed']
        new_avg = (current_avg * (episodes - 1) + processing_time) / episodes
        self._compression_performance['avg_compression_time'] = new_avg
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'MemoryCompressor', 'compression_cycle', 0, False
        )

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'profit_memory': [(mem[0].tolist(), mem[1]) for mem in self.profit_memory[-100:]],  # Last 100
            'loss_memory': [(mem[0].tolist(), mem[1]) for mem in self.loss_memory[-100:]],  # Last 100
            'intuition_vector': self.intuition_vector.tolist(),
            'profit_direction': self.profit_direction.tolist(),
            'loss_direction': self.loss_direction.tolist(),
            'genome': self.genome.copy(),
            'compression_count': self._compression_count,
            'compression_performance': self._compression_performance.copy(),
            'circuit_breaker': self.circuit_breaker.copy(),
            'health_status': self._health_status
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'profit_memory' in state:
            self.profit_memory = [(np.array(mem[0]), mem[1]) for mem in state['profit_memory']]
        
        if 'loss_memory' in state:
            self.loss_memory = [(np.array(mem[0]), mem[1]) for mem in state['loss_memory']]
        
        if 'intuition_vector' in state:
            self.intuition_vector = np.array(state['intuition_vector'], dtype=np.float32)
        
        if 'profit_direction' in state:
            self.profit_direction = np.array(state['profit_direction'], dtype=np.float32)
        
        if 'loss_direction' in state:
            self.loss_direction = np.array(state['loss_direction'], dtype=np.float32)
        
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'compression_count' in state:
            self._compression_count = state['compression_count']
        
        if 'compression_performance' in state:
            self._compression_performance.update(state['compression_performance'])
        
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
            'total_memories': len(self.profit_memory) + len(self.loss_memory),
            'compression_count': self._compression_count,
            'intuition_strength': float(np.linalg.norm(self.intuition_vector))
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def propose_action(self, obs: Any = None, **kwargs) -> np.ndarray:
        """Legacy compatibility for action proposal"""
        # Return intuition vector as action guidance
        if np.linalg.norm(self.intuition_vector) > 0:
            # Convert to 2D action space
            return np.array([self.intuition_vector[0], self.intuition_vector[1] if len(self.intuition_vector) > 1 else 0.0])
        return np.array([0.0, 0.0])
    
    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        return float(np.linalg.norm(self.intuition_vector))