# ─────────────────────────────────────────────────────────────
# File: modules/memory/playbook_memory.py
# [ROCKET] PRODUCTION-READY Playbook Memory System
# Enhanced with SmartInfoBus integration & advanced pattern recognition
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@dataclass
class PlaybookConfig:
    """Configuration for Playbook Memory"""
    max_entries: int = 500
    k: int = 5
    profit_weight: float = 2.0
    context_weight: float = 1.5
    similarity_threshold: float = 0.7
    memory_decay: float = 0.98
    context_features_weight: float = 0.3
    recency_weight: float = 0.1
    
    # Performance thresholds
    max_processing_time_ms: float = 150
    circuit_breaker_threshold: int = 3
    min_quality_threshold: float = 0.5
    
    # Analysis parameters
    max_recall_history: int = 100
    pattern_effectiveness_window: int = 50
    quality_prune_threshold: float = 0.3


@module(
    name="PlaybookMemory",
    version="3.0.0",
    category="memory",
    provides=["playbook_recall", "pattern_memory", "sequence_quality", "memory_analytics"],
    requires=["trades", "actions", "market_data", "prices"],
    description="Advanced playbook memory with context-aware pattern recognition and SmartInfoBus integration",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class PlaybookMemory(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced playbook memory system with SmartInfoBus integration.
    Provides context-aware pattern recognition and profitable trade recall.
    """

    def __init__(self, 
                 config: Optional[PlaybookConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        # Store config first before calling super().__init__()
        self.playbook_config = config or PlaybookConfig()
        super().__init__()
        
        # Ensure our config is preserved after BaseModule initialization
        self.config = self.playbook_config
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize playbook state
        self._initialize_playbook_state()
        
        # Start monitoring after all initialization is complete
        
        self._start_monitoring()
        
        
        
        self.logger.info(
            format_operator_message(
                "📚", "PLAYBOOK_MEMORY_INITIALIZED",
                details=f"Max entries: {self.config.max_entries}, K-neighbors: {self.config.k}",
                result="Context-aware pattern memory ready",
                context="memory_system"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for playbook memory"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="PlaybookMemory", 
            log_path="logs/playbook_memory.log", 
            max_lines=3000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PlaybookMemory", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for memory operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        # Note: _start_monitoring() moved to end of initialization

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        """Initialize genome-based parameters"""
        if genome:
            self.genome = {
                "max_entries": int(genome.get("max_entries", self.config.max_entries)),
                "k": int(genome.get("k", self.config.k)),
                "profit_weight": float(genome.get("profit_weight", self.config.profit_weight)),
                "context_weight": float(genome.get("context_weight", self.config.context_weight)),
                "similarity_threshold": float(genome.get("similarity_threshold", self.config.similarity_threshold)),
                "memory_decay": float(genome.get("memory_decay", self.config.memory_decay)),
                "context_features_weight": float(genome.get("context_features_weight", self.config.context_features_weight)),
                "recency_weight": float(genome.get("recency_weight", self.config.recency_weight))
            }
        else:
            self.genome = {
                "max_entries": self.config.max_entries,
                "k": self.config.k,
                "profit_weight": self.config.profit_weight,
                "context_weight": self.config.context_weight,
                "similarity_threshold": self.config.similarity_threshold,
                "memory_decay": self.config.memory_decay,
                "context_features_weight": self.config.context_features_weight,
                "recency_weight": self.config.recency_weight
            }

    def _initialize_playbook_state(self):
        """Initialize playbook memory state"""
        # Core memory storage
        self._features = []
        self._actions = []
        self._pnls = []
        self._contexts = []
        self._timestamps = []
        self._trade_metadata = []
        
        # Enhanced tracking
        self._recall_history = deque(maxlen=self.config.max_recall_history)
        self._pattern_effectiveness = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})
        self._context_patterns = defaultdict(int)
        self._similarity_scores = deque(maxlen=self.config.pattern_effectiveness_window)
        
        # Performance analytics
        self._memory_quality_score = 0.0
        self._prediction_accuracy = 0.0
        self._pattern_diversity = 0.0
        self._recall_efficiency = 0.0
        
        # ML components
        self._nbrs = None
        self._weighted_nbrs = None
        self._scaler = StandardScaler()
        
        # Adaptive parameters
        self._adaptive_params = {
            'dynamic_k': self.genome["k"],
            'context_importance': self.genome["context_weight"],
            'profit_bias': self.genome["profit_weight"],
            'quality_threshold': self.config.min_quality_threshold
        }

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_memory_health()
                    self._analyze_memory_performance()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    def _initialize(self):
        """Initialize module"""
        try:
            # Set initial memory status in SmartInfoBus
            initial_status = {
                "memory_entries": 0,
                "patterns_identified": 0,
                "memory_quality": 0.0,
                "recall_efficiency": 0.0
            }
            
            self.smart_bus.set(
                'playbook_recall',
                initial_status,
                module='PlaybookMemory',
                thesis="Initial playbook memory status"
            )
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process playbook memory operations"""
        start_time = time.time()
        
        try:
            # Extract trading data
            trade_data = await self._extract_trade_data(**inputs)
            
            if not trade_data:
                return await self._handle_no_data_fallback()
            
            # Process new trades
            memory_result = await self._process_memory_operations(trade_data)
            
            # Perform recall if requested
            if trade_data.get('recall_requested', False):
                recall_result = await self._perform_memory_recall(trade_data)
                memory_result.update(recall_result)
            
            # Update memory analytics
            analytics_result = await self._update_memory_analytics()
            memory_result.update(analytics_result)
            
            # Generate thesis
            thesis = await self._generate_memory_thesis(trade_data, memory_result)
            
            # Update SmartInfoBus
            await self._update_memory_smart_bus(memory_result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return memory_result
            
        except Exception as e:
            return await self._handle_memory_error(e, start_time)

    async def _extract_trade_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract trade data from SmartInfoBus"""
        try:
            # Get recent trades
            trades = self.smart_bus.get('trades', 'PlaybookMemory') or []
            
            # Get actions
            actions = self.smart_bus.get('actions', 'PlaybookMemory') or []
            
            # Get market data
            market_data = self.smart_bus.get('market_data', 'PlaybookMemory') or {}
            
            # Get prices
            prices = self.smart_bus.get('prices', 'PlaybookMemory') or {}
            
            # Extract context from market data
            context = self._extract_standard_context(market_data)
            
            return {
                'trades': trades,
                'actions': actions,
                'market_data': market_data,
                'prices': prices,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'recall_requested': inputs.get('recall_requested', False),
                'query_features': inputs.get('query_features', None)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract trade data: {e}")
            return None

    def _extract_standard_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standard market context"""
        return {
            'regime': market_data.get('regime', 'unknown'),
            'volatility_level': market_data.get('volatility_level', 'medium'),
            'session': market_data.get('session', 'unknown'),
            'drawdown_pct': market_data.get('drawdown_pct', 0.0),
            'exposure_pct': market_data.get('exposure_pct', 0.0),
            'position_count': market_data.get('position_count', 0),
            'timestamp': datetime.now().isoformat()
        }

    async def _process_memory_operations(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory storage operations"""
        try:
            trades_processed = 0
            
            # Process new trades
            for trade in trade_data.get('trades', []):
                features = self._extract_trade_features(trade, trade_data)
                action = self._extract_trade_action(trade)
                pnl = trade.get('pnl', 0.0)
                context = trade_data['context']
                
                await self._record_trade_memory(features, action, pnl, context)
                trades_processed += 1
            
            # Update ML models if needed
            if len(self._features) >= self.genome["k"]:
                await self._fit_memory_models()
            
            return {
                'trades_processed': trades_processed,
                'memory_size': len(self._features),
                'memory_utilization': len(self._features) / self.genome["max_entries"],
                'models_fitted': self._nbrs is not None
            }
            
        except Exception as e:
            self.logger.error(f"Memory operations failed: {e}")
            return self._create_fallback_response("memory operations failed")

    def _extract_trade_features(self, trade: Dict[str, Any], trade_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from trade and market context"""
        features = []
        
        # Market regime features
        regime = trade_data['context'].get('regime', 'unknown')
        regime_encoding = {'trending': [1, 0, 0], 'volatile': [0, 1, 0], 'ranging': [0, 0, 1], 'unknown': [0.33, 0.33, 0.33]}
        features.extend(regime_encoding.get(regime, [0.33, 0.33, 0.33]))
        
        # Volatility features
        vol_level = trade_data['context'].get('volatility_level', 'medium')
        vol_value = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0}.get(vol_level, 0.5)
        features.append(vol_value)
        
        # Risk context
        features.extend([
            trade_data['context'].get('drawdown_pct', 0.0) / 100.0,
            trade_data['context'].get('exposure_pct', 0.0) / 100.0,
            trade_data['context'].get('position_count', 0) / 10.0
        ])
        
        # Session features
        session = trade_data['context'].get('session', 'unknown')
        session_encoding = {'asian': [1, 0, 0], 'european': [0, 1, 0], 'american': [0, 0, 1], 'closed': [0, 0, 0]}
        features.extend(session_encoding.get(session, [0.25, 0.25, 0.25]))
        
        # Trade-specific features
        features.extend([
            float(trade.get('size', 0.0)),
            float(trade.get('confidence', 0.5)),
            1.0 if trade.get('side') == 'buy' else -1.0 if trade.get('side') == 'sell' else 0.0
        ])
        
        # Price context
        prices = trade_data.get('prices', {})
        symbol = trade.get('symbol', 'EUR/USD')
        if symbol in prices:
            current_price = prices[symbol]
            entry_price = trade.get('price', current_price)
            price_change = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            features.extend([current_price / 2.0, price_change])
        else:
            features.extend([0.5, 0.0])
        
        return np.array(features, dtype=np.float32)

    def _extract_trade_action(self, trade: Dict[str, Any]) -> np.ndarray:
        """Extract action representation from trade"""
        size = trade.get('size', 0.0)
        side = trade.get('side', 'hold')
        
        if side == 'buy':
            action = [size, 0.0]
        elif side == 'sell':
            action = [-size, 0.0]
        else:
            action = [0.0, 0.0]
        
        return np.array(action, dtype=np.float32)

    async def _record_trade_memory(self, features: np.ndarray, actions: np.ndarray, 
                                  pnl: float, context: Dict[str, Any]):
        """Record trade in memory with context"""
        try:
            # Apply memory decay to existing entries
            if self._pnls:
                self._pnls = [p * self.genome["memory_decay"] for p in self._pnls]
            
            # Memory management
            if len(self._features) >= self.genome["max_entries"]:
                # Remove oldest entries
                self._features.pop(0)
                self._actions.pop(0)
                self._pnls.pop(0)
                self._contexts.pop(0)
                self._timestamps.pop(0)
                self._trade_metadata.pop(0)
            
            # Add new trade
            self._features.append(features.copy())
            self._actions.append(actions.copy())
            self._pnls.append(pnl)
            self._contexts.append(context.copy())
            self._timestamps.append(datetime.now())
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'pnl': pnl,
                'context': context.copy(),
                'regime': context.get('regime', 'unknown'),
                'volatility_level': context.get('volatility_level', 'medium')
            }
            self._trade_metadata.append(metadata)
            
            # Update pattern tracking
            self._update_pattern_tracking(context, pnl)
            
            # Update trading metrics
            self._update_trading_metrics({'pnl': pnl})
            
        except Exception as e:
            self.logger.error(f"Trade recording failed: {e}")

    def _update_pattern_tracking(self, context: Dict[str, Any], pnl: float):
        """Update pattern effectiveness tracking"""
        regime = context.get('regime', 'unknown')
        vol_level = context.get('volatility_level', 'medium')
        session = context.get('session', 'unknown')
        
        pattern_key = f"{regime}_{vol_level}_{session}"
        self._context_patterns[pattern_key] += 1
        
        # Update pattern effectiveness
        if pnl > 0:
            self._pattern_effectiveness[pattern_key]['wins'] += 1
        elif pnl < 0:
            self._pattern_effectiveness[pattern_key]['losses'] += 1
        
        self._pattern_effectiveness[pattern_key]['total_pnl'] += pnl

    async def _fit_memory_models(self):
        """Fit ML models for memory recall"""
        try:
            if len(self._features) < self.genome["k"]:
                return
            
            # Prepare feature matrix
            X = np.vstack(self._features)
            X_scaled = self._scaler.fit_transform(X)
            
            # Fit KNN models
            k_effective = min(self._adaptive_params['dynamic_k'], len(X))
            self._nbrs = NearestNeighbors(n_neighbors=k_effective, metric='euclidean')
            self._nbrs.fit(X_scaled)
            
            # Fit weighted KNN
            k_extended = min(k_effective * 2, len(X))
            self._weighted_nbrs = NearestNeighbors(n_neighbors=k_extended, metric='euclidean')
            self._weighted_nbrs.fit(X_scaled)
            
            # Update quality score
            profitable_count = sum(1 for p in self._pnls if p > 0)
            self._memory_quality_score = (profitable_count / len(self._pnls)) * 100
            
        except Exception as e:
            self.logger.error(f"Model fitting failed: {e}")

    async def _perform_memory_recall(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform memory recall operation"""
        try:
            query_features = trade_data.get('query_features')
            if query_features is None:
                return {'recall_performed': False, 'reason': 'no_query_features'}
            
            if self._nbrs is None or len(self._features) == 0:
                return self._create_empty_recall_result()
            
            # Prepare query features
            if query_features.ndim == 1:
                query_features = query_features.reshape(1, -1)
            
            # Normalize features
            normalized_features = self._scaler.transform(query_features)
            
            # Find similar trades
            k_search = min(self._adaptive_params['dynamic_k'], len(self._features))
            distances, indices = self._nbrs.kneighbors(normalized_features, n_neighbors=k_search)
            
            indices = indices[0]
            distances = distances[0]
            
            # Extract similar trade data
            similar_pnls = [self._pnls[i] for i in indices if i < len(self._pnls)]
            similar_actions = [self._actions[i] for i in indices if i < len(self._actions)]
            
            # Calculate predictions
            expected_pnl = float(np.mean(similar_pnls)) if similar_pnls else 0.0
            confidence = float(np.exp(-np.mean(distances))) if len(distances) > 0 else 0.0
            
            # Track recall
            recall_record = {
                'timestamp': datetime.now().isoformat(),
                'expected_pnl': expected_pnl,
                'confidence': confidence,
                'similar_trades': len(indices)
            }
            self._recall_history.append(recall_record)
            
            return {
                'recall_performed': True,
                'expected_pnl': expected_pnl,
                'confidence': confidence,
                'similar_trades': len(indices),
                'profitable_matches': sum(1 for pnl in similar_pnls if pnl > 0)
            }
            
        except Exception as e:
            self.logger.error(f"Memory recall failed: {e}")
            return {'recall_performed': False, 'error': str(e)}

    def _create_empty_recall_result(self) -> Dict[str, Any]:
        """Create empty recall result"""
        return {
            'recall_performed': True,
            'expected_pnl': 0.0,
            'confidence': 0.0,
            'similar_trades': 0,
            'profitable_matches': 0
        }

    async def _update_memory_analytics(self) -> Dict[str, Any]:
        """Update memory analytics"""
        try:
            # Memory utilization
            memory_utilization = len(self._features) / self.genome["max_entries"]
            
            # Pattern diversity
            unique_patterns = len(self._pattern_effectiveness)
            self._pattern_diversity = min(1.0, unique_patterns / 20.0)
            
            # Recall efficiency
            if self._recall_history:
                recent_recalls = list(self._recall_history)[-10:]
                self._recall_efficiency = np.mean([r['confidence'] for r in recent_recalls])
            
            # Prediction accuracy
            if self._pattern_effectiveness:
                total_profitable = sum(p['wins'] for p in self._pattern_effectiveness.values())
                total_trades = sum(p['wins'] + p['losses'] for p in self._pattern_effectiveness.values())
                self._prediction_accuracy = total_profitable / max(total_trades, 1)
            
            return {
                'memory_utilization': memory_utilization,
                'pattern_diversity': self._pattern_diversity,
                'recall_efficiency': self._recall_efficiency,
                'prediction_accuracy': self._prediction_accuracy,
                'memory_quality': self._memory_quality_score
            }
            
        except Exception as e:
            self.logger.error(f"Analytics update failed: {e}")
            return {'analytics_updated': False}

    async def _generate_memory_thesis(self, trade_data: Dict[str, Any], 
                                    memory_result: Dict[str, Any]) -> str:
        """Generate comprehensive memory thesis"""
        try:
            # Memory metrics
            memory_size = len(self._features)
            memory_util = memory_result.get('memory_utilization', 0.0)
            quality_score = self._memory_quality_score
            
            # Pattern analysis
            total_patterns = len(self._pattern_effectiveness)
            profitable_patterns = sum(1 for p in self._pattern_effectiveness.values() if p['total_pnl'] > 0)
            
            # Performance metrics
            trades_processed = memory_result.get('trades_processed', 0)
            recall_performed = memory_result.get('recall_performed', False)
            
            thesis_parts = [
                f"Playbook Memory: {memory_size} entries ({memory_util:.1%} capacity) with {quality_score:.1f} quality score",
                f"Pattern Recognition: {total_patterns} unique patterns, {profitable_patterns} profitable",
                f"Learning Effectiveness: {self._prediction_accuracy:.2f} prediction accuracy, {self._recall_efficiency:.2f} recall efficiency"
            ]
            
            if trades_processed > 0:
                thesis_parts.append(f"Processed {trades_processed} new trades with context-aware feature extraction")
            
            if recall_performed:
                expected_pnl = memory_result.get('expected_pnl', 0.0)
                confidence = memory_result.get('confidence', 0.0)
                thesis_parts.append(f"Memory recall: {expected_pnl:.2f} expected PnL with {confidence:.2f} confidence")
            
            # Model status
            models_fitted = memory_result.get('models_fitted', False)
            if models_fitted:
                thesis_parts.append(f"ML models fitted with {self._adaptive_params['dynamic_k']} neighbors")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Memory thesis generation failed: {str(e)} - Pattern learning continuing"

    async def _update_memory_smart_bus(self, memory_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with memory results"""
        try:
            # Playbook recall data
            recall_data = {
                'memory_entries': len(self._features),
                'memory_quality': self._memory_quality_score,
                'prediction_accuracy': self._prediction_accuracy,
                'recall_efficiency': self._recall_efficiency,
                'last_recall': memory_result.get('expected_pnl', 0.0) if memory_result.get('recall_performed') else None
            }
            
            self.smart_bus.set(
                'playbook_recall',
                recall_data,
                module='PlaybookMemory',
                thesis=thesis
            )
            
            # Pattern memory
            pattern_data = {
                'total_patterns': len(self._pattern_effectiveness),
                'profitable_patterns': sum(1 for p in self._pattern_effectiveness.values() if p['total_pnl'] > 0),
                'pattern_diversity': self._pattern_diversity,
                'best_pattern': max(self._pattern_effectiveness.items(), 
                                  key=lambda x: x[1]['total_pnl'])[0] if self._pattern_effectiveness else None
            }
            
            self.smart_bus.set(
                'pattern_memory',
                pattern_data,
                module='PlaybookMemory',
                thesis=f"Pattern memory: {pattern_data['total_patterns']} patterns identified"
            )
            
            # Sequence quality
            quality_data = {
                'memory_utilization': memory_result.get('memory_utilization', 0.0),
                'quality_score': self._memory_quality_score,
                'models_fitted': memory_result.get('models_fitted', False),
                'adaptive_k': self._adaptive_params['dynamic_k']
            }
            
            self.smart_bus.set(
                'sequence_quality',
                quality_data,
                module='PlaybookMemory',
                thesis="Memory sequence quality and model fitness assessment"
            )
            
            # Memory analytics
            analytics_data = {
                'total_recalls': len(self._recall_history),
                'recent_performance': np.mean([r['confidence'] for r in list(self._recall_history)[-5:]]) if len(self._recall_history) >= 5 else 0.0,
                'memory_health': self._health_status,
                'circuit_breaker_state': self.circuit_breaker['state']
            }
            
            self.smart_bus.set(
                'memory_analytics',
                analytics_data,
                module='PlaybookMemory',
                thesis="Memory analytics and system health monitoring"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no trade data is available"""
        self.logger.warning("No trade data available - using cached memory state")
        
        return {
            'memory_size': len(self._features),
            'memory_quality': self._memory_quality_score,
            'total_patterns': len(self._pattern_effectiveness),
            'fallback_reason': 'no_trade_data'
        }

    async def _handle_memory_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle memory operation errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "PlaybookMemory")
        explanation = self.english_explainer.explain_error(
            "PlaybookMemory", str(error), "memory operations"
        )
        
        self.logger.error(
            format_operator_message(
                "[CRASH]", "MEMORY_OPERATION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="memory_system"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'memory_size': len(self._features),
            'memory_quality': self._memory_quality_score,
            'total_patterns': len(self._pattern_effectiveness),
            'fallback_reason': reason,
            'circuit_breaker_state': self.circuit_breaker['state']
        }

    def _update_memory_health(self):
        """Update memory health metrics"""
        try:
            # Check memory utilization
            utilization = len(self._features) / self.genome["max_entries"]
            if utilization > 0.9:
                self._health_status = 'warning'
            elif utilization < 0.1:
                self._health_status = 'warning'
            else:
                self._health_status = 'healthy'
            
            # Check model fitness
            if len(self._features) >= self.genome["k"] and self._nbrs is None:
                self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_memory_performance(self):
        """Analyze memory performance metrics"""
        try:
            # Check recall quality
            if self._recall_history:
                recent_quality = np.mean([r['confidence'] for r in list(self._recall_history)[-10:]])
                if recent_quality > 0.7:
                    self.logger.info(
                        format_operator_message(
                            "[TARGET]", "HIGH_QUALITY_RECALLS",
                            avg_confidence=f"{recent_quality:.2f}",
                            recent_recalls=len(list(self._recall_history)[-10:]),
                            context="memory_performance"
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'PlaybookMemory', 'memory_cycle', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'PlaybookMemory', 'memory_cycle', 0, False
        )

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in memory recommendations"""
        if not isinstance(action, dict):
            return 0.5
            
        base_confidence = 0.5
        
        # Confidence from memory quality
        if len(self._features) > 0:
            memory_quality = self._memory_quality_score / 100.0
            base_confidence += memory_quality * 0.3
        
        # Confidence from pattern effectiveness
        if self._pattern_effectiveness:
            avg_effectiveness = np.mean([
                p['total_pnl'] / max(p['wins'] + p['losses'], 1)
                for p in self._pattern_effectiveness.values()
            ])
            if avg_effectiveness > 0:
                base_confidence += min(0.3, avg_effectiveness / 100.0)
        
        # Confidence from recall efficiency
        base_confidence += self._recall_efficiency * 0.2
        
        # Circuit breaker consideration
        if self.circuit_breaker['state'] == 'OPEN':
            base_confidence *= 0.5
        
        # Memory size factor
        memory_factor = min(1.0, len(self._features) / 100.0)
        base_confidence += memory_factor * 0.1
        
        return float(max(0.0, min(1.0, base_confidence)))
        base_confidence += self._prediction_accuracy * 0.3
        
        # Confidence from recall efficiency
        base_confidence += self._recall_efficiency * 0.2
        
        # Confidence from data volume
        data_confidence = min(0.2, len(self._features) / self.genome["max_entries"] * 0.2)
        base_confidence += data_confidence
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'config': self.config.__dict__,
            'genome': self.genome.copy(),
            'features': [f.tolist() for f in self._features],
            'actions': [a.tolist() for a in self._actions],
            'pnls': self._pnls.copy(),
            'contexts': self._contexts.copy(),
            'timestamps': [t.isoformat() for t in self._timestamps],
            'pattern_effectiveness': dict(self._pattern_effectiveness),
            'context_patterns': dict(self._context_patterns),
            'performance_metrics': {
                'memory_quality_score': self._memory_quality_score,
                'prediction_accuracy': self._prediction_accuracy,
                'pattern_diversity': self._pattern_diversity,
                'recall_efficiency': self._recall_efficiency
            },
            'adaptive_params': self._adaptive_params.copy(),
            'health_status': self._health_status,
            'circuit_breaker': self.circuit_breaker.copy()
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'features' in state:
            self._features = [np.array(f, dtype=np.float32) for f in state['features']]
        
        if 'actions' in state:
            self._actions = [np.array(a, dtype=np.float32) for a in state['actions']]
        
        if 'pnls' in state:
            self._pnls = state['pnls']
        
        if 'contexts' in state:
            self._contexts = state['contexts']
        
        if 'timestamps' in state:
            self._timestamps = [datetime.fromisoformat(t) for t in state['timestamps']]
        
        if 'pattern_effectiveness' in state:
            self._pattern_effectiveness = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
                                                     state['pattern_effectiveness'])
        
        if 'context_patterns' in state:
            self._context_patterns = defaultdict(int, state['context_patterns'])
        
        if 'performance_metrics' in state:
            metrics = state['performance_metrics']
            self._memory_quality_score = metrics.get('memory_quality_score', 0.0)
            self._prediction_accuracy = metrics.get('prediction_accuracy', 0.0)
            self._pattern_diversity = metrics.get('pattern_diversity', 0.0)
            self._recall_efficiency = metrics.get('recall_efficiency', 0.0)
        
        if 'adaptive_params' in state:
            self._adaptive_params.update(state['adaptive_params'])
        
        if 'health_status' in state:
            self._health_status = state['health_status']
        
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])
        
        # Refit models if we have enough data
        if len(self._features) >= self.genome["k"]:
            asyncio.create_task(self._fit_memory_models())

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'memory_size': len(self._features),
            'memory_quality': self._memory_quality_score,
            'models_fitted': self._nbrs is not None
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def step(self, features: Optional[np.ndarray] = None, actions: Optional[np.ndarray] = None, 
             pnl: Optional[float] = None, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Legacy compatibility step method"""
        if features is not None and actions is not None and pnl is not None:
            context = context or {}
            asyncio.create_task(self._record_trade_memory(features, actions, pnl, context))

    def recall(self, features: np.ndarray, context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Legacy compatibility recall method"""
        trade_data = {
            'query_features': features,
            'context': context or {}
        }
        # Run async recall synchronously for compatibility
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._perform_memory_recall(trade_data))
            return result
        finally:
            loop.close()

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Legacy compatibility for action proposal"""
        return {
            'action': [0.0, 0.0],
            'memory_confidence': 0.5,
            'pattern_match': None,
            'reason': 'legacy_default'
        }

    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        # Run async calculate_confidence synchronously for compatibility
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            action = kwargs.get('action', {'action': [0.0, 0.0]})
            result = loop.run_until_complete(self.calculate_confidence(action, obs=obs, **kwargs))
            return result
        finally:
            loop.close()