# ─────────────────────────────────────────────────────────────
# File: modules/memory/playbook_memory.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from collections import deque, defaultdict
import datetime
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, extract_standard_context


class PlaybookMemory(Module, AnalysisMixin, TradingMixin):
    def __init__(self, max_entries: int = 500, k: int = 5,
                 profit_weight: float = 2.0, context_weight: float = 1.5,
                 debug: bool = True, genome: Optional[Dict[str, Any]] = None, **kwargs):
        # Ensure these exist before the base-class state initializer runs
        self.k = k
        self.max_entries = max_entries
        self.profit_weight = profit_weight
        self.context_weight = context_weight

        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=200,
            **kwargs
        )
        super().__init__(config)

        # Initialize genome parameters
        self._initialize_genome_parameters(genome, max_entries, k, profit_weight, context_weight)

        # Enhanced state initialization
        self._initialize_module_state()

        # Initialize playbook components
        self._initialize_playbook_components()

        self.log_operator_info(
            "Playbook memory initialized",
            max_entries=self.max_entries,
            k_neighbors=self.k,
            profit_weight=f"{self.profit_weight:.2f}",
            context_weight=f"{self.context_weight:.2f}",
            features_enabled="context-aware"
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict], max_entries: int,
                                    k: int, profit_weight: float, context_weight: float):
        """Initialize genome-based parameters"""
        if genome:
            self.max_entries = int(genome.get("max_entries", max_entries))
            self.k = int(genome.get("k", k))
            self.profit_weight = float(genome.get("profit_weight", profit_weight))
            self.context_weight = float(genome.get("context_weight", context_weight))
            self.similarity_threshold = float(genome.get("similarity_threshold", 0.7))
            self.memory_decay = float(genome.get("memory_decay", 0.98))
            self.context_features_weight = float(genome.get("context_features_weight", 0.3))
            self.recency_weight = float(genome.get("recency_weight", 0.1))
        else:
            self.max_entries = max_entries
            self.k = k
            self.profit_weight = profit_weight
            self.context_weight = context_weight
            self.similarity_threshold = 0.7
            self.memory_decay = 0.98
            self.context_features_weight = 0.3
            self.recency_weight = 0.1

        # Store genome for evolution
        self.genome = {
            "max_entries": self.max_entries,
            "k": self.k,
            "profit_weight": self.profit_weight,
            "context_weight": self.context_weight,
            "similarity_threshold": self.similarity_threshold,
            "memory_decay": self.memory_decay,
            "context_features_weight": self.context_features_weight,
            "recency_weight": self.recency_weight
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Playbook storage
        self._features = []
        self._actions = []
        self._pnls = []
        self._contexts = []
        self._timestamps = []
        self._trade_metadata = []
        
        # Enhanced tracking
        self._recall_history = deque(maxlen=100)
        self._pattern_effectiveness = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})
        self._context_patterns = defaultdict(int)
        self._similarity_scores = deque(maxlen=50)
        
        # Performance analytics
        self._memory_quality_score = 0.0
        self._prediction_accuracy = 0.0
        self._pattern_diversity = 0.0
        self._recall_efficiency = 0.0
        
        # Adaptive parameters
        self._adaptive_params = {
            'dynamic_k': self.k,
            'context_importance': self.context_weight,
            'profit_bias': self.profit_weight,
            'quality_threshold': 0.5
        }

    def _initialize_playbook_components(self):
        """Initialize ML components"""
        try:
            # K-nearest neighbors models
            self._nbrs = None
            self._weighted_nbrs = None
            self._scaler = StandardScaler()
            
            # Feature engineering
            self._feature_importance = {}
            self._context_encoder = {}
            
            # Pattern recognition
            self._pattern_clusters = {}
            self._success_patterns = deque(maxlen=100)
            
            self.log_operator_info("Playbook components initialized successfully")
            
        except Exception as e:
            self.log_operator_error(f"Playbook component initialization failed: {e}")
            self._update_health_status("ERROR", f"Init failed: {e}")

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_trading_state()
        
        # Clear playbook data
        self._features.clear()
        self._actions.clear()
        self._pnls.clear()
        self._contexts.clear()
        self._timestamps.clear()
        self._trade_metadata.clear()
        
        # Reset tracking
        self._recall_history.clear()
        self._pattern_effectiveness.clear()
        self._context_patterns.clear()
        self._similarity_scores.clear()
        
        # Reset components
        self._nbrs = None
        self._weighted_nbrs = None
        self._scaler = StandardScaler()
        
        # Reset performance metrics
        self._memory_quality_score = 0.0
        self._prediction_accuracy = 0.0
        self._pattern_diversity = 0.0
        self._recall_efficiency = 0.0
        
        # Reset adaptive parameters
        self._adaptive_params = {
            'dynamic_k': self.k,
            'context_importance': self.context_weight,
            'profit_bias': self.profit_weight,
            'quality_threshold': 0.5
        }

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Process trades from InfoBus
        if info_bus:
            self._process_trades_from_info_bus(info_bus)
        
        # Process manual recording from kwargs
        if 'features' in kwargs and 'actions' in kwargs and 'pnl' in kwargs:
            self._record_trade_data(
                kwargs['features'],
                kwargs['actions'],
                kwargs['pnl'],
                kwargs.get('context', {}),
                info_bus
            )
        
        # Update performance metrics
        self._update_playbook_performance()
        
        # Adapt parameters based on performance
        self._adapt_parameters()

    def _process_trades_from_info_bus(self, info_bus: InfoBus):
        """Process recent trades from InfoBus"""
        
        recent_trades = info_bus.get('recent_trades', [])
        if not recent_trades:
            return
        
        # Extract market context
        market_context = extract_standard_context(info_bus)
        
        for trade in recent_trades:
            # Create features from trade and context
            features = self._extract_trade_features(trade, info_bus)
            
            # Get action representation
            action = self._extract_trade_action(trade)
            
            # Record the trade
            self._record_trade_data(
                features,
                action,
                trade.get('pnl', 0.0),
                market_context,
                info_bus
            )

    def _extract_trade_features(self, trade: Dict[str, Any], info_bus: InfoBus) -> np.ndarray:
        """Extract features from trade and market context"""
        
        features = []
        
        # Market regime features
        regime = InfoBusExtractor.get_market_regime(info_bus)
        regime_encoding = {'trending': [1, 0, 0], 'volatile': [0, 1, 0], 'ranging': [0, 0, 1], 'unknown': [0.33, 0.33, 0.33]}
        features.extend(regime_encoding.get(regime, [0.33, 0.33, 0.33]))
        
        # Volatility features
        vol_level = InfoBusExtractor.get_volatility_level(info_bus)
        vol_value = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0}.get(vol_level, 0.5)
        features.append(vol_value)
        
        # Risk context
        features.extend([
            InfoBusExtractor.get_drawdown_pct(info_bus) / 100.0,
            InfoBusExtractor.get_exposure_pct(info_bus) / 100.0,
            InfoBusExtractor.get_position_count(info_bus) / 10.0
        ])
        
        # Session features
        session = InfoBusExtractor.get_session(info_bus)
        session_encoding = {'asian': [1, 0, 0], 'european': [0, 1, 0], 'american': [0, 0, 1], 'closed': [0, 0, 0]}
        features.extend(session_encoding.get(session, [0.25, 0.25, 0.25]))
        
        # Trade-specific features
        features.extend([
            float(trade.get('size', 0.0)),
            float(trade.get('confidence', 0.5)),
            1.0 if trade.get('side') == 'buy' else -1.0 if trade.get('side') == 'sell' else 0.0
        ])
        
        # Price context from market data
        prices = info_bus.get('prices', {})
        symbol = trade.get('symbol', 'EUR/USD')
        if symbol in prices:
            current_price = prices[symbol]
            entry_price = trade.get('price', current_price)
            price_change = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            features.extend([current_price / 2.0, price_change])  # Normalize
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

    def _record_trade_data(self, features: np.ndarray, actions: np.ndarray, pnl: float,
                          context: Dict[str, Any], info_bus: Optional[InfoBus] = None):
        """Enhanced trade recording with context integration"""
        
        try:
            # Create enhanced metadata
            metadata = {
                'timestamp': datetime.datetime.now().isoformat(),
                'pnl': pnl,
                'context': context.copy(),
                'step_idx': info_bus.get('step_idx', self._step_count) if info_bus else self._step_count,
                'market_regime': context.get('regime', 'unknown'),
                'volatility_level': context.get('volatility_level', 'medium'),
                'session': context.get('session', 'unknown'),
                'drawdown_pct': context.get('drawdown_pct', 0.0),
                'exposure_pct': context.get('exposure_pct', 0.0)
            }
            
            # Apply memory decay to existing entries
            if self._pnls:
                self._pnls = [pnl * self.memory_decay for pnl in self._pnls]
            
            # Memory management
            if len(self._features) >= self.max_entries:
                # Remove oldest entries
                removed_pnl = self._pnls.pop(0)
                self._features.pop(0)
                self._actions.pop(0)
                self._contexts.pop(0)
                self._timestamps.pop(0)
                self._trade_metadata.pop(0)
                
                self.log_operator_info(
                    f"Memory full - removed oldest entry",
                    removed_pnl=f"€{removed_pnl:.2f}",
                    new_pnl=f"€{pnl:.2f}"
                )
            
            # Add new trade
            self._features.append(features.copy())
            self._actions.append(actions.copy())
            self._pnls.append(pnl)
            self._contexts.append(context.copy())
            self._timestamps.append(datetime.datetime.now())
            self._trade_metadata.append(metadata)
            
            # Update pattern tracking
            self._update_pattern_tracking(context, pnl)
            
            # Update trading metrics via mixin
            self._update_trading_metrics({'pnl': pnl})
            
            # Refit models if we have enough data
            if len(self._features) >= self.k:
                self._fit_models()
            
            self.log_operator_info(
                f"Trade recorded",
                memory_size=f"{len(self._features)}/{self.max_entries}",
                pnl=f"€{pnl:.2f}",
                regime=context.get('regime', 'unknown'),
                total_pnl=f"€{self._total_pnl:.2f}"
            )
            
        except Exception as e:
            self.log_operator_error(f"Trade recording failed: {e}")

    def _update_pattern_tracking(self, context: Dict[str, Any], pnl: float):
        """Update pattern effectiveness tracking"""
        
        # Track context patterns
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
        
        # Track successful patterns
        if pnl > 0:
            self._success_patterns.append({
                'pattern': pattern_key,
                'pnl': pnl,
                'context': context.copy(),
                'timestamp': datetime.datetime.now().isoformat()
            })

    def _fit_models(self):
        """Enhanced model fitting with profit weighting"""
        
        try:
            if len(self._features) < self.k:
                return
            
            # Prepare feature matrix
            X = np.vstack(self._features)
            
            # Normalize features
            X_scaled = self._scaler.fit_transform(X)
            
            # Create profit-based weights
            weights = np.array(self._pnls)
            positive_weights = np.where(weights > 0,
                                      1 + weights / 100.0 * self.profit_weight,
                                      1 / (1 + np.abs(weights) / 100.0))
            
            # Recency weights
            recency_weights = np.exp(np.arange(len(weights)) * self.recency_weight / len(weights))
            
            # Combined weights
            combined_weights = positive_weights * recency_weights
            
            # Fit standard KNN
            k_effective = min(self._adaptive_params['dynamic_k'], len(X))
            self._nbrs = NearestNeighbors(n_neighbors=k_effective, metric='euclidean')
            self._nbrs.fit(X_scaled)
            
            # Fit weighted KNN (using more neighbors for filtering)
            k_extended = min(k_effective * 2, len(X))
            self._weighted_nbrs = NearestNeighbors(n_neighbors=k_extended, metric='euclidean')
            self._weighted_nbrs.fit(X_scaled)
            
            # Update performance metrics
            profitable_count = sum(1 for p in self._pnls if p > 0)
            self._memory_quality_score = (profitable_count / len(self._pnls)) * 100
            
            self.log_operator_info(
                f"Models fitted",
                memory_size=len(X),
                profitable_trades=f"{profitable_count}/{len(self._pnls)}",
                k_neighbors=k_effective,
                quality_score=f"{self._memory_quality_score:.1f}"
            )
            
        except Exception as e:
            self.log_operator_error(f"Model fitting failed: {e}")

    def recall(self, features: np.ndarray, context: Optional[Dict[str, Any]] = None,
              info_bus: Optional[InfoBus] = None) -> Dict[str, Any]:
        """Enhanced context-aware recall with InfoBus integration"""
        
        try:
            # Use InfoBus context if available
            if info_bus and not context:
                context = extract_standard_context(info_bus)
            
            recall_id = len(self._recall_history) + 1
            
            if self._nbrs is None or len(self._features) == 0:
                return self._create_empty_recall_result(recall_id)
            
            # Prepare query features
            query_features = self._prepare_query_features(features)
            
            # Perform enhanced recall
            recall_result = self._perform_enhanced_recall(query_features, context, recall_id)
            
            # Track recall performance
            self._track_recall_performance(recall_result, context)
            
            return recall_result
            
        except Exception as e:
            self.log_operator_error(f"Recall failed: {e}")
            return self._create_empty_recall_result(0)

    def _prepare_query_features(self, features: np.ndarray) -> np.ndarray:
        """Prepare and normalize query features"""
        
        # Ensure proper shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize using fitted scaler
        try:
            normalized_features = self._scaler.transform(features)
            return normalized_features
        except Exception:
            # Fallback if scaler not fitted
            return features

    def _perform_enhanced_recall(self, query_features: np.ndarray, 
                                context: Optional[Dict[str, Any]], recall_id: int) -> Dict[str, Any]:
        """Perform enhanced recall with context awareness"""
        
        try:
            # Find similar trades
            k_search = min(self._adaptive_params['dynamic_k'] * 2, len(self._features))
            distances, indices = self._weighted_nbrs.kneighbors(query_features, n_neighbors=k_search)
            
            indices = indices[0]
            distances = distances[0]
            
            # Filter by context similarity if available
            if context:
                context_scores = []
                for idx in indices:
                    if idx < len(self._contexts):
                        ctx_score = self._calculate_context_similarity(context, self._contexts[idx])
                        context_scores.append(ctx_score)
                    else:
                        context_scores.append(0.5)  # Default score
                context_scores = np.array(context_scores)
            else:
                context_scores = np.ones(len(indices))
            
            # Combined scoring
            similarity_scores = np.exp(-distances) 
            combined_scores = (0.7 * similarity_scores + 
                             0.3 * context_scores * self._adaptive_params['context_importance'])
            
            # Select top-k
            k_final = min(self._adaptive_params['dynamic_k'], len(indices))
            top_indices_mask = np.argsort(combined_scores)[-k_final:]
            top_indices = indices[top_indices_mask]
            top_scores = combined_scores[top_indices_mask]
            
            # Extract similar trade data
            similar_pnls = [self._pnls[i] for i in top_indices if i < len(self._pnls)]
            similar_actions = [self._actions[i] for i in top_indices if i < len(self._actions)]
            similar_contexts = [self._contexts[i] for i in top_indices if i < len(self._contexts)]
            
            # Calculate weighted predictions
            expected_pnl = self._calculate_expected_pnl(similar_pnls, top_scores)
            suggested_action = self._calculate_suggested_action(similar_actions, similar_pnls, top_scores)
            confidence = self._calculate_recall_confidence(similar_pnls, top_scores, context_scores)
            
            # Create result
            result = {
                'recall_id': recall_id,
                'expected_pnl': expected_pnl,
                'confidence': confidence,
                'suggested_action': suggested_action,
                'similar_trades': len(top_indices),
                'best_similar_pnl': float(max(similar_pnls)) if similar_pnls else 0.0,
                'avg_similarity': float(np.mean(top_scores)),
                'context_match': float(np.mean(context_scores[top_indices_mask])) if context else 1.0,
                'profitable_matches': sum(1 for pnl in similar_pnls if pnl > 0),
                'pattern_strength': self._assess_pattern_strength(similar_contexts, context)
            }
            
            self.log_operator_info(
                f"Recall #{recall_id} completed",
                expected_pnl=f"€{expected_pnl:.2f}",
                confidence=f"{confidence:.3f}",
                similar_trades=len(top_indices),
                profitable_matches=result['profitable_matches'],
                pattern_strength=f"{result['pattern_strength']:.3f}"
            )
            
            return result
            
        except Exception as e:
            self.log_operator_error(f"Enhanced recall failed: {e}")
            return self._create_empty_recall_result(recall_id)

    def _calculate_context_similarity(self, ctx1: Dict[str, Any], ctx2: Dict[str, Any]) -> float:
        """Enhanced context similarity calculation"""
        
        try:
            score = 1.0
            
            # Regime matching (high importance)
            if ctx1.get('regime') == ctx2.get('regime'):
                score *= self.context_weight
            else:
                score *= 0.6
            
            # Volatility similarity
            vol1 = ctx1.get('volatility_level', 'medium')
            vol2 = ctx2.get('volatility_level', 'medium')
            vol_levels = {'low': 1, 'medium': 2, 'high': 3, 'extreme': 4}
            vol_diff = abs(vol_levels.get(vol1, 2) - vol_levels.get(vol2, 2))
            score *= max(0.5, 1.0 - vol_diff * 0.2)
            
            # Session matching
            if ctx1.get('session') == ctx2.get('session'):
                score *= 1.2
            else:
                score *= 0.8
            
            # Risk level similarity
            dd1 = ctx1.get('drawdown_pct', 0)
            dd2 = ctx2.get('drawdown_pct', 0)
            dd_diff = abs(dd1 - dd2)
            score *= max(0.7, 1.0 - dd_diff / 20.0)  # 20% difference = 0.7 score
            
            return min(score, 2.0)  # Cap at 2.0
            
        except Exception as e:
            self.log_operator_warning(f"Context similarity calculation failed: {e}")
            return 1.0

    def _calculate_expected_pnl(self, similar_pnls: List[float], scores: np.ndarray) -> float:
        """Calculate weighted expected PnL"""
        
        if not similar_pnls:
            return 0.0
        
        # Apply profit bias to scores
        profit_adjusted_scores = []
        for pnl, score in zip(similar_pnls, scores):
            if pnl > 0:
                adjusted_score = score * self._adaptive_params['profit_bias']
            else:
                adjusted_score = score / self._adaptive_params['profit_bias']
            profit_adjusted_scores.append(adjusted_score)
        
        profit_adjusted_scores = np.array(profit_adjusted_scores)
        weights = profit_adjusted_scores / (profit_adjusted_scores.sum() + 1e-8)
        
        expected_pnl = float(np.average(similar_pnls, weights=weights))
        return expected_pnl

    def _calculate_suggested_action(self, similar_actions: List[np.ndarray], 
                                   similar_pnls: List[float], scores: np.ndarray) -> np.ndarray:
        """Calculate suggested action from profitable trades"""
        
        if not similar_actions:
            return np.zeros(2, dtype=np.float32)
        
        # Focus on profitable trades
        profitable_indices = [i for i, pnl in enumerate(similar_pnls) if pnl > 0]
        
        if profitable_indices:
            profitable_actions = [similar_actions[i] for i in profitable_indices]
            profitable_scores = scores[profitable_indices]
            
            # Normalize scores
            weights = profitable_scores / (profitable_scores.sum() + 1e-8)
            suggested_action = np.average(profitable_actions, axis=0, weights=weights)
        else:
            # Fallback to all trades if no profitable ones
            weights = scores / (scores.sum() + 1e-8)
            suggested_action = np.average(similar_actions, axis=0, weights=weights)
        
        return suggested_action.astype(np.float32)

    def _calculate_recall_confidence(self, similar_pnls: List[float], 
                                   scores: np.ndarray, context_scores: np.ndarray) -> float:
        """Calculate confidence in recall result"""
        
        if not similar_pnls:
            return 0.0
        
        # Base confidence from similarity
        similarity_confidence = float(np.mean(scores))
        
        # Context confidence
        context_confidence = float(np.mean(context_scores))
        
        # PnL consistency confidence
        if len(similar_pnls) > 1:
            pnl_std = np.std(similar_pnls)
            pnl_confidence = np.exp(-pnl_std / 50.0)  # Lower std = higher confidence
        else:
            pnl_confidence = 0.5
        
        # Profitable ratio confidence
        profitable_ratio = sum(1 for pnl in similar_pnls if pnl > 0) / len(similar_pnls)
        profit_confidence = profitable_ratio
        
        # Combined confidence
        confidence = (0.4 * similarity_confidence + 
                     0.2 * context_confidence + 
                     0.2 * pnl_confidence + 
                     0.2 * profit_confidence)
        
        return float(np.clip(confidence, 0.0, 1.0))

    def _assess_pattern_strength(self, similar_contexts: List[Dict[str, Any]], 
                                query_context: Optional[Dict[str, Any]]) -> float:
        """Assess strength of identified pattern"""
        
        if not similar_contexts or not query_context:
            return 0.5
        
        # Count context pattern occurrences
        pattern_key = f"{query_context.get('regime', 'unknown')}_{query_context.get('volatility_level', 'medium')}_{query_context.get('session', 'unknown')}"
        
        if pattern_key in self._pattern_effectiveness:
            pattern_data = self._pattern_effectiveness[pattern_key]
            total_trades = pattern_data['wins'] + pattern_data['losses']
            if total_trades > 0:
                win_rate = pattern_data['wins'] / total_trades
                avg_pnl = pattern_data['total_pnl'] / total_trades
                
                # Combine win rate and average PnL for strength
                strength = (win_rate * 0.7 + min(1.0, avg_pnl / 50.0) * 0.3)
                return float(np.clip(strength, 0.0, 1.0))
        
        return 0.5

    def _create_empty_recall_result(self, recall_id: int) -> Dict[str, Any]:
        """Create empty recall result for bootstrap scenarios"""
        
        return {
            'recall_id': recall_id,
            'expected_pnl': 0.0,
            'confidence': 0.0,
            'suggested_action': np.zeros(2, dtype=np.float32),
            'similar_trades': 0,
            'best_similar_pnl': 0.0,
            'avg_similarity': 0.0,
            'context_match': 0.0,
            'profitable_matches': 0,
            'pattern_strength': 0.0
        }

    def _track_recall_performance(self, recall_result: Dict[str, Any], context: Optional[Dict[str, Any]]):
        """Track recall performance for continuous improvement"""
        
        recall_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'recall_id': recall_result['recall_id'],
            'confidence': recall_result['confidence'],
            'similar_trades': recall_result['similar_trades'],
            'expected_pnl': recall_result['expected_pnl'],
            'pattern_strength': recall_result['pattern_strength'],
            'context': context.copy() if context else {}
        }
        
        self._recall_history.append(recall_record)
        
        # Update similarity scores for analysis
        self._similarity_scores.append(recall_result['avg_similarity'])
        
        # Update performance metrics
        self._update_performance_metric('recall_confidence', recall_result['confidence'])
        self._update_performance_metric('pattern_strength', recall_result['pattern_strength'])

    def _update_playbook_performance(self):
        """Update playbook performance metrics"""
        
        try:
            # Memory utilization
            memory_utilization = len(self._features) / self.max_entries
            
            # Pattern diversity
            unique_patterns = len(self._pattern_effectiveness)
            self._pattern_diversity = min(1.0, unique_patterns / 20.0)  # Normalize to 20 patterns
            
            # Recall efficiency
            if self._recall_history:
                recent_recalls = list(self._recall_history)[-10:]
                avg_confidence = np.mean([r['confidence'] for r in recent_recalls])
                self._recall_efficiency = avg_confidence
            
            # Prediction accuracy (based on profitable patterns)
            if self._pattern_effectiveness:
                total_profitable = sum(p['wins'] for p in self._pattern_effectiveness.values())
                total_trades = sum(p['wins'] + p['losses'] for p in self._pattern_effectiveness.values())
                self._prediction_accuracy = total_profitable / max(total_trades, 1)
            
            # Update performance metrics
            self._update_performance_metric('memory_utilization', memory_utilization)
            self._update_performance_metric('pattern_diversity', self._pattern_diversity)
            self._update_performance_metric('recall_efficiency', self._recall_efficiency)
            self._update_performance_metric('prediction_accuracy', self._prediction_accuracy)
            
        except Exception as e:
            self.log_operator_warning(f"Performance update failed: {e}")

    def _adapt_parameters(self):
        """Adapt parameters based on performance"""
        
        try:
            # Adapt k based on memory utilization and recall quality
            if len(self._recall_history) >= 10:
                recent_confidences = [r['confidence'] for r in list(self._recall_history)[-10:]]
                avg_confidence = np.mean(recent_confidences)
                
                if avg_confidence < 0.4:  # Low confidence, try more neighbors
                    self._adaptive_params['dynamic_k'] = min(self.k * 2, len(self._features))
                elif avg_confidence > 0.8:  # High confidence, can use fewer neighbors
                    self._adaptive_params['dynamic_k'] = max(self.k // 2, 3)
                else:
                    self._adaptive_params['dynamic_k'] = self.k
            
            # Adapt context importance based on pattern success
            if self._pattern_effectiveness:
                pattern_success_rates = []
                for pattern_data in self._pattern_effectiveness.values():
                    total = pattern_data['wins'] + pattern_data['losses']
                    if total > 0:
                        pattern_success_rates.append(pattern_data['wins'] / total)
                
                if pattern_success_rates:
                    avg_pattern_success = np.mean(pattern_success_rates)
                    if avg_pattern_success > 0.6:
                        self._adaptive_params['context_importance'] = min(3.0, self.context_weight * 1.1)
                    else:
                        self._adaptive_params['context_importance'] = max(0.5, self.context_weight * 0.9)
            
            # Adapt profit bias based on recent performance
            if len(self._pnls) >= 20:
                recent_pnls = self._pnls[-20:]
                profitable_ratio = sum(1 for pnl in recent_pnls if pnl > 0) / len(recent_pnls)
                
                if profitable_ratio > 0.6:
                    self._adaptive_params['profit_bias'] = min(5.0, self.profit_weight * 1.05)
                elif profitable_ratio < 0.4:
                    self._adaptive_params['profit_bias'] = max(1.0, self.profit_weight * 0.95)
                
        except Exception as e:
            self.log_operator_warning(f"Parameter adaptation failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with playbook metrics"""
        
        try:
            # Memory statistics
            memory_utilization = len(self._features) / self.max_entries
            avg_pnl = self._total_pnl / max(self._trades_processed, 1)
            win_rate = self._get_win_rate()
            
            # Recent performance (last 20 trades)
            recent_pnls = self._pnls[-20:] if self._pnls else [0]
            recent_avg = np.mean(recent_pnls)
            
            # Pattern effectiveness
            pattern_strength = 0.0
            if self._pattern_effectiveness:
                pattern_scores = []
                for pattern_data in self._pattern_effectiveness.values():
                    total = pattern_data['wins'] + pattern_data['losses']
                    if total > 0:
                        score = (pattern_data['wins'] / total) * (pattern_data['total_pnl'] / total / 50.0)
                        pattern_scores.append(score)
                pattern_strength = np.mean(pattern_scores) if pattern_scores else 0.0
            
            # Recall performance
            recall_effectiveness = 0.0
            if self._recall_history:
                recent_recalls = list(self._recall_history)[-10:]
                recall_effectiveness = np.mean([r['confidence'] for r in recent_recalls])
            
            # Adaptive parameters status
            dynamic_k_ratio = self._adaptive_params['dynamic_k'] / self.k
            context_importance = self._adaptive_params['context_importance']
            
            # Combine all components
            observation = np.array([
                memory_utilization,
                avg_pnl / 100.0,  # Normalize
                win_rate,
                recent_avg / 100.0,  # Normalize
                float(len(self._features)) / 100.0,  # Normalize
                pattern_strength,
                recall_effectiveness,
                self._memory_quality_score / 100.0,
                self._prediction_accuracy,
                self._pattern_diversity,
                dynamic_k_ratio,
                context_importance
            ], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(12, dtype=np.float32)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Propose actions based on playbook recall"""
        
        if len(self._features) == 0 or self._nbrs is None:
            return np.zeros(2, dtype=np.float32)
        
        try:
            # Extract current features from InfoBus or observation
            if info_bus:
                current_features = self._extract_current_features_from_info_bus(info_bus)
                context = extract_standard_context(info_bus)
            elif obs is not None:
                current_features = np.asarray(obs, dtype=np.float32).flatten()
                context = None
            else:
                return np.zeros(2, dtype=np.float32)
            
            # Recall similar situations
            recall_result = self.recall(current_features, context, info_bus)
            
            # Extract suggested action
            suggested_action = recall_result.get('suggested_action', np.zeros(2))
            confidence = recall_result.get('confidence', 0.0)
            
            # Scale action by confidence
            scaled_action = suggested_action * max(0.3, confidence)  # Minimum 30% scaling
            
            return scaled_action.astype(np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Action proposal failed: {e}")
            return np.zeros(2, dtype=np.float32)

    def _extract_current_features_from_info_bus(self, info_bus: InfoBus) -> np.ndarray:
        """Extract current market features for recall"""
        
        # Use similar feature extraction as in trade recording
        features = []
        
        # Market regime features
        regime = InfoBusExtractor.get_market_regime(info_bus)
        regime_encoding = {'trending': [1, 0, 0], 'volatile': [0, 1, 0], 'ranging': [0, 0, 1], 'unknown': [0.33, 0.33, 0.33]}
        features.extend(regime_encoding.get(regime, [0.33, 0.33, 0.33]))
        
        # Volatility features
        vol_level = InfoBusExtractor.get_volatility_level(info_bus)
        vol_value = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0}.get(vol_level, 0.5)
        features.append(vol_value)
        
        # Risk context
        features.extend([
            InfoBusExtractor.get_drawdown_pct(info_bus) / 100.0,
            InfoBusExtractor.get_exposure_pct(info_bus) / 100.0,
            InfoBusExtractor.get_position_count(info_bus) / 10.0
        ])
        
        # Session features
        session = InfoBusExtractor.get_session(info_bus)
        session_encoding = {'asian': [1, 0, 0], 'european': [0, 1, 0], 'american': [0, 0, 1], 'closed': [0, 0, 0]}
        features.extend(session_encoding.get(session, [0.25, 0.25, 0.25]))
        
        # Current context features
        features.extend([
            0.0,  # No specific trade size
            0.5,  # Default confidence
            0.0   # No specific side
        ])
        
        # Price context
        prices = info_bus.get('prices', {})
        if prices:
            price_values = list(prices.values())[:2]  # First 2 instruments
            normalized_prices = [min(2.0, max(0.5, p)) / 2.0 for p in price_values]
            features.extend(normalized_prices)
        else:
            features.extend([0.5, 0.5])
        
        return np.array(features, dtype=np.float32)

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return confidence in playbook-based recommendations"""
        
        base_confidence = 0.5
        
        # Confidence from memory quality
        if len(self._features) > 0:
            memory_quality = self._memory_quality_score / 100.0
            base_confidence += memory_quality * 0.3
        
        # Confidence from pattern effectiveness
        base_confidence += self._prediction_accuracy * 0.3
        
        # Confidence from recall efficiency
        base_confidence += self._recall_efficiency * 0.2
        
        # Confidence from data volume
        data_confidence = min(0.2, len(self._features) / self.max_entries * 0.2)
        base_confidence += data_confidence
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def prune_low_quality_memories(self, quality_threshold: Optional[float] = None):
        """Prune low-quality memories based on performance"""
        
        if len(self._features) == 0:
            return
        
        threshold = quality_threshold if quality_threshold is not None else self._adaptive_params['quality_threshold']
        
        # Calculate quality scores for each memory
        quality_scores = []
        for i, (pnl, timestamp) in enumerate(zip(self._pnls, self._timestamps)):
            # Base score from PnL
            pnl_score = np.tanh(pnl / 50.0)  # Normalize to [-1, 1]
            
            # Recency score
            age_days = (datetime.datetime.now() - timestamp).days
            recency_score = np.exp(-age_days / 30.0)  # Decay over 30 days
            
            # Combined quality score
            quality_score = (pnl_score + 1) / 2 * recency_score  # [0, 1] range
            quality_scores.append(quality_score)
        
        # Keep memories above threshold
        keep_indices = [i for i, score in enumerate(quality_scores) if score >= threshold]
        
        if len(keep_indices) < len(self._features):
            # Prune low-quality memories
            self._features = [self._features[i] for i in keep_indices]
            self._actions = [self._actions[i] for i in keep_indices]
            self._pnls = [self._pnls[i] for i in keep_indices]
            self._contexts = [self._contexts[i] for i in keep_indices]
            self._timestamps = [self._timestamps[i] for i in keep_indices]
            self._trade_metadata = [self._trade_metadata[i] for i in keep_indices]
            
            pruned_count = len(quality_scores) - len(keep_indices)
            self.log_operator_info(
                f"Memory pruning completed",
                pruned=pruned_count,
                remaining=len(keep_indices),
                threshold=f"{threshold:.3f}"
            )
            
            # Refit models if significant pruning occurred
            if pruned_count > 5 and len(self._features) >= self.k:
                self._fit_models()

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome"""
        self.max_entries = int(np.clip(genome.get("max_entries", self.max_entries), 100, 1000))
        self.k = int(np.clip(genome.get("k", self.k), 3, 20))
        self.profit_weight = float(np.clip(genome.get("profit_weight", self.profit_weight), 1.0, 5.0))
        self.context_weight = float(np.clip(genome.get("context_weight", self.context_weight), 1.0, 3.0))
        self.similarity_threshold = float(np.clip(genome.get("similarity_threshold", self.similarity_threshold), 0.3, 0.9))
        self.memory_decay = float(np.clip(genome.get("memory_decay", self.memory_decay), 0.90, 0.99))
        self.context_features_weight = float(np.clip(genome.get("context_features_weight", self.context_features_weight), 0.1, 0.5))
        self.recency_weight = float(np.clip(genome.get("recency_weight", self.recency_weight), 0.05, 0.3))
        
        self.genome = {
            "max_entries": self.max_entries,
            "k": self.k,
            "profit_weight": self.profit_weight,
            "context_weight": self.context_weight,
            "similarity_threshold": self.similarity_threshold,
            "memory_decay": self.memory_decay,
            "context_features_weight": self.context_features_weight,
            "recency_weight": self.recency_weight
        }
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with performance-based adjustments"""
        g = self.genome.copy()
        mutations = []
        
        if np.random.rand() < mutation_rate:
            old_val = g["max_entries"]
            g["max_entries"] = int(np.clip(old_val + np.random.randint(-50, 51), 100, 1000))
            mutations.append(f"max_entries: {old_val} → {g['max_entries']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["k"]
            g["k"] = int(np.clip(old_val + np.random.choice([-1, 0, 1]), 3, 20))
            mutations.append(f"k: {old_val} → {g['k']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["profit_weight"]
            g["profit_weight"] = float(np.clip(old_val + np.random.uniform(-0.3, 0.3), 1.0, 5.0))
            mutations.append(f"profit_weight: {old_val:.2f} → {g['profit_weight']:.2f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["context_weight"]
            g["context_weight"] = float(np.clip(old_val + np.random.uniform(-0.2, 0.2), 1.0, 3.0))
            mutations.append(f"context_weight: {old_val:.2f} → {g['context_weight']:.2f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["memory_decay"]
            g["memory_decay"] = float(np.clip(old_val + np.random.uniform(-0.02, 0.02), 0.90, 0.99))
            mutations.append(f"memory_decay: {old_val:.3f} → {g['memory_decay']:.3f}")
        
        if mutations:
            self.log_operator_info(f"Playbook mutation applied", changes=", ".join(mutations))
            
        self.set_genome(g)
        
    def crossover(self, other: "PlaybookMemory") -> "PlaybookMemory":
        """Enhanced crossover with performance-based selection"""
        if not isinstance(other, PlaybookMemory):
            self.log_operator_warning("Crossover with incompatible type")
            return self
        
        # Performance-based crossover
        self_performance = self._memory_quality_score
        other_performance = other._memory_quality_score
        
        # Favor higher performance parent
        if self_performance > other_performance:
            bias = 0.7  # Favor self
        else:
            bias = 0.3  # Favor other
        
        new_g = {k: (self.genome[k] if np.random.rand() < bias else other.genome[k]) for k in self.genome}
        
        child = PlaybookMemory(genome=new_g, debug=self.config.debug)
        
        # Inherit memories from better parent
        if self_performance > other_performance:
            if self._features:
                child._features = self._features.copy()
                child._actions = self._actions.copy()
                child._pnls = self._pnls.copy()
                child._contexts = self._contexts.copy()
                child._timestamps = self._timestamps.copy()
                child._trade_metadata = self._trade_metadata.copy()
                
                # Refit models
                if len(child._features) >= child.k:
                    child._fit_models()
        else:
            if other._features:
                child._features = other._features.copy()
                child._actions = other._actions.copy()
                child._pnls = other._pnls.copy()
                child._contexts = other._contexts.copy()
                child._timestamps = other._timestamps.copy()
                child._trade_metadata = other._trade_metadata.copy()
                
                # Refit models
                if len(child._features) >= child.k:
                    child._fit_models()
        
        return child

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check data consistency
            data_lengths = [len(self._features), len(self._actions), len(self._pnls), 
                          len(self._contexts), len(self._timestamps)]
            if not all(length == data_lengths[0] for length in data_lengths):
                return False
                
            # Check value ranges
            if self._pnls and not all(np.isfinite(pnl) for pnl in self._pnls):
                return False
                
            # Check features validity
            for feature_array in self._features:
                if not np.all(np.isfinite(feature_array)):
                    return False
                    
            # Check actions validity
            for action_array in self._actions:
                if not np.all(np.isfinite(action_array)):
                    return False
                    
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        playbook_details = {
            'memory_info': {
                'total_memories': len(self._features),
                'max_capacity': self.max_entries,
                'utilization': len(self._features) / self.max_entries,
                'profitable_memories': sum(1 for pnl in self._pnls if pnl > 0),
                'avg_memory_pnl': np.mean(self._pnls) if self._pnls else 0.0
            },
            'pattern_info': {
                'unique_patterns': len(self._pattern_effectiveness),
                'pattern_diversity': self._pattern_diversity,
                'most_profitable_pattern': max(self._pattern_effectiveness.items(), 
                                             key=lambda x: x[1]['total_pnl'])[0] if self._pattern_effectiveness else 'none'
            },
            'recall_info': {
                'total_recalls': len(self._recall_history),
                'avg_recall_confidence': np.mean([r['confidence'] for r in self._recall_history]) if self._recall_history else 0.0,
                'recall_efficiency': self._recall_efficiency
            },
            'performance_info': {
                'memory_quality_score': self._memory_quality_score,
                'prediction_accuracy': self._prediction_accuracy,
                'k_neighbors': self._adaptive_params['dynamic_k'],
                'context_importance': self._adaptive_params['context_importance']
            },
            'model_info': {
                'knn_fitted': self._nbrs is not None,
                'weighted_knn_fitted': self._weighted_nbrs is not None,
                'scaler_fitted': hasattr(self._scaler, 'scale_') and self._scaler.scale_ is not None
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(playbook_details)
            return base_details
        
        return playbook_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        
        return {
            "features": [f.tolist() for f in self._features],
            "actions": [a.tolist() for a in self._actions],
            "pnls": self._pnls,
            "contexts": self._contexts,
            "timestamps": [t.isoformat() for t in self._timestamps],
            "trade_metadata": self._trade_metadata[-50:],  # Keep recent metadata only
            "genome": self.genome.copy(),
            "adaptive_params": self._adaptive_params.copy(),
            "pattern_effectiveness": dict(self._pattern_effectiveness),
            "context_patterns": dict(self._context_patterns),
            "performance_metrics": {
                'memory_quality_score': self._memory_quality_score,
                'prediction_accuracy': self._prediction_accuracy,
                'pattern_diversity': self._pattern_diversity,
                'recall_efficiency': self._recall_efficiency
            },
            "recall_history": list(self._recall_history)[-30:],  # Keep recent recalls
            "similarity_scores": list(self._similarity_scores)[-20:],
            "success_patterns": list(self._success_patterns)[-30:],
            "scaler_params": {
                'mean_': self._scaler.mean_.tolist() if hasattr(self._scaler, 'mean_') and self._scaler.mean_ is not None else None,
                'scale_': self._scaler.scale_.tolist() if hasattr(self._scaler, 'scale_') and self._scaler.scale_ is not None else None
            }
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        # Restore core data
        self._features = [np.array(f, dtype=np.float32) for f in module_state.get("features", [])]
        self._actions = [np.array(a, dtype=np.float32) for a in module_state.get("actions", [])]
        self._pnls = module_state.get("pnls", [])
        self._contexts = module_state.get("contexts", [])
        self._timestamps = [datetime.datetime.fromisoformat(t) for t in module_state.get("timestamps", [])]
        self._trade_metadata = module_state.get("trade_metadata", [])
        
        # Restore genome and parameters
        self.set_genome(module_state.get("genome", self.genome))
        self._adaptive_params = module_state.get("adaptive_params", self._adaptive_params)
        
        # Restore pattern tracking
        self._pattern_effectiveness = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
                                                 module_state.get("pattern_effectiveness", {}))
        self._context_patterns = defaultdict(int, module_state.get("context_patterns", {}))
        
        # Restore performance metrics
        performance_metrics = module_state.get("performance_metrics", {})
        self._memory_quality_score = performance_metrics.get('memory_quality_score', 0.0)
        self._prediction_accuracy = performance_metrics.get('prediction_accuracy', 0.0)
        self._pattern_diversity = performance_metrics.get('pattern_diversity', 0.0)
        self._recall_efficiency = performance_metrics.get('recall_efficiency', 0.0)
        
        # Restore tracking data
        self._recall_history = deque(module_state.get("recall_history", []), maxlen=100)
        self._similarity_scores = deque(module_state.get("similarity_scores", []), maxlen=50)
        self._success_patterns = deque(module_state.get("success_patterns", []), maxlen=100)
        
        # Restore scaler
        scaler_params = module_state.get("scaler_params", {})
        if scaler_params.get('mean_') and scaler_params.get('scale_'):
            self._scaler.mean_ = np.array(scaler_params['mean_'])
            self._scaler.scale_ = np.array(scaler_params['scale_'])
        
        # Refit models if we have enough data
        if len(self._features) >= self.k:
            self._fit_models()

    def get_playbook_report(self) -> str:
        """Generate operator-friendly playbook report"""
        
        # Memory statistics
        memory_util = len(self._features) / self.max_entries
        profitable_count = sum(1 for pnl in self._pnls if pnl > 0)
        total_trades = len(self._pnls)
        
        # Performance status
        if self._memory_quality_score > 80:
            performance_status = "🚀 Excellent"
        elif self._memory_quality_score > 60:
            performance_status = "✅ Good"
        elif self._memory_quality_score > 40:
            performance_status = "⚡ Fair"
        else:
            performance_status = "⚠️ Poor"
        
        # Pattern effectiveness
        best_pattern = "None"
        if self._pattern_effectiveness:
            best_pattern_data = max(self._pattern_effectiveness.items(), key=lambda x: x[1]['total_pnl'])
            best_pattern = best_pattern_data[0]
            best_pnl = best_pattern_data[1]['total_pnl']
        
        return f"""
📚 PLAYBOOK MEMORY
═══════════════════════════════════════
💾 Memory: {total_trades:,}/{self.max_entries:,} ({memory_util:.1%})
🎯 Performance: {performance_status} ({self._memory_quality_score:.1f})
💰 Profitable: {profitable_count}/{total_trades} ({profitable_count/max(total_trades,1):.1%})
📈 Prediction Accuracy: {self._prediction_accuracy:.3f}

🏗️ CONFIGURATION
• K-Neighbors: {self._adaptive_params['dynamic_k']} (base: {self.k})
• Profit Weight: {self.profit_weight:.2f}
• Context Weight: {self.context_weight:.2f}
• Memory Decay: {self.memory_decay:.3f}

📊 PATTERN ANALYSIS
• Unique Patterns: {len(self._pattern_effectiveness)}
• Pattern Diversity: {self._pattern_diversity:.3f}
• Best Pattern: {best_pattern}
• Context Importance: {self._adaptive_params['context_importance']:.2f}

🔍 RECALL PERFORMANCE
• Total Recalls: {len(self._recall_history)}
• Avg Confidence: {self._recall_efficiency:.3f}
• Recent Quality: {np.mean([r['confidence'] for r in list(self._recall_history)[-5:]]) if len(self._recall_history) >= 5 else 0:.3f}

🧠 MODEL STATUS
• KNN Fitted: {'✅' if self._nbrs is not None else '❌'}
• Weighted KNN: {'✅' if self._weighted_nbrs is not None else '❌'}
• Scaler Ready: {'✅' if hasattr(self._scaler, 'scale_') and self._scaler.scale_ is not None else '❌'}

💡 RECENT ACTIVITY
• Recalls (last hour): {len([r for r in self._recall_history if (datetime.datetime.now() - datetime.datetime.fromisoformat(r['timestamp'])).total_seconds() < 3600])}
• High-confidence recalls: {len([r for r in self._recall_history if r['confidence'] > 0.7])}
• Profitable patterns: {len([p for p in self._pattern_effectiveness.values() if p['total_pnl'] > 0])}
        """

    # Maintain backward compatibility
    def step(self, features: np.ndarray = None, actions: np.ndarray = None, 
             pnl: float = None, context: Dict[str, Any] = None, **kwargs):
        """Backward compatibility step method"""
        if features is not None and actions is not None and pnl is not None:
            kwargs.update({'features': features, 'actions': actions, 'pnl': pnl, 'context': context})
        self._step_impl(None, **kwargs)

    def record(self, features: np.ndarray, actions: np.ndarray, pnl: float, 
               context: Optional[Dict] = None):
        """Backward compatibility record method"""
        self._record_trade_data(features, actions, pnl, context or {})

    def get_state(self) -> Dict[str, Any]:
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state: Dict[str, Any]):
        """Backward compatibility state method"""
        super().set_state(state)