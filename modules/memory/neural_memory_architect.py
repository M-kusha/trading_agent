# ─────────────────────────────────────────────────────────────
# File: modules/memory/neural_memory_architect.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List
from collections import deque
import datetime
import random

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class NeuralMemoryArchitect(Module, AnalysisMixin, TradingMixin):
    def __init__(self,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 max_len: int = 500,
                 memory_decay: float = 0.95,
                 debug: bool = True,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # --- ensure these attributes exist before Module.__init__ triggers _initialize_module_state()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_len = max_len
        self.memory_decay = memory_decay

        config = ModuleConfig(
            debug=debug,
            max_history=400,
            **kwargs
        )
        super().__init__(config)

        # now override from genome if provided
        self._initialize_genome_parameters(genome, embed_dim, num_heads, max_len, memory_decay)

        # Module.__init__ already called _initialize_module_state(), so
        # no need to call it again here

        # initialize neural nets
        self._initialize_neural_components()

        self.log_operator_info(
            "Neural memory architect initialized",
            embedding_dim=self.embed_dim,
            attention_heads=self.num_heads,
            max_memories=self.max_len,
            memory_decay=f"{self.memory_decay:.3f}",
            device="CPU"
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict], embed_dim: int, 
                                    num_heads: int, max_len: int, memory_decay: float):
        """Initialize genome-based parameters"""
        if genome:
            self.embed_dim = int(genome.get("embed_dim", embed_dim))
            self.num_heads = int(genome.get("num_heads", num_heads))
            self.max_len = int(genome.get("max_len", max_len))
            self.memory_decay = float(genome.get("memory_decay", memory_decay))
            self.importance_threshold = float(genome.get("importance_threshold", 0.3))
            self.retrieval_top_k = int(genome.get("retrieval_top_k", 5))
            self.learning_rate = float(genome.get("learning_rate", 0.001))
            self.attention_dropout = float(genome.get("attention_dropout", 0.1))
        else:
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.max_len = max_len
            self.memory_decay = memory_decay
            self.importance_threshold = 0.3
            self.retrieval_top_k = 5
            self.learning_rate = 0.001
            self.attention_dropout = 0.1

        # Store genome for evolution
        self.genome = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "max_len": self.max_len,
            "memory_decay": self.memory_decay,
            "importance_threshold": self.importance_threshold,
            "retrieval_top_k": self.retrieval_top_k,
            "learning_rate": self.learning_rate,
            "attention_dropout": self.attention_dropout
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Memory storage
        self.buffer = torch.zeros((0, self.embed_dim), dtype=torch.float32)
        self.importance_scores = torch.zeros(0, dtype=torch.float32)
        self.memory_metadata = []
        
        # Enhanced tracking
        self._memory_usage_history = deque(maxlen=200)
        self._retrieval_history = deque(maxlen=100)
        self._importance_evolution = deque(maxlen=500)
        self._attention_patterns = deque(maxlen=50)
        
        # Performance analytics
        self._storage_efficiency = 0.0
        self._retrieval_accuracy = 0.0
        self._memory_turnover_rate = 0.0
        self._neural_performance_score = 100.0
        
        # Learning analytics
        self._learning_curves = {
            'importance_prediction': deque(maxlen=100),
            'attention_focus': deque(maxlen=100),
            'memory_utilization': deque(maxlen=100)
        }
        
        # Adaptive parameters
        self._adaptive_params = {
            'importance_scaling': 1.0,
            'attention_temperature': 1.0,
            'decay_adjustment': 1.0,
            'quality_threshold': 0.5
        }

    def _initialize_neural_components(self):
        """Initialize neural network components with enhanced monitoring"""
        
        try:
            # Set device to CPU
            self.device = torch.device("cpu")
            
            # Encoder network
            self.encoder = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            ).to(self.device)
            
            # Multi-head attention
            self.attn = nn.MultiheadAttention(
                self.embed_dim, 
                self.num_heads,
                dropout=self.attention_dropout,
                batch_first=True
            ).to(self.device)
            
            # Importance prediction head
            self.value_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(self.embed_dim // 2, 1),
                nn.Sigmoid()
            ).to(self.device)
            
            # Context integration network
            self.context_net = nn.Sequential(
                nn.Linear(self.embed_dim + 8, self.embed_dim),  # +8 for context features
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            ).to(self.device)
            
            # Initialize weights
            self._initialize_weights()
            
            self.log_operator_info("Neural components initialized successfully")
            
        except Exception as e:
            self.log_operator_error(f"Neural component initialization failed: {e}")
            self._update_health_status("ERROR", f"Neural init failed: {e}")

    def _initialize_weights(self):
        """Initialize neural network weights"""
        
        for module in [self.encoder, self.value_head, self.context_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_trading_state()
        
        # Clear memory buffers
        self.buffer = torch.zeros((0, self.embed_dim), dtype=torch.float32)
        self.importance_scores = torch.zeros(0, dtype=torch.float32)
        self.memory_metadata.clear()
        
        # Reset tracking
        self._memory_usage_history.clear()
        self._retrieval_history.clear()
        self._importance_evolution.clear()
        self._attention_patterns.clear()
        
        # Reset performance metrics
        self._storage_efficiency = 0.0
        self._retrieval_accuracy = 0.0
        self._memory_turnover_rate = 0.0
        self._neural_performance_score = 100.0
        
        # Reset learning curves
        for curve in self._learning_curves.values():
            curve.clear()
        
        # Reset adaptive parameters
        self._adaptive_params = {
            'importance_scaling': 1.0,
            'attention_temperature': 1.0,
            'decay_adjustment': 1.0,
            'quality_threshold': 0.5
        }

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract experience data
        experience_data = self._extract_experience_data(info_bus, kwargs)
        
        # Store experience if available
        if experience_data.get('source') != 'insufficient_data':
            self._store_experience(experience_data)
        
        # Update neural performance metrics
        self._update_neural_performance()
        
        # Adapt parameters based on performance
        self._adapt_parameters()

    def _extract_experience_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract experience data from InfoBus or kwargs"""
        
        # Try InfoBus first
        if info_bus:
            # Extract comprehensive market state
            market_context = info_bus.get('market_context', {})
            risk_data = info_bus.get('risk', {})
            recent_trades = info_bus.get('recent_trades', [])
            
            # Create observation from current state
            observation = self._extract_observation_from_info_bus(info_bus)
            
            # Calculate reward from recent performance
            reward = self._calculate_reward_from_info_bus(info_bus, recent_trades)
            
            # Extract action if available in kwargs
            action = kwargs.get('action', np.zeros(2))
            
            return {
                'observation': observation,
                'action': action,
                'reward': reward,
                'market_context': market_context,
                'risk_data': risk_data,
                'recent_trades': recent_trades,
                'step_idx': info_bus.get('step_idx', 0),
                'done': False,
                'source': 'info_bus'
            }
        
        # Try kwargs (backward compatibility)
        if 'experience' in kwargs:
            experience = kwargs['experience']
            if isinstance(experience, dict):
                return {**experience, 'source': 'kwargs'}
            else:
                # Convert array experience to dict
                obs = np.asarray(experience, dtype=np.float32).flatten()
                return {
                    'observation': obs,
                    'reward': 0.0,
                    'action': np.zeros(2),
                    'done': False,
                    'source': 'kwargs'
                }
        
        # Return insufficient data marker
        return {'source': 'insufficient_data'}

    def _extract_observation_from_info_bus(self, info_bus: InfoBus) -> np.ndarray:
        """Extract observation from InfoBus state"""
        
        observation = []
        
        # Market regime features
        regime = InfoBusExtractor.get_market_regime(info_bus)
        regime_encoding = {'trending': [1, 0, 0], 'volatile': [0, 1, 0], 'ranging': [0, 0, 1], 'unknown': [0.33, 0.33, 0.33]}
        observation.extend(regime_encoding.get(regime, [0.33, 0.33, 0.33]))
        
        # Volatility features
        vol_level = InfoBusExtractor.get_volatility_level(info_bus)
        vol_value = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0}.get(vol_level, 0.5)
        observation.append(vol_value)
        
        # Risk features
        observation.extend([
            InfoBusExtractor.get_drawdown_pct(info_bus) / 100.0,
            InfoBusExtractor.get_exposure_pct(info_bus) / 100.0,
            InfoBusExtractor.get_position_count(info_bus) / 10.0  # Normalize
        ])
        
        # Session features
        session = InfoBusExtractor.get_session(info_bus)
        session_encoding = {'asian': [1, 0, 0], 'european': [0, 1, 0], 'american': [0, 0, 1], 'closed': [0, 0, 0]}
        observation.extend(session_encoding.get(session, [0.25, 0.25, 0.25]))
        
        # Price and market features
        prices = info_bus.get('prices', {})
        if prices:
            price_values = list(prices.values())[:2]  # First 2 instruments
            # Normalize prices (assuming forex pairs around 1.0-2.0)
            normalized_prices = [min(2.0, max(0.5, p)) / 2.0 for p in price_values]
            observation.extend(normalized_prices)
        else:
            observation.extend([0.5, 0.5])
        
        # Market context features
        market_context = info_bus.get('market_context', {})
        if 'volatility' in market_context:
            vol_data = market_context['volatility']
            if isinstance(vol_data, dict):
                avg_vol = np.mean(list(vol_data.values()))
                observation.append(min(1.0, avg_vol * 50))
            else:
                observation.append(min(1.0, float(vol_data) * 50))
        else:
            observation.append(0.5)
        
        # Extend to embedding dimension
        while len(observation) < self.embed_dim:
            observation.append(0.0)
        
        return np.array(observation[:self.embed_dim], dtype=np.float32)

    def _calculate_reward_from_info_bus(self, info_bus: InfoBus, recent_trades: List[Dict]) -> float:
        """Calculate reward from recent trading performance"""
        
        if not recent_trades:
            return 0.0
        
        # Calculate total PnL from recent trades
        total_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
        
        # Normalize reward
        reward = np.tanh(total_pnl / 100.0)  # Scale to [-1, 1]
        
        # Adjust based on risk metrics
        drawdown = InfoBusExtractor.get_drawdown_pct(info_bus)
        if drawdown > 10:  # Penalize high drawdown
            reward *= 0.7
        
        exposure = InfoBusExtractor.get_exposure_pct(info_bus)
        if exposure > 80:  # Penalize high exposure
            reward *= 0.8
        
        return float(reward)

    def _store_experience(self, experience_data: Dict[str, Any]):
        """Enhanced experience storage with context integration"""
        
        try:
            # Extract observation
            obs = experience_data.get('observation', np.zeros(self.embed_dim))
            if isinstance(obs, np.ndarray):
                if obs.size < self.embed_dim:
                    obs = np.pad(obs, (0, self.embed_dim - obs.size), mode='constant')
                elif obs.size > self.embed_dim:
                    obs = obs[:self.embed_dim]
            else:
                obs = np.zeros(self.embed_dim, dtype=np.float32)
            
            # Create enhanced metadata
            metadata = {
                'reward': experience_data.get('reward', 0.0),
                'action': experience_data.get('action', np.zeros(2)),
                'done': experience_data.get('done', False),
                'step_idx': experience_data.get('step_idx', self._step_count),
                'timestamp': datetime.datetime.now().isoformat(),
                'market_context': experience_data.get('market_context', {}),
                'risk_data': experience_data.get('risk_data', {}),
                'source': experience_data.get('source', 'unknown')
            }
            
            # Add context features for enhanced encoding
            context_features = self._extract_context_features(experience_data)
            
            # Encode observation with context
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
            context_tensor = torch.from_numpy(context_features.astype(np.float32)).unsqueeze(0)
            
            with torch.no_grad():
                # Combine observation with context
                combined_input = torch.cat([obs_tensor, context_tensor], dim=1)
                encoded_obs = self.context_net(combined_input)
                
                # Calculate importance score
                importance = self.value_head(encoded_obs).squeeze()
                
                # Apply adaptive scaling
                importance = importance * self._adaptive_params['importance_scaling']
                
                # Store or replace memory
                self._store_in_buffer(encoded_obs.squeeze(0), importance, metadata)
            
            # Update storage metrics
            self._update_storage_metrics(importance.item(), metadata)
            
        except Exception as e:
            self.log_operator_error(f"Experience storage failed: {e}")

    def _extract_context_features(self, experience_data: Dict[str, Any]) -> np.ndarray:
        """Extract context features for enhanced encoding"""
        
        context_features = []
        
        # Market context
        market_context = experience_data.get('market_context', {})
        context_features.append(float(market_context.get('volatility', {}).get('EUR/USD', 0.01) * 100))
        context_features.append(float(market_context.get('volatility', {}).get('XAU/USD', 0.01) * 100))
        
        # Risk context
        risk_data = experience_data.get('risk_data', {})
        context_features.extend([
            risk_data.get('current_drawdown', 0.0),
            risk_data.get('margin_used', 0.0) / max(risk_data.get('equity', 1.0), 1.0),
            len(risk_data.get('open_positions', []))
        ])
        
        # Reward context
        reward = experience_data.get('reward', 0.0)
        context_features.extend([
            reward,
            np.abs(reward),
            np.sign(reward)
        ])
        
        # Ensure we have exactly 8 context features
        while len(context_features) < 8:
            context_features.append(0.0)
        
        return np.array(context_features[:8], dtype=np.float32)

    def _store_in_buffer(self, encoded_obs: torch.Tensor, importance: torch.Tensor, metadata: Dict):
        """Store memory in buffer with importance-based replacement"""
        
        if len(self.buffer) < self.max_len:
            # Add to buffer
            self.buffer = torch.cat([self.buffer, encoded_obs.unsqueeze(0)], dim=0)
            
            # Apply decay to existing importance scores
            self.importance_scores = self.importance_scores * self.memory_decay * self._adaptive_params['decay_adjustment']
            self.importance_scores = torch.cat([self.importance_scores, importance.unsqueeze(0)])
            
            self.memory_metadata.append(metadata)
            
            self.log_operator_info(
                f"Memory stored",
                buffer_size=f"{len(self.buffer)}/{self.max_len}",
                importance=f"{importance.item():.3f}",
                reward=f"{metadata.get('reward', 0):.3f}"
            )
        else:
            # Replace least important memory
            min_idx = torch.argmin(self.importance_scores).item()
            old_importance = self.importance_scores[min_idx].item()
            
            if importance.item() > old_importance:
                self.buffer[min_idx] = encoded_obs
                self.importance_scores[min_idx] = importance
                self.memory_metadata[min_idx] = metadata
                
                self.log_operator_info(
                    f"Memory replaced",
                    old_importance=f"{old_importance:.3f}",
                    new_importance=f"{importance.item():.3f}",
                    reward=f"{metadata.get('reward', 0):.3f}"
                )
            else:
                self.log_operator_info(
                    f"Memory rejected - insufficient importance",
                    importance=f"{importance.item():.3f}",
                    threshold=f"{old_importance:.3f}"
                )

    def _update_storage_metrics(self, importance_score: float, metadata: Dict):
        """Update storage performance metrics"""
        
        # Track importance evolution
        self._importance_evolution.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'importance': importance_score,
            'reward': metadata.get('reward', 0.0),
            'step': self._step_count
        })
        
        # Update learning curves
        self._learning_curves['importance_prediction'].append(importance_score)
        
        # Calculate storage efficiency
        if len(self.buffer) > 0:
            avg_importance = self.importance_scores.mean().item()
            self._storage_efficiency = avg_importance
            
        # Update performance metrics
        self._update_performance_metric('storage_efficiency', self._storage_efficiency)
        self._update_performance_metric('buffer_utilization', len(self.buffer) / self.max_len)

    def _update_neural_performance(self):
        """Update neural network performance metrics"""
        
        try:
            # Memory utilization
            memory_usage = len(self.buffer) / self.max_len
            self._memory_usage_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'usage': memory_usage,
                'avg_importance': self.importance_scores.mean().item() if len(self.importance_scores) > 0 else 0.0,
                'step': self._step_count
            })
            
            # Update learning curves
            self._learning_curves['memory_utilization'].append(memory_usage)
            
            # Calculate neural performance score
            if len(self._importance_evolution) >= 10:
                recent_importance = [entry['importance'] for entry in list(self._importance_evolution)[-10:]]
                importance_trend = np.polyfit(range(len(recent_importance)), recent_importance, 1)[0]
                
                # Score based on improving importance prediction
                trend_score = min(100, max(0, 50 + importance_trend * 1000))
                utilization_score = memory_usage * 100
                
                self._neural_performance_score = 0.6 * trend_score + 0.4 * utilization_score
            
            # Update performance metrics
            self._update_performance_metric('neural_performance', self._neural_performance_score)
            
        except Exception as e:
            self.log_operator_warning(f"Neural performance update failed: {e}")

    def _adapt_parameters(self):
        """Adapt neural network parameters based on performance"""
        
        try:
            # Adapt importance scaling based on recent storage success
            if len(self._importance_evolution) >= 20:
                recent_importance = [entry['importance'] for entry in list(self._importance_evolution)[-20:]]
                avg_importance = np.mean(recent_importance)
                
                if avg_importance < 0.3:
                    self._adaptive_params['importance_scaling'] = min(2.0, self._adaptive_params['importance_scaling'] * 1.05)
                elif avg_importance > 0.8:
                    self._adaptive_params['importance_scaling'] = max(0.5, self._adaptive_params['importance_scaling'] * 0.95)
            
            # Adapt decay based on memory turnover
            if len(self._memory_usage_history) >= 10:
                recent_usage = [entry['usage'] for entry in list(self._memory_usage_history)[-10:]]
                usage_stability = np.std(recent_usage)
                
                if usage_stability > 0.1:  # High turnover
                    self._adaptive_params['decay_adjustment'] = min(1.2, self._adaptive_params['decay_adjustment'] * 1.02)
                else:  # Stable memory
                    self._adaptive_params['decay_adjustment'] = max(0.8, self._adaptive_params['decay_adjustment'] * 0.98)
            
        except Exception as e:
            self.log_operator_warning(f"Parameter adaptation failed: {e}")

    def retrieve(self, query: np.ndarray, top_k: Optional[int] = None, 
                info_bus: Optional[InfoBus] = None) -> Dict[str, Any]:
        """Enhanced memory retrieval with attention and context awareness"""
        
        try:
            if self.buffer.size(0) == 0:
                return self._create_empty_retrieval_result()
            
            # Use provided top_k or default
            k = top_k if top_k is not None else self.retrieval_top_k
            k = min(k, len(self.buffer))
            
            # Prepare query
            if query.size < self.embed_dim:
                query = np.pad(query, (0, self.embed_dim - query.size), mode='constant')
            elif query.size > self.embed_dim:
                query = query[:self.embed_dim]
            
            # Add context if available
            if info_bus:
                context_features = self._extract_context_features({'market_context': info_bus.get('market_context', {}), 'risk_data': info_bus.get('risk', {})})
                enhanced_query = np.concatenate([query, context_features])
                
                # Process through context network
                query_tensor = torch.from_numpy(enhanced_query.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    processed_query = self.context_net(query_tensor).squeeze(0)
            else:
                processed_query = torch.from_numpy(query.astype(np.float32))
            
            # Attention-based retrieval
            retrieval_result = self._perform_attention_retrieval(processed_query, k)
            
            # Track retrieval
            self._track_retrieval(retrieval_result, query, info_bus)
            
            return retrieval_result
            
        except Exception as e:
            self.log_operator_error(f"Memory retrieval failed: {e}")
            return self._create_empty_retrieval_result()

    def _perform_attention_retrieval(self, query: torch.Tensor, k: int) -> Dict[str, Any]:
        """Perform attention-based memory retrieval"""
        
        try:
            # Prepare query and memory for attention
            query_seq = query.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
            memory_seq = self.buffer.unsqueeze(0)        # [1, mem_len, embed_dim]
            
            # Apply attention
            with torch.no_grad():
                # Apply temperature scaling
                temp = self._adaptive_params['attention_temperature']
                
                # Multi-head attention
                attn_out, attn_weights = self.attn(
                    query_seq, memory_seq, memory_seq,
                    need_weights=True
                )
                
                if attn_weights is not None:
                    # Apply temperature
                    attn_weights = torch.softmax(attn_weights.squeeze(0) / temp, dim=-1)
                    
                    # Combine with importance scores for final ranking
                    combined_scores = (0.7 * attn_weights + 
                                     0.3 * self.importance_scores.unsqueeze(0))
                    
                    # Get top-k
                    scores, indices = torch.topk(combined_scores.squeeze(0), k)
                    
                    retrieved_embeddings = self.buffer[indices].cpu().numpy()
                    retrieved_metadata = [self.memory_metadata[i] for i in indices]
                    retrieved_scores = scores.cpu().numpy()
                    
                    # Store attention pattern for analysis
                    self._attention_patterns.append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'attention_weights': attn_weights.cpu().numpy(),
                        'top_indices': indices.cpu().numpy(),
                        'scores': scores.cpu().numpy()
                    })
                    
                else:
                    # Fallback to importance-based retrieval
                    scores, indices = torch.topk(self.importance_scores, k)
                    retrieved_embeddings = self.buffer[indices].cpu().numpy()
                    retrieved_metadata = [self.memory_metadata[i] for i in indices]
                    retrieved_scores = scores.cpu().numpy()
            
            self.log_operator_info(
                f"Memory retrieval completed",
                retrieved_count=k,
                top_score=f"{retrieved_scores[0]:.3f}",
                method="attention-based"
            )
            
            return {
                "embeddings": retrieved_embeddings,
                "metadata": retrieved_metadata,
                "scores": retrieved_scores,
                "retrieval_method": "attention",
                "query_processed": True
            }
            
        except Exception as e:
            self.log_operator_error(f"Attention retrieval failed: {e}")
            # Fallback to simple retrieval
            return self._fallback_retrieval(k)

    def _fallback_retrieval(self, k: int) -> Dict[str, Any]:
        """Fallback retrieval method"""
        
        # Simple importance-based retrieval
        if len(self.importance_scores) > 0:
            scores, indices = torch.topk(self.importance_scores, k)
            retrieved_embeddings = self.buffer[indices].cpu().numpy()
            retrieved_metadata = [self.memory_metadata[i] for i in indices]
            retrieved_scores = scores.cpu().numpy()
        else:
            # Return most recent
            retrieved_embeddings = self.buffer[-k:].cpu().numpy()
            retrieved_metadata = self.memory_metadata[-k:]
            retrieved_scores = np.ones(k) * 0.5
        
        self.log_operator_warning("Using fallback retrieval method")
        
        return {
            "embeddings": retrieved_embeddings,
            "metadata": retrieved_metadata,
            "scores": retrieved_scores,
            "retrieval_method": "fallback",
            "query_processed": False
        }

    def _create_empty_retrieval_result(self) -> Dict[str, Any]:
        """Create empty retrieval result"""
        
        return {
            "embeddings": np.zeros((1, self.embed_dim), dtype=np.float32),
            "metadata": [{'empty': True}],
            "scores": np.zeros(1, dtype=np.float32),
            "retrieval_method": "empty",
            "query_processed": False
        }

    def _track_retrieval(self, retrieval_result: Dict[str, Any], query: np.ndarray, 
                        info_bus: Optional[InfoBus]):
        """Track retrieval performance"""
        
        retrieval_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'step': self._step_count,
            'method': retrieval_result.get('retrieval_method', 'unknown'),
            'retrieved_count': len(retrieval_result.get('embeddings', [])),
            'top_score': float(retrieval_result.get('scores', [0])[0]),
            'query_norm': float(np.linalg.norm(query)),
            'has_context': info_bus is not None
        }
        
        self._retrieval_history.append(retrieval_record)
        
        # Update retrieval accuracy if we have reward feedback
        if len(self._retrieval_history) >= 2:
            # Simple heuristic: higher scores should correlate with better outcomes
            recent_retrievals = list(self._retrieval_history)[-10:]
            scores = [r['top_score'] for r in recent_retrievals]
            if len(scores) > 1:
                score_consistency = 1.0 - np.std(scores)
                self._retrieval_accuracy = max(0.0, score_consistency)
                
        # Update learning curves
        self._learning_curves['attention_focus'].append(retrieval_record['top_score'])
        
        # Update performance metrics
        self._update_performance_metric('retrieval_accuracy', self._retrieval_accuracy)

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with neural metrics"""
        
        try:
            # Memory statistics
            memory_usage = len(self.buffer) / self.max_len
            avg_importance = self.importance_scores.mean().item() if len(self.importance_scores) > 0 else 0.0
            
            # Recent performance
            recent_rewards = []
            for metadata in self.memory_metadata[-10:]:
                if isinstance(metadata, dict) and 'reward' in metadata:
                    recent_rewards.append(metadata['reward'])
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            
            # Neural network performance
            neural_performance = self._neural_performance_score / 100.0
            
            # Attention effectiveness
            attention_effectiveness = 0.0
            if self._attention_patterns:
                recent_pattern = self._attention_patterns[-1]
                attention_weights = recent_pattern.get('attention_weights', np.array([]))
                if len(attention_weights) > 0:
                    # Measure attention focus (entropy-based)
                    attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8))
                    max_entropy = np.log(len(attention_weights))
                    attention_effectiveness = 1.0 - (attention_entropy / max_entropy) if max_entropy > 0 else 0.0
            
            # Learning progress
            learning_progress = 0.0
            if len(self._importance_evolution) >= 10:
                recent_importance = [entry['importance'] for entry in list(self._importance_evolution)[-10:]]
                importance_trend = np.polyfit(range(len(recent_importance)), recent_importance, 1)[0]
                learning_progress = np.tanh(importance_trend * 10)  # Normalize to [-1, 1]
            
            # Adaptive parameters status
            importance_scaling = self._adaptive_params['importance_scaling']
            decay_adjustment = self._adaptive_params['decay_adjustment']
            
            # Memory quality indicators
            memory_turnover = self._memory_turnover_rate
            storage_efficiency = self._storage_efficiency
            
            # Combine all components
            observation = np.array([
                memory_usage,
                avg_importance,
                avg_reward,
                neural_performance,
                attention_effectiveness,
                learning_progress,
                importance_scaling,
                decay_adjustment,
                memory_turnover,
                storage_efficiency,
                self._retrieval_accuracy,
                float(len(self.buffer))
            ], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(12, dtype=np.float32)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Propose actions based on memory insights"""
        
        # Determine action dimension
        action_dim = 2
        if hasattr(obs, 'shape') and len(obs.shape) > 0:
            action_dim = obs.shape[0]
        
        # Base action on memory-derived insights
        if len(self.buffer) > 0 and info_bus:
            # Retrieve relevant memories for current context
            current_obs = self._extract_observation_from_info_bus(info_bus)
            retrieval_result = self.retrieve(current_obs, top_k=3, info_bus=info_bus)
            
            # Extract actions from retrieved memories
            retrieved_metadata = retrieval_result.get('metadata', [])
            retrieved_scores = retrieval_result.get('scores', [])
            
            if retrieved_metadata and len(retrieved_scores) > 0:
                # Weight actions by retrieval scores and rewards
                weighted_actions = []
                total_weight = 0.0
                
                for metadata, score in zip(retrieved_metadata, retrieved_scores):
                    if isinstance(metadata, dict) and 'action' in metadata:
                        action = np.asarray(metadata['action'], dtype=np.float32)
                        reward = metadata.get('reward', 0.0)
                        
                        # Weight by both retrieval score and historical reward
                        weight = score * max(0.1, 1.0 + reward)  # Ensure positive weight
                        
                        if len(action) >= action_dim:
                            weighted_actions.append(action[:action_dim] * weight)
                            total_weight += weight
                
                if weighted_actions and total_weight > 0:
                    # Average weighted actions
                    combined_action = np.sum(weighted_actions, axis=0) / total_weight
                    
                    # Scale based on confidence
                    confidence_scale = min(1.0, self._retrieval_accuracy + 0.3)
                    action = combined_action * confidence_scale
                    
                    return action.astype(np.float32)
        
        # Fallback to neutral action
        return np.zeros(action_dim, dtype=np.float32)

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return confidence in memory-based recommendations"""
        
        base_confidence = 0.5
        
        # Confidence from memory quality
        if len(self.buffer) > 0:
            memory_quality = self.importance_scores.mean().item()
            base_confidence += memory_quality * 0.3
        
        # Confidence from neural performance
        neural_confidence = self._neural_performance_score / 100.0 * 0.2
        base_confidence += neural_confidence
        
        # Confidence from retrieval accuracy
        base_confidence += self._retrieval_accuracy * 0.2
        
        # Confidence from data volume
        data_confidence = min(0.2, len(self.buffer) / self.max_len * 0.2)
        base_confidence += data_confidence
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def prune_memories(self, threshold: Optional[float] = None):
        """Enhanced memory pruning with adaptive thresholds"""
        
        if len(self.buffer) == 0:
            return
        
        prune_threshold = threshold if threshold is not None else self.importance_threshold
        prune_threshold *= self._adaptive_params['quality_threshold']
        
        # Find memories to keep
        mask = self.importance_scores > prune_threshold
        memories_before = len(self.buffer)
        
        if mask.any():
            self.buffer = self.buffer[mask]
            self.importance_scores = self.importance_scores[mask]
            self.memory_metadata = [m for i, m in enumerate(self.memory_metadata) if mask[i]]
        
        memories_after = len(self.buffer)
        pruned_count = memories_before - memories_after
        
        if pruned_count > 0:
            self.log_operator_info(
                f"Memory pruning completed",
                pruned=pruned_count,
                remaining=memories_after,
                threshold=f"{prune_threshold:.3f}"
            )

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with network rebuilding if needed"""
        old_embed_dim = self.embed_dim
        old_num_heads = self.num_heads
        
        self.embed_dim = int(np.clip(genome.get("embed_dim", self.embed_dim), 16, 128))
        self.num_heads = int(np.clip(genome.get("num_heads", self.num_heads), 2, 16))
        self.max_len = int(np.clip(genome.get("max_len", self.max_len), 100, 1000))
        self.memory_decay = float(np.clip(genome.get("memory_decay", self.memory_decay), 0.8, 0.99))
        self.importance_threshold = float(np.clip(genome.get("importance_threshold", self.importance_threshold), 0.1, 0.8))
        self.retrieval_top_k = int(np.clip(genome.get("retrieval_top_k", self.retrieval_top_k), 1, 20))
        self.learning_rate = float(np.clip(genome.get("learning_rate", self.learning_rate), 0.0001, 0.01))
        self.attention_dropout = float(np.clip(genome.get("attention_dropout", self.attention_dropout), 0.0, 0.5))
        
        self.genome = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "max_len": self.max_len,
            "memory_decay": self.memory_decay,
            "importance_threshold": self.importance_threshold,
            "retrieval_top_k": self.retrieval_top_k,
            "learning_rate": self.learning_rate,
            "attention_dropout": self.attention_dropout
        }
        
        # Rebuild networks if architecture changed
        if old_embed_dim != self.embed_dim or old_num_heads != self.num_heads:
            try:
                # Save memory content if possible
                old_metadata = self.memory_metadata.copy() if hasattr(self, 'memory_metadata') else []
                
                # Rebuild neural components
                self._initialize_neural_components()
                
                # Reset memory buffers
                self.buffer = torch.zeros((0, self.embed_dim), dtype=torch.float32)
                self.importance_scores = torch.zeros(0, dtype=torch.float32)
                self.memory_metadata = old_metadata[:0]  # Clear but keep structure
                
                self.log_operator_info(f"Neural networks rebuilt: embed_dim={self.embed_dim}, heads={self.num_heads}")
                
            except Exception as e:
                self.log_operator_error(f"Network rebuild failed: {e}")
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with neural network weight mutation"""
        g = self.genome.copy()
        mutations = []
        
        # Architectural mutations
        if np.random.rand() < mutation_rate:
            old_val = g["embed_dim"]
            # Prefer powers of 2 for efficiency
            options = [16, 32, 64, 128]
            g["embed_dim"] = random.choice(options)
            mutations.append(f"embed_dim: {old_val} → {g['embed_dim']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["num_heads"]
            # Ensure heads divide embed_dim
            possible_heads = [h for h in [2, 4, 8, 16] if g["embed_dim"] % h == 0]
            if possible_heads:
                g["num_heads"] = random.choice(possible_heads)
                mutations.append(f"num_heads: {old_val} → {g['num_heads']}")
                
        if np.random.rand() < mutation_rate:
            old_val = g["max_len"]
            g["max_len"] = int(np.clip(old_val + np.random.randint(-50, 51), 100, 1000))
            mutations.append(f"max_len: {old_val} → {g['max_len']}")
            
        # Parameter mutations
        if np.random.rand() < mutation_rate:
            old_val = g["memory_decay"]
            g["memory_decay"] = float(np.clip(old_val + np.random.uniform(-0.02, 0.02), 0.8, 0.99))
            mutations.append(f"decay: {old_val:.3f} → {g['memory_decay']:.3f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["importance_threshold"]
            g["importance_threshold"] = float(np.clip(old_val + np.random.uniform(-0.1, 0.1), 0.1, 0.8))
            mutations.append(f"threshold: {old_val:.2f} → {g['importance_threshold']:.2f}")
        
        if mutations:
            self.log_operator_info(f"Neural memory mutation applied", changes=", ".join(mutations))
            
        # Neural weight mutation
        if np.random.rand() < mutation_rate * 0.5:
            noise_std = 0.05
            with torch.no_grad():
                for module in [self.encoder, self.value_head, self.context_net]:
                    for param in module.parameters():
                        if param.requires_grad:
                            param.data += noise_std * torch.randn_like(param.data)
            
            self.log_operator_info(f"Neural weights mutated with std={noise_std}")
        
        self.set_genome(g)
        
    def crossover(self, other: "NeuralMemoryArchitect") -> "NeuralMemoryArchitect":
        """Enhanced crossover with neural weight mixing"""
        if not isinstance(other, NeuralMemoryArchitect):
            self.log_operator_warning("Crossover with incompatible type")
            return self
        
        # Performance-based crossover
        self_performance = self._neural_performance_score
        other_performance = other._neural_performance_score
        
        # Favor higher performance parent
        if self_performance > other_performance:
            bias = 0.7  # Favor self
        else:
            bias = 0.3  # Favor other
        
        new_g = {k: (self.genome[k] if np.random.rand() < bias else other.genome[k]) for k in self.genome}
        
        child = NeuralMemoryArchitect(genome=new_g, debug=self.config.debug)
        
        # Neural weight crossover (if architectures match)
        if (self.embed_dim == other.embed_dim and 
            self.num_heads == other.num_heads and 
            self.embed_dim == child.embed_dim and 
            self.num_heads == child.num_heads):
            
            try:
                with torch.no_grad():
                    for module_name in ['encoder', 'value_head', 'context_net']:
                        child_module = getattr(child, module_name)
                        self_module = getattr(self, module_name)
                        other_module = getattr(other, module_name)
                        
                        for (p_child, p_self, p_other) in zip(
                            child_module.parameters(),
                            self_module.parameters(),
                            other_module.parameters()
                        ):
                            if p_child.shape == p_self.shape == p_other.shape:
                                mask = torch.rand_like(p_self) > 0.5
                                p_child.data = torch.where(mask, p_self.data, p_other.data)
                
                self.log_operator_info("Neural weight crossover completed")
                
            except Exception as e:
                self.log_operator_warning(f"Neural weight crossover failed: {e}")
        
        # Inherit memory from better parent
        if self_performance > other_performance:
            if len(self.buffer) > 0:
                child.buffer = self.buffer.clone()
                child.importance_scores = self.importance_scores.clone()
                child.memory_metadata = self.memory_metadata.copy()
        else:
            if len(other.buffer) > 0:
                child.buffer = other.buffer.clone()
                child.importance_scores = other.importance_scores.clone()
                child.memory_metadata = other.memory_metadata.copy()
        
        return child

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check tensor dimensions
            if self.buffer.size(0) != len(self.importance_scores):
                return False
            if self.buffer.size(1) != self.embed_dim:
                return False
            if len(self.memory_metadata) != len(self.buffer):
                return False
                
            # Check value ranges
            if not torch.all(torch.isfinite(self.buffer)):
                return False
            if not torch.all(torch.isfinite(self.importance_scores)):
                return False
            if not torch.all((self.importance_scores >= 0) & (self.importance_scores <= 1)):
                return False
                
            # Check neural network state
            for module in [self.encoder, self.attn, self.value_head, self.context_net]:
                for param in module.parameters():
                    if not torch.all(torch.isfinite(param.data)):
                        return False
                        
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        neural_details = {
            'memory_info': {
                'buffer_size': len(self.buffer),
                'max_capacity': self.max_len,
                'utilization': len(self.buffer) / self.max_len,
                'avg_importance': self.importance_scores.mean().item() if len(self.importance_scores) > 0 else 0.0
            },
            'neural_info': {
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'performance_score': self._neural_performance_score,
                'storage_efficiency': self._storage_efficiency,
                'retrieval_accuracy': self._retrieval_accuracy
            },
            'adaptive_info': {
                'importance_scaling': self._adaptive_params['importance_scaling'],
                'attention_temperature': self._adaptive_params['attention_temperature'],
                'decay_adjustment': self._adaptive_params['decay_adjustment'],
                'quality_threshold': self._adaptive_params['quality_threshold']
            },
            'learning_info': {
                'importance_evolution_size': len(self._importance_evolution),
                'retrieval_history_size': len(self._retrieval_history),
                'attention_patterns_size': len(self._attention_patterns),
                'memory_usage_history_size': len(self._memory_usage_history)
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(neural_details)
            return base_details
        
        return neural_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        
        # Convert tensors and save neural state
        neural_state = {
            'encoder': self.encoder.state_dict(),
            'attn': self.attn.state_dict(),
            'value_head': self.value_head.state_dict(),
            'context_net': self.context_net.state_dict()
        }
        
        return {
            "neural_state": neural_state,
            "buffer": self.buffer.cpu().numpy().tolist(),
            "importance_scores": self.importance_scores.cpu().numpy().tolist(),
            "memory_metadata": self.memory_metadata[-100:],  # Keep recent only
            "genome": self.genome.copy(),
            "adaptive_params": self._adaptive_params.copy(),
            "storage_efficiency": self._storage_efficiency,
            "retrieval_accuracy": self._retrieval_accuracy,
            "neural_performance_score": self._neural_performance_score,
            "importance_evolution": list(self._importance_evolution)[-50:],  # Keep recent
            "retrieval_history": list(self._retrieval_history)[-30:],
            "memory_usage_history": list(self._memory_usage_history)[-50:],
            "learning_curves": {k: list(v)[-30:] for k, v in self._learning_curves.items()}
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        # Restore tensors
        buffer_data = module_state.get("buffer", [])
        if buffer_data:
            self.buffer = torch.tensor(buffer_data, dtype=torch.float32)
        else:
            self.buffer = torch.zeros((0, self.embed_dim), dtype=torch.float32)
            
        importance_data = module_state.get("importance_scores", [])
        if importance_data:
            self.importance_scores = torch.tensor(importance_data, dtype=torch.float32)
        else:
            self.importance_scores = torch.zeros(0, dtype=torch.float32)
        
        # Restore metadata
        self.memory_metadata = module_state.get("memory_metadata", [])
        
        # Restore neural networks
        neural_state = module_state.get("neural_state", {})
        try:
            if "encoder" in neural_state:
                self.encoder.load_state_dict(neural_state["encoder"])
            if "attn" in neural_state:
                self.attn.load_state_dict(neural_state["attn"])
            if "value_head" in neural_state:
                self.value_head.load_state_dict(neural_state["value_head"])
            if "context_net" in neural_state:
                self.context_net.load_state_dict(neural_state["context_net"])
        except Exception as e:
            self.log_operator_warning(f"Neural state restoration failed: {e}")
        
        # Restore other state
        self.set_genome(module_state.get("genome", self.genome))
        self._adaptive_params = module_state.get("adaptive_params", self._adaptive_params)
        self._storage_efficiency = module_state.get("storage_efficiency", 0.0)
        self._retrieval_accuracy = module_state.get("retrieval_accuracy", 0.0)
        self._neural_performance_score = module_state.get("neural_performance_score", 100.0)
        
        # Restore tracking data
        self._importance_evolution = deque(module_state.get("importance_evolution", []), maxlen=500)
        self._retrieval_history = deque(module_state.get("retrieval_history", []), maxlen=100)
        self._memory_usage_history = deque(module_state.get("memory_usage_history", []), maxlen=200)
        
        # Restore learning curves
        learning_curves_data = module_state.get("learning_curves", {})
        for curve_name, curve_data in learning_curves_data.items():
            if curve_name in self._learning_curves:
                self._learning_curves[curve_name] = deque(curve_data, maxlen=100)

    def get_neural_memory_report(self) -> str:
        """Generate operator-friendly neural memory report"""
        
        # Memory statistics
        memory_util = len(self.buffer) / self.max_len
        avg_importance = self.importance_scores.mean().item() if len(self.importance_scores) > 0 else 0.0
        
        # Performance status
        if self._neural_performance_score > 80:
            performance_status = "🚀 Excellent"
        elif self._neural_performance_score > 60:
            performance_status = "✅ Good"
        elif self._neural_performance_score > 40:
            performance_status = "⚡ Fair"
        else:
            performance_status = "⚠️ Poor"
        
        # Learning trend
        learning_trend = "📊 Stable"
        if len(self._importance_evolution) >= 10:
            recent_importance = [entry['importance'] for entry in list(self._importance_evolution)[-10:]]
            if len(recent_importance) >= 2:
                trend = np.polyfit(range(len(recent_importance)), recent_importance, 1)[0]
                if trend > 0.01:
                    learning_trend = "📈 Improving"
                elif trend < -0.01:
                    learning_trend = "📉 Declining"
        
        return f"""
🧠 NEURAL MEMORY ARCHITECT
═══════════════════════════════════════
💾 Memory: {len(self.buffer):,}/{self.max_len:,} ({memory_util:.1%})
🎯 Performance: {performance_status} ({self._neural_performance_score:.1f})
📈 Learning: {learning_trend}
⚡ Avg Importance: {avg_importance:.3f}

🏗️ ARCHITECTURE
• Embedding Dim: {self.embed_dim}
• Attention Heads: {self.num_heads}
• Memory Decay: {self.memory_decay:.3f}
• Dropout: {self.attention_dropout:.2f}

📊 PERFORMANCE METRICS
• Storage Efficiency: {self._storage_efficiency:.3f}
• Retrieval Accuracy: {self._retrieval_accuracy:.3f}
• Memory Turnover: {self._memory_turnover_rate:.3f}

🔧 ADAPTIVE PARAMETERS
• Importance Scaling: {self._adaptive_params['importance_scaling']:.2f}
• Attention Temperature: {self._adaptive_params['attention_temperature']:.2f}
• Decay Adjustment: {self._adaptive_params['decay_adjustment']:.2f}
• Quality Threshold: {self._adaptive_params['quality_threshold']:.2f}

📈 LEARNING ANALYTICS
• Importance Evolution: {len(self._importance_evolution)} records
• Retrieval History: {len(self._retrieval_history)} queries
• Attention Patterns: {len(self._attention_patterns)} analyses
• Usage History: {len(self._memory_usage_history)} snapshots

💡 RECENT ACTIVITY
• Storage Events: {len([e for e in self._importance_evolution if e['importance'] > 0.5])}
• High-Value Retrievals: {len([r for r in self._retrieval_history if r['top_score'] > 0.7])}
• Attention Focus: {'High' if len(self._attention_patterns) > 0 and np.mean([np.max(p['attention_weights']) for p in self._attention_patterns[-3:] if 'attention_weights' in p]) > 0.7 else 'Low'}
        """

    # Maintain backward compatibility
    def step(self, experience: Any = None, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, experience=experience, **kwargs)

    def get_state(self) -> Dict[str, Any]:
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state: Dict[str, Any]):
        """Backward compatibility state method"""
        super().set_state(state)