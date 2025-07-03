
# ─────────────────────────────────────────────────────────────
# File: modules/features/multiscale_feature_engine.py  
# Enhanced with new infrastructure - keeping PyTorch functionality
# ─────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor
from modules.features.advanced_feature_engine import AdvancedFeatureEngine



class MultiScaleFeatureEngine(Module, AnalysisMixin):
    """
    Enhanced multiscale feature engine with infrastructure integration.
    Class name unchanged - just enhanced capabilities!
    """

    def __init__(self,
                 afe: AdvancedFeatureEngine,
                 embed_dim: int = 32,
                 debug: bool = False,
                 **kwargs):
        # ─── Module-specific configuration ─────────────────────────────
        self.afe = afe
        self.embed_dim = embed_dim
        self.in_dim = afe.out_dim   # must exist before base-class init
        self.out_dim = embed_dim    # must exist before base-class init

        # ─── Initialize the Module base (this will call _initialize_module_state) ─
        config = ModuleConfig(
            debug=debug,
            max_history=100,
            **kwargs
        )
        super().__init__(config)

        # ─── Neural network setup ────────────────────────────────────────
        self._initialize_neural_networks()

        self.log_operator_info(
            "Multiscale feature engine initialized",
            input_dim=self.in_dim,
            output_dim=self.out_dim,
            device=str(self.device),
            afe_windows=self.afe.windows
        )

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        
        # Neural network state
        self.last_embedding = np.zeros(self.out_dim, dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enhanced tracking
        self._embedding_history = []
        self._neural_performance = {'forward_passes': 0, 'errors': 0}
        self._attention_weights_history = []
        self._model_health_score = 100.0

    def _initialize_neural_networks(self):
        """Initialize PyTorch neural network components"""
        
        try:
            # Neural layers with enhanced architecture
            self.proj = nn.Sequential(
                nn.Linear(self.in_dim, self.embed_dim),
                nn.ReLU(),
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(0.1),
            )
            
            # Attention mechanism
            self.to_q = nn.Linear(self.embed_dim, self.embed_dim)
            self.to_k = nn.Linear(self.embed_dim, self.embed_dim)
            self.to_v = nn.Linear(self.embed_dim, self.embed_dim)
            
            # Output layers
            self.out = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
            
            # Initialize weights and move to device
            self._init_weights()
            self._move_to_device()
            
            self.log_operator_info(f"Neural networks initialized on {self.device}")
            
        except Exception as e:
            self.log_operator_error(f"Neural network initialization failed: {e}")
            self._update_health_status("ERROR", f"NN init failed: {e}")
            raise

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Module-specific reset
        self.last_embedding.fill(0.0)
        self.afe.reset()
        self._embedding_history.clear()
        self._neural_performance = {'forward_passes': 0, 'errors': 0}
        self._attention_weights_history.clear()
        self._model_health_score = 100.0

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and multi-timeframe extraction"""
        
        # Extract multi-timeframe data
        price_data = self._extract_multiscale_prices(info_bus, kwargs)
        
        # Process with enhanced error handling
        self._process_multiscale_features(price_data)

    def _extract_multiscale_prices(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract prices for multiple timeframes from InfoBus or kwargs"""
        
        price_data = {'h1': None, 'h4': None, 'd1': None}
        
        # Try InfoBus first
        if info_bus:
            # Extract from features if structured by timeframe
            features = info_bus.get('features', {})
            for tf in ['h1', 'h4', 'd1']:
                tf_key = f'price_{tf}'
                if tf_key in features:
                    price_data[tf] = self._validate_price_series(features[tf_key])
            
            # Fallback to current prices for all timeframes
            if all(v is None for v in price_data.values()):
                current_prices = info_bus.get('prices', {})
                if current_prices:
                    price_array = np.array(list(current_prices.values()))
                    # Use same prices for all timeframes if specific TF data unavailable
                    price_data['h1'] = price_array
                    price_data['h4'] = price_array
                    price_data['d1'] = price_array
        
        # Try kwargs (backward compatibility)
        for tf in ['h1', 'h4', 'd1']:
            if f'price_{tf}' in kwargs:
                price_data[tf] = self._validate_price_series(kwargs[f'price_{tf}'])
        
        return price_data

    def _validate_price_series(self, data) -> Optional[np.ndarray]:
        """Validate and clean price series data"""
        
        if data is None:
            return None
        
        try:
            arr = np.asarray(data, dtype=np.float32)
            
            # Filter valid prices
            valid_mask = np.isfinite(arr) & (arr > 0)
            valid_prices = arr[valid_mask]
            
            return valid_prices if valid_prices.size > 0 else None
            
        except Exception as e:
            self.log_operator_warning(f"Price series validation failed: {e}")
            return None

    def _process_multiscale_features(self, price_data: Dict[str, np.ndarray]):
        """Process multiscale features with enhanced neural network pipeline"""
        
        try:
            # Prepare price series with fallbacks
            h1, h4, d1 = self._prepare_price_series_with_fallbacks(price_data)
            
            # Extract features using AFE
            features = self._extract_multiscale_features(h1, h4, d1)
            
            # Neural network processing
            embedding = self._neural_network_forward(features)
            
            # Update state and metrics
            self._update_embedding_state(embedding, features)
            
        except Exception as e:
            self.log_operator_error(f"Multiscale processing failed: {e}")
            self._handle_processing_error(e)

    def _prepare_price_series_with_fallbacks(self, price_data: Dict[str, np.ndarray]) -> tuple:
        """Prepare price series with intelligent fallbacks"""
        
        h1 = price_data.get('h1')
        h4 = price_data.get('h4')
        d1 = price_data.get('d1')
        
        # Apply fallback strategy
        if all(series is None or series.size == 0 for series in [h1, h4, d1]):
            # Use AFE buffer if available
            if len(self.afe.price_buffer) > 0:
                fallback_prices = np.array(list(self.afe.price_buffer)[-28:], dtype=np.float32)
                self.log_operator_info("Using AFE buffer for all timeframes")
                h1 = h4 = d1 = fallback_prices
            else:
                # Generate synthetic data
                synthetic_prices = self.afe._generate_synthetic_prices_enhanced(30)
                self.log_operator_warning("Using synthetic prices for all timeframes")
                h1 = h4 = d1 = synthetic_prices
        else:
            # Fill missing timeframes
            available_series = next((series for series in [d1, h4, h1] if series is not None and series.size > 0), None)
            
            if d1 is None or d1.size == 0:
                d1 = available_series
            if h4 is None or h4.size == 0:
                h4 = available_series if available_series is not None else d1
            if h1 is None or h1.size == 0:
                h1 = available_series if available_series is not None else h4
        
        return h1, h4, d1

    def _extract_multiscale_features(self, h1: np.ndarray, h4: np.ndarray, d1: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features for each timeframe using AFE"""
        
        features = {}
        
        for name, prices in [('h1', h1), ('h4', h4), ('d1', d1)]:
            try:
                feature_vector = self.afe.transform(prices)
                
                # Validate feature dimensions
                if feature_vector.size != self.in_dim:
                    self.log_operator_error(f"AFE output dimension mismatch: {feature_vector.size} != {self.in_dim}")
                    feature_vector = np.zeros(self.in_dim, dtype=np.float32)
                
                features[name] = feature_vector
                
            except Exception as e:
                self.log_operator_error(f"Feature extraction failed for {name}: {e}")
                features[name] = np.zeros(self.in_dim, dtype=np.float32)
        
        return features

    def _neural_network_forward(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Enhanced neural network forward pass with monitoring"""
        
        try:
            self._neural_performance['forward_passes'] += 1
            
            with torch.no_grad():
                # Convert to tensors
                f1_t = torch.from_numpy(features['h1']).to(self.device)
                f4_t = torch.from_numpy(features['h4']).to(self.device)
                fD_t = torch.from_numpy(features['d1']).to(self.device)
                
                # Project to embedding space
                x1, x4, xD = self.proj(f1_t), self.proj(f4_t), self.proj(fD_t)
                X = torch.stack((x1, x4, xD), dim=0).unsqueeze(0)  # (1, 3, embed_dim)
                
                # Multi-head attention
                Q, K, V = self.to_q(X), self.to_k(X), self.to_v(X)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim)
                attention_weights = torch.softmax(scores, dim=-1)
                attended = torch.matmul(attention_weights, V)
                
                # Store attention weights for analysis
                self._store_attention_weights(attention_weights)
                
                # Output processing
                pooled = attended.mean(dim=1).squeeze(0)
                output = self.out(pooled) + pooled  # Residual connection
                
                # Convert back to numpy
                embedding = output.cpu().numpy().astype(np.float32)
                embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
                
                return embedding
                
        except Exception as e:
            self._neural_performance['errors'] += 1
            self.log_operator_error(f"Neural network forward pass failed: {e}")
            raise

    def _store_attention_weights(self, attention_weights: torch.Tensor):
        """Store attention weights for analysis"""
        try:
            weights = attention_weights.cpu().numpy()
            self._attention_weights_history.append({
                'timestamp': np.datetime64('now').astype(str),
                'weights': weights.copy()
            })
            
            # Keep history manageable
            if len(self._attention_weights_history) > 50:
                self._attention_weights_history = self._attention_weights_history[-25:]
                
        except Exception:
            pass  # Don't fail if attention storage fails

    def _update_embedding_state(self, embedding: np.ndarray, features: Dict[str, np.ndarray]):
        """Update embedding state and performance metrics"""
        
        self.last_embedding = embedding
        
        # Store in history
        self._embedding_history.append({
            'timestamp': np.datetime64('now').astype(str),
            'embedding': embedding.copy(),
            'embedding_norm': float(np.linalg.norm(embedding)),
            'feature_norms': {k: float(np.linalg.norm(v)) for k, v in features.items()}
        })
        
        # Keep history manageable
        if len(self._embedding_history) > self.config.max_history:
            self._embedding_history = self._embedding_history[-self.config.max_history//2:]
        
        # Update performance metrics
        self._update_performance_metric('embedding_norm', np.linalg.norm(embedding))
        self._update_performance_metric('neural_success_rate', 
            1.0 - (self._neural_performance['errors'] / max(self._neural_performance['forward_passes'], 1)))

    def _handle_processing_error(self, error: Exception):
        """Handle processing errors gracefully"""
        
        self._neural_performance['errors'] += 1
        
        # Use fallback embedding
        if len(self._embedding_history) > 0:
            self.last_embedding = self._embedding_history[-1]['embedding'].copy()
            self.log_operator_warning("Using last known good embedding due to processing error")
        else:
            self.last_embedding = np.zeros(self.out_dim, dtype=np.float32)
            self.log_operator_warning("Using zero embedding due to processing error")
        
        # Update health status
        error_rate = self._neural_performance['errors'] / max(self._neural_performance['forward_passes'], 1)
        if error_rate > 0.1:
            self._update_health_status("DEGRADED", f"High error rate: {error_rate:.1%}")

    def step(self, price_h1: Optional[np.ndarray] = None, price_h4: Optional[np.ndarray] = None, 
             price_d1: Optional[np.ndarray] = None, **kwargs):
        """Backward compatibility wrapper"""
        kwargs.update({
            'price_h1': price_h1,
            'price_h4': price_h4, 
            'price_d1': price_d1
        })
        self._step_impl(None, **kwargs)

    def _init_weights(self):
        """Enhanced weight initialization"""
        for module in [self.proj, self.to_q, self.to_k, self.to_v, self.out]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _move_to_device(self):
        """Move all neural network components to device"""
        try:
            for module in [self.proj, self.to_q, self.to_k, self.to_v, self.out]:
                module.to(self.device)
            self.log_operator_info(f"Neural networks moved to {self.device}")
        except Exception as e:
            self.log_operator_error(f"Failed to move networks to device: {e}")
            # Fallback to CPU
            self.device = torch.device("cpu")
            for module in [self.proj, self.to_q, self.to_k, self.to_v, self.out]:
                module.to(self.device)

    def _get_observation_impl(self) -> np.ndarray:
        """Enhanced observation with validation"""
        return np.nan_to_num(self.last_embedding, nan=0.0, posinf=1.0, neginf=-1.0).copy()

    def _check_state_integrity(self) -> bool:
        """Enhanced health check for neural network components"""
        try:
            # Check embedding dimensions
            if self.last_embedding.size != self.out_dim:
                return False
            
            # Check neural network health
            error_rate = self._neural_performance['errors'] / max(self._neural_performance['forward_passes'], 1)
            if error_rate > 0.3:
                return False
            
            # Check AFE health
            if hasattr(self.afe, '_health_status') and self.afe._health_status != "OK":
                return False
            
            # Check device availability
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                return False
            
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details with neural network metrics"""
        base_details = super()._get_health_details()
        
        nn_details = {
            'neural_network': {
                'device': str(self.device),
                'forward_passes': self._neural_performance['forward_passes'],
                'errors': self._neural_performance['errors'],
                'success_rate': f"{(1.0 - self._neural_performance['errors'] / max(self._neural_performance['forward_passes'], 1)):.1%}",
                'embedding_norm': float(np.linalg.norm(self.last_embedding))
            },
            'architecture': {
                'input_dim': self.in_dim,
                'output_dim': self.out_dim,
                'embed_dim': self.embed_dim
            },
            'attention_analysis': {
                'weight_history_size': len(self._attention_weights_history),
                'embedding_history_size': len(self._embedding_history)
            },
            'afe_health': self.afe.get_health_status() if hasattr(self.afe, 'get_health_status') else 'N/A'
        }
        
        if base_details:
            base_details.update(nn_details)
            return base_details
        
        return nn_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management with neural network state"""
        return {
            'last_embedding': self.last_embedding.tolist(),
            'neural_performance': self._neural_performance.copy(),
            'embedding_history': self._embedding_history[-10:],  # Keep recent only
            'attention_weights_history': self._attention_weights_history[-5:],  # Keep recent only
            'afe_state': self.afe.get_state() if hasattr(self.afe, 'get_state') else None,
            'device': str(self.device)
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        self.last_embedding = np.array(module_state.get('last_embedding', self.last_embedding), dtype=np.float32)
        self._neural_performance = module_state.get('neural_performance', {'forward_passes': 0, 'errors': 0})
        self._embedding_history = module_state.get('embedding_history', [])
        self._attention_weights_history = module_state.get('attention_weights_history', [])
        
        # Restore AFE state
        afe_state = module_state.get('afe_state')
        if afe_state and hasattr(self.afe, 'set_state'):
            self.afe.set_state(afe_state)

    def get_multiscale_analysis_report(self) -> str:
        """Generate operator-friendly multiscale analysis report"""
        
        success_rate = 1.0 - (self._neural_performance['errors'] / max(self._neural_performance['forward_passes'], 1))
        embedding_norm = np.linalg.norm(self.last_embedding)
        
        # Analyze attention patterns if available
        attention_analysis = "No data"
        if self._attention_weights_history:
            recent_weights = self._attention_weights_history[-1]['weights']
            if recent_weights.size > 0:
                # Find dominant timeframe
                avg_attention = np.mean(recent_weights, axis=(0, 1))
                timeframes = ['H1', 'H4', 'D1']
                dominant_tf = timeframes[np.argmax(avg_attention)] if len(avg_attention) >= 3 else "Unknown"
                attention_analysis = f"Focus: {dominant_tf} ({np.max(avg_attention):.1%})"
        
        return f"""
🧠 MULTISCALE FEATURE ENGINE
═══════════════════════════════════════
🏗️ Architecture: {self.in_dim}D → {self.embed_dim}D (3 timeframes)
🖥️ Device: {self.device} | Success Rate: {success_rate:.1%}

📊 NEURAL NETWORK STATUS
• Forward Passes: {self._neural_performance['forward_passes']}
• Errors: {self._neural_performance['errors']}
• Current Embedding Norm: {embedding_norm:.4f}
• Attention Pattern: {attention_analysis}

🔧 FEATURE ENGINE STATUS
{self.afe.get_feature_analysis_report() if hasattr(self.afe, 'get_feature_analysis_report') else 'AFE status unavailable'}

📈 EMBEDDING TRENDS
• History Size: {len(self._embedding_history)} snapshots
• Attention History: {len(self._attention_weights_history)} snapshots
• Health Score: {self._get_performance_metric('neural_success_rate', 100):.1f}/100
        """

    # Maintain backward compatibility
    def get_state(self):
        """Backward compatibility state method"""
        base_state = super().get_state()
        return base_state

    def set_state(self, state):
        """Backward compatibility state method"""
        super().set_state(state)