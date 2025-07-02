# ─────────────────────────────────────────────────────────────
# File: modules/models/world_model.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, Optional, List, Tuple
from collections import deque
import datetime
import random

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, extract_standard_context


class EnhancedWorldModel(Module, AnalysisMixin, TradingMixin):
    """
    Enhanced world model with infrastructure integration.
    Uses LSTM networks for market state prediction and scenario simulation.
    """
    
    def __init__(self, input_size: int = 16, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, lr: float = 1e-3, device: str = "cpu",
                 debug: bool = True, genome: Optional[Dict[str, Any]] = None, **kwargs):
        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=300,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome, input_size, hidden_size, num_layers, dropout, lr)
        
        # Enhanced state initialization
        self._initialize_module_state()
        
        # Initialize neural components
        self._initialize_neural_components(device)
        
        self.log_operator_info(
            "Enhanced world model initialized",
            input_features=self.input_size,
            hidden_units=self.hidden_size,
            lstm_layers=self.num_layers,
            dropout_rate=f"{self.dropout:.2f}",
            learning_rate=f"{self.learning_rate:.1e}",
            device=str(self.device)
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict], input_size: int,
                                    hidden_size: int, num_layers: int, dropout: float, lr: float):
        """Initialize genome-based parameters"""
        if genome:
            self.input_size = int(genome.get("input_size", input_size))
            self.hidden_size = int(genome.get("hidden_size", hidden_size))
            self.num_layers = int(genome.get("num_layers", num_layers))
            self.dropout = float(genome.get("dropout", dropout))
            self.learning_rate = float(genome.get("learning_rate", lr))
            self.sequence_length = int(genome.get("sequence_length", 50))
            self.batch_size = int(genome.get("batch_size", 64))
            self.prediction_horizon = int(genome.get("prediction_horizon", 10))
            self.gradient_clip = float(genome.get("gradient_clip", 1.0))
            self.weight_decay = float(genome.get("weight_decay", 1e-5))
        else:
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.learning_rate = lr
            self.sequence_length = 50
            self.batch_size = 64
            self.prediction_horizon = 10
            self.gradient_clip = 1.0
            self.weight_decay = 1e-5

        # Store genome for evolution
        self.genome = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "prediction_horizon": self.prediction_horizon,
            "gradient_clip": self.gradient_clip,
            "weight_decay": self.weight_decay
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # World model state
        self._market_history = deque(maxlen=self.sequence_length * 2)
        self._prediction_history = deque(maxlen=100)
        self._training_history = deque(maxlen=50)
        self._scenario_cache = {}
        
        # Enhanced tracking
        self._model_performance = {
            'prediction_accuracy': 0.0,
            'training_loss': float('inf'),
            'validation_error': float('inf'),
            'scenario_quality': 0.0
        }
        
        # Prediction tracking
        self._recent_predictions = deque(maxlen=20)
        self._prediction_errors = deque(maxlen=100)
        self._feature_importance = {}
        
        # Training analytics
        self._training_curves = {
            'loss': deque(maxlen=100),
            'accuracy': deque(maxlen=100),
            'gradient_norm': deque(maxlen=100)
        }
        
        # Model status
        self._is_trained = False
        self._last_training_time = None
        self._model_confidence = 0.0
        self._stability_score = 1.0

    def _initialize_neural_components(self, device: str):
        """Initialize neural network components"""
        
        try:
            # Set device
            self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
            
            # Main LSTM network
            self.lstm = nn.LSTM(
                self.input_size,
                self.hidden_size,
                self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0.0,
                bidirectional=False
            )
            
            # Prediction heads
            self.price_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, 2)  # Price change for 2 instruments
            )
            
            self.volatility_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, 2)  # Volatility for 2 instruments
            )
            
            self.regime_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, 3),  # 3 regime classes
                nn.Softmax(dim=1)
            )
            
            # Context integration network
            self.context_encoder = nn.Sequential(
                nn.Linear(8, self.hidden_size // 4),  # 8 context features
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, self.hidden_size // 4)
            )
            
            # Attention mechanism for feature importance
            self.attention = nn.MultiheadAttention(
                self.hidden_size, 
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            
            # Move to device
            self.to(self.device)
            
            # Initialize optimizer
            self.optimizer = optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Loss functions
            self.mse_loss = nn.MSELoss()
            self.ce_loss = nn.CrossEntropyLoss()
            
            # Initialize weights
            self._initialize_weights()
            
            self.log_operator_info("Neural components initialized successfully")
            
        except Exception as e:
            self.log_operator_error(f"Neural component initialization failed: {e}")
            self._update_health_status("ERROR", f"Neural init failed: {e}")

    def _initialize_weights(self):
        """Initialize neural network weights"""
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_trading_state()
        
        # Clear world model state
        self._market_history.clear()
        self._prediction_history.clear()
        self._training_history.clear()
        self._scenario_cache.clear()
        
        # Reset tracking
        self._recent_predictions.clear()
        self._prediction_errors.clear()
        self._feature_importance.clear()
        
        # Reset performance metrics
        self._model_performance = {
            'prediction_accuracy': 0.0,
            'training_loss': float('inf'),
            'validation_error': float('inf'),
            'scenario_quality': 0.0
        }
        
        # Reset training curves
        for curve in self._training_curves.values():
            curve.clear()
        
        # Reset model status
        self._is_trained = False
        self._last_training_time = None
        self._model_confidence = 0.0
        self._stability_score = 1.0

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract and process market data
        if info_bus:
            market_features = self._extract_market_features(info_bus)
            self._update_market_history(market_features, info_bus)
            
            # Make predictions if model is trained
            if self._is_trained and len(self._market_history) >= self.sequence_length:
                predictions = self._generate_predictions(info_bus)
                self._track_prediction_performance(predictions, info_bus)
        
        # Process manual training data from kwargs
        if 'training_data' in kwargs:
            self._process_training_data(kwargs['training_data'])
        
        # Update model performance metrics
        self._update_model_performance()

    def _extract_market_features(self, info_bus: InfoBus) -> np.ndarray:
        """Extract comprehensive market features from InfoBus"""
        
        features = []
        
        # Price features
        prices = info_bus.get('prices', {})
        if prices:
            price_values = list(prices.values())[:2]  # First 2 instruments
            for price in price_values:
                features.append(price / 2000.0)  # Normalize price
            
            # Price changes if we have history
            if len(self._market_history) > 0:
                prev_prices = self._market_history[-1][:2]
                price_changes = [(p - pp) / pp if pp > 0 else 0.0 for p, pp in zip(price_values, prev_prices)]
                features.extend(price_changes)
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([1.0, 1.0, 0.0, 0.0])  # Default values
        
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
        
        # Market context features
        market_context = info_bus.get('market_context', {})
        if 'volatility' in market_context:
            vol_data = market_context['volatility']
            if isinstance(vol_data, dict):
                avg_vol = np.mean(list(vol_data.values()))
                features.append(min(1.0, avg_vol * 50))
            else:
                features.append(min(1.0, float(vol_data) * 50))
        else:
            features.append(0.5)
        
        # Extend or truncate to target input size
        while len(features) < self.input_size:
            features.append(0.0)
        
        return np.array(features[:self.input_size], dtype=np.float32)

    def _update_market_history(self, features: np.ndarray, info_bus: InfoBus):
        """Update market history with enhanced metadata"""
        
        # Create enhanced market state record
        market_state = {
            'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
            'step_idx': info_bus.get('step_idx', self._step_count),
            'features': features.copy(),
            'context': extract_standard_context(info_bus),
            'prices': info_bus.get('prices', {}),
            'risk_snapshot': info_bus.get('risk', {})
        }
        
        self._market_history.append(market_state)
        
        # Update feature importance tracking
        self._update_feature_importance(features, info_bus)

    def _update_feature_importance(self, features: np.ndarray, info_bus: InfoBus):
        """Track feature importance for interpretability"""
        
        # Simple feature importance based on variance
        feature_names = [
            'price_1', 'price_2', 'price_change_1', 'price_change_2',
            'regime_trending', 'regime_volatile', 'regime_ranging',
            'volatility_level', 'drawdown_pct', 'exposure_pct', 'position_count',
            'session_asian', 'session_european', 'session_american', 'market_volatility'
        ]
        
        for i, (name, value) in enumerate(zip(feature_names[:len(features)], features)):
            if name not in self._feature_importance:
                self._feature_importance[name] = {'values': deque(maxlen=100), 'variance': 0.0}
            
            self._feature_importance[name]['values'].append(value)
            
            # Update variance estimate
            values = list(self._feature_importance[name]['values'])
            if len(values) > 1:
                self._feature_importance[name]['variance'] = np.var(values)

    def _generate_predictions(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Generate model predictions with enhanced context"""
        
        try:
            # Prepare input sequence
            sequence_features = [state['features'] for state in list(self._market_history)[-self.sequence_length:]]
            X = np.vstack(sequence_features)
            X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)
            
            # Context features
            context = extract_standard_context(info_bus)
            context_features = self._encode_context_features(context)
            context_tensor = torch.from_numpy(context_features).unsqueeze(0).to(self.device)
            
            self.eval()
            with torch.no_grad():
                # LSTM forward pass
                lstm_out, _ = self.lstm(X_tensor)
                hidden_state = lstm_out[:, -1, :]  # Last hidden state
                
                # Apply attention mechanism
                attended_out, attention_weights = self.attention(
                    hidden_state.unsqueeze(1),
                    lstm_out,
                    lstm_out
                )
                
                # Integrate context
                context_encoded = self.context_encoder(context_tensor)
                combined_features = torch.cat([attended_out.squeeze(1), context_encoded], dim=1)
                
                # Generate predictions
                price_pred = self.price_head(combined_features)
                vol_pred = self.volatility_head(combined_features)
                regime_pred = self.regime_head(combined_features)
                
                predictions = {
                    'price_changes': price_pred.cpu().numpy()[0],
                    'volatility': vol_pred.cpu().numpy()[0],
                    'regime_probs': regime_pred.cpu().numpy()[0],
                    'confidence': self._calculate_prediction_confidence(attention_weights),
                    'attention_weights': attention_weights.cpu().numpy()[0],
                    'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
                    'context': context.copy()
                }
            
            self._recent_predictions.append(predictions)
            
            self.log_operator_info(
                f"Predictions generated",
                price_changes=f"[{predictions['price_changes'][0]:.4f}, {predictions['price_changes'][1]:.4f}]",
                volatility=f"[{predictions['volatility'][0]:.4f}, {predictions['volatility'][1]:.4f}]",
                regime=f"{np.argmax(predictions['regime_probs'])}",
                confidence=f"{predictions['confidence']:.3f}"
            )
            
            return predictions
            
        except Exception as e:
            self.log_operator_error(f"Prediction generation failed: {e}")
            return self._create_empty_predictions()

    def _encode_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode context into feature vector"""
        
        features = []
        
        # Regime encoding
        regime = context.get('regime', 'unknown')
        regime_values = {'trending': 1.0, 'volatile': 0.5, 'ranging': 0.0, 'unknown': 0.33}
        features.append(regime_values.get(regime, 0.33))
        
        # Session encoding
        session = context.get('session', 'unknown')
        session_values = {'asian': 0.25, 'european': 0.5, 'american': 0.75, 'closed': 0.0}
        features.append(session_values.get(session, 0.0))
        
        # Risk features
        features.extend([
            context.get('drawdown_pct', 0.0) / 100.0,
            context.get('exposure_pct', 0.0) / 100.0,
            context.get('position_count', 0.0) / 10.0,
            context.get('risk_score', 0.0) / 100.0
        ])
        
        # Market features
        features.extend([
            context.get('consensus', 0.5),
            context.get('votes_summary', {}).get('avg_confidence', 0.5)
        ])
        
        return np.array(features, dtype=np.float32)

    def _calculate_prediction_confidence(self, attention_weights: torch.Tensor) -> float:
        """Calculate confidence based on attention distribution"""
        
        # Higher entropy = lower confidence (attention is spread out)
        attention_probs = attention_weights.squeeze().cpu().numpy()
        if len(attention_probs) > 1:
            entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-8))
            max_entropy = np.log(len(attention_probs))
            confidence = 1.0 - (entropy / max_entropy)
        else:
            confidence = 0.5
        
        return float(np.clip(confidence, 0.0, 1.0))

    def _track_prediction_performance(self, predictions: Dict[str, Any], info_bus: InfoBus):
        """Track prediction accuracy for performance monitoring"""
        
        # Compare with actual market movements if available
        if len(self._recent_predictions) >= 2:
            # Get previous prediction and current actual values
            prev_prediction = self._recent_predictions[-2]
            current_prices = info_bus.get('prices', {})
            
            if current_prices and len(self._market_history) >= 2:
                prev_prices = self._market_history[-2]['prices']
                
                # Calculate actual price changes
                actual_changes = []
                predicted_changes = prev_prediction['price_changes']
                
                for symbol in list(current_prices.keys())[:2]:
                    if symbol in prev_prices and prev_prices[symbol] > 0:
                        actual_change = (current_prices[symbol] - prev_prices[symbol]) / prev_prices[symbol]
                        actual_changes.append(actual_change)
                
                if len(actual_changes) == 2:
                    # Calculate prediction error
                    mse_error = np.mean((np.array(actual_changes) - predicted_changes) ** 2)
                    self._prediction_errors.append(mse_error)
                    
                    # Update prediction accuracy
                    if len(self._prediction_errors) >= 10:
                        recent_errors = list(self._prediction_errors)[-10:]
                        avg_error = np.mean(recent_errors)
                        self._model_performance['prediction_accuracy'] = max(0.0, 1.0 - avg_error * 100)

    def fit_on_history(self, validation_split: float = 0.2, epochs: int = 10) -> Dict[str, float]:
        """Train model on accumulated market history"""
        
        if len(self._market_history) < self.sequence_length + 10:
            self.log_operator_warning(
                f"Insufficient data for training",
                required=self.sequence_length + 10,
                available=len(self._market_history)
            )
            return {'loss': float('inf'), 'val_loss': float('inf')}
        
        try:
            # Prepare training data
            X, Y_price, Y_vol, Y_regime = self._prepare_training_data()
            
            if len(X) == 0:
                return {'loss': float('inf'), 'val_loss': float('inf')}
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            Y_price_train, Y_price_val = Y_price[:split_idx], Y_price[split_idx:]
            Y_vol_train, Y_vol_val = Y_vol[:split_idx], Y_vol[split_idx:]
            Y_regime_train, Y_regime_val = Y_regime[:split_idx], Y_regime[split_idx:]
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.stack(X_train),
                torch.stack(Y_price_train),
                torch.stack(Y_vol_train),
                torch.stack(Y_regime_train)
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training loop
            self.train()
            best_val_loss = float('inf')
            training_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for X_batch, Y_price_batch, Y_vol_batch, Y_regime_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    Y_price_batch = Y_price_batch.to(self.device)
                    Y_vol_batch = Y_vol_batch.to(self.device)
                    Y_regime_batch = Y_regime_batch.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    lstm_out, _ = self.lstm(X_batch)
                    hidden_state = lstm_out[:, -1, :]
                    
                    # Predictions
                    price_pred = self.price_head(hidden_state)
                    vol_pred = self.volatility_head(hidden_state)
                    regime_pred = self.regime_head(hidden_state)
                    
                    # Calculate losses
                    price_loss = self.mse_loss(price_pred, Y_price_batch)
                    vol_loss = self.mse_loss(vol_pred, Y_vol_batch)
                    regime_loss = self.ce_loss(regime_pred, Y_regime_batch.long())
                    
                    # Combined loss
                    total_loss = price_loss + vol_loss + 0.5 * regime_loss
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
                    
                    self.optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                training_losses.append(avg_loss)
                
                # Validation
                val_loss = self._validate_model(X_val, Y_price_val, Y_vol_val, Y_regime_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # Update training curves
                self._training_curves['loss'].append(avg_loss)
                self._training_curves['accuracy'].append(1.0 / (1.0 + val_loss))
                
                self.log_operator_info(
                    f"Training epoch {epoch+1}/{epochs}",
                    train_loss=f"{avg_loss:.6f}",
                    val_loss=f"{val_loss:.6f}",
                    best_val=f"{best_val_loss:.6f}"
                )
            
            # Update model status
            self._is_trained = True
            self._last_training_time = datetime.datetime.now()
            self._model_performance['training_loss'] = training_losses[-1]
            self._model_performance['validation_error'] = best_val_loss
            self._model_confidence = max(0.1, 1.0 / (1.0 + best_val_loss))
            
            # Record training session
            training_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'epochs': epochs,
                'final_loss': training_losses[-1],
                'best_val_loss': best_val_loss,
                'data_points': len(X),
                'model_confidence': self._model_confidence
            }
            self._training_history.append(training_record)
            
            self.log_operator_info(
                f"Model training completed",
                epochs=epochs,
                final_loss=f"{training_losses[-1]:.6f}",
                validation_loss=f"{best_val_loss:.6f}",
                confidence=f"{self._model_confidence:.3f}",
                data_points=len(X)
            )
            
            return {
                'loss': training_losses[-1],
                'val_loss': best_val_loss,
                'confidence': self._model_confidence
            }
            
        except Exception as e:
            self.log_operator_error(f"Model training failed: {e}")
            return {'loss': float('inf'), 'val_loss': float('inf')}

    def _prepare_training_data(self) -> Tuple[List[torch.Tensor], ...]:
        """Prepare training data from market history"""
        
        X, Y_price, Y_vol, Y_regime = [], [], [], []
        
        history_list = list(self._market_history)
        
        for i in range(self.sequence_length, len(history_list)):
            # Input sequence
            sequence = [state['features'] for state in history_list[i-self.sequence_length:i]]
            X.append(torch.from_numpy(np.vstack(sequence)).float())
            
            # Target values
            current_state = history_list[i]
            prev_state = history_list[i-1]
            
            # Price changes
            current_prices = current_state.get('prices', {})
            prev_prices = prev_state.get('prices', {})
            
            price_changes = []
            for symbol in list(current_prices.keys())[:2]:
                if symbol in prev_prices and prev_prices[symbol] > 0:
                    change = (current_prices[symbol] - prev_prices[symbol]) / prev_prices[symbol]
                    price_changes.append(change)
                else:
                    price_changes.append(0.0)
            
            # Ensure 2 price changes
            while len(price_changes) < 2:
                price_changes.append(0.0)
            
            Y_price.append(torch.tensor(price_changes[:2], dtype=torch.float32))
            
            # Volatility (simplified)
            volatility = current_state['context'].get('volatility_level', 'medium')
            vol_values = {'low': [0.2, 0.2], 'medium': [0.5, 0.5], 'high': [0.8, 0.8], 'extreme': [1.0, 1.0]}
            Y_vol.append(torch.tensor(vol_values.get(volatility, [0.5, 0.5]), dtype=torch.float32))
            
            # Regime
            regime = current_state['context'].get('regime', 'unknown')
            regime_idx = {'trending': 0, 'volatile': 1, 'ranging': 2, 'unknown': 1}
            Y_regime.append(torch.tensor(regime_idx.get(regime, 1), dtype=torch.long))
        
        return X, Y_price, Y_vol, Y_regime

    def _validate_model(self, X_val: List[torch.Tensor], Y_price_val: List[torch.Tensor],
                       Y_vol_val: List[torch.Tensor], Y_regime_val: List[torch.Tensor]) -> float:
        """Validate model performance"""
        
        if not X_val:
            return float('inf')
        
        self.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for i in range(len(X_val)):
                X_batch = X_val[i].unsqueeze(0).to(self.device)
                Y_price_batch = Y_price_val[i].unsqueeze(0).to(self.device)
                Y_vol_batch = Y_vol_val[i].unsqueeze(0).to(self.device)
                Y_regime_batch = Y_regime_val[i].unsqueeze(0).to(self.device)
                
                # Forward pass
                lstm_out, _ = self.lstm(X_batch)
                hidden_state = lstm_out[:, -1, :]
                
                # Predictions
                price_pred = self.price_head(hidden_state)
                vol_pred = self.volatility_head(hidden_state)
                regime_pred = self.regime_head(hidden_state)
                
                # Calculate losses
                price_loss = self.mse_loss(price_pred, Y_price_batch)
                vol_loss = self.mse_loss(vol_pred, Y_vol_batch)
                regime_loss = self.ce_loss(regime_pred, Y_regime_batch.long())
                
                total_loss += (price_loss + vol_loss + 0.5 * regime_loss).item()
        
        return total_loss / len(X_val)

    def simulate_scenarios(self, steps: int = 10, num_scenarios: int = 5) -> List[Dict[str, Any]]:
        """Generate multiple market scenarios"""
        
        if not self._is_trained or len(self._market_history) < self.sequence_length:
            self.log_operator_warning("Model not ready for scenario simulation")
            return []
        
        try:
            scenarios = []
            
            for scenario_id in range(num_scenarios):
                scenario = self._generate_single_scenario(steps, scenario_id, num_scenarios)
                scenarios.append(scenario)
            
            # Cache scenarios for reuse
            self._scenario_cache = {
                'timestamp': datetime.datetime.now().isoformat(),
                'scenarios': scenarios,
                'parameters': {'steps': steps, 'num_scenarios': num_scenarios}
            }
            
            # Update scenario quality metric
            scenario_diversity = self._calculate_scenario_diversity(scenarios)
            self._model_performance['scenario_quality'] = scenario_diversity
            
            self.log_operator_info(
                f"Generated {num_scenarios} scenarios",
                steps=steps,
                diversity_score=f"{scenario_diversity:.3f}"
            )
            
            return scenarios
            
        except Exception as e:
            self.log_operator_error(f"Scenario simulation failed: {e}")
            return []

    def _generate_single_scenario(self, steps: int, scenario_id: int, num_scenarios: int) -> Dict[str, Any]:
        """Generate a single market scenario"""
        
        # Start with current market state
        current_sequence = [state['features'] for state in list(self._market_history)[-self.sequence_length:]]
        scenario_path = []
        
        self.eval()
        with torch.no_grad():
            for step in range(steps):
                # Prepare input
                X = torch.from_numpy(np.vstack(current_sequence)).unsqueeze(0).float().to(self.device)
                
                # Generate prediction
                lstm_out, _ = self.lstm(X)
                hidden_state = lstm_out[:, -1, :]
                
                price_pred = self.price_head(hidden_state)
                vol_pred = self.volatility_head(hidden_state)
                regime_pred = self.regime_head(hidden_state)
                
                # Add some noise for scenario diversity
                noise_scale = 0.1 * (scenario_id + 1) / num_scenarios
                price_pred += torch.randn_like(price_pred) * noise_scale
                vol_pred += torch.randn_like(vol_pred) * noise_scale * 0.5
                
                # Create step prediction
                step_prediction = {
                    'step': step,
                    'price_changes': price_pred.cpu().numpy()[0],
                    'volatility': vol_pred.cpu().numpy()[0],
                    'regime_probs': regime_pred.cpu().numpy()[0],
                    'predicted_regime': int(torch.argmax(regime_pred, dim=1).cpu().numpy()[0])
                }
                
                scenario_path.append(step_prediction)
                
                # Update sequence for next prediction
                # Create new features based on prediction
                new_features = self._prediction_to_features(step_prediction, current_sequence[-1])
                current_sequence = current_sequence[1:] + [new_features]
        
        return {
            'scenario_id': scenario_id,
            'steps': steps,
            'path': scenario_path,
            'summary': self._summarize_scenario(scenario_path)
        }

    def _prediction_to_features(self, prediction: Dict[str, Any], prev_features: np.ndarray) -> np.ndarray:
        """Convert prediction back to feature representation"""
        
        new_features = prev_features.copy()
        
        # Update price features (first 2 elements)
        if len(prediction['price_changes']) >= 2:
            # Apply price changes to previous prices
            new_features[0] *= (1 + prediction['price_changes'][0])
            new_features[1] *= (1 + prediction['price_changes'][1])
            
            # Update price change features
            new_features[2] = prediction['price_changes'][0]
            new_features[3] = prediction['price_changes'][1]
        
        # Update regime features (positions 4-6)
        regime_probs = prediction['regime_probs']
        if len(regime_probs) >= 3:
            new_features[4:7] = regime_probs
        
        # Update volatility feature (position 7)
        if len(prediction['volatility']) >= 1:
            new_features[7] = prediction['volatility'][0]
        
        return new_features

    def _summarize_scenario(self, scenario_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize scenario for analysis"""
        
        if not scenario_path:
            return {}
        
        price_changes = [step['price_changes'] for step in scenario_path]
        volatilities = [step['volatility'] for step in scenario_path]
        regimes = [step['predicted_regime'] for step in scenario_path]
        
        return {
            'total_return_1': np.sum([pc[0] for pc in price_changes if len(pc) > 0]),
            'total_return_2': np.sum([pc[1] for pc in price_changes if len(pc) > 1]),
            'avg_volatility_1': np.mean([v[0] for v in volatilities if len(v) > 0]),
            'avg_volatility_2': np.mean([v[1] for v in volatilities if len(v) > 1]),
            'regime_distribution': {
                'trending': sum(1 for r in regimes if r == 0) / len(regimes),
                'volatile': sum(1 for r in regimes if r == 1) / len(regimes),
                'ranging': sum(1 for r in regimes if r == 2) / len(regimes)
            },
            'max_drawdown': self._calculate_max_drawdown([pc[0] for pc in price_changes if len(pc) > 0])
        }

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown))

    def _calculate_scenario_diversity(self, scenarios: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for scenarios"""
        
        if len(scenarios) < 2:
            return 0.0
        
        # Compare scenario summaries
        returns_1 = [s['summary'].get('total_return_1', 0) for s in scenarios]
        returns_2 = [s['summary'].get('total_return_2', 0) for s in scenarios]
        
        # Diversity as standard deviation of returns
        diversity_1 = np.std(returns_1) if len(returns_1) > 1 else 0.0
        diversity_2 = np.std(returns_2) if len(returns_2) > 1 else 0.0
        
        return float((diversity_1 + diversity_2) / 2)

    def _create_empty_predictions(self) -> Dict[str, Any]:
        """Create empty predictions for error scenarios"""
        
        return {
            'price_changes': np.zeros(2),
            'volatility': np.ones(2) * 0.5,
            'regime_probs': np.array([0.33, 0.33, 0.34]),
            'confidence': 0.0,
            'attention_weights': np.ones(self.sequence_length) / self.sequence_length,
            'timestamp': datetime.datetime.now().isoformat(),
            'context': {}
        }

    def _update_model_performance(self):
        """Update comprehensive model performance metrics"""
        
        try:
            # Model confidence based on recent training
            if self._training_history:
                recent_training = self._training_history[-1]
                training_quality = 1.0 / (1.0 + recent_training.get('best_val_loss', float('inf')))
                self._model_confidence = training_quality
            
            # Stability score based on prediction consistency
            if len(self._recent_predictions) >= 5:
                confidences = [p['confidence'] for p in list(self._recent_predictions)[-5:]]
                self._stability_score = 1.0 - np.std(confidences)
            
            # Update performance metrics
            self._update_performance_metric('model_confidence', self._model_confidence)
            self._update_performance_metric('stability_score', self._stability_score)
            self._update_performance_metric('prediction_accuracy', self._model_performance['prediction_accuracy'])
            
        except Exception as e:
            self.log_operator_warning(f"Performance update failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with model metrics"""
        
        try:
            # Model status
            model_trained = float(self._is_trained)
            model_confidence = self._model_confidence
            stability_score = self._stability_score
            
            # Recent prediction quality
            prediction_quality = 0.0
            if self._recent_predictions:
                recent_confidences = [p['confidence'] for p in list(self._recent_predictions)[-5:]]
                prediction_quality = np.mean(recent_confidences)
            
            # Training performance
            training_loss = 1.0 / (1.0 + self._model_performance['training_loss']) if self._model_performance['training_loss'] != float('inf') else 0.0
            validation_error = 1.0 / (1.0 + self._model_performance['validation_error']) if self._model_performance['validation_error'] != float('inf') else 0.0
            
            # Data availability
            data_sufficiency = min(1.0, len(self._market_history) / (self.sequence_length * 2))
            
            # Feature importance diversity
            feature_diversity = 0.0
            if self._feature_importance:
                variances = [info['variance'] for info in self._feature_importance.values()]
                feature_diversity = np.mean(variances) if variances else 0.0
            
            # Scenario quality
            scenario_quality = self._model_performance['scenario_quality']
            
            # Latest predictions if available
            latest_price_change = 0.0
            latest_volatility = 0.5
            if self._recent_predictions:
                latest = self._recent_predictions[-1]
                latest_price_change = np.mean(latest['price_changes'])
                latest_volatility = np.mean(latest['volatility'])
            
            # Combine all components
            observation = np.array([
                model_trained,
                model_confidence,
                stability_score,
                prediction_quality,
                training_loss,
                validation_error,
                data_sufficiency,
                feature_diversity,
                scenario_quality,
                latest_price_change,
                latest_volatility,
                self._model_performance['prediction_accuracy']
            ], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(12, dtype=np.float32)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Propose actions based on world model predictions"""
        
        if not self._is_trained or not self._recent_predictions:
            return np.zeros(2, dtype=np.float32)
        
        try:
            # Get latest prediction
            latest_prediction = self._recent_predictions[-1]
            
            # Extract predicted price changes
            price_changes = latest_prediction['price_changes']
            confidence = latest_prediction['confidence']
            
            # Scale actions by confidence and predicted magnitude
            action_scaling = confidence * 0.5  # Conservative scaling
            
            # Convert price changes to trading actions
            # Positive change -> buy signal, negative -> sell signal
            actions = price_changes * action_scaling
            
            # Apply additional constraints
            actions = np.clip(actions, -1.0, 1.0)  # Limit action magnitude
            
            return actions.astype(np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Action proposal failed: {e}")
            return np.zeros(2, dtype=np.float32)

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return confidence in world model predictions"""
        
        base_confidence = 0.5
        
        # Confidence from model training
        if self._is_trained:
            base_confidence += self._model_confidence * 0.3
        
        # Confidence from prediction consistency
        base_confidence += self._stability_score * 0.2
        
        # Confidence from data sufficiency
        data_confidence = min(0.2, len(self._market_history) / (self.sequence_length * 2) * 0.2)
        base_confidence += data_confidence
        
        # Confidence from recent predictions
        if self._recent_predictions:
            recent_confidences = [p['confidence'] for p in list(self._recent_predictions)[-3:]]
            avg_prediction_confidence = np.mean(recent_confidences)
            base_confidence += avg_prediction_confidence * 0.3
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with network rebuilding if needed"""
        old_hidden_size = self.hidden_size
        old_num_layers = self.num_layers
        
        self.hidden_size = int(np.clip(genome.get("hidden_size", self.hidden_size), 32, 256))
        self.num_layers = int(np.clip(genome.get("num_layers", self.num_layers), 1, 4))
        self.dropout = float(np.clip(genome.get("dropout", self.dropout), 0.0, 0.5))
        self.learning_rate = float(np.clip(genome.get("learning_rate", self.learning_rate), 1e-5, 1e-2))
        self.sequence_length = int(np.clip(genome.get("sequence_length", self.sequence_length), 20, 100))
        self.batch_size = int(np.clip(genome.get("batch_size", self.batch_size), 16, 128))
        self.prediction_horizon = int(np.clip(genome.get("prediction_horizon", self.prediction_horizon), 5, 20))
        self.gradient_clip = float(np.clip(genome.get("gradient_clip", self.gradient_clip), 0.5, 2.0))
        self.weight_decay = float(np.clip(genome.get("weight_decay", self.weight_decay), 1e-6, 1e-3))
        
        self.genome = {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "prediction_horizon": self.prediction_horizon,
            "gradient_clip": self.gradient_clip,
            "weight_decay": self.weight_decay
        }
        
        # Rebuild networks if architecture changed
        if old_hidden_size != self.hidden_size or old_num_layers != self.num_layers:
            try:
                self.log_operator_info(f"Rebuilding neural networks: hidden={self.hidden_size}, layers={self.num_layers}")
                self._initialize_neural_components(str(self.device))
                
                # Reset training status
                self._is_trained = False
                self._model_confidence = 0.0
                
            except Exception as e:
                self.log_operator_error(f"Network rebuild failed: {e}")
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with neural network weight mutation"""
        g = self.genome.copy()
        mutations = []
        
        # Architectural mutations
        if np.random.rand() < mutation_rate:
            old_val = g["hidden_size"]
            # Prefer powers of 2 for efficiency
            options = [32, 64, 128, 256]
            g["hidden_size"] = random.choice(options)
            mutations.append(f"hidden_size: {old_val} → {g['hidden_size']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["num_layers"]
            g["num_layers"] = int(np.clip(old_val + np.random.choice([-1, 0, 1]), 1, 4))
            mutations.append(f"num_layers: {old_val} → {g['num_layers']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["sequence_length"]
            g["sequence_length"] = int(np.clip(old_val + np.random.randint(-10, 11), 20, 100))
            mutations.append(f"sequence_length: {old_val} → {g['sequence_length']}")
            
        # Parameter mutations
        if np.random.rand() < mutation_rate:
            old_val = g["learning_rate"]
            g["learning_rate"] = float(np.clip(old_val * np.random.uniform(0.5, 2.0), 1e-5, 1e-2))
            mutations.append(f"learning_rate: {old_val:.1e} → {g['learning_rate']:.1e}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["dropout"]
            g["dropout"] = float(np.clip(old_val + np.random.uniform(-0.1, 0.1), 0.0, 0.5))
            mutations.append(f"dropout: {old_val:.2f} → {g['dropout']:.2f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["batch_size"]
            options = [16, 32, 64, 128]
            g["batch_size"] = random.choice(options)
            mutations.append(f"batch_size: {old_val} → {g['batch_size']}")
        
        if mutations:
            self.log_operator_info(f"World model mutation applied", changes=", ".join(mutations))
            
        # Neural weight mutation
        if np.random.rand() < mutation_rate * 0.3:
            noise_std = 0.02
            with torch.no_grad():
                for param in self.parameters():
                    if param.requires_grad:
                        param.data += noise_std * torch.randn_like(param.data)
            
            self.log_operator_info(f"Neural weights mutated with std={noise_std}")
        
        self.set_genome(g)
        
    def crossover(self, other: "EnhancedWorldModel") -> "EnhancedWorldModel":
        """Enhanced crossover with neural weight mixing"""
        if not isinstance(other, EnhancedWorldModel):
            self.log_operator_warning("Crossover with incompatible type")
            return self
        
        # Performance-based crossover
        self_performance = self._model_confidence
        other_performance = other._model_confidence
        
        # Favor higher performance parent
        if self_performance > other_performance:
            bias = 0.7  # Favor self
        else:
            bias = 0.3  # Favor other
        
        new_g = {k: (self.genome[k] if np.random.rand() < bias else other.genome[k]) for k in self.genome}
        
        child = EnhancedWorldModel(genome=new_g, debug=self.config.debug)
        
        # Neural weight crossover (if architectures match)
        if (self.hidden_size == other.hidden_size and 
            self.num_layers == other.num_layers and 
            self.hidden_size == child.hidden_size and 
            self.num_layers == child.num_layers):
            
            try:
                with torch.no_grad():
                    for child_param, self_param, other_param in zip(
                        child.parameters(),
                        self.parameters(),
                        other.parameters()
                    ):
                        if child_param.shape == self_param.shape == other_param.shape:
                            mask = torch.rand_like(self_param) > 0.5
                            child_param.data = torch.where(mask, self_param.data, other_param.data)
                
                self.log_operator_info("Neural weight crossover completed")
                
            except Exception as e:
                self.log_operator_warning(f"Neural weight crossover failed: {e}")
        
        # Inherit best training data
        if self_performance > other_performance and self._market_history:
            child._market_history = self._market_history.copy()
        elif other._market_history:
            child._market_history = other._market_history.copy()
        
        return child

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check neural network parameters
            for param in self.parameters():
                if not torch.all(torch.isfinite(param.data)):
                    return False
                    
            # Check data consistency
            if len(self._recent_predictions) > 0:
                for pred in self._recent_predictions:
                    if not all(np.isfinite(pred['price_changes'])):
                        return False
                    if not all(np.isfinite(pred['volatility'])):
                        return False
                        
            # Check performance metrics
            for value in self._model_performance.values():
                if not np.isfinite(value):
                    return False
                    
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        model_details = {
            'model_info': {
                'is_trained': self._is_trained,
                'model_confidence': self._model_confidence,
                'stability_score': self._stability_score,
                'last_training': self._last_training_time.isoformat() if self._last_training_time else None
            },
            'architecture_info': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length
            },
            'performance_info': self._model_performance.copy(),
            'data_info': {
                'market_history_size': len(self._market_history),
                'prediction_history_size': len(self._prediction_history),
                'training_sessions': len(self._training_history),
                'recent_predictions': len(self._recent_predictions)
            },
            'training_info': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'gradient_clip': self.gradient_clip,
                'weight_decay': self.weight_decay
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(model_details)
            return base_details
        
        return model_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        
        # Convert PyTorch state
        model_state = {
            'model_state_dict': {k: v.cpu().numpy().tolist() for k, v in self.state_dict().items()},
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        return {
            "model_state": model_state,
            "genome": self.genome.copy(),
            "model_performance": self._model_performance.copy(),
            "market_history": [
                {
                    'timestamp': state['timestamp'],
                    'step_idx': state['step_idx'],
                    'features': state['features'].tolist(),
                    'context': state['context'],
                    'prices': state['prices']
                }
                for state in list(self._market_history)[-100:]  # Keep recent history only
            ],
            "prediction_history": list(self._prediction_history)[-50:],
            "training_history": list(self._training_history)[-20:],
            "recent_predictions": [
                {
                    'price_changes': pred['price_changes'].tolist(),
                    'volatility': pred['volatility'].tolist(),
                    'regime_probs': pred['regime_probs'].tolist(),
                    'confidence': pred['confidence'],
                    'timestamp': pred['timestamp'],
                    'context': pred['context']
                }
                for pred in list(self._recent_predictions)[-10:]
            ],
            "model_status": {
                'is_trained': self._is_trained,
                'last_training_time': self._last_training_time.isoformat() if self._last_training_time else None,
                'model_confidence': self._model_confidence,
                'stability_score': self._stability_score
            },
            "feature_importance": {k: {'variance': v['variance']} for k, v in self._feature_importance.items()},
            "training_curves": {k: list(v)[-30:] for k, v in self._training_curves.items()}
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        # Restore genome and rebuild if needed
        self.set_genome(module_state.get("genome", self.genome))
        
        # Restore model state
        model_state = module_state.get("model_state", {})
        if "model_state_dict" in model_state:
            try:
                # Convert numpy arrays back to tensors
                state_dict = {}
                for k, v in model_state["model_state_dict"].items():
                    state_dict[k] = torch.tensor(v, dtype=torch.float32)
                
                self.load_state_dict(state_dict)
                
                if "optimizer_state_dict" in model_state:
                    self.optimizer.load_state_dict(model_state["optimizer_state_dict"])
                    
            except Exception as e:
                self.log_operator_warning(f"Model state restoration failed: {e}")
        
        # Restore performance metrics
        self._model_performance = module_state.get("model_performance", self._model_performance)
        
        # Restore market history
        market_history_data = module_state.get("market_history", [])
        self._market_history.clear()
        for state_data in market_history_data:
            state = {
                'timestamp': state_data['timestamp'],
                'step_idx': state_data['step_idx'],
                'features': np.array(state_data['features'], dtype=np.float32),
                'context': state_data['context'],
                'prices': state_data['prices'],
                'risk_snapshot': state_data.get('risk_snapshot', {})
            }
            self._market_history.append(state)
        
        # Restore prediction history
        self._prediction_history = deque(module_state.get("prediction_history", []), maxlen=100)
        self._training_history = deque(module_state.get("training_history", []), maxlen=50)
        
        # Restore recent predictions
        recent_predictions_data = module_state.get("recent_predictions", [])
        self._recent_predictions.clear()
        for pred_data in recent_predictions_data:
            pred = {
                'price_changes': np.array(pred_data['price_changes'], dtype=np.float32),
                'volatility': np.array(pred_data['volatility'], dtype=np.float32),
                'regime_probs': np.array(pred_data['regime_probs'], dtype=np.float32),
                'confidence': pred_data['confidence'],
                'timestamp': pred_data['timestamp'],
                'context': pred_data['context']
            }
            self._recent_predictions.append(pred)
        
        # Restore model status
        model_status = module_state.get("model_status", {})
        self._is_trained = model_status.get('is_trained', False)
        self._model_confidence = model_status.get('model_confidence', 0.0)
        self._stability_score = model_status.get('stability_score', 1.0)
        
        if model_status.get('last_training_time'):
            self._last_training_time = datetime.datetime.fromisoformat(model_status['last_training_time'])
        
        # Restore feature importance
        feature_importance_data = module_state.get("feature_importance", {})
        self._feature_importance.clear()
        for name, data in feature_importance_data.items():
            self._feature_importance[name] = {
                'values': deque(maxlen=100),
                'variance': data['variance']
            }
        
        # Restore training curves
        training_curves_data = module_state.get("training_curves", {})
        for curve_name, curve_data in training_curves_data.items():
            if curve_name in self._training_curves:
                self._training_curves[curve_name] = deque(curve_data, maxlen=100)

    def get_world_model_report(self) -> str:
        """Generate operator-friendly world model report"""
        
        # Model status
        if self._is_trained:
            if self._model_confidence > 0.8:
                model_status = "🚀 Excellent"
            elif self._model_confidence > 0.6:
                model_status = "✅ Good"
            elif self._model_confidence > 0.4:
                model_status = "⚡ Fair"
            else:
                model_status = "⚠️ Poor"
        else:
            model_status = "❌ Untrained"
        
        # Prediction trend
        prediction_trend = "📊 Stable"
        if len(self._recent_predictions) >= 3:
            recent_confidences = [p['confidence'] for p in list(self._recent_predictions)[-3:]]
            if len(recent_confidences) >= 2:
                trend = recent_confidences[-1] - recent_confidences[0]
                if trend > 0.1:
                    prediction_trend = "📈 Improving"
                elif trend < -0.1:
                    prediction_trend = "📉 Declining"
        
        # Training status
        training_status = "No training"
        if self._training_history:
            last_training = self._training_history[-1]
            training_time = datetime.datetime.fromisoformat(last_training['timestamp'])
            time_ago = datetime.datetime.now() - training_time
            if time_ago.total_seconds() < 3600:
                training_status = f"Recent ({time_ago.seconds//60}m ago)"
            else:
                training_status = f"Stale ({time_ago.days}d ago)"
        
        return f"""
🌍 ENHANCED WORLD MODEL
═══════════════════════════════════════
🧠 Model Status: {model_status} ({self._model_confidence:.3f})
📈 Predictions: {prediction_trend}
🎯 Stability: {self._stability_score:.3f}
⏰ Training: {training_status}

🏗️ ARCHITECTURE
• Input Features: {self.input_size}
• Hidden Units: {self.hidden_size}
• LSTM Layers: {self.num_layers}
• Sequence Length: {self.sequence_length}
• Dropout: {self.dropout:.2f}

📊 PERFORMANCE METRICS
• Prediction Accuracy: {self._model_performance['prediction_accuracy']:.3f}
• Training Loss: {self._model_performance['training_loss']:.6f}
• Validation Error: {self._model_performance['validation_error']:.6f}
• Scenario Quality: {self._model_performance['scenario_quality']:.3f}

💾 DATA STATUS
• Market History: {len(self._market_history)}/{self.sequence_length * 2} required
• Recent Predictions: {len(self._recent_predictions)}
• Training Sessions: {len(self._training_history)}
• Feature Importance: {len(self._feature_importance)} tracked

🔧 TRAINING CONFIG
• Learning Rate: {self.learning_rate:.1e}
• Batch Size: {self.batch_size}
• Gradient Clip: {self.gradient_clip}
• Weight Decay: {self.weight_decay:.1e}

📈 RECENT ACTIVITY
• Predictions (last hour): {len([p for p in self._recent_predictions if (datetime.datetime.now() - datetime.datetime.fromisoformat(p['timestamp'])).total_seconds() < 3600])}
• High-confidence predictions: {len([p for p in self._recent_predictions if p['confidence'] > 0.7])}
• Scenario cache: {'Available' if self._scenario_cache else 'Empty'}
• Feature diversity: {len([f for f in self._feature_importance.values() if f['variance'] > 0.01])} active features
        """

    # Maintain backward compatibility
    def step(self, market_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, market_data=market_data, **kwargs)

    def fit(self, feature1: np.ndarray, feature2: np.ndarray, seq_len: int = 50,
            batch_size: int = 64, epochs: int = 5) -> float:
        """Backward compatibility fit method"""
        # Store simple training data
        self._process_training_data({'feature1': feature1, 'feature2': feature2})
        
        # Use enhanced training if enough data
        if len(self._market_history) >= self.sequence_length + 10:
            result = self.fit_on_history(epochs=epochs)
            return result.get('loss', float('inf'))
        
        return float('inf')

    def simulate(self, init_returns: np.ndarray, init_vol: np.ndarray, steps: int = 10) -> np.ndarray:
        """Backward compatibility simulate method"""
        scenarios = self.simulate_scenarios(steps=steps, num_scenarios=1)
        
        if scenarios:
            # Extract price changes from first scenario
            scenario = scenarios[0]
            price_changes = [step['price_changes'] for step in scenario['path']]
            return np.array(price_changes, dtype=np.float32)
        
        return np.zeros((steps, 2), dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state: Dict[str, Any]):
        """Backward compatibility state method"""
        super().set_state(state)

    def _process_training_data(self, training_data: Dict[str, Any]):
        """Process manual training data"""
        # This is a placeholder for backward compatibility
        # Enhanced training uses market history automatically
        pass


# Alias for backward compatibility
RNNWorldModel = EnhancedWorldModel