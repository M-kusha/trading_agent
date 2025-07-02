# ─────────────────────────────────────────────────────────────
# File: modules/market/liquidity_heatmap_layer.py
# Enhanced with new infrastructure - 75% less boilerplate!
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
from typing import Any, List, Tuple, Dict, Optional
import tensorflow as tf

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin, VotingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class LiquidityHeatmapLayer(Module, AnalysisMixin, VotingMixin):
    """
    Enhanced liquidity heatmap layer with infrastructure integration.
    Class name unchanged - just enhanced capabilities!
    """
    
    def __init__(self, action_dim: int, debug: bool = True, genome: dict = None, weights: list = None, **kwargs):
        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=200,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome, action_dim)
        
        # Enhanced state initialization
        self._initialize_module_state()
        
        # Initialize neural network
        self._initialize_neural_network(weights)
        
        self.log_operator_info(
            "Liquidity heatmap layer initialized",
            action_dim=self.action_dim,
            lstm_units=self.lstm_units,
            sequence_length=self.seq_len,
            device="CPU (TensorFlow)"
        )

    def _initialize_genome_parameters(self, genome: Optional[dict], action_dim: int):
        """Initialize genome-based parameters"""
        if genome:
            self.lstm_units = int(genome.get("lstm_units", 32))
            self.seq_len = int(genome.get("seq_len", 10))
            self.dense_units = int(genome.get("dense_units", 2))
            self.train_epochs = int(genome.get("train_epochs", 10))
        else:
            self.lstm_units = 32
            self.seq_len = 10
            self.dense_units = 2
            self.train_epochs = 10

        self.action_dim = action_dim
        
        # Store genome for evolution
        self.genome = {
            "lstm_units": self.lstm_units,
            "seq_len": self.seq_len,
            "dense_units": self.dense_units,
            "train_epochs": self.train_epochs,
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        self._initialize_voting_state()
        
        # Liquidity specific state
        self.bids: List[Tuple[float, float]] = []
        self.asks: List[Tuple[float, float]] = []
        self.history = deque(maxlen=200)
        self._trained = False
        
        # Enhanced tracking
        self._liquidity_scores = deque(maxlen=100)
        self._prediction_accuracy = 0.0
        self._model_performance_score = 100.0
        self._training_history = []
        self._market_conditions = {"spread": 0.0, "depth": 0.0, "volatility": 0.0}

    def _initialize_neural_network(self, weights: Optional[list]):
        """Initialize TensorFlow neural network with enhanced monitoring"""
        
        try:
            with tf.device("/CPU:0"):
                self._model = self._build_enhanced_lstm()
                
            if weights is not None:
                self._set_weights_safe(weights)
                
            self.log_operator_info("Neural network initialized successfully")
            
        except Exception as e:
            self.log_operator_error(f"Neural network initialization failed: {e}")
            self._update_health_status("ERROR", f"NN init failed: {e}")
            # Create dummy model for safety
            self._model = self._create_dummy_model()

    def _build_enhanced_lstm(self):
        """Build enhanced LSTM model with monitoring"""
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.seq_len, 2)),
            tf.keras.layers.LSTM(self.lstm_units, return_sequences=False),
            tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
            tf.keras.layers.Dense(self.dense_units * 2, activation='relu'),
            tf.keras.layers.Dense(self.dense_units),
        ])
        
        # Enhanced compilation with metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def _create_dummy_model(self):
        """Create dummy model for fallback"""
        with tf.device("/CPU:0"):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.seq_len, 2)),
                tf.keras.layers.Dense(self.dense_units)
            ])
            model.compile(optimizer="adam", loss="mse")
        return model

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_voting_state()
        
        # Module-specific reset
        self.bids.clear()
        self.asks.clear()
        self.history.clear()
        self._trained = False
        self._liquidity_scores.clear()
        self._prediction_accuracy = 0.0
        self._model_performance_score = 100.0
        self._training_history.clear()
        self._market_conditions = {"spread": 0.0, "depth": 0.0, "volatility": 0.0}

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract market data for liquidity analysis
        market_data = self._extract_liquidity_data(info_bus, kwargs)
        
        # Process liquidity with enhanced analytics
        self._process_liquidity_data(market_data)
        
        # Train model if ready
        self._train_model_if_ready()

    def _extract_liquidity_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract liquidity data from InfoBus or simulate from market conditions"""
        
        # Try InfoBus first
        if info_bus:
            # Look for liquidity data in module_data
            module_data = info_bus.get('module_data', {})
            if 'liquidity' in module_data:
                return module_data['liquidity']
            
            # Extract from market context
            market_context = info_bus.get('market_context', {})
            volatility = market_context.get('volatility', {})
            
            # Get market status for session info
            market_status = info_bus.get('market_status', {})
            session = market_status.get('session', 'unknown')
            liquidity_score = market_status.get('liquidity_score', 0.5)
            
            return {
                'volatility': volatility,
                'session': session,
                'liquidity_score': liquidity_score,
                'regime': InfoBusExtractor.get_market_regime(info_bus),
                'exposure': InfoBusExtractor.get_exposure_pct(info_bus)
            }
        
        # Try kwargs (backward compatibility)
        if "env" in kwargs:
            env = kwargs["env"]
            return self._extract_from_env(env)
        
        # Fallback to default simulation
        return {'volatility': 0.01, 'session': 'unknown', 'liquidity_score': 0.5}

    def _extract_from_env(self, env) -> Dict[str, Any]:
        """Extract liquidity data from environment"""
        
        try:
            if hasattr(env, 'get_volatility_profile'):
                vol_profile = env.get_volatility_profile()
                instrument = env.instruments[0] if hasattr(env, 'instruments') and env.instruments else 'default'
                volatility = vol_profile.get(instrument, 0.01)
            else:
                volatility = 0.01
                
            return {
                'volatility': volatility,
                'session': 'trading',
                'liquidity_score': 0.5,
                'source': 'env'
            }
            
        except Exception as e:
            self.log_operator_warning(f"Failed to extract from env: {e}")
            return {'volatility': 0.01, 'session': 'unknown', 'liquidity_score': 0.5}

    def _process_liquidity_data(self, market_data: Dict[str, Any]):
        """Process liquidity data with enhanced simulation and tracking"""
        
        try:
            # Extract or simulate liquidity metrics
            spread, depth = self._simulate_liquidity_metrics(market_data)
            
            # Update market conditions tracking
            self._market_conditions.update({
                'spread': spread,
                'depth': depth,
                'volatility': market_data.get('volatility', 0.01)
            })
            
            # Store in history
            self.history.append((spread, depth))
            
            # Calculate and track liquidity score
            liquidity_score = self._calculate_liquidity_score(spread, depth)
            self._liquidity_scores.append(liquidity_score)
            
            # Update performance metrics
            self._update_performance_metric('liquidity_score', liquidity_score)
            self._update_performance_metric('spread', spread)
            self._update_performance_metric('depth', depth)
            
            # Log significant changes
            if len(self._liquidity_scores) > 1:
                score_change = liquidity_score - self._liquidity_scores[-2]
                if abs(score_change) > 0.2:
                    self.log_operator_info(
                        f"Significant liquidity change detected",
                        score=f"{liquidity_score:.3f}",
                        change=f"{score_change:+.3f}",
                        spread=f"{spread:.6f}",
                        depth=f"{depth:.1f}"
                    )
            
        except Exception as e:
            self.log_operator_error(f"Liquidity processing failed: {e}")

    def _simulate_liquidity_metrics(self, market_data: Dict[str, Any]) -> Tuple[float, float]:
        """Enhanced liquidity simulation based on market conditions"""
        
        # Base volatility
        if isinstance(market_data.get('volatility'), dict):
            # Multiple instruments - use average
            vol_values = list(market_data['volatility'].values())
            volatility = np.mean(vol_values) if vol_values else 0.01
        else:
            volatility = float(market_data.get('volatility', 0.01))
        
        # Session-based adjustments
        session = market_data.get('session', 'unknown')
        session_multipliers = {
            'asian': 0.7,      # Lower liquidity
            'european': 1.0,   # Normal liquidity
            'american': 1.2,   # Higher liquidity
            'unknown': 0.8
        }
        session_mult = session_multipliers.get(session, 0.8)
        
        # Regime-based adjustments
        regime = market_data.get('regime', 'unknown')
        regime_adjustments = {
            'trending': {'spread_mult': 0.9, 'depth_mult': 1.1},
            'volatile': {'spread_mult': 1.4, 'depth_mult': 0.7},
            'ranging': {'spread_mult': 0.8, 'depth_mult': 1.2},
            'unknown': {'spread_mult': 1.0, 'depth_mult': 1.0}
        }
        regime_adj = regime_adjustments.get(regime, regime_adjustments['unknown'])
        
        # Calculate spread (higher volatility = wider spread)
        base_spread = volatility * np.random.uniform(0.5, 1.5)
        spread = base_spread * regime_adj['spread_mult']
        
        # Calculate depth (higher volatility = lower depth, adjusted by session)
        base_depth = (1000 / (1 + volatility * 10)) * session_mult
        depth = base_depth * regime_adj['depth_mult'] * np.random.uniform(0.8, 1.2)
        
        return float(spread), float(depth)

    def _calculate_liquidity_score(self, spread: float, depth: float) -> float:
        """Enhanced liquidity score calculation"""
        
        # Lower spread and higher depth = better liquidity
        spread_score = np.exp(-spread * 1000)  # Penalize wide spreads heavily
        depth_score = np.log1p(depth) / 10      # Reward high depth logarithmically
        
        # Combined score with weighting
        combined_score = 0.6 * spread_score + 0.4 * depth_score
        
        return float(np.clip(combined_score, 0.0, 1.0))

    def _train_model_if_ready(self):
        """Enhanced model training with monitoring"""
        
        if self._trained or len(self.history) < 2 * self.seq_len:
            return
            
        try:
            X, y = self._prepare_training_data()
            if X.size == 0:
                return
                
            # Train with enhanced monitoring
            history = self._model.fit(
                X, y, 
                epochs=self.train_epochs,
                validation_split=0.2,
                verbose=0,
                batch_size=min(32, len(X)),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
                ]
            )
            
            self._trained = True
            
            # Store training metrics
            final_loss = history.history['loss'][-1]
            final_mae = history.history['mae'][-1] if 'mae' in history.history else 0
            
            self._training_history.append({
                'timestamp': np.datetime64('now').astype(str),
                'loss': final_loss,
                'mae': final_mae,
                'training_samples': len(X)
            })
            
            # Update performance metrics
            self._update_performance_metric('training_loss', final_loss)
            self._update_performance_metric('training_mae', final_mae)
            
            self.log_operator_info(
                f"LSTM model trained successfully",
                samples=len(X),
                epochs=self.train_epochs,
                final_loss=f"{final_loss:.6f}",
                architecture=f"{self.lstm_units}→{self.dense_units}"
            )
            
        except Exception as e:
            self.log_operator_error(f"Model training failed: {e}")
            self._update_health_status("DEGRADED", f"Training failed: {e}")

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced training data preparation"""
        
        X, y = [], []
        rows = list(self.history)
        
        for i in range(len(rows) - self.seq_len):
            sequence = rows[i:i + self.seq_len]
            target = rows[i + self.seq_len]
            
            # Validate sequence
            if all(len(row) == 2 for row in sequence) and len(target) == 2:
                X.append(sequence)
                y.append(target)
        
        if not X:
            return np.array([]), np.array([])
            
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Data validation
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            self.log_operator_warning("Non-finite values in training data")
            # Clean data
            finite_mask = np.all(np.isfinite(X.reshape(len(X), -1)), axis=1) & np.all(np.isfinite(y), axis=1)
            X = X[finite_mask]
            y = y[finite_mask]
        
        return X, y

    def predict_liquidity(self, steps: int = 4) -> Tuple[float, float]:
        """Enhanced liquidity prediction with confidence assessment"""
        
        if not self._trained or len(self.history) < self.seq_len:
            # Return current conditions if model not ready
            if self.history:
                return self.history[-1]
            return 0.0, 0.0
        
        try:
            # Prepare input sequence
            seq = np.array([list(self.history)[-self.seq_len:]], dtype=np.float32)
            
            # Validate input
            if not np.all(np.isfinite(seq)):
                self.log_operator_warning("Non-finite values in prediction input")
                return self.history[-1] if self.history else (0.0, 0.0)
            
            # Make prediction
            pred = self._model.predict(seq, verbose=0)[0]
            
            # Validate prediction
            if not np.all(np.isfinite(pred)):
                self.log_operator_warning("Model produced non-finite prediction")
                return self.history[-1] if self.history else (0.0, 0.0)
            
            # Update prediction tracking
            if len(self.history) >= 2:
                actual = self.history[-1]
                if len(self._training_history) > 0:
                    # Simple prediction accuracy tracking
                    error = np.mean(np.abs(np.array(pred) - np.array(actual)))
                    self._prediction_accuracy = 0.9 * self._prediction_accuracy + 0.1 * (1.0 / (1.0 + error))
                    self._update_performance_metric('prediction_accuracy', self._prediction_accuracy)
            
            return float(pred[0]), float(pred[1])
            
        except Exception as e:
            self.log_operator_error(f"Liquidity prediction failed: {e}")
            return self.history[-1] if self.history else (0.0, 0.0)

    def current_score(self) -> float:
        """Enhanced current liquidity score"""
        
        if not self.history:
            return 1.0
            
        spread, depth = self.history[-1]
        return self._calculate_liquidity_score(spread, depth)

    def step(self, **kwargs):
        """Backward compatibility wrapper"""
        self._step_impl(None, **kwargs)

    def _get_observation_impl(self) -> np.ndarray:
        """Enhanced observation components"""
        
        base_score = self.current_score()
        
        # Additional features
        volatility_estimate = self._market_conditions.get('volatility', 0.01)
        model_confidence = self._prediction_accuracy if self._trained else 0.0
        
        return np.array([
            base_score,
            volatility_estimate,
            model_confidence,
            float(self._trained)
        ], dtype=np.float32)

    def propose_action(self, obs: Any, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Enhanced action proposal with liquidity-based adjustments"""
        
        liq_score = self.current_score()
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        if liq_score < 0.2:
            # Very poor liquidity - avoid trading
            self.log_operator_warning(f"Very poor liquidity detected: {liq_score:.3f}")
            return action
        
        # Enhanced position sizing based on liquidity conditions
        if liq_score > 0.7:
            # Good liquidity - normal trading
            base_size = 0.4
            duration = 0.6
        elif liq_score > 0.4:
            # Moderate liquidity - reduced size
            base_size = 0.25
            duration = 0.4
        else:
            # Poor liquidity - very small positions
            base_size = 0.1
            duration = 0.2
        
        # Apply volatility adjustment
        vol_adj = 1.0 / (1.0 + self._market_conditions.get('volatility', 0.01) * 50)
        adjusted_size = base_size * liq_score * vol_adj
        
        # Fill action array
        for i in range(0, self.action_dim, 2):
            action[i] = adjusted_size
            if i + 1 < self.action_dim:
                action[i + 1] = duration
                
        return action

    def confidence(self, obs: Any, info_bus: Optional[InfoBus] = None) -> float:
        """Enhanced confidence based on liquidity and model performance"""
        
        liq_score = self.current_score()
        model_confidence = self._prediction_accuracy if self._trained else 0.5
        
        # Combine liquidity and model confidence
        combined_conf = 0.7 * liq_score + 0.3 * model_confidence
        
        return float(np.clip(combined_conf, 0.1, 1.0))

    # Neural network management methods
    def _get_weights_safe(self):
        """Safely get model weights"""
        try:
            return self._model.get_weights()
        except Exception as e:
            self.log_operator_warning(f"Failed to get weights: {e}")
            return []

    def _set_weights_safe(self, weights):
        """Safely set model weights"""
        try:
            self._model.set_weights([np.copy(w) for w in weights])
        except Exception as e:
            self.log_operator_warning(f"Failed to set weights: {e}")

    def _weights_like(self, other):
        """Check if weights are compatible"""
        try:
            ws1 = self._get_weights_safe()
            ws2 = other._get_weights_safe()
            return len(ws1) == len(ws2) and all(w1.shape == w2.shape for w1, w2 in zip(ws1, ws2))
        except:
            return False

    def clone_with_weights(self):
        """Create clone with same weights and genome"""
        return LiquidityHeatmapLayer(
            self.action_dim,
            debug=self.config.debug,
            genome=self.genome.copy(),
            weights=self._get_weights_safe(),
        )

    # Enhanced evolutionary methods
    def get_genome(self):
        """Get evolutionary genome"""
        return self.genome.copy()

    def set_genome(self, genome):
        """Set evolutionary genome with model rebuilding if needed"""
        old_units = self.lstm_units
        
        self.lstm_units = int(genome.get("lstm_units", self.lstm_units))
        self.seq_len = int(genome.get("seq_len", self.seq_len))
        self.dense_units = int(genome.get("dense_units", self.dense_units))
        self.train_epochs = int(genome.get("train_epochs", self.train_epochs))
        
        self.genome = {
            "lstm_units": self.lstm_units,
            "seq_len": self.seq_len,
            "dense_units": self.dense_units,
            "train_epochs": self.train_epochs,
        }
        
        # Rebuild model if architecture changed
        if old_units != self.lstm_units:
            try:
                self._model = self._build_enhanced_lstm()
                self._trained = False
                self.log_operator_info(f"Model rebuilt with {self.lstm_units} LSTM units")
            except Exception as e:
                self.log_operator_error(f"Model rebuild failed: {e}")

    def mutate(self, mutation_rate=0.2, weight_mutate_std=0.05):
        """Enhanced mutation with architecture and weight changes"""
        
        g = self.genome.copy()
        
        # Mutate architecture
        if np.random.rand() < mutation_rate:
            g["lstm_units"] = int(np.clip(self.lstm_units + np.random.randint(-8, 9), 8, 128))
        if np.random.rand() < mutation_rate:
            g["seq_len"] = int(np.clip(self.seq_len + np.random.randint(-2, 3), 5, 20))
        if np.random.rand() < mutation_rate:
            g["dense_units"] = int(np.clip(self.dense_units + np.random.randint(-1, 2), 1, 8))
        if np.random.rand() < mutation_rate:
            g["train_epochs"] = int(np.clip(self.train_epochs + np.random.randint(-2, 3), 2, 20))
            
        self.set_genome(g)

        # Mutate weights if trained
        if self._trained:
            try:
                weights = self._get_weights_safe()
                mutated = []
                for w in weights:
                    if np.issubdtype(w.dtype, np.floating):
                        noise = np.random.randn(*w.shape) * weight_mutate_std
                        mutated.append(w + noise)
                    else:
                        mutated.append(w)
                self._set_weights_safe(mutated)
            except Exception as e:
                self.log_operator_warning(f"Weight mutation failed: {e}")

    def crossover(self, other, weight_mix_prob=0.5):
        """Enhanced crossover with architecture and weight mixing"""
        
        # Mix architecture
        g1, g2 = self.genome, other.genome
        new_g = {k: np.random.choice([g1[k], g2[k]]) for k in g1}

        # Mix weights only if architectures match
        new_ws = None
        if self._weights_like(other):
            try:
                ws1, ws2 = self._get_weights_safe(), other._get_weights_safe()
                new_ws = []
                for w1, w2 in zip(ws1, ws2):
                    if np.issubdtype(w1.dtype, np.floating) and w1.shape == w2.shape:
                        mask = np.random.rand(*w1.shape) < weight_mix_prob
                        mixed = np.where(mask, w1, w2)
                        new_ws.append(mixed)
                    else:
                        new_ws.append(w1)
            except Exception as e:
                self.log_operator_warning(f"Weight crossover failed: {e}")
                new_ws = None

        return LiquidityHeatmapLayer(
            self.action_dim, 
            debug=self.config.debug, 
            genome=new_g, 
            weights=new_ws
        )

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check history is reasonable
            if len(self.history) > 250:  # Should not exceed maxlen significantly
                return False
                
            # Check model exists
            if not hasattr(self, '_model'):
                return False
                
            # Check training status is consistent
            if self._trained and len(self.history) < self.seq_len:
                return False
                
            # Check liquidity scores are valid
            if self._liquidity_scores and not all(0 <= score <= 1 for score in self._liquidity_scores):
                return False
                
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        liquidity_details = {
            'liquidity_info': {
                'current_score': self.current_score(),
                'history_size': len(self.history),
                'market_conditions': self._market_conditions.copy()
            },
            'model_info': {
                'trained': self._trained,
                'architecture': f"LSTM({self.lstm_units}) -> Dense({self.dense_units})",
                'prediction_accuracy': self._prediction_accuracy,
                'training_history_size': len(self._training_history)
            },
            'genome_config': self.genome.copy(),
            'action_dim': self.action_dim
        }
        
        if base_details:
            base_details.update(liquidity_details)
            return base_details
        
        return liquidity_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        return {
            'history': list(self.history),
            'trained': bool(self._trained),
            'genome': self.genome.copy(),
            'weights': self._get_weights_safe() if self._trained else None,
            'liquidity_scores': list(self._liquidity_scores),
            'prediction_accuracy': self._prediction_accuracy,
            'model_performance_score': self._model_performance_score,
            'training_history': self._training_history[-10:],  # Keep recent only
            'market_conditions': self._market_conditions.copy()
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        self.history = deque(module_state.get("history", []), maxlen=200)
        self._trained = bool(module_state.get("trained", False))
        self.set_genome(module_state.get("genome", self.genome))
        
        weights = module_state.get("weights")
        if weights is not None and self._trained:
            self._set_weights_safe(weights)
            
        self._liquidity_scores = deque(module_state.get("liquidity_scores", []), maxlen=100)
        self._prediction_accuracy = module_state.get("prediction_accuracy", 0.0)
        self._model_performance_score = module_state.get("model_performance_score", 100.0)
        self._training_history = module_state.get("training_history", [])
        self._market_conditions = module_state.get("market_conditions", 
            {"spread": 0.0, "depth": 0.0, "volatility": 0.0})

    def get_liquidity_analysis_report(self) -> str:
        """Generate operator-friendly liquidity analysis report"""
        
        current_score = self.current_score()
        
        # Model status
        model_status = "✅ Trained" if self._trained else "⚠️ Training"
        if self._trained:
            model_status += f" (Accuracy: {self._prediction_accuracy:.1%})"
        
        # Liquidity trend
        if len(self._liquidity_scores) >= 5:
            recent_avg = np.mean(list(self._liquidity_scores)[-5:])
            earlier_avg = np.mean(list(self._liquidity_scores)[-10:-5]) if len(self._liquidity_scores) >= 10 else recent_avg
            if recent_avg > earlier_avg + 0.05:
                trend = "📈 Improving"
            elif recent_avg < earlier_avg - 0.05:
                trend = "📉 Declining"
            else:
                trend = "➡️ Stable"
        else:
            trend = "📊 Insufficient data"
        
        return f"""
💧 LIQUIDITY HEATMAP ANALYSIS
═══════════════════════════════════════
📊 Current Score: {current_score:.3f} ({self._liquidity_description()})
📈 Trend: {trend}
🧠 Model Status: {model_status}

🔧 NEURAL NETWORK
• Architecture: LSTM({self.lstm_units}) → Dense({self.dense_units})
• Training Data: {len(self.history)} samples
• Sequence Length: {self.seq_len}
• Training Epochs: {self.train_epochs}

📊 MARKET CONDITIONS
• Spread: {self._market_conditions['spread']:.6f}
• Depth: {self._market_conditions['depth']:.1f}
• Volatility: {self._market_conditions['volatility']:.4f}

🎯 PERFORMANCE METRICS
• Prediction Accuracy: {self._prediction_accuracy:.1%}
• Score History: {len(self._liquidity_scores)} snapshots
• Training Sessions: {len(self._training_history)}
        """

    def _liquidity_description(self) -> str:
        """Human-readable liquidity description"""
        score = self.current_score()
        if score > 0.8:
            return "Excellent"
        elif score > 0.6:
            return "Good"
        elif score > 0.4:
            return "Moderate"
        elif score > 0.2:
            return "Poor"
        else:
            return "Very Poor"

    # Maintain backward compatibility
    def get_state(self):
        """Backward compatibility state method"""
        base_state = super().get_state()
        return base_state

    def set_state(self, state):
        """Backward compatibility state method"""
        super().set_state(state)