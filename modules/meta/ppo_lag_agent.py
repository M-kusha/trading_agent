# ─────────────────────────────────────────────────────────────
# File: modules/meta/ppo_lag_agent.py
# Enhanced with InfoBus integration & advanced market adaptation
# ─────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class MarketAwarePPONetwork(nn.Module):
    """Advanced PPO network with market-aware architecture"""
    
    def __init__(self, obs_size: int, act_size: int, lag_window: int, hidden_size: int = 128):
        super().__init__()
        
        self.obs_size = obs_size
        self.act_size = act_size
        self.lag_window = lag_window
        self.lag_features = 4  # returns, volatility, volume, spread
        
        # Calculate extended observation size
        self.extended_obs_size = obs_size + (lag_window * self.lag_features) + 6  # +6 for position features
        
        # Observation preprocessing
        self.obs_normalizer = nn.BatchNorm1d(self.extended_obs_size, momentum=0.1)
        
        # Market context encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(lag_window * self.lag_features, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Main feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.extended_obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Market-aware actor with attention
        self.actor_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_size),
            nn.Tanh()
        )
        
        # Dual critic heads for improved value estimation
        self.value_head_1 = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.value_head_2 = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Adaptive action std with market conditioning
        self.log_std_base = nn.Parameter(torch.log(torch.ones(act_size) * 0.1))
        self.std_conditioner = nn.Linear(hidden_size // 2, act_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Advanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor, market_lags: torch.Tensor):
        """Enhanced forward pass with market adaptation"""
        try:
            # Validate inputs
            if torch.any(torch.isnan(obs)):
                obs = torch.nan_to_num(obs)
            if torch.any(torch.isnan(market_lags)):
                market_lags = torch.nan_to_num(market_lags)
            
            batch_size = obs.shape[0]
            
            # Normalize observations
            if batch_size > 1:
                obs_norm = self.obs_normalizer(obs)
            else:
                obs_norm = obs  # Skip batch norm for single samples
            
            # Extract market context
            market_features = self.market_encoder(market_lags)
            
            # Main feature extraction
            main_features = self.feature_extractor(obs_norm)
            
            # Apply attention for market-aware processing
            main_features_expanded = main_features.unsqueeze(1)  # Add sequence dimension
            attended_features, _ = self.actor_attention(
                main_features_expanded, main_features_expanded, main_features_expanded
            )
            attended_features = attended_features.squeeze(1)  # Remove sequence dimension
            
            # Combine features
            combined_features = torch.cat([attended_features, market_features], dim=-1)
            
            # Actor output with market-conditioned std
            action_mean = self.actor_head(combined_features)
            
            # Adaptive standard deviation based on market conditions
            std_adjustment = torch.sigmoid(self.std_conditioner(market_features))
            action_std = torch.exp(self.log_std_base) * (0.5 + std_adjustment)
            action_std = torch.clamp(action_std, 0.01, 1.0)
            
            # Dual critic values
            value_1 = self.value_head_1(combined_features)
            value_2 = self.value_head_2(combined_features)
            value = torch.min(value_1, value_2)  # Conservative value estimate
            
            # Validate outputs
            if torch.any(torch.isnan(action_mean)):
                action_mean = torch.zeros_like(action_mean)
            if torch.any(torch.isnan(action_std)):
                action_std = torch.ones_like(action_std) * 0.1
            if torch.any(torch.isnan(value)):
                value = torch.zeros_like(value)
            
            return action_mean, action_std, value
            
        except Exception as e:
            # Safe fallback
            batch_size = obs.shape[0] if obs.ndim > 1 else 1
            return (
                torch.zeros(batch_size, self.act_size, device=obs.device),
                torch.ones(batch_size, self.act_size, device=obs.device) * 0.1,
                torch.zeros(batch_size, 1, device=obs.device)
            )


class PPOLagAgent(Module, AnalysisMixin, TradingMixin):
    """
    Enhanced PPO-Lag agent with InfoBus integration and advanced market adaptation.
    Incorporates lagged market features and intelligent position awareness for trading.
    """
    
    def __init__(self, obs_size: int, act_size: int = 2, hidden_size: int = 128, 
                 lr: float = 1e-4, lag_window: int = 20, 
                 adv_decay: float = 0.95, vol_scaling: bool = True,
                 position_aware: bool = True, device: str = "cpu", 
                 debug: bool = True, **kwargs):
        
        # Enhanced configuration
        config = ModuleConfig(
            debug=debug,
            max_history=1000,
            health_check_interval=180,
            performance_window=200,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.device = torch.device(device)
        self.obs_size = obs_size
        self.act_size = act_size
        self.lag_window = lag_window
        self.adv_decay = adv_decay
        self.vol_scaling = vol_scaling
        self.position_aware = position_aware
        self.learning_rate = lr
        
        # Enhanced network architecture
        self.network = MarketAwarePPONetwork(
            obs_size, act_size, lag_window, hidden_size
        ).to(self.device)
        
        # Separate optimizers for different components
        self.actor_optimizer = optim.Adam(
            list(self.network.feature_extractor.parameters()) +
            list(self.network.actor_attention.parameters()) +
            list(self.network.actor_head.parameters()) +
            [self.network.log_std_base] +
            list(self.network.std_conditioner.parameters()),
            lr=lr, eps=1e-5
        )
        
        self.critic_optimizer = optim.Adam(
            list(self.network.market_encoder.parameters()) +
            list(self.network.value_head_1.parameters()) +
            list(self.network.value_head_2.parameters()),
            lr=lr * 2, eps=1e-5  # Higher LR for critics
        )
        
        # Enhanced PPO hyperparameters
        self.clip_eps = 0.1
        self.value_coeff = 0.5
        self.entropy_coeff = 0.001
        self.gae_lambda = 0.95
        self.gamma = 0.99
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.batch_size = 64
        self.target_kl = 0.01
        
        # Lag buffers for market features
        self.price_buffer = deque(maxlen=lag_window)
        self.volume_buffer = deque(maxlen=lag_window)
        self.spread_buffer = deque(maxlen=lag_window)
        self.volatility_buffer = deque(maxlen=lag_window)
        
        # Market adaptation state
        self.market_regime_adaptation = {
            'trending': {'std_multiplier': 1.2, 'clip_adjustment': 0.0},
            'volatile': {'std_multiplier': 0.8, 'clip_adjustment': 0.05},
            'ranging': {'std_multiplier': 1.0, 'clip_adjustment': -0.02},
            'unknown': {'std_multiplier': 1.0, 'clip_adjustment': 0.0}
        }
        
        # Experience buffer
        self.buffer = {
            'observations': [],
            'market_features': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'advantages': [],
            'returns': [],
            'dones': []
        }
        
        # Advanced tracking
        self.episode_rewards = deque(maxlen=200)
        self.episode_lengths = deque(maxlen=200)
        self.market_performance = defaultdict(lambda: {'rewards': deque(maxlen=100), 'count': 0})
        self.volatility_performance = defaultdict(lambda: {'rewards': deque(maxlen=100), 'count': 0})
        
        # Position and risk tracking
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.position_history = deque(maxlen=1000)
        self.risk_metrics = {
            'max_position': 0.0,
            'avg_position': 0.0,
            'position_volatility': 0.0,
            'risk_adjusted_return': 0.0
        }
        
        # Training statistics
        self.training_stats = {
            'total_updates': 0,
            'episodes_completed': 0,
            'actor_loss_trend': 0.0,
            'critic_loss_trend': 0.0,
            'kl_divergence': 0.0,
            'explained_variance': 0.0,
            'policy_entropy': 0.0,
            'advantage_mean': 0.0,
            'advantage_std': 0.0
        }
        
        # Adaptive parameters
        self.running_adv_std = 1.0
        self.adaptive_clip_eps = self.clip_eps
        self.adaptive_lr_factor = 1.0
        
        # Enhanced logging with rotation
        self.logger = RotatingLogger(
            "PPOLagAgent",
            "logs/strategy/ppo/ppo_lag_agent.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("PPOLagAgent")
        
        self.log_operator_info(
            "🎯 Enhanced PPO-Lag Agent initialized",
            obs_size=obs_size,
            extended_obs_size=self.network.extended_obs_size,
            act_size=act_size,
            lag_window=lag_window,
            position_aware=position_aware,
            vol_scaling=vol_scaling,
            network_params=sum(p.numel() for p in self.network.parameters())
        )
    
    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_trading_state()
        
        # Clear experience buffer
        for key in self.buffer:
            self.buffer[key].clear()
        
        # Clear market buffers
        self.price_buffer.clear()
        self.volume_buffer.clear()
        self.spread_buffer.clear()
        self.volatility_buffer.clear()
        
        # Reset tracking
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.market_performance.clear()
        self.volatility_performance.clear()
        
        # Reset position state
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.position_history.clear()
        
        # Reset training statistics
        self.training_stats = {
            'total_updates': 0,
            'episodes_completed': 0,
            'actor_loss_trend': 0.0,
            'critic_loss_trend': 0.0,
            'kl_divergence': 0.0,
            'explained_variance': 0.0,
            'policy_entropy': 0.0,
            'advantage_mean': 0.0,
            'advantage_std': 0.0
        }
        
        # Reset adaptive parameters
        self.running_adv_std = 1.0
        self.adaptive_clip_eps = self.clip_eps
        self.adaptive_lr_factor = 1.0
        
        self.log_operator_info("🔄 PPO-Lag Agent reset - all state cleared")
    
    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and market adaptation"""
        
        if info_bus:
            # Extract and process market context
            context = extract_standard_context(info_bus)
            self._update_market_buffers_from_info_bus(info_bus, context)
            
            # Adapt to current market conditions
            self._adapt_to_market_conditions(context)
            
            # Update performance tracking
            self._update_performance_from_info_bus(info_bus, context)
            
            # Publish agent status
            self._publish_agent_status(info_bus)
    
    def _update_market_buffers_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Update market buffers from InfoBus data"""
        
        try:
            # Extract market data
            prices = info_bus.get('prices', {})
            market_data = info_bus.get('market_data', {})
            
            # Update with current market conditions
            current_price = 0.0
            current_volume = 0.0
            current_spread = 0.0
            
            # Get price from first available instrument
            if prices:
                current_price = list(prices.values())[0]
            
            # Get market metrics
            if market_data:
                current_volume = market_data.get('volume', 0.0)
                current_spread = market_data.get('spread', 0.0)
            
            # Calculate volatility from context
            vol_level = context.get('volatility_level', 'medium')
            vol_mapping = {'low': 0.5, 'medium': 1.0, 'high': 2.0, 'extreme': 4.0}
            current_volatility = vol_mapping.get(vol_level, 1.0)
            
            # Update buffers
            self.update_market_buffers(current_price, current_volume, current_spread, current_volatility)
            
        except Exception as e:
            self.log_operator_error(f"Market buffer update failed: {e}")
    
    def update_market_buffers(self, price: float, volume: float, spread: float, volatility: float):
        """Enhanced market buffer update with validation"""
        
        try:
            # Validate inputs
            if np.isnan(price) or price <= 0:
                price = self.price_buffer[-1] if self.price_buffer else 1.0
            if np.isnan(volume) or volume < 0:
                volume = 0.0
            if np.isnan(spread) or spread < 0:
                spread = 0.0
            if np.isnan(volatility) or volatility <= 0:
                volatility = 1.0
            
            # Calculate returns
            if len(self.price_buffer) > 0:
                last_price = self.price_buffer[-1]
                if last_price > 0:
                    price_return = (price - last_price) / last_price
                else:
                    price_return = 0.0
            else:
                price_return = 0.0
            
            # Validate and clamp return
            if np.isnan(price_return) or abs(price_return) > 0.1:  # Cap at 10%
                price_return = 0.0
            
            # Update buffers
            self.price_buffer.append(price_return)
            self.volume_buffer.append(volume)
            self.spread_buffer.append(spread)
            self.volatility_buffer.append(volatility)
            
            self.log_operator_debug(
                f"Market buffers updated",
                price_return=f"{price_return:.5f}",
                volume=f"{volume:.3f}",
                spread=f"{spread:.5f}",
                volatility=f"{volatility:.3f}"
            )
            
        except Exception as e:
            self.log_operator_error(f"Buffer update failed: {e}")
    
    def get_lag_features(self) -> np.ndarray:
        """Extract enhanced lagged features"""
        
        try:
            # Get buffer contents with padding
            price_lags = list(self.price_buffer) + [0] * (self.lag_window - len(self.price_buffer))
            volume_lags = list(self.volume_buffer) + [0] * (self.lag_window - len(self.volume_buffer))
            spread_lags = list(self.spread_buffer) + [0] * (self.lag_window - len(self.spread_buffer))
            vol_lags = list(self.volatility_buffer) + [1] * (self.lag_window - len(self.volatility_buffer))
            
            # Interleave features for better temporal representation
            features = []
            for i in range(self.lag_window):
                features.extend([
                    price_lags[i],
                    vol_lags[i], 
                    volume_lags[i],
                    spread_lags[i]
                ])
            
            result = np.array(features, dtype=np.float32)
            
            # Validate and normalize features
            if np.any(np.isnan(result)):
                self.log_operator_error(f"NaN in lag features: {result}")
                result = np.nan_to_num(result)
            
            # Apply feature scaling
            result = np.clip(result, -5.0, 5.0)
            
            return result
            
        except Exception as e:
            self.log_operator_error(f"Lag feature extraction failed: {e}")
            return np.zeros(self.lag_window * 4, dtype=np.float32)
    
    def _adapt_to_market_conditions(self, context: Dict[str, Any]):
        """Advanced market condition adaptation"""
        
        regime = context.get('regime', 'unknown')
        vol_level = context.get('volatility_level', 'medium')
        
        # Get adaptation parameters
        adaptation = self.market_regime_adaptation.get(regime, self.market_regime_adaptation['unknown'])
        
        # Adapt clipping epsilon
        base_clip = self.clip_eps
        self.adaptive_clip_eps = base_clip + adaptation['clip_adjustment']
        self.adaptive_clip_eps = np.clip(self.adaptive_clip_eps, 0.05, 0.3)
        
        # Adapt learning rate
        if regime == 'volatile':
            self.adaptive_lr_factor = 0.8  # Slower learning in volatile markets
        elif regime == 'trending':
            self.adaptive_lr_factor = 1.2  # Faster learning in trending markets
        else:
            self.adaptive_lr_factor = 1.0
        
        # Update optimizer learning rates
        for optimizer in [self.actor_optimizer, self.critic_optimizer]:
            for param_group in optimizer.param_groups:
                base_lr = self.learning_rate if optimizer == self.actor_optimizer else self.learning_rate * 2
                param_group['lr'] = base_lr * self.adaptive_lr_factor
        
        # Adapt entropy coefficient based on volatility
        if vol_level == 'extreme':
            self.entropy_coeff = min(0.005, self.entropy_coeff * 0.9)  # Reduce exploration
        elif vol_level == 'low':
            self.entropy_coeff = min(0.02, self.entropy_coeff * 1.05)  # Increase exploration
        
        self.log_operator_debug(
            f"Market adaptation applied",
            regime=regime,
            vol_level=vol_level,
            adaptive_clip=f"{self.adaptive_clip_eps:.3f}",
            lr_factor=f"{self.adaptive_lr_factor:.3f}",
            entropy_coeff=f"{self.entropy_coeff:.5f}"
        )
    
    def record_step(self, obs_vec: np.ndarray, reward: float, 
                   price: float = 0, volume: float = 0, 
                   spread: float = 0, volatility: float = 1,
                   position: float = 0, unrealized_pnl: float = 0,
                   done: bool = False):
        """Enhanced step recording with comprehensive market data"""
        
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                self.log_operator_error(f"NaN in observation vector: {obs_vec}")
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                self.log_operator_error("NaN reward, setting to 0")
                reward = 0.0
            if np.isnan(position):
                position = 0.0
            if np.isnan(unrealized_pnl):
                unrealized_pnl = 0.0
            
            # Update market buffers
            self.update_market_buffers(price, volume, spread, volatility)
            
            # Get lag features
            lag_features = self.get_lag_features()
            
            # Build extended observation
            if self.position_aware:
                # Enhanced position features
                position_features = np.array([
                    position,                           # Current position
                    unrealized_pnl,                    # Unrealized PnL
                    position * volatility,             # Risk-adjusted position
                    np.sign(position) * min(abs(position), 1.0),  # Position direction
                    abs(position) / max(abs(position), 1.0),      # Position size ratio
                    float(len(self.position_history)) / 1000.0    # Experience level
                ], dtype=np.float32)
                
                extended_obs = np.concatenate([obs_vec, lag_features, position_features])
            else:
                extended_obs = np.concatenate([obs_vec, lag_features])
            
            # Validate extended observation
            if np.any(np.isnan(extended_obs)):
                self.log_operator_error(f"NaN in extended observation: {extended_obs}")
                extended_obs = np.nan_to_num(extended_obs)
            
            # Pad or truncate to expected size
            if len(extended_obs) < self.network.extended_obs_size:
                padding = np.zeros(self.network.extended_obs_size - len(extended_obs), dtype=np.float32)
                extended_obs = np.concatenate([extended_obs, padding])
            elif len(extended_obs) > self.network.extended_obs_size:
                extended_obs = extended_obs[:self.network.extended_obs_size]
            
            # Convert to tensors
            obs_tensor = torch.as_tensor(extended_obs, dtype=torch.float32, device=self.device)
            market_tensor = torch.as_tensor(lag_features, dtype=torch.float32, device=self.device)
            
            # Get network outputs
            with torch.no_grad():
                action_mean, action_std, value = self.network(obs_tensor.unsqueeze(0), market_tensor.unsqueeze(0))
                
                # Apply volatility scaling if enabled
                if self.vol_scaling and volatility > 0:
                    vol_adjustment = np.sqrt(1.0 / volatility)
                    action_std = action_std * vol_adjustment
                
                # Apply position-aware action scaling
                if self.position_aware and abs(position) > 0.8:
                    position_penalty = 1.0 - abs(position) * 0.2
                    action_std = action_std * position_penalty
                
                # Clamp action std
                action_std = torch.clamp(action_std, 0.01, 0.5)
                
                # Sample action
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # Validate outputs
                if torch.any(torch.isnan(action)):
                    self.log_operator_error("NaN in sampled action")
                    action = torch.zeros_like(action)
                if torch.any(torch.isnan(log_prob)):
                    self.log_operator_error("NaN in log probability")
                    log_prob = torch.zeros_like(log_prob)
                if torch.any(torch.isnan(value)):
                    self.log_operator_error("NaN in value estimate")
                    value = torch.zeros_like(value)
            
            # Store in buffer
            self.buffer['observations'].append(obs_tensor)
            self.buffer['market_features'].append(market_tensor)
            self.buffer['actions'].append(action.squeeze(0))
            self.buffer['log_probs'].append(log_prob.squeeze(0))
            self.buffer['values'].append(value.squeeze(0))
            self.buffer['rewards'].append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            self.buffer['dones'].append(torch.tensor(done, dtype=torch.bool, device=self.device))
            
            # Update position tracking
            self.position = position
            self.unrealized_pnl = unrealized_pnl
            self.position_history.append({
                'position': position,
                'unrealized_pnl': unrealized_pnl,
                'timestamp': datetime.datetime.now(),
                'volatility': volatility
            })
            
            # Update risk metrics
            self._update_risk_metrics()
            
            # Update trading metrics
            self._update_trading_metrics({'pnl': reward})
            
            self.log_operator_debug(
                f"Step recorded",
                reward=f"{reward:.3f}",
                position=f"{position:.3f}",
                unrealized_pnl=f"{unrealized_pnl:.3f}",
                action_mean=f"{action_mean.mean().item():.3f}",
                value=f"{value.item():.3f}",
                buffer_size=len(self.buffer['rewards'])
            )
            
        except Exception as e:
            self.log_operator_error(f"Step recording failed: {e}")
    
    def _update_risk_metrics(self):
        """Update position and risk metrics"""
        
        if len(self.position_history) >= 10:
            recent_positions = [p['position'] for p in list(self.position_history)[-50:]]
            recent_pnls = [p['unrealized_pnl'] for p in list(self.position_history)[-50:]]
            
            self.risk_metrics['max_position'] = max(abs(p) for p in recent_positions)
            self.risk_metrics['avg_position'] = np.mean(np.abs(recent_positions))
            self.risk_metrics['position_volatility'] = np.std(recent_positions)
            
            if self.risk_metrics['position_volatility'] > 0:
                avg_return = np.mean(recent_pnls)
                self.risk_metrics['risk_adjusted_return'] = avg_return / self.risk_metrics['position_volatility']
            else:
                self.risk_metrics['risk_adjusted_return'] = 0.0
    
    def end_episode(self, gamma: float = None, final_value: float = 0.0):
        """Enhanced episode ending with advanced PPO updates"""
        
        try:
            if len(self.buffer['rewards']) < 10:
                self.log_operator_warning(f"Episode too short ({len(self.buffer['rewards'])} steps), skipping")
                self._clear_buffer()
                return
            
            gamma = gamma if gamma is not None else self.gamma
            
            self.log_operator_info(
                f"Episode ending",
                steps=len(self.buffer['rewards']),
                total_reward=f"{sum(r.item() for r in self.buffer['rewards']):.3f}",
                final_position=f"{self.position:.3f}"
            )
            
            # Calculate GAE advantages and returns
            self._compute_gae_advantages(gamma, final_value)
            
            # Perform advanced PPO updates
            update_stats = self._perform_ppo_updates()
            
            # Update training statistics
            self._update_training_statistics(update_stats)
            
            # Update market performance tracking
            self._update_market_performance_tracking()
            
            # Clear buffer
            self._clear_buffer()
            
        except Exception as e:
            self.log_operator_error(f"Episode ending failed: {e}")
            self._clear_buffer()
    
    def _compute_gae_advantages(self, gamma: float, final_value: float):
        """Enhanced GAE computation with validation"""
        
        try:
            rewards = torch.stack(self.buffer['rewards'])
            values = torch.stack(self.buffer['values'])
            dones = torch.stack(self.buffer['dones'])
            
            # Validate tensors
            if torch.any(torch.isnan(rewards)):
                self.log_operator_error("NaN in rewards")
                rewards = torch.nan_to_num(rewards)
            if torch.any(torch.isnan(values)):
                self.log_operator_error("NaN in values")
                values = torch.nan_to_num(values)
            
            # GAE computation
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            
            next_value = final_value
            next_advantage = 0
            
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    next_value = 0
                    next_advantage = 0
                
                # TD error
                td_error = rewards[t] + gamma * next_value - values[t]
                
                # GAE advantage
                advantages[t] = td_error + gamma * self.gae_lambda * next_advantage
                
                # Return
                returns[t] = rewards[t] + gamma * next_value
                
                next_value = values[t]
                next_advantage = advantages[t]
            
            # Adaptive advantage normalization
            if advantages.std() > 1e-6:
                # Update running std
                self.running_adv_std = self.adv_decay * self.running_adv_std + (1 - self.adv_decay) * advantages.std().item()
                
                # Normalize advantages
                advantages = advantages / (self.running_adv_std + 1e-8)
            
            # Store in buffer
            self.buffer['advantages'] = [advantages[i] for i in range(len(advantages))]
            self.buffer['returns'] = [returns[i] for i in range(len(returns))]
            
            # Track episode statistics
            episode_reward = float(rewards.sum())
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(rewards))
            
            # Update performance metrics
            self._update_performance_metric('episode_reward', episode_reward)
            self._update_performance_metric('advantage_mean', advantages.mean().item())
            self._update_performance_metric('advantage_std', advantages.std().item())
            
        except Exception as e:
            self.log_operator_error(f"GAE computation failed: {e}")
            # Fallback computation
            self._fallback_advantage_computation(gamma, final_value)
    
    def _fallback_advantage_computation(self, gamma: float, final_value: float):
        """Fallback advantage computation"""
        
        returns = []
        running_return = final_value
        for reward in reversed(self.buffer['rewards']):
            running_return = reward.item() + gamma * running_return
            returns.insert(0, running_return)
        
        values = [v.item() for v in self.buffer['values']]
        advantages = [ret - val for ret, val in zip(returns, values)]
        
        # Normalize
        if len(advantages) > 1:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            if adv_std > 1e-6:
                advantages = [(a - adv_mean) / adv_std for a in advantages]
        
        self.buffer['returns'] = [torch.tensor(r, device=self.device) for r in returns]
        self.buffer['advantages'] = [torch.tensor(a, device=self.device) for a in advantages]
    
    def _perform_ppo_updates(self) -> Dict[str, float]:
        """Advanced PPO updates with market awareness"""
        
        try:
            # Prepare data
            observations = torch.stack(self.buffer['observations'])
            market_features = torch.stack(self.buffer['market_features'])
            actions = torch.stack(self.buffer['actions'])
            old_log_probs = torch.stack(self.buffer['log_probs'])
            returns = torch.stack(self.buffer['returns'])
            advantages = torch.stack(self.buffer['advantages'])
            
            # Validate data
            for name, tensor in [('observations', observations), ('actions', actions), 
                               ('old_log_probs', old_log_probs), ('returns', returns), 
                               ('advantages', advantages), ('market_features', market_features)]:
                if torch.any(torch.isnan(tensor)):
                    self.log_operator_error(f"NaN in {name}")
                    tensor = torch.nan_to_num(tensor)
            
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0
            total_kl_div = 0
            update_count = 0
            
            # Multiple epochs
            for epoch in range(self.ppo_epochs):
                # Shuffle data
                indices = torch.randperm(len(observations), device=self.device)
                
                for start in range(0, len(observations), self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    # Get batch
                    batch_obs = observations[batch_indices]
                    batch_market = market_features[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    
                    # Forward pass
                    action_mean, action_std, values = self.network(batch_obs, batch_market)
                    
                    # Create distribution
                    dist = torch.distributions.Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)
                    
                    # Calculate KL divergence for early stopping
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    
                    # Early stopping if KL too large
                    if kl_div > self.target_kl * 2:
                        self.log_operator_warning(f"Early stopping due to large KL: {kl_div:.6f}")
                        break
                    
                    # Policy loss
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    ratio = torch.clamp(ratio, 0.1, 10.0)  # Stability clamp
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.adaptive_clip_eps, 1 + self.adaptive_clip_eps) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss with clipping
                    value_pred_clipped = values.squeeze(-1)
                    value_loss = F.mse_loss(value_pred_clipped, batch_returns)
                    
                    # Entropy loss
                    entropy_loss = -entropy.mean()
                    
                    # Total losses
                    total_actor_loss_batch = actor_loss + self.entropy_coeff * entropy_loss
                    total_critic_loss_batch = self.value_coeff * value_loss
                    
                    # Validate losses
                    if torch.isnan(total_actor_loss_batch) or torch.isnan(total_critic_loss_batch):
                        self.log_operator_error("NaN loss detected, skipping batch")
                        continue
                    
                    # Actor update
                    self.actor_optimizer.zero_grad()
                    total_actor_loss_batch.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.network.feature_extractor.parameters()) +
                        list(self.network.actor_attention.parameters()) +
                        list(self.network.actor_head.parameters()) +
                        [self.network.log_std_base] +
                        list(self.network.std_conditioner.parameters()),
                        self.max_grad_norm
                    )
                    self.actor_optimizer.step()
                    
                    # Critic update
                    self.critic_optimizer.zero_grad()
                    total_critic_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.network.market_encoder.parameters()) +
                        list(self.network.value_head_1.parameters()) +
                        list(self.network.value_head_2.parameters()),
                        self.max_grad_norm
                    )
                    self.critic_optimizer.step()
                    
                    # Accumulate statistics
                    total_actor_loss += actor_loss.item()
                    total_critic_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    total_kl_div += kl_div.item()
                    update_count += 1
            
            # Calculate averages
            if update_count > 0:
                update_stats = {
                    'actor_loss': total_actor_loss / update_count,
                    'critic_loss': total_critic_loss / update_count,
                    'entropy': total_entropy / update_count,
                    'kl_divergence': total_kl_div / update_count,
                    'updates_performed': update_count
                }
            else:
                update_stats = {
                    'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0,
                    'kl_divergence': 0.0, 'updates_performed': 0
                }
            
            return update_stats
            
        except Exception as e:
            self.log_operator_error(f"PPO update failed: {e}")
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0, 'kl_divergence': 0.0, 'updates_performed': 0}
    
    def _update_training_statistics(self, update_stats: Dict[str, float]):
        """Update comprehensive training statistics"""
        
        self.training_stats['total_updates'] += 1
        self.training_stats['episodes_completed'] += 1
        
        # Update loss trends
        if self.training_stats['total_updates'] > 1:
            self.training_stats['actor_loss_trend'] = (
                0.9 * self.training_stats['actor_loss_trend'] + 
                0.1 * update_stats['actor_loss']
            )
            self.training_stats['critic_loss_trend'] = (
                0.9 * self.training_stats['critic_loss_trend'] + 
                0.1 * update_stats['critic_loss']
            )
        else:
            self.training_stats['actor_loss_trend'] = update_stats['actor_loss']
            self.training_stats['critic_loss_trend'] = update_stats['critic_loss']
        
        # Update other metrics
        self.training_stats['kl_divergence'] = update_stats['kl_divergence']
        self.training_stats['policy_entropy'] = update_stats['entropy']
        
        # Calculate explained variance
        if len(self.buffer['returns']) > 0 and len(self.buffer['values']) > 0:
            returns_np = np.array([r.item() for r in self.buffer['returns']])
            values_np = np.array([v.item() for v in self.buffer['values']])
            
            if np.var(returns_np) > 1e-6:
                explained_var = 1 - np.var(returns_np - values_np) / np.var(returns_np)
                self.training_stats['explained_variance'] = max(0, explained_var)
        
        # Log training progress
        if self.training_stats['episodes_completed'] % 10 == 0:
            self.log_operator_info(
                f"📊 Training progress update",
                episodes=self.training_stats['episodes_completed'],
                avg_reward=f"{np.mean(list(self.episode_rewards)[-10:]) if self.episode_rewards else 0:.3f}",
                actor_loss=f"{update_stats['actor_loss']:.6f}",
                critic_loss=f"{update_stats['critic_loss']:.6f}",
                kl_div=f"{update_stats['kl_divergence']:.6f}",
                entropy=f"{update_stats['entropy']:.6f}",
                explained_var=f"{self.training_stats['explained_variance']:.3f}"
            )
    
    def _update_market_performance_tracking(self):
        """Update market-specific performance tracking"""
        
        if len(self.market_context_history) > 0 and len(self.episode_rewards) > 0:
            latest_context = self.market_context_history[-1]
            latest_reward = self.episode_rewards[-1]
            
            # Track performance by regime
            regime = latest_context.get('regime', 'unknown')
            self.market_performance[regime]['rewards'].append(latest_reward)
            self.market_performance[regime]['count'] += 1
            
            # Track performance by volatility
            vol_level = latest_context.get('volatility_level', 'medium')
            self.volatility_performance[vol_level]['rewards'].append(latest_reward)
            self.volatility_performance[vol_level]['count'] += 1
    
    def _clear_buffer(self):
        """Clear experience buffer"""
        for key in self.buffer:
            self.buffer[key].clear()
    
    def select_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Enhanced action selection with market awareness"""
        
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                self.log_operator_error("NaN in observation tensor")
                obs_tensor = torch.nan_to_num(obs_tensor)
            
            # Pad observation if needed
            if obs_tensor.shape[-1] < self.network.extended_obs_size:
                padding = torch.zeros(
                    (*obs_tensor.shape[:-1], self.network.extended_obs_size - obs_tensor.shape[-1]),
                    device=obs_tensor.device, dtype=obs_tensor.dtype
                )
                obs_tensor = torch.cat([obs_tensor, padding], dim=-1)
            elif obs_tensor.shape[-1] > self.network.extended_obs_size:
                obs_tensor = obs_tensor[..., :self.network.extended_obs_size]
            
            # Extract market features from observation
            lag_features_size = self.lag_window * 4
            market_features = obs_tensor[..., -lag_features_size:]
            
            with torch.no_grad():
                action_mean, action_std, _ = self.network(obs_tensor, market_features)
                
                # Sample action
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                
                # Validate action
                if torch.any(torch.isnan(action)):
                    self.log_operator_error("NaN in selected action")
                    action = torch.zeros_like(action)
                
                # Apply reasonable bounds
                action = torch.clamp(action, -2.0, 2.0)
                
                return action
            
        except Exception as e:
            self.log_operator_error(f"Action selection failed: {e}")
            batch_size = obs_tensor.shape[0] if obs_tensor.ndim > 1 else 1
            return torch.zeros(batch_size, self.act_size, device=self.device)
    
    # ═══════════════════════════════════════════════════════════════════
    # OBSERVATION AND STATUS METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation with market and position awareness"""
        
        try:
            # Position metrics
            position_norm = np.tanh(self.position)  # Normalized position
            unrealized_pnl_norm = np.tanh(self.unrealized_pnl / 100.0)  # Normalized PnL
            
            # Risk metrics
            max_pos_norm = np.tanh(self.risk_metrics['max_position'])
            risk_adj_return = np.tanh(self.risk_metrics['risk_adjusted_return'])
            
            # Performance metrics
            if len(self.episode_rewards) > 0:
                avg_reward = float(np.mean(list(self.episode_rewards)[-10:]))
                reward_trend = 0.0
                if len(self.episode_rewards) >= 5:
                    recent_rewards = list(self.episode_rewards)[-5:]
                    reward_trend = float(np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0])
            else:
                avg_reward = 0.0
                reward_trend = 0.0
            
            # Training metrics
            episodes_norm = min(1.0, self.training_stats['episodes_completed'] / 1000.0)
            kl_div_norm = np.tanh(self.training_stats['kl_divergence'] * 100)
            entropy_norm = np.tanh(self.training_stats['policy_entropy'] * 10)
            
            # Market adaptation metrics
            adaptive_clip_norm = self.adaptive_clip_eps / 0.3
            adaptive_lr_norm = self.adaptive_lr_factor
            
            # Buffer and experience metrics
            buffer_fullness = len(self.buffer['rewards']) / 200.0
            experience_level = min(1.0, len(self.position_history) / 1000.0)
            
            observation = np.array([
                position_norm,              # Current position
                unrealized_pnl_norm,        # Unrealized PnL  
                max_pos_norm,              # Maximum position taken
                risk_adj_return,           # Risk-adjusted returns
                avg_reward / 100.0,        # Average episode reward
                reward_trend / 10.0,       # Reward trend
                episodes_norm,             # Training progress
                kl_div_norm,              # KL divergence
                entropy_norm,             # Policy entropy
                adaptive_clip_norm,       # Adaptive clipping
                adaptive_lr_norm,         # Learning rate adaptation
                buffer_fullness,          # Experience buffer state
                experience_level,         # Overall experience
                self.training_stats['explained_variance']  # Value function quality
            ], dtype=np.float32)
            
            # Validate observation
            if np.any(np.isnan(observation)):
                self.log_operator_error(f"NaN in observation: {observation}")
                observation = np.nan_to_num(observation)
            
            # Clamp to reasonable ranges
            observation = np.clip(observation, -3.0, 3.0)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(14, dtype=np.float32)
    
    def _publish_agent_status(self, info_bus: InfoBus):
        """Publish comprehensive agent status to InfoBus"""
        
        agent_status = {
            'agent_type': 'ppo-lag',
            'training_stats': self.training_stats.copy(),
            'position_metrics': {
                'current_position': self.position,
                'unrealized_pnl': self.unrealized_pnl,
                'risk_metrics': self.risk_metrics.copy()
            },
            'performance_metrics': {
                'episodes_completed': len(self.episode_rewards),
                'avg_episode_reward': float(np.mean(list(self.episode_rewards)[-10:])) if self.episode_rewards else 0.0,
                'best_episode_reward': float(max(self.episode_rewards)) if self.episode_rewards else 0.0,
                'buffer_size': len(self.buffer['rewards']),
                'running_adv_std': self.running_adv_std
            },
            'market_adaptation': {
                'adaptive_clip_eps': self.adaptive_clip_eps,
                'adaptive_lr_factor': self.adaptive_lr_factor,
                'entropy_coeff': self.entropy_coeff
            },
            'market_performance': {
                regime: {
                    'avg_reward': float(np.mean(list(data['rewards'])[-10:])) if len(data['rewards']) >= 10 else 0.0,
                    'count': data['count']
                }
                for regime, data in self.market_performance.items()
            },
            'volatility_performance': {
                vol_level: {
                    'avg_reward': float(np.mean(list(data['rewards'])[-10:])) if len(data['rewards']) >= 10 else 0.0,
                    'count': data['count']
                }
                for vol_level, data in self.volatility_performance.items()
            },
            'lag_features': {
                'lag_window': self.lag_window,
                'price_buffer_size': len(self.price_buffer),
                'vol_scaling': self.vol_scaling,
                'position_aware': self.position_aware
            }
        }
        
        InfoBusUpdater.update_agent_status(info_bus, 'ppo-lag', agent_status)
    
    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT AND REPORTING
    # ═══════════════════════════════════════════════════════════════════
    
    def get_agent_report(self) -> str:
        """Generate comprehensive agent report"""
        
        # Performance statistics
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(list(self.episode_rewards)[-10:])
            best_reward = max(self.episode_rewards)
            reward_std = np.std(list(self.episode_rewards)[-10:])
        else:
            avg_reward = 0.0
            best_reward = 0.0
            reward_std = 0.0
        
        # Market performance summary
        market_summary = {}
        for regime, data in self.market_performance.items():
            if len(data['rewards']) >= 5:
                market_summary[regime] = {
                    'avg': np.mean(list(data['rewards'])[-10:]),
                    'count': data['count']
                }
        
        return f"""
🎯 PPO-LAG AGENT STATUS
═══════════════════════════════════════
🧠 Network: {sum(p.numel() for p in self.network.parameters()):,} parameters
📊 Episodes: {self.training_stats['episodes_completed']}
🔄 Updates: {self.training_stats['total_updates']}

💰 PERFORMANCE
• Avg Reward (10ep): {avg_reward:.3f}
• Best Reward: {best_reward:.3f}
• Reward Std: {reward_std:.3f}
• Explained Variance: {self.training_stats['explained_variance']:.3f}

📈 TRAINING METRICS
• Actor Loss Trend: {self.training_stats['actor_loss_trend']:.6f}
• Critic Loss Trend: {self.training_stats['critic_loss_trend']:.6f}
• KL Divergence: {self.training_stats['kl_divergence']:.6f}
• Policy Entropy: {self.training_stats['policy_entropy']:.6f}

💼 POSITION METRICS
• Current Position: {self.position:.3f}
• Unrealized PnL: €{self.unrealized_pnl:.2f}
• Max Position: {self.risk_metrics['max_position']:.3f}
• Risk-Adj Return: {self.risk_metrics['risk_adjusted_return']:.3f}

🌍 MARKET ADAPTATION
• Adaptive Clip: {self.adaptive_clip_eps:.3f}
• LR Factor: {self.adaptive_lr_factor:.3f}
• Entropy Coeff: {self.entropy_coeff:.6f}
• Running Adv Std: {self.running_adv_std:.3f}

📊 LAG FEATURES
• Window: {self.lag_window}
• Price Buffer: {len(self.price_buffer)}/{self.lag_window}
• Vol Scaling: {'✅' if self.vol_scaling else '❌'}
• Position Aware: {'✅' if self.position_aware else '❌'}

🌍 MARKET PERFORMANCE
{chr(10).join([f"• {regime}: {data['avg']:.3f} ({data['count']} episodes)" for regime, data in market_summary.items()])}

⚙️ BUFFER STATUS
• Size: {len(self.buffer['rewards'])}/200
• Experience: {len(self.position_history):,} records
        """
    
    # Legacy compatibility and additional methods
    def step(self, *args, **kwargs):
        """Legacy step method"""
        pass
    
    def get_weights(self) -> Dict[str, Any]:
        """Get network weights"""
        return {'network': self.network.state_dict()}
    
    def get_gradients(self) -> Dict[str, Any]:
        """Get current gradients"""
        gradients = {}
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().numpy()
            else:
                gradients[name] = None
        return gradients