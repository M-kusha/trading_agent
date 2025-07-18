# ─────────────────────────────────────────────────────────────
# File: modules/meta/ppo_lag_agent.py
# [ROCKET] PRODUCTION-READY PPO-Lag Agent System
# Enhanced with SmartInfoBus integration & advanced market adaptation
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


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
                nn.init.orthogonal_(m.weight, gain=1)
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


@dataclass
class PPOLagConfig:
    """Configuration for PPO-Lag Agent"""
    obs_size: int = 64
    act_size: int = 2
    hidden_size: int = 128
    lr: float = 1e-4
    lag_window: int = 20
    adv_decay: float = 0.95
    vol_scaling: bool = True
    position_aware: bool = True
    device: str = "cpu"
    
    # Performance thresholds
    max_processing_time_ms: float = 300
    circuit_breaker_threshold: int = 3
    min_episode_length: int = 10
    
    # PPO parameters
    clip_eps: float = 0.1
    value_coeff: float = 0.5
    entropy_coeff: float = 0.001
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    target_kl: float = 0.01


@module(
    name="PPOLagAgent",
    version="3.0.0",
    category="meta",
    provides=["agent_status", "training_metrics", "position_metrics", "market_adaptation"],
    requires=["trades", "actions", "market_data", "training_signals"],
    description="Advanced PPO-Lag agent with market adaptation and SmartInfoBus integration",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class PPOLagAgent(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced PPO-Lag agent with SmartInfoBus integration.
    Incorporates lagged market features and intelligent position awareness for trading.
    """

    def __init__(self, 
                 config: Optional[PPOLagConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        # Store the custom config to preserve it
        custom_config = config or PPOLagConfig()
        super().__init__()
        # Restore the custom config after BaseModule initialization
        self.config = custom_config
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize agent state
        self._initialize_agent_state()
        
        # Start monitoring after all initialization is complete
        
        self._start_monitoring()
        
        
        
        self.logger.info(
            format_operator_message(
                "[TARGET]", "PPO_LAG_AGENT_INITIALIZED",
                details=f"Obs size: {self.config.obs_size}, Lag window: {self.config.lag_window}",
                result="Market-aware PPO agent ready",
                context="ppo_lag_training"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for PPO-Lag agent"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="PPOLagAgent", 
            log_path="logs/ppo_lag_agent.log", 
            max_lines=3000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PPOLagAgent", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for training operations
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
                "obs_size": int(genome.get("obs_size", self.config.obs_size)),
                "act_size": int(genome.get("act_size", self.config.act_size)),
                "hidden_size": int(genome.get("hidden_size", self.config.hidden_size)),
                "lr": float(genome.get("lr", self.config.lr)),
                "lag_window": int(genome.get("lag_window", self.config.lag_window)),
                "adv_decay": float(genome.get("adv_decay", self.config.adv_decay)),
                "vol_scaling": bool(genome.get("vol_scaling", self.config.vol_scaling)),
                "position_aware": bool(genome.get("position_aware", self.config.position_aware))
            }
        else:
            self.genome = {
                "obs_size": self.config.obs_size,
                "act_size": self.config.act_size,
                "hidden_size": self.config.hidden_size,
                "lr": self.config.lr,
                "lag_window": self.config.lag_window,
                "adv_decay": self.config.adv_decay,
                "vol_scaling": self.config.vol_scaling,
                "position_aware": self.config.position_aware
            }

    def _initialize_agent_state(self):
        """Initialize PPO-Lag agent state"""
        # Core parameters
        self.device = torch.device(self.config.device)
        
        # Enhanced network architecture
        self.network = MarketAwarePPONetwork(
            self.genome["obs_size"], 
            self.genome["act_size"], 
            self.genome["lag_window"], 
            self.genome["hidden_size"]
        ).to(self.device)
        
        # Separate optimizers for different components
        self.actor_optimizer = optim.Adam(
            list(self.network.feature_extractor.parameters()) +
            list(self.network.actor_attention.parameters()) +
            list(self.network.actor_head.parameters()) +
            [self.network.log_std_base] +
            list(self.network.std_conditioner.parameters()),
            lr=self.genome["lr"], eps=1e-5
        )
        
        self.critic_optimizer = optim.Adam(
            list(self.network.market_encoder.parameters()) +
            list(self.network.value_head_1.parameters()) +
            list(self.network.value_head_2.parameters()),
            lr=self.genome["lr"] * 2, eps=1e-5  # Higher LR for critics
        )
        
        # Market adaptation state
        self.market_regime_adaptation = {
            'trending': {'std_multiplier': 1.2, 'clip_adjustment': 0.0},
            'volatile': {'std_multiplier': 0.8, 'clip_adjustment': 0.05},
            'ranging': {'std_multiplier': 1.0, 'clip_adjustment': -0.02},
            'unknown': {'std_multiplier': 1.0, 'clip_adjustment': 0.0}
        }
        
        # Lag buffers for market features
        self.price_buffer = deque(maxlen=self.genome["lag_window"])
        self.volume_buffer = deque(maxlen=self.genome["lag_window"])
        self.spread_buffer = deque(maxlen=self.genome["lag_window"])
        self.volatility_buffer = deque(maxlen=self.genome["lag_window"])
        
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
        self.market_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'rewards': deque(maxlen=100), 'count': 0})
        self.volatility_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'rewards': deque(maxlen=100), 'count': 0})
        
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
        self.adaptive_clip_eps = self.config.clip_eps
        self.adaptive_lr_factor = 1.0

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_agent_health()
                    self._analyze_agent_performance()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    def _initialize(self):
        """Initialize module"""
        try:
            # Set initial agent status in SmartInfoBus
            initial_status = {
                "agent_type": "ppo-lag",
                "episodes_completed": 0,
                "training_active": False,
                "network_parameters": sum(p.numel() for p in self.network.parameters())
            }
            
            self.smart_bus.set(
                'agent_status',
                initial_status,
                module='PPOLagAgent',
                thesis="Initial PPO-Lag agent status"
            )
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process PPO-Lag agent operations"""
        start_time = time.time()
        
        try:
            # Extract agent data
            agent_data = await self._extract_agent_data(**inputs)
            
            if not agent_data:
                return await self._handle_no_data_fallback()
            
            # Update market buffers
            market_result = await self._update_market_buffers(agent_data)
            
            # Process training step if data available
            training_result = await self._process_training_step(agent_data)
            market_result.update(training_result)
            
            # Adapt to market conditions
            adaptation_result = await self._adapt_to_market_conditions(agent_data)
            market_result.update(adaptation_result)
            
            # Update performance tracking
            performance_result = await self._update_performance_tracking(agent_data)
            market_result.update(performance_result)
            
            # Generate thesis
            thesis = await self._generate_agent_thesis(agent_data, market_result)
            
            # Update SmartInfoBus
            await self._update_agent_smart_bus(market_result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return market_result
            
        except Exception as e:
            return await self._handle_agent_error(e, start_time)

    async def _extract_agent_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract agent data from SmartInfoBus"""
        try:
            # Get recent trades
            trades = self.smart_bus.get('trades', 'PPOLagAgent') or []
            
            # Get actions
            actions = self.smart_bus.get('actions', 'PPOLagAgent') or []
            
            # Get market data
            market_data = self.smart_bus.get('market_data', 'PPOLagAgent') or {}
            
            # Get training signals
            training_signals = self.smart_bus.get('training_signals', 'PPOLagAgent') or {}
            
            # Extract context from market data
            context = self._extract_standard_context(market_data)
            
            return {
                'trades': trades,
                'actions': actions,
                'market_data': market_data,
                'training_signals': training_signals,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'obs_vec': inputs.get('obs_vec', None),
                'reward': inputs.get('reward', None),
                'done': inputs.get('done', False)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract agent data: {e}")
            return None

    def _extract_standard_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standard market context"""
        return {
            'regime': market_data.get('regime', 'unknown'),
            'volatility_level': market_data.get('volatility_level', 'medium'),
            'session': market_data.get('session', 'unknown'),
            'price': market_data.get('price', 0.0),
            'volume': market_data.get('volume', 0.0),
            'spread': market_data.get('spread', 0.0),
            'volatility': market_data.get('volatility', 1.0),
            'timestamp': datetime.now().isoformat()
        }

    async def _update_market_buffers(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update market buffers from agent data"""
        try:
            context = agent_data['context']
            
            # Extract current market conditions
            current_price = context.get('price', 0.0)
            current_volume = context.get('volume', 0.0)
            current_spread = context.get('spread', 0.0)
            current_volatility = context.get('volatility', 1.0)
            
            # Update buffers with validation
            self.update_market_buffers(current_price, current_volume, current_spread, current_volatility)
            
            return {
                'market_buffers_updated': True,
                'buffer_sizes': {
                    'price': len(self.price_buffer),
                    'volume': len(self.volume_buffer),
                    'spread': len(self.spread_buffer),
                    'volatility': len(self.volatility_buffer)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Market buffer update failed: {e}")
            return {'market_buffers_updated': False, 'error': str(e)}

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
            
        except Exception as e:
            self.logger.error(f"Buffer update failed: {e}")

    def get_lag_features(self) -> np.ndarray:
        """Extract enhanced lagged features"""
        try:
            # Get buffer contents with padding
            price_lags = list(self.price_buffer) + [0] * (self.genome["lag_window"] - len(self.price_buffer))
            volume_lags = list(self.volume_buffer) + [0] * (self.genome["lag_window"] - len(self.volume_buffer))
            spread_lags = list(self.spread_buffer) + [0] * (self.genome["lag_window"] - len(self.spread_buffer))
            vol_lags = list(self.volatility_buffer) + [1] * (self.genome["lag_window"] - len(self.volatility_buffer))
            
            # Interleave features for better temporal representation
            features = []
            for i in range(self.genome["lag_window"]):
                features.extend([
                    price_lags[i],
                    vol_lags[i], 
                    volume_lags[i],
                    spread_lags[i]
                ])
            
            result = np.array(features, dtype=np.float32)
            
            # Validate and normalize features
            if np.any(np.isnan(result)):
                result = np.nan_to_num(result)
            
            # Apply feature scaling
            result = np.clip(result, -5.0, 5.0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lag feature extraction failed: {e}")
            return np.zeros(self.genome["lag_window"] * 4, dtype=np.float32)

    async def _adapt_to_market_conditions(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced market condition adaptation"""
        try:
            context = agent_data['context']
            regime = context.get('regime', 'unknown')
            vol_level = context.get('volatility_level', 'medium')
            
            # Get adaptation parameters
            adaptation = self.market_regime_adaptation.get(regime, self.market_regime_adaptation['unknown'])
            
            # Adapt clipping epsilon
            base_clip = self.config.clip_eps
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
                    base_lr = self.genome["lr"] if optimizer == self.actor_optimizer else self.genome["lr"] * 2
                    param_group['lr'] = base_lr * self.adaptive_lr_factor
            
            # Adapt entropy coefficient based on volatility
            if vol_level == 'extreme':
                self.config.entropy_coeff = min(0.005, self.config.entropy_coeff * 0.9)  # Reduce exploration
            elif vol_level == 'low':
                self.config.entropy_coeff = min(0.02, self.config.entropy_coeff * 1.05)  # Increase exploration
            
            return {
                'market_adapted': True,
                'regime': regime,
                'vol_level': vol_level,
                'adaptive_clip': self.adaptive_clip_eps,
                'lr_factor': self.adaptive_lr_factor,
                'entropy_coeff': self.config.entropy_coeff
            }
            
        except Exception as e:
            self.logger.error(f"Market adaptation failed: {e}")
            return {'market_adapted': False, 'error': str(e)}

    async def _process_training_step(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process training step if data available"""
        try:
            obs_vec = agent_data.get('obs_vec')
            reward = agent_data.get('reward')
            done = agent_data.get('done', False)
            
            if obs_vec is not None and reward is not None:
                # Process the training step
                await self._record_training_step(obs_vec, reward, agent_data, done)
                
                # Check for episode completion
                if done:
                    await self._end_training_episode()
                
                return {
                    'training_step_processed': True,
                    'reward': reward,
                    'episode_done': done,
                    'buffer_size': len(self.buffer['rewards'])
                }
            
            return {
                'training_step_processed': False,
                'reason': 'insufficient_data'
            }
            
        except Exception as e:
            self.logger.error(f"Training step processing failed: {e}")
            return {'training_step_processed': False, 'error': str(e)}

    async def _record_training_step(self, obs_vec: np.ndarray, reward: float, 
                                   agent_data: Dict[str, Any], done: bool = False):
        """Record training step with comprehensive market data"""
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                reward = 0.0
            
            context = agent_data['context']
            
            # Extract position information from context
            position = context.get('position', 0.0)
            unrealized_pnl = context.get('unrealized_pnl', 0.0)
            
            # Get lag features
            lag_features = self.get_lag_features()
            
            # Build extended observation
            if self.genome["position_aware"]:
                # Enhanced position features
                position_features = np.array([
                    position,                           # Current position
                    unrealized_pnl,                    # Unrealized PnL
                    position * context.get('volatility', 1.0),  # Risk-adjusted position
                    np.sign(position) * min(abs(position), 1.0),  # Position direction
                    abs(position) / max(abs(position), 1.0),      # Position size ratio
                    float(len(self.position_history)) / 1000.0    # Experience level
                ], dtype=np.float32)
                
                extended_obs = np.concatenate([obs_vec, lag_features, position_features])
            else:
                extended_obs = np.concatenate([obs_vec, lag_features])
            
            # Validate extended observation
            if np.any(np.isnan(extended_obs)):
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
                
                # Apply market adaptations
                if self.genome["vol_scaling"] and context.get('volatility', 1.0) > 0:
                    vol_adjustment = np.sqrt(1.0 / context.get('volatility', 1.0))
                    action_std = action_std * vol_adjustment
                
                # Apply position-aware action scaling
                if self.genome["position_aware"] and abs(position) > 0.8:
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
                    action = torch.zeros_like(action)
                if torch.any(torch.isnan(log_prob)):
                    log_prob = torch.zeros_like(log_prob)
                if torch.any(torch.isnan(value)):
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
                'timestamp': datetime.now(),
                'volatility': context.get('volatility', 1.0)
            })
            
            # Update risk metrics
            self._update_risk_metrics()
            
            # Update trading metrics
            self._update_trading_metrics({'pnl': reward})
            
        except Exception as e:
            self.logger.error(f"Training step recording failed: {e}")

    async def _end_training_episode(self):
        """End training episode with PPO updates"""
        try:
            if len(self.buffer['rewards']) < self.config.min_episode_length:
                self.logger.warning(f"Episode too short ({len(self.buffer['rewards'])} steps), skipping")
                self._clear_buffer()
                return
            
            # Calculate GAE advantages and returns
            await self._compute_gae_advantages()
            
            # Perform PPO updates
            update_stats = await self._perform_ppo_updates()
            
            # Update training statistics
            self._update_training_statistics(update_stats)
            
            # Track episode completion
            episode_reward = sum(r.item() for r in self.buffer['rewards'])
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(self.buffer['rewards']))
            self.training_stats['episodes_completed'] += 1
            
            # Clear buffer
            self._clear_buffer()
            
        except Exception as e:
            self.logger.error(f"Episode ending failed: {e}")
            self._clear_buffer()

    async def _compute_gae_advantages(self, final_value: float = 0.0):
        """Enhanced GAE computation with validation"""
        try:
            rewards = torch.stack(self.buffer['rewards'])
            values = torch.stack(self.buffer['values'])
            dones = torch.stack(self.buffer['dones'])
            
            # Validate tensors
            if torch.any(torch.isnan(rewards)):
                rewards = torch.nan_to_num(rewards)
            if torch.any(torch.isnan(values)):
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
                td_error = rewards[t] + self.config.gamma * next_value - values[t]
                
                # GAE advantage
                advantages[t] = td_error + self.config.gamma * self.config.gae_lambda * next_advantage
                
                # Return
                returns[t] = rewards[t] + self.config.gamma * next_value
                
                next_value = values[t]
                next_advantage = advantages[t]
            
            # Adaptive advantage normalization
            if advantages.std() > 1e-6:
                # Update running std
                self.running_adv_std = self.genome["adv_decay"] * self.running_adv_std + (1 - self.genome["adv_decay"]) * advantages.std().item()
                
                # Normalize advantages
                advantages = advantages / (self.running_adv_std + 1e-8)
            
            # Store in buffer
            self.buffer['advantages'] = [advantages[i] for i in range(len(advantages))]
            self.buffer['returns'] = [returns[i] for i in range(len(returns))]
            
        except Exception as e:
            self.logger.error(f"GAE computation failed: {e}")
            # Fallback computation
            await self._fallback_advantage_computation()

    async def _fallback_advantage_computation(self, final_value: float = 0.0):
        """Fallback advantage computation"""
        try:
            returns = []
            running_return = final_value
            for reward in reversed(self.buffer['rewards']):
                running_return = reward.item() + self.config.gamma * running_return
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
            
        except Exception as e:
            self.logger.error(f"Fallback advantage computation failed: {e}")

    async def _perform_ppo_updates(self) -> Dict[str, float]:
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
                    tensor = torch.nan_to_num(tensor)
            
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0
            total_kl_div = 0
            update_count = 0
            
            # Multiple epochs
            for epoch in range(self.config.ppo_epochs):
                # Shuffle data
                indices = torch.randperm(len(observations), device=self.device)
                
                for start in range(0, len(observations), self.config.batch_size):
                    end = start + self.config.batch_size
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
                    if kl_div > self.config.target_kl * 2:
                        break
                    
                    # Policy loss
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    ratio = torch.clamp(ratio, 0.1, 10.0)  # Stability clamp
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.adaptive_clip_eps, 1 + self.adaptive_clip_eps) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_pred_clipped = values.squeeze(-1)
                    value_loss = F.mse_loss(value_pred_clipped, batch_returns)
                    
                    # Entropy loss
                    entropy_loss = -entropy.mean()
                    
                    # Total losses
                    total_actor_loss_batch = actor_loss + self.config.entropy_coeff * entropy_loss
                    total_critic_loss_batch = self.config.value_coeff * value_loss
                    
                    # Validate losses
                    if torch.isnan(total_actor_loss_batch) or torch.isnan(total_critic_loss_batch):
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
                        self.config.max_grad_norm
                    )
                    self.actor_optimizer.step()
                    
                    # Critic update
                    self.critic_optimizer.zero_grad()
                    total_critic_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.network.market_encoder.parameters()) +
                        list(self.network.value_head_1.parameters()) +
                        list(self.network.value_head_2.parameters()),
                        self.config.max_grad_norm
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
            self.logger.error(f"PPO update failed: {e}")
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0, 'kl_divergence': 0.0, 'updates_performed': 0}

    def _update_training_statistics(self, update_stats: Dict[str, float]):
        """Update comprehensive training statistics"""
        self.training_stats['total_updates'] += 1
        
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

    def _update_risk_metrics(self):
        """Update position and risk metrics"""
        if len(self.position_history) >= 10:
            recent_positions = [p['position'] for p in list(self.position_history)[-50:]]
            recent_pnls = [p['unrealized_pnl'] for p in list(self.position_history)[-50:]]
            
            self.risk_metrics['max_position'] = max(abs(p) for p in recent_positions)
            self.risk_metrics['avg_position'] = float(np.mean(np.abs(recent_positions)))
            self.risk_metrics['position_volatility'] = float(np.std(recent_positions))
            
            if self.risk_metrics['position_volatility'] > 0:
                avg_return = np.mean(recent_pnls)
                self.risk_metrics['risk_adjusted_return'] = float(avg_return / self.risk_metrics['position_volatility'])
            else:
                self.risk_metrics['risk_adjusted_return'] = 0.0

    async def _update_performance_tracking(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update performance tracking metrics"""
        try:
            # Track market-specific performance
            if len(self.episode_rewards) > 0:
                latest_reward = self.episode_rewards[-1]
                context = agent_data['context']
                
                # Track performance by regime
                regime = context.get('regime', 'unknown')
                self.market_performance[regime]['rewards'].append(latest_reward)
                self.market_performance[regime]['count'] += 1
                
                # Track performance by volatility
                vol_level = context.get('volatility_level', 'medium')
                self.volatility_performance[vol_level]['rewards'].append(latest_reward)
                self.volatility_performance[vol_level]['count'] += 1
            
            return {
                'performance_tracked': True,
                'total_episodes': len(self.episode_rewards),
                'avg_reward': float(np.mean(list(self.episode_rewards)[-10:])) if self.episode_rewards else 0.0,
                'market_regimes_tracked': len(self.market_performance),
                'volatility_levels_tracked': len(self.volatility_performance)
            }
            
        except Exception as e:
            self.logger.error(f"Performance tracking update failed: {e}")
            return {'performance_tracked': False, 'error': str(e)}

    def _clear_buffer(self):
        """Clear experience buffer"""
        for key in self.buffer:
            self.buffer[key].clear()

    async def _generate_agent_thesis(self, agent_data: Dict[str, Any], 
                                    agent_result: Dict[str, Any]) -> str:
        """Generate comprehensive agent thesis"""
        try:
            # Training metrics
            episodes_completed = self.training_stats['episodes_completed']
            total_updates = self.training_stats['total_updates']
            network_params = sum(p.numel() for p in self.network.parameters())
            
            # Performance metrics
            avg_reward = float(np.mean(list(self.episode_rewards)[-10:])) if self.episode_rewards else 0.0
            market_adapted = agent_result.get('market_adapted', False)
            
            thesis_parts = [
                f"PPO-Lag Agent: {episodes_completed} episodes, {total_updates} updates with {network_params:,} parameters",
                f"Performance: {avg_reward:.3f} avg reward with market adaptation {'active' if market_adapted else 'inactive'}"
            ]
            
            # Training details
            if agent_result.get('training_step_processed', False):
                reward = agent_result.get('reward', 0.0)
                buffer_size = agent_result.get('buffer_size', 0)
                thesis_parts.append(f"Training: {reward:.3f} reward step, {buffer_size} buffer entries")
            
            # Market adaptation details
            if market_adapted:
                regime = agent_result.get('regime', 'unknown')
                adaptive_clip = agent_result.get('adaptive_clip', 0.1)
                thesis_parts.append(f"Market adaptation: {regime} regime with {adaptive_clip:.3f} clip epsilon")
            
            # Position tracking
            thesis_parts.append(f"Position tracking: {self.position:.3f} current, {len(self.position_history)} history entries")
            
            # Risk metrics
            max_position = self.risk_metrics['max_position']
            risk_adj_return = self.risk_metrics['risk_adjusted_return']
            thesis_parts.append(f"Risk metrics: {max_position:.3f} max position, {risk_adj_return:.3f} risk-adjusted return")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Agent thesis generation failed: {str(e)} - PPO-Lag training continuing"

    async def _update_agent_smart_bus(self, agent_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with agent results"""
        try:
            # Agent status
            agent_status = {
                'agent_type': 'ppo-lag',
                'episodes_completed': self.training_stats['episodes_completed'],
                'total_updates': self.training_stats['total_updates'],
                'network_parameters': sum(p.numel() for p in self.network.parameters()),
                'training_active': len(self.buffer['rewards']) > 0,
                'buffer_size': len(self.buffer['rewards'])
            }
            
            self.smart_bus.set(
                'agent_status',
                agent_status,
                module='PPOLagAgent',
                thesis=thesis
            )
            
            # Training metrics
            training_metrics = {
                'training_stats': self.training_stats.copy(),
                'episode_rewards': list(self.episode_rewards)[-10:],
                'episode_lengths': list(self.episode_lengths)[-10:],
                'adaptive_parameters': {
                    'clip_eps': self.adaptive_clip_eps,
                    'lr_factor': self.adaptive_lr_factor,
                    'entropy_coeff': self.config.entropy_coeff
                }
            }
            
            self.smart_bus.set(
                'training_metrics',
                training_metrics,
                module='PPOLagAgent',
                thesis=f"Training metrics: {self.training_stats['episodes_completed']} episodes completed"
            )
            
            # Position metrics
            position_metrics = {
                'current_position': self.position,
                'unrealized_pnl': self.unrealized_pnl,
                'risk_metrics': self.risk_metrics.copy(),
                'position_history_size': len(self.position_history)
            }
            
            self.smart_bus.set(
                'position_metrics',
                position_metrics,
                module='PPOLagAgent',
                thesis="Position tracking and risk metrics"
            )
            
            # Market adaptation
            market_adaptation = {
                'lag_window': self.genome["lag_window"],
                'buffer_sizes': {
                    'price': len(self.price_buffer),
                    'volume': len(self.volume_buffer),
                    'spread': len(self.spread_buffer),
                    'volatility': len(self.volatility_buffer)
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
                'adaptation_settings': {
                    'vol_scaling': self.genome["vol_scaling"],
                    'position_aware': self.genome["position_aware"]
                }
            }
            
            self.smart_bus.set(
                'market_adaptation',
                market_adaptation,
                module='PPOLagAgent',
                thesis="Market adaptation and lag feature processing"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no agent data is available"""
        self.logger.warning("No agent data available - using cached state")
        
        return {
            'agent_type': 'ppo-lag',
            'episodes_completed': self.training_stats['episodes_completed'],
            'network_parameters': sum(p.numel() for p in self.network.parameters()),
            'fallback_reason': 'no_agent_data'
        }

    async def _handle_agent_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle agent operation errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "PPOLagAgent")
        explanation = self.english_explainer.explain_error(
            "PPOLagAgent", str(error), "agent operations"
        )
        
        self.logger.error(
            format_operator_message(
                "[CRASH]", "AGENT_OPERATION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="ppo_lag_training"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'agent_type': 'ppo-lag',
            'episodes_completed': self.training_stats['episodes_completed'],
            'network_parameters': sum(p.numel() for p in self.network.parameters()),
            'fallback_reason': reason,
            'circuit_breaker_state': self.circuit_breaker['state']
        }

    def _update_agent_health(self):
        """Update agent health metrics"""
        try:
            # Check training progress
            if self.training_stats['episodes_completed'] > 10:
                recent_rewards = list(self.episode_rewards)[-10:]
                avg_reward = np.mean(recent_rewards)
                
                if avg_reward < -20:  # Poor performance
                    self._health_status = 'warning'
                elif avg_reward > 10:  # Good performance
                    self._health_status = 'healthy'
                else:
                    self._health_status = 'healthy'
            
            # Check for NaN in network parameters
            has_nan = any(torch.isnan(p).any() for p in self.network.parameters())
            if has_nan:
                self._health_status = 'critical'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_agent_performance(self):
        """Analyze agent performance metrics"""
        try:
            if len(self.episode_rewards) > 20:
                recent_performance = np.mean(list(self.episode_rewards)[-10:])
                overall_performance = np.mean(list(self.episode_rewards))
                
                if recent_performance > overall_performance * 1.2:
                    self.logger.info(
                        format_operator_message(
                            "[CHART]", "PERFORMANCE_IMPROVEMENT",
                            recent_avg=f"{recent_performance:.3f}",
                            overall_avg=f"{overall_performance:.3f}",
                            episodes=len(self.episode_rewards),
                            context="agent_performance"
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'PPOLagAgent', 'agent_cycle', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'PPOLagAgent', 'agent_cycle', 0, False
        )

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in agent recommendations"""
        try:
            base_confidence = 0.5
            
            # Confidence from training progress
            if self.training_stats['episodes_completed'] > 10:
                recent_rewards = list(self.episode_rewards)[-10:]
                if recent_rewards:
                    avg_reward = np.mean(recent_rewards)
                    reward_std = np.std(recent_rewards)
                    
                    # Higher average reward increases confidence
                    if avg_reward > 5:
                        base_confidence += 0.3
                    elif avg_reward < -5:
                        base_confidence -= 0.2
                    
                    # Lower variance increases confidence
                    if reward_std < 10:
                        base_confidence += 0.2
            
            # Confidence from model stability
            explained_variance = self.training_stats['explained_variance']
            base_confidence += explained_variance * 0.2
            
            # Confidence from position management
            if self.risk_metrics['risk_adjusted_return'] > 0:
                base_confidence += 0.1
            
            # Action-specific confidence adjustments
            if isinstance(action, dict):
                action_magnitude = action.get('magnitude', 0.5)
                if action_magnitude > 0.8:  # High confidence actions
                    base_confidence += 0.1
                elif action_magnitude < 0.3:  # Low confidence actions
                    base_confidence -= 0.1
            
            return float(np.clip(base_confidence, 0.1, 1.0))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5
            recent_rewards = list(self.episode_rewards)[-10:]
            avg_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)
            
            # Higher average reward increases confidence
            if avg_reward > 5:
                base_confidence += 0.3
            elif avg_reward < -5:
                base_confidence -= 0.2
            
            # Lower variance increases confidence
            if reward_std < 10:
                base_confidence += 0.2
        
        # Confidence from model stability
        explained_variance = self.training_stats['explained_variance']
        base_confidence += explained_variance * 0.2
        
        # Confidence from position management
        if self.risk_metrics['risk_adjusted_return'] > 0:
            base_confidence += 0.1
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'config': self.config.__dict__,
            'genome': self.genome.copy(),
            'training_stats': self.training_stats.copy(),
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'risk_metrics': self.risk_metrics.copy(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'market_performance': {k: {'rewards': list(v['rewards']), 'count': v['count']} for k, v in self.market_performance.items()},
            'volatility_performance': {k: {'rewards': list(v['rewards']), 'count': v['count']} for k, v in self.volatility_performance.items()},
            'position_history': list(self.position_history)[-100:],  # Keep recent history
            'adaptive_parameters': {
                'running_adv_std': self.running_adv_std,
                'adaptive_clip_eps': self.adaptive_clip_eps,
                'adaptive_lr_factor': self.adaptive_lr_factor
            },
            'network_state': self.network.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'health_status': self._health_status,
            'circuit_breaker': self.circuit_breaker.copy()
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'training_stats' in state:
            self.training_stats.update(state['training_stats'])
        
        if 'position' in state:
            self.position = state['position']
        
        if 'unrealized_pnl' in state:
            self.unrealized_pnl = state['unrealized_pnl']
        
        if 'risk_metrics' in state:
            self.risk_metrics.update(state['risk_metrics'])
        
        if 'episode_rewards' in state:
            self.episode_rewards = deque(state['episode_rewards'], maxlen=200)
        
        if 'episode_lengths' in state:
            self.episode_lengths = deque(state['episode_lengths'], maxlen=200)
        
        if 'market_performance' in state:
            for k, v in state['market_performance'].items():
                self.market_performance[k]['rewards'] = deque(v['rewards'], maxlen=100)
                self.market_performance[k]['count'] = v['count']
        
        if 'volatility_performance' in state:
            for k, v in state['volatility_performance'].items():
                self.volatility_performance[k]['rewards'] = deque(v['rewards'], maxlen=100)
                self.volatility_performance[k]['count'] = v['count']
        
        if 'position_history' in state:
            self.position_history = deque(state['position_history'], maxlen=1000)
        
        if 'adaptive_parameters' in state:
            params = state['adaptive_parameters']
            self.running_adv_std = params.get('running_adv_std', 1.0)
            self.adaptive_clip_eps = params.get('adaptive_clip_eps', self.config.clip_eps)
            self.adaptive_lr_factor = params.get('adaptive_lr_factor', 1.0)
        
        if 'network_state' in state:
            self.network.load_state_dict(state['network_state'])
        
        if 'actor_optimizer_state' in state:
            self.actor_optimizer.load_state_dict(state['actor_optimizer_state'])
        
        if 'critic_optimizer_state' in state:
            self.critic_optimizer.load_state_dict(state['critic_optimizer_state'])
        
        if 'health_status' in state:
            self._health_status = state['health_status']
        
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'episodes_completed': self.training_stats['episodes_completed'],
            'network_parameters': sum(p.numel() for p in self.network.parameters()),
            'buffer_size': len(self.buffer['rewards'])
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def record_step(self, obs_vec: np.ndarray, reward: float, **market_data):
        """Legacy compatibility for step recording"""
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                reward = 0.0
            
            # Create agent data structure
            agent_data = {
                'obs_vec': obs_vec,
                'reward': reward,
                'context': {
                    'price': market_data.get('price', 0.0),
                    'volume': market_data.get('volume', 0.0),
                    'spread': market_data.get('spread', 0.0),
                    'volatility': market_data.get('volatility', 1.0),
                    'position': market_data.get('position', 0.0),
                    'unrealized_pnl': market_data.get('unrealized_pnl', 0.0)
                }
            }
            
            # Run async processing
            asyncio.create_task(self._record_training_step(obs_vec, reward, agent_data))
            
        except Exception as e:
            self.logger.error(f"Step recording failed: {e}")

    def end_episode(self, *args, **kwargs):
        """Legacy compatibility for episode ending"""
        try:
            asyncio.create_task(self._end_training_episode())
        except Exception as e:
            self.logger.error(f"Episode ending failed: {e}")

    def select_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Enhanced action selection with market awareness"""
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
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
            lag_features_size = self.genome["lag_window"] * 4
            market_features = obs_tensor[..., -lag_features_size:]
            
            with torch.no_grad():
                action_mean, action_std, _ = self.network(obs_tensor, market_features)
                
                # Sample action
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                
                # Validate action
                if torch.any(torch.isnan(action)):
                    action = torch.zeros_like(action)
                
                # Apply reasonable bounds
                action = torch.clamp(action, -2.0, 2.0)
                
                return action
            
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            batch_size = obs_tensor.shape[0] if obs_tensor.ndim > 1 else 1
            return torch.zeros(batch_size, self.genome["act_size"], device=self.device)

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
                observation = np.nan_to_num(observation)
            
            # Clamp to reasonable ranges
            observation = np.clip(observation, -3.0, 3.0)
            
            return observation
            
        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(14, dtype=np.float32)

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

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose action based on current agent state"""
        try:
            # Extract observation if available
            obs_vec = inputs.get('obs_vec')
            if obs_vec is not None:
                # Convert to tensor
                obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0).to(self.device)
                lag_features = torch.FloatTensor(self.get_lag_features()).unsqueeze(0).to(self.device)
                
                # Get action from network
                with torch.no_grad():
                    action_mean, action_std, value = self.network(obs_tensor, lag_features)
                    action = torch.normal(action_mean, action_std)
                    action = torch.clamp(action, -1.0, 1.0)
                
                return {
                    'action_type': 'trading_signal',
                    'action_values': action.cpu().numpy().flatten().tolist(),
                    'confidence': float(torch.mean(1.0 / (1.0 + action_std)).item()),
                    'value_estimate': float(value.item()),
                    'reasoning': f"PPO-Lag agent action based on {len(self.price_buffer)} market observations"
                }
            else:
                # Default action when no observation available
                return {
                    'action_type': 'no_action',
                    'action_values': [0.0] * self.genome["act_size"],
                    'confidence': 0.1,
                    'value_estimate': 0.0,
                    'reasoning': 'No observation data available for action selection'
                }
                
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                'action_type': 'error',
                'action_values': [0.0] * self.genome["act_size"],
                'confidence': 0.0,
                'value_estimate': 0.0,
                'reasoning': f'Action proposal error: {str(e)}'
            }

    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        # Since calculate_confidence is now async, return a basic confidence
        base_confidence = 0.5
        
        # Confidence from training progress
        if hasattr(self, 'training_stats') and self.training_stats['episodes_completed'] > 10:
            recent_rewards = list(self.episode_rewards)[-10:] if hasattr(self, 'episode_rewards') else []
            if recent_rewards:
                avg_reward = np.mean(recent_rewards)
                if avg_reward > 5:
                    base_confidence += 0.3
                elif avg_reward < -5:
                    base_confidence -= 0.2
        
        return float(np.clip(base_confidence, 0.1, 1.0))