# ─────────────────────────────────────────────────────────────
# File: modules/meta/ppo_agent.py
# [ROCKET] PRODUCTION-READY PPO Agent System
# Advanced PPO with SmartInfoBus integration and neural networks
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
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


@dataclass
class PPOConfig:
    """Configuration for PPO Agent"""
    obs_size: int = 10
    act_size: int = 2
    hidden_size: int = 64
    learning_rate: float = 3e-4
    device: str = "cpu"
    
    # PPO hyperparameters
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    
    # Performance thresholds
    max_processing_time_ms: float = 500
    circuit_breaker_threshold: int = 3
    min_performance_score: float = 0.3
    
    # Training parameters
    buffer_size: int = 2048
    early_stopping_patience: int = 100
    lr_decay_patience: int = 50


class EnhancedPPONetwork(nn.Module):
    """Enhanced PPO network with improved architecture"""
    
    def __init__(self, obs_size: int, act_size: int, hidden_size: int = 64):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_size * 2)  # mean and log_std
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Special initialization for policy output
        for module in self.policy_head[-1:]:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1)
    
    def forward(self, obs: torch.Tensor):
        """Forward pass"""
        features = self.feature_extractor(obs)
        
        # Policy output
        policy_out = self.policy_head(features)
        action_mean = policy_out[..., :policy_out.size(-1)//2]
        action_log_std = policy_out[..., policy_out.size(-1)//2:]
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        # Value output
        value = self.value_head(features)
        
        return action_mean, action_log_std, value


@module(
    name="PPOAgent",
    version="3.0.0",
    category="meta",
    provides=[
        "policy_actions", "agent_performance", "training_metrics", "policy_gradients",
        "actions", "observations", "rewards", "training_signals", "training_data"
    ],
    requires=["observations", "rewards", "market_data", "training_signals"],
    description="Advanced PPO agent with SmartInfoBus integration for autonomous trading",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    is_voting_member=True
)
class PPOAgent(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced PPO agent with SmartInfoBus integration.
    Provides robust policy optimization with comprehensive monitoring and automation.
    """

    def __init__(self, 
                 config: Optional[Union[PPOConfig, Dict[str, Any]]] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        # Store the custom config to preserve it
        if isinstance(config, dict):
            # Convert dict config to PPOConfig
            custom_config = PPOConfig(**{k: v for k, v in config.items() if k in PPOConfig.__dataclass_fields__})
        else:
            custom_config = config or PPOConfig()
        
        super().__init__(**kwargs)  # Don't pass config to BaseModule
        # Restore the custom config after BaseModule initialization
        self.config = custom_config
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize PPO state
        self._initialize_ppo_state()
        
        # Initialize neural components
        self._initialize_neural_components()
        
        self.logger.info(
            format_operator_message(
                "[BOT]", "PPO_AGENT_INITIALIZED",
                details=f"Obs: {self.config.obs_size}, Actions: {self.config.act_size}, Hidden: {self.config.hidden_size}",
                result="PPO agent ready for training",
                context="ppo_initialization"
            )
        )
        
        # Start monitoring after all initialization is complete
        self._start_monitoring()
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for PPO agent"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="PPOAgent", 
            log_path="logs/meta/ppo_agent.log", 
            max_lines=5000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PPOAgent", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for neural operations
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
                "learning_rate": float(genome.get("learning_rate", self.config.learning_rate)),
                "clip_eps": float(genome.get("clip_eps", self.config.clip_eps)),
                "value_coeff": float(genome.get("value_coeff", self.config.value_coeff)),
                "entropy_coeff": float(genome.get("entropy_coeff", self.config.entropy_coeff)),
                "gae_lambda": float(genome.get("gae_lambda", self.config.gae_lambda)),
                "gamma": float(genome.get("gamma", self.config.gamma))
            }
            # Update config with genome values
            for key, value in self.genome.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            self.genome = {
                "obs_size": self.config.obs_size,
                "act_size": self.config.act_size,
                "hidden_size": self.config.hidden_size,
                "learning_rate": self.config.learning_rate,
                "clip_eps": self.config.clip_eps,
                "value_coeff": self.config.value_coeff,
                "entropy_coeff": self.config.entropy_coeff,
                "gae_lambda": self.config.gae_lambda,
                "gamma": self.config.gamma
            }

    def _initialize_ppo_state(self):
        """Initialize PPO-specific state"""
        # Device setup
        self.device = torch.device(self.config.device)
        
        # Experience buffer
        self.buffer = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'advantages': [],
            'returns': [],
            'dones': []
        }
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_losses = deque(maxlen=100)
        
        # Training statistics
        self.training_stats = {
            'total_updates': 0,
            'episodes_completed': 0,
            'best_episode_reward': -np.inf,
            'avg_episode_reward': 0.0,
            'policy_loss_trend': 0.0,
            'value_loss_trend': 0.0,
            'entropy_trend': 0.0,
            'gradient_norm': 0.0,
            'explained_variance': 0.0,
            'learning_rate': self.config.learning_rate
        }
        
        # Action tracking
        self.last_action = np.zeros(self.config.act_size, dtype=np.float32)
        self.action_history = deque(maxlen=1000)
        self.action_statistics = {
            'mean_action': np.zeros(self.config.act_size),
            'action_std': np.ones(self.config.act_size),
            'action_range': np.ones(self.config.act_size),
            'exploration_level': 0.5
        }
        
        # Market context integration
        self.market_context_history = deque(maxlen=50)
        self.context_performance = defaultdict(lambda: {'rewards': [], 'count': 0})
        
        # Learning adaptation
        self.early_stopping_counter = 0
        self.best_performance = -np.inf
        self.performance_plateau_counter = 0
        
        # Neural performance metrics
        self._neural_performance = {
            'forward_passes': 0,
            'backward_passes': 0,
            'average_loss': 0.0,
            'gradient_stability': 1.0
        }

    def _initialize_neural_components(self):
        """Initialize neural network components"""
        try:
            # Main network
            self.network = EnhancedPPONetwork(
                self.config.obs_size, 
                self.config.act_size, 
                self.config.hidden_size
            ).to(self.device)
            
            # Optimizer with improved settings
            self.optimizer = optim.Adam(
                self.network.parameters(), 
                lr=self.config.learning_rate,
                eps=1e-5,
                weight_decay=1e-4
            )
            
            # Learning rate scheduler
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.8, 
                patience=self.config.lr_decay_patience,
                verbose=False
            )
            
            self.logger.info("Neural components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Neural component initialization failed: {e}")
            self._health_status = 'error'

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_ppo_health()
                    self._analyze_learning_progress()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    def _initialize(self):
        """Initialize module"""
        try:
            # Set initial PPO status in SmartInfoBus
            initial_status = {
                "episodes_completed": 0,
                "training_updates": 0,
                "average_reward": 0.0,
                "learning_rate": self.config.learning_rate,
                "performance_score": 0.0
            }
            
            self.smart_bus.set(
                'agent_performance',
                initial_status,
                module='PPOAgent',
                thesis="Initial PPO agent performance status"
            )
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process PPO agent operations"""
        start_time = time.time()
        
        try:
            # Extract PPO data
            ppo_data = await self._extract_ppo_data(**inputs)
            
            if not ppo_data:
                return await self._handle_no_data_fallback()
            
            # Process action selection if observation provided
            action_result = {}
            if 'observation' in ppo_data:
                action_result = await self._process_action_selection(ppo_data)
            
            # Process training if experience provided
            training_result = {}
            if 'experience' in ppo_data:
                training_result = await self._process_training(ppo_data)
            
            # Update agent metrics
            metrics_result = await self._update_agent_metrics()
            
            # Combine results
            result = {**action_result, **training_result, **metrics_result}
            
            # Generate thesis
            thesis = await self._generate_ppo_thesis(ppo_data, result)
            
            # Update SmartInfoBus
            await self._update_ppo_smart_bus(result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return result
            
        except Exception as e:
            return await self._handle_ppo_error(e, start_time)

    async def _extract_ppo_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract PPO data from SmartInfoBus"""
        try:
            # Get observations
            observations = self.smart_bus.get('observations', 'PPOAgent')
            
            # Get rewards
            rewards = self.smart_bus.get('rewards', 'PPOAgent')
            
            # Get market data
            market_data = self.smart_bus.get('market_data', 'PPOAgent') or {}
            
            # Get training signals
            training_signals = self.smart_bus.get('training_signals', 'PPOAgent') or {}
            
            # Get direct inputs
            observation = inputs.get('observation')
            experience = inputs.get('experience')
            
            return {
                'observations': observations,
                'rewards': rewards,
                'market_data': market_data,
                'training_signals': training_signals,
                'observation': observation,
                'experience': experience,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract PPO data: {e}")
            return None

    async def _process_action_selection(self, ppo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process action selection"""
        try:
            observation = ppo_data['observation']
            
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
            else:
                obs_tensor = torch.tensor([observation], dtype=torch.float32).to(self.device)
            
            # Ensure correct shape
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Forward pass
            with torch.no_grad():
                action_mean, action_log_std, value = self.network(obs_tensor)
                
                # Create action distribution
                action_std = torch.exp(action_log_std)
                dist = torch.distributions.Normal(action_mean, action_std)
                
                # Sample action
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # Convert to numpy
                action_np = action.squeeze().cpu().numpy()
                log_prob_np = log_prob.item()
                value_np = value.squeeze().cpu().item()
            
            # Update action tracking
            self.last_action = action_np
            self.action_history.append(action_np.copy())
            self._update_action_statistics()
            
            # Update neural performance
            self._neural_performance['forward_passes'] += 1
            
            return {
                'action_selected': True,
                'action': action_np.tolist(),
                'log_prob': log_prob_np,
                'value_estimate': value_np,
                'action_std': action_std.squeeze().cpu().numpy().tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            return {'action_selected': False, 'error': str(e)}

    async def _process_training(self, ppo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process PPO training"""
        try:
            experience = ppo_data['experience']
            
            # Add experience to buffer
            if isinstance(experience, dict):
                for key in ['observation', 'action', 'reward', 'log_prob', 'value', 'done']:
                    if key in experience:
                        if key == 'observation':
                            self.buffer['observations'].append(experience[key])
                        elif key == 'action':
                            self.buffer['actions'].append(experience[key])
                        elif key == 'reward':
                            self.buffer['rewards'].append(experience[key])
                        elif key == 'log_prob':
                            self.buffer['log_probs'].append(experience[key])
                        elif key == 'value':
                            self.buffer['values'].append(experience[key])
                        elif key == 'done':
                            self.buffer['dones'].append(experience[key])
            
            # Check if buffer is ready for training
            if len(self.buffer['observations']) >= self.config.batch_size:
                update_result = await self._perform_policy_update()
                return {'training_performed': True, 'update_result': update_result}
            else:
                return {'training_performed': False, 'buffer_size': len(self.buffer['observations'])}
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {'training_performed': False, 'error': str(e)}

    async def _perform_policy_update(self) -> Dict[str, Any]:
        """Perform PPO policy update"""
        try:
            # Compute advantages and returns
            self._compute_gae_returns()
            
            # Convert buffer to tensors
            observations = torch.tensor(np.array(self.buffer['observations']), dtype=torch.float32).to(self.device)
            actions = torch.tensor(np.array(self.buffer['actions']), dtype=torch.float32).to(self.device)
            old_log_probs = torch.tensor(np.array(self.buffer['log_probs']), dtype=torch.float32).to(self.device)
            advantages = torch.tensor(np.array(self.buffer['advantages']), dtype=torch.float32).to(self.device)
            returns = torch.tensor(np.array(self.buffer['returns']), dtype=torch.float32).to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Training loop
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            grad_norm = 0.0  # Initialize grad_norm
            values = None  # Initialize values
            
            for epoch in range(self.config.ppo_epochs):
                # Forward pass
                action_mean, action_log_std, values = self.network(observations)
                
                # Create distribution
                action_std = torch.exp(action_log_std)
                dist = torch.distributions.Normal(action_mean, action_std)
                
                # Calculate new log probs and entropy
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                
                # Calculate ratio and clipped surrogate loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.config.value_coeff * value_loss + self.config.entropy_coeff * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
            
            # Update statistics
            avg_policy_loss = total_policy_loss / self.config.ppo_epochs
            avg_value_loss = total_value_loss / self.config.ppo_epochs
            avg_entropy_loss = total_entropy_loss / self.config.ppo_epochs
            
            self.policy_losses.append(avg_policy_loss)
            self.value_losses.append(avg_value_loss)
            self.entropy_losses.append(avg_entropy_loss)
            
            # Update training stats
            self.training_stats['total_updates'] += 1
            self.training_stats['policy_loss_trend'] = avg_policy_loss
            self.training_stats['value_loss_trend'] = avg_value_loss
            self.training_stats['entropy_trend'] = avg_entropy_loss
            self.training_stats['gradient_norm'] = float(grad_norm)
            
            # Calculate explained variance
            with torch.no_grad():
                if values is not None:
                    explained_var = 1 - torch.var(returns - values.squeeze()) / torch.var(returns)
                    self.training_stats['explained_variance'] = float(explained_var)
                else:
                    explained_var = 0.0
                    self.training_stats['explained_variance'] = 0.0
            
            # Update neural performance
            self._neural_performance['backward_passes'] += 1
            self._neural_performance['average_loss'] = avg_policy_loss + avg_value_loss
            
            # Clear buffer
            self._clear_buffer()
            
            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy_loss': avg_entropy_loss,
                'gradient_norm': float(grad_norm),
                'explained_variance': float(explained_var),
                'epochs_completed': self.config.ppo_epochs
            }
            
        except Exception as e:
            self.logger.error(f"Policy update failed: {e}")
            return {'error': str(e)}

    def _compute_gae_returns(self):
        """Compute GAE advantages and returns"""
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        dones = np.array(self.buffer['dones'])
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - dones[t]
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        
        self.buffer['advantages'] = advantages.tolist()
        self.buffer['returns'] = returns.tolist()

    def _update_action_statistics(self):
        """Update action statistics"""
        if len(self.action_history) > 10:
            actions = np.array(list(self.action_history)[-100:])  # Last 100 actions
            
            self.action_statistics['mean_action'] = np.mean(actions, axis=0)
            self.action_statistics['action_std'] = np.std(actions, axis=0)
            self.action_statistics['action_range'] = np.ptp(actions, axis=0)
            
            # Calculate exploration level
            action_entropy = -np.sum(self.action_statistics['action_std'] * np.log(self.action_statistics['action_std'] + 1e-8))
            self.action_statistics['exploration_level'] = float(np.clip(action_entropy / self.config.act_size, 0, 1))

    def _clear_buffer(self):
        """Clear experience buffer"""
        for key in self.buffer:
            self.buffer[key].clear()

    async def _update_agent_metrics(self) -> Dict[str, Any]:
        """Update agent performance metrics"""
        try:
            # Calculate performance score
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(list(self.episode_rewards)[-10:])
                performance_score = max(0, min(1, (avg_reward + 100) / 200))  # Normalize to 0-1
            else:
                avg_reward = 0.0
                performance_score = 0.0
            
            # Update training stats
            self.training_stats['avg_episode_reward'] = avg_reward
            
            # Learning rate adaptation
            if len(self.episode_rewards) > 0:
                self.lr_scheduler.step(avg_reward)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_stats['learning_rate'] = current_lr
            
            return {
                'agent_metrics': {
                    'performance_score': performance_score,
                    'average_reward': avg_reward,
                    'episodes_completed': self.training_stats['episodes_completed'],
                    'training_updates': self.training_stats['total_updates'],
                    'exploration_level': self.action_statistics['exploration_level'],
                    'learning_rate': self.training_stats['learning_rate']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Agent metrics update failed: {e}")
            return {'agent_metrics': {'error': str(e)}}

    async def _generate_ppo_thesis(self, ppo_data: Dict[str, Any], 
                                 result: Dict[str, Any]) -> str:
        """Generate comprehensive PPO thesis"""
        try:
            # Performance metrics
            episodes = self.training_stats['episodes_completed']
            updates = self.training_stats['total_updates']
            avg_reward = self.training_stats['avg_episode_reward']
            
            # Training metrics
            policy_loss = self.training_stats['policy_loss_trend']
            value_loss = self.training_stats['value_loss_trend']
            explained_var = self.training_stats['explained_variance']
            
            thesis_parts = [
                f"PPO Agent Performance: {episodes} episodes completed with {updates} policy updates",
                f"Average reward: {avg_reward:.2f} with explained variance {explained_var:.2f}",
                f"Learning progress: Policy loss {policy_loss:.4f}, Value loss {value_loss:.4f}"
            ]
            
            # Action analysis
            if result.get('action_selected', False):
                action = result.get('action', [0, 0])
                value_est = result.get('value_estimate', 0)
                thesis_parts.append(f"Action selected: [{action[0]:.3f}, {action[1]:.3f}] with value estimate {value_est:.3f}")
            
            # Training analysis
            if result.get('training_performed', False):
                update_result = result.get('update_result', {})
                grad_norm = update_result.get('gradient_norm', 0)
                thesis_parts.append(f"Policy updated with gradient norm {grad_norm:.4f}")
            
            # Exploration analysis
            exploration = self.action_statistics['exploration_level']
            thesis_parts.append(f"Exploration level: {exploration:.2f} maintaining learning diversity")
            
            # Learning rate adaptation
            current_lr = self.training_stats['learning_rate']
            if current_lr != self.config.learning_rate:
                thesis_parts.append(f"Learning rate adapted to {current_lr:.2e} for optimization")
            
            # Performance assessment
            if len(self.episode_rewards) > 10:
                recent_trend = np.mean(list(self.episode_rewards)[-5:]) - np.mean(list(self.episode_rewards)[-10:-5])
                if recent_trend > 0.1:
                    thesis_parts.append("Recent performance trend: IMPROVING")
                elif recent_trend < -0.1:
                    thesis_parts.append("Recent performance trend: DECLINING")
                else:
                    thesis_parts.append("Recent performance trend: STABLE")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"PPO thesis generation failed: {str(e)} - Agent continuing with basic functionality"

    async def _update_ppo_smart_bus(self, result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with PPO results"""
        try:
            # Policy actions
            if result.get('action_selected', False):
                action_data = {
                    'action': result.get('action', [0, 0]),
                    'log_prob': result.get('log_prob', 0.0),
                    'value_estimate': result.get('value_estimate', 0.0),
                    'action_std': result.get('action_std', [1.0, 1.0]),
                    'exploration_level': self.action_statistics['exploration_level']
                }
                
                self.smart_bus.set(
                    'policy_actions',
                    action_data,
                    module='PPOAgent',
                    thesis=thesis
                )
            
            # Agent performance
            agent_metrics = result.get('agent_metrics', {})
            performance_data = {
                'performance_score': agent_metrics.get('performance_score', 0.0),
                'average_reward': agent_metrics.get('average_reward', 0.0),
                'episodes_completed': agent_metrics.get('episodes_completed', 0),
                'training_updates': agent_metrics.get('training_updates', 0),
                'learning_rate': agent_metrics.get('learning_rate', self.config.learning_rate)
            }
            
            self.smart_bus.set(
                'agent_performance',
                performance_data,
                module='PPOAgent',
                thesis="PPO agent performance metrics and learning progress"
            )
            
            # Training metrics
            if result.get('training_performed', False):
                training_data = {
                    'policy_loss': self.training_stats['policy_loss_trend'],
                    'value_loss': self.training_stats['value_loss_trend'],
                    'entropy': self.training_stats['entropy_trend'],
                    'gradient_norm': self.training_stats['gradient_norm'],
                    'explained_variance': self.training_stats['explained_variance'],
                    'total_updates': self.training_stats['total_updates']
                }
                
                self.smart_bus.set(
                    'training_metrics',
                    training_data,
                    module='PPOAgent',
                    thesis="PPO training metrics and optimization progress"
                )
            
            # Policy gradients info
            gradient_data = {
                'gradient_norm': self.training_stats['gradient_norm'],
                'learning_rate': self.training_stats['learning_rate'],
                'network_parameters': sum(p.numel() for p in self.network.parameters()),
                'forward_passes': self._neural_performance['forward_passes'],
                'backward_passes': self._neural_performance['backward_passes']
            }
            
            self.smart_bus.set(
                'policy_gradients',
                gradient_data,
                module='PPOAgent',
                thesis="Policy gradient information and neural network performance"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no PPO data is available"""
        self.logger.warning("No PPO data available - returning current state")
        
        return {
            'episodes_completed': self.training_stats['episodes_completed'],
            'training_updates': self.training_stats['total_updates'],
            'average_reward': self.training_stats['avg_episode_reward'],
            'exploration_level': self.action_statistics['exploration_level'],
            'fallback_reason': 'no_ppo_data'
        }

    async def _handle_ppo_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle PPO errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "PPOAgent")
        explanation = self.english_explainer.explain_error(
            "PPOAgent", str(error), "PPO training"
        )
        
        self.logger.error(
            format_operator_message(
                "[CRASH]", "PPO_AGENT_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="ppo_training"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'episodes_completed': self.training_stats['episodes_completed'],
            'training_updates': self.training_stats['total_updates'],
            'average_reward': self.training_stats['avg_episode_reward'],
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    def _update_ppo_health(self):
        """Update PPO health metrics"""
        try:
            # Check if all required attributes are initialized
            if not hasattr(self, 'episode_rewards') or not hasattr(self, 'training_stats'):
                return  # Skip if not fully initialized yet
                
            # Check learning progress
            if len(self.episode_rewards) > 20:
                recent_performance = np.mean(list(self.episode_rewards)[-10:])
                if recent_performance < -50:  # Poor performance threshold
                    self._health_status = 'warning'
                elif recent_performance > 50:  # Good performance
                    self._health_status = 'healthy'
            
            # Check gradient stability
            if self.training_stats['gradient_norm'] > 10.0:
                self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_learning_progress(self):
        """Analyze learning progress"""
        try:
            # Check if all required attributes are initialized
            if not hasattr(self, 'episode_rewards'):
                return  # Skip if not fully initialized yet
                
            if len(self.episode_rewards) >= 20:
                # Check for learning plateau
                recent_rewards = list(self.episode_rewards)[-10:]
                older_rewards = list(self.episode_rewards)[-20:-10]
                
                recent_avg = np.mean(recent_rewards)
                older_avg = np.mean(older_rewards)
                
                improvement = recent_avg - older_avg
                
                if improvement > 5.0:
                    self.logger.info(
                        format_operator_message(
                            "[CHART]", "LEARNING_PROGRESS_GOOD",
                            improvement=f"{improvement:.2f}",
                            recent_avg=f"{recent_avg:.2f}",
                            context="learning_analysis"
                        )
                    )
                elif improvement < -5.0:
                    self.logger.warning(
                        format_operator_message(
                            "📉", "LEARNING_REGRESSION",
                            regression=f"{improvement:.2f}",
                            recent_avg=f"{recent_avg:.2f}",
                            context="learning_analysis"
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Learning progress analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'PPOAgent', 'processing_cycle', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'PPOAgent', 'processing_cycle', 0, False
        )

    # Legacy compatibility methods
    def record_step(self, obs_vec: np.ndarray, reward: float, done: bool = False, **kwargs):
        """Legacy compatibility for step recording"""
        experience = {
            'observation': obs_vec,
            'reward': reward,
            'done': done
        }
        
        # Add action and log_prob if available
        if hasattr(self, 'last_action'):
            experience['action'] = self.last_action
        if hasattr(self, '_last_log_prob'):
            experience['log_prob'] = self._last_log_prob
        if hasattr(self, '_last_value'):
            experience['value'] = self._last_value
        
        # Store in buffer
        for key, value in experience.items():
            if key == 'observation':
                self.buffer['observations'].append(value)
            elif key == 'action':
                self.buffer['actions'].append(value)
            elif key == 'reward':
                self.buffer['rewards'].append(value)
            elif key == 'log_prob':
                self.buffer['log_probs'].append(value)
            elif key == 'value':
                self.buffer['values'].append(value)
            elif key == 'done':
                self.buffer['dones'].append(value)

    def select_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Legacy compatibility for action selection"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            obs_np = obs_tensor.cpu().numpy()
            ppo_data = {'observation': obs_np}
            result = loop.run_until_complete(self._process_action_selection(ppo_data))
            
            if result.get('action_selected', False):
                action = result['action']
                self._last_log_prob = result['log_prob']
                self._last_value = result['value_estimate']
                return torch.tensor(action, dtype=torch.float32)
            else:
                return torch.zeros(self.config.act_size, dtype=torch.float32)
        finally:
            loop.close()

    def end_episode(self, **kwargs):
        """Legacy compatibility for episode end"""
        if len(self.buffer['rewards']) > 0:
            episode_reward = sum(self.buffer['rewards'])
            self.episode_rewards.append(episode_reward)
            self.training_stats['episodes_completed'] += 1
            
            # Update best performance
            if episode_reward > self.training_stats['best_episode_reward']:
                self.training_stats['best_episode_reward'] = episode_reward

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'training_stats': self.training_stats.copy(),
            'action_statistics': self.action_statistics.copy(),
            'genome': self.genome.copy(),
            'episodes_completed': self.training_stats['episodes_completed'],
            'total_updates': self.training_stats['total_updates'],
            'best_performance': self.best_performance,
            'circuit_breaker': self.circuit_breaker.copy(),
            'health_status': self._health_status,
            'network_state': self.network.state_dict() if hasattr(self, 'network') else {},
            'optimizer_state': self.optimizer.state_dict() if hasattr(self, 'optimizer') else {}
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'training_stats' in state:
            self.training_stats.update(state['training_stats'])
        
        if 'action_statistics' in state:
            self.action_statistics.update(state['action_statistics'])
        
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'best_performance' in state:
            self.best_performance = state['best_performance']
        
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])
        
        if 'health_status' in state:
            self._health_status = state['health_status']
        
        # Restore network and optimizer states
        if 'network_state' in state and hasattr(self, 'network'):
            try:
                self.network.load_state_dict(state['network_state'])
            except Exception as e:
                self.logger.warning(f"Failed to restore network state: {e}")
        
        if 'optimizer_state' in state and hasattr(self, 'optimizer'):
            try:
                self.optimizer.load_state_dict(state['optimizer_state'])
            except Exception as e:
                self.logger.warning(f"Failed to restore optimizer state: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'episodes_completed': self.training_stats['episodes_completed'],
            'average_reward': self.training_stats['avg_episode_reward'],
            'learning_rate': self.training_stats['learning_rate']
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        # Return confidence based on performance and exploration
        performance_confidence = max(0, min(1, (self.training_stats['avg_episode_reward'] + 50) / 100))
        exploration_confidence = 1.0 - self.action_statistics['exploration_level']
        
        return (performance_confidence + exploration_confidence) / 2

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose PPO-based action"""
        try:
            # Get observation from inputs
            obs = inputs.get('observation', inputs.get('observations', inputs.get('market_data')))
            
            if obs is not None:
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                else:
                    obs_tensor = torch.tensor([obs], dtype=torch.float32)
                
                # Use the PPO network to select action
                with torch.no_grad():
                    action_mean, _, _ = self.network(obs_tensor.unsqueeze(0))
                    action = action_mean.squeeze(0).numpy()
                
                # Store for next time
                self.last_action = action
                
                return {
                    'action_type': 'ppo_policy_action',
                    'action': action.tolist(),
                    'confidence': 0.8,
                    'reasoning': f'PPO policy action based on {len(obs)} observations',
                    'action_stats': {
                        'mean': float(np.mean(action)),
                        'std': float(np.std(action)),
                        'max': float(np.max(action)),
                        'min': float(np.min(action))
                    }
                }
            else:
                # Fallback action
                fallback_action = getattr(self, 'last_action', np.zeros(2, dtype=np.float32))
                return {
                    'action_type': 'fallback_action',
                    'action': fallback_action.tolist() if hasattr(fallback_action, 'tolist') else [0.0, 0.0],
                    'confidence': 0.3,
                    'reasoning': 'No observation provided, using fallback action'
                }
                
        except Exception as e:
            self.logger.error(f"PPO action proposal failed: {e}")
            return {
                'action_type': 'no_action',
                'confidence': 0.0,
                'reasoning': f'PPO action proposal error: {str(e)}',
                'error': str(e)
            }
    
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in PPO action"""
        try:
            if not isinstance(action, dict):
                return 0.0
            
            # Base confidence from training performance
            base_confidence = max(0, min(1, (self.training_stats['avg_episode_reward'] + 50) / 100))
            
            # Action quality assessment
            if 'action' in action and isinstance(action['action'], (list, np.ndarray)):
                action_values = np.array(action['action'])
                
                # Check for extreme actions (lower confidence)
                if np.any(np.abs(action_values) > 2.0):
                    action_confidence = 0.3
                elif np.any(np.abs(action_values) > 1.0):
                    action_confidence = 0.6
                else:
                    action_confidence = 0.9
                
                # Check action variance (too low = overconfident, too high = uncertain)
                action_std = np.std(action_values)
                if 0.1 < action_std < 0.8:
                    variance_confidence = 0.8
                else:
                    variance_confidence = 0.5
            else:
                action_confidence = 0.5
                variance_confidence = 0.5
            
            # Training stability confidence
            if len(self.policy_losses) > 10:
                recent_losses = list(self.policy_losses)[-10:]  # Convert to list for slicing
                loss_std = np.std(recent_losses)
                stability_confidence = max(0.2, 1.0 - loss_std)
            else:
                stability_confidence = 0.5
            
            # Exploration vs exploitation balance
            exploration_confidence = 1.0 - self.action_statistics.get('exploration_level', 0.5)
            
            # Combine confidences
            combined_confidence = (
                base_confidence * 0.4 +
                action_confidence * 0.3 +
                variance_confidence * 0.1 +
                stability_confidence * 0.1 +
                exploration_confidence * 0.1
            )
            
            return float(np.clip(combined_confidence, 0.1, 1.0))
            
        except Exception as e:
            self.logger.error(f"PPO confidence calculation failed: {e}")
            return 0.5