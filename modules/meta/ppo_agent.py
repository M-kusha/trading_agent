# ─────────────────────────────────────────────────────────────
# File: modules/meta/ppo_agent.py
# Enhanced with InfoBus integration & advanced features
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


class EnhancedPPONetwork(nn.Module):
    """Enhanced PPO network with advanced features"""
    
    def __init__(self, obs_size: int, act_size: int, hidden_size: int = 64):
        super().__init__()
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor network with improved architecture
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_size),
            nn.Tanh()
        )
        
        # Critic network with value decomposition
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Action std parameter (learnable)
        self.log_std = nn.Parameter(torch.log(torch.ones(act_size) * 0.1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor):
        """Forward pass with enhanced error handling"""
        try:
            # Validate input
            if torch.any(torch.isnan(obs)):
                obs = torch.nan_to_num(obs)
            
            # Feature extraction
            features = self.feature_extractor(obs)
            
            # Actor output
            action_mean = self.actor_head(features)
            action_std = torch.exp(torch.clamp(self.log_std, -5, 2))
            
            # Critic output
            value = self.value_head(features)
            
            # Validate outputs
            if torch.any(torch.isnan(action_mean)):
                action_mean = torch.zeros_like(action_mean)
            if torch.any(torch.isnan(action_std)):
                action_std = torch.ones_like(action_std) * 0.1
            if torch.any(torch.isnan(value)):
                value = torch.zeros_like(value)
            
            return action_mean, action_std, value
            
        except Exception as e:
            batch_size = obs.shape[0] if obs.ndim > 1 else 1
            act_size = self.log_std.shape[0]
            return (
                torch.zeros(batch_size, act_size, device=obs.device),
                torch.ones(batch_size, act_size, device=obs.device) * 0.1,
                torch.zeros(batch_size, 1, device=obs.device)
            )


class PPOAgent(Module, AnalysisMixin, TradingMixin):
    """
    Enhanced PPO agent with InfoBus integration and advanced features.
    Provides robust policy optimization with comprehensive monitoring and automation.
    """
    
    def __init__(self, obs_size: int, act_size: int = 2, hidden_size: int = 64, 
                 lr: float = 3e-4, device: str = "cpu", debug: bool = True,
                 **kwargs):
        
        # Enhanced configuration
        config = ModuleConfig(
            debug=debug,
            max_history=500,
            health_check_interval=120,
            performance_window=100,
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
        self.learning_rate = lr
        
        # Enhanced network architecture
        self.network = EnhancedPPONetwork(obs_size, act_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # PPO hyperparameters with improved defaults
        self.clip_eps = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        self.gae_lambda = 0.95
        self.gamma = 0.99
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.batch_size = 64
        
        # Experience buffer with enhanced storage
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
            'explained_variance': 0.0
        }
        
        # Action tracking and analysis
        self.last_action = np.zeros(act_size, dtype=np.float32)
        self.action_history = deque(maxlen=1000)
        self.action_statistics = {
            'mean_action': np.zeros(act_size),
            'action_std': np.ones(act_size),
            'action_range': np.ones(act_size),
            'exploration_level': 0.5
        }
        
        # Advanced features
        self.adaptive_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=50, verbose=False
        )
        self.early_stopping_patience = 100
        self.early_stopping_counter = 0
        self.best_performance = -np.inf
        
        # Market context integration
        self.market_context_history = deque(maxlen=50)
        self.context_performance = defaultdict(lambda: {'rewards': [], 'count': 0})
        
        # Enhanced logging with rotation
        self.logger = RotatingLogger(
            "PPOAgent",
            "logs/strategy/ppo/ppo_agent.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("PPOAgent")
        
        self.log_operator_info(
            "🎯 Enhanced PPO Agent initialized",
            obs_size=obs_size,
            act_size=act_size,
            hidden_size=hidden_size,
            learning_rate=f"{lr:.2e}",
            device=device,
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
        
        # Reset tracking
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.policy_losses.clear()
        self.value_losses.clear()
        self.entropy_losses.clear()
        
        # Reset training statistics
        self.training_stats = {
            'total_updates': 0,
            'episodes_completed': 0,
            'best_episode_reward': -np.inf,
            'avg_episode_reward': 0.0,
            'policy_loss_trend': 0.0,
            'value_loss_trend': 0.0,
            'entropy_trend': 0.0,
            'gradient_norm': 0.0,
            'explained_variance': 0.0
        }
        
        # Reset action tracking
        self.last_action = np.zeros(self.act_size, dtype=np.float32)
        self.action_history.clear()
        self.action_statistics = {
            'mean_action': np.zeros(self.act_size),
            'action_std': np.ones(self.act_size),
            'action_range': np.ones(self.act_size),
            'exploration_level': 0.5
        }
        
        # Reset adaptive components
        self.early_stopping_counter = 0
        self.best_performance = -np.inf
        
        # Reset market context
        self.market_context_history.clear()
        self.context_performance.clear()
        
        self.log_operator_info("🔄 PPO Agent reset - all state cleared")
    
    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if info_bus:
            # Extract and store market context
            context = extract_standard_context(info_bus)
            self.market_context_history.append(context)
            
            # Update performance tracking from InfoBus
            self._update_performance_from_info_bus(info_bus, context)
            
            # Adapt learning based on market conditions
            self._adapt_to_market_conditions(context)
            
            # Publish agent status
            self._publish_agent_status(info_bus)
    
    def _update_performance_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Update performance metrics from InfoBus"""
        
        # Extract recent performance
        recent_trades = info_bus.get('recent_trades', [])
        if recent_trades:
            total_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
            self._update_trading_metrics({'pnl': total_pnl})
            
            # Update context-specific performance
            regime = context.get('regime', 'unknown')
            self.context_performance[regime]['rewards'].append(total_pnl)
            self.context_performance[regime]['count'] += 1
    
    def _adapt_to_market_conditions(self, context: Dict[str, Any]):
        """Adapt agent behavior based on market conditions"""
        
        regime = context.get('regime', 'unknown')
        volatility = context.get('volatility_level', 'medium')
        
        # Adjust exploration based on market conditions
        if regime == 'volatile' or volatility == 'extreme':
            # Reduce exploration in volatile markets
            self.entropy_coeff = max(0.005, self.entropy_coeff * 0.95)
        elif regime == 'trending' and volatility == 'low':
            # Increase exploration in stable trending markets
            self.entropy_coeff = min(0.02, self.entropy_coeff * 1.02)
        
        # Adjust learning rate based on regime performance
        if regime in self.context_performance:
            regime_performance = self.context_performance[regime]
            if len(regime_performance['rewards']) >= 10:
                avg_reward = np.mean(regime_performance['rewards'][-10:])
                if avg_reward < 0:
                    # Poor performance in this regime, increase learning
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = min(param_group['lr'] * 1.02, 1e-3)
                elif avg_reward > 10:
                    # Good performance, stabilize learning
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * 0.98, 1e-5)
    
    def record_step(self, obs_vec: np.ndarray, reward: float, done: bool = False, **kwargs):
        """Enhanced step recording with comprehensive validation"""
        
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                self.log_operator_error(f"NaN in observation vector: {obs_vec}")
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                self.log_operator_error("NaN reward, setting to 0")
                reward = 0.0
            
            # Convert to tensors
            obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)
            
            # Get action and value from network
            with torch.no_grad():
                action_mean, action_std, value = self.network(obs_tensor.unsqueeze(0))
                
                # Create action distribution
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # Validate network outputs
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
            self.buffer['actions'].append(action.squeeze(0))
            self.buffer['log_probs'].append(log_prob.squeeze(0))
            self.buffer['values'].append(value.squeeze(0))
            self.buffer['rewards'].append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            self.buffer['dones'].append(torch.tensor(done, dtype=torch.bool, device=self.device))
            
            # Update action tracking
            self.last_action = action.cpu().numpy().squeeze(0)
            self.action_history.append(self.last_action.copy())
            self._update_action_statistics()
            
            # Update trading metrics
            self._update_trading_metrics({'pnl': reward})
            
            self.log_operator_debug(
                f"Step recorded",
                reward=f"{reward:.3f}",
                action_mean=f"{self.last_action.mean():.3f}",
                value_estimate=f"{value.item():.3f}",
                buffer_size=len(self.buffer['rewards'])
            )
            
        except Exception as e:
            self.log_operator_error(f"Step recording failed: {e}")
    
    def _update_action_statistics(self):
        """Update action statistics for analysis"""
        
        if len(self.action_history) >= 10:
            recent_actions = np.array(list(self.action_history)[-50:])
            
            self.action_statistics['mean_action'] = np.mean(recent_actions, axis=0)
            self.action_statistics['action_std'] = np.std(recent_actions, axis=0)
            self.action_statistics['action_range'] = np.ptp(recent_actions, axis=0)
            
            # Calculate exploration level
            action_diversity = np.mean(self.action_statistics['action_std'])
            self.action_statistics['exploration_level'] = min(1.0, action_diversity / 0.5)
    
    def end_episode(self, gamma: float = None, final_value: float = 0.0):
        """Enhanced episode ending with GAE and improved updates"""
        
        try:
            if len(self.buffer['rewards']) < 5:
                self.log_operator_warning(f"Episode too short ({len(self.buffer['rewards'])} steps), skipping update")
                self._clear_buffer()
                return
            
            gamma = gamma if gamma is not None else self.gamma
            
            self.log_operator_info(
                f"Episode ending",
                steps=len(self.buffer['rewards']),
                total_reward=f"{sum(r.item() for r in self.buffer['rewards']):.3f}"
            )
            
            # Calculate advantages and returns using GAE
            self._compute_gae_returns(gamma, final_value)
            
            # Perform PPO updates
            update_stats = self._update_policy()
            
            # Update training statistics
            self._update_training_statistics(update_stats)
            
            # Check for early stopping
            self._check_early_stopping()
            
            # Clear buffer
            self._clear_buffer()
            
            # Update adaptive components
            if len(self.episode_rewards) > 0:
                self.adaptive_lr_scheduler.step(self.episode_rewards[-1])
            
        except Exception as e:
            self.log_operator_error(f"Episode ending failed: {e}")
            self._clear_buffer()
    
    def _compute_gae_returns(self, gamma: float, final_value: float):
        """Compute Generalized Advantage Estimation"""
        
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
            
            # Calculate returns and advantages
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
            
            # Bootstrap final value
            next_value = final_value
            next_advantage = 0
            
            # Compute backwards
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    next_value = 0
                    next_advantage = 0
                
                # Calculate TD error
                td_error = rewards[t] + gamma * next_value - values[t]
                
                # Calculate advantage using GAE
                advantages[t] = td_error + gamma * self.gae_lambda * next_advantage
                
                # Calculate return
                returns[t] = rewards[t] + gamma * next_value
                
                next_value = values[t]
                next_advantage = advantages[t]
            
            # Normalize advantages
            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Store in buffer
            self.buffer['advantages'] = [advantages[i] for i in range(len(advantages))]
            self.buffer['returns'] = [returns[i] for i in range(len(returns))]
            
            # Track episode reward
            episode_reward = float(rewards.sum())
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(rewards))
            
        except Exception as e:
            self.log_operator_error(f"GAE computation failed: {e}")
            # Fallback to simple advantage calculation
            returns = []
            running_return = final_value
            for reward in reversed(self.buffer['rewards']):
                running_return = reward + gamma * running_return
                returns.insert(0, running_return)
            
            values = [v.item() for v in self.buffer['values']]
            advantages = [ret - val for ret, val in zip(returns, values)]
            
            self.buffer['returns'] = [torch.tensor(r, device=self.device) for r in returns]
            self.buffer['advantages'] = [torch.tensor(a, device=self.device) for a in advantages]
    
    def _update_policy(self) -> Dict[str, float]:
        """Perform PPO policy updates"""
        
        try:
            # Prepare data
            observations = torch.stack(self.buffer['observations'])
            actions = torch.stack(self.buffer['actions'])
            old_log_probs = torch.stack(self.buffer['log_probs'])
            returns = torch.stack(self.buffer['returns'])
            advantages = torch.stack(self.buffer['advantages'])
            
            # Validate data
            for name, tensor in [('observations', observations), ('actions', actions), 
                               ('old_log_probs', old_log_probs), ('returns', returns), 
                               ('advantages', advantages)]:
                if torch.any(torch.isnan(tensor)):
                    self.log_operator_error(f"NaN in {name}")
                    tensor = torch.nan_to_num(tensor)
            
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            total_grad_norm = 0
            
            # Multiple epochs of updates
            for epoch in range(self.ppo_epochs):
                # Create mini-batches
                indices = torch.randperm(len(observations), device=self.device)
                
                for start in range(0, len(observations), self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    # Get batch data
                    batch_obs = observations[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    
                    # Forward pass
                    action_mean, action_std, values = self.network(batch_obs)
                    
                    # Create distribution
                    dist = torch.distributions.Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)
                    
                    # Calculate policy loss
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    # Clamp ratio for stability
                    ratio = torch.clamp(ratio, 0.1, 10.0)
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Calculate value loss
                    value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                    
                    # Calculate entropy loss
                    entropy_loss = -entropy.mean()
                    
                    # Total loss
                    total_loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
                    
                    # Validate loss
                    if torch.isnan(total_loss):
                        self.log_operator_error("NaN loss detected, skipping update")
                        continue
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    
                    # Update
                    self.optimizer.step()
                    
                    # Accumulate statistics
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    total_grad_norm += grad_norm.item()
            
            # Calculate averages
            num_updates = self.ppo_epochs * max(1, len(observations) // self.batch_size)
            
            update_stats = {
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'entropy_loss': total_entropy_loss / num_updates,
                'grad_norm': total_grad_norm / num_updates
            }
            
            # Store losses for tracking
            self.policy_losses.append(update_stats['policy_loss'])
            self.value_losses.append(update_stats['value_loss'])
            self.entropy_losses.append(update_stats['entropy_loss'])
            
            return update_stats
            
        except Exception as e:
            self.log_operator_error(f"Policy update failed: {e}")
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0, 'grad_norm': 0}
    
    def _update_training_statistics(self, update_stats: Dict[str, float]):
        """Update comprehensive training statistics"""
        
        self.training_stats['total_updates'] += 1
        self.training_stats['episodes_completed'] += 1
        
        if len(self.episode_rewards) > 0:
            latest_reward = self.episode_rewards[-1]
            
            # Update best reward
            if latest_reward > self.training_stats['best_episode_reward']:
                self.training_stats['best_episode_reward'] = latest_reward
            
            # Update average reward
            self.training_stats['avg_episode_reward'] = float(np.mean(list(self.episode_rewards)[-50:]))
        
        # Update loss trends
        if len(self.policy_losses) >= 10:
            recent_policy_losses = list(self.policy_losses)[-10:]
            self.training_stats['policy_loss_trend'] = np.polyfit(range(len(recent_policy_losses)), recent_policy_losses, 1)[0]
        
        if len(self.value_losses) >= 10:
            recent_value_losses = list(self.value_losses)[-10:]
            self.training_stats['value_loss_trend'] = np.polyfit(range(len(recent_value_losses)), recent_value_losses, 1)[0]
        
        # Update from current update
        self.training_stats['gradient_norm'] = update_stats['grad_norm']
        
        # Calculate explained variance
        if len(self.buffer['returns']) > 0 and len(self.buffer['values']) > 0:
            returns_np = np.array([r.item() for r in self.buffer['returns']])
            values_np = np.array([v.item() for v in self.buffer['values']])
            
            if np.var(returns_np) > 1e-6:
                explained_var = 1 - np.var(returns_np - values_np) / np.var(returns_np)
                self.training_stats['explained_variance'] = max(0, explained_var)
        
        # Log statistics periodically
        if self.training_stats['episodes_completed'] % 10 == 0:
            self.log_operator_info(
                f"📊 Training statistics update",
                episodes=self.training_stats['episodes_completed'],
                avg_reward=f"{self.training_stats['avg_episode_reward']:.3f}",
                best_reward=f"{self.training_stats['best_episode_reward']:.3f}",
                policy_loss=f"{update_stats['policy_loss']:.6f}",
                value_loss=f"{update_stats['value_loss']:.6f}",
                explained_variance=f"{self.training_stats['explained_variance']:.3f}"
            )
    
    def _check_early_stopping(self):
        """Check for early stopping conditions"""
        
        if len(self.episode_rewards) >= 20:
            recent_performance = np.mean(list(self.episode_rewards)[-20:])
            
            if recent_performance > self.best_performance:
                self.best_performance = recent_performance
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.log_operator_warning(
                    f"Early stopping triggered",
                    patience=self.early_stopping_patience,
                    best_performance=f"{self.best_performance:.3f}",
                    recent_performance=f"{recent_performance:.3f}"
                )
    
    def _clear_buffer(self):
        """Clear experience buffer"""
        for key in self.buffer:
            self.buffer[key].clear()
    
    def select_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Enhanced action selection with validation"""
        
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                self.log_operator_error("NaN in observation tensor")
                obs_tensor = torch.nan_to_num(obs_tensor)
            
            with torch.no_grad():
                action_mean, action_std, _ = self.network(obs_tensor)
                
                # Create distribution and sample
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                
                # Validate action
                if torch.any(torch.isnan(action)):
                    self.log_operator_error("NaN in action output")
                    action = torch.zeros_like(action)
                
                # Clamp action to reasonable range
                action = torch.clamp(action, -2.0, 2.0)
                
                return action
            
        except Exception as e:
            self.log_operator_error(f"Action selection failed: {e}")
            batch_size = obs_tensor.shape[0] if obs_tensor.ndim > 1 else 1
            return torch.zeros(batch_size, self.act_size, device=self.device)
    
    # ═══════════════════════════════════════════════════════════════════
    # OBSERVATION AND ANALYSIS METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with comprehensive agent state"""
        
        try:
            # Action statistics
            action_mean = float(self.action_statistics['mean_action'].mean())
            action_std = float(self.action_statistics['action_std'].mean())
            exploration_level = float(self.action_statistics['exploration_level'])
            
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
            
            # Training progress
            episodes_normalized = min(1.0, self.training_stats['episodes_completed'] / 1000.0)
            
            # Loss trends
            policy_loss_trend = float(self.training_stats['policy_loss_trend'])
            value_loss_trend = float(self.training_stats['value_loss_trend'])
            
            # Explained variance
            explained_variance = float(self.training_stats['explained_variance'])
            
            # Current learning rate
            current_lr = float(self.optimizer.param_groups[0]['lr']) / 1e-3  # Normalize
            
            observation = np.array([
                action_mean,           # Current action bias
                action_std,            # Action exploration level
                exploration_level,     # Calculated exploration
                avg_reward / 100.0,    # Normalized average reward
                reward_trend / 10.0,   # Normalized reward trend
                episodes_normalized,   # Training progress
                policy_loss_trend,     # Policy improvement trend
                value_loss_trend,      # Value learning trend
                explained_variance,    # Value function quality
                current_lr,            # Adaptive learning rate
                float(len(self.buffer['rewards'])) / 200.0,  # Buffer fullness
                self.entropy_coeff / 0.02  # Current exploration coefficient
            ], dtype=np.float32)
            
            # Validate observation
            if np.any(np.isnan(observation)):
                self.log_operator_error(f"NaN in observation: {observation}")
                observation = np.nan_to_num(observation)
            
            # Clamp to reasonable ranges
            observation = np.clip(observation, -5.0, 5.0)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(12, dtype=np.float32)
    
    def _publish_agent_status(self, info_bus: InfoBus):
        """Publish agent status to InfoBus"""
        
        agent_status = {
            'agent_type': 'ppo',
            'training_stats': self.training_stats.copy(),
            'action_statistics': {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in self.action_statistics.items()
            },
            'performance_metrics': {
                'episodes_completed': len(self.episode_rewards),
                'avg_episode_reward': float(np.mean(list(self.episode_rewards)[-10:])) if self.episode_rewards else 0.0,
                'best_episode_reward': float(max(self.episode_rewards)) if self.episode_rewards else 0.0,
                'recent_policy_loss': float(self.policy_losses[-1]) if self.policy_losses else 0.0,
                'recent_value_loss': float(self.value_losses[-1]) if self.value_losses else 0.0,
                'buffer_size': len(self.buffer['rewards'])
            },
            'learning_parameters': {
                'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                'entropy_coeff': self.entropy_coeff,
                'clip_eps': self.clip_eps,
                'early_stopping_counter': self.early_stopping_counter
            },
            'context_performance': {
                regime: {
                    'avg_reward': float(np.mean(data['rewards'][-10:])) if len(data['rewards']) >= 10 else 0.0,
                    'count': data['count']
                }
                for regime, data in self.context_performance.items()
            }
        }
        
        InfoBusUpdater.update_agent_status(info_bus, 'ppo', agent_status)
    
    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT AND PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════
    
    def get_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        base_state = super().get_state()
        
        agent_state = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_stats': self.training_stats.copy(),
            'action_statistics': {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in self.action_statistics.items()
            },
            'hyperparameters': {
                'clip_eps': self.clip_eps,
                'value_coeff': self.value_coeff,
                'entropy_coeff': self.entropy_coeff,
                'gae_lambda': self.gae_lambda,
                'gamma': self.gamma
            },
            'performance_history': {
                'episode_rewards': list(self.episode_rewards),
                'episode_lengths': list(self.episode_lengths),
                'policy_losses': list(self.policy_losses),
                'value_losses': list(self.value_losses)
            },
            'early_stopping_state': {
                'best_performance': self.best_performance,
                'early_stopping_counter': self.early_stopping_counter
            },
            'context_performance': {
                regime: {
                    'rewards': data['rewards'][-50:],  # Keep recent history
                    'count': data['count']
                }
                for regime, data in self.context_performance.items()
            },
            'last_action': self.last_action.tolist()
        }
        
        if base_state:
            base_state.update(agent_state)
            return base_state
        
        return agent_state
    
    def set_state(self, state: Dict[str, Any], strict: bool = False):
        """Enhanced state restoration"""
        super().set_state(state)
        
        try:
            # Restore network and optimizer
            if 'network_state' in state:
                self.network.load_state_dict(state['network_state'], strict=strict)
            if 'optimizer_state' in state:
                self.optimizer.load_state_dict(state['optimizer_state'])
            
            # Restore training statistics
            if 'training_stats' in state:
                self.training_stats.update(state['training_stats'])
            
            # Restore action statistics
            if 'action_statistics' in state:
                for key, value in state['action_statistics'].items():
                    if isinstance(value, list):
                        self.action_statistics[key] = np.array(value)
                    else:
                        self.action_statistics[key] = value
            
            # Restore hyperparameters
            if 'hyperparameters' in state:
                hyperparams = state['hyperparameters']
                self.clip_eps = hyperparams.get('clip_eps', self.clip_eps)
                self.value_coeff = hyperparams.get('value_coeff', self.value_coeff)
                self.entropy_coeff = hyperparams.get('entropy_coeff', self.entropy_coeff)
                self.gae_lambda = hyperparams.get('gae_lambda', self.gae_lambda)
                self.gamma = hyperparams.get('gamma', self.gamma)
            
            # Restore performance history
            if 'performance_history' in state:
                history = state['performance_history']
                self.episode_rewards = deque(history.get('episode_rewards', []), maxlen=100)
                self.episode_lengths = deque(history.get('episode_lengths', []), maxlen=100)
                self.policy_losses = deque(history.get('policy_losses', []), maxlen=100)
                self.value_losses = deque(history.get('value_losses', []), maxlen=100)
            
            # Restore early stopping state
            if 'early_stopping_state' in state:
                es_state = state['early_stopping_state']
                self.best_performance = es_state.get('best_performance', -np.inf)
                self.early_stopping_counter = es_state.get('early_stopping_counter', 0)
            
            # Restore context performance
            if 'context_performance' in state:
                for regime, data in state['context_performance'].items():
                    self.context_performance[regime]['rewards'] = data.get('rewards', [])
                    self.context_performance[regime]['count'] = data.get('count', 0)
            
            # Restore last action
            if 'last_action' in state:
                self.last_action = np.array(state['last_action'], dtype=np.float32)
            
            self.log_operator_info("✅ PPO Agent state restored successfully")
            
        except Exception as e:
            self.log_operator_error(f"State restoration failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS AND REPORTING METHODS
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
        
        # Learning status
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Context performance summary
        context_summary = {}
        for regime, data in self.context_performance.items():
            if len(data['rewards']) >= 5:
                context_summary[regime] = {
                    'avg': np.mean(data['rewards'][-10:]),
                    'count': data['count']
                }
        
        return f"""
🎯 PPO AGENT STATUS
═══════════════════════════════════════
🧠 Network: {sum(p.numel() for p in self.network.parameters()):,} parameters
📊 Episodes: {self.training_stats['episodes_completed']}
🎯 Updates: {self.training_stats['total_updates']}

💰 PERFORMANCE
• Avg Reward (10ep): {avg_reward:.3f}
• Best Reward: {best_reward:.3f}
• Reward Std: {reward_std:.3f}
• Explained Variance: {self.training_stats['explained_variance']:.3f}

📈 LEARNING METRICS
• Learning Rate: {current_lr:.2e}
• Policy Loss Trend: {self.training_stats['policy_loss_trend']:.6f}
• Value Loss Trend: {self.training_stats['value_loss_trend']:.6f}
• Entropy Coeff: {self.entropy_coeff:.4f}

🎭 ACTION ANALYSIS
• Mean Action: {self.action_statistics['mean_action'].mean():.3f}
• Action Std: {self.action_statistics['action_std'].mean():.3f}
• Exploration Level: {self.action_statistics['exploration_level']:.3f}
• Buffer Size: {len(self.buffer['rewards'])}/200

⚡ EARLY STOPPING
• Best Performance: {self.best_performance:.3f}
• Patience Counter: {self.early_stopping_counter}/{self.early_stopping_patience}

🌍 CONTEXT PERFORMANCE
{chr(10).join([f"• {regime}: {data['avg']:.3f} ({data['count']} trades)" for regime, data in context_summary.items()])}

🔧 HYPERPARAMETERS
• Clip Epsilon: {self.clip_eps}
• Value Coeff: {self.value_coeff}
• GAE Lambda: {self.gae_lambda}
• Gamma: {self.gamma}
        """
    
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
    
    # Legacy compatibility
    def step(self, *args, **kwargs):
        """Legacy step method"""
        pass