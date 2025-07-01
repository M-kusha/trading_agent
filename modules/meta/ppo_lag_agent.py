# ─────────────────────────────────────────────────────────────
# modules/meta/ppo_lag_agent.py

from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict
from collections import deque
from modules.core.core import Module

class PPOLagAgent(nn.Module, Module):
    def __init__(self, 
                 obs_size: int,
                 act_size: int = 2,
                 hidden_size: int = 128,
                 lr: float = 1e-4,
                 lag_window: int = 20,
                 adv_decay: float = 0.95,
                 vol_scaling: bool = True,
                 position_aware: bool = True,
                 device: str = "cpu",
                 debug: bool = True):
        super().__init__()
        
        self.device = torch.device(device)
        self.debug = debug
        self.lag_window = lag_window
        self.adv_decay = adv_decay
        self.vol_scaling = vol_scaling
        self.position_aware = position_aware
        self.obs_size = obs_size
        self.act_size = act_size
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"PPOLagAgent_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/ppo/ppo_lag_agent.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Expand input size for lagged features
        lag_features = 4  # returns, volatility, volume, spread
        position_features = 3 if position_aware else 0
        self.extended_obs_size = obs_size + (lag_window * lag_features) + position_features
        
        self.logger.info(f"PPOLagAgent initializing - obs_size={obs_size}, extended_size={self.extended_obs_size}")
        
        try:
            # Enhanced actor network with proper initialization
            self.actor = nn.Sequential(
                nn.Linear(self.extended_obs_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, act_size),
                nn.Tanh()
            )
            
            # Market-aware critic
            self.value_encoder = nn.Sequential(
                nn.Linear(self.extended_obs_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            self.market_encoder = nn.Sequential(
                nn.Linear(lag_window * lag_features, hidden_size // 2),
                nn.ReLU()
            )
            
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            
            # FIX: Proper weight initialization
            self._initialize_weights()
            
            self.to(self.device)
            
            # Optimizers
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_opt = optim.Adam(
                list(self.value_encoder.parameters()) + 
                list(self.market_encoder.parameters()) + 
                list(self.value_head.parameters()), 
                lr=lr * 2
            )
            
            # PPO hyperparameters
            self.clip_eps = 0.1
            self.value_coeff = 0.5
            self.entropy_coeff = 0.001
            self.max_grad_norm = 0.5
            
            # Lag buffers
            self.price_buffer = deque(maxlen=lag_window)
            self.volume_buffer = deque(maxlen=lag_window)
            self.spread_buffer = deque(maxlen=lag_window)
            self.vol_buffer = deque(maxlen=lag_window)
            
            # Episode buffers
            self.buffer = {
                "obs": [], "actions": [], "logp": [], "values": [], 
                "rewards": [], "market_features": []
            }
            
            # State tracking
            self.running_adv_std = 1.0
            self.last_action = np.zeros(act_size, dtype=np.float32)
            self.position = 0.0
            self.unrealized_pnl = 0.0
            self.total_trades = 0
            self._step_count = 0
            
            self.logger.info("PPOLagAgent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing PPOLagAgent: {e}")
            raise

    def _initialize_weights(self):
        """FIX: Proper weight initialization to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Small gain to prevent explosion
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.logger.info("Network weights initialized")

    def update_market_buffers(self, price: float, volume: float, spread: float, volatility: float):
        """Update lagged market features with validation"""
        try:
            # Validate inputs
            if np.isnan(price):
                self.logger.error("NaN price, using last valid or 0")
                price = self.price_buffer[-1] if self.price_buffer else 0.0
            if np.isnan(volume):
                volume = 0.0
            if np.isnan(spread):
                spread = 0.0
            if np.isnan(volatility) or volatility <= 0:
                volatility = 1.0
                
            if len(self.price_buffer) > 1:
                last_price = self.price_buffer[-1]
                ret = (price - last_price) / last_price if last_price > 0 else 0
            else:
                ret = 0
                
            # Validate return
            if np.isnan(ret) or abs(ret) > 0.1:  # Cap at 10% return
                ret = 0.0
                
            self.price_buffer.append(ret)
            self.volume_buffer.append(volume)
            self.spread_buffer.append(spread)
            self.vol_buffer.append(volatility)
            
            self.logger.debug(f"Market buffers updated: ret={ret:.5f}, vol={volatility:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error updating market buffers: {e}")

    def get_lag_features(self) -> np.ndarray:
        """Extract lagged features with validation"""
        try:
            price_lags = list(self.price_buffer) + [0] * (self.lag_window - len(self.price_buffer))
            volume_lags = list(self.volume_buffer) + [0] * (self.lag_window - len(self.volume_buffer))
            spread_lags = list(self.spread_buffer) + [0] * (self.lag_window - len(self.spread_buffer))
            vol_lags = list(self.vol_buffer) + [1] * (self.lag_window - len(self.vol_buffer))
            
            features = []
            for i in range(self.lag_window):
                features.extend([price_lags[i], vol_lags[i], volume_lags[i], spread_lags[i]])
                
            result = np.array(features, dtype=np.float32)
            
            # Validate features
            if np.any(np.isnan(result)):
                self.logger.error(f"NaN in lag features: {result}")
                result = np.nan_to_num(result)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting lag features: {e}")
            return np.zeros(self.lag_window * 4, dtype=np.float32)

    def forward(self, obs: torch.Tensor, market_lags: torch.Tensor):
        """Forward pass with NaN validation"""
        try:
            # Validate inputs
            if torch.any(torch.isnan(obs)):
                self.logger.error("NaN in observation input")
                obs = torch.nan_to_num(obs)
            if torch.any(torch.isnan(market_lags)):
                self.logger.error("NaN in market lags input")
                market_lags = torch.nan_to_num(market_lags)
                
            action_logits = self.actor(obs)
            
            value_features = self.value_encoder(obs)
            market_features = self.market_encoder(market_lags)
            combined = torch.cat([value_features, market_features], dim=-1)
            value = self.value_head(combined)
            
            # Validate outputs
            if torch.any(torch.isnan(action_logits)):
                self.logger.error("NaN in action logits")
                action_logits = torch.zeros_like(action_logits)
            if torch.any(torch.isnan(value)):
                self.logger.error("NaN in value output")
                value = torch.zeros_like(value)
            
            return action_logits, value
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {e}")
            # Return safe defaults
            batch_size = obs.shape[0]
            return (torch.zeros(batch_size, self.act_size, device=self.device),
                    torch.zeros(batch_size, 1, device=self.device))

    def record_step(self, obs_vec: np.ndarray, reward: float, 
                   price: float = 0, volume: float = 0, 
                   spread: float = 0, volatility: float = 1,
                   position: float = 0, unrealized_pnl: float = 0):
        """Record step with market data and comprehensive validation"""
        self._step_count += 1
        
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                self.logger.error(f"NaN in observation vector: {obs_vec}")
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                self.logger.error("NaN reward, setting to 0")
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
                position_features = np.array([position, unrealized_pnl, position * volatility], dtype=np.float32)
                extended_obs = np.concatenate([obs_vec, lag_features, position_features])
            else:
                extended_obs = np.concatenate([obs_vec, lag_features])
                
            # Validate extended observation
            if np.any(np.isnan(extended_obs)):
                self.logger.error(f"NaN in extended observation: {extended_obs}")
                extended_obs = np.nan_to_num(extended_obs)
                
            # Pad if necessary
            if len(extended_obs) < self.extended_obs_size:
                padding = np.zeros(self.extended_obs_size - len(extended_obs), dtype=np.float32)
                extended_obs = np.concatenate([extended_obs, padding])
            elif len(extended_obs) > self.extended_obs_size:
                extended_obs = extended_obs[:self.extended_obs_size]
                
            # Convert to tensors
            obs_t = torch.as_tensor(extended_obs, dtype=torch.float32, device=self.device)
            market_t = torch.as_tensor(lag_features, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                mu, value = self.forward(obs_t.unsqueeze(0), market_t.unsqueeze(0))
                
                # Volatility-scaled exploration
                if self.vol_scaling and volatility > 0:
                    action_std = 0.1 / np.sqrt(volatility)
                else:
                    action_std = 0.1
                    
                # FIX: Clamp action std to prevent NaN
                action_std = np.clip(action_std, 0.01, 0.5)
                    
                dist = torch.distributions.Normal(mu, action_std)
                action = dist.rsample()
                
                # Position-aware action scaling
                if self.position_aware and abs(position) > 0.8:
                    action = action * (1 - abs(position))
                    
                logp = dist.log_prob(action).sum(dim=-1)
                
                # Validate outputs
                if torch.any(torch.isnan(action)):
                    self.logger.error("NaN in sampled action")
                    action = torch.zeros_like(action)
                if torch.any(torch.isnan(logp)):
                    self.logger.error("NaN in log probability")
                    logp = torch.zeros_like(logp)

            # Store in buffer
            self.buffer["obs"].append(obs_t)
            self.buffer["actions"].append(action.squeeze(0))
            self.buffer["logp"].append(logp.squeeze(0))
            self.buffer["values"].append(value.squeeze(0))
            self.buffer["rewards"].append(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
            self.buffer["market_features"].append(market_t)
            
            # Update state
            self.last_action = action.cpu().numpy().squeeze(0)
            self.position = position
            self.unrealized_pnl = unrealized_pnl
            self.total_trades += 1
            
            self.logger.debug(f"Step {self._step_count} recorded: reward={reward:.3f}, action_mean={self.last_action.mean():.3f}")
            
        except Exception as e:
            self.logger.error(f"Error in record_step: {e}")

    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95):
        """GAE with advantage smoothing and NaN protection"""
        try:
            rewards = torch.stack(self.buffer["rewards"])
            values = torch.stack(self.buffer["values"])
            
            # Validate inputs
            if torch.any(torch.isnan(rewards)):
                self.logger.error("NaN in rewards buffer")
                rewards = torch.nan_to_num(rewards)
            if torch.any(torch.isnan(values)):
                self.logger.error("NaN in values buffer")
                values = torch.nan_to_num(values)
            
            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            # Compute advantages with GAE
            advantages = []
            adv = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                td_error = rewards[t] + gamma * next_value - values[t]
                adv = td_error + gamma * lam * adv
                advantages.insert(0, adv)
                
            advantages = torch.stack(advantages)
            
            # Validate advantages
            if torch.any(torch.isnan(advantages)):
                self.logger.error("NaN in computed advantages")
                advantages = torch.nan_to_num(advantages)
            
            # Smooth advantages
            adv_std = advantages.std()
            if torch.isnan(adv_std) or adv_std == 0:
                adv_std = torch.tensor(1.0, device=self.device)
                
            self.running_adv_std = self.adv_decay * self.running_adv_std + (1 - self.adv_decay) * adv_std.item()
            advantages = advantages / (self.running_adv_std + 1e-8)
            
            return advantages, returns
            
        except Exception as e:
            self.logger.error(f"Error computing advantages: {e}")
            # Return safe defaults
            n_steps = len(self.buffer["rewards"])
            return (torch.zeros(n_steps, device=self.device), 
                    torch.zeros(n_steps, device=self.device))

    def end_episode(self, gamma: float = 0.99):
        """Episode ending with updates and comprehensive error handling"""
        try:
            if len(self.buffer["rewards"]) < 10:
                self.logger.warning(f"Episode too short ({len(self.buffer['rewards'])} steps), skipping update")
                for k in self.buffer:
                    self.buffer[k].clear()
                return
                
            self.logger.info(f"Ending episode with {len(self.buffer['rewards'])} steps")
            
            # Stack tensors with validation
            obs = torch.stack(self.buffer["obs"])
            actions = torch.stack(self.buffer["actions"])
            logp_old = torch.stack(self.buffer["logp"])
            values_old = torch.stack(self.buffer["values"])
            market_features = torch.stack(self.buffer["market_features"])
            
            # Validate stacked tensors
            for name, tensor in [("obs", obs), ("actions", actions), ("logp_old", logp_old), 
                               ("values_old", values_old), ("market_features", market_features)]:
                if torch.any(torch.isnan(tensor)):
                    self.logger.error(f"NaN in stacked {name}")
                    tensor = torch.nan_to_num(tensor)
            
            # Compute advantages
            advantages, returns = self.compute_advantages(gamma)
            
            # Multiple epochs
            for epoch in range(4):
                try:
                    indices = torch.randperm(len(obs), device=self.device)
                    
                    # Mini-batch updates
                    batch_size = min(64, len(obs))
                    for i in range(0, len(obs), batch_size):
                        batch_idx = indices[i:i+batch_size]
                        
                        obs_batch = obs[batch_idx]
                        act_batch = actions[batch_idx]
                        logp_old_batch = logp_old[batch_idx]
                        adv_batch = advantages[batch_idx]
                        ret_batch = returns[batch_idx]
                        market_batch = market_features[batch_idx]
                        
                        # Forward pass
                        mu, value = self.forward(obs_batch, market_batch)
                        
                        # Recompute log probs
                        dist = torch.distributions.Normal(mu, 0.1)
                        logp = dist.log_prob(act_batch).sum(dim=-1)
                        
                        # PPO losses
                        ratio = (logp - logp_old_batch).exp()
                        
                        # Clamp ratio to prevent extreme values
                        ratio = torch.clamp(ratio, 0.1, 10.0)
                        
                        surr1 = ratio * adv_batch
                        surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_batch
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        value_clipped = values_old[batch_idx] + torch.clamp(
                            value.squeeze(-1) - values_old[batch_idx], -self.clip_eps, self.clip_eps
                        )
                        value_loss1 = F.mse_loss(value.squeeze(-1), ret_batch)
                        value_loss2 = F.mse_loss(value_clipped, ret_batch)
                        value_loss = torch.max(value_loss1, value_loss2)
                        
                        # Entropy
                        entropy_loss = -dist.entropy().mean()
                        
                        # Total loss
                        loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
                        
                        # Validate loss
                        if torch.isnan(loss):
                            self.logger.error("NaN loss detected, skipping update")
                            continue
                        
                        # Update
                        self.actor_opt.zero_grad()
                        self.critic_opt.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                        
                        self.actor_opt.step()
                        self.critic_opt.step()
                        
                except Exception as e:
                    self.logger.error(f"Error in epoch {epoch}, batch {i}: {e}")
                    continue
                    
            # Log stats
            total_reward = sum(r.item() for r in self.buffer["rewards"])
            self.logger.info(f"Episode complete: Reward={total_reward:.2f}, Steps={len(self.buffer['rewards'])}")
            
            # Clear buffers
            for k in self.buffer:
                self.buffer[k].clear()
                
        except Exception as e:
            self.logger.error(f"Error in end_episode: {e}")
            # Clear buffers even on error
            for k in self.buffer:
                self.buffer[k].clear()

    def get_observation_components(self) -> np.ndarray:
        """Return 4 components for MetaRLController compatibility"""
        try:
            # Validate last_action
            if np.any(np.isnan(self.last_action)):
                self.logger.error("NaN in last_action")
                self.last_action = np.nan_to_num(self.last_action)
            if np.isnan(self.position):
                self.position = 0.0
            if np.isnan(self.unrealized_pnl):
                self.unrealized_pnl = 0.0
                
            observation = np.array([
                float(self.last_action.mean()),
                float(self.last_action.std()),
                float(self.position),
                float(self.unrealized_pnl)
            ], dtype=np.float32)
            
            # Final validation
            if np.any(np.isnan(observation)):
                self.logger.error(f"NaN in observation components: {observation}")
                observation = np.nan_to_num(observation)
                
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def select_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Action selection with comprehensive validation"""
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                self.logger.error("NaN in observation tensor")
                obs_tensor = torch.nan_to_num(obs_tensor)
            
            # Pad observation if needed
            if obs_tensor.shape[-1] < self.extended_obs_size:
                padding = torch.zeros(
                    (*obs_tensor.shape[:-1], self.extended_obs_size - obs_tensor.shape[-1]),
                    device=obs_tensor.device, dtype=obs_tensor.dtype
                )
                obs_tensor = torch.cat([obs_tensor, padding], dim=-1)
            elif obs_tensor.shape[-1] > self.extended_obs_size:
                obs_tensor = obs_tensor[..., :self.extended_obs_size]
                
            # Extract market features
            market_features = obs_tensor[..., -self.lag_window*4:]
            
            with torch.no_grad():
                action, _ = self.forward(obs_tensor, market_features)
                
                # Validate action
                if torch.any(torch.isnan(action)):
                    self.logger.error("NaN in selected action")
                    action = torch.zeros_like(action)
                    
            return action
            
        except Exception as e:
            self.logger.error(f"Error in select_action: {e}")
            # Return safe default
            batch_size = obs_tensor.shape[0] if obs_tensor.ndim > 1 else 1
            return torch.zeros(batch_size, self.act_size, device=self.device)

    def reset(self):
        """Reset agent with comprehensive cleanup"""
        try:
            for k in self.buffer:
                self.buffer[k].clear()
            self.last_action = np.zeros(self.act_size, dtype=np.float32)
            self.position = 0.0
            self.unrealized_pnl = 0.0
            self.price_buffer.clear()
            self.volume_buffer.clear()
            self.spread_buffer.clear()
            self.vol_buffer.clear()
            self.running_adv_std = 1.0
            self._step_count = 0
            self.total_trades = 0
            self.logger.info("PPOLagAgent reset complete")
        except Exception as e:
            self.logger.error(f"Error in reset: {e}")

    def step(self, *args, **kwargs):
        pass

    def get_state(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "value_encoder": self.value_encoder.state_dict(),
            "market_encoder": self.market_encoder.state_dict(),
            "value_head": self.value_head.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "last_action": self.last_action.tolist(),
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "running_adv_std": self.running_adv_std,
            "total_trades": self.total_trades,
            "step_count": self._step_count,
        }

    def set_state(self, state: Dict, strict: bool = False):
        try:
            self.actor.load_state_dict(state["actor"], strict=strict)
            self.value_encoder.load_state_dict(state["value_encoder"], strict=strict)
            self.market_encoder.load_state_dict(state["market_encoder"], strict=strict)
            self.value_head.load_state_dict(state["value_head"], strict=strict)
            self.actor_opt.load_state_dict(state["actor_opt"])
            self.critic_opt.load_state_dict(state["critic_opt"])
            self.last_action = np.array(state.get("last_action", [0,0]), dtype=np.float32)
            self.position = state.get("position", 0.0)
            self.unrealized_pnl = state.get("unrealized_pnl", 0.0)
            self.running_adv_std = state.get("running_adv_std", 1.0)
            self.total_trades = state.get("total_trades", 0)
            self._step_count = state.get("step_count", 0)
            self.logger.info("PPOLagAgent state loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")

    def get_weights(self) -> Dict[str, any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": {
                "value_encoder": self.value_encoder.state_dict(),
                "market_encoder": self.market_encoder.state_dict(),
                "value_head": self.value_head.state_dict()
            }
        }

    def get_gradients(self) -> Dict[str, any]:
        grads = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.cpu().numpy()
        return grads
