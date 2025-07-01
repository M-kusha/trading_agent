# ─────────────────────────────────────────────────────────────
# modules/meta/ppo_agent.py

from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any
from modules.core.core import Module


class PPOAgent(nn.Module, Module):
    def __init__(self, obs_size, act_size=2, hidden_size=64, lr=3e-4, device="cpu", debug=True):
        super().__init__()
        self.device = torch.device(device)
        self.debug = debug
        self.obs_size = obs_size
        self.act_size = act_size

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"PPOAgent_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/ppo/ppo_agent.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        try:
            self.actor = nn.Sequential(
                nn.Linear(obs_size, hidden_size), 
                nn.Tanh(),
                nn.Linear(hidden_size, act_size), 
                nn.Tanh()
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_size, hidden_size), 
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
            
            # FIX: Proper initialization
            self._initialize_weights()
            
            self.to(self.device)
            self.opt = optim.Adam(self.parameters(), lr=lr)
            self.clip_eps = 0.2
            self.value_coeff = 0.5
            self.entropy_coeff = 0.01

            self.buffer = {k: [] for k in ["obs", "actions", "logp", "values", "rewards"]}
            self.last_action = np.zeros(act_size, dtype=np.float32)
            self._step_count = 0
            
            self.logger.info(f"PPOAgent initialized - obs_size={obs_size}, act_size={act_size}")
            
        except Exception as e:
            self.logger.error(f"Error initializing PPOAgent: {e}")
            raise

    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor):
        try:
            # Validate input
            if torch.any(torch.isnan(obs)):
                self.logger.error("NaN in observation")
                obs = torch.nan_to_num(obs)
                
            mu = self.actor(obs)
            value = self.critic(obs)
            
            # Validate outputs
            if torch.any(torch.isnan(mu)):
                self.logger.error("NaN in actor output")
                mu = torch.zeros_like(mu)
            if torch.any(torch.isnan(value)):
                self.logger.error("NaN in critic output")
                value = torch.zeros_like(value)
                
            return mu, value
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {e}")
            batch_size = obs.shape[0] if obs.ndim > 1 else 1
            return (torch.zeros(batch_size, self.act_size, device=self.device),
                    torch.zeros(batch_size, 1, device=self.device))

    def record_step(self, obs_vec, reward, **kwargs):
        """FIX: Accept market data kwargs for compatibility with enhanced validation"""
        self._step_count += 1
        
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                self.logger.error("NaN in observation vector")
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                self.logger.error("NaN reward, setting to 0")
                reward = 0.0
                
            obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                mu, value = self.forward(obs_t.unsqueeze(0))
                
            dist = torch.distributions.Normal(mu, 0.1)
            action = dist.rsample()
            logp = dist.log_prob(action).sum(dim=-1)
            
            # Validate action and logp
            if torch.any(torch.isnan(action)):
                self.logger.error("NaN in sampled action")
                action = torch.zeros_like(action)
            if torch.any(torch.isnan(logp)):
                self.logger.error("NaN in log probability")
                logp = torch.zeros_like(logp)

            self.buffer["obs"].append(obs_t)
            self.buffer["actions"].append(action.squeeze(0))
            self.buffer["logp"].append(logp.squeeze(0))
            self.buffer["values"].append(value.squeeze(0))
            self.buffer["rewards"].append(
                torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            )
            self.last_action = action.cpu().numpy().squeeze(0)
            
            self.logger.debug(f"Step {self._step_count} recorded: reward={reward:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error in record_step: {e}")

    def end_episode(self, gamma=0.99):
        try:
            if not self.buffer["rewards"]: 
                self.logger.warning("Empty episode buffer")
                return

            self.logger.info(f"Ending episode with {len(self.buffer['rewards'])} steps")

            obs = torch.stack(self.buffer["obs"])
            actions = torch.stack(self.buffer["actions"])
            logp_old = torch.stack(self.buffer["logp"])
            values = torch.stack(self.buffer["values"])
            rewards = torch.stack(self.buffer["rewards"])

            # Validate tensors
            for name, tensor in [("obs", obs), ("actions", actions), ("logp_old", logp_old), 
                               ("values", values), ("rewards", rewards)]:
                if torch.any(torch.isnan(tensor)):
                    self.logger.error(f"NaN in {name}")
                    tensor = torch.nan_to_num(tensor)

            returns = []
            R = 0.0
            for r in reversed(rewards.tolist()):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advantages = returns - values

            # Validate advantages
            if torch.any(torch.isnan(advantages)):
                self.logger.error("NaN in advantages")
                advantages = torch.nan_to_num(advantages)

            for epoch in range(4):
                try:
                    mu, value = self.forward(obs)
                    dist = torch.distributions.Normal(mu, 0.1)
                    logp = dist.log_prob(actions).sum(dim=-1)
                    ratio = (logp - logp_old).exp()
                    
                    # Clamp ratio
                    ratio = torch.clamp(ratio, 0.1, 10.0)
                    
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value.squeeze(-1), returns)
                    entropy_loss = -dist.entropy().mean()
                    loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
                    
                    # Validate loss
                    if torch.isnan(loss):
                        self.logger.error(f"NaN loss in epoch {epoch}, skipping")
                        continue
                        
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.opt.step()
                    
                except Exception as e:
                    self.logger.error(f"Error in epoch {epoch}: {e}")
                    continue

            total_reward = sum(r.item() for r in rewards)
            self.logger.info(f"Episode complete: reward={total_reward:.2f}")

            for k in self.buffer:
                self.buffer[k].clear()
                
        except Exception as e:
            self.logger.error(f"Error in end_episode: {e}")
            for k in self.buffer:
                self.buffer[k].clear()

    def get_observation_components(self):
        """FIX: Return 4 components for compatibility with validation"""
        try:
            if np.any(np.isnan(self.last_action)):
                self.logger.error("NaN in last_action")
                self.last_action = np.nan_to_num(self.last_action)
                
            observation = np.array([
                float(self.last_action.mean()), 
                float(self.last_action.std()),
                0.0,  # Placeholder for position
                0.0   # Placeholder for unrealized_pnl
            ], dtype=np.float32)
            
            if np.any(np.isnan(observation)):
                self.logger.error("NaN in observation components")
                observation = np.nan_to_num(observation)
                
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_state(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt": self.opt.state_dict(),
            "last_action": self.last_action.tolist(),
            "step_count": self._step_count,
        }

    def set_state(self, state, strict=False):
        try:
            self.actor.load_state_dict(state["actor"], strict=strict)
            self.critic.load_state_dict(state["critic"], strict=strict)
            self.opt.load_state_dict(state["opt"])
            self.last_action = np.array(state.get("last_action", [0,0]), dtype=np.float32)
            self._step_count = state.get("step_count", 0)
            self.logger.info("PPOAgent state loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")

    def reset(self):
        try:
            for k in self.buffer: 
                self.buffer[k].clear()
            self.last_action = np.zeros(self.act_size, dtype=np.float32)
            self._step_count = 0
            self.logger.info("PPOAgent reset complete")
        except Exception as e:
            self.logger.error(f"Error in reset: {e}")

    def step(self, *args, **kwargs):
        pass

    def select_action(self, obs_tensor):
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                self.logger.error("NaN in observation tensor")
                obs_tensor = torch.nan_to_num(obs_tensor)
                
            with torch.no_grad():
                action = self.actor(obs_tensor)
                
            # Validate output
            if torch.any(torch.isnan(action)):
                self.logger.error("NaN in action output")
                action = torch.zeros_like(action)
                
            return action
            
        except Exception as e:
            self.logger.error(f"Error in select_action: {e}")
            batch_size = obs_tensor.shape[0] if obs_tensor.ndim > 1 else 1
            return torch.zeros(batch_size, self.act_size, device=self.device)

    def get_gradients(self) -> Dict[str, Any]:
        grads = {}
        for name, param in self.named_parameters():
            grads[name] = param.grad.cpu().numpy() if param.grad is not None else None
        return grads

    def get_weights(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }
