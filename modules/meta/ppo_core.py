#!/usr/bin/env python3
"""
PPO Core - Pure RL Engine
=========================

This module contains the pure reinforcement learning components of PPO,
completely decoupled from SmartInfoBus, voting, and instrument logic.

Responsibilities:
- EnhancedPPONetwork: Neural network architecture
- PPOCore: Experience buffer, GAE, PPO update, action selection

Input: obs: np.ndarray
Output: actions: np.ndarray, value: float, log_prob: float

Version: 3.0.0 (Extracted from monolithic PPOAgent)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PPOCoreConfig:
    """
    Configuration for PPOCore (pure RL engine).
    
    Note: obs_size should be set to match your observation builder.
    The default of 64 matches PPO_OBS_SIZE v4.0 from ppo_observation_builder
    (includes world_model and trading_mode features).
    """
    # Network dimensions
    obs_size: int = 64
    act_size: int = 2             # (trust_score, position_size_score)
    hidden_size: int = 128
    
    # Device
    device: str = "cpu"
    
    # Learning rate
    learning_rate: float = 3e-4
    
    # PPO hyperparameters
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    
    # Buffer settings
    batch_size: int = 64          # Min samples before update
    buffer_size: int = 2048       # Max buffer size (soft cap)
    
    # Debug
    debug: bool = False


# ═══════════════════════════════════════════════════════════════════
# NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════

class EnhancedPPONetwork(nn.Module):
    """
    Enhanced PPO network with actor-critic architecture.
    
    Architecture:
    - Shared feature extractor (2 hidden layers with dropout)
    - Policy head (actor) - outputs mean and log_std for Gaussian policy
    - Value head (critic) - outputs state value estimate
    """

    def __init__(self, obs_size: int, act_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_size = hidden_size

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Policy head (actor) - outputs mean and log_std
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_size * 2),  # mean and log_std
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                # Pylance expects an int for "gain"; use a small integer
                nn.init.orthogonal_(mod.weight, gain=2)
                nn.init.zeros_(mod.bias)

        # Special initialization for policy output (smaller scale for stability)
        last = self.policy_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=1)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor of shape (batch, obs_size)

        Returns:
            action_mean: (batch, act_size) - mean of Gaussian policy
            action_log_std: (batch, act_size) - log std of Gaussian policy
            value: (batch, 1) - state value estimate
        """
        features = self.feature_extractor(obs)

        # Policy output
        policy_out = self.policy_head(features)
        half = policy_out.size(-1) // 2
        action_mean = policy_out[..., :half]
        action_log_std = policy_out[..., half:]
        action_log_std = torch.clamp(action_log_std, -20.0, 2.0)

        # Value output
        value = self.value_head(features)

        return action_mean, action_log_std, value
    
    def get_action_distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Get the action distribution for given observations."""
        action_mean, action_log_std, _ = self.forward(obs)
        action_std = torch.exp(action_log_std)
        return torch.distributions.Normal(action_mean, action_std)


# ═══════════════════════════════════════════════════════════════════
# PPO CORE ENGINE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PPOCoreStats:
    """Statistics from PPO training."""
    total_updates: int = 0
    episodes_completed: int = 0
    best_episode_reward: float = -np.inf
    avg_episode_reward: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    gradient_norm: float = 0.0
    explained_variance: float = 0.0
    learning_rate: float = 3e-4


class PPOCore:
    """
    Pure PPO RL engine - no SmartInfoBus, no voting, no instruments.
    
    This class handles:
    - Neural network management
    - Experience buffer
    - GAE advantage computation
    - PPO policy updates
    - Action selection (forward pass)
    
    Usage:
        core = PPOCore(config)
        
        # Action selection
        action, log_prob, value = core.select_action(obs)
        
        # Record experience
        core.record_step(obs, action, reward, done, log_prob, value)
        
        # Update policy
        stats = core.update()
    """
    
    def __init__(self, config: Optional[PPOCoreConfig] = None) -> None:
        self.config: PPOCoreConfig = config or PPOCoreConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize network
        self.network = EnhancedPPONetwork(
            obs_size=self.config.obs_size,
            act_size=self.config.act_size,
            hidden_size=self.config.hidden_size,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
            weight_decay=1e-4,
        )
        
        # Learning rate scheduler (Reduce on plateau of episode reward)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.8,
            patience=50,
        )
        
        # Experience buffer
        self.buffer: Dict[str, List[Any]] = {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "advantages": [],
            "returns": [],
        }
        
        # Statistics
        self.stats: PPOCoreStats = PPOCoreStats(learning_rate=self.config.learning_rate)
        self.episode_rewards: deque[float] = deque(maxlen=100)
        self.episode_lengths: deque[int] = deque(maxlen=100)
        
        # Action tracking for convenience (used by shell / training path)
        self.last_action: np.ndarray = np.zeros(self.config.act_size, dtype=np.float32)
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0
    
    # ─────────────────────────────────────────────────────────────
    # Action Selection
    # ─────────────────────────────────────────────────────────────
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Ensure observation is 1D float32 of correct length, padding/truncating as needed.
        """
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.config.obs_size:
            new_obs = np.zeros(self.config.obs_size, dtype=np.float32)
            copy_size = min(arr.shape[0], self.config.obs_size)
            new_obs[:copy_size] = arr[:copy_size]
            arr = new_obs
        return arr
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given observation.
        
        Args:
            obs: Observation array of shape (obs_size,) or compatible
            deterministic: If True, use mean action instead of sampling
        
        Returns:
            action: Action array of shape (act_size,)
            log_prob: Log probability of the action
            value: State value estimate
        """
        obs_arr = self._normalize_obs(obs)
        obs_tensor = torch.from_numpy(obs_arr).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, action_log_std, value = self.network(obs_tensor)
            action_std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            if deterministic:
                action_tensor = action_mean
            else:
                action_tensor = dist.sample()
            
            log_prob_tensor = dist.log_prob(action_tensor).sum(dim=-1)
            
            # Convert to numpy / scalars
            action_np = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            log_prob_np = float(log_prob_tensor.item())
            value_np = float(value.squeeze().item())
        
        # Store for record_step fallback
        self.last_action = action_np
        self._last_log_prob = log_prob_np
        self._last_value = value_np
        
        return action_np, log_prob_np, value_np
    
    def get_value(self, obs: np.ndarray) -> float:
        """Get value estimate for observation without selecting action."""
        obs_arr = self._normalize_obs(obs)
        obs_tensor = torch.from_numpy(obs_arr).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            _, _, value = self.network(obs_tensor)
            return float(value.squeeze().item())
    
    # ─────────────────────────────────────────────────────────────
    # Experience Recording
    # ─────────────────────────────────────────────────────────────
    
    def record_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: Optional[float] = None,
        value: Optional[float] = None,
    ) -> None:
        """
        Record a single step of experience.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            done: Episode terminated?
            log_prob: Log probability (uses stored value if None)
            value: Value estimate (uses stored value if None)
        """
        # Soft capacity guard to avoid unbounded growth
        if len(self.buffer["observations"]) >= self.config.buffer_size:
            # Oldest experiences are dropped; training loop is expected to
            # call update() frequently enough to avoid heavy truncation.
            for key in ("observations", "actions", "rewards", "dones", "log_probs", "values"):
                if self.buffer[key]:
                    self.buffer[key].pop(0)
        
        self.buffer["observations"].append(self._normalize_obs(obs))
        self.buffer["actions"].append(np.asarray(action, dtype=np.float32))
        self.buffer["rewards"].append(float(reward))
        self.buffer["dones"].append(float(done))
        self.buffer["log_probs"].append(
            float(log_prob) if log_prob is not None else self._last_log_prob
        )
        self.buffer["values"].append(
            float(value) if value is not None else self._last_value
        )
    
    def end_episode(self, final_reward: Optional[float] = None) -> None:
        """
        Mark end of episode and record statistics.
        
        Args:
            final_reward: Optional final episode reward (computed from buffer if None)
        """
        if self.buffer["rewards"]:
            if final_reward is not None:
                episode_reward = float(final_reward)
            else:
                episode_reward = float(sum(self.buffer["rewards"]))
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(self.buffer["rewards"]))
            self.stats.episodes_completed += 1
            
            if episode_reward > self.stats.best_episode_reward:
                self.stats.best_episode_reward = episode_reward
            
            if self.episode_rewards:
                recent = list(self.episode_rewards)[-10:]
                self.stats.avg_episode_reward = float(np.mean(recent))
    
    # ─────────────────────────────────────────────────────────────
    # Policy Update
    # ─────────────────────────────────────────────────────────────
    
    def should_update(self) -> bool:
        """Check if buffer has enough samples for policy update."""
        return len(self.buffer["observations"]) >= self.config.batch_size
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform PPO policy update.
        
        Returns:
            Dictionary with training statistics, or None if not enough samples.
        """
        if not self.should_update():
            return None  # Not enough samples
        
        # Compute GAE advantages and returns
        self._compute_gae()
        
        # Convert buffer to tensors
        observations = torch.tensor(
            np.array(self.buffer["observations"], dtype=np.float32),
            dtype=torch.float32,
        ).to(self.device)
        actions = torch.tensor(
            np.array(self.buffer["actions"], dtype=np.float32),
            dtype=torch.float32,
        ).to(self.device)
        old_log_probs = torch.tensor(
            np.array(self.buffer["log_probs"], dtype=np.float32),
            dtype=torch.float32,
        ).to(self.device)
        advantages = torch.tensor(
            np.array(self.buffer["advantages"], dtype=np.float32),
            dtype=torch.float32,
        ).to(self.device)
        returns = torch.tensor(
            np.array(self.buffer["returns"], dtype=np.float32),
            dtype=torch.float32,
        ).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        grad_norm = 0.0
        values_tensor: Optional[torch.Tensor] = None
        
        for _ in range(self.config.ppo_epochs):
            # Forward pass
            action_mean, action_log_std, values_tensor = self.network(observations)
            action_std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            # New log probs and entropy
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # PPO clipped surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.clip_eps,
                1.0 + self.config.clip_eps,
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (guard against optional values_tensor for type checkers)
            assert values_tensor is not None
            value_loss = F.mse_loss(values_tensor.squeeze(), returns)
            
            # Entropy loss (negative because we want to maximize entropy)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (
                policy_loss
                + self.config.value_coeff * value_loss
                + self.config.entropy_coeff * entropy_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.max_grad_norm,
            )
            grad_norm = float(grad_norm_t)
            
            self.optimizer.step()
            
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy_loss += float(entropy_loss.item())
        
        # Average losses
        n_epochs = float(self.config.ppo_epochs)
        avg_policy_loss = total_policy_loss / n_epochs
        avg_value_loss = total_value_loss / n_epochs
        avg_entropy_loss = total_entropy_loss / n_epochs
        
        # Explained variance
        explained_var = 0.0
        if values_tensor is not None:
            with torch.no_grad():
                var_y = torch.var(returns)
                var_diff = torch.var(returns - values_tensor.squeeze())
                explained_var = float(1.0 - var_diff / (var_y + 1e-8))
        
        # Update stats
        self.stats.total_updates += 1
        self.stats.policy_loss = avg_policy_loss
        self.stats.value_loss = avg_value_loss
        self.stats.entropy_loss = avg_entropy_loss
        self.stats.gradient_norm = grad_norm
        self.stats.explained_variance = explained_var
        self.stats.learning_rate = float(self.optimizer.param_groups[0]["lr"])
        
        # Clear buffer
        self._clear_buffer()
        
        # Update learning rate scheduler
        if self.episode_rewards:
            self.lr_scheduler.step(self.stats.avg_episode_reward)
        
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "gradient_norm": grad_norm,
            "explained_variance": explained_var,
            "total_updates": self.stats.total_updates,
        }
    
    def _compute_gae(self) -> None:
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.buffer["rewards"], dtype=np.float32)
        values = np.array(self.buffer["values"], dtype=np.float32)
        dones = np.array(self.buffer["dones"], dtype=np.float32)
        
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            last_gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
            )
            advantages[t] = last_gae
        
        returns = advantages + values
        
        self.buffer["advantages"] = advantages.tolist()
        self.buffer["returns"] = returns.tolist()
    
    def _clear_buffer(self) -> None:
        """Clear experience buffer."""
        for key in self.buffer:
            self.buffer[key] = []
    
    # ─────────────────────────────────────────────────────────────
    # State Management
    # ─────────────────────────────────────────────────────────────
    
    def get_state(self) -> Dict[str, Any]:
        """Get full state for persistence."""
        return {
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "stats": {
                "total_updates": self.stats.total_updates,
                "episodes_completed": self.stats.episodes_completed,
                "best_episode_reward": self.stats.best_episode_reward,
                "avg_episode_reward": self.stats.avg_episode_reward,
                "learning_rate": self.stats.learning_rate,
            },
            # Keep config export minimal and stable
            "config": {
                "obs_size": self.config.obs_size,
                "act_size": self.config.act_size,
                "hidden_size": self.config.hidden_size,
            },
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        if "network_state" in state:
            try:
                self.network.load_state_dict(state["network_state"])
            except Exception:
                # Shape mismatch or partial load; skip silently
                pass
        
        if "optimizer_state" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state"])
            except Exception:
                pass
        
        if "scheduler_state" in state:
            try:
                self.lr_scheduler.load_state_dict(state["scheduler_state"])
            except Exception:
                pass
        
        if "stats" in state:
            stats = state["stats"]
            self.stats.total_updates = stats.get("total_updates", 0)
            self.stats.episodes_completed = stats.get("episodes_completed", 0)
            self.stats.best_episode_reward = stats.get("best_episode_reward", -np.inf)
            self.stats.avg_episode_reward = stats.get("avg_episode_reward", 0.0)
            self.stats.learning_rate = stats.get("learning_rate", self.config.learning_rate)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save(self.get_state(), path)
    
    def load(self, path: str) -> None:
        """Load model from file."""
        state = torch.load(path, map_location=self.device)
        self.set_state(state)
