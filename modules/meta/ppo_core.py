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

Version: 3.1.0 (Mini-batch PPO, fixed episode stats, rich training stats)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

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
    
    ACTION SEMANTICS (v4.1 - Autonomous PPO):
    ===========================================
    PPO outputs a 2D continuous action in [-1, 1]:
    
    action[0] = direction_score ∈ [-1.0, 1.0]
        - > +0.3 ⇒ LONG signal
        - < -0.3 ⇒ SHORT signal  
        - |score| ≤ 0.3 ⇒ FLAT (uncertain/no position)
        - Magnitude indicates conviction strength
    
    action[1] = size_score ∈ [-1.0, 1.0]
        - Mapped to [0.0, 1.0] for position sizing
        - Then scaled by risk/memory/mode gates
    
    The direction_score is the PRIMARY autonomous signal from PPO.
    In training, PPO learns direction entirely from market observations.
    In live trading, direction is blended with experts based on autonomy phase.
    """

    # Network dimensions
    obs_size: int = 64
    act_size: int = 2  # (direction_score, size_score)
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
    gamma: float = 0.95  # SHORT-TERM: ~5h horizon matches M15 + ExitEngine timeouts
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4

    # Buffer / update settings
    batch_size: int = 64  # Minimum samples before update AND mini-batch size
    buffer_size: int = 2048  # Soft cap on buffer length

    # Direction thresholds (for interpreting direction_score)
    direction_long_threshold: float = 0.3   # score > this = LONG
    direction_short_threshold: float = -0.3  # score < this = SHORT

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

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        hidden_size: int = 128,
    ) -> None:
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
    """High-level statistics from PPO training."""
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
    - PPO policy updates (mini-batch, multi-epoch)
    - Action selection (forward pass)

    Usage:
        core = PPOCore(config)

        # Action selection
        action, log_prob, value = core.select_action(obs)

        # Record experience
        core.record_step(obs, action, reward, done, log_prob, value)

        # Mark episode end (for stats only)
        if done:
            core.end_episode()

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
        self.stats: PPOCoreStats = PPOCoreStats(
            learning_rate=self.config.learning_rate
        )
        self.episode_rewards: Deque[float] = deque(maxlen=100)
        self.episode_lengths: Deque[int] = deque(maxlen=100)

        # Episode segmentation within the buffer
        self._episode_start_index: int = 0

        # Action / value tracking (for record_step defaults)
        self.last_action: np.ndarray = np.zeros(
            self.config.act_size,
            dtype=np.float32,
        )
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0

        # Training diagnostics for external shells (e.g. PPOAgentShell)
        self._training_stats: Dict[str, Any] = {}
        self._recent_rewards: Deque[float] = deque(maxlen=1000)
        self._total_steps: int = 0
                # Optional SB3 policy backend (inference-only).
        # When loaded, select_action() uses SB3 PPO.predict() instead of the internal torch network.
        self._sb3_model: Optional[Any] = None
        self._sb3_model_path: Optional[str] = None
        self._sb3_instruments: List[str] = []

    @staticmethod
    def _norm_symbol(sym: Any) -> str:
        if not isinstance(sym, str):
            return ""
        return sym.upper().replace("/", "").replace("_", "").replace("-", "")

    def set_instruments(self, instruments: List[str]) -> None:
        """Set instrument ordering for multi-instrument SB3 action slicing."""
        try:
            self._sb3_instruments = [self._norm_symbol(s) for s in (instruments or []) if s]
        except Exception:
            self._sb3_instruments = []


        # Optional SB3 policy backend (inference-only).
        # When loaded, select_action() uses SB3 PPO.predict() instead of the internal torch network.
        self._sb3_model: Optional[Any] = None
        self._sb3_model_path: Optional[str] = None
        self._sb3_instruments: List[str] = []

    @staticmethod
    def _norm_symbol(sym: Any) -> str:
        if not isinstance(sym, str):
            return ""
        return sym.upper().replace("/", "").replace("_", "").replace("-", "")

    def set_instruments(self, instruments: List[str]) -> None:
        """Set instrument ordering for multi-instrument SB3 action slicing."""
        try:
            self._sb3_instruments = [self._norm_symbol(s) for s in (instruments or []) if s]
        except Exception:
            self._sb3_instruments = []

        # Optional SB3 policy backend (inference-only).
        # When loaded, select_action() uses SB3 PPO.predict() instead of the internal torch network.
        self._sb3_model: Optional[Any] = None
        self._sb3_model_path: Optional[str] = None
        self._sb3_instruments: List[str] = []

    @staticmethod
    def _norm_symbol(sym: Any) -> str:
        if not isinstance(sym, str):
            return ""
        return sym.upper().replace("/", "").replace("_", "").replace("-", "")

    def set_instruments(self, instruments: List[str]) -> None:
        """Set instrument ordering for multi-instrument SB3 action slicing."""
        try:
            self._sb3_instruments = [self._norm_symbol(s) for s in (instruments or []) if s]
        except Exception:
            self._sb3_instruments = []

    # ─────────────────────────────────────────────────────────────
    # Action Selection
    # ─────────────────────────────────────────────────────────────

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Make sure observation is 1D float32 of correct length,
        padding/truncating as needed.
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
        instrument: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given observation.
        
        ACTION SEMANTICS (v4.1 - Autonomous PPO):
        =========================================
        Returns action array of shape (2,) with values in [-1, 1]:
        
        action[0] = direction_score:
            - > +0.3 ⇒ LONG signal
            - < -0.3 ⇒ SHORT signal
            - |score| ≤ 0.3 ⇒ FLAT
            
        action[1] = size_score:
            - Raw position size signal in [-1, 1]
            - Mapped to [0, 1] by caller: (size_score + 1) / 2
            - Then scaled by risk/memory/mode gates

        Args:
            obs: Observation array of shape (obs_size,) or compatible
            deterministic: If True, use mean action instead of sampling

        Returns:
            action: Action array of shape (act_size,) with values in [-1, 1]
            log_prob: Log probability of the action
            value: State value estimate
        """
        # SB3 backend (inference-only)
        # IMPORTANT: normalize to the SB3 model's observation_space size (not PPOCoreConfig.obs_size),
        # otherwise misconfigured configs can silently truncate/pad and skew inference or crash predict().
        if self._sb3_model is not None:
            try:
                obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                try:
                    shape = getattr(self._sb3_model.observation_space, "shape", None)
                    expected = int(shape[0]) if shape and len(shape) == 1 else None
                except Exception:
                    expected = None

                if expected is not None and obs_arr.shape[0] != expected:
                    fixed = np.zeros(expected, dtype=np.float32)
                    copy_size = min(obs_arr.shape[0], expected)
                    fixed[:copy_size] = obs_arr[:copy_size]
                    obs_arr = fixed

                action_full, _ = self._sb3_model.predict(obs_arr, deterministic=deterministic)
                action_full_arr = np.asarray(action_full, dtype=np.float32).reshape(-1)
                action_np = self._slice_sb3_action(action_full_arr, instrument=instrument)
                action_np = np.clip(action_np, -1.0, 1.0).astype(np.float32)

                # SB3 predict() doesn't expose log_prob/value; keep placeholders.
                log_prob_np = 0.0
                value_np = 0.0

                self.last_action = action_np
                self._last_log_prob = log_prob_np
                self._last_value = value_np
                return action_np, log_prob_np, value_np
            except Exception:
                # Fall back to the internal torch policy if SB3 predict fails.
                pass

        obs_arr = self._normalize_obs(obs)
                # SB3 backend (inference-only)
        if self._sb3_model is not None:
            try:
                action_full, _ = self._sb3_model.predict(obs_arr, deterministic=deterministic)
                action_full_arr = np.asarray(action_full, dtype=np.float32).reshape(-1)
                action_np = self._slice_sb3_action(action_full_arr, instrument=instrument)
                action_np = np.clip(action_np, -1.0, 1.0).astype(np.float32)

                # SB3 predict() doesn't expose log_prob/value; keep placeholders.
                log_prob_np = 0.0
                value_np = 0.0

                self.last_action = action_np
                self._last_log_prob = log_prob_np
                self._last_value = value_np
                return action_np, log_prob_np, value_np
            except Exception:
                # Fall back to the internal torch policy if SB3 predict fails.
                pass


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
            
            # CRITICAL: Clip actions to [-1, 1] range
            # The Gaussian policy can output values outside this range, but:
            # - direction_score must be in [-1, 1] for threshold logic to work
            # - Scores > 1.0 would always trigger LONG, scores < -1.0 always SHORT
            # - This ensures proper balance between LONG/SHORT/FLAT decisions
            action_np = np.clip(action_np, -1.0, 1.0)
            
            log_prob_np = float(log_prob_tensor.item())
            value_np = float(value.squeeze().item())

        # Store for record_step fallback
        self.last_action = action_np
        self._last_log_prob = log_prob_np
        self._last_value = value_np

        return action_np, log_prob_np, value_np
    def _slice_sb3_action(self, action: np.ndarray, instrument: Optional[str]) -> np.ndarray:
        """
        Convert an SB3 multi-instrument action vector into the 2D (direction_score, size_score)
        slice expected by ArbiterLogic for a single instrument.
        """
        try:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.zeros(0, dtype=np.float32)

        if arr.size <= 0:
            return np.zeros(self.config.act_size, dtype=np.float32)

        # If already a single-instrument action, trim/pad to act_size.
        if arr.size <= self.config.act_size:
            out = np.zeros(self.config.act_size, dtype=np.float32)
            out[: min(arr.size, self.config.act_size)] = arr[: self.config.act_size]
            return out

        insts = self._sb3_instruments
        if insts and arr.size >= 2 * len(insts):
            idx = 0
            if instrument:
                norm = self._norm_symbol(instrument)
                try:
                    idx = insts.index(norm)
                except ValueError:
                    idx = 0

            start = 2 * idx
            if start + 2 <= arr.size:
                return arr[start : start + 2]

        # Fallback: treat first two dims as (direction, size)
        return arr[:2]

    def _slice_sb3_action(self, action: np.ndarray, instrument: Optional[str]) -> np.ndarray:
        """
        Convert an SB3 multi-instrument action vector into the 2D (direction_score, size_score)
        slice expected by ArbiterLogic for a single instrument.
        """
        try:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.zeros(0, dtype=np.float32)

        if arr.size <= 0:
            return np.zeros(self.config.act_size, dtype=np.float32)

        # If already a single-instrument action, trim/pad to act_size.
        if arr.size <= self.config.act_size:
            out = np.zeros(self.config.act_size, dtype=np.float32)
            out[: min(arr.size, self.config.act_size)] = arr[: self.config.act_size]
            return out

        insts = self._sb3_instruments
        if insts and arr.size >= 2 * len(insts):
            idx = 0
            if instrument:
                norm = self._norm_symbol(instrument)
                try:
                    idx = insts.index(norm)
                except ValueError:
                    idx = 0

            start = 2 * idx
            if start + 2 <= arr.size:
                return arr[start : start + 2]

        # Fallback: treat first two dims as (direction, size)
        return arr[:2]

    def _slice_sb3_action(self, action: np.ndarray, instrument: Optional[str]) -> np.ndarray:
        """
        Convert an SB3 multi-instrument action vector into the 2D (direction_score, size_score)
        slice expected by ArbiterLogic for a single instrument.
        """
        try:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.zeros(0, dtype=np.float32)

        if arr.size <= 0:
            return np.zeros(self.config.act_size, dtype=np.float32)

        # If already a single-instrument action, trim/pad to act_size.
        if arr.size <= self.config.act_size:
            out = np.zeros(self.config.act_size, dtype=np.float32)
            out[: min(arr.size, self.config.act_size)] = arr[: self.config.act_size]
            return out

        insts = self._sb3_instruments
        if insts and arr.size >= 2 * len(insts):
            idx = 0
            if instrument:
                norm = self._norm_symbol(instrument)
                try:
                    idx = insts.index(norm)
                except ValueError:
                    idx = 0

            start = 2 * idx
            if start + 2 <= arr.size:
                return arr[start : start + 2]

        # Fallback: treat first two dims as (direction, size)
        return arr[:2]

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
            # Drop oldest experience across all keys
            for key in ("observations", "actions", "rewards", "dones", "log_probs", "values"):
                if self.buffer[key]:
                    self.buffer[key].pop(0)
            # Episode indexing remains conservative: _episode_start_index
            # may point earlier than actual start, but that is harmless.

        obs_norm = self._normalize_obs(obs)

        self.buffer["observations"].append(obs_norm)
        self.buffer["actions"].append(np.asarray(action, dtype=np.float32))
        self.buffer["rewards"].append(float(reward))
        self.buffer["dones"].append(float(done))
        self.buffer["log_probs"].append(
            float(log_prob) if log_prob is not None else self._last_log_prob
        )
        self.buffer["values"].append(
            float(value) if value is not None else self._last_value
        )

        self._recent_rewards.append(float(reward))
        self._total_steps += 1

    def end_episode(self, final_reward: Optional[float] = None) -> None:
        """
        Mark end of episode and record statistics.

        This uses the portion of the buffer **since the last episode end**
        (tracked by _episode_start_index) so that episode stats are not
        contaminated by previous episodes.

        Args:
            final_reward: Optional final episode reward (if precomputed).
        """
        start_idx = self._episode_start_index
        end_idx = len(self.buffer["rewards"])

        if end_idx <= start_idx:
            return  # no new steps since last episode

        rewards_segment = self.buffer["rewards"][start_idx:end_idx]

        if final_reward is not None:
            episode_reward = float(final_reward)
        else:
            episode_reward = float(sum(rewards_segment))

        episode_length = end_idx - start_idx

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.stats.episodes_completed += 1

        if episode_reward > self.stats.best_episode_reward:
            self.stats.best_episode_reward = episode_reward

        if self.episode_rewards:
            recent = list(self.episode_rewards)[-10:]
            self.stats.avg_episode_reward = float(np.mean(recent))

        # Next episode starts at current buffer end
        self._episode_start_index = end_idx

    # ─────────────────────────────────────────────────────────────
    # Policy Update
    # ─────────────────────────────────────────────────────────────

    def should_update(self) -> bool:
        """Check if buffer has enough samples for policy update."""
        return len(self.buffer["observations"]) >= self.config.batch_size

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform PPO policy update (mini-batch, multi-epoch).

        Returns:
            Dictionary with training statistics, or None if not enough samples.
        """
        if not self.should_update():
            return None  # Not enough samples

        # Compute GAE advantages and returns
        self._compute_gae()

        # Convert buffer to tensors (full batch)
        observations = torch.as_tensor(
            np.array(self.buffer["observations"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.array(self.buffer["actions"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            np.array(self.buffer["log_probs"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.as_tensor(
            np.array(self.buffer["advantages"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.as_tensor(
            np.array(self.buffer["returns"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = observations.size(0)
        mini_batch_size = min(self.config.batch_size, num_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_grad_norm = 0.0
        num_updates = 0

        # PPO update loop with mini-batches
        indices = torch.arange(num_samples, device=self.device)

        for _ in range(self.config.ppo_epochs):
            permutation = indices[torch.randperm(num_samples, device=self.device)]

            for start in range(0, num_samples, mini_batch_size):
                end = start + mini_batch_size
                batch_idx = permutation[start:end]

                obs_b = observations[batch_idx]
                act_b = actions[batch_idx]
                adv_b = advantages[batch_idx]
                ret_b = returns[batch_idx]
                old_log_b = old_log_probs[batch_idx]

                action_mean, action_log_std, values_b = self.network(obs_b)
                action_std = torch.exp(action_log_std)
                dist = torch.distributions.Normal(action_mean, action_std)

                new_log_probs = dist.log_prob(act_b).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - old_log_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_b.squeeze(), ret_b)

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    + self.config.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )
                grad_norm = float(grad_norm_t)

                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy_loss += float(entropy_loss.item())
                total_grad_norm += grad_norm
                num_updates += 1

        # Averages across all mini-batches and epochs
        denom = max(num_updates, 1)
        avg_policy_loss = total_policy_loss / denom
        avg_value_loss = total_value_loss / denom
        avg_entropy_loss = total_entropy_loss / denom
        avg_grad_norm = total_grad_norm / denom

        # Explained variance: fresh forward pass on full batch
        with torch.no_grad():
            _, _, values_eval = self.network(observations)
            v_pred = values_eval.squeeze()
            var_y = torch.var(returns)
            var_diff = torch.var(returns - v_pred)
            explained_var = float(1.0 - var_diff / (var_y + 1e-8))

        # Update stats
        self.stats.total_updates += 1
        self.stats.policy_loss = avg_policy_loss
        self.stats.value_loss = avg_value_loss
        self.stats.entropy_loss = avg_entropy_loss
        self.stats.gradient_norm = avg_grad_norm
        self.stats.explained_variance = explained_var
        self.stats.learning_rate = float(self.optimizer.param_groups[0]["lr"])

        # Update learning rate scheduler
        if self.episode_rewards:
            self.lr_scheduler.step(self.stats.avg_episode_reward)

        # Build training_stats snapshot for external consumers (e.g. PPOAgentShell)
        avg_episode_length = (
            int(float(np.mean(self.episode_lengths)))
            if self.episode_lengths
            else 0
        )

        self._training_stats = {
            "total_steps": self._total_steps,
            "episodes": self.stats.episodes_completed,
            "best_episode_reward": self.stats.best_episode_reward,
            "avg_reward": self.stats.avg_episode_reward,
            "avg_episode_length": avg_episode_length,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "gradient_norm": avg_grad_norm,
            "explained_variance": explained_var,
            "total_updates": self.stats.total_updates,
            "learning_rate": self.stats.learning_rate,
        }

        # Clear buffer after update
        self._clear_buffer()

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "gradient_norm": avg_grad_norm,
            "explained_variance": explained_var,
            "total_updates": self.stats.total_updates,
        }

    def _compute_gae(self) -> None:
        """Compute Generalized Advantage Estimation over the current buffer."""
        rewards = np.array(self.buffer["rewards"], dtype=np.float32)
        values = np.array(self.buffer["values"], dtype=np.float32)
        dones = np.array(self.buffer["dones"], dtype=np.float32)

        n = len(rewards)
        if n == 0:
            self.buffer["advantages"] = []
            self.buffer["returns"] = []
            return

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        # We treat non-terminal last state as bootstrapped from itself
        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            if t == n - 1:
                next_value = values[t]  # bootstrap when trajectory is truncated
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * non_terminal - values[t]
            last_gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * non_terminal * last_gae
            )
            advantages[t] = last_gae

        returns = advantages + values

        self.buffer["advantages"] = advantages.tolist()
        self.buffer["returns"] = returns.tolist()

    def _clear_buffer(self) -> None:
        """Clear experience buffer (keeps episode stats and counters)."""
        for key in self.buffer:
            self.buffer[key] = []
        # When buffer is cleared, next episode (for end_episode) starts at 0
        self._episode_start_index = 0

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
            "counters": {
                "total_steps": self._total_steps,
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
            self.stats.learning_rate = stats.get(
                "learning_rate",
                self.config.learning_rate,
            )

        if "counters" in state:
            counters = state["counters"]
            self._total_steps = counters.get("total_steps", self._total_steps)

    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save(self.get_state(), path)

    def load(self, path: str) -> None:
        """Load model from file."""
        if str(path).lower().endswith(".zip"):
            # SB3 model (ModernTradingEnv training output)
            from stable_baselines3 import PPO as SB3PPO  # type: ignore[import-not-found]

            self._sb3_model = SB3PPO.load(path, device="cpu")
            self._sb3_model_path = str(path)
            return

        # Native torch PPOCore checkpoint
        self._sb3_model = None
        self._sb3_model_path = None
        state = torch.load(path, map_location=self.device)
        self.set_state(state)