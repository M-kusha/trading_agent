#!/usr/bin/env python3

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from sb3_contrib import MaskablePPO
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MaskablePPO = None  # type: ignore
    MASKABLE_PPO_AVAILABLE = False


@dataclass
class PPOCoreConfig:


    obs_size: int = 84
    act_size: int = 2
    hidden_size: int = 128


    device: str = "cpu"


    learning_rate: float = 3e-4


    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.95
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4


    batch_size: int = 64
    buffer_size: int = 2048


    direction_long_threshold: float = 0.3
    direction_short_threshold: float = -0.3


    size_buckets: Tuple[float, ...] = (0.35, 0.60, 0.85, 1.10)


    @property
    def n_discrete_actions(self) -> int:
        return 2 * len(self.size_buckets) + 2


    debug: bool = False


@dataclass
class DiscreteActionDecoded:
    intent: str
    size_mult: float
    action_id: int

    def to_continuous(self) -> Tuple[float, float]:
        if self.intent == "long":
            direction_score = 0.7
        elif self.intent == "short":
            direction_score = -0.7
        else:
            direction_score = 0.0


        if self.size_mult > 0:

            size_score = (self.size_mult - 0.725) / 0.375
            size_score = float(np.clip(size_score, -1.0, 1.0))
        else:
            size_score = 0.0

        return direction_score, size_score


def decode_discrete_action(action_id: int, size_buckets: Tuple[float, ...]) -> DiscreteActionDecoded:
    K = len(size_buckets)
    ACTION_HOLD = 0
    ACTION_LONG_START = 1
    ACTION_SHORT_START = 1 + K
    ACTION_CLOSE = 1 + 2 * K

    a = int(action_id)

    if a == ACTION_HOLD:
        return DiscreteActionDecoded("hold", 0.0, a)
    elif a == ACTION_CLOSE:
        return DiscreteActionDecoded("close", 0.0, a)
    elif ACTION_LONG_START <= a < ACTION_SHORT_START:
        idx = a - ACTION_LONG_START
        return DiscreteActionDecoded("long", float(size_buckets[idx]), a)
    elif ACTION_SHORT_START <= a < ACTION_CLOSE:
        idx = a - ACTION_SHORT_START
        return DiscreteActionDecoded("short", float(size_buckets[idx]), a)
    else:
        return DiscreteActionDecoded("hold", 0.0, a)


class EnhancedPPONetwork(nn.Module):

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


        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )


        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_size * 2),
        )


        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.orthogonal_(mod.weight, gain=2)
                nn.init.zeros_(mod.bias)


        last = self.policy_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=1)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(obs)


        policy_out = self.policy_head(features)
        half = policy_out.size(-1) // 2
        action_mean = policy_out[..., :half]
        action_log_std = policy_out[..., half:]
        action_log_std = torch.clamp(action_log_std, -20.0, 2.0)


        value = self.value_head(features)

        return action_mean, action_log_std, value

    def get_action_distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        action_mean, action_log_std, _ = self.forward(obs)
        action_std = torch.exp(action_log_std)
        return torch.distributions.Normal(action_mean, action_std)


@dataclass
class PPOCoreStats:
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

    def __init__(self, config: Optional[PPOCoreConfig] = None) -> None:
        self.config: PPOCoreConfig = config or PPOCoreConfig()
        self.device = torch.device(self.config.device)


        self.network = EnhancedPPONetwork(
            obs_size=self.config.obs_size,
            act_size=self.config.act_size,
            hidden_size=self.config.hidden_size,
        ).to(self.device)


        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
            weight_decay=1e-4,
        )


        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.8,
            patience=50,
        )


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


        self.stats: PPOCoreStats = PPOCoreStats(
            learning_rate=self.config.learning_rate
        )
        self.episode_rewards: Deque[float] = deque(maxlen=100)
        self.episode_lengths: Deque[int] = deque(maxlen=100)


        self._episode_start_index: int = 0


        self.last_action: np.ndarray = np.zeros(
            self.config.act_size,
            dtype=np.float32,
        )
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0


        self._training_stats: Dict[str, Any] = {}
        self._recent_rewards: Deque[float] = deque(maxlen=1000)
        self._total_steps: int = 0


        self._sb3_model: Optional[Any] = None
        self._sb3_model_path: Optional[str] = None
        self._sb3_instruments: List[str] = []


        self._is_maskable_ppo: bool = False
        self._action_mask_fn: Optional[Callable[[], np.ndarray]] = None
        self._last_discrete_action: Optional[DiscreteActionDecoded] = None

    @staticmethod
    def _norm_symbol(sym: Any) -> str:
        if not isinstance(sym, str):
            return ""
        return sym.upper().replace("/", "").replace("_", "").replace("-", "")

    def set_instruments(self, instruments: List[str]) -> None:
        try:
            self._sb3_instruments = [self._norm_symbol(s) for s in (instruments or []) if s]
        except Exception:
            self._sb3_instruments = []


    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.config.obs_size:
            new_obs = np.zeros(self.config.obs_size, dtype=np.float32)
            copy_size = min(arr.shape[0], self.config.obs_size)
            new_obs[:copy_size] = arr[:copy_size]
            arr = new_obs
        return arr

    def _action_name(self, action_id: int) -> str:
        names = ["HOLD", "L35%", "L60%", "L85%", "L110%", "S35%", "S60%", "S85%", "S110%", "CLOSE"]
        if 0 <= action_id < len(names):
            return names[action_id]
        return f"A{action_id}"

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        instrument: Optional[str] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        self._last_discrete_action = None


        if self._sb3_model is not None:
            try:
                obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)


                try:
                    shape = getattr(self._sb3_model.observation_space, "shape", None)
                    expected = int(shape[0]) if shape and len(shape) == 1 else None
                except Exception:
                    expected = None


                if expected is not None and obs_arr.shape[0] != expected:

                    logging.warning(
                        f"[PPOCore] Obs size mismatch: got {obs_arr.shape[0]}, "
                        f"expected {expected}. Padding/truncating (may affect predictions!)"
                    )
                    fixed = np.zeros(expected, dtype=np.float32)
                    copy_size = min(obs_arr.shape[0], expected)
                    fixed[:copy_size] = obs_arr[:copy_size]
                    obs_arr = fixed


                if self._is_maskable_ppo:

                    mask = action_mask
                    if mask is None and self._action_mask_fn is not None:
                        try:
                            mask = self._action_mask_fn()
                        except Exception as e:
                            logging.debug(f"[PPOCore] Action mask function failed: {e}")
                            mask = None


                    if mask is not None:
                        action_id, _ = self._sb3_model.predict(
                            obs_arr,
                            deterministic=deterministic,
                            action_masks=mask.reshape(1, -1) if mask.ndim == 1 else mask
                        )
                    else:

                        action_id, _ = self._sb3_model.predict(obs_arr, deterministic=deterministic)


                    self._inference_count = getattr(self, '_inference_count', 0) + 1
                    if self._inference_count % 50 == 1:
                        try:

                            obs_tensor = self._sb3_model.policy.obs_to_tensor(obs_arr.reshape(1, -1))[0]
                            with torch.no_grad():
                                dist = self._sb3_model.policy.get_distribution(obs_tensor)
                                probs = dist.distribution.probs.cpu().numpy().flatten()

                                top_indices = probs.argsort()[-3:][::-1]
                                prob_str = ", ".join([
                                    f"{self._action_name(i)}:{probs[i]:.1%}"
                                    for i in top_indices
                                ])
                                logging.info(f"[PPO] 🧠 Model thinking: {prob_str}")
                        except Exception as e:
                            logging.debug(f"[PPO] Couldn't get action probs: {e}")


                    action_id_int = int(action_id.item() if hasattr(action_id, 'item') else action_id)
                    decoded = decode_discrete_action(action_id_int, self.config.size_buckets)
                    self._last_discrete_action = decoded


                    direction_score, size_score = decoded.to_continuous()
                    action_np = np.array([direction_score, size_score], dtype=np.float32)


                    if decoded.intent == "hold":
                        logging.debug(
                            f"[PPO] {instrument or 'MULTI'}: HOLD (waiting for better setup)"
                        )
                    else:
                        logging.info(
                            f"[PPO] 🎯 {instrument or 'MULTI'}: {decoded.intent.upper()} "
                            f"(size={decoded.size_mult:.0%}) → action_id={action_id_int}"
                        )


                else:
                    action_full, _ = self._sb3_model.predict(obs_arr, deterministic=deterministic)
                    action_full_arr = np.asarray(action_full, dtype=np.float32).reshape(-1)
                    action_np = self._slice_sb3_action(action_full_arr, instrument=instrument)
                    action_np = np.clip(action_np, -1.0, 1.0).astype(np.float32)


                log_prob_np = 0.0
                value_np = 0.0

                self.last_action = action_np
                self._last_log_prob = log_prob_np
                self._last_value = value_np
                return action_np, log_prob_np, value_np

            except Exception as e:
                logging.warning(f"[PPOCore] SB3 predict failed: {e}, falling back to torch network")


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


            action_np = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)


            action_np = np.clip(action_np, -1.0, 1.0)

            log_prob_np = float(log_prob_tensor.item())
            value_np = float(value.squeeze().item())


        self.last_action = action_np
        self._last_log_prob = log_prob_np
        self._last_value = value_np

        return action_np, log_prob_np, value_np

    def _slice_sb3_action(self, action: np.ndarray, instrument: Optional[str]) -> np.ndarray:
        try:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.zeros(0, dtype=np.float32)

        if arr.size <= 0:
            return np.zeros(self.config.act_size, dtype=np.float32)


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


        return arr[:2]

    def get_value(self, obs: np.ndarray) -> float:
        obs_arr = self._normalize_obs(obs)
        obs_tensor = torch.from_numpy(obs_arr).to(self.device).unsqueeze(0)

        with torch.no_grad():
            _, _, value = self.network(obs_tensor)
            return float(value.squeeze().item())


    def record_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: Optional[float] = None,
        value: Optional[float] = None,
    ) -> None:

        if len(self.buffer["observations"]) >= self.config.buffer_size:

            for key in ("observations", "actions", "rewards", "dones", "log_probs", "values"):
                if self.buffer[key]:
                    self.buffer[key].pop(0)


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
        start_idx = self._episode_start_index
        end_idx = len(self.buffer["rewards"])

        if end_idx <= start_idx:
            return

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


        self._episode_start_index = end_idx


    def should_update(self) -> bool:
        return len(self.buffer["observations"]) >= self.config.batch_size

    def update(self) -> Optional[Dict[str, float]]:
        if not self.should_update():
            return None


        self._compute_gae()


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


        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = observations.size(0)
        mini_batch_size = min(self.config.batch_size, num_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_grad_norm = 0.0
        num_updates = 0


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


                ratio = torch.exp(new_log_probs - old_log_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_b.squeeze(), ret_b)


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


        denom = max(num_updates, 1)
        avg_policy_loss = total_policy_loss / denom
        avg_value_loss = total_value_loss / denom
        avg_entropy_loss = total_entropy_loss / denom
        avg_grad_norm = total_grad_norm / denom


        with torch.no_grad():
            _, _, values_eval = self.network(observations)
            v_pred = values_eval.squeeze()
            var_y = torch.var(returns)
            var_diff = torch.var(returns - v_pred)
            explained_var = float(1.0 - var_diff / (var_y + 1e-8))


        self.stats.total_updates += 1
        self.stats.policy_loss = avg_policy_loss
        self.stats.value_loss = avg_value_loss
        self.stats.entropy_loss = avg_entropy_loss
        self.stats.gradient_norm = avg_grad_norm
        self.stats.explained_variance = explained_var
        self.stats.learning_rate = float(self.optimizer.param_groups[0]["lr"])


        if self.episode_rewards:
            self.lr_scheduler.step(self.stats.avg_episode_reward)


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


        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            if t == n - 1:
                next_value = values[t]
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
        for key in self.buffer:
            self.buffer[key] = []

        self._episode_start_index = 0


    def get_state(self) -> Dict[str, Any]:
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

            "config": {
                "obs_size": self.config.obs_size,
                "act_size": self.config.act_size,
                "hidden_size": self.config.hidden_size,
            },
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if "network_state" in state:
            try:
                self.network.load_state_dict(state["network_state"])
            except Exception:

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
        torch.save(self.get_state(), path)

    def load(self, path: str) -> None:
        if str(path).lower().endswith(".zip"):

            self._is_maskable_ppo = False


            if MASKABLE_PPO_AVAILABLE and MaskablePPO is not None:
                try:
                    self._sb3_model = MaskablePPO.load(path, device="cpu")
                    self._sb3_model_path = str(path)
                    self._is_maskable_ppo = True


                    action_space = getattr(self._sb3_model, "action_space", None)
                    if action_space is not None:
                        space_type = type(action_space).__name__
                        if "Discrete" in space_type:
                            n_actions = int(getattr(action_space, "n", 0))
                            logging.info(
                                f"[PPOCore] Loaded MaskablePPO (Discrete, {n_actions} actions) from {path}"
                            )
                        else:
                            logging.info(
                                f"[PPOCore] Loaded MaskablePPO ({space_type}) from {path}"
                            )
                    return
                except Exception as e:
                    logging.debug(f"[PPOCore] MaskablePPO.load failed: {e}, trying PPO.load")


            from stable_baselines3 import PPO as SB3PPO  # type: ignore[import-not-found]
            self._sb3_model = SB3PPO.load(path, device="cpu")
            self._sb3_model_path = str(path)
            self._is_maskable_ppo = False
            logging.info(f"[PPOCore] Loaded PPO (continuous) from {path}")
            return


        self._sb3_model = None
        self._sb3_model_path = None
        self._is_maskable_ppo = False
        state = torch.load(path, map_location=self.device)
        self.set_state(state)
        logging.info(f"[PPOCore] Loaded native PyTorch checkpoint from {path}")

    def set_action_mask_fn(self, fn: Callable[[], np.ndarray]) -> None:
        self._action_mask_fn = fn

    @property
    def is_discrete_action_space(self) -> bool:
        return self._is_maskable_ppo

    @property
    def last_discrete_action(self) -> Optional[DiscreteActionDecoded]:
        return self._last_discrete_action
