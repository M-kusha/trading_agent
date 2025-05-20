#!/usr/bin/env python3
"""
utils/meta_learning.py

Wraps your AdaptiveMetaLearner in a Stable-Baselines3 callback
so that you can call it at the end of each rollout.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

# ──────────────────────────────────────────────────────────────────────────────
# Your existing meta-learner
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveMetaLearner:
    def __init__(self, model: nn.Module, meta_lr: float = 1e-4, adaptation_steps: int = 3):
        """
        model: typically your policy network (e.g. SAC policy)
        """
        self.model = model
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        self.logger = logging.getLogger("meta_learner")
        self.logger.setLevel(logging.INFO)
        self.grad_norm_avg = 0.5

    def compute_meta_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        timing_error: float = 0.0
    ) -> torch.Tensor:
        """
        Example meta-loss: MSE + small penalty if mean of targets is negative
        """
        base_loss = torch.mean((predictions - targets) ** 2)
        penalty = torch.relu(-targets.mean()) * 0.1
        timing_penalty = timing_error ** 2
        return base_loss + penalty + timing_penalty

    def adapt(self, experiences: Dict[str, torch.Tensor]) -> float:
        total_loss = 0.0
        total_grad_norm = 0.0
        smoothing = 0.1

        for step in range(self.adaptation_steps):
            self.optimizer.zero_grad()
            preds = self._forward_pass(experiences)
            loss = self.compute_meta_loss(preds, experiences["targets"])
            loss.backward()

            # measure grad norm
            grad_sq = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_sq += param.grad.norm(2).item() ** 2
            grad_norm = grad_sq ** 0.5
            total_grad_norm += grad_norm

            # adaptive clip
            self.grad_norm_avg = (1 - smoothing) * self.grad_norm_avg + smoothing * grad_norm
            clip_value = max(0.5, self.grad_norm_avg * 1.2)
            nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

            self.optimizer.step()
            total_loss += loss.item()
            self.logger.info(
                f"Meta step {step+1}/{self.adaptation_steps}, "
                f"loss={loss.item():.6f}, grad_norm={grad_norm:.6f}"
            )

        avg_loss = total_loss / self.adaptation_steps
        avg_grad_norm = total_grad_norm / self.adaptation_steps
        # expose to model for inspection if you like
        setattr(self.model, "current_meta_loss", avg_loss)
        setattr(self.model, "gradient_norm", avg_grad_norm)
        return avg_loss

    def _forward_pass(self, experiences: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        If the wrapped model exposes `.actor` we call it,
        otherwise we assume the model itself is callable.
        """
        if hasattr(self.model, "actor"):
            return self.model.actor(experiences["observations"])
        return self.model(experiences["observations"])


# ──────────────────────────────────────────────────────────────────────────────
# Updated SB3 callback that applies one round of meta-adaptation
# and supports off-policy algorithms like SAC
# ──────────────────────────────────────────────────────────────────────────────

class MetaLearningCallback(BaseCallback):
    """
    After each rollout, grab the rollout_buffer (for on-policy algos)
    or a sample from the replay_buffer (for off-policy algos),
    and feed it into the AdaptiveMetaLearner.
    """

    def __init__(
        self,
        meta_learner: AdaptiveMetaLearner,
        off_policy_sample_size: int = 1024,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.meta_learner = meta_learner
        self._logger = logging.getLogger("MetaLearningCallback")
        self.off_policy_sample_size = off_policy_sample_size

    def _on_rollout_end(self) -> None:
        # If the model has a rollout_buffer (on-policy), use it directly
        if hasattr(self.model, "rollout_buffer"):
            buf = self.model.rollout_buffer
            obs = buf.observations
            targets = buf.advantages
        # Otherwise, fall back to replay_buffer (off-policy)
        elif hasattr(self.model, "replay_buffer"):
            rb = self.model.replay_buffer
            # Determine current buffer size
            current_size = getattr(rb, "size", None)
            if callable(current_size):
                current_size = rb.size()
            if not current_size:
                self._logger.warning("Replay buffer is empty, skipping meta-learning")
                return
            batch_size = min(current_size, self.off_policy_sample_size)
            try:
                sample = rb.sample(batch_size)
                obs = sample.observations
                # Use actions as pseudo-targets for meta-learning
                targets = sample.actions
            except Exception as e:
                self._logger.warning(
                    f"Failed to sample from replay_buffer ({e}), skipping meta-learning"
                )
                return
        else:
            self._logger.warning(
                "No rollout_buffer or replay_buffer found, skipping meta-learning"
            )
            return

        # Prepare a torch.Tensor dict for adapt()
        experiences: Dict[str, Any] = {
            "observations": torch.as_tensor(obs, dtype=torch.float32),
            "targets":      torch.as_tensor(targets, dtype=torch.float32),
        }

        loss = self.meta_learner.adapt(experiences)
        self._logger.info(f"[MetaLearningCallback] Adaptation loss: {loss:.6f}")

    def _on_step(self) -> bool:
        # Called at every environment step; we do nothing here.
        # Must return True to continue training.
        return True
