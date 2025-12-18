# envs/curriculum_env_wrapper.py
"""
Curriculum Environment Wrapper
==============================

Wraps PropFirmTradingEnv with stage-specific modifications based on curriculum configuration.

Fixes/Enhancements:
- Safe attribute patching (works even if env config schema differs slightly)
- Restores defaults via captured original config for disable/enable toggles
- Adds stage-sync method for VecEnv env_method broadcasting (Subproc-safe strategy)
- Records modified_reward into info for cleaner logging
"""

from __future__ import annotations

import copy
import logging
from datetime import time as dtime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.curriculum_config import (
    CurriculumStage,
    CurriculumStageConfig,
    ExecutionDifficulty,
    RewardShaping,
    TradingConstraints,
    get_stage_config,
)
from envs.curriculum_manager import CurriculumManager

if TYPE_CHECKING:
    from envs.prop_firm_env import PropFirmTradingEnv  # pragma: no cover


logger = logging.getLogger("curriculum_wrapper")


def _set_if_present(obj: Any, attr: str, value: Any) -> None:
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, value)
        except Exception:
            pass


def _get_if_present(obj: Any, attr: str, default: Any = None) -> Any:
    return getattr(obj, attr, default)


class CurriculumEnvWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that applies curriculum stage configurations to PropFirmTradingEnv.
    """

    def __init__(
        self,
        env: gym.Env,
        curriculum_manager: CurriculumManager,
        auto_update_curriculum: bool = True,
        record_metrics: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(env)

        # Best-effort concrete typing
        self.env: "PropFirmTradingEnv" = env  # type: ignore[assignment]

        self.curriculum_manager = curriculum_manager
        self.auto_update_curriculum = auto_update_curriculum
        self.record_metrics = record_metrics
        self.verbose = verbose

        self._original_config = copy.deepcopy(_get_if_present(self.env, "config", None))

        self._episode_reward: float = 0.0
        self._episode_length: int = 0
        self._episode_trades: int = 0

        # Stage sync hook for VecEnv env_method broadcasting
        self._pending_stage: Optional[CurriculumStage] = None
        self._pending_reason: str = ""

        self._apply_stage_config()

    @property
    def current_stage(self) -> CurriculumStage:
        return self.curriculum_manager.current_stage

    @property
    def stage_config(self) -> CurriculumStageConfig:
        return get_stage_config(self.current_stage)

    def set_curriculum_stage(self, stage: int | CurriculumStage, reason: str = "sync") -> None:
        """
        SubprocVecEnv-safe: master process can broadcast a stage to each worker via env_method.
        We DO NOT switch mid-episode; we apply on next reset.
        """
        try:
            s = stage if isinstance(stage, CurriculumStage) else CurriculumStage(int(stage))
        except Exception:
            return
        self._pending_stage = s
        self._pending_reason = reason

    def _maybe_apply_pending_stage(self) -> None:
        if self._pending_stage is None:
            return
        self.curriculum_manager.force_stage(self._pending_stage, reason=self._pending_reason or "sync")
        self._pending_stage = None
        self._pending_reason = ""

    def _apply_stage_config(self) -> None:
        stage_cfg = self.stage_config
        env_cfg = _get_if_present(self.env, "config", None)
        if env_cfg is None:
            return

        # Apply execution difficulty
        self._apply_execution_config(stage_cfg.execution)

        # Apply reward shaping
        self._apply_reward_config(stage_cfg.rewards)

        # Apply constraints
        self._apply_constraint_config(stage_cfg.constraints)

        # Episode params
        _set_if_present(env_cfg, "max_steps_per_episode", stage_cfg.max_steps_per_episode)

        # Observation flags (best-effort: only if config supports them)
        _set_if_present(env_cfg, "include_memory_features", stage_cfg.include_memory_features)
        _set_if_present(env_cfg, "include_world_model_features", stage_cfg.include_world_model_features)
        _set_if_present(env_cfg, "include_expert_signals", stage_cfg.include_expert_signals)

        if self.verbose:
            logger.info(
                f"Applied curriculum stage: {stage_cfg.name} "
                f"(stage={stage_cfg.stage.name}, "
                f"max_dd={stage_cfg.constraints.max_drawdown_limit:.0%}, "
                f"entry_thr={stage_cfg.constraints.entry_quality_threshold:.2f})"
            )

    def _apply_execution_config(self, exec_cfg: ExecutionDifficulty) -> None:
        env_cfg = _get_if_present(self.env, "config", None)
        if env_cfg is None:
            return

        exec_obj = _get_if_present(env_cfg, "execution", None)
        if exec_obj is not None:
            _set_if_present(exec_obj, "base_spread_points", exec_cfg.base_spread_points)
            _set_if_present(exec_obj, "spread_mult_range", exec_cfg.spread_mult_range)
            _set_if_present(exec_obj, "max_spread_points", exec_cfg.max_spread_points)
            _set_if_present(exec_obj, "slippage_points_sigma", exec_cfg.slippage_points_sigma)
            _set_if_present(exec_obj, "slippage_mult_range", exec_cfg.slippage_mult_range)
            _set_if_present(exec_obj, "max_slippage_points", exec_cfg.max_slippage_points)
            _set_if_present(exec_obj, "commission_per_lot", exec_cfg.commission_per_lot)
            _set_if_present(exec_obj, "latency_bars", exec_cfg.latency_bars)

        # Domain randomization toggles (schema varies across envs)
        _set_if_present(env_cfg, "domain_randomization_enabled", exec_cfg.enable_randomization)
        _set_if_present(env_cfg, "spread_mult_range", exec_cfg.spread_randomization_range)
        _set_if_present(env_cfg, "slippage_mult_range", exec_cfg.slippage_randomization_range)
        _set_if_present(env_cfg, "latency_bars_range", exec_cfg.latency_randomization_range)
        _set_if_present(env_cfg, "volatility_scale_range", exec_cfg.volatility_scale_range)

    def _apply_reward_config(self, reward_cfg: RewardShaping) -> None:
        env_cfg = _get_if_present(self.env, "config", None)
        if env_cfg is None:
            return

        r = _get_if_present(env_cfg, "reward", None)
        if r is None:
            return

        # Base scaling
        _set_if_present(r, "reward_scale", reward_cfg.reward_scale)
        _set_if_present(r, "loss_multiplier", reward_cfg.loss_multiplier)

        # R-multiple
        _set_if_present(r, "r_multiple_bonus_threshold", reward_cfg.r_multiple_bonus_threshold)
        _set_if_present(r, "r_multiple_bonus_scale", reward_cfg.r_multiple_bonus_scale)
        _set_if_present(r, "r_multiple_bonus_cap", reward_cfg.r_multiple_bonus_cap)

        # MAE efficiency
        _set_if_present(r, "mae_efficiency_enabled", reward_cfg.mae_efficiency_enabled)
        _set_if_present(r, "mae_efficiency_scale", reward_cfg.mae_efficiency_scale)
        _set_if_present(r, "mae_efficiency_threshold", reward_cfg.mae_efficiency_threshold)

        # Time efficiency
        _set_if_present(r, "time_efficiency_enabled", reward_cfg.time_efficiency_enabled)
        _set_if_present(r, "time_efficiency_scale", reward_cfg.time_efficiency_scale)
        _set_if_present(r, "optimal_trade_bars", reward_cfg.optimal_trade_bars)
        _set_if_present(r, "max_trade_bars_for_bonus", reward_cfg.max_trade_bars_for_bonus)

        # Exit quality
        _set_if_present(r, "exit_quality_enabled", reward_cfg.exit_quality_enabled)
        _set_if_present(r, "trailing_stop_bonus", reward_cfg.trailing_stop_bonus)
        _set_if_present(r, "agent_close_bonus", reward_cfg.agent_close_bonus)
        _set_if_present(r, "hard_stop_penalty", reward_cfg.hard_stop_penalty)
        _set_if_present(r, "risk_liquidation_penalty", reward_cfg.risk_liquidation_penalty)

        # Truncation
        _set_if_present(r, "truncation_winner_discount", reward_cfg.truncation_winner_discount)
        _set_if_present(r, "truncation_loser_extra_penalty", reward_cfg.truncation_loser_extra_penalty)

        # Entry quality
        _set_if_present(r, "entry_quality_integration", reward_cfg.entry_quality_integration)
        _set_if_present(r, "entry_quality_weight", reward_cfg.entry_quality_weight)

        # Drawdown shaping
        _set_if_present(r, "dd_shaping_enabled", reward_cfg.dd_shaping_enabled)
        _set_if_present(r, "dd_threshold", reward_cfg.dd_threshold)
        _set_if_present(r, "dd_penalty_scale", reward_cfg.dd_penalty_scale)
        _set_if_present(r, "dd_severity_exponent", reward_cfg.dd_severity_exponent)
        _set_if_present(r, "dd_severity_cap", reward_cfg.dd_severity_cap)

        # Streaks
        _set_if_present(r, "streak_modifier_enabled", reward_cfg.streak_modifier_enabled)
        _set_if_present(r, "win_streak_bonus_per_win", reward_cfg.win_streak_bonus_per_win)
        _set_if_present(r, "loss_streak_penalty_per_loss", reward_cfg.loss_streak_penalty_per_loss)

        # Anti-churn
        _set_if_present(r, "anti_churn_enabled", reward_cfg.anti_churn_enabled)
        _set_if_present(r, "daily_trade_soft_limit", reward_cfg.daily_trade_soft_limit)
        _set_if_present(r, "churn_penalty_per_trade", reward_cfg.churn_penalty_per_trade)

        # Block penalties
        _set_if_present(r, "hard_block_penalty", reward_cfg.hard_block_penalty)
        _set_if_present(r, "soft_block_penalty", reward_cfg.soft_block_penalty)

        # Per-step shaping
        _set_if_present(r, "per_step_shaping_enabled", reward_cfg.per_step_shaping_enabled)
        _set_if_present(r, "holding_cost_per_bar", reward_cfg.holding_cost_per_bar)
        _set_if_present(r, "opportunity_bonus_scale", reward_cfg.opportunity_bonus_scale)

        # Clipping
        _set_if_present(r, "min_reward", reward_cfg.min_reward)
        _set_if_present(r, "max_reward", reward_cfg.max_reward)

    def _apply_constraint_config(self, constraints: TradingConstraints) -> None:
        env_cfg = _get_if_present(self.env, "config", None)
        if env_cfg is None:
            return

        # Position limits
        _set_if_present(env_cfg, "max_positions", constraints.max_positions)

        # Trade limits
        _set_if_present(env_cfg, "max_trades_per_day", constraints.max_trades_per_day)
        _set_if_present(env_cfg, "max_trades_per_session", constraints.max_trades_per_session)
        _set_if_present(env_cfg, "max_consecutive_losses", constraints.max_consecutive_losses)

        # Weekend block
        _set_if_present(env_cfg, "allow_weekend_holding", not constraints.enforce_weekend_block)

        # No-new-trades window
        if constraints.enforce_no_new_trades_window:
            _set_if_present(env_cfg, "no_new_trades_start", dtime(18, 0))
            _set_if_present(env_cfg, "no_new_trades_end", dtime(9, 0))
        else:
            _set_if_present(env_cfg, "no_new_trades_start", dtime(0, 0))
            _set_if_present(env_cfg, "no_new_trades_end", dtime(0, 0))

        # Hard close + final exit window
        if constraints.enforce_hard_close:
            _set_if_present(env_cfg, "hard_close_time", dtime(22, 0))
            _set_if_present(env_cfg, "final_exit_window_minutes", 60)
        else:
            _set_if_present(env_cfg, "hard_close_time", dtime(23, 59))
            _set_if_present(env_cfg, "final_exit_window_minutes", 1)

        _set_if_present(env_cfg, "min_minutes_between_entries", constraints.min_minutes_between_entries)
        _set_if_present(env_cfg, "min_minutes_after_loss", constraints.min_minutes_after_loss)

        # Session windows: if disabled, widen windows if schema supports it
        if not constraints.enforce_session_windows and self._original_config is not None:
            # Try common patterns; if not present, no-op
            _set_if_present(env_cfg, "session_start", dtime(0, 0))
            _set_if_present(env_cfg, "session_end", dtime(23, 59))
            _set_if_present(env_cfg, "prime_session_start", dtime(0, 0))
            _set_if_present(env_cfg, "prime_session_end", dtime(23, 59))
        elif constraints.enforce_session_windows and self._original_config is not None:
            # Restore original if those fields exist
            for attr in ("session_start", "session_end", "prime_session_start", "prime_session_end"):
                if hasattr(env_cfg, attr) and hasattr(self._original_config, attr):
                    _set_if_present(env_cfg, attr, getattr(self._original_config, attr))

        # Drawdown limits
        _set_if_present(env_cfg, "daily_drawdown_limit", constraints.daily_drawdown_limit)
        _set_if_present(env_cfg, "max_drawdown_limit", constraints.max_drawdown_limit)
        _set_if_present(env_cfg, "daily_dd_safety_buffer", constraints.daily_dd_safety_buffer)
        _set_if_present(env_cfg, "max_dd_safety_buffer", constraints.max_dd_safety_buffer)
        _set_if_present(env_cfg, "emergency_close_threshold", constraints.emergency_close_threshold)

        # Entry quality gate
        _set_if_present(env_cfg, "entry_quality_gate_enabled", constraints.entry_quality_gate_enabled)
        _set_if_present(env_cfg, "entry_quality_threshold", constraints.entry_quality_threshold)

        # Stops / risk mgmt
        _set_if_present(env_cfg, "hard_stop_loss_eur", constraints.hard_stop_loss_eur)
        _set_if_present(env_cfg, "soft_stop_loss_eur", constraints.soft_stop_loss_eur)
        _set_if_present(env_cfg, "trailing_activation_eur", constraints.trailing_activation_eur)
        _set_if_present(env_cfg, "trailing_retrace_pct", constraints.trailing_retrace_pct)
        _set_if_present(env_cfg, "time_decay_hours", constraints.time_decay_hours)

        # Risk per trade
        _set_if_present(env_cfg, "risk_per_trade_pct", constraints.risk_per_trade_pct)
        _set_if_present(env_cfg, "max_risk_per_trade_pct", constraints.max_risk_per_trade_pct)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Apply any pending stage updates before starting a new episode
        self._maybe_apply_pending_stage()

        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_trades = 0

        self._apply_stage_config()

        obs, info = self.env.reset(seed=seed, options=options)

        info["curriculum_stage"] = self.current_stage.name
        info["curriculum_stage_idx"] = int(self.current_stage.value)
        info["curriculum_stage_epoch"] = int(self.curriculum_manager.current_stage_epoch)

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Curriculum-specific modifications (lightweight; base env remains primary)
        modified_reward = self._modify_reward(float(reward), info)

        self._episode_reward += modified_reward
        self._episode_length += 1

        trade_count = int(info.get("trade_count", 0) or 0)
        if trade_count > self._episode_trades:
            self._episode_trades = trade_count

        # Curriculum info for loggers/callbacks
        info["curriculum_stage"] = self.current_stage.name
        info["curriculum_stage_idx"] = int(self.current_stage.value)
        info["curriculum_stage_epoch"] = int(self.curriculum_manager.current_stage_epoch)
        info["modified_reward"] = float(modified_reward)

        if terminated or truncated:
            self._handle_episode_end(info)

        return obs, float(modified_reward), terminated, truncated, info

    def _modify_reward(self, reward: float, info: Dict[str, Any]) -> float:
        stage_cfg = self.stage_config
        reward_cfg = stage_cfg.rewards

        modified = reward

        # Exploration bonus
        if reward_cfg.exploration_bonus > 0:
            if bool(info.get("has_position", False)) or bool(info.get("pending_entry", False)):
                modified += float(reward_cfg.exploration_bonus)

        # Directional accuracy amplification
        if reward_cfg.directional_accuracy_weight != 1.0:
            last_pnl = float(info.get("last_net_trade_pnl", 0.0) or 0.0)
            if last_pnl > 0:
                comps = info.get("reward_components", {}) or {}
                base_pnl_component = float(comps.get("base_pnl", 0.0) or 0.0)
                if base_pnl_component > 0:
                    bonus = base_pnl_component * (float(reward_cfg.directional_accuracy_weight) - 1.0)
                    modified += bonus

        # Reward blending during stage transitions
        # If we're in a transition period, blend with previous stage reward characteristics
        blend_factor = self.curriculum_manager.reward_blend_factor
        if blend_factor < 1.0 and self.curriculum_manager._previous_stage_config is not None:
            # blend_factor = 0.0 means 100% previous, 1.0 means 100% current
            prev_cfg = self.curriculum_manager._previous_stage_config
            prev_reward_cfg = prev_cfg.rewards
            
            # Compute what the reward modification would have been under the previous stage
            prev_modified = reward
            if prev_reward_cfg.exploration_bonus > 0:
                if bool(info.get("has_position", False)) or bool(info.get("pending_entry", False)):
                    prev_modified += float(prev_reward_cfg.exploration_bonus)
            
            if prev_reward_cfg.directional_accuracy_weight != 1.0:
                last_pnl = float(info.get("last_net_trade_pnl", 0.0) or 0.0)
                if last_pnl > 0:
                    comps = info.get("reward_components", {}) or {}
                    base_pnl_component = float(comps.get("base_pnl", 0.0) or 0.0)
                    if base_pnl_component > 0:
                        bonus = base_pnl_component * (float(prev_reward_cfg.directional_accuracy_weight) - 1.0)
                        prev_modified += bonus
            
            # Blend: lerp from previous (blend_factor=0) to current (blend_factor=1)
            modified = prev_modified * (1.0 - blend_factor) + modified * blend_factor

        return float(modified)

    def _handle_episode_end(self, info: Dict[str, Any]) -> None:
        if self.record_metrics:
            self.curriculum_manager.record_episode_from_info(
                info=info,
                episode_reward=self._episode_reward,
                episode_length=self._episode_length,
            )

        if self.auto_update_curriculum:
            changed, new_stage = self.curriculum_manager.update()
            if changed and new_stage is not None and self.verbose:
                logger.info(f"Curriculum stage changed to: {new_stage.name} (applies next reset)")

    def get_curriculum_progress(self) -> Dict[str, Any]:
        return self.curriculum_manager.get_progress_report()

    def action_masks(self) -> np.ndarray:
        # sb3-contrib MaskablePPO expects action_masks()
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()  # type: ignore[misc]
        if hasattr(self.env, "get_action_mask"):
            return self.env.get_action_mask()  # type: ignore[misc]
        n_actions = cast(spaces.Discrete, self.action_space).n
        return np.ones(n_actions, dtype=np.bool_)

    def get_action_mask(self) -> np.ndarray:
        return self.action_masks()


class CurriculumVecEnvWrapper:
    """
    VecEnv wrapper that aggregates episode stats into a MASTER CurriculumManager and can
    broadcast stage changes to sub-envs using env_method (works with SubprocVecEnv/DummyVecEnv).

    NOTE:
    - A single CurriculumManager cannot be shared across subprocess workers.
      Use this wrapper to keep a master manager in the main process, then
      broadcast stage changes to workers that expose set_curriculum_stage().
    """

    def __init__(
        self,
        vec_env: Any,
        curriculum_manager: CurriculumManager,
        auto_update_curriculum: bool = True,
        broadcast_stage: bool = True,
        verbose: bool = True,
    ) -> None:
        self.vec_env = vec_env
        self.curriculum_manager = curriculum_manager
        self.auto_update_curriculum = auto_update_curriculum
        self.broadcast_stage = broadcast_stage
        self.verbose = verbose

        self.n_envs = int(getattr(vec_env, "num_envs", 1))
        self._episode_rewards = [0.0] * self.n_envs
        self._episode_lengths = [0] * self.n_envs

    @property
    def current_stage(self) -> CurriculumStage:
        return self.curriculum_manager.current_stage

    def _broadcast(self, stage: CurriculumStage, reason: str) -> None:
        if not self.broadcast_stage:
            return
        if not hasattr(self.vec_env, "env_method"):
            return
        try:
            self.vec_env.env_method("set_curriculum_stage", int(stage.value), reason)
        except Exception:
            # Not all envs implement set_curriculum_stage; ignore safely
            pass

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        # Broadcast current stage at the start (helps workers sync after reload)
        self._broadcast(self.current_stage, reason="reset_sync")
        return self.vec_env.reset(*args, **kwargs)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        obs, rewards, dones, infos = self.vec_env.step_wait()

        for i in range(self.n_envs):
            self._episode_rewards[i] += float(rewards[i])
            self._episode_lengths[i] += 1

        any_done = False
        for i, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue
            any_done = True

            # Record into master manager using info as provided (SB3 puts episode data here)
            self.curriculum_manager.record_episode_from_info(
                info=info,
                episode_reward=self._episode_rewards[i],
                episode_length=self._episode_lengths[i],
            )

            self._episode_rewards[i] = 0.0
            self._episode_lengths[i] = 0

            info["curriculum_stage"] = self.current_stage.name
            info["curriculum_stage_idx"] = int(self.current_stage.value)
            info["curriculum_stage_epoch"] = int(self.curriculum_manager.current_stage_epoch)

        if self.auto_update_curriculum and any_done:
            changed, new_stage = self.curriculum_manager.update()
            if changed and new_stage is not None:
                if self.verbose:
                    logger.info(f"[VecEnv] Curriculum stage changed to: {new_stage.name} (broadcasting)")
                self._broadcast(new_stage, reason="stage_change")

        return obs, rewards, dones, infos

    def __getattr__(self, name: str) -> Any:
        return getattr(self.vec_env, name)


def make_curriculum_env(
    env_factory: Callable[[], gym.Env],
    curriculum_manager: CurriculumManager,
    auto_update: bool = True,
    record_metrics: bool = True,
    verbose: bool = True,
) -> CurriculumEnvWrapper:
    base_env = env_factory()
    return CurriculumEnvWrapper(
        env=base_env,
        curriculum_manager=curriculum_manager,
        auto_update_curriculum=auto_update,
        record_metrics=record_metrics,
        verbose=verbose,
    )
