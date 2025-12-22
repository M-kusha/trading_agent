# envs/curriculum_env_wrapper.py
"""
Curriculum Environment Wrapper for Trading RL Agent
====================================================

Wraps the base trading environment with curriculum learning capabilities.

Enhancements in this version (v2.0):
- Recovery protocol integration with modified rewards/constraints
- Mixed-stage sampling for catastrophic forgetting prevention
- Review session support
- Skill-based tracking per episode
- Entropy penalty integration
- Dynamic stage selection (effective stage vs current stage)
- Enhanced reward blending during transitions
- Comprehensive episode info augmentation

Usage:
    from envs.curriculum_env_wrapper import CurriculumEnvWrapper
    from envs.curriculum_manager import CurriculumManager
    
    manager = CurriculumManager(initial_stage=CurriculumStage.FOUNDATION)
    env = CurriculumEnvWrapper(base_env, manager)
    
    obs, info = env.reset()
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.curriculum_config import (
    CurriculumStage,
    CurriculumStageConfig,
    ExecutionDifficulty,
    RewardShaping,
    TradingConstraints,
    DataDifficulty,
    TradingSkill,
    get_stage_config,
)
from envs.curriculum_manager import (
    CurriculumManager,
    EpisodeMetrics,
    RollingStats,
    SkillAssessment,
    RecoveryProtocolState,
)

logger = logging.getLogger("curriculum_env_wrapper")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


class CurriculumEnvWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that applies curriculum learning to a trading environment.
    
    Features:
    - Dynamically adjusts environment parameters based on curriculum stage
    - Applies reward shaping according to stage configuration
    - Tracks episode metrics and reports to CurriculumManager
    - Handles stage transitions with reward blending
    - Supports recovery protocols with modified rewards/constraints
    - Implements mixed-stage sampling and review sessions
    - Integrates entropy penalties for exploration management
    
    The wrapper expects the base environment to have certain methods/attributes:
    - set_execution_params(difficulty: ExecutionDifficulty)
    - set_constraints(constraints: TradingConstraints)
    - set_data_difficulty(difficulty: DataDifficulty) [optional]
    - get_episode_stats() -> Dict[str, Any]
    
    If these methods don't exist, the wrapper will skip those configurations.
    """
    
    def __init__(
        self,
        env: gym.Env,
        manager: CurriculumManager,
        apply_reward_shaping: bool = False,  # Default False: env already shapes rewards
        apply_execution_difficulty: bool = True,
        apply_constraints: bool = True,
        apply_data_difficulty: bool = True,
        verbose: bool = True,
        stage_change_callback: Optional[Callable[[CurriculumStage, CurriculumStage], None]] = None,
    ) -> None:
        """
        Initialize the curriculum wrapper.
        
        Args:
            env: Base trading environment to wrap
            manager: CurriculumManager instance for progression tracking
            apply_reward_shaping: Whether to apply ADDITIONAL stage-specific reward shaping.
                                  Default is False because PropFirmTradingEnv already applies
                                  comprehensive reward shaping in _compute_trade_reward().
                                  Setting this to True would cause DOUBLE reward shaping!
                                  Only enable if using a base env without built-in reward shaping.
            apply_execution_difficulty: Whether to apply execution difficulty settings
            apply_constraints: Whether to apply trading constraints
            apply_data_difficulty: Whether to apply data difficulty filtering
            verbose: Whether to log curriculum events
            stage_change_callback: Optional callback on stage changes
        """
        super().__init__(env)
        
        self.manager = manager
        self.apply_reward_shaping = apply_reward_shaping
        self.apply_execution_difficulty = apply_execution_difficulty
        self.apply_constraints = apply_constraints
        self.apply_data_difficulty = apply_data_difficulty
        self.verbose = verbose
        self.stage_change_callback = stage_change_callback
        
        # Track the last applied stage to detect changes
        self._last_applied_stage: Optional[CurriculumStage] = None
        self._last_applied_epoch: int = 0
        
        # Effective stage (may differ from manager.current_stage during reviews/sampling)
        self._effective_stage: CurriculumStage = manager.current_stage
        self._effective_config: CurriculumStageConfig = get_stage_config(self._effective_stage)
        
        # Episode tracking
        self._episode_timesteps: int = 0
        self._episode_reward: float = 0.0
        self._episode_raw_reward: float = 0.0  # Before shaping
        self._episode_shaped_reward: float = 0.0  # Shaping component only
        self._episode_entropy_penalty: float = 0.0
        
        # Per-step tracking for detailed metrics
        self._step_rewards: List[float] = []
        self._step_raw_rewards: List[float] = []
        
        # Exit quality tracking within episode
        self._episode_trailing_stops: int = 0
        self._episode_agent_closes: int = 0
        self._episode_hard_stops: int = 0
        self._episode_risk_liquidations: int = 0
        self._episode_other_exits: int = 0
        
        # Trade quality tracking
        self._episode_entry_qualities: List[float] = []
        self._episode_r_multiples: List[float] = []
        self._episode_maes: List[float] = []
        self._episode_mfes: List[float] = []
        self._episode_bars_held: List[int] = []
        
        # Recovery protocol state cache
        self._active_recovery: bool = False
        self._recovery_reward_mods: Dict[str, float] = {}
        self._recovery_constraint_mods: Dict[str, float] = {}
        
        # Apply initial configuration
        self._apply_stage_config()
        
        if self.verbose:
            logger.info(
                f"CurriculumEnvWrapper initialized at stage {manager.current_stage.name} "
                f"(epoch={manager.current_stage_epoch})"
            )
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Current curriculum stage from manager."""
        return self.manager.current_stage
    
    @property
    def effective_stage(self) -> CurriculumStage:
        """Effective stage for current episode (may differ during reviews)."""
        return self._effective_stage
    
    @property
    def stage_config(self) -> CurriculumStageConfig:
        """Current effective stage configuration."""
        return self._effective_config
    
    @property
    def is_in_review(self) -> bool:
        """Whether currently in a review session."""
        return self.manager.review_state.is_active()
    
    @property
    def is_in_recovery(self) -> bool:
        """Whether currently in a recovery protocol."""
        return self.manager.recovery_state.is_active()
    
    # -------------------------------------------------------------------------
    # Stage Configuration Application
    # -------------------------------------------------------------------------
    
    def _apply_stage_config(self, force: bool = False) -> None:
        """Apply current stage configuration to the environment."""
        # Determine effective stage (considers reviews and mixed sampling)
        self._effective_stage = self.manager.get_effective_stage()
        self._effective_config = get_stage_config(self._effective_stage)
        
        stage = self._effective_stage
        epoch = self.manager.current_stage_epoch
        config = self._effective_config
        
        # Check if we need to reapply
        if not force and self._last_applied_stage == stage and self._last_applied_epoch == epoch:
            # Check if recovery state changed
            recovery = self.manager.recovery_state
            if self._active_recovery == recovery.is_active():
                return
        
        # Update recovery state cache
        recovery = self.manager.recovery_state
        self._active_recovery = recovery.is_active()
        self._recovery_reward_mods = recovery.reward_modifications.copy() if recovery.is_active() else {}
        self._recovery_constraint_mods = recovery.constraint_modifications.copy() if recovery.is_active() else {}
        
        # Apply execution difficulty
        if self.apply_execution_difficulty and hasattr(self.env, "set_execution_params"):
            try:
                self.env.set_execution_params(config.execution)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Failed to set execution params: {e}")
        
        # Apply constraints (with recovery modifications)
        if self.apply_constraints and hasattr(self.env, "set_constraints"):
            try:
                constraints = self._get_effective_constraints(config.constraints)
                self.env.set_constraints(constraints)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Failed to set constraints: {e}")
        
        # CRITICAL: Apply reward configuration from curriculum stage
        # This ensures that when stage changes, the reward config is properly updated.
        # Without this, the agent might use Foundation's lenient loss_multiplier in Discipline stage!
        if hasattr(self.env, "set_reward_config"):
            try:
                reward_shaping = self._get_effective_reward_shaping()
                self.env.set_reward_config(reward_shaping)  # type: ignore[attr-defined]
                if self.verbose:
                    logger.info(
                        f"Applied reward config for {stage.name}: "
                        f"scale={reward_shaping.reward_scale}, loss_mult={reward_shaping.loss_multiplier}"
                    )
            except Exception as e:
                logger.warning(f"Failed to set reward config: {e}")
        
        # Apply data difficulty
        if self.apply_data_difficulty and hasattr(self.env, "set_data_difficulty"):
            try:
                self.env.set_data_difficulty(config.data_difficulty)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Failed to set data difficulty: {e}")
        
        # Update tracking
        self._last_applied_stage = stage
        self._last_applied_epoch = epoch
        
        if self.verbose and (force or stage != self._last_applied_stage):
            extra = ""
            if self.is_in_review:
                extra = " [REVIEW]"
            elif self.is_in_recovery:
                extra = f" [RECOVERY: {recovery.trigger_reason}]"
            logger.debug(f"Applied stage config: {stage.name} (epoch={epoch}){extra}")
    
    def _get_effective_constraints(self, base: TradingConstraints) -> TradingConstraints:
        """Get constraints with recovery modifications applied."""
        if not self._active_recovery or not self._recovery_constraint_mods:
            return base
        
        # Create modified copy
        constraints = copy.deepcopy(base)
        
        for name, multiplier in self._recovery_constraint_mods.items():
            if hasattr(constraints, name):
                current = getattr(constraints, name)
                if isinstance(current, (int, float)):
                    new_val = current * multiplier
                    if isinstance(current, int):
                        new_val = int(new_val)
                    setattr(constraints, name, new_val)
        
        return constraints
    
    def _get_effective_reward_shaping(self) -> RewardShaping:
        """Get reward shaping config with recovery modifications applied."""
        base = self._effective_config.rewards
        
        if not self._active_recovery or not self._recovery_reward_mods:
            return base
        
        # Create modified copy
        rewards = copy.deepcopy(base)
        
        for name, value in self._recovery_reward_mods.items():
            if hasattr(rewards, name):
                setattr(rewards, name, value)
        
        return rewards
    
    # -------------------------------------------------------------------------
    # Reward Shaping
    # -------------------------------------------------------------------------
    
    def _shape_reward(
        self,
        raw_reward: float,
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Apply curriculum-based reward shaping."""
        if not self.apply_reward_shaping:
            return raw_reward
        
        config = self._get_effective_reward_shaping()
        shaped = raw_reward * config.reward_scale
        shaping_component = 0.0
        
        # Loss multiplier
        if raw_reward < 0:
            shaped *= config.loss_multiplier
        
        # Track exit for exit quality bonus/penalty
        exit_type = info.get("exit_type", info.get("close_reason", ""))
        if exit_type:
            self._track_exit_type(exit_type)
            
            if config.exit_quality_enabled:
                exit_bonus = self._compute_exit_quality_bonus(exit_type, config)
                shaped += exit_bonus
                shaping_component += exit_bonus
        
        # R-multiple bonus
        r_multiple = _safe_float(info.get("r_multiple", 0.0), 0.0)
        if r_multiple > config.r_multiple_bonus_threshold:
            bonus = min(
                (r_multiple - config.r_multiple_bonus_threshold) * config.r_multiple_bonus_scale,
                config.r_multiple_bonus_cap,
            )
            shaped += bonus
            shaping_component += bonus
            self._episode_r_multiples.append(r_multiple)
        elif r_multiple != 0:
            self._episode_r_multiples.append(r_multiple)
        
        # MAE efficiency bonus
        if config.mae_efficiency_enabled:
            mae = _safe_float(info.get("mae", 0.0), 0.0)
            mfe = _safe_float(info.get("mfe", 0.0), 0.0)
            if mae > 0:
                self._episode_maes.append(mae)
            if mfe > 0:
                self._episode_mfes.append(mfe)
                
            if mae > 0 and mfe > 0:
                efficiency = mfe / mae
                if efficiency > config.mae_efficiency_threshold:
                    bonus = min(efficiency * config.mae_efficiency_scale, 0.5)
                    shaped += bonus
                    shaping_component += bonus
        
        # Time efficiency bonus
        if config.time_efficiency_enabled:
            bars_held = info.get("bars_held", info.get("trade_duration_bars", 0))
            if bars_held and bars_held > 0:
                self._episode_bars_held.append(bars_held)
                
                if bars_held <= config.max_trade_bars_for_bonus:
                    # Optimal is around optimal_trade_bars
                    deviation = abs(bars_held - config.optimal_trade_bars)
                    max_deviation = config.max_trade_bars_for_bonus - config.optimal_trade_bars
                    if max_deviation > 0:
                        time_score = 1.0 - (deviation / max_deviation)
                        bonus = time_score * config.time_efficiency_scale
                        shaped += bonus
                        shaping_component += bonus
        
        # Entry quality integration
        if config.entry_quality_integration:
            entry_quality = _safe_float(info.get("entry_quality", 0.5), 0.5)
            self._episode_entry_qualities.append(entry_quality)
            
            eq_modifier = (entry_quality - 0.5) * 2 * config.entry_quality_weight
            shaped += eq_modifier
            shaping_component += eq_modifier
        
        # Drawdown penalty
        if config.dd_shaping_enabled:
            current_dd = _safe_float(info.get("current_drawdown", info.get("drawdown", 0.0)), 0.0)
            if current_dd > config.dd_threshold:
                severity = min(
                    ((current_dd - config.dd_threshold) / config.dd_threshold) ** config.dd_severity_exponent,
                    config.dd_severity_cap,
                )
                penalty = severity * config.dd_penalty_scale
                shaped -= penalty
                shaping_component -= penalty
        
        # Streak modifiers
        if config.streak_modifier_enabled:
            consecutive_wins = info.get("consecutive_wins", 0)
            consecutive_losses = info.get("consecutive_losses", 0)
            
            if consecutive_wins > 1:
                bonus = min(consecutive_wins * config.win_streak_bonus_per_win, 0.2)
                shaped += bonus
                shaping_component += bonus
            elif consecutive_losses > 1:
                penalty = min(consecutive_losses * config.loss_streak_penalty_per_loss, 0.3)
                shaped -= penalty
                shaping_component -= penalty
        
        # Anti-churn penalty
        if config.anti_churn_enabled:
            trades_today = info.get("trades_today", info.get("daily_trades", 0))
            if trades_today > config.daily_trade_soft_limit:
                excess = trades_today - config.daily_trade_soft_limit
                penalty = excess * config.churn_penalty_per_trade
                shaped -= penalty
                shaping_component -= penalty
        
        # Block penalties
        if info.get("hard_blocked", False):
            shaped -= config.hard_block_penalty
            shaping_component -= config.hard_block_penalty
        elif info.get("soft_blocked", False):
            shaped -= config.soft_block_penalty
            shaping_component -= config.soft_block_penalty
        
        # Truncation handling
        if truncated and not terminated:
            trade_pnl = _safe_float(info.get("unrealized_pnl", info.get("trade_pnl", 0.0)), 0.0)
            if trade_pnl > 0:
                shaped *= (1.0 - config.truncation_winner_discount)
            elif trade_pnl < 0:
                shaped -= abs(trade_pnl) * config.truncation_loser_extra_penalty
                shaping_component -= abs(trade_pnl) * config.truncation_loser_extra_penalty
        
        # Exploration bonus (decays with stage progress)
        if config.exploration_bonus > 0:
            stage_progress = self.manager.stage_episodes / max(self._effective_config.competence.min_episodes, 100)
            decay = np.exp(-stage_progress * 2)
            exploration = config.exploration_bonus * decay
            shaped += exploration
            shaping_component += exploration
        
        # Entropy penalty from manager
        entropy_penalty = self.manager.get_entropy_penalty()
        if entropy_penalty != 0:
            shaped -= entropy_penalty
            self._episode_entropy_penalty += entropy_penalty
        
        # Apply reward blending during transitions
        # blend_factor: 0.0 at start -> 1.0 at end
        # At transition START, we use mostly OLD config (smooth transition)
        # At transition END, we use fully NEW config
        # This prevents sudden reward signal changes that destabilize learning
        if self.manager.is_in_transition and self.manager._previous_stage_config is not None:
            blend_factor = self.manager.reward_blend_factor
            if blend_factor < 1.0:
                # Compute reward with previous config
                prev_shaped = self._shape_reward_with_config(
                    raw_reward, info, terminated, truncated,
                    self.manager._previous_stage_config.rewards
                )
                shaped = blend_factor * shaped + (1.0 - blend_factor) * prev_shaped
        
        # Clamp final reward
        shaped = _clamp(shaped, config.min_reward, config.max_reward)
        
        # Track shaping contribution
        self._episode_shaped_reward += shaping_component
        
        return shaped
    
    def _shape_reward_with_config(
        self,
        raw_reward: float,
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
        config: RewardShaping,
    ) -> float:
        """Shape reward using a specific config (for blending)."""
        shaped = raw_reward * config.reward_scale
        
        if raw_reward < 0:
            shaped *= config.loss_multiplier
        
        # Simplified shaping for blending (just core components)
        r_multiple = _safe_float(info.get("r_multiple", 0.0), 0.0)
        if r_multiple > config.r_multiple_bonus_threshold:
            bonus = min(
                (r_multiple - config.r_multiple_bonus_threshold) * config.r_multiple_bonus_scale,
                config.r_multiple_bonus_cap,
            )
            shaped += bonus
        
        return _clamp(shaped, config.min_reward, config.max_reward)
    
    def _compute_exit_quality_bonus(self, exit_type: str, config: RewardShaping) -> float:
        """Compute exit quality bonus/penalty."""
        exit_lower = exit_type.lower()
        
        if "trailing" in exit_lower:
            return config.trailing_stop_bonus
        elif "agent" in exit_lower or "manual" in exit_lower:
            return config.agent_close_bonus
        elif "hard" in exit_lower or "stop_loss" in exit_lower:
            return -config.hard_stop_penalty
        elif "liquidation" in exit_lower or "risk" in exit_lower or "margin" in exit_lower:
            return -config.risk_liquidation_penalty
        
        return 0.0
    
    def _track_exit_type(self, exit_type: str) -> None:
        """Track exit type for episode statistics."""
        exit_lower = exit_type.lower()
        
        if "trailing" in exit_lower:
            self._episode_trailing_stops += 1
        elif "agent" in exit_lower or "manual" in exit_lower:
            self._episode_agent_closes += 1
        elif "hard" in exit_lower or "stop_loss" in exit_lower:
            self._episode_hard_stops += 1
        elif "liquidation" in exit_lower or "risk" in exit_lower or "margin" in exit_lower:
            self._episode_risk_liquidations += 1
        else:
            self._episode_other_exits += 1
    
    # -------------------------------------------------------------------------
    # Gym Interface
    # -------------------------------------------------------------------------
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and apply current stage configuration."""
        # Apply stage configuration (handles effective stage selection)
        self._apply_stage_config()
        
        # Reset episode tracking
        self._episode_timesteps = 0
        self._episode_reward = 0.0
        self._episode_raw_reward = 0.0
        self._episode_shaped_reward = 0.0
        self._episode_entropy_penalty = 0.0
        self._step_rewards = []
        self._step_raw_rewards = []
        
        # Reset exit tracking
        self._episode_trailing_stops = 0
        self._episode_agent_closes = 0
        self._episode_hard_stops = 0
        self._episode_risk_liquidations = 0
        self._episode_other_exits = 0
        
        # Reset trade quality tracking
        self._episode_entry_qualities = []
        self._episode_r_multiples = []
        self._episode_maes = []
        self._episode_mfes = []
        self._episode_bars_held = []
        
        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Augment info with curriculum state
        info = self._augment_info(info)
        
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute action and apply curriculum-based processing."""
        # Step base environment
        obs, raw_reward_sf, terminated, truncated, info = self.env.step(action)
        raw_reward: float = float(raw_reward_sf)  # Cast SupportsFloat to float
        
        self._episode_timesteps += 1
        self._episode_raw_reward += raw_reward
        self._step_raw_rewards.append(raw_reward)
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(raw_reward, info, terminated, truncated)
        self._episode_reward += shaped_reward
        self._step_rewards.append(shaped_reward)
        
        # Update manager transition state
        self.manager.step_transition_state(timesteps=1)
        
        # Augment info
        info = self._augment_info(info, raw_reward=raw_reward, shaped_reward=shaped_reward)
        
        # Handle episode end
        if terminated or truncated:
            self._on_episode_end(info)
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _augment_info(
        self,
        info: Dict[str, Any],
        raw_reward: Optional[float] = None,
        shaped_reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Augment info dict with curriculum information."""
        info["curriculum"] = {
            "current_stage": self.manager.current_stage.name,
            "effective_stage": self._effective_stage.name,
            "stage_epoch": self.manager.current_stage_epoch,
            "stage_episodes": self.manager.stage_episodes,
            "stage_timesteps": self.manager.stage_timesteps,
            "total_episodes": self.manager.total_episodes,
            "total_timesteps": self.manager.total_timesteps,
            "is_in_transition": self.manager.is_in_transition,
            "is_in_review": self.is_in_review,
            "is_in_recovery": self.is_in_recovery,
            "reward_blend_factor": self.manager.reward_blend_factor,
            "lr_multiplier": self.manager.get_lr_multiplier(),
        }
        
        if raw_reward is not None:
            info["curriculum"]["raw_reward"] = raw_reward
        if shaped_reward is not None:
            info["curriculum"]["shaped_reward"] = shaped_reward
        
        # Add recovery info if active
        if self.is_in_recovery:
            recovery = self.manager.recovery_state
            info["curriculum"]["recovery"] = {
                "focus_skill": recovery.focus_skill.value if recovery.focus_skill else None,
                "episodes_remaining": recovery.episodes_remaining,
                "trigger_reason": recovery.trigger_reason,
            }
        
        return info
    
    def _on_episode_end(self, info: Dict[str, Any]) -> None:
        """Handle end of episode: record metrics and check for stage transitions."""
        # Get episode stats from base environment
        episode_stats = {}
        if hasattr(self.env, "get_episode_stats"):
            try:
                episode_stats = self.env.get_episode_stats() or {}  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Failed to get episode stats: {e}")
        
        # Merge with info
        merged_info = {**episode_stats, **info}
        
        # Add our tracked exit quality distribution
        merged_info["episode_stats"] = merged_info.get("episode_stats", {})
        merged_info["episode_stats"]["exit_quality_distribution"] = {
            "trailing_stop": self._episode_trailing_stops,
            "agent_close": self._episode_agent_closes,
            "hard_stop": self._episode_hard_stops,
            "risk_liquidation": self._episode_risk_liquidations,
            "other": self._episode_other_exits,
        }
        
        # Add computed averages
        if self._episode_entry_qualities:
            merged_info["episode_stats"]["avg_entry_quality"] = float(np.mean(self._episode_entry_qualities))
        if self._episode_r_multiples:
            merged_info["episode_stats"]["avg_r_multiple"] = float(np.mean(self._episode_r_multiples))
        if self._episode_maes:
            merged_info["episode_stats"]["avg_mae"] = float(np.mean(self._episode_maes))
        if self._episode_mfes:
            merged_info["episode_stats"]["avg_mfe"] = float(np.mean(self._episode_mfes))
        if self._episode_bars_held:
            merged_info["episode_stats"]["avg_bars_held"] = float(np.mean(self._episode_bars_held))
        
        # Add reward breakdown
        merged_info["episode_stats"]["raw_reward"] = self._episode_raw_reward
        merged_info["episode_stats"]["shaped_reward"] = self._episode_reward
        merged_info["episode_stats"]["shaping_contribution"] = self._episode_shaped_reward
        merged_info["episode_stats"]["entropy_penalty_total"] = self._episode_entropy_penalty
        
        # Record to manager with the effective stage used for this episode
        # This ensures mixed-stage samples get recorded to the correct stage's history
        self.manager.record_episode_from_info(
            info=merged_info,
            episode_reward=self._episode_reward,
            episode_length=self._episode_timesteps,
            effective_stage=self._effective_stage,
        )
        
        # Check for stage transitions
        old_stage = self.manager.current_stage
        transitioned, new_stage = self.manager.update()
        
        if transitioned and new_stage is not None:
            if self.verbose:
                direction = "↑" if new_stage.value > old_stage.value else "↓"
                logger.info(f"Stage transition: {old_stage.name} {direction} {new_stage.name}")
            
            if self.stage_change_callback is not None:
                try:
                    self.stage_change_callback(old_stage, new_stage)
                except Exception as e:
                    logger.warning(f"Stage change callback error: {e}")
    
    # -------------------------------------------------------------------------
    # Additional Interface Methods
    # -------------------------------------------------------------------------
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get comprehensive curriculum information."""
        stats = self.manager.get_rolling_stats()
        meets_promotion, promotion_results = self.manager.check_promotion_criteria()
        
        return {
            "current_stage": self.manager.current_stage.name,
            "effective_stage": self._effective_stage.name,
            "stage_epoch": self.manager.current_stage_epoch,
            "stage_episodes": self.manager.stage_episodes,
            "stage_timesteps": self.manager.stage_timesteps,
            "total_episodes": self.manager.total_episodes,
            "total_timesteps": self.manager.total_timesteps,
            "promotion_ready": meets_promotion,
            "is_in_transition": self.manager.is_in_transition,
            "is_in_review": self.is_in_review,
            "is_in_recovery": self.is_in_recovery,
            "is_plateaued": self.manager.learning_velocity.is_plateaued(),
            "rolling_stats": asdict(stats),
            "skill_assessment": self.manager.skill_assessment.to_dict() if self.manager.skill_assessment else None,
            "composite_score": self.manager.composite_score.to_dict() if self.manager.composite_score else None,
            "learning_velocity": self.manager.learning_velocity.to_dict(),
        }
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get detailed progress report from manager."""
        return self.manager.get_progress_report()
    
    def update_entropy(self, entropy: float) -> None:
        """Update current policy entropy (call from training loop)."""
        self.manager.update_entropy(entropy)
    
    def force_stage(self, stage: CurriculumStage, reason: str = "manual") -> None:
        """Force transition to a specific stage."""
        old_stage = self.manager.current_stage
        self.manager.force_stage(stage, reason)
        self._apply_stage_config(force=True)
        
        if self.stage_change_callback is not None:
            try:
                self.stage_change_callback(old_stage, stage)
            except Exception as e:
                logger.warning(f"Stage change callback error: {e}")
    
    def save_curriculum_state(self, path: str) -> None:
        """Save curriculum manager state to file."""
        from pathlib import Path
        self.manager.save(Path(path))
    
    def load_curriculum_state(self, path: str) -> None:
        """Load curriculum manager state from file."""
        from pathlib import Path
        loaded = CurriculumManager.load(
            Path(path),
            auto_promote=self.manager.auto_promote,
            auto_demote=self.manager.auto_demote,
            verbose=self.manager.verbose,
        )
        
        # Transfer state
        self.manager.current_stage = loaded.current_stage
        self.manager._current_stage_epoch = loaded._current_stage_epoch
        self.manager.total_timesteps = loaded.total_timesteps
        self.manager.total_episodes = loaded.total_episodes
        self.manager.stage_timesteps = loaded.stage_timesteps
        self.manager.stage_episodes = loaded.stage_episodes
        self.manager._history = loaded._history
        self.manager._transitions = loaded._transitions
        self.manager._stage_timesteps_total = loaded._stage_timesteps_total
        self.manager._stage_episodes_total = loaded._stage_episodes_total
        self.manager._stage_epoch_counter = loaded._stage_epoch_counter
        self.manager._demotion_analyzer = loaded._demotion_analyzer
        self.manager._learning_velocity = loaded._learning_velocity
        self.manager._rolling_stats_dirty = True
        
        # Reapply configuration
        self._apply_stage_config(force=True)
        
        if self.verbose:
            logger.info(f"Loaded curriculum state: stage={self.manager.current_stage.name}")
    
    def get_lr_multiplier(self) -> float:
        """Get current learning rate multiplier for warmup."""
        return self.manager.get_lr_multiplier()
    
    def get_effective_constraints(self) -> TradingConstraints:
        """Get current effective trading constraints."""
        return self._get_effective_constraints(self._effective_config.constraints)
    
    def get_effective_reward_config(self) -> RewardShaping:
        """Get current effective reward shaping configuration."""
        return self._get_effective_reward_shaping()
    
    def __repr__(self) -> str:
        return (
            f"CurriculumEnvWrapper("
            f"stage={self.manager.current_stage.name}, "
            f"effective={self._effective_stage.name}, "
            f"epoch={self.manager.current_stage_epoch}, "
            f"episodes={self.manager.total_episodes})"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def make_curriculum_env(
    env_factory: Callable[[], gym.Env],
    initial_stage: CurriculumStage = CurriculumStage.FOUNDATION,
    manager_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[CurriculumEnvWrapper, CurriculumManager]:
    """
    Factory function to create a curriculum-wrapped environment.
    
    Args:
        env_factory: Callable that creates the base environment
        initial_stage: Starting curriculum stage
        manager_kwargs: Keyword arguments for CurriculumManager
        wrapper_kwargs: Keyword arguments for CurriculumEnvWrapper
    
    Returns:
        Tuple of (wrapped_env, manager)
    
    Example:
        env, manager = make_curriculum_env(
            env_factory=lambda: TradingEnv(config),
            initial_stage=CurriculumStage.FOUNDATION,
            manager_kwargs={"verbose": True},
            wrapper_kwargs={"apply_reward_shaping": True},
        )
    """
    manager_kwargs = manager_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    
    # Create manager
    manager = CurriculumManager(initial_stage=initial_stage, **manager_kwargs)
    
    # Create base environment
    base_env = env_factory()
    
    # Wrap with curriculum
    wrapped_env = CurriculumEnvWrapper(base_env, manager, **wrapper_kwargs)
    
    return wrapped_env, manager


def make_vectorized_curriculum_env(
    env_factory: Callable[[], gym.Env],
    n_envs: int,
    initial_stage: CurriculumStage = CurriculumStage.FOUNDATION,
    manager_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[CurriculumEnvWrapper], CurriculumManager]:
    """
    Factory function to create multiple curriculum-wrapped environments sharing one manager.
    
    All environments share the same CurriculumManager, which aggregates their experiences.
    
    Args:
        env_factory: Callable that creates base environments
        n_envs: Number of environments to create
        initial_stage: Starting curriculum stage
        manager_kwargs: Keyword arguments for CurriculumManager
        wrapper_kwargs: Keyword arguments for CurriculumEnvWrapper
    
    Returns:
        Tuple of (list_of_wrapped_envs, shared_manager)
    """
    manager_kwargs = manager_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    
    # Create single shared manager
    manager = CurriculumManager(initial_stage=initial_stage, **manager_kwargs)
    
    # Create wrapped environments
    envs = []
    for i in range(n_envs):
        base_env = env_factory()
        # Only first env is verbose to avoid log spam
        env_wrapper_kwargs = {**wrapper_kwargs, "verbose": (i == 0) and wrapper_kwargs.get("verbose", True)}
        wrapped_env = CurriculumEnvWrapper(base_env, manager, **env_wrapper_kwargs)
        envs.append(wrapped_env)
    
    return envs, manager