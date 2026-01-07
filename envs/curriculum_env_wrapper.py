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
    
    manager = CurriculumManager(initial_stage=CurriculumStage.EXPLORER)
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
# DUP-2 FIX: Use shared utilities instead of local duplicates
from envs.shared_utils import clamp, safe_float, get_envs_logger

logger = get_envs_logger("curriculum_env_wrapper")


# DEPRECATED: Use shared_utils.clamp() instead
def _clamp(v: float, lo: float, hi: float) -> float:
    return clamp(v, lo, hi)


# DEPRECATED: Use shared_utils.safe_float() instead
def _safe_float(x: Any, default: float = 0.0) -> float:
    return safe_float(x, default)


def _safe_float_impl(x: Any, default: float = 0.0) -> float:
    """Legacy implementation - kept for reference during transition."""
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
        apply_reward_shaping: bool = False,  # DEPRECATED: Always False. Reward shaping is done in env.
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
            apply_reward_shaping: DEPRECATED - Always ignored. Reward shaping is handled
                                  exclusively by PropFirmTradingEnv._compute_trade_reward().
                                  Curriculum stages modify rewards via env.set_reward_config().
                                  This parameter is kept only for backward compatibility.
            apply_execution_difficulty: Whether to apply execution difficulty settings
            apply_constraints: Whether to apply trading constraints
            apply_data_difficulty: Whether to apply data difficulty filtering
            verbose: Whether to log curriculum events
            stage_change_callback: Optional callback on stage changes
        """
        super().__init__(env)
        
        # AUDIT FIX (CRIT-1): apply_reward_shaping REMOVED - it caused DOUBLE reward shaping.
        # Reward shaping is now ONLY in PropFirmTradingEnv._compute_trade_reward().
        # Curriculum stages modify rewards via env.set_reward_config().
        if apply_reward_shaping:
            import warnings
            warnings.warn(
                "apply_reward_shaping=True is DEPRECATED and IGNORED. "
                "Reward shaping is handled by PropFirmTradingEnv. "
                "Curriculum stages modify rewards via env.set_reward_config().",
                DeprecationWarning,
                stacklevel=2
            )
        
        self.manager = manager
        self.apply_reward_shaping = False  # Always False - wrapper does NOT shape rewards
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
        
        # NOTE: Exit type and trade quality tracking was REMOVED.
        # These are now exclusively tracked by PropFirmTradingEnv.get_episode_stats().
        # The wrapper merges base env stats at episode end.
        
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
    # Gym Interface
    # -------------------------------------------------------------------------
    # NOTE: Reward shaping was REMOVED from wrapper.
    # Reward shaping is handled EXCLUSIVELY by PropFirmTradingEnv._compute_trade_reward().
    # Curriculum stages modify rewards via env.set_reward_config().
    # See AUDIT_ENVS_TRAIN_2026_01.md for rationale.

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
        
        # NOTE: Exit type and trade quality tracking removed.
        # Now exclusively tracked by PropFirmTradingEnv.get_episode_stats().
        
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
        
        # AUDIT FIX (CRIT-1): NO wrapper reward shaping.
        # Reward shaping is done ONLY by PropFirmTradingEnv._compute_trade_reward().
        # Wrapper passes through raw_reward unchanged.
        shaped_reward = raw_reward  # Pass-through, no double shaping
        self._episode_reward += shaped_reward
        self._step_rewards.append(shaped_reward)
        
        # Update manager transition state
        self.manager.step_transition_state(timesteps=1)
        
        # Augment info
        info = self._augment_info(info, raw_reward=raw_reward, shaped_reward=shaped_reward)
        
        # Handle episode end - returns modified info with episode_stats
        if terminated or truncated:
            info = self._on_episode_end(info)
        
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
    
    def _on_episode_end(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle end of episode: record metrics and check for stage transitions.
        
        Returns the modified info dict with episode_stats included.
        """
        # Get episode stats from base environment (authoritative source)
        base_episode_stats = {}
        if hasattr(self.env, "get_episode_stats"):
            try:
                base_episode_stats = self.env.get_episode_stats() or {}  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Failed to get episode stats: {e}")
        
        # Ensure episode_stats exists in info
        if "episode_stats" not in info:
            info["episode_stats"] = {}
        
        # Merge base env's episode_stats into info's episode_stats
        # This includes the authoritative exit_quality_distribution from actual trade results
        for k, v in base_episode_stats.items():
            if k not in info["episode_stats"]:
                info["episode_stats"][k] = v
        
        # NOTE: Wrapper-side tracking fallbacks removed.
        # All stats now come from PropFirmTradingEnv.get_episode_stats().
        
        # Add reward breakdown
        info["episode_stats"]["raw_reward"] = self._episode_raw_reward
        info["episode_stats"]["shaped_reward"] = self._episode_reward
        info["episode_stats"]["shaping_contribution"] = self._episode_shaped_reward
        info["episode_stats"]["entropy_penalty_total"] = self._episode_entropy_penalty
        
        # Build EpisodeMetrics for on_episode_end()
        # CRIT FIX: Use on_episode_end() instead of record_episode_from_info() + update()
        # This ensures proper idempotency (no double-counting) and deterministic transition handling
        ep_stats = info.get("episode_stats", {})
        from envs.curriculum_manager import EpisodeMetrics
        
        metrics = EpisodeMetrics(
            total_pnl=safe_float(ep_stats.get("total_pnl", info.get("total_pnl", 0.0)), 0.0),
            win_rate=clamp(safe_float(ep_stats.get("win_rate", info.get("win_rate", 0.0)), 0.0), 0.0, 1.0),
            trade_count=int(safe_float(ep_stats.get("total_trades", ep_stats.get("trade_count", 0)), 0)),
            winning_trades=int(safe_float(ep_stats.get("winning_trades", 0), 0)),
            losing_trades=int(safe_float(ep_stats.get("losing_trades", 0), 0)),
            max_drawdown=clamp(safe_float(ep_stats.get("max_drawdown", info.get("drawdown", 0.0)), 0.0), 0.0, 1.0),
            daily_drawdown=clamp(safe_float(ep_stats.get("daily_drawdown", 0.0), 0.0), 0.0, 1.0),
            dd_breach=bool(ep_stats.get("dd_breach", info.get("dd_breach", False))),
            avg_r_multiple=safe_float(ep_stats.get("avg_r_multiple", 0.0), 0.0),
            profit_factor=clamp(safe_float(ep_stats.get("profit_factor", 0.0), 0.0), 0.0, 10.0),
            avg_entry_quality=clamp(safe_float(ep_stats.get("avg_entry_quality", 0.5), 0.5), 0.0, 1.0),
            episode_length=self._episode_timesteps,
            episode_reward=self._episode_reward,
            termination_reason=str(ep_stats.get("termination_reason", "") or ""),
            global_episode_idx=self.manager.total_episodes + 1,
        )
        
        # Parse exit distribution if available
        exit_dist = ep_stats.get("exit_quality_distribution", {})
        if exit_dist:
            metrics.trailing_stop_exits = int(safe_float(exit_dist.get("trailing_stop", 0), 0))
            metrics.agent_close_exits = int(safe_float(exit_dist.get("agent_close", 0), 0))
            metrics.hard_stop_exits = int(safe_float(exit_dist.get("hard_stop", 0), 0))
            metrics.risk_liquidation_exits = int(safe_float(exit_dist.get("risk_liquidation", 0), 0))
        
        # Use on_episode_end() for deterministic episode recording and transition handling
        # This ensures idempotency (won't double-count if called twice)
        result = self.manager.on_episode_end(
            metrics=metrics,
            timesteps=self._episode_timesteps,
            effective_stage=self._effective_stage,
            check_transitions=True,
        )
        
        # Check for stage transitions from on_episode_end result
        old_stage = self.manager.current_stage
        transitioned = result.get("promoted", False) or result.get("demoted", False)
        new_stage_name = result.get("transition_to")
        
        if transitioned and new_stage_name:
            new_stage = CurriculumStage[new_stage_name]
            if self.verbose:
                direction = "↑" if result.get("promoted") else "↓"
                logger.info(f"Stage transition: {old_stage.name} {direction} {new_stage.name}")
            
            if self.stage_change_callback is not None:
                try:
                    self.stage_change_callback(old_stage, new_stage)
                except Exception as e:
                    logger.warning(f"Stage change callback error: {e}")
        
        # Add transition result to info for debugging
        info["episode_stats"]["transition_result"] = result
        
        return info
    
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
        
        # Restore new v2.1 fields for idempotency and state consistency
        self.manager._last_episode_end_idx = loaded._last_episode_end_idx
        self.manager._processed_episode_ids = loaded._processed_episode_ids
        self.manager._processed_episode_id_set = loaded._processed_episode_id_set
        self.manager._current_entropy = loaded._current_entropy
        self.manager._review_tick_episode = loaded._review_tick_episode
        self.manager._recovery_state = loaded._recovery_state
        self.manager._review_state = loaded._review_state
        
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
    initial_stage: CurriculumStage = CurriculumStage.EXPLORER,
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
            initial_stage=CurriculumStage.EXPLORER,
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
    initial_stage: CurriculumStage = CurriculumStage.EXPLORER,
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