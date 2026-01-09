"""
train/callbacks/curriculum_callback.py

CurriculumTrainingCallback - Training callback with curriculum integration.
Extends VecEpisodeTradingCallback with curriculum stage tracking,
automatic progression, LR warmup, and automatic checkpointing.

Extracted from train_prop_firm.py for modularity.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ..controllers import SmartEntropyController, TrainingHealthWatchdog

logger = logging.getLogger(__name__)


class CurriculumTrainingCallback(BaseCallback):
    """
    Training callback with curriculum integration.
    
    Extends VecEpisodeTradingCallback with curriculum stage tracking,
    automatic progression, LR warmup, and automatic checkpointing.
    
    Supports goal-based training termination when curriculum goals are met.
    
    LR Warmup Fix: Uses explicit closure factory to avoid variable capture bugs.
    """
    
    def __init__(
        self,
        curriculum_manager: Any,  # CurriculumManager
        total_timesteps: int,
        log_interval_steps: int = 50_000,
        save_path: str = "logs/curriculum",
        verbose: int = 1,
        enable_lr_warmup: bool = True,
        enable_checkpoints: bool = True,
        enable_entropy_schedule: bool = True,  # Adapt entropy by stage
        base_ent_coef: float = 0.02,  # Base entropy coefficient
        # NEW: Full adaptive control suite
        enable_adaptive_clip_range: bool = True,  # Adapt clip_range based on KL
        enable_adaptive_lr: bool = True,  # Adapt LR based on training dynamics
        base_clip_range: float = 0.2,  # Base clip range
        # Goal-based stopping parameters
        goal_based_stopping: bool = False,
        max_hours: Optional[float] = None,
        plateau_stop: bool = True,
        plateau_threshold_episodes: int = 500,
        max_demotions_from_same_stage: int = 5,
        mastery_confirmation_episodes: int = 100,
    ):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.total_timesteps = total_timesteps
        self.log_interval_steps = log_interval_steps
        self.save_path = Path(save_path)
        self.enable_lr_warmup = enable_lr_warmup
        self.enable_checkpoints = enable_checkpoints
        self.enable_entropy_schedule = enable_entropy_schedule
        self.base_ent_coef = base_ent_coef
        
        # NEW: Adaptive control suite
        self.enable_adaptive_clip_range = enable_adaptive_clip_range
        self.enable_adaptive_lr = enable_adaptive_lr
        self.base_clip_range = base_clip_range
        
        # Goal-based stopping
        self.goal_based_stopping = goal_based_stopping
        self.max_hours = max_hours
        self.plateau_stop = plateau_stop
        self.plateau_threshold_episodes = plateau_threshold_episodes
        self.max_demotions_from_same_stage = max_demotions_from_same_stage
        self.mastery_confirmation_episodes = mastery_confirmation_episodes
        
        # NEW: Training health watchdog
        self._health_watchdog = TrainingHealthWatchdog(
            ev_critical_threshold=-0.5,
            ev_warning_threshold=0.05,
            ev_consecutive_failures=5,
            check_interval_steps=25_000,
            auto_stop=False,  # Just alert, don't stop (set True to auto-stop)
        )
        
        self._last_log = 0
        self._last_metrics_save: float = 0.0  # Time-based metrics saving
        self._n_envs = 1
        self._last_ent_update_step = 0  # Track when we last updated entropy
        self._last_clip_update_step = 0  # Track when we last updated clip range
        self._last_lr_update_step = 0  # Track when we last updated learning rate
        
        # Episode tracking - AUDIT FIX: Use bounded deques to prevent memory leak on long runs
        MAX_EPISODE_HISTORY = 5000
        self._ep_rewards: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_pnls: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_win_rates: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_drawdowns: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_trades: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_lens: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        
        # AUDIT FIX (CRIT-1): Cumulative tracking for O(1) totals
        self._cumulative_pnl: float = 0.0
        self._cumulative_trades: int = 0
        
        # NEW: Trading quality metrics per episode - also bounded
        self._ep_profit_factors: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_r_multiples: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_entry_quality: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._exit_reason_counts: Dict[str, int] = {}
        
        # Reward component tracking for dashboard signals tab
        self._reward_component_totals: Dict[str, float] = {}
        self._reward_component_counts: Dict[str, int] = {}
        
        # Per-stage statistics tracking for dashboard Stage Progress tab
        # Maps stage_name -> {metrics dict}
        self._per_stage_stats: Dict[str, Dict[str, Any]] = {}
        self._current_stage_name: Optional[str] = None
        
        # PPO diagnostics (updated from logger)
        self._ppo_diagnostics: Dict[str, float] = {}
        self._n_updates: int = 0
        
        self._cur_rewards: List[float] = []
        self._cur_lens: List[int] = []
        
        # Stage tracking - bounded to prevent unbounded growth
        self._stage_history: deque = deque(maxlen=200)
        self._start_time: Optional[float] = None
        self._base_lr: Optional[float] = None
        self._current_lr: Optional[float] = None  # Track actual current LR (survives lr_schedule changes)
        
        # Adaptive control tracking (rolling windows for intelligent adjustment)
        self._entropy_history: deque = deque(maxlen=20)
        self._kl_history: deque = deque(maxlen=20)  # KL divergence for clip range
        self._clip_fraction_history: deque = deque(maxlen=20)  # Clip fraction tracking
        self._policy_loss_history: deque = deque(maxlen=20)  # Policy loss for LR
        self._value_loss_history: deque = deque(maxlen=20)  # Value loss for LR
        self._reward_history: deque = deque(maxlen=50)  # Recent rewards for performance
        
        # SMART ENTROPY CONTROLLER (PID-based, prevents oscillation)
        self._smart_entropy_controller = SmartEntropyController(initial_ent_coef=base_ent_coef)
        
        # Register transition callback
        if self.curriculum_manager is not None:
            self.curriculum_manager.on_transition_callback = self._on_stage_transition
    
    @property
    def episodes_done(self) -> int:
        """Number of completed episodes."""
        return len(self._ep_rewards)
    
    def _on_stage_transition(
        self,
        transition_type: str,
        old_stage: Any,
        new_stage: Any,
        info: Dict[str, Any],
    ) -> None:
        """Called when curriculum stage changes."""
        logger.info(f"🔄 Stage transition: {transition_type} ({old_stage.name} → {new_stage.name})")
        
        # Checkpoint on transition
        if self.enable_checkpoints and self.model is not None:
            try:
                checkpoint_dir = self.save_path / "stage_checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model
                model_path = checkpoint_dir / f"model_{old_stage.name}_to_{new_stage.name}_{self.num_timesteps}.zip"
                self.model.save(str(model_path))
                
                # Save curriculum state
                state_path = checkpoint_dir / f"curriculum_{old_stage.name}_to_{new_stage.name}_{self.num_timesteps}.json"
                self.curriculum_manager.save(state_path)
                
                logger.info(f"  📁 Checkpoint saved: {model_path.name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Checkpoint failed: {e}")
    
    def _on_training_start(self) -> None:
        env = self.training_env
        self._n_envs = int(getattr(env, "num_envs", 1))
        self._cur_rewards = [0.0] * self._n_envs
        self._cur_lens = [0] * self._n_envs
        self._start_time = time.time()
        self.training_start_time = self._start_time  # For goal-based stopping
        
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Store base learning rate and initialize current LR tracker
        if self.model is not None:
            self._base_lr = float(self.model.learning_rate) if not callable(self.model.learning_rate) else None
            self._current_lr = self._base_lr  # Initialize tracker to base LR
        
        # Save initial metrics file so dashboard sees data immediately
        self._save_live_metrics()
    
    def _apply_lr_warmup(self) -> None:
        """Apply learning rate warmup based on curriculum transition state.
        
        FIXED: Uses explicit value capture to avoid closure bugs where
        the lambda captures 'new_lr' by reference instead of value.
        """
        if not self.enable_lr_warmup or self.curriculum_manager is None or self.model is None:
            return
        
        if self._base_lr is None:
            return
        
        # Get LR multiplier from curriculum manager
        lr_mult = self.curriculum_manager.get_lr_multiplier()
        
        if lr_mult < 1.0:
            # Apply reduced LR during warmup
            new_lr = self._base_lr * lr_mult
            
            # FIXED: Explicitly capture new_lr value to avoid closure bug
            # The 'lr=new_lr' default argument captures the VALUE, not reference
            if hasattr(self.model, 'lr_schedule'):
                self.model.lr_schedule = lambda _, lr=new_lr: lr
            elif hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = new_lr
        else:
            # Warmup complete - restore base LR
            if hasattr(self.model, 'lr_schedule'):
                base = self._base_lr
                self.model.lr_schedule = lambda _, lr=base: lr
            elif hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = self._base_lr
    
    def _apply_entropy_schedule(self) -> None:
        """PID-based entropy controller - smooth, stable, no oscillation.
        
        This uses proper control theory (PID) instead of threshold-based logic.
        The PID controller:
        1. Tracks toward a TARGET entropy (not just bounds)
        2. Uses derivative term to prevent oscillation
        3. Uses integral term to eliminate steady-state error
        4. Has built-in damping and cooldown periods
        
        Uses NORMALIZED entropy targets for portability across:
        - Action-space changes
        - Masking intensity changes
        - New instruments/regimes
        
        This completely replaces the old oscillating threshold-based controller.
        """
        if not self.enable_entropy_schedule or self.curriculum_manager is None or self.model is None:
            return
        
        # Get current state
        current_entropy = self._entropy_history[-1] if self._entropy_history else 0.7
        current_ent_coef = float(getattr(self.model, 'ent_coef', self.base_ent_coef))
        stage = self.curriculum_manager.current_stage
        stage_value = int(stage.value)
        
        # Update smart controller's stage awareness
        self._smart_entropy_controller.on_stage_change(stage_value)
        
        # Calculate steps since last check
        steps_elapsed = self.num_timesteps - self._last_ent_update_step
        if steps_elapsed < 5_000:  # Minimum check interval
            return
            
        self._last_ent_update_step = self.num_timesteps
        
        # ENTROPY NORMALIZATION: Estimate number of valid actions from environment
        # This makes entropy targets portable across action space changes and masking
        n_valid_actions: Optional[int] = None
        try:
            if hasattr(self, 'training_env') and self.training_env is not None:
                venv = self.training_env
                # Unwrap to get base env with action_masks
                current_env: Any = venv
                while hasattr(current_env, 'venv'):
                    current_env = getattr(current_env, 'venv')
                if hasattr(current_env, 'envs'):
                    envs_list = getattr(current_env, 'envs')
                    if envs_list and len(envs_list) > 0:
                        base_env: Any = envs_list[0]
                        while hasattr(base_env, 'env'):
                            if hasattr(base_env, 'action_masks') and callable(getattr(base_env, 'action_masks')):
                                mask = base_env.action_masks()
                                n_valid_actions = int(np.sum(mask))
                                break
                            base_env = base_env.env
                        if n_valid_actions is None and hasattr(base_env, 'action_masks'):
                            mask = base_env.action_masks()
                            n_valid_actions = int(np.sum(mask))
        except Exception as e:
            logger.debug(f"Could not get action masks for entropy normalization: {e}")
        
        # Get PID recommendation with normalized entropy
        new_ent_coef, reason, should_apply = self._smart_entropy_controller.get_ent_coef(
            current_entropy=current_entropy,
            current_ent_coef=current_ent_coef,
            timesteps_elapsed=steps_elapsed,
            n_valid_actions=n_valid_actions,
        )
        
        # Apply if recommended
        if should_apply and hasattr(self.model, 'ent_coef'):
            setattr(self.model, 'ent_coef', new_ent_coef)
            if self.verbose >= 1:
                norm_target = self._smart_entropy_controller.STAGE_TARGETS_NORMALIZED[stage_value]
                max_h = self._smart_entropy_controller._get_max_entropy(n_valid_actions)
                raw_target = norm_target * max_h
                norm_current = current_entropy / max_h if max_h > 0 else 0
                logger.info(
                    f"🎛️ Entropy PID: {reason} | "
                    f"ent_coef: {current_ent_coef:.4f} → {new_ent_coef:.4f} | "
                    f"entropy: {current_entropy:.3f} (norm: {norm_current:.2f}) | "
                    f"target: {raw_target:.2f} (norm: {norm_target:.2f}) | "
                    f"valid_actions: {n_valid_actions or 'est'}"
                )

    def _apply_adaptive_clip_range(self) -> None:
        """ADAPTIVE clip range controller based on KL divergence.
        
        PPO's clip range controls how much the policy can change in one update.
        - Too small: Training is too slow, wastes samples
        - Too large: Policy changes too much, training becomes unstable
        
        The key signal is KL divergence:
        - High KL (>0.02): Policy changing too fast → reduce clip range
        - Low KL (<0.005): Policy barely changing → increase clip range
        - Sweet spot (~0.01): Policy changing at healthy rate
        
        Also monitors clip_fraction:
        - High clip_fraction (>0.3): Too many updates being clipped → increase range
        - Low clip_fraction (<0.05): Clip rarely triggers → might reduce range
        """
        if not self.enable_adaptive_clip_range or self.model is None:
            return
        
        # Check every 10k steps
        if self.num_timesteps - self._last_clip_update_step < 10_000:
            return
        
        self._last_clip_update_step = self.num_timesteps
        
        # AUDIT FIX: Get current clip_range value correctly
        # SB3 schedules expect progress_remaining (1.0 at start -> 0.0 at end)
        # Using 1.0 always returns the INITIAL value, not current
        progress = float(getattr(self.model, "_current_progress_remaining", 1.0))
        
        clip_range_attr = getattr(self.model, 'clip_range', self.base_clip_range)
        current_clip: float = self.base_clip_range
        if callable(clip_range_attr):
            try:
                raw_result = clip_range_attr(progress)
                if isinstance(raw_result, (int, float)):
                    current_clip = float(raw_result)
            except Exception as e:
                logger.debug(f"Could not evaluate clip_range schedule: {e}")
        elif isinstance(clip_range_attr, (int, float)):
            current_clip = float(clip_range_attr)
        
        current_kl = self._kl_history[-1] if self._kl_history else 0.01
        current_clip_frac = self._clip_fraction_history[-1] if self._clip_fraction_history else 0.1
        
        # === ADAPTIVE LOGIC ===
        target_clip = current_clip
        adjustment_reason = "stable"
        
        # 1. EMERGENCY: Clip fraction dangerously low (policy stuck)
        if current_clip_frac < 0.05:  # Raised threshold from 0.03
            # Almost no updates being clipped = policy frozen, need bigger steps
            # More aggressive: 50% increase, higher ceiling
            target_clip = min(0.40, current_clip * 1.50)  # 50% increase (was 25%)
            adjustment_reason = f"EMERGENCY: clip_frac {current_clip_frac:.3f} critically low - major boost"
        
        # 2. KL-based adjustment (primary signal)
        elif len(self._kl_history) >= 3:
            avg_kl = sum(list(self._kl_history)[-5:]) / min(5, len(self._kl_history))
            
            if avg_kl > 0.025:
                # Policy changing too fast - tighten clip range
                target_clip = current_clip * 0.9
                adjustment_reason = f"KL too high ({avg_kl:.4f}) - tightening"
            elif avg_kl < 0.005 and current_clip_frac < 0.1:
                # Policy barely changing - loosen clip range
                target_clip = current_clip * 1.15  # More aggressive than before
                adjustment_reason = f"KL too low ({avg_kl:.4f}) - loosening"
            elif current_clip_frac < 0.05:
                # Low clip fraction even with OK KL - still too conservative
                target_clip = current_clip * 1.1
                adjustment_reason = f"Low clip_frac ({current_clip_frac:.3f}) - slight boost"
        
        # 3. Clip fraction override (secondary signal)
        if current_clip_frac > 0.35:
            # Too many updates being clipped - increase range
            target_clip = max(target_clip, current_clip * 1.15)
            adjustment_reason = f"High clip_frac ({current_clip_frac:.2f}) - widening"
        
        # 4. Clamp to reasonable bounds
        target_clip = max(0.1, min(0.4, target_clip))  # PPO typically uses 0.1-0.3
        
        # === APPLY CHANGE ===
        if hasattr(self.model, 'clip_range') and abs(current_clip - target_clip) > 0.01:
            # SB3 expects clip_range as a callable schedule, not raw float
            # Create a constant function that returns the target value
            def constant_clip_schedule(progress: float, val: float = target_clip) -> float:
                return val
            
            setattr(self.model, 'clip_range', constant_clip_schedule)
            if self.verbose >= 1:
                logger.info(
                    f"Adaptive clip_range: {adjustment_reason} | "
                    f"{current_clip:.3f} -> {target_clip:.3f} | "
                    f"KL: {current_kl:.4f}, clip_frac: {current_clip_frac:.2f}"
                )

    def _apply_adaptive_learning_rate(self) -> None:
        """ADAPTIVE learning rate controller based on training dynamics.
        
        Monitors multiple signals to adjust LR:
        1. Loss plateau: If losses stop improving, reduce LR
        2. Loss instability: If losses spike, reduce LR
        3. Performance stagnation: If rewards plateau, try LR adjustment
        4. Stage-based scaling: Later stages may need finer updates
        
        Uses a multiplicative adjustment with momentum to avoid oscillation.
        
        CRITICAL FIX: Previous version was too aggressive - reduced LR 24x in 1 hour!
        Now: longer interval, stricter conditions, hard floor at 30% of base LR.
        """
        if not self.enable_adaptive_lr or self.model is None or self._base_lr is None:
            return
        
        # Check every 50k steps (was 20k - too frequent!)
        if self.num_timesteps - self._last_lr_update_step < 50_000:
            return
        
        self._last_lr_update_step = self.num_timesteps
        
        # AUDIT FIX: Get current LR from optimizer (most reliable source)
        current_lr = self._current_lr or self._base_lr
        try:
            if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
                current_lr = float(self.model.policy.optimizer.param_groups[0]["lr"])
        except Exception:
            pass  # Fall back to tracked/base value
        
        # Need enough history for meaningful decisions
        if len(self._policy_loss_history) < 5 or len(self._reward_history) < 10:
            return
        
        # === GATHER SIGNALS ===
        recent_p_loss = list(self._policy_loss_history)[-10:]
        recent_v_loss = list(self._value_loss_history)[-10:]
        recent_rewards = list(self._reward_history)[-20:]
        
        # Loss trends (negative slope = improving)
        p_loss_slope = (recent_p_loss[-1] - recent_p_loss[0]) / max(len(recent_p_loss), 1)
        v_loss_slope = (recent_v_loss[-1] - recent_v_loss[0]) / max(len(recent_v_loss), 1)
        
        # Loss variance (high = unstable)
        p_loss_var = float(np.var(recent_p_loss)) if len(recent_p_loss) > 1 else 0
        v_loss_var = float(np.var(recent_v_loss)) if len(recent_v_loss) > 1 else 0
        
        # Reward trend
        reward_early = sum(recent_rewards[:10]) / 10
        reward_late = sum(recent_rewards[-10:]) / 10
        reward_improving = reward_late > reward_early + 0.5
        
        # === ADAPTIVE LOGIC ===
        lr_multiplier = 1.0
        adjustment_reason = "stable"
        
        # MINIMUM LR FLOOR: Never go below 30% of base LR - this was killing training!
        lr_floor = self._base_lr * 0.3  # e.g., 3e-4 * 0.3 = 9e-5
        
        # 1. INSTABILITY: High loss variance → reduce LR (but gently)
        if p_loss_var > 0.1 or v_loss_var > 1.0:
            lr_multiplier = 0.9  # Reduced from 0.8 - less aggressive
            adjustment_reason = f"Unstable losses (p_var={p_loss_var:.3f}, v_var={v_loss_var:.3f})"
        
        # 2. PLATEAU: Only reduce if REALLY stuck (much stricter condition)
        # Previously this triggered every 20k steps and killed the LR
        elif p_loss_slope > 0.01 and v_loss_slope > 0.01 and not reward_improving:
            # Only reduce if losses are actually INCREASING (not just flat)
            lr_multiplier = 0.95  # Reduced from 0.9 - gentler reduction
            adjustment_reason = "Losses increasing - slight LR reduction"
        
        # 3. HEALTHY IMPROVEMENT: Good progress → slight increase (explore more)
        elif p_loss_slope < -0.01 and reward_improving:
            lr_multiplier = 1.1  # Increased from 1.05 - more aggressive recovery
            adjustment_reason = "Healthy progress - LR increase"
        
        # === APPLY CHANGE ===
        target_lr = current_lr * lr_multiplier
        
        # Clamp to reasonable bounds - CRITICAL: floor at 30% of base, not 1e-6!
        target_lr = max(lr_floor, min(self._base_lr * 2, target_lr))
        
        if abs(current_lr - target_lr) / current_lr > 0.05:  # >5% change
            # Apply to model schedule
            if hasattr(self.model, 'lr_schedule'):
                self.model.lr_schedule = lambda _, lr=target_lr: lr
            if hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = target_lr
            
            # AUDIT FIX: Also push directly to optimizer for immediate effect
            try:
                if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
                    for g in self.model.policy.optimizer.param_groups:
                        g["lr"] = float(target_lr)
            except Exception:
                pass
            
            # CRITICAL: Persist adjusted value for future calls
            self._current_lr = target_lr
            
            if self.verbose >= 1:
                logger.info(
                    f"Adaptive LR: {adjustment_reason} | "
                    f"{current_lr:.2e} -> {target_lr:.2e} | "
                    f"p_loss_slope: {p_loss_slope:.4f}, reward_improving: {reward_improving}"
                )

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if rewards is None or dones is None or infos is None:
            return True
        
        # Update transition state and all adaptive controls
        # F-6 FIX: Skip step_transition_state here - PropFirmTradingEnv already calls it
        # per env step (timesteps=1). Calling again here would double-count transitions.
        if self.curriculum_manager is not None:
            # self.curriculum_manager.step_transition_state(timesteps=self._n_envs)  # F-6: REMOVED - env handles this
            self._apply_lr_warmup()
            self._apply_entropy_schedule()  # Adapt entropy by stage
            self._apply_adaptive_clip_range()  # Adapt clip range by KL
            self._apply_adaptive_learning_rate()  # Adapt LR by training dynamics

        rewards_arr = np.array(rewards, dtype=np.float64).reshape(-1)
        dones_arr = np.array(dones, dtype=np.bool_).reshape(-1)

        for i in range(min(self._n_envs, len(rewards_arr))):
            self._cur_rewards[i] += float(rewards_arr[i])
            self._cur_lens[i] += 1

        for i, done in enumerate(dones_arr[:self._n_envs]):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            ep_reward = self._cur_rewards[i]
            ep_len = self._cur_lens[i]
            
            finfo = info.get("terminal_info", info)
            
            # Get episode_stats if available (contains detailed metrics)
            ep_stats = finfo.get("episode_stats", info.get("episode_stats", {}))
            
            # Record basic metrics
            self._ep_rewards.append(ep_reward)
            self._ep_lens.append(ep_len)
            pnl = float(finfo.get("total_pnl", info.get("total_pnl", 0.0)))
            trades = int(finfo.get("trade_count", info.get("trade_count", 0)))
            self._ep_pnls.append(pnl)
            self._ep_win_rates.append(float(finfo.get("win_rate", info.get("win_rate", 0.0))))
            self._ep_drawdowns.append(float(finfo.get("drawdown", info.get("drawdown", 0.0))))
            self._ep_trades.append(trades)
            
            # Track reward for adaptive LR controller
            self._reward_history.append(ep_reward)
            
            # AUDIT FIX (CRIT-1): Accumulate for O(1) totals
            self._cumulative_pnl += pnl
            self._cumulative_trades += trades
            
            # Record trading quality metrics
            self._ep_profit_factors.append(float(ep_stats.get("profit_factor", finfo.get("profit_factor", 0.0))))
            self._ep_r_multiples.append(float(ep_stats.get("avg_r_multiple", finfo.get("avg_r_multiple", 0.0))))
            self._ep_entry_quality.append(float(ep_stats.get("avg_entry_quality", finfo.get("avg_entry_quality", 0.5))))
            
            # Track exit reasons from episode_stats (key is exit_quality_distribution)
            exit_dist = ep_stats.get("exit_quality_distribution", ep_stats.get("exit_distribution", {}))
            for reason, count in exit_dist.items():
                self._exit_reason_counts[reason] = self._exit_reason_counts.get(reason, 0) + int(count)
            
            # Track reward components from episode_stats for dashboard Signals tab
            reward_components = ep_stats.get("reward_components", {}) or {}
            for comp_name, comp_data in reward_components.items():
                if isinstance(comp_data, dict):
                    total = float(comp_data.get("total", 0.0))
                    count = int(comp_data.get("count", 0))
                else:
                    total = float(comp_data)
                    count = 1
                self._reward_component_totals[comp_name] = self._reward_component_totals.get(comp_name, 0.0) + total
                self._reward_component_counts[comp_name] = self._reward_component_counts.get(comp_name, 0) + count
            
            # Track per-stage statistics for Stage Progress dashboard tab
            if self.curriculum_manager is not None:
                stage_name = self.curriculum_manager.current_stage.name
            else:
                stage_name = finfo.get("curriculum_stage", info.get("curriculum_stage", "unknown"))
            self._record_stage_episode_stats(
                stage_name=stage_name,
                pnl=pnl,
                win_rate=float(finfo.get("win_rate", info.get("win_rate", 0.0))),
                drawdown=float(finfo.get("drawdown", info.get("drawdown", 0.0))),
                trades=trades,
                profit_factor=float(ep_stats.get("profit_factor", finfo.get("profit_factor", 0.0))),
                r_multiple=float(ep_stats.get("avg_r_multiple", finfo.get("avg_r_multiple", 0.0))),
                reward=ep_reward,
            )
            
            # NOTE: episode_transition_tick is called by PropFirmTradingEnv._episode_end_hook()
            # via record_episode_from_info(), so we don't call it here to avoid double-counting
            
            # Track stage transitions (get from curriculum manager directly)
            if self.curriculum_manager is not None:
                current_stage = self.curriculum_manager.current_stage.name
            else:
                current_stage = finfo.get("curriculum_stage", info.get("curriculum_stage", "unknown"))
            if self._stage_history and self._stage_history[-1]["stage"] != current_stage:
                self._stage_history.append({
                    "stage": current_stage,
                    "timestep": self.num_timesteps,
                    "episode": len(self._ep_rewards),
                })
            elif not self._stage_history:
                self._stage_history.append({
                    "stage": current_stage,
                    "timestep": self.num_timesteps,
                    "episode": 1,
                })
            
            # Reset tracking
            self._cur_rewards[i] = 0.0
            self._cur_lens[i] = 0
            
            # Save live metrics
            self._save_live_metrics()

        # Collect PPO diagnostics from SB3 logger
        self._update_ppo_diagnostics()
        
        # === TRAINING HEALTH CHECK ===
        # Run the watchdog to detect critical issues early
        if hasattr(self, '_health_watchdog') and self._health_watchdog is not None:
            ev = self._ppo_diagnostics.get('explained_variance', 0.5)
            mean_reward = float(np.mean(list(self._ep_rewards)[-50:])) if self._ep_rewards else 0
            win_rate = float(np.mean(list(self._ep_win_rates)[-50:])) if self._ep_win_rates else 0.5
            entropy = float(self._entropy_history[-1]) if self._entropy_history else 0.7
            
            should_stop, reason = self._health_watchdog.check(
                explained_variance=ev,
                mean_reward=mean_reward,
                win_rate=win_rate / 100.0 if win_rate > 1 else win_rate,  # Normalize if percentage
                entropy=entropy,
                timestep=self.num_timesteps,
            )
            # If auto_stop is True in watchdog and critical failure detected
            if should_stop:
                logger.error(f"⛔ TRAINING HALTED BY WATCHDOG: {reason}")
                self._save_live_metrics()
                return False

        # Periodic logging
        if self.num_timesteps - self._last_log >= self.log_interval_steps:
            self._log_progress()
            self._last_log = self.num_timesteps
        
        # Save metrics every 1 second (time-based, not step-based)
        now = time.time()
        if now - self._last_metrics_save >= 1.0:
            self._save_live_metrics()
            self._last_metrics_save = now

        # Goal-based stopping check (only if enabled)
        if self.goal_based_stopping and self.curriculum_manager is not None:
            # Check periodically (after each episode completion is enough)
            if self.episodes_done > 0 and self.episodes_done % 10 == 0:
                should_stop, reason = self.curriculum_manager.should_stop_training(
                    max_timesteps=None,  # Let SB3 handle max timesteps
                    max_episodes=None,
                    max_hours=self.max_hours,
                    start_time=self.training_start_time,
                    plateau_stop=self.plateau_stop,
                    plateau_threshold_episodes=self.plateau_threshold_episodes,
                    max_demotions_from_same_stage=self.max_demotions_from_same_stage,
                    mastery_confirmation_episodes=self.mastery_confirmation_episodes,
                )
                if should_stop:
                    logger.info(f"🎯 Goal-based stopping triggered: {reason}")
                    self._log_progress()  # Final log
                    self._save_live_metrics()  # Final save
                    return False  # Stop training

        return True
    
    def _update_ppo_diagnostics(self) -> None:
        """Extract PPO training diagnostics from the model/logger."""
        if self.model is None:
            return
        
        # Try to get diagnostics from the model's logger
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logger_obj = self.model.logger
                
                # SB3 stores values in name_to_value dict
                if hasattr(logger_obj, 'name_to_value'):
                    values = logger_obj.name_to_value
                    
                    # Extract common PPO metrics
                    # F-5 FIX: Use 'train/entropy' (true entropy) not 'train/entropy_loss' (scaled by -ent_coef)
                    self._ppo_diagnostics['policy_loss'] = float(values.get('train/policy_gradient_loss', values.get('train/policy_loss', 0)))
                    self._ppo_diagnostics['value_loss'] = float(values.get('train/value_loss', 0))
                    self._ppo_diagnostics['entropy'] = float(values.get('train/entropy', values.get('train/entropy_loss', 0)))
                    self._ppo_diagnostics['kl_divergence'] = float(values.get('train/approx_kl', 0))
                    self._ppo_diagnostics['clip_fraction'] = float(values.get('train/clip_fraction', 0))
                    self._ppo_diagnostics['explained_variance'] = float(values.get('train/explained_variance', 0))
                    self._ppo_diagnostics['learning_rate'] = float(values.get('train/learning_rate', 0))
                    self._ppo_diagnostics['clip_range'] = float(values.get('train/clip_range', 0.2))
                    
                    # Track n_updates
                    n_updates = values.get('train/n_updates', 0)
                    if n_updates > 0:
                        self._n_updates = int(n_updates)
                    
                    # Update curriculum manager with current entropy for entropy-based reward shaping
                    if self.curriculum_manager is not None and self._ppo_diagnostics['entropy'] != 0:
                        entropy_val = abs(self._ppo_diagnostics['entropy'])
                        self.curriculum_manager.update_entropy(entropy_val)
                        # Track for adaptive entropy controller
                        self._entropy_history.append(entropy_val)
                    
                    # Track KL and clip_fraction for adaptive clip range controller
                    kl_val = self._ppo_diagnostics.get('kl_divergence', 0)
                    clip_frac = self._ppo_diagnostics.get('clip_fraction', 0)
                    if kl_val > 0:
                        self._kl_history.append(kl_val)
                    if clip_frac > 0:
                        self._clip_fraction_history.append(clip_frac)
                    
                    # Track losses for adaptive LR controller
                    p_loss = abs(self._ppo_diagnostics.get('policy_loss', 0))
                    v_loss = abs(self._ppo_diagnostics.get('value_loss', 0))
                    if p_loss > 0:
                        self._policy_loss_history.append(p_loss)
                    if v_loss > 0:
                        self._value_loss_history.append(v_loss)
            
            # Alternative: get from model attributes
            if self._n_updates == 0 and hasattr(self.model, '_n_updates'):
                self._n_updates = int(self.model._n_updates)
            
            # Fallback: get learning rate from model if SB3 logger didn't provide it
            if self._ppo_diagnostics.get('learning_rate', 0) == 0 and hasattr(self.model, 'learning_rate'):
                lr = self.model.learning_rate
                if callable(lr):
                    # Learning rate schedule - evaluate at current progress
                    progress = self.num_timesteps / max(getattr(self.model, '_total_timesteps', 1), 1)
                    lr = lr(1.0 - progress)
                self._ppo_diagnostics['learning_rate'] = float(lr)
                
        except Exception:
            pass  # Silently fail - diagnostics are optional
    
    def _record_stage_episode_stats(
        self,
        stage_name: str,
        pnl: float,
        win_rate: float,
        drawdown: float,
        trades: int,
        profit_factor: float,
        r_multiple: float,
        reward: float,
    ) -> None:
        """
        Record episode statistics for a specific stage.
        This enables per-stage comparisons on the dashboard.
        """
        if stage_name not in self._per_stage_stats:
            self._per_stage_stats[stage_name] = {
                "stage_name": stage_name,
                "episodes": 0,
                "total_pnl": 0.0,
                "total_trades": 0,
                "total_wins": 0,
                "total_losses": 0,
                "total_drawdown": 0.0,
                "total_reward": 0.0,
                "sum_profit_factor": 0.0,
                "sum_r_multiple": 0.0,
                "pnl_values": [],  # Keep last N for variance
                "win_rate_values": [],
                "first_episode": len(self._ep_rewards),
                "first_timestep": self.num_timesteps,
                "last_episode": 0,
                "last_timestep": 0,
            }
        
        stats = self._per_stage_stats[stage_name]
        stats["episodes"] += 1
        stats["total_pnl"] += pnl
        stats["total_trades"] += trades
        stats["total_drawdown"] += drawdown
        stats["total_reward"] += reward
        stats["sum_profit_factor"] += profit_factor
        stats["sum_r_multiple"] += r_multiple
        stats["last_episode"] = len(self._ep_rewards)
        stats["last_timestep"] = self.num_timesteps
        
        # Track wins/losses (win_rate is 0-1 range)
        if trades > 0:
            wins = int(round(win_rate * trades))
            losses = trades - wins
            stats["total_wins"] += wins
            stats["total_losses"] += losses
        
        # Keep last 100 values for variance calculation
        stats["pnl_values"].append(pnl)
        stats["win_rate_values"].append(win_rate)
        if len(stats["pnl_values"]) > 100:
            stats["pnl_values"] = stats["pnl_values"][-100:]
        if len(stats["win_rate_values"]) > 100:
            stats["win_rate_values"] = stats["win_rate_values"][-100:]
    
    def _get_stage_comparison_data(self) -> Dict[str, Any]:
        """
        Compute stage comparison data for the dashboard.
        Returns per-stage metrics and stage-to-stage improvements.
        """
        stage_order = [
            "EXPLORER", "EXPERIMENTER", "TREND_STUDENT", "SESSION_STUDENT",
            "TIMING_STUDENT", "INTEGRATOR", "RISK_MANAGER", "STRATEGIST",
            "PROFESSIONAL", "LIVE_READY"
        ]
        
        stages_data = []
        prev_stats = None
        
        for stage_name in stage_order:
            stats = self._per_stage_stats.get(stage_name)
            if stats is None or stats["episodes"] == 0:
                continue
            
            episodes = stats["episodes"]
            total_trades = stats["total_trades"]
            
            # Calculate averages
            avg_pnl = stats["total_pnl"] / episodes if episodes > 0 else 0
            avg_trades = total_trades / episodes if episodes > 0 else 0
            avg_drawdown = stats["total_drawdown"] / episodes if episodes > 0 else 0
            avg_reward = stats["total_reward"] / episodes if episodes > 0 else 0
            avg_profit_factor = stats["sum_profit_factor"] / episodes if episodes > 0 else 0
            avg_r_multiple = stats["sum_r_multiple"] / episodes if episodes > 0 else 0
            
            # Win rate from total wins/losses
            total_trades_wl = stats["total_wins"] + stats["total_losses"]
            win_rate = stats["total_wins"] / total_trades_wl if total_trades_wl > 0 else 0
            
            # Calculate variance from stored values
            pnl_std = float(np.std(stats["pnl_values"])) if len(stats["pnl_values"]) > 1 else 0
            win_rate_std = float(np.std(stats["win_rate_values"])) if len(stats["win_rate_values"]) > 1 else 0
            
            # Build stage entry
            stage_entry = {
                "stage_name": stage_name,
                "stage_index": stage_order.index(stage_name),
                "episodes": episodes,
                "timesteps": stats["last_timestep"] - stats["first_timestep"],
                "total_trades": total_trades,
                "total_pnl": stats["total_pnl"],
                "avg_pnl": avg_pnl,
                "pnl_std": pnl_std,
                "win_rate": win_rate * 100,  # As percentage
                "win_rate_std": win_rate_std * 100,
                "avg_drawdown": avg_drawdown * 100,  # As percentage
                "avg_trades": avg_trades,
                "avg_reward": avg_reward,
                "avg_profit_factor": avg_profit_factor,
                "avg_r_multiple": avg_r_multiple,
                "first_episode": stats["first_episode"],
                "last_episode": stats["last_episode"],
            }
            
            # Calculate improvements from previous stage
            if prev_stats is not None:
                prev_win_rate = prev_stats.get("win_rate", 0)
                prev_pnl = prev_stats.get("avg_pnl", 0)
                prev_pf = prev_stats.get("avg_profit_factor", 0)
                prev_reward = prev_stats.get("avg_reward", 0)
                
                stage_entry["improvement"] = {
                    "win_rate_delta": (win_rate * 100) - prev_win_rate,
                    "pnl_delta": avg_pnl - prev_pnl,
                    "profit_factor_delta": avg_profit_factor - prev_pf,
                    "reward_delta": avg_reward - prev_reward,
                }
            else:
                stage_entry["improvement"] = None
            
            stages_data.append(stage_entry)
            prev_stats = stage_entry
        
        # Calculate overall progression (first to current)
        overall_improvement = {}
        if len(stages_data) >= 2:
            first = stages_data[0]
            last = stages_data[-1]
            overall_improvement = {
                "win_rate_delta": last["win_rate"] - first["win_rate"],
                "pnl_delta": last["avg_pnl"] - first["avg_pnl"],
                "profit_factor_delta": last["avg_profit_factor"] - first["avg_profit_factor"],
                "reward_delta": last["avg_reward"] - first["avg_reward"],
                "stages_progressed": last["stage_index"] - first["stage_index"],
            }
        
        return {
            "stages": stages_data,
            "overall_improvement": overall_improvement,
            "current_stage": self.curriculum_manager.current_stage.name if self.curriculum_manager else "N/A",
            "total_stages_visited": len(stages_data),
        }
    
    def _log_progress(self) -> None:
        """Log training progress."""
        if not self._ep_rewards:
            return

        # AUDIT FIX: deques don't support slicing - convert to list first
        n = min(50, len(self._ep_rewards))
        rewards = list(self._ep_rewards)[-n:]
        pnls = list(self._ep_pnls)[-n:]
        win_rates = list(self._ep_win_rates)[-n:]
        drawdowns = list(self._ep_drawdowns)[-n:]
        trades = list(self._ep_trades)[-n:]

        mean_reward = float(np.mean(rewards))
        mean_pnl = float(np.mean(pnls))
        mean_wr = float(np.mean(win_rates))
        max_dd = float(np.max(drawdowns))
        mean_trades = float(np.mean(trades))

        progress = 100.0 * (self.num_timesteps / max(self.total_timesteps, 1))
        
        stage = self.curriculum_manager.current_stage.name if self.curriculum_manager else "N/A"

        logger.info(
            f"Step {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1f}%) | "
            f"Stage: {stage} | "
            f"Reward {mean_reward:+.3f} | PnL €{mean_pnl:+.0f} | "
            f"WR {mean_wr:.1%} | Trades {mean_trades:.1f} | MaxDD {max_dd:.1%}"
        )
    
    def _save_live_metrics(self) -> None:
        """Save metrics for dashboard."""
        self._save_live_metrics_inline()
    
    def _save_live_metrics_inline(self) -> None:
        """Inline implementation of metrics saving."""
        try:
            metrics_file = Path("logs/training/live_metrics.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            n_recent = min(100, len(self._ep_rewards))
            
            # Compute FPS
            fps = 0.0
            if self._start_time is not None and self.num_timesteps > 0:
                elapsed = time.time() - self._start_time
                if elapsed > 0:
                    fps = self.num_timesteps / elapsed
            
            # Calculate ETA
            eta_seconds = 0.0
            if self._start_time and self.num_timesteps > 0:
                elapsed = time.time() - self._start_time
                remaining = self.total_timesteps - self.num_timesteps
                rate = self.num_timesteps / max(elapsed, 1)
                eta_seconds = remaining / max(rate, 1)
            
            # Calculate summary stats
            mean_reward = float(np.mean(list(self._ep_rewards)[-50:])) if self._ep_rewards else 0.0
            mean_pnl = float(np.mean(list(self._ep_pnls)[-50:])) if self._ep_pnls else 0.0
            total_pnl = self._cumulative_pnl
            mean_win_rate = float(np.mean(list(self._ep_win_rates)[-50:])) if self._ep_win_rates else 0.0
            max_drawdown = float(np.max(list(self._ep_drawdowns)[-50:])) if self._ep_drawdowns else 0.0
            mean_trades = float(np.mean(list(self._ep_trades)[-50:])) if self._ep_trades else 0.0
            total_trades = self._cumulative_trades
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "timesteps": self.num_timesteps,
                    "total_timesteps": self.total_timesteps,
                    "progress_pct": 100.0 * (self.num_timesteps / max(self.total_timesteps, 1)),
                    "total_episodes": len(self._ep_rewards),
                    "eta_seconds": eta_seconds,
                },
                "learning": {
                    "fps": fps,
                    "n_updates": self._n_updates,
                    "mean_reward": mean_reward,
                    "total_pnl": total_pnl,
                    "policy_loss": self._ppo_diagnostics.get('policy_loss', 0),
                    "value_loss": self._ppo_diagnostics.get('value_loss', 0),
                    "entropy": self._ppo_diagnostics.get('entropy', 0),
                    "kl_divergence": self._ppo_diagnostics.get('kl_divergence', 0),
                    "clip_fraction": self._ppo_diagnostics.get('clip_fraction', 0),
                    "explained_variance": self._ppo_diagnostics.get('explained_variance', 0),
                    "learning_rate": self._ppo_diagnostics.get('learning_rate', 0),
                },
                "trading": {
                    "total_trades": total_trades,
                    "mean_trades": mean_trades,
                    "mean_win_rate": mean_win_rate * 100,
                    "max_drawdown": max_drawdown * 100,
                },
                "curriculum_stage": self.curriculum_manager.current_stage.name if self.curriculum_manager else "N/A",
                "recent_rewards": [float(x) for x in list(self._ep_rewards)[-n_recent:]],
                "recent_pnls": [float(x) for x in list(self._ep_pnls)[-n_recent:]],
                "stage_history": list(self._stage_history),
                "stage_comparison": self._get_stage_comparison_data(),
                # Legacy flat fields
                "timesteps": self.num_timesteps,
                "total_timesteps": self.total_timesteps,
                "mean_reward": mean_reward,
                "mean_pnl": mean_pnl,
                "total_pnl": total_pnl,
                "mean_win_rate": mean_win_rate,
                "max_drawdown": max_drawdown,
            }
            
            # Atomic write
            tmp_file = metrics_file.with_suffix('.json.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f)
            
            try:
                tmp_file.replace(metrics_file)
            except PermissionError:
                import shutil
                try:
                    shutil.copy2(tmp_file, metrics_file)
                    tmp_file.unlink(missing_ok=True)
                except Exception:
                    with open(metrics_file, 'w', encoding='utf-8') as f:
                        json.dump(metrics, f)
                    tmp_file.unlink(missing_ok=True)
                    
        except Exception as e:
            import traceback
            logger.error(f"Failed to save live metrics: {e}")
            traceback.print_exc()
