# envs/curriculum_manager.py
"""
Curriculum Manager for Trading RL Agent
========================================

Tracks agent competence across multiple metrics and determines when to promote
(or demote) the agent to the next curriculum stage.

Enhancements in this version (v2.0):
- Learning velocity tracking with plateau detection
- Skill-based decomposed competency assessment
- Demotion diagnosis and recovery protocols
- Composite competence scoring with hard floors
- Adaptive threshold relaxation
- Mixed-stage sampling for catastrophic forgetting prevention
- Review session scheduling
- Entropy tracking and management
- Rich progress reporting with actionable recommendations

Key Principles:
1. Progress by STATISTICAL CONFIDENCE, not time or luck
2. Multiple metrics must ALL pass thresholds (no single-metric gaming)
3. Stability matters as much as performance (low variance required)
4. Demotion is possible if performance degrades significantly
5. Rolling window evaluation prevents overfitting to recent episodes
6. Skill decomposition enables targeted improvement
7. Adaptive thresholds prevent permanent plateaus
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Deque, Dict, List, Optional, Tuple, Set
from zoneinfo import ZoneInfo

import numpy as np

from envs.curriculum.curriculum_config import (
    CurriculumStage,
    CurriculumStageConfig,
    CompetenceThresholds,
    TradingSkill,
    SkillRequirements,
    CompositeScoringConfig,
    AdaptiveThresholdConfig,
    RecoveryProtocolConfig,
    MIN_EVALUATION_EPISODES,
    get_stage_config,
    get_next_stage,
    get_previous_stage,
    get_stage_progression,
    # NOTE: DataDifficulty, TransitionSettings, EntropyTargets, 
    # MixedStageSamplingConfig, ReviewSessionConfig are accessed via 
    # CurriculumStageConfig attributes (e.g., stage_config.mixed_stage_sampling)
    # rather than as direct type annotations, so they're not imported here.
)

# DUP-2 FIX: Use shared utilities for common functions
from envs.core.shared_utils import (
    safe_float as _sf,
    safe_int as _si,
    clamp as _cl,
    wilson_interval as _wi,
    mean_ci_normal as _mci,
    get_envs_logger,
    iso_timestamp,
)

# Import from curriculum subpackage
from envs.curriculum.metrics import (
    EpisodeMetrics,
    RollingStats,
    LearningVelocity,
    CompositeScore,
    compute_composite_score,
    compute_adjusted_thresholds,
)
from envs.curriculum.skills import (
    SkillAssessment,
    DemotionRecord,
    DemotionAnalyzer,
)
from envs.curriculum.protocols import (
    RecoveryProtocolState,
    ReviewSessionState,
)

# Phase 1-4 hardening modules
from envs.curriculum.curriculum_invariants import (
    CurriculumInvariantChecker,
    reconcile_trade_accounting,
    AntiGamingChecker,
    InvariantViolation,
)
from envs.curriculum.validation_gates import (
    ValidationGateChecker,
    ValidationGateConfig,
    StressTestRunner,
    StressTestConfig,
)
from envs.curriculum.regime_skill_assessment import (
    RegimeSkillAssessment,
    TradeWithRegime,
)

logger = get_envs_logger("curriculum_manager")

STATE_VERSION = "2.1"  # Bumped for Phase 1-4 additions
DEFAULT_TZ = "Europe/Berlin"

# Limits for persistence
DEMOTION_HISTORY_LIMIT = 200
PROCESSED_EPISODE_ID_LIMIT = 5000
MIN_TRADES_FOR_WILSON_GATE_DEFAULT = 30


# =============================================================================
# Utility Functions (Aliases to shared_utils for backward compatibility)
# =============================================================================

def _now_iso(tz: str = DEFAULT_TZ) -> str:
    try:
        return datetime.now(tz=ZoneInfo(tz)).isoformat()
    except Exception as e:
        logger.debug(f"Timezone fallback for {tz}: {e}")
        return iso_timestamp()


# DEPRECATED: Use envs.shared_utils.safe_float() directly
def _safe_float(x: Any, default: float = 0.0) -> float:
    return _sf(x, default)


# DEPRECATED: Use envs.shared_utils.safe_int() directly
def _safe_int(x: Any, default: int = 0) -> int:
    return _si(x, default)


# DEPRECATED: Use envs.shared_utils.clamp() directly
def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value to range [lo, hi], handling NaN/inf safely."""
    return _cl(v, lo, hi)


# DEPRECATED: Use envs.shared_utils.wilson_interval() directly
def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a Bernoulli proportion.
    Returns (low, high). For promotion we usually use the LOWER bound.
    """
    return _wi(k, n, z)


# DEPRECATED: Use envs.shared_utils.mean_ci_normal() directly
def _mean_ci_normal(mean: float, std: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    return _mci(mean, std, n, z)


# =============================================================================
# Safe Stage Progression Helpers
# =============================================================================
# These helpers ensure stage sampling/review work correctly even if
# CurriculumStage enum values are reordered or non-contiguous.

def _get_progression() -> List[CurriculumStage]:
    """Get the canonical stage progression list."""
    return get_stage_progression()


def _stage_to_index(stage: CurriculumStage) -> int:
    """Convert a stage to its index in the progression (0-based)."""
    prog = _get_progression()
    try:
        return prog.index(stage)
    except ValueError:
        return 0


def _index_to_stage(idx: int) -> CurriculumStage:
    """Convert a progression index to a stage (clamped to valid range)."""
    prog = _get_progression()
    idx = max(0, min(idx, len(prog) - 1))
    return prog[idx]


def _foundation_stage() -> CurriculumStage:
    """Get the foundation stage (first stage in progression).
    
    This avoids hard-coding CurriculumStage.EXPLORER, making the manager
    resilient to any stage naming scheme in curriculum_config.py.
    """
    return _get_progression()[0]


def _linear_regression_slope(y: np.ndarray) -> float:
    """Compute slope of linear regression for trend detection."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    try:
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])
    except Exception as e:
        logger.debug(f"Linear regression failed: {e}")
        return 0.0



# =============================================================================
# Main Curriculum Manager
# =============================================================================

class CurriculumManager:
    """
    Manages curriculum progression for the trading RL agent.
    
    Version 2.0 Features:
    - Learning velocity tracking with plateau detection
    - Skill-based decomposed competency assessment
    - Demotion diagnosis and recovery protocols
    - Composite competence scoring with hard floors
    - Adaptive threshold relaxation
    - Mixed-stage sampling for catastrophic forgetting prevention
    - Review session scheduling
    - Entropy tracking and management
    """
    
    def __init__(
        self,
        initial_stage: Optional[CurriculumStage] = None,
        max_history_size: int = 1000,
        auto_promote: bool = True,
        auto_demote: bool = True,
        verbose: bool = True,
        tz: str = DEFAULT_TZ,
        on_transition_callback: Optional[Callable] = None,
        rng_seed: Optional[int] = None,
        validation_evaluator: Optional[Callable[[CurriculumStageConfig], Dict[str, Any]]] = None,
        bars_per_trading_day: int = 96,  # M15 default; override for other timeframes
    ) -> None:
        # Use foundation stage if not specified (avoids hard-coding EXPLORER)
        if initial_stage is None:
            initial_stage = _foundation_stage()
        self.current_stage = initial_stage
        self.max_history_size = max_history_size
        self.auto_promote = auto_promote
        self.auto_demote = auto_demote
        self.verbose = verbose
        self.tz = tz
        self.on_transition_callback = on_transition_callback
        self.validation_evaluator = validation_evaluator
        self.bars_per_trading_day = bars_per_trading_day
        
        # Random number generator for mixed-stage sampling
        self._rng = np.random.default_rng(rng_seed)
        
        # Metrics history per stage
        self._history: Dict[CurriculumStage, Deque[EpisodeMetrics]] = {
            stage: deque(maxlen=max_history_size) for stage in CurriculumStage
        }
        
        # Totals per stage (lifetime)
        self._stage_timesteps_total: Dict[CurriculumStage, int] = {stage: 0 for stage in CurriculumStage}
        self._stage_episodes_total: Dict[CurriculumStage, int] = {stage: 0 for stage in CurriculumStage}
        
        # Current stage-epoch counters
        self.stage_timesteps: int = 0
        self.stage_episodes: int = 0
        
        # Stage epoch counters
        self._stage_epoch_counter: Dict[CurriculumStage, int] = {stage: 0 for stage in CurriculumStage}
        self._current_stage_epoch: int = 0
        
        # Promotion/demotion history
        self._transitions: List[Dict[str, Any]] = []
        
        # Cached rolling stats
        self._rolling_stats: Optional[RollingStats] = None
        self._rolling_stats_dirty: bool = True
        
        # Track totals
        self.total_timesteps: int = 0
        self.total_episodes: int = 0
        
        # Transition state tracking
        self._transition_cooldown_remaining: int = 0
        self._reward_blend_remaining: int = 0
        self._previous_stage_config: Optional[CurriculumStageConfig] = None
        # Persistable provenance for reward blending across resumes
        self._previous_stage_name: Optional[str] = None
        self._lr_warmup_active: bool = False
        self._lr_warmup_steps_remaining: int = 0
        self._lr_warmup_factor: float = 1.0
        
        # New v2.0 components
        self._learning_velocity = LearningVelocity()
        self._skill_assessment: Optional[SkillAssessment] = None
        self._demotion_analyzer = DemotionAnalyzer()
        self._recovery_state = RecoveryProtocolState()
        self._review_state = ReviewSessionState()
        self._composite_score: Optional[CompositeScore] = None
        
        # Phase 1-4 hardening components (v2.1)
        self._invariant_checker = CurriculumInvariantChecker(verbose=verbose)
        self._anti_gaming_checker = AntiGamingChecker()
        self._validation_gate = ValidationGateChecker()
        self._stress_tester = StressTestRunner()
        self._regime_assessment: Optional[RegimeSkillAssessment] = None
        
        # Regime trade accumulator for RegimeSkillAssessment
        # Bounded to prevent memory issues during long training runs
        self._regime_trades: Deque[TradeWithRegime] = deque(maxlen=2000)
        self._regime_assessment_dirty: bool = True

        
        # Episode-end call tracking (Phase 1.2: ensure single call per episode)
        self._last_episode_end_idx: int = -1
        
        # True idempotency tracking: track processed episode IDs (bounded to avoid memory leak)
        self._processed_episode_ids: Deque[int] = deque()
        self._processed_episode_id_set: Set[int] = set()
        
        # Review tick latch: prevents double-ticking review counters within same episode
        self._review_tick_episode: int = -1
        
        # Effective stage caching (Phase 1.3: prevent non-determinism)
        # Caches the effective stage per "next episode index" so multiple calls
        # within the same episode setup return consistent results
        self._cached_effective_stage: Optional[CurriculumStage] = None
        self._cached_effective_stage_episode: int = -1
        
        # Current entropy (updated externally if available)
        self._current_entropy: float = -1.0
        
        # Initialize stage
        self._enter_stage(self.current_stage, reason="init")
        
        logger.info(f"CurriculumManager v{STATE_VERSION} initialized at stage: {initial_stage.name}")
    
    # -------------------------------------------------------------------------
    # Stage Entry/Exit
    # -------------------------------------------------------------------------
    
    def _enter_stage(
        self, 
        stage: CurriculumStage, 
        reason: str,
        demoted_from_stage: Optional[CurriculumStage] = None,
    ) -> None:
        """
        Enter a new stage, setting up all transition effects.
        
        Args:
            stage: The stage to enter
            reason: Why we're entering ("init", "promotion", "demotion", etc.)
            demoted_from_stage: If reason="demotion", the stage we failed at (for recovery)
        """
        # Save previous config for reward blending
        if hasattr(self, 'current_stage') and self.current_stage != stage:
            self._previous_stage_config = get_stage_config(self.current_stage)
            self._previous_stage_name = self.current_stage.name
        
        self._stage_epoch_counter[stage] += 1
        self._current_stage_epoch = self._stage_epoch_counter[stage]
        self.stage_timesteps = 0
        self.stage_episodes = 0
        self._rolling_stats_dirty = True
        
        # Get new stage config
        new_config = get_stage_config(stage)
        transition = new_config.transition
        
        # Setup transition cooldown
        self._transition_cooldown_remaining = transition.transition_cooldown_episodes
        
        # Setup reward blending
        if transition.reward_blend_enabled and self._previous_stage_config is not None:
            self._reward_blend_remaining = transition.reward_blend_episodes
        else:
            self._reward_blend_remaining = 0
        
        # Setup LR warmup
        if transition.lr_warmup_enabled and reason != "init":
            self._lr_warmup_active = True
            self._lr_warmup_steps_remaining = transition.lr_warmup_steps
            self._lr_warmup_factor = transition.lr_warmup_factor
        else:
            self._lr_warmup_active = False
            self._lr_warmup_steps_remaining = 0
            self._lr_warmup_factor = 1.0
        
        # Reset learning velocity for new stage and wire stage config thresholds
        self._learning_velocity = LearningVelocity()
        at = new_config.adaptive_thresholds
        self._learning_velocity.plateau_threshold = float(at.plateau_improvement_threshold)
        
        # Check if recovery protocol should be triggered (for demotions)
        if reason == "demotion" and demoted_from_stage is not None:
            recovery_config = new_config.recovery_protocol
            # Use the stage we demoted FROM to check failure count and diagnose
            # This ensures we track repeated failures at the higher stage correctly
            self._recovery_state = self._demotion_analyzer.create_recovery_protocol(
                demoted_from_stage, recovery_config
            )
            if self._recovery_state.is_active() and self.verbose:
                logger.info(
                    f"  Recovery protocol activated: {self._recovery_state.trigger_reason}, "
                    f"{self._recovery_state.episodes_remaining} episodes"
                )
        else:
            # Clear recovery on promotion
            self._recovery_state = RecoveryProtocolState()
        
        if self.verbose:
            logger.info(f"Entered stage {stage.name} (epoch={self._current_stage_epoch}, reason={reason})")
            if self._lr_warmup_active:
                logger.info(f"  LR warmup: {transition.lr_warmup_factor:.0%} → 100% over {transition.lr_warmup_steps:,} steps")
            if self._reward_blend_remaining > 0:
                logger.info(f"  Reward blend: {self._reward_blend_remaining} episodes")
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def stage_config(self) -> CurriculumStageConfig:
        return get_stage_config(self.current_stage)
    
    @property
    def competence_thresholds(self) -> CompetenceThresholds:
        """Get thresholds, possibly adjusted for plateau (with proximity check)."""
        base = self.stage_config.competence
        config = self.stage_config.adaptive_thresholds
        promotion_threshold = self.stage_config.composite_scoring.promotion_threshold
        return compute_adjusted_thresholds(
            base, 
            self._learning_velocity, 
            config,
            composite_score=self._composite_score,
            promotion_threshold=promotion_threshold,
        )
    
    @property
    def current_stage_epoch(self) -> int:
        return self._current_stage_epoch
    
    @property
    def is_in_transition(self) -> bool:
        return self._lr_warmup_active or self._reward_blend_remaining > 0
    
    def is_learning_plateaued(self) -> bool:
        """Check if learning has plateaued using stage-configured threshold."""
        threshold = self.stage_config.adaptive_thresholds.plateau_episodes_threshold
        return self._learning_velocity.is_plateaued(threshold)
    
    @property
    def reward_blend_factor(self) -> float:
        """Get blend factor: 1.0 = fully new config, 0.0 = fully old config."""
        if self._reward_blend_remaining <= 0 or self._previous_stage_config is None:
            return 1.0
        
        transition = self.stage_config.transition
        total_blend = transition.reward_blend_episodes
        if total_blend <= 0:
            return 1.0
        
        progress = 1.0 - (self._reward_blend_remaining / total_blend)
        return _clamp(progress, 0.0, 1.0)
    
    @property
    def learning_velocity(self) -> LearningVelocity:
        return self._learning_velocity
    
    @property
    def skill_assessment(self) -> Optional[SkillAssessment]:
        return self._skill_assessment
    
    @property
    def recovery_state(self) -> RecoveryProtocolState:
        return self._recovery_state
    
    @property
    def review_state(self) -> ReviewSessionState:
        return self._review_state
    
    @property
    def composite_score(self) -> Optional[CompositeScore]:
        return self._composite_score
    
    # -------------------------------------------------------------------------
    # Transition State
    # -------------------------------------------------------------------------
    
    def get_lr_multiplier(self, base_steps_since_transition: int = 0) -> float:
        """Get learning rate multiplier for warmup."""
        if not self._lr_warmup_active:
            return 1.0
        
        transition = self.stage_config.transition
        if transition.lr_warmup_steps <= 0:
            return 1.0
        
        warmup_progress = 1.0 - (self._lr_warmup_steps_remaining / transition.lr_warmup_steps)
        warmup_progress = _clamp(warmup_progress, 0.0, 1.0)
        
        return self._lr_warmup_factor + (1.0 - self._lr_warmup_factor) * warmup_progress
    
    def step_transition_state(self, timesteps: int = 1) -> None:
        """Update transition state counters (call each training step)."""
        if self._lr_warmup_active and self._lr_warmup_steps_remaining > 0:
            self._lr_warmup_steps_remaining -= timesteps
            if self._lr_warmup_steps_remaining <= 0:
                self._lr_warmup_active = False
                self._lr_warmup_factor = 1.0
                if self.verbose:
                    logger.info(f"LR warmup complete for stage {self.current_stage.name}")
    
    def _seen_episode_id(self, eid: int) -> bool:
        """Check if an episode ID has already been processed."""
        return eid in self._processed_episode_id_set
    
    def _mark_episode_id(self, eid: int) -> None:
        """Mark an episode ID as processed (bounded to prevent memory leak)."""
        # Manual bounded eviction so the set stays consistent without O(n) rebuilds.
        if len(self._processed_episode_ids) >= PROCESSED_EPISODE_ID_LIMIT:
            old = self._processed_episode_ids.popleft()
            self._processed_episode_id_set.discard(old)
        self._processed_episode_ids.append(eid)
        self._processed_episode_id_set.add(eid)
    
    def episode_transition_tick(self) -> None:
        """Update per-episode transition counters."""
        if self._transition_cooldown_remaining > 0:
            self._transition_cooldown_remaining -= 1
        
        if self._reward_blend_remaining > 0:
            self._reward_blend_remaining -= 1
            if self._reward_blend_remaining <= 0 and self.verbose:
                logger.info(f"Reward blending complete for stage {self.current_stage.name}")
        
        # Tick recovery protocol
        if self._recovery_state.is_active():
            self._recovery_state.tick()
            if not self._recovery_state.is_active() and self.verbose:
                logger.info(f"Recovery protocol complete for stage {self.current_stage.name}")
    
    # -------------------------------------------------------------------------
    # Deterministic Episode End (Phase 1.2)
    # -------------------------------------------------------------------------
    
    def on_episode_end(
        self,
        metrics: EpisodeMetrics,
        timesteps: int = 0,
        effective_stage: Optional[CurriculumStage] = None,
        check_transitions: bool = True,
    ) -> Dict[str, Any]:
        """
        Single entry point for episode completion. Ensures deterministic handling.
        
        This method:
        1. Records the episode (with invariant checking)
        2. Advances all counters exactly once
        3. Checks for transitions if enabled
        4. Returns the resulting state
        
        MUST be called exactly once per episode to maintain counter integrity.
        Calling multiple times for the same episode is a no-op (idempotent).
        
        Args:
            metrics: Episode metrics
            timesteps: Timesteps in episode
            effective_stage: Stage actually trained on (for mixed-stage)
            check_transitions: Whether to check for promotion/demotion
            
        Returns:
            Dict with episode results and any transition information
        """
        result: Dict[str, Any] = {
            "episode_idx": self.total_episodes + 1,
            "stage": self.current_stage.name,
            "stage_epoch": self._current_stage_epoch,
            "promoted": False,
            "demoted": False,
            "transition_to": None,
            "invariant_violations": [],
        }
        
        # True idempotency check using episode ID (not just sequential counter)
        # Use global_episode_idx from metrics if available, otherwise fall back to sequential
        gid = int(getattr(metrics, "global_episode_idx", 0) or 0)
        # If gid is missing/stale (common after resume or per-env counters), fall back to manager sequence.
        # This avoids permanent "duplicate" skipping after resume.
        if gid <= 0 or gid <= self.total_episodes:
            episode_id = self.total_episodes + 1
        else:
            episode_id = gid
        
        if self._seen_episode_id(episode_id):
            logger.debug(f"on_episode_end: duplicate episode_id={episode_id}, skipping")
            result["skipped"] = True
            result["skip_reason"] = f"duplicate_episode_id:{episode_id}"
            return result
        
        # Fallback idempotency check (sequential counter)
        next_idx = self.total_episodes + 1
        if next_idx <= self._last_episode_end_idx:
            logger.warning(
                f"on_episode_end called for episode {next_idx} but already processed "
                f"up to {self._last_episode_end_idx}. Skipping to maintain counter integrity."
            )
            result["skipped"] = True
            return result
        
        # Run invariant checker on incoming metrics
        violations = self._invariant_checker.check_episode(
            metrics=asdict(metrics),
            stage_name=self.current_stage.name,
            stage_epoch=self._current_stage_epoch,
            episode_idx=next_idx,
        )
        result["invariant_violations"] = [str(v) for v in violations]
        
        # Record episode (this handles counters and history)
        self.record_episode(metrics, timesteps=timesteps, effective_stage=effective_stage)
        
        # Mark as processed (both systems)
        self._last_episode_end_idx = self.total_episodes
        self._mark_episode_id(episode_id)
        
        # Check transitions if enabled and not in cooldown
        if check_transitions and self._transition_cooldown_remaining <= 0:
            # Check promotion first
            if self.auto_promote:
                promoted, new_stage = self.try_promote()
                result["promoted"] = promoted
                if promoted:
                    result["transition_to"] = new_stage.name if new_stage else None
            
            # Check demotion if not promoted
            if not result["promoted"] and self.auto_demote:
                demoted, new_stage = self.try_demote()
                result["demoted"] = demoted
                if demoted:
                    result["transition_to"] = new_stage.name if new_stage else None
        
        result["stage_after"] = self.current_stage.name
        result["cooldown_remaining"] = self._transition_cooldown_remaining
        
        return result
    
    # -------------------------------------------------------------------------
    # Entropy Management
    # -------------------------------------------------------------------------
    
    def update_entropy(self, entropy: float) -> None:
        """Update current policy entropy (call from training loop)."""
        self._current_entropy = entropy
    
    def get_entropy_penalty(self) -> float:
        """Get entropy penalty based on current entropy and targets.
        
        NOTE: This is currently UNUSED. Entropy control is done via PPO's ent_coef
        in train_prop_firm.py (_apply_entropy_schedule). This method exists as a
        hook for reward-shaping based entropy control if needed in the future.
        
        If you enable this, DO NOT also use ent_coef control (double-control causes
        instability). Apply as: shaped_reward = env_reward - get_entropy_penalty()
        """
        if self._current_entropy < 0:
            return 0.0  # Entropy not available
        
        targets = self.stage_config.entropy_targets
        
        if self._current_entropy < targets.min_entropy:
            return (targets.min_entropy - self._current_entropy) * targets.low_entropy_penalty_scale
        elif self._current_entropy > targets.max_entropy:
            return (self._current_entropy - targets.max_entropy) * targets.high_entropy_penalty_scale
        
        return 0.0
    
    # -------------------------------------------------------------------------
    # Mixed-Stage Sampling
    # -------------------------------------------------------------------------
    
    def sample_training_stage(self) -> CurriculumStage:
        """
        Sample a stage for the next episode.
        
        Used for mixed-stage training to prevent catastrophic forgetting.
        Uses the configured weights: current_stage_weight, recent_stages_weight,
        and foundation_weight to determine sampling probabilities.
        """
        config = self.stage_config.mixed_stage_sampling
        
        if not config.enabled:
            return self.current_stage
        
        r = self._rng.random()
        
        # Normalize weights to sum to 1.0 (in case they don't)
        total_weight = (
            config.current_stage_weight +
            config.recent_stages_weight +
            config.foundation_weight
        )
        if total_weight <= 0:
            return self.current_stage
        
        current_prob = config.current_stage_weight / total_weight
        recent_prob = config.recent_stages_weight / total_weight
        # foundation_prob = config.foundation_weight / total_weight  # remainder
        
        if r < current_prob:
            return self.current_stage
        elif r < current_prob + recent_prob:
            # Sample from recent stages using progression index (not .value)
            cur_idx = _stage_to_index(self.current_stage)
            min_idx = max(0, cur_idx - config.recent_stage_depth)
            if min_idx >= cur_idx:
                return self.current_stage
            sampled_idx = int(self._rng.integers(min_idx, cur_idx))
            return _index_to_stage(sampled_idx)
        else:
            # Foundation stage (using foundation_weight) - use _foundation_stage()
            return _foundation_stage()
    
    # -------------------------------------------------------------------------
    # Review Sessions
    # -------------------------------------------------------------------------
    
    def check_review_session(self) -> Optional[CurriculumStage]:
        """
        Check if a review session should be triggered.
        
        Returns the stage to review, or None if no review needed.
        
        NOTE: This method uses a per-episode latch to prevent double-ticking
        of review counters if called multiple times within the same episode.
        """
        config = self.stage_config.review_session
        
        if not config.enabled:
            return None
        
        # min_stage_for_review is guaranteed non-None by ReviewSessionConfig.__post_init__
        if config.min_stage_for_review is None:
            return None  # Should never happen, but satisfies type checker
        
        # Use progression index for comparison (not .value)
        cur_idx = _stage_to_index(self.current_stage)
        min_review_idx = _stage_to_index(config.min_stage_for_review)
        if cur_idx < min_review_idx:
            return None
        
        # Prevent double-ticking within the same episode
        next_ep = self.total_episodes + 1
        already_ticked = (self._review_tick_episode == next_ep)
        if not already_ticked:
            self._review_tick_episode = next_ep
        
        # Check return-to-home latch (one episode after review ends)
        if getattr(self._review_state, 'return_to_home_latch', False):
            self._review_state.return_to_home_latch = False
            # Return current_stage to ensure predictable post-review behavior
            return self.current_stage
        
        # Handle active review
        if self._review_state.in_review:
            # Only decrement if we haven't already ticked this episode
            if not already_ticked:
                self._review_state.review_episodes_remaining -= 1
            if self._review_state.review_episodes_remaining <= 0:
                # End review AFTER running the final intended review episode.
                # Previously this returned home immediately, effectively shortening review by 1.
                final_review_stage = self._review_state.review_stage

                # End review, set latch to return home for next episode
                self._review_state.in_review = False
                self._review_state.episodes_since_review = 0
                self._review_state.return_to_home_latch = True  # Will return home next call
                if self.verbose:
                    home_name = self._review_state.home_stage.name if self._review_state.home_stage else "current"
                    logger.info(f"Review session complete, returning to {home_name} next episode")
                return final_review_stage  # Run the last review episode on the review stage
            return self._review_state.review_stage
        
        # Check if review should start (only tick counter if not already ticked)
        if not already_ticked:
            self._review_state.episodes_since_review += 1
        
        if self._review_state.episodes_since_review >= config.review_frequency:
            # Start review using progression indices (not .value)
            cur_idx = _stage_to_index(self.current_stage)
            min_review_idx = max(0, cur_idx - config.review_depth)
            if min_review_idx < cur_idx:
                review_stage_idx = int(self._rng.integers(min_review_idx, cur_idx))
                review_stage = _index_to_stage(review_stage_idx)
                
                self._review_state.in_review = True
                self._review_state.review_stage = review_stage
                self._review_state.review_episodes_remaining = config.review_duration
                self._review_state.home_stage = self.current_stage
                
                if self.verbose:
                    logger.info(f"Starting review session on {review_stage.name} ({config.review_duration} episodes)")
                
                return review_stage
        
        return None
    
    def get_effective_stage(self) -> CurriculumStage:
        """
        Get the effective stage for the current episode.
        
        Considers review sessions and mixed-stage sampling.
        
        IMPORTANT: Results are cached per episode to ensure determinism.
        Multiple calls within the same episode setup will return the same stage.
        Cache is cleared after episode is recorded.
        """
        # Check cache first - prevents non-determinism from multiple calls
        next_ep = self.total_episodes + 1
        if (self._cached_effective_stage_episode == next_ep 
            and self._cached_effective_stage is not None):
            return self._cached_effective_stage
        
        # Compute effective stage (exactly once per episode)
        stage: CurriculumStage
        
        # One-episode latch: after review ends, return to home/current stage once
        # This must be checked BEFORE check_review_session to avoid immediate mixed-stage sampling
        if getattr(self._review_state, 'return_to_home_latch', False):
            self._review_state.return_to_home_latch = False
            stage = self._review_state.home_stage or self.current_stage
        else:
            # Check review session first
            review_stage = self.check_review_session()
            if review_stage is not None:
                stage = review_stage
            else:
                # Then mixed-stage sampling
                stage = self.sample_training_stage()
        
        # Cache result
        self._cached_effective_stage = stage
        self._cached_effective_stage_episode = next_ep
        
        return stage
    
    def _clear_effective_stage_cache(self) -> None:
        """Clear the effective stage cache after episode is recorded."""
        self._cached_effective_stage = None
        self._cached_effective_stage_episode = -1
    
    # -------------------------------------------------------------------------
    # Episode Recording
    # -------------------------------------------------------------------------
    
    def record_episode(
        self, 
        metrics: EpisodeMetrics, 
        timesteps: int = 0,
        effective_stage: Optional[CurriculumStage] = None,
    ) -> None:
        """
        Record an episode's metrics.
        
        Args:
            metrics: Episode metrics to record
            timesteps: Number of timesteps in the episode
            effective_stage: If mixed-stage sampling was used, the stage that was 
                           actually trained on (may differ from current_stage).
                           If None, uses current_stage.
        """
        # Determine which stage to record to
        # If effective_stage provided (from mixed-stage sampling), use it for history
        # but still track progression against current_stage
        record_to_stage = effective_stage if effective_stage is not None else self.current_stage
        
        # Fill curriculum metadata
        # FIX: When effective_stage is provided, always use record_to_stage.name
        # (don't keep the pre-set stage_name from record_episode_from_info)
        if effective_stage is not None:
            metrics.stage_name = record_to_stage.name
        else:
            metrics.stage_name = metrics.stage_name or record_to_stage.name
        
        # FIX: Use the epoch counter for the stage we're recording to, not current_stage
        if record_to_stage == self.current_stage:
            metrics.stage_epoch = metrics.stage_epoch or self._current_stage_epoch
        else:
            metrics.stage_epoch = self._stage_epoch_counter.get(record_to_stage, 0)
        
        metrics.global_episode_idx = metrics.global_episode_idx or (self.total_episodes + 1)
        metrics.policy_entropy = self._current_entropy
        if not metrics.timestamp:
            metrics.timestamp = _now_iso(self.tz)
        
        # Record to the effective stage's history
        self._history[record_to_stage].append(metrics)
        
        # Track total stats for the effective stage
        self._stage_timesteps_total[record_to_stage] += timesteps
        self._stage_episodes_total[record_to_stage] += 1
        
        # Only track current stage progress if we're on the current stage
        # (mixed-stage episodes don't count toward promotion)
        if record_to_stage == self.current_stage:
            self.stage_timesteps += timesteps
            self.stage_episodes += 1
        
        self.total_timesteps += timesteps
        self.total_episodes += 1
        self._rolling_stats_dirty = True
        
        # Clear effective stage cache now that episode is recorded
        self._clear_effective_stage_cache()
        
        # Update learning velocity (only for current stage episodes)
        if record_to_stage == self.current_stage:
            self._learning_velocity.update({
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "avg_pnl": metrics.total_pnl,
                "max_drawdown": metrics.max_drawdown,  # FIX: Key must match LOWER_IS_BETTER
            })
        
        # Transition tick
        self.episode_transition_tick()
    
    def record_episode_from_info(
        self,
        info: Dict[str, Any],
        episode_reward: float,
        episode_length: int,
        effective_stage: Optional[CurriculumStage] = None,
    ) -> None:
        """
        Record episode from environment info dict.
        
        Args:
            info: Environment info dictionary
            episode_reward: Total episode reward
            episode_length: Episode length in timesteps
            effective_stage: Stage that was actually trained on (for mixed-stage sampling)
        """
        ep_stats = info.get("episode_stats", {}) or {}
        
        # Parse metrics - using canonical naming (trade_count, max_drawdown)
        total_pnl = _safe_float(info.get("total_pnl", ep_stats.get("total_pnl", 0.0)), 0.0)
        win_rate = _clamp(_safe_float(info.get("win_rate", ep_stats.get("win_rate", 0.0)), 0.0), 0.0, 1.0)
        
        # trade_count: from step info or episode_stats (both now use same name)
        trade_count = _safe_int(info.get("trade_count", ep_stats.get("trade_count", 0)), 0)
        winning_trades = _safe_int(ep_stats.get("winning_trades", info.get("winning_trades", 0)), 0)
        losing_trades = _safe_int(ep_stats.get("losing_trades", info.get("losing_trades", 0)), 0)
        
        # TRADE ACCOUNTING RECONCILIATION (Issue #4 fix)
        # Ensure wins + losses <= trade_count and log if inconsistent
        computed_sum = winning_trades + losing_trades
        if computed_sum > trade_count and trade_count > 0:
            # Warn and reconcile: use wins+losses as the source of truth
            logger.warning(
                f"Trade accounting mismatch: wins({winning_trades})+losses({losing_trades})={computed_sum} > "
                f"trade_count({trade_count}). Using wins+losses as denominator."
            )
            trade_count = computed_sum
        elif (winning_trades == 0 and losing_trades == 0) and trade_count > 0:
            # Estimate from win_rate if we have trades but no win/loss breakdown
            approx_wins = int(round(win_rate * trade_count))
            winning_trades = max(0, min(trade_count, approx_wins))
            losing_trades = max(0, trade_count - winning_trades)
        
        # max_drawdown: prefer episode_stats (peak DD) over step info current drawdown
        max_dd = _clamp(_safe_float(info.get("max_drawdown", ep_stats.get("max_drawdown", info.get("drawdown", 0.0))), 0.0), 0.0, 1.0)
        daily_dd = _clamp(_safe_float(info.get("daily_drawdown", ep_stats.get("daily_drawdown", 0.0)), 0.0), 0.0, 1.0)
        
        termination_reason = str(info.get("termination_reason", ep_stats.get("termination_reason", "")) or "")
        
        dd_breach = bool(info.get("dd_breach", ep_stats.get("dd_breach", False)))
        if not dd_breach and termination_reason:
            low = termination_reason.lower()
            dd_breach = ("drawdown" in low) or ("dd breach" in low) or ("daily_dd" in low) or ("max_dd" in low)
        
        pf_raw = ep_stats.get("profit_factor", info.get("profit_factor", 0.0))
        pf = _safe_float(pf_raw, 0.0)
        if math.isinf(pf) or pf > 1e6:
            pf = 10.0
        pf = _clamp(pf, 0.0, 10.0)
        
        hit_max_consec = bool(info.get("hit_max_consecutive_losses", ep_stats.get("hit_max_consecutive_losses", False)))
        if not hit_max_consec and termination_reason:
            hit_max_consec = "consecutive" in termination_reason.lower()
        
        # Parse exit quality distribution
        # CloseReason enum values from prop_firm_env.py:
        #   GOOD: trailing_stop, agent_close
        #   BAD: hard_stop, emergency_close, risk_liquidation
        #   NEUTRAL: time_decay, hard_close, weekend_flatten, daily_limit_safety, episode_truncate_flatten
        exit_dist = ep_stats.get("exit_quality_distribution", {}) or {}
        
        # Define known exit types for robust parsing
        GOOD_EXITS = {"trailing_stop", "agent_close"}
        BAD_EXITS = {"hard_stop", "emergency_close", "risk_liquidation"}
        NEUTRAL_EXITS = {"time_decay", "hard_close", "weekend_flatten", 
                        "daily_limit_safety", "episode_truncate_flatten", "episode_truncate"}
        
        trailing_stops = _safe_int(exit_dist.get("trailing_stop", 0), 0)
        agent_closes = _safe_int(exit_dist.get("agent_close", 0), 0)
        
        # Hard stop includes emergency close (both are risk management failures)
        hard_stops = _safe_int(exit_dist.get("hard_stop", 0), 0) + _safe_int(exit_dist.get("emergency_close", 0), 0)
        
        risk_liquidations = _safe_int(exit_dist.get("risk_liquidation", 0), 0)
        
        # Neutral exits (not bad, but not demonstrating exit skill)
        neutral_exits = sum(
            _safe_int(exit_dist.get(key, 0), 0) 
            for key in NEUTRAL_EXITS
        )
        
        # Handle any unrecognized exit types (future-proofing)
        known_keys = GOOD_EXITS | BAD_EXITS | NEUTRAL_EXITS
        unknown_exits = sum(
            _safe_int(count, 0) 
            for key, count in exit_dist.items() 
            if key not in known_keys
        )
        if unknown_exits > 0:
            logger.debug(f"Found {unknown_exits} exits with unrecognized types, treating as neutral")
        
        tracked_exits = trailing_stops + agent_closes + hard_stops + risk_liquidations + neutral_exits + unknown_exits
        raw_other_exits = trade_count - tracked_exits
        
        # Safety: if raw_other_exits is negative, it means exit_dist double-counted something
        # In this case, trust exit_dist totals
        if raw_other_exits < 0:
            logger.debug(f"Exit count mismatch: trade_count={trade_count}, tracked={tracked_exits}")
            raw_other_exits = 0
        
        other_exits = raw_other_exits + neutral_exits + unknown_exits  # Include neutral in 'other' for skill assessment
        
        metrics = EpisodeMetrics(
            total_pnl=total_pnl,
            win_rate=win_rate,
            trade_count=trade_count,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown=max_dd,
            daily_drawdown=daily_dd,
            dd_breach=dd_breach,
            avg_r_multiple=_safe_float(ep_stats.get("avg_r_multiple", info.get("avg_r_multiple", 0.0)), 0.0),
            profit_factor=pf,
            avg_mae=_safe_float(ep_stats.get("avg_mae", info.get("avg_mae", 0.0)), 0.0),
            avg_mfe=_safe_float(ep_stats.get("avg_mfe", info.get("avg_mfe", 0.0)), 0.0),
            avg_bars_held=_safe_float(ep_stats.get("avg_bars_held", info.get("avg_bars_held", 0.0)), 0.0),
            avg_entry_quality=_clamp(_safe_float(ep_stats.get("avg_entry_quality", info.get("avg_entry_quality", 0.5)), 0.5), 0.0, 1.0),
            consecutive_losses=_safe_int(info.get("consecutive_losses", ep_stats.get("consecutive_losses", 0)), 0),
            consecutive_wins=_safe_int(info.get("consecutive_wins", ep_stats.get("consecutive_wins", 0)), 0),
            max_consecutive_losses_reached=_safe_int(ep_stats.get("max_consecutive_losses_reached", 
                                                                   info.get("consecutive_losses", ep_stats.get("consecutive_losses", 0))), 0),
            hit_max_consecutive_losses=hit_max_consec,
            trailing_stop_exits=trailing_stops,
            agent_close_exits=agent_closes,
            hard_stop_exits=hard_stops,
            risk_liquidation_exits=risk_liquidations,
            other_exits=other_exits,  # Already clamped to >= 0 above
            episode_length=_safe_int(episode_length, 0),
            episode_reward=_safe_float(episode_reward, 0.0),
            termination_reason=termination_reason,
            stage_name=self.current_stage.name,
            stage_epoch=self._current_stage_epoch,
            global_episode_idx=self.total_episodes + 1,
            policy_entropy=self._current_entropy,
            timestamp=_now_iso(self.tz),
        )
        
        # Extract regime-tagged trades for skill assessment (Phase 2.2)
        self._accumulate_regime_trades(ep_stats.get("trades_with_regime", []))
        
        # Use on_episode_end to record AND check transitions (promotion/demotion)
        # Previously this called record_episode directly, which skipped transition checks!
        self.on_episode_end(
            metrics=metrics,
            timesteps=episode_length,
            effective_stage=effective_stage,
            check_transitions=True,  # Enable automatic promotion/demotion
        )
    
    def _accumulate_regime_trades(self, trades_data: List[Dict[str, Any]]) -> None:
        """
        Accumulate regime-tagged trades for RegimeSkillAssessment.
        
        Args:
            trades_data: List of trade dicts with regime info from prop_firm_env
        """
        if not trades_data:
            return
        
        # Map string regime names to enums
        from envs.curriculum.regime_skill_assessment import (
            VolatilityRegime, TrendRegime, SessionRegime, SpreadRegime
        )
        
        vol_map = {
            "low": VolatilityRegime.LOW,
            "medium": VolatilityRegime.MEDIUM,
            "high": VolatilityRegime.HIGH,
        }
        trend_map = {
            "strong_trend": TrendRegime.STRONG_TREND,
            "weak_trend": TrendRegime.WEAK_TREND,
            "ranging": TrendRegime.RANGING,
        }
        session_map = {
            "asian": SessionRegime.ASIAN,
            "london": SessionRegime.LONDON,
            "overlap": SessionRegime.LONDON_NY_OVERLAP,
            "ny": SessionRegime.NY,
            "off_hours": SessionRegime.OFF_HOURS,
        }
        spread_map = {
            "tight": SpreadRegime.TIGHT,
            "normal": SpreadRegime.NORMAL,
            "wide": SpreadRegime.WIDE,
        }
        
        for td in trades_data:
            try:
                trade = TradeWithRegime(
                    pnl=float(td.get("pnl", 0.0)),
                    r_multiple=float(td.get("r_multiple", 0.0)),
                    is_winner=bool(td.get("is_winner", False)),
                    bars_held=int(td.get("bars_held", 0)),
                    mae=float(td.get("mae", 0.0)),
                    mfe=float(td.get("mfe", 0.0)),
                    entry_quality=float(td.get("entry_quality", 0.5)),
                    exit_type=str(td.get("exit_type", "")),
                    volatility_regime=vol_map.get(td.get("volatility_regime", "medium"), VolatilityRegime.MEDIUM),
                    trend_regime=trend_map.get(td.get("trend_regime", "ranging"), TrendRegime.RANGING),
                    session_regime=session_map.get(td.get("session_regime", "off_hours"), SessionRegime.OFF_HOURS),
                    spread_regime=spread_map.get(td.get("spread_regime", "normal"), SpreadRegime.NORMAL),
                )
                self._regime_trades.append(trade)
                self._regime_assessment_dirty = True
            except Exception as e:
                logger.debug(f"Failed to parse regime trade: {e}")
    
    def _update_regime_assessment(self) -> None:
        """Update RegimeSkillAssessment from accumulated trades."""
        if not self._regime_assessment_dirty:
            return
        
        if len(self._regime_trades) >= 50:  # Need minimum sample for meaningful assessment
            self._regime_assessment = RegimeSkillAssessment.from_trades(list(self._regime_trades))
            self._regime_assessment_dirty = False
        else:
            self._regime_assessment = None
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def _current_epoch_window(self, window_size: int) -> List[EpisodeMetrics]:
        """Get episodes from current stage epoch only."""
        history = list(self._history[self.current_stage])
        epoch = self._current_stage_epoch
        filtered = [m for m in history if m.stage_epoch == epoch]
        return filtered[-window_size:] if len(filtered) > window_size else filtered
    
    def get_rolling_stats(self, force_refresh: bool = False) -> RollingStats:
        """Compute rolling statistics over evaluation window."""
        if not force_refresh and not self._rolling_stats_dirty and self._rolling_stats is not None:
            return self._rolling_stats
        
        thresholds = self.stage_config.competence
        window_size = thresholds.evaluation_window
        
        window = self._current_epoch_window(window_size)
        
        if not window:
            self._rolling_stats = RollingStats(window_size=0)
            self._rolling_stats_dirty = False
            return self._rolling_stats
        
        # Extract arrays
        pnls = np.array([_safe_float(m.total_pnl, 0.0) for m in window], dtype=np.float64)
        win_rates = np.array([_clamp(_safe_float(m.win_rate, 0.0), 0.0, 1.0) for m in window], dtype=np.float64)
        trade_counts = np.array([max(0.0, float(m.trade_count)) for m in window], dtype=np.float64)
        drawdowns = np.array([_clamp(_safe_float(m.max_drawdown, 0.0), 0.0, 1.0) for m in window], dtype=np.float64)
        dd_breaches = np.array([bool(m.dd_breach) for m in window], dtype=np.bool_)
        profit_factors = np.array([_clamp(_safe_float(m.profit_factor, 0.0), 0.0, 10.0) for m in window], dtype=np.float64)
        r_multiples = np.array([_safe_float(m.avg_r_multiple, 0.0) for m in window], dtype=np.float64)
        entry_qualities = np.array([_clamp(_safe_float(m.avg_entry_quality, 0.5), 0.0, 1.0) for m in window], dtype=np.float64)
        consec_loss_breaches = np.array([bool(m.hit_max_consecutive_losses) for m in window], dtype=np.bool_)
        # Track peak consecutive losses per episode (more informative than just breach rate)
        max_consec_losses = np.array([max(0, m.max_consecutive_losses_reached) for m in window], dtype=np.int32)
        entropies = np.array([m.policy_entropy for m in window if m.policy_entropy >= 0], dtype=np.float64)
        
        # Basic stats (use ddof=1 for sample std - proper for CI calculation)
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        mean_win_rate = float(np.mean(win_rates))
        std_win_rate = float(np.std(win_rates, ddof=1)) if len(win_rates) > 1 else 0.0
        mean_trade_count = float(np.mean(trade_counts))
        
        total_trades = int(np.sum(trade_counts))
        total_wins = int(sum(max(0, m.winning_trades) for m in window))
        total_losses = int(sum(max(0, m.losing_trades) for m in window))
        
        mean_drawdown = float(np.mean(drawdowns))
        max_drawdown_seen = float(np.max(drawdowns)) if len(drawdowns) else 0.0
        dd_breach_rate = float(np.mean(dd_breaches)) if len(dd_breaches) else 0.0
        
        # Profit factor: Handle zero-trade episodes properly.
        # Previously filtered out zeros which inflates PF unfairly.
        # Now: Include zeros as 0.0 (penalizes idle episodes) OR require min trades.
        # Using floor of 0.0 for episodes without trades (they contribute nothing).
        mean_profit_factor = float(np.mean(profit_factors)) if len(profit_factors) else 0.0
        
        mean_r_multiple = float(np.mean(r_multiples)) if len(r_multiples) else 0.0
        mean_entry_quality = float(np.mean(entry_qualities)) if len(entry_qualities) else 0.5
        consecutive_loss_breach_rate = float(np.mean(consec_loss_breaches)) if len(consec_loss_breaches) else 0.0
        # Average of peak consecutive losses per episode
        avg_max_consecutive_losses = float(np.mean(max_consec_losses)) if len(max_consec_losses) else 0.0
        # Rate of episodes with 3+ consecutive losses (catches problematic streaks earlier)
        consecutive_loss_streak_rate = float(np.mean(max_consec_losses >= 3)) if len(max_consec_losses) else 0.0
        
        # Sharpe/Sortino
        if std_pnl > 1e-9:
            sharpe = mean_pnl / std_pnl
        else:
            sharpe = mean_pnl if mean_pnl > 0 else 0.0
        
        negative_pnls = pnls[pnls < 0]
        if negative_pnls.size:
            downside_std = float(np.std(negative_pnls))
            sortino = mean_pnl / downside_std if downside_std > 1e-9 else sharpe
        else:
            sortino = sharpe * 1.5 if mean_pnl > 0 else 0.0
        
        win_loss_ratio = total_wins / max(total_losses, 1)
        
        # Wilson confidence intervals for trade-level win rate
        # Note: This uses pooled trades across episodes for statistical power.
        # FIX: Use max(total_trades, total_wins + total_losses) to account for breakevens
        # (total_trades may include breakevens not counted in wins+losses)
        wl_n = max(total_trades, total_wins + total_losses, 0)
        win_low, win_high = _wilson_interval(total_wins, wl_n, z=1.96) if wl_n > 0 else (0.0, 0.0)
        
        # Mean PnL CI
        ci_low, ci_high = _mean_ci_normal(mean_pnl, std_pnl, n=len(window), z=1.96)
        
        # Entropy stats - track sample count for gating
        entropy_samples = len(entropies)
        
        # Entropy stats
        mean_entropy = float(np.mean(entropies)) if len(entropies) > 0 else -1.0
        std_entropy = float(np.std(entropies)) if len(entropies) > 1 else 0.0
        
        # Exit quality rates
        total_exits = sum(
            m.trailing_stop_exits + m.agent_close_exits + m.hard_stop_exits + 
            m.risk_liquidation_exits + m.other_exits
            for m in window
        )
        if total_exits > 0:
            trailing_stop_rate = sum(m.trailing_stop_exits for m in window) / total_exits
            agent_close_rate = sum(m.agent_close_exits for m in window) / total_exits
            hard_stop_rate = sum(m.hard_stop_exits for m in window) / total_exits
            risk_liquidation_rate = sum(m.risk_liquidation_exits for m in window) / total_exits
        else:
            trailing_stop_rate = agent_close_rate = hard_stop_rate = risk_liquidation_rate = 0.0
        
        stats = RollingStats(
            window_size=len(window),
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            mean_win_rate=mean_win_rate,
            std_win_rate=std_win_rate,
            mean_trade_count=mean_trade_count,
            total_trades=total_trades,
            total_wins=total_wins,
            total_losses=total_losses,
            win_rate_trade_weighted=float(total_wins / max(total_trades, 1)),  # Trade-pooled win rate
            mean_drawdown=mean_drawdown,
            max_drawdown_seen=max_drawdown_seen,
            dd_breach_rate=dd_breach_rate,
            mean_profit_factor=mean_profit_factor,
            mean_r_multiple=mean_r_multiple,
            mean_entry_quality=mean_entry_quality,
            consecutive_loss_breach_rate=consecutive_loss_breach_rate,
            avg_max_consecutive_losses=avg_max_consecutive_losses,
            consecutive_loss_streak_rate=consecutive_loss_streak_rate,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            win_loss_ratio=float(win_loss_ratio),
            win_rate_wilson_low=float(win_low),
            win_rate_wilson_high=float(win_high),
            pnl_mean_ci_low=float(ci_low),
            pnl_mean_ci_high=float(ci_high),
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            entropy_samples=entropy_samples,
            trailing_stop_rate=trailing_stop_rate,
            agent_close_rate=agent_close_rate,
            hard_stop_rate=hard_stop_rate,
            risk_liquidation_rate=risk_liquidation_rate,
        )
        
        self._rolling_stats = stats
        self._rolling_stats_dirty = False
        
        # Update skill assessment (pass configurable bars_per_trading_day)
        self._skill_assessment = SkillAssessment.from_episode_results(
            window, stats, bars_per_trading_day=self.bars_per_trading_day
        )
        
        # Update composite score
        self._composite_score = compute_composite_score(
            stats,
            self.stage_config.competence,
            self.stage_config.composite_scoring,
        )
        
        return stats
    
    # -------------------------------------------------------------------------
    # Promotion/Demotion Criteria
    # -------------------------------------------------------------------------
    
    def check_promotion_criteria(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if promotion criteria are met."""
        thresholds = self.competence_thresholds  # Uses adaptive thresholds
        base_thresholds = self.stage_config.competence
        stats = self.get_rolling_stats()
        
        results: Dict[str, Any] = {
            "stage": self.current_stage.name,
            "stage_epoch": self._current_stage_epoch,
            "evaluation_window": stats.window_size,
            "checks": {},
            "thresholds_relaxed": thresholds != base_thresholds,
        }
        
        all_passed = True
        
        # Minimum episodes/timesteps in current stage
        passed = self.stage_episodes >= thresholds.min_episodes
        results["checks"]["min_episodes"] = {
            "required": thresholds.min_episodes,
            "actual": self.stage_episodes,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        passed = self.stage_timesteps >= thresholds.min_timesteps
        results["checks"]["min_timesteps"] = {
            "required": thresholds.min_timesteps,
            "actual": self.stage_timesteps,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        # Data sufficiency
        min_required = max(MIN_EVALUATION_EPISODES, min(thresholds.evaluation_window, max(1, thresholds.min_episodes // 2)))
        if stats.window_size < min_required:
            results["checks"]["insufficient_data"] = {
                "required": min_required,
                "actual": stats.window_size,
                "passed": False,
            }
            results["promotion_ready"] = False
            results["stats"] = asdict(stats)
            return False, results
        
        # Performance checks
        passed = stats.mean_profit_factor >= thresholds.min_profit_factor
        results["checks"]["profit_factor"] = {
            "required": thresholds.min_profit_factor,
            "actual": stats.mean_profit_factor,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        passed = stats.mean_drawdown <= thresholds.max_avg_drawdown
        results["checks"]["avg_drawdown"] = {
            "required": thresholds.max_avg_drawdown,
            "actual": stats.mean_drawdown,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        passed = stats.mean_pnl >= thresholds.min_avg_pnl
        results["checks"]["avg_pnl"] = {
            "required": thresholds.min_avg_pnl,
            "actual": stats.mean_pnl,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        # R-Multiple check
        if thresholds.min_avg_r_multiple > 0:
            passed = stats.mean_r_multiple >= thresholds.min_avg_r_multiple
            results["checks"]["r_multiple"] = {
                "required": thresholds.min_avg_r_multiple,
                "actual": stats.mean_r_multiple,
                "passed": passed,
            }
            all_passed = all_passed and passed
        
        # Win rate checks
        passed_mean = stats.mean_win_rate >= thresholds.min_win_rate
        results["checks"]["win_rate_mean"] = {
            "required": thresholds.min_win_rate,
            "actual": stats.mean_win_rate,
            "passed": passed_mean,
        }
        
        # Wilson bound gate: use separate threshold if provided; otherwise default slightly lower
        wilson_required = getattr(thresholds, "min_win_rate_wilson_low", None)
        if wilson_required is None:
            wilson_required = max(0.0, float(thresholds.min_win_rate) - 0.05)
        
        # If we don't have enough pooled trades, skip Wilson gating to avoid false negatives
        min_trades_for_wilson = getattr(thresholds, "min_trades_for_wilson_gate", MIN_TRADES_FOR_WILSON_GATE_DEFAULT)
        pooled_trades_for_wilson = max(stats.total_wins + stats.total_losses, 0)
        if pooled_trades_for_wilson < int(min_trades_for_wilson):
            passed_wilson = True  # Skip - insufficient data for reliable Wilson bound
            wilson_note = f"Skipped Wilson gate (pooled_trades={pooled_trades_for_wilson} < {int(min_trades_for_wilson)})"
        else:
            passed_wilson = stats.win_rate_wilson_low >= float(wilson_required)
            wilson_note = "Trade-level 95% Wilson lower bound"
        
        results["checks"]["win_rate_wilson_low"] = {
            "required": float(wilson_required),
            "actual": stats.win_rate_wilson_low,
            "passed": passed_wilson,
            "note": wilson_note,
            "pooled_trades": pooled_trades_for_wilson,
        }
        all_passed = all_passed and passed_mean and passed_wilson
        
        # Consistency checks
        passed = stats.std_win_rate <= thresholds.max_win_rate_std
        results["checks"]["win_rate_stability"] = {
            "required": thresholds.max_win_rate_std,
            "actual": stats.std_win_rate,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        passed = stats.std_pnl <= thresholds.max_pnl_std
        results["checks"]["pnl_stability"] = {
            "required": thresholds.max_pnl_std,
            "actual": stats.std_pnl,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        passed = stats.mean_trade_count >= thresholds.min_trade_count_avg
        results["checks"]["trade_activity"] = {
            "required": thresholds.min_trade_count_avg,
            "actual": stats.mean_trade_count,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        # Behavior checks
        passed = stats.dd_breach_rate <= thresholds.max_dd_breach_rate
        results["checks"]["dd_breach_rate"] = {
            "required": thresholds.max_dd_breach_rate,
            "actual": stats.dd_breach_rate,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        passed = stats.consecutive_loss_breach_rate <= thresholds.max_consecutive_loss_rate
        results["checks"]["consecutive_loss_rate"] = {
            "required": thresholds.max_consecutive_loss_rate,
            "actual": stats.consecutive_loss_breach_rate,
            "passed": passed,
        }
        all_passed = all_passed and passed
        
        # Entropy check - use EntropyTargets.min_entropy for single source of truth
        # (same threshold used for penalties and promotion gating)
        # FIX: Only gate on entropy if we have sufficient samples (≥80% of window)
        entropy_targets = self.stage_config.entropy_targets
        min_entropy_samples = int(stats.window_size * 0.8)  # Require 80% coverage
        has_sufficient_entropy_samples = stats.entropy_samples >= min_entropy_samples
        
        if entropy_targets.use_in_promotion and entropy_targets.min_entropy > 0 and stats.mean_entropy >= 0:
            if has_sufficient_entropy_samples:
                passed = stats.mean_entropy >= entropy_targets.min_entropy
            else:
                # Insufficient samples - skip gating (pass by default)
                passed = True
            results["checks"]["entropy"] = {
                "required": entropy_targets.min_entropy,
                "actual": stats.mean_entropy,
                "passed": passed,
                "entropy_samples": stats.entropy_samples,
                "min_samples_required": min_entropy_samples,
                "note": "Skipped due to insufficient samples" if not has_sufficient_entropy_samples else "Using EntropyTargets.min_entropy",
            }
            all_passed = all_passed and passed
        
        # Skill requirements check
        skill_reqs = self.stage_config.skill_requirements
        if skill_reqs.required_skills and self._skill_assessment:
            skill_passed, skill_results = self._skill_assessment.check_requirements(skill_reqs)
            results["checks"]["skills"] = skill_results
            all_passed = all_passed and skill_passed
        
        # Composite score check - ADDITIONAL requirement, NOT an override
        # The composite score provides a holistic view but does NOT bypass traditional checks
        # NOTE: self._composite_score is computed against BASE thresholds (in get_rolling_stats)
        # If thresholds are relaxed, we re-compute against EFFECTIVE thresholds for the gate
        if self._composite_score and self.stage_config.composite_scoring.enabled:
            # If thresholds were relaxed, compute composite against effective thresholds
            if thresholds != base_thresholds:
                effective_composite = compute_composite_score(
                    stats,
                    thresholds,  # effective/adjusted thresholds
                    self.stage_config.composite_scoring,
                )
                results["composite_score"] = effective_composite.to_dict()
                results["composite_score_base"] = self._composite_score.to_dict()
                composite_passed = effective_composite.promotion_ready
                composite_actual = effective_composite.total_score
                composite_meets_floors = effective_composite.meets_hard_floors
            else:
                # No relaxation - use cached composite
                results["composite_score"] = self._composite_score.to_dict()
                composite_passed = self._composite_score.promotion_ready
                composite_actual = self._composite_score.total_score
                composite_meets_floors = self._composite_score.meets_hard_floors
            
            results["checks"]["composite_score"] = {
                "required": self.stage_config.composite_scoring.promotion_threshold,
                "actual": composite_actual,
                "passed": composite_passed,
                "meets_hard_floors": composite_meets_floors,
            }
            all_passed = all_passed and composite_passed
        
        # Phase 2: Anti-gaming checks
        # Run anti-gaming analysis to detect degenerate strategies
        gaming_results = self._anti_gaming_checker.run_all_checks(asdict(stats))
        aggregate_gaming, gaming_concerns = self._anti_gaming_checker.get_aggregate_gaming_score(gaming_results)
        
        results["anti_gaming"] = {
            "aggregate_score": aggregate_gaming,
            "concerns": gaming_concerns,
            "checks": {name: r.gaming_score for name, r in gaming_results.items()},
        }
        
        # Block promotion if significant gaming detected (score > 0.6)
        if aggregate_gaming > 0.6:
            results["checks"]["anti_gaming"] = {
                "required": "< 0.6",
                "actual": aggregate_gaming,
                "passed": False,
                "concerns": gaming_concerns,
            }
            all_passed = False
        elif aggregate_gaming > 0.3:
            # Warning level - don't block but flag
            results["checks"]["anti_gaming"] = {
                "required": "< 0.6",
                "actual": aggregate_gaming,
                "passed": True,
                "warning": True,
                "concerns": gaming_concerns,
            }
        
        # Phase 2.2: Regime-based skill assessment
        # For late stages (>= TIMING_STUDENT), require regime adaptation
        stage_idx = _stage_to_index(self.current_stage)
        if stage_idx >= 4:  # TIMING_STUDENT and above
            self._update_regime_assessment()
            if self._regime_assessment is not None:
                regime_score = self._regime_assessment.adaptation_score
                regime_coverage = self._regime_assessment.regime_coverage
                
                results["regime_assessment"] = {
                    "adaptation_score": regime_score,
                    "regime_coverage": regime_coverage,
                    "volatility_handling": self._regime_assessment.volatility_handling,
                    "trend_following": self._regime_assessment.trend_following,
                    "session_awareness": self._regime_assessment.session_awareness,
                    "cost_resilience": self._regime_assessment.cost_resilience,
                    "total_trades": self._regime_assessment.total_trades,
                    "confidence": self._regime_assessment.confidence,
                }
                
                # Require minimum adaptation score for late stages
                min_adaptation = 0.35 if stage_idx >= 6 else 0.25  # Stricter for prop stages
                if regime_score < min_adaptation:
                    results["checks"]["regime_adaptation"] = {
                        "required": min_adaptation,
                        "actual": regime_score,
                        "passed": False,
                        "reason": "Performance too inconsistent across market regimes",
                    }
                    all_passed = False
                else:
                    results["checks"]["regime_adaptation"] = {
                        "required": min_adaptation,
                        "actual": regime_score,
                        "passed": True,
                    }
                
                # Require minimum regime coverage for late stages
                min_coverage = 0.5 if stage_idx >= 6 else 0.3
                if regime_coverage < min_coverage:
                    results["checks"]["regime_coverage"] = {
                        "required": min_coverage,
                        "actual": regime_coverage,
                        "passed": False,
                        "reason": "Not trading across enough market conditions",
                    }
                    all_passed = False
                else:
                    results["checks"]["regime_coverage"] = {
                        "required": min_coverage,
                        "actual": regime_coverage,
                        "passed": True,
                    }
                
                # Phase 3: Stress resilience check for prop stages (>= 6)
                # Uses cost_resilience and volatility_handling from regime assessment
                if stage_idx >= 6:
                    cost_resilience = self._regime_assessment.cost_resilience
                    vol_handling = self._regime_assessment.volatility_handling
                    
                    # Must handle stress conditions reasonably well
                    min_stress_score = 0.35
                    stress_score = (cost_resilience + vol_handling) / 2
                    
                    results["stress_resilience"] = {
                        "cost_resilience": cost_resilience,
                        "volatility_handling": vol_handling,
                        "combined_score": stress_score,
                        "min_required": min_stress_score,
                    }
                    
                    if stress_score < min_stress_score:
                        results["checks"]["stress_resilience"] = {
                            "required": min_stress_score,
                            "actual": stress_score,
                            "passed": False,
                            "reason": "Poor performance under stress conditions (wide spreads, high volatility)",
                        }
                        all_passed = False
                    else:
                        results["checks"]["stress_resilience"] = {
                            "required": min_stress_score,
                            "actual": stress_score,
                            "passed": True,
                        }
            else:
                results["regime_assessment"] = {"status": "insufficient_data"}
        
        results["promotion_ready"] = all_passed
        results["stats"] = asdict(stats)
        
        return all_passed, results
    
    def check_demotion_criteria(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if demotion criteria are met."""
        config = self.stage_config
        if not config.allow_demotion:
            return False, {"reason": "demotion_disabled"}
        
        # Can't demote below foundation stage (Stage 0)
        if _stage_to_index(self.current_stage) == 0:
            return False, {"reason": "at_foundation"}
        
        thresholds = self.stage_config.competence
        stats = self.get_rolling_stats()
        
        results: Dict[str, Any] = {
            "stage": self.current_stage.name,
            "stage_epoch": self._current_stage_epoch,
            "evaluation_window": stats.window_size,
            "checks": {},
        }
        
        # Need enough data
        min_episodes_for_demotion = max(50, thresholds.evaluation_window // 2)
        if stats.window_size < min_episodes_for_demotion:
            results["should_demote"] = False
            results["reason"] = "insufficient_data"
            results["stats"] = asdict(stats)
            return False, results
        
        # Check composite score for demotion
        if self._composite_score and config.composite_scoring.enabled:
            if self._composite_score.demotion_risk:
                results["should_demote"] = True
                results["reason"] = "composite_score_below_threshold"
                results["composite_score"] = self._composite_score.to_dict()
                results["stats"] = asdict(stats)
                return True, results
        
        # Traditional demotion checks
        demotion_factor = 0.70
        critical_failures = 0
        failure_reasons: List[str] = []
        
        if stats.mean_win_rate < thresholds.min_win_rate * demotion_factor:
            critical_failures += 1
            failure_reasons.append("win_rate_critical")
            results["checks"]["win_rate_critical"] = True
        
        if stats.mean_profit_factor < thresholds.min_profit_factor * demotion_factor:
            critical_failures += 1
            failure_reasons.append("profit_factor_critical")
            results["checks"]["profit_factor_critical"] = True
        
        if stats.mean_drawdown > thresholds.max_avg_drawdown * (1.0 / demotion_factor):
            critical_failures += 1
            failure_reasons.append("drawdown_critical")
            results["checks"]["drawdown_critical"] = True
        
        if stats.dd_breach_rate > thresholds.max_dd_breach_rate * (1.0 / demotion_factor):
            critical_failures += 1
            failure_reasons.append("dd_breach_critical")
            results["checks"]["dd_breach_critical"] = True
        
        should_demote = critical_failures >= 2
        
        results["critical_failures"] = critical_failures
        results["failure_reasons"] = failure_reasons
        results["should_demote"] = should_demote
        results["stats"] = asdict(stats)
        
        return should_demote, results
    
    def _internal_validation_check(self, promotion_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal validation gate based on performance stability and generalization proxies.
        
        Since we don't have separate validation data, we check:
        1. Win rate stability across recent episodes (std < threshold)
        2. Profit factor stability (no wild swings)
        3. Recent vs overall performance ratio (catch late overfitting)
        4. Safety metrics (DD breach rate must stay low)
        
        Returns:
            Dict with "passed" bool and diagnostic info
        """
        stats = self.get_rolling_stats()
        val_cfg = self.stage_config.validation
        stage_idx = _stage_to_index(self.current_stage)
        
        result: Dict[str, Any] = {
            "type": "internal_stability_check",
            "passed": True,
            "checks": {},
        }
        
        # Check 1: Win rate stability (low std = consistent performance)
        # Use recent history for stability check
        recent_window = min(50, stats.window_size)
        recent_episodes = self._current_epoch_window(recent_window)
        
        if len(recent_episodes) >= 20:
            recent_wr = [ep.win_rate for ep in recent_episodes]
            wr_std = float(np.std(recent_wr))
            max_wr_std = 0.15 if stage_idx >= 6 else 0.20  # Stricter for prop stages
            
            wr_stable = wr_std <= max_wr_std
            result["checks"]["win_rate_stability"] = {
                "std": wr_std,
                "max_allowed": max_wr_std,
                "passed": wr_stable,
            }
            if not wr_stable:
                result["passed"] = False
        
        # Check 2: Recent vs overall performance (detect late-episode overfitting)
        if len(recent_episodes) >= 20 and stats.window_size >= 50:
            recent_mean_wr = float(np.mean([ep.win_rate for ep in recent_episodes]))
            overall_mean_wr = stats.mean_win_rate
            
            # Recent should be at least 85% of overall (no severe late degradation)
            min_ratio = val_cfg.min_performance_ratio if hasattr(val_cfg, 'min_performance_ratio') else 0.85
            if overall_mean_wr > 0.01:
                perf_ratio = recent_mean_wr / overall_mean_wr
                ratio_ok = perf_ratio >= min_ratio
                result["checks"]["performance_ratio"] = {
                    "recent_wr": recent_mean_wr,
                    "overall_wr": overall_mean_wr,
                    "ratio": perf_ratio,
                    "min_required": min_ratio,
                    "passed": ratio_ok,
                }
                if not ratio_ok:
                    result["passed"] = False
        
        # Check 3: Safety check (DD breach rate must be acceptable)
        max_dd_breach_rate = 0.05 if stage_idx >= 6 else 0.10
        dd_breach_ok = stats.dd_breach_rate <= max_dd_breach_rate
        result["checks"]["safety"] = {
            "dd_breach_rate": stats.dd_breach_rate,
            "max_allowed": max_dd_breach_rate,
            "passed": dd_breach_ok,
        }
        if not dd_breach_ok:
            result["passed"] = False
        
        # Check 4: For late stages, require regime stability from RegimeSkillAssessment
        if stage_idx >= 6 and self._regime_assessment is not None:
            adaptation = self._regime_assessment.adaptation_score
            min_adaptation = 0.4
            adapt_ok = adaptation >= min_adaptation
            result["checks"]["regime_stability"] = {
                "adaptation_score": adaptation,
                "min_required": min_adaptation,
                "passed": adapt_ok,
            }
            if not adapt_ok:
                result["passed"] = False
        
        return result
    
    # -------------------------------------------------------------------------
    # Promotion/Demotion Execution
    # -------------------------------------------------------------------------

    
    def try_promote(self) -> Tuple[bool, Optional[CurriculumStage]]:
        """Attempt to promote to next stage."""
        if not self.auto_promote:
            return False, None
        
        config = self.stage_config
        if config.is_terminal:
            return False, None
        
        next_stage = get_next_stage(self.current_stage)
        if next_stage is None:
            return False, None
        
        # Respect transition cooldown
        if self._transition_cooldown_remaining > 0:
            return False, None
        
        meets_criteria, results = self.check_promotion_criteria()
        if not meets_criteria:
            return False, None
        
        # Holdout validation gate (if enabled and evaluator provided)
        val_cfg = self.stage_config.validation
        if val_cfg.enabled:
            if self.validation_evaluator is None:
                # Use internal validation gate based on rolling stats stability
                val_result = self._internal_validation_check(results)
                results["validation"] = val_result
                if not val_result.get("passed", False):
                    if self.verbose:
                        logger.info(f"Internal validation failed; blocking promotion. Details: {val_result}")
                    return False, None
            else:
                try:
                    val_result = self.validation_evaluator(self.stage_config)
                    # Expect val_result: {"passed": bool, "performance_ratio": float, ...}
                    results["validation"] = val_result
                    if not val_result.get("passed", False):
                        if self.verbose:
                            logger.info(f"Validation failed; blocking promotion. Details: {val_result}")
                        return False, None
                except Exception as e:
                    logger.warning(f"Validation evaluator error: {e}; skipping validation gate.")
        
        old_stage = self.current_stage
        self.current_stage = next_stage
        self._enter_stage(next_stage, reason="promotion")
        
        transition_info = {
            "type": "promotion",
            "from_stage": old_stage.name,
            "to_stage": next_stage.name,
            "timestamp": _now_iso(self.tz),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "criteria_results": results,
        }
        self._transitions.append(transition_info)
        
        if self.verbose:
            logger.info(f"🎓 PROMOTION: {old_stage.name} → {next_stage.name} (epoch={self._current_stage_epoch})")
        
        # Trigger callback
        if self.on_transition_callback is not None:
            try:
                self.on_transition_callback("promotion", old_stage, next_stage, transition_info)
            except Exception as e:
                logger.warning(f"Transition callback error: {e}")
        
        return True, next_stage
    
    def try_demote(self) -> Tuple[bool, Optional[CurriculumStage]]:
        """Attempt to demote to previous stage."""
        if not self.auto_demote:
            return False, None
        
        previous_stage = get_previous_stage(self.current_stage)
        if previous_stage is None:
            return False, None
        
        # Respect transition cooldown
        if self._transition_cooldown_remaining > 0:
            return False, None
        
        should_demote, results = self.check_demotion_criteria()
        if not should_demote:
            return False, None
        
        old_stage = self.current_stage
        
        # Record demotion for analysis
        stats = self.get_rolling_stats()
        failure_reasons = results.get("failure_reasons", [])
        self._demotion_analyzer.record_demotion(
            from_stage=old_stage,
            to_stage=previous_stage,
            failure_reasons=failure_reasons,
            skill_assessment=self._skill_assessment,
            stats=stats,
            global_episode=self.total_episodes,
        )
        
        self.current_stage = previous_stage
        # Pass old_stage so recovery protocol checks failures from the stage we failed at
        self._enter_stage(previous_stage, reason="demotion", demoted_from_stage=old_stage)
        
        transition_info = {
            "type": "demotion",
            "from_stage": old_stage.name,
            "to_stage": previous_stage.name,
            "timestamp": _now_iso(self.tz),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "criteria_results": results,
            "failure_reasons": failure_reasons,
        }
        self._transitions.append(transition_info)
        
        if self.verbose:
            logger.warning(f"📉 DEMOTION: {old_stage.name} → {previous_stage.name} (epoch={self._current_stage_epoch})")
        
        # Trigger callback
        if self.on_transition_callback is not None:
            try:
                self.on_transition_callback("demotion", old_stage, previous_stage, transition_info)
            except Exception as e:
                logger.warning(f"Transition callback error: {e}")
        
        return True, previous_stage
    
    def update(self) -> Tuple[bool, Optional[CurriculumStage]]:
        """Check and execute any warranted stage transitions."""
        promoted, new_stage = self.try_promote()
        if promoted:
            return True, new_stage
        
        demoted, new_stage = self.try_demote()
        if demoted:
            return True, new_stage
        
        return False, None
    
    def force_stage(self, stage: CurriculumStage, reason: str = "manual") -> None:
        """Force transition to a specific stage."""
        old_stage = self.current_stage
        self.current_stage = stage
        self._enter_stage(stage, reason=f"force:{reason}")
        
        self._transitions.append({
            "type": "force",
            "from_stage": old_stage.name,
            "to_stage": stage.name,
            "reason": reason,
            "timestamp": _now_iso(self.tz),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
        })
        
        if self.verbose:
            logger.info(f"⚡ FORCE STAGE: {old_stage.name} → {stage.name} (reason={reason})")
    
    # -------------------------------------------------------------------------
    # Progress Reporting
    # -------------------------------------------------------------------------
    
    def _get_regime_assessment_report(self) -> Dict[str, Any]:
        """Get regime skill assessment report for dashboard."""
        self._update_regime_assessment()
        
        if self._regime_assessment is None:
            return {
                "status": "insufficient_data",
                "total_trades": len(self._regime_trades),
                "min_required": 50,
            }
        
        ra = self._regime_assessment
        
        return {
            "status": "active",
            "total_trades": ra.total_trades,
            "confidence": ra.confidence,
            
            # Main skill scores
            "scores": {
                "adaptation": ra.adaptation_score,
                "volatility_handling": ra.volatility_handling,
                "trend_following": ra.trend_following,
                "session_awareness": ra.session_awareness,
                "cost_resilience": ra.cost_resilience,
            },
            
            # Coverage
            "regime_coverage": ra.regime_coverage,
            
            # Weaknesses
            "weaknesses": ra.get_weaknesses(threshold=0.4),
            
            # Breakdown by regime (for detailed view)
            "volatility_breakdown": {
                k.value: {
                    "trades": v.trade_count,
                    "win_rate": v.win_rate,
                    "avg_r": v.avg_r,
                }
                for k, v in ra.volatility_performance.items()
            },
            "trend_breakdown": {
                k.value: {
                    "trades": v.trade_count,
                    "win_rate": v.win_rate,
                    "avg_r": v.avg_r,
                }
                for k, v in ra.trend_performance.items()
            },
            "session_breakdown": {
                k.value: {
                    "trades": v.trade_count,
                    "win_rate": v.win_rate,
                    "avg_r": v.avg_r,
                }
                for k, v in ra.session_performance.items()
            },
            "spread_breakdown": {
                k.value: {
                    "trades": v.trade_count,
                    "win_rate": v.win_rate,
                    "avg_r": v.avg_r,
                }
                for k, v in ra.spread_performance.items()
            },
        }
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Generate comprehensive progress report."""
        stats = self.get_rolling_stats()
        meets_promotion, promotion_results = self.check_promotion_criteria()
        meets_demotion, demotion_results = self.check_demotion_criteria()
        
        progression = get_stage_progression()
        stage_idx = progression.index(self.current_stage)
        
        # Build phase_info for dashboard display
        reward_cfg = self.stage_config.rewards if self.stage_config else None
        bonuses_enabled = False
        if reward_cfg:
            # Bonuses are "enabled" if any meaningful bonus scale is > 0
            bonuses_enabled = (
                getattr(reward_cfg, 'trailing_stop_bonus', 0.0) > 0.01 or
                getattr(reward_cfg, 'r_multiple_bonus_scale', 0.0) > 0.01 or
                getattr(reward_cfg, 'agent_close_bonus', 0.0) > 0.01
            )
        
        phase_info = {
            "phase_num": stage_idx,
            "phase_name": self.current_stage.name,
            "description": getattr(self.stage_config, 'description', '') if self.stage_config else '',
            "bonuses_enabled": bonuses_enabled,
            "loss_multiplier_range": f"{getattr(reward_cfg, 'loss_penalty_mult', 1.0):.1f}x" if reward_cfg else "1.0x",
            "market_difficulty": "Progressive" if stage_idx < 5 else "Full",
        }
        
        # Identify promotion blockers with correct gap calculation
        # For "min_*" metrics: gap = required - actual (need to increase)
        # For "max_*" metrics: gap = actual - required (need to decrease)
        blockers = []
        for check_name, check_data in promotion_results.get("checks", {}).items():
            if isinstance(check_data, dict) and not check_data.get("passed", True):
                required = check_data.get("required", 0)
                actual = check_data.get("actual", 0)
                
                # Determine if this is a "max" metric (lower is better)
                is_max_metric = any(kw in check_name.lower() for kw in [
                    "max_", "std", "breach", "drawdown", "loss"
                ])
                
                if is_max_metric:
                    # For max metrics: violation = actual - required (positive = bad)
                    gap = actual - required
                else:
                    # For min metrics: violation = required - actual (positive = shortfall)
                    gap = required - actual
                
                blockers.append({
                    "metric": check_name,
                    "gap": gap,
                    "required": required,
                    "actual": actual,
                    "is_max_metric": is_max_metric,
                })
        blockers.sort(key=lambda x: abs(x.get("gap", 0)), reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(blockers)
        
        # Estimate episodes to promotion (rough heuristic)
        estimated_episodes = 0
        if blockers and not meets_promotion:
            velocity = self._learning_velocity
            avg_improvement = velocity.get_average_improvement()
            if avg_improvement > 0:
                avg_gap = np.mean([abs(b["gap"]) for b in blockers if b["gap"] is not None])
                estimated_episodes = int(avg_gap / avg_improvement * 10)
            else:
                estimated_episodes = -1  # Can't estimate
        
        return {
            "version": STATE_VERSION,
            "current_stage": self.current_stage.name,
            "current_stage_epoch": self._current_stage_epoch,
            "stage_index": stage_idx,
            "total_stages": len(progression),
            "progress_pct": (stage_idx / (len(progression) - 1)) * 100 if len(progression) > 1 else 100,
            
            # Dashboard phase info
            "phase_info": phase_info,
            
            # Totals
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "stage_timesteps": self.stage_timesteps,
            "stage_episodes": self.stage_episodes,
            
            # Stats
            "rolling_stats": asdict(stats),
            
            # Promotion/Demotion
            "promotion_ready": meets_promotion,
            "promotion_checks": promotion_results.get("checks", {}),
            "promotion_blockers": blockers[:5],
            "demotion_risk": meets_demotion,
            "demotion_checks": demotion_results.get("checks", {}),
            
            # Anti-gaming (Phase 2)
            "anti_gaming": promotion_results.get("anti_gaming", {}),
            
            # Estimates
            "estimated_episodes_to_promotion": estimated_episodes,
            "recommendations": recommendations,
            
            # New v2.0 features
            "learning_velocity": self._learning_velocity.to_dict(),
            "is_plateaued": self._learning_velocity.is_plateaued(),
            "skill_assessment": self._skill_assessment.to_dict() if self._skill_assessment else None,
            "composite_score": self._composite_score.to_dict() if self._composite_score else None,
            "recovery_protocol": self._recovery_state.to_dict(),
            "review_session": self._review_state.to_dict(),
            "demotion_analysis": self._demotion_analyzer.to_dict(),
            
            # Phase 2.1 hardening
            "invariant_summary": self._invariant_checker.get_summary(),
            
            # Phase 2.2 - Regime skill assessment
            "regime_assessment": self._get_regime_assessment_report(),
            
            # Transition state
            "is_in_transition": self.is_in_transition,
            "transition_cooldown_remaining": self._transition_cooldown_remaining,
            "reward_blend_remaining": self._reward_blend_remaining,
            "reward_blend_factor": self.reward_blend_factor,
            "lr_warmup_active": self._lr_warmup_active,
            "lr_warmup_steps_remaining": self._lr_warmup_steps_remaining,
            "lr_multiplier": self.get_lr_multiplier(),
            
            # History
            "transitions_count": len(self._transitions),
            "recent_transitions": self._transitions[-5:],
        }
    
    def _generate_recommendations(self, blockers: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on blockers."""
        recs = []
        
        for blocker in blockers[:3]:
            metric = blocker.get("metric", "")
            
            if "win_rate" in metric:
                recs.append("Focus on entry quality - wait for higher-confidence setups")
            elif "drawdown" in metric:
                recs.append("Reduce position sizes or tighten stops to control drawdown")
            elif "profit_factor" in metric:
                recs.append("Improve risk-reward: let winners run longer, cut losers faster")
            elif "stability" in metric or "std" in metric:
                recs.append("Maintain consistent approach across different market conditions")
            elif "trade" in metric and "activity" in metric:
                recs.append("Increase trading activity with quality entries")
            elif "dd_breach" in metric:
                recs.append("Critical: Avoid drawdown breaches - reduce risk per trade")
            elif "consecutive_loss" in metric:
                recs.append("Take breaks after consecutive losses to reset")
            elif "skill" in metric:
                if self._skill_assessment and self._skill_assessment.weakest_skills:
                    weak = self._skill_assessment.weakest_skills[0].value
                    recs.append(f"Focus on improving {weak.replace('_', ' ')}")
        
        # Add velocity-based recommendation (use stage-configured threshold)
        if self.is_learning_plateaued():
            recs.append("Learning has plateaued - consider adjusting strategy or hyperparameters")
        
        # Deduplicate
        return list(dict.fromkeys(recs))
    
    # -------------------------------------------------------------------------
    # Goal-Based Training Termination
    # -------------------------------------------------------------------------
    
    def should_stop_training(
        self,
        *,
        max_timesteps: Optional[int] = None,
        max_episodes: Optional[int] = None,
        max_hours: Optional[float] = None,
        start_time: Optional[float] = None,
        plateau_stop: bool = True,
        plateau_threshold_episodes: int = 500,
        max_demotions_from_same_stage: int = 5,
        mastery_confirmation_episodes: int = 100,
    ) -> Tuple[bool, str]:
        """
        Check if training should stop based on curriculum goals.
        
        Use this instead of fixed timesteps for goal-oriented training.
        Training stops when:
        1. Agent reaches MASTERY stage AND maintains it for confirmation episodes
        2. Safety caps are hit (max_timesteps, max_episodes, max_hours)
        3. Learning has plateaued for too long (optional)
        4. Agent has been demoted from the same stage too many times
        
        Args:
            max_timesteps: Safety cap on total timesteps (None = no cap)
            max_episodes: Safety cap on total episodes (None = no cap)
            max_hours: Safety cap on training hours (None = no cap)
            start_time: Training start time from time.time() for hours cap
            plateau_stop: Whether to stop on extended plateau
            plateau_threshold_episodes: Episodes of no improvement before plateau stop
            max_demotions_from_same_stage: Stop if repeatedly failing same stage
            mastery_confirmation_episodes: Episodes to confirm MASTERY is stable
            
        Returns:
            (should_stop, reason): Tuple of bool and explanation string
        """
        # 1. Check if MASTERY achieved and confirmed
        if self.stage_config.is_terminal:
            # At terminal stage - check if stable
            if self.stage_episodes >= mastery_confirmation_episodes:
                # Check we're still meeting criteria (not about to be demoted)
                meets_demotion, _ = self.check_demotion_criteria()
                if not meets_demotion:
                    return True, f"GOAL_ACHIEVED: Reached {self.current_stage.name} and maintained for {self.stage_episodes} episodes"
        
        # 2. Safety caps
        if max_timesteps is not None and self.total_timesteps >= max_timesteps:
            return True, f"MAX_TIMESTEPS: Reached {self.total_timesteps:,} timesteps"
        
        if max_episodes is not None and self.total_episodes >= max_episodes:
            return True, f"MAX_EPISODES: Reached {self.total_episodes:,} episodes"
        
        if max_hours is not None and start_time is not None:
            import time
            elapsed_hours = (time.time() - start_time) / 3600.0
            if elapsed_hours >= max_hours:
                return True, f"MAX_HOURS: Training ran for {elapsed_hours:.1f} hours"
        
        # 3. Plateau detection
        if plateau_stop and self._learning_velocity.is_plateaued(plateau_threshold_episodes):
            return True, f"PLATEAU: No improvement for {self._learning_velocity.plateau_episodes} episodes"
        
        # 4. Repeated failure at same stage
        for stage, count in self._demotion_analyzer.stage_failure_counts.items():
            if count >= max_demotions_from_same_stage:
                return True, f"REPEATED_FAILURE: Demoted from {stage.name} {count} times"
        
        return False, "TRAINING"
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status for goal-based training.
        
        Returns dict with progress info for logging/dashboard.
        """
        progression = get_stage_progression()
        stage_idx = progression.index(self.current_stage)
        
        return {
            "current_stage": self.current_stage.name,
            "stage_index": stage_idx,
            "total_stages": len(progression),
            "progress_pct": (stage_idx / max(len(progression) - 1, 1)) * 100,
            "is_terminal": self.stage_config.is_terminal,
            "stage_episodes": self.stage_episodes,
            "total_episodes": self.total_episodes,
            "total_timesteps": self.total_timesteps,
            "plateau_episodes": self._learning_velocity.plateau_episodes,
            "is_plateaued": self._learning_velocity.is_plateaued(),
            "demotion_counts": {s.name: c for s, c in self._demotion_analyzer.stage_failure_counts.items()},
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def save(self, path: Path) -> None:
        """Save curriculum state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        history_serialized: Dict[str, List[Dict[str, Any]]] = {}
        for stage, episodes in self._history.items():
            history_serialized[stage.name] = [asdict(ep) for ep in episodes]
        
        state = {
            "version": STATE_VERSION,
            "tz": self.tz,
            "current_stage": self.current_stage.name,
            "current_stage_epoch": self._current_stage_epoch,
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "stage_timesteps_total": {s.name: t for s, t in self._stage_timesteps_total.items()},
            "stage_episodes_total": {s.name: e for s, e in self._stage_episodes_total.items()},
            "stage_epoch_counter": {s.name: e for s, e in self._stage_epoch_counter.items()},
            "stage_timesteps_current": self.stage_timesteps,
            "stage_episodes_current": self.stage_episodes,
            "history": history_serialized,
            "transitions": self._transitions,
            "demotion_analyzer": self._demotion_analyzer.to_dict(),
            "recovery_state": self._recovery_state.to_dict(),
            "review_state": self._review_state.to_dict(),
            "learning_velocity": {
                "plateau_episodes": self._learning_velocity.plateau_episodes,
                "improvement_rates": self._learning_velocity.improvement_rates,
                "metric_history": {
                    k: list(v)[-100:] for k, v in self._learning_velocity.metric_history.items()
                } if hasattr(self._learning_velocity, 'metric_history') else {},
            },
            # Transition counters (prevents behavior change on resume)
            "transition_cooldown_remaining": self._transition_cooldown_remaining,
            "reward_blend_remaining": self._reward_blend_remaining,
            # Needed to restore reward blending correctly across resumes
            "previous_stage": self._previous_stage_name,
            "lr_warmup_active": self._lr_warmup_active,
            "lr_warmup_steps_remaining": self._lr_warmup_steps_remaining,
            "lr_warmup_factor": self._lr_warmup_factor,
            # Episode tracking
            "last_episode_end_idx": self._last_episode_end_idx,
            # True idempotency tracking (bounded list of processed episode IDs)
            "processed_episode_ids": list(self._processed_episode_ids),
            # Current entropy state
            "current_entropy": self._current_entropy,
            # Review tick latch
            "review_tick_episode": self._review_tick_episode,
            # RNG state for reproducible mixed-stage sampling
            "rng_state": self._rng.bit_generator.state,
            "saved_at": _now_iso(self.tz),
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=False)
        
        logger.info(f"Curriculum state saved to {path}")
    
    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "CurriculumManager":
        """Load curriculum state from file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        # Version compatibility check
        loaded_version = state.get("version", "1.0")
        if loaded_version != STATE_VERSION:
            logger.warning(
                f"Loading checkpoint from v{loaded_version} (current: v{STATE_VERSION}). "
                f"New features (recovery_state, review_state, learning_velocity) may use defaults."
            )
        
        tz = state.get("tz", DEFAULT_TZ)
        current_stage = CurriculumStage[state["current_stage"]]
        
        manager = cls(initial_stage=current_stage, tz=tz, **kwargs)
        manager.total_timesteps = _safe_int(state.get("total_timesteps", 0), 0)
        manager.total_episodes = _safe_int(state.get("total_episodes", 0), 0)
        
        # Restore totals
        for stage_name, timesteps in (state.get("stage_timesteps_total", {}) or {}).items():
            try:
                manager._stage_timesteps_total[CurriculumStage[stage_name]] = _safe_int(timesteps, 0)
            except KeyError:
                logger.debug(f"Unknown stage in timesteps_total: {stage_name}")
        
        for stage_name, episodes in (state.get("stage_episodes_total", {}) or {}).items():
            try:
                manager._stage_episodes_total[CurriculumStage[stage_name]] = _safe_int(episodes, 0)
            except KeyError:
                logger.debug(f"Unknown stage in episodes_total: {stage_name}")
        
        # Restore epoch counters
        for stage_name, epoch in (state.get("stage_epoch_counter", {}) or {}).items():
            try:
                manager._stage_epoch_counter[CurriculumStage[stage_name]] = _safe_int(epoch, 0)
            except KeyError:
                logger.debug(f"Unknown stage in epoch_counter: {stage_name}")
        
        manager._current_stage_epoch = _safe_int(
            state.get("current_stage_epoch", manager._current_stage_epoch),
            manager._current_stage_epoch
        )
        manager.stage_timesteps = _safe_int(state.get("stage_timesteps_current", 0), 0)
        manager.stage_episodes = _safe_int(state.get("stage_episodes_current", 0), 0)
        
        # Restore history
        manager._history = {stage: deque(maxlen=manager.max_history_size) for stage in CurriculumStage}
        for stage_name, episodes_data in (state.get("history", {}) or {}).items():
            try:
                stage = CurriculumStage[stage_name]
            except KeyError:
                logger.debug(f"Unknown stage in history: {stage_name}")
                continue
            for ep_data in (episodes_data or []):
                try:
                    manager._history[stage].append(EpisodeMetrics.from_dict(ep_data))
                except Exception as e:
                    logger.debug(f"Skipping malformed episode in history: {e}")
                    continue
        
        manager._transitions = state.get("transitions", []) or []
        
        # Restore demotion analyzer
        demotion_data = state.get("demotion_analyzer", {})
        if demotion_data:
            manager._demotion_analyzer.load_from_dict(demotion_data)
        
        # Restore recovery state (FIXED: was missing)
        recovery_data = state.get("recovery_state", {})
        if recovery_data:
            manager._recovery_state = RecoveryProtocolState.from_dict(recovery_data)
        
        # Restore review state (FIXED: was missing)
        review_data = state.get("review_state", {})
        if review_data:
            manager._review_state = ReviewSessionState.from_dict(review_data)
        
        # Restore learning velocity
        velocity_data = state.get("learning_velocity", {})
        if velocity_data:
            manager._learning_velocity.plateau_episodes = velocity_data.get("plateau_episodes", 0)
            manager._learning_velocity.improvement_rates = velocity_data.get("improvement_rates", {})
            # Restore metric history if available
            metric_history = velocity_data.get("metric_history", {})
            if metric_history and hasattr(manager._learning_velocity, 'metric_history'):
                for k, v in metric_history.items():
                    manager._learning_velocity.metric_history[k] = deque(v, maxlen=100)
        
        # Restore transition counters (prevents behavior change on resume)
        manager._transition_cooldown_remaining = _safe_int(state.get("transition_cooldown_remaining", 0), 0)
        manager._reward_blend_remaining = _safe_int(state.get("reward_blend_remaining", 0), 0)
        manager._lr_warmup_active = bool(state.get("lr_warmup_active", False))
        manager._lr_warmup_steps_remaining = _safe_int(state.get("lr_warmup_steps_remaining", 0), 0)
        manager._lr_warmup_factor = _safe_float(state.get("lr_warmup_factor", 1.0), 1.0)

        # Restore reward blend provenance (prevents behavior change on resume)
        prev_stage_name = state.get("previous_stage", None)
        manager._previous_stage_name = prev_stage_name
        if prev_stage_name and manager._reward_blend_remaining > 0:
            try:
                manager._previous_stage_config = get_stage_config(CurriculumStage[prev_stage_name])
            except Exception as e:
                logger.warning(f"Could not restore previous_stage_config for blending: {e}; disabling blend.")
                manager._previous_stage_config = None
                manager._reward_blend_remaining = 0
        
        # Restore episode tracking
        manager._last_episode_end_idx = _safe_int(state.get("last_episode_end_idx", -1), -1)
        
        # Restore true idempotency tracking
        processed_ids = state.get("processed_episode_ids", [])
        if processed_ids:
            # Keep only the most recent IDs (bounded)
            trimmed = list(processed_ids)[-PROCESSED_EPISODE_ID_LIMIT:]
            manager._processed_episode_ids = deque(trimmed)
            manager._processed_episode_id_set = set(trimmed)
        
        # Restore current entropy state
        manager._current_entropy = _safe_float(state.get("current_entropy", -1.0), -1.0)
        
        # Restore review tick latch
        manager._review_tick_episode = _safe_int(state.get("review_tick_episode", -1), -1)
        
        # Restore RNG state for reproducible mixed-stage sampling
        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                manager._rng.bit_generator.state = rng_state
            except Exception as e:
                logger.warning(f"Could not restore RNG state: {e}")
        
        manager._rolling_stats_dirty = True
        
        logger.info(f"Curriculum state loaded from {path} (stage={current_stage.name}, epoch={manager._current_stage_epoch})")
        return manager
    
    def __repr__(self) -> str:
        return (
            f"CurriculumManager(stage={self.current_stage.name}, "
            f"epoch={self._current_stage_epoch}, "
            f"episodes={self.total_episodes}, "
            f"timesteps={self.total_timesteps:,})"
        )