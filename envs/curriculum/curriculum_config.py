# envs/curriculum_config.py
"""
Curriculum Configuration for Trading RL Agent
==============================================

Defines curriculum progression from early learning to live-ready discipline.

This file re-exports TYPES from envs/curriculum/config/ subpackage for backward compatibility,
and defines the stage factory functions + validators in one place.

Enhancements in this version (v2.0):
- Skill-based competency requirements per stage
- Entropy targets for exploration management
- Composite scoring configuration
- Recovery protocol definitions
- Validation configuration
- Adaptive threshold settings
- Mixed-stage sampling configuration
- Review session scheduling
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

# Import all types from curriculum.config subpackage
from envs.curriculum.config import (
    COMPOSITE_HARD_FLOOR_KEYS,
    COMPOSITE_WEIGHT_KEYS,
    MIN_EVALUATION_EPISODES,
    AdaptiveThresholdConfig,
    # Thresholds
    CompetenceThresholds,
    CompositeScoringConfig,
    # Stages
    CurriculumStage,
    CurriculumStageConfig,
    DataDifficulty,
    EntropyTargets,
    # Execution
    ExecutionDifficulty,
    MarketRegime,
    MixedStageSamplingConfig,
    # Protocols
    RecoveryProtocolConfig,
    ReviewSessionConfig,
    RewardShaping,
    SkillRequirements,
    # Constraints
    TradingConstraints,
    TradingSkill,
    TransitionSettings,
    ValidationConfig,
    is_valid_threshold_field,
)

# =============================================================================
# Shared Curriculum Constants
# =============================================================================

# Granular time-of-day quality map for session timing rewards.
TIME_OF_DAY_QUALITY = {
    "00:00-04:00": 0.3,  # Asian session - low quality
    "04:00-08:00": 0.5,  # Asian/London overlap - medium
    "08:00-12:00": 0.9,  # London prime - high
    "12:00-16:00": 0.8,  # London/NY overlap - high
    "16:00-20:00": 0.7,  # NY afternoon - medium
    "20:00-00:00": 0.4,  # Late NY - low
}

# Entry certainty reward bands.
ENTRY_CERTAINTY_BONUS = {
    "0.70-0.80": 0.05,
    "0.80-0.90": 0.12,
    "0.90-1.00": 0.20,
}

# Dynamic patience multipliers based on market state.
PATIENCE_BONUS_MULTIPLIER = {
    "low_volatility": 0.5,
    "high_volatility": 2.0,
    "trending": 1.5,
    "ranging": 0.8,
}



# =============================================================================
# Stage Factory Functions
# =============================================================================
# 
# 4-PHASE "FIRST GRADE TO UNIVERSITY" CURRICULUM
# ===============================================
# 
# PHASE 0: DISCOVERY (Stages 0-1) - "Kindergarten"
#   - Focus: Pure exploration, learn market patterns
#   - Rewards: HIGH exploration bonus, minimal penalties
#   - Entropy: HIGH (0.50-0.60) - MUST explore
#   - Goal: Don't develop "don't trade" habit
#
# PHASE 1: FOUNDATION (Stages 2-4) - "Elementary School"  
#   - Focus: ONE concept per stage (trend, session, timing)
#   - Rewards: Gradual introduction of profit incentives
#   - Entropy: MODERATE-HIGH (0.30-0.40) - still exploring
#   - Goal: Learn WHY each concept matters
#
# PHASE 2: DEVELOPMENT (Stages 5-7) - "High School"
#   - Focus: COMBINE learned concepts into strategies
#   - Rewards: All components enabled, increasing strength
#   - Entropy: MODERATE (0.12-0.22) - strategy forming
#   - Goal: Build coherent trading approach
#
# PHASE 3: MASTERY (Stages 8-9) - "University"
#   - Focus: Prop firm constraints, live-ready execution
#   - Rewards: Full strength + variance penalties
#   - Entropy: LOW (0.05-0.08) - converged strategy
#   - Goal: Execute strategy under real constraints
#
# =============================================================================


def get_explorer_config() -> CurriculumStageConfig:
    """
    PHASE 0 - DISCOVERY: Stage 0 (EXPLORER)
    ========================================
    Pure market observation. NO trading penalties.
    
    Goal: Learn market patterns WITHOUT developing "don't trade" habit.
    
    ENABLED: High exploration bonus, light PnL signal
    DISABLED: All penalties (anti-churn, DD shaping, etc.)
    MARKET: Very easy
    ENTROPY: 0.60 minimum - MUST explore widely
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.EXPLORER,
        name="Explorer",
        description="DISCOVERY Phase: Pure exploration. Learn market patterns.",
        execution=ExecutionDifficulty(
            # VERY EASY - no execution friction
            base_spread_points=0.01,
            spread_mult_range=(1.0, 1.0),
            max_spread_points=0.02,
            slippage_points_sigma=0.0,
            slippage_mult_range=(1.0, 1.0),
            max_slippage_points=0.0,
            commission_per_lot=0.0,
            latency_bars=0,
            enable_randomization=False,
            # DATA SPREAD: DISABLED - use synthetic training wheels
            use_data_spread=False,
            data_spread_scale=0.0,
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 0 DISCOVERY: Pure exploration, NO penalties
            # ============================================================================
            reward_scale=5.0,                 # STANDARDIZED: Same scale across stages to reduce critic shock
            loss_multiplier=1.0,
            
            # PnL DOMINANCE (v6.0): Low during discovery - don't overwhelm exploration
            pnl_scale_factor=150.0,           # FIX: PnL must dominate over exploration bonus
            max_shaping_to_pnl_ratio=0.5,     # FIX: Reduce shaping freedom to prevent reward hacking
            
            # Execution costs: DISABLED during discovery - frictionless exploration
            execution_cost_visibility_enabled=False,
            execution_cost_reward_scale=0.0,
            
            # Good loss cuts: DISABLED during discovery
            good_loss_cut_enabled=False,
            good_loss_cut_bonus=0.0,
            
            # Cost erosion: DISABLED during discovery
            cost_erosion_penalty_enabled=False,
            
            # NO profit incentives yet - just observe
            r_multiple_bonus_threshold=99.0,
            r_multiple_bonus_scale=0.0,
            r_multiple_bonus_cap=0.0,
            
            mae_efficiency_enabled=False,
            mae_efficiency_scale=0.0,
            mae_efficiency_threshold=99.0,
            
            time_efficiency_enabled=False,
            time_efficiency_scale=0.0,
            optimal_trade_bars=12,
            max_trade_bars_for_bonus=48,
            
            exit_quality_enabled=False,
            trailing_stop_bonus=0.0,
            agent_close_bonus=0.0,
            hard_stop_penalty=0.0,
            risk_liquidation_penalty=0.0,
            
            truncation_winner_discount=0.10,  # Very light
            truncation_loser_extra_penalty=0.05,
            
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            # DD SHAPING: DISABLED - let agent explore freely
            dd_shaping_enabled=False,
            dd_threshold=0.50,
            dd_penalty_scale=0.0,
            dd_severity_exponent=1.0,
            dd_severity_cap=0.0,
            
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            # ANTI-CHURN: ENABLED with mild penalty to prevent overtrading habit
            # FIX: Was disabled, causing 380 trades/ep. Now soft limit at 25.
            anti_churn_enabled=True,
            daily_trade_soft_limit=25,
            churn_penalty_per_trade=0.02,  # Mild penalty to discourage excessive trading
            
            hard_block_penalty=0.0,
            soft_block_penalty=0.0,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            # ACTIVITY CONSISTENCY: MILD for exploration stage
            # Let agent explore, but gently nudge away from extreme overtrading
            # Real learning comes from PnL signal (execution costs hurt naturally)
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=8.0,  # Target ~12 trades/1500 steps
            activity_deviation_penalty_scale=0.3,  # MILD: allow exploration
            activity_deviation_penalty_cap=10.0,  # Cap at -10 (not -30)
            min_trades_penalty=0.3,  # Penalty if < 20% of target
            
            # EXPLORATION BONUS: SMALL - encourage activity but PnL must dominate
            # FIX: Was 0.10 which dominated PnL signal causing reward hacking
            exploration_bonus=0.02,
            directional_accuracy_weight=0.5,  # Light directional signal
            min_reward=-50.0,  # FIX: Was -1.5, clipping destroyed gradients
            max_reward=50.0,   # FIX: Was 1.5
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=50,            # Very permissive
            max_trades_per_session=25,
            max_consecutive_losses=20,        # Very permissive
            loss_layer_stop=19,               # Block at 19 to prevent reaching 20 = no breach
            # Session budget (v5.5): Very permissive for exploration
            session_loss_limit_pct=0.99,      # Effectively infinite
            session_consecutive_loss_limit=99,
            enforce_session_windows=False,
            enforce_no_new_trades_window=False,
            enforce_weekend_block=False,
            enforce_hard_close=False,
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.50,        # Very loose
            max_drawdown_limit=0.50,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.45,
            entry_quality_gate_enabled=True,  # FIX: Enable mild quality gate
            entry_quality_threshold=0.10,  # FIX: Lowest threshold (EXPERIMENTER=0.12, increases from here)
            hard_stop_loss_eur=1000.0,
            soft_stop_loss_eur=800.0,
            trailing_activation_eur=200.0,
            trailing_retrace_pct=0.50,
            time_decay_hours=24.0,
            risk_per_trade_pct=0.01,
            max_risk_per_trade_pct=0.02,
        ),
        competence=CompetenceThresholds(
            # DISCOVERY: Just need to explore
            min_episodes=100,
            min_timesteps=100_000,
            min_win_rate=0.0,                 # No performance requirements
            min_profit_factor=0.0,
            max_avg_drawdown=0.50,            # Very loose
            min_avg_pnl=-5000.0,              # Can lose money
            min_avg_r_multiple=-1.0,
            min_entropy=0.60,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=1.0,             # Ignore variance
            max_pnl_std=50000.0,
            min_trade_count_avg=5.0,          # Must be trading
            max_dd_breach_rate=1.0,           # Ignore DD breaches
            max_consecutive_loss_rate=1.0,
            evaluation_window=50,
        ),
        max_steps_per_episode=1500,
        include_memory_features=False,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=False,                 # Can't demote from Stage 0
        data_difficulty=DataDifficulty(
            # EASY MARKET: Clear patterns to observe
            volatility_percentile_range=(0.0, 0.30),
            min_trend_clarity=0.5,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=True,
            exclude_market_open_close=True,
            prefer_recent_data=False,
            recent_data_weight=1.0,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
            ],
            regime_sampling_weights={
                "trending_up": 0.50,
                "trending_down": 0.50,
            },
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=False,
            lr_warmup_factor=1.0,
            lr_warmup_steps=0,
            reward_blend_enabled=False,
            reward_blend_episodes=0,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=20,
        ),
        skill_requirements=SkillRequirements(
            # No skill requirements - just explore
            required_skills={},
            min_confidence=0.0,
            require_all_skills=False,
            weighted_threshold=0.0,
        ),
        entropy_targets=EntropyTargets(
            # HIGH entropy required - MUST explore
            min_entropy=0.60,
            max_entropy=1.50,
            low_entropy_penalty_scale=0.20,   # Strong penalty for not exploring
            high_entropy_penalty_scale=0.0,   # No penalty for too much exploration
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.40,         # Easy promotion
            demotion_threshold=0.0,           # No demotion
            hard_floors={},                   # No hard requirements
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=False,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=False,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=False,
        ),
        review_session=ReviewSessionConfig(
            enabled=False,
        ),
    )


def get_experimenter_config() -> CurriculumStageConfig:
    """
    PHASE 0 - DISCOVERY: Stage 1 (EXPERIMENTER)
    ============================================
    Light outcome signals. Learn that trades have consequences.
    
    Goal: Start associating actions with outcomes, still exploring freely.
    
    ENABLED: Light PnL signal, very light DD awareness, exploration bonus
    DISABLED: Anti-churn, most penalties
    MARKET: Very easy
    ENTROPY: 0.50 minimum - still high exploration
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.EXPERIMENTER,
        name="Experimenter",
        description="DISCOVERY Phase: Light outcome signals. Actions have consequences.",
        execution=ExecutionDifficulty(
            base_spread_points=0.02,
            spread_mult_range=(1.0, 1.0),
            max_spread_points=0.05,
            slippage_points_sigma=0.0,
            slippage_mult_range=(1.0, 1.0),
            max_slippage_points=0.0,
            commission_per_lot=0.0,
            latency_bars=0,
            enable_randomization=False,
            # DATA SPREAD: DISABLED - still using training wheels
            use_data_spread=False,
            data_spread_scale=0.0,
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 0 DISCOVERY: Light outcome signals
            # ============================================================================
            reward_scale=5.0,                 # STANDARDIZED: Same scale across stages
            loss_multiplier=1.0,
            
            # PnL DOMINANCE (v6.0): CRITICAL - PnL must dominate exploration bonus
            pnl_scale_factor=150.0,           # FIX: Was 80, exploration bonus was 2.3x PnL signal!
            max_shaping_to_pnl_ratio=0.5,     # FIX: Reduce shaping freedom to prevent reward hacking
            
            # Execution costs: LIGHT during discovery
            execution_cost_visibility_enabled=True,
            execution_cost_reward_scale=0.2,  # Light cost signal
            
            # Good loss cuts: Starting to enable
            good_loss_cut_enabled=True,
            good_loss_cut_bonus=0.03,
            good_loss_cut_efficiency_threshold=0.4,
            good_loss_cut_max_bonus=0.06,
            
            # Cost erosion: DISABLED during discovery
            cost_erosion_penalty_enabled=False,
            
            # TINY profit incentive - just a hint
            r_multiple_bonus_threshold=2.0,
            r_multiple_bonus_scale=0.01,
            r_multiple_bonus_cap=0.02,
            
            mae_efficiency_enabled=False,
            mae_efficiency_scale=0.0,
            mae_efficiency_threshold=99.0,
            
            time_efficiency_enabled=False,
            time_efficiency_scale=0.0,
            optimal_trade_bars=12,
            max_trade_bars_for_bonus=48,
            
            exit_quality_enabled=False,
            trailing_stop_bonus=0.0,
            agent_close_bonus=0.0,
            hard_stop_penalty=0.0,
            risk_liquidation_penalty=0.0,
            
            truncation_winner_discount=0.15,
            truncation_loser_extra_penalty=0.08,
            
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            # DD SHAPING: VERY LIGHT - just awareness
            dd_shaping_enabled=True,
            dd_threshold=0.15,
            dd_penalty_scale=0.2,
            dd_severity_exponent=1.0,
            dd_severity_cap=0.3,
            
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            # ANTI-CHURN: ENABLED with light penalty (between EXPLORER=25 and TREND_STUDENT=12)
            # FIX: Was disabled, now soft limit at 18 with light penalty
            anti_churn_enabled=True,
            daily_trade_soft_limit=18,
            churn_penalty_per_trade=0.018,
            
            hard_block_penalty=0.005,
            soft_block_penalty=0.002,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            # ACTIVITY CONSISTENCY: Slightly stronger than EXPLORER
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=8.0,  # Target ~12 trades/1500 steps
            activity_deviation_penalty_scale=0.4,  # Gradual increase
            activity_deviation_penalty_cap=12.0,  # Gradual increase
            min_trades_penalty=0.35,
            
            # EXPLORATION BONUS: REDUCED - PnL must be primary signal
            # FIX: Was 0.08 which dominated PnL signal causing reward hacking
            exploration_bonus=0.015,
            directional_accuracy_weight=0.7,
            min_reward=-50.0,  # FIX: Was -2.0, clipping destroyed gradients
            max_reward=50.0,   # FIX: Was 2.0
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=30,  # FIX: Reduced from 40 (soft limit is 18)
            max_trades_per_session=15,  # FIX: Reduced from 20
            max_consecutive_losses=15,
            loss_layer_stop=14,               # Block at 14 to prevent reaching 15 = no breach
            # Session budget (v5.5): Very permissive for experimentation
            session_loss_limit_pct=0.99,      # Effectively infinite
            session_consecutive_loss_limit=99,
            enforce_session_windows=False,
            enforce_no_new_trades_window=False,
            enforce_weekend_block=False,
            enforce_hard_close=False,
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.40,
            max_drawdown_limit=0.40,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.35,
            entry_quality_gate_enabled=True,  # FIX: Enable mild quality gate
            entry_quality_threshold=0.12,  # FIX: Light filter (lower than TREND_STUDENT)
            hard_stop_loss_eur=800.0,
            soft_stop_loss_eur=600.0,
            trailing_activation_eur=150.0,
            trailing_retrace_pct=0.45,
            time_decay_hours=18.0,
            risk_per_trade_pct=0.008,
            max_risk_per_trade_pct=0.015,
        ),
        competence=CompetenceThresholds(
            # DISCOVERY: Explore + light performance awareness
            min_episodes=150,
            min_timesteps=200_000,
            min_win_rate=0.25,                # Very low bar
            min_profit_factor=0.3,            # Can lose money
            max_avg_drawdown=0.35,            # Loose
            min_avg_pnl=-3000.0,
            min_avg_r_multiple=-0.5,
            min_entropy=0.50,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=0.50,
            max_pnl_std=30000.0,
            min_trade_count_avg=5.0,
            max_dd_breach_rate=0.50,
            max_consecutive_loss_rate=0.40,
            evaluation_window=50,
        ),
        max_steps_per_episode=1500,
        include_memory_features=False,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 0.35),
            min_trend_clarity=0.45,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=True,
            exclude_market_open_close=True,
            prefer_recent_data=False,
            recent_data_weight=1.0,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
            ],
            regime_sampling_weights={
                "trending_up": 0.50,
                "trending_down": 0.50,
            },
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.5,
            lr_warmup_steps=5_000,
            reward_blend_enabled=True,
            reward_blend_episodes=10,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=25,
        ),
        skill_requirements=SkillRequirements(
            required_skills={},
            min_confidence=0.0,
            require_all_skills=False,
            weighted_threshold=0.0,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.50,
            max_entropy=1.30,
            low_entropy_penalty_scale=0.18,
            high_entropy_penalty_scale=0.01,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.45,
            demotion_threshold=0.15,
            hard_floors={},
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=False,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=False,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=False,
        ),
        review_session=ReviewSessionConfig(
            enabled=False,
        ),
    )


def get_trend_student_config() -> CurriculumStageConfig:
    """
    PHASE 1 - FOUNDATION: Stage 2 (TREND_STUDENT)
    ==============================================
    First concept: Learn that trading WITH trend is better.
    
    Goal: Discover trend alignment improves outcomes.
    
    ENABLED: R-multiple bonus, light DD shaping, light anti-churn
    NEW: Directional accuracy weight increased
    ENTROPY: 0.40 minimum - still exploring
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.TREND_STUDENT,
        name="Trend Student",
        description="FOUNDATION Phase: Learn trend alignment. Trade with the trend.",
        execution=ExecutionDifficulty(
            base_spread_points=0.05,
            spread_mult_range=(0.95, 1.08),
            max_spread_points=0.12,
            slippage_points_sigma=0.01,
            slippage_mult_range=(0.9, 1.15),
            max_slippage_points=0.05,
            commission_per_lot=0.0,
            latency_bars=0,
            enable_randomization=True,
            spread_randomization_range=(0.95, 1.05),
            slippage_randomization_range=(0.95, 1.08),
            latency_randomization_range=(0, 0),
            volatility_scale_range=(0.95, 1.05),
            # DATA SPREAD: 25% - starting to feel real spreads
            use_data_spread=True,
            data_spread_scale=0.25,
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 1 FOUNDATION: Learn trend alignment
            # ============================================================================
            reward_scale=5.0,
            loss_multiplier=1.0,
            
            # PnL DOMINANCE (v6.0): Foundation phase - PnL starts to matter
            pnl_scale_factor=150.0,           # FIX: Was 120, exploration bonus ratio was 0.57x
            max_shaping_to_pnl_ratio=0.5,     # FIX: Tighten shaping cap to prevent reward hacking
            
            # Execution costs: Moderate visibility
            execution_cost_visibility_enabled=True,
            execution_cost_reward_scale=0.35,
            
            # Good loss cuts: Enabled
            good_loss_cut_enabled=True,
            good_loss_cut_bonus=0.07,  # D2 TUNING: Start building good loss-cut habit
            good_loss_cut_efficiency_threshold=0.35,
            good_loss_cut_max_bonus=0.12,  # D2 TUNING: Stronger max bonus
            
            # Cost erosion: Light
            cost_erosion_penalty_enabled=True,
            cost_erosion_threshold=0.6,       # Trigger at 60% cost ratio
            cost_erosion_penalty_scale=0.08,
            cost_erosion_penalty_cap=0.15,
            
            # R-multiple bonus - reward good trades
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.02,
            r_multiple_bonus_cap=0.04,
            
            mae_efficiency_enabled=False,
            mae_efficiency_scale=0.0,
            mae_efficiency_threshold=99.0,
            
            time_efficiency_enabled=False,
            time_efficiency_scale=0.0,
            optimal_trade_bars=12,
            max_trade_bars_for_bonus=36,
            
            exit_quality_enabled=True,        # ENABLED: Even beginners should learn exits matter
            trailing_stop_bonus=0.12,         # D2 TUNING: Stronger trailing incentive early
            agent_close_bonus=-0.05,          # D2 TUNING: Discourage cutting winners
            hard_stop_penalty=0.05,           # D2 TUNING: Slightly stronger
            risk_liquidation_penalty=0.05,    # D2 TUNING: Slightly stronger
            
            truncation_winner_discount=0.20,
            truncation_loser_extra_penalty=0.10,
            
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            # MARKET STRUCTURE: Light intro - just S/R awareness
            market_structure_enabled=True,
            sr_proximity_bonus=0.10,          # Light: reward good S/R entries
            sr_proximity_penalty=0.12,        # Light: punish bad S/R entries
            structure_alignment_bonus=0.0,    # Not yet
            bos_alignment_bonus=0.0,          # Not yet
            order_block_entry_bonus=0.0,      # Not yet
            
            # DD SHAPING: Light
            dd_shaping_enabled=True,
            dd_threshold=0.10,
            dd_penalty_scale=0.3,
            dd_severity_exponent=1.1,
            dd_severity_cap=0.5,
            
            # STREAKS: Enable early to teach loss avoidance (Jan 2026 FIX)
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.01,      # Mild win streak bonus
            loss_streak_penalty_per_loss=0.03,  # Mild loss streak penalty
            
            # ANTI-CHURN: Light - learning stage needs room to explore
            anti_churn_enabled=True,
            daily_trade_soft_limit=12,     # D2 TUNING: Earlier soft limit
            churn_penalty_per_trade=0.025, # D2 TUNING: Slightly stronger penalty
            
            hard_block_penalty=0.01,
            soft_block_penalty=0.005,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,

            # OBSERVATION PERIOD: Learn to watch before trading
            observation_period_required=True,
            min_bars_observation_before_entry=20,
            observation_completion_bonus=0.06,
            premature_entry_penalty=0.10,
            
            # ACTIVITY CONSISTENCY: ENABLED - maintain activity level
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=8.0,
            activity_deviation_penalty_scale=0.5,  # Moderate - still learning
            activity_deviation_penalty_cap=15.0,
            min_trades_penalty=0.3,
            
            # EXPLORATION: Reduced to keep PnL dominant - must be <= EXPERIMENTER (0.015)
            exploration_bonus=0.015,          # FIX: Was 0.025, now monotone with EXPERIMENTER
            directional_accuracy_weight=1.2,  # INCREASED - reward trend alignment
            min_reward=-100.0,  # FIX: Was -2.5
            max_reward=100.0,   # FIX: Was 2.5
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=25,
            max_trades_per_session=12,
            max_consecutive_losses=10,
            loss_layer_stop=9,                # Block at 9 to prevent reaching 10 = no breach
            # Session budget (v5.5): Starting to introduce session awareness
            session_loss_limit_pct=0.50,      # 50% max session loss
            session_consecutive_loss_limit=20,
            enforce_session_windows=False,
            enforce_no_new_trades_window=False,
            enforce_weekend_block=False,
            enforce_hard_close=False,
            observation_period_required=True,
            min_bars_observation_before_entry=20,
            min_bars_between_entries=2,        # Bar-based pacing (timeframe-agnostic)
            min_bars_after_loss=4,             # Bar-based cooldown after loss
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.25,
            max_drawdown_limit=0.30,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.25,
            entry_quality_gate_enabled=True,  # FIX: Enable for monotonicity
            entry_quality_threshold=0.20,  # FIX: Between EXPERIMENTER(0.12) and TIMING_STUDENT(0.40)
            hard_stop_loss_eur=450.0,         # REDUCED from 600 - limit max damage
            soft_stop_loss_eur=320.0,         # REDUCED from 400
            trailing_activation_eur=100.0,    # REDUCED from 120
            trailing_retrace_pct=0.40,
            time_decay_hours=12.0,
            risk_per_trade_pct=0.006,
            max_risk_per_trade_pct=0.012,
        ),
        competence=CompetenceThresholds(
            min_episodes=200,
            min_timesteps=300_000,
            min_win_rate=0.35,
            min_profit_factor=0.7,
            max_avg_drawdown=0.20,
            min_avg_pnl=-500.0,
            min_avg_r_multiple=-0.1,
            min_entropy=0.40,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=0.30,
            max_pnl_std=10000.0,
            min_trade_count_avg=4.0,
            max_dd_breach_rate=0.25,
            max_consecutive_loss_rate=0.25,
            evaluation_window=60,
        ),
        max_steps_per_episode=1800,
        include_memory_features=False,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 0.45),
            min_trend_clarity=0.35,           # Prefer clearer trends for learning
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=True,
            exclude_market_open_close=True,
            prefer_recent_data=False,
            recent_data_weight=1.0,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
            ],
            regime_sampling_weights={
                "trending_up": 0.50,
                "trending_down": 0.50,
            },
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.5,
            lr_warmup_steps=8_000,
            reward_blend_enabled=True,
            reward_blend_episodes=15,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=30,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.TREND_ALIGNMENT: 0.35,
            },
            min_confidence=0.4,
            require_all_skills=False,
            weighted_threshold=0.35,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.40,
            max_entropy=1.00,
            low_entropy_penalty_scale=0.15,
            high_entropy_penalty_scale=0.02,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.55,
            demotion_threshold=0.20,
            hard_floors={},
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=100,
            max_relaxation=0.12,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=80,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.80,
            recent_stages_weight=0.15,
            foundation_weight=0.05,
        ),
        review_session=ReviewSessionConfig(
            enabled=False,
        ),
    )


def get_session_student_config() -> CurriculumStageConfig:
    """
    PHASE 1 - FOUNDATION: Stage 3 (SESSION_STUDENT)
    ================================================
    Second concept: Learn that session timing matters.
    
    Goal: Discover that trading during good sessions improves outcomes.
    
    ENABLED: Previous + Exit quality (trailing stops)
    NEW: Session awareness in constraints
    ENTROPY: 0.35 minimum - still exploring
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.SESSION_STUDENT,
        name="Session Student",
        description="FOUNDATION Phase: Learn session awareness. When matters.",
        execution=ExecutionDifficulty(
            base_spread_points=0.08,
            spread_mult_range=(0.92, 1.12),
            max_spread_points=0.20,
            slippage_points_sigma=0.02,
            slippage_mult_range=(0.9, 1.2),
            max_slippage_points=0.08,
            commission_per_lot=0.0,
            latency_bars=0,
            enable_randomization=True,
            spread_randomization_range=(0.93, 1.08),
            slippage_randomization_range=(0.93, 1.10),
            latency_randomization_range=(0, 0),
            volatility_scale_range=(0.93, 1.08),
            # DATA SPREAD: 40% - moderate real spread exposure
            use_data_spread=True,
            data_spread_scale=0.40,
        ),
        rewards=RewardShaping(
            reward_scale=5.5,
            loss_multiplier=1.0,
            
            # PnL DOMINANCE (v6.0): SESSION_STUDENT - Ramping up PnL signal
            pnl_scale_factor=160.0,           # Growing PnL dominance
            max_shaping_to_pnl_ratio=0.5,     # FIX: Was 0.6, must be <= TREND_STUDENT (0.5)
            
            # Execution costs: Growing visibility
            execution_cost_visibility_enabled=True,
            execution_cost_reward_scale=0.4,
            
            # Good loss cuts: Enabled
            good_loss_cut_enabled=True,
            good_loss_cut_bonus=0.08,  # D2 TUNING: Stronger loss-cut habit
            good_loss_cut_efficiency_threshold=0.3,
            good_loss_cut_max_bonus=0.14,  # D2 TUNING: Stronger max bonus
            
            # Cost erosion: Moderate
            cost_erosion_penalty_enabled=True,
            cost_erosion_threshold=0.55,
            cost_erosion_penalty_scale=0.12,
            cost_erosion_penalty_cap=0.25,
            
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.03,
            r_multiple_bonus_cap=0.06,
            
            mae_efficiency_enabled=False,
            mae_efficiency_scale=0.0,
            mae_efficiency_threshold=99.0,
            
            time_efficiency_enabled=False,
            time_efficiency_scale=0.0,
            optimal_trade_bars=12,
            max_trade_bars_for_bonus=36,
            
            # EXIT QUALITY: NEW - Learn to hold winners
            # STRONGLY prefer trailing stops - this is where we teach patience!
            exit_quality_enabled=True,
            trailing_stop_bonus=0.20,    # D2 TUNING: Strong reward for letting winners run
            agent_close_bonus=-0.07,     # D2 TUNING: Stronger discouragement of cutting winners
            hard_stop_penalty=0.07,      # D2 TUNING: Meaningful penalty for hitting stop loss
            risk_liquidation_penalty=0.10,  # Strong penalty for risk liquidation
            
            truncation_winner_discount=0.20,
            truncation_loser_extra_penalty=0.10,
            
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            # SESSION TIMING: NEW - Learn trading hours
            session_timing_enabled=True,      # Teach agent to prefer prime hours
            off_hours_trade_penalty=0.10,     # Penalty for off-hours trades
            prime_hours_trade_bonus=0.03,     # Bonus for prime hours trades
            time_of_day_quality=TIME_OF_DAY_QUALITY,
            
            dd_shaping_enabled=True,
            dd_threshold=0.08,
            dd_penalty_scale=0.4,
            dd_severity_exponent=1.2,
            dd_severity_cap=0.6,
            
            # STREAKS: Progressive scaling (Jan 2026 FIX)
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.012,     # Slight increase from Stage 2
            loss_streak_penalty_per_loss=0.05,  # Moderate penalty for loss streaks
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=10,   # D2 TUNING: Earlier soft limit
            churn_penalty_per_trade=0.032,  # D2 TUNING: Slightly stronger penalty
            
            hard_block_penalty=0.015,
            soft_block_penalty=0.008,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            # ACTIVITY CONSISTENCY: ENABLED - maintain activity in sessions
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=8.0,
            activity_deviation_penalty_scale=0.6,  # Progressive increase
            activity_deviation_penalty_cap=18.0,
            min_trades_penalty=0.25,
            
            # FIX: Was 0.05 which caused exploration spike vs earlier stages
            # Must be <= TREND_STUDENT (0.015) for monotonicity
            exploration_bonus=0.015,
            directional_accuracy_weight=1.15,
            min_reward=-100.0,  # FIX: Was -2.8
            max_reward=100.0,   # FIX: Was 2.8
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=20,
            max_trades_per_session=10,
            max_consecutive_losses=8,
            loss_layer_stop=7,                # Block at 7 to prevent reaching 8 = no breach
            # Session budget (v5.5): KEY stage for session budget learning
            session_loss_limit_pct=0.20,      # 20% max session loss
            session_consecutive_loss_limit=10,
            enforce_session_windows=True,      # KEY: Session awareness
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=False,
            min_bars_between_entries=3,        # Bar-based pacing (timeframe-agnostic)
            min_bars_after_loss=5,             # Bar-based cooldown after loss
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.20,
            max_drawdown_limit=0.25,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.22,
            entry_quality_gate_enabled=True,  # FIX: Enable for monotonicity
            entry_quality_threshold=0.30,  # FIX: Between TREND_STUDENT(0.20) and TIMING_STUDENT(0.40)
            hard_stop_loss_eur=450.0,         # MONOTONICITY FIX: Must not increase from Stage 2 (was 500)
            soft_stop_loss_eur=350.0,
            trailing_activation_eur=100.0,
            trailing_retrace_pct=0.40,
            time_decay_hours=10.0,
            risk_per_trade_pct=0.006,
            max_risk_per_trade_pct=0.012,
        ),
        competence=CompetenceThresholds(
            min_episodes=250,
            min_timesteps=400_000,
            min_win_rate=0.38,
            min_profit_factor=0.80,
            max_avg_drawdown=0.18,
            min_avg_pnl=-300.0,
            min_avg_r_multiple=-0.05,
            min_entropy=0.35,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=0.28,
            max_pnl_std=9000.0,
            min_trade_count_avg=4.0,
            max_dd_breach_rate=0.22,
            max_consecutive_loss_rate=0.22,
            evaluation_window=70,
        ),
        max_steps_per_episode=1800,
        include_memory_features=False,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 0.55),
            min_trend_clarity=0.25,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=True,
            exclude_market_open_close=False,
            prefer_recent_data=False,
            recent_data_weight=1.0,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
            ],
            regime_sampling_weights={
                "trending_up": 0.40,
                "trending_down": 0.40,
                "ranging": 0.20,
            },
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.5,
            lr_warmup_steps=8_000,
            reward_blend_enabled=True,
            reward_blend_episodes=15,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=35,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.TREND_ALIGNMENT: 0.40,
                TradingSkill.EXIT_QUALITY: 0.30,
            },
            min_confidence=0.4,
            require_all_skills=False,
            weighted_threshold=0.38,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.35,
            max_entropy=0.90,
            low_entropy_penalty_scale=0.14,
            high_entropy_penalty_scale=0.02,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.55,
            demotion_threshold=0.22,
            hard_floors={},
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=100,
            max_relaxation=0.12,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=100,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.78,
            recent_stages_weight=0.17,
            foundation_weight=0.05,
        ),
        review_session=ReviewSessionConfig(
            enabled=False,
        ),
    )


def get_timing_student_config() -> CurriculumStageConfig:
    """
    PHASE 1 - FOUNDATION: Stage 4 (TIMING_STUDENT)
    ===============================================
    Third concept: Learn entry quality improves outcomes.
    
    Goal: Discover that better entries lead to better risk/reward.
    
    ENABLED: Previous + Entry quality integration
    ENTROPY: 0.30 minimum - strategy starting to form
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.TIMING_STUDENT,
        name="Timing Student",
        description="FOUNDATION Phase: Learn entry timing. Quality entries matter.",
        execution=ExecutionDifficulty(
            base_spread_points=0.10,
            spread_mult_range=(0.90, 1.15),
            max_spread_points=0.25,
            slippage_points_sigma=0.02,
            slippage_mult_range=(0.88, 1.25),
            max_slippage_points=0.10,
            commission_per_lot=0.0,
            latency_bars=0,
            enable_randomization=True,
            spread_randomization_range=(0.92, 1.10),
            slippage_randomization_range=(0.92, 1.12),
            latency_randomization_range=(0, 1),
            volatility_scale_range=(0.92, 1.10),
            # DATA SPREAD: 55% - getting serious about costs
            use_data_spread=True,
            data_spread_scale=0.55,
        ),
        rewards=RewardShaping(
            reward_scale=6.0,
            loss_multiplier=1.0,
            
            # ================================================================
            # PnL DOMINANCE (v6.0): TIMING_STUDENT - PnL must drive learning!
            # ================================================================
            # This is the stage where profitability decoupling was most severe.
            # With pnl_scale_factor=200, a €100 profit → 0.001 * 200 * 6.0 = 1.2
            # This is NOW dominant over shaping bonuses (0.1-0.2 range)
            pnl_scale_factor=200.0,           # FULL PnL dominance
            max_shaping_to_pnl_ratio=0.5,     # Strict: shaping <= 50% of |base_pnl|
            
            # Execution costs: FULLY visible - agent must feel costs
            execution_cost_visibility_enabled=True,
            execution_cost_reward_scale=0.5,  # Full cost signal
            
            # Good loss cuts: ENABLED - teach agent to control exits
            good_loss_cut_enabled=True,
            good_loss_cut_bonus=0.10,     # D2 AUDIT: Increased from 0.08 to encourage controlled loss cuts
            good_loss_cut_efficiency_threshold=0.3,
            good_loss_cut_max_bonus=0.15,
            
            # Cost erosion: ENABLED - penalize overtrading
            cost_erosion_penalty_enabled=True,
            cost_erosion_threshold=0.5,       # Trigger at 50% cost ratio
            cost_erosion_penalty_scale=0.15,
            cost_erosion_penalty_cap=0.30,
            
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.04,
            r_multiple_bonus_cap=0.08,
            
            mae_efficiency_enabled=False,
            mae_efficiency_scale=0.0,
            mae_efficiency_threshold=99.0,
            
            time_efficiency_enabled=False,
            time_efficiency_scale=0.0,
            optimal_trade_bars=10,
            max_trade_bars_for_bonus=32,
            
            exit_quality_enabled=True,
            # D2 AUDIT FIX: Stronger trailing incentive to shift exit mix
            trailing_stop_bonus=0.30,    # AUDIT: Increased from 0.22 to encourage trailing exits
            agent_close_bonus=-0.10,     # AUDIT: Increased penalty from -0.08 to discourage premature closes
            hard_stop_penalty=0.05,      # Keep: Punish bad entries that hit stop
            risk_liquidation_penalty=0.06,
            
            truncation_winner_discount=0.10,   # GPT FIX: Reduce truncation stress (was 0.22)
            truncation_loser_extra_penalty=0.05,  # GPT FIX: Reduce truncation stress (was 0.12)
            
            # ENTRY QUALITY: NEW
            entry_quality_integration=True,
            entry_quality_weight=0.10,

            # SETUP QUALITY: Introduce confluence awareness
            setup_quality_enabled=True,
            setup_quality_threshold=0.70,
            setup_quality_bonus_scale=0.15,
            hasty_entry_penalty=0.08,

            # ENTRY CERTAINTY: confidence-based rewards
            certainty_threshold=0.70,
            entry_certainty_bonus=ENTRY_CERTAINTY_BONUS,
            low_certainty_penalty=0.15,
            
            # SESSION TIMING: Continue from Stage 3
            session_timing_enabled=True,
            off_hours_trade_penalty=0.12,     # Slightly stronger
            prime_hours_trade_bonus=0.04,
            time_of_day_quality=TIME_OF_DAY_QUALITY,
            
            # MARKET STRUCTURE: REBALANCED - penalties were dominating (GPT FIX)
            market_structure_enabled=True,
            sr_proximity_bonus=0.18,          # Good S/R entry reward
            sr_proximity_penalty=0.15,        # GPT FIX: Reduced from 0.25 - was too harsh
            structure_alignment_bonus=0.12,   # Moderate structure alignment bonus
            bos_alignment_bonus=0.0,          # Not yet - too advanced for Stage 4
            order_block_entry_bonus=0.0,      # Not yet - too advanced for Stage 4
            
            # DIVERGENCE: REBALANCED - penalties were too harsh (GPT FIX)
            divergence_awareness_enabled=True,
            divergence_contra_penalty=0.12,   # GPT FIX: Reduced from 0.20
            divergence_aligned_bonus=0.12,    # Reward respecting divergence
            overbought_long_penalty=0.10,     # GPT FIX: Reduced from 0.18
            oversold_short_penalty=0.10,      # GPT FIX: Reduced from 0.18
            
            # REGIME: Not yet
            regime_awareness_enabled=False,
            
            dd_shaping_enabled=True,
            dd_threshold=0.06,
            dd_penalty_scale=0.5,
            dd_severity_exponent=1.3,
            dd_severity_cap=0.8,
            
            # STREAKS: Stronger penalty (Jan 2026 FIX)
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.015,     # Progressive increase
            loss_streak_penalty_per_loss=0.08,  # Stronger penalty for loss streaks
            
            anti_churn_enabled=True,
            # D2 AUDIT FIX: Tighter churn controls to reduce overtrading
            daily_trade_soft_limit=10,   # MONOTONICITY FIX: Must not increase from Stage 3 (was 12)
            churn_penalty_per_trade=0.035, # AUDIT: Increased from 0.03 for stronger discouragement
            
            hard_block_penalty=0.02,
            soft_block_penalty=0.01,
            
            # PER-STEP SHAPING: Enable patience teaching
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0,     # No holding cost yet
            patience_shaping_enabled=True,
            patience_bonus_per_bar=0.001, # Small bonus for waiting when setups are weak
            patience_quality_threshold=0.35,
            dynamic_patience_enabled=True,
            patience_bonus_base=0.001,
            patience_bonus_multiplier=PATIENCE_BONUS_MULTIPLIER,
            per_step_min=-0.03,
            per_step_max=0.03,
            
            # ACTIVITY CONSISTENCY: ENABLED - maintain minimum activity
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=6.0,
            activity_deviation_penalty_scale=0.7,  # Progressive increase
            activity_deviation_penalty_cap=20.0,
            min_trades_penalty=0.2,
            
            exploration_bonus=0.015,  # FIX: Was 0.02, must be <= Stage 3 for monotonicity
            directional_accuracy_weight=1.1,
            min_reward=-100.0,  # FIX: Was -3.0
            max_reward=100.0,   # FIX: Was 3.0
        ),
        constraints=TradingConstraints(
            max_positions=1,
            # D2 AUDIT FIX: Tighter trade limits to reduce churn
            max_trades_per_day=14,        # AUDIT: Reduced from 18
            max_trades_per_session=7,     # AUDIT: Reduced from 9
            max_consecutive_losses=7,
            loss_layer_stop=6,                # Block at 6 to prevent reaching 7 = no breach
            # Session budget (v5.5): Tightening further
            session_loss_limit_pct=0.15,      # 15% max session loss
            session_consecutive_loss_limit=8,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            # Bar-based pacing (timeframe-agnostic)
            min_bars_between_entries=4,
            min_bars_after_loss=6,
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.15,
            max_drawdown_limit=0.20,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.18,
            entry_quality_gate_enabled=True,
            # D2 AUDIT FIX: Stricter entry quality gate
            entry_quality_threshold=0.35,   # Progressive ladder: 0.35 -> 0.42 -> 0.47 -> 0.50
            min_setup_quality_for_entry=0.55,
            hard_stop_loss_eur=400.0,
            soft_stop_loss_eur=280.0,
            # D2 AUDIT FIX: More sensitive trailing to capture more winners
            trailing_activation_eur=80.0,   # AUDIT: Reduced from 90 to trigger trailing sooner
            trailing_retrace_pct=0.30,      # AUDIT: Reduced from 0.38 for tighter trailing
            # D2 AUDIT FIX: Shorter time decay to reduce truncation episodes
            time_decay_hours=6.0,           # AUDIT: Reduced from 8.0
            # D2 AUDIT FIX: Slightly lower risk to reduce loss magnitude
            risk_per_trade_pct=0.004,       # AUDIT: Reduced from 0.005
            max_risk_per_trade_pct=0.008,   # AUDIT: Reduced from 0.01
        ),
        competence=CompetenceThresholds(
            min_episodes=300,
            min_timesteps=500_000,
            min_win_rate=0.40,
            min_profit_factor=0.85,   # GPT FIX: Relaxed from 0.90 to unlock progression
            max_avg_drawdown=0.15,
            min_avg_pnl=-200.0,       # GPT FIX: Relaxed from -100 (intermediate target)
            min_avg_r_multiple=-0.05, # GPT FIX: Allow slight negative R while learning
            min_entropy=0.30,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=0.25,
            max_pnl_std=8000.0,
            min_trade_count_avg=4.0,
            min_avg_bars_between_trades=3.0,
            max_dd_breach_rate=0.18,
            max_consecutive_loss_rate=0.20,  # FIX: Was 0.30 which broke monotonicity from Stage 3 (0.22)
            evaluation_window=80,
        ),
        max_steps_per_episode=2000,
        include_memory_features=True,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 0.65),
            min_trend_clarity=0.15,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=False,
            exclude_market_open_close=False,
            prefer_recent_data=False,
            recent_data_weight=1.0,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
            ],
            regime_sampling_weights={
                "trending_up": 0.35,
                "trending_down": 0.35,
                "ranging": 0.30,
            },
            include_setup_maturity_metrics=True,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.45,
            lr_warmup_steps=10_000,
            reward_blend_enabled=True,
            reward_blend_episodes=20,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=40,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.TREND_ALIGNMENT: 0.45,
                TradingSkill.EXIT_QUALITY: 0.40,
                TradingSkill.ENTRY_TIMING: 0.35,
            },
            min_confidence=0.45,
            require_all_skills=False,
            weighted_threshold=0.42,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.30,
            max_entropy=0.80,
            low_entropy_penalty_scale=0.12,
            high_entropy_penalty_scale=0.03,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.58,
            demotion_threshold=0.25,
            hard_floors={
                "max_drawdown": 0.15,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=90,
            max_relaxation=0.10,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=120,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.75,
            recent_stages_weight=0.18,
            foundation_weight=0.07,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=200,
            review_duration=40,
            review_depth=2,
        ),
    )


def get_integrator_config() -> CurriculumStageConfig:
    """
    PHASE 2 - DEVELOPMENT: Stage 5 (INTEGRATOR)
    ============================================
    Combine trend + session + entry into coherent approach.
    
    Goal: Integrate learned concepts. Trade in trend, in good session, with good entry.
    
    ENABLED: All previous + MAE efficiency
    ENTROPY: 0.22 minimum - strategy forming
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.INTEGRATOR,
        name="Integrator",
        description="DEVELOPMENT Phase: Combine skills. Trend + Session + Entry.",
        execution=ExecutionDifficulty(
            base_spread_points=0.12,
            spread_mult_range=(0.88, 1.25),
            max_spread_points=0.35,
            slippage_points_sigma=0.03,
            slippage_mult_range=(0.85, 1.35),
            max_slippage_points=0.15,
            commission_per_lot=0.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.90, 1.15),
            slippage_randomization_range=(0.88, 1.20),
            latency_randomization_range=(0, 1),
            volatility_scale_range=(0.90, 1.12),
            # DATA SPREAD: 70% - near-realistic costs
            use_data_spread=True,
            data_spread_scale=0.70,
        ),
        rewards=RewardShaping(
            reward_scale=7.0,
            loss_multiplier=1.0,
            
            r_multiple_bonus_threshold=1.4,
            r_multiple_bonus_scale=0.06,
            r_multiple_bonus_cap=0.10,
            
            # MAE EFFICIENCY: NEW
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.03,
            mae_efficiency_threshold=1.8,
            
            time_efficiency_enabled=False,
            time_efficiency_scale=0.0,
            optimal_trade_bars=10,
            max_trade_bars_for_bonus=30,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.32,    # D2 FIX: Must be >= TIMING_STUDENT (0.30)
            agent_close_bonus=-0.10,     # GPT FIX: Stronger penalty
            hard_stop_penalty=0.06,
            risk_liquidation_penalty=0.08,
            
            truncation_winner_discount=0.12,   # GPT FIX: Reduced from 0.22
            truncation_loser_extra_penalty=0.06,  # GPT FIX: Reduced from 0.12
            
            entry_quality_integration=True,
            entry_quality_weight=0.15,

            # SETUP QUALITY: stronger confluence awareness
            setup_quality_enabled=True,
            setup_quality_threshold=0.70,
            setup_quality_bonus_scale=0.15,
            hasty_entry_penalty=0.08,

            # ENTRY CERTAINTY: confidence-based rewards
            certainty_threshold=0.70,
            entry_certainty_bonus=ENTRY_CERTAINTY_BONUS,
            low_certainty_penalty=0.15,
            
            # SESSION TIMING: Strengthened
            session_timing_enabled=True,
            off_hours_trade_penalty=0.15,
            prime_hours_trade_bonus=0.05,
            time_of_day_quality=TIME_OF_DAY_QUALITY,
            
            # MARKET STRUCTURE: REBALANCED (GPT FIX)
            market_structure_enabled=True,
            sr_proximity_bonus=0.22,          # Good S/R entry reward
            sr_proximity_penalty=0.18,        # GPT FIX: Reduced from 0.30
            structure_alignment_bonus=0.15,   # Moderate
            bos_alignment_bonus=0.12,         # BOS confirmation bonus
            order_block_entry_bonus=0.10,     # Order block awareness
            
            # DIVERGENCE: REBALANCED (GPT FIX)
            divergence_awareness_enabled=True,
            divergence_contra_penalty=0.15,   # GPT FIX: Reduced from 0.25
            divergence_aligned_bonus=0.14,    # Reward divergence awareness
            overbought_long_penalty=0.12,     # GPT FIX: Reduced from 0.22
            oversold_short_penalty=0.12,      # GPT FIX: Reduced from 0.22
            
            # REGIME: Introduce lightly
            regime_awareness_enabled=True,
            risk_off_aggressive_penalty=0.04,
            high_vol_size_penalty=0.03,
            
            dd_shaping_enabled=True,
            dd_threshold=0.05,
            dd_penalty_scale=0.6,
            dd_severity_exponent=1.35,
            dd_severity_cap=0.9,
            
            # STREAKS: Near-final penalty level (Jan 2026 FIX)
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.018,     # Progressive increase
            loss_streak_penalty_per_loss=0.10,  # Strong penalty - approaching Stage 6 level
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=10,   # MONOTONICITY FIX: Must not increase from Stage 3-4 (was 12)
            churn_penalty_per_trade=0.04, # GPT FIX: Reduced from 0.05
            
            hard_block_penalty=0.025,
            soft_block_penalty=0.012,
            
            # PER-STEP SHAPING: Strengthen patience teaching
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0002,  # Light holding cost - encourage selectivity
            patience_shaping_enabled=True,
            patience_bonus_per_bar=0.0012, # Slightly stronger patience bonus
            patience_quality_threshold=0.38,
            dynamic_patience_enabled=True,
            patience_bonus_base=0.0012,
            patience_bonus_multiplier=PATIENCE_BONUS_MULTIPLIER,
            per_step_min=-0.04,
            per_step_max=0.04,

            # STRATEGIC PATIENCE: reward skipping setups before entering
            strategic_patience_enabled=True,
            setup_rejection_bonus=0.03,
            max_setup_rejections_for_bonus=5,

            # DELIBERATION TIME: reward thinking before acting
            deliberation_time_tracking=True,
            min_deliberation_bars=2,
            optimal_deliberation_range=(3, 8),
            too_fast_penalty=0.08,
            deliberation_quality_bonus=0.05,
            
            # ACTIVITY CONSISTENCY: Maintain discipline
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=5.0,  # Smooth decline into selectivity
            activity_deviation_penalty_scale=0.8,  # Strong
            activity_deviation_penalty_cap=20.0,
            min_trades_penalty=0.2,
            
            exploration_bonus=0.015,
            directional_accuracy_weight=1.05,
            min_reward=-200.0,  # FIX: Was -3.5
            max_reward=200.0,   # FIX: Was 3.5
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=14,
            max_trades_per_session=7,
            max_consecutive_losses=6,
            loss_layer_stop=5,                # Block at 5 to prevent reaching 6 = no breach
            # Session budget (v5.5): Prop firm discipline emerging
            session_loss_limit_pct=0.10,      # 10% max session loss
            session_consecutive_loss_limit=6,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_bars_between_entries=4,
            min_bars_after_loss=6,
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.10,
            max_drawdown_limit=0.15,
            daily_dd_safety_buffer=0.005,
            max_dd_safety_buffer=0.01,
            emergency_close_threshold=0.13,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.42,     # Progressive ladder: 0.35 -> 0.42 -> 0.47 -> 0.50
            min_setup_quality_for_entry=0.60,
            hard_stop_loss_eur=350.0,
            soft_stop_loss_eur=230.0,
            trailing_activation_eur=80.0,
            trailing_retrace_pct=0.35,
            time_decay_hours=7.0,
            risk_per_trade_pct=0.004,
            max_risk_per_trade_pct=0.008,
        ),
        competence=CompetenceThresholds(
            min_episodes=350,
            min_timesteps=700_000,
            min_win_rate=0.45,
            min_profit_factor=1.0,
            max_avg_drawdown=0.12,
            min_avg_pnl=0.0,
            min_avg_r_multiple=0.05,
            min_entropy=0.22,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=0.22,
            max_pnl_std=7000.0,
            min_trade_count_avg=4.0,
            min_avg_bars_between_trades=4.0,
            min_setup_skipped_per_episode=2.0,
            min_entry_certainty_avg=0.55,
            min_avg_setup_quality=0.60,
            max_fomo_trade_rate=0.20,
            max_revenge_trade_rate=0.20,
            max_dd_breach_rate=0.15,
            max_consecutive_loss_rate=0.18,
            evaluation_window=90,
        ),
        max_steps_per_episode=2200,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 0.75),
            min_trend_clarity=0.0,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=False,
            exclude_market_open_close=False,
            prefer_recent_data=False,
            recent_data_weight=1.0,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
            ],
            regime_sampling_weights={
                "trending_up": 0.30,
                "trending_down": 0.30,
                "ranging": 0.20,
                "high_volatility": 0.20,
            },
            include_setup_maturity_metrics=True,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.4,
            lr_warmup_steps=12_000,
            reward_blend_enabled=True,
            reward_blend_episodes=25,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=50,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.TREND_ALIGNMENT: 0.50,
                TradingSkill.EXIT_QUALITY: 0.45,
                TradingSkill.ENTRY_TIMING: 0.42,
                TradingSkill.DRAWDOWN_CONTROL: 0.50,
                TradingSkill.PATIENCE: 0.35,
                TradingSkill.SELECTIVITY: 0.30,
                TradingSkill.CERTAINTY: 0.40,
                TradingSkill.SETUP_QUALITY: 0.35,
            },
            min_confidence=0.5,
            require_all_skills=False,
            weighted_threshold=0.48,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.22,
            max_entropy=0.60,
            low_entropy_penalty_scale=0.10,
            high_entropy_penalty_scale=0.04,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.62,
            demotion_threshold=0.28,
            hard_floors={
                "max_drawdown": 0.12,
                "profit_factor": 0.95,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=80,
            max_relaxation=0.08,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=150,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.72,
            recent_stages_weight=0.20,
            foundation_weight=0.08,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=250,
            review_duration=50,
            review_depth=3,
        ),
        # Validation before promotion (Stage 5+)
        validation=ValidationConfig(
            enabled=True,
            validation_episodes=50,
            min_performance_ratio=0.75,  # Lighter requirement for early validation
            max_performance_drop=0.25,
            required_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
            ],
            min_episodes_per_regime=10,
        ),
    )


def get_risk_manager_config() -> CurriculumStageConfig:
    """
    PHASE 2 - DEVELOPMENT: Stage 6 (RISK_MANAGER)
    ==============================================
    Add position sizing and risk control.
    
    Goal: Learn capital preservation. Manage drawdowns actively.
    
    ENABLED: All previous + Streak modifiers + Strong DD shaping
    ENTROPY: 0.15 minimum - strategy solidifying
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.RISK_MANAGER,
        name="Risk Manager",
        description="DEVELOPMENT Phase: Add risk control. Capital preservation.",
        execution=ExecutionDifficulty(
            base_spread_points=0.15,
            spread_mult_range=(0.85, 1.35),
            max_spread_points=0.50,
            slippage_points_sigma=0.04,
            slippage_mult_range=(0.80, 1.45),
            max_slippage_points=0.20,
            commission_per_lot=1.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.88, 1.20),
            slippage_randomization_range=(0.85, 1.25),
            latency_randomization_range=(0, 2),
            volatility_scale_range=(0.88, 1.15),
            # DATA SPREAD: 85% - near-full realistic costs
            use_data_spread=True,
            data_spread_scale=0.85,
        ),
        rewards=RewardShaping(
            reward_scale=8.0,
            loss_multiplier=1.0,
            
            r_multiple_bonus_threshold=1.4,
            r_multiple_bonus_scale=0.08,
            r_multiple_bonus_cap=0.12,
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.05,
            mae_efficiency_threshold=1.6,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.02,
            optimal_trade_bars=9,
            max_trade_bars_for_bonus=28,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.34,    # D2 FIX: Must be >= INTEGRATOR (0.32)
            agent_close_bonus=-0.10,     # Discourage cutting winners
            hard_stop_penalty=0.07,
            risk_liquidation_penalty=0.10,
            
            truncation_winner_discount=0.12,   # GPT FIX: Reduced from 0.25
            truncation_loser_extra_penalty=0.08,  # GPT FIX: Reduced from 0.15
            
            entry_quality_integration=True,
            entry_quality_weight=0.18,

            # SETUP QUALITY: stronger confluence awareness
            setup_quality_enabled=True,
            setup_quality_threshold=0.70,
            setup_quality_bonus_scale=0.15,
            hasty_entry_penalty=0.08,

            # ENTRY CERTAINTY: confidence-based rewards
            certainty_threshold=0.70,
            entry_certainty_bonus=ENTRY_CERTAINTY_BONUS,
            low_certainty_penalty=0.15,
            
            # SESSION TIMING: Full strength
            session_timing_enabled=True,
            off_hours_trade_penalty=0.18,
            prime_hours_trade_bonus=0.06,
            time_of_day_quality=TIME_OF_DAY_QUALITY,
            
            # MARKET STRUCTURE: REBALANCED (GPT FIX)
            market_structure_enabled=True,
            sr_proximity_bonus=0.25,          # Good S/R entry reward
            sr_proximity_penalty=0.20,        # GPT FIX: Reduced from 0.35
            structure_alignment_bonus=0.18,   # Structure alignment
            bos_alignment_bonus=0.15,         # BOS confirmation bonus
            order_block_entry_bonus=0.12,     # Order block awareness
            
            # DIVERGENCE: REBALANCED (GPT FIX)
            divergence_awareness_enabled=True,
            divergence_contra_penalty=0.18,   # GPT FIX: Reduced from 0.30
            divergence_aligned_bonus=0.16,    # Reward divergence awareness
            overbought_long_penalty=0.14,     # GPT FIX: Reduced from 0.25
            oversold_short_penalty=0.14,      # GPT FIX: Reduced from 0.25
            
            # REGIME: Full strength
            regime_awareness_enabled=True,
            risk_off_aggressive_penalty=0.08,
            high_vol_size_penalty=0.06,
            
            # DD SHAPING: Strong
            dd_shaping_enabled=True,
            dd_threshold=0.04,
            dd_penalty_scale=0.8,
            dd_severity_exponent=1.45,
            dd_severity_cap=1.2,
            
            # STREAKS: NEW
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.02,
            loss_streak_penalty_per_loss=0.10,  # BOOSTED: Was 0.05 - harsher penalty for loss streaks
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=10,   # GPT FIX: Increased from 6
            churn_penalty_per_trade=0.05, # GPT FIX: Reduced from 0.06
            
            hard_block_penalty=0.03,
            soft_block_penalty=0.015,
            
            # PER-STEP SHAPING: Full patience teaching
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0003,  # Light holding cost
            patience_shaping_enabled=True,
            patience_bonus_per_bar=0.002,  # BOOSTED: Was 0.0015
            patience_quality_threshold=0.40,
            dynamic_patience_enabled=True,
            patience_bonus_base=0.002,
            patience_bonus_multiplier=PATIENCE_BONUS_MULTIPLIER,
            per_step_min=-0.06,  # Monotonic: -0.04 → -0.06 → -0.08 → -0.10 → -0.12
            per_step_max=0.05,   # Monotonic: 0.04 → 0.05 → 0.05 → 0.05 → 0.05

            # DELIBERATION TIME: reward thinking before acting
            deliberation_time_tracking=True,
            min_deliberation_bars=2,
            optimal_deliberation_range=(3, 8),
            too_fast_penalty=0.08,
            deliberation_quality_bonus=0.05,

            # WIN-RATE PRESERVATION
            win_rate_preservation_enabled=True,
            current_win_rate_threshold=0.45,
            selectivity_bonus=0.08,
            win_rate_decay_penalty=0.10,
            
            # LOSS STREAK CAUTION: Start teaching this at RISK_MANAGER
            loss_streak_caution_enabled=True,
            loss_streak_caution_base=0.025,  # Lighter than STRATEGIST - introduce gradually
            loss_streak_caution_cap=0.20,
            
            # ACTIVITY CONSISTENCY: Quality over quantity
            # Target moderate activity matching Stage 5 (was 31.0 which contradicts anti-churn tightening)
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=4.0,  # Progressive selectivity
            activity_deviation_penalty_scale=0.8,  # STRONG: match PnL signal
            activity_deviation_penalty_cap=12.0,  # Match PnL scale
            min_trades_penalty=0.15,
            
            exploration_bonus=0.01,
            directional_accuracy_weight=1.0,
            min_reward=-200.0,  # FIX: Was -4.0
            max_reward=200.0,   # FIX: Was 4.0
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=12,
            max_trades_per_session=6,
            max_consecutive_losses=5,
            loss_layer_stop=4,                # Block at 4 to prevent reaching 5 = no breach
            # Session budget (v5.5): Risk-focused tightening
            session_loss_limit_pct=0.08,      # 8% max session loss
            session_consecutive_loss_limit=5,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_bars_between_entries=4,
            min_bars_after_loss=6,
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.06,
            max_drawdown_limit=0.10,
            daily_dd_safety_buffer=0.006,
            max_dd_safety_buffer=0.01,
            emergency_close_threshold=0.09,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.47,
            min_setup_quality_for_entry=0.65,
            hard_stop_loss_eur=300.0,
            soft_stop_loss_eur=200.0,
            trailing_activation_eur=70.0,
            trailing_retrace_pct=0.32,
            time_decay_hours=6.0,
            risk_per_trade_pct=0.0035,
            max_risk_per_trade_pct=0.007,
        ),
        competence=CompetenceThresholds(
            min_episodes=400,
            min_timesteps=900_000,
            min_win_rate=0.48,
            min_profit_factor=1.10,
            max_avg_drawdown=0.08,
            min_avg_pnl=50.0,
            min_avg_r_multiple=0.08,
            min_entropy=0.15,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=0.18,
            max_pnl_std=6000.0,
            min_trade_count_avg=4.0,
            min_avg_bars_between_trades=5.0,
            min_setup_skipped_per_episode=2.0,
            min_entry_certainty_avg=0.60,
            min_avg_setup_quality=0.65,
            max_fomo_trade_rate=0.15,
            max_revenge_trade_rate=0.15,
            max_dd_breach_rate=0.10,
            max_consecutive_loss_rate=0.16,  # FIX: Was 0.25 which broke monotonicity from Stage 5 (0.18)
            evaluation_window=100,
        ),
        max_steps_per_episode=2400,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 0.85),
            min_trend_clarity=0.0,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=False,
            exclude_market_open_close=False,
            prefer_recent_data=False,
            recent_data_weight=1.0,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
            ],
            regime_sampling_weights={
                "trending_up": 0.28,
                "trending_down": 0.28,
                "ranging": 0.22,
                "high_volatility": 0.22,
            },
            include_setup_maturity_metrics=True,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.35,
            lr_warmup_steps=15_000,
            reward_blend_enabled=True,
            reward_blend_episodes=30,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=60,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.TREND_ALIGNMENT: 0.55,
                TradingSkill.EXIT_QUALITY: 0.50,
                TradingSkill.ENTRY_TIMING: 0.48,
                TradingSkill.DRAWDOWN_CONTROL: 0.60,
                TradingSkill.PATIENCE: 0.40,  # FIX: Relaxed from 0.55 - 3 trades/day is acceptable for active gold trading
                TradingSkill.SELECTIVITY: 0.35,
                TradingSkill.CERTAINTY: 0.45,
                TradingSkill.DISCIPLINE: 0.40,
                TradingSkill.SETUP_QUALITY: 0.40,
            },
            min_confidence=0.55,
            require_all_skills=False,
            weighted_threshold=0.52,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.15,
            max_entropy=0.50,
            low_entropy_penalty_scale=0.08,
            high_entropy_penalty_scale=0.05,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.68,
            demotion_threshold=0.32,
            hard_floors={
                "max_drawdown": 0.08,
                "profit_factor": 1.05,
                "dd_breach_rate": 0.10,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=75,
            max_relaxation=0.06,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=180,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.70,
            recent_stages_weight=0.22,
            foundation_weight=0.08,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=300,
            review_duration=60,
            review_depth=4,
        ),
        # Validation before promotion (Stage 6)
        validation=ValidationConfig(
            enabled=True,
            validation_episodes=70,
            min_performance_ratio=0.78,
            max_performance_drop=0.22,
            required_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
            ],
            min_episodes_per_regime=12,
        ),
    )


def get_strategist_config() -> CurriculumStageConfig:
    """
    PHASE 2 - DEVELOPMENT: Stage 7 (STRATEGIST)
    ============================================
    Combine everything into coherent strategy.
    
    Goal: Demonstrate consistent execution. Strategy integration.
    
    ENABLED: All features, full integration
    ENTROPY: 0.12 minimum - consistent behavior expected
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.STRATEGIST,
        name="Strategist",
        description="DEVELOPMENT Phase: Strategy integration. Consistent execution.",
        execution=ExecutionDifficulty(
            base_spread_points=0.18,
            spread_mult_range=(0.80, 1.45),
            max_spread_points=0.60,
            slippage_points_sigma=0.05,
            slippage_mult_range=(0.75, 1.55),
            max_slippage_points=0.25,
            commission_per_lot=1.5,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.85, 1.25),
            slippage_randomization_range=(0.80, 1.35),
            latency_randomization_range=(0, 2),
            volatility_scale_range=(0.85, 1.18),
            # DATA SPREAD: 100% - FULL realistic costs (real FTMO spreads)
            use_data_spread=True,
            data_spread_scale=1.0,
        ),
        rewards=RewardShaping(
            reward_scale=9.0,
            loss_multiplier=1.05,
            
            r_multiple_bonus_threshold=1.3,
            r_multiple_bonus_scale=0.10,
            r_multiple_bonus_cap=0.15,
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.06,
            mae_efficiency_threshold=1.5,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.025,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=25,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.36,    # D2 FIX: Must be >= RISK_MANAGER (0.34)
            agent_close_bonus=-0.10,     # Discourage cutting winners
            hard_stop_penalty=0.08,
            risk_liquidation_penalty=0.12,
            
            truncation_winner_discount=0.12,   # GPT FIX: Reduced from 0.22
            truncation_loser_extra_penalty=0.10,  # GPT FIX: Reduced from 0.18
            
            entry_quality_integration=True,
            entry_quality_weight=0.20,

            # SETUP QUALITY: strong confluence awareness
            setup_quality_enabled=True,
            setup_quality_threshold=0.70,
            setup_quality_bonus_scale=0.15,
            hasty_entry_penalty=0.08,

            # ENTRY CERTAINTY: confidence-based rewards
            certainty_threshold=0.70,
            entry_certainty_bonus=ENTRY_CERTAINTY_BONUS,
            low_certainty_penalty=0.15,
            
            # SESSION TIMING: Full strength
            session_timing_enabled=True,
            off_hours_trade_penalty=0.20,
            prime_hours_trade_bonus=0.08,
            time_of_day_quality=TIME_OF_DAY_QUALITY,
            
            # MARKET STRUCTURE: REBALANCED (GPT FIX)
            market_structure_enabled=True,
            sr_proximity_bonus=0.28,          # Good S/R entry reward
            sr_proximity_penalty=0.22,        # GPT FIX: Reduced from 0.38
            structure_alignment_bonus=0.20,   # Structure alignment
            bos_alignment_bonus=0.16,         # BOS confirmation bonus
            order_block_entry_bonus=0.14,     # Order block awareness
            
            # DIVERGENCE: REBALANCED (GPT FIX)
            divergence_awareness_enabled=True,
            divergence_contra_penalty=0.20,   # GPT FIX: Reduced from 0.35
            divergence_aligned_bonus=0.18,    # Reward divergence awareness
            overbought_long_penalty=0.16,     # GPT FIX: Reduced from 0.28
            oversold_short_penalty=0.16,      # GPT FIX: Reduced from 0.28
            
            # REGIME: Full strength
            regime_awareness_enabled=True,
            risk_off_aggressive_penalty=0.10,
            high_vol_size_penalty=0.08,
            
            dd_shaping_enabled=True,
            dd_threshold=0.035,
            dd_penalty_scale=0.9,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.3,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.025,
            loss_streak_penalty_per_loss=0.35,  # BOOSTED: Was 0.20 - need stronger signal to stop loss streaks
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=8,
            churn_penalty_per_trade=0.06,
            
            hard_block_penalty=0.035,
            soft_block_penalty=0.018,
            
            # PER-STEP SHAPING: Strong patience discipline
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0004,
            patience_shaping_enabled=True,
            patience_bonus_per_bar=0.003,  # BOOSTED: Was 0.0018 - stronger reward for waiting
            patience_quality_threshold=0.42,
            dynamic_patience_enabled=True,
            patience_bonus_base=0.003,
            patience_bonus_multiplier=PATIENCE_BONUS_MULTIPLIER,
            per_step_min=-0.08,  # Monotonic: -0.06 → -0.08 → -0.10 → -0.12
            per_step_max=0.05,   # Monotonic: 0.04 → 0.05 → 0.05 → 0.05

            # DELIBERATION TIME: reward thinking before acting
            deliberation_time_tracking=True,
            min_deliberation_bars=2,
            optimal_deliberation_range=(3, 8),
            too_fast_penalty=0.08,
            deliberation_quality_bonus=0.05,

            # WIN-RATE PRESERVATION
            win_rate_preservation_enabled=True,
            current_win_rate_threshold=0.45,
            selectivity_bonus=0.08,
            win_rate_decay_penalty=0.10,

            # COMPOUNDING SUCCESS
            compounding_success_enabled=True,
            consecutive_quality_trades_bonus=[0.0, 0.02, 0.05, 0.09, 0.14],
            quality_trade_r_multiple=1.0,
            quality_trade_entry_quality=0.6,
            quality_trade_exit_type="trailing_stop",
            
            # LOSS STREAK CAUTION: Penalize entries while tilted
            loss_streak_caution_enabled=True,
            loss_streak_caution_base=0.04,   # Escalating penalty
            loss_streak_caution_cap=0.08,    # Capped to fit bounds
            
            # ACTIVITY CONSISTENCY: Quality focus
            # Target moderate activity that doesn't spike vs RISK_MANAGER (6.0)
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=3.5,  # Progressive selectivity
            activity_deviation_penalty_scale=1.0,  # STRONGER: strategic discipline
            activity_deviation_penalty_cap=12.0,  # Match PnL scale
            min_trades_penalty=0.1,
            
            exploration_bonus=0.005,
            directional_accuracy_weight=1.0,
            min_reward=-200.0,  # FIX: Was -4.5
            max_reward=200.0,   # FIX: Was 4.5
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=10,
            max_trades_per_session=5,
            max_consecutive_losses=5,  # MONOTONIC: Same as RISK_MANAGER, only tighten rate
            loss_layer_stop=4,                # Block at 4 to prevent reaching 5 = no breach
            # Session budget (v5.5): Near-prop-firm discipline
            session_loss_limit_pct=0.05,      # 5% max session loss
            session_consecutive_loss_limit=4,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_bars_between_entries=4,
            min_bars_after_loss=10,
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.055,
            max_drawdown_limit=0.095,
            daily_dd_safety_buffer=0.007,
            max_dd_safety_buffer=0.012,
            emergency_close_threshold=0.085,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.50,
            min_setup_quality_for_entry=0.70,
            hard_stop_loss_eur=280.0,
            soft_stop_loss_eur=180.0,
            trailing_activation_eur=75.0,
            trailing_retrace_pct=0.30,
            time_decay_hours=5.5,
            risk_per_trade_pct=0.0032,
            max_risk_per_trade_pct=0.0065,
        ),
        competence=CompetenceThresholds(
            min_episodes=500,
            min_timesteps=1_100_000,
            min_win_rate=0.50,
            min_profit_factor=1.18,
            max_avg_drawdown=0.07,
            min_avg_pnl=80.0,
            min_avg_r_multiple=0.10,
            min_entropy=0.12,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=0.16,
            max_pnl_std=5500.0,
            min_trade_count_avg=4.5,
            min_avg_bars_between_trades=6.0,
            min_setup_skipped_per_episode=2.5,
            min_entry_certainty_avg=0.62,
            min_avg_setup_quality=0.70,
            max_fomo_trade_rate=0.12,
            max_revenge_trade_rate=0.12,
            max_dd_breach_rate=0.08,
            max_consecutive_loss_rate=0.14,  # FIX: Was 0.20, now monotonically decreasing from Stage 6 (0.16)
            evaluation_window=120,
        ),
        max_steps_per_episode=2600,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 0.90),
            min_trend_clarity=0.0,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=False,
            exclude_market_open_close=False,
            prefer_recent_data=True,
            recent_data_weight=1.1,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.LOW_VOLATILITY,
                "news_volatility",
            ],
            regime_sampling_weights={
                "trending_up": 0.22,
                "trending_down": 0.22,
                "ranging": 0.18,
                "high_volatility": 0.18,
                "low_volatility": 0.10,
                "news_volatility": 0.10,
            },
            include_setup_maturity_metrics=True,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.32,
            lr_warmup_steps=15_000,
            reward_blend_enabled=True,
            reward_blend_episodes=35,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=70,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.TREND_ALIGNMENT: 0.60,
                TradingSkill.EXIT_QUALITY: 0.55,
                TradingSkill.ENTRY_TIMING: 0.52,
                TradingSkill.DRAWDOWN_CONTROL: 0.65,
                TradingSkill.PATIENCE: 0.45,  # FIX: Relaxed from 0.58 - ~3 trades/day acceptable
                TradingSkill.RISK_REWARD: 0.50,
                TradingSkill.SELECTIVITY: 0.45,
                TradingSkill.CERTAINTY: 0.50,
                TradingSkill.DISCIPLINE: 0.45,
                TradingSkill.SETUP_QUALITY: 0.45,
            },
            min_confidence=0.58,
            require_all_skills=False,
            weighted_threshold=0.55,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.12,
            max_entropy=0.45,
            low_entropy_penalty_scale=0.10,
            high_entropy_penalty_scale=0.06,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.72,
            demotion_threshold=0.35,
            hard_floors={
                "max_drawdown": 0.07,
                "profit_factor": 1.12,
                "dd_breach_rate": 0.08,
                "win_rate": 0.48,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=80,
            max_relaxation=0.05,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=200,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.68,
            recent_stages_weight=0.24,
            foundation_weight=0.08,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=350,
            review_duration=70,
            review_depth=5,
        ),
        # Validation before promotion (Stage 7)
        validation=ValidationConfig(
            enabled=True,
            validation_episodes=85,
            min_performance_ratio=0.82,
            max_performance_drop=0.18,
            required_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
            ],
            min_episodes_per_regime=15,
        ),
    )


def get_professional_config() -> CurriculumStageConfig:
    """
    PHASE 3 - MASTERY: Stage 8 (PROFESSIONAL)
    ==========================================
    Prop firm constraints. Real-world pressure.
    
    Goal: Trade under prop firm rules. Strict limits.
    
    ENABLED: Full prop firm constraints, tighter DD limits
    ENTROPY: 0.08 minimum - highly consistent behavior expected
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.PROFESSIONAL,
        name="Professional",
        description="MASTERY Phase: Prop firm constraints. Real trading pressure.",
        execution=ExecutionDifficulty(
            base_spread_points=0.20,
            spread_mult_range=(0.75, 1.55),
            max_spread_points=0.70,
            slippage_points_sigma=0.06,
            slippage_mult_range=(0.70, 1.65),
            max_slippage_points=0.30,
            commission_per_lot=2.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.80, 1.30),
            slippage_randomization_range=(0.75, 1.40),
            latency_randomization_range=(0, 2),
            volatility_scale_range=(0.82, 1.22),
            spread_shock_enabled=True,
            spread_shock_probability=0.01,
            spread_shock_multiplier=2.0,
            # DATA SPREAD: 100% - FULL realistic costs (real FTMO spreads)
            use_data_spread=True,
            data_spread_scale=1.0,
        ),
        rewards=RewardShaping(
            reward_scale=10.0,
            loss_multiplier=1.08,
            
            r_multiple_bonus_threshold=1.2,
            r_multiple_bonus_scale=0.12,
            r_multiple_bonus_cap=0.18,
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.08,
            mae_efficiency_threshold=1.4,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.03,
            optimal_trade_bars=7,
            max_trade_bars_for_bonus=22,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.38,    # D2 FIX: Must be >= STRATEGIST (0.36)
            agent_close_bonus=-0.12,     # Cutting winners is unprofessional
            hard_stop_penalty=0.10,
            risk_liquidation_penalty=0.15,
            
            truncation_winner_discount=0.12,   # GPT FIX: Reduced from 0.20
            truncation_loser_extra_penalty=0.12,  # GPT FIX: Reduced from 0.20
            
            entry_quality_integration=True,
            entry_quality_weight=0.22,

            # SETUP QUALITY: pro-level confluence
            setup_quality_enabled=True,
            setup_quality_threshold=0.70,
            setup_quality_bonus_scale=0.15,
            hasty_entry_penalty=0.08,

            # ENTRY CERTAINTY: confidence-based rewards
            certainty_threshold=0.70,
            entry_certainty_bonus=ENTRY_CERTAINTY_BONUS,
            low_certainty_penalty=0.15,
            
            # SESSION TIMING: Full strength for Pro level
            session_timing_enabled=True,
            off_hours_trade_penalty=0.25,
            prime_hours_trade_bonus=0.10,
            time_of_day_quality=TIME_OF_DAY_QUALITY,
            
            # MARKET STRUCTURE: REBALANCED (GPT FIX)
            market_structure_enabled=True,
            sr_proximity_bonus=0.32,          # Strong S/R entry reward
            sr_proximity_penalty=0.25,        # GPT FIX: Reduced from 0.45
            structure_alignment_bonus=0.24,   # Structure alignment
            bos_alignment_bonus=0.18,         # BOS confirmation
            order_block_entry_bonus=0.16,     # Order block awareness
            
            # DIVERGENCE: REBALANCED (GPT FIX)
            divergence_awareness_enabled=True,
            divergence_contra_penalty=0.22,   # GPT FIX: Reduced from 0.40
            divergence_aligned_bonus=0.20,    # Reward divergence awareness
            overbought_long_penalty=0.18,     # GPT FIX: Reduced from 0.35
            oversold_short_penalty=0.18,      # GPT FIX: Reduced from 0.35
            
            # REGIME: Full strength
            regime_awareness_enabled=True,
            risk_off_aggressive_penalty=0.15,
            high_vol_size_penalty=0.12,
            
            dd_shaping_enabled=True,
            dd_threshold=0.030,
            dd_penalty_scale=1.0,
            dd_severity_exponent=1.55,
            dd_severity_cap=1.35,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.03,
            loss_streak_penalty_per_loss=0.30,  # FIX: Was 0.55. Smooth progression: 0.10→0.20→0.30→0.40
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=7,    # GPT FIX: Increased from 4
            churn_penalty_per_trade=0.08, # GPT FIX: Reduced from 0.10
            
            hard_block_penalty=0.04,
            soft_block_penalty=0.02,
            
            # PER-STEP SHAPING: Professional patience mastery
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0005,  # Real holding cost
            patience_shaping_enabled=True,
            patience_bonus_per_bar=0.003,  # BOOSTED: Was 0.002
            patience_quality_threshold=0.45,
            dynamic_patience_enabled=True,
            patience_bonus_base=0.003,
            patience_bonus_multiplier=PATIENCE_BONUS_MULTIPLIER,
            per_step_min=-0.10,  # WIDENED: Allow loss_streak_caution penalty room
            per_step_max=0.05,

            # DELIBERATION TIME: reward thinking before acting
            deliberation_time_tracking=True,
            min_deliberation_bars=2,
            optimal_deliberation_range=(3, 8),
            too_fast_penalty=0.08,
            deliberation_quality_bonus=0.05,

            # WIN-RATE PRESERVATION
            win_rate_preservation_enabled=True,
            current_win_rate_threshold=0.45,
            selectivity_bonus=0.08,
            win_rate_decay_penalty=0.10,

            # COMPOUNDING SUCCESS
            compounding_success_enabled=True,
            consecutive_quality_trades_bonus=[0.0, 0.02, 0.05, 0.09, 0.14],
            quality_trade_r_multiple=1.0,
            quality_trade_entry_quality=0.6,
            quality_trade_exit_type="trailing_stop",

            # PSYCHOLOGICAL FACTORS: live trader discipline
            psychological_factors_enabled=True,
            fear_of_missing_out_penalty=0.15,
            revenge_trading_penalty=0.25,
            overconfidence_penalty=0.12,
            
            # LOSS STREAK CAUTION: Strong at this stage
            loss_streak_caution_enabled=True,
            loss_streak_caution_base=0.05,   # Stronger than STRATEGIST
            loss_streak_caution_cap=0.35,
            
            # ACTIVITY CONSISTENCY: Near-live discipline
            # Target moderate activity that doesn't spike vs STRATEGIST (5.5)
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=3.0,  # Progressive selectivity
            activity_deviation_penalty_scale=1.2,  # STRONG: specialist discipline
            activity_deviation_penalty_cap=10.0,  # Match PnL scale
            min_trades_penalty=0.1,
            
            exploration_bonus=0.0,
            directional_accuracy_weight=1.0,
            min_reward=-300.0,  # FIX: Was -5.0
            max_reward=300.0,   # FIX: Was 5.0
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=8,
            max_trades_per_session=4,
            max_consecutive_losses=4,
            loss_layer_stop=3,                # Block at 3 to prevent reaching 4 = no breach
            # Session budget (v5.5): Prop firm discipline
            session_loss_limit_pct=0.04,      # 4% max session loss
            session_consecutive_loss_limit=3,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_bars_between_entries=4,
            min_bars_after_loss=12,
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.09,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.08,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.55,
            min_setup_quality_for_entry=0.72,
            hard_stop_loss_eur=260.0,
            soft_stop_loss_eur=170.0,
            trailing_activation_eur=80.0,
            trailing_retrace_pct=0.28,
            time_decay_hours=5.0,
            risk_per_trade_pct=0.003,
            max_risk_per_trade_pct=0.006,
        ),
        competence=CompetenceThresholds(
            min_episodes=600,
            min_timesteps=1_400_000,
            min_win_rate=0.52,
            min_profit_factor=1.25,
            max_avg_drawdown=0.06,
            min_avg_pnl=100.0,
            min_avg_r_multiple=0.12,
            min_entropy=0.08,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=0.14,
            min_trades_per_episode_for_win_rate_stability=3,
            max_win_rate_wilson_width=0.12,
            max_pnl_std=5000.0,
            min_trade_count_avg=5.0,
            min_avg_bars_between_trades=7.0,
            min_setup_skipped_per_episode=3.0,
            min_entry_certainty_avg=0.65,
            min_avg_setup_quality=0.75,
            max_fomo_trade_rate=0.10,
            max_revenge_trade_rate=0.10,
            max_dd_breach_rate=0.06,
            max_consecutive_loss_rate=0.14,  # MONOTONIC: Same as STRATEGIST, count already tightened (5→4)
            max_mask_collapse_rate=0.12,
            max_stop_mode_rate=0.10,
            consistency_streak_required=5,
            consistency_streak_criteria={
                "win_rate": 0.48,
                "avg_bars_between_trades": 8.0,
                "entry_certainty_avg": 0.65,
                "fomo_trade_rate": 0.10,
            },
            evaluation_window=140,
        ),
        max_steps_per_episode=2800,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 1.0),
            min_trend_clarity=0.0,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=False,
            exclude_market_open_close=False,
            prefer_recent_data=True,
            recent_data_weight=1.2,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.LOW_VOLATILITY,
                "news_volatility",
            ],
            regime_sampling_weights={
                "trending_up": 0.20,
                "trending_down": 0.20,
                "ranging": 0.18,
                "high_volatility": 0.18,
                "low_volatility": 0.12,
                "news_volatility": 0.12,
            },
            include_setup_maturity_metrics=True,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.28,
            lr_warmup_steps=18_000,
            reward_blend_enabled=True,
            reward_blend_episodes=40,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=80,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.TREND_ALIGNMENT: 0.65,
                TradingSkill.EXIT_QUALITY: 0.60,
                TradingSkill.ENTRY_TIMING: 0.55,
                TradingSkill.DRAWDOWN_CONTROL: 0.70,
                TradingSkill.PATIENCE: 0.50,  # FIX: Relaxed from 0.62 - ~2.5 trades/day
                TradingSkill.RISK_REWARD: 0.55,
                TradingSkill.CONSISTENCY: 0.55,
                TradingSkill.SELECTIVITY: 0.50,
                TradingSkill.CERTAINTY: 0.55,
                TradingSkill.DISCIPLINE: 0.55,
                TradingSkill.SETUP_QUALITY: 0.50,
            },
            min_confidence=0.62,
            require_all_skills=False,
            weighted_threshold=0.58,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.08,
            max_entropy=0.40,
            low_entropy_penalty_scale=0.12,
            high_entropy_penalty_scale=0.08,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.75,
            demotion_threshold=0.38,
            hard_floors={
                "max_drawdown": 0.06,
                "profit_factor": 1.18,
                "dd_breach_rate": 0.06,
                "win_rate": 0.50,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=90,
            max_relaxation=0.04,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=220,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.65,
            recent_stages_weight=0.26,
            foundation_weight=0.09,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=400,
            review_duration=80,
            review_depth=6,
        ),
        validation=ValidationConfig(
            enabled=True,
            validation_episodes=100,
            min_performance_ratio=0.85,
            max_performance_drop=0.15,
            required_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
            ],
            min_episodes_per_regime=20,
        ),
    )


def get_live_ready_config() -> CurriculumStageConfig:
    """
    PHASE 3 - MASTERY: Stage 9 (LIVE_READY) - TERMINAL
    ===================================================
    Final stage. Live-ready consistency.
    
    Goal: Prove consistent profitability for live deployment.
    
    ENABLED: All features, strictest standards
    ENTROPY: 0.05 minimum - machine-level consistency expected
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.LIVE_READY,
        name="Live Ready",
        description="MASTERY Phase: Terminal stage. Prove live-ready consistency.",
        execution=ExecutionDifficulty(
            base_spread_points=0.22,
            spread_mult_range=(0.70, 1.65),
            max_spread_points=0.80,
            slippage_points_sigma=0.06,
            slippage_mult_range=(0.65, 1.75),
            max_slippage_points=0.35,
            commission_per_lot=3.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.75, 1.35),
            slippage_randomization_range=(0.70, 1.45),
            latency_randomization_range=(0, 3),
            volatility_scale_range=(0.80, 1.25),
            spread_shock_enabled=True,
            spread_shock_probability=0.015,
            spread_shock_multiplier=2.5,
            # DATA SPREAD: 100% - FULL realistic costs (real FTMO spreads)
            use_data_spread=True,
            data_spread_scale=1.0,
        ),
        rewards=RewardShaping(
            reward_scale=10.0,
            loss_multiplier=1.10,
            
            r_multiple_bonus_threshold=1.2,
            r_multiple_bonus_scale=0.15,
            r_multiple_bonus_cap=0.20,
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.10,
            mae_efficiency_threshold=1.3,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.035,
            optimal_trade_bars=6,
            max_trade_bars_for_bonus=20,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.40,    # D2 FIX: Maximum - must be >= PROFESSIONAL (0.38)
            agent_close_bonus=-0.12,     # Strongest penalty for premature exits
            hard_stop_penalty=0.12,
            risk_liquidation_penalty=0.20,
            
            truncation_winner_discount=0.10,   # GPT FIX: Reduced from 0.18
            truncation_loser_extra_penalty=0.12,  # GPT FIX: Reduced from 0.22
            
            entry_quality_integration=True,
            entry_quality_weight=0.25,

            # SETUP QUALITY: highest confluence standard
            setup_quality_enabled=True,
            setup_quality_threshold=0.70,
            setup_quality_bonus_scale=0.15,
            hasty_entry_penalty=0.08,

            # ENTRY CERTAINTY: confidence-based rewards
            certainty_threshold=0.70,
            entry_certainty_bonus=ENTRY_CERTAINTY_BONUS,
            low_certainty_penalty=0.15,
            
            # SESSION TIMING: MAXIMUM strength for Live Ready
            session_timing_enabled=True,
            off_hours_trade_penalty=0.30,
            prime_hours_trade_bonus=0.12,
            time_of_day_quality=TIME_OF_DAY_QUALITY,
            
            # MARKET STRUCTURE: REBALANCED (GPT FIX)
            market_structure_enabled=True,
            sr_proximity_bonus=0.35,          # Strong S/R entry reward
            sr_proximity_penalty=0.28,        # GPT FIX: Reduced from 0.50
            structure_alignment_bonus=0.28,   # Structure alignment
            bos_alignment_bonus=0.22,         # BOS confirmation
            order_block_entry_bonus=0.18,     # Order block awareness
            
            # DIVERGENCE: REBALANCED (GPT FIX)
            divergence_awareness_enabled=True,
            divergence_contra_penalty=0.25,   # GPT FIX: Reduced from 0.45
            divergence_aligned_bonus=0.22,    # Reward divergence awareness
            overbought_long_penalty=0.20,     # GPT FIX: Reduced from 0.40
            oversold_short_penalty=0.20,      # GPT FIX: Reduced from 0.40
            
            # REGIME: MAXIMUM strength
            regime_awareness_enabled=True,
            risk_off_aggressive_penalty=0.20,
            high_vol_size_penalty=0.15,
            
            dd_shaping_enabled=True,
            dd_threshold=0.025,
            dd_penalty_scale=1.1,
            dd_severity_exponent=1.6,
            dd_severity_cap=1.4,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.035,
            loss_streak_penalty_per_loss=0.40,  # FIX: Was 0.60. Smooth progression: 0.10→0.20→0.30→0.40
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=6,    # GPT FIX: Increased from 3
            churn_penalty_per_trade=0.10, # GPT FIX: Reduced from 0.12
            
            hard_block_penalty=0.045,
            soft_block_penalty=0.022,
            
            # PER-STEP SHAPING: Live-ready patience mastery
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0005,  # Real holding cost
            patience_shaping_enabled=True,
            patience_bonus_per_bar=0.004,  # BOOSTED: Was 0.002 - maximum patience reward
            patience_quality_threshold=0.48,
            dynamic_patience_enabled=True,
            patience_bonus_base=0.004,
            patience_bonus_multiplier=PATIENCE_BONUS_MULTIPLIER,
            per_step_min=-0.12,  # WIDENED: Allow loss_streak_caution penalty room
            per_step_max=0.05,

            # DELIBERATION TIME: reward thinking before acting
            deliberation_time_tracking=True,
            min_deliberation_bars=2,
            optimal_deliberation_range=(3, 8),
            too_fast_penalty=0.08,
            deliberation_quality_bonus=0.05,

            # WIN-RATE PRESERVATION
            win_rate_preservation_enabled=True,
            current_win_rate_threshold=0.45,
            selectivity_bonus=0.08,
            win_rate_decay_penalty=0.10,

            # COMPOUNDING SUCCESS
            compounding_success_enabled=True,
            consecutive_quality_trades_bonus=[0.0, 0.02, 0.05, 0.09, 0.14],
            quality_trade_r_multiple=1.0,
            quality_trade_entry_quality=0.6,
            quality_trade_exit_type="trailing_stop",

            # PSYCHOLOGICAL FACTORS: maximum discipline
            psychological_factors_enabled=True,
            fear_of_missing_out_penalty=0.15,
            revenge_trading_penalty=0.25,
            overconfidence_penalty=0.12,
            
            # LOSS STREAK CAUTION: Maximum at live-ready
            loss_streak_caution_enabled=True,
            loss_streak_caution_base=0.06,   # Strongest penalty
            loss_streak_caution_cap=0.40,
            
            # ACTIVITY CONSISTENCY: Live-ready discipline
            # Target moderate activity that doesn't spike vs PROFESSIONAL (5.0)
            activity_consistency_enabled=True,
            target_trades_per_1k_steps=3.0,  # Progressive selectivity
            activity_deviation_penalty_scale=1.5,  # VERY STRONG: live discipline
            activity_deviation_penalty_cap=10.0,  # Match PnL scale
            min_trades_penalty=0.1,
            
            exploration_bonus=0.0,
            directional_accuracy_weight=1.0,
            min_reward=-300.0,  # FIX: Was -5.0
            max_reward=300.0,   # FIX: Was 5.0
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=7,
            max_trades_per_session=4,
            max_consecutive_losses=4,  # MONOTONIC: Same as PROFESSIONAL, only tighten rate
            loss_layer_stop=3,                # Live-ready strict discipline
            # Session budget (v5.5): FTMO-ready strict discipline
            session_loss_limit_pct=0.03,      # 3% max session loss (FTMO daily is 5%)
            session_consecutive_loss_limit=3,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_bars_between_entries=4,
            min_bars_after_loss=16,
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.048,
            max_drawdown_limit=0.085,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.075,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.60,
            min_setup_quality_for_entry=0.75,
            hard_stop_loss_eur=250.0,
            soft_stop_loss_eur=160.0,
            trailing_activation_eur=85.0,
            trailing_retrace_pct=0.26,
            time_decay_hours=4.5,
            risk_per_trade_pct=0.0028,
            max_risk_per_trade_pct=0.0055,
        ),
        competence=CompetenceThresholds(
            min_episodes=700,
            min_timesteps=1_600_000,
            min_win_rate=0.54,
            min_profit_factor=1.32,
            max_avg_drawdown=0.055,
            min_avg_pnl=120.0,
            min_avg_r_multiple=0.14,
            min_entropy=0.05,
            max_win_rate_std=0.12,
            min_trades_per_episode_for_win_rate_stability=3,
            max_win_rate_wilson_width=0.10,
            max_pnl_std=4500.0,
            min_trade_count_avg=5.5,
            min_avg_bars_between_trades=8.0,
            min_setup_skipped_per_episode=3.0,
            min_entry_certainty_avg=0.68,
            min_avg_setup_quality=0.78,
            max_fomo_trade_rate=0.08,
            max_revenge_trade_rate=0.08,
            max_dd_breach_rate=0.05,
            max_consecutive_loss_rate=0.12,  # MONOTONIC: Tighter than PROFESSIONAL (0.14)
            max_mask_collapse_rate=0.10,
            max_stop_mode_rate=0.08,
            consistency_streak_required=5,
            consistency_streak_criteria={
                "win_rate": 0.52,
                "avg_bars_between_trades": 8.0,
                "entry_certainty_avg": 0.68,
                "fomo_trade_rate": 0.08,
            },
            evaluation_window=160,
        ),
        max_steps_per_episode=3000,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        is_terminal=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 1.0),
            min_trend_clarity=0.0,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=False,
            exclude_market_open_close=False,
            prefer_recent_data=True,
            recent_data_weight=1.3,
            allowed_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.LOW_VOLATILITY,
                "news_volatility",
            ],
            regime_sampling_weights={
                "trending_up": 0.20,
                "trending_down": 0.20,
                "ranging": 0.18,
                "high_volatility": 0.18,
                "low_volatility": 0.12,
                "news_volatility": 0.12,
            },
            include_setup_maturity_metrics=True,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.25,
            lr_warmup_steps=20_000,
            reward_blend_enabled=True,
            reward_blend_episodes=45,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=100,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.TREND_ALIGNMENT: 0.70,
                TradingSkill.EXIT_QUALITY: 0.65,
                TradingSkill.ENTRY_TIMING: 0.60,
                TradingSkill.DRAWDOWN_CONTROL: 0.75,
                TradingSkill.PATIENCE: 0.55,  # FIX: Relaxed from 0.68 - ~2 trades/day for expert
                TradingSkill.RISK_REWARD: 0.60,
                TradingSkill.CONSISTENCY: 0.65,
                TradingSkill.SELECTIVITY: 0.55,
                TradingSkill.CERTAINTY: 0.60,
                TradingSkill.DISCIPLINE: 0.60,
                TradingSkill.SETUP_QUALITY: 0.55,
            },
            min_confidence=0.68,
            require_all_skills=False,
            weighted_threshold=0.62,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.05,
            max_entropy=0.35,
            low_entropy_penalty_scale=0.15,
            high_entropy_penalty_scale=0.10,
            use_in_promotion=False,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.80,
            demotion_threshold=0.40,
            hard_floors={
                "max_drawdown": 0.055,
                "profit_factor": 1.25,
                "dd_breach_rate": 0.05,
                "win_rate": 0.52,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=100,
            max_relaxation=0.03,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=250,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.62,
            recent_stages_weight=0.28,
            foundation_weight=0.10,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=450,
            review_duration=90,
            review_depth=7,
        ),
        validation=ValidationConfig(
            enabled=True,
            validation_episodes=120,
            min_performance_ratio=0.88,
            max_performance_drop=0.12,
            required_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.LOW_VOLATILITY,
            ],
            min_episodes_per_regime=22,
        ),
    )


# =============================================================================
# Registry and Helpers
# =============================================================================

CURRICULUM_CONFIGS: Dict[CurriculumStage, Callable[[], CurriculumStageConfig]] = {
    # Phase 0: DISCOVERY
    CurriculumStage.EXPLORER: get_explorer_config,
    CurriculumStage.EXPERIMENTER: get_experimenter_config,
    # Phase 1: FOUNDATION
    CurriculumStage.TREND_STUDENT: get_trend_student_config,
    CurriculumStage.SESSION_STUDENT: get_session_student_config,
    CurriculumStage.TIMING_STUDENT: get_timing_student_config,
    # Phase 2: DEVELOPMENT
    CurriculumStage.INTEGRATOR: get_integrator_config,
    CurriculumStage.RISK_MANAGER: get_risk_manager_config,
    CurriculumStage.STRATEGIST: get_strategist_config,
    # Phase 3: MASTERY
    CurriculumStage.PROFESSIONAL: get_professional_config,
    CurriculumStage.LIVE_READY: get_live_ready_config,
}


def get_stage_config(stage: CurriculumStage) -> CurriculumStageConfig:
    factory = CURRICULUM_CONFIGS.get(stage)
    if factory is None:
        raise ValueError(f"Unknown curriculum stage: {stage}")
    return factory()


def get_all_stage_configs() -> Dict[CurriculumStage, CurriculumStageConfig]:
    return {stage: get_stage_config(stage) for stage in CurriculumStage}


def get_stage_progression() -> List[CurriculumStage]:
    return sorted(CurriculumStage, key=lambda s: s.value)


def get_next_stage(current: CurriculumStage) -> Optional[CurriculumStage]:
    progression = get_stage_progression()
    try:
        idx = progression.index(current)
        if idx < len(progression) - 1:
            return progression[idx + 1]
    except ValueError:
        pass
    return None


def get_previous_stage(current: CurriculumStage) -> Optional[CurriculumStage]:
    progression = get_stage_progression()
    try:
        idx = progression.index(current)
        if idx > 0:
            return progression[idx - 1]
    except ValueError:
        pass
    return None


def validate_stage_config(cfg: CurriculumStageConfig) -> List[str]:
    """
    Validates stage configuration for consistency.
    Returns list of human-readable issues.
    """
    issues: List[str] = []
    c = cfg.competence
    
    if not (0.0 <= c.min_win_rate <= 1.0):
        issues.append(f"{cfg.stage.name}: min_win_rate out of [0,1]")
    if not (0.0 <= c.max_avg_drawdown <= 1.0):
        issues.append(f"{cfg.stage.name}: max_avg_drawdown out of [0,1]")
    if c.evaluation_window < MIN_EVALUATION_EPISODES:
        issues.append(f"{cfg.stage.name}: evaluation_window < MIN_EVALUATION_EPISODES ({MIN_EVALUATION_EPISODES})")
    if c.min_episodes < MIN_EVALUATION_EPISODES:
        issues.append(f"{cfg.stage.name}: min_episodes < MIN_EVALUATION_EPISODES ({MIN_EVALUATION_EPISODES})")
    if c.min_avg_bars_between_trades < 0.0:
        issues.append(f"{cfg.stage.name}: min_avg_bars_between_trades must be >= 0")
    if c.min_setup_skipped_per_episode < 0.0:
        issues.append(f"{cfg.stage.name}: min_setup_skipped_per_episode must be >= 0")
    if not (0.0 <= c.min_entry_certainty_avg <= 1.0):
        issues.append(f"{cfg.stage.name}: min_entry_certainty_avg out of [0,1]: {c.min_entry_certainty_avg}")
    if not (0.0 <= c.min_avg_setup_quality <= 1.0):
        issues.append(f"{cfg.stage.name}: min_avg_setup_quality out of [0,1]: {c.min_avg_setup_quality}")
    if not (0.0 <= c.max_fomo_trade_rate <= 1.0):
        issues.append(f"{cfg.stage.name}: max_fomo_trade_rate out of [0,1]: {c.max_fomo_trade_rate}")
    if not (0.0 <= c.max_revenge_trade_rate <= 1.0):
        issues.append(f"{cfg.stage.name}: max_revenge_trade_rate out of [0,1]: {c.max_revenge_trade_rate}")
    if c.consistency_streak_required < 0:
        issues.append(f"{cfg.stage.name}: consistency_streak_required must be >= 0")
    if getattr(c, "consistency_streak_criteria", None):
        for k, v in (c.consistency_streak_criteria or {}).items():
            try:
                if float(v) < 0:
                    issues.append(f"{cfg.stage.name}: consistency_streak_criteria[{k}] must be >= 0")
            except Exception:
                issues.append(f"{cfg.stage.name}: consistency_streak_criteria[{k}] must be numeric")
    
    # Validate entropy targets
    e = cfg.entropy_targets
    if e.min_entropy > e.max_entropy:
        issues.append(f"{cfg.stage.name}: min_entropy > max_entropy")
    
    # AUDIT FIX: Single source of truth for entropy gating
    # If entropy is used in promotion, competence.min_entropy and entropy_targets.min_entropy
    # must match to avoid confusing mismatches (e.g., competence=0.30, targets=0.35)
    if e.use_in_promotion and abs(c.min_entropy - e.min_entropy) > 1e-6:
        issues.append(
            f"{cfg.stage.name}: competence.min_entropy ({c.min_entropy}) != "
            f"entropy_targets.min_entropy ({e.min_entropy}) while use_in_promotion=True. "
            f"This causes promotion gating confusion - pick ONE value for both."
        )
    
    # -------------------------------------------------------------------------
    # Constraint sanity checks (prevents silent "impossible" or "unsafe" stages)
    # -------------------------------------------------------------------------
    tc = cfg.constraints
    if tc.max_positions < 1:
        issues.append(f"{cfg.stage.name}: constraints.max_positions < 1 (invalid)")

    # Risk per trade should be sane and ordered
    if not (0.0 < tc.risk_per_trade_pct <= tc.max_risk_per_trade_pct <= 1.0):
        issues.append(
            f"{cfg.stage.name}: risk_per_trade_pct ({tc.risk_per_trade_pct}) and/or "
            f"max_risk_per_trade_pct ({tc.max_risk_per_trade_pct}) invalid or unordered"
        )

    # Drawdown limits must be ordered and within [0,1]
    if not (0.0 < tc.daily_drawdown_limit <= 1.0 and 0.0 < tc.max_drawdown_limit <= 1.0):
        issues.append(f"{cfg.stage.name}: drawdown limits must be within (0,1]")
    if tc.daily_drawdown_limit > tc.max_drawdown_limit + 1e-9:
        issues.append(
            f"{cfg.stage.name}: daily_drawdown_limit ({tc.daily_drawdown_limit}) > "
            f"max_drawdown_limit ({tc.max_drawdown_limit})"
        )

    # Emergency close should not exceed max DD limit
    if tc.emergency_close_threshold > tc.max_drawdown_limit + 1e-9:
        issues.append(
            f"{cfg.stage.name}: emergency_close_threshold ({tc.emergency_close_threshold}) > "
            f"max_drawdown_limit ({tc.max_drawdown_limit})"
        )

    # Trailing retrace should be a percentage in (0,1)
    if not (0.0 < tc.trailing_retrace_pct < 1.0):
        issues.append(f"{cfg.stage.name}: trailing_retrace_pct out of (0,1): {tc.trailing_retrace_pct}")

    # Stop loss ordering: soft <= hard (soft triggers warning, hard triggers stop)
    if tc.soft_stop_loss_eur > tc.hard_stop_loss_eur + 1e-9:
        issues.append(
            f"{cfg.stage.name}: soft_stop_loss_eur ({tc.soft_stop_loss_eur}) > "
            f"hard_stop_loss_eur ({tc.hard_stop_loss_eur}) - soft should be <= hard"
        )

    # Entry quality threshold should be in [0, 1]
    if not (0.0 <= tc.entry_quality_threshold <= 1.0):
        issues.append(f"{cfg.stage.name}: entry_quality_threshold out of [0,1]: {tc.entry_quality_threshold}")

    # Setup quality entry gate should be in [0, 1]
    min_setup_q = float(getattr(tc, "min_setup_quality_for_entry", 0.0) or 0.0)
    if not (0.0 <= min_setup_q <= 1.0):
        issues.append(f"{cfg.stage.name}: min_setup_quality_for_entry out of [0,1]: {min_setup_q}")

    # Time decay hours should be positive
    if tc.time_decay_hours <= 0.0:
        issues.append(f"{cfg.stage.name}: time_decay_hours must be > 0: {tc.time_decay_hours}")

    # Timing constraints should be non-negative
    if getattr(tc, "min_bars_between_entries", 0) < 0:
        issues.append(f"{cfg.stage.name}: min_bars_between_entries must be >= 0")
    if getattr(tc, "min_bars_after_loss", 0) < 0:
        issues.append(f"{cfg.stage.name}: min_bars_after_loss must be >= 0")
    if tc.min_minutes_between_entries < 0:
        issues.append(f"{cfg.stage.name}: min_minutes_between_entries must be >= 0")
    if tc.min_minutes_after_loss < 0:
        issues.append(f"{cfg.stage.name}: min_minutes_after_loss must be >= 0")

    # Trade limits: soft limit should not exceed hard cap
    if tc.max_trades_per_session > tc.max_trades_per_day:
        issues.append(
            f"{cfg.stage.name}: max_trades_per_session ({tc.max_trades_per_session}) > "
            f"max_trades_per_day ({tc.max_trades_per_day})"
        )

    # -------------------------------------------------------------------------
    # Execution sanity checks (ranges, ordering, and non-negative friction)
    # -------------------------------------------------------------------------
    ex = cfg.execution
    def _range_ok(r) -> bool:
        return isinstance(r, tuple) and len(r) == 2 and r[0] <= r[1]

    if ex.base_spread_points < 0.0 or ex.max_spread_points < 0.0:
        issues.append(f"{cfg.stage.name}: spread points must be >= 0")
    if ex.max_spread_points + 1e-12 < ex.base_spread_points:
        issues.append(
            f"{cfg.stage.name}: max_spread_points ({ex.max_spread_points}) < base_spread_points ({ex.base_spread_points})"
        )
    if ex.slippage_points_sigma < 0.0 or ex.max_slippage_points < 0.0:
        issues.append(f"{cfg.stage.name}: slippage params must be >= 0")

    if not hasattr(ex, "spread_mult_range") or not _range_ok(ex.spread_mult_range) or ex.spread_mult_range[0] <= 0.0:
        issues.append(f"{cfg.stage.name}: invalid spread_mult_range: {getattr(ex, 'spread_mult_range', None)}")
    if not hasattr(ex, "slippage_mult_range") or not _range_ok(ex.slippage_mult_range) or ex.slippage_mult_range[0] <= 0.0:
        issues.append(f"{cfg.stage.name}: invalid slippage_mult_range: {getattr(ex, 'slippage_mult_range', None)}")

    # Only require randomization ranges when randomization is enabled.
    if getattr(ex, "enable_randomization", False):
        if not hasattr(ex, "volatility_scale_range") or not _range_ok(ex.volatility_scale_range) or ex.volatility_scale_range[0] <= 0.0:
            issues.append(f"{cfg.stage.name}: invalid volatility_scale_range: {getattr(ex, 'volatility_scale_range', None)}")
        lr = getattr(ex, "latency_randomization_range", None)
        if not (isinstance(lr, tuple) and len(lr) == 2 and lr[0] <= lr[1]):
            issues.append(f"{cfg.stage.name}: invalid latency_randomization_range: {lr}")

        # If you model spread shocks, validate bounds.
        if getattr(ex, "spread_shock_enabled", False):
            p = getattr(ex, "spread_shock_probability", None)
            m = getattr(ex, "spread_shock_multiplier", None)
            if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
                issues.append(f"{cfg.stage.name}: spread_shock_probability must be in [0,1], got {p}")
            if not (isinstance(m, (int, float)) and m is not None and m >= 1.0):
                issues.append(f"{cfg.stage.name}: spread_shock_multiplier must be >= 1.0, got {m}")

    # -------------------------------------------------------------------------
    # Reward sanity checks (clipping + per-step bounds)
    # -------------------------------------------------------------------------
    rw = cfg.rewards
    if rw.min_reward > rw.max_reward:
        issues.append(f"{cfg.stage.name}: rewards.min_reward > rewards.max_reward")
    if rw.per_step_shaping_enabled and getattr(rw, "per_step_min", -0.01) > getattr(rw, "per_step_max", 0.01):
        issues.append(f"{cfg.stage.name}: per-step shaping bounds invalid (per_step_min > per_step_max)")

    # -------------------------------------------------------------------------
    # Reward "core knobs" presence (prevents silent regime changes in Stage 4+)
    # -------------------------------------------------------------------------
    # From TIMING_STUDENT onward, you rely on explicit PnL dominance / anti-hack controls.
    if cfg.stage.value >= CurriculumStage.TIMING_STUDENT.value:
        required_reward_fields = (
            "pnl_scale_factor",
            "max_shaping_to_pnl_ratio",
            "execution_cost_visibility_enabled",
            "execution_cost_reward_scale",
        )
        missing = [k for k in required_reward_fields if not hasattr(rw, k)]
        if missing:
            issues.append(
                f"{cfg.stage.name}: rewards missing core fields {missing}. "
                "This can silently revert to defaults and break PnL dominance / cost visibility assumptions."
            )

    # Validate composite scoring weights - keys must match compute_composite_score() components
    cs = cfg.composite_scoring
    if cs.enabled:
        if not getattr(cs, "weights", None):
            issues.append(
                f"{cfg.stage.name}: composite_scoring.enabled=True but weights is empty/missing. "
                "Either provide default weights summing to 1.0 in CompositeScoringConfig, or set cs.weights explicitly."
            )
        weight_sum = sum(cs.weights.values())
        # FIX: Tighten tolerance from 0.1 to 0.03 - loose tolerance enables reward hacking
        if abs(weight_sum - 1.0) > 0.03:
            issues.append(f"{cfg.stage.name}: composite scoring weights sum to {weight_sum:.3f}, expected 1.0 ± 0.03")
        
        # Canonicalize composite keys so legacy configs ("max_drawdown") validate correctly
        from envs.curriculum.config.registry import canonicalize_composite_key
        canon_weight_keys = {canonicalize_composite_key(k) for k in cs.weights.keys()}
        canon_floor_keys = {canonicalize_composite_key(k) for k in cs.hard_floors.keys()}

        # Validate weight keys against allowed composite weight keys
        bad_weight_keys = canon_weight_keys - COMPOSITE_WEIGHT_KEYS
        if bad_weight_keys:
            issues.append(f"{cfg.stage.name}: composite_scoring.weights has unknown keys: {sorted(bad_weight_keys)}. Allowed: {sorted(COMPOSITE_WEIGHT_KEYS)}")
        
        # Validate hard_floors keys against allowed hard floor keys
        bad_floor_keys = canon_floor_keys - COMPOSITE_HARD_FLOOR_KEYS
        if bad_floor_keys:
            issues.append(f"{cfg.stage.name}: composite_scoring.hard_floors has unknown keys: {sorted(bad_floor_keys)}. Allowed: {sorted(COMPOSITE_HARD_FLOOR_KEYS)}")
    
    # Validate adaptive thresholds - keys must be valid threshold fields
    # Only validate when enabled to avoid false positives from defaults
    at = cfg.adaptive_thresholds
    if at.enabled:
        for metric in at.relaxable_metrics:
            if not is_valid_threshold_field(metric):
                issues.append(f"{cfg.stage.name}: adaptive_thresholds.relaxable_metrics '{metric}' is not a valid threshold field")
        
        for metric in at.never_relax:
            if not is_valid_threshold_field(metric):
                issues.append(f"{cfg.stage.name}: adaptive_thresholds.never_relax '{metric}' is not a valid threshold field")
    
    # Validate skill requirements
    sr = cfg.skill_requirements
    for skill, threshold in sr.required_skills.items():
        if not (0.0 <= threshold <= 1.0):
            issues.append(f"{cfg.stage.name}: skill {skill.value} threshold {threshold} out of [0,1]")
    
    # Check for dead code: expert_signal_dropout with include_expert_signals=False
    dropout = getattr(cfg, "expert_signal_dropout", 0.0)
    if not cfg.include_expert_signals and dropout > 0.0:
        issues.append(f"{cfg.stage.name}: expert_signal_dropout={dropout} has no effect when include_expert_signals=False (suggest setting to 0.0)")
    
    return issues


def validate_all_configs() -> Dict[CurriculumStage, List[str]]:
    """Validate all stage configurations."""
    return {stage: validate_stage_config(get_stage_config(stage)) for stage in CurriculumStage}


def validate_curriculum_monotonicity() -> List[str]:
    """
    Validate curriculum monotonicity - ensures harder stages are actually harder.
    
    Checks:
    - DD limits should be non-increasing (tighter at higher stages)
    - Randomization ranges should be non-decreasing (more variance at higher stages)
    - Competence thresholds should be non-decreasing (stricter requirements)
    - never_relax metrics should never appear in relaxable_metrics
    
    Returns:
        List of issues found (empty = valid)
    """
    issues: List[str] = []
    configs = [get_stage_config(stage) for stage in get_stage_progression()]
    
    for i in range(1, len(configs)):
        prev, curr = configs[i-1], configs[i]
        prev_name, curr_name = prev.stage.name, curr.stage.name
        
        # DD limits should get tighter (non-increasing)
        if curr.competence.max_avg_drawdown > prev.competence.max_avg_drawdown:
            issues.append(f"{curr_name}: max_avg_drawdown ({curr.competence.max_avg_drawdown}) > {prev_name} ({prev.competence.max_avg_drawdown})")
        
        # DD breach rate should get tighter (non-increasing)
        if curr.competence.max_dd_breach_rate > prev.competence.max_dd_breach_rate:
            issues.append(f"{curr_name}: max_dd_breach_rate ({curr.competence.max_dd_breach_rate}) > {prev_name} ({prev.competence.max_dd_breach_rate})")
        
        # Win rate should be non-decreasing (stricter requirements)
        if curr.competence.min_win_rate < prev.competence.min_win_rate - 0.02:  # Allow 2% tolerance
            issues.append(f"{curr_name}: min_win_rate ({curr.competence.min_win_rate}) < {prev_name} ({prev.competence.min_win_rate})")
        
        # Profit factor should be non-decreasing
        if curr.competence.min_profit_factor < prev.competence.min_profit_factor - 0.05:  # Allow 0.05 tolerance
            issues.append(f"{curr_name}: min_profit_factor ({curr.competence.min_profit_factor}) < {prev_name} ({prev.competence.min_profit_factor})")
        
        # Commission should be non-decreasing (more realistic over time)
        if curr.execution.commission_per_lot < prev.execution.commission_per_lot:
            issues.append(f"{curr_name}: commission_per_lot ({curr.execution.commission_per_lot}) < {prev_name} ({prev.execution.commission_per_lot})")

        # Patience/discipline thresholds should be non-decreasing (or non-increasing for max rates)
        if curr.competence.min_avg_bars_between_trades < prev.competence.min_avg_bars_between_trades - 1e-6:
            issues.append(
                f"{curr_name}: min_avg_bars_between_trades ({curr.competence.min_avg_bars_between_trades}) < "
                f"{prev_name} ({prev.competence.min_avg_bars_between_trades})"
            )
        if curr.competence.min_setup_skipped_per_episode < prev.competence.min_setup_skipped_per_episode - 1e-6:
            issues.append(
                f"{curr_name}: min_setup_skipped_per_episode ({curr.competence.min_setup_skipped_per_episode}) < "
                f"{prev_name} ({prev.competence.min_setup_skipped_per_episode})"
            )
        if curr.competence.min_entry_certainty_avg < prev.competence.min_entry_certainty_avg - 1e-6:
            issues.append(
                f"{curr_name}: min_entry_certainty_avg ({curr.competence.min_entry_certainty_avg}) < "
                f"{prev_name} ({prev.competence.min_entry_certainty_avg})"
            )
        if curr.competence.min_avg_setup_quality < prev.competence.min_avg_setup_quality - 1e-6:
            issues.append(
                f"{curr_name}: min_avg_setup_quality ({curr.competence.min_avg_setup_quality}) < "
                f"{prev_name} ({prev.competence.min_avg_setup_quality})"
            )
        if curr.competence.max_fomo_trade_rate > prev.competence.max_fomo_trade_rate + 1e-9:
            issues.append(
                f"{curr_name}: max_fomo_trade_rate ({curr.competence.max_fomo_trade_rate}) > "
                f"{prev_name} ({prev.competence.max_fomo_trade_rate})"
            )
        if curr.competence.max_revenge_trade_rate > prev.competence.max_revenge_trade_rate + 1e-9:
            issues.append(
                f"{curr_name}: max_revenge_trade_rate ({curr.competence.max_revenge_trade_rate}) > "
                f"{prev_name} ({prev.competence.max_revenge_trade_rate})"
            )

        # -------------------------------------------------------------
        # NEW: Constraint monotonicity (risk + DD tightening over stages)
        # -------------------------------------------------------------
        if curr.constraints.daily_drawdown_limit > prev.constraints.daily_drawdown_limit + 1e-9:
            issues.append(
                f"{curr_name}: constraints.daily_drawdown_limit ({curr.constraints.daily_drawdown_limit}) > "
                f"{prev_name} ({prev.constraints.daily_drawdown_limit})"
            )
        if curr.constraints.max_drawdown_limit > prev.constraints.max_drawdown_limit + 1e-9:
            issues.append(
                f"{curr_name}: constraints.max_drawdown_limit ({curr.constraints.max_drawdown_limit}) > "
                f"{prev_name} ({prev.constraints.max_drawdown_limit})"
            )
        if curr.constraints.risk_per_trade_pct > prev.constraints.risk_per_trade_pct + 1e-12:
            issues.append(
                f"{curr_name}: risk_per_trade_pct ({curr.constraints.risk_per_trade_pct}) > "
                f"{prev_name} ({prev.constraints.risk_per_trade_pct})"
            )

        # New: pacing should generally tighten (prevents late-stage churn)
        if getattr(curr.constraints, "min_bars_between_entries", 0) < getattr(prev.constraints, "min_bars_between_entries", 0):
            issues.append(
                f"{curr_name}: min_bars_between_entries ({getattr(curr.constraints, 'min_bars_between_entries', 0)}) < "
                f"{prev_name} ({getattr(prev.constraints, 'min_bars_between_entries', 0)})"
            )
        if curr.constraints.min_minutes_between_entries < prev.constraints.min_minutes_between_entries:
            issues.append(
                f"{curr_name}: min_minutes_between_entries ({curr.constraints.min_minutes_between_entries}) < "
                f"{prev_name} ({prev.constraints.min_minutes_between_entries})"
            )

        # Post-loss cooldown should be non-decreasing (more discipline at higher stages)
        if getattr(curr.constraints, "min_bars_after_loss", 0) < getattr(prev.constraints, "min_bars_after_loss", 0):
            issues.append(
                f"{curr_name}: min_bars_after_loss ({getattr(curr.constraints, 'min_bars_after_loss', 0)}) < "
                f"{prev_name} ({getattr(prev.constraints, 'min_bars_after_loss', 0)})"
            )
        if curr.constraints.min_minutes_after_loss < prev.constraints.min_minutes_after_loss:
            issues.append(
                f"{curr_name}: min_minutes_after_loss ({curr.constraints.min_minutes_after_loss}) < "
                f"{prev_name} ({prev.constraints.min_minutes_after_loss})"
            )

        # Entry quality threshold should be non-decreasing (stricter quality gates)
        if curr.constraints.entry_quality_threshold < prev.constraints.entry_quality_threshold - 0.01:  # 1% tolerance
            issues.append(
                f"{curr_name}: entry_quality_threshold ({curr.constraints.entry_quality_threshold}) < "
                f"{prev_name} ({prev.constraints.entry_quality_threshold})"
            )

        # Setup quality threshold should be non-decreasing (stricter confluence gates)
        curr_setup_q = float(getattr(curr.constraints, "min_setup_quality_for_entry", 0.0) or 0.0)
        prev_setup_q = float(getattr(prev.constraints, "min_setup_quality_for_entry", 0.0) or 0.0)
        if curr_setup_q < prev_setup_q - 0.01:
            issues.append(
                f"{curr_name}: min_setup_quality_for_entry ({curr_setup_q}) < "
                f"{prev_name} ({prev_setup_q})"
            )

        # -------------------------------------------------------------
        # NEW: Reward monotonicity (prevent spikes that confuse learning)
        # -------------------------------------------------------------
        # Exploration bonus should be non-increasing (agent learns, explores less)
        curr_exp = getattr(curr.rewards, "exploration_bonus", 0.0)
        prev_exp = getattr(prev.rewards, "exploration_bonus", 0.0)
        if curr_exp > prev_exp + 0.005:  # Allow tiny tolerance
            issues.append(
                f"{curr_name}: exploration_bonus ({curr_exp}) > {prev_name} ({prev_exp}) - "
                "exploration should decrease as agent learns"
            )

        # Daily trade soft limit should be non-increasing (more selective)
        curr_soft = getattr(curr.rewards, "daily_trade_soft_limit", 50)
        prev_soft = getattr(prev.rewards, "daily_trade_soft_limit", 50)
        if curr_soft > prev_soft:
            issues.append(
                f"{curr_name}: daily_trade_soft_limit ({curr_soft}) > {prev_name} ({prev_soft})"
            )

        # Activity target should not spike dramatically (prevents curriculum shock)
        curr_target = getattr(curr.rewards, "target_trades_per_1k_steps", 10.0)
        prev_target = getattr(prev.rewards, "target_trades_per_1k_steps", 10.0)
        if curr_target > prev_target * 1.5:  # Allow some variation but not 5x jumps
            issues.append(
                f"{curr_name}: target_trades_per_1k_steps ({curr_target}) > 1.5x {prev_name} ({prev_target}) - "
                "activity target should not spike dramatically"
            )

        # Max trades per day should be non-increasing
        if curr.constraints.max_trades_per_day > prev.constraints.max_trades_per_day:
            issues.append(
                f"{curr_name}: max_trades_per_day ({curr.constraints.max_trades_per_day}) > "
                f"{prev_name} ({prev.constraints.max_trades_per_day})"
            )

        # -------------------------------------------------------------
        # NEW: Entropy floor should be non-increasing across curriculum
        # -------------------------------------------------------------
        if curr.entropy_targets.min_entropy > prev.entropy_targets.min_entropy + 1e-9:
            issues.append(
                f"{curr_name}: entropy_targets.min_entropy ({curr.entropy_targets.min_entropy}) > "
                f"{prev_name} ({prev.entropy_targets.min_entropy})"
            )

        # -------------------------------------------------------------
        # NEW: Once randomization is enabled, it should not be disabled later
        # -------------------------------------------------------------
        if prev.execution.enable_randomization and not curr.execution.enable_randomization:
            issues.append(f"{curr_name}: enable_randomization turned OFF after being ON in {prev_name}")
    
    # Validate never_relax vs relaxable_metrics for adaptive thresholds
    for cfg in configs:
        at = cfg.adaptive_thresholds
        overlap = at.relaxable_metrics & at.never_relax
        if overlap:
            issues.append(f"{cfg.stage.name}: metrics in both relaxable and never_relax: {overlap}")
        
        # Note: Individual metric validation is now done in validate_stage_config()
    
    return issues


def print_curriculum_summary() -> None:
    """Print a summary of the curriculum configuration for debugging."""
    print("\n" + "=" * 80)
    print("10-STAGE PROP FIRM CURRICULUM SUMMARY")
    print("=" * 80)
    
    phases = {
        "PHASE 0 - DISCOVERY": [CurriculumStage.EXPLORER, CurriculumStage.EXPERIMENTER],
        "PHASE 1 - FOUNDATION": [CurriculumStage.TREND_STUDENT, CurriculumStage.SESSION_STUDENT, CurriculumStage.TIMING_STUDENT],
        "PHASE 2 - DEVELOPMENT": [CurriculumStage.INTEGRATOR, CurriculumStage.RISK_MANAGER, CurriculumStage.STRATEGIST],
        "PHASE 3 - MASTERY": [CurriculumStage.PROFESSIONAL, CurriculumStage.LIVE_READY],
    }
    
    for phase_name, stages in phases.items():
        print(f"\n{phase_name}")
        print("-" * 40)
        for stage in stages:
            cfg = get_stage_config(stage)
            c = cfg.competence
            e = cfg.execution
            ent = cfg.entropy_targets
            print(f"  Stage {stage.value}: {cfg.name}")
            print(f"    WR >= {c.min_win_rate:.0%}, PF >= {c.min_profit_factor:.2f}, DD <= {c.max_avg_drawdown:.1%}")
            print(f"    Entropy target: {ent.min_entropy:.2f}, Commission: ${e.commission_per_lot}/lot")
    
    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    config_issues = validate_all_configs()
    mono_issues = validate_curriculum_monotonicity()
    
    all_issues = []
    for stage, issues in config_issues.items():
        all_issues.extend(issues)
    all_issues.extend(mono_issues)
    
    if all_issues:
        print("\n[!] Issues found:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] All validations passed!")
    
    print("\n" + "=" * 80)
